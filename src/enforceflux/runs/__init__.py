"""Run-artifact contract shared by every stage driver.

Each stage writes under ``{outputs_root}/{run_name}/{stage}/`` and lands the
same three required files so downstream consumers can read a run without
introspection:

* ``manifest.json`` — stage, run name, git sha, timestamps, and the
  role-keyed list of payload files (each with sha256).
* ``config.snapshot.yaml`` — fully-resolved config that produced the run.
* ``logs/run.log`` — optional, attached with :func:`attach_file_logger`.

Payload files are recorded by ``role`` (a stable string like
``"concentration_field"`` or ``"H_columns"``); downstream stages look them up
by role so filenames can change without breaking the pipeline.

The output tree is a contract: consumers open ``manifest.json`` first, then
trust the declared role→file mapping. Writers MUST call
:meth:`RunDir.record_output` for every non-log file they produce.

See ``.claude/plans/i-want-to-make-luminous-lynx.md`` for the full contract.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUTS_ROOT = _REPO_ROOT / "runs"

OUTPUT_CONTRACT_VERSION = "1.0"


@dataclass
class _OutputEntry:
    """One recorded payload file inside a RunDir."""

    path: str  # relative to RunDir.root
    role: str
    sha256: str | None = None  # filled in on finalize


@dataclass
class RunDir:
    """A single stage's output directory, with a manifest under construction."""

    root: Path
    stage: str
    run_name: str
    started_at: float
    inputs: dict[str, str] = field(default_factory=dict)  # stage → RunDir path
    _outputs: list[_OutputEntry] = field(default_factory=list)
    _extra: dict[str, Any] = field(default_factory=dict)
    _config_snapshot: Any = None

    # ------------------------------------------------------------------
    # Paths and outputs
    # ------------------------------------------------------------------

    def path(self, *parts: str) -> Path:
        """Return an absolute path inside this run dir, mkdir'ing parents."""
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def record_output(self, relpath: str, role: str) -> None:
        """Declare that ``relpath`` (relative to ``root``) is a payload file.

        The ``role`` key is the stable identifier downstream stages use to
        look the file up via :meth:`UpstreamRef.file`. Filenames can change
        across versions; roles cannot within a contract version.
        """
        for existing in self._outputs:
            if existing.path == relpath:
                existing.role = role
                return
        self._outputs.append(_OutputEntry(path=relpath, role=role))

    def add_manifest_field(self, key: str, value: Any) -> None:
        """Attach an extra top-level field to the manifest."""
        self._extra[key] = value

    def snapshot_config(self, config: Any) -> None:
        """Record the resolved config; dumped as YAML on :meth:`finalize`.

        Accepts a plain dict (typical for YAML-loaded configs) or a
        dataclass tree; dataclasses are serialized with ``asdict``.
        """
        self._config_snapshot = config

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def finalize(self) -> dict[str, Any]:
        """Write manifest + config snapshot; return the completion contract.

        A declared-but-missing file is a writer bug and aborts finalize —
        an absent manifest means the run crashed or is still in progress.
        """
        missing = [
            entry.path
            for entry in self._outputs
            if not self.root.joinpath(entry.path).is_file()
        ]
        if missing:
            raise RuntimeError(
                "cannot finalize RunDir; declared outputs are missing: "
                + ", ".join(missing)
            )

        for entry in self._outputs:
            entry.sha256 = _sha256(self.root / entry.path)

        if self._config_snapshot is not None:
            snap = self.path("config.snapshot.yaml")
            snap.write_text(
                yaml.safe_dump(_to_plain(self._config_snapshot), sort_keys=False)
            )
            # Snapshot is metadata, not a payload — no role, tracked separately.

        manifest = {
            "contract_version": OUTPUT_CONTRACT_VERSION,
            "stage": self.stage,
            "run_name": self.run_name,
            "status": "complete",
            "output_dir": str(self.root),
            "enforceflux_version": _package_version(),
            "git_sha": _git_sha(),
            "started_at": _fmt_time(self.started_at),
            "finished_at": _fmt_time(time.time()),
            "inputs": self.inputs,
            "outputs": [
                {"path": e.path, "role": e.role, "sha256": e.sha256}
                for e in self._outputs
            ],
            **self._extra,
        }
        manifest_path = self.path("manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, default=_json_default)
        )
        return {
            "output_dir": str(self.root),
            "manifest": str(manifest_path),
            "files": {e.role: str(self.root / e.path) for e in self._outputs},
        }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def open_run_dir(
    *,
    stage: str,
    run_name: str,
    outputs_root: str | Path | None = None,
    inputs: dict[str, str] | None = None,
) -> RunDir:
    """Create ``{outputs_root}/{run_name}/{stage}/`` and return a RunDir.

    Prior contents are NOT deleted; a rerun overwrites files with the same
    name and appends to the outputs list on :meth:`RunDir.finalize`.
    """
    base = Path(outputs_root).resolve() if outputs_root else DEFAULT_OUTPUTS_ROOT
    root = base / _sanitize(run_name) / _sanitize(stage)
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    return RunDir(
        root=root,
        stage=stage,
        run_name=run_name,
        started_at=time.time(),
        inputs=dict(inputs or {}),
    )


def attach_file_logger(run_dir: RunDir, level: int = logging.INFO) -> logging.Handler:
    """Route stdlib logging into ``{run_dir}/logs/run.log`` (append mode)."""
    log_path = run_dir.path("logs", "run.log")
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(handler)
    return handler


# ---------------------------------------------------------------------------
# Upstream reader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UpstreamRef:
    """Handle to a completed upstream RunDir, opened via its manifest."""

    root: Path
    manifest: dict[str, Any]

    @property
    def stage(self) -> str:
        return self.manifest["stage"]

    @property
    def run_name(self) -> str:
        return self.manifest["run_name"]

    def file(self, role: str) -> Path:
        """Absolute path of the payload file with the given ``role``."""
        for entry in self.manifest.get("outputs", []):
            if entry.get("role") == role:
                return self.root / entry["path"]
        available = sorted({e.get("role", "?") for e in self.manifest.get("outputs", [])})
        raise KeyError(
            f"upstream {self.stage!r} manifest has no role={role!r}; "
            f"available roles: {available}"
        )

    def has(self, role: str) -> bool:
        return any(e.get("role") == role for e in self.manifest.get("outputs", []))


def read_upstream(path: str | Path) -> UpstreamRef:
    """Open a prior RunDir at ``path`` and return an :class:`UpstreamRef`.

    ``path`` may point to the RunDir itself or the manifest.json inside it.
    """
    p = Path(path).resolve()
    if p.is_file() and p.name == "manifest.json":
        root, manifest_path = p.parent, p
    else:
        root = p
        manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no manifest.json under {root}")
    manifest = json.loads(manifest_path.read_text())
    got = manifest.get("contract_version")
    if got != OUTPUT_CONTRACT_VERSION:
        raise ValueError(
            f"upstream manifest at {manifest_path} declares contract_version "
            f"{got!r}, this build expects {OUTPUT_CONTRACT_VERSION!r}"
        )
    return UpstreamRef(root=root, manifest=manifest)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _sanitize(name: str) -> str:
    bad = '/\\ \t\n"\''
    out = "".join("_" if c in bad else c for c in str(name))
    return out or "unnamed"


def _fmt_time(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return None


def _package_version() -> str:
    try:
        from importlib.metadata import version
        return version("enforceflux")
    except Exception:
        return "unknown"


def _to_plain(obj: Any) -> Any:
    """Convert dataclasses / nested containers to plain YAML-friendly types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_plain(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return obj


def _json_default(o: Any) -> Any:
    if isinstance(o, Path):
        return str(o)
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    if hasattr(o, "tolist"):
        try:
            return o.tolist()
        except Exception:
            pass
    return str(o)


from enforceflux.runs.stage_config import StageConfig, load_stage_config

__all__ = [
    "OUTPUT_CONTRACT_VERSION",
    "RunDir",
    "StageConfig",
    "UpstreamRef",
    "attach_file_logger",
    "load_stage_config",
    "open_run_dir",
    "read_upstream",
]
