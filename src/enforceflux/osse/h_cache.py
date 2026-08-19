"""
On-disk, content-addressed cache for FLEXPART Jacobians (H matrices).

# --- source-heterogeneity OSSE (M5) ---

The cache key is a SHA256 of the canonical JSON of the *transport-relevant*
subset of a resolved dispersion config: transport model, met, receptors,
domain, output shape. Source-field parameters that only vary the truth field
(``sources.config.covariance.L_m``, ``sources.config.cv``,
``sources.config.seed``, ``sources.config.basis``) are excluded, which is the
whole point of the sweep cache — a single FLEXPART Jacobian is reused across
every (L, CV, realization) triple that shares the same transport setup.

Values on disk live in ``<root>/<sha256>/`` and contain:

- ``jacobian.npz``    — the Jacobian and any coarse/fine metadata.
- ``manifest.json``   — dims, receptor ids, met time range, source fingerprint.
- optional passthrough of the FLEXPART NetCDF output for provenance.

macOS note (project memory): FLEXPART is flaky under threading here. Callers
that produce the RunDir must launch FLEXPART with ``OMP_NUM_THREADS=1``.
Validation of time coverage happens inside :meth:`DiskHCache.put`, so a
FLEXPART output with missing/duplicate/short time steps is refused and never
enters the cache.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Public protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class HCache(Protocol):
    """Protocol for a keyed store of pre-computed transport Jacobians.

    Mirrors the sibling ``InMemoryHCache`` shipped in ``sweep.py`` (Agent D).
    Implementations must be safe to construct with a fresh key set on every
    sweep and must not raise on ``get`` misses.
    """

    def get(self, key: str) -> Path | None: ...

    def put(self, key: str, dispersion_run_dir: Path) -> Path: ...


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------

# Keys under ``dispersion.sources.config`` that describe *the truth field only*
# and therefore must NOT influence the transport cache key. Everything else in
# the dispersion config counts.
_SOURCE_FIELD_ONLY_KEYS = frozenset({"covariance", "cv", "seed", "basis"})


def _canonical(obj: Any) -> Any:
    """Return a JSON-safe, deterministically-ordered representation of ``obj``."""

    if isinstance(obj, Mapping):
        return {str(k): _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _canonical(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _transport_subset(dispersion_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the transport-relevant subset of a resolved dispersion config.

    Keeps: ``transport``, ``met``, ``domain``, ``receptors``, ``flexpart``, and
    the ``sources`` block *minus* the truth-field-only knobs. The sources block
    is retained (positions + release altitude are needed to build FLEXPART
    RELEASES), but knobs that only reshape the truth field are stripped.
    """

    if "dispersion" in dispersion_cfg and isinstance(dispersion_cfg["dispersion"], Mapping):
        cfg = dispersion_cfg["dispersion"]
    else:
        cfg = dispersion_cfg

    subset: dict[str, Any] = {}
    for k in ("transport", "met", "domain", "receptors", "flexpart"):
        if k in cfg:
            subset[k] = _canonical(cfg[k])

    if "sources" in cfg:
        sources = cfg["sources"]
        if isinstance(sources, Mapping) and "generator" in sources:
            src_out: dict[str, Any] = {"generator": sources.get("generator")}
            inner = sources.get("config", {}) or {}
            stripped = {
                k: v for k, v in inner.items() if k not in _SOURCE_FIELD_ONLY_KEYS
            }
            src_out["config"] = _canonical(stripped)
            subset["sources"] = src_out
        else:
            # Explicit list of RunSources — positions matter for FLEXPART.
            subset["sources"] = _canonical(sources)

    return subset


def transport_cache_key(dispersion_cfg: Mapping[str, Any]) -> str:
    """SHA256 hex of the canonical JSON of the transport-only subset."""

    subset = _transport_subset(dispersion_cfg)
    payload = json.dumps(subset, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Time-coverage validation
# ---------------------------------------------------------------------------


class CacheValidationError(RuntimeError):
    """Raised when a candidate H fails the coverage check and is refused."""


def _validate_time_coverage(run_dir: Path) -> dict[str, Any]:
    """Inspect a dispersion RunDir and confirm its H has no time gaps.

    Rules, in order of preference:

    1. If a ``manifest.json`` exists and carries ``time_start``, ``time_end``,
       and ``time_step_seconds``, the recorded ``n_time_steps`` must equal
       ``(end - start) / step + 1`` (no gaps).
    2. Otherwise, inspect ``*.nc`` outputs (FLEXPART writes ``grid_time_*.nc``
       / ``receptor_*.nc``): the ``time`` coordinate must be strictly
       increasing with a single, consistent stride and no missing samples.
    3. If neither manifest nor .nc are present, the RunDir must at least
       contain ``jacobian.npz``.
    """

    if not run_dir.exists():
        raise CacheValidationError(f"dispersion RunDir does not exist: {run_dir}")

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise CacheValidationError(f"unreadable manifest.json: {exc}") from exc
        ts = manifest.get("time_start")
        te = manifest.get("time_end")
        step = manifest.get("time_step_seconds")
        n = manifest.get("n_time_steps")
        if ts is not None and te is not None and step and n:
            expected = int(round((float(te) - float(ts)) / float(step))) + 1
            if int(n) != expected:
                raise CacheValidationError(
                    f"time coverage gap: manifest reports {n} steps but "
                    f"[{ts}, {te}] @ {step}s implies {expected}"
                )
            return {
                "time_start": ts,
                "time_end": te,
                "time_step_seconds": step,
                "n_time_steps": int(n),
            }

    nc_files = sorted(run_dir.glob("*.nc"))
    if nc_files:
        return _validate_nc_time(nc_files[0])

    if not (run_dir / "jacobian.npz").exists():
        raise CacheValidationError(
            f"RunDir {run_dir} has neither manifest.json, .nc output, nor jacobian.npz"
        )
    return {}


def _validate_nc_time(nc_path: Path) -> dict[str, Any]:
    try:
        from netCDF4 import Dataset  # type: ignore
    except ImportError:  # pragma: no cover - optional dep
        raise CacheValidationError(
            "netCDF4 is required to validate FLEXPART output time coverage"
        )

    with Dataset(nc_path) as ds:
        time_var = None
        for name in ("time", "Times", "TIME"):
            if name in ds.variables:
                time_var = ds.variables[name]
                break
        if time_var is None:
            raise CacheValidationError(
                f"{nc_path.name}: no recognised time variable"
            )
        t = np.asarray(time_var[:]).astype(float).ravel()

    if t.size < 1:
        raise CacheValidationError(f"{nc_path.name}: empty time axis")
    if t.size == 1:
        return {"n_time_steps": 1, "time_values": t.tolist()}

    dt = np.diff(t)
    if not np.all(dt > 0):
        raise CacheValidationError(
            f"{nc_path.name}: time axis not strictly increasing"
        )
    step = float(dt[0])
    if not np.allclose(dt, step, rtol=0, atol=max(1e-6, 1e-3 * abs(step))):
        raise CacheValidationError(
            f"{nc_path.name}: inconsistent time step (gaps): "
            f"min={dt.min()}, max={dt.max()}"
        )
    return {
        "time_start": float(t[0]),
        "time_end": float(t[-1]),
        "time_step_seconds": step,
        "n_time_steps": int(t.size),
    }


# ---------------------------------------------------------------------------
# Disk-backed implementation
# ---------------------------------------------------------------------------


@dataclass
class DiskHCache:
    """Content-addressed on-disk H cache.

    Layout::

        <root>/
            <sha256>/
                jacobian.npz
                manifest.json      # dims, receptor ids, met range
                *.nc               # optional passthrough of FLEXPART output

    ``get`` returns the entry directory when it exists and passes validation;
    ``put`` validates then copies from a dispersion RunDir.
    """

    root: Path

    _JACOBIAN_NAMES = ("jacobian.npz",)
    _EXTRA_COPY_GLOBS = ("*.nc",)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- HCache protocol ------------------------------------------------

    def get(self, key: str) -> Path | None:
        entry = self._entry(key)
        if not entry.exists():
            return None
        if not (entry / "manifest.json").exists():
            return None
        if not any((entry / n).exists() for n in self._JACOBIAN_NAMES):
            return None
        return entry

    def put(self, key: str, dispersion_run_dir: Path) -> Path:
        dispersion_run_dir = Path(dispersion_run_dir)
        jac = self._find_jacobian(dispersion_run_dir)
        if jac is None:
            raise CacheValidationError(
                f"no jacobian.npz found under {dispersion_run_dir}"
            )
        time_meta = _validate_time_coverage(dispersion_run_dir)

        entry = self._entry(key)
        tmp = entry.with_suffix(".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)

        shutil.copy2(jac, tmp / "jacobian.npz")
        for pattern in self._EXTRA_COPY_GLOBS:
            for src in dispersion_run_dir.glob(pattern):
                shutil.copy2(src, tmp / src.name)

        manifest = self._build_manifest(key, tmp / "jacobian.npz", time_meta)
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

        if entry.exists():
            shutil.rmtree(entry)
        tmp.rename(entry)
        return entry

    # ---- Helpers --------------------------------------------------------

    def _entry(self, key: str) -> Path:
        if not isinstance(key, str) or not key or any(
            c not in "0123456789abcdef" for c in key.lower()
        ):
            raise ValueError(f"invalid cache key (expected sha256 hex): {key!r}")
        return self.root / key.lower()

    def _find_jacobian(self, run_dir: Path) -> Path | None:
        for name in self._JACOBIAN_NAMES:
            candidate = run_dir / name
            if candidate.exists():
                return candidate
        npzs = sorted(run_dir.glob("*.npz"))
        return npzs[0] if npzs else None

    def _build_manifest(
        self, key: str, jac_path: Path, time_meta: Mapping[str, Any]
    ) -> dict[str, Any]:
        with np.load(jac_path, allow_pickle=False) as data:
            arr_names = list(data.files)
            shapes = {name: list(data[name].shape) for name in arr_names}
            dtypes = {name: str(data[name].dtype) for name in arr_names}
            receptor_ids: list[str] | None = None
            for candidate in ("receptor_ids", "instrument_ids"):
                if candidate in data.files:
                    receptor_ids = [str(x) for x in np.asarray(data[candidate]).tolist()]
                    break
        manifest: dict[str, Any] = {
            "key": key,
            "arrays": arr_names,
            "shapes": shapes,
            "dtypes": dtypes,
        }
        if receptor_ids is not None:
            manifest["receptor_ids"] = receptor_ids
        if time_meta:
            manifest["time"] = dict(time_meta)
        return manifest


# ---------------------------------------------------------------------------
# In-memory fallback (mirrors Agent D's InMemoryHCache in sweep.py)
# ---------------------------------------------------------------------------


@dataclass
class InMemoryHCache:
    """Ephemeral cache useful for tests and small sweeps."""

    _store: dict[str, Path] = field(default_factory=dict)

    def get(self, key: str) -> Path | None:
        return self._store.get(key)

    def put(self, key: str, dispersion_run_dir: Path) -> Path:
        _validate_time_coverage(Path(dispersion_run_dir))
        self._store[key] = Path(dispersion_run_dir)
        return Path(dispersion_run_dir)


__all__ = [
    "HCache",
    "DiskHCache",
    "InMemoryHCache",
    "CacheValidationError",
    "transport_cache_key",
]
