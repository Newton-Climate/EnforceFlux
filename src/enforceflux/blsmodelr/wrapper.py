"""Subprocess bridge to bLSmodelR.

Marshals a :class:`BlsRequest` to JSON, invokes ``Rscript run_bls.R``, and
reads the resulting long-form CSV back into a :class:`BlsRunResult`. The R
script is a thin shim (``r/run_bls.R``); this module owns the contract and
is the sole entry point used by downstream framework code.

The wrapper deliberately does *not* discretize source polygons or diagnose
turbulence from LES — those live in ``footprint.py`` and ``met_from_les.py``
(items 3 and 4 in the integration plan) and hand pre-built ``BlsSource`` /
``BlsInterval`` lists to :meth:`BlsWrapper.run`.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

_R_SCRIPT = Path(__file__).with_name("r") / "run_bls.R"
_INSTALL_SCRIPT = Path(__file__).with_name("r") / "install_bls.R"


# ── request/response types ────────────────────────────────────────────────


@dataclass(frozen=True)
class BlsSensor:
    """One point receptor in the bLS local metric frame (metres)."""

    name: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class BlsSource:
    """One source polygon in the bLS local metric frame (metres)."""

    name: str
    polygon_xy: Sequence[tuple[float, float]]


@dataclass(frozen=True)
class BlsInterval:
    """One MOST turbulence window feeding bLS.

    Canonical construction is via
    :func:`enforceflux.blsmodelr.met_from_sonic.interval_from_sonic`, which
    processes a raw sonic timeseries (real or LES-sampled) into these fields.

    Use :meth:`from_ec_stats` when the only available input is a met station's
    already-processed EC output (u*, L, wind, σ) — that path skips the
    sonic-noise and rotation model, so real-data runs constructed this way
    cannot claim the fraternal-twin guarantee an OSSE via ``interval_from_sonic``
    provides.
    """

    id: str
    u_star: float
    L: float
    z0: float
    wind_dir_deg: float
    wind_speed: float
    z_ref: float
    sd_u: float | None = None
    sd_v: float | None = None
    sd_w: float | None = None

    @classmethod
    def from_ec_stats(
        cls,
        *,
        id: str,
        u_star: float,
        L: float,
        z0: float,
        wind_dir_deg: float,
        wind_speed: float,
        z_ref: float,
        sd_u: float | None = None,
        sd_v: float | None = None,
        sd_w: float | None = None,
    ) -> "BlsInterval":
        """Construct from pre-processed EC stats.

        See class docstring for the caveat about fraternal-twin guarantees.
        """
        return cls(
            id=id, u_star=u_star, L=L, z0=z0,
            wind_dir_deg=wind_dir_deg, wind_speed=wind_speed, z_ref=z_ref,
            sd_u=sd_u, sd_v=sd_v, sd_w=sd_w,
        )


@dataclass(frozen=True)
class BlsModelParams:
    n_particles: int = 50_000
    max_traj_s: int = 200
    seed: int = 1


@dataclass(frozen=True)
class BlsRequest:
    sensors: Sequence[BlsSensor]
    sources: Sequence[BlsSource]
    intervals: Sequence[BlsInterval]
    model: BlsModelParams = field(default_factory=BlsModelParams)

    def to_json_dict(self) -> dict:
        return {
            "sensors":   [asdict(s) for s in self.sensors],
            "sources":   [asdict(s) for s in self.sources],
            "intervals": [asdict(i) for i in self.intervals],
            "model":     asdict(self.model),
        }


@dataclass(frozen=True)
class BlsRunResult:
    """Long-form CxE table returned by bLS.

    Columns:
      sensor, source, interval  (str keys matching the request)
      cxe, cxe_se               ((kg m-3) / (kg m-2 s-1))
      n_particles_used          (int)

    Downstream code reshapes ``cxe`` into the (instrument × source) Jacobian
    columns of :class:`~enforceflux.core.base.ForwardModelResult.g`.
    """

    sensor: np.ndarray
    source: np.ndarray
    interval: np.ndarray
    cxe: np.ndarray
    cxe_se: np.ndarray
    n_particles_used: np.ndarray
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.cxe.size)


# ── wrapper ───────────────────────────────────────────────────────────────


class BlsWrapper:
    """Invoke the bLSmodelR R shim as a subprocess.

    Config keys
    -----------
    rscript : str, default "Rscript"
        Path to the Rscript binary.
    workdir : str | Path | None
        Directory for the JSON request and CSV output. A temp dir is used
        when omitted; when supplied, files are kept for inspection.
    dry_run : bool, default False
        Pass ``--dry-run`` to the shim to use its analytic stub instead of
        calling bLSmodelR. Useful for CI and for smoke-testing plumbing on
        machines without R.
    timeout_s : float | None
        Subprocess timeout. ``None`` (default) waits indefinitely.
    env : dict[str, str] | None
        Extra environment variables layered onto ``os.environ``.
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = dict(config or {})
        self.rscript = str(cfg.get("rscript", "Rscript"))
        workdir = cfg.get("workdir")
        self.workdir = Path(workdir) if workdir else None
        self.dry_run = bool(cfg.get("dry_run", False))
        self.timeout_s = cfg.get("timeout_s")
        self.env_extra = dict(cfg.get("env") or {})
        self.script_path = Path(cfg.get("script_path", _R_SCRIPT))
        if not self.script_path.exists():
            raise FileNotFoundError(f"bLS R shim not found: {self.script_path}")

    # ── env checks ───────────────────────────────────────────────────────

    def check_rscript(self) -> str:
        """Return the resolved Rscript path or raise with an actionable message."""
        resolved = shutil.which(self.rscript)
        if resolved is None:
            raise FileNotFoundError(
                f"Rscript executable {self.rscript!r} not on PATH. "
                f"See docs/apps/blsmodelr.md for install steps."
            )
        return resolved

    def install_r_deps(self) -> subprocess.CompletedProcess[str]:
        """Run install_bls.R to install bLSmodelR and its deps into the R env."""
        self.check_rscript()
        return subprocess.run(
            [self.rscript, str(_INSTALL_SCRIPT)],
            check=True,
            capture_output=True,
            text=True,
            env=self._env(),
        )

    # ── main entry point ─────────────────────────────────────────────────

    def run(self, request: BlsRequest) -> BlsRunResult:
        self.check_rscript()
        work = self._open_workdir()
        try:
            cfg_path = work / "request.json"
            out_path = work / "footprints.csv"
            cfg_path.write_text(json.dumps(request.to_json_dict(), indent=2))
            cmd = [self.rscript, str(self.script_path),
                   "--config", str(cfg_path), "--out", str(out_path)]
            if self.dry_run:
                cmd.append("--dry-run")
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.timeout_s, env=self._env(),
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"bLSmodelR shim failed (exit {proc.returncode}).\n"
                    f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                )
            if not out_path.exists():
                raise RuntimeError(
                    f"bLSmodelR shim produced no output CSV at {out_path}.\n"
                    f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                )
            result = _read_csv(out_path)
            return BlsRunResult(
                **result,
                meta={
                    "cmd": cmd,
                    "stdout": proc.stdout,
                    "workdir": str(work),
                    "dry_run": self.dry_run,
                },
            )
        finally:
            self._close_workdir(work)

    # ── internals ────────────────────────────────────────────────────────

    def _open_workdir(self) -> Path:
        if self.workdir is None:
            self._tempdir = tempfile.TemporaryDirectory(prefix="enforceflux_bls_")
            return Path(self._tempdir.name)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._tempdir = None
        return self.workdir

    def _close_workdir(self, work: Path) -> None:
        if getattr(self, "_tempdir", None) is not None:
            self._tempdir.cleanup()
            self._tempdir = None

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.env_extra)
        return env


def _read_csv(path: Path) -> dict:
    sensor, source, interval = [], [], []
    cxe, cxe_se, n_used = [], [], []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        expected = {"sensor", "source", "interval", "cxe", "cxe_se", "n_particles_used"}
        missing = expected - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                f"bLS output CSV missing columns {sorted(missing)}; "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            sensor.append(row["sensor"])
            source.append(row["source"])
            interval.append(row["interval"])
            cxe.append(float(row["cxe"]))
            cxe_se.append(float(row["cxe_se"]))
            n_used.append(int(row["n_particles_used"]))
    return {
        "sensor":   np.array(sensor, dtype=object),
        "source":   np.array(source, dtype=object),
        "interval": np.array(interval, dtype=object),
        "cxe":      np.asarray(cxe, dtype=float),
        "cxe_se":   np.asarray(cxe_se, dtype=float),
        "n_particles_used": np.asarray(n_used, dtype=int),
    }
