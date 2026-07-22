"""Run a MicroHH case: write inputs, then invoke ``microhh init`` / ``microhh run``.

MicroHH's CLI workflow is::

    microhh init <case_name>    # build grid + restart files from .ini/.prof
    microhh run  <case_name>    # integrate

Both are invoked from the case directory. This runner owns the subprocess
plumbing and a ``dry_run`` path that writes inputs without executing — which is
the only path available until the MicroHH binary is cloned and compiled.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from enforceflux.microhh.case import write_case
from enforceflux.microhh.sim_config import MicroHHConfig


@dataclass(frozen=True)
class MicroHHRunResult:
    case_dir: Path
    ini_path: Path
    input_nc_path: Path
    output_path: Path | None
    executed: bool
    meta: dict


class MicroHHRunner:
    """Prepares and (optionally) executes a MicroHH case."""

    def __init__(self, config: MicroHHConfig) -> None:
        self.config = config

    def prepare(self) -> dict[str, Path]:
        """Write ``.ini`` and ``<case>_input.nc`` into the case directory."""
        return write_case(self.config)

    def clean_outputs(self) -> None:
        """Remove prior restart/output artifacts so ``init`` can re-run.

        MicroHH refuses to overwrite restart files (``grid.0000000`` etc.), so a
        re-run must start from a clean directory. Keeps the two inputs written
        by :meth:`prepare` (``<case>.ini`` and ``<case>_input.nc``).
        """
        cfg = self.config
        # The 2-D surface-flux BC is an INPUT written by prepare(), not run
        # output — deleting it would make `init` fail on the next run.
        keep = {f"{cfg.case_name}.ini", f"{cfg.case_name}_input.nc",
                f"{cfg.scalar_name}_bot_in.0000000"}
        d = cfg.case_dir
        patterns = (
            "grid.*", "fftwplan.*", "time.*", "*.restart", "*.xy.*", "*.xz.*",
            "*.yz.*", "*.column.*.nc", "*.nc", "rhoref.*", "thermo_basestate.*",
            "*_gradbot.*", "d*dz_mo.*", "u.0*", "v.0*", "w.0*", "th.0*",
            f"{cfg.scalar_name}.0*", "p.0*", "b.0*",
        )
        for pat in patterns:
            for path in glob.glob(str(d / pat)):
                if os.path.basename(path) in keep:
                    continue
                try:
                    os.remove(path)
                except OSError:
                    pass

    def run(self, *, dry_run: bool = False) -> MicroHHRunResult:
        cfg = self.config
        paths = self.prepare()
        meta: dict = {
            "backend": "microhh",
            "case_name": cfg.case_name,
            "grid": (cfg.grid.itot, cfg.grid.jtot, cfg.grid.ktot),
            "n_cells": cfg.grid.itot * cfg.grid.jtot * cfg.grid.ktot,
            "dx_m": cfg.grid.dx,
            "num_workers": cfg.num_workers,
            "decomposition": cfg.decomposition,   # (npx, npy)
        }

        if dry_run:
            meta["prepared"] = True
            return MicroHHRunResult(
                case_dir=cfg.case_dir, ini_path=paths["ini"], input_nc_path=paths["input_nc"],
                output_path=None, executed=False, meta=meta,
            )

        if not cfg.executable.exists():
            raise FileNotFoundError(
                f"MicroHH executable not found at {cfg.executable}. Clone and build "
                "MicroHH (https://github.com/microhh/microhh), then set "
                "microhh.executable in the case YAML. Use dry_run=True to generate "
                "inputs without executing."
            )

        self.clean_outputs()
        self._invoke(["init", cfg.case_name])
        self._invoke(["run", cfg.case_name])
        meta["executed"] = True
        return MicroHHRunResult(
            case_dir=cfg.case_dir, ini_path=paths["ini"], input_nc_path=paths["input_nc"],
            output_path=cfg.output_path, executed=True, meta=meta,
        )

    def _launcher(self) -> list[str]:
        """MPI launcher prefix, empty for a serial run.

        Both ``init`` and ``run`` must use the same rank count: ``init`` writes
        the decomposed restart files that ``run`` reads back.
        """
        n = self.config.num_workers
        if n <= 1:
            return []
        exe = shutil.which("mpirun") or shutil.which("mpiexec")
        if exe is None:
            raise RuntimeError(
                f"num_workers={n} needs an MPI launcher, but neither mpirun nor "
                "mpiexec is on PATH. Install one (brew install open-mpi) and "
                "rebuild MicroHH (the default build is MPI):\n"
                "    make install-microhh\n"
                "or set num_workers: 1 to run serially. Note MicroHH cannot "
                "combine MPI with CUDA."
            )
        return [exe, "-n", str(n)]

    def _invoke(self, args: list[str]) -> None:
        cmd = [*self._launcher(), str(self.config.executable), *args]
        subprocess.run(cmd, cwd=str(self.config.case_dir), check=True)
