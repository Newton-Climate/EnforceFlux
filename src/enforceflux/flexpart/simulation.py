"""YAML-driven FLEXPART simulation: file preparation, execution, output post-processing."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from enforceflux.flexpart.sim_config import SimulationConfig, load_simulation_config
from enforceflux.flexpart.sources import DiffuseSource, PointSource

# Re-export for callers that import from this module directly (backward compat).
__all__ = [
    "FlexpartSimulation",
    "SimulationConfig",
    "load_simulation_config",
    "PointSource",
    "DiffuseSource",
]


class FlexpartSimulation:
    """Forward FLEXPART simulation driven by a YAML config.

    Handles point sources and diffuse area sources, runs FLEXPART, and writes
    a clean CF-1.8 NetCDF output file with descriptive variable names.

    Usage::

        sim = FlexpartSimulation.from_yaml("config.yaml")
        output_path = sim.run()

    Call :meth:`prepare` to write the FLEXPART input files without running.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FlexpartSimulation":
        """Create a :class:`FlexpartSimulation` from a YAML config file."""
        return cls(load_simulation_config(path))

    # ── Public API ────────────────────────────────────────────────────────────

    def prepare(self) -> Path:
        """Write FLEXPART input files without executing. Returns the run directory."""
        run_dir, options_dir, output_dir, pathnames = self._setup_run_dir()
        self._write_pathnames(pathnames, options_dir, output_dir)
        self._write_command(options_dir / "COMMAND")
        self._write_outgrid(options_dir / "OUTGRID")
        self._write_releases(options_dir / "RELEASES")
        return run_dir

    def run(self) -> Path:
        """Prepare inputs, execute FLEXPART, write output NetCDF. Returns output path."""
        run_dir, options_dir, output_dir, pathnames = self._setup_run_dir()
        self._write_pathnames(pathnames, options_dir, output_dir)
        self._write_command(options_dir / "COMMAND")
        self._write_outgrid(options_dir / "OUTGRID")
        self._write_releases(options_dir / "RELEASES")
        self._execute(run_dir, pathnames)
        return self._write_output_netcdf(output_dir)

    # ── Run-dir setup ─────────────────────────────────────────────────────────

    def _setup_run_dir(self) -> tuple[Path, Path, Path, Path]:
        cfg = self.config
        run_dir    = cfg.run_dir
        options_dir = run_dir / "options"
        output_dir  = run_dir / "output"
        pathnames   = run_dir / "pathnames"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        shutil.copytree(cfg.options_dir, options_dir, dirs_exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        return run_dir, options_dir, output_dir, pathnames

    # ── FLEXPART input writers ────────────────────────────────────────────────

    def _write_pathnames(self, path: Path, options_dir: Path, output_dir: Path) -> None:
        cfg = self.config
        path.write_text(
            "\n".join([str(options_dir), str(output_dir),
                       str(cfg.meteo_dir), str(cfg.available_file)]) + "\n"
        )

    def _write_command(self, path: Path) -> None:
        cfg = self.config
        step    = cfg.output_step_s
        sync    = cfg.n_sync_s
        per_src = 1 if cfg.output_per_source else 0
        if cfg.ldirect == -1:       # FLEXPART requires this for backward runs
            per_src = 1
        path.write_text(
            "&COMMAND\n"
            f" LDIRECT=               {cfg.ldirect},\n"
            f" IBDATE=         {cfg.start.strftime('%Y%m%d')},\n"
            f" IBTIME=           {cfg.start.strftime('%H%M%S')},\n"
            f" IEDATE=         {cfg.end.strftime('%Y%m%d')},\n"
            f" IETIME=           {cfg.end.strftime('%H%M%S')},\n"
            f" LOUTSTEP=           {step},\n"
            f" LOUTAVER=           {step},\n"
            f" LOUTSAMPLE=          {min(step, sync)},\n"
            f" LOUTRESTART=       86400,\n"
            f" LRECOUTSTEP=        {step},\n"
            f" LRECOUTAVER=        {step},\n"
            f" LRECOUTSAMPLE=      {min(step, sync)},\n"
            f" LSYNCTIME=           {sync},\n"
            f" CTL=          -5.0000000,\n"
            f" IFINE=                 4,\n"
            f" IOUT=                  9,\n"
            f" IPOUT=                 0,\n"
            f" LSUBGRID=              0,\n"
            f" LCONVECTION=           1,\n"
            f" LTURBULENCE=           1,\n"
            f" LTURBULENCE_MESO=      0,\n"
            f" LAGESPECTRA=           0,\n"
            f" IPIN=                  0,\n"
            f" IOUTPUTFOREACHRELEASE= {per_src},\n"
            f" IFLUX=                 0,\n"
            f" MDOMAINFILL=           0,\n"
            f" IND_SOURCE=            1,\n"
            f" IND_RECEPTOR=          1,\n"
            f" MQUASILAG=             0,\n"
            f" NESTED_OUTPUT=         0,\n"
            f" LNETCDFOUT=            1,\n"
            f" LINIT_COND=            0,\n"
            f" SFC_ONLY=              0,\n"
            f" CBLFLAG=               0,\n"
            f" NXSHIFT=             {cfg.nxshift},\n"
            f" MAXTHREADGRID=         1,\n"
            f" MAXFILESIZE=       10000,\n"
            f" LOGVERTINTERP=         0,\n"
            f" LCMOUTPUT=             0,\n"
            f" /\n"
        )

    def _write_outgrid(self, path: Path) -> None:
        cfg = self.config
        numx = max(1, round((cfg.domain_lon_max - cfg.domain_lon_min) / cfg.domain_dx))
        numy = max(1, round((cfg.domain_lat_max - cfg.domain_lat_min) / cfg.domain_dy))
        heights_str = ", ".join(f"{h:.1f}" for h in cfg.heights_m)
        path.write_text(
            "&OUTGRID\n"
            f" OUTLON0=    {cfg.domain_lon_min:.4f},\n"
            f" OUTLAT0=     {cfg.domain_lat_min:.4f},\n"
            f" NUMXGRID=       {numx},\n"
            f" NUMYGRID=       {numy},\n"
            f" DXOUT=        {cfg.domain_dx:.4f},\n"
            f" DYOUT=        {cfg.domain_dy:.4f},\n"
            f" OUTHEIGHTS=  {heights_str},\n"
            f" /\n"
        )

    def _write_releases(self, path: Path) -> None:
        cfg = self.config
        blocks: list[str] = []
        idx = 1
        for src in cfg.sources:
            idate1 = src.start.strftime("%Y%m%d")
            itime1 = src.start.strftime("%H%M%S")
            idate2 = src.end.strftime("%Y%m%d")
            itime2 = src.end.strftime("%H%M%S")
            dur_s  = (src.end - src.start).total_seconds()
            if isinstance(src, PointSource):
                blocks.append(self._release_block(
                    lon1=src.lon, lat1=src.lat, lon2=src.lon, lat2=src.lat,
                    z1=src.alt_m, z2=src.alt_m,
                    idate1=idate1, itime1=itime1, idate2=idate2, itime2=itime2,
                    mass_kg=src.emission_rate_kg_s * dur_s,
                    n_parts=src.n_particles, comment=src.id,
                ))
                idx += 1
            elif isinstance(src, DiffuseSource):
                for (lon1, lat1, lon2, lat2, mass_kg) in src.cells():
                    blocks.append(self._release_block(
                        lon1=lon1, lat1=lat1, lon2=lon2, lat2=lat2,
                        z1=src.alt_m, z2=src.alt_m,
                        idate1=idate1, itime1=itime1, idate2=idate2, itime2=itime2,
                        mass_kg=mass_kg, n_parts=src.n_particles_per_cell,
                        comment=f"{src.id}_{idx}",
                    ))
                    idx += 1
        header = (
            "&RELEASES_CTRL\n"
            f" NSPEC      =           1,\n"
            f" SPECNUM_REL=          {cfg.species_number},\n"
            " /\n"
        )
        path.write_text(header + "\n".join(blocks))

    @staticmethod
    def _release_block(
        lon1: float, lat1: float, lon2: float, lat2: float,
        z1: float, z2: float,
        idate1: str, itime1: str, idate2: str, itime2: str,
        mass_kg: float, n_parts: int, comment: str,
    ) -> str:
        return (
            "&RELEASE\n"
            f" IDATE1  =       {idate1},\n"
            f" ITIME1  =         {itime1},\n"
            f" IDATE2  =       {idate2},\n"
            f" ITIME2  =         {itime2},\n"
            f" LON1    =        {lon1:.6f},\n"
            f" LON2    =        {lon2:.6f},\n"
            f" LAT1    =        {lat1:.6f},\n"
            f" LAT2    =        {lat2:.6f},\n"
            f" Z1      =        {z1:.3f},\n"
            f" Z2      =        {z2:.3f},\n"
            f" ZKIND   =              1,\n"
            f" MASS    =       {mass_kg:.6E},\n"
            f" PARTS   =          {n_parts},\n"
            f" COMMENT =    \"{comment[:16]}\",\n"
            f" /\n"
        )

    # ── Execution ─────────────────────────────────────────────────────────────

    def _execute(self, run_dir: Path, pathnames: Path) -> None:
        exe = self.config.executable
        if not exe.exists():
            raise FileNotFoundError(f"FLEXPART executable not found: {exe}")
        subprocess.run(
            [str(exe), pathnames.name],   # filename only: FLEXPART opens relative to cwd
            cwd=run_dir, check=True, env=os.environ.copy(),
        )

    # ── Output NetCDF post-processing ─────────────────────────────────────────

    def _write_output_netcdf(self, output_dir: Path) -> Path:
        try:
            from netCDF4 import Dataset
        except ImportError as exc:
            raise RuntimeError("netCDF4 is required: pip install netCDF4") from exc
        import numpy as np

        for pattern in ("grid_time_*.nc", "grid_conc_*.nc", "*.nc"):
            candidates = sorted(output_dir.glob(pattern))
            if candidates:
                break
        else:
            raise FileNotFoundError(f"No NetCDF output found in {output_dir}")

        cfg = self.config
        out_path = cfg.output_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        zlib   = cfg.output_compress
        clevel = 4 if zlib else 0
        with Dataset(candidates[0]) as src:
            with Dataset(out_path, "w", format="NETCDF4") as dst:
                self._build_output(src, dst, np=np, zlib=zlib, clevel=clevel)
        return out_path

    def _build_output(self, src: Any, dst: Any, *, np: Any, zlib: bool, clevel: int) -> None:
        cfg = self.config
        dst.title            = "FLEXPART forward methane transport simulation"
        dst.species          = cfg.species_name
        dst.simulation_start = cfg.start.isoformat()
        dst.simulation_end   = cfg.end.isoformat()
        dst.source_ids       = ", ".join(s.id for s in cfg.sources)
        dst.n_point_sources  = sum(1 for s in cfg.sources if isinstance(s, PointSource))
        dst.n_diffuse_sources = sum(1 for s in cfg.sources if isinstance(s, DiffuseSource))
        dst.Conventions      = "CF-1.8"

        for name, dim in src.dimensions.items():
            dst.createDimension(name, None if dim.isunlimited() else len(dim))

        _rename = {"spec001": "ch4_concentration", "spec001_mr": "ch4_mixing_ratio"}
        for name, var in src.variables.items():
            out_name = _rename.get(name, name)
            v = dst.createVariable(out_name, var.dtype, var.dimensions,
                                   zlib=zlib, complevel=clevel)
            v[:] = np.asarray(var[:])
            for attr in var.ncattrs():
                v.setncattr(attr, var.getncattr(attr))
            if out_name == "ch4_concentration":
                v.long_name     = "CH4 mass concentration"
                v.standard_name = "mass_concentration_of_methane_in_air"
                v.units         = getattr(var, "units", "ng m-3")
            elif out_name == "ch4_mixing_ratio":
                v.long_name = "CH4 mass mixing ratio"
                v.units     = getattr(var, "units", "ng kg-1")
