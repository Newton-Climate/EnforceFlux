"""YAML-driven forward FLEXPART simulation for methane transport."""
from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Source types ─────────────────────────────────────────────────────────────

@dataclass
class PointSource:
    """Single-location methane release (e.g. landfill, well pad)."""
    id: str
    lon: float
    lat: float
    alt_m: float
    emission_rate_kg_s: float   # kg s⁻¹ — total emission rate
    start: datetime
    end: datetime
    n_particles: int = 10_000


@dataclass
class DiffuseSource:
    """Area emission discretized into a regular lat/lon grid of FLEXPART releases.

    Typical use: rice paddy fields, wetlands, agricultural zones.
    """
    id: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    alt_m: float
    emission_flux_kg_m2_s: float    # kg m⁻² s⁻¹
    start: datetime
    end: datetime
    cell_size_deg: float = 0.1          # size of each release cell in degrees
    n_particles_per_cell: int = 1_000

    def cells(self) -> list[tuple[float, float, float, float, float]]:
        """Yield (lon1, lat1, lon2, lat2, mass_kg) for each discretized cell."""
        R = 6_371_000.0  # m
        duration = (self.end - self.start).total_seconds()
        result = []
        lon = self.lon_min
        while lon < self.lon_max - 1e-9:
            lon2 = min(lon + self.cell_size_deg, self.lon_max)
            lat = self.lat_min
            while lat < self.lat_max - 1e-9:
                lat2 = min(lat + self.cell_size_deg, self.lat_max)
                lat_c = math.radians((lat + lat2) / 2.0)
                dx = math.radians(lon2 - lon) * R * math.cos(lat_c)
                dy = math.radians(lat2 - lat) * R
                area_m2 = abs(dx * dy)
                mass_kg = self.emission_flux_kg_m2_s * area_m2 * duration
                result.append((lon, lat, lon2, lat2, mass_kg))
                lat += self.cell_size_deg
            lon += self.cell_size_deg
        return result


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class SimulationConfig:
    executable: Path
    options_dir: Path
    available_file: Path
    meteo_dir: Path
    run_dir: Path
    start: datetime
    end: datetime
    output_step_s: int
    domain_lon_min: float
    domain_lat_min: float
    domain_lon_max: float
    domain_lat_max: float
    domain_dx: float
    domain_dy: float
    heights_m: list[float]
    sources: list[PointSource | DiffuseSource]
    output_path: Path
    species_name: str = "CH4"
    species_number: int = 24      # numeric SPECNUM_REL; must match SPECIES_0XX file in options/SPECIES/
    nxshift: int = -9999          # grid shift for global met data; -9999 = FLEXPART auto-detect (359 ECMWF, 0 GFS)
    n_sync_s: int = 900
    output_compress: bool = True
    output_per_source: bool = False
    ldirect: int = 1              # 1 = forward transport; -1 = backward (footprint/Jacobian) mode


# ── YAML loader ───────────────────────────────────────────────────────────────

def _parse_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(s.rstrip("Z"))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def load_simulation_config(yaml_path: str | Path) -> SimulationConfig:
    """Load a :class:`SimulationConfig` from a YAML file.

    Relative paths inside the YAML are resolved against the YAML file's directory.
    """
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: pip install pyyaml") from exc

    yaml_path = Path(yaml_path).resolve()
    data = yaml.safe_load(yaml_path.read_text())
    base = yaml_path.parent

    def _p(s: str) -> Path:
        p = Path(s)
        return p if p.is_absolute() else (base / p).resolve()

    fp = data["flexpart"]
    sim = data["simulation"]
    dom = data["domain"]
    out = data.get("output", {})
    spec = data.get("species", {})

    sources: list[PointSource | DiffuseSource] = []
    for s in data.get("sources", []):
        kind = s["type"].lower()
        t0 = _parse_dt(s.get("start", sim["start"]))
        t1 = _parse_dt(s.get("end", sim["end"]))
        if kind == "point":
            sources.append(PointSource(
                id=str(s["id"]),
                lon=float(s["lon"]),
                lat=float(s["lat"]),
                alt_m=float(s.get("alt_m", 10.0)),
                emission_rate_kg_s=float(s["emission_rate_kg_s"]),
                start=t0,
                end=t1,
                n_particles=int(s.get("n_particles", 10_000)),
            ))
        elif kind == "diffuse":
            sources.append(DiffuseSource(
                id=str(s["id"]),
                lon_min=float(s["lon_min"]),
                lon_max=float(s["lon_max"]),
                lat_min=float(s["lat_min"]),
                lat_max=float(s["lat_max"]),
                alt_m=float(s.get("alt_m", 2.0)),
                emission_flux_kg_m2_s=float(s["emission_flux_kg_m2_s"]),
                start=t0,
                end=t1,
                cell_size_deg=float(s.get("cell_size_deg", 0.1)),
                n_particles_per_cell=int(s.get("n_particles_per_cell", 1_000)),
            ))
        else:
            raise ValueError(f"Unknown source type '{kind}' for source '{s.get('id')}'")

    return SimulationConfig(
        executable=_p(fp["executable"]),
        options_dir=_p(fp["options_dir"]),
        available_file=_p(fp["available_file"]),
        meteo_dir=_p(fp["meteo_dir"]),
        run_dir=_p(fp.get("run_dir", "runs/simulation")),
        start=_parse_dt(sim["start"]),
        end=_parse_dt(sim["end"]),
        output_step_s=int(sim.get("output_step_seconds", 3600)),
        n_sync_s=int(sim.get("sync_seconds", 900)),
        domain_lon_min=float(dom["lon_min"]),
        domain_lat_min=float(dom["lat_min"]),
        domain_lon_max=float(dom["lon_max"]),
        domain_lat_max=float(dom["lat_max"]),
        domain_dx=float(dom.get("dx", 0.1)),
        domain_dy=float(dom.get("dy", 0.1)),
        heights_m=[float(h) for h in dom.get("heights_m", [100.0, 500.0, 1000.0])],
        sources=sources,
        output_path=_p(out.get("path", "outputs/simulation.nc")),
        species_name=str(spec.get("name", "CH4")),
        species_number=int(spec.get("number", 24)),
        nxshift=int(sim.get("nxshift", -9999)),
        output_compress=bool(out.get("compress", True)),
        output_per_source=bool(out.get("per_source", False)),
    )


# ── Main class ────────────────────────────────────────────────────────────────

class FlexpartSimulation:
    """Forward FLEXPART simulation driven by a YAML config.

    Handles point sources and diffuse area sources (e.g. rice paddy emissions),
    runs FLEXPART, and writes a clean NetCDF output file.

    Usage::

        sim = FlexpartSimulation.from_yaml("config.yaml")
        output_path = sim.run()

    Or call :meth:`prepare` to write the FLEXPART input files without running.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FlexpartSimulation":
        """Create a :class:`FlexpartSimulation` from a YAML config file."""
        return cls(load_simulation_config(path))

    # ── Public API ────────────────────────────────────────────────────────────

    def prepare(self) -> Path:
        """Write FLEXPART input files to the run directory without executing.

        Returns the run directory path. Useful for inspecting or modifying
        inputs before running.
        """
        cfg = self.config
        run_dir, options_dir, output_dir, pathnames = self._setup_run_dir()
        self._write_pathnames(pathnames, options_dir, output_dir)
        self._write_command(options_dir / "COMMAND")
        self._write_outgrid(options_dir / "OUTGRID")
        self._write_releases(options_dir / "RELEASES")
        return run_dir

    def run(self) -> Path:
        """Prepare inputs, execute FLEXPART, write output NetCDF.

        Returns the path to the output NetCDF file.
        """
        cfg = self.config
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
        run_dir = cfg.run_dir
        options_dir = run_dir / "options"
        output_dir = run_dir / "output"
        pathnames = run_dir / "pathnames"

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
            "\n".join([
                str(options_dir),
                str(output_dir),
                str(cfg.meteo_dir),
                str(cfg.available_file),
            ]) + "\n"
        )

    def _write_command(self, path: Path) -> None:
        cfg = self.config
        step = cfg.output_step_s
        sync = cfg.n_sync_s
        per_src = 1 if cfg.output_per_source else 0
        # FLEXPART requires IOUTPUTFOREACHRELEASE=1 in backward mode
        if cfg.ldirect == -1:
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
        # SPECNUM_REL is an integer index; FLEXPART opens SPECIES/SPECIES_0XX where XX is that number.
        spec_num = cfg.species_number

        blocks: list[str] = []
        idx = 1
        for src in cfg.sources:
            idate1 = src.start.strftime("%Y%m%d")
            itime1 = src.start.strftime("%H%M%S")
            idate2 = src.end.strftime("%Y%m%d")
            itime2 = src.end.strftime("%H%M%S")
            dur_s = (src.end - src.start).total_seconds()

            if isinstance(src, PointSource):
                mass_kg = src.emission_rate_kg_s * dur_s
                blocks.append(self._release_block(
                    lon1=src.lon, lat1=src.lat,
                    lon2=src.lon, lat2=src.lat,
                    z1=src.alt_m, z2=src.alt_m,
                    idate1=idate1, itime1=itime1,
                    idate2=idate2, itime2=itime2,
                    mass_kg=mass_kg,
                    n_parts=src.n_particles,
                    comment=src.id,
                ))
                idx += 1

            elif isinstance(src, DiffuseSource):
                for (lon1, lat1, lon2, lat2, mass_kg) in src.cells():
                    blocks.append(self._release_block(
                        lon1=lon1, lat1=lat1,
                        lon2=lon2, lat2=lat2,
                        z1=src.alt_m, z2=src.alt_m,
                        idate1=idate1, itime1=itime1,
                        idate2=idate2, itime2=itime2,
                        mass_kg=mass_kg,
                        n_parts=src.n_particles_per_cell,
                        comment=f"{src.id}_{idx}",
                    ))
                    idx += 1

        header = (
            "&RELEASES_CTRL\n"
            f" NSPEC      =           1,\n"
            f" SPECNUM_REL=          {spec_num},\n"
            " /\n"
        )
        path.write_text(header + "\n".join(blocks))

    @staticmethod
    def _release_block(
        lon1: float, lat1: float,
        lon2: float, lat2: float,
        z1: float, z2: float,
        idate1: str, itime1: str,
        idate2: str, itime2: str,
        mass_kg: float,
        n_parts: int,
        comment: str,
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
        # Pass pathnames as its filename only; FLEXPART opens it relative to cwd=run_dir.
        subprocess.run(
            [str(exe), pathnames.name],
            cwd=run_dir,
            check=True,
            env=os.environ.copy(),
        )

    # ── Output NetCDF ─────────────────────────────────────────────────────────

    def _write_output_netcdf(self, output_dir: Path) -> Path:
        try:
            from netCDF4 import Dataset
        except ImportError as exc:
            raise RuntimeError("netCDF4 is required: pip install netCDF4") from exc
        import numpy as np

        # FLEXPART names gridded output grid_time_*.nc (or grid_conc_*.nc)
        candidates = sorted(output_dir.glob("grid_time_*.nc"))
        if not candidates:
            candidates = sorted(output_dir.glob("grid_conc_*.nc"))
        if not candidates:
            candidates = sorted(output_dir.glob("*.nc"))
        if not candidates:
            raise FileNotFoundError(f"No NetCDF output found in {output_dir}")

        cfg = self.config
        out_path = cfg.output_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        zlib = cfg.output_compress
        clevel = 4 if zlib else 0

        with Dataset(candidates[0]) as src:
            with Dataset(out_path, "w", format="NETCDF4") as dst:
                self._build_output(src, dst, np=np, zlib=zlib, clevel=clevel)

        return out_path

    def _build_output(
        self, src: Any, dst: Any, *, np: Any, zlib: bool, clevel: int
    ) -> None:
        cfg = self.config

        dst.title = "FLEXPART forward methane transport simulation"
        dst.species = cfg.species_name
        dst.simulation_start = cfg.start.isoformat()
        dst.simulation_end = cfg.end.isoformat()
        dst.source_ids = ", ".join(s.id for s in cfg.sources)
        dst.n_point_sources = sum(1 for s in cfg.sources if isinstance(s, PointSource))
        dst.n_diffuse_sources = sum(1 for s in cfg.sources if isinstance(s, DiffuseSource))
        dst.Conventions = "CF-1.8"

        # Dimensions
        for name, dim in src.dimensions.items():
            dst.createDimension(name, None if dim.isunlimited() else len(dim))

        # Variables — rename FLEXPART's spec001* to descriptive names
        _rename = {
            "spec001":    "ch4_concentration",
            "spec001_mr": "ch4_mixing_ratio",
        }
        for name, var in src.variables.items():
            out_name = _rename.get(name, name)
            v = dst.createVariable(
                out_name, var.dtype, var.dimensions, zlib=zlib, complevel=clevel
            )
            v[:] = np.asarray(var[:])
            for attr in var.ncattrs():
                v.setncattr(attr, var.getncattr(attr))
            if out_name == "ch4_concentration":
                v.long_name = "CH4 mass concentration"
                v.standard_name = "mass_concentration_of_methane_in_air"
                v.units = getattr(var, "units", "ng m-3")
            elif out_name == "ch4_mixing_ratio":
                v.long_name = "CH4 mass mixing ratio"
                v.units = getattr(var, "units", "ng kg-1")
