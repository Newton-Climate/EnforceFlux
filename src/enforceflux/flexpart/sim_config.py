"""SimulationConfig dataclass and YAML loader for FLEXPART runs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from enforceflux.flexpart.sources import DiffuseSource, PointSource


@dataclass
class SimulationConfig:
    """All parameters needed to prepare and run one FLEXPART simulation."""

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
    species_number: int = 24     # SPECNUM_REL; must match SPECIES_0XX in options/SPECIES/
    nxshift: int = -9999         # -9999 = FLEXPART auto-detect (359 ECMWF, 0 GFS)
    n_sync_s: int = 900
    output_compress: bool = True
    output_per_source: bool = False
    ldirect: int = 1             # +1 = forward transport; -1 = backward (footprint) mode


# ─── YAML loader ─────────────────────────────────────────────────────────────

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

    fp   = data["flexpart"]
    sim  = data["simulation"]
    dom  = data["domain"]
    out  = data.get("output", {})
    spec = data.get("species", {})

    sources: list[PointSource | DiffuseSource] = []
    for s in data.get("sources", []):
        kind = s["type"].lower()
        t0 = _parse_dt(s.get("start", sim["start"]))
        t1 = _parse_dt(s.get("end",   sim["end"]))
        if kind == "point":
            sources.append(PointSource(
                id=str(s["id"]),
                lon=float(s["lon"]), lat=float(s["lat"]),
                alt_m=float(s.get("alt_m", 10.0)),
                emission_rate_kg_s=float(s["emission_rate_kg_s"]),
                start=t0, end=t1,
                n_particles=int(s.get("n_particles", 10_000)),
            ))
        elif kind == "diffuse":
            sources.append(DiffuseSource(
                id=str(s["id"]),
                lon_min=float(s["lon_min"]), lon_max=float(s["lon_max"]),
                lat_min=float(s["lat_min"]), lat_max=float(s["lat_max"]),
                alt_m=float(s.get("alt_m", 2.0)),
                emission_flux_kg_m2_s=float(s["emission_flux_kg_m2_s"]),
                start=t0, end=t1,
                cell_size_deg=float(s.get("cell_size_deg", 0.1)),
                n_particles_per_cell=int(s.get("n_particles_per_cell", 1_000)),
            ))
        else:
            raise ValueError(f"Unknown source type {kind!r} for source {s.get('id')!r}")

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
