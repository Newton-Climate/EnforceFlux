"""MicroHHConfig dataclass and YAML loader for MicroHH LES cases.

The counterpart to :mod:`enforceflux.flexpart.sim_config`. One YAML fully
describes a MicroHH case: the box grid, the wind-aligned projection, the
large-scale forcing, the scalar sources, and the column receptors. The loader
resolves relative paths against the YAML file's directory (same contract as the
FLEXPART loader).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from enforceflux.microhh.sources import MicroHHPointSource, MicroHHReceptor


@dataclass(frozen=True)
class BoxGrid:
    """MicroHH ``[grid]`` block: cell counts and physical extents (m)."""

    itot: int
    jtot: int
    ktot: int
    xsize: float
    ysize: float
    zsize: float

    @property
    def dx(self) -> float:
        return self.xsize / self.itot

    @property
    def dy(self) -> float:
        return self.ysize / self.jtot

    @property
    def dz(self) -> float:
        return self.zsize / self.ktot


@dataclass(frozen=True)
class Forcing:
    """Large-scale forcing / surface roughness for a dry-ABL case.

    Constant geostrophic wind + surface roughness give a self-contained
    idealised run. When ERA5-nudged profiles are wired in later (see
    ``forcing.py``), these become the fallback/initial values.
    """

    u_geo: float = 3.0        # mean wind along box +x / west-inflow (m/s)
    v_geo: float = 0.0        # cross-stream component (m/s)
    z0m: float = 0.1          # momentum roughness length (m)
    z0h: float = 0.01         # scalar roughness length (m)
    thl_surface_K: float = 300.0        # mixed-layer potential temperature (K)
    thl_lapse_K_per_m: float = 0.003    # free-atmosphere stability (K/m)
    boundary_layer_height_m: float = 1000.0
    inversion_strength_K: float = 2.0   # capping-inversion Δθ (K)
    inversion_depth_m: float = 100.0    # inversion thickness (m)
    surface_heat_flux_K_m_s: float = 0.1  # sbot[th], drives convection (K m/s)


@dataclass(frozen=True)
class MicroHHConfig:
    """Everything needed to write and run one MicroHH case."""

    executable: Path
    case_dir: Path
    case_name: str

    grid: BoxGrid
    forcing: Forcing

    # Wind-aligned box projection.
    origin_lon: float
    origin_lat: float
    x_bearing_deg: float
    source_x0: float
    source_y0: float

    sources: list[MicroHHPointSource]
    receptors: list[MicroHHReceptor]

    output_path: Path

    # Timing (seconds).
    start: datetime
    spinup_s: int = 7200
    runtime_s: int = 21600
    # Hard cap on the adaptive timestep (s). The actual step is usually set by
    # the CFL limit, so this only bites in very slow flow; 60 s matches
    # MicroHH's shipped cases.
    dt_max_s: float = 60.0
    sampletime_s: int = 10

    scalar_name: str = "ch4"
    # Multiplier on each source's physical emission_rate_kg_s when writing the
    # MicroHH strength. 1.0 → physically calibrated run (kg/s). Set to a
    # reference value only for unit-response (Jacobian) runs; the scalar is
    # linear so results rescale exactly.
    emission_scale: float = 1.0

    extra_ini: dict = field(default_factory=dict)


# ─── YAML loader ─────────────────────────────────────────────────────────────

def _parse_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(str(s).rstrip("Z"))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def load_microhh_config(yaml_path: str | Path) -> MicroHHConfig:
    """Load a :class:`MicroHHConfig` from a YAML file."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required: pip install pyyaml") from exc

    yaml_path = Path(yaml_path).resolve()
    data = yaml.safe_load(yaml_path.read_text())
    base = yaml_path.parent

    def _p(s: str) -> Path:
        p = Path(s)
        return p if p.is_absolute() else (base / p).resolve()

    mh = data["microhh"]
    sim = data["simulation"]
    g = data["grid"]
    dom = data["domain"]
    frc = data.get("forcing", {})
    spec = data.get("species", {})
    out = data.get("output", {})

    grid = BoxGrid(
        itot=int(g["itot"]), jtot=int(g["jtot"]), ktot=int(g["ktot"]),
        xsize=float(g["xsize"]), ysize=float(g["ysize"]), zsize=float(g["zsize"]),
    )

    forcing = Forcing(
        u_geo=float(frc.get("u_geo", 3.0)),
        v_geo=float(frc.get("v_geo", 0.0)),
        z0m=float(frc.get("z0m", 0.1)),
        z0h=float(frc.get("z0h", 0.01)),
        thl_surface_K=float(frc.get("thl_surface_K", 300.0)),
        thl_lapse_K_per_m=float(frc.get("thl_lapse_K_per_m", 0.003)),
        boundary_layer_height_m=float(frc.get("boundary_layer_height_m", 1000.0)),
        inversion_strength_K=float(frc.get("inversion_strength_K", 2.0)),
        inversion_depth_m=float(frc.get("inversion_depth_m", 100.0)),
        surface_heat_flux_K_m_s=float(frc.get("surface_heat_flux_K_m_s", 0.1)),
    )

    # Default source placement: near the upwind edge, centred cross-stream.
    source_x0 = float(dom.get("source_x0", 0.15 * grid.xsize))
    source_y0 = float(dom.get("source_y0", 0.5 * grid.ysize))

    sources = [
        MicroHHPointSource(
            id=str(s["id"]),
            lon=float(s["lon"]), lat=float(s["lat"]),
            alt_m=float(s.get("alt_m", 5.0)),
            emission_rate_kg_s=float(s["emission_rate_kg_s"]),
            sigma_x_m=float(s.get("sigma_x_m", 15.0)),
            sigma_y_m=float(s.get("sigma_y_m", 15.0)),
            sigma_z_m=float(s.get("sigma_z_m", 5.0)),
        )
        for s in data.get("sources", [])
    ]

    receptors = [
        MicroHHReceptor(
            id=str(r["id"]),
            lon=float(r["lon"]), lat=float(r["lat"]),
            alt_m=float(r.get("alt_m", 10.0)),
        )
        for r in data.get("instruments", data.get("receptors", []))
    ]

    case_name = str(sim.get("name", "case"))

    return MicroHHConfig(
        executable=_p(mh["executable"]),
        case_dir=_p(mh.get("case_dir", f"runs/microhh/{case_name}")),
        case_name=case_name,
        grid=grid,
        forcing=forcing,
        origin_lon=float(dom["origin_lon"]),
        origin_lat=float(dom["origin_lat"]),
        x_bearing_deg=float(dom.get("x_bearing_deg", 0.0)),
        source_x0=source_x0,
        source_y0=source_y0,
        sources=sources,
        receptors=receptors,
        output_path=_p(out.get("path", f"runs/microhh/{case_name}/{case_name}.column.nc")),
        start=_parse_dt(sim["start"]),
        spinup_s=int(sim.get("spinup_seconds", 7200)),
        runtime_s=int(sim.get("runtime_seconds", 21600)),
        dt_max_s=float(sim.get("dt_max", 60.0)),
        sampletime_s=int(sim.get("sampletime", 10)),
        scalar_name=str(spec.get("name", "ch4")),
        emission_scale=float(spec.get("emission_scale", 1.0)),
        extra_ini=dict(data.get("extra_ini", {})),
    )
