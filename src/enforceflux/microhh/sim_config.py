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
class SurfaceFluxPatch:
    """A square area source realised as a 2-D surface-flux boundary condition.

    This is the physically right representation of a field-scale surface
    emitter (a rice paddy, a landfill cap, a wetland). The alternative this
    codebase started with — tiling the area with volumetric Gaussian blobs —
    injects the emission through a layer as deep as the blob's sigma_z, which
    destroys the near-surface vertical gradient that a low-mounted instrument
    actually measures.

    The square is axis-aligned in GEOGRAPHIC metres (east/north of the domain
    origin), so in the wind-aligned LES box it appears rotated. Rasterising
    handles that; do not pre-rotate it.
    """

    id: str
    lon: float
    lat: float
    side_m: float
    emission_rate_kg_s: float

    @property
    def area_m2(self) -> float:
        return self.side_m * self.side_m

    @property
    def flux_kg_m2_s(self) -> float:
        """Uniform surface flux over the patch."""
        return self.emission_rate_kg_s / self.area_m2


@dataclass(frozen=True)
class BoxGrid:
    """MicroHH ``[grid]`` block: cell counts and physical extents (m)."""

    itot: int
    jtot: int
    ktot: int
    xsize: float
    ysize: float
    zsize: float
    # Optional near-surface vertical stretching. With both set, the first cell
    # is `dz_surface_m` thick and cells grow geometrically until they reach
    # `dz_max_m`, then stay uniform. Leave unset for a uniform grid.
    #
    # This exists because a uniform grid that spans a 2 km boundary layer in ~64
    # levels puts its first scalar level ~16 m up — far above the 0.5-3 m where
    # open-path beams and EC towers actually sit, and the surface-layer profile
    # is logarithmic, so that height difference is worth tens of ppb.
    dz_surface_m: float | None = None
    dz_max_m: float | None = None

    @property
    def dx(self) -> float:
        return self.xsize / self.itot

    @property
    def dy(self) -> float:
        return self.ysize / self.jtot

    @property
    def dz(self) -> float:
        """Uniform layer thickness. Only meaningful when not stretched."""
        return self.zsize / self.ktot

    @property
    def stretched(self) -> bool:
        return self.dz_surface_m is not None and self.dz_max_m is not None

    def levels(self):
        """Full-level heights ``z`` (m), uniform or stretched.

        MicroHH reads these from ``<case>_input.nc`` and derives the staggered
        levels as midpoints (``swspatialorder=2``), so any monotonic profile is
        admissible provided the top full level stays below ``zsize``.
        """
        import numpy as np

        if not self.stretched:
            return np.arange(0.5 * self.dz, self.zsize, self.dz)

        dz0, dz_max = float(self.dz_surface_m), float(self.dz_max_m)
        if dz0 <= 0 or dz_max < dz0:
            raise ValueError(
                f"Need 0 < dz_surface_m ({dz0}) <= dz_max_m ({dz_max})."
            )

        def thicknesses(ratio: float):
            dz, out = dz0, []
            for _ in range(self.ktot):
                out.append(dz)
                dz = min(dz * ratio, dz_max)
            return np.asarray(out)

        # Bisect the growth ratio so the layers exactly fill zsize.
        lo, hi = 1.0, 1.5
        if thicknesses(hi).sum() < self.zsize:
            raise ValueError(
                f"ktot={self.ktot} cannot span zsize={self.zsize} m from "
                f"dz_surface_m={dz0} m even at a 1.5x growth ratio. Raise ktot, "
                "raise dz_max_m, or lower zsize."
            )
        if thicknesses(lo).sum() > self.zsize:
            raise ValueError(
                f"ktot={self.ktot} uniform layers of dz_surface_m={dz0} m "
                f"already exceed zsize={self.zsize} m. Lower ktot or dz_surface_m."
            )
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if thicknesses(mid).sum() < self.zsize:
                lo = mid
            else:
                hi = mid
        dz = thicknesses(0.5 * (lo + hi))
        # Full levels sit at layer centres: z_k = sum(dz[:k]) + dz[k]/2.
        return np.cumsum(dz) - 0.5 * dz

    @property
    def growth_ratio(self) -> float:
        """Cell-to-cell growth ratio near the surface (1.0 when uniform)."""
        z = self.levels()
        if not self.stretched or z.size < 3:
            return 1.0
        return float((z[2] - z[1]) / (z[1] - z[0]))


def decompose_workers(num_workers: int, grid: BoxGrid) -> tuple[int, int]:
    """Split ``num_workers`` MPI ranks into a ``(npx, npy)`` decomposition.

    MicroHH decomposes x **and z** by ``npx`` and y by ``npy``, and its
    transposes impose (``src/grid.cxx``)::

        itot % npx == 0        itot % npy == 0
        jtot % npy == 0        jtot % npx == 0   (only when npy > 1)
        ktot % npx == 0

    The ``ktot % npx`` rule is the one that usually bites: a vertical extent
    that is not a multiple of npx fails even when the horizontal grid divides
    cleanly. Among the valid splits the most balanced is chosen, which
    minimises the halo-exchange surface.

    Raises with the grid and the rules when no valid split exists.
    """
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    if num_workers == 1:
        return (1, 1)

    candidates: list[tuple[int, int, int]] = []
    for npx in range(1, num_workers + 1):
        if num_workers % npx:
            continue
        npy = num_workers // npx
        if grid.itot % npx or grid.itot % npy:
            continue
        if grid.jtot % npy:
            continue
        if npy > 1 and grid.jtot % npx:
            continue
        if grid.ktot % npx:
            continue
        candidates.append((abs(npx - npy), npx, npy))

    if not candidates:
        raise ValueError(
            f"num_workers={num_workers} cannot be decomposed for grid "
            f"itot={grid.itot}, jtot={grid.jtot}, ktot={grid.ktot}. MicroHH needs "
            "itot%npx==0, itot%npy==0, jtot%npy==0, ktot%npx==0 (and jtot%npx==0 "
            "when npy>1), where npx*npy==num_workers. Pick a worker count whose "
            "factors divide the grid — note ktot must divide by npx."
        )
    _, npx, npy = min(candidates)
    return npx, npy


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

    # MPI ranks for the run. 1 = serial (the default build). >1 requires a
    # MicroHH built with -DUSEMPI=TRUE and an mpirun/mpiexec on PATH; the
    # runner launches through it. The rank count is decomposed into (npx, npy)
    # by :func:`decompose_workers`, which the grid must divide evenly.
    num_workers: int = 1

    # Cross-section planes, in box coordinates: ``cross_xy_m`` is a height and
    # ``cross_xz_m`` a box-y. Both default to the FIRST source's height and y,
    # which means adding or reordering sources silently moves the slice — set
    # them explicitly whenever two runs' cross-sections must be comparable.
    cross_xy_m: float | None = None
    cross_xz_m: float | None = None

    # Area sources applied as a 2-D bottom boundary condition rather than as
    # volumetric blobs. Both may be used together: a point leak stays a blob,
    # a field becomes a surface flux.
    surface_flux_patches: tuple[SurfaceFluxPatch, ...] = ()

    extra_ini: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Fail at load time, not four minutes into a run.
        decompose_workers(self.num_workers, self.grid)

    @property
    def decomposition(self) -> tuple[int, int]:
        """``(npx, npy)`` MPI decomposition for this run."""
        return decompose_workers(self.num_workers, self.grid)


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
        dz_surface_m=(float(g["dz_surface_m"]) if "dz_surface_m" in g else None),
        dz_max_m=(float(g["dz_max_m"]) if "dz_max_m" in g else None),
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
    cross = data.get("cross", {}) or {}

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
        num_workers=int(mh.get("num_workers", 1)),
        cross_xy_m=(float(cross["xy_m"]) if "xy_m" in cross else None),
        cross_xz_m=(float(cross["xz_m"]) if "xz_m" in cross else None),
        surface_flux_patches=tuple(
            SurfaceFluxPatch(
                id=str(p["id"]), lon=float(p["lon"]), lat=float(p["lat"]),
                side_m=float(p["side_m"]),
                emission_rate_kg_s=float(p["emission_rate_kg_s"]),
            )
            for p in (data.get("surface_flux_patches") or [])
        ),
        extra_ini=dict(data.get("extra_ini", {})),
    )
