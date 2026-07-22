"""
Experiment: Open-Path vs Eddy Covariance for diffuse methane sources.

Science question
----------------
For a spatially distributed (diffuse) source field — modelling a rice-paddy
complex, wetland, or landfill — which instrument network better constrains
individual source cells?

  OP  — open-path sensors  → path-integrated concentration [ppm]
  EC  — eddy-covariance towers  → turbulent surface flux [nmol m⁻² s⁻¹]

The two observable types differ in:
  • Spatial sensitivity: EC responds to the turbulent flux footprint upwind of
    the tower; OP path-averages concentration along a 1 km crosswind beam
    (9 Gaussian-weighted receptor samples per beam).
  • Error budget: OP has 3 ppb instrument precision with a noise-limited
    detection threshold for enhancements (DL=0 above background); EC detects
    ~2 nmol m⁻² s⁻¹.  Both observable types additionally carry a 20 %
    transport/representation error added in quadrature.

Scenario
--------
8 source cells in a 2×4 grid (±600 m × ±250 m), emitting 2–4 kg CH₄ hr⁻¹ each.
Prior is 3 ± 3 kg hr⁻¹ per cell (uncertain, slightly over-estimated in the east).

Analysis
--------
1. Run a forward FLEXPART simulation with one unit-emission release per source.
2. Sample the time-dependent concentration field at shared OP/EC locations.
3. Build cadence-aware synthetic observations for OP, EC, and combined networks.
4. Fisher Information, DFS, averaging kernel, and OE inversion diagnostics.
5. Network-scaling study: grow each network one instrument at a time (centre-out
   along the downwind arc) and find the minimum count whose mean posterior
   uncertainty reduction reaches 50 %.
6. Publication-ready figures saved to  runs/diffuse_source_experiment/.

Instruments sit on a downwind arc (radius 1 km) rather than a full ring: the
ERA5 10-m winds for the simulation window blow from the SE toward the NW
(Delta-breeze up-valley flow), so the arc is centred on that outflow direction
where the plume actually goes.

Run from repo root:
    python examples/diffuse_source_comparison.py
"""
import dataclasses
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")

from enforceflux.instrument import INSTRUMENT_DB, Instrument, InstrumentOperator
from enforceflux.flexpart import (
    FlexpartSimulation,
    PointSource,
    SimulationConfig,
    build_ec_observation_operator_from_backward_runs,
)
from enforceflux.inversion import oe_from_linear
from enforceflux.analysis import (
    analyze_information_content,
    run_ablation_study,
    summarize_ablation,
    plot_averaging_kernel,
    plot_dfs_per_source,
    plot_posterior_uncertainty,
    plot_flux_comparison,
    plot_ablation_comparison,
    plot_inversion_summary,
    plot_forward_operator,
    plot_eigenspectrum,
)
from enforceflux.models.config import DomainConfig
from enforceflux.models.source import Source

OUTPUT_DIR = REPO_ROOT / "runs" / "diffuse_source_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FLEXPART_OUTPUT_NC = OUTPUT_DIR / "flexpart_unit_sources.nc"
FLEXPART_RUN_DIR = REPO_ROOT / "runs" / "dsc_fp_run"

RNG = np.random.default_rng(42)

# ── Unit conversions ──────────────────────────────────────────────────────────

KG_HR_TO_KG_S = 1.0 / 3600.0
CH4_NMOL_PER_KG = 1e9 / 16.04e-3        # nmol per kg CH4
CH4_NG_M3_PER_PPM = 655_800.0           # ng m⁻³ per ppm (25 °C, 1 atm)

# ── Sources: 8 cells in a 2×4 grid ───────────────────────────────────────────

SRC_X = np.array([-600, -200,  200,  600, -600, -200,  200,  600], dtype=float)
SRC_Y = np.array([-250, -250, -250, -250,  250,  250,  250,  250], dtype=float)
N_SRC = len(SRC_X)

# True fluxes [kg/hr]: western cells strongest (e.g. a seep zone on the west side)
Q_TRUE_KG_HR = np.array([4.0, 3.5, 3.0, 2.5,
                          3.5, 3.0, 2.5, 2.0])
Q_TRUE_KG_S  = Q_TRUE_KG_HR * KG_HR_TO_KG_S

# Prior: uniform 3 kg/hr ± 3 kg/hr (1-sigma in log-space would be ≈100 %)
Q_PRIOR_KG_S     = np.full(N_SRC, 3.0 * KG_HR_TO_KG_S)
Q_PRIOR_STD_KG_S = np.full(N_SRC, 3.0 * KG_HR_TO_KG_S)

SRC_NAMES = [f"S{i+1}" for i in range(N_SRC)]

# ── Instrument placement ──────────────────────────────────────────────────────

# Mean ERA5 10-m wind at the site over the simulation window is
# (u, v) ≈ (-0.8, +0.8) m/s: flow from the SE toward the NW.  Instruments go
# on a downwind arc centred on that outflow bearing.
DOWNWIND_DIR_DEG = 133.0            # transport direction, CCW from east (math convention)
ARC_HALF_WIDTH_DEG = 75.0           # arc spans downwind bearing ± this
SENSOR_RADIUS_M = 1000.0

# 12 candidate slots on the arc, ordered centre-out so the first n slots are
# always a sensible n-instrument network for the scaling study.
N_SLOTS = 12
_offsets = np.linspace(-ARC_HALF_WIDTH_DEG, ARC_HALF_WIDTH_DEG, N_SLOTS)
_slot_order = np.argsort(np.abs(_offsets), kind="stable")
_slot_angles = np.radians(DOWNWIND_DIR_DEG + _offsets[_slot_order])
SLOT_X = SENSOR_RADIUS_M * np.cos(_slot_angles)
SLOT_Y = SENSOR_RADIUS_M * np.sin(_slot_angles)

# OP sampling of the forward field is free, so OP uses every slot; each EC
# tower costs 48 backward FLEXPART runs, so EC keeps the 6 innermost slots.
N_OP = N_SLOTS
N_EC = 6
OP_X = SLOT_X.copy()          # beam midpoints
OP_Y = SLOT_Y.copy()
EC_X = SLOT_X[:N_EC].copy()
EC_Y = SLOT_Y[:N_EC].copy()

# OP beams: 1 km paths oriented crosswind (perpendicular to the mean transport
# direction) and centred on each slot, represented by N_PATH_PTS receptor
# samples that the line-integral operator Gaussian-weights along the beam.
OP_PATH_LENGTH_M = 1000.0
OP_PATH_BEARING_DEG = ((90.0 - DOWNWIND_DIR_DEG) + 90.0) % 360.0   # compass, crosswind
N_PATH_PTS = 9
_beam_rad = np.radians(OP_PATH_BEARING_DEG)
_beam_ux, _beam_uy = np.sin(_beam_rad), np.cos(_beam_rad)
OP_ANCHOR_X = OP_X - 0.5 * OP_PATH_LENGTH_M * _beam_ux    # beam start points
OP_ANCHOR_Y = OP_Y - 0.5 * OP_PATH_LENGTH_M * _beam_uy
_beam_frac = np.linspace(0.0, 1.0, N_PATH_PTS)
OP_REC_X = (OP_ANCHOR_X[:, None] + OP_PATH_LENGTH_M * _beam_frac[None, :] * _beam_ux).ravel()
OP_REC_Y = (OP_ANCHOR_Y[:, None] + OP_PATH_LENGTH_M * _beam_frac[None, :] * _beam_uy).ravel()
OP_RECEPTOR_MAP = [
    [i * N_PATH_PTS + k for k in range(N_PATH_PTS)] for i in range(N_OP)
]
SITE_LON0 = -121.75
SITE_LAT0 = 39.15
SIM_START = datetime(2020, 7, 1, 0, 0, tzinfo=timezone.utc)
SIM_END = datetime(2020, 7, 2, 0, 0, tzinfo=timezone.utc)
FLEXPART_METEO_DIR = REPO_ROOT / "runs" / "sacramento_valley_2020" / "meteo_july"
FLEXPART_AVAILABLE = FLEXPART_METEO_DIR / "AVAILABLE"
FLEXPART_EXECUTABLE = REPO_ROOT / "flexpart" / "src" / "FLEXPART"
FLEXPART_OPTIONS_DIR = REPO_ROOT / "flexpart" / "tests" / "default_options"
FLEXPART_OPTIONS_TEMPLATE_DIR = REPO_ROOT / "runs" / "dsc_fp_opts"
FLEXPART_OUTPUT_STEP_S = 1800
BASE_TIME_STEP_S = 60.0
SOURCE_AREA_M2 = 400.0 * 500.0


def _local_xy_to_lonlat(
    x_m: np.ndarray,
    y_m: np.ndarray,
    lon0: float = SITE_LON0,
    lat0: float = SITE_LAT0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert small local Cartesian offsets to lon/lat around a site centre."""
    earth_radius_m = 6_371_000.0
    lat = lat0 + np.degrees(y_m / earth_radius_m)
    lon = lon0 + np.degrees(x_m / (earth_radius_m * np.cos(np.radians(lat0))))
    return lon, lat


SRC_LON, SRC_LAT = _local_xy_to_lonlat(SRC_X, SRC_Y)
OP_LON, OP_LAT = _local_xy_to_lonlat(OP_X, OP_Y)
OP_REC_LON, OP_REC_LAT = _local_xy_to_lonlat(OP_REC_X, OP_REC_Y)
EC_LON, EC_LAT = _local_xy_to_lonlat(EC_X, EC_Y)


# ── Observation error covariance ──────────────────────────────────────────────

# The OSSE observable is enhancement above background, so the OP detection
# limit is noise-limited (~σ), not the 40 ppb absolute-concentration threshold
# in the literature entry: a 3 ppb-precision path average resolves any
# enhancement once the background is differenced away.
OP_DB = dataclasses.replace(INSTRUMENT_DB["OP"]["good"], detection_limit=0.0)
EC_DB = INSTRUMENT_DB["EC"]["good"]

# Transport/representation error, as a fraction of the clean signal, added in
# quadrature to instrument noise for BOTH observable types.  For OP this
# dominates the 3 ppb instrument precision; for EC it stands in for
# footprint-model error.  Applied to Se and to the synthetic y_obs draws so
# the data are consistent with the assumed error budget.
REP_ERR_FRAC = 0.20

SA = Q_PRIOR_STD_KG_S ** 2                         # (N_SRC,) (kg/s)²


def _build_instruments() -> tuple[list[Instrument], list[Instrument]]:
    """Create co-located OP and EC instruments for the shared simulator."""
    op_instruments = [
        Instrument(
            id=f"OP{i+1}", tech_id="OP", x=float(x), y=float(y),
            path_length_m=OP_PATH_LENGTH_M,
            path_bearing_deg=OP_PATH_BEARING_DEG,
            params_override=OP_DB,
        )
        for i, (x, y) in enumerate(zip(OP_ANCHOR_X, OP_ANCHOR_Y))
    ]
    ec_instruments = [
        Instrument(id=f"EC{i+1}", tech_id="EC", x=float(lon), y=float(lat), z=3.0)
        for i, (lon, lat) in enumerate(zip(EC_LON, EC_LAT))
    ]
    return op_instruments, ec_instruments


def _build_ec_sources() -> tuple[list[Source], np.ndarray]:
    """Create source support for the EC backward-footprint operator."""
    sources = [
        Source(
            id=name,
            kind="point",
            x=float(lon),
            y=float(lat),
            flux_true=float(q_true),
            flux_prior_mean=float(q_prior),
            flux_prior_std=float(q_std),
        )
        for name, lon, lat, q_true, q_prior, q_std in zip(
            SRC_NAMES, SRC_LON, SRC_LAT, Q_TRUE_KG_S, Q_PRIOR_KG_S, Q_PRIOR_STD_KG_S
        )
    ]
    return sources, np.full(len(sources), SOURCE_AREA_M2, dtype=float)


def _build_wgs84_domain() -> DomainConfig:
    margin_deg = 0.02
    lon_all = np.concatenate([SRC_LON, OP_LON, EC_LON])
    lat_all = np.concatenate([SRC_LAT, OP_LAT, EC_LAT])
    return DomainConfig(
        x_min=float(lon_all.min() - margin_deg),
        x_max=float(lon_all.max() + margin_deg),
        y_min=float(lat_all.min() - margin_deg),
        y_max=float(lat_all.max() + margin_deg),
        grid_spacing=0.005,
        crs="EPSG:4326",
        crs_wgs84="EPSG:4326",
    )


def _build_flexpart_config() -> SimulationConfig:
    """Forward FLEXPART setup with one unit-emission point release per source."""
    if not FLEXPART_EXECUTABLE.exists():
        raise FileNotFoundError(f"FLEXPART executable not found: {FLEXPART_EXECUTABLE}")
    if not FLEXPART_AVAILABLE.exists():
        raise FileNotFoundError(f"AVAILABLE file not found: {FLEXPART_AVAILABLE}")

    lon_all = np.concatenate([SRC_LON, OP_LON, EC_LON])
    lat_all = np.concatenate([SRC_LAT, OP_LAT, EC_LAT])
    margin_deg = 0.02

    sources = [
        PointSource(
            id=name,
            lon=float(lon),
            lat=float(lat),
            alt_m=2.0,
            emission_rate_kg_s=1.0,
            start=SIM_START,
            end=SIM_END,
            n_particles=50_000,
        )
        for name, lon, lat in zip(SRC_NAMES, SRC_LON, SRC_LAT)
    ]

    return SimulationConfig(
        executable=FLEXPART_EXECUTABLE,
        options_dir=_prepare_flexpart_options_template(),
        available_file=FLEXPART_AVAILABLE,
        meteo_dir=FLEXPART_METEO_DIR,
        run_dir=FLEXPART_RUN_DIR,
        start=SIM_START,
        end=SIM_END,
        output_step_s=FLEXPART_OUTPUT_STEP_S,
        domain_lon_min=float(lon_all.min() - margin_deg),
        domain_lat_min=float(lat_all.min() - margin_deg),
        domain_lon_max=float(lon_all.max() + margin_deg),
        domain_lat_max=float(lat_all.max() + margin_deg),
        domain_dx=0.005,
        domain_dy=0.005,
        heights_m=[100.0, 500.0, 1000.0],
        sources=sources,
        output_path=FLEXPART_OUTPUT_NC,
        species_name="CH4",
        species_number=24,
        nxshift=0,
        n_sync_s=900,
        output_compress=True,
        output_per_source=True,
        ldirect=1,
    )


def _prepare_flexpart_options_template() -> Path:
    """Create an example-local FLEXPART options tree with a numbered CH4 species file."""
    species_dir = FLEXPART_OPTIONS_TEMPLATE_DIR / "SPECIES"
    if FLEXPART_OPTIONS_TEMPLATE_DIR.exists():
        shutil.rmtree(FLEXPART_OPTIONS_TEMPLATE_DIR)
    shutil.copytree(FLEXPART_OPTIONS_DIR, FLEXPART_OPTIONS_TEMPLATE_DIR)

    species_target = species_dir / "SPECIES_024"
    if not species_target.exists():
        ch4_template = species_dir / "SPECIES_CH4"
        if not ch4_template.exists():
            raise FileNotFoundError(f"CH4 species template not found: {ch4_template}")
        shutil.copy2(ch4_template, species_target)
    return FLEXPART_OPTIONS_TEMPLATE_DIR


def _forward_output_complete(nc_path: Path) -> bool:
    """True when the forward output covers the full simulation window.

    FLEXPART's NetCDF writer can SIGTRAP mid-run on macOS, leaving a valid
    but truncated file that would otherwise be reused silently.
    """
    expected_span_s = (SIM_END - SIM_START).total_seconds()
    try:
        from netCDF4 import Dataset
        with Dataset(nc_path) as ds:
            if "time" not in ds.dimensions or len(ds.dimensions["time"]) == 0:
                return False
            times = np.asarray(ds["time"][:], dtype=float)
    except Exception:
        return False
    return float(times.max()) >= expected_span_s - FLEXPART_OUTPUT_STEP_S


def _run_flexpart_unit_sources(max_attempts: int = 3) -> Path:
    """Run or reuse the per-source FLEXPART simulation used to build G(t)."""
    if FLEXPART_OUTPUT_NC.exists():
        if _forward_output_complete(FLEXPART_OUTPUT_NC):
            return FLEXPART_OUTPUT_NC
        FLEXPART_OUTPUT_NC.unlink(missing_ok=True)
    for attempt in range(1, max_attempts + 1):
        shutil.rmtree(FLEXPART_RUN_DIR / "output", ignore_errors=True)
        sim = FlexpartSimulation(_build_flexpart_config())
        nc_path = sim.run()
        if _forward_output_complete(nc_path):
            return nc_path
        print(f"  forward FLEXPART output truncated (attempt {attempt}/{max_attempts}); retrying…")
        Path(nc_path).unlink(missing_ok=True)
    raise RuntimeError(
        f"Forward FLEXPART output never covered the full window after {max_attempts} attempts."
    )


def _load_flexpart_op_series() -> tuple[np.ndarray, np.ndarray]:
    """
    Return FLEXPART-derived OP operator time series plus timestamps.

    The field is sampled at every beam receptor point (N_OP × N_PATH_PTS) and
    converted to ppm / (kg/s); the line-integral instrument operator later
    Gaussian-weights the points along each beam into a path average.
    """
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise RuntimeError("netCDF4 is required to read FLEXPART outputs") from exc

    nc_path = _run_flexpart_unit_sources()
    with Dataset(nc_path) as ds:
        if "ch4_mixing_ratio" not in ds.variables:
            raise KeyError(f"No ch4_mixing_ratio variable found in {nc_path}")

        raw = np.asarray(ds["ch4_mixing_ratio"][:], dtype=float)
        # FLEXPART output: (nageclass, pointspec, time, height, lat, lon)
        raw = raw[0, :N_SRC, :, 0, :, :]
        raw = np.transpose(raw, (1, 0, 2, 3))   # (time, source, lat, lon)

        times_s = np.asarray(ds["time"][:], dtype=float)
        lons = np.asarray(ds["longitude"][:], dtype=float)
        lats = np.asarray(ds["latitude"][:], dtype=float)

    op_series_ng = np.zeros((len(times_s), N_OP * N_PATH_PTS, N_SRC), dtype=float)
    for i, (lon, lat) in enumerate(zip(OP_REC_LON, OP_REC_LAT)):
        ix = int(np.argmin(np.abs(lons - lon)))
        iy = int(np.argmin(np.abs(lats - lat)))
        op_series_ng[:, i, :] = raw[:, :, iy, ix]
    g_op_series = op_series_ng / CH4_NG_M3_PER_PPM

    return g_op_series, times_s


def _expand_step_series(
    g_step_series: np.ndarray,
    step_times_s: np.ndarray,
    times_s: np.ndarray,
) -> np.ndarray:
    """Broadcast piecewise-constant transport slices onto the base time grid."""
    step_times = np.asarray(step_times_s, dtype=float)
    target_times = np.asarray(times_s, dtype=float)
    indices = np.searchsorted(step_times, target_times, side="right") - 1
    indices = np.clip(indices, 0, len(step_times) - 1)
    return g_step_series[indices]


def _build_time_grid(step_times_s: np.ndarray) -> np.ndarray:
    """Minute grid spanning the FLEXPART transport window."""
    end_s = float(np.max(step_times_s))
    return np.arange(0.0, end_s + BASE_TIME_STEP_S, BASE_TIME_STEP_S)


def _build_ec_g_series(times_s: np.ndarray, ec_instruments: list[Instrument]) -> np.ndarray:
    """Build EC Jacobian rows from time-dependent backward FLEXPART footprints."""
    sources, source_areas_m2 = _build_ec_sources()
    ec_sample_times_s = times_s[np.isclose(np.mod(times_s - times_s[0], EC_DB.cadence_s), 0.0)]
    ec_operator = build_ec_observation_operator_from_backward_runs(
        base_config=_build_flexpart_config(),
        domain=_build_wgs84_domain(),
        instruments=ec_instruments,
        sources=sources,
        source_areas_m2=source_areas_m2,
        sample_times_s=ec_sample_times_s,
        lookback_s=float(EC_DB.cadence_s),
        runner_config={"cache": True, "surface_only": True, "species_number": 24},
    )
    return _expand_step_series(ec_operator.g, ec_operator.times_s, times_s)


def _flatten_valid_series(
    H_g: np.ndarray,
    y_clean: np.ndarray,
    y_obs: np.ndarray,
    valid_mask: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a time-series simulation to valid observations only."""
    valid_rows = valid_mask.reshape(-1)
    g_flat = H_g.reshape(-1, H_g.shape[-1])[valid_rows]
    y_clean_flat = y_clean.reshape(-1)[valid_rows]
    y_obs_flat = y_obs.reshape(-1)[valid_rows]
    r_flat = np.diagonal(R, axis1=1, axis2=2).reshape(-1)[valid_rows]
    n_time, n_inst = valid_mask.shape
    inst_idx_flat = np.tile(np.arange(n_inst), n_time)[valid_rows]
    return g_flat, y_clean_flat, y_obs_flat, r_flat, inst_idx_flat


# ── Network-scaling study ─────────────────────────────────────────────────────

UR_TARGET = 0.50   # mean uncertainty-reduction target


def _network_scaling_curve(
    g_flat: np.ndarray,
    se_flat: np.ndarray,
    inst_idx: np.ndarray,
    n_max: int,
) -> np.ndarray:
    """Mean uncertainty reduction as instruments are added centre-out along the arc."""
    mean_ur = np.zeros(n_max)
    for n in range(1, n_max + 1):
        sel = inst_idx < n
        if not np.any(sel):
            continue
        _, _, post = analyze_information_content(
            g_flat[sel], se_flat[sel], SA, source_names=SRC_NAMES)
        mean_ur[n - 1] = float(post.uncertainty_reduction.mean())
    return mean_ur


def _instruments_for_target(mean_ur: np.ndarray, target: float = UR_TARGET) -> int | None:
    """Smallest network size whose mean uncertainty reduction meets the target."""
    hits = np.nonzero(mean_ur >= target)[0]
    return int(hits[0]) + 1 if hits.size else None


# ── Main experiment ────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("Open-Path vs Eddy Covariance  |  Diffuse CH₄ Source OSSE")
    print("=" * 70)

    # ── Build FLEXPART-derived G(t) ──────────────────────────────────
    G_op_step, step_times_s = _load_flexpart_op_series()
    times_s = _build_time_grid(step_times_s)
    G_op_series = _expand_step_series(G_op_step, step_times_s, times_s)
    x_true_series = np.repeat(Q_TRUE_KG_S[None, :], len(times_s), axis=0)

    op_instruments, ec_instruments = _build_instruments()
    G_ec_series = _build_ec_g_series(times_s, ec_instruments)
    G_ec_mean = np.nanmean(G_ec_series, axis=0)
    op_result = InstrumentOperator(op_instruments, rng=RNG).simulate_time_series(
        G_op_series, x_true_series, times_s,
        receptor_map=OP_RECEPTOR_MAP,
        receptor_x=OP_REC_X, receptor_y=OP_REC_Y,
    )
    G_op_mean = np.mean(op_result.H_g, axis=0)   # path-integrated operator
    ec_result = InstrumentOperator(ec_instruments, rng=RNG).simulate_time_series(
        G_ec_series, x_true_series, times_s
    )

    G_op_valid, y_clean_op, y_obs_op, Se_op_valid, op_inst_idx = _flatten_valid_series(
        op_result.H_g, op_result.y_clean, op_result.y_obs, op_result.valid_mask, op_result.R
    )
    G_ec_valid, y_clean_ec, y_obs_ec, Se_ec_valid, ec_inst_idx = _flatten_valid_series(
        ec_result.H_g, ec_result.y_clean, ec_result.y_obs, ec_result.valid_mask, ec_result.R
    )

    # Add transport/representation error in quadrature (Se) and to the
    # synthetic observations so data and error budget stay consistent.
    rep_var_op = (REP_ERR_FRAC * y_clean_op) ** 2
    rep_var_ec = (REP_ERR_FRAC * y_clean_ec) ** 2
    Se_op_valid = Se_op_valid + rep_var_op
    Se_ec_valid = Se_ec_valid + rep_var_ec
    y_obs_op = y_obs_op + RNG.normal(0.0, np.sqrt(rep_var_op))
    y_obs_ec = y_obs_ec + RNG.normal(0.0, np.sqrt(rep_var_ec))

    G_all = np.vstack([G_op_valid, G_ec_valid])
    Se_all = np.concatenate([Se_op_valid, Se_ec_valid])

    print(f"\nSources       : {N_SRC} cells, total true = {Q_TRUE_KG_HR.sum():.1f} kg/hr")
    print(f"OP network    : {N_OP} downwind-arc beams ({OP_PATH_LENGTH_M:.0f} m crosswind, "
          f"{N_PATH_PTS} pts), σ_obs = {OP_DB.sigma_abs*1e3:.0f} ppb")
    print(f"EC network    : {N_EC} downwind-arc towers,  σ_obs = {EC_DB.sigma_abs:.1f} nmol m⁻² s⁻¹")
    print(f"Rep. error    : {REP_ERR_FRAC*100:.0f}% of clean signal, both observable types")
    print(f"Prior         : {Q_PRIOR_KG_S[0]*3600:.1f} ± {Q_PRIOR_STD_KG_S[0]*3600:.1f} kg/hr per source")
    dt_s = times_s[1] - times_s[0] if len(times_s) > 1 else BASE_TIME_STEP_S
    print("Transport     : FLEXPART forward OP fields + time-dependent backward EC footprints")
    print(f"Time grid     : {len(times_s)} steps at {dt_s:.0f} s spacing ({times_s[-1]/3600:.1f} hr)")

    # ── Simulate observations ────────────────────────────────────────
    op_sample_slots = np.isfinite(op_result.y_clean).sum()
    ec_sample_slots = np.isfinite(ec_result.y_clean).sum()
    valid_op = op_result.valid_mask
    valid_ec = ec_result.valid_mask

    op_first_idx = np.where(np.isfinite(op_result.y_clean).any(axis=1))[0][0]
    ec_first_idx = np.where(np.isfinite(ec_result.y_clean).any(axis=1))[0][0]
    _op_ppb = " ".join(f"{v*1e3:6.1f}" for v in op_result.y_clean[op_first_idx])
    _ec_sig = " ".join(f"{v:7.1f}" for v in ec_result.y_clean[ec_first_idx])
    print(
        f"\n  OP valid: {valid_op.sum()}/{op_sample_slots} sampled obs"
        f"  (cadence={OP_DB.cadence_s:.0f}s, DL={OP_DB.detection_limit*1e3:.0f} ppb)"
    )
    print(f"    first sampled y_clean [ppb]: [{_op_ppb}]")
    print(
        f"\n  EC valid: {valid_ec.sum()}/{ec_sample_slots} sampled obs"
        f"  (cadence={EC_DB.cadence_s:.0f}s, DL={EC_DB.detection_limit:.1f} nmol m⁻² s⁻¹)"
    )
    print(f"    first sampled y_clean [nmol m⁻² s⁻¹]: [{_ec_sig}]")

    # ── Information analysis ─────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Information analysis (Fisher / DFS / posterior uncertainty)")
    print("─" * 70)

    fisher_op, dof_op, post_op = analyze_information_content(
        G_op_valid, Se_op_valid, SA, source_names=SRC_NAMES)
    fisher_ec, dof_ec, post_ec = analyze_information_content(
        G_ec_valid, Se_ec_valid, SA, source_names=SRC_NAMES)
    obs_groups = {
        "OP": np.arange(len(Se_all)) < len(Se_op_valid),
        "EC": np.arange(len(Se_all)) >= len(Se_op_valid),
    }
    fisher_all, dof_all, post_all = analyze_information_content(
        G_all, Se_all, SA, obs_groups=obs_groups, source_names=SRC_NAMES)

    def _fmt_dfs(label, dof):
        per_src = " ".join(f"{v:.2f}" for v in dof.dfs_per_source)
        return f"  {label:<14} DFS_total={dof.dfs_total:5.2f}   per src: [{per_src}]"

    print(_fmt_dfs("OP only",  dof_op))
    print(_fmt_dfs("EC only",  dof_ec))
    print(_fmt_dfs("OP + EC",  dof_all))

    print(f"\n  Uncertainty reduction (1 − σ_post/σ_prior):")
    print(f"  {'Source':<8} {'OP only':>10} {'EC only':>10} {'Combined':>10}")
    print(f"  {'-'*42}")
    for k, name in enumerate(SRC_NAMES):
        print(f"  {name:<8} "
              f"{post_op.uncertainty_reduction[k]*100:>9.1f}%"
              f"{post_ec.uncertainty_reduction[k]*100:>10.1f}%"
              f"{post_all.uncertainty_reduction[k]*100:>10.1f}%")

    # ── Network-scaling study ────────────────────────────────────────
    print("\n" + "─" * 70)
    print(f"Network scaling (instruments needed for {UR_TARGET*100:.0f}% mean uncertainty reduction)")
    print("─" * 70)

    ur_curve_op = _network_scaling_curve(G_op_valid, Se_op_valid, op_inst_idx, N_OP)
    ur_curve_ec = _network_scaling_curve(G_ec_valid, Se_ec_valid, ec_inst_idx, N_EC)
    n_req_op = _instruments_for_target(ur_curve_op)
    n_req_ec = _instruments_for_target(ur_curve_ec)

    def _fmt_scaling(label, curve, n_req, n_max):
        per_n = " ".join(f"{v*100:5.1f}" for v in curve)
        verdict = (f"target reached with {n_req} instrument(s)" if n_req is not None
                   else f"target NOT reached within {n_max} instruments")
        print(f"  {label} mean UR by network size [%]: [{per_n}]")
        print(f"  {label} → {verdict}")

    _fmt_scaling("OP", ur_curve_op, n_req_op, N_OP)
    _fmt_scaling("EC", ur_curve_ec, n_req_ec, N_EC)

    # ── Ablation study ───────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Ablation study (DFS by scenario)")
    print("─" * 70)

    ablation = run_ablation_study(G_all, Se_all, SA, obs_groups,
                                  source_names=SRC_NAMES)
    table = summarize_ablation(ablation)
    for key, row in table.items():
        print(f"  {key:<20} DFS={row['dfs_total']:.3f}  "
              f"mean UR={row['uncertainty_reduction_mean']*100:.1f}%")

    # ── OE inversions ────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Optimal-estimation inversions")
    print("─" * 70)

    # Use valid observations only; replace G and Se accordingly
    def _inversion(G, Se, y_obs, label):
        valid = ~np.isnan(y_obs)
        if valid.sum() == 0:
            print(f"  {label}: no valid observations, skipping.")
            return None
        result = oe_from_linear(
            G[valid], y_obs[valid], Q_PRIOR_KG_S, SA, Se[valid],
            source_names=SRC_NAMES,
        )
        rmse = float(np.sqrt(np.mean((result.x_posterior - Q_TRUE_KG_S) ** 2))) * 3600
        print(f"  {label:<14}  RMSE={rmse:.3f} kg/hr   "
              f"DFS={np.trace(result.averaging_kernel):.2f}")
        return result

    y_obs_all = np.concatenate([y_obs_op, y_obs_ec])
    res_op  = _inversion(G_op_valid,  Se_op_valid,  y_obs_op,  "OP only")
    res_ec  = _inversion(G_ec_valid,  Se_ec_valid,  y_obs_ec,  "EC only")
    res_all = _inversion(G_all, Se_all, y_obs_all, "OP + EC")

    print(f"\n  True  flux [kg/hr]: {' '.join(f'{v:5.1f}' for v in Q_TRUE_KG_HR)}")
    print(f"  Prior flux [kg/hr]: {' '.join(f'{v:5.1f}' for v in Q_PRIOR_KG_S*3600)}")
    if res_op:
        x_op_hr = res_op.x_posterior * 3600
        print(f"  OP post [kg/hr]:    {' '.join(f'{v:5.1f}' for v in x_op_hr)}")
    if res_ec:
        x_ec_hr = res_ec.x_posterior * 3600
        print(f"  EC post [kg/hr]:    {' '.join(f'{v:5.1f}' for v in x_ec_hr)}")
    if res_all:
        x_all_hr = res_all.x_posterior * 3600
        print(f"  OP+EC post[kg/hr]:  {' '.join(f'{v:5.1f}' for v in x_all_hr)}")

    # ── Visualisations ────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print(f"Saving figures → {OUTPUT_DIR.relative_to(REPO_ROOT)}/")
    print("─" * 70)

    _save_figures(
        G_op_mean, G_ec_mean, G_all, Se_all,
        fisher_op, fisher_ec, fisher_all,
        dof_op, dof_ec, dof_all,
        post_op, post_ec, post_all,
        ablation,
        res_op, res_ec, res_all,
        ur_curve_op, ur_curve_ec, n_req_op, n_req_ec,
    )

    _save_text_summary(dof_op, dof_ec, dof_all, post_op, post_ec, post_all, table,
                       ur_curve_op, ur_curve_ec, n_req_op, n_req_ec)

    print("=" * 70)
    print("Done.")


# ── Figure helpers ─────────────────────────────────────────────────────────────

def _save_fig(fig, name: str) -> None:
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  ✓ {path.name}")


def _save_figures(
    G_op, G_ec, G_all, Se_all,
    fisher_op, fisher_ec, fisher_all,
    dof_op, dof_ec, dof_all,
    post_op, post_ec, post_all,
    ablation,
    res_op, res_ec, res_all,
    ur_curve_op, ur_curve_ec, n_req_op, n_req_ec,
):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    op_labels = [f"OP{i+1}" for i in range(N_OP)]
    ec_labels = [f"EC{i+1}" for i in range(N_EC)]

    # 1. Forward operators side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_forward_operator(G_op * 1e3, SRC_NAMES, op_labels, ax=axes[0],
                          title="OP Mean FLEXPART Operator (ppb / kg·s⁻¹)")
    plot_forward_operator(G_ec / 1e3, SRC_NAMES, ec_labels, ax=axes[1],
                          title="EC Mean FLEXPART Proxy (μmol m⁻² s⁻¹ / kg·s⁻¹)")
    fig.suptitle("Forward Operators", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "01_forward_operators.png")

    # 2. Averaging kernels (3-panel)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_averaging_kernel(dof_op.averaging_kernel,  SRC_NAMES, ax=axes[0], title="Averaging Kernel — OP")
    plot_averaging_kernel(dof_ec.averaging_kernel,  SRC_NAMES, ax=axes[1], title="Averaging Kernel — EC")
    plot_averaging_kernel(dof_all.averaging_kernel, SRC_NAMES, ax=axes[2], title="Averaging Kernel — OP+EC")
    fig.suptitle("Averaging Kernels (ideal retrieval = identity matrix)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "02_averaging_kernels.png")

    # 3. DFS per source (3-panel)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_dfs_per_source(dof_op.dfs_per_source,  SRC_NAMES, ax=axes[0], title="DFS — OP")
    plot_dfs_per_source(dof_ec.dfs_per_source,  SRC_NAMES, ax=axes[1], title="DFS — EC")
    plot_dfs_per_source(dof_all.dfs_per_source, SRC_NAMES, ax=axes[2], title="DFS — OP+EC")
    fig.suptitle(f"Degrees of Freedom for Signal  (n_sources={N_SRC})",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "03_dfs_per_source.png")

    # 4. Posterior uncertainty comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_posterior_uncertainty(post_op.prior_sigma  * 3600, post_op.posterior_sigma  * 3600,
                               SRC_NAMES, ax=axes[0], title="Uncertainty — OP")
    plot_posterior_uncertainty(post_ec.prior_sigma  * 3600, post_ec.posterior_sigma  * 3600,
                               SRC_NAMES, ax=axes[1], title="Uncertainty — EC")
    plot_posterior_uncertainty(post_all.prior_sigma * 3600, post_all.posterior_sigma * 3600,
                               SRC_NAMES, ax=axes[2], title="Uncertainty — OP+EC")
    for ax in axes:
        ax.set_ylabel("Flux uncertainty (kg hr⁻¹)")
    fig.suptitle("Prior vs Posterior Uncertainty", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "04_posterior_uncertainty.png")

    # 5. Ablation comparison
    fig, ax = plt.subplots(figsize=(7, 3.5))
    plot_ablation_comparison(ablation, ax=ax, title="DFS by Observation Scenario")
    _save_fig(fig, "05_ablation_dfs.png")

    # 6. Eigenspectrum comparison
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    plot_eigenspectrum(fisher_op.eigenvalues,  ax=axes[0], title="FIM Eigenspectrum — OP",  color="steelblue")
    plot_eigenspectrum(fisher_ec.eigenvalues,  ax=axes[1], title="FIM Eigenspectrum — EC",  color="darkorange")
    plot_eigenspectrum(fisher_all.eigenvalues, ax=axes[2], title="FIM Eigenspectrum — OP+EC", color="forestgreen")
    fig.suptitle("Fisher Information Matrix Eigenspectra",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "06_eigenspectra.png")

    # 7. Flux comparison (if inversions succeeded)
    if res_all is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_flux_comparison(
            Q_PRIOR_KG_S * 3600,
            res_all.x_posterior * 3600,
            x_true=Q_TRUE_KG_HR,
            source_names=SRC_NAMES,
            posterior_sigma=np.sqrt(np.diag(res_all.posterior_cov)) * 3600,
            ax=ax,
            title="OP+EC Combined: Prior vs Posterior vs True Fluxes (kg hr⁻¹)",
        )
        _save_fig(fig, "07_flux_comparison_combined.png")

    # 8. Inversion summary for combined scenario
    if res_all is not None:
        fig, axes = plot_inversion_summary(
            res_all, fisher=fisher_all, dof=dof_all, posterior=post_all,
            source_names=SRC_NAMES,
        )
        _save_fig(fig, "08_inversion_summary_combined.png")

    # 9. Network-scaling curves
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(ur_curve_op) + 1), ur_curve_op * 100,
            "o-", color="steelblue", label="OP")
    ax.plot(np.arange(1, len(ur_curve_ec) + 1), ur_curve_ec * 100,
            "s-", color="darkorange", label="EC")
    ax.axhline(UR_TARGET * 100, color="crimson", linestyle="--", linewidth=1,
               label=f"{UR_TARGET*100:.0f}% target")
    for n_req, color in ((n_req_op, "steelblue"), (n_req_ec, "darkorange")):
        if n_req is not None:
            ax.axvline(n_req, color=color, linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("Number of instruments (added centre-out along downwind arc)")
    ax.set_ylabel("Mean uncertainty reduction (%)")
    ax.set_title("Network Scaling: Mean Uncertainty Reduction vs Network Size",
                 fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend()
    _save_fig(fig, "09_network_scaling.png")

    # 10. OP vs EC scenario map
    _save_scenario_map()


def _save_scenario_map() -> None:
    """Top-down view of the source field and instrument placement."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(7, 7))

    # Source cells
    sc = ax.scatter(SRC_X, SRC_Y, c=Q_TRUE_KG_HR, s=250, cmap="YlOrRd",
                    vmin=0, zorder=3, edgecolors="black", linewidths=0.8)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("True flux (kg hr⁻¹)")
    for k, (x, y, name) in enumerate(zip(SRC_X, SRC_Y, SRC_NAMES)):
        ax.text(x, y - 80, name, ha="center", va="top", fontsize=7)

    # Downwind sensor arc
    theta_arc = np.radians(DOWNWIND_DIR_DEG
                           + np.linspace(-ARC_HALF_WIDTH_DEG, ARC_HALF_WIDTH_DEG, 100))
    ax.plot(SENSOR_RADIUS_M * np.cos(theta_arc), SENSOR_RADIUS_M * np.sin(theta_arc),
            "--", color="steelblue", linewidth=0.8, alpha=0.5)
    ax.scatter(OP_X, OP_Y, s=100, marker="^", color="steelblue", zorder=4,
               edgecolors="black", linewidths=0.6)
    for i, (x, y) in enumerate(zip(OP_X, OP_Y)):
        ax.text(x * 1.12, y * 1.12, f"OP{i+1}", ha="center", va="center",
                fontsize=7.5, color="steelblue")
        ax.plot(
            [OP_ANCHOR_X[i], OP_ANCHOR_X[i] + OP_PATH_LENGTH_M * _beam_ux],
            [OP_ANCHOR_Y[i], OP_ANCHOR_Y[i] + OP_PATH_LENGTH_M * _beam_uy],
            color="steelblue", linewidth=1.2, alpha=0.6, zorder=3,
        )

    # EC towers (innermost arc slots)
    ax.scatter(EC_X, EC_Y, s=120, marker="s", color="darkorange", zorder=4,
               edgecolors="black", linewidths=0.6)
    for i, (x, y) in enumerate(zip(EC_X, EC_Y)):
        ax.text(x + 70, y + 70, f"EC{i+1}", ha="left", va="bottom",
                fontsize=7.5, color="darkorange")

    # Mean transport direction (plume travels toward the arc)
    wind_dir = np.radians(DOWNWIND_DIR_DEG)
    ax.annotate(
        "", xy=(500 * np.cos(wind_dir), 500 * np.sin(wind_dir)),
        xytext=(-500 * np.cos(wind_dir), -500 * np.sin(wind_dir)),
        arrowprops=dict(arrowstyle="-|>", color="dimgray", linewidth=1.5),
        zorder=2,
    )
    ax.text(650 * np.cos(wind_dir), 650 * np.sin(wind_dir) - 120,
            "mean wind", ha="center", fontsize=8, color="dimgray", style="italic")

    ax.set_xlim(-1600, 1600)
    ax.set_ylim(-1600, 1600)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.set_title("Scenario Layout: Sources + Downwind OP/EC Sensor Arc", fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.4, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.4, alpha=0.5)
    ax.grid(alpha=0.2)

    legend_handles = [
        mpatches.Patch(color="steelblue",  label="OP beam (1 km crosswind)"),
        mpatches.Patch(color="darkorange", label="EC tower (innermost slots)"),
        mpatches.Patch(color="#f4a460",    label="Source cell (colour=flux)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    _save_fig(fig, "00_scenario_map.png")


def _save_text_summary(dof_op, dof_ec, dof_all, post_op, post_ec, post_all, ablation_table,
                       ur_curve_op, ur_curve_ec, n_req_op, n_req_ec):
    lines = [
        "=" * 70,
        "DIFFUSE SOURCE OSSE — RESULT SUMMARY",
        "=" * 70,
        "",
        "Sources",
        f"  N = {N_SRC}, total true = {Q_TRUE_KG_HR.sum():.1f} kg/hr",
        f"  True:  {' '.join(f'{v:.1f}' for v in Q_TRUE_KG_HR)} kg/hr",
        f"  Prior: {Q_PRIOR_KG_S[0]*3600:.1f} ± {Q_PRIOR_STD_KG_S[0]*3600:.1f} kg/hr per cell",
        "",
        "Information content (DFS)",
        f"  OP only : {dof_op.dfs_total:.3f}",
        f"  EC only : {dof_ec.dfs_total:.3f}",
        f"  OP + EC : {dof_all.dfs_total:.3f}",
        "",
        "Mean uncertainty reduction",
        f"  OP only : {post_op.uncertainty_reduction.mean()*100:.1f} %",
        f"  EC only : {post_ec.uncertainty_reduction.mean()*100:.1f} %",
        f"  OP + EC : {post_all.uncertainty_reduction.mean()*100:.1f} %",
        "",
        f"Network scaling (instruments for ≥{UR_TARGET*100:.0f}% mean uncertainty reduction)",
        f"  OP curve [%]: {' '.join(f'{v*100:.1f}' for v in ur_curve_op)}",
        f"  EC curve [%]: {' '.join(f'{v*100:.1f}' for v in ur_curve_ec)}",
        f"  OP needs : {n_req_op if n_req_op is not None else f'> {len(ur_curve_op)} (not reached)'}",
        f"  EC needs : {n_req_ec if n_req_ec is not None else f'> {len(ur_curve_ec)} (not reached)'}",
        "",
        "Ablation study",
    ]
    for key, row in ablation_table.items():
        lines.append(f"  {key:<20}  DFS={row['dfs_total']:.3f}  "
                     f"mean UR={row['uncertainty_reduction_mean']*100:.1f}%")
    lines += ["", "=" * 70]
    out = OUTPUT_DIR / "summary.txt"
    out.write_text("\n".join(lines))
    print(f"  ✓ summary.txt")

    import json
    payload = {
        "source_names": SRC_NAMES,
        "q_true_kg_hr": Q_TRUE_KG_HR.tolist(),
        "ur_per_source": {
            "OP": post_op.uncertainty_reduction.tolist(),
            "EC": post_ec.uncertainty_reduction.tolist(),
            "combined": post_all.uncertainty_reduction.tolist(),
        },
        "dfs_total": {
            "OP": float(dof_op.dfs_total),
            "EC": float(dof_ec.dfs_total),
            "combined": float(dof_all.dfs_total),
        },
        "network_scaling": {
            "ur_target": UR_TARGET,
            "OP": {"mean_ur": ur_curve_op.tolist(), "n_for_target": n_req_op},
            "EC": {"mean_ur": ur_curve_ec.tolist(), "n_for_target": n_req_ec},
        },
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"  ✓ results.json")


if __name__ == "__main__":
    main()
