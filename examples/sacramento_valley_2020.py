"""
Sacramento Valley 2020 — Diffuse CH₄ OSSE

Workflow
--------
1. Download ERA5 reanalysis (ECMWF CDS) for April and July 2020.
2. Run FLEXPART forward simulations for each month (diffuse rice-paddy source).
3. Load concentration outputs and compute monthly-mean wind statistics.
4. Build a Gaussian forward operator G calibrated to ERA5 winds.
5. Run Fisher / DFS / averaging-kernel analysis for each month.
6. Run Bayesian OE inversion and compare April vs July.
7. Save all figures and a summary to runs/sacramento_valley_2020/.

Prerequisites
-------------
- ERA5 credentials in ~/.cdsapirc  (see https://cds.climate.copernicus.eu/how-to-api)
- cdsapi and eccodes installed:  pip install cdsapi eccodes
- FLEXPART binary present at     flexpart/src/FLEXPART
- matplotlib:                    pip install matplotlib

Run from repo root
------------------
    python examples/sacramento_valley_2020.py
"""
import argparse
import sys
import shutil
import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from enforceflux.meteo.era5 import (
    ERA5Downloader,
    available_covers_window,
    is_flexpart_meteo_compatible,
)
from enforceflux.flexpart.simulation import FlexpartSimulation
from enforceflux.inversion import oe_from_linear
from enforceflux.analysis import (
    analyze_information_content,
    run_ablation_study,
    plot_forward_operator,
    plot_averaging_kernel,
    plot_dfs_per_source,
    plot_posterior_uncertainty,
    plot_ablation_comparison,
    plot_flux_comparison,
    create_simulation_movie_from_netcdf,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Output directory ─────────────────────────────────────────────────────────

RUN_DIR = REPO_ROOT / "runs" / "sacramento_valley_2020"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# ── Sacramento Valley geometry ───────────────────────────────────────────────

# ERA5 / FLEXPART meteorology domain (buffer around the valley)
METEO_BBOX = (-124.5, 36.5, -118.0, 42.0)   # lon_min, lat_min, lon_max, lat_max

# Analysis source grid over the rice-paddy zone (Yolo–Tehama counties).
# Higher spatial resolution: 0.25° cells across the 2°×2° valley box → 8 × 8.
GRID_RES_DEG = 0.25
_GRID_LON = (-122.5, -120.5)   # W, E edges
_GRID_LAT = (38.5, 40.5)       # S, N edges
_SRC_LON_CTRS = np.arange(_GRID_LON[0] + GRID_RES_DEG / 2, _GRID_LON[1], GRID_RES_DEG)  # W → E
_SRC_LAT_CTRS = np.arange(_GRID_LAT[0] + GRID_RES_DEG / 2, _GRID_LAT[1], GRID_RES_DEG)  # S → N
N_LON, N_LAT = len(_SRC_LON_CTRS), len(_SRC_LAT_CTRS)
SRC_LON, SRC_LAT = np.meshgrid(_SRC_LON_CTRS, _SRC_LAT_CTRS)     # (N_LAT, N_LON)
SRC_LON = SRC_LON.ravel()
SRC_LAT = SRC_LAT.ravel()
N_SRC = len(SRC_LON)

# Source names — grid position, row-major S→N, W→E (resolution-agnostic)
SRC_NAMES = [f"r{r}c{c}" for r in range(N_LAT) for c in range(N_LON)]

# True emission fluxes (kg m⁻² s⁻¹) for each season
# Cell area at lat~39.5°N for the chosen grid resolution
CELL_AREA_M2 = GRID_RES_DEG * GRID_RES_DEG * (111_000.0 ** 2) * np.cos(np.radians(39.5))
FLUX_APRIL = 2.0e-10   # pre-season  (~0.17 mg m⁻² hr⁻¹)
FLUX_JULY  = 1.5e-9    # peak season (~5.4  mg m⁻² hr⁻¹)

# True emission rates per cell (kg s⁻¹) used as "ground truth" for the OSSE
Q_TRUE_APRIL = np.full(N_SRC, FLUX_APRIL * CELL_AREA_M2)
Q_TRUE_JULY  = np.full(N_SRC, FLUX_JULY  * CELL_AREA_M2)

# Prior: 50 % uncertainty on each source cell
Q_PRIOR_APRIL = Q_TRUE_APRIL.copy()
Q_PRIOR_JULY  = Q_TRUE_JULY.copy()
SA_PRIOR_STD_FRAC = 0.50   # σ_prior = 50 % of prior mean

# ── Instrument network ───────────────────────────────────────────────────────

# Open-path (OP) sensors: ring around the valley on ridgelines / passes
OP_LON = np.array([-122.4, -122.4, -122.4, -122.0, -120.7, -121.5])
OP_LAT = np.array([ 38.1,   39.0,   40.0,   40.8,   39.0,   38.0])
OP_LABELS = ["SW-Carquinez", "W-Williams", "NW-Willows",
             "N-Redding",    "E-Foothills", "SE-Delta"]
N_OP = len(OP_LON)

# Eddy-covariance (EC) towers: inside the rice-paddy zone
EC_LON = np.array([-122.0, -121.8, -121.6, -122.1, -121.5])
EC_LAT = np.array([ 39.0,   39.5,   38.8,   40.0,   39.3])
EC_LABELS = ["Colusa-EC", "Glenn-EC", "Yolo-EC", "Tehama-EC", "Sutter-EC"]
N_EC = len(EC_LON)

# Observation error variances
SE_OP_VAR = (3.0e-3) ** 2    # (3 ppb)²  in ppm²
SE_EC_VAR = (2.0)    ** 2    # (2 nmol m⁻² s⁻¹)²


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — ERA5 download
# ═══════════════════════════════════════════════════════════════════════════════

def download_era5(start: str, end: str, meteo_dir: Path) -> bool:
    """Download ERA5 for one month. Returns True on success."""
    available = meteo_dir / "AVAILABLE"
    if available.exists():
        if is_flexpart_meteo_compatible(meteo_dir, available):
            if available_covers_window(available, start, end, timestep_hours=3):
                log.info(
                    "[ERA5] FLEXPART-compatible AVAILABLE already covers %s → %s, skipping: %s",
                    start,
                    end,
                    meteo_dir,
                )
                return True
            log.warning(
                "[ERA5] AVAILABLE is compatible but does not cover %s → %s. Refreshing %s",
                start,
                end,
                meteo_dir,
            )
        else:
            log.warning(
                "[ERA5] Existing meteorology is not FLEXPART-compatible (missing model-level 'pv'). "
                "Refreshing %s",
                meteo_dir,
            )
        shutil.rmtree(meteo_dir, ignore_errors=True)
        meteo_dir.mkdir(parents=True, exist_ok=True)
    try:
        dl = ERA5Downloader(
            output_dir=meteo_dir,
            timestep_hours=3,
            vertical_mode="model_levels",
            model_level_grid_deg=0.25,
            model_level_allow_global_fallback=False,
            cleanup_raw_daily_grib=True,
        )
        result = dl.download(start=start, end=end, bbox=METEO_BBOX)
        log.info("[ERA5] Downloaded %d files → %s", result.n_timesteps, result.available_file)
        return True
    except RuntimeError as exc:
        log.warning("[ERA5] Download failed: %s", exc)
        log.warning("[ERA5] Ensure cdsapi is installed and ~/.cdsapirc is configured.")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — FLEXPART simulation
# ═══════════════════════════════════════════════════════════════════════════════

def run_flexpart_sim(yaml_path: Path, output_nc: Path) -> bool:
    """Prepare and run FLEXPART for one month. Returns True on success."""
    if output_nc.exists():
        log.info("[FLEXPART] Output exists, skipping: %s", output_nc.name)
        return True
    try:
        from enforceflux.core.base import ITransportSimulation
        from enforceflux.utils.plugin_registry import get_plugin

        simulation = get_plugin(
            "enforceflux.transport_simulation", "flexpart", ITransportSimulation
        )()
        simulation.simulate(
            [], None, {"sim_config": str(yaml_path), "output_path": str(output_nc)}
        )
        log.info("[FLEXPART] Wrote %s", output_nc)
        return True
    except FileNotFoundError as exc:
        log.warning("[FLEXPART] %s", exc)
        return False
    except Exception as exc:
        log.warning("[FLEXPART] Run failed: %s", exc)
        return False


def run_flexpart_sim_window(
    yaml_path: Path,
    output_nc: Path,
    start_iso: str,
    end_iso: str,
) -> bool:
    """Run FLEXPART over a specific time window for fast smoke tests.

    Kept on the direct ``FlexpartSimulation`` API (not the transport_simulation
    plugin) because it imperatively rewrites the simulation window *and* each
    source's start/end times — reconfiguration beyond the config-driven plugin's
    scalar override surface.
    """
    if output_nc.exists():
        log.info("[FLEXPART] Output exists, skipping: %s", output_nc.name)
        return True

    try:
        t0 = datetime.fromisoformat(start_iso).replace(tzinfo=timezone.utc)
        t1 = datetime.fromisoformat(end_iso).replace(tzinfo=timezone.utc)
        sim = FlexpartSimulation.from_yaml(yaml_path)
        sim.config.start = t0
        sim.config.end = t1
        sim.config.output_path = output_nc
        for src in sim.config.sources:
            src.start = t0
            src.end = t1
        sim.run()
        log.info("[FLEXPART] Wrote %s", output_nc)
        return True
    except FileNotFoundError as exc:
        log.warning("[FLEXPART] %s", exc)
        return False
    except Exception as exc:
        log.warning("[FLEXPART] Run failed: %s", exc)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Extract ERA5 mean wind statistics
# ═══════════════════════════════════════════════════════════════════════════════

# Monthly-mean 10 m winds for Sacramento Valley (fallback climatological values)
# July: dominant SW delta breeze (Carquinez Strait intrusion)
# April: mixed, often light southerly
_CLIM_WIND = {
    "april": (1.0, 2.5),    # (u_ms, v_ms) — light southerly
    "july":  (3.0, 1.5),    # strong SW delta breeze
}


def extract_wind_mean(meteo_dir: Path, month_key: str) -> tuple[float, float]:
    """Return (u10_mean, v10_mean) in m/s from ERA5 daily GRIB files.

    Falls back to climatological values if eccodes is unavailable or files
    are missing.
    """
    try:
        import eccodes
    except ImportError:
        log.info("[WIND] eccodes not available — using climatological winds for %s", month_key)
        return _CLIM_WIND[month_key]

    u_all, v_all = [], []
    grib_files = sorted(meteo_dir.glob("ERA5_sl_*.grib"))
    if not grib_files:
        # If daily raw files were cleaned up, fallback to merged FLEXPART files.
        grib_files = sorted(
            p for p in meteo_dir.glob("EA*")
            if p.is_file() and p.name.startswith("EA") and len(p.name) == 12 and p.name[2:].isdigit()
        )

    for grib_file in grib_files:
        try:
            with open(grib_file, "rb") as fh:
                while True:
                    msg = eccodes.codes_grib_new_from_file(fh)
                    if msg is None:
                        break
                    try:
                        sn = eccodes.codes_get(msg, "shortName")
                        if sn == "10u":
                            u_all.extend(eccodes.codes_get_values(msg).tolist())
                        elif sn == "10v":
                            v_all.extend(eccodes.codes_get_values(msg).tolist())
                    finally:
                        eccodes.codes_release(msg)
        except Exception:
            continue

    if u_all and v_all:
        u_mean, v_mean = float(np.mean(u_all)), float(np.mean(v_all))
        log.info("[WIND] %s: u10=%.2f m/s, v10=%.2f m/s (from ERA5)", month_key, u_mean, v_mean)
        return u_mean, v_mean

    log.info("[WIND] No ERA5 wind data found — using climatological values for %s", month_key)
    return _CLIM_WIND[month_key]


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Build Gaussian forward operator G
# ═══════════════════════════════════════════════════════════════════════════════

def _lonlat_to_m(lon: np.ndarray, lat: np.ndarray,
                 lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert lon/lat to local Cartesian (x, y) in metres relative to (lon0, lat0)."""
    RE = 6_371_000.0
    lat_c = np.radians((lat + lat0) / 2.0)
    x = np.radians(lon - lon0) * RE * np.cos(lat_c)
    y = np.radians(lat - lat0) * RE
    return x, y


def build_G_op(src_lon, src_lat, inst_lon, inst_lat,
               wind_u: float, wind_v: float,
               sigma_base_m: float = 8_000.0,
               amplitude: float = 120.0) -> np.ndarray:
    """Build OP sensitivity matrix in ppm / (kg s⁻¹).

    Uses a Gaussian plume model; only downwind sensor positions receive signal.
    Dispersion width scales as σ(d) = σ_base × (d / 50_000)^0.75.
    """
    wind_spd = max(np.hypot(wind_u, wind_v), 0.5)
    wd_x, wd_y = wind_u / wind_spd, wind_v / wind_spd   # unit wind vector

    m, n = len(inst_lon), len(src_lon)
    G = np.zeros((m, n))

    lon0 = float(np.mean(np.concatenate([src_lon, inst_lon])))
    lat0 = float(np.mean(np.concatenate([src_lat, inst_lat])))
    sx, sy = _lonlat_to_m(src_lon, src_lat, lon0, lat0)
    ix, iy = _lonlat_to_m(inst_lon, inst_lat, lon0, lat0)

    for i in range(m):
        for j in range(n):
            dx, dy = ix[i] - sx[j], iy[i] - sy[j]
            downwind = dx * wd_x + dy * wd_y
            if downwind <= 0:
                continue
            crosswind = dx * wd_y - dy * wd_x   # perpendicular component
            sigma = sigma_base_m * (downwind / 50_000.0) ** 0.75
            G[i, j] = amplitude * np.exp(-0.5 * (crosswind / sigma) ** 2)
    return G


def build_G_ec(src_lon, src_lat, inst_lon, inst_lat,
               sigma_fp_m: float = 3_000.0,
               amplitude: float = 3.2e6) -> np.ndarray:
    """Build EC sensitivity matrix in nmol m⁻² s⁻¹ / (kg s⁻¹).

    Isotropic Gaussian footprint; radius ~σ_fp.
    """
    m, n = len(inst_lon), len(src_lon)
    G = np.zeros((m, n))

    lon0 = float(np.mean(np.concatenate([src_lon, inst_lon])))
    lat0 = float(np.mean(np.concatenate([src_lat, inst_lat])))
    sx, sy = _lonlat_to_m(src_lon, src_lat, lon0, lat0)
    ix, iy = _lonlat_to_m(inst_lon, inst_lat, lon0, lat0)

    for i in range(m):
        for j in range(n):
            r2 = (ix[i] - sx[j]) ** 2 + (iy[i] - sy[j]) ** 2
            G[i, j] = amplitude * np.exp(-0.5 * r2 / sigma_fp_m ** 2)
    return G


def build_forward_operator(wind_u: float, wind_v: float
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Return (G_all, Se_all) for the combined OP + EC network."""
    G_op = build_G_op(SRC_LON, SRC_LAT, OP_LON, OP_LAT, wind_u, wind_v)
    G_ec = build_G_ec(SRC_LON, SRC_LAT, EC_LON, EC_LAT)
    G_all = np.vstack([G_op, G_ec])
    Se_all = np.concatenate([
        np.full(N_OP, SE_OP_VAR),
        np.full(N_EC, SE_EC_VAR),
    ])
    return G_all, Se_all


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Load FLEXPART output
# ═══════════════════════════════════════════════════════════════════════════════

def load_concentration_stats(nc_path: Path) -> dict | None:
    """Read FLEXPART NetCDF and return summary stats (peak conc, mean, etc.)."""
    try:
        import netCDF4 as nc4
    except ImportError:
        log.warning("[NC] netCDF4 not available")
        return None
    if not nc_path.exists():
        return None
    with nc4.Dataset(nc_path) as ds:
        # Prefer mixing ratio field if present, else mass concentration
        if "ch4_mixing_ratio" in ds.variables:
            field = np.asarray(ds["ch4_mixing_ratio"][:])
        elif "ch4_concentration" in ds.variables:
            field = np.asarray(ds["ch4_concentration"][:])
        else:
            log.warning("[NC] No CH4 field found in %s", nc_path.name)
            return None
        field = field[np.isfinite(field)]
        return {
            "peak_ppb":  float(field.max() * 1e3),
            "mean_ppb":  float(field[field > 0].mean() * 1e3) if (field > 0).any() else 0.0,
            "n_nonzero": int((field > 0).sum()),
            "units":     "ppb (mixing ratio × 1000)",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — Per-month analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_month(
    month_key: str,
    q_true: np.ndarray,
    q_prior: np.ndarray,
    wind_u: float,
    wind_v: float,
    rng: np.random.Generator,
) -> dict:
    """Run information analysis + OE inversion for one month. Returns result dict."""
    G_all, Se_all = build_forward_operator(wind_u, wind_v)
    Sa_diag = (SA_PRIOR_STD_FRAC * q_prior) ** 2

    n_obs = N_OP + N_EC
    mask_op = np.zeros(n_obs, dtype=bool); mask_op[:N_OP] = True
    mask_ec = np.zeros(n_obs, dtype=bool); mask_ec[N_OP:] = True
    obs_groups = {"OP": mask_op, "EC": mask_ec}

    # ── Information content ──────────────────────────────────────────────────
    fisher, dof, posterior = analyze_information_content(
        G=G_all, Se=Se_all, Sa=Sa_diag,
        obs_groups=obs_groups, source_names=SRC_NAMES,
    )
    ablation = run_ablation_study(
        G=G_all, Se=Se_all, Sa=Sa_diag,
        obs_groups=obs_groups, source_names=SRC_NAMES,
    )

    # ── Synthetic observations ────────────────────────────────────────────────
    y_clean = G_all @ q_true
    noise = rng.normal(0, np.sqrt(Se_all))
    y_obs  = y_clean + noise

    # Simulate 10 % dropout for each instrument independently
    dropout_mask = rng.random(len(y_obs)) < 0.10
    y_obs[dropout_mask] = np.nan

    valid = ~np.isnan(y_obs)
    log.info("[%s] %d / %d valid observations", month_key.upper(), valid.sum(), len(y_obs))

    # ── OE inversion (OP-only, EC-only, combined) ────────────────────────────
    def _invert(G, Se, y, label):
        v = ~np.isnan(y)
        if v.sum() == 0:
            log.warning("[%s] No valid observations for %s", month_key.upper(), label)
            return None
        return oe_from_linear(
            G[v], y[v], q_prior, np.diag(Sa_diag), Se[v],
            source_names=SRC_NAMES,
        )

    y_op, y_ec = y_obs[:N_OP], y_obs[N_OP:]
    oe_op  = _invert(G_all[:N_OP],  Se_all[:N_OP],  y_op,  "OP")
    oe_ec  = _invert(G_all[N_OP:],  Se_all[N_OP:],  y_ec,  "EC")
    oe_all = _invert(G_all,          Se_all,          y_obs, "OP+EC")

    def _rmse(oe, unit=1.0):
        if oe is None:
            return float("nan")
        return float(np.sqrt(np.mean((oe.x_posterior - q_true) ** 2))) * unit

    return {
        "month":      month_key,
        "wind_u":     wind_u,
        "wind_v":     wind_v,
        "G_all":      G_all,
        "Se_all":     Se_all,
        "Sa_diag":    Sa_diag,
        "q_true":     q_true,
        "q_prior":    q_prior,
        "y_obs":      y_obs,
        "y_clean":    y_clean,
        "fisher":     fisher,
        "dof":        dof,
        "posterior":  posterior,
        "ablation":   ablation,
        "oe_op":      oe_op,
        "oe_ec":      oe_ec,
        "oe_all":     oe_all,
        "rmse_op":    _rmse(oe_op),
        "rmse_ec":    _rmse(oe_ec),
        "rmse_all":   _rmse(oe_all),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7 — Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=120, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    log.info("  ✓ %s", path.name)


def save_monthly_figures(res: dict, out_dir: Path) -> None:
    """Save per-month diagnostic figures."""
    import matplotlib.pyplot as plt
    tag = res["month"]
    dof  = res["dof"]
    post = res["posterior"]

    # 1. Forward operator heat-map
    fig, ax = plot_forward_operator(
        res["G_all"], source_names=SRC_NAMES,
        obs_names=(OP_LABELS + EC_LABELS),
        title=f"Forward operator G  [{tag.capitalize()}]",
    )
    _save(fig, out_dir / f"{tag}_01_forward_operator.png")

    # 2. Averaging kernel
    fig, ax = plot_averaging_kernel(
        dof.averaging_kernel, source_names=SRC_NAMES,
        title=f"Averaging kernel  [{tag.capitalize()}]",
    )
    _save(fig, out_dir / f"{tag}_02_averaging_kernel.png")

    # 3. DFS per source
    fig, ax = plot_dfs_per_source(
        dof.dfs_per_source, source_names=SRC_NAMES,
        title=f"DFS per source  [{tag.capitalize()}]",
    )
    _save(fig, out_dir / f"{tag}_03_dfs_per_source.png")

    # 4. Posterior uncertainty
    fig, ax = plot_posterior_uncertainty(
        post.prior_sigma, post.posterior_sigma, source_names=SRC_NAMES,
        title=f"Posterior uncertainty  [{tag.capitalize()}]",
    )
    _save(fig, out_dir / f"{tag}_04_posterior_uncertainty.png")

    # 5. Ablation DFS
    fig, ax = plot_ablation_comparison(
        res["ablation"], title=f"Network ablation  [{tag.capitalize()}]",
    )
    _save(fig, out_dir / f"{tag}_05_ablation.png")

    # 6. Flux comparison (combined inversion)
    if res["oe_all"] is not None:
        oe = res["oe_all"]
        fig, ax = plot_flux_comparison(
            oe.x_prior, oe.x_posterior, x_true=res["q_true"],
            source_names=SRC_NAMES,
            posterior_sigma=np.sqrt(np.diag(oe.posterior_cov)),
            title=f"OE flux retrieval  [{tag.capitalize()}]",
        )
        _save(fig, out_dir / f"{tag}_06_flux_comparison.png")

    # 7. Scenario map
    _save_scenario_map(res, out_dir / f"{tag}_00_scenario_map.png")


def save_simulation_movie(nc_path: Path, month_key: str, out_dir: Path) -> None:
    """Write a GIF showing the concentration field evolution over time."""
    if not nc_path.exists():
        log.info("[MOVIE] NetCDF output missing, skipping %s movie", month_key)
        return

    movie_path = out_dir / f"{month_key}_07_concentration_evolution.gif"
    create_simulation_movie_from_netcdf(
        nc_path,
        output_path=movie_path,
        level_index=0,
        release_index=0,
        fps=4,
        title_prefix=f"{month_key.capitalize()} CH4 concentration",
    )
    log.info("  ✓ %s", movie_path.name)


def _save_scenario_map(res: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.cm as cm

    fig, ax = plt.subplots(figsize=(7, 7))
    tag = res["month"]

    # Source cells — colour by true flux
    q_plot = res["q_true"] * 1e6   # kg/s → μg/s for display scale
    norm = plt.Normalize(q_plot.min(), q_plot.max())
    cmap = cm.YlOrRd
    for k in range(N_SRC):
        rect = plt.Rectangle(
            (SRC_LON[k] - GRID_RES_DEG / 2, SRC_LAT[k] - GRID_RES_DEG / 2),
            GRID_RES_DEG, GRID_RES_DEG,
            facecolor=cmap(norm(q_plot[k])), alpha=0.6, linewidth=0.4, edgecolor="grey",
        )
        ax.add_patch(rect)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="True flux (μg s⁻¹ per cell)", shrink=0.7)

    # OP sensors
    ax.scatter(OP_LON, OP_LAT, s=70, marker="^", color="steelblue",
               zorder=5, label="OP sensor")
    for k, lbl in enumerate(OP_LABELS):
        ax.annotate(lbl, (OP_LON[k], OP_LAT[k]),
                    fontsize=6, textcoords="offset points", xytext=(3, 3))

    # EC towers
    ax.scatter(EC_LON, EC_LAT, s=70, marker="s", color="darkorange",
               zorder=5, label="EC tower")
    for k, lbl in enumerate(EC_LABELS):
        ax.annotate(lbl, (EC_LON[k], EC_LAT[k]),
                    fontsize=6, textcoords="offset points", xytext=(3, 3))

    # Wind arrow (domain-centre)
    cx, cy = -121.5, 39.5
    spd = np.hypot(res["wind_u"], res["wind_v"])
    ax.annotate("", xy=(cx + res["wind_u"] / spd * 0.4, cy + res["wind_v"] / spd * 0.4),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color="navy", lw=2))
    ax.text(cx + 0.05, cy - 0.2,
            f"ERA5 mean wind\n{spd:.1f} m/s", fontsize=7, color="navy")

    ax.set_xlim(-123.0, -120.0)
    ax.set_ylim(38.0, 41.2)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"Sacramento Valley — {tag.capitalize()} 2020\nInstrument network & source grid")
    ax.legend(handles=[
        mpatches.Patch(color="steelblue",  label="OP sensor"),
        mpatches.Patch(color="darkorange", label="EC tower"),
        mpatches.Patch(color="#d9a84e",    label="Source cell (colour=flux)"),
    ], fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    _save(fig, path)


def save_comparison_figure(res_apr: dict, res_jul: dict, out_dir: Path) -> None:
    """Side-by-side comparison of DFS, uncertainty reduction, and wind."""
    import matplotlib.pyplot as plt

    labels = SRC_NAMES
    x = np.arange(N_SRC)
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Sacramento Valley 2020: April vs July", fontsize=14)

    # (0,0) DFS per source
    ax = axes[0, 0]
    ax.bar(x - width/2, res_apr["dof"].dfs_per_source, width,
           label="April", color="steelblue", alpha=0.8)
    ax.bar(x + width/2, res_jul["dof"].dfs_per_source, width,
           label="July",  color="darkorange", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("DFS per source"); ax.set_title("Degrees of freedom per source")
    ax.legend()

    # (0,1) Uncertainty reduction
    ax = axes[0, 1]
    ur_apr = res_apr["posterior"].uncertainty_reduction * 100
    ur_jul = res_jul["posterior"].uncertainty_reduction * 100
    ax.bar(x - width/2, ur_apr, width, label="April", color="steelblue", alpha=0.8)
    ax.bar(x + width/2, ur_jul, width, label="July",  color="darkorange", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Uncertainty reduction (%)")
    ax.set_title("Posterior uncertainty reduction"); ax.legend()

    # (1,0) Total DFS bar
    ax = axes[1, 0]
    for i, (tag, res) in enumerate([("April", res_apr), ("July", res_jul)]):
        dfs_op  = res["ablation"]["OP"].dfs_total
        dfs_ec  = res["ablation"]["EC"].dfs_total
        dfs_all = res["ablation"]["OP+EC"].dfs_total
        ax.bar([i*4, i*4+1, i*4+2], [dfs_op, dfs_ec, dfs_all],
               color=["steelblue", "darkorange", "seagreen"], alpha=0.85)
        ax.text(i*4,   dfs_op  + 0.05, f"{dfs_op:.2f}",  ha="center", fontsize=7)
        ax.text(i*4+1, dfs_ec  + 0.05, f"{dfs_ec:.2f}",  ha="center", fontsize=7)
        ax.text(i*4+2, dfs_all + 0.05, f"{dfs_all:.2f}", ha="center", fontsize=7)
    ax.set_xticks([0,1,2, 4,5,6])
    ax.set_xticklabels(["OP","EC","OP+EC","OP","EC","OP+EC"])
    ax.set_ylabel("Total DFS"); ax.set_title("Total DFS by network subset")
    for x_pos, label in [(1, "April"), (5, "July")]:
        ax.text(x_pos, ax.get_ylim()[1] * 0.9, label, ha="center",
                fontsize=9, fontweight="bold")

    # (1,1) RMSE comparison
    ax = axes[1, 1]
    scenarios = ["OP", "EC", "OP+EC"]
    rmse_apr = [res_apr["rmse_op"], res_apr["rmse_ec"], res_apr["rmse_all"]]
    rmse_jul = [res_jul["rmse_op"], res_jul["rmse_ec"], res_jul["rmse_all"]]
    xi = np.arange(3)
    bars_a = ax.bar(xi - width/2, [r * 1e9 for r in rmse_apr], width,
                    label="April", color="steelblue", alpha=0.85)
    bars_j = ax.bar(xi + width/2, [r * 1e9 for r in rmse_jul], width,
                    label="July",  color="darkorange", alpha=0.85)
    ax.set_xticks(xi); ax.set_xticklabels(scenarios)
    ax.set_ylabel("RMSE (ng s⁻¹ per cell)")
    ax.set_title("OE inversion RMSE"); ax.legend()

    fig.tight_layout()
    _save(fig, out_dir / "comparison_april_vs_july.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Sacramento Valley 2020 diffuse CH4 OSSE")
    parser.add_argument(
        "--smoke-1day",
        action="store_true",
        help="Run a fast 1-day smoke test for April and July 2020.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Sacramento Valley 2020 — Diffuse CH₄ OSSE")
    if args.smoke_1day:
        print("Mode: 1-day smoke test (fail fast)")
    print("=" * 70)

    rng = np.random.default_rng(42)

    april_end = "2020-04-01T21:00" if args.smoke_1day else "2020-04-30T21:00"
    july_end = "2020-07-01T21:00" if args.smoke_1day else "2020-07-31T21:00"
    april_nc = "sacval_april_2020_1day.nc" if args.smoke_1day else "sacval_april_2020.nc"
    july_nc = "sacval_july_2020_1day.nc" if args.smoke_1day else "sacval_july_2020.nc"

    months = {
        "april": {
            "start": "2020-04-01T00:00",
            "end":   april_end,
            "yaml":  REPO_ROOT / "examples" / "sacval_april_2020.yaml",
            "nc":    REPO_ROOT / "outputs"  / april_nc,
            "q_true":  Q_TRUE_APRIL,
            "q_prior": Q_PRIOR_APRIL,
        },
        "july": {
            "start": "2020-07-01T00:00",
            "end":   july_end,
            "yaml":  REPO_ROOT / "examples" / "sacval_july_2020.yaml",
            "nc":    REPO_ROOT / "outputs"  / july_nc,
            "q_true":  Q_TRUE_JULY,
            "q_prior": Q_PRIOR_JULY,
        },
    }

    # ── PHASE 1-2: ERA5 download + FLEXPART ──────────────────────────────────
    for key, m in months.items():
        print(f"\n── {key.upper()} ─────────────────────────────────────────────")

        meteo_dir = RUN_DIR / f"meteo_{key}"
        print(f"[ERA5]    downloading {m['start'][:7]} → {meteo_dir.relative_to(REPO_ROOT)}")
        ok_era5 = download_era5(m["start"], m["end"], meteo_dir)

        if ok_era5:
            print(f"[FLEXPART] running simulation → {m['nc'].relative_to(REPO_ROOT)}")
            if args.smoke_1day:
                run_flexpart_sim_window(m["yaml"], m["nc"], m["start"], m["end"])
            else:
                run_flexpart_sim(m["yaml"], m["nc"])
        else:
            print("[FLEXPART] skipping — no ERA5 data available")

    # ── PHASE 3-7: Analysis (runs even without ERA5 / FLEXPART) ──────────────
    print("\n" + "=" * 70)
    print("Information analysis & OE inversion")
    print("=" * 70)

    results = {}
    for key, m in months.items():
        meteo_dir = RUN_DIR / f"meteo_{key}"
        wind_u, wind_v = extract_wind_mean(meteo_dir, key)

        # Optional: load FLEXPART stats for reporting
        nc_stats = load_concentration_stats(m["nc"])
        if nc_stats:
            print(f"\n  [{key}] FLEXPART peak: {nc_stats['peak_ppb']:.2f} ppb "
                  f"| mean (>0): {nc_stats['mean_ppb']:.3f} ppb")

        print(f"\n  [{key}] ERA5 mean wind: u={wind_u:.2f} m/s, v={wind_v:.2f} m/s "
              f"({np.degrees(np.arctan2(wind_v, wind_u)):.0f}° from E)")

        res = analyse_month(key, m["q_true"], m["q_prior"], wind_u, wind_v, rng)
        results[key] = res

        dof = res["dof"]
        post = res["posterior"]
        print(f"  DFS (OP only): {res['ablation']['OP'].dfs_total:.2f}  |  "
              f"EC only: {res['ablation']['EC'].dfs_total:.2f}  |  "
              f"OP+EC: {res['ablation']['OP+EC'].dfs_total:.2f}  (max={N_SRC})")
        print(f"  Mean UR: OP={np.mean(post.uncertainty_reduction[:N_OP])*100:.0f}%  "
              f"(combined={np.mean(post.uncertainty_reduction)*100:.0f}%)")
        print(f"  RMSE [kg/s]:  OP={res['rmse_op']:.3e}  "
              f"EC={res['rmse_ec']:.3e}  combined={res['rmse_all']:.3e}")

    # ── PHASE 7: Figures ──────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print(f"Saving figures → {RUN_DIR.relative_to(REPO_ROOT)}/")
    print("─" * 70)

    try:
        import matplotlib
        matplotlib.use("Agg")

        for key, res in results.items():
            save_monthly_figures(res, RUN_DIR)
            save_simulation_movie(months[key]["nc"], key, RUN_DIR)

        save_comparison_figure(results["april"], results["july"], RUN_DIR)

    except ImportError:
        print("[PLOT] matplotlib not available — install with: pip install matplotlib")

    # ── Summary report ────────────────────────────────────────────────────────
    summary_name = "summary_1day.txt" if args.smoke_1day else "summary.txt"
    summary_path = RUN_DIR / summary_name
    _write_summary(results, summary_path)
    print(f"\n  ✓ {summary_path.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 70)
    print("Done.")


def _write_summary(results: dict, path: Path) -> None:
    lines = [
        "Sacramento Valley 2020 — Diffuse CH₄ OSSE Summary",
        "=" * 60,
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Source grid: {N_LON} × {N_LAT} = {N_SRC} cells at {GRID_RES_DEG}° (Yolo–Tehama counties)",
        f"Instruments: {N_OP} OP sensors + {N_EC} EC towers",
        "",
    ]
    for key, res in results.items():
        abl = res["ablation"]
        post = res["posterior"]
        ur_mean = np.mean(post.uncertainty_reduction) * 100
        lines += [
            f"── {key.upper()} 2020 ──────────────────────────────────────",
            f"  True flux (April): {FLUX_APRIL:.2e} kg m⁻² s⁻¹  "
            f"| (July): {FLUX_JULY:.2e} kg m⁻² s⁻¹",
            f"  ERA5 mean wind: u={res['wind_u']:.2f} m/s, v={res['wind_v']:.2f} m/s",
            f"  Total DFS  — OP: {abl['OP'].dfs_total:.2f}  "
            f"EC: {abl['EC'].dfs_total:.2f}  "
            f"OP+EC: {abl['OP+EC'].dfs_total:.2f}  (max={N_SRC})",
            f"  Mean uncertainty reduction (OP+EC): {ur_mean:.0f} %",
            f"  OE RMSE  — OP: {res['rmse_op']:.3e} kg/s  "
            f"EC: {res['rmse_ec']:.3e} kg/s  combined: {res['rmse_all']:.3e} kg/s",
            "",
        ]
    lines += [
        "Source true emission rates",
        "  April: {:.3e} kg/s per cell  ({:.2f} kg/hr per cell)".format(
            Q_TRUE_APRIL[0], Q_TRUE_APRIL[0] * 3600),
        "  July:  {:.3e} kg/s per cell  ({:.2f} kg/hr per cell)".format(
            Q_TRUE_JULY[0],  Q_TRUE_JULY[0]  * 3600),
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
