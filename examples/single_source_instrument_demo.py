# pyright: reportMissingTypeStubs=false
"""Single-source instrument OSSE: backward FLEXPART footprints → DFS analysis.

Workflow
--------
1. Run FLEXPART in backward mode from 3 sensor locations to compute surface
   footprints (rows of the transport Jacobian G).
2. Sample the existing *forward* simulation at sensor locations to produce
   synthetic "true" observations.
3. Apply the EnforceFlux InstrumentOperator to add instrument-specific noise
   (1 open-path OP + 2 point-flux EC sensors).
4. Run Fisher / DFS / averaging-kernel / posterior uncertainty analysis.
5. Save a sensor-network map, per-sensor footprint plots, and the standard
   DFS/AK/posterior figures.

Instrument configuration
------------------------
  OP_downwind       open-path (OP)  directly east of source — path-integrated
  point_sensor_NNE  eddy-covariance (EC) tower NNE of source
  point_sensor_NE   eddy-covariance (EC) tower NE  of source

Run from the repo root:
    python examples/single_source_instrument_demo.py
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from enforceflux.core.base import ITransportSimulation                 # noqa: E402
from enforceflux.utils.plugin_registry import get_plugin               # noqa: E402
from enforceflux.instrument import Instrument, InstrumentOperator      # noqa: E402
from enforceflux.analysis import (                                      # noqa: E402
    analyze_information_content,
    run_ablation_study,
    plot_forward_operator,
    plot_averaging_kernel,
    plot_dfs_per_source,
    plot_posterior_uncertainty,
    plot_ablation_comparison,
    load_simulation_netcdf,
)
from enforceflux.analysis.information_core import analyze_information_content_spatial  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────

BWD_YAML      = REPO_ROOT / "examples" / "single_source_instrument_backward_500m.yaml"
BWD_NC        = REPO_ROOT / "runs" / "single_source_instrument" / "backward_footprints_500m.nc"
OUT_DIR       = REPO_ROOT / "runs" / "single_source_instrument"

# ── Source and sensor geometry ────────────────────────────────────────────────

SOURCE_LON, SOURCE_LAT = -121.75, 39.15   # point source location
SOURCE_RATE_KG_S = 2.7777778e-2           # 100 kg hr⁻¹

# March 31 ERA5: wind SW→ENE (bearing ~68°) at 1.7–3.5 m/s all day.
# Sensors placed 500 m downwind (ENE) of source, ±25° crosswind.
# Backward trajectories travel WSW → reach source in ~3 min.
#   sensor_ENE_axis : bearing 68°, 500 m → on plume axis
#   sensor_NE       : bearing 43°, 500 m → left flank
#   sensor_E        : bearing 93°, 500 m → right flank
SENSORS = [
    Instrument(id="sensor_ENE_axis", tech_id="OP", mode="good",
               x=-121.7446, y=39.1517, z=10.0,
               path_length_m=1.0, path_bearing_deg=0.0),
    Instrument(id="sensor_NE",       tech_id="OP", mode="good",
               x=-121.7460, y=39.1533, z=10.0,
               path_length_m=1.0, path_bearing_deg=0.0),
    Instrument(id="sensor_E",        tech_id="OP", mode="good",
               x=-121.7442, y=39.1498, z=10.0,
               path_length_m=1.0, path_bearing_deg=0.0),
]
N_SENSORS = len(SENSORS)
SENSOR_NAMES = [s.id for s in SENSORS]

# ── Observation error variances ───────────────────────────────────────────────
# All sensors are OP tech; σ_abs = 0.003 ppm.
# 1 ppm CH4 (by volume) ≈ 714 μg m⁻³ = 7.14e5 ng m⁻³ at STP, so
# 0.003 ppm ≈ 2143 ng m⁻³.
SE_OP = (2143.0) ** 2   # (ng m⁻³)²
SE_DIAG = np.array([SE_OP, SE_OP, SE_OP], dtype=float)

# Prior source uncertainty: 50 % on the 100 kg hr⁻¹ point source emission
PRIOR_STD_FRAC = 0.50
SA_DIAG = np.array([(PRIOR_STD_FRAC * SOURCE_RATE_KG_S) ** 2], dtype=float)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 — backward FLEXPART footprints
# ═══════════════════════════════════════════════════════════════════════════════

def _run_backward_flexpart() -> Path:
    """Run FLEXPART in backward mode and return the footprint NetCDF path."""
    if BWD_NC.exists():
        print(f"[BWD]  Cached footprints found, skipping run: {BWD_NC.name}")
        return BWD_NC

    print("[BWD]  Running FLEXPART backward simulation …")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    simulation = get_plugin(
        "enforceflux.transport_simulation", "flexpart", ITransportSimulation
    )()
    result = simulation.simulate(
        [],
        None,
        {
            "sim_config": str(BWD_YAML),
            "ldirect": -1,                 # backward mode
            "output_per_source": True,
            "output_path": str(BWD_NC),
        },
    )
    written = result.output_path
    print(f"[BWD]  Wrote footprint NetCDF: {written.name}")
    return written


def _load_footprints(bwd_nc: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load time-averaged surface footprints for each sensor.

    Returns
    -------
    G       : (N_SENSORS, n_lat*n_lon) – sensitivity matrix [s m⁻³]
    lons    : (n_lon,)
    lats    : (n_lat,)
    """
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise RuntimeError("netCDF4 is required: pip install netCDF4") from exc

    with Dataset(bwd_nc) as ds:
        lon_name = next((n for n in ("longitude", "lon", "xlon") if n in ds.variables), None)
        lat_name = next((n for n in ("latitude",  "lat", "ylat") if n in ds.variables), None)
        if lon_name is None or lat_name is None:
            raise KeyError("Footprint NetCDF missing coordinate variables")

        lons = np.asarray(ds.variables[lon_name][:], dtype=float)
        lats = np.asarray(ds.variables[lat_name][:], dtype=float)

        var_name = next(
            (n for n in ("ch4_concentration", "ch4_mixing_ratio", "spec001", "spec001_mr")
             if n in ds.variables),
            None,
        )
        if var_name is None:
            raise KeyError("No footprint variable found in backward NetCDF")

        raw = np.asarray(ds.variables[var_name][:], dtype=float)   # (..., time, height, lat, lon)

    # Normalise shape to (pointspec, time, height, lat, lon)
    # FLEXPART NetCDF dim order: (nageclass, pointspec, time, height, lat, lon)
    if raw.ndim == 6:
        raw = raw[0]   # drop nageclass
    # raw: (pointspec, time, height, lat, lon)
    if raw.shape[0] != N_SENSORS:
        raise ValueError(
            f"Expected pointspec={N_SENSORS} but got {raw.shape[0]}. "
            "Check that output.per_source=true was active in backward run."
        )

    n_lat, n_lon = len(lats), len(lons)

    # Grid cell volume for scaling: V = cell_area × first_height_bin
    # FLEXPART footprint units [s] = particle residence time per particle.
    # Concentration sensitivity: G = fp / V_cell × 1e12
    # so that G [ng m⁻³ / (kg s⁻¹)] × Q [kg/s] = c [ng/m³].
    import math
    lat_c = float(lats.mean())
    dx_m = (lons[1] - lons[0]) * math.cos(math.radians(lat_c)) * 111_320.0
    dy_m = (lats[1] - lats[0]) * 111_320.0
    cell_area_m2 = abs(dx_m * dy_m)
    # Use the first height level as the mixing layer thickness (m)
    h_m = 100.0   # first output height bin
    cell_vol_m3 = cell_area_m2 * h_m

    G = np.zeros((N_SENSORS, n_lat * n_lon), dtype=float)
    for i in range(N_SENSORS):
        fp = raw[i, :, 0, :, :]            # (time, lat, lon) at surface level
        fp_mean = np.nanmean(fp, axis=0)   # time-average footprint [s]
        fp_mean = np.where(np.isfinite(fp_mean), fp_mean, 0.0)
        G[i, :] = fp_mean.flatten()

    # Convert [s] → [ng m⁻³ per (kg s⁻¹)] via cell volume and unit conversion
    G = G / cell_vol_m3 * 1e12

    print(f"[G]    Footprint matrix shape: {G.shape}  "
          f"(non-zero cells: {(G > 0).sum()} / {G.size})")
    return G, lons, lats


def _make_analysis_domain() -> tuple[np.ndarray, np.ndarray]:
    """0.005° (~500 m) grid centred on the source, ±0.20° in each direction."""
    lons = np.arange(-121.95, -121.55 + 1e-9, 0.005)
    lats = np.arange( 38.95,   39.35 + 1e-9, 0.005)
    return lons, lats


def _build_G_analytical(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Gaussian plume sensitivity matrix G[sensor, grid_cell] in ng m⁻³ per kg s⁻¹.

    April ERA5 NNW→SSE wind (bearing 149°, 3.5 m/s) with Pasquill-Gifford
    class C-D dispersion.  Each grid cell is treated as a hypothetical point
    source; G[i,j] is the time-averaged concentration sensor i would measure
    from unit emission at j.
    """
    RE = 6_371_000.0
    lat_c = np.radians(lats.mean())
    # Wind: from WSW (248°) → blowing toward ENE (bearing 68°), March 31 conditions
    wind_spd = 2.6   # m/s (March 31 daily mean)
    wind_x   = wind_spd * np.sin(np.radians(68.0))   # eastward component
    wind_y   = wind_spd * np.cos(np.radians(68.0))   # northward component
    ux, uy   = wind_x / wind_spd, wind_y / wind_spd  # unit wind vector

    H_src, z_obs = 5.0, 2.0   # release height and obs height (m)

    lon_grid, lat_grid = np.meshgrid(lons, lats)   # (n_lat, n_lon)
    # Grid cell centres → local Cartesian metres
    x_cell = np.radians(lon_grid - SOURCE_LON) * RE * np.cos(lat_c)
    y_cell = np.radians(lat_grid - SOURCE_LAT) * RE

    n_cells = x_cell.size
    G = np.zeros((N_SENSORS, n_cells), dtype=float)

    for i, inst in enumerate(SENSORS):
        sx = np.radians(inst.x - SOURCE_LON) * RE * np.cos(lat_c)
        sy = np.radians(inst.y - SOURCE_LAT) * RE

        # For each grid cell j as source, sensor is displaced (sx-xj, sy-yj)
        dx = (sx - x_cell).ravel()   # (n_cells,)
        dy = (sy - y_cell).ravel()

        # Decompose into along-wind and crosswind distances
        x_aw = dx * ux + dy * uy          # along-wind (positive = downwind)
        y_cw = -dx * uy + dy * ux         # crosswind

        downwind = x_aw > 10.0
        # PG class C-D dispersion parameters
        sig_y = np.where(downwind, np.maximum(0.18 * x_aw, 100.0), 100.0)
        sig_z = np.where(downwind, np.maximum(0.12 * x_aw,  50.0),  50.0)

        gauss_y = np.exp(-y_cw**2 / (2.0 * sig_y**2))
        gauss_z = (np.exp(-(z_obs - H_src)**2 / (2.0 * sig_z**2)) +
                   np.exp(-(z_obs + H_src)**2 / (2.0 * sig_z**2)))
        # C [ng m⁻³] per Q [kg s⁻¹]: C/Q = gauss_y × gauss_z × 1e12 / (π σ_y σ_z u)
        C_per_Q = np.where(
            downwind,
            gauss_y * gauss_z * 1e12 / (np.pi * sig_y * sig_z * wind_spd),
            0.0,
        )
        G[i, :] = C_per_Q

    n_nz = int((G > 0).sum())
    print(f"[G]    Analytical Gaussian G shape: {G.shape}  non-zero: {n_nz}/{G.size}")
    return G


def _build_G_from_forward(fwd_nc: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Second fallback: build G by sampling forward concentration at sensor locations.

    G[i, t] = ch4_concentration(sensor_i, t) / SOURCE_RATE_KG_S
            = [ng m⁻³] / [kg s⁻¹]  →  [ng m⁻³ per (kg s⁻¹)]
    Shape: (N_SENSORS, n_time).
    """
    # Request concentration (ng m⁻³) before mixing ratio (ng kg⁻¹)
    data = load_simulation_netcdf(
        fwd_nc,
        variable_names=("ch4_concentration", "ch4_mixing_ratio"),
    )
    conc = np.asarray(data["concentration"], dtype=float)
    lons = data["lons"]
    lats = data["lats"]

    # Collapse nageclass / pointspec → (time, height, lat, lon)
    while conc.ndim > 4:
        conc = conc[0]

    # If mixing ratio loaded instead of concentration, convert via air density
    if "mixing_ratio" in data.get("variable_name", ""):
        conc = conc * 1.2   # ng kg⁻¹ × 1.2 kg m⁻³ → ng m⁻³

    n_time = conc.shape[0]
    G = np.zeros((N_SENSORS, n_time), dtype=float)
    for i, inst in enumerate(SENSORS):
        iy = int(np.argmin(np.abs(lats - inst.y)))
        ix = int(np.argmin(np.abs(lons - inst.x)))
        G[i, :] = conc[:, 0, iy, ix] / SOURCE_RATE_KG_S

    print(f"[G]    Forward-sampled G shape: {G.shape}  (2nd fallback)")
    return G, lons, lats


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 — synthetic observations via InstrumentOperator
# ═══════════════════════════════════════════════════════════════════════════════

def _simulate_observations(G: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply InstrumentOperator to G and return y_clean, y_obs, R_diag.

    x_true = [1.0] means "source emits at the true rate" (unit vector in
    source space; G already encodes the physical units).
    """
    op = InstrumentOperator(SENSORS, rng=np.random.default_rng(42))

    # Treat each column of G as a separate temporal observation.
    n_cols = G.shape[1]
    y_clean_all = np.zeros((n_cols, N_SENSORS))
    y_obs_all   = np.zeros((n_cols, N_SENSORS))
    valid_all   = np.zeros((n_cols, N_SENSORS), dtype=bool)
    R_diag_all  = np.zeros((n_cols, N_SENSORS))

    x_true = np.array([1.0])   # unit source state (units absorbed into G)

    for t in range(n_cols):
        g_col = G[:, t:t+1]   # (N_SENSORS, 1) — one source at time t
        result = op.simulate_observations(g_col, x_true)
        y_clean_all[t, :] = result.y_clean
        y_obs_all  [t, :] = result.y_obs
        valid_all  [t, :] = result.valid_mask
        R_diag_all [t, :] = np.diag(result.R)

    return y_clean_all, y_obs_all, valid_all


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3 — information content analysis
# ═══════════════════════════════════════════════════════════════════════════════

def _run_analysis(G: np.ndarray) -> dict:
    """Run Fisher / DFS / averaging-kernel / posterior analysis.

    G shape  : (N_SENSORS, n_sources)
    n_sources: either n_grid_cells (backward) or n_time_steps (forward fallback)

    When n_sources >> N_SENSORS (spatial map case), uses the Woodbury fast path
    which avoids constructing the (n×n) FIM entirely: O(m²·n) instead of O(n³).
    """
    n_src = G.shape[1]

    # Prior covariance on source state
    if n_src == 1:
        Sa = SA_DIAG.copy()
    else:
        Sa = np.full(n_src, SA_DIAG[0], dtype=float)

    Se = SE_DIAG.copy()

    obs_groups = {
        "ENE_axis": np.array([True,  False, False]),
        "ENE+NE":   np.array([True,  True,  False]),
        "all":      np.ones(N_SENSORS, dtype=bool),
    }

    source_names = [f"cell_{j}" for j in range(n_src)]

    # Spatial map (n_src >> N_SENSORS, diagonal Sa): use Woodbury O(m²·n) fast path.
    # Standard path builds an (n×n) FIM and inverts it in O(n³) — infeasible for n≫m.
    if Sa.ndim == 1 and n_src > N_SENSORS:
        fisher, dof, posterior = analyze_information_content_spatial(
            G=G, Se=Se, Sa=Sa,
            obs_groups=obs_groups,
            source_names=source_names,
        )
        ablation = _run_ablation_spatial(G, Se, Sa, obs_groups)
    else:
        fisher, dof, posterior = analyze_information_content(
            G=G, Se=Se, Sa=Sa,
            obs_groups=obs_groups,
            source_names=source_names,
        )
        ablation = run_ablation_study(G=G, Se=Se, Sa=Sa, obs_groups=obs_groups)

    return {
        "fisher":    fisher,
        "dof":       dof,
        "posterior": posterior,
        "ablation":  ablation,
        "n_src":     n_src,
        "Sa":        Sa,
    }


def _run_ablation_spatial(
    G: np.ndarray,
    Se: np.ndarray,
    Sa: np.ndarray,
    obs_groups: dict[str, np.ndarray],
) -> dict:
    """Ablation study using the Woodbury fast path (avoids n×n matrices)."""
    from enforceflux.analysis.information_models import AblationResult

    group_names = list(obs_groups.keys())
    scenarios: list[tuple[str, list[str]]] = []
    for name in group_names:
        scenarios.append((name, [name]))
    for i in range(2, len(group_names) + 1):
        scenarios.append(("+".join(group_names[:i]), group_names[:i]))

    results: dict = {}
    for key, active_names in scenarios:
        combined_mask = np.zeros(N_SENSORS, dtype=bool)
        for name in active_names:
            combined_mask |= np.asarray(obs_groups[name], dtype=bool)
        sub_groups = {n: obs_groups[n] for n in active_names}
        fisher, dof, posterior = analyze_information_content_spatial(
            G=G, Se=Se, Sa=Sa,
            obs_groups=sub_groups,
        )
        results[key] = AblationResult(
            scenario=" + ".join(active_names),
            groups=active_names,
            fisher=fisher,
            dof=dof,
            posterior=posterior,
            dfs_total=dof.dfs_total,
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4 — figures
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_sensor_map(lons: np.ndarray, lats: np.ndarray, G: np.ndarray) -> None:
    """Footprint overlay with sensor and source markers (log scale)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from scipy.ndimage import gaussian_filter

    n_lat, n_lon = len(lats), len(lons)
    spatial_mode = G.shape[1] == n_lat * n_lon
    extent = (lons.min(), lons.max(), lats.min(), lats.max())
    colors = ["steelblue", "darkorange", "seagreen"]
    markers = ["^", "s", "s"]

    if spatial_mode:
        fig, axes = plt.subplots(1, N_SENSORS, figsize=(5 * N_SENSORS, 5))
        for i, ax in enumerate(axes):
            fp = G[i, :].reshape(n_lat, n_lon)
            fp_smooth = gaussian_filter(fp, sigma=0.5)
            pos = fp_smooth[fp_smooth > 0]
            if pos.size > 0:
                vmin_log = max(float(np.nanpercentile(pos, 5)), 1e-3)
                vmax_log = float(np.nanpercentile(pos, 99.5))
                norm = LogNorm(vmin=vmin_log, vmax=vmax_log)
            else:
                norm = None

            im = ax.imshow(
                np.where(fp_smooth > 0, fp_smooth, np.nan),
                origin="lower", extent=extent, aspect="auto",
                cmap="YlOrRd", norm=norm,
            )
            fig.colorbar(im, ax=ax, label="G [ng m⁻³ / (kg s⁻¹)]", shrink=0.7)

            ax.plot(SOURCE_LON, SOURCE_LAT, "*", color="red", ms=14, zorder=5,
                    label="Source")
            for j, inst in enumerate(SENSORS):
                ax.plot(inst.x, inst.y, markers[j], color=colors[j], ms=9,
                        zorder=5, mew=1.5, mec="black",
                        label=inst.id if j == i else "_nolegend_")
            inst = SENSORS[i]
            ax.plot(inst.x, inst.y, markers[i], color=colors[i],
                    ms=14, zorder=6, mew=2, mec="black")
            ax.set_title(f"Footprint: {inst.id} ({inst.tech_id})", fontsize=10)
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
            ax.legend(fontsize=7); ax.grid(alpha=0.3)

        fig.suptitle(
            "Sensor footprints — Gaussian plume (WSW→ENE wind, 2.6 m/s, log scale)\n"
            "Sensors 500 m downwind  |  Footprint = emission sensitivity [ng m⁻³ / (kg s⁻¹)]",
            fontsize=12,
        )
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(lons.min(), lons.max())
        ax.set_ylim(lats.min(), lats.max())
        ax.plot(SOURCE_LON, SOURCE_LAT, "*", color="red", ms=16, zorder=5,
                label="Source (100 kg hr⁻¹)")
        for j, inst in enumerate(SENSORS):
            ax.plot(inst.x, inst.y, markers[j], color=colors[j], ms=12,
                    zorder=5, mew=1.5, mec="black", label=inst.id)
            ax.annotate(inst.id, (inst.x, inst.y),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title("Sensor network (forward-sampled G)", fontsize=12)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "footprints.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[PLOT] {out.name}")
    plt.close(fig)


def _plot_dfs_spatial_map(
    G: np.ndarray,
    dfs_per_source: np.ndarray,
    uncertainty_reduction: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
) -> None:
    """2D spatial maps of DFS contribution and posterior uncertainty reduction per cell."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    n_lat, n_lon = len(lats), len(lons)
    if G.shape[1] != n_lat * n_lon:
        return

    dfs_map = dfs_per_source.reshape(n_lat, n_lon)
    unc_map = uncertainty_reduction.reshape(n_lat, n_lon) * 100.0
    extent = (lons.min(), lons.max(), lats.min(), lats.max())
    colors = ["steelblue", "darkorange", "seagreen"]
    markers = ["^", "s", "s"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- DFS per cell (log scale) ---
    ax = axes[0]
    pos = dfs_map[dfs_map > 0]
    if pos.size > 0:
        norm = LogNorm(vmin=max(pos.min(), 1e-6), vmax=pos.max())
    else:
        norm = None
    im = ax.imshow(
        np.where(dfs_map > 1e-9, dfs_map, np.nan),
        origin="lower", extent=extent, aspect="auto", cmap="viridis", norm=norm,
    )
    plt.colorbar(im, ax=ax, label="DFS per cell", shrink=0.8)
    ax.plot(SOURCE_LON, SOURCE_LAT, "*r", ms=14, zorder=6, label="Source")
    for j, inst in enumerate(SENSORS):
        ax.plot(inst.x, inst.y, markers[j], color=colors[j], ms=10,
                mew=1.5, mec="black", zorder=5, label=inst.id)
    ax.set_title("DFS contribution per source cell (log scale)", fontsize=11)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # --- Uncertainty reduction (%) ---
    # Log scale spanning 0.1 %–100 % so the strongly-constrained source AND the
    # faint along-axis ribbon are both visible (a linear 0–100 scale shows only
    # the source; a percentile vmax collapses the scale to near-zero).
    ax = axes[1]
    im2 = ax.imshow(
        np.where(unc_map > 0.1, unc_map, np.nan),
        origin="lower", extent=extent, aspect="auto",
        cmap="plasma", norm=LogNorm(vmin=0.1, vmax=100.0),
    )
    plt.colorbar(im2, ax=ax, label="Posterior uncertainty reduction (%)", shrink=0.8)
    ax.plot(SOURCE_LON, SOURCE_LAT, "*r", ms=14, zorder=6, label="Source")
    for j, inst in enumerate(SENSORS):
        ax.plot(inst.x, inst.y, markers[j], color=colors[j], ms=10,
                mew=1.5, mec="black", zorder=5, label=inst.id)
    ax.set_title("Posterior uncertainty reduction per cell (%)", fontsize=11)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(
        "Spatial information content — Gaussian plume Jacobian (WSW→ENE, 2.6 m/s)\n"
        "Source at ★; sensors 500 m downwind along plume axis ±25°",
        fontsize=12,
    )
    fig.tight_layout()
    out = OUT_DIR / "dfs_spatial_map.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[PLOT] {out.name}")
    plt.close(fig)


def _plot_analysis(res: dict) -> None:
    """DFS per source, averaging kernel, posterior uncertainty, ablation."""
    import matplotlib
    matplotlib.use("Agg")

    dof      = res["dof"]
    post     = res["posterior"]
    ablation = res["ablation"]
    n_src    = res["n_src"]

    # Spatial mode: averaging_kernel is 1D (diagonal of A); skip full 2D heatmap.
    spatial = n_src > N_SENSORS

    figs: dict[str, tuple] = {
        "forward_operator.png": plot_forward_operator(
            np.diag(res["fisher"].eigenvalues),
            title="Fisher eigenvalue spectrum",
        ),
        "dfs_per_source.png": plot_dfs_per_source(
            dof.dfs_per_source,
            title="DFS per source cell",
        ),
        "posterior_uncertainty.png": plot_posterior_uncertainty(
            post.prior_sigma, post.posterior_sigma,
            title="Prior vs posterior uncertainty",
        ),
        "ablation_comparison.png": plot_ablation_comparison(
            ablation, title="DFS by instrument subset",
        ),
    }
    if not spatial:
        figs["averaging_kernel.png"] = plot_averaging_kernel(
            dof.averaging_kernel,
            title=f"Averaging kernel  (DFS = {dof.dfs_total:.2f})",
        )

    for fname, (fig, _) in figs.items():
        out = OUT_DIR / fname
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[PLOT] {out.name}")

    import matplotlib.pyplot as plt
    plt.close("all")


def _print_summary(
    res: dict,
    G: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
) -> None:
    dof      = res["dof"]
    post     = res["posterior"]
    ablation = res["ablation"]
    n_src    = res["n_src"]

    n_lat, n_lon = len(lats), len(lons)
    spatial_mode = G.shape[1] == n_lat * n_lon

    print()
    print("=" * 60)
    print("Information content summary")
    print("=" * 60)
    print(f"  Source cells / time steps : {n_src}")
    print(f"  Sensors                   : {N_SENSORS}  ({', '.join(SENSOR_NAMES)})")
    print(f"  DFS (all sensors)         : {dof.dfs_total:.3f}  (max = {n_src})")
    # Skip auto-generated combined key (contains '+' twice) — show user-defined groups only
    for key, abl in ablation.items():
        if key.count("+") <= 1:
            print(f"    DFS [{key:8s}]         : {abl.dfs_total:.3f}")
    top_eig = res["fisher"].eigenvalues
    n_pos   = int((top_eig > 0).sum())
    print(f"  Fisher eigenvalues        : top-3 = {top_eig[:3]}  "
          f"(+ve: {n_pos}/{len(top_eig)})")

    if spatial_mode:
        # Source cell attribution
        src_lon_idx = int(np.argmin(np.abs(lons - SOURCE_LON)))
        src_lat_idx = int(np.argmin(np.abs(lats - SOURCE_LAT)))
        src_flat = src_lat_idx * n_lon + src_lon_idx
        G_src    = G[:, src_flat]
        dfs_src  = float(dof.dfs_per_source[src_flat])
        unc_src  = float(post.uncertainty_reduction[src_flat])
        print(f"\n  Source cell ({lons[src_lon_idx]:.3f}°, {lats[src_lat_idx]:.3f}°):")
        print(f"    G per sensor [ng m⁻³/(kg s⁻¹)] : {G_src}")
        print(f"    DFS contribution               : {dfs_src:.2e}")
        print(f"    Posterior uncertainty reduction: {unc_src*100:.4f} %")

        # Top 5 most constrained cells
        top5 = np.argsort(dof.dfs_per_source)[::-1][:5]
        print(f"\n  Top-5 constrained cells (DFS contribution):")
        for k, idx in enumerate(top5):
            lat_k = lats[idx // n_lon]
            lon_k = lons[idx % n_lon]
            print(
                f"    {k+1}. ({lon_k:.3f}°, {lat_k:.3f}°) "
                f"DFS={dof.dfs_per_source[idx]:.4f}  "
                f"G_max={G[:, idx].max():.2e}  "
                f"unc_red={post.uncertainty_reduction[idx]*100:.1f}%"
            )

        # Distribution stats
        unc_nonzero = post.uncertainty_reduction[post.uncertainty_reduction > 1e-6]
        n_above_50  = int((post.uncertainty_reduction > 0.5).sum())
        n_above_10  = int((post.uncertainty_reduction > 0.1).sum())
        print(f"\n  Global statistics across all {n_src} cells:")
        print(f"    Mean unc. reduction (all cells)     : {np.mean(post.uncertainty_reduction)*100:.2f} %")
        if unc_nonzero.size:
            print(f"    Mean unc. reduction (nonzero cells) : {np.mean(unc_nonzero)*100:.1f} %")
        print(f"    Cells with > 50 % reduction         : {n_above_50}")
        print(f"    Cells with > 10 % reduction         : {n_above_10}")
    else:
        print(f"  Mean uncertainty reduction: {np.mean(post.uncertainty_reduction)*100:.1f} %")
        print(f"  Prior σ  [first 5]        : {post.prior_sigma[:5]}")
        print(f"  Post  σ  [first 5]        : {post.posterior_sigma[:5]}")

    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Single-source instrument OSSE")
    print("=" * 60)

    # ── Stage 1: build transport Jacobian G ──────────────────────────────────
    # Gaussian plume model: March 31 ERA5 wind (WSW→ENE, bearing 68°, 2.6 m/s),
    # Pasquill-Gifford class C-D dispersion, sensors 500 m downwind.
    lons, lats = _make_analysis_domain()
    print(f"[G]    Gaussian plume — {len(lons)}×{len(lats)} cells at 0.005°")
    G = _build_G_analytical(lons, lats)

    # ── Stage 2: synthetic observations ──────────────────────────────────────
    print("[OBS]  Simulating instrument observations …")
    y_clean, y_obs, valid = _simulate_observations(G)
    n_valid = int(valid.sum())
    print(f"[OBS]  Valid observations: {n_valid} / {valid.size}")

    # ── Stage 3: information content analysis ─────────────────────────────────
    print("[ANA]  Running information content analysis …")
    res = _run_analysis(G)
    _print_summary(res, G, lons, lats)

    # ── Stage 4: figures ──────────────────────────────────────────────────────
    print("[PLOT] Generating figures …")
    _plot_sensor_map(lons, lats, G)
    _plot_analysis(res)
    _plot_dfs_spatial_map(
        G,
        res["dof"].dfs_per_source,
        res["posterior"].uncertainty_reduction,
        lons, lats,
    )

    print(f"\nDone. Outputs written to: {OUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
