from pathlib import Path
from typing import Any

import numpy as np

from flux_helpers import (
    build_y_and_se,
    infer_time_size,
    prepare_sim_transport,
    sample_step_response,
    step_to_impulse,
    toeplitz_convolution_block,
)

# --- source-heterogeneity OSSE (M2) ---
from enforceflux.source_fields.basis import load_mapping
from enforceflux.source_fields.prior import build_prior_covariance
# --- end M2 ---


def _time_resolved_G(
    cvar,
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    n_sources: int,
    site_lons: np.ndarray,
    site_lats: np.ndarray,
    level_index: int,
    n_time_kernel: int,
    n_time_obs: int,
    n_flux: int,
) -> np.ndarray:
    """Assemble the block-Toeplitz transport matrix G.

    For every (receptor ``i``, source ``j``) pair the simulation gives a step
    response (concentration at the receptor from a sustained unit-rate release);
    first-differencing turns it into the impulse kernel ``h_ij(tau)``, and each
    kernel becomes a lower-triangular Toeplitz block. Stacking the blocks maps
    the flat emission state (source-major, window-minor; length
    ``n_sources * n_flux``) to the flat observation vector (receptor-major,
    time-minor; length ``n_sites * n_time_obs``).
    """
    n_sites = len(site_lons)
    G = np.zeros((n_sites * n_time_obs, n_sources * n_flux), dtype=float)
    for j in range(n_sources):
        for i in range(n_sites):
            step = sample_step_response(
                cvar,
                lons,
                lats,
                release_index=j,
                level_index=level_index,
                lon=float(site_lons[i]),
                lat=float(site_lats[i]),
                n_time=n_time_kernel,
            )
            impulse = step_to_impulse(step)
            block = toeplitz_convolution_block(impulse, n_time_obs, n_flux)
            G[i * n_time_obs : (i + 1) * n_time_obs, j * n_flux : (j + 1) * n_flux] = block
    return G


# --- source-heterogeneity OSSE (M2) ---
def build_from_prebuilt_operator(
    cfg: dict[str, Any],
    dispersion_up: Any,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, list[str], str, dict[str, Any], int,
    np.ndarray, np.ndarray, dict[str, Any],
]:
    """Consume a prebuilt fine-grid Jacobian + basis mapping from dispersion.

    Returns G_coarse, y_obs (synthesized from H_fine @ x_true + noise),
    Se, source_names (coarse cell ids), variable-name placeholder, obs_meta,
    n_flux (=1), x_prior (coarse), S_a (coarse), and diagnostics metadata
    (L_true_m, L_B_m, inverse_crime_flag).
    """
    from netCDF4 import Dataset

    jac = np.load(dispersion_up.file("jacobian"))
    G_fine = np.asarray(jac["G"], dtype=float)
    column_labels = [str(c) for c in np.asarray(jac["column_labels"]).tolist()]

    mapping = load_mapping(dispersion_up.file("basis_mapping"))
    W = np.asarray(mapping.W, dtype=float)
    n_fine_per_coarse = W.sum(axis=1)
    if np.any(n_fine_per_coarse == 0):
        raise ValueError("basis mapping has a coarse cell with no fine children")
    # Distributor form: per-fine-cell weights within a coarse parent (rows sum
    # to 1 across the parent's children). H_coarse[:, c] is the observation a
    # unit per-fine-cell emission distributed uniformly through coarse cell c
    # would produce — the right transform for a per-cell inversion state.
    W_dist = W / n_fine_per_coarse[:, None]
    if G_fine.shape[1] != W.shape[1]:
        raise ValueError(
            f"jacobian has {G_fine.shape[1]} source columns but basis mapping "
            f"expects {W.shape[1]} fine cells"
        )
    G_coarse = G_fine @ W_dist.T  # (n_obs, n_coarse)

    truth_path = dispersion_up.file("truth_field")
    with Dataset(truth_path) as ds:
        F_true = np.asarray(ds.variables["F_true"][:], dtype=float)
        L_true_m = float(getattr(ds, "L_true_m", 0.0))
    x_true_fine = F_true.ravel() * mapping.fine_cell_areas_m2  # emission per fine cell

    inv_cfg = cfg.get("inversion", {}) or {}
    obs_cfg = cfg.get("observations", {}) or {}
    seed = int(obs_cfg.get("random_seed", 42))
    sigma_default = float(obs_cfg.get("default_sigma", 1.0))
    add_noise = bool(obs_cfg.get("add_noise", False))

    y_clean = G_fine @ x_true_fine
    Se = np.full(y_clean.shape[0], sigma_default ** 2, dtype=float)
    if add_noise:
        rng = np.random.default_rng(seed)
        y_obs = y_clean + rng.normal(0.0, sigma_default, size=y_clean.shape[0])
    else:
        y_obs = y_clean.copy()

    n_coarse = W.shape[0]
    prior_mean = float(inv_cfg.get("prior_flux_kg_s", 0.0))
    x_prior = np.full(n_coarse, prior_mean, dtype=float)

    prior_cov_cfg = inv_cfg.get("prior_covariance") or {}
    prior_model = str(prior_cov_cfg.get("model", "diagonal")).strip().lower()
    L_B_m = float(prior_cov_cfg.get("L_B_m", 0.0))
    sigma_kg_s = float(prior_cov_cfg.get("sigma_kg_s", 1.0e-6))
    if prior_model == "gaussian_process":
        Sa = build_prior_covariance(mapping, sigma_kg_s, L_B_m, model="exponential")
    else:
        prior_var = float(inv_cfg.get("prior_variance", sigma_kg_s ** 2))
        Sa = np.full(n_coarse, prior_var, dtype=float)

    source_names = [f"coarse_{i:05d}" for i in range(n_coarse)]
    inverse_crime = bool(L_B_m > 0 and abs(L_B_m - L_true_m) <= 1e-6 * max(L_B_m, L_true_m))

    obs_meta = {
        "mode": "prebuilt_operator",
        "input_mode": "prebuilt_operator",
        "n_observations_total": int(y_obs.size),
        "n_observations_used": int(y_obs.size),
        "n_time": 1,
        "n_flux_windows": 1,
        "add_noise": add_noise,
        "random_seed": seed,
        "default_sigma": sigma_default,
    }
    diagnostics = {
        "L_true_m": L_true_m,
        "L_B_m": L_B_m,
        "inverse_crime_flag": inverse_crime,
        "prior_covariance_model": prior_model,
        "n_fine_cells": int(W.shape[1]),
        "n_coarse_cells": int(n_coarse),
        "fine_column_count": len(column_labels),
    }
    return G_coarse, y_obs, Se, source_names, "concentration", obs_meta, 1, x_prior, Sa, diagnostics
# --- end M2 ---


def build_from_receptors_mode(
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], str, Path, dict[str, Any], int]:
    from netCDF4 import Dataset

    input_cfg = cfg.get("input", {})
    sim_nc = Path(input_cfg.get("simulation_netcdf", "")).expanduser().resolve()
    if not sim_nc.exists():
        raise FileNotFoundError(f"Simulation NetCDF not found: {sim_nc}")

    level_index = int(input_cfg.get("level_index", 0))
    variable_name_cfg = input_cfg.get("variable_name")

    receptors = cfg.get("receptors", [])
    if not receptors:
        raise ValueError("At least one receptor is required in receptors[] for input.mode=simulation_receptors")

    site_lons = np.array([float(r["lon"]) for r in receptors], dtype=float)
    site_lats = np.array([float(r["lat"]) for r in receptors], dtype=float)

    with Dataset(sim_nc) as ds:
        vname, lons, lats, cvar, n_sources, source_names = prepare_sim_transport(ds, variable_name_cfg)
        n_time = infer_time_size(cvar)
        # One flux window per simulation timestep; observations share that base.
        n_flux = n_time
        n_time_obs = n_time

        G = _time_resolved_G(
            cvar,
            lons,
            lats,
            n_sources=n_sources,
            site_lons=site_lons,
            site_lats=site_lats,
            level_index=level_index,
            n_time_kernel=n_time,
            n_time_obs=n_time_obs,
            n_flux=n_flux,
        )

    y_obs, Se, obs_meta = build_y_and_se(
        cfg, G, receptors, n_time_obs=n_time_obs, n_sources=n_sources, n_flux=n_flux
    )
    obs_meta["input_mode"] = "simulation_receptors"
    obs_meta["n_time"] = int(n_time_obs)
    obs_meta["n_flux_windows"] = int(n_flux)
    obs_meta["n_observations_total"] = int(len(y_obs))
    obs_meta["n_observations_used"] = int(len(y_obs))

    return G, y_obs, Se, source_names, vname, sim_nc, obs_meta, n_flux


def build_from_instrument_mode(
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], str, Path, dict[str, Any], int]:
    from netCDF4 import Dataset

    input_cfg = cfg.get("input", {})
    sim_nc = Path(input_cfg.get("simulation_netcdf", "")).expanduser().resolve()
    if not sim_nc.exists():
        raise FileNotFoundError(f"Simulation NetCDF not found: {sim_nc}")

    inst_nc = Path(input_cfg.get("instrument_netcdf", "")).expanduser().resolve()
    if not inst_nc.exists():
        raise FileNotFoundError(f"Instrument NetCDF not found: {inst_nc}")

    level_index = int(input_cfg.get("level_index", 0))
    variable_name_cfg = input_cfg.get("variable_name")

    from flux_helpers import find_var

    with Dataset(inst_nc) as ds_i:
        y_name = find_var(ds_i, ("y_obs", "observation", "observations"))
        if y_name is None:
            raise KeyError("Instrument NetCDF must include y_obs/observation variable")
        y_grid = np.asarray(ds_i.variables[y_name][:], dtype=float)
        if y_grid.ndim != 2:
            raise ValueError(f"Expected instrument y_obs shape (time, instrument), got {y_grid.shape}")
        n_time_i, n_inst = y_grid.shape

        valid_name = find_var(ds_i, ("valid_mask",))
        valid_grid = np.asarray(ds_i.variables[valid_name][:], dtype=bool) if valid_name else np.isfinite(y_grid)

        nvar_name = find_var(ds_i, ("noise_variance",))
        if nvar_name is not None:
            se_grid = np.asarray(ds_i.variables[nvar_name][:], dtype=float)
            if se_grid.shape != y_grid.shape:
                raise ValueError(
                    f"noise_variance shape {se_grid.shape} must match y_obs shape {y_grid.shape}"
                )
        else:
            sigma_default = float(cfg.get("observations", {}).get("default_sigma", 1.0))
            se_grid = np.full_like(y_grid, sigma_default**2, dtype=float)

        lon_name = find_var(ds_i, ("instrument_lon", "lon", "longitude"))
        lat_name = find_var(ds_i, ("instrument_lat", "lat", "latitude"))
        if lon_name is None or lat_name is None:
            raise KeyError(
                "Instrument NetCDF must include instrument_lon and instrument_lat variables"
            )
        inst_lons = np.asarray(ds_i.variables[lon_name][:], dtype=float).reshape(-1)
        inst_lats = np.asarray(ds_i.variables[lat_name][:], dtype=float).reshape(-1)
        if len(inst_lons) != n_inst or len(inst_lats) != n_inst:
            raise ValueError("Instrument coordinate vectors must match instrument dimension length")

    with Dataset(sim_nc) as ds_s:
        vname, lons, lats, cvar, n_sources, source_names = prepare_sim_transport(ds_s, variable_name_cfg)
        n_time_s = infer_time_size(cvar)
        # Flux windows are set by the simulation's time base (kernel length);
        # observations may run longer — lags past the kernel contribute zero
        # (the plume's transport memory is finite), so no timestep reuse.
        n_flux = n_time_s

        G = _time_resolved_G(
            cvar,
            lons,
            lats,
            n_sources=n_sources,
            site_lons=inst_lons,
            site_lats=inst_lats,
            level_index=level_index,
            n_time_kernel=n_time_s,
            n_time_obs=n_time_i,
            n_flux=n_flux,
        )

    # Reorder instrument grids from (time, inst) to instrument-major, time-minor
    # so the flat observation index matches G's row order (i * n_time_i + t).
    y_flat = y_grid.T.reshape(-1)
    valid_flat = valid_grid.T.reshape(-1) & np.isfinite(y_flat)
    se_flat = se_grid.T.reshape(-1)

    se_valid = se_flat[valid_flat]
    y_valid = y_flat[valid_flat]
    if np.any(se_valid <= 0):
        raise ValueError("All observation variances must be positive in instrument mode")

    G_valid = G[valid_flat]

    obs_meta = {
        "mode": "instrument_netcdf",
        "input_mode": "instrument_netcdf",
        "instrument_netcdf": str(inst_nc),
        "y_variable": y_name,
        "n_time": int(n_time_i),
        "n_flux_windows": int(n_flux),
        "n_observations_total": int(y_flat.size),
        "n_observations_used": int(valid_flat.sum()),
    }
    return G_valid, y_valid, se_valid, source_names, vname, sim_nc, obs_meta, n_flux
