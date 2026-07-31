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
