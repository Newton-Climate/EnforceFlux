from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from flux_helpers import (
    build_y_and_se,
    extract_field_for_release,
    infer_time_size,
    prepare_sim_transport,
    sample_nearest,
)


def build_from_receptors_mode(
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], str, Path, dict[str, Any]]:
    from netCDF4 import Dataset

    input_cfg = cfg.get("input", {})
    sim_nc = Path(input_cfg.get("simulation_netcdf", "")).expanduser().resolve()
    if not sim_nc.exists():
        raise FileNotFoundError(f"Simulation NetCDF not found: {sim_nc}")

    time_index = int(input_cfg.get("time_index", 0))
    level_index = int(input_cfg.get("level_index", 0))
    variable_name_cfg = input_cfg.get("variable_name")

    receptors = cfg.get("receptors", [])
    if not receptors:
        raise ValueError("At least one receptor is required in receptors[] for input.mode=simulation_receptors")

    with Dataset(sim_nc) as ds:
        vname, lons, lats, cvar, n_sources, source_names = prepare_sim_transport(ds, variable_name_cfg)

        G = np.zeros((len(receptors), n_sources), dtype=float)
        for j in range(n_sources):
            field = extract_field_for_release(
                cvar,
                release_index=j,
                time_index=time_index,
                level_index=level_index,
            )
            for i, rec in enumerate(receptors):
                G[i, j] = sample_nearest(
                    field,
                    lons,
                    lats,
                    float(rec["lon"]),
                    float(rec["lat"]),
                )

    y_obs, Se, obs_meta = build_y_and_se(cfg, G, receptors)
    obs_meta["input_mode"] = "simulation_receptors"
    obs_meta["n_observations_total"] = int(len(y_obs))
    obs_meta["n_observations_used"] = int(len(y_obs))

    return G, y_obs, Se, source_names, vname, sim_nc, obs_meta


def build_from_instrument_mode(
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], str, Path, dict[str, Any]]:
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
    reuse_last_sim_time = bool(input_cfg.get("reuse_last_sim_time", True))

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

        if n_time_i > n_time_s and not reuse_last_sim_time:
            raise ValueError(
                f"Instrument has {n_time_i} timesteps but simulation has {n_time_s}. "
                "Set input.reuse_last_sim_time: true to reuse final simulation timestep."
            )

        G_grid = np.zeros((n_time_i, n_inst, n_sources), dtype=float)
        for t in range(n_time_i):
            t_sim = t if t < n_time_s else (n_time_s - 1)
            for j in range(n_sources):
                field = extract_field_for_release(
                    cvar,
                    release_index=j,
                    time_index=t_sim,
                    level_index=level_index,
                )
                for i in range(n_inst):
                    G_grid[t, i, j] = sample_nearest(field, lons, lats, inst_lons[i], inst_lats[i])

    y_flat = y_grid.reshape(-1)
    valid_flat = valid_grid.reshape(-1) & np.isfinite(y_flat)
    se_flat = se_grid.reshape(-1)
    se_valid = se_flat[valid_flat]
    y_valid = y_flat[valid_flat]

    if np.any(se_valid <= 0):
        raise ValueError("All observation variances must be positive in instrument mode")

    G_flat = G_grid.reshape(-1, n_sources)
    G_valid = G_flat[valid_flat]

    obs_meta = {
        "mode": "instrument_netcdf",
        "input_mode": "instrument_netcdf",
        "instrument_netcdf": str(inst_nc),
        "y_variable": y_name,
        "n_observations_total": int(y_flat.size),
        "n_observations_used": int(valid_flat.sum()),
        "reuse_last_sim_time": reuse_last_sim_time,
    }
    return G_valid, y_valid, se_valid, source_names, vname, sim_nc, obs_meta
