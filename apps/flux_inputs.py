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


def _assert_operator_obs_units(jacobian_units: str, y_units: str) -> dict[str, str]:
    """Enforce that G maps state-flux units (kg s-1) to the observation units.

    ``jacobian.npz['units']`` is a string of the shape ``"<obs> / (<state>)"``
    (e.g. ``"ng m-3 / (kg s-1)"``). We split on the first ``" / "`` and check:

      * the obs half matches ``y_obs.units`` exactly (case- and space-sensitive
        so units like ``"kg m-3"`` and ``"ng m-3"`` don't silently mix);
      * the state half canonicalises to ``kg s-1`` so the flux stage's state
        vector is unambiguously per-cell emission rate.

    Returns a metadata dict describing the checked units, to be stamped into
    ``obs_meta`` / ``summary.json`` so downstream users can reproduce which
    physical products were paired.
    """
    def _norm(s: str) -> str:
        return " ".join((s or "").strip().split())

    j = _norm(jacobian_units)
    y = _norm(y_units)
    if not j:
        raise ValueError(
            "Operator jacobian.npz is missing a `units` field — the flux "
            "inversion cannot verify units × Jacobian == observations. "
            "Rebuild the dispersion output with a units-tagged operator."
        )
    if " / " not in j:
        raise ValueError(
            f"Jacobian units {j!r} must be formatted '<obs> / (<state>)' "
            f"so the flux stage can check compatibility with y_obs."
        )
    obs_side, state_side = (part.strip() for part in j.split(" / ", 1))
    state_side = state_side.strip("()")
    if _norm(state_side) != "kg s-1":
        raise ValueError(
            f"Jacobian state units {state_side!r} must be 'kg s-1' — the flux "
            f"stage's state vector is per-source total emission rate in kg/s."
        )
    if not y:
        raise ValueError(
            "y_obs is missing a `units` attribute in the instrument NetCDF; "
            "instrument stage must stamp units so unit checks are enforceable."
        )
    if _norm(obs_side) != y:
        raise ValueError(
            "Units mismatch between Jacobian and observations:\n"
            f"  Jacobian obs half : {obs_side!r}\n"
            f"  y_obs units       : {y!r}\n"
            "Rebuild the dispersion operator with a matching units contract, "
            "or convert the instrument stage's obs to the operator's obs units."
        )
    return {
        "jacobian_units": j,
        "y_obs_units": y,
        "state_units": _norm(state_side),
        "obs_units": _norm(obs_side),
    }


def _time_resolved_G(
    cvar,
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_sources: int,
    site_x: np.ndarray,
    site_y: np.ndarray,
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
    n_sites = len(site_x)
    G = np.zeros((n_sites * n_time_obs, n_sources * n_flux), dtype=float)
    for j in range(n_sources):
        for i in range(n_sites):
            step = sample_step_response(
                cvar,
                x,
                y,
                release_index=j,
                level_index=level_index,
                site_x=float(site_x[i]),
                site_y=float(site_y[i]),
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


def build_from_prebuilt_operator_with_instrument(
    cfg: dict[str, Any], dispersion_up: Any, instrument_netcdf: Path,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, list[str], str, dict[str, Any], int,
    np.ndarray, np.ndarray, dict[str, Any],
]:
    """Pair an operator-mode Jacobian with pseudo-observations from instruments.

    Operator Jacobians are written time-major/receptor-minor; instrument files
    are flattened receptor-major/time-minor to match the standard flux path.
    """
    from netCDF4 import Dataset
    from flux_helpers import find_var

    jac = np.load(dispersion_up.file("jacobian"))
    G_fine = np.asarray(jac["G"], dtype=float)
    jacobian_units = str(jac["units"]) if "units" in jac.files else ""
    mapping = load_mapping(dispersion_up.file("basis_mapping"))
    W = np.asarray(mapping.W, dtype=float)
    counts = W.sum(axis=1)
    if np.any(counts == 0) or G_fine.shape[1] != W.shape[1]:
        raise ValueError("Operator Jacobian and basis mapping are incompatible")

    with Dataset(instrument_netcdf) as ds:
        y_name = find_var(ds, ("y_obs", "observation", "observations"))
        if y_name is None:
            raise KeyError("Instrument NetCDF must include y_obs/observation")
        y_grid = np.asarray(ds.variables[y_name][:], dtype=float)
        y_units = str(getattr(ds.variables[y_name], "units", "") or "")
        if y_grid.ndim != 2:
            raise ValueError(f"Expected y_obs shape (time, instrument), got {y_grid.shape}")
        n_time, n_inst = y_grid.shape
        valid_name = find_var(ds, ("valid_mask",))
        valid_grid = np.asarray(ds.variables[valid_name][:], dtype=bool) if valid_name else np.isfinite(y_grid)
        variance_name = find_var(ds, ("noise_variance",))
        if variance_name is None:
            sigma = float(cfg.get("observations", {}).get("default_sigma", 1.0))
            variance_grid = np.full_like(y_grid, sigma ** 2)
        else:
            variance_grid = np.asarray(ds.variables[variance_name][:], dtype=float)
            if variance_grid.shape != y_grid.shape:
                raise ValueError("noise_variance must have the same shape as y_obs")

    # A time-resolved operator can only cover the window its turbulence
    # intervals were built for. Selecting the matching observation frames is
    # explicit rather than inferred: obs.nc carries a frame index, not seconds,
    # so an index range is the only unambiguous way to say it.
    index_range = cfg.get("input", {}).get("time_index_range")
    if index_range is not None:
        i0, i1 = (int(v) for v in index_range)
        if not 0 <= i0 < i1 <= n_time:
            raise ValueError(
                f"input.time_index_range {list(index_range)} is not a valid "
                f"half-open range within the {n_time} observation frames"
            )
        y_grid = y_grid[i0:i1]
        valid_grid = valid_grid[i0:i1]
        variance_grid = variance_grid[i0:i1]
        n_time = i1 - i0
        obs_meta_time_index_range = [i0, i1]
    else:
        obs_meta_time_index_range = None

    # ── units contract (state × Jacobian = y_obs) ─────────────────────────
    # Jacobian.units must parse as "<obs_units> / (<state_units>)". We require
    # <obs_units> == y_obs.units and <state_units> to be a per-source-flux unit
    # so the inversion's state vector reads as kg s-1 (per cell) unambiguously.
    # Any mismatch is fatal — silently rescaling here would move a physical
    # error into a numeric one that reads as an underdetermined inversion.
    units_meta = _assert_operator_obs_units(jacobian_units, y_units)

    # A backward LPDM footprint represents one window-integrated observation
    # per instrument.  Reduce the LES pseudo-observation time series over the
    # same window before pairing it with that Jacobian.
    time_reduce = str(cfg.get("input", {}).get("time_reduce", "none")).lower()
    if G_fine.shape[0] == n_inst and n_time > 1 and time_reduce == "mean":
        usable = valid_grid & np.isfinite(y_grid) & np.isfinite(variance_grid)
        counts_by_inst = usable.sum(axis=0)
        if not np.all(counts_by_inst > 0):
            raise ValueError("Cannot window-average an instrument with no valid LES samples")
        weights = usable.astype(float)
        y_grid = (np.where(usable, y_grid, 0.0).sum(axis=0) / counts_by_inst)[None, :]
        variance_grid = (
            (np.where(usable, variance_grid, 0.0).sum(axis=0) / counts_by_inst**2)[None, :]
        )
        valid_grid = np.ones((1, n_inst), dtype=bool)
        n_time = 1

    if time_reduce == "mean" and G_fine.shape[0] == n_time * n_inst and n_time > 1:
        # A time-resolved operator has a row per (interval, instrument), so the
        # averaging branch above never fires. Silently keeping the full time
        # series under a config that says "mean" would misreport what was
        # inverted; make the operator and the config agree explicitly.
        raise ValueError(
            f"input.time_reduce is 'mean', but the operator is time-resolved "
            f"({G_fine.shape[0]} rows = {n_time} intervals x {n_inst} "
            "instruments). Set input.time_reduce: none to invert the time "
            "series, or rebuild the operator with interval_reduce: mean."
        )
    if G_fine.shape[0] != n_time * n_inst:
        selected = (
            f" (after selecting frames {obs_meta_time_index_range})"
            if obs_meta_time_index_range else ""
        )
        raise ValueError(
            f"Operator has {G_fine.shape[0]} rows, but instrument file has "
            f"{n_time} × {n_inst} observations{selected}. Set "
            "input.time_reduce: mean for a window-integrated backward LPDM "
            "operator, or — for a time-resolved operator — set "
            "input.time_index_range to the frames its intervals were built "
            f"for ({G_fine.shape[0] // n_inst} of them)."
        )
    row_order = np.asarray([t * n_inst + i for i in range(n_inst) for t in range(n_time)])
    G_coarse = G_fine[row_order] @ (W / counts[:, None]).T
    y_flat = y_grid.T.reshape(-1)
    variance_flat = variance_grid.T.reshape(-1)
    valid = valid_grid.T.reshape(-1) & np.isfinite(y_flat) & np.isfinite(variance_flat)
    if not np.any(valid) or np.any(variance_flat[valid] <= 0):
        raise ValueError("Instrument observations must include valid, positive variances")

    inv_cfg = cfg.get("inversion", {}) or {}
    prior_cfg = inv_cfg.get("prior_covariance") or {}
    n_coarse = W.shape[0]
    x_prior = np.full(n_coarse, float(inv_cfg.get("prior_flux_kg_s", 0.0)))
    prior_model = str(prior_cfg.get("model", "diagonal")).strip().lower()
    L_B_m = float(prior_cfg.get("L_B_m", 0.0))
    sigma_kg_s = float(prior_cfg.get("sigma_kg_s", 1.0e-6))
    Sa = (build_prior_covariance(mapping, sigma_kg_s, L_B_m, model="exponential")
          if prior_model == "gaussian_process"
          else np.full(n_coarse, float(inv_cfg.get("prior_variance", sigma_kg_s ** 2))))
    with Dataset(dispersion_up.file("truth_field")) as ds:
        L_true_m = float(getattr(ds, "L_true_m", 0.0))
        truth_fine = np.asarray(ds.variables["F_true"][:], dtype=float).ravel() * mapping.fine_cell_areas_m2

    # Each observation's sensitivity to a spatially uniform total flux
    # [ng m-3 / (kg s-1)]. The multiplicative representation-error term needs
    # exactly this, and only the code holding the basis mapping can form it —
    # for a one-cell state it is the state column itself, but for a resolved
    # state no single column carries it.
    uniform_template = (
        np.asarray(mapping.fine_cell_areas_m2, dtype=float)
        / float(np.sum(mapping.fine_cell_areas_m2))
    )
    uniform_total_sensitivity = (G_fine[row_order] @ uniform_template)[valid]

    obs_meta = {
        "mode": "instrument_netcdf", "input_mode": "instrument_netcdf",
        "instrument_netcdf": str(instrument_netcdf), "y_variable": y_name,
        "n_time": int(n_time), "n_flux_windows": 1,
        "uniform_total_sensitivity": uniform_total_sensitivity.tolist(),
        "n_observations_total": int(y_flat.size), "n_observations_used": int(valid.sum()),
        "units": units_meta,
        "time_index_range": obs_meta_time_index_range,
        "time_reduce": time_reduce,
    }
    diagnostics = {
        "L_true_m": L_true_m, "L_B_m": L_B_m,
        "inverse_crime_flag": bool(L_B_m > 0 and abs(L_B_m - L_true_m) <= 1e-6 * max(L_B_m, L_true_m)),
        "prior_covariance_model": prior_model, "n_fine_cells": int(W.shape[1]),
        "n_coarse_cells": int(n_coarse), "observation_source": "instrument_operator",
    }
    total_only = bool(inv_cfg.get("total_only", False))
    if total_only:
        template_name = str(inv_cfg.get("total_template", "truth_shape"))
        q_true = float(truth_fine.sum())
        if q_true <= 0.0:
            raise ValueError("Total source flux must be positive")
        if template_name == "truth_shape":
            template = truth_fine / q_true
        elif template_name == "uniform":
            # A spatially uniform flux density assigns total emissions in
            # proportion to cell area.  It is deliberately independent of the
            # heterogeneous truth and therefore suitable for paired sensor
            # comparisons without oracle knowledge of the source pattern.
            areas = np.asarray(mapping.fine_cell_areas_m2, dtype=float)
            template = areas / areas.sum()
        else:
            raise ValueError(
                "total_template must be 'truth_shape' or 'uniform', "
                f"got {template_name!r}"
            )
        G_coarse = (G_fine[row_order] @ template).reshape(-1, 1)
        x_prior = np.array([float(inv_cfg.get("prior_total_flux_kg_s", 0.0))])
        Sa = np.array([float(inv_cfg.get("prior_total_variance", 1.0e-4))])
        names = ["Q_total"]
    else:
        names = [f"coarse_{i:05d}" for i in range(n_coarse)]
    diagnostics["total_only"] = total_only
    diagnostics["inversion_template"] = template_name if total_only else "coarse_gp"
    return G_coarse[valid], y_flat[valid], variance_flat[valid], names, "concentration", obs_meta, 1, x_prior, Sa, diagnostics


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

    site_x = np.array([float(r["x_m"]) for r in receptors], dtype=float)
    site_y = np.array([float(r["y_m"]) for r in receptors], dtype=float)

    with Dataset(sim_nc) as ds:
        vname, x, y, cvar, n_sources, source_names = prepare_sim_transport(ds, variable_name_cfg)
        n_time = infer_time_size(cvar)
        # One flux window per simulation timestep; observations share that base.
        n_flux = n_time
        n_time_obs = n_time

        G = _time_resolved_G(
            cvar,
            x,
            y,
            n_sources=n_sources,
            site_x=site_x,
            site_y=site_y,
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
    from enforceflux.coordinates import frame_from_canonical

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
        attrs = {name: ds_i.getncattr(name) for name in ds_i.ncattrs()}
        if str(attrs.get("Conventions", "")) != "EnforceFlux-canonical-3":
            raise ValueError("Instrument NetCDF must use EnforceFlux-canonical-3")
        frame_from_canonical(attrs)
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

        x_name = find_var(ds_i, ("instrument_x_m",))
        y_name = find_var(ds_i, ("instrument_y_m",))
        if x_name is None or y_name is None:
            raise KeyError(
                "Instrument NetCDF must include instrument_x_m and instrument_y_m variables"
            )
        inst_x = np.asarray(ds_i.variables[x_name][:], dtype=float).reshape(-1)
        inst_y = np.asarray(ds_i.variables[y_name][:], dtype=float).reshape(-1)
        if len(inst_x) != n_inst or len(inst_y) != n_inst:
            raise ValueError("Instrument coordinate vectors must match instrument dimension length")

    with Dataset(sim_nc) as ds_s:
        vname, x, y, cvar, n_sources, source_names = prepare_sim_transport(ds_s, variable_name_cfg)
        n_time_s = infer_time_size(cvar)
        # Flux windows are set by the simulation's time base (kernel length);
        # observations may run longer — lags past the kernel contribute zero
        # (the plume's transport memory is finite), so no timestep reuse.
        n_flux = n_time_s

        G = _time_resolved_G(
            cvar,
            x,
            y,
            n_sources=n_sources,
            site_x=inst_x,
            site_y=inst_y,
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
