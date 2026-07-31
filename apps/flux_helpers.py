from pathlib import Path
from typing import Any

import numpy as np


def require_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: pip install pyyaml") from exc
    return yaml


def find_var(ds, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in ds.variables:
            return name
    return None


def sample_nearest(field_2d: np.ndarray, lons: np.ndarray, lats: np.ndarray, lon: float, lat: float) -> float:
    iy = int(np.argmin(np.abs(lats - lat)))
    ix = int(np.argmin(np.abs(lons - lon)))

    if field_2d.shape == (len(lats), len(lons)):
        return float(field_2d[iy, ix])
    if field_2d.shape == (len(lons), len(lats)):
        return float(field_2d[ix, iy])

    raise ValueError(
        "Unable to map concentration field to lat/lon axes. "
        f"Field shape={field_2d.shape}, lat={len(lats)}, lon={len(lons)}"
    )


def parse_source_names(ds, n_sources: int) -> list[str]:
    raw = getattr(ds, "source_ids", "")
    if raw:
        ids = [s.strip() for s in str(raw).split(",") if s.strip()]
        if len(ids) == n_sources:
            return ids
    return [f"source_{i}" for i in range(n_sources)]


def extract_field_for_release(
    var,
    release_index: int,
    *,
    time_index: int,
    level_index: int,
) -> np.ndarray:
    dims = [d.lower() for d in var.dimensions]

    idx: list[object] = []
    for d in dims:
        if d in ("latitude", "lat", "ylat", "longitude", "lon", "xlon"):
            idx.append(slice(None))
        elif d in ("time", "times"):
            idx.append(time_index)
        elif d in ("height", "level", "lev", "z"):
            idx.append(level_index)
        elif d in ("releases", "release", "pointspec"):
            idx.append(release_index)
        elif d in ("nageclass",):
            idx.append(0)
        else:
            idx.append(0)

    arr = np.asarray(var[tuple(idx)], dtype=float)
    return np.asarray(np.squeeze(arr), dtype=float)


def infer_n_sources(var) -> int:
    dims = [d.lower() for d in var.dimensions]
    for dim_name, dim_obj in zip(dims, var.shape):
        if dim_name in ("releases", "release", "pointspec"):
            return int(dim_obj)
    return 1


def infer_time_size(var) -> int:
    dims = [d.lower() for d in var.dimensions]
    for dim_name, dim_obj in zip(dims, var.shape):
        if dim_name in ("time", "times"):
            return int(dim_obj)
    return 1


def sample_step_response(
    cvar,
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    release_index: int,
    level_index: int,
    lon: float,
    lat: float,
    n_time: int,
) -> np.ndarray:
    """Concentration time series at ``(lon, lat)`` for one source release.

    The simulation runs each source at a *sustained* unit emission rate, so the
    per-timestep concentration at a fixed point is the transport **step
    response** ``s(tau)`` — the concentration a lag ``tau`` after a constant
    unit-rate emission began. Returned length is ``n_time``.
    """
    s = np.empty(n_time, dtype=float)
    for t in range(n_time):
        field = extract_field_for_release(
            cvar,
            release_index=release_index,
            time_index=t,
            level_index=level_index,
        )
        s[t] = sample_nearest(field, lons, lats, lon, lat)
    return s


def step_to_impulse(step: np.ndarray) -> np.ndarray:
    """First-difference a step response into a per-window impulse response.

    If ``s(tau)`` is the concentration from a sustained unit-rate emission, the
    contribution of a *single* window of unit-rate emission at lag ``tau`` is
    ``h(tau) = s(tau) - s(tau - 1)`` with ``s(-1) = 0`` (so ``h(0) = s(0)``).
    This ``h`` is the convolution kernel that maps a time-varying emission rate
    to concentration.
    """
    return np.diff(np.asarray(step, dtype=float), prepend=0.0)


def toeplitz_convolution_block(impulse: np.ndarray, n_time: int, n_flux: int) -> np.ndarray:
    """Lower-triangular Toeplitz block for one source→receptor pair.

    Maps a per-window emission-rate series (length ``n_flux``) to a
    concentration series (length ``n_time``) by discrete convolution::

        block[t, k] = impulse[t - k]   for 0 <= t - k < len(impulse), else 0

    Causality (emission in window ``k`` only affects times ``t >= k``) makes the
    block lower-triangular; the plume's finite memory zeroes lags past the
    kernel length.
    """
    h = np.asarray(impulse, dtype=float)
    block = np.zeros((n_time, n_flux), dtype=float)
    for k in range(n_flux):
        lag_hi = min(n_time - k, h.size)
        if lag_hi <= 0:
            continue
        # rows t = k .. k + lag_hi - 1 get h[0 .. lag_hi - 1]
        block[k : k + lag_hi, k] = h[:lag_hi]
    return block


def broadcast_state(values, n_sources: int, n_flux: int, *, what: str) -> np.ndarray:
    """Broadcast a per-source or time-resolved quantity to the flat state layout.

    The flat state vector is source-major, window-minor: index ``j * n_flux + k``
    is source ``j`` in flux window ``k``. Accepts:

    * scalar                         → same value everywhere
    * shape ``(n_sources,)``         → constant in time, one value per source
    * shape ``(n_sources, n_flux)``  → fully time-resolved
    * shape ``(n_sources * n_flux,)``→ already flat
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return np.full(n_sources * n_flux, float(arr), dtype=float)
    if arr.shape == (n_sources,):
        return np.repeat(arr, n_flux)
    if arr.shape == (n_sources, n_flux):
        return arr.reshape(-1)
    if arr.shape == (n_sources * n_flux,):
        return arr
    raise ValueError(
        f"{what} must be scalar, length n_sources ({n_sources}), "
        f"shape (n_sources, n_flux) = ({n_sources}, {n_flux}), or length "
        f"n_sources*n_flux ({n_sources * n_flux}); got shape {arr.shape}"
    )


def prepare_sim_transport(ds, variable_name_cfg: str | None):
    var_candidates = (
        str(variable_name_cfg),
        "ch4_mixing_ratio",
        "ch4_concentration",
        "spec001_mr",
        "spec001",
    ) if variable_name_cfg else (
        "ch4_mixing_ratio",
        "ch4_concentration",
        "spec001_mr",
        "spec001",
    )
    vname = find_var(ds, tuple(var_candidates))
    if vname is None:
        raise KeyError("No concentration variable found. Set input.variable_name in YAML.")

    lon_name = find_var(ds, ("longitude", "lon", "xlon"))
    lat_name = find_var(ds, ("latitude", "lat", "ylat"))
    if lon_name is None or lat_name is None:
        raise KeyError("Simulation NetCDF must include longitude and latitude variables")

    lons = np.asarray(ds.variables[lon_name][:], dtype=float)
    lats = np.asarray(ds.variables[lat_name][:], dtype=float)
    cvar = ds.variables[vname]
    n_sources = infer_n_sources(cvar)

    declared_sources = int(getattr(ds, "n_point_sources", 0)) + int(
        getattr(ds, "n_diffuse_sources", 0)
    )
    if declared_sources > 1 and n_sources == 1:
        raise ValueError(
            "Simulation output does not contain per-source release fields. "
            "Rerun simulation with output.per_source: true in simulation YAML "
            "so flux_main can build a multi-source transport matrix G."
        )

    source_names = parse_source_names(ds, n_sources)
    return vname, lons, lats, cvar, n_sources, source_names


def _receptor_time_series(value, n_time_obs: int, what: str) -> np.ndarray:
    """A per-receptor scalar or list coerced to a length-``n_time_obs`` series."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n_time_obs, float(arr), dtype=float)
    if arr.shape == (n_time_obs,):
        return arr
    raise ValueError(
        f"receptor {what} must be a scalar or a list of length n_time ({n_time_obs}); "
        f"got shape {arr.shape}"
    )


def build_y_and_se(
    cfg: dict[str, Any],
    G: np.ndarray,
    receptors: list[dict[str, Any]],
    *,
    n_time_obs: int,
    n_sources: int,
    n_flux: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Assemble the time-resolved observation vector ``y`` and noise variance.

    Observations are laid out receptor-major, time-minor: index
    ``i * n_time_obs + t`` is receptor ``i`` at observation time ``t`` — the same
    row order the time-resolved ``G`` uses.
    """
    obs_cfg = cfg.get("observations", {})
    mode = str(obs_cfg.get("mode", "provided")).strip().lower()
    if mode not in {"provided", "synthetic_from_truth"}:
        raise ValueError("observations.mode must be 'provided' or 'synthetic_from_truth'")

    sigma_default = float(obs_cfg.get("default_sigma", 1.0))

    def sigma_grid() -> np.ndarray:
        rows = [
            np.maximum(
                _receptor_time_series(r.get("sigma", sigma_default), n_time_obs, "sigma"),
                1e-12,
            )
            for r in receptors
        ]
        return np.stack(rows, axis=0)  # (n_recv, n_time_obs)

    if mode == "provided":
        y_rows = []
        for r in receptors:
            if "observed" not in r:
                raise ValueError("each receptor needs an 'observed' time series in provided mode")
            y_rows.append(_receptor_time_series(r["observed"], n_time_obs, "observed"))
        y_obs = np.stack(y_rows, axis=0).reshape(-1)
        Se = (sigma_grid() ** 2).reshape(-1)
        return y_obs, Se, {"mode": mode, "n_time": n_time_obs}

    truth = obs_cfg.get("truth_flux_kg_s")
    if truth is None:
        raise ValueError("observations.truth_flux_kg_s is required for synthetic_from_truth mode")
    x_truth = broadcast_state(truth, n_sources, n_flux, what="observations.truth_flux_kg_s")

    y_clean = G @ x_truth
    sigma = sigma_grid()
    add_noise = bool(obs_cfg.get("add_noise", False))
    seed = int(obs_cfg.get("random_seed", 42))
    if add_noise:
        rng = np.random.default_rng(seed)
        y_obs = y_clean + rng.normal(0.0, sigma.reshape(-1))
    else:
        y_obs = y_clean

    Se = (sigma**2).reshape(-1)
    meta = {
        "mode": mode,
        "truth_flux_kg_s": x_truth.reshape(n_sources, n_flux).tolist(),
        "add_noise": add_noise,
        "random_seed": seed,
        "n_time": n_time_obs,
    }
    return y_obs, Se, meta


def build_prior(
    cfg: dict[str, Any], n_sources: int, n_flux: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Prior mean and diagonal covariance over the flat ``n_sources * n_flux`` state.

    Per-source values are held constant across the ``n_flux`` flux windows; pass
    a ``(n_sources, n_flux)`` array to vary the prior in time.
    """
    inv_cfg = cfg.get("inversion", {})

    x_prior = broadcast_state(
        inv_cfg.get("prior_flux_kg_s", 0.0), n_sources, n_flux, what="inversion.prior_flux_kg_s"
    )

    prior_cov_diag = inv_cfg.get("prior_covariance_diag")
    if prior_cov_diag is not None:
        Sa = broadcast_state(
            prior_cov_diag, n_sources, n_flux, what="inversion.prior_covariance_diag"
        )
    else:
        prior_var = inv_cfg.get("prior_variance")
        if prior_var is not None:
            Sa = np.full(n_sources * n_flux, float(prior_var), dtype=float)
        else:
            frac = float(inv_cfg.get("prior_sigma_fraction", 0.5))
            sigma = np.maximum(np.abs(x_prior) * frac, 1e-12)
            Sa = sigma**2

    return x_prior, Sa
