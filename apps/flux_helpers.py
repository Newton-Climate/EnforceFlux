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


def build_y_and_se(
    cfg: dict[str, Any],
    G: np.ndarray,
    receptors: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    obs_cfg = cfg.get("observations", {})
    mode = str(obs_cfg.get("mode", "provided")).strip().lower()

    if mode not in {"provided", "synthetic_from_truth"}:
        raise ValueError("observations.mode must be 'provided' or 'synthetic_from_truth'")

    if mode == "provided":
        y_obs = np.array([float(r["observed"]) for r in receptors], dtype=float)
        sigma = np.array(
            [float(r.get("sigma", obs_cfg.get("default_sigma", 1.0))) for r in receptors],
            dtype=float,
        )
        sigma = np.maximum(sigma, 1e-12)
        Se = sigma**2
        return y_obs, Se, {"mode": mode}

    truth = obs_cfg.get("truth_flux_kg_s")
    if truth is None:
        raise ValueError("observations.truth_flux_kg_s is required for synthetic_from_truth mode")

    truth_arr = np.asarray(truth, dtype=float)
    if truth_arr.ndim == 0:
        truth_arr = np.full(G.shape[1], float(truth_arr), dtype=float)
    if truth_arr.shape[0] != G.shape[1]:
        raise ValueError(
            "truth_flux_kg_s must be scalar or length n_sources. "
            f"Got {truth_arr.shape[0]} for n_sources={G.shape[1]}"
        )

    y_clean = G @ truth_arr
    sigma_default = float(obs_cfg.get("default_sigma", 1.0))
    sigma = np.array(
        [float(r.get("sigma", sigma_default)) for r in receptors],
        dtype=float,
    )
    sigma = np.maximum(sigma, 1e-12)

    add_noise = bool(obs_cfg.get("add_noise", False))
    seed = int(obs_cfg.get("random_seed", 42))
    if add_noise:
        rng = np.random.default_rng(seed)
        y_obs = y_clean + rng.normal(0.0, sigma)
    else:
        y_obs = y_clean

    Se = sigma**2
    meta = {
        "mode": mode,
        "truth_flux_kg_s": truth_arr.tolist(),
        "add_noise": add_noise,
        "random_seed": seed,
    }
    return y_obs, Se, meta


def build_prior(cfg: dict[str, Any], n_sources: int) -> tuple[np.ndarray, np.ndarray]:
    inv_cfg = cfg.get("inversion", {})

    prior_flux = inv_cfg.get("prior_flux_kg_s", 0.0)
    x_prior = np.asarray(prior_flux, dtype=float)
    if x_prior.ndim == 0:
        x_prior = np.full(n_sources, float(x_prior), dtype=float)
    if x_prior.shape[0] != n_sources:
        raise ValueError(
            "inversion.prior_flux_kg_s must be scalar or length n_sources. "
            f"Got {x_prior.shape[0]} for n_sources={n_sources}"
        )

    prior_cov_diag = inv_cfg.get("prior_covariance_diag")
    if prior_cov_diag is not None:
        Sa = np.asarray(prior_cov_diag, dtype=float)
        if Sa.shape[0] != n_sources:
            raise ValueError(
                "inversion.prior_covariance_diag length must match n_sources"
            )
    else:
        prior_var = inv_cfg.get("prior_variance")
        if prior_var is not None:
            Sa = np.full(n_sources, float(prior_var), dtype=float)
        else:
            frac = float(inv_cfg.get("prior_sigma_fraction", 0.5))
            sigma = np.maximum(np.abs(x_prior) * frac, 1e-12)
            Sa = sigma**2

    return x_prior, Sa
