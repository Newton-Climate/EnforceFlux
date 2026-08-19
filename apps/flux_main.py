#!/usr/bin/env python3
"""Flux inversion stage — build G and solve for posterior fluxes.

Reads upstream artifacts by role, runs optimal estimation, emits a
posterior + diagnostics bundle under runs/<run.name>/flux/.

Roles:
  in : dispersion → concentration_field  (concentration.nc)  — always required
       obs        → obs                   (obs.nc)             — optional, from
                                                                 instrument or obs
                                                                 stage; enables
                                                                 mode: instrument_netcdf
  out: flux.summary   → summary.json      — posterior + diagnostics
       flux.matrices  → matrices.npz      — G, S_a, S_e, gain, averaging kernel
       flux.posterior → posterior.csv     — one row per (source, flux_window)

Usage:
    enforceflux flux --config configs/main/flux.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# --- source-heterogeneity OSSE (M8) ---
def _build_background_nuisance(bg_cfg, block, obs_meta, n_obs, input_mode):
    """Return (g_beta, sigma_beta, meta) for the flux.inversion.background block."""
    if not bg_cfg:
        return None, None, {"model": "none"}
    model = str(bg_cfg.get("model", "none")).strip().lower()
    if model == "none":
        return None, None, {"model": "none"}
    sigma = float(bg_cfg.get("sigma"))
    if model == "constant":
        g_beta = np.ones((n_obs, 1), dtype=float)
        return g_beta, np.array([sigma], dtype=float), {"model": "constant", "sigma": sigma, "n_beta": 1}
    if model == "gradient":
        if input_mode != "simulation_receptors":
            raise NotImplementedError(
                "flux.inversion.background.model=gradient requires simulation_receptors input mode"
            )
        receptors = block.get("receptors") or []
        if not receptors:
            raise ValueError("gradient background requires receptors[] with lon/lat")
        n_time = int(obs_meta.get("n_time", n_obs // len(receptors)))
        lons = np.array([float(r["lon"]) for r in receptors], dtype=float)
        lats = np.array([float(r["lat"]) for r in receptors], dtype=float)
        x_obs = np.repeat(lons, n_time)
        y_obs = np.repeat(lats, n_time)
        if x_obs.size != n_obs:
            raise ValueError(
                f"gradient background expected {n_obs} obs, receptors*time gave {x_obs.size}"
            )
        g_beta = np.stack([np.ones(n_obs), x_obs, y_obs], axis=1)
        sigma_arr = np.full(3, sigma, dtype=float)
        return g_beta, sigma_arr, {"model": "gradient", "sigma": sigma, "n_beta": 3}
    raise ValueError(f"flux.inversion.background.model must be constant|gradient|none, got {model!r}")
# --- end source-heterogeneity OSSE (M8) ---


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build G from dispersion output and run OE flux inversion"
    )
    parser.add_argument("--config", required=True, help="Path to flux-stage YAML")
    return parser


def main() -> None:
    from enforceflux.inversion import oe_from_linear, optimize_oe
    from enforceflux.inversion.bayesian import bayesian_linear_inversion
    from enforceflux.runs import load_stage_config, open_run_dir, read_upstream
    from flux_helpers import build_prior
    from flux_inputs import build_from_instrument_mode, build_from_receptors_mode

    args = build_parser().parse_args()

    stage_cfg = load_stage_config(args.config, expected_stage="flux")
    block = dict(stage_cfg.block)

    # Resolve upstream artifacts before touching the helpers so the injected
    # cfg dict looks like the pre-refactor shape they still expect.
    if "dispersion" not in stage_cfg.inputs:
        raise ValueError(
            f"{stage_cfg.yaml_path}: `inputs.dispersion:` is required — "
            f"flux needs a dispersion RunDir to build the transport matrix G."
        )
    dispersion_up = read_upstream(stage_cfg.inputs["dispersion"])
    sim_nc = dispersion_up.file("concentration_field")

    input_block = dict(block.get("input") or {})
    input_block["simulation_netcdf"] = str(sim_nc)
    input_mode = str(input_block.get("mode", "simulation_receptors")).strip().lower()
    if input_mode not in {"simulation_receptors", "instrument_netcdf"}:
        raise ValueError("flux.input.mode must be 'simulation_receptors' or 'instrument_netcdf'")

    if input_mode == "instrument_netcdf":
        if "obs" not in stage_cfg.inputs:
            raise ValueError(
                f"{stage_cfg.yaml_path}: `inputs.obs:` is required when "
                f"flux.input.mode is 'instrument_netcdf' — point it at an "
                f"instrument or obs RunDir."
            )
        obs_up = read_upstream(stage_cfg.inputs["obs"])
        input_block["instrument_netcdf"] = str(obs_up.file("obs"))
    else:
        obs_up = None

    # Repackage into the shape the existing helpers expect.
    legacy_cfg = {
        "input": input_block,
        "receptors": block.get("receptors") or [],
        "observations": block.get("observations") or {},
        "inversion": block.get("inversion") or {},
    }

    if input_mode == "simulation_receptors":
        G, y_obs, Se, source_names, vname, _, obs_meta, n_flux = build_from_receptors_mode(legacy_cfg)
    else:
        G, y_obs, Se, source_names, vname, _, obs_meta, n_flux = build_from_instrument_mode(legacy_cfg)

    n_sources = len(source_names)
    x_prior, Sa = build_prior(legacy_cfg, n_sources, n_flux)

    inv_cfg = legacy_cfg["inversion"]
    method = str(inv_cfg.get("method", "linear")).strip().lower()
    if method not in {"linear", "nonlinear"}:
        raise ValueError("flux.inversion.method must be 'linear' or 'nonlinear'")

    # --- source-heterogeneity OSSE (M8) ---
    g_beta, sigma_beta, background_meta = _build_background_nuisance(
        inv_cfg.get("background"), block, obs_meta, len(y_obs), input_mode
    )
    # --- end source-heterogeneity OSSE (M8) ---

    if method == "linear":
        # --- source-heterogeneity OSSE (M8) ---
        if g_beta is not None:
            result = bayesian_linear_inversion(
                g=np.asarray(G, dtype=float), y=y_obs,
                x_prior=np.asarray(x_prior, dtype=float),
                s_a=Sa, r=Se, source_names=source_names,
                g_beta=g_beta, sigma_beta=sigma_beta,
            )
        else:
            result = oe_from_linear(
                G=G, y=y_obs, x_prior=x_prior, Sa=Sa, Se=Se, source_names=source_names,
            )
        # --- end source-heterogeneity OSSE (M8) ---
    else:
        result = optimize_oe(
            F=lambda x: G @ x, y=y_obs, x_prior=x_prior, Sa=Sa, Se=Se,
            n_iter=int(inv_cfg.get("n_iter", 20)),
            lam0=float(inv_cfg.get("lam0", 1e-3)),
            lam_factor=float(inv_cfg.get("lam_factor", 10.0)),
            eps=float(inv_cfg.get("eps", 1e-4)),
            fd_step=float(inv_cfg.get("fd_step", 1e-5)),
            source_names=source_names,
        )

    n_flux_windows = int(G.shape[1] // n_sources)
    x_prior_ts = np.asarray(x_prior, dtype=float).reshape(n_sources, n_flux_windows)
    x_opt_ts = np.asarray(result.x_posterior, dtype=float).reshape(n_sources, n_flux_windows)
    post_sigma_ts = np.sqrt(
        np.maximum(np.diag(result.posterior_cov), 0.0)
    ).reshape(n_sources, n_flux_windows)

    # ── Open RunDir and write outputs into it ──────────────────────────────
    run_dir = open_run_dir(
        stage="flux",
        run_name=stage_cfg.run_name,
        outputs_root=stage_cfg.outputs_root,
        inputs={k: str(v) for k, v in stage_cfg.inputs.items()},
    )
    run_dir.snapshot_config(stage_cfg.snapshot)

    matrices_path = run_dir.path("matrices.npz")
    np.savez(
        matrices_path,
        G=G, y_obs=y_obs, Se_diag=Se,
        x_prior=x_prior, Sa_diag=Sa if Sa.ndim == 1 else np.diag(Sa),
        x_opt=result.x_posterior,
        y_prior=result.y_prior, y_opt=result.y_posterior,
        Sx=result.posterior_cov, averaging_kernel=result.averaging_kernel,
    )
    run_dir.record_output("matrices.npz", role="matrices")

    posterior_csv = run_dir.path("posterior.csv")
    with posterior_csv.open("w", newline="") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["source", "flux_window", "x_prior_kg_s", "x_opt_kg_s", "posterior_sigma_kg_s"])
        for i, name in enumerate(source_names):
            for k in range(n_flux_windows):
                writer.writerow([
                    name, k,
                    float(x_prior_ts[i, k]), float(x_opt_ts[i, k]),
                    float(post_sigma_ts[i, k]),
                ])
    run_dir.record_output("posterior.csv", role="posterior")

    summary = {
        "input_mode": input_mode,
        "concentration_variable": vname,
        "upstream_dispersion": str(dispersion_up.root),
        "upstream_obs": str(obs_up.root) if obs_up is not None else None,
        "n_observations": int(len(y_obs)),
        "n_sources": int(n_sources),
        "n_flux_windows": int(n_flux_windows),
        "source_names": source_names,
        "method": method,
        "observation_mode": obs_meta,
        "x_prior_kg_s": x_prior_ts.tolist(),
        "x_opt_kg_s": x_opt_ts.tolist(),
        "posterior_sigma_kg_s": post_sigma_ts.tolist(),
        "converged": bool(result.converged),
        "n_iter": int(result.n_iter),
    }
    # --- source-heterogeneity OSSE (M8) ---
    summary["background_nuisance"] = background_meta
    if result.posterior_beta_mean is not None:
        beta_sigma = np.sqrt(np.maximum(np.diag(result.posterior_beta_cov), 0.0))
        summary["background_nuisance"]["posterior_beta_mean"] = result.posterior_beta_mean.tolist()
        summary["background_nuisance"]["posterior_beta_sigma"] = beta_sigma.tolist()
    # --- end source-heterogeneity OSSE (M8) ---
    summary_path = run_dir.path("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    run_dir.record_output("summary.json", role="summary")

    contract = run_dir.finalize()

    print("EnforceFlux flux")
    print(f"Config       : {stage_cfg.yaml_path}")
    print(f"Run name     : {stage_cfg.run_name}")
    print(f"Run dir      : {run_dir.root}")
    print(f"Dispersion   : {dispersion_up.root}")
    if obs_up is not None:
        print(f"Obs          : {obs_up.root}")
    print(f"G shape      : {G.shape}  (obs x sources*windows)")
    print(f"Flux windows : {n_flux_windows} per source")
    print(f"Inversion    : {method}")
    print(f"Converged    : {result.converged}")
    print(f"Summary      : {summary_path}")
    print(f"Matrices     : {matrices_path}")
    print(f"Posterior    : {posterior_csv}")
    print(f"Manifest     : {contract['manifest']}")


if __name__ == "__main__":
    main()
