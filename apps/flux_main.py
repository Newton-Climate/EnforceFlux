#!/usr/bin/env python3
"""Build flux inversion matrices from simulation output and run OE inversion.

Usage:
    python apps/flux_main.py --config apps/flux_main.yaml
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build G from simulation NetCDF and run optimal-estimation flux inversion"
    )
    parser.add_argument("--config", required=True, help="Path to flux inversion YAML config")
    return parser


def main() -> None:
    from enforceflux.inversion import oe_from_linear, optimize_oe
    from flux_helpers import build_prior, require_yaml
    from flux_inputs import build_from_instrument_mode, build_from_receptors_mode
    from flux_outputs import write_flux_outputs

    parser = build_parser()
    args = parser.parse_args()

    yaml = require_yaml()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = yaml.safe_load(config_path.read_text()) or {}

    input_cfg = cfg.get("input", {})
    input_mode = str(input_cfg.get("mode", "simulation_receptors")).strip().lower()
    if input_mode not in {"simulation_receptors", "instrument_netcdf"}:
        raise ValueError("input.mode must be 'simulation_receptors' or 'instrument_netcdf'")

    if input_mode == "simulation_receptors":
        G, y_obs, Se, source_names, vname, sim_nc, obs_meta = build_from_receptors_mode(cfg)
    else:
        G, y_obs, Se, source_names, vname, sim_nc, obs_meta = build_from_instrument_mode(cfg)

    n_sources = G.shape[1]
    x_prior, Sa = build_prior(cfg, n_sources)

    inv_cfg = cfg.get("inversion", {})
    method = str(inv_cfg.get("method", "linear")).strip().lower()
    if method not in {"linear", "nonlinear"}:
        raise ValueError("inversion.method must be 'linear' or 'nonlinear'")

    if method == "linear":
        result = oe_from_linear(
            G=G,
            y=y_obs,
            x_prior=x_prior,
            Sa=Sa,
            Se=Se,
            source_names=source_names,
        )
    else:
        result = optimize_oe(
            F=lambda x: G @ x,
            y=y_obs,
            x_prior=x_prior,
            Sa=Sa,
            Se=Se,
            n_iter=int(inv_cfg.get("n_iter", 20)),
            lam0=float(inv_cfg.get("lam0", 1e-3)),
            lam_factor=float(inv_cfg.get("lam_factor", 10.0)),
            eps=float(inv_cfg.get("eps", 1e-4)),
            fd_step=float(inv_cfg.get("fd_step", 1e-5)),
            source_names=source_names,
        )

    summary = {
        "input_simulation_netcdf": str(sim_nc),
        "input_mode": input_mode,
        "concentration_variable": vname,
        "n_observations": int(len(y_obs)),
        "n_sources": int(G.shape[1]),
        "source_names": source_names,
        "method": method,
        "observation_mode": obs_meta,
        "x_prior_kg_s": x_prior.tolist(),
        "x_opt_kg_s": np.asarray(result.x_posterior, dtype=float).tolist(),
        "converged": bool(result.converged),
        "n_iter": int(result.n_iter),
        "outputs": {
            "summary_json": "",
            "matrices_npz": "",
            "posterior_csv": "",
        },
    }

    out_json, out_npz, out_csv = write_flux_outputs(
        cfg,
        G=G,
        y_obs=y_obs,
        Se=Se,
        x_prior=x_prior,
        Sa=Sa,
        result=result,
        source_names=source_names,
        summary_extra=summary,
    )

    summary["outputs"] = {
        "summary_json": str(out_json),
        "matrices_npz": str(out_npz),
        "posterior_csv": str(out_csv),
    }
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    print("EnforceFlux flux_main")
    print(f"Config        : {config_path}")
    print(f"Input NC      : {sim_nc}")
    print(f"Input mode    : {input_mode}")
    print(f"G shape       : {G.shape}")
    print(f"Inversion     : {method}")
    print(f"Converged     : {result.converged}")
    print(f"Summary JSON  : {out_json}")
    print(f"Matrices NPZ  : {out_npz}")
    print(f"Posterior CSV : {out_csv}")


if __name__ == "__main__":
    main()
