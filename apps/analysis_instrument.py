from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from enforceflux.analysis import (
    analyze_information_content,
    plot_averaging_kernel,
    plot_dfs_per_source,
    plot_forward_operator,
    plot_posterior_uncertainty,
)

from analysis_common import find_var, instrument_group_masks, jsonify


def analyze_instrument(
    nc_path: Path,
    cfg: dict[str, Any],
    viz_dir: Path,
    viz_enabled: bool,
) -> dict[str, Any]:
    from netCDF4 import Dataset

    analysis_cfg = cfg.get("analysis", {})

    with Dataset(nc_path) as ds:
        y_obs_name = find_var(ds, ("y_obs", "observation", "observations"))
        y_clean_name = find_var(ds, ("y_clean", "sampled_concentration"))
        valid_name = find_var(ds, ("valid_mask",))
        nvar_name = find_var(ds, ("noise_variance",))
        g_name = find_var(ds, ("transport_operator", "H_g", "G"))

        if y_obs_name is None or y_clean_name is None:
            raise KeyError("Instrument NetCDF must include y_obs and y_clean (or sampled_concentration)")

        y_obs = np.asarray(ds.variables[y_obs_name][:], dtype=float)
        y_clean = np.asarray(ds.variables[y_clean_name][:], dtype=float)

        if y_obs.ndim != 2:
            raise ValueError(f"Expected y_obs with shape (time, instrument), got {y_obs.shape}")

        n_time, n_inst = y_obs.shape
        valid = np.asarray(ds.variables[valid_name][:], dtype=bool) if valid_name else np.isfinite(y_obs)
        nvar = np.asarray(ds.variables[nvar_name][:], dtype=float) if nvar_name else np.ones_like(y_obs)

        y_obs_flat = y_obs.reshape(-1)
        y_clean_flat = y_clean.reshape(-1)
        valid_flat = valid.reshape(-1) & np.isfinite(y_obs_flat)
        se_diag_flat = nvar.reshape(-1)

        if g_name is not None:
            g_raw = np.asarray(ds.variables[g_name][:], dtype=float)
            if g_raw.ndim == 3:
                g_all = g_raw.reshape(-1, g_raw.shape[-1])
            elif g_raw.ndim == 2:
                g_all = g_raw
            else:
                raise ValueError(f"Unsupported transport_operator shape: {g_raw.shape}")
        else:
            # Fallback: one-source pseudo-Jacobian from clean signal.
            g_all = y_clean_flat.reshape(-1, 1)

        if g_all.shape[0] != y_obs_flat.shape[0]:
            raise ValueError(
                f"G rows ({g_all.shape[0]}) must match number of observations ({y_obs_flat.shape[0]})"
            )

        g_valid = g_all[valid_flat]
        se_valid = se_diag_flat[valid_flat]

        if g_valid.size == 0:
            raise ValueError("No valid observations available for analysis")

        n_src = g_valid.shape[1]
        prior_diag_cfg = analysis_cfg.get("prior_covariance_diag")
        if isinstance(prior_diag_cfg, list):
            Sa = np.asarray(prior_diag_cfg, dtype=float)
            if Sa.shape[0] != n_src:
                raise ValueError(
                    "analysis.prior_covariance_diag length must match number of source columns in G"
                )
        else:
            prior_var = float(analysis_cfg.get("prior_variance", 1.0))
            Sa = np.full(n_src, prior_var, dtype=float)

        obs_groups = instrument_group_masks(n_time, n_inst, valid_flat)
        obs_groups = {k: v[valid_flat] for k, v in obs_groups.items()}

        fisher, dof, posterior = analyze_information_content(
            G=g_valid,
            Se=se_valid,
            Sa=Sa,
            obs_groups=obs_groups,
            source_names=[f"source_{i}" for i in range(n_src)],
        )

        summary: dict[str, Any] = {
            "type": "instrument",
            "netcdf": str(nc_path),
            "y_obs_variable": y_obs_name,
            "y_clean_variable": y_clean_name,
            "n_time": int(n_time),
            "n_instruments": int(n_inst),
            "n_observations_total": int(y_obs_flat.size),
            "n_observations_valid": int(valid_flat.sum()),
            "n_sources": int(n_src),
            "dfs_total": float(dof.dfs_total),
            "dfs_per_group": jsonify(dof.dfs_per_group),
            "dfs_per_source": jsonify(dof.dfs_per_source),
            "fisher_eigenvalues": jsonify(fisher.eigenvalues),
            "posterior_sigma": jsonify(posterior.posterior_sigma),
            "prior_sigma": jsonify(posterior.prior_sigma),
            "uncertainty_reduction": jsonify(posterior.uncertainty_reduction),
        }

        if viz_enabled:
            viz_dir.mkdir(parents=True, exist_ok=True)

            fig1, _ = plot_forward_operator(g_valid, title="Valid Observation Jacobian")
            p1 = viz_dir / "instrument_forward_operator.png"
            fig1.savefig(p1, dpi=150)

            fig2, _ = plot_averaging_kernel(dof.averaging_kernel, title="Averaging Kernel")
            p2 = viz_dir / "instrument_averaging_kernel.png"
            fig2.savefig(p2, dpi=150)

            fig3, _ = plot_dfs_per_source(dof.dfs_per_source, title="DFS per Source")
            p3 = viz_dir / "instrument_dfs_per_source.png"
            fig3.savefig(p3, dpi=150)

            fig4, _ = plot_posterior_uncertainty(
                posterior.prior_sigma,
                posterior.posterior_sigma,
                title="Prior vs Posterior Uncertainty",
            )
            p4 = viz_dir / "instrument_posterior_uncertainty.png"
            fig4.savefig(p4, dpi=150)

            summary["plots"] = [str(p1), str(p2), str(p3), str(p4)]

        return summary
