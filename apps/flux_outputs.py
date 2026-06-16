import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def write_flux_outputs(
    cfg: dict[str, Any],
    *,
    G: np.ndarray,
    y_obs: np.ndarray,
    Se: np.ndarray,
    x_prior: np.ndarray,
    Sa: np.ndarray,
    result,
    source_names: list[str],
    summary_extra: dict[str, Any],
) -> tuple[Path, Path, Path]:
    out_cfg = cfg.get("output", {})
    out_json = Path(out_cfg.get("summary_json", "outputs/flux_inversion_summary.json")).expanduser().resolve()
    out_npz = Path(out_cfg.get("matrices_npz", "outputs/flux_inversion_matrices.npz")).expanduser().resolve()
    out_csv = Path(out_cfg.get("posterior_csv", "outputs/flux_posterior.csv")).expanduser().resolve()

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_npz,
        G=G,
        y_obs=y_obs,
        Se_diag=Se,
        x_prior=x_prior,
        Sa_diag=Sa if Sa.ndim == 1 else np.diag(Sa),
        x_opt=result.x_posterior,
        y_prior=result.y_prior,
        y_opt=result.y_posterior,
        Sx=result.posterior_cov,
        averaging_kernel=result.averaging_kernel,
    )

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "x_prior_kg_s", "x_opt_kg_s", "posterior_sigma_kg_s"])
        post_sigma = np.sqrt(np.maximum(np.diag(result.posterior_cov), 0.0))
        for i, name in enumerate(source_names):
            writer.writerow([name, float(x_prior[i]), float(result.x_posterior[i]), float(post_sigma[i])])

    out_json.write_text(json.dumps(summary_extra, indent=2) + "\n")
    return out_json, out_npz, out_csv
