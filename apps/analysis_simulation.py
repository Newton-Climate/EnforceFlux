from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from enforceflux.analysis import load_simulation_netcdf, plot_simulation_heatmap_from_netcdf


def analyze_simulation(
    nc_path: Path,
    cfg: dict[str, Any],
    viz_dir: Path,
    viz_enabled: bool,
) -> dict[str, Any]:
    variable_names = tuple(
        cfg.get("input", {}).get("simulation_variable_names", ["ch4_mixing_ratio", "ch4_concentration"])
    )
    payload = load_simulation_netcdf(nc_path, variable_names=variable_names)

    c = np.asarray(payload["concentration"], dtype=float)
    finite = c[np.isfinite(c)]
    nonzero = finite[finite > 0]

    summary: dict[str, Any] = {
        "type": "simulation",
        "netcdf": str(nc_path),
        "variable": str(payload["variable_name"]),
        "units": str(payload["units"]),
        "shape": list(c.shape),
        "n_finite": int(finite.size),
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "mean": float(np.mean(finite)) if finite.size else None,
        "mean_nonzero": float(np.mean(nonzero)) if nonzero.size else 0.0,
        "n_nonzero": int(nonzero.size),
    }

    if viz_enabled:
        viz_dir.mkdir(parents=True, exist_ok=True)
        time_index = int(cfg.get("visualization", {}).get("time_index", 0))
        level_index = int(cfg.get("visualization", {}).get("level_index", 0))
        release_index = int(cfg.get("visualization", {}).get("release_index", 0))
        heatmap_path = viz_dir / "simulation_heatmap.png"

        fig, _ = plot_simulation_heatmap_from_netcdf(
            nc_path,
            time_index=time_index,
            level_index=level_index,
            release_index=release_index,
        )
        fig.savefig(heatmap_path, dpi=150)
        summary["heatmap_png"] = str(heatmap_path)

    return summary
