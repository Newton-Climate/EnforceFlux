#!/usr/bin/env python3
"""Run reproducible analysis from simulation or instrument NetCDF outputs.

Usage:
    python apps/analysis_main.py --config apps/analysis_main.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis_common import jsonify, require_yaml
from analysis_instrument import analyze_instrument
from analysis_simulation import analyze_simulation


def _maybe_add_wind_rose(
    summary: dict,
    cfg: dict,
    *,
    default_input_nc: Path,
    viz_dir: Path,
    viz_enabled: bool,
) -> None:
    if not viz_enabled:
        return

    wind_cfg = cfg.get("visualization", {}).get("wind_rose", {})
    if not bool(wind_cfg.get("enabled", False)):
        return

    from enforceflux.analysis import plot_wind_rose_from_netcdf

    src_nc = Path(wind_cfg.get("netcdf", str(default_input_nc))).expanduser().resolve()
    if not src_nc.exists():
        raise FileNotFoundError(f"Wind-rose NetCDF not found: {src_nc}")

    u_var = str(wind_cfg.get("u_var", "u10"))
    v_var = str(wind_cfg.get("v_var", "v10"))
    n_dir_bins = int(wind_cfg.get("n_dir_bins", 16))
    calm_threshold = float(wind_cfg.get("calm_threshold", 0.2))
    title = str(wind_cfg.get("title", "Wind Rose"))
    cmap = str(wind_cfg.get("cmap", "viridis"))
    out_name = str(wind_cfg.get("output_name", "wind_rose.png"))

    viz_dir.mkdir(parents=True, exist_ok=True)
    out_path = viz_dir / out_name

    fig, _ax, payload = plot_wind_rose_from_netcdf(
        src_nc,
        u_var=u_var,
        v_var=v_var,
        title=title,
        n_dir_bins=n_dir_bins,
        calm_threshold=calm_threshold,
        cmap=cmap,
    )
    fig.savefig(out_path, dpi=150)

    summary["wind_rose"] = {
        "netcdf": str(src_nc),
        "u_var": u_var,
        "v_var": v_var,
        "n_dir_bins": n_dir_bins,
        "calm_threshold": calm_threshold,
        "plot_png": str(out_path),
        "sample_count": int(payload.get("sample_count", 0)),
        "calm_fraction": float(payload.get("calm_fraction", 0.0)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run information/visual analysis from simulation or instrument NetCDF"
    )
    parser.add_argument("--config", required=True, help="Path to analysis YAML config")
    return parser


def main() -> None:
    from netCDF4 import Dataset

    parser = build_parser()
    args = parser.parse_args()

    yaml = require_yaml()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = yaml.safe_load(config_path.read_text()) or {}
    input_cfg = cfg.get("input", {})
    nc_path = Path(input_cfg.get("netcdf", "")).expanduser().resolve()
    if not nc_path.exists():
        raise FileNotFoundError(f"Input NetCDF not found: {nc_path}")

    kind = str(input_cfg.get("kind", "auto")).strip().lower()
    if kind not in {"auto", "simulation", "instrument"}:
        raise ValueError("input.kind must be one of: auto, simulation, instrument")

    viz_cfg = cfg.get("visualization", {})
    viz_enabled = bool(viz_cfg.get("enabled", False))
    viz_dir = Path(viz_cfg.get("directory", "outputs/analysis_plots")).expanduser().resolve()

    if kind == "auto":
        with Dataset(nc_path) as ds:
            has_instrument = any(name in ds.variables for name in ("y_obs", "y_clean", "valid_mask"))
            kind = "instrument" if has_instrument else "simulation"

    if kind == "simulation":
        summary = analyze_simulation(nc_path, cfg, viz_dir, viz_enabled)
    else:
        summary = analyze_instrument(nc_path, cfg, viz_dir, viz_enabled)

    _maybe_add_wind_rose(
        summary,
        cfg,
        default_input_nc=nc_path,
        viz_dir=viz_dir,
        viz_enabled=viz_enabled,
    )

    out_json = Path(cfg.get("output", {}).get("summary_json", "outputs/analysis_summary.json"))
    out_json = out_json.expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(jsonify(summary), indent=2) + "\n")

    print("EnforceFlux analysis_main")
    print(f"Config      : {config_path}")
    print(f"Input kind  : {kind}")
    print(f"Input NC    : {nc_path}")
    print(f"Summary JSON: {out_json}")
    if viz_enabled:
        print(f"Plots dir   : {viz_dir}")


if __name__ == "__main__":
    main()
