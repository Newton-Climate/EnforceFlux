#!/usr/bin/env python3
"""Analysis stage — information-content diagnostics + plots.

Reads either an instrument RunDir (preferred; gives DFS, averaging kernel,
posterior uncertainty from the operator + noise model) or a dispersion
RunDir (simulation-only diagnostics), and emits a summary bundle under
runs/<run.name>/analysis/.

Roles:
  in : instrument → obs                  (obs.nc)             — preferred
       dispersion → concentration_field  (concentration.nc)   — fallback
  out: analysis.summary → summary.json
       analysis.plots   → plots/*.png    (when visualization.enabled)

Usage:
    enforceflux analysis --config configs/main/analysis.yaml
"""
import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis_common import jsonify
from analysis_instrument import analyze_instrument
from analysis_simulation import analyze_simulation


def _maybe_add_wind_rose(
    summary: dict, cfg: dict, *,
    default_input_nc: Path, viz_dir: Path, viz_enabled: bool,
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
    out_name = str(wind_cfg.get("output_name", "wind_rose.png"))
    viz_dir.mkdir(parents=True, exist_ok=True)
    out_path = viz_dir / out_name
    fig, _ax, payload = plot_wind_rose_from_netcdf(
        src_nc, u_var=u_var, v_var=v_var,
        title=str(wind_cfg.get("title", "Wind Rose")),
        n_dir_bins=n_dir_bins, calm_threshold=calm_threshold,
        cmap=str(wind_cfg.get("cmap", "viridis")),
    )
    fig.savefig(out_path, dpi=150)
    summary["wind_rose"] = {
        "netcdf": str(src_nc),
        "u_var": u_var, "v_var": v_var,
        "n_dir_bins": n_dir_bins, "calm_threshold": calm_threshold,
        "plot_png": str(out_path),
        "sample_count": int(payload.get("sample_count", 0)),
        "calm_fraction": float(payload.get("calm_fraction", 0.0)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Information-content analysis of a dispersion or instrument RunDir"
    )
    parser.add_argument("--config", required=True, help="Path to analysis-stage YAML")
    return parser


def main() -> None:
    from netCDF4 import Dataset
    from enforceflux.runs import load_stage_config, open_run_dir, read_upstream

    args = build_parser().parse_args()

    stage_cfg = load_stage_config(args.config, expected_stage="analysis")
    block = stage_cfg.block

    # Prefer instrument RunDir (full info-content diagnostics); fall back to
    # dispersion (simulation-only diagnostics). At least one is required.
    instrument_up = None
    dispersion_up = None
    nc_path: Path
    if "instrument" in stage_cfg.inputs:
        instrument_up = read_upstream(stage_cfg.inputs["instrument"])
        nc_path = instrument_up.file("obs")
        kind_default = "instrument"
    elif "dispersion" in stage_cfg.inputs:
        dispersion_up = read_upstream(stage_cfg.inputs["dispersion"])
        nc_path = dispersion_up.file("concentration_field")
        kind_default = "simulation"
    else:
        raise ValueError(
            f"{stage_cfg.yaml_path}: `inputs.instrument:` or `inputs.dispersion:` "
            f"is required."
        )

    input_cfg = dict(block.get("input") or {})
    kind = str(input_cfg.get("kind", kind_default)).strip().lower()
    if kind not in {"auto", "simulation", "instrument"}:
        raise ValueError("analysis.input.kind must be one of: auto, simulation, instrument")
    if kind == "auto":
        with Dataset(nc_path) as ds:
            has_instrument = any(v in ds.variables for v in ("y_obs", "y_clean", "valid_mask"))
            kind = "instrument" if has_instrument else "simulation"

    run_dir = open_run_dir(
        stage="analysis",
        run_name=stage_cfg.run_name,
        outputs_root=stage_cfg.outputs_root,
        inputs={k: str(v) for k, v in stage_cfg.inputs.items()},
    )
    run_dir.snapshot_config(stage_cfg.snapshot)

    viz_cfg = block.get("visualization") or {}
    viz_enabled = bool(viz_cfg.get("enabled", False))
    viz_dir = run_dir.path("plots") if viz_enabled else run_dir.root / "plots"

    # The analyze_* helpers read the config from the "input"/"analysis"/
    # "visualization" keys; wrap the analysis block so it matches.
    legacy_cfg = {
        "input": {
            **input_cfg,
            "netcdf": str(nc_path),
            "kind": kind,
        },
        "analysis": block.get("analysis") or {},
        "visualization": {**viz_cfg, "directory": str(viz_dir)},
        "output": block.get("output") or {},
    }

    if kind == "simulation":
        summary = analyze_simulation(nc_path, legacy_cfg, viz_dir, viz_enabled)
    else:
        summary = analyze_instrument(nc_path, legacy_cfg, viz_dir, viz_enabled)

    _maybe_add_wind_rose(
        summary, legacy_cfg,
        default_input_nc=nc_path, viz_dir=viz_dir, viz_enabled=viz_enabled,
    )

    summary["upstream"] = {
        "instrument": str(instrument_up.root) if instrument_up is not None else None,
        "dispersion": str(dispersion_up.root) if dispersion_up is not None else None,
    }

    summary_path = run_dir.path("summary.json")
    summary_path.write_text(json.dumps(jsonify(summary), indent=2) + "\n")
    run_dir.record_output("summary.json", role="summary")

    if viz_enabled and viz_dir.is_dir():
        for png in sorted(viz_dir.glob("*.png")):
            run_dir.record_output(f"plots/{png.name}", role=f"plot_{png.stem}")

    contract = run_dir.finalize()

    print("EnforceFlux analysis")
    print(f"Config     : {stage_cfg.yaml_path}")
    print(f"Run name   : {stage_cfg.run_name}")
    print(f"Run dir    : {run_dir.root}")
    print(f"Input kind : {kind}")
    print(f"Input NC   : {nc_path}")
    print(f"Summary    : {summary_path}")
    if viz_enabled:
        print(f"Plots dir  : {viz_dir}")
    print(f"Manifest   : {contract['manifest']}")


if __name__ == "__main__":
    main()
