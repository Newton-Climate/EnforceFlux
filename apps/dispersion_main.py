#!/usr/bin/env python3
"""Run any transport model from one stage YAML.

AERMOD, FLEXPART, and MicroHH all run from the same file — switch models by
changing ``dispersion.transport.model`` — and all return the same result
shape, with simulation output written as a canonical
``concentration(time, y, x)`` NetCDF in ng m-3.

The dispersion stage is a source stage: it consumes no upstream RunDir.
Its declared output roles are:

* ``concentration_field`` — the canonical NetCDF (``concentration.nc``).

Usage:
    enforceflux dispersion --config configs/dispersion/single_source_aermod.yaml
    enforceflux dispersion --config configs/dispersion/main.yaml --model flexpart
    enforceflux dispersion --config configs/dispersion/main.yaml --mode operator
    enforceflux dispersion --config configs/dispersion/main.yaml --dry-run
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a transport model from a dispersion-stage YAML config"
    )
    parser.add_argument("--config", required=True, help="Path to the dispersion-stage YAML")
    parser.add_argument(
        "--model",
        choices=["aermod", "flexpart", "microhh"],
        help="Override dispersion.transport.model from the config.",
    )
    parser.add_argument(
        "--mode",
        choices=["simulation", "operator"],
        help="Override dispersion.transport.mode from the config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Generate the model's input files without executing it. The only "
            "path available when a Fortran binary is not built."
        ),
    )
    parser.add_argument(
        "--print-jacobian",
        action="store_true",
        help="In operator mode, print the full Jacobian rather than a summary.",
    )
    return parser


def main() -> None:
    from enforceflux.runs import load_stage_config, open_run_dir
    from enforceflux.transport import TransportRunConfig, run_transport

    args = build_parser().parse_args()

    stage_cfg = load_stage_config(args.config, expected_stage="dispersion")

    block = dict(stage_cfg.block)
    if args.model or args.mode:
        transport = dict(block.get("transport") or {})
        if args.model:
            transport["model"] = args.model
        if args.mode:
            transport["mode"] = args.mode
        block["transport"] = transport

    run_dir = open_run_dir(
        stage="dispersion",
        run_name=stage_cfg.run_name,
        outputs_root=stage_cfg.outputs_root,
        inputs={k: str(v) for k, v in stage_cfg.inputs.items()},
    )
    run_dir.snapshot_config(stage_cfg.snapshot)

    output_path = run_dir.path("concentration.nc")

    run = TransportRunConfig.from_dict(
        block,
        base_dir=stage_cfg.yaml_dir,
        output_path=output_path,
    )

    print("EnforceFlux dispersion")
    print(f"Config     : {stage_cfg.yaml_path}")
    print(f"Run name   : {stage_cfg.run_name}")
    print(f"Run dir    : {run_dir.root}")
    print(f"Model      : {run.model}")
    print(f"Mode       : {run.mode}")
    print(f"Sources    : {len(run.sources)}")
    print(f"Receptors  : {len(run.receptors)}")
    size_x, size_y = run.domain.size_m
    print(f"Origin     : {run.domain.origin_lon}, {run.domain.origin_lat}")
    print(f"Domain     : x {run.domain.x_min:g}..{run.domain.x_max:g} m, "
          f"y {run.domain.y_min:g}..{run.domain.y_max:g} m "
          f"({size_x/1000:.1f} x {size_y/1000:.1f} km) @ {run.domain.spacing_m} m")
    lon_min, lat_min, lon_max, lat_max = run.domain.bounds_lonlat
    print(f"  (derived : {lon_min:.4f}..{lon_max:.4f} lon, "
          f"{lat_min:.4f}..{lat_max:.4f} lat)")
    if run.start and run.end:
        print(f"Window     : {run.start.isoformat()} -> {run.end.isoformat()}")
    print(f"Output     : {run.output.path}")
    print()

    result = run_transport(run, dry_run=args.dry_run)

    if result.met is not None:
        print(f"Meteorology: {len(result.met)} records from "
              f"{result.met.provenance.get('source', 'unknown')}, "
              f"{result.met.start:%Y-%m-%d %H:%M} -> {result.met.end:%Y-%m-%d %H:%M}")
        print()

    print(result.summary())

    if args.print_jacobian and result.g is not None:
        print("\nJacobian [" + result.units + "]")
        header = "  ".join(f"{c:>14s}" for c in result.column_labels)
        print(f"{'observation':>34s}  {header}")
        for label, row in zip(result.row_labels, result.g):
            name = label if isinstance(label, str) else " / ".join(str(p) for p in label)
            values = "  ".join(f"{v:14.5g}" for v in row)
            print(f"{name:>34s}  {values}")

    if output_path.is_file():
        run_dir.record_output("concentration.nc", role="concentration_field")
        contract = run_dir.finalize()
        print()
        print(f"Manifest   : {contract['manifest']}")
    else:
        print()
        print(f"(no {output_path.name} written — dry-run or model prepared inputs only; "
              f"manifest not finalized)")


if __name__ == "__main__":
    main()
