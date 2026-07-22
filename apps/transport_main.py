#!/usr/bin/env python3
"""Run any transport model from one YAML config.

AERMOD, FLEXPART, and MicroHH all run from the same file — switch models by
changing ``transport.model`` — and all return the same result shape, with
simulation output written as a canonical ``concentration(time, y, x)`` NetCDF
in ng m-3.

Usage:
    python apps/transport_main.py --config apps/transport_main.yaml
    python apps/transport_main.py --config apps/transport_main.yaml --model flexpart
    python apps/transport_main.py --config apps/transport_main.yaml --mode operator
    python apps/transport_main.py --config apps/transport_main.yaml --dry-run
"""
import argparse
import dataclasses
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a transport model from a shared YAML config"
    )
    parser.add_argument("--config", required=True, help="Path to the transport YAML config")
    parser.add_argument(
        "--model",
        choices=["aermod", "flexpart", "microhh"],
        help="Override transport.model from the config.",
    )
    parser.add_argument(
        "--mode",
        choices=["simulation", "operator"],
        help="Override transport.mode from the config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Override the canonical output NetCDF path.",
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
    from enforceflux.transport import TransportRunConfig, run_transport

    args = build_parser().parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    run = TransportRunConfig.from_file(config_path)
    overrides = {}
    if args.model:
        overrides["model"] = args.model
    if args.mode:
        overrides["mode"] = args.mode
    if args.output:
        output = args.output if args.output.is_absolute() else (Path.cwd() / args.output)
        overrides["output"] = dataclasses.replace(run.output, path=output.resolve())
    if overrides:
        run = dataclasses.replace(run, **overrides)

    print("EnforceFlux transport_main")
    print(f"Config     : {config_path}")
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


if __name__ == "__main__":
    main()
