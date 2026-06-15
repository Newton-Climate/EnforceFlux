#!/usr/bin/env python3
"""Run FLEXPART simulation from a YAML config.

Usage:
    python apps/simulation_main.py --config apps/simulation_main.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _load_mode(yaml_path: Path) -> str:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: pip install pyyaml") from exc

    data = yaml.safe_load(yaml_path.read_text()) or {}
    sim = data.get("simulation", {})
    mode = str(sim.get("mode", "forward")).strip().lower()
    if mode not in {"forward", "backward"}:
        raise ValueError(
            f"Invalid simulation.mode={mode!r}. Expected 'forward' or 'backward'."
        )
    return mode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FLEXPART with EnforceFlux using YAML configuration"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to simulation YAML config",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write FLEXPART input files but do not execute FLEXPART",
    )
    return parser


def main() -> None:
    from enforceflux.flexpart import FlexpartSimulation

    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    sim = FlexpartSimulation.from_yaml(config_path)
    mode = _load_mode(config_path)
    sim.config.ldirect = 1 if mode == "forward" else -1

    print("EnforceFlux simulation_main")
    print(f"Config     : {config_path}")
    print(f"Mode       : {mode} (ldirect={sim.config.ldirect})")
    print(
        f"Window     : {sim.config.start.isoformat()} -> {sim.config.end.isoformat()}"
    )
    print(f"Sources    : {len(sim.config.sources)}")
    print(f"Run dir    : {sim.config.run_dir}")
    print(f"Output NC  : {sim.config.output_path}")

    if args.prepare_only:
        run_dir = sim.prepare()
        print(f"Prepared FLEXPART input files in: {run_dir}")
        return

    output_nc = sim.run()
    print(f"Simulation complete. Wrote: {output_nc}")


if __name__ == "__main__":
    main()
