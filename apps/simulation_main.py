#!/usr/bin/env python3
"""Run FLEXPART simulation from a YAML config.

Usage:
    python apps/simulation_main.py --config apps/simulation_main.yaml
"""
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
    from enforceflux.core.base import ITransportSimulation
    from enforceflux.flexpart.sim_config import load_simulation_config
    from enforceflux.utils.plugin_registry import get_plugin

    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    mode = _load_mode(config_path)
    ldirect = 1 if mode == "forward" else -1

    # Load the config for the run summary; execution goes through the registry.
    sim_config = load_simulation_config(config_path)

    print("EnforceFlux simulation_main")
    print(f"Config     : {config_path}")
    print(f"Mode       : {mode} (ldirect={ldirect})")
    print(
        f"Window     : {sim_config.start.isoformat()} -> {sim_config.end.isoformat()}"
    )
    print(f"Sources    : {len(sim_config.sources)}")
    print(f"Run dir    : {sim_config.run_dir}")
    print(f"Output NC  : {sim_config.output_path}")

    simulation = get_plugin(
        "enforceflux.transport_simulation", "flexpart", ITransportSimulation
    )()
    result = simulation.simulate(
        [],
        None,
        {
            "sim_config": str(config_path),
            "ldirect": ldirect,
            "dry_run": args.prepare_only,
        },
    )

    if args.prepare_only:
        print(f"Prepared FLEXPART input files in: {result.meta['run_dir']}")
        return

    print(f"Simulation complete. Wrote: {result.output_path}")


if __name__ == "__main__":
    main()
