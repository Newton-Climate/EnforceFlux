#!/usr/bin/env python3
"""OSSE sweep driver — run the M4 (L, CV, N, seed) grid and write parquet.

Usage:
    enforceflux osse-sweep --config configs/source_heterogeneity_e1_aermod/sweep.yaml
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run an EnforceFlux OSSE sweep")
    p.add_argument("--config", required=True, help="Path to sweep YAML")
    return p


def main() -> None:
    from enforceflux.osse.sweep import SweepConfig, run_sweep

    args = build_parser().parse_args()
    cfg = SweepConfig.from_yaml(Path(args.config))
    parquet = run_sweep(cfg)
    print("EnforceFlux osse-sweep")
    print(f"Config    : {args.config}")
    print(f"Run name  : {cfg.name}")
    print(f"Parquet   : {parquet}")


if __name__ == "__main__":
    main()
