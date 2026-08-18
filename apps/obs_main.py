"""Real-observation ingress stage.

Reads a user-supplied observation file, normalizes units / QC, and emits an
``obs.parquet`` artifact with the same schema as ``apps/instrument_main.py``
so downstream ``flux`` runs cannot tell virtual and real obs apart.

Payload wiring (parquet write + RunDir manifest) lands with the run-artifact
contract in ``src/enforceflux/runs/``; this stub advertises the subcommand and
the argument shape so the unified CLI dispatcher is complete now.
"""
from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enforceflux obs",
        description="Ingest real observations into the pipeline (obs.parquet).",
    )
    parser.add_argument("--config", required=True, help="Path to obs-stage YAML config")
    return parser


def main() -> int:
    build_parser().parse_args()
    print(
        "obs stage: ingress wiring is added with the RunDir contract (D2). "
        "This stub reserves the subcommand.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
