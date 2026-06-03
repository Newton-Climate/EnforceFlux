from __future__ import annotations

import argparse

from enforceflux.config import load_config
from enforceflux.osse import run_osse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a simple EnforceFlux OSSE")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    output = run_osse(config)

    print("OSSE summary")
    print(f"Sources: {len(output.x_true)} Instruments: {len(output.y)}")
    print(f"Posterior mean: {output.inversion.x_posterior}")
    print(f"Posterior std:  {output.metrics.posterior_std}")
    print(f"Jacobian rank: {output.metrics.jacobian_rank}")
    print(f"Null space dim: {output.metrics.null_space_dimension}")
    print(f"Condition #:  {output.metrics.condition_number:.3g}")


if __name__ == "__main__":
    main()
