#!/usr/bin/env python3
"""Plot the focused 500 m OP versus point network sweep.

The data comprise the L_source=200 m, CV=2 LES nature run, with five network
layouts at each network size.  Lines show the median across layouts and faint
markers show individual layouts.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
NETWORK_SIZES = (1, 2, 4, 8, 16)
MODES = (("total_uniform", "Domain-total state"), ("spatial_gp", "Spatial GP state"))
STYLE = {
    "point": {"color": "#0072B2", "label": "Centered point sensors"},
    "op": {"color": "#D55E00", "label": "500 m open-path beams"},
}


def read_rows(input_path: Path) -> list[dict[str, float | str]]:
    with input_path.open(newline="") as stream:
        raw = list(csv.DictReader(stream))
    rows = []
    for row in raw:
        if int(row["L_source_m"]) != 200 or float(row["cv"]) != 2.0:
            continue
        rows.append({
            "n": int(row["n_instruments"]),
            "technology": row["technology"],
            "mode": row["mode"],
            "q_true": float(row["Q_true_kg_s"]),
            "q_hat": float(row["Q_hat_kg_s"]),
            "error": float(row["E_Q"]),
        })
    expected = len(NETWORK_SIZES) * 2 * 2 * 5
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} focused-sweep rows, found {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-model", choices=("flexpart", "aermod"), default="flexpart")
    args = parser.parse_args()
    suffix = "_aermod" if args.operator_model == "aermod" else ""
    input_path = ROOT / f"notebooks/hetero_experiments/paired_observability_results{suffix}.csv"
    output = Path(__file__).with_name(f"paired_op_point_network_sweep{suffix}.pdf")
    rows = read_rows(input_path)
    y_max = max(1.0e4, 10.0 ** np.ceil(np.log10(max(float(row["error"]) for row in rows))))
    grouped: dict[tuple[str, str, int], list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["mode"]), str(row["technology"]), int(row["n"]))].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=True, constrained_layout=True)
    for ax, (mode, title) in zip(axes, MODES):
        for technology, style in STYLE.items():
            medians = []
            for n in NETWORK_SIZES:
                values = np.array([float(r["error"]) for r in grouped[(mode, technology, n)]])
                # Jitter makes the five layout replicates distinguishable without
                # implying an additional uncertainty model.
                offsets = np.linspace(-0.10, 0.10, len(values))
                ax.scatter(
                    np.asarray(n, float) * (1.0 + offsets), values,
                    color=style["color"], alpha=0.28, s=28, linewidths=0, zorder=2,
                )
                medians.append(float(np.median(values)))
            ax.plot(
                NETWORK_SIZES, medians, marker="o", markersize=5, linewidth=2.4,
                color=style["color"], label=style["label"], zorder=3,
            )

        ax.axhline(0.20, color="0.25", linestyle=":", linewidth=1.2, label="20% target")
        ax.axhline(1.00, color="0.55", linestyle="--", linewidth=1.0, label="100% error")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(NETWORK_SIZES, [str(n) for n in NETWORK_SIZES])
        ax.set_ylim(0.008, y_max)
        ax.set_title(title)
        ax.set_xlabel("Number of colocated instruments")
        ax.grid(axis="y", which="both", color="0.88", linewidth=0.7)

    axes[0].set_ylabel(r"Total-emission relative error, $|Q_{hat}-Q_{true}|/Q_{true}$")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle(
        "LES nature run: L = 200 m, CV = 2; five layouts per network size\n"
        "Faint markers are layouts; solid lines are medians; priors are broad and zero-centred",
        fontsize=11,
    )
    fig.savefig(output, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
