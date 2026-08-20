#!/usr/bin/env python3
"""Compare matched FLEXPART and Gaussian-plume inversions for the LES truth."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).with_name("flexpart_gaussian_inversion_comparison.pdf")
NETWORK_SIZES = (8, 16)
MODELS = (
    ("FLEXPART", ROOT / "notebooks/hetero_experiments/paired_observability_results.csv"),
    ("Gaussian plume", ROOT / "notebooks/hetero_experiments/paired_observability_results_aermod.csv"),
)
MODES = (("total_uniform", "Domain-total state"), ("spatial_gp", "Spatial GP state"))
STYLE = {
    "point": {"color": "#0072B2", "label": "Centered point sensors"},
    "op": {"color": "#D55E00", "label": "500 m open-path beams"},
}


def rows_for(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="") as stream:
        raw = csv.DictReader(stream)
        rows = [
            {
                "n": int(row["n_instruments"]),
                "technology": row["technology"],
                "mode": row["mode"],
                "error": float(row["E_Q"]),
            }
            for row in raw
            if int(row["L_source_m"]) == 200
            and float(row["cv"]) == 2.0
            and int(row["n_instruments"]) in NETWORK_SIZES
        ]
    expected = len(NETWORK_SIZES) * len(STYLE) * len(MODES) * 5
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} matched rows in {path}, found {len(rows)}")
    return rows


def main() -> None:
    model_rows = [(name, rows_for(path)) for name, path in MODELS]
    ymax = 10.0 ** np.ceil(
        np.log10(max(float(row["error"]) for _, rows in model_rows for row in rows))
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), sharex=True, sharey=True,
                             constrained_layout=True)

    for row_index, (model_name, rows) in enumerate(model_rows):
        grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
        for row in rows:
            grouped[(str(row["mode"]), str(row["technology"]), int(row["n"]))].append(
                float(row["error"])
            )
        for col_index, (mode, mode_name) in enumerate(MODES):
            axis = axes[row_index, col_index]
            for technology, style in STYLE.items():
                medians = []
                for n in NETWORK_SIZES:
                    values = np.asarray(grouped[(mode, technology, n)], float)
                    offsets = np.linspace(-0.07, 0.07, len(values))
                    axis.scatter(n * (1.0 + offsets), values, s=31, alpha=0.27,
                                 color=style["color"], linewidths=0)
                    medians.append(float(np.median(values)))
                axis.plot(NETWORK_SIZES, medians, marker="o", markersize=5,
                          linewidth=2.4, color=style["color"], label=style["label"])
            axis.axhline(0.20, color="0.20", linestyle=":", linewidth=1.1)
            axis.axhline(1.00, color="0.55", linestyle="--", linewidth=1.0)
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.set_xticks(NETWORK_SIZES, [str(n) for n in NETWORK_SIZES])
            axis.set_ylim(0.008, ymax)
            axis.grid(axis="y", which="both", color="0.88", linewidth=0.7)
            if row_index == 0:
                axis.set_title(mode_name)
            if col_index == 0:
                axis.set_ylabel(
                    f"{model_name}\nrelative total-emission error"
                )
            if row_index == len(MODELS) - 1:
                axis.set_xlabel("Number of colocated instruments")

    axes[0, 0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle(
        "Same LES nature run (L = 200 m, CV = 2): transport-model comparison\n"
        "Faint markers: five layouts; solid lines: medians; dotted: 20% target; dashed: 100% error",
        fontsize=11,
    )
    fig.savefig(OUTPUT, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
