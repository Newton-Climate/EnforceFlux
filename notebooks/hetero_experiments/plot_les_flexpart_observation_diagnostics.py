#!/usr/bin/env python3
"""Diagnose LES observations against FLEXPART predictions at the true total flux."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[2]
COUNTS = (2, 3, 4, 8, 16)
LS = (200, 500, 1000)
CVS = (0.5, 1.0, 2.0)
COLORS = {"point": "#0072B2", "op": "#D55E00"}
LABELS = {"point": "Point", "op": "400 m OP"}


def slug(L: int, cv: float) -> str:
    return f"l{L}_cv{str(cv).replace('.', 'p')}"


def records() -> list[dict]:
    result = []
    for cv in CVS:
        for L in LS:
            case = slug(L, cv)
            truth_path = ROOT / (
                f"runs/source_heterogeneity_les_source_{case}/dispersion/truth_field.nc"
            )
            with Dataset(truth_path) as ds:
                q_true = float(np.sum(ds["F_true"][:] * ds["cell_area_m2"][:]))
            for n in COUNTS:
                for tech in ("point", "op"):
                    name = f"source_heterogeneity_les_source_{case}_n{n}_july_flex_{tech}"
                    matrices = np.load(ROOT / "runs" / name / "flux/matrices.npz")
                    g = np.asarray(matrices["G"], float).reshape(-1)
                    observed = np.asarray(matrices["y_obs"], float).reshape(-1)
                    predicted = g * q_true
                    active = (predicted > 0.0) & (observed > 0.0)
                    result.append({
                        "cv": cv, "L": L, "n": n, "tech": tech,
                        "observed": observed, "predicted": predicted,
                        "active": active,
                        "ratio": float(np.median(observed[active] / predicted[active])),
                    })
    return result


def main() -> None:
    rows = records()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25), constrained_layout=True)

    ax = axes[0]
    positive_values = []
    for tech in ("point", "op"):
        predicted = np.concatenate([r["predicted"][r["active"]] for r in rows if r["tech"] == tech])
        observed = np.concatenate([r["observed"][r["active"]] for r in rows if r["tech"] == tech])
        positive_values.extend((predicted, observed))
        ax.scatter(predicted, observed, s=16, alpha=0.38, color=COLORS[tech],
                   edgecolors="none", label=LABELS[tech])
    lo = min(float(a.min()) for a in positive_values)
    hi = max(float(a.max()) for a in positive_values)
    ax.plot([lo, hi], [lo, hi], color=".15", lw=1.2, ls=":", label="1:1")
    ax.set(xscale="log", yscale="log", xlim=(lo * 0.7, hi * 1.4), ylim=(lo * 0.7, hi * 1.4),
           xlabel=r"FLEXPART prediction at $Q_{true}$ (ng m$^{-3}$)",
           ylabel=r"LES pseudo-observation (ng m$^{-3}$)",
           title="Observation-level model comparison")
    ax.legend(frameon=False, loc="lower right")
    point_rows = [r for r in rows if r["tech"] == "point"]
    op_rows = [r for r in rows if r["tech"] == "op"]
    point_zero = sum(int(np.sum(r["predicted"] <= 0)) for r in point_rows)
    op_zero = sum(int(np.sum(r["predicted"] <= 0)) for r in op_rows)
    point_total = sum(len(r["predicted"]) for r in point_rows)
    op_total = sum(len(r["predicted"]) for r in op_rows)
    ax.text(0.03, 0.97,
            f"Zero FLEX sensitivity: point {point_zero}/{point_total}; OP {op_zero}/{op_total}",
            transform=ax.transAxes, va="top", fontsize=8)

    ax = axes[1]
    for tech, style in (("point", "-"), ("op", "--")):
        medians, lower, upper = [], [], []
        for n in COUNTS:
            ratios = np.array([r["ratio"] for r in rows if r["tech"] == tech and r["n"] == n])
            medians.append(float(np.median(ratios)))
            lower.append(float(np.percentile(ratios, 25)))
            upper.append(float(np.percentile(ratios, 75)))
        ax.fill_between(COUNTS, lower, upper, color=COLORS[tech], alpha=0.14)
        ax.plot(COUNTS, medians, style, color=COLORS[tech], marker="o",
                label=LABELS[tech])
    ax.axhline(1.0, color=".15", lw=1.2, ls=":")
    ax.set(xscale="log", yscale="log", xlabel="Number of sensors",
           ylabel=r"Median concentration ratio, LES / FLEXPART($Q_{true}$)",
           title="Systematic concentration residual")
    ax.set_xticks(COUNTS, [str(n) for n in COUNTS])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(frameon=False)
    ax.text(0.03, 0.04, "Bands: interquartile range across 9 L/CV cases",
            transform=ax.transAxes, fontsize=8)

    fig.suptitle("LES nature observations versus FLEXPART inversion operator")
    output = ROOT / "notebooks/hetero_experiments/les_flexpart_observation_diagnostics.pdf"
    fig.savefig(output, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
