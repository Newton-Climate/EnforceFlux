#!/usr/bin/env python3
"""Build deterministic, accuracy-based deck outputs from the bLS/LES runs.

This is intentionally not a Monte-Carlo success-probability analysis.  Every
cell is one paired OSSE result and the primary metric is relative total-flux
error::

    E_Q = abs(Q_hat - Q_true) / Q_true

The script scans completed runs, writes audit-friendly CSVs, and renders the
four result figures needed by the preliminary deck.  Threshold products use
the explicit wording "smallest tested N" because accuracy need not improve
monotonically under transport-model mismatch.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = ROOT / "runs"
OUT_ROOT = ROOT / "notebooks/hetero_experiments"
FIG_ROOT = OUT_ROOT / "figures"

DEFAULT_NS = (1, 2, 3, 4, 5)
DEFAULT_LS = (50, 100, 200)
DEFAULT_CVS = (0.5, 1.0, 2.0)
TECHS = ("op", "point_match")
E_TARGET = 0.20

RUN_RE = re.compile(
    r"^source_heterogeneity_les_source_l(?P<L>\d+)_cv(?P<cv>[0-9p]+)_"
    r"n(?P<n>\d+)_(?P<tech>op|point_match)$"
)


def _float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _scalar_nested(value) -> float | None:
    while isinstance(value, list) and value:
        value = value[0]
    return _float(value)


def collect_rows(ns: tuple[int, ...]) -> list[dict]:
    rows: list[dict] = []
    allowed_n = set(ns)
    for run_dir in sorted(RUNS_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        match = RUN_RE.match(run_dir.name)
        if not match:
            continue
        n = int(match["n"])
        if n not in allowed_n:
            continue
        L = int(match["L"])
        cv = float(match["cv"].replace("p", "."))
        tech = match["tech"]
        analysis = _read_json(run_dir / "analysis/summary.json")
        flux = _read_json(run_dir / "flux/summary.json")
        e_q = _float((analysis.get("source_heterogeneity") or {}).get("E_Q"))
        q_hat = _scalar_nested(flux.get("x_opt_kg_s"))
        post_sigma = _scalar_nested(flux.get("posterior_sigma_kg_s"))
        q_true = None
        if e_q is not None and q_hat is not None:
            # All current fields are exactly normalized to this configured total.
            q_true = 0.027778
        rows.append({
            "L_m": L,
            "cv": cv,
            "n": n,
            "technology": tech,
            "E_Q": e_q,
            "accuracy": (1.0 - e_q) if e_q is not None else None,
            "Q_hat_kg_s": q_hat,
            "Q_true_kg_s": q_true,
            "posterior_sigma_kg_s": post_sigma,
            "meets_E_Q_0p20": bool(e_q is not None and e_q <= E_TARGET),
            "run": run_dir.name,
            "status": "ok" if e_q is not None else "missing_summary",
        })
    return rows


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _row_map(rows: list[dict]) -> dict[tuple[int, float, int, str], dict]:
    return {
        (int(r["L_m"]), float(r["cv"]), int(r["n"]), str(r["technology"])): r
        for r in rows if r.get("E_Q") is not None
    }


def threshold_rows(rows: list[dict], ns: tuple[int, ...]) -> list[dict]:
    by_key = _row_map(rows)
    out = []
    for L in DEFAULT_LS:
        for cv in DEFAULT_CVS:
            for tech in TECHS:
                values = [(n, by_key.get((L, cv, n, tech))) for n in ns]
                passing = [n for n, row in values if row and row["E_Q"] <= E_TARGET]
                first = min(passing) if passing else None
                sustained = None
                for i, (n, row) in enumerate(values):
                    tail = [tail_row for _tail_n, tail_row in values[i:]]
                    if tail and all(r is not None and r["E_Q"] <= E_TARGET for r in tail):
                        sustained = n
                        break
                errors = [row["E_Q"] for _n, row in values if row]
                nonmonotone = any(errors[i + 1] > errors[i] for i in range(len(errors) - 1))
                out.append({
                    "L_m": L,
                    "cv": cv,
                    "technology": tech,
                    "E_Q_target": E_TARGET,
                    "smallest_tested_N_meeting_target": first,
                    "smallest_tested_N_sustaining_target": sustained,
                    "nonmonotone_with_N": nonmonotone,
                    "n_tested": len(errors),
                })
    return out


def difference_rows(rows: list[dict]) -> list[dict]:
    by_key = _row_map(rows)
    out = []
    for L in DEFAULT_LS:
        for cv in DEFAULT_CVS:
            for n in sorted({int(r["n"]) for r in rows}):
                op = by_key.get((L, cv, n, "op"))
                pt = by_key.get((L, cv, n, "point_match"))
                e_op = op["E_Q"] if op else None
                e_pt = pt["E_Q"] if pt else None
                out.append({
                    "L_m": L,
                    "cv": cv,
                    "n": n,
                    "E_Q_op": e_op,
                    "E_Q_point": e_pt,
                    "point_minus_op_error": (e_pt - e_op) if e_op is not None and e_pt is not None else None,
                    "more_accurate": (
                        "op" if e_op is not None and e_pt is not None and e_op < e_pt
                        else "point" if e_op is not None and e_pt is not None and e_pt < e_op
                        else "tie" if e_op is not None and e_pt is not None
                        else "undefined"
                    ),
                })
    return out


def equivalence_rows(rows: list[dict], ns: tuple[int, ...], op_reference_n: int) -> list[dict]:
    """Accuracy equivalence: smallest tested point N with E_point <= E_op(ref)."""
    by_key = _row_map(rows)
    out = []
    for L in DEFAULT_LS:
        for cv in DEFAULT_CVS:
            op = by_key.get((L, cv, op_reference_n, "op"))
            e_ref = op["E_Q"] if op else None
            n_point = None
            e_point = None
            if e_ref is not None:
                for n in ns:
                    point = by_key.get((L, cv, n, "point_match"))
                    if point and point["E_Q"] <= e_ref:
                        n_point = n
                        e_point = point["E_Q"]
                        break
            out.append({
                "L_m": L,
                "cv": cv,
                "N_op_reference": op_reference_n,
                "E_Q_op_reference": e_ref,
                "N_point_equivalent_smallest_tested": n_point,
                "E_Q_point_at_equivalence": e_point,
                "K_points_per_op": (n_point / op_reference_n) if n_point else None,
                "status": "ok" if n_point else "undefined",
            })
    return out


def _log_edges(values: np.ndarray) -> np.ndarray:
    lv = np.log10(values.astype(float))
    out = np.empty(values.size + 1)
    out[1:-1] = 10 ** ((lv[:-1] + lv[1:]) / 2)
    out[0] = 10 ** (lv[0] - (lv[1] - lv[0]) / 2)
    out[-1] = 10 ** (lv[-1] + (lv[-1] - lv[-2]) / 2)
    return out


def _linear_edges(values: np.ndarray) -> np.ndarray:
    out = np.empty(values.size + 1)
    out[1:-1] = (values[:-1] + values[1:]) / 2
    out[0] = values[0] - (values[1] - values[0]) / 2
    out[-1] = values[-1] + (values[-1] - values[-2]) / 2
    return out


def plot_accuracy_vs_cv(rows: list[dict], ns: tuple[int, ...], comparison_n: int, out: Path) -> None:
    by_key = _row_map(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, constrained_layout=True)
    palette = ("#0072B2", "#009E73", "#D55E00", "#CC79A7", "#E69F00")
    colors = {L: palette[i % len(palette)] for i, L in enumerate(DEFAULT_LS)}
    for ax, tech in zip(axes, TECHS):
        for L in DEFAULT_LS:
            vals = [by_key.get((L, cv, comparison_n, tech)) for cv in DEFAULT_CVS]
            y = [v["E_Q"] if v else np.nan for v in vals]
            ax.plot(DEFAULT_CVS, y, marker="o", lw=2, color=colors[L], label=f"L = {L} m")
        ax.axhline(E_TARGET, color="0.25", ls="--", lw=1.2, label="20% target")
        ax.set_xlabel("Source-field coefficient of variation, CV")
        ax.set_title("400 m open path" if tech == "op" else "Midpoint-matched point sensors")
        ax.set_xticks(DEFAULT_CVS)
        ax.grid(alpha=0.22)
    axes[0].set_ylabel(r"Relative total-flux error  $E_Q$")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Preliminary retrieval accuracy at N = {comparison_n} (one source and meteorological realization)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_phase(rows: list[dict], ns: tuple[int, ...], out: Path) -> None:
    by_key = _row_map(rows)
    ls = np.asarray(DEFAULT_LS, float)
    nvals = np.asarray(ns, float)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for ir, tech in enumerate(TECHS):
        for ic, cv in enumerate(DEFAULT_CVS):
            z = np.full((len(ns), len(DEFAULT_LS)), np.nan)
            for iy, n in enumerate(ns):
                for ix, L in enumerate(DEFAULT_LS):
                    row = by_key.get((L, cv, n, tech))
                    if row:
                        z[iy, ix] = row["E_Q"]
            ax = axes[ir, ic]
            image = ax.pcolormesh(
                _log_edges(ls), _log_edges(nvals), z,
                norm=LogNorm(vmin=0.02, vmax=1.0), cmap="viridis_r", shading="flat",
            )
            X, Y = np.meshgrid(ls, nvals)
            if np.isfinite(z).sum() >= 4 and np.nanmin(z) <= E_TARGET <= np.nanmax(z):
                ax.contour(X, Y, z, levels=[E_TARGET], colors="white", linewidths=1.8, linestyles="--")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(DEFAULT_LS, [str(v) for v in DEFAULT_LS])
            ax.set_yticks(ns, [str(v) for v in ns])
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
            if ir == 0:
                ax.set_title(f"CV = {cv:g}")
            if ic == 0:
                ax.set_ylabel(("Open path\n" if tech == "op" else "Point matched\n") + "N instruments")
            if ir == 1:
                ax.set_xlabel("Correlation length L [m]")
    if image is not None:
        fig.colorbar(image, ax=axes, label=r"Relative total-flux error  $E_Q$")
    fig.suptitle("Preliminary deterministic accuracy surface; dashed contour marks 20% error")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_op_point_difference(rows: list[dict], comparison_n: int, out: Path) -> None:
    by_key = _row_map(rows)
    op = np.full((len(DEFAULT_CVS), len(DEFAULT_LS)), np.nan)
    pt = np.full_like(op, np.nan)
    for iy, cv in enumerate(DEFAULT_CVS):
        for ix, L in enumerate(DEFAULT_LS):
            ro = by_key.get((L, cv, comparison_n, "op"))
            rp = by_key.get((L, cv, comparison_n, "point_match"))
            if ro:
                op[iy, ix] = ro["E_Q"]
            if rp:
                pt[iy, ix] = rp["E_Q"]
    delta = pt - op
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9), constrained_layout=True)
    xe = _log_edges(np.asarray(DEFAULT_LS, float))
    ye = _linear_edges(np.asarray(DEFAULT_CVS, float))
    vmax = max(np.nanmax(op), np.nanmax(pt), E_TARGET)
    for ax, data, title in zip(axes[:2], (pt, op), ("Point sensors", "Open path")):
        im = ax.pcolormesh(xe, ye, data, vmin=0, vmax=vmax, cmap="viridis_r", shading="flat")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label=r"$E_Q$ (lower is better)")
    dmax = max(abs(float(np.nanmin(delta))), abs(float(np.nanmax(delta))), 1e-6)
    imd = axes[2].pcolormesh(
        xe, ye, delta, norm=TwoSlopeNorm(vmin=-dmax, vcenter=0, vmax=dmax),
        cmap="RdBu", shading="flat",
    )
    axes[2].set_title(r"Point error $-$ OP error")
    fig.colorbar(imd, ax=axes[2], label="Positive = open path more accurate")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(DEFAULT_LS, [str(v) for v in DEFAULT_LS])
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_yticks(DEFAULT_CVS, [str(v) for v in DEFAULT_CVS])
        ax.set_xlabel("L [m]")
    axes[0].set_ylabel("CV")
    fig.suptitle(f"Technology comparison at N = {comparison_n}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_equivalence(rows: list[dict], out: Path) -> None:
    z = np.full((len(DEFAULT_CVS), len(DEFAULT_LS)), np.nan)
    for row in rows:
        if row["K_points_per_op"] is None:
            continue
        iy = DEFAULT_CVS.index(float(row["cv"]))
        ix = DEFAULT_LS.index(int(row["L_m"]))
        z[iy, ix] = float(row["K_points_per_op"])
    fig, ax = plt.subplots(figsize=(6.8, 4.7), constrained_layout=True)
    im = ax.pcolormesh(
        _log_edges(np.asarray(DEFAULT_LS, float)),
        _linear_edges(np.asarray(DEFAULT_CVS, float)), z,
        cmap="magma", shading="flat", vmin=1,
        vmax=max(2, float(np.nanmax(z))) if np.isfinite(z).any() else 2,
    )
    for iy, cv in enumerate(DEFAULT_CVS):
        for ix, L in enumerate(DEFAULT_LS):
            label = "—" if not np.isfinite(z[iy, ix]) else f"{z[iy, ix]:g}"
            ax.text(L, cv, label, ha="center", va="center", color="white", fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks(DEFAULT_LS, [str(v) for v in DEFAULT_LS])
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_yticks(DEFAULT_CVS, [str(v) for v in DEFAULT_CVS])
    ax.set_xlabel("Correlation length L [m]")
    ax.set_ylabel("CV")
    ax.set_title("Smallest tested point network matching one open path by accuracy")
    fig.colorbar(im, ax=ax, label=r"$K = N_{point}/N_{OP}$")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ns", default=",".join(str(n) for n in DEFAULT_NS))
    parser.add_argument("--comparison-n", type=int, default=4)
    parser.add_argument("--op-reference-n", type=int, default=1)
    parser.add_argument("--strict", action="store_true", help="Fail unless the L × CV × N × technology grid is complete.")
    args = parser.parse_args()

    ns = tuple(dict.fromkeys(int(v.strip()) for v in args.ns.split(",") if v.strip()))
    rows = collect_rows(ns)
    expected = len(DEFAULT_LS) * len(DEFAULT_CVS) * len(ns) * len(TECHS)
    ok = sum(r["status"] == "ok" for r in rows)
    if args.strict and ok != expected:
        raise SystemExit(f"incomplete deck sweep: {ok}/{expected} completed cells")

    summary_fields = [
        "L_m", "cv", "n", "technology", "E_Q", "accuracy", "Q_hat_kg_s",
        "Q_true_kg_s", "posterior_sigma_kg_s", "meets_E_Q_0p20", "run", "status",
    ]
    _write_csv(OUT_ROOT / "hetero_observability_summary.csv", rows, summary_fields)

    thresholds = threshold_rows(rows, ns)
    _write_csv(
        OUT_ROOT / "hetero_observability_smallest_tested_n.csv", thresholds,
        ["L_m", "cv", "technology", "E_Q_target", "smallest_tested_N_meeting_target",
         "smallest_tested_N_sustaining_target", "nonmonotone_with_N", "n_tested"],
    )

    differences = difference_rows(rows)
    _write_csv(
        OUT_ROOT / "op_vs_point_accuracy_difference.csv", differences,
        ["L_m", "cv", "n", "E_Q_op", "E_Q_point", "point_minus_op_error", "more_accurate"],
    )

    equivalence = equivalence_rows(rows, ns, args.op_reference_n)
    _write_csv(
        OUT_ROOT / "op_point_equivalence_accuracy.csv", equivalence,
        ["L_m", "cv", "N_op_reference", "E_Q_op_reference",
         "N_point_equivalent_smallest_tested", "E_Q_point_at_equivalence",
         "K_points_per_op", "status"],
    )

    plot_accuracy_vs_cv(rows, ns, args.comparison_n, FIG_ROOT / "hetero_accuracy_vs_cv.pdf")
    plot_phase(rows, ns, FIG_ROOT / "hetero_observability_phase_diagram.pdf")
    plot_op_point_difference(rows, args.comparison_n, FIG_ROOT / "op_vs_point_network_heterogeneity.pdf")
    plot_equivalence(equivalence, FIG_ROOT / "op_point_equivalence_accuracy.pdf")

    print(f"completed cells: {ok}/{expected}")
    print(f"summary: {OUT_ROOT / 'hetero_observability_summary.csv'}")
    print(f"figures: {FIG_ROOT}")


if __name__ == "__main__":
    main()
