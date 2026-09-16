#!/usr/bin/env python3
"""Figures for the quadratic-form sigma_rep validation. Run run_validation.py first.

    python make_figures.py              # 8-seed study -> figures/fig*.{png,pdf,txt}
    python make_figures.py --nseeds 20  # reads *_s20.csv  -> figures/fig*_s20.*

Each figure is written as PNG + PDF with a matching .txt: alt text, then the
plotted values as CSV. Every live Figure is passed through figure_qc and the
report printed verbatim.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import stats  # noqa: E402

from quadratic_rep import receptors  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
QC = os.environ.get("FIGURE_QC_DIR")
if QC:
    sys.path.insert(0, QC)
    from figure_qc import check_figure, format_report
else:
    check_figure = None

plt.rcParams.update({
    "figure.constrained_layout.use": True,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10, "legend.fontsize": 8,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})

# Okabe-Ito; every series also differs by marker and linestyle.
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
SERIES = [  # column, label, colour, marker, linestyle
    ("emp_total", "Empirical, total error", OI["black"], "s", "-"),
    ("emp_pure", "Empirical, perfect transport", OI["blue"], "o", "-"),
    ("pred_loglinear", "Predicted, log-linear (as specified)", OI["vermillion"], "^", "--"),
    ("pred_lognormal", "Predicted, exact lognormal", OI["orange"], "v", ":"),
    ("pred_mc", "Predicted, generator Monte Carlo", OI["green"], "D", "-."),
]
CV_STYLE = {0.5: (OI["sky"], "o"), 1.0: (OI["orange"], "s"), 2.0: (OI["vermillion"], "^")}
L_STYLE = {100.0: (OI["sky"], "o"), 250.0: (OI["orange"], "s"), 500.0: (OI["vermillion"], "^")}
GEOM_FILL = {"point": "full", "open_path": "none"}
GEOM_LABEL = {"point": "Point sensor", "open_path": "Open path"}
SFX, M = "", 8  # set by main(): file suffix and seeds per condition


def finish(fig, stem: str, alt: str, table: pd.DataFrame) -> None:
    OUT.mkdir(exist_ok=True)
    stem = f"{stem}{SFX}"
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}")
    (OUT / f"{stem}.txt").write_text(alt.strip() + "\n\n" + table.to_csv(index=False, float_format="%.4g"))
    if check_figure is not None:
        print(f"--- QC {stem}")
        print(format_report(check_figure(fig)))
    plt.close(fig)


# ------------------------------------------------------------ 1. pooled


def fig_pooled(pooled: pd.DataFrame, null: pd.DataFrame) -> None:
    panels = [("CV", "sd", "CV (dimensionless)", "Pooled SD of total-flux error (%)"),
              ("L_m", "sd", "L (m)", "Pooled SD of total-flux error (%)"),
              ("L_m", "mae", "L (m)", "Mean absolute total-flux error (%)")]
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.8), sharey="col")
    rows = []
    for i, geom in enumerate(("point", "open_path")):
        for j, (by, stat, xl, yl) in enumerate(panels):
            ax = axes[i, j]
            d = pooled[(pooled.geometry == geom) & (pooled.pooled_by == by)].sort_values("value")
            x = np.arange(len(d))
            nb = null[(null.geometry == geom) & (null.pooled_by == by) & (null.statistic == stat)].sort_values("value")
            ax.fill_between(x, 100 * nb.null_q025, 100 * nb.null_q975, color="0.85", lw=0,
                            label=f"Shared-seed null, 95% ({M}-seed replays)")
            rows += [{"geometry": geom, "pooled_by": by, "statistic": stat, "value": v,
                      "series": f"null {q}", "percent": 100 * yy}
                     for q in ("q025", "median", "q975") for v, yy in zip(nb.value, nb[f"null_{q}"])]
            for col, lab, c, mk, ls in SERIES:
                y = 100 * d[f"{stat}_{col}"].to_numpy()
                ax.plot(x, y, color=c, marker=mk, ls=ls, lw=1.4, ms=5,
                        mfc=c if col.startswith("emp") else "white", label=lab)
                rows += [{"geometry": geom, "pooled_by": by, "statistic": stat, "value": v,
                          "series": lab, "percent": yy} for v, yy in zip(d.value, y)]
            ax.set_xticks(x, [f"{v:g}" for v in d.value])
            ax.set_xlim(-0.3, len(d) - 0.7)
            ax.set_ylim(0, 60 if stat == "sd" else 45)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(f"{GEOM_LABEL[geom]}: {stat.upper()} by {xl.split()[0]}", fontsize=9)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=2, frameon=False)
    t = pd.DataFrame(rows)
    # Every claim below is computed, so the text stays true for any seed count.
    sd = null[null.statistic == "sd"]
    l500 = (sd.pooled_by == "L_m") & (sd.value == 500)
    p_rest, p_500 = sd[~l500].study_percentile_in_null, sd[l500].study_percentile_in_null
    n_above = sum((pooled[f"sd_pred_{v}"] > pooled.sd_emp_total).sum() for v in ("loglinear", "lognormal", "mc"))
    alt = ("Six line panels, point sensors on the top row and open paths on the bottom, comparing "
           f"pooled total-flux error statistics (percent, {pooled.n_runs.iloc[0]} inversions per point, "
           f"{M} seeds) with quadratic-form predictions pooled the same way. Columns show sample SD by CV, "
           "sample SD by correlation length L, and mean absolute error by L (the statistic behind the "
           "published 10.4, 11.8 and 19.0 percent). A grey band shows where the perfect-transport "
           f"statistic falls in 95 percent of 200 replays of the study's shared {M}-seed design under "
           f"the generator. Predicted SD exceeds empirical total-error SD in {n_above} of "
           f"{3 * len(pooled)} predictor-by-point comparisons. The empirical perfect-transport SD sits "
           f"between percentiles {p_rest.min():.0f} and {p_rest.max():.0f} of the band, except at "
           f"L = 500 m (percentiles {p_500.min():.0f} to {p_500.max():.0f}).")
    finish(fig, "fig1_pooled_comparison", alt, t)


# ------------------------------------------------------------ 2. scatter


def fig_scatter(cond: pd.DataFrame, metrics: pd.DataFrame, variant: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.2))
    v = np.concatenate([cond[f"sd_pred_{variant}"] ** 2, cond.sd_emp_total_lo ** 2,
                        cond.sd_emp_pure_lo ** 2, cond.sd_emp_total_hi ** 2, cond.sd_emp_pure_hi ** 2])
    lim = (10 ** (np.floor(2 * np.log10(v.min())) / 2), 10 ** (np.ceil(2 * np.log10(v.max())) / 2))
    rows = []
    for i, (t, tlab) in enumerate((("total", "total error"), ("pure", "perfect-transport error"))):
        met = metrics[(metrics.target == t) & (metrics.variant == variant)].iloc[0]
        for j, (key, style, nm) in enumerate((("CV", CV_STYLE, "CV"), ("L_m", L_STYLE, "L"))):
            ax = axes[i, j]
            ax.plot(lim, lim, color="0.5", lw=1, ls="--")
            for val, (c, mk) in style.items():
                for geom, fill in GEOM_FILL.items():
                    d = cond[(cond[key] == val) & (cond.geometry == geom)]
                    xp = d[f"sd_pred_{variant}"] ** 2
                    ye = d[f"sd_emp_{t}"] ** 2
                    # Error bars as plain lines: figure_qc misreads errorbar's LineCollection offsets.
                    for xx, lo, hi in zip(xp, d[f"sd_emp_{t}_lo"] ** 2, d[f"sd_emp_{t}_hi"] ** 2):
                        ax.plot([xx, xx], [lo, hi], color=c, lw=0.6, alpha=0.7)
                    ax.plot(xp, ye, mk, color=c, mfc=c if fill == "full" else "white", ms=5)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(*lim); ax.set_ylim(*lim)
            ax.set_aspect("equal")
            ax.set_xlabel(f"Predicted variance, {variant} (fraction$^2$)")
            ax.set_ylabel(f"Empirical {M}-seed variance (fraction$^2$)")
            ax.set_title(f"{tlab.capitalize()}, by {nm}", fontsize=9)
            ax.text(0.03, 0.97, f"r = {met.pearson_r_variance:.2f} (variance)\n"
                    f"RMSE = {met.rmse_sd_pp:.1f} pp (SD)\n"
                    f"median ratio pred/emp SD = {met.median_ratio_pred_over_emp:.2f}",
                    transform=ax.transAxes, va="top", fontsize=8)
            handles = [Line2D([], [], color=c, marker=mk, ls="none", label=f"{nm} = {v:g}")
                       for v, (c, mk) in style.items()]
            handles += [Line2D([], [], color="0.3", marker="o", mfc=("0.3" if f == "full" else "white"),
                               ls="none", label=GEOM_LABEL[g]) for g, f in GEOM_FILL.items()]
            ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    for r in cond.itertuples():
        rows.append({"geometry": r.geometry, "n": r.n, "CV": r.CV, "L_m": r.L_m,
                     "var_pred": getattr(r, f"sd_pred_{variant}") ** 2,
                     "var_emp_total": r.sd_emp_total**2, "var_emp_pure": r.sd_emp_pure**2})
    m_t = metrics[(metrics.target == "total") & (metrics.variant == variant)].iloc[0]
    m_p = metrics[(metrics.target == "pure") & (metrics.variant == variant)].iloc[0]
    alt = (f"Four log-log scatter panels of predicted ({variant} covariance) versus empirical {M}-seed "
           f"error variance for all 72 conditions (geometry, n, CV, L), with chi-square 95 percent "
           f"intervals on the empirical values and a dashed 1:1 line. Top row uses the total "
           f"retrieval error (r = {m_t.pearson_r_variance:.2f}, Spearman {m_t.spearman_rho_sd:.2f}, "
           f"median predicted/empirical SD {m_t.median_ratio_pred_over_emp:.2f}), bottom row the "
           f"perfect-transport error (r = {m_p.pearson_r_variance:.2f}, Spearman {m_p.spearman_rho_sd:.2f}, "
           f"median ratio {m_p.median_ratio_pred_over_emp:.2f}); left panels colour by CV, right by L, "
           f"filled markers for point sensors and open markers for open paths.")
    finish(fig, f"fig2_scatter_{variant}", alt, pd.DataFrame(rows))


# ------------------------------------------------------------ 3. z / PIT


def fig_z(z: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.1))
    q = np.linspace(0.005, 0.995, 200)
    rows = []
    for col, lab, c, mk in (("z_total_loglinear", "Total / log-linear", OI["black"], "o"),
                            ("z_pure_mc", "Pure / Monte Carlo", OI["green"], "D")):
        emp = np.quantile(z[col], q)
        th = stats.norm.ppf(q)
        axes[0].plot(th, emp, mk, color=c, ms=2.5, mfc="white", label=lab, markevery=4)
        rows += [{"panel": "qq", "series": lab, "normal_quantile": a, "z_quantile": b}
                 for a, b in zip(th[::10], emp[::10])]
    axes[0].plot([-3, 3], [-3, 3], color="0.5", ls="--", lw=1)
    axes[0].set_xlim(-3, 3); axes[0].set_ylim(-3, 3)
    axes[0].set_xlabel("Normal quantile (dimensionless)")
    axes[0].set_ylabel("z = error / predicted SD (dimensionless)")
    axes[0].set_title(f"QQ, all {len(z)} inversions", fontsize=9)
    axes[0].legend(frameon=False, loc="lower right")

    bins = np.linspace(0, 1, 11)
    cnt, _ = np.histogram(z.pit_pure_mc, bins)
    axes[1].bar(bins[:-1], cnt, width=0.1, align="edge", color=OI["green"], edgecolor="white",
                hatch="//", label="Stored fields")
    axes[1].axhline(len(z) / 10, color="0.3", ls="--", lw=1, label="Uniform")
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, max(cnt.max(), len(z) / 10) * 1.25)
    axes[1].set_xlabel("PIT of e_pure (dimensionless)")
    axes[1].set_ylabel("Inversions (count)")
    axes[1].set_title(f"Are the {M} seeds typical draws?", fontsize=9)
    axes[1].legend(frameon=False, loc="upper right")
    rows += [{"panel": "pit", "series": "count", "normal_quantile": b, "z_quantile": c}
             for b, c in zip(bins[:-1], cnt)]

    cond = z.groupby(["geometry", "n", "CV", "L_m"])
    sd_by = cond.z_pure_mc.std(ddof=1).reset_index()
    for cv, (c, mk) in CV_STYLE.items():
        d = sd_by[sd_by.CV == cv]
        axes[2].plot(d.L_m + {0.5: -15, 1.0: 0, 2.0: 15}[cv], d.z_pure_mc, mk, color=c, ms=4,
                     mfc="white", label=f"CV = {cv:g}")
    axes[2].plot([50, 550], [1, 1], color="0.5", ls="--", lw=1)
    axes[2].set_xlim(50, 550); axes[2].set_ylim(0, max(2.0, 1.1 * sd_by.z_pure_mc.max()))
    axes[2].set_xticks([100, 250, 500])
    axes[2].set_xlabel("L (m)")
    axes[2].set_ylabel("SD of z per condition (dimensionless)")
    axes[2].set_title("SD of z by condition", fontsize=9)
    axes[2].legend(frameon=False, loc="upper left")
    ks = stats.kstest(z.pit_pure_mc, "uniform")
    alt = (f"Three panels testing the prediction inversion by inversion ({len(z)} inversions, {M} seeds). "
           "Left: normal QQ plot of each error divided by its condition's predicted SD, against a 1:1 "
           f"line; SD of z is {z.z_total_loglinear.std():.2f} (total, log-linear) and {z.z_pure_mc.std():.2f} "
           "(pure, Monte Carlo). Middle: histogram of where each stored field's perfect-transport error "
           f"falls in the generator's own distribution; a flat histogram means typical draws "
           f"(Kolmogorov-Smirnov D = {ks.statistic:.2f}, p = {ks.pvalue:.1g}). Right: per-condition SD of z "
           "against L for each CV, where 1 marks a calibrated prediction.")
    finish(fig, "fig3_case_level", alt, pd.DataFrame(rows))


# ------------------------------------------------------------ 4. kernels


def cumulative_level(w: np.ndarray, frac: float) -> float:
    s = np.sort(w[w > 0])[::-1]
    return float(s[min(np.searchsorted(np.cumsum(s) / s.sum(), frac), s.size - 1)])


def fig_kernels(k: dict, inv: pd.DataFrame) -> None:
    x, y = k["x"], k["y"]
    cases = [("point", 1), ("open_path", 1), ("point", 4), ("open_path", 4)]
    vmax = max(np.abs(k[f"{g}_n{n}_dw"]).max() for g, n in cases) * 625
    pts = []
    for g, n in cases:
        for rc in receptors(inv[(inv.geometry == g) & (inv.n == n)].run.iloc[0]):
            h = rc.get("path_length_m", 0) / 2
            pts += [(rc["x_m"] - h, rc["y_m"] - h), (rc["x_m"] + h, rc["y_m"] + h)]
    pts = np.array(pts + [(-500, -500), (500, 500)])
    XLIM = (pts[:, 0].min() - 60, pts[:, 0].max() + 60)
    YLIM = (pts[:, 1].min() - 60, pts[:, 1].max() + 60)
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 7.0), sharex=True, sharey=True)
    rows = []
    for ax, (g, n) in zip(axes.ravel(), cases):
        dw = k[f"{g}_n{n}_dw"].reshape(y.size, x.size) * 625  # in units of the uniform weight
        w = k[f"{g}_n{n}_w"].reshape(y.size, x.size)
        m = ax.pcolormesh(x, y, dw, cmap="RdBu_r", shading="nearest", rasterized=True,
                          norm=TwoSlopeNorm(0, vmin=-1, vmax=vmax))
        for frac, ls in ((0.5, "-"), (0.9, "--")):
            ax.contour(x, y, w, levels=[cumulative_level(w, frac)], colors="k", linewidths=1, linestyles=ls)
        run = inv[(inv.geometry == g) & (inv.n == n)].run.iloc[0]
        for rc in receptors(run):
            if "path_length_m" in rc:
                b = np.deg2rad(rc["path_bearing_deg"]); h = rc["path_length_m"] / 2
                ax.plot([rc["x_m"] - h * np.sin(b), rc["x_m"] + h * np.sin(b)],
                        [rc["y_m"] - h * np.cos(b), rc["y_m"] + h * np.cos(b)],
                        color=OI["green"], lw=2.5)
            else:
                ax.plot(rc["x_m"], rc["y_m"], "D", color=OI["green"], mec="k", ms=5)
        ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_aspect("equal")
        ax.add_patch(plt.Rectangle((-500, -500), 1000, 1000, fill=False, lw=0.8, ec="0.3"))
        ax.set_xlabel("East-west distance (m)"); ax.set_ylabel("North-south distance (m)")
        ax.set_title(f"{GEOM_LABEL[g]}, n = {n}", fontsize=9)
        nz = np.sum(w > 0) / w.size
        rows.append({"design": f"{g}_n{n}", "dw_min_x_uniform": dw.min(), "dw_max_x_uniform": dw.max(),
                     "frac_cells_w_positive": nz,
                     "cells_in_50pct": int(np.sum(w >= cumulative_level(w, 0.5))),
                     "cells_in_90pct": int(np.sum(w >= cumulative_level(w, 0.9))),
                     "N_times_sum_dw_sq": float(np.sum(dw**2) / 625)})
    cb = fig.colorbar(m, ax=axes, shrink=0.7, label="dw x N (1 = uniform weight; -1 = unseen cell)")
    cb.set_ticks([-1, -0.5, 0, *range(10, int(vmax) + 1, 10)])
    cb.ax.tick_params(labelsize=8)
    fig.legend(handles=[Line2D([], [], color="k", ls="-", label="50% cumulative w_g"),
                        Line2D([], [], color="k", ls="--", label="90% cumulative w_g"),
                        Line2D([], [], color=OI["green"], marker="D", ls="none", label="Point sensor"),
                        Line2D([], [], color=OI["green"], lw=2.5, label="Open path")],
               loc="outside lower center", ncol=4, frameon=False)
    t = pd.DataFrame(rows)
    alt = ("Four maps of the mismatch kernel dw over the 1 km source domain, expressed as a multiple of "
           "the uniform weight 1/625, for point sensors and open paths at n = 1 and n = 4, with 50 and 90 "
           "percent cumulative-footprint contours and the receptors. Blue cells (-1) are unseen by the "
           "footprint; red cells are overweighted. Point sensors concentrate weight in a narrow upwind "
           "wedge, open paths spread it along the path, and four sensors widen coverage; see the table "
           "for the fraction of cells with non-zero weight and the peak overweighting.")
    finish(fig, "fig4_mismatch_kernels", alt, t)


def main(nseeds: int) -> int:
    global SFX, M
    SFX, M = ("" if nseeds == 8 else f"_s{nseeds}"), nseeds
    cond = pd.read_csv(HERE / f"conditions{SFX}.csv")
    if not (cond.m_realizations == nseeds).all():
        raise AssertionError(f"conditions{SFX}.csv is not a {nseeds}-seed table")
    pooled = pd.read_csv(HERE / f"pooled{SFX}.csv")
    metrics = pd.read_csv(HERE / f"metrics{SFX}.csv")
    z = pd.read_csv(HERE / f"inversions_z{SFX}.csv")
    fig_pooled(pooled, pd.read_csv(HERE / f"pooled_null{SFX}.csv"))
    for v in ("loglinear", "mc"):
        fig_scatter(cond, metrics, v)
    fig_z(z)
    if nseeds == 8:  # the kernels do not depend on the seeds
        inv = pd.read_csv(HERE.parent / "rep_error_scaling" / "inversions.csv")
        fig_kernels(dict(np.load(HERE / "kernels.npz")), inv)
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nseeds", type=int, default=8)
    raise SystemExit(main(ap.parse_args().nseeds))
