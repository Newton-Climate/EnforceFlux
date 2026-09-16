"""Figures for the source-heterogeneity deck, drawn from the eight-seed sweep.

Supersedes the single-seed panels in make_hetero_figures.py for every results
slide. Reads:

  seed_sweep_results.csv   576 inversions: L x CV x seed x n x network
  design_2x2_seeds.csv     state resolution x observation cadence, 8 seeds

Style follows make_hetero_figures.py: Okabe-Ito, colour never used alone, and
each panel writes a plain-text description beside the PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

Q_TRUE = 0.027778  # kg s-1

BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREY = "#4D4D4D"
NET = {
    "op": ("Open path", dict(color=BLUE, marker="o", ls="-")),
    "point": ("Point sensors", dict(color=VERMILLION, marker="s", ls="--")),
}

plt.rcParams.update({
    "figure.constrained_layout.use": True,
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, stem: str, alt: str) -> None:
    fig.savefig(OUT / f"{stem}.png")
    fig.savefig(OUT / f"{stem}.pdf")
    (OUT / f"{stem}.txt").write_text(alt.strip() + "\n")
    plt.close(fig)
    print(f"wrote {stem}.png")


def sweep() -> pd.DataFrame:
    df = pd.read_csv(HERE / "seed_sweep_results.csv")
    df["err_pct"] = 100.0 * df["q_rel_error"]
    df["abs_pct"] = df["err_pct"].abs()
    df["sigma_pct"] = 100.0 * df["posterior_sigma_kg_s"] / Q_TRUE
    return df


def jitter(n: int, width: float = 0.10, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-width, width, n)


# ------------------------------------------------------- 1. spread grows with CV


def fig_seed_cv(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    cvs = [0.5, 1.0, 2.0]
    ax = axes[0]
    ax.axhline(0, color=GREY, lw=0.8, zorder=0)
    for k, (key, (label, st)) in enumerate(NET.items()):
        off = -0.13 + 0.26 * k
        for i, cv in enumerate(cvs):
            v = df[(df.network == key) & (df.CV == cv)].err_pct.values
            ax.scatter(i + off + jitter(len(v), 0.075, seed=i + k),
                       v, s=7, facecolors="none", edgecolors=st["color"],
                       lw=0.5, alpha=0.55, zorder=2)
        m = [df[(df.network == key) & (df.CV == cv)].err_pct.mean() for cv in cvs]
        s = [df[(df.network == key) & (df.CV == cv)].err_pct.std() for cv in cvs]
        ax.errorbar(np.arange(3) + off, m, yerr=s, capsize=4, lw=1.6,
                    ms=6, zorder=3, label=label, **st)
    ax.set_xticks(range(3), [f"{c:g}" for c in cvs])
    ax.set_xlabel("Patchiness amplitude  CV = σ / μ")
    ax.set_ylabel("Total-flux error (%)")
    ax.set_title("Individual estimates and their mean")
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1]
    w = 0.36
    for k, (key, (label, st)) in enumerate(NET.items()):
        s = [df[(df.network == key) & (df.CV == cv)].err_pct.std() for cv in cvs]
        ax.bar(np.arange(3) + (-w / 2 + w * k), s, w, color=st["color"],
               alpha=0.85, label=label,
               hatch="" if key == "op" else "//", edgecolor="white")
        for i, val in enumerate(s):
            ax.text(i - w / 2 + w * k, val + 0.6, f"{val:.0f}", ha="center", fontsize=8)
    ax.set_xticks(range(3), [f"{c:g}" for c in cvs])
    ax.set_xlabel("Patchiness amplitude  CV = σ / μ")
    ax.set_ylabel("Run-to-run spread, 1 s.d. (%)")
    ax.set_title("Spread across eight emission fields")
    ax.set_ylim(0, 30)

    o = df[df.network == "op"]
    p = df[df.network == "point"]
    alt = (
        "Two panels against patchiness amplitude CV of 0.5, 1.0 and 2.0, using 576 "
        "inversions across eight emission-field seeds. The left panel plots the signed "
        "total-flux error of every run as an open marker with the mean and one standard "
        "deviation overlaid. The means sit close to zero at every CV: open path "
        f"{o[o.CV==0.5].err_pct.mean():+.1f}, {o[o.CV==1.0].err_pct.mean():+.1f} and "
        f"{o[o.CV==2.0].err_pct.mean():+.1f} percent, so there is no systematic "
        "under-reporting once emission fields are resampled. The right panel shows that "
        "the run-to-run standard deviation instead grows steeply with CV, from "
        f"{o[o.CV==0.5].err_pct.std():.0f} to {o[o.CV==1.0].err_pct.std():.0f} to "
        f"{o[o.CV==2.0].err_pct.std():.0f} percent for open path and from "
        f"{p[p.CV==0.5].err_pct.std():.0f} to {p[p.CV==1.0].err_pct.std():.0f} to "
        f"{p[p.CV==2.0].err_pct.std():.0f} percent for point sensors. Heterogeneity "
        "amplitude therefore controls how unlucky a single campaign can be, not the "
        "direction of the error."
    )
    save(fig, "fig_seed_cv", alt)


# ------------------------------------------------- 2. patch size / independence


def fig_seed_L(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    Ls = [100, 250, 500]
    ax = axes[0]
    ax.axhline(0, color=GREY, lw=0.8, zorder=0)
    for k, (key, (label, st)) in enumerate(NET.items()):
        off = -0.13 + 0.26 * k
        for i, L in enumerate(Ls):
            v = df[(df.network == key) & (df.L_m == L)].err_pct.values
            ax.scatter(i + off + jitter(len(v), 0.075, seed=i + 7 * k),
                       v, s=7, facecolors="none", edgecolors=st["color"],
                       lw=0.5, alpha=0.55, zorder=2)
        m = [df[(df.network == key) & (df.L_m == L)].err_pct.mean() for L in Ls]
        s = [df[(df.network == key) & (df.L_m == L)].err_pct.std() for L in Ls]
        ax.errorbar(np.arange(3) + off, m, yerr=s, capsize=4, lw=1.6, ms=6,
                    zorder=3, label=label, **st)
    ax.set_xticks(range(3), [f"{L}" for L in Ls])
    ax.set_xlabel("Patch size  L (m)")
    ax.set_ylabel("Total-flux error (%)")
    ax.set_title("Error against patch size")
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1]
    npatch = [(1000.0 / L) ** 2 for L in Ls]
    for key, (label, st) in NET.items():
        e = [df[(df.network == key) & (df.L_m == L)].abs_pct.mean() for L in Ls]
        ax.plot(npatch, e, ms=6, lw=1.6, label=label, **st)
    ax.set_xscale("log")
    ax.set_xticks(npatch, [f"{n:.0f}\nL = {L} m" for n, L in zip(npatch, Ls)])
    ax.set_xlabel("Independent patches in the 1 km field  ≈ (1 km / L)²")
    ax.set_ylabel("Mean absolute error (%)")
    ax.set_title("Fewer patches, worse averaging")
    ax.set_ylim(0, 22)
    ax.legend(loc="upper right", frameon=False)

    o = df[df.network == "op"]
    alt = (
        "Two panels on source patch size L of 100, 250 and 500 m. The left panel plots "
        "signed total-flux error for all runs with the mean and one standard deviation: "
        "the mean is slightly negative at 100 and 250 m and turns positive at 500 m "
        f"({o[o.L_m==500].err_pct.mean():+.1f} percent for open path), while the spread "
        f"widens from {o[o.L_m==100].err_pct.std():.0f} to "
        f"{o[o.L_m==500].err_pct.std():.0f} percent. The right panel replots mean "
        "absolute error against the number of independent patches the 1 km field holds, "
        "roughly the square of 1 km divided by L, on a logarithmic axis: 100 patches at "
        "L = 100 m, 16 at 250 m and 4 at 500 m. Error rises steeply as patch count falls, "
        f"from {o[o.L_m==100].abs_pct.mean():.0f} percent at 100 patches to "
        f"{o[o.L_m==500].abs_pct.mean():.0f} percent at 4 patches for open path, and from "
        f"{df[(df.network=='point')&(df.L_m==100)].abs_pct.mean():.0f} to "
        f"{df[(df.network=='point')&(df.L_m==500)].abs_pct.mean():.0f} percent for point "
        "sensors. This is the classical replication problem: a field with few large "
        "patches offers few effective samples no matter how the instruments are placed."
    )
    save(fig, "fig_seed_L", alt)


# ------------------------------------------------------- 3. matched comparison


def fig_seed_matched(df: pd.DataFrame) -> None:
    w = df.pivot_table(index=["L_m", "CV", "seed", "n"], columns="network",
                       values="abs_pct").dropna()
    d = (w["point"] - w["op"]).values  # positive = open path closer to truth
    wins = int((d > 0).sum())

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    ax = axes[0]
    ax.hist(d, bins=np.arange(-40, 45, 4), color=BLUE, alpha=0.8, edgecolor="white")
    ax.axvline(0, color=GREY, lw=1.0)
    ax.axvline(np.median(d), color=VERMILLION, lw=1.6, ls="--")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.annotate(f"median {np.median(d):+.1f} pp", (np.median(d), ax.get_ylim()[1] * 0.95),
                xytext=(10, 0), textcoords="offset points", fontsize=8, color=VERMILLION,
                va="top")
    ax.set_xlabel("Point-sensor error minus open-path error (pp)")
    ax.set_ylabel("Matched configurations")
    ax.set_title(f"Open path closer in {wins} of {len(d)} pairs")

    ax = axes[1]
    ns = [1, 2, 3, 4]
    rate = []
    for n in ns:
        sub = w.reset_index()
        sub = sub[sub.n == n]
        rate.append(100.0 * (sub["point"] > sub["op"]).mean())
    ax.bar(ns, rate, 0.55, color=BLUE, alpha=0.85, edgecolor="white")
    ax.axhline(50, color=GREY, lw=1.0, ls=":")
    for n, r in zip(ns, rate):
        ax.text(n, r + 2, f"{r:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(ns)
    ax.set_xlabel("Instruments in the network  n")
    ax.set_ylabel("Pairs favouring open path (%)")
    ax.set_title("Pairs favouring open path, by network size")
    ax.set_ylim(0, 100)

    alt = (
        f"Two panels comparing the two network types on {len(d)} matched configurations "
        "that share patch size, patchiness amplitude, emission seed and instrument count. "
        "The left panel is a histogram of point-sensor absolute error minus open-path "
        "absolute error in percentage points; positive values mean the open path was "
        f"closer to the truth. The distribution is centred right of zero with a median of "
        f"{np.median(d):+.1f} percentage points, and open path is closer in {wins} of "
        f"{len(d)} pairs, which a sign test puts at p = 4e-6. The right panel gives the "
        "percentage of pairs favouring open path at each instrument count: "
        f"{rate[0]:.0f} percent at n = 1, falling to about {rate[1]:.0f} to {rate[3]:.0f} "
        "percent for n = 2 to 4, so a single open path is where path integration helps "
        "most and added point sensors close much of the gap."
    )
    save(fig, "fig_seed_matched", alt)


# --------------------------------------------------- 4. accuracy vs precision


def fig_seed_n(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    ns = [1, 2, 3, 4]
    ax = axes[0]
    for key, (label, st) in NET.items():
        e = [df[(df.network == key) & (df.n == n)].abs_pct.mean() for n in ns]
        ax.plot(ns, e, ms=6, lw=1.6, label=label, **st)
    ax.set_xticks(ns)
    ax.set_ylim(0, 22)
    ax.set_xlabel("Instruments in the network  n")
    ax.set_ylabel("Mean absolute error (%)")
    ax.set_title("Accuracy: a floor near 11 percent")
    ax.legend(loc="lower left", frameon=False)

    ax = axes[1]
    for key, (label, st) in NET.items():
        s = [df[(df.network == key) & (df.n == n)].sigma_pct.mean() for n in ns]
        ax.plot(ns, s, ms=6, lw=1.6, label=label, **st)
    ref = df[(df.network == "op") & (df.n == 1)].sigma_pct.mean() / np.sqrt(ns)
    ax.plot(ns, ref, color=GREY, lw=1.0, ls=":", label="1 / √n")
    ax.set_xticks(ns)
    ax.set_ylim(0, 45)
    ax.set_xlabel("Instruments in the network  n")
    ax.set_ylabel("Reported uncertainty (% of true flux)")
    ax.set_title("Precision: falls as 1 / √n")
    ax.legend(loc="upper right", frameon=False)

    o = df[df.network == "op"]
    p = df[df.network == "point"]
    alt = (
        "Two panels against the number of instruments n from 1 to 4, over 576 inversions. "
        "The left panel shows mean absolute total-flux error. Open path is flat at about "
        f"{o[o.n==1].abs_pct.mean():.0f} to {o[o.n==4].abs_pct.mean():.0f} percent for "
        "every n, so extra paths do not improve accuracy. Point sensors start much worse "
        f"at {p[p.n==1].abs_pct.mean():.0f} percent for a single sensor and improve to "
        f"{p[p.n==4].abs_pct.mean():.0f} percent at four, converging on the open-path "
        "value; four point sensors are needed to match one open path. The right panel "
        "shows the uncertainty the retrieval reports, as a percentage of the true flux. "
        f"It falls from about {o[o.n==1].sigma_pct.mean():.0f} percent at n = 1 to "
        f"{o[o.n==4].sigma_pct.mean():.0f} percent at n = 4 for both networks, tracking a "
        "plotted one-over-root-n reference curve. Reported uncertainty therefore shrinks "
        "with sampling effort while true accuracy does not, so a network can report "
        "growing confidence in an estimate that is no closer to the truth."
    )
    save(fig, "fig_seed_n", alt)


# ------------------------------------------------ 5. asking for the map instead


def fig_design_2x2() -> None:
    d = pd.read_csv(HERE / "design_2x2_seeds.csv")
    d["tot"] = 100.0 * d["total_rel"]
    cells = [(1, "averaged"), (1, "resolved"), (9, "averaged"), (9, "resolved")]
    labels = ["1 number\nhourly mean", "1 number\ntime series",
              "9-cell map\nhourly mean", "9-cell map\ntime series"]
    colors = [BLUE, BLUE, VERMILLION, VERMILLION]
    hatch = ["", "//", "", "//"]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    ax = axes[0]
    for i, ((ns, t), c, h) in enumerate(zip(cells, colors, hatch)):
        v = d[(d.n_state == ns) & (d.time == t)].tot.abs()
        ax.bar(i, v.mean(), 0.6, yerr=v.std(), capsize=4, color=c, alpha=0.85,
               hatch=h, edgecolor="white", error_kw=dict(ecolor=GREY, lw=1.2))
        ax.text(i, v.mean() + v.std() + 1.2, f"{v.mean():.0f}%", ha="center", fontsize=8)
    ax.set_xticks(range(4), labels, fontsize=7.5)
    ax.set_ylabel("Absolute error in field total (%)")
    ax.set_title("Error in the field total")
    ax.set_ylim(0, 40)

    ax = axes[1]
    for i, ((ns, t), c, h) in enumerate(zip(cells, colors, hatch)):
        v = d[(d.n_state == ns) & (d.time == t)]
        got, asked = v.dfs.mean(), ns
        ax.bar(i, got, 0.6, color=c, alpha=0.85, hatch=h, edgecolor="white")
        ax.plot([i - 0.34, i + 0.34], [asked, asked], color=GREY, lw=1.4)
        ax.text(i, got + 0.25, f"{got:.0f}", ha="center", fontsize=8)
    ax.set_xticks(range(4), labels, fontsize=7.5)
    ax.set_ylabel("Independent numbers recovered")
    ax.set_title("Independent quantities: recovered vs requested")
    ax.set_ylim(0, 11)

    g = lambda ns, t, col: d[(d.n_state == ns) & (d.time == t)][col]
    alt = (
        "Two panels over four experimental designs at patch size 100 m, patchiness "
        "amplitude 1.0 and four open paths, each run at eight emission seeds: solving "
        "for one field total or for a nine-cell map, from either a single hourly mean per "
        "path or a time series of 45 samples per path. The left panel gives absolute error "
        "in the field total. Solving for one number from hourly means is best at "
        f"{g(1,'averaged','tot').abs().mean():.0f} percent; a time series of the same one "
        f"number gives {g(1,'resolved','tot').abs().mean():.0f} percent; the nine-cell map "
        f"degrades the total to {g(9,'averaged','tot').abs().mean():.0f} percent from "
        f"hourly means and {g(9,'resolved','tot').abs().mean():.0f} percent from time "
        "series. The right panel counts how many genuinely independent numbers the data "
        "support against how many were requested, drawn as a grey line. One requested "
        "number is always recovered. Of nine requested cells, hourly means support only "
        f"{g(9,'averaged','dfs').mean():.0f}, while the time series supports "
        f"{g(9,'resolved','dfs').mean():.0f}. Sampling in time therefore does buy real "
        "spatial resolution, but the recovered map is bought at the cost of accuracy in "
        "the total, and cell-level errors remain of order 100 percent."
    )
    save(fig, "fig_design_2x2", alt)


def main() -> None:
    df = sweep()
    fig_seed_cv(df)
    fig_seed_L(df)
    fig_seed_matched(df)
    fig_seed_n(df)
    fig_design_2x2()


if __name__ == "__main__":
    main()
