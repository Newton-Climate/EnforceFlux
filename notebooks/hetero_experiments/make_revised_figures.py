#!/usr/bin/env python3
"""Publication-style figures for the revised representativeness-error deck.

All panels use the existing stored ensembles. Confidence intervals are
cluster bootstraps over the eight source-seed labels, preserving the repeated
L/CV/n structure within a resampled seed. No new simulation is run.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / "figures" / "revised"
OUT.mkdir(parents=True, exist_ok=True)
Q_TRUE = 0.027778
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREY = "#4D4D4D"
LIGHT = "#C9D3D7"

mpl.use("Agg")
plt.rcParams.update({
    "figure.constrained_layout.use": True,
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def data():
    d = pd.read_csv(HERE / "seed_sweep_results.csv")
    d["err_pct"] = 100 * d.q_rel_error
    d["abs_pct"] = d.err_pct.abs()
    d["sigma_pct"] = 100 * d.posterior_sigma_kg_s / Q_TRUE
    return d


def cluster_ci(frame, value, statistic=np.mean, reps=2000, seed=20260913):
    """Percentile CI after resampling seed labels as intact clusters."""
    rng = np.random.default_rng(seed)
    labels = np.sort(frame.seed.unique())
    vals = []
    for _ in range(reps):
        sampled = rng.choice(labels, size=len(labels), replace=True)
        pieces = [frame[frame.seed == s][value].to_numpy() for s in sampled]
        vals.append(statistic(np.concatenate(pieces)))
    return np.quantile(vals, [0.025, 0.975])


def save(fig, stem, alt):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight")
    (OUT / f"{stem}.txt").write_text(alt.strip() + "\n")
    plt.close(fig)


def jitter(n, center, seed, width=.10):
    return center + np.random.default_rng(seed).uniform(-width, width, n)


def cv_panel(d):
    p = d[d.network == "point"]
    cvs = [0.5, 1.0, 2.0]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.0))
    ax = axes[0]
    ax.axhline(0, color=GREY, lw=1)
    means, los, his = [], [], []
    for i, cv in enumerate(cvs):
        sub = p[p.CV == cv]
        ax.scatter(jitter(len(sub), i, 11+i), sub.err_pct, s=10, facecolor="none",
                   edgecolor=VERMILLION, alpha=.38, lw=.55)
        lo, hi = cluster_ci(sub, "err_pct")
        means.append(sub.err_pct.mean()); los.append(lo); his.append(hi)
    ax.errorbar(range(3), means, yerr=[np.array(means)-los, np.array(his)-means],
                color=VERMILLION, marker="s", ls="--", capsize=4, lw=1.8,
                label="Mean, 95% seed-cluster bootstrap CI")
    ax.set_xticks(range(3), [str(c) for c in cvs])
    ax.set_xlabel("Emission-field coefficient of variation, CV")
    ax.set_ylabel("Signed total-flux error (%)")
    ax.set_title("Bias remains small relative to realization spread")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    seed_sd = p.groupby(["CV", "seed"]).err_pct.std().reset_index()
    for i, cv in enumerate(cvs):
        vals = seed_sd[seed_sd.CV == cv].err_pct
        ax.scatter(jitter(len(vals), i, 31+i, .06), vals, s=24, facecolor="white",
                   edgecolor=VERMILLION, lw=1)
    pooled = [p[p.CV == cv].err_pct.std(ddof=1) for cv in cvs]
    ax.plot(range(3), pooled, color=VERMILLION, marker="s", ls="--", lw=1.8,
            label="Pooled sample SD")
    ax.set_xticks(range(3), [str(c) for c in cvs])
    ax.set_xlabel("Emission-field coefficient of variation, CV")
    ax.set_ylabel("Run-to-run spread, sample SD (%)")
    ax.set_title("Precision degrades as emission contrast grows")
    ax.set_ylim(0, 31)
    ax.legend(frameon=False, loc="upper left")
    alt = ("Two panels for 288 point-sensor inversions. The horizontal axis is source-field CV "
           "at 0.5, 1.0, and 2.0. Left: individual signed total-flux errors and mean with a "
           "95 percent seed-cluster bootstrap interval; means are +4.3, +2.3, and -0.5 percent. "
           "Right: realization spread rises from 9.9 to 16.8 to 26.4 percent. The scientific "
           "meaning is that greater emission contrast broadens campaign outcomes without a "
           "consistent error direction in these simulations.")
    save(fig, "fig_cv_verified", alt)


def L_panel(d):
    p = d[d.network == "point"]
    Ls = [100, 250, 500]
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    means=[]; los=[]; his=[]
    for i,L in enumerate(Ls):
        sub=p[p.L_m==L]
        ax.scatter(jitter(len(sub), i, 70+i, .12), sub.abs_pct, s=11,
                   facecolor="none", edgecolor=VERMILLION, alpha=.38, lw=.55)
        lo,hi=cluster_ci(sub,"abs_pct")
        means.append(sub.abs_pct.mean());los.append(lo);his.append(hi)
    ax.errorbar(range(3),means,yerr=[np.array(means)-los,np.array(his)-means],
                color=VERMILLION,marker="s",ls="--",capsize=4,lw=1.8,
                label="Mean absolute error, 95% seed-cluster bootstrap CI")
    ax.set_xticks(range(3),[str(v) for v in Ls])
    ax.set_xlabel("Source-field correlation length, L (m)")
    ax.set_ylabel("Absolute total-flux error (%)")
    ax.set_ylim(0,103)
    ax.set_title("Longer correlation scales reduce effective spatial averaging")
    ax.legend(frameon=False,loc="upper left")
    for i,m in enumerate(means): ax.text(i,m+5,f"{m:.1f}%",ha="center",color=VERMILLION)
    alt=("Absolute total-flux error for point-sensor inversions versus source-field correlation "
         "length L of 100, 250, and 500 metres. Pale points show all 96 inversions at each L; "
         "squares show mean absolute errors of 10.4, 11.8, and 19.0 percent with 95 percent "
         "seed-cluster bootstrap intervals. The worst observed point-sensor error is 97.5 "
         "percent. Under this geometry, spatial averaging becomes less effective as L grows.")
    save(fig,"fig_L_verified",alt)


def paired_panel(d):
    w=d.pivot(index=["L_m","CV","seed","n"],columns="network",values="abs_pct").dropna().reset_index()
    w["diff"]=w.point-w.op
    fig,axes=plt.subplots(1,2,figsize=(7.4,4.0))
    ax=axes[0]
    ax.scatter(w.point,w.op,c=w.n,cmap="viridis",s=17,alpha=.55,edgecolor="none")
    lim=max(w.point.max(),w.op.max())*1.03
    ax.plot([0,lim],[0,lim],color=GREY,ls=":",lw=1.2)
    ax.set_xlim(0,lim);ax.set_ylim(0,lim)
    ax.set_xlabel("Point-sensor absolute error (%)")
    ax.set_ylabel("Open-path absolute error (%)")
    ax.set_title("Each dot is a matched configuration")
    ax.text(.98,.05,"Below line = open path closer",transform=ax.transAxes,ha="right",color=BLUE)
    ax=axes[1]
    seedmeans=w.groupby("seed")["diff"].mean()
    ax.axhline(0,color=GREY,lw=1)
    ax.scatter(range(8),seedmeans,color=BLUE,s=36,zorder=3)
    ax.vlines(range(8),0,seedmeans,color=BLUE,alpha=.65,lw=2)
    ax.set_xticks(range(8),[str(i) for i in range(8)])
    ax.set_xlabel("Source-seed label")
    ax.set_ylabel("Mean paired improvement (percentage points)")
    ax.set_title("Seven of eight seed clusters favor open path")
    alt=("Left: paired scatter of open-path absolute error against point-sensor absolute error "
         "for 288 configurations matched on L, CV, source seed, and instrument count. Of these, "
         "182 lie below the equality line; the median point-minus-path error is 3.56 percentage "
         "points. Right: the paired improvement averaged within each of eight source-seed "
         "clusters; seven are positive. A one-sided sign test on seed clusters gives p equals "
         "0.035, whereas treating all repeated configurations as independent gives 4.42e-6.")
    save(fig,"fig_geometry_paired_verified",alt)


def n_panel(d):
    ns=[1,2,3,4]
    fig,axes=plt.subplots(1,2,figsize=(7.4,4.0))
    for key,label,color,marker in [("point","Point sensors",VERMILLION,"s"),("op","Open path",BLUE,"o")]:
        z=d[d.network==key]
        means=[];los=[];his=[]
        for i,n in enumerate(ns):
            sub=z[z.n==n];lo,hi=cluster_ci(sub,"abs_pct",seed=90+i)
            means.append(sub.abs_pct.mean());los.append(lo);his.append(hi)
            axes[0].scatter(jitter(len(sub),n,1000+i+(0 if key=='point' else 20),.055),sub.abs_pct,
                            s=8,facecolor="none",edgecolor=color,alpha=.25,lw=.45)
        axes[0].errorbar(ns,means,yerr=[np.array(means)-los,np.array(his)-means],color=color,
                         marker=marker,ls="--" if key=="point" else "-",capsize=3,lw=1.8,label=label)
        sig=[z[z.n==n].sigma_pct.mean() for n in ns]
        axes[1].plot(ns,sig,color=color,marker=marker,ls="--" if key=="point" else "-",lw=1.8,label=label)
    axes[0].set_xlabel("Instruments in network, n");axes[0].set_ylabel("Absolute total-flux error (%)")
    axes[0].set_xticks(ns);axes[0].set_ylim(0,103);axes[0].set_title("Empirical error reaches a geometry-dependent floor")
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("Instruments in network, n");axes[1].set_ylabel("Reported posterior sigma (% of true flux)")
    axes[1].set_xticks(ns);axes[1].set_ylim(0,43);axes[1].set_title("Reported inversion uncertainty keeps falling")
    axes[1].legend(frameon=False)
    alt=("Two panels versus instrument count n from one to four. Left: all individual absolute "
         "errors are faint points and means have 95 percent seed-cluster bootstrap intervals. "
         "Point-sensor mean absolute error falls from 18.8 to 12.9, 12.1, and 11.1 percent; "
         "open-path error is nearly flat at 11.4, 11.4, 10.7, and 10.9 percent. Right: reported "
         "posterior sigma falls from about 35 to 21 percent for point sensors and 39 to 21 percent "
         "for open paths. Under the tested fence geometry, additional instruments reduce the "
         "reported inversion uncertainty faster than the observed error.")
    save(fig,"fig_n_verified",alt)


def map_panel():
    x=pd.read_csv(HERE/"design_2x2_seeds.csv")
    x["abs_total_pct"]=100*x.total_rel.abs()
    cells=[(1,"averaged"),(1,"resolved"),(9,"averaged"),(9,"resolved")]
    labels=["1 total\nmean","1 total\ntime","9 cells\nmean","9 cells\ntime"]
    fig,axes=plt.subplots(1,2,figsize=(7.4,4.0))
    for i,(state,time) in enumerate(cells):
        sub=x[(x.n_state==state)&(x.time==time)]
        color=BLUE if state==1 else VERMILLION
        axes[0].scatter(jitter(len(sub),i,200+i,.08),sub.abs_total_pct,s=26,facecolor="white",edgecolor=color,lw=1)
        axes[0].plot(i,sub.abs_total_pct.mean(),marker="_",ms=18,mew=2.2,color=color)
        axes[1].scatter(jitter(len(sub),i,300+i,.08),sub.dfs,s=26,facecolor="white",edgecolor=color,lw=1)
        axes[1].plot(i,sub.dfs.mean(),marker="_",ms=18,mew=2.2,color=color)
        axes[1].plot([i-.18,i+.18],[state,state],color=GREY,lw=1.2,ls=":")
    for ax in axes: ax.set_xticks(range(4),labels,fontsize=8.5)
    axes[0].set_ylabel("Absolute error in field total (%)");axes[0].set_ylim(0,42)
    axes[0].set_title("Spatial flexibility can degrade the recovered total")
    axes[1].set_ylabel("Degrees of freedom for signal");axes[1].set_ylim(0,10.5)
    axes[1].set_title("Time variation adds spatial information")
    alt=("Four designs compare retrieving one field total or a nine-cell spatial map from hourly "
         "means or time-resolved observations, with eight source realizations each. Individual "
         "runs are open circles and horizontal ticks are means. Mean absolute total-flux errors "
         "are 6.2, 7.4, 9.8, and 23.3 percent. Degrees of freedom for signal are 1, 1, 4, and 9. "
         "Time-resolved data support the nine requested spatial quantities, but the more flexible "
         "retrieval has worse total-flux accuracy in this experiment.")
    save(fig,"fig_map_total_verified",alt)


def main():
    d=data();cv_panel(d);L_panel(d);paired_panel(d);n_panel(d);map_panel()
    print(f"wrote revised figures to {OUT}")


if __name__=="__main__":
    main()
