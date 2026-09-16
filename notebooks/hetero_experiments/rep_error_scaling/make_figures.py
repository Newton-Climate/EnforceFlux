#!/usr/bin/env python3
"""Publication figures for the representativeness-error scaling analysis.

Run after ``scaling_analysis.py``. Style follows ``make_seed_figures.py``:
Okabe-Ito, colour never load-bearing on its own, and a plain-text description
written beside every PNG so the figure can be read without seeing it.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from scaling_core import MODELS, SQRT_A, fit_model, loco_cv  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

# Okabe-Ito
BLUE, VERM, GREEN, ORANGE, GREY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#4D4D4D"
CV_COLOR = {0.5: BLUE, 1.0: ORANGE, 2.0: VERM}
L_MARKER = {100.0: "o", 250.0: "s", 500.0: "^"}
GEOM_STYLE = {
    "point": dict(color=VERM, marker="s", ls="--", label="Point sensors"),
    "open_path": dict(color=BLUE, marker="o", ls="-", label="Open path"),
}

plt.rcParams.update({
    "figure.constrained_layout.use": True,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 7.5, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, stem: str, alt: str) -> None:
    fig.savefig(OUT / f"{stem}.png")
    fig.savefig(OUT / f"{stem}.pdf")
    (OUT / f"{stem}.txt").write_text(alt.strip() + "\n")
    plt.close(fig)
    print(f"wrote {stem}.png")


def cond_legend(fig, y: float = -0.06) -> None:
    """CV as colour and L as marker, both redundant, below the panels."""
    h = [Line2D([], [], color=CV_COLOR[c], marker="o", ls="none", ms=5,
                label=f"CV = {c:g}") for c in (0.5, 1.0, 2.0)]
    h += [Line2D([], [], color=GREY, marker=L_MARKER[L], ls="none", ms=5,
                 mfc="none", label=f"L = {L:g} m") for L in (100.0, 250.0, 500.0)]
    fig.legend(handles=h, frameon=False, ncol=6, loc="lower center",
               bbox_to_anchor=(0.5, y))


# ------------------------------------------------------------------ fig 1


def _log_edges(v: np.ndarray) -> np.ndarray:
    lv = np.log(v)
    mid = np.exp((lv[:-1] + lv[1:]) / 2)
    return np.concatenate([[v[0] ** 2 / mid[0]], mid, [v[-1] ** 2 / mid[-1]]])


def _lin_edges(v: np.ndarray) -> np.ndarray:
    mid = (v[:-1] + v[1:]) / 2
    return np.concatenate([[2 * v[0] - mid[0]], mid, [2 * v[-1] - mid[-1]]])


def fig1(inv: pd.DataFrame, cond: pd.DataFrame) -> None:
    """CV-L response before any dimensional collapse, on the fine mesh.

    ``inv`` and ``cond`` are the mesh tables from ``phase_mesh.py``. The top
    strip keeps the original 3 x 3 conditions; the phase diagrams below use
    every mesh point, with the original nine in bold.
    """
    sub = inv[(inv.geometry == "point") & (inv["n"] == 1)]
    c = cond[(cond.geometry == "point") & (cond["n"] == 1)]
    cvs, ls = [0.5, 1.0, 2.0], [100.0, 250.0, 500.0]
    mcv = np.array(sorted(c.CV.unique()))
    mL = np.array(sorted(c.L_m.unique()))
    m_real = int(c.m_realizations.min())
    chol = c[c.sampler == "cholesky"].L_m
    L_switch = (float(mL[mL < chol.min()].max()) * float(chol.min())) ** 0.5 \
        if len(chol) and (mL < chol.min()).any() else None

    fig = plt.figure(figsize=(9.8, 6.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.15])
    ax = fig.add_subplot(gs[0, :])
    ax.axhline(0, color=GREY, lw=0.8, zorder=0)
    rng = np.random.default_rng(0)
    pos, labels = [], []
    for i, cv in enumerate(cvs):
        for j, L in enumerate(ls):
            x = i * 3.5 + j
            g = sub[(sub.CV == cv) & (sub.L_m == L)]
            e = 100 * g.e_signed.to_numpy(float)
            ax.scatter(x + rng.uniform(-0.16, 0.16, len(e)), e, s=16,
                       facecolors="none", edgecolors=CV_COLOR[cv], lw=0.8,
                       marker=L_MARKER[L], zorder=2)
            r = c[(c.CV == cv) & (c.L_m == L)].iloc[0]
            ax.errorbar(x, 100 * r.bias, yerr=100 * r.sigma_rep, color=CV_COLOR[cv],
                        marker="_", ms=14, capsize=5, lw=1.8, zorder=3)
            pos.append(x)
            labels.append(f"{L:g}")
    ax.set_xticks(pos, labels)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.16 * (hi - lo))
    for i, cv in enumerate(cvs):
        ax.text(i * 3.5 + 1, ax.get_ylim()[1] * 0.98, f"CV = {cv:g}",
                ha="center", va="top", fontsize=8.5, color=CV_COLOR[cv])
    ax.set_xlabel("Source correlation length L (m), grouped by emission contrast CV")
    ax.set_ylabel("Total-flux error (%)")
    ax.set_title("Every realization, point sensors, n = 1, original nine conditions "
                 f"({m_real} source fields each; bar = mean ± 1 s.d.)")

    xe, ye = _log_edges(mL), _lin_edges(mcv)
    sd_max = 100 * max(c.sigma_rep.max(), c.sd_pred_loglinear.max())
    panels = [
        ("sigma_rep", "Spread σ_rep, total error (%)", "viridis", (0.0, sd_max)),
        ("sd_pred_loglinear", "Closed form √(δwᵀΣδw),\nperfect transport (%)",
         "viridis", (0.0, sd_max)),
        ("bias", "Mean bias (%)", "RdBu_r", None),
        ("RMSE", "RMSE (%)", "viridis", None),
    ]
    for k, (col, title, cmap, lim) in enumerate(panels):
        a = fig.add_subplot(gs[1, k])
        M = np.array([[100 * c[(c.CV == cv) & (c.L_m == L)][col].iloc[0]
                       for L in mL] for cv in mcv])
        if lim is None:
            v = float(np.abs(M).max())
            lim = (-v if col == "bias" else 0.0, v)
        im = a.pcolormesh(xe, ye, M, cmap=cmap, vmin=lim[0], vmax=lim[1],
                          shading="flat", rasterized=True)
        for i, cv in enumerate(mcv):
            for j, L in enumerate(mL):
                f = (M[i, j] - lim[0]) / (lim[1] - lim[0])
                dark = f < 0.5 if cmap == "viridis" else abs(f - 0.5) > 0.3
                a.text(L, cv, f"{M[i, j]:.0f}", ha="center", va="center",
                       fontsize=5.5, color="white" if dark else "black",
                       fontweight="bold" if (cv in cvs and L in ls) else "normal")
        if L_switch is not None:
            a.axvline(L_switch, color="black", ls=":", lw=1.0)
        a.set_xscale("log")
        a.set_xlim(xe[0], xe[-1])
        a.set_xticks(mL, [f"{L:g}" for L in mL], rotation=90, fontsize=7)
        a.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        a.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        a.set_yticks(mcv, [f"{cv:g}" for cv in mcv], fontsize=7)
        a.set_xlabel("L (m)")
        if k == 0:
            a.set_ylabel("CV")
        else:
            a.set_yticklabels([])
        a.set_title(title, fontsize=8)
        fig.colorbar(im, ax=a, fraction=0.05, pad=0.02)

    sd = c.set_index(["CV", "L_m"])
    ratio = c.sigma_rep / c.sd_pred_loglinear
    ratio_pure = c.sigma_rep_pure / c.sd_pred_loglinear

    def row(col: str, cv: float) -> str:
        return ", ".join(f"{100 * sd.loc[(cv, L), col]:.0f}" for L in mL)

    def column(col: str, L: float) -> str:
        return ", ".join(f"{100 * sd.loc[(cv, L), col]:.0f}" for cv in mcv)

    L_mid = float(mL[len(mL) // 2])
    switch = (f"A dotted line at L = {L_switch:.0f} m marks where the source-field "
              "generator switches from circulant embedding to padded Cholesky, so "
              "seed k is a different field on either side of it; common random "
              "numbers hold only within each side." if L_switch else "")
    alt = f"""
Figure 1. CV-L response surface for point sensors with n = 1, before any
dimensional collapse, on a {len(mcv)} x {len(mL)} mesh (CV {mcv[0]:g} to
{mcv[-1]:g}, L {mL[0]:g} to {mL[-1]:g} m) with {m_real} source-field seeds per
mesh point. Top panel: every inversion for the original nine conditions
(CV 0.5, 1, 2 by L 100, 250, 500 m) as signed total-flux error in percent,
grouped by CV then by L, with the condition mean and one standard deviation.
Colour encodes CV and marker shape encodes L; both repeat the axis grouping.
Bottom row: four phase diagrams, CV on the vertical axis and L on a log
horizontal axis, each cell labelled with its value in percent, the original
nine in bold. Panels: run-to-run spread of the total error; the closed-form
perfect-transport spread sqrt(AK^2 dw^T Sigma dw) with Sigma = ln(1 + CV^2)
exp(-r/L), on the same colour scale as the spread; mean bias; RMSE. {switch}

Spread of the total error by L, left to right, at CV = {mcv[0]:g}: {row('sigma_rep', mcv[0])} percent;
at CV = {mcv[-1]:g}: {row('sigma_rep', mcv[-1])} percent.
Spread by CV, bottom to top, at L = {L_mid:g} m: {column('sigma_rep', L_mid)} percent.
Closed form by L at CV = {mcv[-1]:g}: {row('sd_pred_loglinear', mcv[-1])} percent.
Across the {len(c)} mesh points the empirical spread is {ratio.median():.2f} times the
closed form for total error (range {ratio.min():.2f} to {ratio.max():.2f}) and
{ratio_pure.median():.2f} times for perfect-transport error (range
{ratio_pure.min():.2f} to {ratio_pure.max():.2f}).
Bias by L at CV = {mcv[-1]:g}: {row('bias', mcv[-1])} percent.

CSV of the phase-diagram values (percent):
CV,L_m,sampler,bias,sigma_rep_total,sigma_rep_perfect_transport,sigma_closed_form,MAE,RMSE
""" + "\n".join(
        f"{r.CV:g},{r.L_m:g},{r.sampler},{100*r.bias:.2f},{100*r.sigma_rep:.2f},"
        f"{100*r.sigma_rep_pure:.2f},{100*r.sd_pred_loglinear:.2f},"
        f"{100*r.MAE:.2f},{100*r.RMSE:.2f}"
        for r in c.sort_values(["CV", "L_m"]).itertuples())
    save(fig, "fig1_response_surface", alt)


# ------------------------------------------------------------------ fig 2


def fig2(cond: pd.DataFrame) -> None:
    """Does H = CV/sqrt(N_eff) collapse the nine conditions?"""
    c = cond[(cond.geometry == "point") & (cond["n"] == 1)].sort_values(
        ["CV", "L_m"])
    get = lambda k: fit_model(next(m for m in MODELS if m.name == k), c)
    f0, f2, f3, f4 = get("M0"), get("M2"), get("M3"), get("M4")

    # Points at a shared x are nudged apart along x so all nine stay visible;
    # the nudge is cosmetic and keyed to the variable *not* on that axis.
    NUDGE = {0.5: 0.94, 1.0: 1.0, 2.0: 1.06}
    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.9))
    panels = [
        ("CV", "Emission contrast  CV", "L_m", None),
        ("l_star", "Correlation length  $L/\\sqrt{A}$", "CV", None),
        ("H_simple", "$H = CV\\,L/\\sqrt{A}$", None, f2),
        ("_m4", "Fitted  $a\\,\\sigma_{log}^{\\alpha}(L/\\sqrt{A})^{\\beta}$ (%)",
         None, None),
    ]
    for ax, (col, xlabel, nudge_by, fit) in zip(axes, panels):
        x = f4["pred"] * 100 if col == "_m4" else c[col].to_numpy(float)
        for k, r in enumerate(c.itertuples()):
            xv = x[k]
            if nudge_by == "L_m":
                xv *= {100.0: 0.94, 250.0: 1.0, 500.0: 1.06}[r.L_m]
            elif nudge_by == "CV":
                xv *= NUDGE[r.CV]
            ax.errorbar(xv, 100 * r.sigma_rep,
                        yerr=[[100 * (r.sigma_rep - r.sigma_rep_lo)],
                              [100 * (r.sigma_rep_hi - r.sigma_rep)]],
                        color=CV_COLOR[r.CV], marker=L_MARKER[r.L_m], ms=6,
                        capsize=2.5, lw=1.0, mfc="none", mew=1.4, ls="none")
        if fit is not None:
            xx = np.array([c[col].min() * 0.8, c[col].max() * 1.25])
            ax.plot(xx, 100 * fit["a"] * xx, color=GREY, ls=":", lw=1.5,
                    label=f"proposed, slope 1\n$R^2$ = {fit['r2_log']:.2f}")
            ax.legend(frameon=False, loc="lower right", fontsize=7)
        if col == "_m4":
            lim = np.array([x.min() * 0.75, x.max() * 1.3])
            ax.plot(lim, lim, color=GREY, ls="--", lw=1.0,
                    label=f"1:1\n$R^2$ = {f4['r2_log']:.2f}")
            ax.legend(frameon=False, loc="lower right", fontsize=7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25, which="major", lw=0.4)
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    axes[0].set_xticks([0.5, 1.0, 2.0])
    axes[1].set_xticks([0.1, 0.25, 0.5])
    axes[2].set_xticks([0.05, 0.1, 0.25, 0.5, 1.0])
    axes[0].set_ylabel("Run-to-run spread  $\\sigma_{rep}$ (%)")
    for ax, t in zip(axes, ["A. against CV alone", "B. against L alone",
                            "C. against the proposed H",
                            "D. against the fitted power law"]):
        ax.set_title(t, fontsize=8.5)
    h = [Line2D([], [], color=CV_COLOR[cv], marker="o", ls="none", ms=5,
                label=f"CV = {cv:g}") for cv in (0.5, 1.0, 2.0)]
    h += [Line2D([], [], color=GREY, marker=L_MARKER[L], ls="none", ms=5,
                 mfc="none", label=f"L = {L:g} m") for L in (100.0, 250.0, 500.0)]
    fig.legend(handles=h, frameon=False, ncol=6, loc="lower center",
               bbox_to_anchor=(0.5, -0.09))
    fig.suptitle("Point sensors, n = 1. Bars are 95% bootstrap intervals over "
                 "8 source realizations.", fontsize=8.5)

    alt = f"""
Figure 2. Scaling collapse test, point sensors with n = 1, all axes logarithmic.
Run-to-run spread sigma_rep in percent is plotted against four candidate
predictors: CV alone (A), L/sqrt(A) alone (B), the proposed
H = CV/sqrt(N_eff) = CV*L/sqrt(A) (C), and the fitted free-exponent power law
a * sigma_log^alpha * (L/sqrt(A))^beta (D). All nine CV-L conditions appear in
every panel; colour encodes CV and marker shape encodes L, so a failure of the
collapse shows as colour or shape structure rather than a single curve. Points
sharing an x value are nudged sideways by a few percent so none is hidden; the
nudge is cosmetic. Error bars are 95 percent percentile bootstrap intervals over
the 8 source realizations and are wide -- typically a factor of two -- because
each condition rests on only 8 fields.

The proposed H does not collapse the conditions. Against H the nine points still
separate by CV, and a slope-1 line through them explains R^2 = {f2['r2_log']:.2f} of the
variance in log sigma_rep, which is worse than CV alone (R^2 = {f0['r2_log']:.2f}). Freeing the
exponents gives alpha = {f3['exponents']['CV'][0]:.2f} +/- {f3['exponents']['CV'][1]:.2f} on CV and beta = {f3['exponents']['l_star'][0]:.2f} +/- {f3['exponents']['l_star'][1]:.2f} on
L/sqrt(A), R^2 = {f3['r2_log']:.2f}. Replacing CV by the log-field amplitude
sigma_log = sqrt(ln(1+CV^2)) gives alpha = {f4['exponents']['sigma_log'][0]:.2f} +/- {f4['exponents']['sigma_log'][1]:.2f}, consistent with 1, and the
nine points fall onto the one-to-one line in panel D with R^2 = {f4['r2_log']:.2f}.

CSV of the plotted points:
CV,L_m,H_simple,sigma_rep_pct,boot_lo_pct,boot_hi_pct,M4_pred_pct
""" + "\n".join(
        f"{r.CV:g},{r.L_m:g},{r.H_simple:.4f},{100*r.sigma_rep:.2f},"
        f"{100*r.sigma_rep_lo:.2f},{100*r.sigma_rep_hi:.2f},{100*p_:.2f}"
        for r, p_ in zip(c.itertuples(), f4["pred"]))
    save(fig, "fig2_collapse", alt)


# ------------------------------------------------------------------ fig 3


def fig3(cond: pd.DataFrame) -> None:
    """Observed versus predicted for each candidate model."""
    c = cond[(cond.geometry == "point") & (cond["n"] == 1)].sort_values(
        ["CV", "L_m"])
    show = ["M0", "M1", "M2", "M3", "M4", "M5"]
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.0), sharex=True, sharey=True)
    lines = []
    for ax, name in zip(axes.ravel(), show):
        m = next(x for x in MODELS if x.name == name)
        f = fit_model(m, c)
        lo = min(f["obs"].min(), f["pred"].min()) * 0.7
        hi = max(f["obs"].max(), f["pred"].max()) * 1.4
        ax.plot([100 * lo, 100 * hi], [100 * lo, 100 * hi], color=GREY,
                lw=0.9, ls="--", zorder=0)
        for r, p, o in zip(c.itertuples(), f["pred"], f["obs"]):
            ax.plot(100 * p, 100 * o, color=CV_COLOR[r.CV],
                    marker=L_MARKER[r.L_m], ms=6, mfc="none", mew=1.4, ls="none")
        ax.set_xscale("log")
        ax.set_yscale("log")
        cvr = loco_cv(m, c)
        ax.set_title(f"{name}: {m.label}", fontsize=8.5)
        ax.text(0.04, 0.95, f"$R^2$ = {f['r2_log']:.2f}\nLOCO = {cvr:.2f}\n"
                            f"BIC = {f['bic']:.1f}",
                transform=ax.transAxes, va="top", fontsize=7.5)
        ax.grid(alpha=0.25, which="both", lw=0.4)
        lines.append((name, m.label, f, cvr))
    for ax in axes[1]:
        ax.set_xlabel("Predicted $\\sigma_{rep}$ (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed $\\sigma_{rep}$ (%)")
    cond_legend(fig, -0.045)
    fig.suptitle("Observed versus predicted, point sensors n = 1. LOCO is the "
                 "leave-one-CV-L-condition-out RMSE in log space (lower is "
                 "better).", fontsize=8.5)

    alt = """
Figure 3. Observed versus predicted run-to-run spread for six candidate scaling
models, point sensors with n = 1, both axes logarithmic and percent. Each panel
shows the nine CV-L conditions against a one-to-one dashed line; colour encodes
CV and marker shape encodes L. Annotated in each panel are the in-sample R^2 in
log space, the leave-one-condition-out RMSE in log space (LOCO, the honest
generalization number given only nine design points), and BIC.

Model,form,R2_log,LOCO_rmse_log,BIC,a,alpha,beta
""" + "\n".join(
        f"{n},{lab},{f['r2_log']:.3f},{cvr:.3f},{f['bic']:.2f},{f['a']:.3f},"
        + (f"{f['exponents'][list(f['exponents'])[0]][0]:.3f}"
           if f["exponents"] else "fixed")
        + ","
        + (f"{f['exponents'][list(f['exponents'])[1]][0]:.3f}"
           if len(f["exponents"]) > 1 else "fixed")
        for n, lab, f, cvr in lines)
    save(fig, "fig3_model_comparison", alt)


# ------------------------------------------------------------------ fig 4


def fig4(cond: pd.DataFrame, nd: pd.DataFrame) -> None:
    """Does sensor number shift the scaling?"""
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))
    shades = {1: 0.25, 2: 0.45, 3: 0.7, 4: 1.0}
    for ax, geom in zip(axes[:2], ["point", "open_path"]):
        base = GEOM_STYLE[geom]["color"]
        for n in (1, 2, 3, 4):
            d = cond[(cond.geometry == geom) & (cond["n"] == n)].sort_values(
                "H_simple")
            ax.plot(d.H_simple, 100 * d.sigma_rep, marker="os^D"[n - 1], ms=4.5,
                    lw=1.1, alpha=shades[n], color=base, mfc="none",
                    label=f"n = {n}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$H = CV\\,L/\\sqrt{A}$")
        ax.set_title(f"{'AB'[geom == 'open_path']}. "
                     f"{GEOM_STYLE[geom]['label']}", fontsize=8.5)
        ax.legend(frameon=False, fontsize=7)
        ax.grid(alpha=0.25, which="both", lw=0.4)
    axes[0].set_ylabel("$\\sigma_{rep}$ (%)")

    ax = axes[2]
    for geom in ("point", "open_path"):
        st = GEOM_STYLE[geom]
        d = cond[cond.geometry == geom].groupby("n").sigma_rep.mean()
        ax.plot(d.index, 100 * d.values, color=st["color"], marker=st["marker"],
                ls=st["ls"], ms=5, label=st["label"])
    ref = 100 * cond[cond.geometry == "point"].groupby("n").sigma_rep.mean().iloc[0]
    nn = np.array([1, 2, 3, 4], float)
    ax.plot(nn, ref / np.sqrt(nn), color=GREY, ls=":", lw=1.4,
            label="$1/\\sqrt{n}$ reference")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Number of instruments  n")
    ax.set_ylabel("$\\sigma_{rep}$ (%), mean over CV and L")
    ax.set_title("C. Replication gain", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.25, lw=0.4)

    pt = nd[nd.geometry == "point"].iloc[0]
    op = nd[nd.geometry == "open_path"].iloc[0]
    alt = f"""
Figure 4. Sensor-number dependence. Panels A and B plot sigma_rep in percent
against H = CV*L/sqrt(A) on log-log axes for n = 1 to 4, separately for point
sensors and open paths; increasing n is drawn with increasing colour intensity
and a different marker. The nine conditions collapse onto only six distinct
values of H (CV*L is degenerate for 0.5x500, 1.0x250 and for 1.0x500, 2.0x250),
which is why the curves zig-zag rather than rise monotonically -- itself a
visible symptom of the collapse failing. Panel C shows sigma_rep averaged over
CV and L against n, with a 1/sqrt(n) reference curve anchored at the n = 1
point-sensor value.

Adding log n to the scaling gives an exponent of {pt.gamma_n:+.3f} (95% CI {pt.gamma_lo:+.3f} to
{pt.gamma_hi:+.3f}) for point sensors, consistent with the 1/sqrt(n) value of -0.5, and
{op.gamma_n:+.3f} (95% CI {op.gamma_lo:+.3f} to {op.gamma_hi:+.3f}) for open paths, consistent with zero. The
open-path curves for n = 1 to 4 lie almost on top of one another. This is a
consequence of the experimental design: total open-path length is held at
1000 m, so n = 1 is one 1000 m path and n = 4 is four 250 m paths covering the
same ground. Subdividing a fixed path buys nothing; adding independent point
sensors buys 1/sqrt(n).

geometry,gamma_n,gamma_lo,gamma_hi,alpha_CV,beta_L
""" + "\n".join(
        f"{r.geometry},{r.gamma_n:.3f},{r.gamma_lo:.3f},{r.gamma_hi:.3f},"
        f"{r.alpha_CV:.3f},{r.beta_L:.3f}" for r in nd.itertuples())
    save(fig, "fig4_sensor_number", alt)


# ------------------------------------------------------------------ fig 5


def fig5(cond: pd.DataFrame, fits: pd.DataFrame) -> None:
    """Point versus open path: coefficient or functional form?"""
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))
    ax = axes[0]
    for geom in ("point", "open_path"):
        st = GEOM_STYLE[geom]
        d = cond[(cond.geometry == geom) & (cond["n"] == 1)].sort_values("H_simple")
        ax.errorbar(d.H_simple, 100 * d.sigma_rep,
                    yerr=[100 * (d.sigma_rep - d.sigma_rep_lo),
                          100 * (d.sigma_rep_hi - d.sigma_rep)],
                    color=st["color"], marker=st["marker"], ls=st["ls"],
                    ms=5, capsize=2.5, lw=1.2, mfc="none", label=st["label"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$H = CV\\,L/\\sqrt{A}$")
    ax.set_ylabel("$\\sigma_{rep}$ (%)")
    ax.set_title("A. n = 1", fontsize=8.5)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, which="both", lw=0.4)

    m3 = fits[fits.model == "M3"]
    for ax, (col, se, lab, ref) in zip(axes[1:], [
        ("alpha", "alpha_se", "CV exponent  α", 1.0),
        ("beta", "beta_se", "$L/\\sqrt{A}$ exponent  β", 1.0),
    ]):
        for k, geom in enumerate(("point", "open_path")):
            st = GEOM_STYLE[geom]
            d = m3[m3.geometry == geom].sort_values("n")
            ax.errorbar(d["n"] + (k - 0.5) * 0.12, d[col], yerr=1.96 * d[se],
                        color=st["color"], marker=st["marker"], ls=st["ls"],
                        ms=5, capsize=3, lw=1.2, mfc="none", label=st["label"])
        ax.axhline(ref, color=GREY, ls=":", lw=1.4,
                   label="first-order prediction")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("Number of instruments  n")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25, lw=0.4)
    axes[1].set_title("B. amplitude exponent", fontsize=8.5)
    axes[2].set_title("C. length exponent", fontsize=8.5)
    axes[2].set_ylim(top=1.12)
    axes[2].legend(frameon=False, fontsize=7, loc="lower right")

    alt = """
Figure 5. Point versus open-path geometry. Panel A plots sigma_rep against
H = CV*L/sqrt(A) on log-log axes at n = 1 for both geometries, with 95 percent
bootstrap intervals. Panels B and C show the fitted free exponents of
sigma_rep = a * CV^alpha * (L/sqrt(A))^beta for each geometry at each n, with 95
percent confidence intervals and a dotted line at the first-order prediction of
1.

Open-path sampling lowers the coefficient without changing the shape: at every
n the two geometries have overlapping alpha and beta intervals, and the length
exponent sits near 0.4 to 0.55 for both, nowhere near 1. The advantage of the
open path is largest at n = 1, where a single 1000 m path is compared against a
single point, and it shrinks as the point network grows.

geometry,n,a,alpha,alpha_se,beta,beta_se
""" + "\n".join(
        f"{r.geometry},{r.n},{r.a:.3f},{r.alpha:.3f},{r.alpha_se:.3f},"
        f"{r.beta:.3f},{r.beta_se:.3f}"
        for r in m3.sort_values(["geometry", "n"]).itertuples())
    save(fig, "fig5_geometry", alt)


# ------------------------------------------------------------------ fig 6


def fig6(cond: pd.DataFrame, cal: pd.DataFrame) -> None:
    """Is the reported uncertainty calibrated, and does H predict where it fails?"""
    d = cond.merge(cal, on=["geometry", "n", "CV", "L_m"])
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9))

    ax = axes[0]
    for geom in ("point", "open_path"):
        st = GEOM_STYLE[geom]
        g = d[d.geometry == geom]
        ax.plot(g.H_simple, g.R_mean, ls="none", marker=st["marker"], ms=4.5,
                mfc="none", color=st["color"], label=st["label"])
    x = np.log(d.H_simple)
    b = np.polyfit(x, d.R_mean, 1)
    xx = np.linspace(x.min(), x.max(), 40)
    ax.plot(np.exp(xx), np.polyval(b, xx), color=GREY, lw=1.4, ls="--")
    r = np.corrcoef(x, d.R_mean)[0, 1]
    ax.text(0.96, 0.06, f"$r$ = {r:.2f} vs log $H$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("$H = CV\\,L/\\sqrt{A}$")
    ax.set_ylabel("$R = |error| / \\sigma_{reported}$")
    ax.set_title("A. error against reported σ", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.25, lw=0.4)

    ax = axes[1]
    # H takes six distinct values across the nine CV-L conditions, so coverage
    # is aggregated on those exactly rather than by an arbitrary window.
    from statsmodels.stats.proportion import proportion_confint

    d = d.sort_values("H_simple")
    hv = np.sort(d.H_simple.unique())
    for k, (col, lab, cc, ls) in enumerate([
        ("cover68_reported", "reported $\\sigma$", VERM, "-"),
        ("cover68_widened", "widened by $\\sigma_{rep}(H)$", GREEN, "--"),
    ]):
        y, lo, hi = [], [], []
        for h in hv:
            g = d[d.H_simple == h]
            k_succ = int(round((g[col] * 8).sum()))
            n_tot = int(8 * len(g))
            y.append(k_succ / n_tot)
            a, b = proportion_confint(k_succ, n_tot, method="wilson")
            lo.append(a)
            hi.append(b)
        y = np.array(y)
        ax.errorbar(hv * (1 + 0.03 * (2 * k - 1)), y,
                    yerr=[y - np.array(lo), np.array(hi) - y],
                    color=cc, ls=ls, marker="os"[k], ms=5, capsize=3, lw=1.4,
                    mfc="none", label=lab)
    ax.axhline(0.68, color=GREY, lw=1.0, ls=":")
    ax.text(hv[0], 0.687, "nominal 0.68", fontsize=7, color=GREY)
    ax.set_xscale("log")
    ax.set_ylim(0.3, 1.05)
    ax.set_xlabel("$H = CV\\,L/\\sqrt{A}$")
    ax.set_ylabel("Empirical coverage of the 68% interval")
    ax.set_title("B. interval coverage\n(Wilson intervals, 8 seeds x designs)",
                 fontsize=8.5)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.grid(alpha=0.25, lw=0.4)

    ax = axes[2]
    for geom in ("point", "open_path"):
        st = GEOM_STYLE[geom]
        g = d[d.geometry == geom]
        ax.plot(100 * g.sigma_reported, 100 * g.sigma_rep, ls="none",
                marker=st["marker"], ms=4.5, mfc="none", color=st["color"],
                label=st["label"])
    lim = [0, 105]
    ax.plot(lim, lim, color=GREY, ls="--", lw=1.0)
    ax.set_xlim(0, 45)
    ax.set_ylim(0, 55)
    ax.set_xlabel("Reported $\\sigma$ / $Q_{true}$ (%)")
    ax.set_ylabel("Empirical $\\sigma_{rep}$ (%)")
    ax.set_title("C. reported vs actual spread", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=0.25, lw=0.4)

    byH = d.groupby("H_simple")[
        ["R_mean", "cover68_reported", "cover68_widened", "cover95_reported",
         "cover95_widened", "logscore_reported", "logscore_widened",
         "sigma_rep", "sigma_reported"]].mean()
    hi_, lo_ = byH.index.max(), byH.index.min()
    alt = f"""
Figure 6. Uncertainty calibration across all 72 conditions. Panel A plots
R = |error| / sigma_reported against H on a log x-axis, one point per condition,
with a least-squares line against log H. Panel B shows empirical coverage of the
nominal 68 percent interval at each of the six distinct values of H, for the
reported posterior sigma and for the widened
sigma_total^2 = sigma_reported^2 + sigma_rep(H)^2, where sigma_rep(H) comes from
a leave-one-condition-out fit so it is never calibrated on the condition it
scores. Error bars are Wilson intervals on the underlying binomial counts.
Panel C plots empirical sigma_rep against reported sigma, both as a percentage
of the true flux, with a one-to-one line.

R rises steeply and systematically with H (r = {r:.2f} against log H), from
{byH.loc[lo_,'R_mean']:.2f} at the lowest H to {byH.loc[hi_,'R_mean']:.2f} at the highest, so H does predict when formal
precision stops describing actual error. Coverage of the nominal 68 percent
interval is {byH.loc[lo_,'cover68_reported']:.2f} at low H -- the reported sigma is far too conservative there,
because the inversion configuration prescribes a 30 percent fractional
representation error regardless of the source field -- and falls monotonically
to {byH.loc[hi_,'cover68_reported']:.2f} at H = {hi_:.2f}, below nominal. Adding the empirical sigma_rep(H) term
restores the top end to {byH.loc[hi_,'cover68_widened']:.2f} but leaves the low-H conditions even more
over-covered: the log score improves at the highest H ({byH.loc[hi_,'logscore_reported']:.2f} to
{byH.loc[hi_,'logscore_widened']:.2f}) but the mean over all H falls from {byH.logscore_reported.mean():.2f} to {byH.logscore_widened.mean():.2f}, so the widening
fixes the tail at the cost of the bulk. Panel C shows why a single prescribed
sigma cannot do both -- reported sigma takes only eight values, set by the
measurement design, while sigma_rep varies over a factor of ten with the source
field at fixed design.

H,R_mean,cover68_reported,cover68_widened,cover95_reported,cover95_widened,logscore_reported,logscore_widened
""" + "\n".join(
        f"{i:.3f},{r_.R_mean:.3f},{r_.cover68_reported:.3f},"
        f"{r_.cover68_widened:.3f},{r_.cover95_reported:.3f},"
        f"{r_.cover95_widened:.3f},{r_.logscore_reported:.3f},"
        f"{r_.logscore_widened:.3f}" for i, r_ in byH.iterrows())
    save(fig, "fig6_calibration", alt)


def main() -> int:
    inv = pd.read_csv(HERE / "inversions.csv").merge(
        pd.read_csv(HERE / "error_decomposition.csv"), on="run")
    cond = pd.read_csv(HERE / "conditions.csv")
    fits = pd.read_csv(HERE / "model_fits.csv")
    cal = pd.read_csv(HERE / "calibration.csv")
    nd = pd.DataFrame(json.loads((HERE / "results.json").read_text())["n_dependence"])

    # fig1 is the phase diagram on the fine mesh (phase_mesh.py); the scaling
    # figures below stay on the original 8-seed 3 x 3 study.
    fig1(pd.read_csv(HERE / "inversions_mesh.csv"),
         pd.read_csv(HERE / "conditions_mesh.csv"))
    fig2(cond)
    fig3(cond)
    fig4(cond, nd)
    fig5(cond, fits)
    fig6(cond, cal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
