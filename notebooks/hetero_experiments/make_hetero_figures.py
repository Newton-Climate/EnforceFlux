"""Figures for the source-heterogeneity observability deck.

Every panel is drawn from data on disk: the 72 inversion summaries aggregated by
analyze_hetero_osse.py, the stored lognormal truth fields, the instrument configs,
and the gridded GHGI/CDL rasters used for the motivating context figure.

Run analyze_hetero_osse.py first; this script reads hetero_osse_results.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import xarray as xr
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = REPO / "runs"
CONFIGS = REPO / "configs" / "hetero_rice_paddy_test" / "inversions"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

Q_TRUE = 0.027778  # kg s-1
# load_ghgi_rice returns Tg CH4 per year per cell; CDL areas are hectares.
TG_PER_HA_TO_KG_PER_HA = 1.0e9
DOMAIN = 500.0  # metres, half-width of the 1 km square source domain
WIND_FROM_DEG = 35.0
WIND_SPEED = 1.75  # m s-1
FENCE_BEARING_DEG = 305.0

# Okabe-Ito, colourblind safe. Never colour alone: every series also varies
# marker and linestyle.
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREY = "#4D4D4D"
GEOM_STYLE = {
    "Open path": dict(color=BLUE, marker="o", ls="-"),
    "Point": dict(color=VERMILLION, marker="s", ls="--"),
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


def save(fig, stem: str, alt: str) -> Path:
    png = OUT / f"{stem}.png"
    fig.savefig(png)
    fig.savefig(OUT / f"{stem}.pdf")
    (OUT / f"{stem}.txt").write_text(alt.strip() + "\n")
    return png


def ef_grid() -> dict:
    """Gridded GHGI rice emissions and CDL rice area, cached after first build.

    Regenerating reads a 3 GB CDL raster window, so the result is cached beside
    this script; delete ef_grid.npz to force a rebuild.
    """
    cache = HERE / "ef_grid.npz"
    if not cache.exists():
        sys.path.insert(0, str(REPO / "data" / "presentation_figures"))
        import make_figures as mf

        rice, lons, lats = mf.load_ghgi_rice(mf.NC_FILE)
        area = mf.aggregate_cdl_rice(lons, lats)
        np.savez(cache, rice=rice, lons=lons, lats=lats, area=area)
    return np.load(cache)


def results() -> pd.DataFrame:
    return pd.read_csv(HERE / "hetero_osse_results.csv")


def truth_field(L: int, cv: float) -> xr.Dataset:
    cvs = f"{cv:.1f}".replace(".", "p")
    run = RUNS / f"source_heterogeneity_les_rice_paddy_l{L}_cv{cvs}_wind3_n1_op_gp" / "dispersion"
    return xr.open_dataset(run / "truth_field.nc")


# ---------------------------------------------------------------- source fields


def fig_source_fields():
    """The actual 3 x 3 lognormal nature fields, shared colour scale."""
    Ls, cvs = [100, 250, 500], [0.5, 1.0, 2.0]
    fields = {(L, cv): truth_field(L, cv) for L in Ls for cv in cvs}
    # Emission rate per unit area in mg m-2 s-1 for a readable colourbar.
    scaled = {k: ds.F_true.values * 1e6 for k, ds in fields.items()}
    vmax = max(v.max() for v in scaled.values())
    # The fields are lognormal, so a linear scale collapses eight of nine panels
    # into one dark tone. One shared log scale keeps the panels comparable and
    # still lets the eye verify that the totals match.
    vmin = min(v[v > 0].min() for v in scaled.values())
    norm = LogNorm(vmin=max(vmin, vmax / 1e3), vmax=vmax)

    fig, axes = plt.subplots(3, 3, figsize=(7.0, 6.6), sharex=True, sharey=True)
    for i, L in enumerate(Ls):
        for j, cv in enumerate(cvs):
            ax = axes[i, j]
            ds = fields[(L, cv)]
            im = ax.pcolormesh(
                ds.x.values, ds.y.values, scaled[(L, cv)],
                cmap="cividis", norm=norm, shading="nearest",
            )
            ax.set_aspect("equal")
            ax.set_xticks([-400, 0, 400])
            ax.set_yticks([-400, 0, 400])
            if i == 0:
                ax.set_title(f"CV = {cv:.1f}")
            # Outer panels only: repeating identical labels on a shared-axis
            # grid adds clutter without adding information.
            if j == 0:
                ax.set_ylabel(f"L = {L} m\ny (m)")
            else:
                ax.tick_params(labelleft=False)
            if i == 2:
                ax.set_xlabel("x (m)")
            else:
                ax.tick_params(labelbottom=False)
    cb = fig.colorbar(im, ax=axes, shrink=0.75, label="Emission rate (mg m$^{-2}$ s$^{-1}$)")
    cb.ax.tick_params(labelsize=8)
    tot = {k: float((ds.F_true.values * ds.cell_area_m2.values).sum()) for k, ds in fields.items()}
    spread = max(tot.values()) - min(tot.values())
    alt = (
        "A three by three grid of maps of the simulated rice-paddy emission field over a "
        "1 by 1 km domain, with correlation length L increasing down the rows (100, 250, 500 m) "
        "and coefficient of variation CV increasing across the columns (0.5, 1.0, 2.0). All nine "
        "panels share one logarithmic colour scale in milligrams per square metre per second, "
        "chosen because the fields are lognormal, and all nine "
        f"integrate to the same total flux of 0.027778 kilograms per second, agreeing to {spread:.1e} "
        "kilograms per second. Raising CV brightens and darkens individual cells; raising L merges "
        "them into larger coherent patches. All nine panels use the same random seed, so the "
        "hotspots sit in the same places throughout."
    )
    return fig, "fig_source_fields", alt, tot


# ------------------------------------------------------------------- geometry


def _instruments(n: int, geom: str) -> list[dict]:
    cfg = yaml.safe_load((CONFIGS / f"n{n}_{geom}_instrument.yaml").read_text())
    return cfg["instrument"]["instruments"]


CLEARANCE_M = [None]


def fig_geometry():
    """Real receptor geometry: fence sits downwind and outside the source domain."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), sharex=True, sharey=True)
    brg = np.deg2rad(FENCE_BEARING_DEG)
    unit = np.array([np.sin(brg), np.cos(brg)])

    for ax, (geom, label) in zip(axes, [("point", "Point sensor, n = 4"), ("op", "Open path, n = 4")]):
        ax.add_patch(Rectangle(
            (-DOMAIN, -DOMAIN), 2 * DOMAIN, 2 * DOMAIN,
            facecolor="#DDDDDD", edgecolor=GREY, lw=1.0, zorder=1,
        ))
        ax.text(0, 0, "source\ndomain\n1 x 1 km", ha="center", va="center",
                fontsize=8, color=GREY, zorder=2)

        for inst in _instruments(4, geom):
            p0 = np.array([inst["x_m"], inst["y_m"]])
            if geom == "op":
                p1 = p0 + inst["path_length_m"] * unit
                ax.plot(*zip(p0, p1), color=BLUE, lw=3.0, solid_capstyle="butt", zorder=4)
                # End caps, so four contiguous paths do not read as one line.
                perp = np.array([-unit[1], unit[0]])
                for end in (p0, p1):
                    ax.plot(*zip(end - 28 * perp, end + 28 * perp),
                            color="white", lw=1.4, zorder=5)
            else:
                ax.plot(*p0, marker="s", color=VERMILLION, ms=7, zorder=4)

        # Full fence line, common to both geometries.
        f0 = np.array([-1.7, -873.6])
        f1 = f0 + 1000.0 * unit
        ax.plot(*zip(f0, f1), color=GREY, lw=0.8, ls=":", zorder=3)

        # Wind arrow: blows from 35 degrees, i.e. towards 215 degrees.
        to = np.deg2rad(WIND_FROM_DEG + 180.0)
        w0 = np.array([620.0, 620.0])
        ax.annotate("", xy=w0 + 420 * np.array([np.sin(to), np.cos(to)]), xytext=w0,
                    arrowprops=dict(arrowstyle="-|>", lw=2.0, color="black"))
        ax.text(700, 700, f"wind {WIND_SPEED} m s$^{{-1}}$\nfrom {WIND_FROM_DEG:.0f}$\\degree$",
                fontsize=8, ha="left", va="bottom")

        ax.set_title(label)
        ax.set_aspect("equal")
        ax.set_xlim(-1050, 1050)
        ax.set_ylim(-1050, 1050)
        ax.set_xlabel("x, east (m)")
        ax.set_ylabel("y, north (m)")

    handles = [
        Line2D([], [], color=VERMILLION, marker="s", ls="none", ms=7, label="Point sensor"),
        Line2D([], [], color=BLUE, lw=3, label="Open path, 250 m each"),
        Line2D([], [], color=GREY, lw=0.8, ls=":", label="1 km crosswind fence"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    # Clearance measured from the fence line to the nearest point of the square,
    # which is the downwind corner, not the edge midpoint.
    normal = np.array([-unit[1], unit[0]])
    f_mid = np.array([-1.7, -873.6])
    normal = normal if normal @ f_mid < 0 else -normal
    corners = np.array([[-DOMAIN, -DOMAIN], [-DOMAIN, DOMAIN], [DOMAIN, -DOMAIN], [DOMAIN, DOMAIN]])
    clearance = abs(f_mid @ normal) - max(c @ -normal for c in corners)
    CLEARANCE_M[0] = clearance

    alt = (
        "Two plan-view maps of the measurement geometry, point sensors on the left and open path on "
        "the right, both for four instruments. A grey square marks the 1 by 1 km source domain "
        "centred on the origin. A dotted line marks a 1 km fence running on a bearing of 305 degrees "
        "from the point 2 m east, 874 m south of centre. The fence is perpendicular to a 1.75 metre "
        "per second wind blowing from 35 degrees, and it clears the nearest, downwind corner of the "
        "domain by only 20 m, although it is 217 m beyond the domain edge measured along the wind "
        "axis. The open-path "
        "panel tiles the fence with four contiguous 250 m paths; the point panel places four sensors at "
        "the midpoints of those same four segments. All instruments are at 2 m height."
    )
    return fig, "fig_geometry", alt, None


# ------------------------------------------------------------ headline: bias


def _jitter(n: int, width: float = 0.09, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-width, width, n)


def fig_bias_cv(df: pd.DataFrame):
    """Signed error against CV, with every individual run shown."""
    cvs = [0.5, 1.0, 2.0]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(0.0, color="black", lw=1.0, zorder=1)

    for k, (geom, style) in enumerate(GEOM_STYLE.items()):
        offs = -0.16 + 0.32 * k
        means, xs = [], []
        for i, cv in enumerate(cvs):
            v = df[(df.geometry == geom) & (df.CV == cv)].bias_total_pct.values
            x = i + offs
            ax.scatter(x + _jitter(len(v), seed=k * 10 + i), v, s=18,
                       facecolor="none", edgecolor=style["color"],
                       marker=style["marker"], lw=0.9, alpha=0.85, zorder=3)
            means.append(v.mean())
            xs.append(x)
        ax.plot(xs, means, color=style["color"], marker=style["marker"], ls=style["ls"],
                ms=9, lw=2.0, mec="white", mew=1.2, zorder=4, label=f"{geom} (mean of 12)")

    ax.set_xticks(range(len(cvs)))
    ax.set_xticklabels([f"CV = {c:.1f}" for c in cvs])
    ax.set_xlim(-0.5, len(cvs) - 0.5)
    ax.set_ylim(-30, 12)
    ax.set_ylabel("Signed total-flux error (%)\nnegative = underestimate")
    ax.set_xlabel("Source heterogeneity amplitude, CV (dimensionless)")
    ax.legend(loc="lower left", frameon=False)
    ax.text(0.98, 0.96, "12 runs per point (3 correlation lengths x 4 network sizes)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=GREY)

    stats = df.groupby(["geometry", "CV"]).bias_total_pct.agg(["mean", "min", "max", "count"])
    frac_under = df.assign(u=df.bias_total_pct < 0).groupby(["geometry", "CV"]).u.mean()
    alt = (
        "A scatter and line plot of signed total-flux retrieval error in percent against source "
        "heterogeneity amplitude CV, for open-path and point-sensor networks. Individual runs are "
        "open markers and the mean of the twelve runs at each CV is a filled marker joined by a line. "
        "A horizontal line marks zero error. Both geometries move from a small positive bias at CV 0.5 "
        "to a large negative bias at CV 2.0, where the open-path mean is minus 11.9 percent and the "
        "point-sensor mean is minus 12.5 percent, and every one of the twelve open-path runs "
        "underestimates the true flux."
    )
    return fig, "fig_bias_cv", alt, (stats, frac_under)


# ------------------------------------------------------------- decomposition


def fig_decomposition(df: pd.DataFrame):
    """Split the signed error into aggregation and transport-mismatch terms."""
    cvs = [0.5, 1.0, 2.0]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), sharey=True)
    width = 0.34

    for ax, geom in zip(axes, GEOM_STYLE):
        sub = df[df.geometry == geom].groupby("CV")[
            ["bias_aggregation_pct", "bias_transport_pct", "bias_total_pct"]
        ].mean()
        x = np.arange(len(cvs))
        ax.axhline(0.0, color="black", lw=1.0)
        ax.bar(x - width / 2, sub.bias_aggregation_pct, width,
               color=GREY, edgecolor="black", lw=0.6, label="Uniform-template aggregation")
        ax.bar(x + width / 2, sub.bias_transport_pct, width,
               color="white", edgecolor="black", lw=0.6, hatch="///",
               label="LES-versus-bLS transport")
        ax.plot(x, sub.bias_total_pct, color=GEOM_STYLE[geom]["color"],
                marker=GEOM_STYLE[geom]["marker"], ls=GEOM_STYLE[geom]["ls"],
                ms=8, lw=2.0, mec="white", mew=1.0, label="Total (reported)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c:.1f}" for c in cvs])
        ax.set_xlabel("CV (dimensionless)")
        ax.set_title(geom)
    axes[0].set_ylabel("Contribution to signed error (%)")
    axes[1].set_ylabel("Contribution to signed error (%)")
    axes[0].set_ylim(-34, 12)
    axes[0].legend(loc="lower left", frameon=False, fontsize=7.5)

    table = df.groupby(["geometry", "CV"])[
        ["bias_total_pct", "bias_aggregation_pct", "bias_transport_pct"]
    ].mean()
    alt = (
        "Two bar panels, open path on the left and point sensors on the right, decomposing the signed "
        "total-flux error at each CV into the part caused by inverting with a uniform template and the "
        "part caused by the difference between the large-eddy-simulation truth and the backward "
        "Lagrangian inversion operator, with the reported total overlaid as a line. For open path the "
        "two terms have the same sign and aggregation dominates, reaching minus 8.8 percent against "
        "minus 3.1 percent transport at CV 2.0. For point sensors the terms have opposite signs, minus "
        "19.4 percent aggregation against plus 7.0 percent transport at CV 2.0, so the smaller total is "
        "the result of cancellation rather than a smaller underlying error."
    )
    return fig, "fig_decomposition", alt, table


# ------------------------------------------------------------------ L trend


def fig_bias_L(df: pd.DataFrame):
    Ls = [100, 250, 500]
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.axhline(0.0, color="black", lw=1.0, zorder=1)
    for k, (geom, style) in enumerate(GEOM_STYLE.items()):
        offs = -0.14 + 0.28 * k
        means, xs = [], []
        for i, L in enumerate(Ls):
            v = df[(df.geometry == geom) & (df.L_m == L)].bias_total_pct.values
            x = i + offs
            ax.scatter(x + _jitter(len(v), seed=k * 7 + i), v, s=16, facecolor="none",
                       edgecolor=style["color"], marker=style["marker"], lw=0.9, alpha=0.85, zorder=3)
            means.append(v.mean()); xs.append(x)
        ax.plot(xs, means, color=style["color"], marker=style["marker"], ls=style["ls"],
                ms=9, lw=2.0, mec="white", mew=1.2, zorder=4, label=geom)
    ax.set_xticks(range(len(Ls)))
    ax.set_xticklabels([f"L = {L} m\n({L / 40:.1f} source cells)" for L in Ls])
    ax.set_xlim(-0.5, len(Ls) - 0.5)
    ax.set_ylim(-30, 12)
    ax.set_ylabel("Signed total-flux error (%)")
    ax.set_xlabel("Source correlation length, L (m)")
    ax.legend(loc="lower left", frameon=False)

    stats = df.groupby(["geometry", "L_m"]).bias_total_pct.agg(["mean", "min", "max"])
    alt = (
        "A scatter and line plot of signed total-flux error in percent against source correlation "
        "length L of 100, 250 and 500 m, for open-path and point-sensor networks, with the twelve "
        "individual runs at each L shown as open markers behind the mean. Error becomes more negative "
        "as L grows, from minus 1.9 percent at 100 m to minus 8.2 percent at 500 m for open path, but "
        "the spread at each L is far larger than the trend, and the 100 m case spans only 2.5 source "
        "grid cells so it is only marginally resolved."
    )
    return fig, "fig_bias_L", alt, stats


# ------------------------------------------------------- matched comparison


def fig_matched_scatter(df: pd.DataFrame):
    """Open path against point, matched on (L, CV, n). Equal aspect, CV encoded."""
    piv = df.pivot_table(index=["L_m", "CV", "n"], columns="geometry",
                         values="abs_error_pct").reset_index()
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    lim = 28.0
    ax.plot([0, lim], [0, lim], color="black", lw=1.0, zorder=2)
    ax.text(lim * 0.72, lim * 0.76, "1:1", fontsize=9, rotation=45,
            ha="center", va="bottom")
    ax.fill_between([0, lim], [0, lim], lim, color=BLUE, alpha=0.06, zorder=1)
    ax.text(3.0, lim * 0.93, "open path better", fontsize=8.5, color=BLUE, va="top")
    ax.text(lim * 0.30, 0.6, "point better", fontsize=8.5, color=VERMILLION)

    markers = {1: "o", 2: "s", 3: "^", 4: "D"}
    sizes = {0.5: 26, 1.0: 58, 2.0: 118}
    for _, r in piv.iterrows():
        win = r["Open path"] < r["Point"]
        ax.scatter(r["Open path"], r["Point"], s=sizes[r.CV], marker=markers[r.n],
                   facecolor=(BLUE if win else VERMILLION), edgecolor="black",
                   lw=0.5, alpha=0.8, zorder=3)

    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel("Open-path absolute error (%)")
    ax.set_ylabel("Point-sensor absolute error (%)")

    size_h = [Line2D([], [], ls="none", marker="o", ms=np.sqrt(sizes[c]),
                     mfc="none", mec="black", label=f"CV {c:.1f}") for c in sizes]
    mark_h = [Line2D([], [], ls="none", marker=markers[k], ms=6, mfc="none",
                     mec="black", label=f"n = {k}") for k in markers]
    ax.legend(handles=size_h + mark_h, loc="lower right", frameon=False,
              fontsize=7.5, ncol=2, handletextpad=0.6, labelspacing=0.8)

    n_op = int((piv["Open path"] < piv["Point"]).sum())
    alt = (
        "A scatter plot of point-sensor absolute error against open-path absolute error in percent for "
        "all 36 matched configurations, with equal axes from 0 to 28 percent and a 1 to 1 reference "
        "line. Marker size encodes CV and marker shape encodes network size n. Points above the line, "
        f"where open path does better, number {n_op} of 36. The largest markers, which are the CV 2.0 "
        "cases, sit farthest from the origin, and the cases where point sensors win cluster at low "
        "error and low CV near the origin."
    )
    return fig, "fig_matched_scatter", alt, piv


# ------------------------------------------------- network size: two effects


def fig_n_precision(df: pd.DataFrame):
    """Accuracy and precision against n, on separate axes, plus DFS."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8))
    ns = [1, 2, 3, 4]

    ax = axes[0]
    ax.axhline(0.0, color="black", lw=1.0, zorder=1)
    for k, (geom, style) in enumerate(GEOM_STYLE.items()):
        offs = -0.11 + 0.22 * k
        means, xs = [], []
        for i, n in enumerate(ns):
            v = df[(df.geometry == geom) & (df.n == n)].bias_total_pct.values
            ax.scatter(n + offs + _jitter(len(v), 0.06, seed=k * 5 + i), v, s=15,
                       facecolor="none", edgecolor=style["color"], marker=style["marker"],
                       lw=0.9, alpha=0.85, zorder=3)
            means.append(v.mean()); xs.append(n + offs)
        ax.plot(xs, means, color=style["color"], marker=style["marker"], ls=style["ls"],
                ms=8, lw=2.0, mec="white", mew=1.1, zorder=4, label=geom)
    ax.set_xticks(ns); ax.set_xlim(0.5, 4.5); ax.set_ylim(-32, 22)
    ax.set_xlabel("Number of instruments, n (count)")
    ax.set_ylabel("Signed total-flux error (%)")
    ax.set_title("Accuracy: no reliable gain")
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=7.5)

    ax = axes[1]
    for geom, style in GEOM_STYLE.items():
        s = df[df.geometry == geom].groupby("n").posterior_sigma_kg_s.mean() / Q_TRUE * 100
        ax.plot(s.index, s.values, color=style["color"], marker=style["marker"],
                ls=style["ls"], ms=8, lw=2.0, mec="white", mew=1.1, label=geom)
    ref = df[df.geometry == "Open path"].groupby("n").posterior_sigma_kg_s.mean().iloc[0] / Q_TRUE * 100
    ax.plot(ns, [ref / np.sqrt(n) for n in ns], color="black", lw=1.0, ls=":",
            label="1 / sqrt(n) from n = 1")
    ax.set_xticks(ns); ax.set_xlim(0.5, 4.5); ax.set_ylim(0, 45)
    ax.set_xlabel("Number of instruments, n (count)")
    ax.set_ylabel("Posterior sigma (% of true flux)")
    ax.set_title("Precision: falls as 1/sqrt(n) by construction")
    ax.legend(loc="upper right", frameon=False, fontsize=7.5)

    sig = df.groupby(["geometry", "n"]).agg(
        signed_bias_pct=("bias_total_pct", "mean"),
        posterior_sigma_kg_s=("posterior_sigma_kg_s", "mean"),
        dfs=("dfs", "mean"),
    )
    alt = (
        "Two panels against number of instruments n from 1 to 4. The left panel shows signed "
        "total-flux error with individual runs behind the means: neither geometry shows a reliable "
        "improvement in accuracy with n, and the spread at every n is roughly 25 percentage points "
        "wide. The right panel shows posterior sigma as a percentage of the true flux, falling from "
        "39 percent for open path and 35 percent for point sensors at n equals 1 to 21 percent for "
        "both at n equals 4, tracking a plotted 1 over "
        "square root of n reference curve closely, because the assumed representativeness error "
        "is fixed per observation and independent between instruments."
    )
    return fig, "fig_n_precision", alt, sig


# ------------------------------------------------------- motivating context


def fig_ef_context():
    """Single readable panel replacing the dense two-panel motivation figure."""
    cache = ef_grid()
    rice, area = cache["rice"], cache["area"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ef = np.where(area > 100.0, rice / np.where(area > 0, area, np.nan), np.nan)
    ef = ef[np.isfinite(ef)] * TG_PER_HA_TO_KG_PER_HA

    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ax.hist(ef, bins=np.arange(0, 2001, 100), color="#BBBBBB", edgecolor="black", lw=0.5)
    ax.axvspan(300, 600, color=BLUE, alpha=0.12, zorder=0)
    ax.text(450, ax.get_ylim()[1] * 0.97, "IPCC\n300-600", ha="center", va="top",
            fontsize=8, color=BLUE)

    refs = [
        (742, "EPA GHGI 742", VERMILLION, "-"),
        (143, "CARB 143", VERMILLION, "--"),
        (float(np.median(ef)), f"median {np.median(ef):.0f}", "black", ":"),
    ]
    for x, lab, c, ls in refs:
        ax.axvline(x, color=c, ls=ls, lw=1.6, zorder=4)
        ax.annotate(lab, xy=(x, ax.get_ylim()[1] * 0.62), xytext=(6, 0),
                    textcoords="offset points", rotation=90, fontsize=8,
                    color=c, va="center")

    ax.set_xlim(0, 2000)
    ax.set_xlabel("Rice CH$_4$ emission factor (kg ha$^{-1}$ yr$^{-1}$)")
    ax.set_ylabel(f"Number of 0.1 degree grid cells (n = {len(ef)})")
    ax.set_title("Implied emission factors across Sacramento Valley rice cells, 2020")

    alt = (
        f"A histogram of implied rice methane emission factors in kilograms per hectare per year "
        f"across {len(ef)} Sacramento Valley grid cells of 0.1 degree, binned every 100 units from 0 "
        f"to 2000. A shaded band marks the IPCC reference range of 300 to 600 and three vertical lines "
        f"mark the EPA gridded inventory value of 742, the CARB value of 143, and the distribution "
        f"median of {np.median(ef):.0f}. The distribution is broad and right-skewed and the two agency "
        f"estimates differ by a factor of 5.2, with CARB below and EPA above the IPCC range."
    )
    return fig, "fig_ef_context", alt, ef


def fig_sac_map():
    """Clean, uncropped version of the title-slide emission-factor map."""
    cache = ef_grid()
    rice, lons, lats, area = cache["rice"], cache["lons"], cache["lats"], cache["area"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ef = np.where(area > 100.0, rice / np.where(area > 0, area, np.nan), np.nan)
    ef = ef * TG_PER_HA_TO_KG_PER_HA

    fig, ax = plt.subplots(figsize=(5.0, 5.4))
    im = ax.pcolormesh(lons, lats, ef, cmap="cividis", shading="auto",
                       vmin=0.0, vmax=float(np.nanmax(ef)))
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (degrees east)")
    ax.set_ylabel("Latitude (degrees north)")
    ax.set_title("Implied rice CH$_4$ emission factor\nSacramento Valley, 2020")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("kg ha$^{-1}$ yr$^{-1}$")
    ok = np.isfinite(ef)
    lon_e, lat_e = np.meshgrid(lons, lats)
    ax.set_xlim(lon_e[ok].min() - 0.15, lon_e[ok].max() + 0.15)
    ax.set_ylim(lat_e[ok].min() - 0.15, lat_e[ok].max() + 0.15)

    finite = ef[np.isfinite(ef)]
    alt = (
        "A map of the Sacramento Valley showing implied rice methane emission factor per 0.1 degree "
        "grid cell in kilograms per hectare per year, on a cividis colour scale with a labelled "
        f"colourbar. {len(finite)} cells carry rice area above 100 hectares; values range from "
        f"{finite.min():.0f} to {finite.max():.0f} with a median of {np.median(finite):.0f}. The "
        "highest values sit along the western side of the valley."
    )
    return fig, "fig_sac_map", alt, finite


# --------------------------------------------------------------------- driver


def main() -> None:
    df = results()
    builders = [
        lambda: fig_sac_map(),
        lambda: fig_ef_context(),
        lambda: fig_source_fields(),
        lambda: fig_geometry(),
        lambda: fig_bias_cv(df),
        lambda: fig_decomposition(df),
        lambda: fig_bias_L(df),
        lambda: fig_matched_scatter(df),
        lambda: fig_n_precision(df),
    ]
    figs = {}
    for b in builders:
        fig, stem, alt, extra = b()
        path = save(fig, stem, alt)
        figs[stem] = (fig, alt, extra)
        print(f"wrote {path}")
        plt.close(fig)
    return figs


if __name__ == "__main__":
    main()
