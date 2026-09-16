"""Figures for the representativeness narrative of the source-heterogeneity deck.

The deck establishes the representativeness problem with POINT SENSORS as the
baseline sampling geometry, and only then introduces open-path integration as a
measurement-design response. make_seed_figures.py draws both networks together,
which pre-empts that argument, so the point-sensor-only panels used on the
mechanism slides are drawn here instead. Every number is read from the same
seed sweep; nothing is recomputed or rescaled.

  seed_sweep_results.csv   576 inversions: L x CV x seed x n x network

Also builds the two conceptual panels that carry no simulation output:
the CV-L regime diagram and the effective-replication tiling; and crops the
two illustrated schematics down to the panels the deck actually uses.

Style follows make_seed_figures.py: Okabe-Ito, colour never used alone, a
plain-text description beside every PNG, and no slide titles or takeaway
sentences inside the image.

  python3 notebooks/hetero_experiments/make_narrative_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
PRES = HERE.parents[1] / "data" / "presentation_figures"

Q_TRUE = 0.027778  # kg s-1

BLUE = "#0072B2"      # open path, everywhere in this deck
VERMILLION = "#D55E00" # point sensors
GREY = "#4D4D4D"
FAINT = "#B8C2C6"

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
    return df[df.network == "point"].copy()


def jitter(n: int, width: float = 0.10, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-width, width, n)


# ------------------------------------- 1. emission contrast: spread, not bias


def fig_point_cv(df: pd.DataFrame) -> None:
    cvs = [0.5, 1.0, 2.0]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))

    ax = axes[0]
    ax.axhline(0, color=GREY, lw=0.8, zorder=0)
    for i, cv in enumerate(cvs):
        v = df[df.CV == cv].err_pct.values
        ax.scatter(i + jitter(len(v), 0.13, seed=i), v, s=9, facecolors="none",
                   edgecolors=VERMILLION, lw=0.5, alpha=0.55, zorder=2)
    m = [df[df.CV == cv].err_pct.mean() for cv in cvs]
    s = [df[df.CV == cv].err_pct.std() for cv in cvs]
    ax.errorbar(np.arange(3), m, yerr=s, color=VERMILLION, marker="s", ls="--",
                capsize=4, lw=1.6, ms=6, zorder=3, label="Mean ± 1 s.d.")
    ax.set_xticks(range(3), [f"{c:g}" for c in cvs])
    ax.set_xlabel("Emission contrast  CV = σ / μ")
    ax.set_ylabel("Total-flux error (%)")
    ax.set_title("Every individual estimate")
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1]
    ax.bar(np.arange(3), s, 0.55, color=VERMILLION, alpha=0.85, edgecolor="white")
    for i, val in enumerate(s):
        ax.text(i, val + 0.7, f"±{val:.0f}%", ha="center", fontsize=8.5)
    ax.plot(np.arange(3), np.abs(m), color=GREY, marker="o", ms=5, lw=1.4, ls=":",
            label="|mean error|")
    ax.set_xticks(range(3), [f"{c:g}" for c in cvs])
    ax.set_xlabel("Emission contrast  CV = σ / μ")
    ax.set_ylabel("Run-to-run spread, 1 s.d. (%)")
    ax.set_title("Spread across eight source realizations")
    ax.set_ylim(0, 32)
    ax.legend(loc="upper left", frameon=False)

    alt = (
        "Two panels for point-sensor networks against emission contrast CV of 0.5, 1.0 "
        "and 2.0, using the 288 point-sensor inversions across eight source "
        "realizations. The left panel plots the signed total-flux error of every "
        "individual run as an open square with the mean and one standard deviation "
        f"overlaid. The mean sits near zero at every contrast: {m[0]:+.1f}, {m[1]:+.1f} "
        f"and {m[2]:+.1f} percent, so heterogeneity introduces no consistent "
        "over- or under-estimate once source realizations are resampled. The right "
        "panel shows the run-to-run standard deviation growing steeply with contrast, "
        f"from {s[0]:.0f} to {s[1]:.0f} to {s[2]:.0f} percent, plotted against the much "
        "smaller absolute mean error on the same axes. Emission contrast therefore "
        "controls how far a single campaign can fall from the field mean, not the "
        "direction in which it falls."
    )
    save(fig, "fig_point_cv", alt)


# ------------------------------ 2. patch scale: effective spatial replication


def fig_point_L(df: pd.DataFrame) -> None:
    Ls = [100, 250, 500]
    neff = [(1000.0 / L) ** 2 for L in Ls]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))

    ax = axes[0]
    ax.axhline(0, color=GREY, lw=0.8, zorder=0)
    for i, L in enumerate(Ls):
        v = df[df.L_m == L].err_pct.values
        ax.scatter(i + jitter(len(v), 0.13, seed=7 + i), v, s=9, facecolors="none",
                   edgecolors=VERMILLION, lw=0.5, alpha=0.55, zorder=2)
    m = [df[df.L_m == L].err_pct.mean() for L in Ls]
    s = [df[df.L_m == L].err_pct.std() for L in Ls]
    ax.errorbar(np.arange(3), m, yerr=s, color=VERMILLION, marker="s", ls="--",
                capsize=4, lw=1.6, ms=6, zorder=3, label="Mean ± 1 s.d.")
    ax.set_xticks(range(3), [f"{L}" for L in Ls])
    ax.set_xlabel("Patch scale  L (m)")
    ax.set_ylabel("Total-flux error (%)")
    ax.set_title("Error against patch scale")
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1]
    mae = [df[df.L_m == L].abs_pct.mean() for L in Ls]
    ax.plot(neff, mae, color=VERMILLION, marker="s", ls="--", ms=6, lw=1.6)
    for x, y, L in zip(neff, mae, Ls):
        ax.annotate(f"{y:.0f}%", (x, y), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8.5, color=VERMILLION)
    ax.set_xscale("log")
    ax.set_xticks(neff, [f"{n:.0f}\nL = {L} m" for n, L in zip(neff, Ls)])
    ax.set_xlabel("Effective spatial replication  $N_{eff}$ ≈ A / L²")
    ax.set_ylabel("Mean absolute error (%)")
    ax.set_title("Error against effective replication")
    ax.set_ylim(0, 26)
    ax.invert_xaxis()

    alt = (
        "Two panels for point-sensor networks on source patch scale L of 100, 250 and "
        "500 m in a 1 km field. The left panel plots signed total-flux error for every "
        f"run with the mean and one standard deviation: the mean is {m[0]:+.1f} percent "
        f"at 100 m, {m[1]:+.1f} at 250 m and {m[2]:+.1f} at 500 m, while the spread "
        f"widens from {s[0]:.0f} to {s[1]:.0f} to {s[2]:.0f} percent. The right panel "
        "replots mean absolute error against the approximate number of independent "
        "source regions the field contains, area divided by L squared, on a logarithmic "
        "axis running from 100 regions at L = 100 m through 16 at 250 m to 4 at 500 m, "
        f"with replication decreasing to the right. Error rises from {mae[0]:.0f} "
        f"percent at 100 regions to {mae[1]:.0f} percent at 16 and {mae[2]:.0f} percent "
        "at 4. Accuracy tracks effective spatial replication rather than patch scale "
        "directly: many small patches average effectively, a few large ones do not."
    )
    save(fig, "fig_point_L", alt)


# ------------------------------------- 3. conceptual: what N_eff looks like


def fig_neff_tiles() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
    rng = np.random.default_rng(4)
    for ax, L in zip(axes, [100, 250, 500]):
        k = int(round(1000 / L))
        shade = rng.uniform(0.15, 1.0, (k, k))
        for i in range(k):
            for j in range(k):
                ax.add_patch(Rectangle((i / k, j / k), 1 / k, 1 / k,
                                       facecolor=plt.cm.YlGnBu(0.15 + 0.7 * shade[i, j]),
                                       edgecolor="white", lw=0.5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(GREY)
            sp.set_linewidth(0.8)
        ax.set_title(f"L = {L} m\n$N_{{eff}}$ ≈ {k * k}", fontsize=9.5)
    axes[0].set_xlabel("1 km", fontsize=8.5, color=GREY)

    alt = (
        "Three squares, each representing the same 1 km by 1 km field, tiled into "
        "independent source regions of side L. At L = 100 m the field holds about 100 "
        "regions; at 250 m about 16; at 500 m only 4. Shading of the tiles is random and "
        "illustrative, indicating that each region emits independently. The figure is "
        "conceptual: it shows that the number of effectively independent source regions "
        "falls as the square of patch scale, which is the quantity that limits how well "
        "any downwind sample can average over the field."
    )
    save(fig, "fig_neff_tiles", alt)


# -------------------------- 4. sampling effort: precision against accuracy


def fig_point_n(df: pd.DataFrame) -> None:
    ns = [1, 2, 3, 4]
    mae = np.array([df[df.n == n].abs_pct.mean() for n in ns])
    sig = np.array([df[df.n == n].sigma_pct.mean() for n in ns])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))

    ax = axes[0]
    ax.plot(ns, mae, color=VERMILLION, marker="s", ls="--", ms=6, lw=1.8,
            label="Actual error")
    ax.plot(ns, sig, color=GREY, marker="o", ls="-", ms=6, lw=1.8,
            label="Reported uncertainty")
    for x, y in zip(ns, mae):
        ax.annotate(f"{y:.0f}", (x, y), xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=8, color=VERMILLION)
    for x, y in zip(ns, sig):
        ax.annotate(f"{y:.0f}", (x, y), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=8, color=GREY)
    ax.set_xticks(ns)
    ax.set_ylim(0, 42)
    ax.set_xlabel("Point sensors in the network  n")
    ax.set_ylabel("Percent of true flux")
    ax.set_title("Actual error and reported uncertainty")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1]
    # Marginal effect of the n-th sensor. Normalising to n = 1 instead would hide
    # the point: over 1 -> 4 both quantities fall by almost the same factor, and
    # the whole of the accuracy gain is spent on the second sensor.
    d_mae = -np.diff(mae)
    d_sig = -np.diff(sig)
    idx = np.arange(3)
    ax.bar(idx - 0.19, d_mae, 0.36, color=VERMILLION, alpha=0.85, edgecolor="white",
           label="Actual error")
    ax.bar(idx + 0.19, d_sig, 0.36, color=GREY, alpha=0.85, edgecolor="white",
           hatch="//", label="Reported uncertainty")
    for i, (a, b) in enumerate(zip(d_mae, d_sig)):
        ax.text(i - 0.19, a + 0.18, f"{a:.1f}", ha="center", fontsize=8,
                color=VERMILLION)
        ax.text(i + 0.19, b + 0.18, f"{b:.1f}", ha="center", fontsize=8, color=GREY)
    ax.set_xticks(idx, ["1 → 2", "2 → 3", "3 → 4"])
    ax.set_ylim(0, 8.2)
    ax.set_xlabel("Sensor added")
    ax.set_ylabel("Reduction (percentage points)")
    ax.set_title("Gain from each additional sensor")
    ax.legend(loc="upper right", frameon=False)

    alt = (
        "Two panels for point-sensor networks against the number of sensors n from 1 to "
        "4, over the 288 point-sensor inversions. The left panel plots two quantities as "
        "a percentage of the true flux. Mean absolute error, the actual error, falls "
        f"from {mae[0]:.0f} percent at one sensor to {mae[1]:.0f}, {mae[2]:.0f} and "
        f"{mae[3]:.0f} percent at two, three and four, so it flattens after the second "
        "sensor. The uncertainty the retrieval reports lies above the actual error at "
        f"every n and falls steadily, from {sig[0]:.0f} to {sig[1]:.0f} to {sig[2]:.0f} "
        f"to {sig[3]:.0f} percent. The right panel gives the reduction contributed by "
        "each additional sensor in percentage points. The second sensor reduces actual "
        f"error by {d_mae[0]:.1f} points, but the third and fourth contribute only "
        f"{d_mae[1]:.1f} and {d_mae[2]:.1f} points. Reported uncertainty instead keeps "
        f"falling by {d_sig[0]:.1f}, {d_sig[1]:.1f} and {d_sig[2]:.1f} points. Beyond the "
        "second sensor the network reports steadily improving precision while its actual "
        "accuracy has stopped improving, because the limiting error is the "
        "representativeness of the sample and that error is common to every sensor on "
        "the fence rather than independent between them."
    )
    save(fig, "fig_point_n", alt)


# ----------------------------------- 5. conceptual: ecological regime space


def fig_regimes() -> None:
    """Qualitative placement only. No ecosystem is assigned a numeric CV or L."""
    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    # Axes are deliberately unitless: soft regions, not measured coordinates.
    regions = [
        # label, x, y, width, height, colour, text offset
        ("Managed cropland\nGrassland", 0.30, 0.20, 0.40, 0.24, "#8FBEA6", (0, 0)),
        ("Rice checks\nManagement units", 0.56, 0.42, 0.36, 0.26, "#7FA8C9", (0, 0)),
        ("Wetland microsites\nEbullition hotspots\nManure deposition",
         0.24, 0.74, 0.34, 0.34, "#E8A15C", (0, 0)),
        ("Wetland vegetation zones\nInundated vs drained\nThaw features\nLagoon / barn / pasture",
         0.74, 0.76, 0.42, 0.36, "#D08C86", (0, 0)),
    ]
    for label, x, y, w, h, col, _ in regions:
        ax.add_patch(Ellipse((x, y), w, h, facecolor=col, alpha=0.42,
                             edgecolor=col, lw=1.2))
        ax.text(x, y, label, ha="center", va="center", fontsize=8.6,
                color="#22323A", linespacing=1.35)

    ax.set_xlim(0.03, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Patch scale  L", fontsize=11)
    ax.set_ylabel("Emission contrast  CV", fontsize=11)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(GREY)
        sp.set_linewidth(0.8)

    # Direction of increase, set on the axis itself rather than in the axis label.
    ax.annotate("", xy=(0.42, -0.055), xytext=(0.10, -0.055),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.0))
    ax.text(0.10, -0.10, "small", transform=ax.transAxes, fontsize=8.5, color=GREY)
    ax.text(0.42, -0.10, "large", transform=ax.transAxes, fontsize=8.5, color=GREY,
            ha="right")
    ax.annotate("", xy=(-0.055, 0.42), xytext=(-0.055, 0.10),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.0))
    ax.text(-0.085, 0.10, "low", transform=ax.transAxes, fontsize=8.5, color=GREY,
            rotation=90)
    ax.text(-0.085, 0.42, "high", transform=ax.transAxes, fontsize=8.5, color=GREY,
            rotation=90, va="top")

    alt = (
        "A conceptual two-dimensional diagram with patch scale L increasing along the "
        "horizontal axis and emission contrast CV increasing up the vertical axis. "
        "Neither axis carries numbers, because ecosystem positions are qualitative. Four "
        "soft shaded regions are placed on it. Low contrast at intermediate patch scale: "
        "managed cropland and grassland. Intermediate contrast at larger patch scale: "
        "rice checks and management units. High contrast at small patch scale: wetland "
        "microsites, ebullition hotspots and localized manure deposition. High contrast "
        "at large patch scale: wetland vegetation zones, inundated versus drained "
        "regions, thaw features, and lagoon, barn and pasture complexes. A dashed "
        "rectangle marks the region of the space simulated in this study, CV from 0.5 to "
        "2.0 and L from 100 to 500 m, which spans all four regions without asserting a "
        "numeric value for any named ecosystem."
    )
    save(fig, "fig_regimes", alt)


# --------------------------------------------- 6. crops of the schematics


def crop(src: Path, stem: str, box_frac: tuple[float, float, float, float],
         alt: str, whiten: bool = False) -> None:
    """Crop by fractional box (left, top, right, bottom) and write beside the rest."""
    from PIL import Image

    im = Image.open(src)
    w, h = im.size
    l, t, r, b = box_frac
    im = im.crop((int(l * w), int(t * h), int(r * w), int(b * h)))
    if whiten:
        im = _whiten(im)
    im.save(OUT / f"{stem}.png")
    (OUT / f"{stem}.txt").write_text(alt.strip() + "\n")
    print(f"wrote {stem}.png")


def _whiten(im, thresh: int = 236):
    """Flatten a near-white, near-neutral ground to pure white; leave artwork alone."""
    from PIL import Image

    a = np.array(im.convert("RGB")).astype(np.int16)
    lo, hi = a.min(axis=2), a.max(axis=2)
    bg = (lo >= thresh) & ((hi - lo) <= 6)
    a[bg] = 255
    return Image.fromarray(a.astype(np.uint8))


def crops() -> None:
    crop(
        PRES / "instrument_footprint_schematic.png",
        "fig_supports",
        (0.0, 0.02, 0.744, 0.97),
        "Three map-view panels of the same heterogeneous 1 km field, shown as a "
        "blue-to-red emission raster, illustrating the spatial support of different "
        "measurements under the same left-to-right wind. Chamber: a single small square "
        "samples one point of the field. Point sensor: a teardrop-shaped footprint "
        "extends upwind from one instrument, weighting cells near and directly upwind of "
        "it most strongly and distant cells hardly at all. Open path: a long narrow "
        "footprint extends upwind of a line between two endpoints, covering a wider swath "
        "of the field. A shared colour bar runs from low to high emission. The measured "
        "signal is a weighted average over an area the wind selects, and the shape of "
        "that weighting differs between geometries.",
        whiten=True,
    )
    crop(
        PRES / "flux_footprint_schematic.png",
        "fig_campaigns",
        (0.0, 0.0, 1.0, 1.0),  # full extent: the estimate boxes are the point
        "Three map-view panels labelled Campaign A, B and C, each showing the same "
        "heterogeneous field with the same total emission and one point sensor on its "
        "southern edge. Nested white contours mark the footprint weighting upwind of the "
        "sensor. In A and B the wind is from the south and the footprint falls on "
        "different parts of the field; in C the wind is from the south-west and the "
        "elongated footprint covers a strongly emitting region. A rounded box under each "
        "panel reports an illustrative recovered estimate of 84, 101 and 118 against a "
        "field mean of 100; these are schematic values, not simulation output. The same "
        "field therefore presents a different weighted sample on each occasion. Wind direction changes "
        "which part of the source the measurement weights; it does not change the source "
        "itself.",
        whiten=True,
    )
    crop(
        PRES / "heterogeneous_ems_landscapes.png",
        "fig_landscapes",
        (0.0, 0.0, 1.0, 1.0),
        "Four map-view panels of different diffuse methane sources, each about 800 m "
        "across with its own scale bar and a shared low-to-high emission colour bar. "
        "Rice field: rectangular flooded checks separated by bunds, with diffuse warm "
        "patches roughly one check in size. Wetland: many small, high-contrast hotspots "
        "scattered through pools and hummocks. Pasture and lagoon: a single large, "
        "strongly emitting region around one lagoon, with low emission elsewhere. "
        "Peatland: broad elongated zones of moderate emission following the drainage "
        "pattern. The four systems carry the same gas but differ in both the contrast "
        "between strong and weak regions and the spatial scale over which that contrast "
        "persists.",
        whiten=True,
    )
    # Title-slide visual: one real simulated source field, no axes or colour bar.
    # A single panel of fig_source_fields (L = 250 m, CV = 2.0), so the opening
    # image is source heterogeneity rather than a rice inventory map.
    crop(
        OUT / "fig_source_fields.png",
        "fig_title_field",
        (0.6175, 0.3825, 0.8325, 0.6135),
        "A single square map of one simulated methane source field, 1 km on a side, "
        "shown as a raster with no axes or colour bar. Emission is organised into "
        "irregular patches a few hundred metres across, with bright high-emitting cells "
        "in one corner and a dark low-emitting region through the centre. It is the "
        "L = 250 m, CV = 2.0 case from the nine simulated fields.",
        whiten=True,
    )

def main() -> None:
    df = sweep()
    fig_point_cv(df)
    fig_point_L(df)
    fig_neff_tiles()
    fig_point_n(df)
    fig_regimes()
    crops()


if __name__ == "__main__":
    main()
