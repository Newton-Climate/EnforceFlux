"""Plot the paddy CH4 signal seen by an OP beam and three point sensors.

Four sensors over the July-flux rice paddy, in one convective hour of LES:

    OP across paddy      300 m crosswind beam centred on the field
    Point, paddy centre  in-field, at the beam's midpoint
    Point, downwind edge at the field's downwind bund (+150 m)
    Point, 200 m beyond  clear of the field (+350 m from centre)

The figure's point is the comparison against the instrument's own error bars:
every signal here lies below the OP's 10 ppb random error, so the shaded noise
band and the detection limit are drawn as reference lines rather than left to
the reader to imagine.

Usage:
    python examples/plot_open_path_timeseries.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "runs" / "sacramento_convective_compare"

# Categorical slots 1-4 of the validated default palette, in fixed order.
SERIES_COLORS = ("#2a78d6", "#008300", "#e87ba4", "#eda100")
TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED = "#0b0b0b", "#52514e", "#84837c"
SURFACE = "#fcfcfb"


def build_series(root: Path, season: str):
    """Path/point averages in ppb for each sensor, plus the time axis."""
    import sys
    sys.path.insert(0, str(REPO / "examples"))
    from microhh_open_path_operator import KGKG_TO_PPM, PADDY_FLUX, load_xy_slices

    from enforceflux.instrument import path_average_series
    from enforceflux.microhh import load_microhh_config

    cfg = load_microhh_config(root / "paddy" / "microhh_generated.yaml")
    g = cfg.grid
    xax = (np.arange(g.itot) + 0.5) * g.dx
    yax = (np.arange(g.jtot) + 0.5) * g.dy

    times, fields = load_xy_slices(cfg.case_dir, cfg.scalar_name, g.itot, g.jtot)
    keep = times >= cfg.spinup_s
    scale = PADDY_FLUX[season] / PADDY_FLUX["april"]
    fields = fields[keep] * KGKG_TO_PPM * scale * 1000.0        # -> ppb
    minutes = (times[keep] - cfg.spinup_s) / 60.0

    x0, y0 = cfg.source_x0, cfg.source_y0
    sensors = [
        ("OP across paddy (300 m beam)", x0, 300.0),
        ("Point, paddy centre", x0, 0.0),
        ("Point, downwind edge (+150 m)", x0 + 150.0, 0.0),
        ("Point, 200 m beyond field", x0 + 350.0, 0.0),
    ]
    series = [(label, path_average_series(fields, xax, yax, xs, y0, length,
                                          0.0, centred=True))
              for label, xs, length in sensors]
    return minutes, series, cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--season", default="july", choices=("april", "july"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from enforceflux.instrument import INSTRUMENT_DB

    minutes, series, cfg = build_series(args.root, args.season)
    params = INSTRUMENT_DB["OP"]["good"]
    noise_ppb = params.sigma_abs * 1000.0
    limit_ppb = params.detection_limit * 1000.0

    fig, ax = plt.subplots(figsize=(11.5, 6.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # The y-range must show the whole noise band, otherwise the band fills the
    # axes and reads as a background tint instead of an instrument limit.
    top = max(noise_ppb * 1.45, max(v.max() for _, v in series) * 1.25)
    x_end = minutes[-1]
    x_pad = x_end * 1.155

    # Instrument reality, drawn behind the data.
    ax.axhspan(0, noise_ppb, color=TEXT_MUTED, alpha=0.11, lw=0, zorder=0)
    ax.axhline(noise_ppb, color=TEXT_SECONDARY, lw=1.2, alpha=0.6, zorder=1)
    ax.axhline(limit_ppb, color=TEXT_SECONDARY, lw=1.2, ls=(0, (5, 3)),
               alpha=0.6, zorder=1)
    # Label the reference lines in the clear band above the data rather than
    # inline: at y = 5 and y = 10 the plot is dense and inline text overprints
    # the series.
    from matplotlib.lines import Line2D
    ref_handles = [
        Line2D([], [], color=TEXT_SECONDARY, lw=1.2, alpha=0.6,
               label=f"OP random error  1σ = {noise_ppb:.0f} ppb"),
        Line2D([], [], color=TEXT_SECONDARY, lw=1.2, alpha=0.6, ls=(0, (5, 3)),
               label=f"detection limit = {limit_ppb:.0f} ppb"),
    ]
    ref_legend = ax.legend(handles=ref_handles, loc="upper left", frameon=False,
                           fontsize=9.5, labelcolor=TEXT_SECONDARY,
                           handlelength=2.4, borderaxespad=0.6)
    ax.add_artist(ref_legend)

    for (label, values), color in zip(series, SERIES_COLORS):
        ax.plot(minutes, values, color=color, lw=1.7, label=label,
                solid_capstyle="round", zorder=3)

    # Direct labels as well as the legend: three of these hues sit under 3:1 on
    # a light surface, so identity must not rest on colour alone. Space them so
    # near-equal means (2.9 vs 3.5 ppb here) do not overprint.
    anchors = sorted(((v.mean(), lab, c) for (lab, v), c in zip(series, SERIES_COLORS)),
                     key=lambda t: t[0])
    min_gap = top * 0.075
    placed: list[float] = []
    for mean, _label, color in anchors:
        pos = mean if not placed else max(mean, placed[-1] + min_gap)
        placed.append(pos)
        ax.annotate(f"{mean:.1f} ppb", xy=(x_end, mean), xytext=(x_end + top * 0.02, pos),
                    va="center", ha="left", fontsize=9.5, color=color, weight="bold",
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.5,
                                    shrinkA=0, shrinkB=2))

    ax.set_xlim(minutes[0], x_pad)
    ax.set_ylim(0, top)
    ax.set_xlabel("minutes into sampling window  (2020-03-31 18:30–21:30 UTC)",
                  fontsize=10.5, color=TEXT_SECONDARY)
    ax.set_ylabel("CH₄ enhancement  (ppb)", fontsize=10.5, color=TEXT_SECONDARY)
    ax.set_title("Rice paddy CH₄ at July peak flux: open-path beam vs point sensors",
                 fontsize=14.5, color=TEXT_PRIMARY, pad=46, loc="left")
    ax.text(0, 1.075,
            "MicroHH LES, convective hour (H = +205 W m⁻², w* = 1.68 m s⁻¹) · "
            "300×300 m paddy at 11.7 kg day⁻¹",
            transform=ax.transAxes, fontsize=10.5, color=TEXT_SECONDARY)
    ax.text(0, 1.022,
            "Every sensor sits below the instrument's own 1σ random error — "
            "none of these signals is measurable.",
            transform=ax.transAxes, fontsize=10.5, color=TEXT_SECONDARY)

    ax.grid(axis="y", color=TEXT_MUTED, alpha=0.22, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9.5)

    ax.legend(loc="upper right", frameon=False, fontsize=10, ncol=2,
              labelcolor=TEXT_SECONDARY)

    out = args.out or args.root / f"open_path_timeseries_{args.season}.png"
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")

    print(f"\n{'sensor':<32}{'mean':>9}{'max':>9}{'CV':>7}  vs 1σ noise")
    for label, v in series:
        print(f"  {label:<30}{v.mean():>9.2f}{v.max():>9.2f}"
              f"{v.std()/v.mean():>7.2f}  {v.mean()/noise_ppb:>5.2f}×")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
