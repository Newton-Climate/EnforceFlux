"""Animate the paddy CH4 plume at beam height, and plot the sensor timeseries.

Two products from the surface-flux LES run:

1. ``paddy_plume_0p5m.gif`` — the 0.5 m concentration field over 1 h, from the
   60 s cross-sections (91 frames). The paddy footprint is outlined so the
   plume can be read against its source.

2. ``paddy_sensor_timeseries.png`` — six sensors:
     * open-path beams across the field at 0.5, 1.0 and 2.0 m
     * point sensors at the field centre, the downwind bund, and 200 m beyond

   The two families come from different outputs and therefore different
   cadences, which the plot states rather than hides: beams need a horizontal
   FIELD at each height, which only the 3-D dump provides (300 s), while point
   sensors come from column output carrying the full profile at 60 s.

Usage:
    python examples/microhh_paddy_animation.py
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUN = REPO / "runs" / "sacramento_paddy_surface_flux" / "paddy"

TO_PPB = (28.9647 / 16.043) * 1e9
BEAM_HEIGHTS_M = (0.5, 1.0, 2.0)
BEAM_LENGTH_M = 300.0
POINT_HEIGHT_M = 1.0

# Palette slots 1-6, fixed order (validated for adjacent-pair CVD separation).
OP_COLORS = ("#2a78d6", "#1baf7a", "#4a3aa7")        # blue, aqua, violet
PT_COLORS = ("#eb6834", "#eda100", "#e34948")        # orange, yellow, red
TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED = "#0b0b0b", "#52514e", "#84837c"
SURFACE = "#fcfcfb"


def load_cfg(run: Path):
    from enforceflux.microhh import load_microhh_config
    return load_microhh_config(run / "microhh_generated.yaml")


def load_xy(cfg):
    """60 s cross-sections at the first model level -> (t, field[t, y, x]) in ppb."""
    g = cfg.grid
    times, fields = [], []
    for path in sorted(Path(cfg.case_dir).glob(f"{cfg.scalar_name}.xy.*")):
        stamp = path.suffix.lstrip(".")
        if len(stamp) != 7 or not stamp.isdigit():
            continue
        arr = np.fromfile(path, dtype=np.float64)
        if arr.size != g.itot * g.jtot:
            continue
        fields.append(arr.reshape(g.jtot, g.itot))
        times.append(int(stamp))
    order = np.argsort(times)
    return (np.asarray(times, float)[order],
            np.stack([fields[i] for i in order]) * TO_PPB)


def footprint_mask(cfg):
    g = cfg.grid
    bot = np.fromfile(Path(cfg.case_dir) / f"{cfg.scalar_name}_bot_in.0000000",
                      dtype=np.float64).reshape(g.jtot, g.itot)
    return bot > 0


def make_animation(cfg, out_path: Path, fps: int = 8):
    from enforceflux.analysis.visualization_simulation import create_simulation_movie

    g = cfg.grid
    times, fields = load_xy(cfg)
    keep = times >= cfg.spinup_s
    times, fields = times[keep], fields[keep]

    xax = (np.arange(g.itot) + 0.5) * g.dx
    yax = (np.arange(g.jtot) + 0.5) * g.dy
    mask = footprint_mask(cfg)
    js, is_ = np.where(mask)
    x0, x1 = xax[is_.min()], xax[is_.max()]
    y0, y1 = yax[js.min()], yax[js.max()]

    def overlay(ax):
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor="white", lw=1.4, ls="--", alpha=0.9))
        ax.text(x1, y1 + 40, "paddy footprint", color="white", fontsize=8,
                ha="right", va="bottom")
        # Crop to the region that carries signal. At full domain width the
        # plume is a small bright patch in a mostly black frame, which reads
        # as empty at slide size.
        ax.set_xlim(x0 - 400, min(xax[-1], x1 + 1500))
        ax.set_ylim(y0 - 450, min(yax[-1], y1 + 450))

    labels = [f"t = {(t - cfg.spinup_s)/60:5.0f} min" for t in times]
    z0 = g.levels()[0]
    return create_simulation_movie(
        fields, output_path=out_path, lons=xax, lats=yax, time_labels=labels,
        fps=fps, cmap="magma", units="CH4 enhancement (ppb)",
        title_prefix=f"Rice paddy plume at z = {z0:.2f} m",
        xlabel="downwind distance x (m, box frame)",
        ylabel="crosswind distance y (m)",
        figsize=(8.0, 4.6), overlay=overlay,
    )


def beam_series(cfg):
    """OP path averages across the paddy at each beam height, from the 3-D dump."""
    import xarray as xr

    from enforceflux.instrument import path_average_series

    g = cfg.grid
    z = g.levels()
    ds = xr.open_dataset(RUN / "ch4_4d.nc")
    fld = ds[cfg.scalar_name].where(ds.time >= cfg.spinup_s, drop=True)
    t = (fld.time.values - cfg.spinup_s) / 60.0
    xax = (np.arange(g.itot) + 0.5) * g.dx
    yax = (np.arange(g.jtot) + 0.5) * g.dy

    out = []
    for h in BEAM_HEIGHTS_M:
        k = int(np.searchsorted(z, h))
        if k == 0 or k >= z.size:
            k = min(max(k, 0), z.size - 1)
            series = path_average_series(fld.isel(z=k).values * TO_PPB, xax, yax,
                                         cfg.source_x0, cfg.source_y0,
                                         BEAM_LENGTH_M, 0.0, centred=True)
        else:
            # Log-interpolate between the bracketing levels: the surface-layer
            # profile is logarithmic, so linear interpolation would bias low.
            f = (np.log(h) - np.log(z[k - 1])) / (np.log(z[k]) - np.log(z[k - 1]))
            lo = path_average_series(fld.isel(z=k - 1).values * TO_PPB, xax, yax,
                                     cfg.source_x0, cfg.source_y0,
                                     BEAM_LENGTH_M, 0.0, centred=True)
            hi = path_average_series(fld.isel(z=k).values * TO_PPB, xax, yax,
                                     cfg.source_x0, cfg.source_y0,
                                     BEAM_LENGTH_M, 0.0, centred=True)
            series = lo + f * (hi - lo)
        out.append((f"OP beam, {h:g} m", t, series))
    return out


def log_interp_height(profile: np.ndarray, z: np.ndarray, target: float) -> np.ndarray:
    """Interpolate a (time, z) profile to `target` height, logarithmically in z.

    The surface layer varies as ln(z), so linear interpolation between model
    levels biases low — by ~8% between the 0.50 m and 1.52 m levels here. Beams
    and point sensors both go through this, so the two are compared at genuinely
    the same height rather than at whichever level happened to be nearest.
    """
    k = int(np.searchsorted(z, target))
    if k <= 0:
        return profile[:, 0]
    if k >= z.size:
        return profile[:, -1]
    f = (np.log(target) - np.log(z[k - 1])) / (np.log(z[k]) - np.log(z[k - 1]))
    return profile[:, k - 1] + f * (profile[:, k] - profile[:, k - 1])


def point_series(cfg):
    """Point sensors from column output (full profile, 60 s cadence)."""
    import xarray as xr

    z = cfg.grid.levels()
    wanted = [("centre", "field centre"),
              ("edge_down", "downwind bund"),
              ("fetch_350", "200 m beyond")]
    from enforceflux.microhh.geometry import BoxProjection
    proj = BoxProjection(origin_lon=cfg.origin_lon, origin_lat=cfg.origin_lat,
                         x_bearing_deg=cfg.x_bearing_deg,
                         source_x0=cfg.source_x0, source_y0=cfg.source_y0)
    by_id = {r.id: r for r in cfg.receptors}

    out = []
    for rid, label in wanted:
        r = by_id[rid]
        bx, by = proj.to_box(r.lon, r.lat)
        ix, iy = int(round(bx) / cfg.grid.dx), int(round(by) / cfg.grid.dy)
        matches = glob.glob(str(Path(cfg.case_dir) /
                                f"{cfg.case_name}.column.{ix:05d}.{iy:05d}.*.nc"))
        if not matches:
            continue
        col = xr.open_dataset(matches[0], decode_times=False)
        sel = col.time.values >= cfg.spinup_s
        t = (col.time.values[sel] - cfg.spinup_s) / 60.0
        prof = col[cfg.scalar_name].values[sel, :] * TO_PPB
        out.append((f"Point, {label} ({POINT_HEIGHT_M:g} m)", t,
                    log_interp_height(prof, z, POINT_HEIGHT_M)))
    return out


def make_timeseries(cfg, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from enforceflux.instrument import INSTRUMENT_DB

    beams, points = beam_series(cfg), point_series(cfg)
    params = INSTRUMENT_DB["OP"]["good"]
    noise, limit = params.sigma_abs * 1000, params.detection_limit * 1000

    fig, ax = plt.subplots(figsize=(11.5, 6.0), dpi=200)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

    ax.axhspan(0, noise, color=TEXT_MUTED, alpha=0.11, lw=0, zorder=0)
    ax.axhline(noise, color=TEXT_SECONDARY, lw=1.2, alpha=0.6, zorder=1)
    ax.axhline(limit, color=TEXT_SECONDARY, lw=1.2, ls=(0, (5, 3)), alpha=0.6, zorder=1)

    from matplotlib.lines import Line2D
    ref = ax.legend(handles=[
        Line2D([], [], color=TEXT_SECONDARY, lw=1.2, alpha=0.6,
               label=f"OP random error  1σ = {noise:.0f} ppb"),
        Line2D([], [], color=TEXT_SECONDARY, lw=1.2, alpha=0.6, ls=(0, (5, 3)),
               label=f"detection limit = {limit:.0f} ppb")],
        loc="upper left", frameon=False, fontsize=9,
        labelcolor=TEXT_SECONDARY, handlelength=2.4, borderaxespad=0.6)
    ax.add_artist(ref)

    for (label, t, v), c in zip(beams, OP_COLORS):
        ax.plot(t, v, color=c, lw=2.0, marker="o", ms=4.5, label=label, zorder=4)
    for (label, t, v), c in zip(points, PT_COLORS):
        ax.plot(t, v, color=c, lw=1.3, alpha=0.85, label=label, zorder=3)

    top = max(max(v.max() for _, _, v in beams),
              max(v.max() for _, _, v in points), noise) * 1.30
    ax.set_ylim(0, top)
    ax.set_xlim(0, max(t.max() for _, t, _ in points))
    ax.set_xlabel("minutes into sampling window  (2020-03-31 18:30–19:30 UTC)",
                  fontsize=10.5, color=TEXT_SECONDARY)
    ax.set_ylabel("CH₄ enhancement  (ppb)", fontsize=10.5, color=TEXT_SECONDARY)
    ax.set_title("Rice paddy at July flux: open-path beams vs point sensors",
                 fontsize=14.5, color=TEXT_PRIMARY, pad=44, loc="left")
    ax.text(0, 1.070, "Surface-flux LES, 1 m first cell — the near-surface "
            "gradient is resolved, not extrapolated",
            transform=ax.transAxes, fontsize=10.5, color=TEXT_SECONDARY)
    ax.text(0, 1.020, "Beams (markers, 300 s dumps) carry the whole field at "
            "each height; points (lines, 60 s columns) are single locations",
            transform=ax.transAxes, fontsize=10.5, color=TEXT_SECONDARY)

    ax.grid(axis="y", color=TEXT_MUTED, alpha=0.22, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_MUTED); ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9.5)
    ax.legend(loc="upper right", frameon=False, fontsize=9.5, ncol=2,
              labelcolor=TEXT_SECONDARY)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    return out_path, beams, points


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--skip-animation", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.run)
    if not args.skip_animation:
        gif = make_animation(cfg, args.run.parent / "paddy_plume_0p5m.gif", args.fps)
        print(f"animation  -> {gif}")

    png, beams, points = make_timeseries(
        cfg, args.run.parent / "paddy_sensor_timeseries.png")
    print(f"timeseries -> {png}\n")
    print(f"{'sensor':<32}{'mean ppb':>10}{'sd':>8}{'n':>5}")
    for label, _t, v in beams + points:
        print(f"  {label:<30}{v.mean():>10.2f}{v.std():>8.2f}{v.size:>5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ─── Cross-beam vertical section ─────────────────────────────────────────────


def make_cross_section_animation(cfg, out_path: Path, fps: int = 3,
                                 z_top_m: float = 40.0):
    """Animate the y-z plane through the paddy: the observer's view of the beam.

    Horizontal axis is crosswind distance (the beam runs left-right across the
    frame); vertical axis is height. This is the plane an observer sees standing
    perpendicular to a crosswind beam.

    Deliberately NOT drawn with imshow. The vertical grid is stretched (1 m at
    the ground growing to 32 m), and imshow assumes uniform spacing, so it would
    silently stretch the near-surface cells to look as tall as the ones aloft —
    exactly the region the whole run exists to resolve. pcolormesh with the true
    cell edges keeps the geometry honest.

    Only the 300 s [dump] output carries every height, so this has 13 frames
    against the plan view's 61: `[cross] yz` was never requested for this run.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xarray as xr
    from matplotlib.animation import FuncAnimation

    g = cfg.grid
    z = g.levels()
    ds = xr.open_dataset(RUN / "ch4_4d.nc")
    fld = ds[cfg.scalar_name].where(ds.time >= cfg.spinup_s, drop=True)
    times = fld.time.values

    ix = int(round(cfg.source_x0 / g.dx))          # slice through the field centre
    kmax = int(np.searchsorted(z, z_top_m)) + 1
    data = fld.isel(x=ix, z=slice(0, kmax)).values * TO_PPB     # (t, z, y)

    # True cell edges: zh midway between levels, y uniform.
    # Lowest edge nudged off zero so the height axis can be logarithmic. That
    # matters here: on a linear 0-40 m axis the 0.5-2 m beams occupy the bottom
    # 5% of the frame and cannot be told apart, which defeats the figure.
    zh = np.concatenate(([0.05], 0.5 * (z[:kmax - 1] + z[1:kmax]), [z[kmax - 1]]))
    yh = np.arange(g.jtot + 1) * g.dy

    mask = footprint_mask(cfg)
    ys = np.where(mask[:, ix])[0]
    y_lo, y_hi = (yh[ys.min()], yh[ys.max() + 1]) if ys.size else (None, None)

    fig, ax = plt.subplots(figsize=(9.0, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    mesh = ax.pcolormesh(yh, zh, data[0], cmap="magma",
                         vmin=0.0, vmax=float(np.percentile(data, 99.5)))
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label("CH₄ enhancement (ppb)")

    # Zoom to the field and its immediate surroundings: the rest is empty.
    pad = 350.0
    x_lo = max(yh[0], (y_lo or 0) - pad)
    x_hi = min(yh[-1], (y_hi or yh[-1]) + pad)

    for h, c in zip(BEAM_HEIGHTS_M, OP_COLORS):
        ax.axhline(h, color=c, lw=1.6, ls="--", alpha=0.95)
        ax.text(x_lo + 12, h * 1.06, f"{h:g} m beam", color=c, fontsize=9,
                va="bottom", ha="left", weight="bold")
    if y_lo is not None:
        ax.plot([y_lo, y_hi], [0.055, 0.055], color="#02C39A", lw=5,
                solid_capstyle="butt")
        ax.text((y_lo + y_hi) / 2, 0.075, "paddy", color="#02C39A",
                fontsize=9.5, ha="center", va="bottom", weight="bold")

    ax.set_yscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0.05, z_top_m)
    ax.set_yticks([0.1, 0.5, 1, 2, 5, 10, 20, 40])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("crosswind distance y (m) — the beam runs along this axis")
    ax.set_ylabel("height z (m, log scale)")

    def _title(i):
        return (f"Cross-beam section through the paddy  "
                f"(t = {(times[i]-cfg.spinup_s)/60:.0f} min)")

    ax.set_title(_title(0))

    def _update(i):
        mesh.set_array(data[i].ravel())
        ax.set_title(_title(i))
        return (mesh,)

    anim = FuncAnimation(fig, _update, frames=len(times), blit=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer="pillow", fps=fps)
    plt.close(fig)
    return out_path, data, z[:kmax]
