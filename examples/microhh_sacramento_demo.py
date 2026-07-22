"""Single-source MicroHH LES plume demo.

The MicroHH counterpart to gaussian_plume_single_source_demo.py: runs a
reduced-resolution Sacramento case (100 kg/hr point source in a dry convective
boundary layer) end-to-end via the transport-simulation plugin, then renders
the resolved plume (surface + vertical cross-sections) and the receptor time
series so you can validate that the LES output looks physically sensible.

Requires the compiled MicroHH binary (make install-microhh) and the analysis
extra. Run from the repo root:
    python examples/microhh_sacramento_demo.py
    python examples/microhh_sacramento_demo.py --dry-run   # inputs only
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from enforceflux.core.base import ITransportSimulation  # noqa: E402
from enforceflux.microhh import (  # noqa: E402
    load_microhh_config,
    read_cross_xy,
    read_cross_xz,
    read_receptor_series,
)
from enforceflux.microhh.geometry import BoxProjection  # noqa: E402
from enforceflux.utils.plugin_registry import get_plugin  # noqa: E402

CONFIG_YAML = Path(__file__).parent / "microhh_sacramento_demo.yaml"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _plot_plume(cfg, out_png: Path) -> None:
    g = cfg.grid
    proj = BoxProjection(cfg.origin_lon, cfg.origin_lat, cfg.x_bearing_deg,
                         cfg.source_x0, cfg.source_y0)
    xy = read_cross_xy(cfg, k=0)               # (jtot, itot) surface slice
    xz = read_cross_xz(cfg)                     # (ktot, itot) vertical slice
    x = (np.arange(g.itot) + 0.5) * g.dx
    y = (np.arange(g.jtot) + 0.5) * g.dy
    z = (np.arange(g.ktot) + 0.5) * g.dz

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    im1 = ax1.pcolormesh(x, y, np.ma.masked_less(xy, 1e-9), cmap="viridis", shading="auto")
    ax1.plot(cfg.source_x0, cfg.source_y0, "r*", ms=14, label="source")
    for r in cfg.receptors:
        rx, ry = proj.to_box(r.lon, r.lat)
        ax1.plot(rx, ry, "w^", ms=7)
    ax1.set(title="Surface CH4 (final frame)", xlabel="x [m] (downwind)", ylabel="y [m]")
    ax1.set_aspect("equal")
    fig.colorbar(im1, ax=ax1, label="CH4 [kg/kg]")

    im2 = ax2.pcolormesh(x, z, np.ma.masked_less(xz, 1e-9), cmap="viridis", shading="auto")
    ax2.plot(cfg.source_x0, cfg.sources[0].alt_m, "r*", ms=14)
    ax2.set(title="Vertical CH4 (plume centreline)", xlabel="x [m]", ylabel="z [m]",
            ylim=(0, 1200))
    fig.colorbar(im2, ax=ax2, label="CH4 [kg/kg]")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_receptors(cfg, out_png: Path) -> None:
    series = read_receptor_series(cfg, sample_level=0)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for i, rid in enumerate(series.receptor_ids):
        ax.plot(series.times_s / 60.0, series.values[:, i], label=rid)
    ax.set(title="Receptor CH4 time series (~500 m downwind)",
           xlabel="time [min]", ylabel="surface CH4 [kg/kg]")
    ax.legend()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single-source MicroHH LES plume demo.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Write case inputs without running MicroHH.")
    args = parser.parse_args()

    cfg = load_microhh_config(CONFIG_YAML)

    print("=" * 72)
    print("Single-source MicroHH LES plume demo")
    print("=" * 72)
    print(f"Source   : {cfg.sources[0].id}")
    print(f"Grid     : {cfg.grid.itot}x{cfg.grid.jtot}x{cfg.grid.ktot} "
          f"({cfg.grid.dx:g} m), {cfg.grid.itot * cfg.grid.jtot * cfg.grid.ktot:,} cells")
    print(f"Output   : {_display_path(cfg.case_dir)}")

    sim = get_plugin("enforceflux.transport_simulation", "microhh", ITransportSimulation)()
    print("\nRunning MicroHH …" if not args.dry_run else "\nPreparing inputs (dry run) …")
    result = sim.simulate([], None, {"sim_config": str(CONFIG_YAML), "dry_run": args.dry_run})

    print(f"  ini       : {_display_path(Path(result.meta['ini_path']))}")
    print(f"  input.nc  : {_display_path(Path(result.meta['input_nc_path']))}")
    if args.dry_run:
        print("Dry run complete (no MicroHH execution).")
        return

    plume_png = cfg.case_dir / f"{cfg.case_name}_plume.png"
    recep_png = cfg.case_dir / f"{cfg.case_name}_receptors.png"
    _plot_plume(cfg, plume_png)
    _plot_receptors(cfg, recep_png)
    print(f"\nWrote plume  : {_display_path(plume_png)}")
    print(f"Wrote series : {_display_path(recep_png)}")

    xy = read_cross_xy(cfg, k=0)
    print(f"Validation   : surface-CH4 peak={np.nanmax(xy):.3g} kg/kg, "
          f"nonzero cells={int((xy > 1e-9).sum())}/{xy.size}")
    print("Done.")


if __name__ == "__main__":
    main()
