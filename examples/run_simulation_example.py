"""End-to-end example: forward FLEXPART CH4 simulation using bundled test met data.

Usage (from repo root):
    python examples/run_simulation_example.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Make sure the repo is importable when run directly.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enforceflux.flexpart import FlexpartSimulation


CONFIG = Path(__file__).parent / "simulation_test.yaml"
REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    print("=" * 60)
    print("EnforceFlux — FLEXPART CH4 simulation example")
    print("=" * 60)

    # ── Check prerequisites ───────────────────────────────────────
    _check_met_data()

    # ── Build simulation from YAML ────────────────────────────────
    print(f"\nLoading config: {CONFIG.relative_to(REPO_ROOT)}")
    sim = FlexpartSimulation.from_yaml(CONFIG)
    cfg = sim.config

    print(f"  Period      : {cfg.start.strftime('%Y-%m-%d %H:%M')} → {cfg.end.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  Domain      : lon [{cfg.domain_lon_min}, {cfg.domain_lon_max}]  lat [{cfg.domain_lat_min}, {cfg.domain_lat_max}]")
    print(f"  Resolution  : {cfg.domain_dx}° × {cfg.domain_dy}°")
    print(f"  Sources     : {len(cfg.sources)}")
    for src in cfg.sources:
        kind = type(src).__name__
        print(f"    [{kind}] {src.id}")

    # ── Prepare inputs (dry-run first to inspect) ─────────────────
    print("\nPreparing FLEXPART input files …")
    run_dir = sim.prepare()
    _print_releases_summary(run_dir / "options" / "RELEASES")

    # ── Execute FLEXPART ──────────────────────────────────────────
    print("\nRunning FLEXPART …")
    output_nc = sim.run()
    print(f"\nOutput written → {output_nc.relative_to(REPO_ROOT)}")

    # ── Summarise results ─────────────────────────────────────────
    _summarise_output(output_nc)


def _check_met_data() -> None:
    meteo_dir = REPO_ROOT / "flexpart" / "tests" / "testdata"
    needed = ["EC2009010100", "EC2009010106"]
    missing = [f for f in needed if not (meteo_dir / f).exists()]
    if missing:
        print("\nMissing test meteorological files:", missing)
        print("Download them with:")
        print(f"  mkdir -p {meteo_dir.relative_to(REPO_ROOT)}")
        print("  wget -r -nH --cut-dirs=2 --no-parent \\")
        print(f"       --accept=\"{','.join(missing)}\" \\")
        print("       -P flexpart/tests/testdata/ \\")
        print("       https://webdata.wolke.img.univie.ac.at/flexpart/")
        raise SystemExit(1)
    print(f"Met files OK  : {', '.join(needed)}")


def _print_releases_summary(releases_path: Path) -> None:
    text = releases_path.read_text()
    n_releases = text.count("&RELEASE\n")
    print(f"  RELEASES file: {n_releases} release block(s)")
    # Pull MASS values to show total CH4
    import re
    masses = [float(m) for m in re.findall(r"MASS\s*=\s*([\d.E+\-]+)", text)]
    if masses:
        print(f"  Total mass   : {sum(masses):.4g} kg  ({len(masses)} blocks)")


def _summarise_output(nc_path: Path) -> None:
    try:
        from netCDF4 import Dataset
    except ImportError:
        print("(netCDF4 not available — skipping output summary)")
        return

    print("\n── Output summary ──────────────────────────────────────")
    with Dataset(nc_path) as ds:
        print(f"  File        : {nc_path.name}  ({nc_path.stat().st_size / 1e6:.1f} MB)")
        print(f"  Dimensions  : {dict(ds.dimensions)}")

        conc_var = None
        for name in ("ch4_concentration", "spec001_mr", "spec001"):
            if name in ds.variables:
                conc_var = name
                break

        if conc_var:
            data = np.asarray(ds.variables[conc_var][:])
            units = getattr(ds.variables[conc_var], "units", "?")
            print(f"  Variable    : {conc_var}  [{units}]")
            print(f"  Shape       : {data.shape}")
            nonzero = data[data > 0]
            if nonzero.size > 0:
                print(f"  Non-zero    : {nonzero.size} grid cells")
                print(f"  Max conc    : {nonzero.max():.4g} {units}")
                print(f"  Mean (non-zero): {nonzero.mean():.4g} {units}")
            else:
                print("  All values zero — particles may not have reached output grid in 3 h")

        print(f"\n  Global attrs:")
        for attr in ("title", "species", "simulation_start", "simulation_end", "source_ids"):
            val = getattr(ds, attr, None)
            if val:
                print(f"    {attr}: {val}")

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
