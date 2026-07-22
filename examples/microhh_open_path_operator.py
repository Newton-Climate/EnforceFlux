"""Open-path (OP) sensor over a turbulent LES field, vs point and area averages.

Answers three questions against the convective Sacramento LES:

1. Does the OP measurement equal a spatial average?
   Along its own beam, yes by definition: OP = (1/L) * int c ds. But that is a
   1-D line average. Averaging any OTHER region — a 2-D patch, a model grid
   cell, a footprint — is a different operator and gives a different number,
   except in the degenerate case where the field does not vary across the extra
   dimensions being averaged.

2. Does it differ from taking the mean of the concentration fields?
   Not in the mean: path-averaging is linear, so it commutes with time
   averaging. It differs in every higher moment, because the beam is a spatial
   low-pass filter — and it is those moments that set detection.

3. What happens at the July peak paddy flux?
   The paddy field is rescaled by the July/April flux ratio. The scalar is
   passive and the source term linear, so rescaling the field is exact — no
   re-run needed.

The path integral itself now lives in the library
(``enforceflux.instrument.open_path``); this script only drives it.

Usage:
    python examples/microhh_open_path_operator.py
    python examples/microhh_open_path_operator.py --paddy-season april
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "runs" / "sacramento_convective_compare"

# Dry-air CH4 mass mixing ratio (kg/kg) -> mole fraction in ppm.
M_AIR, M_CH4 = 28.9647, 16.043
KGKG_TO_PPM = (M_AIR / M_CH4) * 1e6

# Rice-paddy fluxes from the FLEXPART configs, kg m-2 s-1.
PADDY_FLUX = {"april": 2.0e-10,   # examples/sacval_april_2020.yaml, pre-season
              "july": 1.5e-9}     # examples/sacval_july_2020.yaml, peak season
PADDY_RUN_SEASON = "april"        # what the LES was actually run with

# The headline beam shoots ACROSS the paddy: centred on the field, spanning its
# full 300 m width, crosswind. That is the deployment a grower would actually
# use — the beam interrogates the field's own emissions rather than a plume
# some distance downwind. Note it also crosses the point leak in the leak run,
# so the two remain directly comparable.
PADDY_SIDE_M = 300.0
BEAM_FETCH_M = 0.0                # 0 = across the source itself
BEAM_LENGTH_M = PADDY_SIDE_M
DOWNWIND_M = (0.0, 250.0, 500.0, 1000.0)
PATH_LENGTHS_M = (50.0, 100.0, 200.0, 300.0, 600.0)
CROSSWIND_BEARING_DEG = 0.0       # box +y; the plume runs along box +x


def load_xy_slices(case_dir: Path, scalar: str, itot: int, jtot: int):
    """Read MicroHH's [cross] xy plane series -> (time_s, field[t, y, x])."""
    times, fields = [], []
    for path in sorted(case_dir.glob(f"{scalar}.xy.*")):
        stamp = path.suffix.lstrip(".")
        if len(stamp) != 7 or not stamp.isdigit():
            continue
        arr = np.fromfile(path, dtype=np.float64)
        if arr.size != itot * jtot:
            raise ValueError(f"{path.name}: {arr.size} values, expected {itot*jtot}")
        fields.append(arr.reshape(jtot, itot))
        times.append(int(stamp))
    order = np.argsort(times)
    return np.asarray(times, float)[order], np.stack([fields[i] for i in order])


def moments(x: np.ndarray) -> dict:
    mu, sd = float(x.mean()), float(x.std())
    z = (x - mu) / sd if sd > 0 else np.zeros_like(x)
    return {"mean": mu, "std": sd, "cv": sd / mu if mu > 0 else np.nan,
            "skew": float((z**3).mean()), "kurt": float((z**4).mean()),
            "p95_over_mean": float(np.percentile(x, 95) / mu) if mu > 0 else np.nan}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--paddy-season", choices=sorted(PADDY_FLUX), default="july")
    ap.add_argument("--fetch", type=float, default=BEAM_FETCH_M,
                    help="beam distance downwind of the source centre (m)")
    ap.add_argument("--beam", type=float, default=BEAM_LENGTH_M,
                    help="beam length (m); default spans the paddy")
    args = ap.parse_args()

    from enforceflux.instrument import (
        INSTRUMENT_DB, path_average, path_average_series, simulate_open_path,
    )
    from enforceflux.microhh import load_microhh_config

    variants = ("leak", "paddy")
    cfgs = {v: load_microhh_config(args.root / v / "microhh_generated.yaml")
            for v in variants}
    g = cfgs["leak"].grid
    xax = (np.arange(g.itot) + 0.5) * g.dx
    yax = (np.arange(g.jtot) + 0.5) * g.dy

    scale = PADDY_FLUX[args.paddy_season] / PADDY_FLUX[PADDY_RUN_SEASON]
    print("Open-path operator over a turbulent LES field")
    print(f"paddy season: {args.paddy_season} "
          f"(flux {PADDY_FLUX[args.paddy_season]:.1e} kg m-2 s-1, "
          f"x{scale:g} the simulated {PADDY_RUN_SEASON} flux)")
    params = INSTRUMENT_DB["OP"]["good"]
    detect = params.detection_limit
    print(f"beam: {args.beam:.0f} m crosswind at {args.fetch:.0f} m fetch "
          f"({'across the source' if args.fetch == 0 else 'downwind'})")
    print(f"OP: detection limit {detect*1000:.0f} ppb, noise {params.sigma_abs*1000:.0f} ppb"
          f"  ({detect/params.sigma_abs:.1f} sigma)\n")

    data = {}
    for v in variants:
        cfg = cfgs[v]
        t, f = load_xy_slices(cfg.case_dir, cfg.scalar_name, g.itot, g.jtot)
        keep = t >= cfg.spinup_s
        # Passive scalar + linear source => rescaling the field is exact.
        factor = KGKG_TO_PPM * (scale if v == "paddy" else 1.0)
        data[v] = (t[keep], f[keep] * factor)
        total = sum(s.emission_rate_kg_s for s in cfg.sources)
        total *= scale if v == "paddy" else 1.0
        print(f"{v:>6}: {keep.sum()} samples @ {int(t[1]-t[0])} s, "
              f"emission {total:.3e} kg/s ({total*86400:.2f} kg/day)")

    x_at = lambda d: cfgs["leak"].source_x0 + d          # noqa: E731
    yc = cfgs["leak"].source_y0

    # ---- 1. Line vs area vs point: are these the same "spatial average"? ----
    print("\n" + "=" * 78)
    print("1. IS 'SPATIAL AVERAGING' THE SAME AS THE OP MEASUREMENT?")
    print("=" * 78)
    print("   crosswind beam on the TIME-MEAN field")
    for v in variants:
        _, f = data[v]
        mean_field = f.mean(axis=0)
        line = path_average(mean_field, xax, yax, x_at(args.fetch), yc,
                            args.beam, CROSSWIND_BEARING_DEG, centred=True)
        # Same 200 m extent, but as a square patch instead of a line.
        h = args.beam / 2
        jm = (yax >= yc - h) & (yax < yc + h)
        im = (xax >= x_at(args.fetch) - h) & (xax < x_at(args.fetch) + h)
        area = float(mean_field[np.ix_(jm, im)].mean())
        point = path_average(mean_field, xax, yax, x_at(args.fetch), yc, 0.0, 0.0)
        print(f"\n  {v}")
        print(f"    line  (OP, 1-D beam)        = {line:.6e} ppm")
        print(f"    area  ({args.beam:.0f}x{args.beam:.0f} m patch)   = {area:.6e} ppm  "
              f"({100*(area/line-1):+.1f}% vs OP)")
        print(f"    point (single location)     = {point:.6e} ppm  "
              f"({100*(point/line-1):+.1f}% vs OP)")

    # ---- 2. Linearity: OP mean == mean of the fields ----
    print("\n" + "=" * 78)
    print("2. OP MEAN vs MEAN OF THE CONCENTRATION FIELDS")
    print("=" * 78)
    for v in variants:
        _, f = data[v]
        args_geom = (xax, yax, x_at(args.fetch), yc, args.beam, CROSSWIND_BEARING_DEG)
        A = path_average_series(f, *args_geom, centred=True).mean()
        B = path_average(f.mean(axis=0), *args_geom, centred=True)
        print(f"  {v:<6} time-mean of OP(t) = {A:.6e}   "
              f"OP of time-mean field = {B:.6e}   |A-B|/A = {abs(A-B)/A:.1e}")

    # ---- 3. What path length actually changes ----
    print("\n" + "=" * 78)
    print("3. WHAT THE BEAM CHANGES: FLUCTUATIONS, NOT THE MEAN")
    print("=" * 78)
    for v in variants:
        _, f = data[v]
        print(f"\n  {v}")
        print(f"    {'sensor':<14}{'mean ppm':>12}{'CV':>7}{'skew':>7}"
              f"{'kurt':>7}{'p95/mean':>10}{'detect%':>9}")
        for label, L in [("point", 0.0)] + [(f"OP {int(L)} m", L) for L in PATH_LENGTHS_M]:
            s = path_average_series(f, xax, yax, x_at(args.fetch), yc, L,
                                    CROSSWIND_BEARING_DEG, centred=True)
            m = moments(s)
            det = 100.0 * (s > detect).mean()
            print(f"    {label:<14}{m['mean']:>12.4e}{m['cv']:>7.2f}{m['skew']:>7.2f}"
                  f"{m['kurt']:>7.1f}{m['p95_over_mean']:>10.2f}{det:>9.1f}")

    # ---- 4. Through the real instrument ----
    print("\n" + "=" * 78)
    print("4. THROUGH THE OP INSTRUMENT MODEL (noise, dropouts, detection limit)")
    print("=" * 78)
    out = {}
    for v in variants:
        t, f = data[v]
        s = simulate_open_path(
            f, xax, yax, times_s=t, centred=True, rng=np.random.default_rng(42),
            x0=x_at(args.fetch), y0=yc, path_length_m=args.beam,
            path_bearing_deg=CROSSWIND_BEARING_DEG, id=f"op_{v}",
        )
        out[v] = s
        snr = s.truth.mean() / params.sigma_abs
        print(f"\n  {v}: true path-avg mean      {s.truth.mean():.4e} ppm  "
              f"(SNR vs {params.sigma_abs*1000:.0f} ppb noise = {snr:.2f})")
        print(f"       mean of DETECTED samples {s.mean_of_detected:.4e} ppm  "
              f"(censoring bias x{s.censoring_bias:.2f})")
        print(f"       non-detects as zero      {s.mean_with_nondetects_as_zero:.4e} ppm")
        print(f"       detected {int(s.valid.sum())}/{s.valid.size} "
              f"({100*s.detected_fraction:.0f}%), peak {s.truth.max():.4e} ppm "
              f"vs limit {detect} ppm")

    np.savez(args.root / f"open_path_{args.paddy_season}.npz",
             **{f"{v}_{k}": a for v, s in out.items()
                for k, a in (("time_s", s.times_s), ("truth_ppm", s.truth),
                             ("obs_ppm", s.observed), ("valid", s.valid))})
    print(f"\nwrote {args.root / f'open_path_{args.paddy_season}.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
