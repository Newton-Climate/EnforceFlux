#!/usr/bin/env python3
"""Derive the bLS met interval from a MicroHH donor run's column output.

Reproduces, as code, the by-hand procedure recorded in
configs/hetero_rice_paddy_test/inversions/README.md:

* u* and Obukhov L from MicroHH's SGS-inclusive column diagnostics
  (``ustar``, ``obuk``), averaged over the post-spinup window. The resolved
  2 m covariance alone collapses near an LES wall, so it is not used.
* Wind direction and mean speed from the resolved u, v interpolated to
  ``z_ref`` in the 10 Hz column records.
* sigma_u, sigma_v, sigma_w = 2.5, 2.0, 1.25 x u* (bLS surface-layer ratios),
  because resolved variances also collapse near the wall.

Conventions that the README leaves implicit (vector vs scalar mean speed,
arithmetic vs harmonic mean of L) are all reported; ``--check-1km`` selects
the ones that reproduce the published 1 km interval.

    python derive_bls_met.py CASE_DIR --out bls_met.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# The interval written into configs/hetero_rice_paddy_test/inversions/*.
PUBLISHED_1KM = {"u_star": 0.2941302128367419, "L": -28.74228288678913,
                 "wind_dir_deg": 35.02103178152356, "wind_speed": 1.7509097854801514}
SD_RATIOS = {"sd_u": 2.5, "sd_v": 2.0, "sd_w": 1.25}


def _at_height(field: np.ndarray, z: np.ndarray, z_ref: float) -> np.ndarray:
    return np.array([np.interp(z_ref, z, row) for row in field])


def derive(case_dir: Path, t0: float, t1: float, z_ref: float,
           x_bearing_deg: float) -> dict:
    files = sorted(glob.glob(str(case_dir / "*.column.*.nc")))
    if not files:
        raise FileNotFoundError(f"no column NetCDF files in {case_dir}")
    us, obs, east, north = [], [], [], []
    for f in files:
        with xr.open_dataset(f, decode_times=False) as ds:
            t = np.asarray(ds["time"].values, float)
            m = (t >= t0) & (t <= t1)
            if not m.any():
                continue
            z = np.asarray(ds["z"].values, float)
            u = _at_height(np.asarray(ds["u"].values, float)[m], z, z_ref)
            v = _at_height(np.asarray(ds["v"].values, float)[m], z, z_ref)
            us.append(np.asarray(ds["ustar"].values, float)[m])
            obs.append(np.asarray(ds["obuk"].values, float)[m])
        # Box x points along x_bearing (clockwise from north); box y is 90 deg
        # counter-clockwise of it.
        xb = np.deg2rad(x_bearing_deg)
        yb = xb - np.pi / 2
        east.append(u * np.sin(xb) + v * np.sin(yb))
        north.append(u * np.cos(xb) + v * np.cos(yb))
    if not us:
        raise ValueError(f"no column records in [{t0}, {t1}] s in {case_dir}")
    us, obs = np.concatenate(us), np.concatenate(obs)
    east, north = np.concatenate(east), np.concatenate(north)

    toward = float(np.rad2deg(np.arctan2(east.mean(), north.mean())) % 360.0)
    u_star = float(us.mean())
    variants = {
        "L_arithmetic": float(obs.mean()),
        "L_harmonic": float(1.0 / np.mean(1.0 / obs)),
        "wind_speed_vector_mean": float(np.hypot(east.mean(), north.mean())),
        "wind_speed_scalar_mean": float(np.hypot(east, north).mean()),
    }
    return {
        "case_dir": str(case_dir), "window_s": [t0, t1], "z_ref": z_ref,
        "n_columns": len(files), "n_records": int(us.size),
        "u_star": u_star, "toward_bearing_deg": toward,
        "wind_dir_deg": (toward + 180.0) % 360.0,
        "variants": variants,
    }


def interval(d: dict, L_key: str, speed_key: str, z0: float) -> dict:
    u = d["u_star"]
    return {"id": "les0001", "u_star": u, "L": d["variants"][L_key], "z0": z0,
            "wind_dir_deg": d["wind_dir_deg"],
            "wind_speed": d["variants"][speed_key], "z_ref": d["z_ref"],
            **{k: r * u for k, r in SD_RATIOS.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case_dir", type=Path)
    ap.add_argument("--t0", type=float, default=2700.0, help="window start, s")
    ap.add_argument("--t1", type=float, default=5400.0, help="window end, s")
    ap.add_argument("--z-ref", type=float, default=2.0)
    ap.add_argument("--z0", type=float, default=0.1)
    ap.add_argument("--x-bearing", type=float, default=90.0)
    ap.add_argument("--L-key", default="L_arithmetic")
    # Defaults verified with --check-1km on the 1 km donor: arithmetic L and
    # vector-mean speed reproduce the published interval to ~1e-4; harmonic L
    # (-15%) and scalar-mean speed (+5%) do not.
    ap.add_argument("--speed-key", default="wind_speed_vector_mean")
    ap.add_argument("--check-1km", action="store_true",
                    help="compare every variant with the published 1 km interval")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    d = derive(a.case_dir, a.t0, a.t1, a.z_ref, a.x_bearing)
    print(json.dumps({k: v for k, v in d.items() if k != "variants"}, indent=2))
    print("variants:", json.dumps(d["variants"], indent=2))
    if a.check_1km:
        print("\nagainst the published 1 km interval (relative difference):")
        print(f"  u_star        {d['u_star'] / PUBLISHED_1KM['u_star'] - 1:+.2e}")
        print(f"  wind_dir_deg  {d['wind_dir_deg'] - PUBLISHED_1KM['wind_dir_deg']:+.4f} deg")
        for k in ("L_arithmetic", "L_harmonic"):
            print(f"  {k:22s}{d['variants'][k] / PUBLISHED_1KM['L'] - 1:+.2e}")
        for k in ("wind_speed_vector_mean", "wind_speed_scalar_mean"):
            print(f"  {k:22s}{d['variants'][k] / PUBLISHED_1KM['wind_speed'] - 1:+.2e}")
    if a.out:
        out = {"interval": interval(d, a.L_key, a.speed_key, a.z0),
               "toward_bearing_deg": d["toward_bearing_deg"], "derivation": d,
               "L_key": a.L_key, "speed_key": a.speed_key}
        a.out.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
