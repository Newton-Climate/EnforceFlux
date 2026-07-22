"""Build a near-uniform-wind ERA5 met set so FLEXPART sees the LES's mean wind.

The MicroHH demo is forced with a uniform 3 m/s wind toward bearing 68 deg. The
real ERA5 for 31 Mar 2020 blew a different way, so a fair comparison needs
FLEXPART driven by the *same* mean wind. This clones the april_week met and
overwrites the horizontal wind components (all model levels + 10 m) with the
constant LES wind, leaving thermodynamics/surface fluxes intact.

Important: a *perfectly* constant GRIB field packs to bitsPerValue=0, which
FLEXPART's reader treats as zero — the plume then never advects. We add a tiny
(<0.1 %) deterministic ripple so the field packs normally while remaining
physically uniform.

    python examples/build_matched_met.py
"""
import math
import shutil
import sys
from pathlib import Path

import eccodes as ec
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "runs" / "sacramento_valley_2020" / "meteo_april_week"
DST = REPO_ROOT / "runs" / "sacramento_valley_2020" / "meteo_april_week_matched"

U_SPEED = 3.0          # m/s, matches MicroHH forcing.u_geo
BEARING_DEG = 68.0     # box +x downwind bearing
U_EAST = U_SPEED * math.sin(math.radians(BEARING_DEG))    # 10u / u
V_NORTH = U_SPEED * math.cos(math.radians(BEARING_DEG))   # 10v / v
RIPPLE = 1.0e-3        # fractional ripple amplitude to keep bitsPerValue>0

WIND_TARGET = {"u": U_EAST, "v": V_NORTH, "10u": U_EAST, "10v": V_NORTH}


def _rewrite_winds(src: Path, dst: Path) -> None:
    fin = open(src, "rb")
    fout = open(dst, "wb")
    try:
        while True:
            g = ec.codes_grib_new_from_file(fin)
            if g is None:
                break
            sn = ec.codes_get(g, "shortName")
            if sn in WIND_TARGET:
                n = ec.codes_get_size(g, "values")
                # Uniform target + sub-permille deterministic ripple.
                ramp = RIPPLE * WIND_TARGET[sn] * np.cos(np.arange(n) * 0.1)
                ec.codes_set_values(g, WIND_TARGET[sn] + ramp)
            ec.codes_write(g, fout)
            ec.codes_release(g)
    finally:
        fin.close()
        fout.close()


def main() -> None:
    if not SRC.exists():
        sys.exit(f"Source met not found: {SRC}")
    DST.mkdir(parents=True, exist_ok=True)

    for f in sorted(SRC.iterdir()):
        if f.name.startswith("EA2"):
            _rewrite_winds(f, DST / f.name)
            print(f"  wind-matched {f.name}")
        else:
            shutil.copy2(f, DST / f.name)

    print(f"\nWind set to U={U_SPEED} m/s @ {BEARING_DEG:.0f} deg "
          f"(u={U_EAST:.3f}, v={V_NORTH:.3f} m/s, +/-{RIPPLE:.1%} ripple)")
    print(f"Matched met: {DST.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
