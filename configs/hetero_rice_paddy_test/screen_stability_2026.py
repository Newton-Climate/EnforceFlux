#!/usr/bin/env python3
"""Screen summer 2026 ERA5 hours at the Sac Valley rice site by stability.

Pulls hourly single-level fields at the grid point nearest 39.15 N, 121.75 W
(the site in sac_met_2026-07-22.yaml), computes the Obukhov length from the
ERA5 friction velocity and surface buoyancy flux, and writes one row per hour
plus a per-day summary of the daytime (10-16 PDT) and nighttime (22-04 PDT)
windows. Used to pick the neutral and stable companions to 22 July (unstable).

    python screen_stability_2026.py --start 2026-06-01 --end 2026-09-07 \
        --out-dir runs/stability_screen_2026
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

SITE_LAT, SITE_LON = 39.15, -121.75
VARIABLES = [
    "10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature",
    "2m_dewpoint_temperature", "surface_pressure", "boundary_layer_height",
    "total_cloud_cover", "friction_velocity",
    "surface_sensible_heat_flux", "surface_latent_heat_flux",
]
K, G, CP, LV, RD = 0.4, 9.81, 1004.0, 2.5e6, 287.05


def fetch(start: str, end: str, out_dir: Path) -> list[list[Path]]:
    """One CDS request per month; a whole summer in one trips the cost limit."""
    import cdsapi

    dates = pd.date_range(start, end, freq="D")
    out_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    groups = []
    for (year, month), ds in pd.Series(dates, index=dates).groupby([dates.year, dates.month]):
        dest = out_dir / f"era5_sfc_site_{year}{month:02d}.nc"
        req = {
            "product_type": ["reanalysis"],
            "variable": VARIABLES,
            "year": [f"{year}"],
            "month": [f"{month:02d}"],
            "day": [f"{d.day:02d}" for d in ds.index],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [39.5, -122.0, 39.0, -121.5],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        if not dest.exists():
            client.retrieve("reanalysis-era5-single-levels", req, str(dest))
        # CDS may still hand back a zip (instant + accumulated streams split).
        if zipfile.is_zipfile(dest):
            unz = out_dir / f"unzipped_{year}{month:02d}"
            with zipfile.ZipFile(dest) as z:
                z.extractall(unz)
            groups.append(sorted(unz.glob("*.nc")))
        else:
            groups.append([dest])
    return groups


def _load_one(files: list[Path]) -> pd.DataFrame:
    ds = xr.merge([xr.open_dataset(f) for f in files], compat="override")
    tname = "valid_time" if "valid_time" in ds.dims else "time"
    p = ds.sel(latitude=SITE_LAT, longitude=SITE_LON, method="nearest")
    df = p.to_dataframe().reset_index().set_index(tname)
    df.index = pd.to_datetime(df.index)
    return df


def load(groups: list[list[Path]]) -> pd.DataFrame:
    return pd.concat([_load_one(g) for g in groups]).sort_index()


def stability(df: pd.DataFrame) -> pd.DataFrame:
    # ERA5 fluxes are hourly accumulations (J m-2), positive downward.
    H = -df["sshf"] / 3600.0
    LE = -df["slhf"] / 3600.0
    T, ps = df["t2m"], df["sp"]
    rho = ps / (RD * T)
    ustar = df["zust"].clip(lower=1e-3)
    # Virtual buoyancy flux, w'theta_v' ~ H/(rho cp) + 0.61 T E/rho.
    wtv = H / (rho * CP) + 0.61 * T * (LE / LV) / rho
    L = -(ustar ** 3) * T / (K * G * wtv.where(wtv.abs() > 1e-6, np.sign(wtv) * 1e-6 + 1e-9))
    out = pd.DataFrame({
        "wind10": np.hypot(df["u10"], df["v10"]),
        "wind_dir": (270.0 - np.rad2deg(np.arctan2(df["v10"], df["u10"]))) % 360.0,
        "ustar": df["zust"], "H": H, "LE": LE, "L": L, "zL10": 10.0 / L,
        "blh": df["blh"], "tcc": df["tcc"], "t2m_C": T - 273.15,
    })
    out.index.name = "time_utc"
    out["time_pdt"] = out.index - pd.Timedelta(hours=7)
    return out


def daily(h: pd.DataFrame) -> pd.DataFrame:
    pdt = h.set_index("time_pdt")
    rows = []
    for day, g in pdt.groupby(pdt.index.date):
        dayw = g.between_time("10:00", "16:00")
        # Night following this day: 22:00 PDT through 04:00 PDT next morning.
        nxt = pdt.loc[pd.Timestamp(day) + pd.Timedelta(hours=22):
                      pd.Timestamp(day) + pd.Timedelta(hours=28)]
        rows.append({
            "date": day,
            "day_wind10": dayw["wind10"].mean(), "day_ustar": dayw["ustar"].mean(),
            "day_H": dayw["H"].mean(), "day_zL10": dayw["zL10"].median(),
            "day_blh": dayw["blh"].mean(), "day_tcc": dayw["tcc"].mean(),
            "day_wind_dir": dayw["wind_dir"].median(),
            "night_wind10": nxt["wind10"].mean(), "night_ustar": nxt["ustar"].mean(),
            "night_H": nxt["H"].mean(), "night_zL10": nxt["zL10"].median(),
            "night_blh": nxt["blh"].mean(),
        })
    return pd.DataFrame(rows).set_index("date")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-06-01")
    ap.add_argument("--end", default="2026-09-07")
    ap.add_argument("--out-dir", type=Path, default=Path("runs/stability_screen_2026"))
    a = ap.parse_args()
    files = fetch(a.start, a.end, a.out_dir)
    h = stability(load(files))
    h = h.loc[a.start:pd.Timestamp(a.end) + pd.Timedelta(hours=23)]
    h.to_csv(a.out_dir / "hourly_stability.csv", float_format="%.4g")
    d = daily(h)
    d.to_csv(a.out_dir / "daily_stability.csv", float_format="%.4g")
    print(d.loc[[pd.Timestamp("2026-07-22").date()]].T if pd.Timestamp("2026-07-22").date() in d.index else "")
    print(f"wrote {len(h)} hours, {len(d)} days to {a.out_dir}")


if __name__ == "__main__":
    main()
