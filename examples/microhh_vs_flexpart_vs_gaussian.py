"""Compare CH4 concentration at ~500 m downwind sensors across three models.

Same 100 kg/hr Sacramento point source, three transport models:

- MicroHH   : turbulence-resolving LES (20 m), mass mixing ratio [kg/kg].
- FLEXPART  : Lagrangian, ERA5-driven (~28 km met, parameterized turbulence),
              concentration [ng/m3].
- Gaussian  : analytical steady-state plume with Pasquill-Gifford dispersion.

All converted to ng/m3 (and ppb) and sampled at the three sensor locations.

Run (after the MicroHH demo + forward FLEXPART runs have produced output):
    python examples/microhh_vs_flexpart_vs_gaussian.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from enforceflux.microhh import load_microhh_config, read_receptor_series  # noqa: E402
from enforceflux.microhh.geometry import BoxProjection  # noqa: E402
from enforceflux.microhh.units import (  # noqa: E402
    RHO_AIR,
    gaussian_plume_ground_conc,
    mixing_ratio_to_mass_conc,
)

MH_YAML = REPO_ROOT / "examples" / "microhh_sacramento_demo.yaml"
FP_NC = REPO_ROOT / "runs" / "flexpart_forward_sacramento_march31" / "flexpart_forward.nc"

Q_KG_S = 2.7777778e-2      # 100 kg/hr
U_M_S = 3.0                # mean wind (matches MicroHH forcing.u_geo)
STABILITY = "B"            # unstable (surface heating, moderate wind)
NG_PER_KG = 1.0e12
# ng/m3 -> ppb: X[mol/mol] = chi[kg/m3]*M_air/(M_ch4*rho); *1e9 for ppb
NG_M3_TO_PPB = (1.0 / NG_PER_KG) * (28.9647 / 16.043) / RHO_AIR * 1.0e9


def sensor_geometry(cfg):
    proj = BoxProjection(cfg.origin_lon, cfg.origin_lat, cfg.x_bearing_deg,
                         cfg.source_x0, cfg.source_y0)
    rows = []
    for r in cfg.receptors:
        x, y = proj.to_box(r.lon, r.lat)
        rows.append({"id": r.id, "lon": r.lon, "lat": r.lat,
                     "downwind_m": x - cfg.source_x0, "crosswind_m": y - cfg.source_y0})
    return rows


def microhh_conc(cfg):
    """Time-mean surface CH4 [ng/m3] at each receptor."""
    s = read_receptor_series(cfg, sample_level=0)
    # Average over the release window (drop initial pre-arrival transient).
    mean_kg_kg = np.nanmean(s.values, axis=0)
    return {rid: float(mixing_ratio_to_mass_conc(c) * NG_PER_KG)
            for rid, c in zip(s.receptor_ids, mean_kg_kg)}


def flexpart_conc(geom):
    """Time-mean surface CH4 [ng/m3] at nearest grid cell to each sensor."""
    ds = xr.open_dataset(FP_NC)
    field = ds["ch4_mixing_ratio"].isel(nageclass=0, pointspec=0, height=0)
    out = {}
    for g in geom:
        cell = field.sel(longitude=g["lon"], latitude=g["lat"], method="nearest")
        out[g["id"]] = float(cell.mean("time").values)   # already ng/m3
    return out


def gaussian_conc(geom):
    """Analytical ground-level CH4 [ng/m3] at each sensor."""
    out = {}
    for g in geom:
        c_kg_m3 = gaussian_plume_ground_conc(
            Q_KG_S, x_m=g["downwind_m"], u_m_s=U_M_S,
            h_m=20.0, y_m=g["crosswind_m"], stability=STABILITY)
        out[g["id"]] = c_kg_m3 * NG_PER_KG
    return out


def flexpart_plume_diagnostics():
    """FLEXPART real-ERA5 plume bearing and its own-centerline conc at ~500 m."""
    import math
    ds = xr.open_dataset(FP_NC)
    f = ds["ch4_mixing_ratio"].isel(nageclass=0, pointspec=0, height=0).mean("time").values
    lon, lat = ds.longitude.values, ds.latitude.values
    LON, LAT = np.meshgrid(lon, lat)
    dx = (LON + 121.75) * 111e3 * math.cos(math.radians(39.15))
    dy = (LAT - 39.15) * 111e3
    dist = np.sqrt(dx**2 + dy**2)
    w = f * (dist > 100) * (dist < 1500)
    bearing = math.degrees(math.atan2((w * dx).sum(), (w * dy).sum())) % 360
    ring = (dist > 350) & (dist < 700)
    centerline_ng = float(f[ring].max()) if ring.any() else float("nan")
    return bearing, centerline_ng


def main():
    cfg = load_microhh_config(MH_YAML)
    geom = sensor_geometry(cfg)
    mh = microhh_conc(cfg)
    fp = flexpart_conc(geom)
    ga = gaussian_conc(geom)

    print("=" * 84)
    print(f"CH4 concentration at ~500 m sensors — Q=100 kg/hr, U={U_M_S} m/s, PG-{STABILITY}")
    print("=" * 84)
    hdr = f"{'sensor':16s} {'downwind':>9s} {'cross':>7s} | " \
          f"{'MicroHH':>12s} {'FLEXPART':>12s} {'Gaussian':>12s}   (ng/m3)"
    print(hdr)
    print("-" * 84)
    ids = [g["id"] for g in geom]
    for g in geom:
        rid = g["id"]
        print(f"{rid:16s} {g['downwind_m']:8.0f}m {g['crosswind_m']:+6.0f}m | "
              f"{mh[rid]:12.3e} {fp[rid]:12.3e} {ga[rid]:12.3e}")
    print("-" * 84)
    print("same, in ppb:")
    for g in geom:
        rid = g["id"]
        print(f"{rid:16s} {'':17s}| "
              f"{mh[rid]*NG_M3_TO_PPB:12.3f} {fp[rid]*NG_M3_TO_PPB:12.3f} "
              f"{ga[rid]*NG_M3_TO_PPB:12.3f}   (ppb)")

    # Own-centerline comparison at ~500 m (isolates dispersion from wind dir).
    fp_bearing, fp_center_ng = flexpart_plume_diagnostics()
    mh_center = mh["sensor_ENE_axis"]       # MicroHH on-axis sensor
    ga_center = ga["sensor_ENE_axis"]       # Gaussian on-axis
    print("\n" + "=" * 84)
    print("Each model along ITS OWN plume centreline at ~500 m (dispersion only):")
    print(f"  MicroHH wind bearing = {cfg.x_bearing_deg:.0f} deg (idealised); "
          f"FLEXPART plume bearing = {fp_bearing:.0f} deg (real ERA5, 31 Mar)")
    print(f"  MicroHH  : {mh_center:.3e} ng/m3  ({mh_center*NG_M3_TO_PPB:.0f} ppb)")
    print(f"  FLEXPART : {fp_center_ng:.3e} ng/m3  ({fp_center_ng*NG_M3_TO_PPB:.0f} ppb)")
    print(f"  Gaussian : {ga_center:.3e} ng/m3  ({ga_center*NG_M3_TO_PPB:.0f} ppb)")

    # Bar chart
    x = np.arange(len(ids))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.bar(x - w, [mh[i] for i in ids], w, label="MicroHH (LES)")
    ax.bar(x,     [fp[i] for i in ids], w, label="FLEXPART (Lagrangian)")
    ax.bar(x + w, [ga[i] for i in ids], w, label="Gaussian (analytical)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([i.replace("sensor_", "") for i in ids])
    ax.set_ylabel("mean surface CH4 [ng/m3]")
    ax.set_title("CH4 at ~500 m downwind — three transport models, Q=100 kg/hr")
    ax.legend()
    out_png = REPO_ROOT / "runs" / "microhh_sacramento_demo" / "model_comparison.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\nWrote {out_png.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
