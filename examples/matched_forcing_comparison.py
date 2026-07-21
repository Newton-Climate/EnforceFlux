"""Three-model CH4 comparison under matched forcing (3 m/s @ 68 deg, 100 kg/hr).

All three models see the same mean wind and source; we compare CH4 at the three
~500 m sensors (on-axis + two flanks) and on the plume centreline:

- MicroHH (LES, 20 m)   : resolved turbulent plume (intact column receptors).
- Gaussian (analytical) : smooth ensemble-mean plume, PG unstable dispersion.
- FLEXPART (Lagrangian) : coarse-grid (~500 m) centreline value (real ERA5 that
                          afternoon was convective, ~3.6 m/s -- close to the
                          idealised forcing). Its 10 m grid is SHOT-NOISE limited
                          in a convective BL: the 0-20 m surface layer holds
                          almost no mass and each 10 m cell samples ~0-1
                          particles, so a 10 m plume never forms (see
                          flexpart_forward_sac_fine.yaml).

    python examples/matched_forcing_comparison.py
"""
import dataclasses
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from enforceflux.microhh import load_microhh_config, read_receptor_series  # noqa: E402
from enforceflux.microhh.geometry import BoxProjection  # noqa: E402
from enforceflux.microhh.units import RHO_AIR, gaussian_plume_ground_conc  # noqa: E402

MH_YAML = REPO_ROOT / "examples" / "microhh_sacramento_demo.yaml"
FP_NC = REPO_ROOT / "runs" / "flexpart_forward_sacramento_march31" / "flexpart_forward.nc"

Q_KG_S = 2.7777778e-2
U_M_S = 3.0
KGKG_TO_PPB = (28.9647 / 16.043) * 1.0e9
NGM3_TO_PPB = (1.0 / 1.0e12) / RHO_AIR * (28.9647 / 16.043) * 1.0e9


def geometry(cfg):
    proj = BoxProjection(cfg.origin_lon, cfg.origin_lat, cfg.x_bearing_deg,
                         cfg.source_x0, cfg.source_y0)
    rows = []
    for r in cfg.receptors:
        x, y = proj.to_box(r.lon, r.lat)
        rows.append({"id": r.id, "lon": r.lon, "lat": r.lat,
                     "downwind_m": x - cfg.source_x0, "crosswind_m": y - cfg.source_y0})
    return rows


def flexpart_centerline_ppb():
    import xarray as xr
    if not FP_NC.exists():
        return None
    ds = xr.open_dataset(FP_NC)
    f = ds["ch4_mixing_ratio"].isel(nageclass=0, pointspec=0, height=0).mean("time").values
    lon, lat = ds.longitude.values, ds.latitude.values
    LON, LAT = np.meshgrid(lon, lat)
    d = np.sqrt(((LON + 121.75) * 111e3 * math.cos(math.radians(39.15)))**2
                + ((LAT - 39.15) * 111e3)**2)
    ring = (d > 350) & (d < 700)
    return float(f[ring].max()) * NGM3_TO_PPB if ring.any() else None


def main():
    cfg = load_microhh_config(MH_YAML)
    # Prefer the protected snapshot of the LES output if present (the live case
    # dir can be clobbered by a later re-run).
    snapshot = REPO_ROOT / "runs" / "microhh_matched_snapshot"
    if snapshot.exists() and list(snapshot.glob("*.column.*.nc")):
        cfg = dataclasses.replace(cfg, case_dir=snapshot)
    geom = geometry(cfg)
    s = read_receptor_series(cfg, sample_level=0)
    les = {rid: float(np.nanmean(s.values[:, i]) * KGKG_TO_PPB)
           for i, rid in enumerate(s.receptor_ids)}
    gau = {g["id"]: gaussian_plume_ground_conc(
                Q_KG_S, x_m=g["downwind_m"], u_m_s=U_M_S, h_m=20.0,
                y_m=g["crosswind_m"], stability="B") * 1.0e12 * NGM3_TO_PPB
           for g in geom}
    fp_center = flexpart_centerline_ppb()

    print("=" * 74)
    print("MATCHED FORCING — 3 m/s @ 68 deg, 100 kg/hr — CH4 at ~500 m sensors [ppb]")
    print("=" * 74)
    print(f"{'sensor':16s} {'downwind':>9s} {'cross':>7s} | {'MicroHH':>10s} {'Gaussian':>10s}")
    for g in geom:
        r = g["id"]
        print(f"{r:16s} {g['downwind_m']:8.0f}m {g['crosswind_m']:+6.0f}m | "
              f"{les[r]:10.1f} {gau[r]:10.1f}")
    print("-" * 74)
    on_axis = geom[0]["id"]
    print(f"centreline (on-axis): MicroHH {les[on_axis]:.0f} ppb, "
          f"Gaussian {gau[on_axis]:.0f} ppb"
          + (f", FLEXPART(coarse) {fp_center:.0f} ppb" if fp_center else ""))
    print("FLEXPART 10 m grid: shot-noise limited (no resolved plume) — see note.")

    ids = [g["id"] for g in geom]
    x = np.arange(len(ids)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.bar(x - w / 2, [les[i] for i in ids], w, label="MicroHH (LES, resolved)")
    ax.bar(x + w / 2, [gau[i] for i in ids], w, label="Gaussian (analytical mean)")
    if fp_center:
        ax.axhline(fp_center, color="gray", ls=":", label=f"FLEXPART centreline ({fp_center:.0f} ppb)")
    ax.set_xticks(x); ax.set_xticklabels([i.replace("sensor_", "") for i in ids])
    ax.set_ylabel("mean surface CH4 [ppb]")
    ax.set_title("Matched forcing — CH4 at ~500 m: LES resolves the asymmetric meander")
    ax.legend()
    out = REPO_ROOT / "runs" / "microhh_sacramento_demo" / "matched_forcing_comparison.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nWrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
