"""
Instrument operator demo using the existing FLEXPART forward simulation output.

Demonstrates non-NaN simulated observations for both:
  1. Concentration instruments (OP — open-path optical)
  2. Emission-rate instruments (LP_ESN, BP_GML, IM_LS)

Run from repo root:
    python examples/instrument_operator_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enforceflux.models.instrument import Instrument, InstrumentOperator

# ─── Constants ────────────────────────────────────────────────────────────────

# CH4 concentration to ppm at surface conditions (~25 °C, 1 atm)
# 1 ppm CH4 = 16.04 g/mol / 24.465 L/mol = 655.8 µg/m³ = 655_800 ng/m³
CH4_NG_M3_PER_PPM = 655_800.0

EXISTING_NC = Path(__file__).parent.parent / "outputs" / "example_simulation.nc"

# Ruhr industrial source params from simulation_test.yaml
Q_RUHR_KG_S = 5.0e-3          # kg/s
Q_RUHR_KG_HR = Q_RUHR_KG_S * 3600.0  # ≈ 18 kg/hr


def main() -> None:
    print("=" * 65)
    print("Instrument Operator Demo — existing FLEXPART run")
    print("=" * 65)
    if not EXISTING_NC.exists():
        print(f"\nERROR: {EXISTING_NC} not found.")
        print("Run:  python examples/run_simulation_example.py")
        raise SystemExit(1)

    # ── Load FLEXPART gridded output ──────────────────────────────────
    try:
        import xarray as xr
    except ImportError:
        print("xarray is required: pip install xarray")
        raise SystemExit(1)

    ds = xr.open_dataset(EXISTING_NC)
    # dims: (nageclass=1, pointspec=1, time=3, height=4, lat=65, lon=85)
    # units: ng/m³  (despite the variable name "ch4_mixing_ratio")
    c_field = ds["ch4_mixing_ratio"].values  # full array

    lons = ds["longitude"].values   # 1-D, 1° spacing
    lats = ds["latitude"].values    # 1-D, 1° spacing

    # Last time step, lowest height level (100 m AGL)
    c_surface = c_field[0, 0, -1, 0, :, :]   # (lat, lon)

    print(f"\nFLEXPART output: {EXISTING_NC.name}")
    print(f"  Grid       : {len(lons)}×{len(lats)} cells at 1° resolution")
    print(f"  Sources    : Ruhr industrial ({Q_RUHR_KG_HR:.1f} kg/hr)  +  Paris Basin diffuse")
    nz = c_surface[c_surface > 0]
    print(f"  Surface max: {c_surface.max():.0f} ng/m³  "
          f"({c_surface.max() / CH4_NG_M3_PER_PPM * 1e3:.1f} ppb)")
    print(f"  Non-zero cells: {len(nz)}")

    # ── Part 1: Open-path concentration instruments ────────────────────
    print("\n" + "─" * 65)
    print("Part 1 — Open-Path (OP) instruments at high-concentration cells")
    print("─" * 65)
    print(f"  OP detection limit: 0.04 ppm  |  noise σ: 0.003 ppm (additive)")

    # Pick the 3 grid cells with highest surface concentration
    flat_order = np.argsort(c_surface.ravel())[::-1]
    top_lat_idx, top_lon_idx = np.unravel_index(flat_order[:3], c_surface.shape)

    op_instruments: list[Instrument] = []
    G_op_rows: list[list[float]] = []

    for k, (ilat, ilon) in enumerate(zip(top_lat_idx, top_lon_idx)):
        c_i = float(c_surface[ilat, ilon])         # ng/m³ from FLEXPART
        c_ppm = c_i / CH4_NG_M3_PER_PPM            # ppm
        # G[i, 0]: ppm per (kg/s), so that y_clean = G @ [Q_kg_s] = c_ppm
        g_ij = c_ppm / Q_RUHR_KG_S                 # ppm / (kg/s)
        op_instruments.append(Instrument(
            id=f"OP_{k+1}",
            tech_id="OP",
            x=float(lons[ilon]),
            y=float(lats[ilat]),
        ))
        G_op_rows.append([g_ij])
        print(f"\n  OP_{k+1} at lon={lons[ilon]:.1f}°E, lat={lats[ilat]:.1f}°N")
        print(f"    FLEXPART concentration : {c_i:.0f} ng/m³  ({c_ppm*1e3:.2f} ppb,  {c_ppm:.4f} ppm)")
        print(f"    G[i,0] (ppm per kg/s) : {g_ij:.4f}")

    G_op = np.array(G_op_rows)               # (3, 1)
    x_true_op = np.array([Q_RUHR_KG_S])     # kg/s

    rng = np.random.default_rng(42)
    op_result = InstrumentOperator(op_instruments, rng=rng).simulate_observations(
        G_op, x_true_op
    )

    print(f"\n  {'Instrument':<10} {'y_clean (ppm)':>15} {'y_obs (ppm)':>13} {'valid':>6} {'σ (ppm)':>9}")
    print(f"  {'-'*57}")
    for k, inst in enumerate(op_instruments):
        sigma = float(np.sqrt(op_result.R[k, k]))
        print(f"  {inst.id:<10} {op_result.y_clean[k]:>15.5f} {op_result.y_obs[k]:>13.5f} "
              f"{str(op_result.valid_mask[k]):>6} {sigma:>9.5f}")

    # ── Part 2: Emission-rate instruments (LP_ESN, IM_LS, BP_GML) ────
    print("\n" + "─" * 65)
    print("Part 2 — Emission-rate instruments (LP_ESN, IM_LS, BP_GML)")
    print("─" * 65)
    print(f"  True emission: {Q_RUHR_KG_HR:.1f} kg/hr  ({Q_RUHR_KG_S*1e3:.1f} g/s)")
    print()
    print("  Instrument   mode          DL (kg/hr)  σ_scale   p_drop")
    print("  " + "-" * 53)
    from enforceflux.models.instrument import INSTRUMENT_DB
    for tech in ["LP_ESN", "IM_LS", "BP_GML"]:
        for mode in ["good", "challenging", "bad"]:
            p = INSTRUMENT_DB[tech][mode]
            print(f"  {tech:<12} {mode:<14} {p.detection_limit:>8.2f}  "
                  f"{p.sigma_scale:>6.2f}    {p.dropout_probability:.2f}")

    # Build instruments — one per type/mode combination
    er_instruments: list[Instrument] = []
    G_er_rows: list[list[float]] = []

    for tech in ["LP_ESN", "IM_LS", "BP_GML"]:
        for mode in ["good", "challenging"]:
            er_instruments.append(Instrument(
                id=f"{tech[:4]}_{mode[:4]}",
                tech_id=tech,
                mode=mode,
                x=7.5, y=51.5,   # at Ruhr source location
            ))
            # G[i, 0] converts x_true[kg/s] → observable[kg/hr]: factor = 3600
            G_er_rows.append([3600.0])

    G_er = np.array(G_er_rows)                  # (6, 1)
    x_true_er = np.array([Q_RUHR_KG_S])         # kg/s

    er_result = InstrumentOperator(er_instruments, rng=np.random.default_rng(42)).simulate_observations(
        G_er, x_true_er
    )

    print(f"\n  {'Instrument':<14} {'y_clean (kg/hr)':>16} {'y_obs (kg/hr)':>15} {'valid':>6} {'σ (kg/hr)':>10}")
    print(f"  {'-'*65}")
    for k, inst in enumerate(er_instruments):
        sigma = float(np.sqrt(er_result.R[k, k]))
        yobs_str = f"{er_result.y_obs[k]:.3f}" if not np.isnan(er_result.y_obs[k]) else "NaN"
        print(f"  {inst.id:<14} {er_result.y_clean[k]:>16.3f} {yobs_str:>15} "
              f"{str(er_result.valid_mask[k]):>6} {sigma:>10.3f}")

    print("\n" + "=" * 65)
    print("Done.")


if __name__ == "__main__":
    main()
