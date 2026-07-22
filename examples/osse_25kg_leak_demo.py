"""
OSSE demo — 25 kg/hr methane point source (Ruhr, 2009-01-01).

Workflow
--------
1. Run a 3-hour forward FLEXPART simulation with the 25 kg/hr source.
2. Read the gridded concentration output (ng/m³).
3. Build a G-matrix by extracting concentration at instrument locations and
   dividing by the known emission rate.
4. Apply InstrumentOperator: add heteroscedastic noise + detection limits.
5. Run a Bayesian linear inversion to recover the emission rate.

Run from repo root:
    python examples/osse_25kg_leak_demo.py
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from enforceflux.core.base import ITransportSimulation
from enforceflux.utils.plugin_registry import get_plugin
from enforceflux.instrument import INSTRUMENT_DB, Instrument, InstrumentOperator
from enforceflux.analysis import analyze_information_content
from enforceflux.inversion import oe_from_linear

# ─── Scenario parameters ─────────────────────────────────────────────────────

CONFIG_YAML = Path(__file__).parent / "osse_25kg_leak.yaml"
OUTPUT_NC   = REPO_ROOT / "outputs" / "osse_25kg_leak.nc"

Q_TRUE_KG_HR  = 25.0                          # true emission rate (kg/hr)
Q_TRUE_KG_S   = Q_TRUE_KG_HR / 3600.0         # ≈ 6.944e-3 kg/s

Q_PRIOR_KG_HR = 10.0                          # prior best-guess (kg/hr)
Q_PRIOR_STD   = 15.0                          # prior 1-σ uncertainty (kg/hr)

# CH4 conversion: 1 ppm ≈ 655 800 ng/m³ (at 25 °C, 1 atm)
CH4_NG_M3_PER_PPM = 655_800.0

RNG_SEED = 7


# ─── Helper: build G-matrix from gridded FLEXPART output ─────────────────────

def build_g_matrix(
    ds,
    instruments: list[Instrument],
    op_indices: list[int],          # indices into `instruments` that are OP-type
    er_indices: list[int],          # indices that are emission-rate type
    q_kg_s: float,
) -> np.ndarray:
    """
    Build a (m, 1) G-matrix from the FLEXPART forward concentration field.

    For OP (concentration) instruments rows are in ppm / (kg/s), so that
        y_clean[i]  =  G[i,0] * Q_kg_s  =  c_i_ppm

    For emission-rate (LP_ESN / BP_GML / IM_LS) instrument rows are in
        (kg/hr) / (kg/s)  =  3600, so that
        y_clean[i]  =  3600 * Q_kg_s  =  Q_kg_hr
    """
    c_field = ds["ch4_mixing_ratio"].values     # (nageclass, pointspec, time, height, lat, lon)
    c_surface = c_field[0, 0, -1, 0, :, :]     # last time-step, lowest height (100 m)

    lons = ds["longitude"].values
    lats = ds["latitude"].values

    m = len(instruments)
    G = np.zeros((m, 1))

    for i in op_indices:
        inst = instruments[i]
        ilat = int(np.argmin(np.abs(lats - inst.y)))
        ilon = int(np.argmin(np.abs(lons - inst.x)))
        c_i_ng = float(c_surface[ilat, ilon])          # ng/m³
        c_i_ppm = c_i_ng / CH4_NG_M3_PER_PPM           # ppm
        G[i, 0] = c_i_ppm / q_kg_s                     # ppm / (kg/s)

    for i in er_indices:
        G[i, 0] = 3600.0                                # (kg/hr) / (kg/s)

    return G


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 68)
    print("OSSE — 25 kg/hr methane point source  (Ruhr, 2009-01-01)")
    print("=" * 68)

    # ── Step 1: run (or reuse) FLEXPART simulation ─────────────────
    print(f"\nSource    : ruhr_leak_25kghr  ({Q_TRUE_KG_HR:.1f} kg/hr = {Q_TRUE_KG_S*1e3:.3f} g/s)")
    print(f"Location  : 7.5°E, 51.5°N,  5 m AGL")
    print(f"Period    : 2009-01-01 00:00 → 03:00 UTC  (3 hours)")

    if OUTPUT_NC.exists():
        print(f"\n[cache] Using existing output: {OUTPUT_NC.relative_to(REPO_ROOT)}")
    else:
        print("\nRunning FLEXPART simulation …")
        simulation = get_plugin(
            "enforceflux.transport_simulation", "flexpart", ITransportSimulation
        )()
        simulation.simulate([], None, {"sim_config": str(CONFIG_YAML)})
        print(f"Output written → {OUTPUT_NC.relative_to(REPO_ROOT)}")

    # ── Step 2: read gridded output ─────────────────────────────────
    try:
        import xarray as xr
    except ImportError:
        print("xarray is required: pip install xarray")
        raise SystemExit(1)

    ds = xr.open_dataset(OUTPUT_NC)
    c_field  = ds["ch4_mixing_ratio"].values
    c_surface = c_field[0, 0, -1, 0, :, :]     # (lat, lon) at t=3h, h=100m

    lons = ds["longitude"].values
    lats  = ds["latitude"].values

    nz = c_surface[c_surface > 0]
    print(f"\nConcentration field (100 m AGL, t=3 h)")
    print(f"  Non-zero cells : {len(nz)}")
    print(f"  Peak           : {c_surface.max():.0f} ng/m³  "
          f"({c_surface.max() / CH4_NG_M3_PER_PPM * 1e3:.1f} ppb,  "
          f"{c_surface.max() / CH4_NG_M3_PER_PPM:.4f} ppm)")

    # Top-3 grid cells by concentration → used for OP instruments
    top3_flat = np.argsort(c_surface.ravel())[::-1][:3]
    top3_lat, top3_lon = np.unravel_index(top3_flat, c_surface.shape)

    # ── Step 3: define instruments ──────────────────────────────────
    print("\n" + "─" * 68)
    print("Instrument configuration")
    print("─" * 68)

    instruments: list[Instrument] = []

    # Three OP open-path sensors placed at the three peak concentration cells
    op_indices: list[int] = []
    for k, (ilat, ilon) in enumerate(zip(top3_lat, top3_lon)):
        c_ppb = float(c_surface[ilat, ilon]) / CH4_NG_M3_PER_PPM * 1e3
        print(f"  OP_{k+1}    lon={lons[ilon]:.1f}°E  lat={lats[ilat]:.1f}°N  "
              f"c={c_ppb:.1f} ppb  (DL=40 ppb)")
        op_indices.append(len(instruments))
        instruments.append(Instrument(
            id=f"OP_{k+1}", tech_id="OP", mode="good",
            x=float(lons[ilon]), y=float(lats[ilat]),
        ))

    # Emission-rate instruments monitoring the source (mode = good / challenging)
    er_configs = [
        ("LP_ESN", "good"),
        ("LP_ESN", "challenging"),
        ("BP_GML", "good"),
        ("BP_GML", "challenging"),
        ("IM_LS",  "good"),
    ]
    er_indices: list[int] = []
    for tech, mode in er_configs:
        p = INSTRUMENT_DB[tech][mode]
        print(f"  {tech:<7} [{mode:<11}]  DL={p.detection_limit:.2f} kg/hr  "
              f"σ={p.sigma_scale:.0%}·Q  bias={p.bias_scale:+.0%}·Q  "
              f"p_drop={p.dropout_probability:.0%}")
        er_indices.append(len(instruments))
        instruments.append(Instrument(
            id=f"{tech}_{mode[:4]}", tech_id=tech, mode=mode,
            x=7.5, y=51.5,
        ))

    # ── Step 4: build G and simulate observations ───────────────────
    print("\n" + "─" * 68)
    print("Simulated observations")
    print("─" * 68)

    G = build_g_matrix(ds, instruments, op_indices, er_indices, Q_TRUE_KG_S)
    x_true = np.array([Q_TRUE_KG_S])       # (1,) in kg/s

    op_result = InstrumentOperator(instruments, rng=np.random.default_rng(RNG_SEED))
    obs = op_result.simulate_observations(G, x_true)

    # Display results
    print(f"\n  True emission : {Q_TRUE_KG_HR:.2f} kg/hr\n")
    print(f"  {'ID':<16} {'observable':<20} {'y_clean':>10} {'y_obs':>10} {'σ':>8}  valid")
    print(f"  {'-'*72}")

    for i, inst in enumerate(instruments):
        p = inst.params
        unit = p.observable.replace("_", " ")
        yc   = obs.y_clean[i]
        yo   = obs.y_obs[i]
        sig  = float(np.sqrt(obs.R[i, i]))

        # Format in the right units
        if p.observable == "emission_rate_kg_hr":
            yc_s = f"{yc:>10.3f}"
            yo_s = f"{yo:>10.3f}" if not np.isnan(yo) else f"{'NaN':>10}"
            s_s  = f"{sig:>8.3f}"
        else:
            yc_s = f"{yc:>10.5f}"
            yo_s = f"{yo:>10.5f}" if not np.isnan(yo) else f"{'NaN':>10}"
            s_s  = f"{sig:>8.5f}"

        valid_mark = "✓" if obs.valid_mask[i] else "✗ (dropped)"
        print(f"  {inst.id:<16} {unit:<20} {yc_s} {yo_s} {s_s}  {valid_mark}")

    n_valid = int(obs.valid_mask.sum())
    print(f"\n  {n_valid} / {len(instruments)} observations valid")

    # ── Step 5: Bayesian inversion (emission-rate instruments only) ──
    valid_er = [i for i in er_indices if obs.valid_mask[i]]
    if len(valid_er) < 1:
        print("\n[skip] No valid emission-rate observations — cannot invert.")
        return

    print("\n" + "─" * 68)
    print("Bayesian linear inversion  (emission-rate instruments, kg/hr)")
    print("─" * 68)

    # Collect valid emission-rate obs, convert everything to kg/hr
    G_inv  = obs.H_g[valid_er]                    # (k, 1)  in (kg/hr)/(kg/s)
    y_inv  = obs.y_obs[valid_er]                  # (k,)    in kg/hr
    R_inv  = obs.R[np.ix_(valid_er, valid_er)]    # (k, k)  in (kg/hr)²

    # Convert G and x so the inversion works in kg/hr units throughout:
    #   G_kghr @ x_kghr = y_kghr
    #   G_kghr[i,0] = 1.0  (since G[i,0]=3600 and x_true was kg/s → 3600*kg_s = kg_hr)
    G_kghr    = G_inv / 3600.0                    # now (kg/hr)/(kg/hr) = dimensionless = 1.0
    x_prior   = np.array([Q_PRIOR_KG_HR])
    s_a       = np.array([[Q_PRIOR_STD**2]])

    oe = oe_from_linear(
        G=G_kghr, y=y_inv, x_prior=x_prior, Sa=s_a, Se=R_inv,
        source_names=["ruhr_leak"],
    )

    x_post = float(oe.x_posterior[0])
    x_std  = float(np.sqrt(oe.posterior_cov[0, 0]))
    ak     = float(oe.averaging_kernel[0, 0])

    print(f"\n  Prior             : {Q_PRIOR_KG_HR:.1f} ± {Q_PRIOR_STD:.1f} kg/hr")
    print(f"  True              : {Q_TRUE_KG_HR:.1f} kg/hr")
    print(f"  Posterior         : {x_post:.2f} ± {x_std:.2f} kg/hr")
    print(f"  Averaging kernel  : {ak:.3f}  (1.0 = perfectly observation-constrained)")
    print(f"  Observations used : {len(valid_er)} (from {len(er_indices)} deployed)")

    # ── Information content analysis ──────────────────────────────────
    # Build the full ER G-matrix in kg/hr units and compute Fisher metrics.
    Se_er = np.diag(R_inv) if R_inv.ndim == 2 else R_inv
    fisher, dof, posterior = analyze_information_content(
        G=G_kghr, Se=Se_er, Sa=s_a.ravel(),
        source_names=["ruhr_leak"],
    )
    print(f"\n  DFS               : {dof.dfs_total:.3f}  (1.0 = fully observation-constrained)")
    print(f"  Uncertainty reduc.: {posterior.uncertainty_reduction[0]*100:.0f} %")
    print(f"  Posterior σ       : {posterior.posterior_sigma[0]:.2f} kg/hr")
    print(f"  FIM eigenvalue    : {fisher.eigenvalues[0]:.4g}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    prior_err  = abs(Q_PRIOR_KG_HR - Q_TRUE_KG_HR)
    post_err   = abs(x_post - Q_TRUE_KG_HR)
    print(f"  Prior error          : {prior_err:.1f} kg/hr")
    print(f"  Posterior error      : {post_err:.2f} kg/hr")
    print(f"  Uncertainty reduction: {(1 - x_std/Q_PRIOR_STD)*100:.0f} %")
    print("=" * 68)


if __name__ == "__main__":
    main()
