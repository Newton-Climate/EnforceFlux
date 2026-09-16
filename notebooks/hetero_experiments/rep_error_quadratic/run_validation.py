#!/usr/bin/env python3
"""Compare the quadratic-form sigma_rep against the source-heterogeneity OSSE.

    python run_validation.py              # the published 8-seed study (576 inversions)
    python run_validation.py --nseeds 20  # seeds 0-19 (1440 inversions), outputs suffixed _s20

The 8-seed mode reads ``../rep_error_scaling/{inversions,error_decomposition}.csv``.
Any other seed count builds the table straight from ``runs/`` (``e_signed`` exactly as
``build_table.py`` defines it) and gates its seed 0-7 rows against those CSVs.
Everything is read-only except this directory. Raises on any failed gate.

Outputs (suffix empty for 8 seeds, ``_s<N>`` otherwise)
  gates.csv            integrity checks and their measured values
  conditions.csv       72 rows: geometry x n x CV x L, predicted vs empirical
  inversions_z.csv     one row per inversion: error standardized by prediction
  pooled.csv           the figure pooling (per CV or L value, per network)
  metrics.csv          r, RMSE across the 72 conditions, per target x variant
  mc_errors.npz        Monte Carlo errors per condition
  kernels.npz          w_g and dw per design, for the kernel maps (seed-independent)
"""
from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats

from quadratic_rep import (
    RUNS, covariance, design_weights, grid_from_truth, monte_carlo_phi,
    pure_error, sample_lognormal_field, sigma_rep2, spec_for,
)

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "rep_error_scaling"
N_MC = 4000
MC_SEED = 20260914
VARIANTS = ("loglinear", "lognormal", "mc")
TARGETS = {"total": "e_signed", "pure": "e_pure_representativeness"}
LS, CVS, NS, NETS = (100, 250, 500), (0.5, 1.0, 2.0), (1, 2, 3, 4), {"op": "open_path", "point": "point"}


def build_inversions(seeds) -> pd.DataFrame:
    """One row per seeded inversion, from the run tree, as build_table.py defines e_signed."""
    q_true, rows = {}, []
    for L, cv, s, n, net in itertools.product(LS, CVS, seeds, NS, NETS):
        tag = f"l{L}_cv{cv:.1f}".replace(".", "p") + f"_s{s}_wind3"
        nature_run = f"source_heterogeneity_les_rice_paddy_{tag}_surface"
        run = f"source_heterogeneity_les_rice_paddy_{tag}_n{n}_{net}"
        if nature_run not in q_true:
            with xr.open_dataset(RUNS / nature_run / "dispersion" / "truth_field.nc") as ds:
                q_true[nature_run] = float(ds.attrs["Q_true_kg_s"])
        Q = q_true[nature_run]
        x_opt = float(np.load(RUNS / run / "flux" / "matrices.npz")["x_opt"][0])
        rows.append({"run": run, "L_m": float(L), "CV": cv, "seed": s, "n": n,
                     "geometry": NETS[net], "nature_run": nature_run, "Q_true_kg_s": Q,
                     "e_signed": (x_opt - Q) / Q})
    return pd.DataFrame(rows)


def main(nseeds: int) -> int:
    sfx = "" if nseeds == 8 else f"_s{nseeds}"
    gates = []
    if nseeds == 8:
        inv = pd.read_csv(SRC / "inversions.csv").merge(
            pd.read_csv(SRC / "error_decomposition.csv"), on="run", validate="one_to_one")
    else:
        inv = build_inversions(range(nseeds))
        ref = pd.read_csv(SRC / "inversions.csv").set_index("run")
        old = inv[inv.seed < 8].set_index("run")
        d = float((old.e_signed - ref.e_signed.reindex(old.index)).abs().max())
        gates.append(("rebuilt_e_signed_matches_inversions_csv_seeds0_7", d < 1e-12, d))
        if not d < 1e-12:
            raise AssertionError(f"rebuilt e_signed disagrees with inversions.csv ({d:.2e})")
    if len(inv) != 72 * nseeds:
        raise AssertionError(f"expected {72 * nseeds} inversions, found {len(inv)}")

    # ---- 1. grid and covariance parameterization, straight from the files
    nature = inv[["L_m", "CV", "seed", "nature_run"]].drop_duplicates()
    fields, grid0, regen_err = {}, None, 0.0
    for r in nature.itertuples():
        path = RUNS / r.nature_run / "dispersion" / "truth_field.nc"
        grid, a = grid_from_truth(path)
        grid0 = grid0 or grid
        if grid != grid0:
            raise AssertionError(f"grid differs at {path}")
        if a["covariance_model"] != "exponential" or float(a["L_true_m"]) != r.L_m \
                or float(a["cv"]) != r.CV or int(a["seed"]) != r.seed:
            raise AssertionError(f"design attributes disagree with the inversion table at {path}")
        with xr.open_dataset(path) as ds:
            F = ds["F_true"].values.copy()
        # Regenerate with the plugin's exact recipe: rng = default_rng(seed).
        spec = spec_for(grid, r.L_m, r.CV, Q=float(a["Q_true_kg_s"]), seed=r.seed)
        with warnings.catch_warnings():  # fallback counted in the MC gates below
            warnings.simplefilter("ignore", RuntimeWarning)
            F2 = sample_lognormal_field(spec, np.random.default_rng(r.seed))
        regen_err = max(regen_err, float(np.abs(F2 - F).max() / np.abs(F).max()))
        fields[(r.L_m, r.CV, r.seed)] = F
    gates.append((f"grid_identical_all_{len(nature)}_fields", True,
                  f"nx={grid0.nx} ny={grid0.ny} dx={grid0.dx_m} origin=({grid0.origin_x_m},{grid0.origin_y_m})"))
    gates.append(("fields_regenerated_by_generator_max_rel_err", regen_err < 1e-10, regen_err))
    if regen_err >= 1e-10:
        raise AssertionError(f"generator does not reproduce stored fields ({regen_err:.2e})")

    # ---- 2. effective footprint weights, one per design
    designs = {k: design_weights(k[0], k[1], g.run.iloc[0])
               for k, g in inv.groupby(["geometry", "n"])}
    for k, g in inv.groupby(["geometry", "n"]):
        G0 = designs[k].G_fine
        for run in g.run:
            if not np.array_equal(np.load(RUNS / f"{run}_gp/dispersion/jacobian.npz")["G"], G0):
                raise AssertionError(f"Jacobian varies within design {k} at {run}")
    gates.append(("jacobian_constant_within_each_design", True, len(designs)))

    # The exact linear map must reproduce the existing perfect-transport errors.
    inv["e_pure_recomputed"] = [pure_error(designs[(r.geometry, r.n)], fields[(r.L_m, r.CV, r.seed)])
                                for r in inv.itertuples()]
    if nseeds != 8:
        inv["e_pure_representativeness"] = inv.e_pure_recomputed
        inv["e_transport"] = inv.e_signed - inv.e_pure_representativeness
        dec = pd.read_csv(SRC / "error_decomposition.csv").set_index("run")
        old = inv[inv.seed < 8].set_index("run")
        dmax = float((old.e_pure_recomputed - dec.e_pure_representativeness.reindex(old.index)).abs().max())
    else:
        dmax = float((inv.e_pure_recomputed - inv.e_pure_representativeness).abs().max())
    gates.append(("dw_dot_phi_reproduces_e_pure_seeds0_7_max_abs", dmax < 1e-9, dmax))
    if not dmax < 1e-9:
        raise AssertionError(f"linear map does not reproduce e_pure ({dmax:.2e})")

    # ---- 3. predictions: two analytic covariances and the generator itself
    rows, e_store, pit = [], {}, {}
    for (L, cv), _ in inv.groupby(["L_m", "CV"]):
        spec = spec_for(grid0, L, cv)
        covs = {v: covariance(spec, v) for v in ("loglinear", "lognormal")}
        P, n_chol = monte_carlo_phi(spec, N_MC, MC_SEED + int(L) * 10 + int(cv * 10))
        gates.append((f"mc_cholesky_fallback_draws_L{L:g}_cv{cv:g}", True, f"{n_chol}/{N_MC}"))
        for (geom, n), dw in designs.items():
            e_mc = dw.ak * (P @ dw.dw) + (dw.ak - 1.0)
            row = {"geometry": geom, "n": n, "CV": cv, "L_m": L, "ak": dw.ak,
                   "sd_pred_loglinear": np.sqrt(sigma_rep2(dw, covs["loglinear"])),
                   "sd_pred_lognormal": np.sqrt(sigma_rep2(dw, covs["lognormal"])),
                   "sd_pred_mc": e_mc.std(ddof=1), "mean_pred_mc": e_mc.mean(),
                   "mae_pred_mc": np.abs(e_mc).mean(),
                   "sd_pred_mc_se": e_mc.std(ddof=1) / np.sqrt(2 * (N_MC - 1))}
            sub = inv[(inv.geometry == geom) & (inv.n == n) & (inv.L_m == L) & (inv.CV == cv)]
            m = len(sub)
            # What an m-realization sample SD looks like when the model is exact:
            # heavy lognormal tails bias a small-sample SD low, not just widen it.
            sd_m = e_mc[: (N_MC // m) * m].reshape(-1, m).std(axis=1, ddof=1)
            row |= {"sd_mc_m_median": np.median(sd_m),
                    "sd_mc_m_lo": np.quantile(sd_m, 0.025),
                    "sd_mc_m_hi": np.quantile(sd_m, 0.975)}
            e_store[f"{geom}_n{n}_cv{cv:g}_L{L:g}"] = e_mc
            for rr in sub.itertuples():
                pit[rr.run] = float(np.mean(e_mc <= rr.e_pure_representativeness))
            for t, col in TARGETS.items():
                sd = sub[col].std(ddof=1)
                row |= {f"sd_emp_{t}": sd, f"mean_emp_{t}": sub[col].mean(),
                        f"mae_emp_{t}": sub[col].abs().mean(),
                        f"sd_emp_{t}_lo": sd * np.sqrt((m - 1) / stats.chi2.ppf(0.975, m - 1)),
                        f"sd_emp_{t}_hi": sd * np.sqrt((m - 1) / stats.chi2.ppf(0.025, m - 1))}
            row["m_realizations"] = m
            rows.append(row)
    cond = pd.DataFrame(rows).sort_values(["geometry", "n", "CV", "L_m"])
    cond.to_csv(HERE / f"conditions{sfx}.csv", index=False)
    np.savez_compressed(HERE / f"mc_errors{sfx}.npz", **e_store)
    r_pt = stats.pearsonr(inv.e_pure_representativeness, inv.e_transport)[0]
    gates.append((f"info_corr_e_pure_vs_e_transport_all{len(inv)}", True, round(r_pt, 4)))

    # ---- 4. per-inversion standardized errors
    z = inv[["run", "geometry", "n", "CV", "L_m", "seed", "e_signed",
             "e_pure_representativeness", "e_transport"]].merge(
        cond[["geometry", "n", "CV", "L_m", *[f"sd_pred_{v}" for v in VARIANTS]]],
        on=["geometry", "n", "CV", "L_m"])
    for v in VARIANTS:
        for t, col in TARGETS.items():
            z[f"z_{t}_{v}"] = z[col] / z[f"sd_pred_{v}"]
    # Rank of each stored field's e_pure within the generator's own distribution;
    # uniform on [0, 1] if the seeds are typical draws.
    z["pit_pure_mc"] = z.run.map(pit)
    z.to_csv(HERE / f"inversions_z{sfx}.csv", index=False)

    # ---- 5. metrics across the 72 conditions
    mrows = []
    for t in TARGETS:
        for v in VARIANTS:
            p, e = cond[f"sd_pred_{v}"].to_numpy(), cond[f"sd_emp_{t}"].to_numpy()
            zz = z[f"z_{t}_{v}"]
            mrows.append({
                "target": t, "variant": v,
                "pearson_r_variance": stats.pearsonr(p**2, e**2)[0],
                "pearson_r_sd": stats.pearsonr(p, e)[0],
                "pearson_r_log_sd": stats.pearsonr(np.log(p), np.log(e))[0],
                "spearman_rho_sd": stats.spearmanr(p, e)[0],
                "rmse_sd_pp": 100 * np.sqrt(np.mean((p - e) ** 2)),
                "rmse_variance_pp2": 1e4 * np.sqrt(np.mean((p**2 - e**2) ** 2)),
                "rmse_log10_ratio": np.sqrt(np.mean(np.log10(p / e) ** 2)),
                "median_ratio_pred_over_emp": np.median(p / e),
                "frac_pred_in_emp_chi2_95ci": np.mean((p >= cond[f"sd_emp_{t}_lo"]) & (p <= cond[f"sd_emp_{t}_hi"])),
                "z_sd": zz.std(ddof=1), "z_frac_abs_lt1": np.mean(zz.abs() < 1),
                "z_frac_abs_lt2": np.mean(zz.abs() < 2),
                # Model-independent: is the empirical SD a plausible m-draw SD?
                "frac_emp_in_mc_m_95band": np.mean((e >= cond.sd_mc_m_lo) & (e <= cond.sd_mc_m_hi)),
                "median_ratio_emp_over_mc_m_median": np.median(e / cond.sd_mc_m_median),
            })
    ks = stats.kstest(z.pit_pure_mc, "uniform")
    gates.append(("info_pit_pure_mc_ks_stat_p", True, f"D={ks.statistic:.3f} p={ks.pvalue:.2e}"))
    pd.DataFrame(mrows).to_csv(HERE / f"metrics{sfx}.csv", index=False)

    # ---- 6. the published figure pooling: 3 of L or CV x 4 n x all seeds, per network
    prow = []
    for geom in ("point", "open_path"):
        for by in ("CV", "L_m"):
            for val in sorted(inv[by].unique()):
                sub = inv[(inv.geometry == geom) & (inv[by] == val)]
                c = cond[(cond.geometry == geom) & (cond[by] == val)]
                r = {"geometry": geom, "pooled_by": by, "value": val, "n_runs": len(sub)}
                for t, col in TARGETS.items():
                    r[f"sd_emp_{t}"] = sub[col].std(ddof=1)
                    r[f"mae_emp_{t}"] = sub[col].abs().mean()
                # Pooled variance of a zero-mean mixture of equal-weight conditions.
                for v in ("loglinear", "lognormal"):
                    r[f"sd_pred_{v}"] = np.sqrt(np.mean(c[f"sd_pred_{v}"] ** 2))
                    r[f"mae_pred_{v}"] = np.mean(c[f"sd_pred_{v}"]) * np.sqrt(2 / np.pi)
                r["sd_pred_mc"] = np.sqrt(np.mean(c.sd_pred_mc**2) + np.var(c.mean_pred_mc))
                r["mae_pred_mc"] = c.mae_pred_mc.mean()
                prow.append(r)
    pooled = pd.DataFrame(prow)
    pooled.to_csv(HERE / f"pooled{sfx}.csv", index=False)

    # Tie the 8-seed pooling back to the published numbers before anyone relies on it.
    if nseeds == 8:
        ref = pd.read_csv(HERE.parent / "seed_sweep_results.csv")
        ref = ref[ref.network == "point"]
        for by, stat, want in (("CV", "sd", ref.groupby("CV").q_rel_error.std()),
                               ("L_m", "mae", ref.groupby("L_m").q_rel_error.apply(lambda s: s.abs().mean()))):
            got = pooled[(pooled.geometry == "point") & (pooled.pooled_by == by)].set_index("value")[f"{stat}_emp_total"]
            diff = float((got - want).abs().max())
            gates.append((f"pooled_point_{stat}_by_{by}_matches_published", diff < 1e-5, diff))
            if diff >= 1e-5:
                raise AssertionError(f"pooling by {by} does not reproduce the published figure")

    np.savez(HERE / "kernels.npz",
             **{f"{g}_n{n}_w": d.w_g for (g, n), d in designs.items()},
             **{f"{g}_n{n}_dw": d.dw for (g, n), d in designs.items()},
             x=grid0.cell_centers()[0], y=grid0.cell_centers()[1])
    pd.DataFrame(gates, columns=["gate", "passed", "value"]).to_csv(HERE / f"gates{sfx}.csv", index=False)

    pd.set_option("display.width", 200, "display.precision", 4)
    print(pd.DataFrame(gates, columns=["gate", "passed", "value"]).to_string(index=False))
    print(pooled.to_string(index=False))
    print(pd.DataFrame(mrows).to_string(index=False))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nseeds", type=int, default=8)
    raise SystemExit(main(ap.parse_args().nseeds))
