#!/usr/bin/env python3
"""Test whether representativeness error in the 576-inversion OSSE collapses.

Consumes ``inversions.csv`` (build_table.py), ``field_stats.csv`` /
``footprint_scales.csv`` / ``error_decomposition.csv`` (field_stats.py).

Produces ``conditions.csv``, ``model_fits.csv``, ``calibration.csv``,
``results.json`` and the figures. All statistics live in ``scaling_core.py``,
which is exercised by ``test_scaling_core.py`` before anything here is trusted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scaling_core import (
    CONDITION, DESIGN, MODELS, SQRT_A, add_predictors, bootstrap_exponents,
    condition_metrics, fit_model, loco_cv,
)

HERE = Path(__file__).resolve().parent
PRIMARY = {"geometry": "point", "n": 1}


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inv = pd.read_csv(HERE / "inversions.csv")
    dec = pd.read_csv(HERE / "error_decomposition.csv")
    inv = inv.merge(dec, on="run", validate="1:1")
    fs = pd.read_csv(HERE / "field_stats.csv")
    fp = pd.read_csv(HERE / "footprint_scales.csv")
    return inv, fs, fp


def build_conditions(inv, fs, fp) -> pd.DataFrame:
    c = condition_metrics(inv, "e_signed")
    pure = condition_metrics(inv, "e_pure_representativeness")[
        DESIGN + CONDITION + ["bias", "sigma_rep"]
    ].rename(columns={"bias": "bias_pure", "sigma_rep": "sigma_rep_pure"})
    trans = condition_metrics(inv, "e_transport")[
        DESIGN + CONDITION + ["bias", "sigma_rep"]
    ].rename(columns={"bias": "bias_transport", "sigma_rep": "sigma_rep_transport"})
    c = c.merge(pure, on=DESIGN + CONDITION).merge(trans, on=DESIGN + CONDITION)

    # Calibration: reported sigma is a single scalar per inversion here.
    cal = inv.groupby(DESIGN + CONDITION).apply(
        lambda g: pd.Series({
            "sigma_reported": g.posterior_sigma_kg_s.mean() / g.Q_true_kg_s.mean(),
            "R_mean": np.mean(g.e_abs * g.Q_true_kg_s / g.posterior_sigma_kg_s),
            "cover68": np.mean(
                g.e_abs * g.Q_true_kg_s <= 1.0 * g.posterior_sigma_kg_s),
            "cover95": np.mean(
                g.e_abs * g.Q_true_kg_s <= 1.96 * g.posterior_sigma_kg_s),
            "dfs": g.dfs.mean(),
            "chi2_per_dof": g.chi2_per_dof.mean(skipna=True),
            "prior_influence": g.prior_influence.mean(),
            "ak_diag": g.ak_diag.mean(),
        }), include_groups=False).reset_index()
    c = c.merge(cal, on=DESIGN + CONDITION)
    return add_predictors(c, fs, fp)


def fit_table(cond: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (geom, n), d in cond.groupby(DESIGN):
        for m in MODELS:
            if m.name == "M7":
                continue  # L_H is constant within a design; pooled fit only
            f = fit_model(m, d)
            rows.append({
                "geometry": geom, "n": n, "model": m.name, "form": m.label,
                "a": f["a"], "a_lo": np.exp(f["log_a"] - 1.96 * f["log_a_se"]),
                "a_hi": np.exp(f["log_a"] + 1.96 * f["log_a_se"]),
                "alpha": f["exponents"].get(
                    "CV", f["exponents"].get("sigma_log", (np.nan, np.nan)))[0],
                "alpha_se": f["exponents"].get(
                    "CV", f["exponents"].get("sigma_log", (np.nan, np.nan)))[1],
                "beta": f["exponents"].get(
                    "l_star", f["exponents"].get("Pi", (np.nan, np.nan)))[0],
                "beta_se": f["exponents"].get(
                    "l_star", f["exponents"].get("Pi", (np.nan, np.nan)))[1],
                "r2_log": f["r2_log"], "adj_r2_log": f["adj_r2_log"],
                "rmse_log": f["rmse_log"], "rmse_rel": f["rmse_rel"],
                "aic": f["aic"], "bic": f["bic"],
                "loco_rmse_log": loco_cv(m, d),
                "n_params": f["n_params"], "n_points": f["n_points"],
            })
    return pd.DataFrame(rows)


def pooled_footprint_test(cond: pd.DataFrame) -> dict:
    """Does L/L_H beat L/sqrt(A) once all eight designs are pooled?

    Within one design L_H is a constant, so the two are indistinguishable. They
    separate only across designs, and only then if the design's own level is
    not absorbed by a free intercept — so the comparison is run both with a
    single intercept and with per-design intercepts.
    """
    d = cond.copy()
    y = np.log(d.sigma_rep.to_numpy(float))
    lcv = np.log(d.CV.to_numpy(float))
    out = {}
    for tag, xcol in (("l_star", "l_star"), ("Pi", "Pi")):
        for kind in ("single_intercept", "design_intercepts"):
            if kind == "single_intercept":
                X = np.column_stack([np.ones(len(d)), lcv, np.log(d[xcol])])
            else:
                D = pd.get_dummies(d.geometry + "_n" + d["n"].astype(str),
                                   dtype=float).to_numpy()
                X = np.column_stack([D, lcv, np.log(d[xcol])])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            r = y - X @ beta
            out[f"{tag}__{kind}"] = {
                "rmse_log": float(np.sqrt(np.mean(r**2))),
                "beta_x": float(beta[-1]), "alpha_CV": float(beta[-2]),
                "n_params": int(X.shape[1]),
            }
    # Saturating length: patches smaller than the footprint do not count twice.
    best = None
    for c in np.linspace(0.0, 4.0, 81):
        Leff = np.sqrt(d.L_m**2 + c * d.L_H_m**2) / SQRT_A
        X = np.column_stack([np.ones(len(d)), lcv, np.log(Leff)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        rm = float(np.sqrt(np.mean(r**2)))
        if best is None or rm < best["rmse_log"]:
            best = {"c": float(c), "rmse_log": rm, "alpha_CV": float(beta[1]),
                    "beta_Leff": float(beta[2]), "a": float(np.exp(beta[0])),
                    "n_params": 4}
    out["saturating_Leff"] = best
    return out


def n_dependence(cond: pd.DataFrame) -> pd.DataFrame:
    """Does 1/sqrt(n) describe how sigma_rep falls with sensor count?"""
    rows = []
    for geom, d in cond.groupby("geometry"):
        y = np.log(d.sigma_rep.to_numpy(float))
        X = np.column_stack([np.ones(len(d)), np.log(d.CV), np.log(d.l_star),
                             np.log(d["n"].astype(float))])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        k = X.shape[1]
        se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * (r @ r) / (len(y) - k))
        rows.append({
            "geometry": geom, "gamma_n": float(beta[3]), "gamma_n_se": float(se[3]),
            "gamma_lo": float(beta[3] - 1.96 * se[3]),
            "gamma_hi": float(beta[3] + 1.96 * se[3]),
            "sqrt_n_predicts": bool(
                beta[3] - 1.96 * se[3] <= -0.5 <= beta[3] + 1.96 * se[3]),
            "zero_in_ci": bool(beta[3] - 1.96 * se[3] <= 0 <= beta[3] + 1.96 * se[3]),
            "alpha_CV": float(beta[1]), "beta_L": float(beta[2]),
        })
    return pd.DataFrame(rows)


def calibration_test(inv: pd.DataFrame, cond: pd.DataFrame) -> pd.DataFrame:
    """Coverage before and after adding an empirical sigma_rep(H) term.

    sigma_rep(H) is taken from the M3 fit for the inversion's own design, and
    the fold structure is leave-one-(CV,L)-condition-out so the widening is
    never calibrated on the condition it is scored on.
    """
    from scipy.stats import norm

    m3 = next(m for m in MODELS if m.name == "M3")
    rows = []
    for (geom, n), d in cond.groupby(DESIGN):
        sub = inv[(inv.geometry == geom) & (inv["n"] == n)]
        for cv, L in d[CONDITION].itertuples(index=False):
            train = d[~((d.CV == cv) & (d.L_m == L))]
            f = fit_model(m3, train)
            pred = f["a"] * cv ** f["exponents"]["CV"][0] * \
                (L / SQRT_A) ** f["exponents"]["l_star"][0]
            g = sub[(sub.CV == cv) & (sub.L_m == L)]
            err = (g.Q_hat_kg_s - g.Q_true_kg_s).to_numpy(float)
            s_rep = pred * g.Q_true_kg_s.to_numpy(float)
            s0 = g.posterior_sigma_kg_s.to_numpy(float)
            s1 = np.sqrt(s0**2 + s_rep**2)
            rows.append({
                "geometry": geom, "n": n, "CV": cv, "L_m": L,
                "sigma_rep_pred": float(pred),
                "cover68_reported": float(np.mean(np.abs(err) <= s0)),
                "cover95_reported": float(np.mean(np.abs(err) <= 1.96 * s0)),
                "cover68_widened": float(np.mean(np.abs(err) <= s1)),
                "cover95_widened": float(np.mean(np.abs(err) <= 1.96 * s1)),
                "logscore_reported": float(np.mean(norm.logpdf(err, 0, s0))),
                "logscore_widened": float(np.mean(norm.logpdf(err, 0, s1))),
            })
    return pd.DataFrame(rows)


def main() -> int:
    inv, fs, fp = load()
    cond = build_conditions(inv, fs, fp)
    cond.to_csv(HERE / "conditions.csv", index=False)
    print(f"wrote conditions.csv ({len(cond)} conditions x {cond.shape[1]} cols)")

    fits = fit_table(cond)
    fits.to_csv(HERE / "model_fits.csv", index=False)

    prim = cond[(cond.geometry == PRIMARY["geometry"]) & (cond["n"] == PRIMARY["n"])]
    print("\n=== PRIMARY SUBSET: point sensors, n = 1 (9 conditions x 8 seeds) ===")
    print(prim[["CV", "L_m", "bias", "bias_p", "MAE", "RMSE", "sigma_rep",
                "sigma_rep_lo", "sigma_rep_hi", "H_simple"]]
          .sort_values(["CV", "L_m"]).round(4).to_string(index=False))

    pf = fits[(fits.geometry == "point") & (fits.n == 1)]
    print("\n=== MODEL COMPARISON, point n=1 ===")
    print(pf[["model", "form", "a", "alpha", "alpha_se", "beta", "beta_se",
              "r2_log", "rmse_log", "loco_rmse_log", "aic", "bic"]]
          .round(3).to_string(index=False))

    print("\n=== bootstrap over source realizations, M3, point n=1 ===")
    m3 = next(m for m in MODELS if m.name == "M3")
    bs = bootstrap_exponents(m3, inv, PRIMARY, fs, fp, n_boot=500)
    boot = {}
    for c in ("CV", "l_star"):
        q = np.percentile(bs[c], [2.5, 50, 97.5])
        boot[c] = {"lo": float(q[0]), "med": float(q[1]), "hi": float(q[2]),
                   "excludes_1": bool(q[0] > 1.0 or q[2] < 1.0)}
        print(f"  {c:8s} median {q[1]:.3f}  95% CI [{q[0]:.3f}, {q[2]:.3f}]"
              f"   excludes 1: {boot[c]['excludes_1']}")

    print("\n=== exponents by design (M3) ===")
    print(fits[fits.model == "M3"][
        ["geometry", "n", "a", "alpha", "alpha_se", "beta", "beta_se",
         "r2_log", "loco_rmse_log"]].round(3).to_string(index=False))

    print("\n=== best model per design, by leave-one-condition-out RMSE ===")
    best = fits.loc[fits.groupby(DESIGN).loco_rmse_log.idxmin()]
    print(best[["geometry", "n", "model", "form", "loco_rmse_log",
                "rmse_log", "bic"]].round(3).to_string(index=False))

    print("\n=== pooled footprint test ===")
    pft = pooled_footprint_test(cond)
    for k, v in pft.items():
        print(f"  {k:32s} {v}")
    print("  note: log(Pi) = log(L) - log(L_H) and L_H is constant within a "
          "design, so the two design-intercept fits are identical by "
          "construction. Only the single-intercept rows are informative.")

    print("\n=== sensor-number dependence ===")
    nd = n_dependence(cond)
    print(nd.round(3).to_string(index=False))

    print("\n=== calibration ===")
    cal = calibration_test(inv, cond)
    cal.to_csv(HERE / "calibration.csv", index=False)
    cond2 = cond.merge(cal, on=DESIGN + CONDITION)
    print(cond2.groupby(DESIGN)[
        ["sigma_reported", "sigma_rep", "R_mean", "cover68_reported",
         "cover68_widened", "cover95_reported", "cover95_widened"]
    ].mean().round(3).to_string())
    print("\n  coverage by H tercile (all designs pooled):")
    cond2["H_tercile"] = pd.qcut(cond2.H_simple, 3, labels=["low", "mid", "high"])
    print(cond2.groupby("H_tercile", observed=True)[
        ["H_simple", "sigma_rep", "sigma_reported", "R_mean",
         "cover68_reported", "cover68_widened",
         "cover95_reported", "cover95_widened",
         "logscore_reported", "logscore_widened"]].mean().round(3).to_string())

    print("\n=== error decomposition (SD across seeds), pooled over CV, L ===")
    print(cond.groupby(DESIGN)[
        ["sigma_rep", "sigma_rep_pure", "sigma_rep_transport"]
    ].mean().round(4).to_string())

    print("\n=== bias significance ===")
    nsig = (cond.bias_p < 0.05).sum()
    print(f"  {nsig} of {len(cond)} conditions have bias distinguishable from "
          f"zero at p<0.05 (8 realizations each)")
    print(f"  |bias| / sigma_rep: median {(cond.bias.abs()/cond.sigma_rep).median():.2f}, "
          f"max {(cond.bias.abs()/cond.sigma_rep).max():.2f}")

    results = {
        "primary_subset": PRIMARY,
        "bootstrap_M3_point_n1": boot,
        "pooled_footprint_test": pft,
        "n_dependence": nd.to_dict("records"),
        "n_conditions": int(len(cond)),
        "n_inversions": int(len(inv)),
        "bias_significant_conditions": int(nsig),
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print("\nwrote model_fits.csv, calibration.csv, results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
