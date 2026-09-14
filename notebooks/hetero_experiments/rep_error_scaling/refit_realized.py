#!/usr/bin/env python3
"""Refit the scaling law on the patchiness the 1 km fields actually contain.

domain_size_fields.py showed a 1 km window holds only part of the nominal
contrast at large L (60% of log-variance at L = 500 m) and that realized CV
falls further below nominal as CV rises. If the fitted exponents were an
artifact of those mislabelled covariates, refitting on realized statistics or
dropping the few-patch L = 500 m regime will move them. Same sigma_rep (SD of
e_signed over 8 seeds), same OLS/LOCO code from scaling_core; only the
covariates change.

Variants (per design):
  M3_nom   a*CV^alpha*(L/sqrtA)^beta, nominal CV, all 9 conditions (original)
  R1       nominal CV -> realized within-field CV
  R2       M4 with realized log-SD instead of nominal sigma_log
  R3       M3_nom on L in {100, 250} only
  R4       R1 on L in {100, 250} only
  M2_nom / M2_real   fixed exponents CV*L/sqrtA (nominal / realized CV)

Decision rule, fixed before the results were seen. The 1 km conclusions are
"distorted" for a design if either
  (a) alpha or beta under R1-R4 moves by more than the original M3_nom
      seed-bootstrap 95% half-width, or beta's 95% interval stops excluding 1;
  (b) M2 (nominal or realized) comes within 10% of the best model's LOCO RMSE.

Seed-level check: within each condition, does |e| track that seed's realized
CV? (log|e| on log cv_real with condition fixed effects, per geometry.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from field_stats import load_field  # noqa: E402
from scaling_core import CONDITION, DESIGN, SQRT_A, Model, fit_model, loco_cv  # noqa: E402

N_BOOT = 1000
SEED = 20260913
SMALL_L = (100.0, 250.0)

MODELS = {
    "M3_nom": (Model("M3_nom", "a·CV^α·(L/√A)^β", free=("CV", "l_star")), False),
    "R1": (Model("R1", "a·CVreal^α·(L/√A)^β", free=("cv_real", "l_star")), False),
    "R2": (Model("R2", "a·σlog,real^α·(L/√A)^β", free=("logsd_real", "l_star")), False),
    "R3": (Model("R3", "a·CV^α·(L/√A)^β, L≤250", free=("CV", "l_star")), True),
    "R4": (Model("R4", "a·CVreal^α·(L/√A)^β, L≤250", free=("cv_real", "l_star")), True),
    "M2_nom": (Model("M2_nom", "a·CV·L/√A", fixed=(("CV", 1.0), ("l_star", 1.0))), False),
    "M2_real": (Model("M2_real", "a·CVreal·L/√A",
                      fixed=(("cv_real", 1.0), ("l_star", 1.0))), False),
}
FREE = ["M3_nom", "R1", "R2", "R3", "R4"]


def seed_table() -> pd.DataFrame:
    """Realized within-field statistics of every truth field, per seed."""
    inv = pd.read_csv(HERE / "inversions.csv")
    rows = []
    for (L, cv, s), _ in inv.groupby(["L_m", "CV", "seed"]):
        F, _area = load_field(L, cv, s)
        rows.append({"L_m": L, "CV": cv, "seed": s,
                     "cv_real_seed": float(F.std() / F.mean()),
                     "logvar_real_seed": float(np.log(F).var())})
    return pd.DataFrame(rows)


def conditions(inv: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    """sigma_rep and realized covariates for each (design, condition)."""
    j = inv.merge(seeds, on=["L_m", "CV", "seed"], validate="m:1")
    g = j.groupby(DESIGN + CONDITION)
    d = g.agg(sigma_rep=("e_signed", lambda e: float(np.std(e, ddof=1))),
              cv_real=("cv_real_seed", "mean"),
              logvar_real=("logvar_real_seed", "mean")).reset_index()
    d["logsd_real"] = np.sqrt(d.logvar_real)
    d["l_star"] = d.L_m / SQRT_A
    return d


def fit_all(d: pd.DataFrame) -> dict:
    out = {}
    for name, (m, small) in MODELS.items():
        dd = d[d.L_m.isin(SMALL_L)] if small else d
        f = fit_model(m, dd)
        ex = f["exponents"]
        alpha = next((ex[c] for c in ("CV", "cv_real", "logsd_real") if c in ex),
                     (np.nan, np.nan))
        out[name] = {"alpha": alpha[0], "beta": ex.get("l_star", (np.nan,))[0],
                     "r2_log": f["r2_log"], "loco": loco_cv(m, dd),
                     "n_points": f["n_points"]}
    return out


def bootstrap(inv_design: pd.DataFrame, seeds: pd.DataFrame, rng) -> pd.DataFrame:
    """Resample seeds within each condition; sigma_rep and realized CV move
    together because both come from the same resampled seeds."""
    j = inv_design.merge(seeds, on=["L_m", "CV", "seed"], validate="1:1")
    groups = {c: g for c, g in j.groupby(CONDITION)}
    recs = []
    for _ in range(N_BOOT):
        parts = []
        for c, g in groups.items():
            idx = rng.integers(0, len(g), len(g))
            parts.append(g.iloc[idx])
        d = conditions(pd.concat(parts).drop(columns=["cv_real_seed",
                                                       "logvar_real_seed"]), seeds)
        # conditions() re-merges seeds by (L, CV, seed), so duplicated seeds keep
        # their own realized stats — exactly the resample.
        r = {}
        for name in FREE:
            m, small = MODELS[name]
            dd = d[d.L_m.isin(SMALL_L)] if small else d
            ex = fit_model(m, dd)["exponents"]
            r[f"{name}_alpha"] = next(ex[c][0] for c in ("CV", "cv_real", "logsd_real")
                                      if c in ex)
            r[f"{name}_beta"] = ex["l_star"][0]
        recs.append(r)
    return pd.DataFrame(recs)


def seed_level(inv: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    """log|e| on log(realized CV) with (design, condition) fixed effects."""
    j = inv.merge(seeds, on=["L_m", "CV", "seed"], validate="m:1")
    rows = []
    for geom, g in j.groupby("geometry"):
        g = g[g.e_abs > 0].copy()
        y = np.log(g.e_abs.to_numpy())
        x = np.log(g.cv_real_seed.to_numpy())
        cell = g.groupby(DESIGN + CONDITION).ngroup().to_numpy()
        # Demean within cells = fixed effects.
        yd = y - pd.Series(y).groupby(cell).transform("mean").to_numpy()
        xd = x - pd.Series(x).groupby(cell).transform("mean").to_numpy()
        slope = float(xd @ yd / (xd @ xd))
        resid = yd - slope * xd
        dof = len(y) - cell.max() - 2
        se = float(np.sqrt(resid @ resid / dof / (xd @ xd)))
        # Seeds are shared across the 4 sensor counts, so observations are not
        # independent across n; report the SE as a lower bound.
        rows.append({"geometry": geom, "slope_log_e_on_log_cvreal": slope,
                     "se_naive": se, "n_obs": len(y)})
    return pd.DataFrame(rows)


def main() -> int:
    inv = pd.read_csv(HERE / "inversions.csv")
    seeds = seed_table()
    seeds.to_csv(HERE / "realized_seed_stats.csv", index=False)
    cond = conditions(inv, seeds)
    cond.to_csv(HERE / "realized_conditions.csv", index=False)

    rng = np.random.default_rng(SEED)
    rows, verdicts = [], []
    for (geom, n), d in cond.groupby(DESIGN):
        fits = fit_all(d)
        bs = bootstrap(inv[(inv.geometry == geom) & (inv.n == n)], seeds, rng)
        ci = {c: (bs[c].quantile(0.025), bs[c].quantile(0.975)) for c in bs}
        half = {p: (ci[f"M3_nom_{p}"][1] - ci[f"M3_nom_{p}"][0]) / 2
                for p in ("alpha", "beta")}
        best = min(v["loco"] for v in fits.values())
        flags = []
        for name in FREE:
            f = fits[name]
            rows.append({"geometry": geom, "n": n, "variant": name,
                         "form": MODELS[name][0].label, "n_points": f["n_points"],
                         "alpha": f["alpha"],
                         "alpha_lo": ci[f"{name}_alpha"][0],
                         "alpha_hi": ci[f"{name}_alpha"][1],
                         "beta": f["beta"],
                         "beta_lo": ci[f"{name}_beta"][0],
                         "beta_hi": ci[f"{name}_beta"][1],
                         "r2_log": f["r2_log"], "loco_rmse_log": f["loco"]})
            if name == "M3_nom":
                continue
            for p in ("alpha", "beta"):
                shift = f[p] - fits["M3_nom"][p]
                if abs(shift) > half[p]:
                    flags.append(f"{name} {p} shift {shift:+.2f} > {half[p]:.2f}")
            if ci[f"{name}_beta"][1] >= 1.0:
                flags.append(f"{name} beta CI reaches 1")
        for name in ("M2_nom", "M2_real"):
            f = fits[name]
            rows.append({"geometry": geom, "n": n, "variant": name,
                         "form": MODELS[name][0].label, "n_points": f["n_points"],
                         "r2_log": f["r2_log"], "loco_rmse_log": f["loco"]})
            if f["loco"] <= 1.10 * best:
                flags.append(f"{name} LOCO {f['loco']:.3f} within 10% of best {best:.3f}")
        verdicts.append({"geometry": geom, "n": n,
                         "distorted": bool(flags), "flags": "; ".join(flags)})

    fits_df = pd.DataFrame(rows)
    fits_df.to_csv(HERE / "refit_realized.csv", index=False)
    ver = pd.DataFrame(verdicts)
    ver.to_csv(HERE / "refit_verdict.csv", index=False)
    sl = seed_level(inv, seeds)
    sl.to_csv(HERE / "refit_seed_level.csv", index=False)

    pd.set_option("display.width", 200)
    print("=== realized vs nominal CV by condition ===")
    print(cond.groupby(CONDITION)[["cv_real", "logsd_real"]].first().round(3)
          .assign(cv_ratio=lambda x: x.cv_real / x.index.get_level_values("CV"))
          .to_string())
    print("\n=== exponents (seed-bootstrap 95%) ===")
    print(fits_df.round(3).to_string(index=False))
    print("\n=== verdict per design ===")
    print(ver.to_string(index=False))
    print("\n=== seed-level: log|e| ~ log(realized CV), condition fixed effects ===")
    print(sl.round(3).to_string(index=False))
    (HERE / "refit_realized.json").write_text(json.dumps({
        "n_distorted_designs": int(ver.distorted.sum()),
        "n_designs": len(ver),
        "rule": "alpha/beta shift > original bootstrap 95% half-width, beta CI "
                "reaching 1, or M2 LOCO within 10% of best",
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
