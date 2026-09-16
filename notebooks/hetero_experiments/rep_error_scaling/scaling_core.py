#!/usr/bin/env python3
"""Condition-level metrics and candidate scaling models.

Imported by ``scaling_analysis.py``. Kept separate so the statistics can be
exercised on synthetic data with known exponents (``test_scaling_core.py``)
without drawing any figures.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
A_M2 = 1.0e6
SQRT_A = np.sqrt(A_M2)
N_BOOT = 2000
RNG_SEED = 20260903

DESIGN = ["geometry", "n"]
CONDITION = ["CV", "L_m"]


# ------------------------------------------------------------- metrics


def bootstrap_sd(e: np.ndarray, n_boot: int = N_BOOT, seed: int = 0):
    """Percentile bootstrap CI for the SD of a small sample.

    BCa is not used: with eight realizations the acceleration term is estimated
    from eight jackknife points and is itself noisier than the interval it
    corrects. The chi-square interval is reported alongside as a parametric
    cross-check.
    """
    rng = np.random.default_rng(seed)
    m = len(e)
    draws = rng.integers(0, m, size=(n_boot, m))
    sds = e[draws].std(axis=1, ddof=1)
    return float(np.percentile(sds, 2.5)), float(np.percentile(sds, 97.5)), sds


def chi2_sd_ci(s: float, m: int, alpha: float = 0.05):
    """Parametric CI for an SD from m normal observations."""
    from scipy.stats import chi2

    dof = m - 1
    lo = s * np.sqrt(dof / chi2.ppf(1 - alpha / 2, dof))
    hi = s * np.sqrt(dof / chi2.ppf(alpha / 2, dof))
    return float(lo), float(hi)


def condition_metrics(df: pd.DataFrame, col: str = "e_signed") -> pd.DataFrame:
    """Bias, MAE, RMSE and run-to-run SD for every (design, condition) cell.

    Bias and spread are kept as separate columns throughout; nothing here folds
    one into the other.
    """
    from scipy.stats import ttest_1samp

    rows = []
    for i, (key, g) in enumerate(df.groupby(DESIGN + CONDITION, sort=True)):
        e = g[col].to_numpy(float)
        m = len(e)
        sd = float(e.std(ddof=1))
        lo, hi, _ = bootstrap_sd(e, seed=RNG_SEED + i)
        clo, chi = chi2_sd_ci(sd, m)
        t = ttest_1samp(e, 0.0)
        rows.append({
            "geometry": key[0], "n": key[1], "CV": key[2], "L_m": key[3],
            "m_realizations": m,
            "bias": float(e.mean()),
            "bias_se": float(e.std(ddof=1) / np.sqrt(m)),
            "bias_p": float(t.pvalue),
            "MAE": float(np.abs(e).mean()),
            "RMSE": float(np.sqrt((e**2).mean())),
            "sigma_rep": sd,
            "sigma_rep_lo": lo, "sigma_rep_hi": hi,
            "sigma_rep_chi2_lo": clo, "sigma_rep_chi2_hi": chi,
        })
    return pd.DataFrame(rows)


def add_predictors(df: pd.DataFrame, field_stats: pd.DataFrame,
                   footprints: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(field_stats, on=CONDITION, how="left")
    out = out.merge(footprints[["geometry", "n", "L_H_m"]], on=DESIGN, how="left")
    out["l_star"] = out.L_m / SQRT_A                       # dimensionless L
    out["H_simple"] = out.CV / np.sqrt(out.N_eff_simple)   # == CV*L/sqrt(A)
    out["H_lognormal"] = out.CV / np.sqrt(out.N_eff_lognormal)
    out["H_sigmalog"] = out.sigma_log / np.sqrt(out.N_eff_lognormal)
    out["Pi"] = out.L_m / out.L_H_m                        # source vs footprint
    return out


# ------------------------------------------------------------- models


@dataclass(frozen=True)
class Model:
    name: str
    label: str
    # Fixed-exponent terms: (column, exponent) applied as column**exponent.
    fixed: tuple[tuple[str, float], ...] = ()
    # Free-exponent terms: column names whose log enters as a free regressor.
    free: tuple[str, ...] = ()

    @property
    def n_params(self) -> int:
        return 1 + len(self.free)


MODELS = (
    Model("M0", "a·CV", fixed=(("CV", 1.0),)),
    Model("M1", "a·L/√A", fixed=(("l_star", 1.0),)),
    Model("M2", "a·CV·L/√A", fixed=(("CV", 1.0), ("l_star", 1.0))),
    Model("M3", "a·CV^α·(L/√A)^β", free=("CV", "l_star")),
    Model("M4", "a·σ_log^α·(L/√A)^β", free=("sigma_log", "l_star")),
    Model("M5", "a·CV/√N_eff,lognormal", fixed=(("H_lognormal", 1.0),)),
    Model("M6", "a·σ_log/√N_eff,lognormal", fixed=(("H_sigmalog", 1.0),)),
    Model("M7", "a·CV^α·(L/L_H)^β", free=("CV", "Pi")),
)


def design_matrix(model: Model, d: pd.DataFrame):
    """Log-space design matrix and offset for one candidate model.

    Every model is fitted as ``log y = log a + offset + sum_k p_k log x_k``,
    so models with fixed exponents contribute an offset and free ones a column.
    This keeps AIC/BIC comparable: they differ only in ``n_params``.
    """
    n = len(d)
    offset = np.zeros(n)
    for col, p in model.fixed:
        offset += p * np.log(d[col].to_numpy(float))
    X = [np.ones(n)]
    for col in model.free:
        X.append(np.log(d[col].to_numpy(float)))
    return np.column_stack(X), offset


def fit_model(model: Model, d: pd.DataFrame, y_col: str = "sigma_rep") -> dict:
    """Ordinary least squares of log(y) on the model's log-space regressors.

    OLS rather than weighted: the sampling SE of ``log s`` is
    ``1/sqrt(2(m-1))`` for every condition here, since all conditions have the
    same eight realizations, so weights would be uniform.
    """
    X, offset = design_matrix(model, d)
    y = np.log(d[y_col].to_numpy(float)) - offset
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    rss = float(resid @ resid)
    sigma2 = rss / max(n - k, 1)
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv) * sigma2)

    y_full = np.log(d[y_col].to_numpy(float))
    pred = X @ beta + offset
    ss_tot = float(((y_full - y_full.mean()) ** 2).sum())
    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else np.nan
    adj = 1.0 - (1 - r2) * (n - 1) / max(n - k, 1)
    # Gaussian log-likelihood in log space, for AIC/BIC on a common scale.
    ll = -0.5 * n * (np.log(2 * np.pi * rss / n) + 1)
    return {
        "model": model.name, "label": model.label,
        "log_a": float(beta[0]), "log_a_se": float(se[0]),
        "a": float(np.exp(beta[0])),
        "exponents": {c: (float(b), float(s))
                      for c, b, s in zip(model.free, beta[1:], se[1:])},
        "n_params": k, "n_points": n,
        "rmse_log": float(np.sqrt(rss / n)),
        "rmse_rel": float(np.sqrt(np.mean((np.exp(pred) / np.exp(y_full) - 1) ** 2))),
        "r2_log": float(r2), "adj_r2_log": float(adj),
        "aic": float(2 * k - 2 * ll), "bic": float(k * np.log(n) - 2 * ll),
        "pred": np.exp(pred), "obs": np.exp(y_full),
    }


def loco_cv(model: Model, d: pd.DataFrame, y_col: str = "sigma_rep") -> float:
    """Leave-one-(CV,L)-condition-out prediction error, in log space.

    With nine design points this is the only generalization estimate worth
    quoting; in-sample R^2 on nine points is not.
    """
    errs = []
    for cond, _ in d.groupby(CONDITION):
        mask = (d.CV == cond[0]) & (d.L_m == cond[1])
        train, test = d[~mask], d[mask]
        if len(train) < model.n_params + 1:
            return np.nan
        f = fit_model(model, train, y_col)
        Xt, off = design_matrix(model, test)
        beta = np.array([f["log_a"]] + [f["exponents"][c][0] for c in model.free])
        pred = Xt @ beta + off
        errs.extend(pred - np.log(test[y_col].to_numpy(float)))
    return float(np.sqrt(np.mean(np.asarray(errs) ** 2)))


def bootstrap_exponents(model: Model, inv: pd.DataFrame, design: dict,
                        field_stats: pd.DataFrame, footprints: pd.DataFrame,
                        n_boot: int = 500, seed: int = RNG_SEED) -> pd.DataFrame:
    """Resample source realizations within each condition, refit, repeat.

    The source realization is the independent experimental unit, so the
    resampling is by seed, never by inversion.
    """
    rng = np.random.default_rng(seed)
    sub = inv
    for k, v in design.items():
        sub = sub[sub[k] == v]
    groups = {c: g.e_signed.to_numpy(float)
              for c, g in sub.groupby(CONDITION)}
    keys = list(groups)
    out = []
    for _ in range(n_boot):
        rows = []
        for c in keys:
            e = groups[c]
            samp = e[rng.integers(0, len(e), len(e))]
            rows.append({"CV": c[0], "L_m": c[1],
                         "sigma_rep": float(samp.std(ddof=1))})
        d = pd.DataFrame(rows)
        d = d.merge(field_stats, on=CONDITION).assign(
            **{k: v for k, v in design.items()})
        d = d.merge(footprints[["geometry", "n", "L_H_m"]], on=DESIGN, how="left")
        d["l_star"] = d.L_m / SQRT_A
        d["Pi"] = d.L_m / d.L_H_m
        d["H_lognormal"] = d.CV / np.sqrt(d.N_eff_lognormal)
        d["H_sigmalog"] = d.sigma_log / np.sqrt(d.N_eff_lognormal)
        f = fit_model(model, d)
        rec = {"log_a": f["log_a"]}
        rec.update({c: f["exponents"][c][0] for c in model.free})
        out.append(rec)
    return pd.DataFrame(out)
