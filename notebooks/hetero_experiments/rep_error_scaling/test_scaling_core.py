#!/usr/bin/env python3
"""Recover known exponents from synthetic data before trusting the fits."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scaling_core import MODELS, SQRT_A, fit_model, loco_cv, condition_metrics

CVS = (0.5, 1.0, 2.0)
LS = (100.0, 250.0, 500.0)


def synthetic(a: float, alpha: float, beta: float, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    for cv in CVS:
        for L in LS:
            s = a * cv**alpha * (L / SQRT_A) ** beta
            rows.append({"CV": cv, "L_m": L, "l_star": L / SQRT_A,
                         "sigma_log": np.sqrt(np.log1p(cv**2)),
                         "sigma_rep": s * np.exp(rng.normal(0, noise))})
    return pd.DataFrame(rows)


def main() -> int:
    m3 = next(m for m in MODELS if m.name == "M3")
    for a, alpha, beta in [(0.4, 1.0, 1.0), (0.9, 0.6, 1.4), (0.2, 1.3, 0.5)]:
        d = synthetic(a, alpha, beta, noise=0.02, seed=1)
        f = fit_model(m3, d)
        ea = f["exponents"]["CV"][0]
        eb = f["exponents"]["l_star"][0]
        assert abs(ea - alpha) < 0.03, (ea, alpha)
        assert abs(eb - beta) < 0.03, (eb, beta)
        assert abs(f["a"] - a) / a < 0.05, (f["a"], a)
        print(f"  recovered a={f['a']:.3f} alpha={ea:.3f} beta={eb:.3f} "
              f"(true {a}, {alpha}, {beta})")

    # M2 is M3 with both exponents pinned at 1, so on data generated with
    # alpha=beta=1 the two must agree and M2 must have the lower BIC.
    d = synthetic(0.4, 1.0, 1.0, noise=0.02, seed=7)
    m2 = next(m for m in MODELS if m.name == "M2")
    assert fit_model(m2, d)["bic"] < fit_model(m3, d)["bic"]
    assert loco_cv(m2, d) < 0.1
    print("  M2 preferred over M3 when the truth is alpha=beta=1")

    # Exact data must give exactly zero residual for the correct model.
    d0 = synthetic(0.4, 1.0, 1.0, noise=0.0, seed=0)
    assert fit_model(m2, d0)["rmse_log"] < 1e-12
    print("  zero residual on noiseless data")

    # condition_metrics keeps bias and spread separate.
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "geometry": "point", "n": 1, "CV": 1.0, "L_m": 100.0,
        "e_signed": rng.normal(0.3, 0.1, 8),
    })
    cm = condition_metrics(df).iloc[0]
    assert abs(cm.bias - 0.3) < 0.12 and abs(cm.sigma_rep - 0.1) < 0.06
    assert cm.RMSE > cm.sigma_rep  # bias is not absorbed into the spread
    print("  bias and spread reported separately")
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
