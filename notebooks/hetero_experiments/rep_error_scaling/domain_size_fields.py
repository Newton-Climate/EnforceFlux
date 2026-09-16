#!/usr/bin/env python3
"""Does the 1 km source domain represent the patchiness the labels claim?

Generator-only test, no transport. For each nominal (L, CV) of the rice-paddy
sweep, draw many realizations of the production lognormal generator on 1, 2
and 4 km domains (all at the sweep's 40 m cell) and measure what one realized
field actually contains:

* ``logvar_ratio`` — within-field variance of log F over the nominal
  sigma^2 = log(1+CV^2). The Q_true renormalization divides out each field's
  domain mean, so only within-domain contrast reaches the footprint; theory
  says E[ratio] = 1 - mean_ij rho(r_ij).
* ``cv_ratio`` — realized within-field CV of F over nominal CV, with the
  spread an 8-seed design actually sees.
* ``L_hat_m`` — e-folding scale from the same pooled-ACF estimator
  field_stats.py used (the one that saturated at 179 m for L = 500 m), pooled
  over 8 seeds as in the sweep, reported as median and 10-90% over batches.
* ``N_eff_lognormal`` — 1/mean(rho_F) over the grid, as in field_stats.py.

A crop check draws on 4 km and keeps the central 1 km. The generator is exact
in distribution, so this must match drawing directly on 1 km; if it does, any
shortfall is the window, not the generator.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from enforceflux.source_fields.lognormal_gp import (  # noqa: E402
    FieldGrid,
    LognormalFieldSpec,
    sample_lognormal_field,
)
from field_stats import (  # noqa: E402
    _lag_matrix,
    empirical_acf,
    fit_efolding,
    n_eff_from_rho,
    rho_log,
    rho_lognormal,
)

DX = 40.0
DOMAINS_M = (1000.0, 2000.0, 4000.0)
L_VALUES = (100.0, 250.0, 500.0)
CV_VALUES = (0.5, 1.0, 2.0)
N_SEEDS = 200          # realizations per condition
BATCH = 8              # the sweep pooled L_hat over 8 seeds
Q = 1.0


def draw(n: int, L: float, cv: float, seed: int) -> tuple[np.ndarray, str]:
    grid = FieldGrid(nx=n, ny=n, dx_m=DX)
    spec = LognormalFieldSpec(grid=grid, Q_true_kg_s=Q, L_m=L, cv=cv, seed=seed)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        F = sample_lognormal_field(spec, np.random.default_rng(seed))
    method = "cholesky" if any("Cholesky" in str(x.message) for x in w) else "circulant"
    return F, method


def sampled_acf(fields: list[np.ndarray], dx: float, nbins: int = 24,
                npairs: int = 2_000_000):
    """empirical_acf with random cell pairs instead of all of them.

    Same bins (0 to 0.6 of the domain width), same per-field demeaning and
    pooled normalization; only the pair set is subsampled, because the full
    lag matrix of a 100x100 grid is 1e8 entries per field.
    """
    ny, nx = fields[0].shape
    rng = np.random.default_rng(1)
    i, j = rng.integers(0, nx * ny, (2, npairs))
    R = np.hypot(i % nx - j % nx, i // nx - j // nx) * dx
    edges = np.linspace(0.0, 0.6 * nx * dx, nbins + 1)
    idx = np.digitize(R, edges) - 1
    keep = (idx >= 0) & (idx < nbins)
    num = np.zeros(nbins)
    den = np.zeros(nbins)
    var = 0.0
    for F in fields:
        a = np.log(F).ravel()
        a = a - a.mean()
        prod = a[i] * a[j]
        num += np.bincount(idx[keep], weights=prod[keep], minlength=nbins)
        den += np.bincount(idx[keep], minlength=nbins)
        var += float(a.var())
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, (num / np.maximum(den, 1)) / (var / len(fields))


def summarise(fields: list[np.ndarray], L: float, cv: float, n: int) -> dict:
    s2 = np.log1p(cv**2)
    logvar = np.array([np.log(F).var() for F in fields]) / s2
    cvr = np.array([F.std() / F.mean() for F in fields]) / cv
    acf = empirical_acf if n <= 50 else sampled_acf
    lhat = []
    for b in range(len(fields) // BATCH):
        c, a = acf(fields[b * BATCH:(b + 1) * BATCH], DX)
        lhat.append(fit_efolding(c, a))
    lhat = np.array(lhat)
    R = _lag_matrix(n, n, DX) if n <= 50 else None
    if R is None:
        # 100x100 lag matrix is 1e8 entries; subsample pairs for mean(rho).
        rng = np.random.default_rng(0)
        x = (np.arange(n) + 0.5) * DX
        XX, YY = np.meshgrid(x, x, indexing="xy")
        p = np.column_stack([XX.ravel(), YY.ravel()])
        i, j = rng.integers(0, len(p), (2, 2_000_000))
        R = np.hypot(*(p[i] - p[j]).T)
    mean_rho = float(np.mean(rho_log(L)(R)))
    return {
        "logvar_ratio_mean": logvar.mean(),
        "logvar_ratio_theory": 1.0 - mean_rho,
        "logvar_ratio_p10": np.percentile(logvar, 10),
        "logvar_ratio_p90": np.percentile(logvar, 90),
        "cv_ratio_mean": cvr.mean(),
        "cv_ratio_p10": np.percentile(cvr, 10),
        "cv_ratio_p90": np.percentile(cvr, 90),
        "L_hat_median_m": np.nanmedian(lhat),
        "L_hat_p10_m": np.nanpercentile(lhat, 10),
        "L_hat_p90_m": np.nanpercentile(lhat, 90),
        "N_eff_lognormal": 1.0 / float(np.mean(rho_lognormal(L, cv)(R))),
    }


def main() -> int:
    rows = []
    for D in DOMAINS_M:
        n = int(round(D / DX))
        for L in L_VALUES:
            for cv in CV_VALUES:
                draws = [draw(n, L, cv, s) for s in range(N_SEEDS)]
                fields = [f for f, _ in draws]
                methods = sorted({m for _, m in draws})
                rows.append({"domain_m": D, "L_m": L, "CV": cv, "D_over_L": D / L,
                             "method": "+".join(methods),
                             **summarise(fields, L, cv, n)})
                print(f"D={D:.0f} L={L:.0f} CV={cv}: done ({rows[-1]['method']})",
                      flush=True)

    # Crop check: 4 km draws cropped to the central 1 km vs direct 1 km draws.
    n4, n1 = int(4000 / DX), int(1000 / DX)
    o = (n4 - n1) // 2
    for L in L_VALUES:
        cv = 2.0
        crops = []
        for s in range(N_SEEDS):
            F, _ = draw(n4, L, cv, 10_000 + s)
            c = F[o:o + n1, o:o + n1]
            crops.append(c / c.sum())  # renormalize the window, as the sweep would
        rows.append({"domain_m": 1000.0, "L_m": L, "CV": cv, "D_over_L": 1000.0 / L,
                     "method": "crop_of_4km", **summarise(crops, L, cv, n1)})

    df = pd.DataFrame(rows)
    out = HERE / "domain_size_fields.csv"
    df.to_csv(out, index=False)
    cols = ["domain_m", "L_m", "CV", "D_over_L", "method", "logvar_ratio_mean",
            "logvar_ratio_theory", "cv_ratio_mean", "cv_ratio_p10", "cv_ratio_p90",
            "L_hat_median_m", "L_hat_p10_m", "L_hat_p90_m", "N_eff_lognormal"]
    print(df[cols].round(3).to_string(index=False))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
