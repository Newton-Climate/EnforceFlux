#!/usr/bin/env python3
"""Source-field geometry: effective replication, L validation, error decomposition.

Sections 6-7 of the analysis. Three products, none of which runs a simulation.

**field_stats.csv** — effective spatial replication for each of the nine (CV, L)
conditions. The generator is known in closed form
(``src/enforceflux/source_fields/lognormal_gp.py``), so these are analytic:

* ``N_eff_simple = A / L^2`` — the first-order guess under test.
* ``N_eff_logfield`` — the textbook effective sample size of the *log* field,
  ``1 / mean_ij rho(r_ij)`` with ``rho(r) = exp(-r/L)`` summed over the actual
  25x25 grid. This is what ``A / A_corr`` converges to when L << domain, but it
  stays finite and >= 1 on a bounded domain, which ``A/L^2`` does not.
* ``N_eff_lognormal`` — the same functional applied to the field that is
  actually emitted. For ``F = exp(Z)`` with ``Var(Z) = sigma^2 = log1p(CV^2)``,
  ``rho_F(r) = expm1(sigma^2 exp(-r/L)) / CV^2``. It depends on CV as well as L,
  which is what makes it a genuine alternative rather than a relabelling of L.
* ``L_hat_m`` — the e-folding scale recovered by fitting ``exp(-r/L)`` to the
  empirical autocorrelation of ``log F_true`` pooled over the eight seeds. This
  is the check that L means in the data what the generator says it means.

The empirical autocorrelation is taken of ``log F``, not ``F``: the generator
renormalizes every realization so ``sum(F * area) == Q_true`` exactly, which
pins the domain mean and leaves the realized field's own domain average with
identically zero variance. That fact is itself a result and is carried into the
report — the retrieval error cannot come from sampling the source *total*,
because the total does not vary.

**footprint_scales.csv** — the footprint correlation length ``L_H`` for each of
the eight measurement designs, via
``enforceflux.analysis.footprint_scale.footprint_correlation_length`` applied to
the bLS Jacobian. The Jacobian is verified here to be identical across L, CV and
seed, so there are exactly eight values.

**error_decomposition.csv** — one row per inversion. The retrieval is an
unconstrained weighted least square (verified below to reproduce ``x_opt`` to
machine precision), so replacing the LES-generated observations with the bLS
forward model of the same true field gives the retrieval that *would* have been
obtained with a perfect transport operator:

    Q_pred = [ (g/Se) . (J e_true) + x_prior/Sa ] / [ (g/Se) . g + 1/Sa ]

``e_pure = (Q_pred - Q_true)/Q_true`` is pure source representativeness — the
footprint sampling a heterogeneous field — and ``e_transport = e_total - e_pure``
is everything the bLS-versus-LES operator mismatch contributes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = ROOT / "runs"
sys.path.insert(0, str(ROOT / "src"))

from enforceflux.analysis.footprint_scale import footprint_correlation_length  # noqa: E402
from enforceflux.source_fields.lognormal_gp import FieldGrid  # noqa: E402

A_M2 = 1.0e6
GRID_N = 25
GRID_DX = 40.0

INV = pd.read_csv(HERE / "inversions.csv")
_NATURE = (
    INV[["L_m", "CV", "seed", "nature_run"]].drop_duplicates()
    .set_index(["L_m", "CV", "seed"])["nature_run"].to_dict()
)


# --------------------------------------------------------------- geometry


def _lag_matrix(nx: int, ny: int, dx: float) -> np.ndarray:
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    XX, YY = np.meshgrid(x, y, indexing="xy")
    p = np.column_stack([XX.ravel(), YY.ravel()])
    return np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])


def n_eff_from_rho(rho, R: np.ndarray) -> float:
    """1 / mean(rho) over all cell pairs = effective independent samples.

    ``Var(spatial mean)/Var(point) == mean_ij rho(r_ij)`` for a stationary field
    on a regular grid, so the reciprocal is what the domain is worth.
    """
    return float(1.0 / np.mean(rho(R)))


def rho_log(L: float):
    return lambda r: np.exp(-r / L)


def rho_lognormal(L: float, cv: float):
    s2 = np.log1p(cv**2)
    # expm1(s2) == cv**2 exactly, so cv**2 is the correct normalization.
    return lambda r: np.expm1(s2 * np.exp(-r / L)) / cv**2


def a_corr_lognormal(L: float, cv: float) -> float:
    from scipy.integrate import quad

    f = rho_lognormal(L, cv)
    val, _ = quad(lambda r: float(f(np.array(r))) * r, 0.0, 60.0 * L, limit=400)
    return float(2.0 * np.pi * val)


# --------------------------------------------------------------- fields


def load_field(L: float, cv: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(F_true in kg s-1 m-2, cell areas in m2) for one realization."""
    name = _NATURE[(L, cv, seed)]
    with xr.open_dataset(RUNS / name / "dispersion" / "truth_field.nc") as ds:
        return ds["F_true"].values.copy(), ds["cell_area_m2"].values.copy()


def empirical_acf(fields: list[np.ndarray], dx: float, nbins: int = 24):
    """Isotropic autocorrelation of log(field), pooled over realizations."""
    ny, nx = fields[0].shape
    R = _lag_matrix(nx, ny, dx).ravel()
    edges = np.linspace(0.0, 0.6 * nx * dx, nbins + 1)
    idx = np.digitize(R, edges) - 1
    keep = (idx >= 0) & (idx < nbins)
    num = np.zeros(nbins)
    den = np.zeros(nbins)
    var = 0.0
    for F in fields:
        a = np.log(F).ravel()
        a = a - a.mean()
        prod = np.outer(a, a).ravel()
        num += np.bincount(idx[keep], weights=prod[keep], minlength=nbins)
        den += np.bincount(idx[keep], minlength=nbins)
        var += float(a.var())
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, (num / np.maximum(den, 1)) / (var / len(fields))


def fit_efolding(centres: np.ndarray, acf: np.ndarray) -> float:
    """Least-squares e-folding length from log(acf) over the positive lags.

    Restricted to lags where the empirical acf is still above 0.1; beyond that
    the finite-domain bias of the estimator dominates.
    """
    m = (acf > 0.1) & (centres > 0)
    if m.sum() < 3:
        return np.nan
    slope = np.polyfit(centres[m], np.log(acf[m]), 1)[0]
    return float(-1.0 / slope) if slope < 0 else np.nan


# --------------------------------------------------------------- products


def build_field_stats() -> pd.DataFrame:
    R = _lag_matrix(GRID_N, GRID_N, GRID_DX)
    rows = []
    for (L, cv), _ in INV.groupby(["L_m", "CV"]):
        fields = [load_field(L, cv, s)[0] for s in range(8)]
        centres, acf = empirical_acf(fields, GRID_DX)
        totals = np.array([float((F * load_field(L, cv, s)[1]).sum())
                           for s, F in enumerate(fields)])
        rows.append({
            "L_m": L,
            "CV": cv,
            "sigma_log": float(np.sqrt(np.log1p(cv**2))),
            "N_eff_simple": A_M2 / L**2,
            "N_eff_Acorr_log": A_M2 / (2.0 * np.pi * L**2),
            "N_eff_Acorr_lognormal": A_M2 / a_corr_lognormal(L, cv),
            "N_eff_logfield": n_eff_from_rho(rho_log(L), R),
            "N_eff_lognormal": n_eff_from_rho(rho_lognormal(L, cv), R),
            "L_hat_m": fit_efolding(centres, acf),
            "cv_realized_mean": float(np.mean([F.std() / F.mean() for F in fields])),
            "source_total_sd_kg_s": float(totals.std(ddof=1)),
        })
    return pd.DataFrame(rows).sort_values(["L_m", "CV"]).reset_index(drop=True)


def build_footprint_scales() -> pd.DataFrame:
    fgrid = FieldGrid(nx=GRID_N, ny=GRID_N, dx_m=GRID_DX,
                      origin_x_m=-500.0, origin_y_m=-500.0)
    rows = []
    for geom in ("open_path", "point"):
        for n in (1, 2, 3, 4):
            sel = INV[(INV.geometry == geom) & (INV["n"] == n)].iloc[0]
            d = np.load(RUNS / f"{sel.run}_gp" / "dispersion" / "jacobian.npz",
                        allow_pickle=True)
            rows.append({
                "geometry": geom,
                "n": n,
                "L_H_m": footprint_correlation_length(d["G"], fgrid),
                "path_length_total_m": sel.path_length_total_m,
                "example_run": sel.run,
            })
    return pd.DataFrame(rows)


def _jacobian_cache() -> dict[tuple[str, int], np.ndarray]:
    """One bLS Jacobian per design, after verifying it is design-only."""
    cache: dict[tuple[str, int], np.ndarray] = {}
    for (geom, n), grp in INV.groupby(["geometry", "n"]):
        ref = None
        for run in grp.run:
            G = np.load(RUNS / f"{run}_gp" / "dispersion" / "jacobian.npz")["G"]
            if ref is None:
                ref = G
            elif not np.allclose(ref, G):
                raise AssertionError(
                    f"bLS Jacobian varies within design ({geom}, n={n}) at {run}; "
                    "the operator-reuse assumption in SWEEPS.md does not hold."
                )
        cache[(geom, n)] = ref
        print(f"  Jacobian constant across {len(grp)} runs for {geom} n={n}")
    return cache


def build_decomposition(jac: dict) -> pd.DataFrame:
    fields = {k: load_field(*k) for k in _NATURE}
    rows = []
    wls_err = 0.0
    for r in INV.itertuples():
        M = np.load(RUNS / r.run / "flux" / "matrices.npz")
        g = M["G"].ravel()
        Se = M["Se_diag"]
        Sa = M["Sa_diag"][0]
        xp = M["x_prior"][0]
        den = float((g / Se) @ g + 1.0 / Sa)

        # The retrieval is an unconstrained WLS; confirm before relying on it.
        x_check = float(((g / Se) @ M["y_obs"] + xp / Sa) / den)
        wls_err = max(wls_err, abs(x_check - float(M["x_opt"][0])))

        F, area = fields[(r.L_m, r.CV, r.seed)]
        y_bls = jac[(r.geometry, r.n)] @ (F * area).ravel()
        q_pred = float(((g / Se) @ y_bls + xp / Sa) / den)

        e_pure = (q_pred - r.Q_true_kg_s) / r.Q_true_kg_s
        rows.append({
            "run": r.run,
            "Q_pred_no_transport_error_kg_s": q_pred,
            "e_pure_representativeness": e_pure,
            "e_transport": r.e_signed - e_pure,
        })
    print(f"  WLS reconstruction reproduces x_opt to {wls_err:.3e} kg/s "
          f"(max over {len(INV)} runs)")
    if wls_err > 1e-12:
        raise AssertionError("retrieval is not the unconstrained WLS assumed here")
    return pd.DataFrame(rows)


def main() -> int:
    fs = build_field_stats()
    fs.to_csv(HERE / "field_stats.csv", index=False)
    print("=== field_stats.csv ===")
    print(fs.round(3).to_string(index=False))

    fp = build_footprint_scales()
    fp.to_csv(HERE / "footprint_scales.csv", index=False)
    print("\n=== footprint_scales.csv ===")
    print(fp.round(2).to_string(index=False))

    print("\n=== verifying operator reuse ===")
    jac = _jacobian_cache()
    print("\n=== error decomposition ===")
    dec = build_decomposition(jac)
    dec.to_csv(HERE / "error_decomposition.csv", index=False)
    j = INV.merge(dec, on="run")
    summ = j.groupby(["geometry", "n"])[
        ["e_signed", "e_pure_representativeness", "e_transport"]
    ].std().round(4)
    print("\nrun-to-run SD of each component, pooled over CV and L:")
    print(summ.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
