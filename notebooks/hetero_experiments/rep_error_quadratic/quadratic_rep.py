"""Analytical representativeness-error variance as a quadratic form.

For a total-only retrieval with a uniform source template, the retrieved total
under a *perfect* transport operator is (derivation in README.md)

    Q_hat / Q = AK * (w . phi) + (1 - AK) * x_prior / Q

where ``phi_i = N F_i / sum(F)`` is the field relative to its domain mean,
``AK`` is the scalar averaging kernel and ``w`` is the effective footprint
weighting the weighted least squares actually applies,

    c = (g_tot / Se)^T G_fine,     w = c / sum(c),

which reduces to ``g / sum(g)`` for a single observation. Because ``w_u`` is
uniform and ``sum(phi) == N``, ``w . phi - 1 == dw . phi`` with ``dw = w - w_u``,
so the fractional error is ``AK * dw . phi + (AK - 1)`` and

    Var(e) = AK^2 * dw^T Cov(phi) dw.

Three models of ``Cov(phi)`` are provided, all built from the generator in
``enforceflux.source_fields.lognormal_gp`` rather than reimplemented:

* ``loglinear``  — ``sigma^2 R``, the first-order log-space form under test.
* ``lognormal``  — ``expm1(sigma^2 R)``, the exact covariance of ``exp(Z)``
  before the generator's renormalization to ``Q_true``.
* Monte Carlo    — ``sample_lognormal_field`` itself, renormalization included.

Nothing here writes outside this directory.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from enforceflux.source_fields.lognormal_gp import (  # noqa: E402
    FieldGrid,
    LognormalFieldSpec,
    _correlation,  # private, but it is the generator's exact kernel
    sample_lognormal_field,
)

RUNS = ROOT / "runs"


# --------------------------------------------------------------- geometry


def grid_from_truth(path: Path) -> tuple[FieldGrid, dict]:
    """FieldGrid and design attributes exactly as recorded by the generator."""
    with xr.open_dataset(path) as ds:
        a = dict(ds.attrs)
    grid = FieldGrid(nx=int(a["nx"]), ny=int(a["ny"]), dx_m=float(a["dx_m"]),
                     origin_x_m=float(a["origin_x_m"]), origin_y_m=float(a["origin_y_m"]))
    return grid, a


def spec_for(grid: FieldGrid, L_m: float, cv: float, *, Q: float = 0.027778,
             covariance: str = "exponential", seed: int = 0) -> LognormalFieldSpec:
    return LognormalFieldSpec(grid=grid, Q_true_kg_s=Q, L_m=float(L_m), cv=float(cv),
                              covariance=covariance, seed=int(seed))


def lag_matrix(grid: FieldGrid) -> np.ndarray:
    """Cell-centre separations, raveled in the generator's (y, x) order."""
    x, y = grid.cell_centers()
    XX, YY = np.meshgrid(x, y, indexing="xy")
    p = np.column_stack([XX.ravel(), YY.ravel()])
    return np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])


def covariance(spec: LognormalFieldSpec, variant: str) -> np.ndarray:
    """Covariance of phi over the source grid under one of the analytic models."""
    rho = _correlation(lag_matrix(spec.grid), spec)
    s2 = np.log1p(spec.cv**2)
    if variant == "loglinear":
        return s2 * rho
    if variant == "lognormal":
        return np.expm1(s2 * rho)
    raise ValueError(f"unknown covariance variant {variant!r}")


# --------------------------------------------------------------- footprint


@dataclass(frozen=True)
class DesignWeights:
    geometry: str
    n: int
    G_fine: np.ndarray   # (n_obs, N) bLS Jacobian, per kg s-1 in each cell
    w_g: np.ndarray      # (N,) effective footprint weighting, sums to 1
    dw: np.ndarray       # (N,) w_g - 1/N
    ak: float            # scalar averaging kernel of the total-only retrieval

    @property
    def n_cells(self) -> int:
        return self.w_g.size


def design_weights(geometry: str, n: int, inversion_run: str) -> DesignWeights:
    """Effective WLS weighting for one measurement design.

    ``inversion_run`` is any retrieval run of the design: its ``matrices.npz``
    supplies Se, Sa and the template-collapsed Jacobian, and its ``_gp`` twin
    the per-cell bLS Jacobian.
    """
    G = np.load(RUNS / f"{inversion_run}_gp" / "dispersion" / "jacobian.npz")["G"]
    M = np.load(RUNS / inversion_run / "flux" / "matrices.npz")
    N = G.shape[1]
    g_tot = G @ np.full(N, 1.0 / N)
    if not np.allclose(g_tot, M["G"].ravel(), rtol=1e-12, atol=0):
        raise AssertionError(f"{inversion_run}: inversion G is not G_fine @ uniform template")
    a = g_tot / M["Se_diag"]
    c = a @ G
    w = c / c.sum()
    info = float(a @ g_tot)
    ak = info / (info + 1.0 / float(M["Sa_diag"][0]))
    if float(M["x_prior"][0]) != 0.0:
        raise AssertionError("non-zero prior: the (1 - AK) x_prior term is not modelled")
    return DesignWeights(geometry, n, G, w, w - 1.0 / N, ak)


def receptors(inversion_run: str) -> list[dict]:
    path = RUNS / f"{inversion_run}_gp" / "dispersion" / "config.snapshot.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["dispersion"]["receptors"]


# --------------------------------------------------------------- prediction


def sigma_rep2(dw: DesignWeights, cov: np.ndarray) -> float:
    """Predicted variance of the fractional total-flux error."""
    return float(dw.ak**2 * dw.dw @ cov @ dw.dw)


def phi_of(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float).ravel()
    return F.size * F / F.sum()


def pure_error(dw: DesignWeights, F: np.ndarray) -> float:
    """Exact perfect-transport fractional error for one realized field."""
    return float(dw.ak * dw.dw @ phi_of(F) + (dw.ak - 1.0))


def monte_carlo_phi(spec: LognormalFieldSpec, n_draws: int, seed: int) -> tuple[np.ndarray, int]:
    """(n_draws, N) relative fields drawn by the production generator.

    Also returns how many draws took the generator's padded-Cholesky fallback
    (exact covariance, used when circulant embedding is not PSD) — the same
    path the stored realizations took for that condition.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((n_draws, spec.grid.nx * spec.grid.ny))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        for k in range(n_draws):
            out[k] = phi_of(sample_lognormal_field(spec, rng))
    return out, sum("padded Cholesky" in str(w.message) for w in caught)
