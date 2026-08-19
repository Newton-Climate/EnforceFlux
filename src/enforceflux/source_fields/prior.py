"""Prior covariance builder for the inversion state on the coarse basis."""
from __future__ import annotations

from typing import Literal

import numpy as np

from .basis import BasisMapping
from .lognormal_gp import _correlation


class _CorrShim:
    """Adapter so we can reuse `_correlation` without a full FieldSpec."""

    def __init__(self, model: str, L: float, nu: float = 1.5) -> None:
        self.covariance = model
        self.L_m = L
        self.matern_nu = nu


def build_prior_covariance(
    mapping: BasisMapping,
    sigma_kg_s: float | np.ndarray,
    L_B_m: float,
    model: Literal["exponential", "matern"] = "exponential",
    matern_nu: float = 1.5,
) -> np.ndarray:
    centers = mapping.coarse_centers_m
    n = centers.shape[0]
    diffs = centers[:, None, :] - centers[None, :, :]
    R = np.linalg.norm(diffs, axis=-1)
    C = _correlation(R, _CorrShim(model, L_B_m, matern_nu))
    sigma = np.broadcast_to(np.asarray(sigma_kg_s, dtype=float), (n,))
    S = (sigma[:, None] * sigma[None, :]) * C
    # Symmetrize to kill roundoff-induced asymmetry that eigvalsh would object to.
    S = 0.5 * (S + S.T)
    return S
