from __future__ import annotations

import numpy as np
import pytest

from enforceflux.source_fields.basis import uniform_coarse_basis
from enforceflux.source_fields.lognormal_gp import FieldGrid
from enforceflux.source_fields.prior import build_prior_covariance


def _mapping():
    grid = FieldGrid(nx=16, ny=16, dx_m=50.0)
    return uniform_coarse_basis(grid, coarsen=4)


@pytest.mark.parametrize("L_B", [50.0, 500.0, 5000.0])
@pytest.mark.parametrize("model", ["exponential", "matern"])
def test_prior_covariance_psd(L_B, model):
    mapping = _mapping()
    S = build_prior_covariance(mapping, sigma_kg_s=1e-6, L_B_m=L_B, model=model)
    eigs = np.linalg.eigvalsh(S)
    assert eigs.min() >= -1e-10 * max(1.0, eigs.max())


def test_prior_covariance_diagonal_matches_sigma():
    mapping = _mapping()
    sigma = 3.2e-6
    S = build_prior_covariance(mapping, sigma_kg_s=sigma, L_B_m=500.0)
    np.testing.assert_allclose(np.diag(S), sigma * sigma, rtol=1e-12)

    sigmas = np.linspace(1e-7, 5e-6, mapping.W.shape[0])
    S2 = build_prior_covariance(mapping, sigma_kg_s=sigmas, L_B_m=500.0)
    np.testing.assert_allclose(np.diag(S2), sigmas ** 2, rtol=1e-12)
