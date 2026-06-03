from __future__ import annotations

import numpy as np

from enforceflux.core.base import IInversionEngine
from enforceflux.retrieval.inversion import InversionResult, bayesian_linear_inversion


class BayesianInversionEngine(IInversionEngine):
    def invert(
        self,
        g: np.ndarray,
        y: np.ndarray,
        x_prior: np.ndarray,
        s_a: np.ndarray,
        r: np.ndarray,
    ) -> InversionResult:
        return bayesian_linear_inversion(g=g, y=y, x_prior=x_prior, s_a=s_a, r=r)
