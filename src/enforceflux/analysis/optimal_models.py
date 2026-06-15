"""Result containers for optimal-estimation inversion workflows."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class OEResult(NamedTuple):
    """Result of an Optimal Estimation inversion."""

    x_opt: np.ndarray
    x_prior: np.ndarray
    Sx: np.ndarray
    averaging_kernel: np.ndarray
    y_obs: np.ndarray
    y_prior: np.ndarray
    y_opt: np.ndarray
    cost_history: list
    converged: bool
    n_iter: int
    source_names: list | None
