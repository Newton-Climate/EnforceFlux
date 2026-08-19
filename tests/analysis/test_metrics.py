"""Tests for heterogeneity OSSE recovery metrics (M3)."""

import numpy as np
import pandas as pd
import pytest

from enforceflux.analysis.metrics import (
    n_min,
    success_probability,
    total_emission_error,
)


def test_e_q_zero_when_perfect_recovery():
    rng = np.random.default_rng(0)
    x_true = rng.uniform(0.1, 1.0, size=32)
    areas = np.full_like(x_true, 25.0**2)
    assert total_emission_error(x_true, x_true, areas) == pytest.approx(0.0, abs=1e-15)


def test_success_probability_matches_empirical_indicator():
    e = np.array([0.01, 0.03, 0.05, 0.07, 0.11, 0.20])
    eps = 0.05
    # Indicator count: {0.01, 0.03, 0.05} are <= 0.05 -> 3/6
    assert success_probability(e, eps) == pytest.approx(3 / 6)
    assert success_probability(e, 0.0) == pytest.approx(0.0)
    assert success_probability(e, 1.0) == pytest.approx(1.0)


def test_n_min_monotonic_in_p_star():
    # Synthesize a table where higher N drives lower e_q.
    rng = np.random.default_rng(0)
    rows = []
    for L in [100.0, 500.0]:
        for N in [4, 8, 16, 32, 64, 128, 256]:
            # e_q shrinks with N, plus small noise → nested monotonicity.
            base = 0.20 * (1.0 / np.sqrt(N))
            for _ in range(200):
                rows.append(
                    {
                        "L": L,
                        "N": N,
                        "e_q": max(0.0, base + 0.005 * rng.standard_normal()),
                    }
                )
    df = pd.DataFrame(rows)
    tbl_low = n_min(df, epsilon=0.05, p_star=0.5, group_cols=["L"])
    tbl_high = n_min(df, epsilon=0.05, p_star=0.9, group_cols=["L"])
    for L in [100.0, 500.0]:
        n_low = float(tbl_low.loc[tbl_low["L"] == L, "n_min"].iloc[0])
        n_high = float(tbl_high.loc[tbl_high["L"] == L, "n_min"].iloc[0])
        assert n_high >= n_low
