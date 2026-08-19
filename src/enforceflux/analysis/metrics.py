from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricResults:
    fisher_information: np.ndarray
    null_space_dimension: int
    jacobian_rank: int
    posterior_cov: np.ndarray
    posterior_std: np.ndarray
    averaging_kernel: np.ndarray
    condition_number: float


def compute_metrics(g: np.ndarray, fisher: np.ndarray, posterior_cov: np.ndarray, averaging_kernel: np.ndarray, r_cond: float) -> MetricResults:
    g = np.asarray(g)
    fisher = np.asarray(fisher)
    posterior_cov = np.asarray(posterior_cov)
    averaging_kernel = np.asarray(averaging_kernel)

    u, s, v = np.linalg.svd(g, full_matrices=False)
    if s.size == 0:
        rank = 0
    else:
        tol = r_cond * s.max()
        rank = int((s > tol).sum())

    null_space_dimension = g.shape[1] - rank
    condition_number = float(s.max() / s.min()) if s.size > 0 and s.min() > 0 else float("inf")
    posterior_std = np.sqrt(np.diag(posterior_cov))

    return MetricResults(
        fisher_information=fisher,
        null_space_dimension=null_space_dimension,
        jacobian_rank=rank,
        posterior_cov=posterior_cov,
        posterior_std=posterior_std,
        averaging_kernel=averaging_kernel,
        condition_number=condition_number,
    )


# --- heterogeneity OSSE metrics (M3) ---
# Recovery-based metrics from the source-heterogeneity design memo (eqs. 21-23).
# Pandas is imported lazily inside n_min so it does not become a project-wide dep.


def total_emission_error(
    x_hat: np.ndarray, x_true: np.ndarray, areas: np.ndarray
) -> float:
    """Total-emission recovery error E_Q (memo eq. 21).

    E_Q = |sum(x_hat*area) - sum(x_true*area)| / sum(x_true*area).
    ``x_hat`` and ``x_true`` are per-cell fluxes (kg/s/m^2 or kg/s per cell —
    just be consistent with ``areas``); ``areas`` is the matching per-cell area
    array (or all-ones if ``x`` is already in kg/s per cell).
    """
    x_hat = np.asarray(x_hat, dtype=float)
    x_true = np.asarray(x_true, dtype=float)
    areas = np.asarray(areas, dtype=float)
    q_true = float(np.sum(x_true * areas))
    if q_true == 0.0:
        raise ValueError("total_emission_error: Q_true is zero")
    q_hat = float(np.sum(x_hat * areas))
    return abs(q_hat - q_true) / abs(q_true)


def success_probability(e_q_samples: np.ndarray, epsilon: float) -> float:
    """Empirical P(success) = P(E_Q <= epsilon) over realizations (memo eq. 22)."""
    e = np.asarray(e_q_samples, dtype=float)
    if e.size == 0:
        return float("nan")
    return float(np.mean(e <= float(epsilon)))


def n_min(df, epsilon: float, p_star: float, group_cols: list[str]):
    """Smallest network size N per group meeting P(E_Q<=epsilon) >= p_star (eq. 23).

    ``df`` is a pandas DataFrame with columns ``group_cols + ['N', 'e_q']``.
    Returns a DataFrame with columns ``group_cols + ['n_min']``; ``n_min`` is
    NaN for groups that never reach ``p_star`` at any available N.
    """
    import pandas as pd  # lazy import — not a project dep

    if "N" not in df.columns or "e_q" not in df.columns:
        raise ValueError("n_min: df must have columns 'N' and 'e_q'")

    def _reduce(sub: "pd.DataFrame") -> "pd.Series":
        by_n = (
            sub.groupby("N")["e_q"]
            .apply(lambda s: float((s <= float(epsilon)).mean()))
            .sort_index()
        )
        ok = by_n[by_n >= float(p_star)]
        val = float(ok.index.min()) if len(ok) > 0 else float("nan")
        return pd.Series({"n_min": val})

    if group_cols:
        out = df.groupby(list(group_cols), dropna=False).apply(_reduce).reset_index()
    else:
        out = _reduce(df).to_frame().T
    return out
