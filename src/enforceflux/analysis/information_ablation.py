"""Ablation-study workflows built on top of core information metrics."""
import numpy as np

from enforceflux.analysis.information_core import _group_fim, compute_dof, compute_posterior
from enforceflux.analysis.information_models import AblationResult, FisherResult


def run_ablation_study(
    G: np.ndarray,
    Se: np.ndarray,
    Sa: np.ndarray,
    obs_groups: dict[str, np.ndarray],
    source_names: list[str] | None = None,
) -> dict[str, AblationResult]:
    """Run observation ablation study across individual and cumulative group subsets."""
    G = np.asarray(G, dtype=float)
    Se = np.asarray(Se, dtype=float)
    group_names = list(obs_groups.keys())

    scenarios: list[tuple[str, list[str]]] = []
    for name in group_names:
        scenarios.append((name, [name]))
    for i in range(2, len(group_names) + 1):
        active = group_names[:i]
        scenarios.append(("+".join(active), active))

    results: dict[str, AblationResult] = {}
    for key, active_names in scenarios:
        FIM_per_group: dict[str, np.ndarray] = {}
        n_obs_per_group: dict[str, int] = {}
        for name in active_names:
            mask = np.asarray(obs_groups[name], dtype=bool)
            FIM_per_group[name] = _group_fim(G, Se, mask)
            n_obs_per_group[name] = int(mask.sum())

        FIM_total: np.ndarray = sum(FIM_per_group.values())  # type: ignore[assignment]
        evals, evecs = np.linalg.eigh(FIM_total)
        order = np.argsort(evals)[::-1]

        fisher = FisherResult(
            FIM_total=FIM_total,
            FIM_per_group=FIM_per_group,
            eigenvalues=evals[order],
            eigenvectors=evecs[:, order],
            group_names=active_names,
            n_obs_per_group=n_obs_per_group,
        )
        dof = compute_dof(fisher, Sa, source_names=source_names)
        posterior = compute_posterior(fisher, Sa, source_names=source_names)

        results[key] = AblationResult(
            scenario=" + ".join(active_names),
            groups=active_names,
            fisher=fisher,
            dof=dof,
            posterior=posterior,
            dfs_total=dof.dfs_total,
        )

    return results


def summarize_ablation(ablation: dict[str, AblationResult]) -> dict[str, dict]:
    """Return a compact comparison table from an ablation study."""
    table: dict[str, dict] = {}
    for key, res in ablation.items():
        ur = res.posterior.uncertainty_reduction
        table[key] = {
            "dfs_total": res.dfs_total,
            "dfs_per_group": res.dof.dfs_per_group,
            "uncertainty_reduction_mean": float(np.nanmean(ur)),
        }
    return table
