"""Diagnostics and multi-panel plotting utilities for inversion analysis."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from enforceflux.analysis._viz_base import _make_fig, _require_mpl, _resolve, mticker, plt
from enforceflux.analysis.visualization_static import (
    plot_averaging_kernel,
    plot_concentration_timeseries,
    plot_dfs_per_source,
    plot_flux_comparison,
    plot_posterior_uncertainty,
)


def plot_eigenspectrum(
    eigenvalues: np.ndarray,
    ax=None,
    color: str = "steelblue",
    title: str = "FIM Eigenvalue Spectrum",
    figsize: tuple = (6, 4),
):
    """Semilogy plot of FIM eigenvalues in descending order."""
    fig, ax = _resolve(ax) if ax is not None else _make_fig(figsize)
    evals = np.sort(np.asarray(eigenvalues, dtype=float))[::-1]
    pos_mask = evals > 0

    ax.semilogy(np.arange(1, len(evals) + 1),
                np.where(pos_mask, evals, np.nan),
                "o-", color=color, markersize=5, linewidth=1.5, label="Eigenvalue")
    if pos_mask.any():
        ax.axhline(float(evals[pos_mask].min()), color="gray", linestyle=":",
                   linewidth=0.8, label="Min positive")
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_cost_history(
    cost_history: list | np.ndarray,
    ax=None,
    color: str = "steelblue",
    title: str = "LM Convergence",
    figsize: tuple = (6, 4),
):
    """Line plot of the OE cost J(x) across LM iterations."""
    fig, ax = _resolve(ax) if ax is not None else _make_fig(figsize)
    cost = np.asarray(cost_history, dtype=float)
    ax.plot(np.arange(1, len(cost) + 1), cost, "o-", color=color, markersize=5)
    ax.set_xlabel("LM iteration")
    ax.set_ylabel("Cost J(x)")
    ax.set_title(title)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    return fig, ax


def plot_correlation_matrix(
    correlation_matrix: np.ndarray,
    source_names: Sequence[str] | None = None,
    ax=None,
    cmap: str = "RdBu_r",
    title: str = "Posterior Correlation Matrix",
    figsize: tuple = (6, 5),
):
    """Heatmap of the posterior parameter correlation matrix."""
    fig, ax = _resolve(ax) if ax is not None else _make_fig(figsize)
    corr = np.asarray(correlation_matrix, dtype=float)

    im = ax.imshow(corr, aspect="equal", cmap=cmap, vmin=-1.0, vmax=1.0,
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Correlation")

    if source_names is not None:
        ticks = list(range(len(source_names)))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(list(source_names), rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(list(source_names), fontsize=8)

    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_ablation_comparison(
    ablation: dict,
    ax=None,
    color: str = "steelblue",
    title: str = "DFS by Observation Scenario",
    figsize: tuple | None = None,
):
    """Horizontal bar chart comparing total DFS across ablation scenarios."""
    fig, ax = _resolve(ax) if ax is not None else _make_fig(figsize or (7, max(3, len(ablation) * 0.5 + 1)))

    labels = list(ablation.keys())
    dfs_vals = [res.dfs_total for res in ablation.values()]

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, dfs_vals, color=color, alpha=0.85, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Total DFS")
    ax.set_title(title)
    ax.invert_yaxis()

    for i, v in enumerate(dfs_vals):
        ax.text(v + 0.01 * max(dfs_vals), i, f"{v:.2f}", va="center", fontsize=8)

    fig.tight_layout()
    return fig, ax


def plot_inversion_summary(
    oe_result,
    fisher=None,
    dof=None,
    posterior=None,
    source_names: Sequence[str] | None = None,
    figsize: tuple = (14, 9),
):
    """Six-panel inversion summary figure."""
    _require_mpl()
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    snames = source_names or oe_result.source_names

    x_true = getattr(oe_result, "x_true", None)
    post_sigma = (np.sqrt(np.diag(oe_result.Sx))
                  if oe_result.Sx is not None else None)
    plot_flux_comparison(
        oe_result.x_prior, oe_result.x_opt,
        x_true=x_true,
        source_names=snames,
        posterior_sigma=post_sigma,
        ax=axes[0, 0],
        title="Fluxes: prior vs posterior",
    )

    if posterior is not None:
        plot_posterior_uncertainty(
            posterior.prior_sigma, posterior.posterior_sigma,
            source_names=snames,
            ax=axes[0, 1],
        )
    else:
        axes[0, 1].set_visible(False)

    if dof is not None:
        plot_dfs_per_source(
            dof.dfs_per_source,
            source_names=snames,
            ax=axes[0, 2],
        )
    else:
        axes[0, 2].set_visible(False)

    plot_averaging_kernel(
        oe_result.averaging_kernel,
        source_names=snames,
        ax=axes[1, 0],
    )

    if fisher is not None:
        plot_eigenspectrum(fisher.eigenvalues, ax=axes[1, 1])
    elif oe_result.cost_history:
        plot_cost_history(oe_result.cost_history, ax=axes[1, 1])
    else:
        axes[1, 1].set_visible(False)

    plot_concentration_timeseries(
        oe_result.y_obs,
        oe_result.y_prior,
        oe_result.y_opt,
        ax=axes[1, 2],
        title="Observation fit",
    )

    fig.suptitle("Inversion Summary", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig, axes
