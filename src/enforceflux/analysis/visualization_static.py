"""Static plotting utilities for forward operator and inversion summaries."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from enforceflux.analysis._viz_base import _make_fig, _resolve


def plot_forward_operator(
    G: np.ndarray,
    source_names: Sequence[str] | None = None,
    obs_names: Sequence[str] | None = None,
    ax=None,
    cmap: str = "RdBu_r",
    title: str = "Forward Operator G",
    figsize: tuple = (8, 5),
):
    """Heatmap of the (m x n) forward operator / Jacobian."""
    fig, ax = _resolve(ax) if ax is not None else _make_fig(figsize)
    G = np.asarray(G, dtype=float)

    vmax = np.max(np.abs(G)) or 1.0
    im = ax.imshow(G, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Sensitivity")

    if source_names is not None:
        ax.set_xticks(range(len(source_names)))
        ax.set_xticklabels(list(source_names), rotation=45, ha="right", fontsize=8)
    if obs_names is not None:
        ax.set_yticks(range(len(obs_names)))
        ax.set_yticklabels(list(obs_names), fontsize=8)

    ax.set_xlabel("Source")
    ax.set_ylabel("Observation")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_averaging_kernel(
    A: np.ndarray,
    source_names: Sequence[str] | None = None,
    ax=None,
    cmap: str = "RdBu_r",
    title: str = "Averaging Kernel A",
    figsize: tuple = (6, 5),
):
    """Heatmap of the (n x n) averaging kernel."""
    fig, ax = _resolve(ax) if ax is not None else _make_fig(figsize)
    A = np.asarray(A, dtype=float)

    vmax = min(1.0, float(np.max(np.abs(A))) or 1.0)
    im = ax.imshow(A, aspect="equal", cmap=cmap, vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax)
    cb.ax.set_ylabel("Sensitivity")

    if source_names is not None:
        ticks = list(range(len(source_names)))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(list(source_names), rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(list(source_names), fontsize=8)

    ax.set_xlabel("True source")
    ax.set_ylabel("Retrieved source")
    ax.set_title(f"{title}\nDFS = {float(np.trace(A)):.2f}")
    fig.tight_layout()
    return fig, ax


def plot_dfs_per_source(
    dfs_per_source: np.ndarray,
    source_names: Sequence[str] | None = None,
    ax=None,
    color: str = "steelblue",
    title: str = "Degrees of Freedom for Signal per Source",
    figsize: tuple | None = None,
):
    """Bar chart of per-source DFS (diagonal of the averaging kernel)."""
    dfs = np.asarray(dfs_per_source, dtype=float)
    n = len(dfs)
    labels = list(source_names) if source_names is not None else [str(i) for i in range(n)]

    figsize = figsize or (max(6, n * 0.5), 4)
    fig, ax = _resolve(ax) if ax is not None else _make_fig(figsize)

    ax.bar(range(n), dfs, color=color, edgecolor="white", linewidth=0.5)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8, label="DFS = 1 (fully constrained)")
    ax.axhline(0.5, color="orange", linestyle=":", linewidth=0.8, label="DFS = 0.5")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("DFS")
    ax.set_ylim(bottom=0)
    ax.set_title(f"{title}\nTotal DFS = {float(dfs.sum()):.2f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_posterior_uncertainty(
    prior_sigma: np.ndarray,
    posterior_sigma: np.ndarray,
    source_names: Sequence[str] | None = None,
    ax=None,
    title: str = "Prior vs Posterior Uncertainty",
    figsize: tuple | None = None,
):
    """Grouped bar chart comparing prior sigma and posterior sigma per source."""
    prior_sigma = np.asarray(prior_sigma, dtype=float)
    posterior_sigma = np.asarray(posterior_sigma, dtype=float)
    n = len(prior_sigma)
    labels = list(source_names) if source_names is not None else [str(i) for i in range(n)]
    x = np.arange(n)
    w = 0.35

    figsize = figsize or (max(6, n * 0.6), 4)
    fig, ax = _resolve(ax) if ax is not None else _make_fig(figsize)

    ax.bar(x - w / 2, prior_sigma, w, label="Prior sigma", color="slategray", alpha=0.75)
    ax.bar(x + w / 2, posterior_sigma, w, label="Posterior sigma", color="steelblue", alpha=0.9)

    ur_mean = float(np.nanmean(1.0 - posterior_sigma / (prior_sigma + 1e-300)))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Standard deviation")
    ax.set_title(f"{title}\nMean uncertainty reduction = {ur_mean * 100:.1f} %")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_flux_comparison(
    x_prior: np.ndarray,
    x_opt: np.ndarray,
    x_true: np.ndarray | None = None,
    source_names: Sequence[str] | None = None,
    posterior_sigma: np.ndarray | None = None,
    ax=None,
    title: str = "Prior vs Posterior Fluxes",
    figsize: tuple | None = None,
):
    """Grouped bar chart of prior, posterior, and (optionally) true fluxes."""
    x_prior = np.asarray(x_prior, dtype=float)
    x_opt = np.asarray(x_opt, dtype=float)
    n = len(x_prior)
    labels = list(source_names) if source_names is not None else [str(i) for i in range(n)]
    x_pos = np.arange(n)

    n_bars = 3 if x_true is not None else 2
    w = 0.22 if n_bars == 3 else 0.35
    offsets = np.linspace(-(n_bars - 1) * w / 2, (n_bars - 1) * w / 2, n_bars)

    figsize = figsize or (max(6, n * 0.7), 4)
    fig, ax = _resolve(ax) if ax is not None else _make_fig(figsize)

    ax.bar(x_pos + offsets[0], x_prior, w * 0.9,
           label="Prior", color="slategray", alpha=0.75)
    ax.bar(x_pos + offsets[1], x_opt, w * 0.9,
           label="Posterior", color="steelblue", alpha=0.9,
           yerr=posterior_sigma, capsize=3 if posterior_sigma is not None else 0,
           error_kw={"linewidth": 1.0})
    if x_true is not None:
        ax.bar(x_pos + offsets[2], np.asarray(x_true), w * 0.9,
               label="True", color="forestgreen", alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Flux")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_concentration_timeseries(
    y_obs: np.ndarray,
    y_prior: np.ndarray | None = None,
    y_opt: np.ndarray | None = None,
    obs_names: Sequence[str] | None = None,
    ax=None,
    title: str = "Observed vs Simulated Concentrations",
    figsize: tuple | None = None,
):
    """Scatter + line plot of observed, prior, and posterior model concentrations."""
    y_obs = np.asarray(y_obs, dtype=float)
    m = len(y_obs)
    x_idx = np.arange(m)

    figsize = figsize or (max(8, m * 0.3), 4)
    fig, ax = _resolve(ax) if ax is not None else _make_fig(figsize)

    ax.scatter(x_idx, y_obs, s=20, color="black", zorder=4, label="Observed")
    if y_prior is not None:
        ax.plot(x_idx, np.asarray(y_prior), "--", color="slategray",
                label="Prior", linewidth=1.2)
    if y_opt is not None:
        ax.plot(x_idx, np.asarray(y_opt), "-", color="steelblue",
                label="Posterior", linewidth=1.8)

    if obs_names is not None and len(obs_names) == m:
        ax.set_xticks(x_idx)
        ax.set_xticklabels(list(obs_names), rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Observation index")
    ax.set_ylabel("Concentration / mixing ratio")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig, ax
