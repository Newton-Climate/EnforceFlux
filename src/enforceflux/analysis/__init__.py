"""
enforceflux.analysis — information metrics and visualization.

Submodules
----------
information
    Fisher Information Matrix, averaging kernel, degrees of freedom for signal
    (DFS), posterior covariance, and observation ablation studies.

visualization
    Matplotlib-based plots for forward operators, averaging kernels, DFS bars,
    flux comparisons, concentration timeseries, and multi-panel summaries.

Optimal-estimation and Bayesian inversion algorithms moved to
``enforceflux.inversion``.
"""
from enforceflux.analysis.information import (
    FisherResult,
    DofResult,
    PosteriorResult,
    AblationResult,
    compute_fisher,
    compute_dof,
    compute_posterior,
    analyze_information_content,
    run_ablation_study,
    summarize_ablation,
)
from enforceflux.analysis.metrics import MetricResults, compute_metrics
from enforceflux.analysis.visualization import (
    plot_forward_operator,
    plot_averaging_kernel,
    plot_dfs_per_source,
    plot_posterior_uncertainty,
    plot_flux_comparison,
    plot_concentration_timeseries,
    plot_simulation_heatmap,
    create_simulation_movie,
    load_simulation_netcdf,
    plot_simulation_heatmap_from_netcdf,
    create_simulation_movie_from_netcdf,
    plot_eigenspectrum,
    plot_cost_history,
    plot_correlation_matrix,
    plot_ablation_comparison,
    plot_inversion_summary,
)
from enforceflux.analysis.wind_rose import (
    build_wind_rose,
    plot_wind_rose,
    plot_wind_rose_from_netcdf,
)

__all__ = [
    # information
    "FisherResult",
    "DofResult",
    "PosteriorResult",
    "AblationResult",
    "compute_fisher",
    "compute_dof",
    "compute_posterior",
    "analyze_information_content",
    "run_ablation_study",
    "summarize_ablation",
    # metrics
    "MetricResults",
    "compute_metrics",
    # visualization
    "plot_forward_operator",
    "plot_averaging_kernel",
    "plot_dfs_per_source",
    "plot_posterior_uncertainty",
    "plot_flux_comparison",
    "plot_concentration_timeseries",
    "plot_simulation_heatmap",
    "create_simulation_movie",
    "load_simulation_netcdf",
    "plot_simulation_heatmap_from_netcdf",
    "create_simulation_movie_from_netcdf",
    "plot_eigenspectrum",
    "plot_cost_history",
    "plot_correlation_matrix",
    "plot_ablation_comparison",
    "plot_inversion_summary",
    "build_wind_rose",
    "plot_wind_rose",
    "plot_wind_rose_from_netcdf",
]
