"""
enforceflux.analysis — optimal estimation, information metrics, and visualization.

Submodules
----------
information
    Fisher Information Matrix, averaging kernel, degrees of freedom for signal
    (DFS), posterior covariance, and observation ablation studies.

optimal_estimation
    Levenberg-Marquardt nonlinear OE inversion (``optimize_oe``) and a linear
    Bayesian wrapper (``oe_from_linear``).

visualization
    Matplotlib-based plots for forward operators, averaging kernels, DFS bars,
    flux comparisons, concentration timeseries, and multi-panel summaries.
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
from enforceflux.analysis.optimal_estimation import (
    OEResult,
    optimize_oe,
    oe_from_linear,
)
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
    # optimal_estimation
    "OEResult",
    "optimize_oe",
    "oe_from_linear",
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
