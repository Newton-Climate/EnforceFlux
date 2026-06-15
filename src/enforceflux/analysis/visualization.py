"""Compatibility re-export for visualization utilities.

This module remains as the public import surface while implementation is split
across smaller files for maintainability.
"""
from enforceflux.analysis.visualization_static import (
    plot_forward_operator,
    plot_averaging_kernel,
    plot_dfs_per_source,
    plot_posterior_uncertainty,
    plot_flux_comparison,
    plot_concentration_timeseries,
)
from enforceflux.analysis.visualization_simulation import (
    plot_simulation_heatmap,
    create_simulation_movie,
    load_simulation_netcdf,
    plot_simulation_heatmap_from_netcdf,
    create_simulation_movie_from_netcdf,
)
from enforceflux.analysis.visualization_diagnostics import (
    plot_eigenspectrum,
    plot_cost_history,
    plot_correlation_matrix,
    plot_ablation_comparison,
    plot_inversion_summary,
)

__all__ = [
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
]
