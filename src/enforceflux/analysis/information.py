"""Compatibility re-export for information-theoretic analysis utilities.

This module preserves the existing public import path while implementation
is split into smaller focused modules.
"""
from enforceflux.analysis.information_models import (
    FisherResult,
    DofResult,
    PosteriorResult,
    AblationResult,
)
from enforceflux.analysis.information_core import (
    compute_fisher,
    compute_dof,
    compute_posterior,
    analyze_information_content,
)
from enforceflux.analysis.information_ablation import (
    run_ablation_study,
    summarize_ablation,
)

__all__ = [
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
]
