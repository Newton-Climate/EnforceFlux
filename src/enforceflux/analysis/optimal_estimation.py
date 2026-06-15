"""Compatibility re-export for optimal-estimation inversion utilities.

This module preserves the historical import path while implementation is split
into smaller focused files.
"""
from enforceflux.analysis.optimal_models import OEResult
from enforceflux.analysis.optimal_nonlinear import optimize_oe
from enforceflux.analysis.optimal_linear import oe_from_linear

__all__ = [
    "OEResult",
    "optimize_oe",
    "oe_from_linear",
]
