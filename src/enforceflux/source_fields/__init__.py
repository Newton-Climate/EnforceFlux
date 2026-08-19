"""Source-field generation and basis machinery for OSSE (M1)."""
from .lognormal_gp import FieldGrid, LognormalFieldSpec, sample_lognormal_field
from .basis import (
    BasisMapping,
    uniform_coarse_basis,
    polygon_basis,
    project_flux_to_coarse,
    save_mapping,
    load_mapping,
)
from .prior import build_prior_covariance

__all__ = [
    "FieldGrid",
    "LognormalFieldSpec",
    "sample_lognormal_field",
    "BasisMapping",
    "uniform_coarse_basis",
    "polygon_basis",
    "project_flux_to_coarse",
    "save_mapping",
    "load_mapping",
    "build_prior_covariance",
]
