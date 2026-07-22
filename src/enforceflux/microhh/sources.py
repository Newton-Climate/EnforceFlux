"""MicroHH source and receptor types.

Mirrors :mod:`enforceflux.flexpart.sources`: MicroHH keeps its own light source
representation rather than reusing the planar :class:`enforceflux.models.source.Source`,
because an LES point source needs a finite Gaussian *blob* size (``sigma_*``)
and is specified in lon/lat like the FLEXPART configs.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MicroHHPointSource:
    """A finite-size scalar release for MicroHH's ``[source]`` module.

    ``emission_rate_kg_s`` is the *physical* release rate. Because a passive
    scalar is linear in emission, cases are typically run at a unit rate and
    rescaled afterwards; ``emission_rate_kg_s`` records the intended physical
    value for that scaling and for provenance.
    """

    id: str
    lon: float
    lat: float
    alt_m: float
    emission_rate_kg_s: float
    # Gaussian blob half-widths (m) for the emission kernel — MicroHH sigma_x/y/z.
    sigma_x_m: float = 15.0
    sigma_y_m: float = 15.0
    sigma_z_m: float = 5.0


@dataclass
class MicroHHReceptor:
    """A point receptor sampled via MicroHH's ``[column]`` output.

    One receptor → one column time series of the scalar, which the instrument
    operator later weights/aggregates into an observation.
    """

    id: str
    lon: float
    lat: float
    alt_m: float = 10.0
