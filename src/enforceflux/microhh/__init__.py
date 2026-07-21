"""MicroHH LES integration for plume-scale transport.

Mirrors the FLEXPART integration: a YAML-driven config (:mod:`sim_config`), a
case-input writer (:mod:`case`), a subprocess runner (:mod:`runner`), and a
column-output reader (:mod:`output`). The MicroHH binary is added later; every
piece here supports a ``dry_run`` path so cases can be generated and inspected
first.
"""

from enforceflux.microhh.case import (
    build_ini,
    initial_profiles,
    write_case,
    write_input_nc,
)
from enforceflux.microhh.geometry import BoxProjection
from enforceflux.microhh.output import (
    ReceptorSeries,
    read_cross_xy,
    read_cross_xz,
    read_receptor_series,
)
from enforceflux.microhh.runner import MicroHHRunner, MicroHHRunResult
from enforceflux.microhh.sim_config import (
    BoxGrid,
    Forcing,
    MicroHHConfig,
    load_microhh_config,
)
from enforceflux.microhh.sources import MicroHHPointSource, MicroHHReceptor
from enforceflux.microhh.units import (
    gaussian_plume_ground_conc,
    mixing_ratio_to_mass_conc,
    mixing_ratio_to_ppb,
)

__all__ = [
    "BoxGrid",
    "BoxProjection",
    "Forcing",
    "MicroHHConfig",
    "MicroHHPointSource",
    "MicroHHReceptor",
    "MicroHHRunner",
    "MicroHHRunResult",
    "ReceptorSeries",
    "build_ini",
    "gaussian_plume_ground_conc",
    "initial_profiles",
    "mixing_ratio_to_mass_conc",
    "mixing_ratio_to_ppb",
    "load_microhh_config",
    "read_cross_xy",
    "read_cross_xz",
    "read_receptor_series",
    "write_case",
    "write_input_nc",
]
