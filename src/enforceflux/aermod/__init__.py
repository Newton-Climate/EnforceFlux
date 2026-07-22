"""AERMOD-style dispersion: a differentiable steady-state plume model.

The default near-field transport model for EnforceFlux, replacing the earlier
toy Gaussian operator. The dispersion physics follows AERMOD (Cimorelli et al.
2005) — similarity-scaled turbulence, Briggs plume rise, a bi-Gaussian
convective boundary layer, reflected stable layer — and is implemented in JAX,
so concentrations are differentiable with respect to emissions, geometry, and
meteorology alike.

It is a research reimplementation rather than EPA regulatory AERMOD; see
:mod:`enforceflux.aermod.dispersion` for exactly what is and is not modelled.

Driven entirely from Python::

    from enforceflux.aermod import AermodConfig, AermodModel, Receptor, SurfaceMet

    config = AermodConfig(
        met=[SurfaceMet(wind_speed_m_s=3.0, wind_direction_deg=225.0, stability_class="D")],
        receptors=[Receptor(id="tower", x=500.0, y=500.0, z=3.0)],
        concentration_units="ug_m3_per_g_s",
    )
    model = AermodModel(config)
    g = model.jacobian(sources)              # inversion Jacobian
    field = model.grid_field(sources)        # forward concentration field

The registry plugins ``enforceflux.transport_operator.aermod`` and
``enforceflux.transport_simulation.aermod`` are thin wrappers over exactly this
API.
"""
from enforceflux.aermod.config import (
    AermodConfig,
    DispersionOptions,
    Receptor,
    ReceptorGrid,
    StackParameters,
    SurfaceMet,
)
from enforceflux.aermod.meteorology import MetState, derive_met_state
from enforceflux.aermod.model import (
    AermodModel,
    GridField,
    receptors_from_grid,
    receptors_from_instruments,
)
from enforceflux.aermod.output import write_grid_netcdf

__all__ = [
    "AermodConfig",
    "AermodModel",
    "DispersionOptions",
    "GridField",
    "MetState",
    "Receptor",
    "ReceptorGrid",
    "StackParameters",
    "SurfaceMet",
    "derive_met_state",
    "receptors_from_grid",
    "receptors_from_instruments",
    "write_grid_netcdf",
]
