"""
Backward-compatibility shim.

The instrument operator module has been moved to the ``enforceflux.instrument``
sub-package.  This module re-exports everything so that existing imports of the
form ``from enforceflux.models.instrument import ...`` continue to work.
"""
from enforceflux.instrument import (  # noqa: F401
    INSTRUMENT_DB,
    Instrument,
    InstrumentOperator,
    ObservableType,
    ObservationResult,
    OperatingMode,
    OperatorParams,
    OperatorType,
)

__all__ = [
    "INSTRUMENT_DB",
    "Instrument",
    "InstrumentOperator",
    "ObservableType",
    "ObservationResult",
    "OperatingMode",
    "OperatorParams",
    "OperatorType",
]
