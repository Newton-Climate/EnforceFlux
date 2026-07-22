"""
Instrument operator sub-package.

Encodes 8 real-world methane monitoring instruments (OP, EC, CH, AIR, MSAT,
LP_ESN, IM_LS, BP_GML) with good/challenging/bad operating modes, and
provides the ``InstrumentOperator`` forward model H for OSSE simulations.

Sub-modules
-----------
types    : Type aliases (ObservableType, OperatorType, OperatingMode)
db       : OperatorParams dataclass + INSTRUMENT_DB constant
models   : Instrument deployment dataclass
operator : ObservationResult + TimeSeriesObservationResult + InstrumentOperator
open_path: beam geometry, path integral, and the OP instrument simulation
"""
from enforceflux.instrument.db import INSTRUMENT_DB, OperatorParams
from enforceflux.instrument.models import Instrument
from enforceflux.instrument.operator import (
    InstrumentOperator,
    ObservationResult,
    TimeSeriesObservationResult,
)
from enforceflux.instrument.open_path import (
    OpenPathSeries,
    beam_endpoints,
    beam_samples,
    open_path_instrument,
    path_average,
    path_average_series,
    simulate_open_path,
)
from enforceflux.instrument.types import ObservableType, OperatingMode, OperatorType

__all__ = [
    "INSTRUMENT_DB",
    "Instrument",
    "InstrumentOperator",
    "ObservableType",
    "ObservationResult",
    "OperatingMode",
    "OperatorParams",
    "OperatorType",
    "OpenPathSeries",
    "TimeSeriesObservationResult",
    "beam_endpoints",
    "beam_samples",
    "open_path_instrument",
    "path_average",
    "path_average_series",
    "simulate_open_path",
]
