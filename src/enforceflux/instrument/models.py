"""Instrument deployment dataclass."""
import math
from dataclasses import dataclass, field

import numpy as np

from enforceflux.instrument.db import INSTRUMENT_DB, OperatorParams
from enforceflux.instrument.types import ObservableType, OperatingMode, OperatorType


@dataclass
class Instrument:
    """
    A single instrument deployment.

    ``tech_id`` identifies the instrument type in ``INSTRUMENT_DB``; ``mode``
    selects the operating-condition row (good / challenging / bad).
    """

    id: str
    tech_id: str   # key into INSTRUMENT_DB, e.g. "LP_ESN", "OP", "EC"
    x: float       # domain x-coordinate (m or °lon when CRS is WGS-84)
    y: float       # domain y-coordinate (m or °lat when CRS is WGS-84)
    z: float = 0.0
    mode: OperatingMode = "good"

    # Line-integral geometry for OP / LP_ESN path receptors
    path_length_m: float = 200.0
    path_bearing_deg: float = 0.0    # degrees clockwise from north

    # EC turbulent-footprint geometry
    footprint_sigma_m: float = 100.0
    footprint_wind_dir_deg: float = 270.0   # degrees clockwise from north

    # Column instruments: per-level averaging kernel (None → uniform weighting)
    averaging_kernel: np.ndarray | None = field(default=None, compare=False, repr=False)

    @property
    def params(self) -> OperatorParams:
        try:
            return INSTRUMENT_DB[self.tech_id][self.mode]
        except KeyError:
            raise ValueError(
                f"Unknown tech_id={self.tech_id!r} or mode={self.mode!r}. "
                f"Known tech_ids: {sorted(INSTRUMENT_DB)}"
            ) from None

    @property
    def operator_type(self) -> OperatorType:
        return self.params.operator_type

    @property
    def observable(self) -> ObservableType:
        return self.params.observable

    @property
    def effective_noise_std(self) -> float:
        """Scalar noise estimate for backward compatibility. Prefer ObservationResult.R."""
        p = self.params
        return math.sqrt(p.sigma_abs**2 + p.sigma_scale**2)
