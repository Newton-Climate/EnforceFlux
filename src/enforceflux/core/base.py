from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from enforceflux.instrument import Instrument
from enforceflux.models.source import Source


@dataclass(frozen=True)
class ForwardModelResult:
    """Result of a transport *operator* build: the Jacobian ``g`` = ∂y/∂x."""

    g: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransportSimulationResult:
    """Result of a transport *simulation*: a forward concentration field.

    ``output_path`` points at the gridded concentration NetCDF written by the
    simulation backend (``None`` when only inputs were prepared, e.g. dry runs).
    Field extraction is left to the caller so multi-dimensional output
    (time/height/lat/lon) is not flattened prematurely.
    """

    output_path: Path | None
    meta: dict[str, Any] = field(default_factory=dict)


class ISourceModel(ABC):
    @abstractmethod
    def build_sources(self, config: dict[str, Any], domain: Any) -> list[Source]:
        raise NotImplementedError


class IInstrumentModel(ABC):
    @abstractmethod
    def build_instruments(self, config: dict[str, Any], domain: Any) -> list[Instrument]:
        raise NotImplementedError


class ITransportOperator(ABC):
    """Builds the linear(ized) observation operator ``g`` (Jacobian) for inversion.

    Note the two independent axes: this method always builds the *forward
    operator* (∂observation/∂flux), regardless of whether the underlying
    transport model is integrated forward or backward in time (e.g. FLEXPART
    backward footprints also assemble a forward operator).
    """

    @abstractmethod
    def build_forward_operator(
        self,
        sources: Iterable[Source],
        instruments: Iterable[Instrument],
        domain: Any,
        config: dict[str, Any],
    ) -> ForwardModelResult:
        raise NotImplementedError


class ITransportSimulation(ABC):
    """Runs a forward transport simulation, producing a concentration field.

    Unlike :class:`ITransportOperator` (which returns a Jacobian for inversion),
    a simulation integrates the actual emissions to produce observable
    concentrations — used for synthetic truth, plume fields, and the
    simulate-once / build-G-many-ways workflows.
    """

    @abstractmethod
    def simulate(
        self,
        sources: Iterable[Source],
        domain: Any,
        config: dict[str, Any],
    ) -> TransportSimulationResult:
        raise NotImplementedError


class IInversionEngine(ABC):
    @abstractmethod
    def invert(
        self,
        g: np.ndarray,
        y: np.ndarray,
        x_prior: np.ndarray,
        s_a: np.ndarray,
        r: np.ndarray,
    ) -> Any:
        raise NotImplementedError


@dataclass(frozen=True)
class FluxResult:
    """Estimated flux(es) from any :class:`IFluxEstimator`, regardless of method.

    ``flux`` holds the per-source (or per-tower, for local methods) flux
    estimate. ``meta`` carries method-specific diagnostics — for inversion-based
    estimators this includes the underlying :class:`~enforceflux.inversion.result.InversionResult`.
    """

    flux: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


class IFluxEstimator(ABC):
    """Computes flux estimates from observations, by whatever method the
    plugin implements (transport inversion, eddy covariance, flux-gradient, ...).

    Unlike :class:`IInversionEngine` (a narrow G/prior/covariance contract for
    transport-based inversion specifically), this is the user-facing
    abstraction: a single ``estimate`` call that different flux-computation
    paradigms can implement with whatever inputs they need, via ``config``.
    """

    @abstractmethod
    def estimate(
        self,
        observations: Any,
        config: dict[str, Any],
    ) -> FluxResult:
        raise NotImplementedError
