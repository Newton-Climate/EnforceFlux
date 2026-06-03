from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from enforceflux.models.instrument import Instrument
from enforceflux.models.source import Source


@dataclass(frozen=True)
class ForwardModelResult:
    g: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


class ISourceModel(ABC):
    @abstractmethod
    def build_sources(self, config: dict[str, Any], domain: Any) -> list[Source]:
        raise NotImplementedError


class IInstrumentModel(ABC):
    @abstractmethod
    def build_instruments(self, config: dict[str, Any], domain: Any) -> list[Instrument]:
        raise NotImplementedError


class ITransportModel(ABC):
    @abstractmethod
    def build_forward_operator(
        self,
        sources: Iterable[Source],
        instruments: Iterable[Instrument],
        domain: Any,
        config: dict[str, Any],
    ) -> ForwardModelResult:
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
