from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from enforceflux.config import InstrumentConfig


@dataclass(frozen=True)
class Instrument:
    id: str
    kind: str
    x: float
    y: float
    noise_std: float
    averaging_seconds: float
    z: float = 0.0

    @property
    def effective_noise_std(self) -> float:
        # Placeholder: treat averaging as independent 1-second samples.
        if self.averaging_seconds <= 1:
            return self.noise_std
        return self.noise_std / (self.averaging_seconds ** 0.5)

    @classmethod
    def from_config(cls, config: InstrumentConfig) -> "Instrument":
        return cls(
            id=config.id,
            kind=config.kind,
            x=config.x,
            y=config.y,
            z=getattr(config, "z", 0.0),
            noise_std=config.noise_std,
            averaging_seconds=config.averaging_seconds,
        )


def instruments_from_config(configs: Iterable[InstrumentConfig]) -> list[Instrument]:
    return [Instrument.from_config(cfg) for cfg in configs]
