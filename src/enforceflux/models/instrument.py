from __future__ import annotations

from dataclasses import dataclass


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
        if self.averaging_seconds <= 1:
            return self.noise_std
        return self.noise_std / (self.averaging_seconds ** 0.5)
