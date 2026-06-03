from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from enforceflux.models.instrument import Instrument
from enforceflux.models.source import Source


@dataclass(frozen=True)
class GaussianTransport:
    sigma: float
    wind: tuple[float, float]

    def build_forward_operator(
        self,
        sources: Iterable[Source],
        instruments: Iterable[Instrument],
    ) -> np.ndarray:
        sources = list(sources)
        instruments = list(instruments)
        g = np.zeros((len(instruments), len(sources)))
        norm = 1.0 / (2.0 * np.pi * self.sigma ** 2)

        for i, inst in enumerate(instruments):
            for j, src in enumerate(sources):
                dx = inst.x - src.x - self.wind[0]
                dy = inst.y - src.y - self.wind[1]
                r2 = dx * dx + dy * dy
                g[i, j] = norm * np.exp(-r2 / (2.0 * self.sigma ** 2))

        return g
