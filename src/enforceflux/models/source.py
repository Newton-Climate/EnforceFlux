from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from enforceflux.config import SourceConfig


@dataclass(frozen=True)
class Source:
    id: str
    kind: str
    x: float
    y: float
    flux_true: float
    flux_prior_mean: float
    flux_prior_std: float
    z: float = 0.0

    @classmethod
    def from_config(cls, config: SourceConfig) -> "Source":
        return cls(
            id=config.id,
            kind=config.kind,
            x=config.x,
            y=config.y,
            z=getattr(config, "z", 0.0),
            flux_true=config.flux_true,
            flux_prior_mean=config.flux_prior_mean,
            flux_prior_std=config.flux_prior_std,
        )


def sources_from_config(configs: Iterable[SourceConfig]) -> list[Source]:
    return [Source.from_config(cfg) for cfg in configs]
