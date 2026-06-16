from dataclasses import dataclass


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
