from __future__ import annotations

from typing import Any, Iterable

from enforceflux.core.base import ForwardModelResult, ITransportModel
from enforceflux.models.instrument import Instrument
from enforceflux.models.source import Source
from enforceflux.models.transport import GaussianTransport


def _require_keys(blob: dict, keys: list[str], context: str) -> None:
    missing = [k for k in keys if k not in blob]
    if missing:
        raise ValueError(f"Missing keys in {context}: {missing}")


class GaussianTransportModel(ITransportModel):
    def build_forward_operator(
        self,
        sources: Iterable[Source],
        instruments: Iterable[Instrument],
        domain: Any,
        config: dict[str, Any],
    ) -> ForwardModelResult:
        _require_keys(config, ["sigma", "wind"], "transport.gaussian")
        wind = list(config["wind"])
        if len(wind) != 2:
            raise ValueError("transport.gaussian.wind must be a 2-element list [vx, vy]")
        transport = GaussianTransport(sigma=float(config["sigma"]), wind=(float(wind[0]), float(wind[1])))
        g = transport.build_forward_operator(sources, instruments)
        return ForwardModelResult(g=g, meta={"model": "gaussian"})
