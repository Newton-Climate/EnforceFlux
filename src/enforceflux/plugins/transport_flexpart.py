from __future__ import annotations

from typing import Any, Iterable

from enforceflux.core.base import ForwardModelResult, ITransportModel
from enforceflux.flexpart.wrapper import FlexpartWrapper
from enforceflux.models.instrument import Instrument
from enforceflux.models.source import Source


class FlexpartTransportModel(ITransportModel):
    def build_forward_operator(
        self,
        sources: Iterable[Source],
        instruments: Iterable[Instrument],
        domain: Any,
        config: dict[str, Any],
    ) -> ForwardModelResult:
        wrapper = FlexpartWrapper(domain=domain, config=config)
        result = wrapper.run(sources, instruments)
        return ForwardModelResult(g=result.g, meta=result.meta)
