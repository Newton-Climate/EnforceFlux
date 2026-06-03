from __future__ import annotations

from typing import Any

from enforceflux.core.base import IInstrumentModel
from enforceflux.models.instrument import Instrument


def _require_keys(blob: dict, keys: list[str], context: str) -> None:
    missing = [k for k in keys if k not in blob]
    if missing:
        raise ValueError(f"Missing keys in {context}: {missing}")


class StaticInstrumentModel(IInstrumentModel):
    def build_instruments(self, config: dict[str, Any], domain: Any) -> list[Instrument]:
        instruments_blob = config.get("instruments", [])
        instruments: list[Instrument] = []
        for item in instruments_blob:
            _require_keys(item, ["id", "kind", "x", "y", "noise_std"], "instrument")
            instruments.append(
                Instrument(
                    id=str(item["id"]),
                    kind=str(item["kind"]),
                    x=float(item["x"]),
                    y=float(item["y"]),
                    z=float(item.get("z", item.get("alt", 0.0))),
                    noise_std=float(item["noise_std"]),
                    averaging_seconds=float(item.get("averaging_seconds", 0.0)),
                )
            )
        return instruments
