from __future__ import annotations

from typing import Any

from enforceflux.core.base import ISourceModel
from enforceflux.models.source import Source


def _require_keys(blob: dict, keys: list[str], context: str) -> None:
    missing = [k for k in keys if k not in blob]
    if missing:
        raise ValueError(f"Missing keys in {context}: {missing}")


class StaticSourceModel(ISourceModel):
    def build_sources(self, config: dict[str, Any], domain: Any) -> list[Source]:
        sources_blob = config.get("sources", [])
        sources: list[Source] = []
        for item in sources_blob:
            _require_keys(item, ["id", "kind", "x", "y", "flux_true"], "source")
            sources.append(
                Source(
                    id=str(item["id"]),
                    kind=str(item["kind"]),
                    x=float(item["x"]),
                    y=float(item["y"]),
                    z=float(item.get("z", item.get("alt", 0.0))),
                    flux_true=float(item["flux_true"]),
                    flux_prior_mean=float(item.get("flux_prior_mean", 0.0)),
                    flux_prior_std=float(item.get("flux_prior_std", 1.0)),
                )
            )
        return sources
