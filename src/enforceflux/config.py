from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class DomainConfig:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    grid_spacing: float
    crs: str | None = None
    crs_wgs84: str = "EPSG:4326"


@dataclass(frozen=True)
class SourceConfig:
    id: str
    kind: str
    x: float
    y: float
    flux_true: float
    flux_prior_mean: float
    flux_prior_std: float
    z: float = 0.0


@dataclass(frozen=True)
class InstrumentConfig:
    id: str
    kind: str
    x: float
    y: float
    noise_std: float
    averaging_seconds: float
    z: float = 0.0


@dataclass(frozen=True)
class TransportConfig:
    model: str
    sigma: float
    wind: List[float]


@dataclass(frozen=True)
class InversionConfig:
    r_cond: float = 1e-10


@dataclass(frozen=True)
class ComponentConfig:
    plugin: str
    config: Dict[str, Any]


@dataclass(frozen=True)
class ProjectConfig:
    domain: DomainConfig
    components: Dict[str, ComponentConfig]
    random_seed: Optional[int] = None

    def component(self, name: str) -> ComponentConfig:
        if name not in self.components:
            available = ", ".join(sorted(self.components.keys()))
            raise KeyError(f"Component '{name}' not found. Available: {available}")
        return self.components[name]


def _require_keys(blob: dict, keys: List[str], context: str) -> None:
    missing = [k for k in keys if k not in blob]
    if missing:
        raise ValueError(f"Missing keys in {context}: {missing}")


def _source_from_dict(blob: dict) -> SourceConfig:
    _require_keys(blob, ["id", "kind", "x", "y", "flux_true"], "source")
    flux_prior_mean = float(blob.get("flux_prior_mean", 0.0))
    flux_prior_std = float(blob.get("flux_prior_std", 1.0))
    return SourceConfig(
        id=str(blob["id"]),
        kind=str(blob["kind"]),
        x=float(blob["x"]),
        y=float(blob["y"]),
        z=float(blob.get("z", blob.get("alt", 0.0))),
        flux_true=float(blob["flux_true"]),
        flux_prior_mean=flux_prior_mean,
        flux_prior_std=flux_prior_std,
    )


def _instrument_from_dict(blob: dict) -> InstrumentConfig:
    _require_keys(blob, ["id", "kind", "x", "y", "noise_std"], "instrument")
    return InstrumentConfig(
        id=str(blob["id"]),
        kind=str(blob["kind"]),
        x=float(blob["x"]),
        y=float(blob["y"]),
        z=float(blob.get("z", blob.get("alt", 0.0))),
        noise_std=float(blob["noise_std"]),
        averaging_seconds=float(blob.get("averaging_seconds", 0.0)),
    )


def _domain_from_dict(blob: dict) -> DomainConfig:
    _require_keys(blob, ["x_min", "x_max", "y_min", "y_max", "grid_spacing"], "domain")
    return DomainConfig(
        x_min=float(blob["x_min"]),
        x_max=float(blob["x_max"]),
        y_min=float(blob["y_min"]),
        y_max=float(blob["y_max"]),
        grid_spacing=float(blob["grid_spacing"]),
        crs=blob.get("crs"),
        crs_wgs84=str(blob.get("crs_wgs84", "EPSG:4326")),
    )


def _transport_from_dict(blob: dict) -> TransportConfig:
    _require_keys(blob, ["model", "sigma", "wind"], "transport")
    wind = list(blob["wind"])
    if len(wind) != 2:
        raise ValueError("transport.wind must be a 2-element list [vx, vy]")
    return TransportConfig(
        model=str(blob["model"]),
        sigma=float(blob["sigma"]),
        wind=[float(wind[0]), float(wind[1])],
    )


def _inversion_from_dict(blob: dict) -> InversionConfig:
    return InversionConfig(r_cond=float(blob.get("r_cond", 1e-10)))


def _component_from_dict(blob: dict, context: str) -> ComponentConfig:
    _require_keys(blob, ["plugin"], context)
    config = dict(blob.get("config", {}))
    return ComponentConfig(plugin=str(blob["plugin"]), config=config)


def _legacy_to_components(data: dict) -> Dict[str, ComponentConfig]:
    _require_keys(data, ["sources", "instruments", "transport"], "legacy config")

    transport_cfg = _transport_from_dict(data["transport"])
    transport_model = transport_cfg.model.lower()
    transport_plugin = f"enforceflux.transport.{transport_model}"

    components = {
        "source": ComponentConfig(
            plugin="enforceflux.source.static",
            config={"sources": data["sources"]},
        ),
        "instrument": ComponentConfig(
            plugin="enforceflux.instrument.static",
            config={"instruments": data["instruments"]},
        ),
        "transport": ComponentConfig(
            plugin=transport_plugin,
            config={
                "model": transport_cfg.model,
                "sigma": transport_cfg.sigma,
                "wind": transport_cfg.wind,
            },
        ),
        "inversion": ComponentConfig(
            plugin="enforceflux.inversion.bayesian",
            config=dict(data.get("inversion", {})),
        ),
    }

    return components


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path)
    data = json.loads(path.read_text())
    _require_keys(data, ["domain"], "config")

    domain = _domain_from_dict(data["domain"])

    if "components" in data:
        components_blob = data["components"]
        _require_keys(
            components_blob,
            ["source", "instrument", "transport", "inversion"],
            "components",
        )
        components = {
            name: _component_from_dict(components_blob[name], f"components.{name}")
            for name in ["source", "instrument", "transport", "inversion"]
        }
    else:
        components = _legacy_to_components(data)

    random_seed = data.get("random_seed")

    return ProjectConfig(domain=domain, components=components, random_seed=random_seed)
