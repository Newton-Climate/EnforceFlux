from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


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
class ComponentConfig:
    plugin: str
    config: dict


@dataclass(frozen=True)
class ProjectConfig:
    domain: DomainConfig
    components: dict[str, ComponentConfig]
    random_seed: int | None = None

    def component(self, name: str) -> ComponentConfig:
        if name not in self.components:
            available = ", ".join(sorted(self.components.keys()))
            raise KeyError(f"Component '{name}' not found. Available: {available}")
        return self.components[name]


def _require_keys(blob: dict, keys: list[str], context: str) -> None:
    missing = [k for k in keys if k not in blob]
    if missing:
        raise ValueError(f"Missing keys in {context}: {missing}")


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


def _component_from_dict(blob: dict, context: str) -> ComponentConfig:
    _require_keys(blob, ["plugin"], context)
    return ComponentConfig(plugin=str(blob["plugin"]), config=dict(blob.get("config", {})))


def _legacy_to_components(data: dict) -> dict[str, ComponentConfig]:
    _require_keys(data, ["sources", "instruments", "transport"], "legacy config")

    transport = data["transport"]
    _require_keys(transport, ["model", "sigma", "wind"], "transport")
    wind = list(transport["wind"])
    if len(wind) != 2:
        raise ValueError("transport.wind must be a 2-element list [vx, vy]")

    return {
        "source": ComponentConfig(
            plugin="enforceflux.source.static",
            config={"sources": data["sources"]},
        ),
        "instrument": ComponentConfig(
            plugin="enforceflux.instrument.static",
            config={"instruments": data["instruments"]},
        ),
        "transport": ComponentConfig(
            plugin=f"enforceflux.transport.{transport['model'].lower()}",
            config={
                "model": transport["model"],
                "sigma": float(transport["sigma"]),
                "wind": [float(wind[0]), float(wind[1])],
            },
        ),
        "inversion": ComponentConfig(
            plugin="enforceflux.inversion.bayesian",
            config=dict(data.get("inversion", {})),
        ),
    }


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path)
    data = json.loads(path.read_text())
    _require_keys(data, ["domain"], "config")

    domain = _domain_from_dict(data["domain"])

    if "components" in data:
        blob = data["components"]
        _require_keys(blob, ["source", "instrument", "transport", "inversion"], "components")
        components = {
            name: _component_from_dict(blob[name], f"components.{name}")
            for name in ["source", "instrument", "transport", "inversion"]
        }
    else:
        components = _legacy_to_components(data)

    return ProjectConfig(
        domain=domain,
        components=components,
        random_seed=data.get("random_seed"),
    )
