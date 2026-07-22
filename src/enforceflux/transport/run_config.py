"""The one YAML schema every transport model runs from.

A transport run is described once — which model, forward simulation or
inversion operator, the meteorology, the sources, the receptors, the domain,
the output — and the model is a single line in that file::

    transport:
      model: aermod        # aermod | flexpart | microhh
      mode: simulation     # simulation | operator

The shared keys are **authoritative for every model**. Each model additionally
gets a block (``aermod:``, ``flexpart:``, ``microhh:``) for the settings that
genuinely have no counterpart elsewhere — a Fortran binary path, a particle
count, an LES box. Those blocks may never restate a shared key: sources and
meteorology cannot silently diverge between models, which is the entire point
of running them from one file.

Geometry follows one contract: **``domain.origin_lon``/``origin_lat`` are the
only geographic coordinates in the file.** The domain extent, every source and
every receptor are Cartesian metres east/north of that origin. The lon/lat
bounds FLEXPART needs are *derived* from the origin and the extent
(:attr:`RunDomain.bounds_lonlat`) rather than declared, so a config cannot
carry two descriptions of the same box that disagree.

That choice suits the models: AERMOD's dispersion algebra and MicroHH's LES box
are both dimensional, so they consume the metres directly, and only the
lon/lat-gridded FLEXPART path pays a projection on the way out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

MODELS = ("aermod", "flexpart", "microhh")
MODES = ("simulation", "operator")

Model = Literal["aermod", "flexpart", "microhh"]
Mode = Literal["simulation", "operator"]


def _require(blob: dict, keys: Sequence[str], context: str) -> None:
    missing = [k for k in keys if k not in blob]
    if missing:
        raise ValueError(f"Missing keys in {context}: {missing}")


def _parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(str(value).rstrip("Z"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _reject_geographic(blob: dict[str, Any], context: str) -> None:
    """Enforce the Cartesian contract on a source/receptor entry."""
    geographic = sorted({"lon", "lat", "longitude", "latitude"} & set(blob))
    if geographic:
        raise ValueError(
            f"{context} uses geographic key(s) {geographic}. Only "
            "domain.origin_lon/origin_lat are given in lon/lat; every other "
            "position is Cartesian metres east/north of that origin. Use "
            "x_m/y_m (and alt_m for height)."
        )


@dataclass(frozen=True)
class RunSource:
    """An emission source, in metres east/north of the domain origin."""

    id: str
    x_m: float
    y_m: float
    emission_rate_kg_s: float
    altitude_m: float = 0.0
    # Inversion priors; only meaningful in operator mode.
    prior_mean_kg_s: float | None = None
    prior_std_kg_s: float | None = None

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> "RunSource":
        _reject_geographic(blob, "sources[]")
        _require(blob, ["id", "x_m", "y_m"], "sources[]")
        rate = float(blob.get("emission_rate_kg_s", 0.0))
        return cls(
            id=str(blob["id"]),
            x_m=float(blob["x_m"]),
            y_m=float(blob["y_m"]),
            emission_rate_kg_s=rate,
            altitude_m=float(blob.get("alt_m", 0.0)),
            prior_mean_kg_s=(
                float(blob["prior_mean_kg_s"]) if "prior_mean_kg_s" in blob else None
            ),
            prior_std_kg_s=(
                float(blob["prior_std_kg_s"]) if "prior_std_kg_s" in blob else None
            ),
        )


@dataclass(frozen=True)
class RunReceptor:
    """A measurement location, in metres east/north of the domain origin."""

    id: str
    x_m: float
    y_m: float
    altitude_m: float = 0.0
    path_length_m: float = 0.0
    path_bearing_deg: float = 0.0

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> "RunReceptor":
        _reject_geographic(blob, "receptors[]")
        _require(blob, ["id", "x_m", "y_m"], "receptors[]")
        return cls(
            id=str(blob["id"]),
            x_m=float(blob["x_m"]),
            y_m=float(blob["y_m"]),
            altitude_m=float(blob.get("alt_m", 0.0)),
            path_length_m=float(blob.get("path_length_m", 0.0)),
            path_bearing_deg=float(blob.get("path_bearing_deg", 0.0)),
        )


@dataclass(frozen=True)
class RunDomain:
    """The domain: one geographic origin, everything else Cartesian metres.

    ``origin_lon``/``origin_lat`` are the **only** geographic coordinates in a
    run config. The extent is given in metres east/north of that origin, and
    the geographic bounds the lon/lat-gridded models need are *derived* from it
    (:attr:`lon_min` and friends) rather than declared, so the two can never
    disagree.
    """

    origin_lon: float
    origin_lat: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    spacing_m: float = 500.0
    heights_m: tuple[float, ...] = (100.0,)
    receptor_height_m: float = 2.0

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("domain extents must satisfy max > min")
        if self.spacing_m <= 0.0:
            raise ValueError("domain.spacing_m must be positive")

    @property
    def origin_lonlat(self) -> tuple[float, float]:
        return (self.origin_lon, self.origin_lat)

    @property
    def projection(self) -> "DomainProjection":
        """Local metric frame anchored on the declared origin."""
        return DomainProjection(self.origin_lon, self.origin_lat)

    @property
    def centre(self) -> tuple[float, float]:
        """Box centre in metres — the origin need not be centred."""
        return (0.5 * (self.x_min + self.x_max), 0.5 * (self.y_min + self.y_max))

    @property
    def size_m(self) -> tuple[float, float]:
        return (self.x_max - self.x_min, self.y_max - self.y_min)

    @property
    def bounds_lonlat(self) -> tuple[float, float, float, float]:
        """Derived ``(lon_min, lat_min, lon_max, lat_max)``.

        The corners are projected out and the extremes taken: an
        azimuthal-equidistant box is not exactly a lon/lat rectangle, so using
        the corner envelope keeps the derived bounds a superset of the
        Cartesian domain rather than clipping it.
        """
        proj = self.projection
        xs = (self.x_min, self.x_max)
        ys = (self.y_min, self.y_max)
        corners = [proj.to_lonlat(x, y) for x in xs for y in ys]
        lons = [float(c[0]) for c in corners]
        lats = [float(c[1]) for c in corners]
        return (min(lons), min(lats), max(lons), max(lats))

    @property
    def lon_min(self) -> float:
        return self.bounds_lonlat[0]

    @property
    def lat_min(self) -> float:
        return self.bounds_lonlat[1]

    @property
    def lon_max(self) -> float:
        return self.bounds_lonlat[2]

    @property
    def lat_max(self) -> float:
        return self.bounds_lonlat[3]

    @property
    def spacing_deg(self) -> tuple[float, float]:
        """``spacing_m`` expressed in degrees, for the lon/lat-gridded models."""
        import math

        dlat = self.spacing_m / 111_320.0
        dlon = self.spacing_m / (
            111_320.0 * max(math.cos(math.radians(self.origin_lat)), 1.0e-6)
        )
        return dlon, dlat

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> "RunDomain":
        legacy = sorted({"lon_min", "lat_min", "lon_max", "lat_max"} & set(blob))
        if legacy:
            raise ValueError(
                f"domain declares geographic bounds {legacy}. These are now derived: "
                "give domain.origin_lon/origin_lat plus the Cartesian extent "
                "x_min/x_max/y_min/y_max in metres east/north of that origin."
            )
        _require(
            blob,
            ["origin_lon", "origin_lat", "x_min", "x_max", "y_min", "y_max"],
            "domain",
        )
        return cls(
            origin_lon=float(blob["origin_lon"]),
            origin_lat=float(blob["origin_lat"]),
            x_min=float(blob["x_min"]),
            x_max=float(blob["x_max"]),
            y_min=float(blob["y_min"]),
            y_max=float(blob["y_max"]),
            spacing_m=float(blob.get("spacing_m", 500.0)),
            heights_m=tuple(float(h) for h in blob.get("heights_m", [100.0])),
            receptor_height_m=float(blob.get("receptor_height_m", 2.0)),
        )


@dataclass(frozen=True)
class RunOutput:
    """Where the canonical NetCDF goes."""

    path: Path
    compress: bool = True
    keep_native: bool = True

    @classmethod
    def from_dict(cls, blob: dict[str, Any], base: Path) -> "RunOutput":
        _require(blob, ["path"], "output")
        path = Path(blob["path"])
        return cls(
            path=path if path.is_absolute() else (base / path).resolve(),
            compress=bool(blob.get("compress", True)),
            keep_native=bool(blob.get("keep_native", True)),
        )


@dataclass(frozen=True)
class TransportRunConfig:
    """A complete, model-agnostic transport run."""

    model: Model
    mode: Mode
    domain: RunDomain
    sources: tuple[RunSource, ...]
    output: RunOutput
    receptors: tuple[RunReceptor, ...] = ()
    met: dict[str, Any] = field(default_factory=dict)
    start: datetime | None = None
    end: datetime | None = None
    model_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    base_dir: Path = Path(".")

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"transport.model must be one of {MODELS}, got {self.model!r}")
        if self.mode not in MODES:
            raise ValueError(f"transport.mode must be one of {MODES}, got {self.mode!r}")
        if not self.sources:
            raise ValueError("A transport run needs at least one source")
        if self.mode == "operator" and not self.receptors:
            raise ValueError(
                "operator mode builds an observation × source Jacobian, so it needs "
                "'receptors' — add them, or use mode: simulation for a field."
            )

    @property
    def options(self) -> dict[str, Any]:
        """The active model's own block (empty when it has none)."""
        return self.model_options.get(self.model, {})

    def option(self, key: str, default: Any = None) -> Any:
        return self.options.get(key, default)

    def resolve(self, value: str | Path) -> Path:
        """Resolve a path from a model block against the config's directory."""
        path = Path(value)
        return path if path.is_absolute() else (self.base_dir / path).resolve()

    def projection(self) -> "DomainProjection":
        """The local metric frame for this domain, anchored on the origin."""
        return self.domain.projection

    @property
    def origin_lonlat(self) -> tuple[float, float]:
        """The run's single geographic anchor."""
        return self.domain.origin_lonlat

    def to_lonlat(self, x_m: float, y_m: float) -> tuple[float, float]:
        """Cartesian metres -> ``(lon, lat)``, for the lon/lat-gridded models."""
        lon, lat = self.projection().to_lonlat(x_m, y_m)
        return float(lon), float(lat)

    # ── Loading ──────────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, blob: dict[str, Any], base_dir: Path = Path(".")) -> "TransportRunConfig":
        _require(blob, ["transport", "domain", "sources", "output"], "config")
        transport = blob["transport"]
        _require(transport, ["model"], "transport")

        model = str(transport["model"]).strip().lower()
        mode = str(transport.get("mode", "simulation")).strip().lower()

        model_options = {m: dict(blob.get(m) or {}) for m in MODELS}
        _reject_shadowed_keys(model_options)

        return cls(
            model=model,  # type: ignore[arg-type]
            mode=mode,  # type: ignore[arg-type]
            domain=RunDomain.from_dict(blob["domain"]),
            sources=tuple(RunSource.from_dict(s) for s in blob["sources"]),
            receptors=tuple(RunReceptor.from_dict(r) for r in (blob.get("receptors") or [])),
            met=dict(blob.get("met") or {}),
            output=RunOutput.from_dict(blob["output"], base_dir),
            start=_parse_time(transport["start"]) if "start" in transport else None,
            end=_parse_time(transport["end"]) if "end" in transport else None,
            model_options=model_options,
            base_dir=base_dir,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "TransportRunConfig":
        import yaml

        config_path = Path(path).resolve()
        blob = yaml.safe_load(config_path.read_text())
        if not isinstance(blob, dict):
            raise ValueError(f"Transport config at {config_path} must parse to a mapping")
        return cls.from_dict(blob, base_dir=config_path.parent)


# Shared keys a model block must never restate — allowing them would let one
# model quietly run a different scenario than another from the same file.
_SHARED_KEYS = frozenset({"sources", "receptors", "domain", "met", "output", "transport"})


def _reject_shadowed_keys(model_options: dict[str, dict[str, Any]]) -> None:
    for model, options in model_options.items():
        shadowed = sorted(_SHARED_KEYS & set(options))
        if shadowed:
            raise ValueError(
                f"The '{model}:' block redefines shared key(s) {shadowed}. Shared "
                "settings live at the top level so every model runs the same "
                "scenario; model blocks are only for settings unique to that model."
            )


@dataclass(frozen=True)
class DomainProjection:
    """Local azimuthal-equidistant frame centred on the domain.

    AERMOD needs metres and the config is in degrees. Centring the projection on
    the domain keeps distortion negligible over the near-field extents these
    runs use, and the round trip is exact enough to place sources and receptors.
    """

    centre_lon: float
    centre_lat: float

    def _transformer(self, inverse: bool = False):
        from pyproj import CRS, Transformer

        local = CRS.from_proj4(
            f"+proj=aeqd +lat_0={self.centre_lat} +lon_0={self.centre_lon} "
            "+units=m +datum=WGS84 +no_defs"
        )
        wgs84 = CRS.from_epsg(4326)
        if inverse:
            return Transformer.from_crs(local, wgs84, always_xy=True)
        return Transformer.from_crs(wgs84, local, always_xy=True)

    def to_xy(self, longitude, latitude):
        """Longitude/latitude → local metres (east, north)."""
        return self._transformer().transform(longitude, latitude)

    def to_lonlat(self, x, y):
        """Local metres → longitude/latitude."""
        return self._transformer(inverse=True).transform(x, y)
