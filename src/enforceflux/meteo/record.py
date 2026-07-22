"""The canonical meteorology record — one format, every transport model.

AERMOD, MicroHH, and FLEXPART each want boundary-layer forcing in a different
shape: AERMOD wants similarity parameters as scalars, MicroHH wants an
idealised single-column forcing block in kinematic units, FLEXPART wants GRIB
files on disk. :class:`MetRecord` is the single representation upstream of all
three, so a run is specified once and handed to whichever model.

    ERA5 GRIB ──► MetSeries ──┬──► AERMOD    SurfaceMet
                              ├──► MicroHH   Forcing
                              └──► FLEXPART  FlexpartMetSource (paths)

Every field is SI and sign conventions are fixed here (they differ between the
sources): ``sensible_heat_flux_w_m2`` is **positive upward** (ERA5's own
convention is the opposite and is flipped on read), and
``wind_direction_deg`` is meteorological — the direction the wind blows *from*.

See :mod:`enforceflux.meteo.adapters` for the model-specific conversions and
:mod:`enforceflux.meteo.era5_profile` for the ERA5 reader.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Iterator, Sequence

# Reference pressure for potential temperature [Pa].
REFERENCE_PRESSURE_PA = 100000.0
R_OVER_CP = 0.2857  # R_d/c_p for dry air
AIR_DENSITY = 1.2  # kg m-3
AIR_CP = 1005.0  # J kg-1 K-1


@dataclass(frozen=True)
class MetRecord:
    """Boundary-layer state at one place and time.

    The minimum a dispersion model needs: the wind that transports the plume,
    the turbulence that spreads it (``friction_velocity_m_s`` and
    ``sensible_heat_flux_w_m2``), and the lid that confines it
    (``mixing_height_m``).
    """

    time: datetime
    wind_speed_m_s: float
    wind_direction_deg: float  # direction the wind blows *from*
    temperature_k: float
    mixing_height_m: float
    friction_velocity_m_s: float
    sensible_heat_flux_w_m2: float  # positive upward
    surface_roughness_m: float = 0.1
    surface_pressure_pa: float = REFERENCE_PRESSURE_PA
    reference_height_m: float = 10.0
    # Free-atmosphere potential-temperature gradient above the mixed layer.
    potential_temperature_gradient_k_m: float = 0.01

    def __post_init__(self) -> None:
        if self.wind_speed_m_s < 0.0:
            raise ValueError("wind_speed_m_s must be non-negative")
        if self.surface_roughness_m <= 0.0:
            raise ValueError("surface_roughness_m must be positive")
        if self.mixing_height_m <= 0.0:
            raise ValueError("mixing_height_m must be positive")
        if self.friction_velocity_m_s <= 0.0:
            raise ValueError("friction_velocity_m_s must be positive")

    @property
    def is_convective(self) -> bool:
        """True when the surface is heating the boundary layer."""
        return self.sensible_heat_flux_w_m2 > 0.0

    @property
    def obukhov_length_m(self) -> float:
        """``L = -ρ cp T u*³ / (k g H)``; ``inf`` in the neutral limit."""
        heat = self.sensible_heat_flux_w_m2
        if heat == 0.0:
            return math.inf
        return -(
            AIR_DENSITY * AIR_CP * self.temperature_k * self.friction_velocity_m_s**3
        ) / (0.4 * 9.80665 * heat)

    @property
    def kinematic_heat_flux_k_m_s(self) -> float:
        """Surface heat flux in kinematic units ``w'θ'`` [K m s⁻¹] (MicroHH's unit)."""
        return self.sensible_heat_flux_w_m2 / (AIR_DENSITY * AIR_CP)

    @property
    def potential_temperature_k(self) -> float:
        """Surface potential temperature ``θ = T (p₀/p)^(R/cp)``."""
        return self.temperature_k * (REFERENCE_PRESSURE_PA / self.surface_pressure_pa) ** R_OVER_CP

    @property
    def wind_toward_deg(self) -> float:
        """The direction the wind blows *toward* (the plume's bearing)."""
        return (self.wind_direction_deg + 180.0) % 360.0

    def wind_components_m_s(self) -> tuple[float, float]:
        """Wind vector as ``(eastward, northward)`` components [m s⁻¹]."""
        toward = math.radians(self.wind_toward_deg)
        return self.wind_speed_m_s * math.sin(toward), self.wind_speed_m_s * math.cos(toward)

    def with_roughness(self, surface_roughness_m: float) -> "MetRecord":
        """Copy with a different site roughness (does not rescale ``u*``)."""
        return replace(self, surface_roughness_m=surface_roughness_m)


@dataclass(frozen=True)
class MetSeries:
    """An ordered set of :class:`MetRecord`s at one location.

    ``provenance`` records where the data came from — the ERA5 directory, the
    grid point actually sampled, the ``u*`` convention — so a run stays
    traceable to its forcing.
    """

    records: tuple[MetRecord, ...]
    longitude: float
    latitude: float
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(sorted(self.records, key=lambda r: r.time)))
        if not self.records:
            raise ValueError("MetSeries requires at least one MetRecord")

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[MetRecord]:
        return iter(self.records)

    def __getitem__(self, index):
        return self.records[index]

    @property
    def start(self) -> datetime:
        return self.records[0].time

    @property
    def end(self) -> datetime:
        return self.records[-1].time

    def window(self, start: datetime, end: datetime) -> "MetSeries":
        """The sub-series within ``[start, end]`` inclusive."""
        selected = [r for r in self.records if start <= r.time <= end]
        if not selected:
            raise ValueError(
                f"No met records between {start} and {end}; series covers "
                f"{self.start} to {self.end}."
            )
        return replace(self, records=tuple(selected))

    def daytime(self) -> "MetSeries":
        """Only the convective hours (upward surface heat flux)."""
        selected = [r for r in self.records if r.is_convective]
        if not selected:
            raise ValueError("No convective records in this series")
        return replace(self, records=tuple(selected))

    def mean(self) -> MetRecord:
        """A single record averaging the series.

        Wind is averaged as a vector, so a full day of veering wind correctly
        yields a weak mean rather than a spurious strong one.
        """
        n = len(self.records)
        east = sum(r.wind_components_m_s()[0] for r in self.records) / n
        north = sum(r.wind_components_m_s()[1] for r in self.records) / n
        speed = math.hypot(east, north)
        toward = math.degrees(math.atan2(east, north)) % 360.0
        return MetRecord(
            time=self.records[n // 2].time,
            wind_speed_m_s=speed,
            wind_direction_deg=(toward + 180.0) % 360.0,
            temperature_k=_mean(self.records, "temperature_k"),
            mixing_height_m=_mean(self.records, "mixing_height_m"),
            friction_velocity_m_s=_mean(self.records, "friction_velocity_m_s"),
            sensible_heat_flux_w_m2=_mean(self.records, "sensible_heat_flux_w_m2"),
            surface_roughness_m=self.records[0].surface_roughness_m,
            surface_pressure_pa=_mean(self.records, "surface_pressure_pa"),
            reference_height_m=self.records[0].reference_height_m,
            potential_temperature_gradient_k_m=_mean(
                self.records, "potential_temperature_gradient_k_m"
            ),
        )

    @property
    def directional_consistency(self) -> float:
        """How steady the wind direction is, from 0 (fully reversing) to 1 (constant).

        The ratio of the vector-mean wind speed to the mean scalar speed. A
        series that veers through the compass has a vector mean near zero even
        though every hour had a decent wind — which silently ruins any model
        that takes a single steady forcing. Check this before collapsing a
        series with :meth:`mean`.
        """
        scalar = _mean(self.records, "wind_speed_m_s")
        if scalar == 0.0:
            return 0.0
        return self.mean().wind_speed_m_s / scalar

    def summary(self) -> str:
        """One line per record — the quickest way to eyeball a forcing set."""
        lines = [
            f"MetSeries at ({self.latitude:.4f}, {self.longitude:.4f}), "
            f"{len(self)} records, {self.start:%Y-%m-%d %H:%M} → {self.end:%Y-%m-%d %H:%M}",
            f"  {'time':>16s} {'U':>6s} {'dir':>5s} {'T':>7s} {'zi':>7s} "
            f"{'u*':>6s} {'H':>8s} {'L':>9s}",
        ]
        for r in self.records:
            length = r.obukhov_length_m
            length_str = "     inf" if math.isinf(length) else f"{length:9.1f}"
            lines.append(
                f"  {r.time:%Y-%m-%d %H:%M} {r.wind_speed_m_s:6.2f} "
                f"{r.wind_direction_deg:5.0f} {r.temperature_k:7.2f} "
                f"{r.mixing_height_m:7.0f} {r.friction_velocity_m_s:6.3f} "
                f"{r.sensible_heat_flux_w_m2:8.1f} {length_str}"
            )
        return "\n".join(lines)


def _mean(records: Sequence[MetRecord], attribute: str) -> float:
    return sum(getattr(r, attribute) for r in records) / len(records)
