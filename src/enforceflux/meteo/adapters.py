"""Convert canonical meteorology into each transport model's native format.

One :class:`~enforceflux.meteo.record.MetSeries` in, any model's forcing out::

    series = met_series_from_era5(meteo_dir, lon, lat)

    to_aermod(series)              # list[SurfaceMet]        — direct, per hour
    to_microhh_forcing(series)     # Forcing                 — single idealised column
    to_flexpart(series)            # FlexpartMetSource       — GRIB paths + coverage

The three conversions are deliberately asymmetric, because the models are:

* **AERMOD** consumes boundary-layer scalars, so the mapping is one-to-one and
  lossless — every canonical field lands in a ``SurfaceMet``.
* **MicroHH** is an idealised LES: it takes *one* steady forcing for the whole
  run, so a series must be collapsed (see ``reduce``) and several LES-specific
  quantities (capping-inversion strength and depth) have no ERA5 counterpart
  and keep their defaults.
* **FLEXPART** reads ERA5 GRIB directly and cannot be handed scalars at all.
  Its "conversion" is therefore a validated pointer back at the files the
  series came from — which is why :func:`to_flexpart` refuses to work on a
  series that did not originate from ERA5.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from enforceflux.meteo.record import MetRecord, MetSeries

if TYPE_CHECKING:  # pragma: no cover - typing only
    from enforceflux.aermod.config import SurfaceMet
    from enforceflux.microhh.sim_config import Forcing

Reduce = Literal["none", "mean", "daytime_mean"]


def _as_records(met: MetSeries | MetRecord) -> list[MetRecord]:
    return [met] if isinstance(met, MetRecord) else list(met)


def _reduce(met: MetSeries | MetRecord, reduce: Reduce) -> MetRecord:
    if isinstance(met, MetRecord):
        return met
    if reduce == "mean":
        return met.mean()
    if reduce == "daytime_mean":
        return met.daytime().mean()
    raise ValueError(
        f"reduce={reduce!r} leaves {len(met)} records, but this model takes a "
        "single steady forcing. Use 'mean' or 'daytime_mean', or pass one MetRecord."
    )


# ── AERMOD ───────────────────────────────────────────────────────────────────


def to_aermod(met: MetSeries | MetRecord) -> "list[SurfaceMet]":
    """Canonical met → a list of AERMOD :class:`SurfaceMet` hours.

    Lossless: ``u*``, the surface heat flux, and the mixing height are passed
    through as measured values, so AERMOD uses them directly instead of
    re-deriving them from a Pasquill class. Feed the result to
    ``AermodConfig(met=...)``.
    """
    from enforceflux.aermod.config import SurfaceMet

    return [
        SurfaceMet(
            wind_speed_m_s=r.wind_speed_m_s,
            wind_direction_deg=r.wind_direction_deg,
            temperature_k=r.temperature_k,
            monin_obukhov_length_m=_finite_obukhov(r),
            mixing_height_m=r.mixing_height_m,
            surface_roughness_m=r.surface_roughness_m,
            friction_velocity_m_s=r.friction_velocity_m_s,
            sensible_heat_flux_w_m2=r.sensible_heat_flux_w_m2,
            reference_height_m=r.reference_height_m,
            potential_temperature_gradient_k_m=r.potential_temperature_gradient_k_m,
            timestamp=r.time.isoformat(),
        )
        for r in _as_records(met)
    ]


def _finite_obukhov(record: MetRecord) -> float:
    """``L``, clipped away from the neutral singularity AERMOD cannot represent."""
    length = record.obukhov_length_m
    if math.isinf(length) or abs(length) > 1.0e6:
        return 1.0e6  # effectively neutral
    # |L| below a few metres is outside similarity theory's validity.
    return math.copysign(max(abs(length), 2.0), length)


# ── MicroHH ──────────────────────────────────────────────────────────────────


def to_microhh_forcing(
    met: MetSeries | MetRecord,
    *,
    x_bearing_deg: float | None = None,
    reduce: Reduce = "mean",
    inversion_strength_K: float = 2.0,
    inversion_depth_m: float = 100.0,
    min_directional_consistency: float = 0.6,
) -> "Forcing":
    """Canonical met → a MicroHH :class:`Forcing` block.

    MicroHH runs in a wind-aligned box, so the wind is rotated into box
    coordinates: with ``x_bearing_deg`` equal to the bearing of the box's +x
    axis, ``u_geo`` is the along-box component and ``v_geo`` the cross-box one.
    Pass the case's own ``x_bearing_deg`` to keep them consistent; omit it to
    align the box with the mean wind (``v_geo = 0``).

    ``inversion_strength_K`` and ``inversion_depth_m`` have no ERA5 counterpart
    and stay at their defaults — they describe the capping inversion's
    structure, which ERA5's single ``blh`` value does not resolve.

    The scalar roughness ``z0h`` follows the usual ``z0m/10`` convention.

    Collapsing a long series is refused when the wind direction is not steady
    enough (``min_directional_consistency``): a vector mean over veering wind
    is arithmetically correct but produces an absurdly weak forcing — e.g. ten
    days of Sacramento spring met average to 0.4 m/s despite no calm hour.
    Narrow the window with :meth:`MetSeries.window` instead, or lower the
    threshold deliberately.
    """
    from enforceflux.microhh.sim_config import Forcing

    if isinstance(met, MetSeries) and len(met) > 1 and reduce != "none":
        candidate = met.daytime() if reduce == "daytime_mean" else met
        consistency = candidate.directional_consistency
        if consistency < min_directional_consistency:
            raise ValueError(
                f"Wind direction is too variable to collapse into one steady LES "
                f"forcing: directional consistency {consistency:.2f} < "
                f"{min_directional_consistency:.2f} over {len(candidate)} records "
                f"({candidate.start:%Y-%m-%d %H:%M} → {candidate.end:%Y-%m-%d %H:%M}). "
                "The vector mean would be far weaker than the actual hourly winds. "
                "Select a shorter window with MetSeries.window(), or pass "
                "min_directional_consistency=0 to override."
            )

    record = _reduce(met, reduce)
    bearing = record.wind_toward_deg if x_bearing_deg is None else x_bearing_deg
    # Angle between the wind's travel direction and the box +x axis.
    offset = math.radians(record.wind_toward_deg - bearing)

    return Forcing(
        u_geo=record.wind_speed_m_s * math.cos(offset),
        v_geo=record.wind_speed_m_s * math.sin(offset),
        z0m=record.surface_roughness_m,
        z0h=record.surface_roughness_m / 10.0,
        thl_surface_K=record.potential_temperature_k,
        thl_lapse_K_per_m=record.potential_temperature_gradient_k_m,
        boundary_layer_height_m=record.mixing_height_m,
        inversion_strength_K=inversion_strength_K,
        inversion_depth_m=inversion_depth_m,
        surface_heat_flux_K_m_s=record.kinematic_heat_flux_k_m_s,
    )


def microhh_box_bearing(met: MetSeries | MetRecord, reduce: Reduce = "mean") -> float:
    """The bearing a MicroHH box should use so its +x axis points downwind."""
    return _reduce(met, reduce).wind_toward_deg


# ── FLEXPART ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FlexpartMetSource:
    """The met inputs a FLEXPART run needs, validated against a time window.

    FLEXPART reads ERA5 GRIB itself, so this is a pointer rather than a
    conversion: the directory and ``AVAILABLE`` index, plus whether they
    actually cover the requested period.
    """

    meteo_dir: Path
    available_file: Path
    covers_window: bool
    start: Any = None
    end: Any = None

    def as_config(self) -> dict[str, str]:
        """The ``flexpart:`` keys a simulation YAML expects."""
        return {
            "meteo_dir": str(self.meteo_dir),
            "available_file": str(self.available_file),
        }

    def require_coverage(self) -> "FlexpartMetSource":
        """Raise unless the meteorology covers the requested window."""
        if not self.covers_window:
            raise ValueError(
                f"ERA5 meteorology in {self.meteo_dir} does not cover "
                f"{self.start} → {self.end}. Download the missing timesteps "
                "before running FLEXPART."
            )
        return self


def to_flexpart(
    met: MetSeries,
    *,
    start: Any = None,
    end: Any = None,
    timestep_hours: int = 3,
) -> FlexpartMetSource:
    """Canonical met → the ERA5 files FLEXPART should be pointed at.

    Only meaningful for a series read from ERA5: FLEXPART cannot be driven by
    scalars, so what it needs is the GRIB directory the series came from, with
    its ``AVAILABLE`` index checked for coverage of ``[start, end]`` (defaulting
    to the series' own span).
    """
    from enforceflux.meteo.era5 import available_covers_window

    meteo_dir = met.provenance.get("meteo_dir")
    if not meteo_dir:
        raise ValueError(
            "to_flexpart needs a series read from ERA5 GRIB (its provenance "
            "carries the meteo_dir). FLEXPART cannot be driven by canonical "
            "scalars — it reads the GRIB files directly."
        )

    meteo_path = Path(meteo_dir)
    available = meteo_path / "AVAILABLE"
    window_start = start or met.start
    window_end = end or met.end
    covers = available.exists() and available_covers_window(
        available, window_start, window_end, timestep_hours=timestep_hours
    )
    return FlexpartMetSource(
        meteo_dir=meteo_path,
        available_file=available,
        covers_window=covers,
        start=window_start,
        end=window_end,
    )
