"""The single public coordinate contract used across EnforceFlux.

The run declares one geographic origin.  Every public position is then
``x_m`` metres east and ``y_m`` metres north of that origin.  Model-native
rotations are private implementation details and may never appear in a
canonical field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np

from enforceflux.transport.run_config import DomainProjection


class LocalMetricFrame(Protocol):
    """A metric model frame with a reversible geographic mapping."""

    def local_to_lonlat(self, x_m, y_m): ...
    def lonlat_to_local(self, longitude, latitude): ...


@dataclass(frozen=True)
class EastNorthFrame:
    """Local east/north metres about a geographic origin."""

    projection: DomainProjection

    @classmethod
    def from_origin(cls, origin_lon: float, origin_lat: float) -> "EastNorthFrame":
        return cls(DomainProjection(float(origin_lon), float(origin_lat)))

    def local_to_lonlat(self, x_m, y_m):
        return self.projection.to_lonlat(x_m, y_m)

    def lonlat_to_local(self, longitude, latitude):
        return self.projection.to_xy(longitude, latitude)


def frame_from_canonical(attrs: Mapping[str, Any]) -> LocalMetricFrame:
    """Validate and construct the canonical east/north frame."""
    kind = str(attrs.get("coordinate_frame", "")).strip().lower()
    if kind == "east_north":
        return EastNorthFrame.from_origin(
            float(attrs["frame_center_lon"]), float(attrs["frame_center_lat"])
        )
    if not kind:
        raise ValueError(
            "Canonical field is missing required coordinate_frame metadata; "
            "regenerate it with the current transport adapter"
        )
    raise ValueError(
        f"Unsupported canonical coordinate_frame {kind!r}; only 'east_north' "
        "is public"
    )


def geographic_path_to_local(
    frame: LocalMetricFrame, longitude: float, latitude: float,
    length_m: float, bearing_deg: float,
) -> tuple[float, float, float, float]:
    """Convert a geographic path into local start, length, and bearing.

    Bearings use the common compass convention: degrees clockwise from the
    local +y axis. Transforming the endpoint along with the start avoids any
    assumption about how a model rotates its axes.
    """
    from pyproj import Geod

    x0, y0 = frame.lonlat_to_local(float(longitude), float(latitude))
    if float(length_m) <= 0.0:
        return float(x0), float(y0), 0.0, float(bearing_deg) % 360.0
    lon1, lat1, _ = Geod(ellps="WGS84").fwd(
        float(longitude), float(latitude), float(bearing_deg), float(length_m)
    )
    x1, y1 = frame.lonlat_to_local(lon1, lat1)
    dx, dy = float(x1) - float(x0), float(y1) - float(y0)
    return float(x0), float(y0), float(np.hypot(dx, dy)), float(np.degrees(np.arctan2(dx, dy)) % 360.0)


def local_path_to_geographic(
    frame: LocalMetricFrame, x_m: float, y_m: float,
    length_m: float, bearing_deg: float,
) -> tuple[float, float, float, float]:
    """Convert a local metric path into geographic start, length, and bearing."""
    from pyproj import Geod

    lon0, lat0 = frame.local_to_lonlat(float(x_m), float(y_m))
    if float(length_m) <= 0.0:
        return float(lon0), float(lat0), 0.0, float(bearing_deg) % 360.0
    angle = np.deg2rad(float(bearing_deg))
    x1 = float(x_m) + float(length_m) * float(np.sin(angle))
    y1 = float(y_m) + float(length_m) * float(np.cos(angle))
    lon1, lat1 = frame.local_to_lonlat(x1, y1)
    azimuth, _, distance = Geod(ellps="WGS84").inv(
        float(lon0), float(lat0), float(lon1), float(lat1)
    )
    return float(lon0), float(lat0), float(distance), float(azimuth % 360.0)
