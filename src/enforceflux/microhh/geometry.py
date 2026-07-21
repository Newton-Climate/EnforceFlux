"""Lon/lat ↔ local-metre projection for MicroHH cases.

MicroHH runs in a Cartesian box whose axes are metres from a domain origin.
Our sources and instruments are specified in lon/lat (to stay consistent with
the FLEXPART configs), so we need a small, dependency-free projection that:

1. converts lon/lat to local east/north metres about a reference point
   (equirectangular / local-tangent-plane; adequate over an LES-sized domain
   of a few km), and
2. rotates into a *wind-aligned* frame so the box x-axis points downwind — the
   standard LES setup for a dispersing plume with cyclic/streamwise boundaries.

The source is pinned to a fixed location inside the box (``source_x0`` /
``source_y0``); every other point is placed relative to it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class BoxProjection:
    """Projects lon/lat onto a wind-aligned MicroHH box in metres.

    Parameters
    ----------
    origin_lon, origin_lat
        Reference point (typically the primary source) for the local tangent
        plane.
    x_bearing_deg
        Compass bearing (degrees clockwise from north) that the box **x-axis**
        points toward — i.e. the mean/downwind direction. ``0`` → x points
        north; ``90`` → x points east.
    source_x0, source_y0
        Where the origin point lands inside the box, in metres. Defaults place
        the source near the upwind edge, centred cross-stream, so the plume has
        room to develop downwind.
    """

    origin_lon: float
    origin_lat: float
    x_bearing_deg: float
    source_x0: float
    source_y0: float

    def to_box(self, lon: float, lat: float) -> tuple[float, float]:
        """Return ``(x, y)`` box coordinates in metres for a lon/lat point."""
        lat0 = math.radians(self.origin_lat)
        east = math.radians(lon - self.origin_lon) * _EARTH_RADIUS_M * math.cos(lat0)
        north = math.radians(lat - self.origin_lat) * _EARTH_RADIUS_M

        # Downwind unit vector (box +x) and left-cross-stream unit vector (box
        # +y), forming a right-handed frame in the (east, north) plane.
        b = math.radians(self.x_bearing_deg)
        sin_b, cos_b = math.sin(b), math.cos(b)
        x_down = east * sin_b + north * cos_b
        y_cross = -east * cos_b + north * sin_b

        return (self.source_x0 + x_down, self.source_y0 + y_cross)

    def to_lonlat(self, x, y):
        """Inverse of :meth:`to_box`: box metres → ``(lon, lat)``.

        Accepts scalars or arrays, so a whole cross-section grid can be given
        geographic coordinates in one call.
        """
        import numpy as np

        x_down = np.asarray(x, dtype=float) - self.source_x0
        y_cross = np.asarray(y, dtype=float) - self.source_y0

        # The forward rotation is orthogonal, so the inverse is its transpose.
        b = math.radians(self.x_bearing_deg)
        sin_b, cos_b = math.sin(b), math.cos(b)
        east = x_down * sin_b - y_cross * cos_b
        north = x_down * cos_b + y_cross * sin_b

        lat0 = math.radians(self.origin_lat)
        lon = self.origin_lon + np.degrees(east / (_EARTH_RADIUS_M * math.cos(lat0)))
        lat = self.origin_lat + np.degrees(north / _EARTH_RADIUS_M)
        return lon, lat
