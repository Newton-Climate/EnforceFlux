import pytest

from enforceflux.coordinates import (
    EastNorthFrame,
    frame_from_canonical,
    geographic_path_to_local,
    local_path_to_geographic,
)


ORIGIN_LON = -121.75
ORIGIN_LAT = 39.15


def test_east_north_frame_preserves_compass_bearing():
    frame = EastNorthFrame.from_origin(ORIGIN_LON, ORIGIN_LAT)
    lon, lat, length, bearing = local_path_to_geographic(
        frame, 0.0, 0.0, 500.0, 90.0
    )
    assert length == pytest.approx(500.0, rel=1e-6)
    assert bearing == pytest.approx(90.0, abs=1e-6)
    x, y, local_length, local_bearing = geographic_path_to_local(
        frame, lon, lat, length, bearing
    )
    assert (x, y) == pytest.approx((0.0, 0.0), abs=1e-6)
    assert local_length == pytest.approx(500.0, rel=1e-6)
    assert local_bearing == pytest.approx(90.0, abs=1e-6)


def test_canonical_frame_factory_is_strict_and_explicit():
    with pytest.raises(ValueError, match="missing required coordinate_frame"):
        frame_from_canonical({})
    with pytest.raises(ValueError, match="Unsupported canonical coordinate_frame"):
        frame_from_canonical({"coordinate_frame": "mystery"})

    frame = frame_from_canonical({
        "coordinate_frame": "east_north",
        "frame_center_lon": ORIGIN_LON,
        "frame_center_lat": ORIGIN_LAT,
    })
    assert isinstance(frame, EastNorthFrame)

    with pytest.raises(ValueError, match="only 'east_north'"):
        frame_from_canonical({
        "coordinate_frame": "wind_aligned",
        "frame_center_lon": ORIGIN_LON,
        "frame_center_lat": ORIGIN_LAT,
        "frame_x_bearing_deg": 30.0,
        })
