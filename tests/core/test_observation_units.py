import pytest

from enforceflux.core.observation_units import ObservationSpec, assert_matches


def _spec(**overrides) -> ObservationSpec:
    base = dict(
        variable="ch4",
        unit="ppb",
        averaging_window_s=60.0,
        height_m=10.0,
        path_length_m=0.0,
    )
    base.update(overrides)
    return ObservationSpec(**base)


def test_matching_passes():
    assert_matches(_spec(), _spec())


def test_mismatch_raises():
    actual = _spec(
        variable="co2",
        unit="ppm",
        averaging_window_s=30.0,
        height_m=5.0,
        path_length_m=2.0,
    )
    expected = _spec()
    with pytest.raises(ValueError) as excinfo:
        assert_matches(actual, expected)
    msg = str(excinfo.value)
    for field in ("variable", "unit", "averaging_window_s", "height_m", "path_length_m"):
        assert field in msg


def test_numeric_tolerance():
    a = _spec(averaging_window_s=60.0)
    e = _spec(averaging_window_s=60.0 + 1e-12)
    assert_matches(a, e)

    a2 = _spec(height_m=10.0)
    e2 = _spec(height_m=10.001)
    with pytest.raises(ValueError):
        assert_matches(a2, e2)
