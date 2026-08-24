import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "apps"))
from instrument_main import _parse_instruments, _validate_instruments


def test_instrument_yaml_uses_east_north_metres():
    instruments = _parse_instruments([
        {"id": "op", "tech_id": "OP", "x_m": 100.0, "y_m": -50.0,
         "path_length_m": 200.0, "path_bearing_deg": 90.0}
    ])
    assert (instruments[0].x, instruments[0].y) == (100.0, -50.0)
    assert instruments[0].path_bearing_deg == 90.0


def test_instrument_yaml_rejects_lon_lat():
    with pytest.raises(ValueError, match="use x_m/y_m"):
        _parse_instruments([
            {"id": "old", "tech_id": "OP", "lon": -121.75, "lat": 39.15}
        ])


def test_full_beam_must_fit_east_north_grid():
    instrument = _parse_instruments([
        {"id": "op", "tech_id": "OP", "x_m": 900.0, "y_m": 0.0,
         "path_length_m": 200.0, "path_bearing_deg": 90.0}
    ])
    with pytest.raises(ValueError, match="outside"):
        _validate_instruments(instrument, np.linspace(-1000, 1000, 21),
                              np.linspace(-1000, 1000, 21))
