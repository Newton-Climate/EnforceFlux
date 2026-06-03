import json

from enforceflux.config import load_config


def test_legacy_config_conversion(tmp_path):
    legacy = {
        "domain": {
            "x_min": 0,
            "x_max": 1,
            "y_min": 0,
            "y_max": 1,
            "grid_spacing": 1,
        },
        "sources": [
            {
                "id": "S1",
                "kind": "point",
                "x": 0,
                "y": 0,
                "flux_true": 1.0,
            }
        ],
        "instruments": [
            {
                "id": "I1",
                "kind": "open_path",
                "x": 0,
                "y": 0,
                "noise_std": 0.1,
            }
        ],
        "transport": {"model": "gaussian", "sigma": 100.0, "wind": [0.0, 0.0]},
    }

    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy))

    config = load_config(path)
    assert config.component("source").plugin == "enforceflux.source.static"
    assert config.component("instrument").plugin == "enforceflux.instrument.static"
    assert config.component("transport").plugin == "enforceflux.transport.gaussian"
    assert config.component("inversion").plugin == "enforceflux.inversion.bayesian"


def test_new_config_parsing(tmp_path):
    new_cfg = {
        "domain": {
            "x_min": 0,
            "x_max": 1,
            "y_min": 0,
            "y_max": 1,
            "grid_spacing": 1,
            "crs": "EPSG:32610",
        },
        "components": {
            "source": {"plugin": "enforceflux.source.static", "config": {"sources": []}},
            "instrument": {
                "plugin": "enforceflux.instrument.static",
                "config": {"instruments": []},
            },
            "transport": {
                "plugin": "enforceflux.transport.gaussian",
                "config": {"sigma": 1.0, "wind": [0.0, 0.0]},
            },
            "inversion": {
                "plugin": "enforceflux.inversion.bayesian",
                "config": {"r_cond": 1e-6},
            },
        },
    }

    path = tmp_path / "new.json"
    path.write_text(json.dumps(new_cfg))

    config = load_config(path)
    assert config.domain.crs == "EPSG:32610"
    assert config.component("transport").config["sigma"] == 1.0
