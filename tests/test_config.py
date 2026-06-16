import json

from enforceflux.models.config import load_config


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
            "transport_operator": {
                "plugin": "enforceflux.transport_operator.gaussian",
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
    assert config.component("transport_operator").config["sigma"] == 1.0
