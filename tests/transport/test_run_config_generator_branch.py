from pathlib import Path

from enforceflux.plugins.source_lognormal_field import clear_pending_writes
from enforceflux.transport.run_config import RunSource, TransportRunConfig


BASE_DOMAIN = {
    "origin_lon": -121.75,
    "origin_lat": 39.15,
    "x_min": -400.0, "x_max": 400.0,
    "y_min": -400.0, "y_max": 400.0,
    "spacing_m": 100.0,
}


def _list_sources_blob():
    return {
        "transport": {"model": "aermod", "mode": "simulation",
                      "start": "2020-01-01T00:00:00", "end": "2020-01-01T01:00:00"},
        "domain": BASE_DOMAIN,
        "sources": [
            {"id": "s0", "x_m": 0.0, "y_m": 0.0, "emission_rate_kg_s": 1.0e-3, "alt_m": 2.0},
        ],
        "receptors": [
            {"id": "r0", "x_m": 100.0, "y_m": 100.0, "alt_m": 3.0},
        ],
    }


def _generator_blob():
    return {
        "transport": {"model": "aermod", "mode": "simulation",
                      "start": "2020-01-01T00:00:00", "end": "2020-01-01T01:00:00"},
        "domain": BASE_DOMAIN,
        "sources": {
            "generator": "lognormal_field",
            "config": {
                "Q_true_kg_s": 1.0e-2,
                "grid": {"nx": 4, "ny": 4, "dx_m": 200.0,
                         "origin_x_m": -400.0, "origin_y_m": -400.0},
                "alt_m": 2.0,
                "covariance": {"model": "exponential", "L_m": 200.0},
                "cv": 0.3,
                "seed": 3,
                "basis": {"coarsen": 2},
                "prior": {"mean_kg_s_per_cell": 0.0},
            },
        },
        "receptors": [
            {"id": "r0", "x_m": 100.0, "y_m": 100.0, "alt_m": 3.0},
        ],
    }


def test_list_sources_unchanged(tmp_path):
    clear_pending_writes()
    run = TransportRunConfig.from_dict(
        _list_sources_blob(),
        base_dir=tmp_path,
        output_path=tmp_path / "concentration.nc",
    )
    assert len(run.sources) == 1
    assert run.sources[0].id == "s0"
    assert isinstance(run.sources[0], RunSource)


def test_generator_branch_expands(tmp_path):
    clear_pending_writes()
    run = TransportRunConfig.from_dict(
        _generator_blob(),
        base_dir=tmp_path,
        output_path=tmp_path / "concentration.nc",
    )
    assert len(run.sources) == 4 * 4
    for s in run.sources:
        assert isinstance(s, RunSource)
    total = sum(s.emission_rate_kg_s for s in run.sources)
    assert abs(total - 1.0e-2) / 1.0e-2 < 1.0e-12
    clear_pending_writes()
