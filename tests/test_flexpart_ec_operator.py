import dataclasses
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from enforceflux.flexpart.ec_operator import (
    build_ec_observation_operator_from_backward_runs,
    build_ec_observation_operator_from_flexpart,
)
from enforceflux.flexpart.sim_config import SimulationConfig
from enforceflux.instrument import Instrument
from enforceflux.models.config import DomainConfig
from enforceflux.models.source import Source


def test_build_ec_observation_operator_from_flexpart_normalizes_overlap():
    g_raw = np.array([[[2.0, 1.0], [0.0, 0.0]]])
    areas = np.array([10.0, 30.0])
    times_s = np.array([0.0])

    result = build_ec_observation_operator_from_flexpart(g_raw, areas, times_s)

    expected_row = (np.array([2.0, 1.0]) / (2.0 * 10.0 + 1.0 * 30.0)) * (1e9 / 16.04e-3)
    assert np.allclose(result.g[0, 0], expected_row)
    assert result.valid_mask.tolist() == [[True, False]]
    assert np.all(np.isnan(result.g[0, 1]))


def test_build_ec_observation_operator_from_backward_runs_uses_runner_output(monkeypatch, tmp_path):
    calls = []

    class FakeRunner:
        def __init__(self, base_config, domain, config):
            calls.append((base_config, domain, config))

        def run(self, instruments, sources):
            return dataclasses.make_dataclass("Result", [("g", np.ndarray), ("meta", dict)])(
                g=np.array([[3.0, 1.0]]),
                meta={"runner": "fake"},
            )

    monkeypatch.setattr("enforceflux.flexpart.ec_operator.FlexpartBackwardRunner", FakeRunner)

    base_config = SimulationConfig(
        executable=tmp_path / "FLEXPART",
        options_dir=tmp_path / "opts",
        available_file=tmp_path / "AVAILABLE",
        meteo_dir=tmp_path / "meteo",
        run_dir=tmp_path / "run",
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2020, 1, 1, 1, tzinfo=timezone.utc),
        output_step_s=1800,
        domain_lon_min=0.0,
        domain_lat_min=0.0,
        domain_lon_max=1.0,
        domain_lat_max=1.0,
        domain_dx=0.1,
        domain_dy=0.1,
        heights_m=[100.0],
        sources=[],
        output_path=tmp_path / "out.nc",
    )
    domain = DomainConfig(0.0, 1.0, 0.0, 1.0, 0.1, crs="EPSG:4326")
    instruments = [Instrument(id="EC1", tech_id="EC", x=0.0, y=0.0)]
    sources = [
        Source(id="S1", kind="point", x=0.0, y=0.0, flux_true=1.0, flux_prior_mean=1.0, flux_prior_std=1.0),
        Source(id="S2", kind="point", x=1.0, y=1.0, flux_true=1.0, flux_prior_mean=1.0, flux_prior_std=1.0),
    ]

    result = build_ec_observation_operator_from_backward_runs(
        base_config=base_config,
        domain=domain,
        instruments=instruments,
        sources=sources,
        source_areas_m2=np.array([5.0, 5.0]),
        sample_times_s=np.array([0.0, 1800.0]),
        lookback_s=1800.0,
        runner_config={"dry_run": True},
    )

    assert result.g.shape == (2, 1, 2)
    assert result.valid_mask.tolist() == [[True], [True]]
    assert len(calls) == 2
    assert Path(calls[0][2]["base_run_dir"]).name == "ec_t0000"
