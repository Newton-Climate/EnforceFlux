"""Integration test for the bLSmodelR transport operator plugin.

Uses ``dry_run=True`` so the R shim runs its analytic stub — no bLSmodelR
install required, only ``Rscript`` on ``PATH``.
"""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from enforceflux.instrument import Instrument
from enforceflux.plugins.transport_blsmodelr import BlsTransportOperator


pytestmark = pytest.mark.skipif(
    shutil.which("Rscript") is None,
    reason="Rscript not on PATH; needed even for the bLS dry-run shim.",
)


def _make_instruments() -> list[Instrument]:
    return [
        Instrument(id="s0", tech_id="OP", x=-100.0, y=0.0, z=2.0),
        Instrument(id="s1", tech_id="OP", x=0.0, y=0.0, z=2.0),
        Instrument(id="s2", tech_id="OP", x=100.0, y=50.0, z=2.0),
    ]


def _config() -> dict:
    return {
        "source_grid": {
            "x_bounds": [-500.0, 500.0],
            "y_bounds": [-500.0, 500.0],
            "nx": 10,
            "ny": 10,
        },
        "intervals": [
            {
                "id": "t0",
                "u_star": 0.35,
                "L": -50.0,
                "z0": 0.05,
                "wind_dir_deg": 270.0,
                "wind_speed": 3.0,
                "z_ref": 4.0,
            }
        ],
        "interval_reduce": "mean",
        "wrapper": {"dry_run": True, "rscript": "Rscript"},
    }


def test_bls_transport_operator_builds_jacobian():
    op = BlsTransportOperator()
    instruments = _make_instruments()
    config = _config()

    result = op.build_forward_operator(
        sources=[], instruments=instruments, domain=None, config=config
    )

    nx = config["source_grid"]["nx"]
    ny = config["source_grid"]["ny"]
    assert result.g.shape == (len(instruments), nx * ny)
    assert np.all(np.isfinite(result.g))
    assert np.all(result.g >= 0.0)

    assert result.meta["n_sources"] == nx * ny
    assert result.meta["n_intervals"] == 1
    assert result.meta["cell_area_m2"] == pytest.approx(
        (1000.0 / nx) * (1000.0 / ny)
    )
    assert result.meta["receptor_path_samples"] == 8


def test_bls_transport_operator_averages_open_path_subpoints():
    op = BlsTransportOperator()
    config = _config()
    config["receptor_path_samples"] = 4
    beam = Instrument(
        id="beam", tech_id="OP", x=-200.0, y=-100.0, z=2.0,
        path_length_m=400.0, path_bearing_deg=0.0,
    )

    result = op.build_forward_operator(
        sources=[], instruments=[beam], domain=None, config=config
    )

    assert result.g.shape == (1, 100)
    assert result.meta["receptor_path_samples"] == 4


def test_bls_transport_operator_refuses_single_point_open_path():
    op = BlsTransportOperator()
    config = _config()
    config["receptor_path_samples"] = 1
    beam = Instrument(
        id="beam", tech_id="OP", x=0.0, y=0.0, z=2.0,
        path_length_m=400.0, path_bearing_deg=0.0,
    )

    with pytest.raises(ValueError, match="may not silently degrade"):
        op.build_forward_operator(
            sources=[], instruments=[beam], domain=None, config=config
        )


def _two_intervals(config: dict) -> dict:
    """Same config with a second, differently-forced turbulence window.

    Wind speed must be among the differences: the dry-run shim is an analytic
    stub that ignores direction and u*, so two windows differing only in those
    produce identical rows and the ordering assertion below would hold
    vacuously.
    """
    second = dict(config["intervals"][0])
    second.update(id="t1", u_star=0.20, wind_dir_deg=250.0, wind_speed=2.0)
    config["intervals"] = [config["intervals"][0], second]
    return config


def test_bls_operator_none_gives_one_row_per_interval_and_instrument():
    config = _two_intervals(_config())
    config["interval_reduce"] = "none"
    instruments = _make_instruments()

    result = BlsTransportOperator().build_forward_operator(
        sources=[], instruments=instruments, domain=None, config=config
    )

    n_src = config["source_grid"]["nx"] * config["source_grid"]["ny"]
    assert result.g.shape == (2 * len(instruments), n_src)
    assert np.all(np.isfinite(result.g))


def test_bls_operator_none_is_ordered_interval_major():
    """The flux stage indexes operator rows as ``t * n_instruments + i``."""
    config = _two_intervals(_config())
    instruments = _make_instruments()

    config["interval_reduce"] = "none"
    stacked = BlsTransportOperator().build_forward_operator(
        sources=[], instruments=instruments, domain=None, config=config
    ).g
    config["interval_reduce"] = "mean"
    averaged = BlsTransportOperator().build_forward_operator(
        sources=[], instruments=instruments, domain=None, config=config
    ).g

    n = len(instruments)
    # Interval-major: the two blocks of n rows must average to the mean path.
    np.testing.assert_allclose(
        (stacked[:n] + stacked[n:]) / 2.0, averaged, rtol=1e-9
    )
    # The two intervals are forced differently, so the blocks must differ —
    # otherwise the ordering assertion above would pass on duplicated rows.
    assert not np.allclose(stacked[:n], stacked[n:]), (
        "interval blocks are identical; the ordering assertion above is vacuous"
    )
