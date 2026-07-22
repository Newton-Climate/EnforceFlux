import numpy as np
import pytest

from enforceflux.instrument import INSTRUMENT_DB, Instrument, InstrumentOperator, OperatorParams


@pytest.fixture
def custom_instruments():
    added = []

    def _register(name: str, params: OperatorParams) -> Instrument:
        INSTRUMENT_DB[name] = {"good": params}
        added.append(name)
        return Instrument(id=name, tech_id=name, x=0.0, y=0.0)

    yield _register

    for name in added:
        INSTRUMENT_DB.pop(name, None)


def test_simulate_time_series_respects_cadence_and_window_average(custom_instruments):
    inst = custom_instruments(
        "TEST_CADENCE",
        OperatorParams(
            tech_id="TEST_CADENCE",
            mode="good",
            operator_type="point_flux",
            observable="flux_nmol_m2_s",
            sigma_scale=0.0,
            sigma_abs=0.0,
            bias_scale=0.0,
            bias_abs=0.0,
            detection_limit=0.0,
            dropout_probability=0.0,
            cadence_s=180.0,
        ),
    )
    op = InstrumentOperator([inst], rng=np.random.default_rng(0))
    times_s = np.arange(0.0, 361.0, 60.0)
    g_series = np.ones((len(times_s), 1, 1))
    x_true_series = np.arange(1.0, len(times_s) + 1.0).reshape(-1, 1)

    result = op.simulate_time_series(g_series, x_true_series, times_s)

    assert np.allclose(result.H_g[:, 0, 0], 1.0)
    assert np.isnan(result.y_clean[1, 0])
    assert np.isnan(result.y_clean[2, 0])
    assert result.valid_mask[:, 0].tolist() == [True, False, False, True, False, False, True]
    assert np.allclose(result.y_clean[[0, 3, 6], 0], [1.0, 2.5, 5.5])
    assert np.allclose(result.y_obs[[0, 3, 6], 0], [1.0, 2.5, 5.5])


def test_simulate_time_series_applies_bias_and_relative_noise(custom_instruments):
    inst = custom_instruments(
        "TEST_REL_BIAS",
        OperatorParams(
            tech_id="TEST_REL_BIAS",
            mode="good",
            operator_type="point_flux",
            observable="flux_nmol_m2_s",
            sigma_scale=0.2,
            sigma_abs=1.0,
            bias_scale=0.1,
            bias_abs=0.5,
            detection_limit=0.0,
            dropout_probability=0.0,
            cadence_s=60.0,
        ),
    )
    seed = 7
    op = InstrumentOperator([inst], rng=np.random.default_rng(seed))
    times_s = np.array([0.0, 60.0])
    g_series = np.ones((2, 1, 1))
    x_true_series = np.array([[10.0], [20.0]])

    result = op.simulate_time_series(g_series, x_true_series, times_s)

    expected_sigmas = np.array([
        np.sqrt((0.2 * 10.0) ** 2 + 1.0**2),
        np.sqrt((0.2 * 15.0) ** 2 + 1.0**2),
    ])
    expected_clean = np.array([10.0, 15.0])

    replay_rng = np.random.default_rng(seed)
    expected_obs = []
    for clean, sigma in zip(expected_clean, expected_sigmas):
        expected_obs.append(clean * 1.1 + 0.5 + replay_rng.normal(0.0, sigma))
        replay_rng.random()
    expected_obs = np.array(expected_obs)

    assert np.allclose(result.y_clean[:, 0], expected_clean)
    assert np.allclose(np.sqrt(np.diagonal(result.R, axis1=1, axis2=2)[:, 0]), expected_sigmas)
    assert np.allclose(result.y_obs[:, 0], expected_obs)


def test_simulate_time_series_dropout_and_detection_limit_invalidate_observations(custom_instruments):
    dl_inst = custom_instruments(
        "TEST_DL",
        OperatorParams(
            tech_id="TEST_DL",
            mode="good",
            operator_type="point_flux",
            observable="flux_nmol_m2_s",
            sigma_scale=0.0,
            sigma_abs=0.0,
            bias_scale=0.0,
            bias_abs=0.0,
            detection_limit=5.0,
            dropout_probability=0.0,
            cadence_s=60.0,
        ),
    )
    drop_inst = custom_instruments(
        "TEST_DROP",
        OperatorParams(
            tech_id="TEST_DROP",
            mode="good",
            operator_type="point_flux",
            observable="flux_nmol_m2_s",
            sigma_scale=0.0,
            sigma_abs=0.0,
            bias_scale=0.0,
            bias_abs=0.0,
            detection_limit=0.0,
            dropout_probability=1.0,
            cadence_s=60.0,
        ),
    )
    times_s = np.array([0.0])
    g_series = np.ones((1, 2, 1))
    x_true_series = np.array([[3.0]])

    result = InstrumentOperator([dl_inst, drop_inst], rng=np.random.default_rng(0)).simulate_time_series(
        g_series, x_true_series, times_s
    )

    assert np.isclose(result.y_clean[0, 0], 3.0)
    assert np.isclose(result.y_clean[0, 1], 3.0)
    assert not result.valid_mask[0, 0]
    assert not result.valid_mask[0, 1]
    assert np.isnan(result.y_obs[0, 0])
    assert np.isnan(result.y_obs[0, 1])
    assert np.isinf(result.R[0, 0, 0])
    assert np.isinf(result.R[0, 1, 1])


def test_simulate_observations_backward_compatible(custom_instruments):
    inst = custom_instruments(
        "TEST_BACKWARD",
        OperatorParams(
            tech_id="TEST_BACKWARD",
            mode="good",
            operator_type="point_flux",
            observable="flux_nmol_m2_s",
            sigma_scale=0.0,
            sigma_abs=0.003,
            bias_scale=0.0,
            bias_abs=0.0,
            detection_limit=0.0,
            dropout_probability=0.0,
            cadence_s=60.0,
        ),
    )
    seed = 5
    op = InstrumentOperator([inst], rng=np.random.default_rng(seed))
    g = np.array([[2.0]])
    x_true = np.array([3.0])

    result = op.simulate_observations(g, x_true)

    sigma = INSTRUMENT_DB["TEST_BACKWARD"]["good"].sigma_abs
    replay_rng = np.random.default_rng(seed)
    replay_rng.random()
    expected = 6.0 + replay_rng.normal(0.0, sigma)

    assert np.allclose(result.H_g, g)
    assert np.allclose(result.y_clean, [6.0])
    assert np.allclose(result.y_obs, [expected])
    assert result.valid_mask.tolist() == [True]
    assert np.allclose(result.R, np.diag([sigma**2]))


def test_simulate_time_series_invalid_physical_rows_stay_invalid(custom_instruments):
    inst = custom_instruments(
        "TEST_INVALID_ROW",
        OperatorParams(
            tech_id="TEST_INVALID_ROW",
            mode="good",
            operator_type="point_flux",
            observable="flux_nmol_m2_s",
            sigma_scale=0.0,
            sigma_abs=0.0,
            bias_scale=0.0,
            bias_abs=0.0,
            detection_limit=0.0,
            dropout_probability=0.0,
            cadence_s=60.0,
        ),
    )
    times_s = np.array([0.0, 60.0])
    g_series = np.array([
        [[np.nan]],
        [[2.0]],
    ])
    x_true_series = np.array([[1.0], [1.0]])

    result = InstrumentOperator([inst], rng=np.random.default_rng(0)).simulate_time_series(
        g_series, x_true_series, times_s
    )

    assert np.isnan(result.y_clean[0, 0])
    assert not result.valid_mask[0, 0]
    assert np.isnan(result.y_obs[0, 0])
    assert np.isinf(result.R[0, 0, 0])
    assert np.isclose(result.y_clean[1, 0], 2.0)
    assert result.valid_mask[1, 0]
