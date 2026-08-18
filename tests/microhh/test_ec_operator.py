import numpy as np
import pytest

from enforceflux.microhh.ec_operator import build_ec_observation_operator_from_les


def test_les_ec_operator_recovers_known_covariance_and_source_scaling():
    times = np.arange(0.0, 120.0, 1.0)
    w = np.sin(2.0 * np.pi * times / 20.0)[:, None]
    # q0 = 2*w from a 2 kg/s source; q1 = -w from a 0.5 kg/s source.
    q = np.stack([2.0 * w[:, 0], -w[:, 0]], axis=1)[:, None, :]
    result = build_ec_observation_operator_from_les(
        w_m_s=w,
        scalar_responses_kg_kg=q,
        times_s=times,
        source_emission_rates_kg_s=np.array([2.0, 0.5]),
        air_density_kg_m3=1.0,
        window_s=60.0,
        min_samples=50,
        n_error_blocks=3,
    )

    conversion = 1e9 / 16.04e-3
    # var(sine)=0.5; linear detrending changes the finite-window value slightly,
    # so compare against the exact operation used by the model via source ratio.
    assert result.g.shape == (1, 1, 2)
    assert result.valid_mask[0, 0]
    assert result.g[0, 0, 0] > 0
    assert result.g[0, 0, 1] < 0
    assert result.g[0, 0, 0] / conversion == pytest.approx(
        -0.5 * result.g[0, 0, 1] / conversion, rel=1e-12
    )


def test_les_ec_operator_rejects_low_turbulence_window():
    times = np.arange(0.0, 61.0, 1.0)
    w = np.sin(times)[:, None]
    q = w[:, :, None]
    result = build_ec_observation_operator_from_les(
        w_m_s=w,
        scalar_responses_kg_kg=q,
        times_s=times,
        source_emission_rates_kg_s=np.array([1.0]),
        window_s=60.0,
        min_samples=50,
        min_ustar_m_s=0.1,
        ustar_m_s=np.full_like(w, 0.05),
    )
    assert not result.valid_mask[0, 0]
    assert np.isnan(result.g[0, 0, 0])


def test_les_ec_operator_does_not_renormalize_weak_source_response():
    times = np.arange(0.0, 61.0, 1.0)
    w = np.sin(times)[:, None]
    strong = w[:, 0]
    weak = 1e-4 * w[:, 0]
    q = np.stack([strong, weak], axis=1)[:, None, :]
    result = build_ec_observation_operator_from_les(
        w_m_s=w,
        scalar_responses_kg_kg=q,
        times_s=times,
        source_emission_rates_kg_s=np.ones(2),
        window_s=60.0,
        min_samples=50,
    )
    assert result.g[0, 0, 1] / result.g[0, 0, 0] == pytest.approx(1e-4, rel=1e-10)


def test_cadence_averages_jacobian_over_same_window():
    from enforceflux.instrument import Instrument, InstrumentOperator, OperatorParams

    params = OperatorParams(
        tech_id="X", mode="good", operator_type="point_flux", observable="flux_nmol_m2_s",
        sigma_scale=0.0, sigma_abs=0.0, bias_scale=0.0, bias_abs=0.0,
        detection_limit=0.0, dropout_probability=0.0, cadence_s=2.0,
    )
    inst = Instrument(id="x", tech_id="EC", x=0.0, y=0.0, params_override=params)
    g = np.array([[[1.0]], [[3.0]], [[5.0]]])
    result = InstrumentOperator([inst], rng=np.random.default_rng(0)).simulate_time_series(
        g, np.array([2.0]), np.array([0.0, 1.0, 2.0])
    )
    assert result.y_clean[2, 0] == pytest.approx(6.0)
    assert result.H_g[2, 0, 0] == pytest.approx(3.0)
