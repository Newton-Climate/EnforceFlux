import numpy as np

from enforceflux.flux import EddyCovarianceFluxEstimator, EddyCovarianceWindow


def test_ec_estimator_uses_precomputed_flux():
    estimator = EddyCovarianceFluxEstimator()
    result = estimator.estimate(EddyCovarianceWindow(flux=3.5, n_samples=10), {})

    assert np.allclose(result.flux, [3.5])
    assert result.meta["valid_mask"].tolist() == [True]
    assert result.meta["methods"] == ["flux"]


def test_ec_estimator_computes_covariance_from_primes():
    estimator = EddyCovarianceFluxEstimator()
    window = EddyCovarianceWindow(
        w_prime=np.array([1.0, -1.0, 2.0]),
        c_prime=np.array([4.0, 2.0, -1.0]),
        qc_passed=True,
    )

    result = estimator.estimate(window, {})

    assert np.allclose(result.flux, [0.0])
    assert result.meta["methods"] == ["covariance_from_primes"]


def test_ec_estimator_rejects_failed_qc():
    estimator = EddyCovarianceFluxEstimator()
    result = estimator.estimate(
        [{"flux": 2.0, "qc_passed": False, "timestamp_s": 60.0}],
        {"reject_failed_qc": True},
    )

    assert np.isnan(result.flux[0])
    assert result.meta["valid_mask"].tolist() == [False]
