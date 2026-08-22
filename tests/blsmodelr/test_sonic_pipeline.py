"""Tests for SonicObservation, LES sampler, and the EC processor."""
from __future__ import annotations

import numpy as np
import pytest

from enforceflux.blsmodelr import BlsInterval
from enforceflux.blsmodelr.met_from_sonic import interval_from_sonic
from enforceflux.instrument.sonic import SonicObservation, sample_sonic_from_les


# ── SonicObservation ─────────────────────────────────────────────────────


def _analytic_sonic(instrument_id="S1", n=1000, dt=0.1, u_mean=3.0, v_mean=0.0,
                    theta_mean=300.0, u_prime_amp=0.5, w_prime_amp=0.3,
                    covar_uw=-0.15, covar_wt=0.05, z0=0.05, z=4.0) -> SonicObservation:
    rng = np.random.default_rng(0)
    # Correlated turbulence: seed w', then u' = a*w' + noise, θ' = b*w' + noise.
    wp = rng.normal(0.0, w_prime_amp, size=n)
    a = covar_uw / (w_prime_amp ** 2)
    up = a * wp + rng.normal(0.0, np.sqrt(max(u_prime_amp**2 - a*a*w_prime_amp**2, 1e-6)), size=n)
    b = covar_wt / (w_prime_amp ** 2)
    tp = b * wp + rng.normal(0.0, 0.05, size=n)
    return SonicObservation(
        instrument_id=instrument_id, interval_id="t0",
        x=0.0, y=0.0, z=z,
        times_s=np.arange(n) * dt,
        u=u_mean + up, v=v_mean + np.zeros(n), w=wp,
        theta=theta_mean + tp,
        z0=z0,
    )


def test_sonic_shape_validation():
    with pytest.raises(ValueError, match="times_s length"):
        SonicObservation(
            instrument_id="X", interval_id="t0", x=0.0, y=0.0, z=2.0,
            times_s=np.zeros(10),
            u=np.zeros(5), v=np.zeros(10), w=np.zeros(10), theta=np.zeros(10),
            z0=0.05,
        )


def test_low_rate_warning():
    n = 20
    obs_kwargs = dict(
        instrument_id="slow", interval_id="t0",
        x=0.0, y=0.0, z=2.0,
        u=np.zeros(n), v=np.zeros(n), w=np.zeros(n), theta=np.zeros(n) + 300,
        z0=0.05,
    )
    # 1 Hz → below 5 Hz threshold, must warn.
    with pytest.warns(UserWarning, match="under-resolved"):
        SonicObservation(times_s=np.arange(n, dtype=float), **obs_kwargs)
    # 10 Hz → no warning.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SonicObservation(times_s=np.arange(n) * 0.1, **obs_kwargs)


# ── EC processor ─────────────────────────────────────────────────────────


def test_interval_from_sonic_recovers_wind_and_ustar():
    obs = _analytic_sonic(u_mean=4.0, v_mean=0.0, covar_uw=-0.2)
    iv = interval_from_sonic(obs)
    assert iv.wind_speed == pytest.approx(4.0, abs=0.1)
    # For a westerly wind (u>0, v=0), meteorological "from" direction is 270°.
    assert iv.wind_dir_deg == pytest.approx(270.0, abs=1.0)
    # u* = |⟨u'w'⟩|^(1/2) with v'w'≈0. covar_uw=-0.2 → u* ≈ 0.447.
    assert iv.u_star == pytest.approx(0.447, abs=0.05)
    assert iv.z_ref == 4.0
    assert iv.z0 == 0.05


def test_interval_from_sonic_stable_vs_unstable_L():
    # Positive w'θ' (unstable) → negative L.
    iv_unstable = interval_from_sonic(_analytic_sonic(covar_wt=+0.05))
    iv_stable   = interval_from_sonic(_analytic_sonic(covar_wt=-0.05))
    assert iv_unstable.L < 0
    assert iv_stable.L > 0
    # And L saturates when heat flux is exactly 0 (constant theta).
    obs = _analytic_sonic(covar_wt=0.0)
    obs_const_theta = SonicObservation(
        instrument_id=obs.instrument_id, interval_id=obs.interval_id,
        x=obs.x, y=obs.y, z=obs.z,
        times_s=obs.times_s, u=obs.u, v=obs.v, w=obs.w,
        theta=np.full_like(obs.theta, 300.0),
        z0=obs.z0,
    )
    iv_neutral = interval_from_sonic(obs_const_theta)
    assert abs(iv_neutral.L) >= 9e5


def test_interval_from_sonic_respects_valid_mask():
    obs = _analytic_sonic()
    mask = np.ones_like(obs.u, dtype=bool)
    mask[:500] = False  # drop first half
    obs_masked = SonicObservation(
        instrument_id=obs.instrument_id, interval_id=obs.interval_id,
        x=obs.x, y=obs.y, z=obs.z,
        times_s=obs.times_s, u=obs.u, v=obs.v, w=obs.w, theta=obs.theta,
        z0=obs.z0, valid_mask=mask,
    )
    iv = interval_from_sonic(obs_masked)
    assert np.isfinite(iv.u_star) and iv.u_star > 0


def test_interval_from_sonic_refuses_tiny_windows():
    obs = SonicObservation(
        instrument_id="tiny", interval_id="t0",
        x=0.0, y=0.0, z=2.0,
        times_s=np.arange(4) * 0.1,
        u=np.zeros(4), v=np.zeros(4), w=np.zeros(4), theta=np.zeros(4) + 300,
        z0=0.05,
    )
    with pytest.raises(ValueError, match="valid samples"):
        interval_from_sonic(obs)


# ── LES sampler ──────────────────────────────────────────────────────────


class _Vel:
    def __init__(self, u, v, w):
        self.u, self.v, self.w = u, v, w


def _make_les(nt=200, nz=4, ny=8, nx=8):
    """Uniform u=3 m/s at all heights except z=4 m where u=5 m/s."""
    z = np.array([1.0, 2.0, 3.0, 5.0])  # nz=4
    u = np.zeros((nt, nz, ny, nx))
    u[:, 0] = 3.0; u[:, 1] = 3.0; u[:, 2] = 3.0; u[:, 3] = 5.0
    v = np.zeros_like(u)
    w = np.zeros_like(u)
    theta = np.full((nt, nz, ny, nx), 300.0)
    x = np.linspace(-100.0, 100.0, nx)
    y = np.linspace(-100.0, 100.0, ny)
    times = np.arange(nt) * 0.1
    return _Vel(u, v, w), theta, x, y, z, times


def test_sample_sonic_from_les_uses_nearest_xy_and_linear_z():
    vel, theta, xg, yg, zg, ts = _make_les()
    # Sensor at z=4 m: between z=3 (u=3) and z=5 (u=5) → linear interp gives u=4.
    obs = sample_sonic_from_les(
        velocity_field=vel, theta_field=theta,
        x_grid=xg, y_grid=yg, z_grid=zg, times_s=ts,
        sensor_x=12.3, sensor_y=-8.4, sensor_z=4.0,
        z0=0.05, instrument_id="S1",
    )
    assert obs.u.shape == (ts.size,)
    assert obs.u == pytest.approx(4.0)
    assert obs.theta == pytest.approx(300.0)
    assert obs.z == 4.0
    # Nearest x/y should map to sensor location within one grid step.
    assert abs(xg[obs.meta["ix"]] - 12.3) <= (xg[1] - xg[0])
    assert abs(yg[obs.meta["iy"]] - (-8.4)) <= (yg[1] - yg[0])


def test_sample_sonic_from_les_accepts_1d_theta():
    vel, theta, xg, yg, zg, ts = _make_les()
    theta_1d = np.full(ts.size, 305.0)
    obs = sample_sonic_from_les(
        velocity_field=vel, theta_field=theta_1d,
        x_grid=xg, y_grid=yg, z_grid=zg, times_s=ts,
        sensor_x=0.0, sensor_y=0.0, sensor_z=2.0,
        z0=0.05, instrument_id="S2",
    )
    assert obs.theta == pytest.approx(305.0)


# ── EC-stats backdoor ────────────────────────────────────────────────────


def test_bls_interval_from_ec_stats():
    iv = BlsInterval.from_ec_stats(
        id="ec0", u_star=0.4, L=-30.0, z0=0.05,
        wind_dir_deg=225.0, wind_speed=3.5, z_ref=4.0,
    )
    assert iv.id == "ec0"
    assert iv.u_star == 0.4
    assert iv.L == -30.0
