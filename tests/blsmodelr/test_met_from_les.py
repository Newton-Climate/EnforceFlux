"""Unit tests for LES -> BlsInterval diagnosis (no MicroHH run required)."""
from __future__ import annotations

import numpy as np
import pytest

from enforceflux.blsmodelr import BlsInterval
from enforceflux.blsmodelr.met_from_les import (
    KAPPA,
    G,
    VelocityField,
    intervals_from_les,
    intervals_from_microhh_output,
)


def _sinusoidal_case(nt=100, dt=1.0, period=50.0, U=5.0, A=2.0, W=1.0, theta_amp=1.0,
                    theta_bar=300.0, ny=2, nx=2, nz=3):
    """Build a 4-D velocity field whose window statistics are analytic."""
    t = np.arange(nt) * dt
    s = np.sin(2 * np.pi * t / period)  # (nt,)
    # broadcast to (nt, nz, ny, nx)
    ones = np.ones((nz, ny, nx))
    u = (U + A * s)[:, None, None, None] * ones[None]
    v = np.zeros_like(u)
    w = (W * s)[:, None, None, None] * ones[None]
    theta_ref_1d = theta_bar + theta_amp * s          # (nt,)
    surface_theta_1d = np.full(nt, theta_bar)         # (nt,)
    return VelocityField(u=u, v=v, w=w), theta_ref_1d, surface_theta_1d


def test_windowing_produces_two_intervals():
    vf, tref, tsurf = _sinusoidal_case(nt=100)
    z = np.array([0.0, 10.0, 20.0])
    out = intervals_from_les(
        velocity_field=vf, z=z,
        surface_theta=tsurf, theta_ref=tref,
        z_ref=10.0, z0=0.1, window_s=50.0, dt=1.0,
    )
    assert len(out) == 2
    assert [iv.id for iv in out] == ["t0000", "t0001"]
    assert all(isinstance(iv, BlsInterval) for iv in out)


def test_ustar_L_wind_analytic():
    U, A, W, theta_amp, theta_bar = 5.0, 2.0, 1.0, 1.0, 300.0
    vf, tref, tsurf = _sinusoidal_case(
        nt=100, dt=1.0, period=50.0,
        U=U, A=A, W=W, theta_amp=theta_amp, theta_bar=theta_bar,
    )
    out = intervals_from_les(
        velocity_field=vf, z=np.array([0.0, 10.0, 20.0]),
        surface_theta=tsurf, theta_ref=tref,
        z_ref=10.0, z0=0.1, window_s=50.0, dt=1.0,
    )
    iv = out[0]
    # ⟨u'w'⟩ = A*W/2, ⟨v'w'⟩ = 0  =>  u* = sqrt(A*W/2)
    expected_ustar = ((A * W / 2.0) ** 2) ** 0.25
    assert iv.u_star == pytest.approx(expected_ustar, rel=1e-6)
    # ⟨w'θ'⟩ = W*θ_amp/2  =>  L = -u*^3 * θ_bar / (κ g w'θ')
    wpthp = W * theta_amp / 2.0
    expected_L = -(expected_ustar ** 3) * theta_bar / (KAPPA * G * wpthp)
    assert iv.L == pytest.approx(expected_L, rel=1e-6)
    # Mean wind: u_bar = U, v_bar = 0 → wind from west = 270°.
    assert iv.wind_speed == pytest.approx(U, rel=1e-12)
    assert iv.wind_dir_deg == pytest.approx(270.0, abs=1e-9)
    # sd_u ≈ A/sqrt(2), sd_w ≈ W/sqrt(2), sd_v = 0.
    assert iv.sd_u == pytest.approx(A / np.sqrt(2.0), rel=1e-6)
    assert iv.sd_w == pytest.approx(W / np.sqrt(2.0), rel=1e-6)
    assert iv.sd_v == pytest.approx(0.0, abs=1e-12)
    # Passthroughs.
    assert iv.z_ref == 10.0
    assert iv.z0 == 0.1


def test_zero_heat_flux_saturates_L():
    vf, _, tsurf = _sinusoidal_case(nt=100, theta_amp=0.0)
    tref = np.full(100, 300.0)  # constant → w'θ' = 0
    out = intervals_from_les(
        velocity_field=vf, z=np.array([0.0, 10.0, 20.0]),
        surface_theta=tsurf, theta_ref=tref,
        z_ref=10.0, z0=0.1, window_s=50.0, dt=1.0,
    )
    assert abs(out[0].L) == pytest.approx(1.0e6)


def test_theta_1d_vs_3d_match():
    vf, tref1d, tsurf1d = _sinusoidal_case(nt=100)
    ny, nx = vf.u.shape[2], vf.u.shape[3]
    tref3d = np.broadcast_to(tref1d[:, None, None], (100, ny, nx)).copy()
    tsurf3d = np.broadcast_to(tsurf1d[:, None, None], (100, ny, nx)).copy()
    common = dict(velocity_field=vf, z=np.array([0.0, 10.0, 20.0]),
                  z_ref=10.0, z0=0.1, window_s=50.0, dt=1.0)
    a = intervals_from_les(surface_theta=tsurf1d, theta_ref=tref1d, **common)
    b = intervals_from_les(surface_theta=tsurf3d, theta_ref=tref3d, **common)
    for x, y in zip(a, b):
        assert x.u_star == pytest.approx(y.u_star, rel=1e-12)
        assert x.L == pytest.approx(y.L, rel=1e-12)
        assert x.wind_dir_deg == pytest.approx(y.wind_dir_deg, abs=1e-12)


def test_interpolation_to_z_ref():
    """Field at z=0 is zeros, at z=20 is 2x target → interp at z_ref=10 gives target."""
    nt, ny, nx = 100, 2, 2
    t = np.arange(nt)
    s = np.sin(2 * np.pi * t / 50.0)
    U, A, W = 4.0, 2.0, 1.0
    # target profile at z_ref=10:
    u_ref = (U + A * s)[:, None, None] * np.ones((ny, nx))
    w_ref = (W * s)[:, None, None] * np.ones((ny, nx))
    # z=0 layer is zeros, z=20 is 2x; z_ref=10 must interpolate to *_ref.
    u = np.stack([np.zeros_like(u_ref), u_ref, 2.0 * u_ref], axis=1)
    v = np.zeros_like(u)
    w = np.stack([np.zeros_like(w_ref), w_ref, 2.0 * w_ref], axis=1)
    vf = VelocityField(u=u, v=v, w=w)
    z = np.array([0.0, 10.0, 20.0])

    # Compare to a case whose z_ref sits on the grid at z=10 (index 1).
    out_interp = intervals_from_les(
        velocity_field=vf, z=z,
        surface_theta=np.full(nt, 300.0), theta_ref=np.full(nt, 300.0),
        z_ref=10.0, z0=0.1, window_s=50.0, dt=1.0,
    )

    # Now bracket z_ref between z=5 and z=15 (asymmetric) and confirm interp
    # actually mixes both layers rather than snapping to the nearest one.
    z2 = np.array([5.0, 15.0])
    u2 = np.stack([np.zeros_like(u_ref), 2.0 * u_ref], axis=1)
    w2 = np.stack([np.zeros_like(w_ref), 2.0 * w_ref], axis=1)
    vf2 = VelocityField(u=u2, v=np.zeros_like(u2), w=w2)
    out2 = intervals_from_les(
        velocity_field=vf2, z=z2,
        surface_theta=np.full(nt, 300.0), theta_ref=np.full(nt, 300.0),
        z_ref=10.0, z0=0.1, window_s=50.0, dt=1.0,
    )
    assert out2[0].u_star == pytest.approx(out_interp[0].u_star, rel=1e-12)
    assert out2[0].wind_speed == pytest.approx(out_interp[0].wind_speed, rel=1e-12)


def test_microhh_convenience_requires_receptor_or_grid_index(tmp_path):
    """The convenience needs a real MicroHHConfig; validate its arg-checking."""
    # A stub that satisfies just enough of the surface used before the read.
    class _StubCfg:
        receptors = ()
    with pytest.raises(ValueError, match="receptor_id or both ix and iy"):
        intervals_from_microhh_output(
            _StubCfg(), z_ref=10.0, z0=0.1, window_s=50.0,
        )
