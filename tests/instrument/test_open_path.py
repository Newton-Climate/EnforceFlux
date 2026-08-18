"""Tests for the open-path instrument (enforceflux.instrument.open_path)."""
import math

import numpy as np
import pytest

from enforceflux.instrument import (
    INSTRUMENT_DB,
    Instrument,
    InstrumentOperator,
    beam_endpoints,
    open_path_instrument,
    path_average,
    path_average_series,
    simulate_open_path,
)


@pytest.fixture
def grid():
    x = np.arange(0.0, 1000.0, 10.0)
    y = np.arange(0.0, 800.0, 10.0)
    return x, y


def test_uniform_field_returns_the_constant(grid):
    """A path average of a constant field is that constant, any bearing."""
    x, y = grid
    field = np.full((y.size, x.size), 2.5)
    for bearing in (0.0, 37.0, 90.0, 214.0):
        got = path_average(field, x, y, 500.0, 400.0, 200.0, bearing, centred=True)
        assert got == pytest.approx(2.5, rel=1e-12)


def test_linear_field_gives_the_midpoint_value(grid):
    """For a linear field the arc-length mean equals the value at the centre.

    This is the property that distinguishes a real line integral from
    averaging whichever cell centres happen to lie near the beam.
    """
    x, y = grid
    field = 3.0 + 0.01 * x[None, :] + 0.02 * y[:, None]
    x0, y0 = 500.0, 400.0
    expected = 3.0 + 0.01 * x0 + 0.02 * y0
    for bearing in (0.0, 30.0, 45.0, 90.0, 123.0):
        got = path_average(field, x, y, x0, y0, 300.0, bearing, centred=True)
        assert got == pytest.approx(expected, rel=1e-10)


def test_bearing_convention_is_compass(grid):
    """0 deg points +y (north) and 90 deg points +x (east)."""
    xa, ya, xb, yb = beam_endpoints(100.0, 200.0, 50.0, 0.0)
    assert (xb, yb) == pytest.approx((100.0, 250.0))
    xa, ya, xb, yb = beam_endpoints(100.0, 200.0, 50.0, 90.0)
    assert (xb, yb) == pytest.approx((150.0, 200.0))


def test_opposite_bearings_agree_when_centred(grid):
    """A centred beam is the same segment viewed from either end."""
    x, y = grid
    rng = np.random.default_rng(0)
    field = rng.random((y.size, x.size))
    a = path_average(field, x, y, 500.0, 400.0, 240.0, 55.0, centred=True)
    b = path_average(field, x, y, 500.0, 400.0, 240.0, 235.0, centred=True)
    assert a == pytest.approx(b, rel=1e-10)


def test_zero_length_path_is_a_point_sample(grid):
    """The degenerate limit must be the interpolated point value."""
    x, y = grid
    field = 1.0 + 0.003 * x[None, :] * y[:, None] ** 0.5
    got = path_average(field, x, y, 505.0, 405.0, 0.0, 0.0)
    expected = 1.0 + 0.003 * 505.0 * 405.0**0.5
    assert got == pytest.approx(expected, rel=1e-3)


def test_beam_leaving_the_grid_raises(grid):
    """Running off the domain must fail loudly, not clamp to the edge."""
    x, y = grid
    field = np.ones((y.size, x.size))
    with pytest.raises(ValueError, match="leaves the grid"):
        path_average(field, x, y, 50.0, 400.0, 400.0, 270.0)


def test_series_matches_per_slice_calls(grid):
    """The vectorised path must equal looping over slices."""
    x, y = grid
    rng = np.random.default_rng(1)
    fields = rng.random((7, y.size, x.size))
    series = path_average_series(fields, x, y, 500.0, 400.0, 180.0, 65.0, centred=True)
    per_slice = [path_average(f, x, y, 500.0, 400.0, 180.0, 65.0, centred=True)
                 for f in fields]
    assert series == pytest.approx(per_slice, rel=1e-12)


def test_path_average_commutes_with_time_average(grid):
    """Path-averaging is linear, so it commutes with averaging over time.

    This is the identity behind "the OP mean equals the mean of the fields".
    """
    x, y = grid
    rng = np.random.default_rng(2)
    fields = rng.random((30, y.size, x.size))
    args = (x, y, 500.0, 400.0, 220.0, 41.0)
    time_then_path = path_average(fields.mean(axis=0), *args, centred=True)
    path_then_time = path_average_series(fields, *args, centred=True).mean()
    assert time_then_path == pytest.approx(path_then_time, rel=1e-12)


def test_line_average_matches_analytic_gaussian(grid):
    """The beam average of a Gaussian cross-section has a closed form."""
    x, y = grid
    sigma, half = 80.0, 100.0
    field = np.exp(-(((y[:, None] - 400.0) / sigma) ** 2)) * np.ones_like(x[None, :])
    line = path_average(field, x, y, 500.0, 400.0, 2 * half, 0.0, centred=True)

    # (1/L) * int_{-h}^{h} exp(-(s/sigma)^2) ds = (sigma/2h) sqrt(pi) erf(h/sigma)
    expected = (sigma / (2 * half)) * math.sqrt(math.pi) * math.erf(half / sigma)
    assert line == pytest.approx(expected, rel=2e-3)


def test_area_average_equals_line_only_when_field_is_uniform_along_x(grid):
    """When and why "spatial averaging" is or is not the OP measurement.

    A beam is a 1-D line; an area average is a 2-D patch. They agree only when
    the field does not vary along the extra dimension the patch spans — then
    averaging it adds nothing. Give the plume any along-wind structure and the
    two operators separate.
    """
    x, y = grid
    cross = np.exp(-(((y[:, None] - 400.0) / 80.0) ** 2))
    ys = (y >= 300.0) & (y < 500.0)
    xs = (x >= 400.0) & (x < 600.0)

    # No x-dependence: the extra averaged dimension is degenerate, so they match.
    flat = cross * np.ones_like(x[None, :])
    line = path_average(flat, x, y, 500.0, 400.0, 200.0, 0.0, centred=True)
    assert line == pytest.approx(flat[np.ix_(ys, xs)].mean(), rel=1e-9)

    # Decaying downwind: the patch mixes in concentrations the beam never sees.
    decay = cross * np.exp(-x[None, :] / 500.0)
    line = path_average(decay, x, y, 500.0, 400.0, 200.0, 0.0, centred=True)
    assert not math.isclose(line, decay[np.ix_(ys, xs)].mean(), rel_tol=1e-3)


def test_operator_sample_fields_routes_by_operator_type(grid):
    """line_integral instruments path-average; others sample their location."""
    x, y = grid
    field = np.exp(-(((y[:, None] - 400.0) / 60.0) ** 2)) * np.ones_like(x[None, :])
    fields = field[None, ...]

    op_inst = Instrument(id="op", tech_id="OP", mode="good",
                         x=500.0, y=400.0, z=3.0,
                         path_length_m=300.0, path_bearing_deg=0.0)
    ch_inst = Instrument(id="ch", tech_id="CH", mode="good",
                         x=500.0, y=400.0, z=3.0,
                         path_length_m=300.0, path_bearing_deg=0.0)

    got = InstrumentOperator([op_inst, ch_inst]).sample_fields(
        fields, x, y, centred=True)

    # The chamber sits at the Gaussian peak; the beam averages down its flanks.
    assert got[0, 1] == pytest.approx(1.0, rel=1e-6)
    assert got[0, 0] < 0.95 * got[0, 1]


# ─── The instrument simulation ───────────────────────────────────────────────


def test_op_detection_limit_is_5_ppb():
    """The OP detection limit is 5 ppb, expressed in the ppm observable."""
    params = INSTRUMENT_DB["OP"]["good"]
    assert params.detection_limit == pytest.approx(0.005)
    assert params.observable == "concentration_ppm"


def test_simulate_open_path_recovers_truth_above_the_limit(grid):
    """Well above the detection limit, the reported mean tracks the truth."""
    x, y = grid
    fields = np.full((200, y.size, x.size), 1.0)      # 1 ppm, far above 5 ppb
    s = simulate_open_path(fields, x, y, times_s=np.arange(200) * 60.0,
                           x0=500.0, y0=400.0, path_length_m=200.0,
                           path_bearing_deg=0.0, centred=True,
                           rng=np.random.default_rng(0))
    assert s.truth == pytest.approx(np.ones(200))
    # Only dropouts (5%) should be lost, not the detection limit.
    assert 0.88 < s.detected_fraction < 1.0
    assert s.mean_of_detected == pytest.approx(1.0, abs=0.01)
    assert s.censoring_bias == pytest.approx(1.0, abs=0.01)


def test_sub_threshold_signal_yields_noise_driven_false_detections(grid):
    """The detection limit sits BELOW the noise floor, so it gates almost nothing.

    With sigma_abs = 10 ppb and a 5 ppb limit (0.5 sigma), pure noise clears the
    bar most of the time. The expected rate is computed from the configured
    parameters rather than hardcoded, so this keeps testing the real behaviour
    if either number is retuned.
    """
    x, y = grid
    params = INSTRUMENT_DB["OP"]["good"]
    assert params.detection_limit < 2 * params.sigma_abs   # the premise

    signal = 0.001                                          # 1 ppb, well under
    fields = np.full((4000, y.size, x.size), signal)
    s = simulate_open_path(fields, x, y, x0=500.0, y0=400.0, path_length_m=200.0,
                           path_bearing_deg=0.0, centred=True,
                           rng=np.random.default_rng(0))
    assert s.truth == pytest.approx(np.full(4000, signal))

    # |y| > DL for y ~ N(signal, sigma_abs), times the survival of dropout.
    def upper_tail(z):
        return 0.5 * math.erfc(z / math.sqrt(2.0))

    sigma, limit = params.sigma_abs, params.detection_limit
    expected = (1.0 - params.dropout_probability) * (
        upper_tail((limit - signal) / sigma) + upper_tail((limit + signal) / sigma)
    )
    assert s.detected_fraction == pytest.approx(expected, abs=0.03)
    assert expected > 0.5, "a sub-noise limit should pass most pure-noise samples"

    # Nothing that gets through is a measurement of the 1 ppb truth. The
    # inflation is milder than one might expect because the operator thresholds
    # |y|, so large NEGATIVE noise excursions are also "detected" and partly
    # cancel the positive ones in the mean.
    assert s.censoring_bias > 1.3
    assert np.nanmin(s.observed[s.valid]) < -limit    # negative detections exist


def test_signal_far_below_noise_is_effectively_never_detected(grid):
    """Push the signal far enough down and the threshold does its job."""
    x, y = grid
    fields = np.full((2000, y.size, x.size), 1e-6)         # 1 ppt
    s = simulate_open_path(fields, x, y, x0=500.0, y0=400.0, path_length_m=200.0,
                           path_bearing_deg=0.0, centred=True,
                           rng=np.random.default_rng(0))
    # Still ~10% two-sided noise excursions past |0.005|, but zero information:
    # the detected mean is set by the noise, and is not a measurement of 1 ppt.
    assert abs(s.mean_of_detected) > 100 * float(s.truth.mean())


def test_censoring_bias_is_high_for_an_intermittent_plume(grid):
    """Thresholding a skewed signal keeps the tail and overstates the mean.

    This is the trap in reporting `mean_of_detected` from real OP data, so the
    dataclass exposes the bias rather than leaving it to be rediscovered.
    """
    x, y = grid
    rng = np.random.default_rng(3)
    # Lognormal in time: mostly near zero, occasional large excursions.
    amp = rng.lognormal(mean=math.log(0.004), sigma=1.4, size=400)
    fields = amp[:, None, None] * np.ones((1, y.size, x.size))
    s = simulate_open_path(fields, x, y, x0=500.0, y0=400.0, path_length_m=200.0,
                           path_bearing_deg=0.0, centred=True,
                           rng=np.random.default_rng(4))

    assert 0.0 < s.detected_fraction < 1.0
    # Biased high, though the 10 ppb noise floor dilutes the effect: noise both
    # promotes weak samples and inflates the ones it lets through, so this is
    # an ordering property, not a fixed factor.
    assert s.censoring_bias > 1.2
    # Treating non-detects as zero is much closer to the truth.
    true_mean = s.truth.mean()
    assert abs(s.mean_with_nondetects_as_zero - true_mean) < abs(s.mean_of_detected - true_mean)


def test_open_path_instrument_helper_selects_the_op_operator():
    inst = open_path_instrument(x0=1.0, y0=2.0, path_length_m=300.0,
                                path_bearing_deg=45.0)
    assert inst.tech_id == "OP"
    assert inst.operator_type == "line_integral"
    assert inst.path_length_m == 300.0


def test_noise_is_applied_after_averaging_not_before(grid):
    """Path length must not beat down instrument noise.

    The analyser integrates along the beam and *then* its electronics add
    noise, so the reported scatter is sigma_abs regardless of path length.
    Averaging per-cell noise instead would shrink it as 1/sqrt(n).
    """
    x, y = grid
    fields = np.full((400, y.size, x.size), 1.0)
    scatter = []
    for length in (50.0, 600.0):
        s = simulate_open_path(fields, x, y, x0=500.0, y0=400.0,
                               path_length_m=length, path_bearing_deg=0.0,
                               centred=True, rng=np.random.default_rng(7))
        scatter.append(np.nanstd(s.observed[s.valid]))
    sigma_abs = INSTRUMENT_DB["OP"]["good"].sigma_abs
    for got in scatter:
        assert got == pytest.approx(sigma_abs, rel=0.2)
