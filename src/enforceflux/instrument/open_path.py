"""Everything needed to simulate an open-path (OP) analyser.

Two halves, in one place:

**Beam geometry and the path integral** — :func:`beam_endpoints`,
:func:`beam_samples`, :func:`path_average`, :func:`path_average_series`.

**The instrument** — :func:`simulate_open_path` takes a time series of gridded
concentration fields and returns what the analyser actually records: the beam
average, then noise, dropouts and the detection limit from
``INSTRUMENT_DB['OP']``. :class:`OpenPathSeries` carries the result together
with the censoring diagnostics that an OP over an intermittent plume demands.

An open-path analyser reports the *path-averaged* concentration along its beam,

    c_OP = (1/L) * integral_0^L c(s) ds

which is what the retrieval returns after dividing the measured path-integrated
column by the known path length. This module evaluates that integral against a
concentration field on a regular Cartesian grid.

Why this is not just "average the cells the beam crosses":

* A beam at an arbitrary bearing does not follow grid lines. Sampling nearest
  cells weights the average by how many cell centres happen to fall near the
  beam, which is a function of the bearing, not of the physics.
* The average must be **arc-length weighted**. Uniformly spaced samples along
  the beam with bilinear interpolation give that directly; cell-hit counting
  does not.

Both are handled here by sampling the beam at uniform arc-length intervals and
interpolating the field bilinearly at each sample.

Note on scale: the field is already a grid-cell average, so this returns the
path average *of the resolved field*. A real OP additionally averages the
sub-grid fluctuations, and — importantly for any point-vs-path comparison — a
single grid cell is likewise a cell average, not a true point sample.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "OpenPathSeries",
    "beam_endpoints",
    "beam_samples",
    "open_path_instrument",
    "path_average",
    "path_average_series",
    "simulate_open_path",
]


def beam_endpoints(
    x0: float, y0: float, path_length_m: float, path_bearing_deg: float,
    *, centred: bool = False,
) -> tuple[float, float, float, float]:
    """Beam start/end coordinates for an instrument at ``(x0, y0)``.

    ``path_bearing_deg`` is a compass bearing: 0 = +y (north), 90 = +x (east),
    matching the convention used for receptors elsewhere in EnforceFlux.

    With ``centred=False`` (the default, and what :class:`Instrument` means)
    ``(x0, y0)`` is the analyser and the beam runs from it toward the
    retroreflector. With ``centred=True`` the beam is centred on ``(x0, y0)``,
    which is the convenient form for a transect straddling a plume axis.
    """
    bearing = math.radians(path_bearing_deg)
    dx, dy = path_length_m * math.sin(bearing), path_length_m * math.cos(bearing)
    if centred:
        return x0 - dx / 2, y0 - dy / 2, x0 + dx / 2, y0 + dy / 2
    return x0, y0, x0 + dx, y0 + dy


def beam_samples(
    x0: float, y0: float, x1: float, y1: float, n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``n_samples`` points at uniform arc length along the beam.

    Midpoints of equal sub-intervals rather than endpoints: this is the
    midpoint rule, which integrates exactly over each sub-interval for a
    linear field and avoids double-weighting the two ends.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    frac = (np.arange(n_samples) + 0.5) / n_samples
    return x0 + frac * (x1 - x0), y0 + frac * (y1 - y0)


def _bilinear(field: np.ndarray, x: np.ndarray, y: np.ndarray,
              xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinearly interpolate ``field[y, x]`` at scattered ``(xs, ys)``.

    Samples outside the grid are returned as NaN rather than silently clamped
    to the edge value — a beam running off the domain is a configuration error,
    and clamping would fabricate concentrations at the boundary.
    """
    nx, ny = x.size, y.size
    dx, dy = x[1] - x[0], y[1] - y[0]

    fx = (xs - x[0]) / dx
    fy = (ys - y[0]) / dy
    outside = (fx < 0) | (fx > nx - 1) | (fy < 0) | (fy > ny - 1)

    i0 = np.clip(np.floor(fx).astype(int), 0, nx - 2)
    j0 = np.clip(np.floor(fy).astype(int), 0, ny - 2)
    tx = np.clip(fx - i0, 0.0, 1.0)
    ty = np.clip(fy - j0, 0.0, 1.0)

    v = ((1 - tx) * (1 - ty) * field[..., j0, i0]
         + tx * (1 - ty) * field[..., j0, i0 + 1]
         + (1 - tx) * ty * field[..., j0 + 1, i0]
         + tx * ty * field[..., j0 + 1, i0 + 1])
    return np.where(outside, np.nan, v)


def _resolve_samples(n_samples: int | None, path_length_m: float,
                     x: np.ndarray, y: np.ndarray) -> int:
    """Default to ~4 samples per grid cell along the beam."""
    if n_samples is not None:
        return n_samples
    spacing = min(abs(x[1] - x[0]), abs(y[1] - y[0]))
    return max(2, int(math.ceil(4.0 * path_length_m / spacing)))


def path_average(
    field: np.ndarray, x: np.ndarray, y: np.ndarray,
    x0: float, y0: float, path_length_m: float, path_bearing_deg: float,
    *, centred: bool = False, n_samples: int | None = None,
) -> float:
    """Path-averaged concentration along one beam over a 2-D field.

    ``field`` is ``(ny, nx)`` on the regular axes ``y`` and ``x`` (metres).
    Returns the arc-length mean, i.e. what an ideal OP analyser reports.

    A zero-length path degenerates to a point sample, which is the correct
    limit and lets the same call site serve both instrument types.
    """
    if path_length_m <= 0:
        return float(_bilinear(field, x, y, np.array([x0]), np.array([y0]))[0])

    xa, ya, xb, yb = beam_endpoints(x0, y0, path_length_m, path_bearing_deg,
                                    centred=centred)
    n = _resolve_samples(n_samples, path_length_m, x, y)
    xs, ys = beam_samples(xa, ya, xb, yb, n)
    values = _bilinear(field, x, y, xs, ys)
    if np.isnan(values).any():
        raise ValueError(
            f"Beam from ({xa:.1f}, {ya:.1f}) to ({xb:.1f}, {yb:.1f}) leaves the "
            f"grid x[{x[0]:.1f}, {x[-1]:.1f}] y[{y[0]:.1f}, {y[-1]:.1f}]. "
            "Shorten the path, move the instrument, or widen the domain."
        )
    return float(values.mean())


def path_average_series(
    fields: np.ndarray, x: np.ndarray, y: np.ndarray,
    x0: float, y0: float, path_length_m: float, path_bearing_deg: float,
    *, centred: bool = False, n_samples: int | None = None,
) -> np.ndarray:
    """:func:`path_average` over a ``(nt, ny, nx)`` stack — one value per time.

    The beam geometry is fixed, so the interpolation weights are computed once
    and applied across the whole series.
    """
    if fields.ndim != 3:
        raise ValueError(f"fields must be (nt, ny, nx), got shape {fields.shape}")

    if path_length_m <= 0:
        return _bilinear(fields, x, y, np.array([x0]), np.array([y0]))[:, 0]

    xa, ya, xb, yb = beam_endpoints(x0, y0, path_length_m, path_bearing_deg,
                                    centred=centred)
    n = _resolve_samples(n_samples, path_length_m, x, y)
    xs, ys = beam_samples(xa, ya, xb, yb, n)
    values = _bilinear(fields, x, y, xs, ys)      # (nt, n_samples)
    if np.isnan(values).any():
        raise ValueError(
            f"Beam from ({xa:.1f}, {ya:.1f}) to ({xb:.1f}, {yb:.1f}) leaves the "
            f"grid x[{x[0]:.1f}, {x[-1]:.1f}] y[{y[0]:.1f}, {y[-1]:.1f}]."
        )
    return values.mean(axis=-1)


# ─── The instrument ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OpenPathSeries:
    """What an open-path analyser records over a run, plus censoring diagnostics.

    ``truth`` is the noiseless beam average; ``observed`` is what the instrument
    reports, with NaN wherever a sample dropped out or fell below the detection
    limit. Keep both: the gap between them is the measurement, not an artefact.
    """

    times_s: np.ndarray        # (nt,)
    truth: np.ndarray          # (nt,) noiseless path average
    observed: np.ndarray       # (nt,) reported value, NaN where invalid
    valid: np.ndarray          # (nt,) bool
    detection_limit: float
    path_length_m: float

    @property
    def detected_fraction(self) -> float:
        return float(self.valid.mean())

    @property
    def mean_of_detected(self) -> float:
        """Mean over detected samples only — biased HIGH for a skewed plume.

        Thresholding an intermittent signal keeps the plume-hit tail and
        discards the near-zero majority, so this over-estimates the true mean.
        It is reported because it is what a naive analysis of real data
        computes, not because it is the right estimator.
        """
        return float(np.nanmean(self.observed[self.valid])) if self.valid.any() else float("nan")

    @property
    def mean_with_nondetects_as_zero(self) -> float:
        """Mean treating non-detects as zero — the better of the two simple estimators."""
        return float(np.where(self.valid, np.nan_to_num(self.observed), 0.0).mean())

    @property
    def censoring_bias(self) -> float:
        """``mean_of_detected / true mean``. 1.0 means censoring cost nothing."""
        true_mean = float(self.truth.mean())
        return self.mean_of_detected / true_mean if true_mean > 0 else float("nan")


def open_path_instrument(
    x0: float, y0: float, path_length_m: float, path_bearing_deg: float,
    *, id: str = "op", z: float = 3.0, mode: str = "good",
):
    """An :class:`Instrument` configured as an open-path analyser.

    Thin helper so callers do not have to remember that ``tech_id='OP'`` is what
    selects the line-integral operator and its noise parameters.

    The beam position is ``x0``/``y0`` rather than ``x``/``y`` so that it can be
    passed through :func:`simulate_open_path` without colliding with that
    function's ``x``/``y`` grid axes.
    """
    from enforceflux.instrument.models import Instrument

    return Instrument(
        id=id, tech_id="OP", mode=mode, x=x0, y=y0, z=z,
        path_length_m=path_length_m, path_bearing_deg=path_bearing_deg,
    )


def simulate_open_path(
    fields: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    instrument=None,
    *,
    times_s: np.ndarray | None = None,
    centred: bool = False,
    n_samples: int | None = None,
    rng: np.random.Generator | None = None,
    **beam,
) -> OpenPathSeries:
    """Simulate an OP analyser against a ``(nt, ny, nx)`` concentration field.

    ``x`` and ``y`` are the field's grid axes in metres. Pass an ``instrument``,
    or the beam geometry as keywords (``x0``/``y0``/``path_length_m``/
    ``path_bearing_deg``) forwarded to :func:`open_path_instrument`. Fields must
    already be in the instrument's observable units — ppm for ``tech_id='OP'``.

    The beam average is computed first and the instrument model applied to it,
    which is the physically correct order: the analyser integrates along the
    path and *then* its electronics add noise and impose a detection limit. The
    reverse (noise per grid cell, then average) would wrongly let path-averaging
    beat down instrument noise.
    """
    from enforceflux.instrument.operator import InstrumentOperator

    if instrument is None:
        instrument = open_path_instrument(**beam)

    fields = np.asarray(fields, dtype=float)
    if fields.ndim == 2:
        fields = fields[None, ...]

    truth = path_average_series(
        fields, x, y, instrument.x, instrument.y,
        instrument.path_length_m, instrument.path_bearing_deg,
        centred=centred, n_samples=n_samples,
    )

    op = InstrumentOperator([instrument], rng=rng or np.random.default_rng())
    observed = np.empty_like(truth)
    valid = np.zeros(truth.size, dtype=bool)
    unit = np.array([1.0])
    for k, c in enumerate(truth):
        result = op.simulate_observations(np.array([[c]]), unit)
        observed[k], valid[k] = result.y_obs[0], result.valid_mask[0]

    return OpenPathSeries(
        times_s=(np.arange(truth.size, dtype=float) if times_s is None
                 else np.asarray(times_s, dtype=float)),
        truth=truth, observed=observed, valid=valid,
        detection_limit=instrument.params.detection_limit,
        path_length_m=instrument.path_length_m,
    )
