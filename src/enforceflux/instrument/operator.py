"""
InstrumentOperator: applies the spatial forward operator H and heteroscedastic
noise model to a FLEXPART G-matrix, producing simulated OSSE observations.
"""
import math
from dataclasses import dataclass

import numpy as np

from enforceflux.instrument.models import Instrument


@dataclass(frozen=True)
class ObservationResult:
    """Output of ``InstrumentOperator.simulate_observations``."""

    instruments: tuple[Instrument, ...]
    H_g: np.ndarray         # (m, n) instrument-modified forward operator
    y_clean: np.ndarray     # (m,) noiseless simulated observations
    y_obs: np.ndarray       # (m,) noisy observations; np.nan where invalid
    valid_mask: np.ndarray  # (m,) bool: True where observation is usable
    R: np.ndarray           # (m, m) diagonal noise covariance (inf for invalid)


@dataclass(frozen=True)
class TimeSeriesObservationResult:
    """Output of ``InstrumentOperator.simulate_time_series``."""

    instruments: tuple[Instrument, ...]
    times_s: np.ndarray     # (t,) simulation timestamps in seconds
    H_g: np.ndarray         # (t, m, n) instrument-modified forward operator
    y_clean: np.ndarray     # (t, m) clean, cadence-aggregated observations
    y_obs: np.ndarray       # (t, m) noisy observations; np.nan where invalid
    valid_mask: np.ndarray  # (t, m) bool: True where observation is usable
    R: np.ndarray           # (t, m, m) diagonal noise covariance (inf for invalid)


# ─── Beam-path geometry helper ────────────────────────────────────────────────

def _line_segment_weights(
    rx: np.ndarray,
    ry: np.ndarray,
    x0: float, y0: float,
    x1: float, y1: float,
    bandwidth_m: float,
) -> np.ndarray:
    """Gaussian weights for receptors near a beam-path segment."""
    dx, dy = x1 - x0, y1 - y0
    seg_len2 = dx * dx + dy * dy
    if seg_len2 < 1e-12:
        dist2 = (rx - x0) ** 2 + (ry - y0) ** 2
    else:
        t = np.clip(((rx - x0) * dx + (ry - y0) * dy) / seg_len2, 0.0, 1.0)
        px = x0 + t * dx
        py = y0 + t * dy
        dist2 = (rx - px) ** 2 + (ry - py) ** 2
    return np.exp(-dist2 / (2.0 * bandwidth_m**2))


# ─── Instrument operator ──────────────────────────────────────────────────────

class InstrumentOperator:
    """
    Forward operator H: maps a FLEXPART G-matrix to instrument-space observations.

    Full model:  y = H(G @ x) + ε
    where G is the (m_receptors × n_sources) transport operator,
    H combines/weights receptors per instrument type, and ε is noise.

    Usage
    -----
    op = InstrumentOperator(instruments)
    result = op.simulate_observations(g, x_true)
    # result.y_obs  → synthetic observations (NaN where dropout / below DL)
    # result.R      → diagonal noise covariance
    # result.H_g    → modified operator for Bayesian inversion
    """

    def __init__(
        self,
        instruments: list[Instrument],
        rng: np.random.Generator | None = None,
    ) -> None:
        self.instruments = instruments
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------
    # Spatial operator
    # ------------------------------------------------------------------

    def apply_spatial_operator(
        self,
        g: np.ndarray,
        receptor_map: list[list[int]] | None = None,
        receptor_x: np.ndarray | None = None,
        receptor_y: np.ndarray | None = None,
        receptor_heights_m: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply the spatial forward operator H to the FLEXPART G-matrix.

        Parameters
        ----------
        g : (m_receptors, n_sources) array
        receptor_map : list of lists, optional
            receptor_map[i] = row indices in g belonging to instrument i.
            If None, one-to-one mapping (instrument i → row i).
        receptor_x, receptor_y : (m_receptors,) arrays, optional
            Domain coordinates of each receptor. Required for
            ``line_integral`` and synthetic ``ec_footprint`` fallback with
            multiple receptors.
        receptor_heights_m : (m_receptors,) array, optional
            Height of each receptor. Required for column operators.

        Returns H_g : (m_instruments, n_sources) array
        """
        m_inst = len(self.instruments)
        n_src = g.shape[1]

        if receptor_map is None:
            if g.shape[0] != m_inst:
                raise ValueError(
                    f"G rows ({g.shape[0]}) ≠ instruments ({m_inst}). "
                    "Provide receptor_map when multiple receptors per instrument."
                )
            receptor_map = [[i] for i in range(m_inst)]

        if len(receptor_map) != m_inst:
            raise ValueError(
                f"receptor_map length ({len(receptor_map)}) ≠ instruments ({m_inst})."
            )

        H_g = np.empty((m_inst, n_src), dtype=float)

        for i, inst in enumerate(self.instruments):
            rows = receptor_map[i]
            g_rows = g[rows]
            op = inst.operator_type

            if len(rows) == 1:
                H_g[i] = g_rows[0]
            elif op == "line_integral" and receptor_x is not None:
                H_g[i] = self._line_integral_row(g_rows, rows, inst, receptor_x, receptor_y)
            elif op == "ec_footprint" and receptor_x is not None:
                H_g[i] = self._synthetic_ec_footprint_row(g_rows, rows, inst, receptor_x, receptor_y)
            elif op == "column_satellite" and inst.averaging_kernel is not None:
                H_g[i] = self._column_kernel_row(g_rows, inst.averaging_kernel)
            elif op in ("column_aircraft", "column_satellite") and receptor_heights_m is not None:
                H_g[i] = self._column_uniform_row(g_rows, rows, receptor_heights_m)
            else:
                H_g[i] = g_rows.mean(axis=0)

        return H_g

    def _line_integral_row(
        self, g_rows: np.ndarray, row_indices: list[int],
        inst: Instrument, rx: np.ndarray, ry: np.ndarray,
    ) -> np.ndarray:
        bearing = math.radians(inst.path_bearing_deg)
        x1 = inst.x + inst.path_length_m * math.sin(bearing)
        y1 = inst.y + inst.path_length_m * math.cos(bearing)
        sub_x, sub_y = rx[row_indices], ry[row_indices]
        bandwidth = (
            float(np.mean(np.sqrt(np.diff(sub_x) ** 2 + np.diff(sub_y) ** 2)))
            if len(sub_x) > 1 else inst.path_length_m
        )
        w = _line_segment_weights(sub_x, sub_y, inst.x, inst.y, x1, y1, bandwidth)
        w_sum = w.sum()
        return g_rows.mean(axis=0) if w_sum < 1e-30 else (w[:, None] * g_rows).sum(axis=0) / w_sum

    def _synthetic_ec_footprint_row(
        self, g_rows: np.ndarray, row_indices: list[int],
        inst: Instrument, rx: np.ndarray, ry: np.ndarray,
    ) -> np.ndarray:
        """Synthetic Gaussian fallback for toy EC demos, not a physical EC model."""
        wind_rad = math.radians(inst.footprint_wind_dir_deg)
        cx = inst.x + inst.footprint_sigma_m * math.sin(wind_rad)
        cy = inst.y + inst.footprint_sigma_m * math.cos(wind_rad)
        sub_x, sub_y = rx[row_indices], ry[row_indices]
        dist2 = (sub_x - cx) ** 2 + (sub_y - cy) ** 2
        w = np.exp(-dist2 / (2.0 * inst.footprint_sigma_m**2))
        w_sum = w.sum()
        return g_rows.mean(axis=0) if w_sum < 1e-30 else (w[:, None] * g_rows).sum(axis=0) / w_sum

    def _column_kernel_row(self, g_rows: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        k = kernel[: len(g_rows)]
        total = k.sum()
        k = k / total if total > 0 else np.ones(len(g_rows)) / len(g_rows)
        return (k[:, None] * g_rows).sum(axis=0)

    def _column_uniform_row(
        self, g_rows: np.ndarray, row_indices: list[int], heights_m: np.ndarray,
    ) -> np.ndarray:
        h = heights_m[row_indices]
        dz = np.abs(np.gradient(h)) if len(h) > 1 else np.ones(1)
        dz_sum = dz.sum()
        return g_rows.mean(axis=0) if dz_sum < 1e-30 else (dz[:, None] * g_rows).sum(axis=0) / dz_sum

    # ------------------------------------------------------------------
    # Gridded-field sampling
    # ------------------------------------------------------------------

    def sample_fields(
        self,
        fields: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        *,
        centred: bool = False,
        n_samples: int | None = None,
    ) -> np.ndarray:
        """Sample a ``(nt, ny, nx)`` concentration field for every instrument.

        Instruments whose ``operator_type`` is ``line_integral`` get the true
        arc-length average along their beam; everything else is sampled at its
        own location. Returns ``(nt, m_instruments)``.

        This is the forward path a gridded model needs, and it is what makes
        ``path_length_m`` mean something: sampling one nearest grid cell for an
        open-path analyser reports a point measurement, which has a different
        variance, skewness and detection rate from the path average even though
        — path-averaging being linear — it has almost the same long-run mean.

        ``x`` and ``y`` are the field's axes in metres, and each instrument's
        ``x``/``y`` must be in those same coordinates.
        """
        from enforceflux.instrument.open_path import path_average_series

        fields = np.asarray(fields, dtype=float)
        if fields.ndim == 2:
            fields = fields[None, ...]
        if fields.ndim != 3:
            raise ValueError(f"fields must be (nt, ny, nx), got {fields.shape}")

        out = np.empty((fields.shape[0], len(self.instruments)), dtype=float)
        for i, inst in enumerate(self.instruments):
            length = inst.path_length_m if inst.operator_type == "line_integral" else 0.0
            out[:, i] = path_average_series(
                fields, x, y, inst.x, inst.y, length, inst.path_bearing_deg,
                centred=centred, n_samples=n_samples,
            )
        return out

    # ------------------------------------------------------------------
    # Noise model
    # ------------------------------------------------------------------

    def noise_covariance(self, y_clean: np.ndarray) -> np.ndarray:
        """
        Return diagonal R for given noiseless signal levels.

        Useful for Fisher-information analysis without a stochastic draw.
        """
        var = np.empty(len(self.instruments))
        for i, inst in enumerate(self.instruments):
            p = inst.params
            sigma_i = math.sqrt((p.sigma_scale * abs(float(y_clean[i]))) ** 2 + p.sigma_abs**2)
            var[i] = sigma_i**2
        return np.diag(var)

    # ------------------------------------------------------------------
    # Full simulation
    # ------------------------------------------------------------------

    def simulate_observations(
        self,
        g: np.ndarray,
        x_true: np.ndarray,
        receptor_map: list[list[int]] | None = None,
        receptor_x: np.ndarray | None = None,
        receptor_y: np.ndarray | None = None,
        receptor_heights_m: np.ndarray | None = None,
    ) -> ObservationResult:
        """
        Generate synthetic OSSE observations:  y = H(G @ x) + ε.

        Noise model per instrument i::

            σᵢ = sqrt( (sigma_scale · |ŷᵢ|)² + sigma_abs² )
            yᵢ = ŷᵢ · (1 + bias_scale) + bias_abs + N(0, σᵢ²)

        Observation becomes NaN if dropout is sampled or |yᵢ| < detection_limit.
        """
        H_g = self.apply_spatial_operator(
            g, receptor_map=receptor_map,
            receptor_x=receptor_x, receptor_y=receptor_y,
            receptor_heights_m=receptor_heights_m,
        )
        y_clean = H_g @ x_true

        m = len(self.instruments)
        y_obs = np.empty(m)
        valid = np.ones(m, dtype=bool)
        noise_var = np.zeros(m)

        for i, inst in enumerate(self.instruments):
            p = inst.params
            yc = float(y_clean[i])

            if not np.isfinite(yc):
                y_obs[i] = np.nan
                valid[i] = False
                noise_var[i] = np.inf
                continue

            if self.rng.random() < p.dropout_probability:
                y_obs[i] = np.nan
                valid[i] = False
                noise_var[i] = np.inf
                continue

            sigma_i = math.sqrt((p.sigma_scale * abs(yc)) ** 2 + p.sigma_abs**2)
            noise_var[i] = sigma_i**2
            y_obs[i] = yc * (1.0 + p.bias_scale) + p.bias_abs + self.rng.normal(0.0, sigma_i)

            if p.detection_limit > 0.0 and abs(y_obs[i]) < p.detection_limit:
                y_obs[i] = np.nan
                valid[i] = False
                noise_var[i] = np.inf

        return ObservationResult(
            instruments=tuple(self.instruments),
            H_g=H_g, y_clean=y_clean, y_obs=y_obs,
            valid_mask=valid, R=np.diag(noise_var),
        )

    def simulate_time_series(
        self,
        g_series: np.ndarray,
        x_true_series: np.ndarray,
        times_s: np.ndarray,
        receptor_map: list[list[int]] | None = None,
        receptor_x: np.ndarray | None = None,
        receptor_y: np.ndarray | None = None,
        receptor_heights_m: np.ndarray | None = None,
    ) -> TimeSeriesObservationResult:
        """
        Generate cadence-aware synthetic observations across time.

        Parameters
        ----------
        g_series : (t, m_receptors, n_sources) or (m_receptors, n_sources) array
            Forward operator by timestep. A 2-D array is broadcast across time.
        x_true_series : (t, n_sources) or (n_sources,) array
            True source state by timestep. A 1-D array is broadcast across time.
        times_s : (t,) array
            Monotonic simulation timestamps in seconds.

        Returns
        -------
        TimeSeriesObservationResult
            Per-time clean and noisy observations after cadence, bias, noise,
            dropout, and detection-limit handling.
        """
        times = np.asarray(times_s, dtype=float)
        if times.ndim != 1 or len(times) == 0:
            raise ValueError("times_s must be a non-empty 1-D array.")
        if np.any(np.diff(times) < 0.0):
            raise ValueError("times_s must be monotonically non-decreasing.")

        g_arr = np.asarray(g_series, dtype=float)
        if g_arr.ndim == 2:
            g_arr = np.broadcast_to(g_arr[None, :, :], (len(times),) + g_arr.shape)
        elif g_arr.ndim != 3:
            raise ValueError("g_series must be a 2-D or 3-D array.")
        if g_arr.shape[0] != len(times):
            raise ValueError("g_series and times_s must have the same number of timesteps.")

        x_arr = np.asarray(x_true_series, dtype=float)
        if x_arr.ndim == 1:
            x_arr = np.broadcast_to(x_arr[None, :], (len(times), len(x_arr)))
        elif x_arr.ndim != 2:
            raise ValueError("x_true_series must be a 1-D or 2-D array.")
        if x_arr.shape[0] != len(times):
            raise ValueError("x_true_series and times_s must have the same number of timesteps.")
        if g_arr.shape[2] != x_arr.shape[1]:
            raise ValueError("g_series source dimension must match x_true_series.")

        t_count = len(times)
        m_inst = len(self.instruments)
        n_src = g_arr.shape[2]

        H_g = np.empty((t_count, m_inst, n_src), dtype=float)
        y_native = np.full((t_count, m_inst), np.nan, dtype=float)
        y_clean = np.full((t_count, m_inst), np.nan, dtype=float)
        y_obs = np.full((t_count, m_inst), np.nan, dtype=float)
        valid = np.zeros((t_count, m_inst), dtype=bool)
        noise_var = np.full((t_count, m_inst), np.inf, dtype=float)

        for t_idx in range(t_count):
            H_g[t_idx] = self.apply_spatial_operator(
                g_arr[t_idx],
                receptor_map=receptor_map,
                receptor_x=receptor_x,
                receptor_y=receptor_y,
                receptor_heights_m=receptor_heights_m,
            )
            finite_rows = np.all(np.isfinite(H_g[t_idx]), axis=1)
            if np.any(finite_rows):
                y_native[t_idx, finite_rows] = H_g[t_idx, finite_rows] @ x_arr[t_idx]

        t0 = times[0]
        for i, inst in enumerate(self.instruments):
            p = inst.params
            cadence = float(p.cadence_s)
            if cadence <= 0.0:
                raise ValueError(f"Instrument {inst.id} has non-positive cadence_s={cadence}.")

            for t_idx, t_now in enumerate(times):
                elapsed = t_now - t0
                if elapsed < 0.0:
                    continue
                sample_index = round(elapsed / cadence)
                if not np.isclose(elapsed, sample_index * cadence, atol=1e-9, rtol=0.0):
                    continue

                window_start = t_now - cadence
                window_mask = (times >= window_start) & (times <= t_now)
                if not np.any(window_mask):
                    continue

                window_values = y_native[window_mask, i]
                finite_window = np.isfinite(window_values)
                if not np.any(finite_window):
                    continue

                # The Jacobian row used downstream is H_g at the sample step;
                # a non-finite row cannot back a usable observation.
                if not np.all(np.isfinite(H_g[t_idx, i])):
                    continue

                yc = float(np.mean(window_values[finite_window]))
                y_clean[t_idx, i] = yc

                sigma_i = math.sqrt((p.sigma_scale * abs(yc)) ** 2 + p.sigma_abs**2)
                noise_var[t_idx, i] = sigma_i**2

                y_sample = yc * (1.0 + p.bias_scale) + p.bias_abs + self.rng.normal(0.0, sigma_i)
                if self.rng.random() < p.dropout_probability:
                    noise_var[t_idx, i] = np.inf
                    continue

                if p.detection_limit > 0.0 and abs(y_sample) < p.detection_limit:
                    noise_var[t_idx, i] = np.inf
                    continue

                y_obs[t_idx, i] = y_sample
                valid[t_idx, i] = True

        R = np.zeros((t_count, m_inst, m_inst), dtype=float)
        for t_idx in range(t_count):
            R[t_idx] = np.diag(noise_var[t_idx])

        return TimeSeriesObservationResult(
            instruments=tuple(self.instruments),
            times_s=times.copy(),
            H_g=H_g,
            y_clean=y_clean,
            y_obs=y_obs,
            valid_mask=valid,
            R=R,
        )
