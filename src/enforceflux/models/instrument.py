"""
Instrument operator module for OSSE methane flux monitoring.

Applies instrument-specific forward operators (H) and noise models to
FLEXPART transport output (G-matrix) to produce simulated observations.

Parameters sourced from data/methane_monitoring_tech_comparison.xlsx.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

ObservableType = Literal[
    "concentration_ppm",
    "flux_nmol_m2_s",
    "column_ppb",
    "emission_rate_kg_hr",
]

OperatorType = Literal[
    "line_integral",
    "ec_footprint",
    "point_flux",
    "column_aircraft",
    "column_satellite",
    "multi_path_inversion",
    "plume_imaging",
    "lidar_path_integral",
]

OperatingMode = Literal["good", "challenging", "bad"]


@dataclass(frozen=True)
class OperatorParams:
    """Noise and detection parameters for one instrument type in one operating mode."""

    tech_id: str
    mode: OperatingMode
    operator_type: OperatorType
    observable: ObservableType

    # Noise model: sigma_i = sqrt((sigma_scale * |y|)^2 + sigma_abs^2)
    sigma_scale: float  # relative (fraction of signal); 0 for purely additive
    sigma_abs: float    # absolute additive noise (observable units); 0 for purely relative

    # Bias model: y_obs = y_clean * (1 + bias_scale) + bias_abs + noise
    bias_scale: float
    bias_abs: float

    detection_limit: float    # minimum detectable signal (0 = no limit)
    dropout_probability: float
    cadence_s: float          # nominal reporting interval (seconds)
    source_ids: str = ""      # literature references


# ─── Instrument parameter database (from methane_monitoring_tech_comparison.xlsx) ───

INSTRUMENT_DB: dict[str, dict[str, OperatorParams]] = {
    "OP": {
        "good": OperatorParams(
            tech_id="OP", mode="good",
            operator_type="line_integral", observable="concentration_ppm",
            sigma_scale=0.0, sigma_abs=0.003,
            bias_scale=0.0, bias_abs=0.0,
            detection_limit=0.04, dropout_probability=0.05,
            cadence_s=60.0, source_ids="Ashik2024",
        ),
    },
    "EC": {
        "good": OperatorParams(
            tech_id="EC", mode="good",
            operator_type="ec_footprint", observable="flux_nmol_m2_s",
            sigma_scale=0.0, sigma_abs=2.0,
            bias_scale=0.0, bias_abs=0.0,
            detection_limit=2.0, dropout_probability=0.1,
            cadence_s=1800.0, source_ids="Wang2013",
        ),
    },
    "CH": {
        "good": OperatorParams(
            tech_id="CH", mode="good",
            operator_type="point_flux", observable="flux_nmol_m2_s",
            sigma_scale=0.2, sigma_abs=0.0,
            bias_scale=0.2, bias_abs=0.0,
            detection_limit=0.0, dropout_probability=0.1,
            cadence_s=300.0, source_ids="Levy2011",
        ),
    },
    "AIR": {
        "good": OperatorParams(
            tech_id="AIR", mode="good",
            operator_type="column_aircraft", observable="column_ppb",
            sigma_scale=0.0, sigma_abs=5.0,
            bias_scale=0.0, bias_abs=0.0,
            detection_limit=2.0, dropout_probability=0.1,
            cadence_s=60.0, source_ids="Cusworth2019",
        ),
    },
    "MSAT": {
        "good": OperatorParams(
            tech_id="MSAT", mode="good",
            operator_type="column_satellite", observable="column_ppb",
            sigma_scale=0.0, sigma_abs=1.5,
            bias_scale=0.0, bias_abs=0.0,
            detection_limit=2.0, dropout_probability=0.2,
            cadence_s=86400.0, source_ids="MethaneSAT2023",
        ),
    },
    "LP_ESN": {
        "good": OperatorParams(
            tech_id="LP_ESN", mode="good",
            operator_type="multi_path_inversion", observable="emission_rate_kg_hr",
            sigma_scale=0.27, sigma_abs=0.0,
            bias_scale=0.00, bias_abs=0.0,
            detection_limit=0.06, dropout_probability=0.05,
            cadence_s=900.0, source_ids="S1,S3,S4,A0",
        ),
        "challenging": OperatorParams(
            tech_id="LP_ESN", mode="challenging",
            operator_type="multi_path_inversion", observable="emission_rate_kg_hr",
            sigma_scale=0.40, sigma_abs=0.0,
            bias_scale=0.05, bias_abs=0.0,
            detection_limit=0.06, dropout_probability=0.2,
            cadence_s=900.0, source_ids="S1,S4,A0",
        ),
        "bad": OperatorParams(
            tech_id="LP_ESN", mode="bad",
            operator_type="multi_path_inversion", observable="emission_rate_kg_hr",
            sigma_scale=0.60, sigma_abs=0.0,
            bias_scale=0.10, bias_abs=0.0,
            detection_limit=7.0, dropout_probability=0.8,
            cadence_s=900.0, source_ids="S1,S3,A0",
        ),
    },
    "IM_LS": {
        "good": OperatorParams(
            tech_id="IM_LS", mode="good",
            operator_type="plume_imaging", observable="emission_rate_kg_hr",
            sigma_scale=0.40, sigma_abs=0.0,
            bias_scale=0.15, bias_abs=0.0,
            detection_limit=9.5, dropout_probability=0.1,
            cadence_s=60.0, source_ids="S5,S7,S9,A0",
        ),
        "challenging": OperatorParams(
            tech_id="IM_LS", mode="challenging",
            operator_type="plume_imaging", observable="emission_rate_kg_hr",
            sigma_scale=0.50, sigma_abs=0.0,
            bias_scale=0.25, bias_abs=0.0,
            detection_limit=15.0, dropout_probability=0.4,
            cadence_s=60.0, source_ids="S5,S7,A0",
        ),
        "bad": OperatorParams(
            tech_id="IM_LS", mode="bad",
            operator_type="plume_imaging", observable="emission_rate_kg_hr",
            sigma_scale=0.70, sigma_abs=0.0,
            bias_scale=0.45, bias_abs=0.0,
            detection_limit=15.0, dropout_probability=0.85,
            cadence_s=60.0, source_ids="S5,S7,A0",
        ),
    },
    "BP_GML": {
        "good": OperatorParams(
            tech_id="BP_GML", mode="good",
            operator_type="lidar_path_integral", observable="emission_rate_kg_hr",
            sigma_scale=0.31, sigma_abs=0.0,
            bias_scale=0.08, bias_abs=0.0,
            detection_limit=0.9, dropout_probability=0.05,
            cadence_s=10.0, source_ids="S11,S13,S14,A0",
        ),
        "challenging": OperatorParams(
            tech_id="BP_GML", mode="challenging",
            operator_type="lidar_path_integral", observable="emission_rate_kg_hr",
            sigma_scale=0.40, sigma_abs=0.0,
            bias_scale=0.10, bias_abs=0.0,
            detection_limit=2.3, dropout_probability=0.15,
            cadence_s=10.0, source_ids="S11,S15,A0",
        ),
        "bad": OperatorParams(
            tech_id="BP_GML", mode="bad",
            operator_type="lidar_path_integral", observable="emission_rate_kg_hr",
            sigma_scale=0.60, sigma_abs=0.0,
            bias_scale=0.20, bias_abs=0.0,
            detection_limit=3.0, dropout_probability=0.8,
            cadence_s=10.0, source_ids="S11,S13,A0",
        ),
    },
}


@dataclass
class Instrument:
    """
    A single instrument deployment.

    ``tech_id`` identifies the instrument type in ``INSTRUMENT_DB``; ``mode``
    selects the operating-condition row (good / challenging / bad).
    """

    id: str
    tech_id: str   # key into INSTRUMENT_DB, e.g. "LP_ESN", "OP", "EC"
    x: float       # domain x-coordinate (m)
    y: float       # domain y-coordinate (m)
    z: float = 0.0
    mode: OperatingMode = "good"

    # Line-integral geometry for OP / LP_ESN path receptors
    path_length_m: float = 200.0
    path_bearing_deg: float = 0.0      # degrees clockwise from north

    # EC turbulent-footprint geometry
    footprint_sigma_m: float = 100.0
    footprint_wind_dir_deg: float = 270.0   # degrees clockwise from north

    # Column instruments: per-level averaging kernel (None → uniform weighting)
    averaging_kernel: np.ndarray | None = field(default=None, compare=False, repr=False)

    @property
    def params(self) -> OperatorParams:
        try:
            return INSTRUMENT_DB[self.tech_id][self.mode]
        except KeyError:
            raise ValueError(
                f"Unknown tech_id={self.tech_id!r} or mode={self.mode!r}. "
                f"Known tech_ids: {sorted(INSTRUMENT_DB)}"
            ) from None

    @property
    def operator_type(self) -> OperatorType:
        return self.params.operator_type

    @property
    def observable(self) -> ObservableType:
        return self.params.observable

    @property
    def effective_noise_std(self) -> float:
        """Scalar noise estimate for backward compatibility. Prefer ObservationResult.R."""
        p = self.params
        return math.sqrt(p.sigma_abs**2 + p.sigma_scale**2)


@dataclass(frozen=True)
class ObservationResult:
    """Output of ``InstrumentOperator.simulate_observations``."""

    instruments: tuple[Instrument, ...]
    H_g: np.ndarray         # (m, n) instrument-modified forward operator
    y_clean: np.ndarray     # (m,) noiseless simulated observations
    y_obs: np.ndarray       # (m,) noisy observations; np.nan where invalid
    valid_mask: np.ndarray  # (m,) bool: True where observation is usable
    R: np.ndarray           # (m, m) diagonal noise covariance (inf on diag for invalid)


# ─── Beam-path helper ────────────────────────────────────────────────────────

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


# ─── Instrument operator ─────────────────────────────────────────────────────

class InstrumentOperator:
    """
    Forward operator H: maps a FLEXPART G-matrix to instrument-space observations.

    Full model:  y = H(G @ x) + ε
    where G is the (m_receptors × n_sources) transport operator from
    ``FlexpartRunner``, H combines/weights receptors per instrument type,
    and ε is instrument-specific noise.

    Usage
    -----
    op = InstrumentOperator(instruments)
    result = op.simulate_observations(g, x_true)
    # result.y_obs   → synthetic observations (NaN where dropout / below DL)
    # result.R       → diagonal noise covariance
    # result.H_g     → modified operator for Bayesian inversion
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
            G-matrix from ``FlexpartRunner``.
        receptor_map : list of lists, optional
            ``receptor_map[i]`` = row indices in *g* belonging to instrument *i*.
            If ``None``, assumes one-to-one mapping (instrument i → row i).
        receptor_x, receptor_y : (m_receptors,) arrays, optional
            Domain coordinates of each receptor row. Required for
            ``line_integral`` and ``ec_footprint`` operators when multiple
            receptors are assigned to one instrument.
        receptor_heights_m : (m_receptors,) array, optional
            Height of each receptor. Required for column operators with
            multiple height levels assigned to one instrument.

        Returns
        -------
        H_g : (m_instruments, n_sources) array
        """
        m_inst = len(self.instruments)
        n_src = g.shape[1]

        if receptor_map is None:
            if g.shape[0] != m_inst:
                raise ValueError(
                    f"G rows ({g.shape[0]}) ≠ instruments ({m_inst}). "
                    "Provide receptor_map when multiple receptors belong to one instrument."
                )
            receptor_map = [[i] for i in range(m_inst)]

        if len(receptor_map) != m_inst:
            raise ValueError(
                f"receptor_map length ({len(receptor_map)}) ≠ instruments ({m_inst})."
            )

        H_g = np.empty((m_inst, n_src), dtype=float)

        for i, inst in enumerate(self.instruments):
            rows = receptor_map[i]
            g_rows = g[rows]  # (k, n)
            op = inst.operator_type

            if len(rows) == 1:
                H_g[i] = g_rows[0]
            elif op == "line_integral" and receptor_x is not None:
                H_g[i] = self._line_integral_row(
                    g_rows, rows, inst, receptor_x, receptor_y
                )
            elif op == "ec_footprint" and receptor_x is not None:
                H_g[i] = self._ec_footprint_row(
                    g_rows, rows, inst, receptor_x, receptor_y
                )
            elif op == "column_satellite" and inst.averaging_kernel is not None:
                H_g[i] = self._column_kernel_row(g_rows, inst.averaging_kernel)
            elif op in ("column_aircraft", "column_satellite") and receptor_heights_m is not None:
                H_g[i] = self._column_uniform_row(g_rows, rows, receptor_heights_m)
            else:
                H_g[i] = g_rows.mean(axis=0)

        return H_g

    def _line_integral_row(
        self,
        g_rows: np.ndarray,
        row_indices: list[int],
        inst: Instrument,
        rx: np.ndarray,
        ry: np.ndarray,
    ) -> np.ndarray:
        bearing = math.radians(inst.path_bearing_deg)
        x1 = inst.x + inst.path_length_m * math.sin(bearing)
        y1 = inst.y + inst.path_length_m * math.cos(bearing)

        sub_x = rx[row_indices]
        sub_y = ry[row_indices]
        if len(sub_x) > 1:
            bandwidth = float(np.mean(np.sqrt(np.diff(sub_x) ** 2 + np.diff(sub_y) ** 2)))
        else:
            bandwidth = inst.path_length_m

        w = _line_segment_weights(sub_x, sub_y, inst.x, inst.y, x1, y1, bandwidth)
        w_sum = w.sum()
        if w_sum < 1e-30:
            return g_rows.mean(axis=0)
        return (w[:, None] * g_rows).sum(axis=0) / w_sum

    def _ec_footprint_row(
        self,
        g_rows: np.ndarray,
        row_indices: list[int],
        inst: Instrument,
        rx: np.ndarray,
        ry: np.ndarray,
    ) -> np.ndarray:
        wind_rad = math.radians(inst.footprint_wind_dir_deg)
        cx = inst.x + inst.footprint_sigma_m * math.sin(wind_rad)
        cy = inst.y + inst.footprint_sigma_m * math.cos(wind_rad)

        sub_x = rx[row_indices]
        sub_y = ry[row_indices]
        dist2 = (sub_x - cx) ** 2 + (sub_y - cy) ** 2
        w = np.exp(-dist2 / (2.0 * inst.footprint_sigma_m**2))
        w_sum = w.sum()
        if w_sum < 1e-30:
            return g_rows.mean(axis=0)
        return (w[:, None] * g_rows).sum(axis=0) / w_sum

    def _column_kernel_row(
        self,
        g_rows: np.ndarray,
        kernel: np.ndarray,
    ) -> np.ndarray:
        k = kernel[: len(g_rows)]
        total = k.sum()
        k = k / total if total > 0 else np.ones(len(g_rows)) / len(g_rows)
        return (k[:, None] * g_rows).sum(axis=0)

    def _column_uniform_row(
        self,
        g_rows: np.ndarray,
        row_indices: list[int],
        heights_m: np.ndarray,
    ) -> np.ndarray:
        h = heights_m[row_indices]
        dz = np.abs(np.gradient(h)) if len(h) > 1 else np.ones(1)
        dz_sum = dz.sum()
        if dz_sum < 1e-30:
            return g_rows.mean(axis=0)
        return (dz[:, None] * g_rows).sum(axis=0) / dz_sum

    # ------------------------------------------------------------------
    # Noise model
    # ------------------------------------------------------------------

    def noise_covariance(self, y_clean: np.ndarray) -> np.ndarray:
        """
        Return diagonal R for given noiseless signal levels.

        Useful for Fisher-information analysis without a stochastic draw.
        """
        m = len(self.instruments)
        var = np.empty(m)
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

        Noise model per instrument *i*::

            σᵢ = sqrt( (sigma_scale · |ŷᵢ|)² + sigma_abs² )
            yᵢ = ŷᵢ · (1 + bias_scale) + bias_abs + N(0, σᵢ²)

        The observation is set to ``NaN`` (invalid) if dropout is sampled or
        ``|yᵢ| < detection_limit``.

        Parameters
        ----------
        g       : (m, n) FLEXPART G-matrix from ``FlexpartRunner``
        x_true  : (n,) true source emission rates
        receptor_map, receptor_x/y, receptor_heights_m :
            Optional geometry for spatial operators; see ``apply_spatial_operator``.

        Returns
        -------
        ObservationResult
        """
        H_g = self.apply_spatial_operator(
            g,
            receptor_map=receptor_map,
            receptor_x=receptor_x,
            receptor_y=receptor_y,
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
            H_g=H_g,
            y_clean=y_clean,
            y_obs=y_obs,
            valid_mask=valid,
            R=np.diag(noise_var),
        )
