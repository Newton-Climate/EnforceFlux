"""
Instrument parameter database.

All values sourced from data/methane_monitoring_tech_comparison.xlsx
(sheets: Comparison and OSSE_Modes).
"""
from dataclasses import dataclass

from enforceflux.instrument.types import ObservableType, OperatingMode, OperatorType


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

    detection_limit: float     # minimum detectable signal (0 = no limit)
    dropout_probability: float
    cadence_s: float           # nominal reporting interval (seconds)
    source_ids: str = ""       # literature references


# ─── Instrument parameter database ───────────────────────────────────────────

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
