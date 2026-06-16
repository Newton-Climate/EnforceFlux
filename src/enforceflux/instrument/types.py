"""Type aliases for instrument observable and operator kinds."""
from typing import Literal

ObservableType = Literal[
    "concentration_ppm",    # Path-averaged CH4 concentration (ppm)
    "flux_nmol_m2_s",       # Surface flux (nmol m⁻² s⁻¹)
    "column_ppb",           # Column-averaged CH4 enhancement (ppb)
    "emission_rate_kg_hr",  # Source emission rate (kg CH4 hr⁻¹)
]

OperatorType = Literal[
    "line_integral",          # OP: path-averaged concentration
    "ec_footprint",           # EC: turbulent flux footprint convolution
    "point_flux",             # CH: chamber accumulation at a point
    "column_aircraft",        # AIR: vertical column integral
    "column_satellite",       # MSAT: averaging-kernel-weighted column
    "multi_path_inversion",   # LP_ESN: multi-line + transport inversion → Q
    "plume_imaging",          # IM_LS: plume mass × wind → Q
    "lidar_path_integral",    # BP_GML: active LiDAR plume → Q
]

OperatingMode = Literal["good", "challenging", "bad"]
