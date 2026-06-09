from __future__ import annotations

from typing import Any

from enforceflux.core.base import IInstrumentModel
from enforceflux.instrument import Instrument

# Maps legacy "kind" strings to INSTRUMENT_DB tech_ids
_KIND_TO_TECH_ID: dict[str, str] = {
    "open_path": "OP",
    "eddy_covariance": "EC",
    "flux_chamber": "CH",
    "aircraft": "AIR",
    "satellite": "MSAT",
    "lp_esn": "LP_ESN",
    "im_ls": "IM_LS",
    "bp_gml": "BP_GML",
}


def _require_keys(blob: dict, keys: list[str], context: str) -> None:
    missing = [k for k in keys if k not in blob]
    if missing:
        raise ValueError(f"Missing keys in {context}: {missing}")


class StaticInstrumentModel(IInstrumentModel):
    def build_instruments(self, config: dict[str, Any], domain: Any) -> list[Instrument]:
        instruments_blob = config.get("instruments", [])
        instruments: list[Instrument] = []
        for item in instruments_blob:
            _require_keys(item, ["id", "x", "y"], "instrument")

            # Prefer explicit tech_id; fall back to legacy "kind" mapping
            if "tech_id" in item:
                tech_id = str(item["tech_id"])
            elif "kind" in item:
                kind = str(item["kind"])
                tech_id = _KIND_TO_TECH_ID.get(kind, kind)
            else:
                raise ValueError(
                    f"Instrument {item.get('id')!r} must specify 'tech_id' "
                    f"(e.g. 'OP', 'EC', 'LP_ESN') or legacy 'kind'."
                )

            instruments.append(
                Instrument(
                    id=str(item["id"]),
                    tech_id=tech_id,
                    x=float(item["x"]),
                    y=float(item["y"]),
                    z=float(item.get("z", item.get("alt", 0.0))),
                    mode=str(item.get("mode", "good")),
                    path_length_m=float(item.get("path_length_m", 200.0)),
                    path_bearing_deg=float(item.get("path_bearing_deg", 0.0)),
                    footprint_sigma_m=float(item.get("footprint_sigma_m", 100.0)),
                    footprint_wind_dir_deg=float(item.get("footprint_wind_dir_deg", 270.0)),
                )
            )
        return instruments
