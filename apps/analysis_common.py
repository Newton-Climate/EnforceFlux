from pathlib import Path
from typing import Any

import numpy as np


def require_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: pip install pyyaml") from exc
    return yaml


def find_var(ds, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in ds.variables:
            return name
    return None


def jsonify(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): jsonify(v) for k, v in x.items()}
    if isinstance(x, list):
        return [jsonify(v) for v in x]
    return x


def instrument_group_masks(n_time: int, n_inst: int, valid_flat: np.ndarray) -> dict[str, np.ndarray]:
    groups: dict[str, np.ndarray] = {}
    for inst in range(n_inst):
        mask = np.zeros(n_time * n_inst, dtype=bool)
        mask[inst::n_inst] = True
        groups[f"instrument_{inst}"] = mask & valid_flat
    return groups


def resolve_output_json_path(cfg: dict[str, Any]) -> Path:
    out_json = Path(cfg.get("output", {}).get("summary_json", "outputs/analysis_summary.json"))
    out_json = out_json.expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    return out_json
