#!/usr/bin/env python3
"""Rebuild rice-paddy canonical fields from completed native LES outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from enforceflux.microhh.sim_config import load_microhh_config
from enforceflux.transport import canonical


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs/hetero_rice_paddy_test"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for config_path in sorted(CONFIG_DIR.glob("les_l*_cv*.yaml")):
        shared = yaml.safe_load(config_path.read_text())
        run_name = shared["run"]["name"]
        level = int(shared["dispersion"]["microhh"].get("level_index", 0))
        run_dir = ROOT / "runs" / run_name / "dispersion"
        generated = run_dir / "concentration_microhh/microhh_generated.yaml"
        output = run_dir / "concentration.nc"
        manifest_path = run_dir / "manifest.json"

        case = load_microhh_config(generated)
        field = canonical.from_microhh(case, level=level, meta={"mode": "simulation"})
        canonical.write_canonical(field, output, compress=True)

        manifest = json.loads(manifest_path.read_text())
        for item in manifest["outputs"]:
            if item.get("path") == "concentration.nc":
                item["sha256"] = sha256(output)
                break
        else:
            raise RuntimeError(f"concentration.nc missing from {manifest_path}")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            f"{run_name}: {field.values.shape[0]} frames, "
            f"{field.timestamps[0]}..{field.timestamps[-1]}, "
            f"level={level}, precision={case.precision}"
        )


if __name__ == "__main__":
    main()
