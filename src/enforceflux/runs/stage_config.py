"""Load a stage YAML into (stage_block, run_meta, inputs).

Every runnable EnforceFlux stage config follows the same wrapper shape::

    stage: dispersion | instrument | obs | flux | analysis | met
    run:
      name: <experiment-id>          # groups all stages of one experiment
      outputs_root: runs/             # optional; defaults to repo runs/
    inputs:                           # optional; empty for source stages
      dispersion: runs/<name>/dispersion/
    <stage>:
      # stage-specific block (transport/domain/sources for dispersion, etc.)

The wrapper lets every driver (a) know the run name without re-parsing its
own block, (b) resolve upstream artifacts through the RunDir manifest, and
(c) hand a resolved config snapshot to :meth:`RunDir.snapshot_config`
without introspection.

:func:`load_stage_config` reads the YAML, asserts the ``stage:`` matches
the expected stage the driver is written for, and returns the pieces the
driver needs. Path resolution rules:

* ``run.outputs_root`` — resolved relative to the YAML file's directory
  when relative, absolute path otherwise.
* ``inputs.<stage>`` — same rule; the driver passes these to
  :func:`enforceflux.runs.read_upstream`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StageConfig:
    """The three pieces every stage driver needs from its YAML."""

    stage: str                     # canonical stage name (dispersion, flux, ...)
    run_name: str                  # unique experiment id → runs/<name>/<stage>/
    outputs_root: Path | None      # None → RunDir's default
    inputs: dict[str, Path]        # {upstream stage → resolved RunDir path}
    block: dict[str, Any]          # the <stage>: sub-mapping (parsed opaquely)
    snapshot: dict[str, Any]       # full YAML tree, for RunDir.snapshot_config
    yaml_path: Path                # absolute path to the loaded YAML
    yaml_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "yaml_dir", self.yaml_path.parent)


def load_stage_config(
    path: str | Path, *, expected_stage: str
) -> StageConfig:
    """Parse a stage YAML and validate its ``stage:`` matches ``expected_stage``.

    The driver typically calls this at entry with its own stage name, so a
    dispersion YAML fed to the instrument driver fails loudly rather than
    silently running the wrong pipeline step.
    """
    import yaml

    yaml_path = Path(path).resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"stage config not found: {yaml_path}")
    blob = yaml.safe_load(yaml_path.read_text()) or {}
    if not isinstance(blob, dict):
        raise ValueError(f"stage config at {yaml_path} must parse to a mapping")

    stage = str(blob.get("stage", "")).strip()
    if not stage:
        raise ValueError(
            f"{yaml_path}: missing top-level `stage:` key. Add e.g. "
            f"`stage: {expected_stage}` at the top of the file."
        )
    if stage != expected_stage:
        raise ValueError(
            f"{yaml_path}: stage={stage!r} but this driver expects "
            f"stage={expected_stage!r}. Use the matching driver, or fix the YAML."
        )

    run_meta = blob.get("run") or {}
    if not isinstance(run_meta, dict):
        raise ValueError(f"{yaml_path}: `run:` must be a mapping")
    run_name = str(run_meta.get("name", "")).strip()
    if not run_name:
        raise ValueError(
            f"{yaml_path}: `run.name` is required (groups outputs under "
            f"runs/<name>/{expected_stage}/)."
        )

    outputs_root_raw = run_meta.get("outputs_root")
    if outputs_root_raw is None:
        outputs_root: Path | None = None
    else:
        p = Path(outputs_root_raw)
        outputs_root = p if p.is_absolute() else (yaml_path.parent / p).resolve()

    inputs_raw = blob.get("inputs") or {}
    if not isinstance(inputs_raw, dict):
        raise ValueError(f"{yaml_path}: `inputs:` must be a mapping")
    inputs: dict[str, Path] = {}
    for upstream, upstream_path in inputs_raw.items():
        if upstream_path is None:
            continue
        p = Path(str(upstream_path))
        inputs[str(upstream)] = p if p.is_absolute() else (yaml_path.parent / p).resolve()

    block = blob.get(expected_stage)
    if block is None:
        raise ValueError(
            f"{yaml_path}: missing `{expected_stage}:` block. The stage-specific "
            f"configuration lives under that key."
        )
    if not isinstance(block, dict):
        raise ValueError(f"{yaml_path}: `{expected_stage}:` must be a mapping")

    return StageConfig(
        stage=stage,
        run_name=run_name,
        outputs_root=outputs_root,
        inputs=inputs,
        block=block,
        snapshot=blob,
        yaml_path=yaml_path,
    )


__all__ = ["StageConfig", "load_stage_config"]
