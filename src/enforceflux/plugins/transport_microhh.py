"""Registry-facing MicroHH transport operator (``ITransportOperator``).

Builds a forward Jacobian ``g`` (instrument × source) from plume-scale LES.
Because a passive scalar is *linear* in emission rate, one unit-emission run per
source gives a full Jacobian column by scaling — the same principle as FLEXPART
forward mode, but with LES-resolved turbulence.

Scope note (read before relying on this)
-----------------------------------------
Each column costs a full LES run (hours of compute), so this operator is only
practical for the 1–few-source near-field configs MicroHH is meant for — it is
**not** a substitute for FLEXPART backward mode on a gridded inversion. For most
workflows, use :class:`MicroHHSimulationModel` to generate synthetic truth and
keep FLEXPART for the operator.

Config keys
-----------
sim_config : str
    Path to the MicroHH case YAML (required). Its receptors define the
    instrument rows; its sources define the columns.
reduce : str
    How to collapse each receptor's time series to a scalar Jacobian entry:
    ``"mean"`` (default), ``"max"``, or ``"ec_covariance"``. The latter
    computes a physical EC flux operator from LES-resolved ``w'CH4'`` and
    requires sufficiently frequent MicroHH column output.
dry_run : bool
    Prepare inputs without executing. Returns a zero ``g`` with prepared paths
    in ``meta`` (useful for wiring/inspection before the binary exists).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from enforceflux.core.base import ForwardModelResult, ITransportOperator
from enforceflux.instrument import Instrument
from enforceflux.models.source import Source


class MicroHHTransportOperator(ITransportOperator):
    def build_forward_operator(
        self,
        sources: Iterable[Source],
        instruments: Iterable[Instrument],
        domain: Any,
        config: dict[str, Any],
    ) -> ForwardModelResult:
        from enforceflux.microhh.output import read_receptor_series
        from enforceflux.microhh.runner import MicroHHRunner
        from enforceflux.microhh.sim_config import load_microhh_config

        sim_config_ref = config.get("sim_config")
        if not sim_config_ref:
            raise ValueError(
                "MicroHH transport operator requires config['sim_config']: a path "
                "to a MicroHH case YAML defining the grid, forcing, sources, and "
                "receptors."
            )

        base_cfg = load_microhh_config(self._resolve_path(sim_config_ref))
        dry_run = bool(config.get("dry_run", False))
        reduce = str(config.get("reduce", "mean")).lower()
        _reducer = {"mean": np.mean, "max": np.max}.get(reduce)
        if _reducer is None and reduce != "ec_covariance":
            raise ValueError(
                f"Unknown reduce={reduce!r}; expected 'mean', 'max', or 'ec_covariance'."
            )

        n_inst = len(base_cfg.receptors)
        n_src = len(base_cfg.sources)
        g = np.zeros((n_inst, n_src))
        meta: dict[str, Any] = {"backend": "microhh", "reduce": reduce, "runs": []}
        source_run_configs = []
        source_rates = []

        # One unit-emission LES per source; scale the receptor response to a
        # Jacobian column [concentration / (kg s-1)].
        for j, src in enumerate(base_cfg.sources):
            case_name = f"{base_cfg.case_name}_src{j:02d}"
            single = dataclasses.replace(
                base_cfg,
                case_name=case_name,
                case_dir=base_cfg.case_dir / case_name,
                sources=[src],
            )
            runner = MicroHHRunner(single)
            run = runner.run(dry_run=dry_run)
            meta["runs"].append(
                {"source_id": src.id, "case_dir": str(run.case_dir), "executed": run.executed}
            )
            if dry_run or not run.executed:
                continue

            if reduce == "ec_covariance":
                source_run_configs.append(single)
                source_rates.append(src.emission_rate_kg_s * single.emission_scale)
                continue

            series = read_receptor_series(single)
            # Response per unit physical emission rate [kg/s] → Jacobian column.
            emitted_kg_s = src.emission_rate_kg_s * single.emission_scale
            col = np.array([_reducer(series.values[:, i]) for i in range(n_inst)])
            g[:, j] = col / emitted_kg_s

        if reduce == "ec_covariance" and not dry_run:
            from enforceflux.microhh.ec_operator import (
                build_ec_observation_operator_from_microhh_runs,
            )

            if len(source_run_configs) != n_src:
                raise RuntimeError("Not all source-specific MicroHH runs completed for EC covariance.")
            ec = build_ec_observation_operator_from_microhh_runs(
                source_run_configs,
                measurement_height_m=float(config.get("measurement_height_m", 20.0)),
                source_emission_rates_kg_s=np.asarray(source_rates),
                air_density_kg_m3=float(config.get("air_density_kg_m3", 1.2)),
                window_s=float(config.get("ec_window_s", 1800.0)),
                min_samples=int(config.get("ec_min_samples", 120)),
                n_error_blocks=int(config.get("ec_error_blocks", 6)),
                max_missing_fraction=float(config.get("ec_max_missing_fraction", 0.1)),
                min_ustar_m_s=(
                    None if config.get("ec_min_ustar_m_s") is None
                    else float(config["ec_min_ustar_m_s"])
                ),
            )
            if not np.any(ec.valid_mask):
                raise RuntimeError(
                    "No valid LES EC windows. Use faster MicroHH column sampling, a longer "
                    "runtime, or relax ec_min_samples/QC settings."
                )
            g = np.nanmean(np.where(ec.valid_mask[:, :, None], ec.g, np.nan), axis=0)
            meta["ec"] = {
                **ec.meta,
                "window_times_s": ec.times_s.tolist(),
                "valid_mask": ec.valid_mask.tolist(),
                "random_error": ec.random_error.tolist(),
                "n_samples": ec.n_samples.tolist(),
            }

        meta["units"] = (
            "nmol m-2 s-1 / (kg s-1)"
            if reduce == "ec_covariance"
            else "scalar mixing ratio / (kg s-1)"
        )
        meta["dry_run"] = dry_run
        return ForwardModelResult(g=g, meta=meta)

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        repo_root = Path(__file__).resolve().parents[3]
        return (repo_root / path).resolve()
