"""Registry-facing MicroHH forward simulation (``ITransportSimulation``).

The MicroHH counterpart to :class:`~enforceflux.plugins.simulation_flexpart.FlexpartSimulationModel`.
Runs a plume-scale LES for the emissions in a MicroHH case YAML
(``config["sim_config"]``) and returns the path to the column output — the
near-field synthetic truth used to validate the cheaper FLEXPART/Gaussian
operators.

As with the FLEXPART simulation, the YAML is authoritative for sources, grid,
and forcing, so ``sources``/``domain`` are accepted for interface symmetry but
not used here.

Config keys
-----------
sim_config : str
    Path to the MicroHH case YAML (required).
dry_run : bool
    Write ``.ini``/``.prof`` without executing (the only path until the MicroHH
    binary is built). Returns ``output_path=None``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from enforceflux.core.base import ITransportSimulation, TransportSimulationResult
from enforceflux.models.source import Source


class MicroHHSimulationModel(ITransportSimulation):
    def simulate(
        self,
        sources: Iterable[Source],
        domain: Any,
        config: dict[str, Any],
    ) -> TransportSimulationResult:
        from enforceflux.microhh.runner import MicroHHRunner
        from enforceflux.microhh.sim_config import load_microhh_config

        sim_config_ref = config.get("sim_config")
        if not sim_config_ref:
            raise ValueError(
                "MicroHH simulation requires config['sim_config']: a path to a "
                "MicroHH case YAML defining the executable, grid, forcing, "
                "sources, and receptors."
            )

        cfg = load_microhh_config(self._resolve_path(sim_config_ref))
        runner = MicroHHRunner(cfg)

        dry_run = bool(config.get("dry_run", False))
        result = runner.run(dry_run=dry_run)

        meta = dict(result.meta)
        meta["case_dir"] = str(result.case_dir)
        meta["ini_path"] = str(result.ini_path)
        meta["input_nc_path"] = str(result.input_nc_path)

        return TransportSimulationResult(output_path=result.output_path, meta=meta)

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        repo_root = Path(__file__).resolve().parents[3]
        return (repo_root / path).resolve()
