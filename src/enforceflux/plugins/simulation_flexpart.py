import dataclasses
from pathlib import Path
from typing import Any, Iterable

from enforceflux.core.base import ITransportSimulation, TransportSimulationResult
from enforceflux.models.source import Source


class FlexpartSimulationModel(ITransportSimulation):
    """Registry-facing FLEXPART forward simulation.

    Produces a gridded concentration NetCDF by running FLEXPART for the
    emissions defined in a :class:`~enforceflux.flexpart.sim_config.SimulationConfig`
    YAML (``config["sim_config"]``).  This is the *simulation* counterpart to
    :class:`~enforceflux.plugins.transport_flexpart.FlexpartTransportOperator`
    (which builds a Jacobian instead of a field).

    The YAML is authoritative for sources, meteorology, period, and output grid,
    so the ``sources``/``domain`` arguments are accepted for interface symmetry
    but not used here.

    Config keys
    -----------
    sim_config : str
        Path to the SimulationConfig YAML (required).
    dry_run : bool
        Write FLEXPART inputs without executing.  Returns ``output_path=None``.
    ldirect : int
        Override the integration direction (+1 forward, -1 backward).  Lets a
        single YAML drive either a forward concentration run or a backward
        footprint run.
    output_per_source : bool
        Override ``IOUTPUTFOREACHRELEASE`` (one output grid per release).
    output_path : str
        Override the destination concentration NetCDF path.
    run_dir : str
        Override the FLEXPART run directory.
    """

    def simulate(
        self,
        sources: Iterable[Source],
        domain: Any,
        config: dict[str, Any],
    ) -> TransportSimulationResult:
        from enforceflux.flexpart import FlexpartSimulation
        from enforceflux.flexpart.sim_config import load_simulation_config

        sim_config_ref = config.get("sim_config")
        if not sim_config_ref:
            raise ValueError(
                "FLEXPART simulation requires config['sim_config']: a path to a "
                "SimulationConfig YAML defining the executable, meteorology, "
                "sources, period, and output grid."
            )

        sim_config = load_simulation_config(self._resolve_path(sim_config_ref))

        overrides: dict[str, Any] = {}
        if "ldirect" in config:
            overrides["ldirect"] = int(config["ldirect"])
        if "output_per_source" in config:
            overrides["output_per_source"] = bool(config["output_per_source"])
        if "output_path" in config:
            overrides["output_path"] = self._resolve_path(config["output_path"])
        if "run_dir" in config:
            overrides["run_dir"] = self._resolve_path(config["run_dir"])
        if overrides:
            sim_config = dataclasses.replace(sim_config, **overrides)

        sim = FlexpartSimulation(sim_config)
        meta: dict[str, Any] = {
            "backend": "flexpart",
            "ldirect": sim_config.ldirect,
        }

        if bool(config.get("dry_run", False)):
            run_dir = sim.prepare()
            meta["run_dir"] = str(run_dir)
            meta["prepared"] = True
            return TransportSimulationResult(output_path=None, meta=meta)

        output_path = sim.run()
        meta["output_path"] = str(output_path)
        return TransportSimulationResult(output_path=output_path, meta=meta)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        # Mirror the operator plugin: resolve relative paths against the repo root.
        repo_root = Path(__file__).resolve().parents[3]
        return (repo_root / path).resolve()
