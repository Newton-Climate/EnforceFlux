from pathlib import Path
from typing import Any, Iterable

import numpy as np

from enforceflux.core.base import ForwardModelResult, ITransportOperator
from enforceflux.flexpart.wrapper import FlexpartWrapper
from enforceflux.instrument import Instrument
from enforceflux.models.source import Source


class FlexpartTransportOperator(ITransportOperator):
    """Registry-facing FLEXPART transport operator.

    Exposes both FLEXPART Jacobian modes through the single
    :meth:`build_forward_operator` interface, selected by ``config["mode"]``:

    - ``"forward"`` (default): one unit-emission run per source via
      :class:`~enforceflux.flexpart.runner.FlexpartRunner` (through
      :class:`~enforceflux.flexpart.wrapper.FlexpartWrapper`).  Reads point
      receptors and assembles ``g`` column-by-column.
    - ``"backward"``: one footprint run per instrument via
      :class:`~enforceflux.flexpart.backward.FlexpartBackwardRunner`, assembling
      ``g`` row-by-row.  More efficient when ``n_instruments < n_sources``.

    Both modes return a :class:`ForwardModelResult` whose ``g`` is the
    instrument-by-source Jacobian expected by ``run_osse``.
    """

    def build_forward_operator(
        self,
        sources: Iterable[Source],
        instruments: Iterable[Instrument],
        domain: Any,
        config: dict[str, Any],
    ) -> ForwardModelResult:
        mode = str(config.get("mode", "forward")).lower()
        if mode == "forward":
            return self._build_forward(sources, instruments, domain, config)
        if mode == "backward":
            return self._build_backward(sources, instruments, domain, config)
        raise ValueError(
            f"Unknown FLEXPART transport mode {mode!r}. Expected 'forward' or 'backward'."
        )

    # ── Forward (per-source) ──────────────────────────────────────────────────

    def _build_forward(
        self,
        sources: Iterable[Source],
        instruments: Iterable[Instrument],
        domain: Any,
        config: dict[str, Any],
    ) -> ForwardModelResult:
        wrapper = FlexpartWrapper(domain=domain, config=config)
        result = wrapper.run(sources, instruments)
        meta = dict(result.meta)
        meta.setdefault("mode", "forward")
        return ForwardModelResult(g=result.g, meta=meta)

    # ── Backward (per-receptor footprint) ─────────────────────────────────────

    def _build_backward(
        self,
        sources: Iterable[Source],
        instruments: Iterable[Instrument],
        domain: Any,
        config: dict[str, Any],
    ) -> ForwardModelResult:
        from enforceflux.flexpart.backward import FlexpartBackwardRunner
        from enforceflux.flexpart.sim_config import load_simulation_config

        sim_config_ref = config.get("sim_config")
        if not sim_config_ref:
            raise ValueError(
                "FLEXPART backward mode requires config['sim_config']: a path to a "
                "SimulationConfig YAML providing the executable, meteorology, options "
                "template, simulation period, and output grid."
            )
        base_config = load_simulation_config(self._resolve_path(sim_config_ref))

        runner_keys = (
            "base_run_dir",
            "n_particles",
            "cache",
            "dry_run",
            "surface_only",
            "species_number",
        )
        runner_config = {k: config[k] for k in runner_keys if k in config}

        runner = FlexpartBackwardRunner(
            base_config=base_config, domain=domain, config=runner_config
        )
        result = runner.run(list(instruments), list(sources))

        g = result.g
        meta = dict(result.meta)
        meta["mode"] = "backward"
        meta["units"] = "s m3 kg-1 (raw footprint)"

        # Optional conversion to physical Jacobian units [ng m-3 / (kg s-1)],
        # matching the forward operator. Requires per-source horizontal areas.
        source_areas = config.get("source_areas_m2")
        if source_areas is not None:
            mixing_height_m = float(config.get("mixing_height_m", 100.0))
            g = FlexpartBackwardRunner.to_jacobian(
                np.asarray(g, dtype=float),
                np.asarray(source_areas, dtype=float),
                mixing_height_m=mixing_height_m,
            )
            meta["units"] = "ng m-3 / (kg s-1)"
            meta["mixing_height_m"] = mixing_height_m

        return ForwardModelResult(g=g, meta=meta)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        # Mirror FlexpartWrapper: resolve relative paths against the repo root.
        repo_root = Path(__file__).resolve().parents[3]
        return (repo_root / path).resolve()
