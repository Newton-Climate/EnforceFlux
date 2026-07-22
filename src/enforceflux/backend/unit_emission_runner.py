"""Shared template for backends that build a forward Jacobian via one
unit-emission run per source (FLEXPART forward mode, AERMOD, ...).

Subclasses supply the model-specific steps — writing input files, invoking
the binary, parsing receptor output — while this class owns the parts that
are the same regardless of which transport model is doing the work: the
per-source loop, run-directory caching, lon/lat coordinate transform, and
the unit-emission-rate scaling that turns a raw receptor reading into a
Jacobian column.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from enforceflux.backend.paths import resolve_path
from enforceflux.instrument import Instrument
from enforceflux.models.config import DomainConfig
from enforceflux.models.source import Source


@dataclass(frozen=True)
class UnitRunResult:
    g: np.ndarray
    meta: dict[str, Any]


class UnitEmissionRunner(ABC):
    def __init__(self, domain: DomainConfig, config: dict[str, Any]) -> None:
        self.domain = domain
        self.config = config

    def run(
        self,
        sources: Iterable[Source],
        instruments: Iterable[Instrument],
    ) -> UnitRunResult:
        sources = list(sources)
        instruments = list(instruments)
        transformer = self._build_transformer()

        base_run_dir = self._resolve_path(self.config.get("base_run_dir", self.default_run_dir))
        base_run_dir.mkdir(parents=True, exist_ok=True)

        g = np.zeros((len(instruments), len(sources)))
        meta: dict[str, Any] = {"runs": []}

        for j, source in enumerate(sources):
            run_dir = base_run_dir / f"source_{source.id}"
            output_dir = run_dir / self.config.get("output_dir", "output")

            if self.config.get("dry_run", False):
                self._prepare_run(run_dir, output_dir, source, instruments, transformer)
                meta["runs"].append({"source": source.id, "run_dir": str(run_dir)})
                continue

            if self._should_run(output_dir):
                self._prepare_run(run_dir, output_dir, source, instruments, transformer)
                self._execute(run_dir)

            receptor_values = self._read_receptor_values(output_dir, instruments)
            unit_emission_rate = float(self.config.get("unit_emission_rate", 1.0))
            g[:, j] = receptor_values / unit_emission_rate
            meta["runs"].append(
                {"source": source.id, "run_dir": str(run_dir), "output_dir": str(output_dir)}
            )

        return UnitRunResult(g=g, meta=meta)

    def _should_run(self, output_dir: Path) -> bool:
        if not self.config.get("cache", True):
            return True
        return not any(output_dir.glob(self.output_glob))

    def _build_transformer(self) -> Any:
        if not self.domain.crs:
            raise ValueError(f"domain.crs must be set to use {self.model_name} transport")
        try:
            from pyproj import Transformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("pyproj is required for coordinate conversion") from exc
        return Transformer.from_crs(self.domain.crs, self.domain.crs_wgs84, always_xy=True)

    def _resolve_path(self, value: str | Path) -> Path:
        return resolve_path(value, base=Path.cwd())

    # ── Model-specific hooks ────────────────────────────────────────────────

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model name, used in error messages."""
        raise NotImplementedError

    @property
    @abstractmethod
    def output_glob(self) -> str:
        """Glob pattern used to detect cached output in an output directory."""
        raise NotImplementedError

    @property
    @abstractmethod
    def default_run_dir(self) -> str:
        """Default value for ``config["base_run_dir"]``."""
        raise NotImplementedError

    @abstractmethod
    def _prepare_run(
        self,
        run_dir: Path,
        output_dir: Path,
        source: Source,
        instruments: list[Instrument],
        transformer: Any,
    ) -> None:
        """Write whatever input files the backend needs for a unit-emission
        run of ``source``, observed by ``instruments``."""
        raise NotImplementedError

    @abstractmethod
    def _execute(self, run_dir: Path) -> None:
        """Invoke the model binary for the run prepared in ``run_dir``."""
        raise NotImplementedError

    @abstractmethod
    def _read_receptor_values(
        self, output_dir: Path, instruments: list[Instrument]
    ) -> np.ndarray:
        """Read the per-instrument concentration values from ``output_dir``,
        in instrument order."""
        raise NotImplementedError
