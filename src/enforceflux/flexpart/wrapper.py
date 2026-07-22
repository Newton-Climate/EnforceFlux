import subprocess
from pathlib import Path
from typing import Any, Iterable

from enforceflux.models.config import DomainConfig
from enforceflux.backend import UnitRunResult, resolve_path
from enforceflux.flexpart.build import FlexpartBuildResult, FlexpartCompiler
from enforceflux.flexpart.runner import FlexpartRunner
from enforceflux.instrument import Instrument
from enforceflux.models.source import Source


class FlexpartWrapper:
    """Python-facing wrapper around FLEXPART build and run operations."""

    def __init__(self, domain: DomainConfig, config: dict[str, Any]) -> None:
        self.domain = domain
        self.config = dict(config)
        repo_root = self._repo_root()
        source_dir = self._resolve_path(self.config.get("source_dir", repo_root / "flexpart" / "src"))
        executable = self._resolve_path(self.config.get("executable", source_dir / "FLEXPART"))
        self.compiler = FlexpartCompiler(
            repo_root=repo_root,
            source_dir=source_dir,
            executable_name=executable.name,
            compiler=str(self.config.get("compiler", "gfortran")),
        )

    def compile(
        self,
        *,
        force: bool = False,
        clean: bool = False,
        jobs: int | None = None,
    ) -> FlexpartBuildResult:
        return self.compiler.build(force=force, clean=clean, jobs=jobs)

    def prepare(
        self,
        sources: Iterable[Source],
        instruments: Iterable[Instrument],
    ) -> UnitRunResult:
        dry_run_config = dict(self.config)
        dry_run_config["dry_run"] = True
        runner = FlexpartRunner(domain=self.domain, config=dry_run_config)
        return runner.run(sources, instruments)

    def run(
        self,
        sources: Iterable[Source],
        instruments: Iterable[Instrument],
        *,
        build_if_missing: bool | None = None,
    ) -> UnitRunResult:
        should_build = self.config.get("build_if_missing", False)
        if build_if_missing is not None:
            should_build = build_if_missing

        if should_build and not self.compiler.executable.exists():
            self.compile(
                force=bool(self.config.get("rebuild_flexpart", False)),
                clean=bool(self.config.get("clean_build", False)),
                jobs=self.config.get("build_jobs"),
            )

        run_config = dict(self.config)
        run_config["executable"] = str(self.compiler.executable)
        runner = FlexpartRunner(domain=self.domain, config=run_config)
        return runner.run(sources, instruments)

    def smoke_test_launch(
        self,
        *,
        cwd: str | Path | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return self.compiler.smoke_test(cwd=cwd, args=args, env=env)

    def _resolve_path(self, value: str | Path) -> Path:
        return resolve_path(value, base=self._repo_root())

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[3]
