import platform
from pathlib import Path

import pytest

from enforceflux.config import DomainConfig
from enforceflux.flexpart.build import FlexpartCompiler
from enforceflux.flexpart.runner import FlexpartRunner
from enforceflux.flexpart.wrapper import FlexpartWrapper
from enforceflux.models.instrument import Instrument
from enforceflux.models.source import Source


def _domain() -> DomainConfig:
    return DomainConfig(
        x_min=0,
        x_max=1,
        y_min=0,
        y_max=1,
        grid_spacing=1,
        crs="EPSG:32610",
        crs_wgs84="EPSG:4326",
    )


def _sources() -> list[Source]:
    return [
        Source(
            id="S1",
            kind="point",
            x=500,
            y=500,
            z=10,
            flux_true=1.0,
            flux_prior_mean=0.0,
            flux_prior_std=1.0,
        )
    ]


def _instruments() -> list[Instrument]:
    return [
        Instrument(
            id="I1",
            kind="open_path",
            x=600,
            y=600,
            z=2,
            noise_std=0.1,
            averaging_seconds=60,
        )
    ]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _wrapper_config(tmp_path: Path) -> dict[str, str | bool]:
    repo_root = _repo_root()
    return {
        "base_run_dir": str(tmp_path / "runs"),
        "options_dir": str(repo_root / "flexpart" / "options"),
        "available_file": str(repo_root / "flexpart" / "AVAILABLE"),
        "meteo_dir": str(tmp_path / "meteo"),
        "executable": str(repo_root / "flexpart" / "src" / "FLEXPART"),
    }


def test_flexpart_dry_run(tmp_path):
    domain = _domain()
    sources = _sources()
    instruments = _instruments()
    repo_root = _repo_root()
    config = {
        "base_run_dir": str(tmp_path / "runs"),
        "options_dir": str(repo_root / "flexpart" / "options"),
        "available_file": str(repo_root / "flexpart" / "AVAILABLE"),
        "meteo_dir": str(tmp_path / "meteo"),
        "dry_run": True,
    }

    runner = FlexpartRunner(domain=domain, config=config)
    result = runner.run(sources, instruments)

    run_dir = Path(config["base_run_dir"]) / "source_S1"
    assert run_dir.exists()
    assert (run_dir / "pathnames").exists()
    assert (run_dir / "options" / "RELEASES").exists()
    assert (run_dir / "options" / "RECEPTORS").exists()
    assert result.g.shape == (1, 1)


def test_flexpart_wrapper_prepare(tmp_path):
    wrapper = FlexpartWrapper(domain=_domain(), config=_wrapper_config(tmp_path))
    result = wrapper.prepare(_sources(), _instruments())

    run_dir = tmp_path / "runs" / "source_S1"
    assert run_dir.exists()
    assert (run_dir / "pathnames").exists()
    assert (run_dir / "options" / "RELEASES").exists()
    assert (run_dir / "options" / "RECEPTORS").exists()
    assert result.g.shape == (1, 1)


def test_flexpart_build_plan_uses_portable_flags():
    compiler = FlexpartCompiler(repo_root=_repo_root())
    plan = compiler.plan()

    assert "/opt/homebrew/include" in plan.env["CPATH"]
    assert "/opt/homebrew/lib" in plan.env["LIBRARY_PATH"]

    make_args = " ".join(plan.make_args)
    if platform.system() == "Darwin" and platform.machine().lower() == "arm64":
        assert "-mcmodel=large" not in make_args
        assert "-march=native" not in make_args


@pytest.mark.flexpart_integration
def test_flexpart_wrapper_can_compile_and_launch_binary(tmp_path):
    wrapper = FlexpartWrapper(domain=_domain(), config=_wrapper_config(tmp_path))
    build = wrapper.compile(jobs=2)

    assert build.executable.exists()

    completed = wrapper.smoke_test_launch(cwd=tmp_path)
    combined_output = completed.stdout + completed.stderr

    assert "Welcome to FLEXPART" in combined_output
    assert "FLEXPART is running with METER coordinates." in combined_output
