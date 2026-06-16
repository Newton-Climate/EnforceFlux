import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class FlexpartBuildPlan:
    source_dir: Path
    executable: Path
    env: dict[str, str]
    make_args: tuple[str, ...]


@dataclass(frozen=True)
class FlexpartBuildResult:
    executable: Path
    rebuilt: bool
    plan: FlexpartBuildPlan


class FlexpartCompiler:
    """Compile the vendored FLEXPART source tree for the local machine."""

    def __init__(
        self,
        repo_root: str | Path | None = None,
        source_dir: str | Path | None = None,
        executable_name: str = "FLEXPART",
        makefile: str = "makefile_gfortran",
        compiler: str = "gfortran",
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root is not None else _default_repo_root()
        self.source_dir = (
            Path(source_dir) if source_dir is not None else self.repo_root / "flexpart" / "src"
        )
        self.executable = self.source_dir / executable_name
        self.makefile = makefile
        self.compiler = compiler

    def plan(
        self,
        *,
        eta: bool = False,
        serial: bool = False,
        debug: bool = False,
    ) -> FlexpartBuildPlan:
        include_dirs = self._discover_dirs("include")
        lib_dirs = self._discover_dirs("lib")
        self._validate_toolchain(include_dirs=include_dirs, lib_dirs=lib_dirs)

        env = os.environ.copy()
        env["CPATH"] = self._merge_path_var(env.get("CPATH"), include_dirs)
        env["LIBRARY_PATH"] = self._merge_path_var(env.get("LIBRARY_PATH"), lib_dirs)

        make_args = [
            f"FC={self.compiler}",
            f"eta={'yes' if eta else 'no'}",
        ]
        if serial:
            make_args.append("SERIAL=yes")
        if debug:
            make_args.append("DEBUG=yes")

        if self._needs_portable_overrides():
            make_args.extend(
                self._portable_make_overrides(
                    include_dirs=include_dirs,
                    lib_dirs=lib_dirs,
                    eta=eta,
                )
            )

        return FlexpartBuildPlan(
            source_dir=self.source_dir,
            executable=self.executable,
            env=env,
            make_args=tuple(make_args),
        )

    def build(
        self,
        *,
        force: bool = False,
        clean: bool = False,
        eta: bool = False,
        serial: bool = False,
        debug: bool = False,
        jobs: int | None = None,
    ) -> FlexpartBuildResult:
        plan = self.plan(eta=eta, serial=serial, debug=debug)

        if self.executable.exists() and not force and not clean:
            return FlexpartBuildResult(executable=self.executable, rebuilt=False, plan=plan)

        self._sync_gitversion_file()

        if clean or force:
            subprocess.run(
                ["make", "cleanall", "-f", self.makefile],
                cwd=self.source_dir,
                check=True,
                env=plan.env,
            )

        command = ["make"]
        if jobs is not None:
            command.append(f"-j{jobs}")
        command.extend(["-f", self.makefile, *plan.make_args])
        subprocess.run(command, cwd=self.source_dir, check=True, env=plan.env)

        if not self.executable.exists():
            raise RuntimeError(f"Expected FLEXPART executable was not created: {self.executable}")

        return FlexpartBuildResult(executable=self.executable, rebuilt=True, plan=plan)

    def smoke_test(
        self,
        *,
        cwd: str | Path | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if not self.executable.exists():
            raise FileNotFoundError(f"FLEXPART executable not found: {self.executable}")

        run_env = os.environ.copy()
        if env:
            run_env.update({k: str(v) for k, v in env.items()})

        return subprocess.run(
            [str(self.executable), *(args or [])],
            cwd=Path(cwd) if cwd is not None else self.source_dir,
            env=run_env,
            capture_output=True,
            text=True,
            check=False,
        )

    def _discover_dirs(self, subdir: str) -> list[Path]:
        candidates: list[Path] = []
        common_roots = [Path("/opt/homebrew"), Path("/usr/local")]
        for root in common_roots:
            path = root / subdir
            if path.exists():
                candidates.append(path)

        for formula in ("eccodes", "netcdf", "netcdf-fortran"):
            prefix = self._brew_prefix(formula)
            if prefix is None:
                continue
            path = prefix / subdir
            if path.exists():
                candidates.append(path)

        return self._dedupe_paths(candidates)

    def _validate_toolchain(self, *, include_dirs: list[Path], lib_dirs: list[Path]) -> None:
        missing_tools = [name for name in ("make", self.compiler) if shutil.which(name) is None]
        if missing_tools:
            missing = ", ".join(missing_tools)
            raise RuntimeError(f"Missing build tools for FLEXPART: {missing}")

        if not any((path / "eccodes.mod").exists() for path in include_dirs):
            raise RuntimeError(
                "Unable to find ecCodes Fortran headers. Install `eccodes` and ensure it is linked."
            )
        if not any((path / "netcdf.mod").exists() for path in include_dirs):
            raise RuntimeError(
                "Unable to find NetCDF Fortran headers. Install `netcdf-fortran` and ensure it is linked."
            )
        if not any((path / "libeccodes_f90.dylib").exists() or (path / "libeccodes_f90.a").exists() for path in lib_dirs):
            raise RuntimeError(
                "Unable to find ecCodes Fortran libraries. Install `eccodes` and ensure it is linked."
            )
        if not any((path / "libnetcdff.dylib").exists() or (path / "libnetcdff.a").exists() for path in lib_dirs):
            raise RuntimeError(
                "Unable to find NetCDF Fortran libraries. Install `netcdf-fortran` and ensure it is linked."
            )

    def _brew_prefix(self, formula: str) -> Path | None:
        brew = shutil.which("brew")
        if brew is None:
            return None
        result = subprocess.run(
            [brew, "--prefix", formula],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        path = Path(result.stdout.strip())
        return path if path.exists() else None

    def _sync_gitversion_file(self) -> None:
        git_version = self._git_version()
        file_path = self.source_dir / "gitversion.txt"
        current = file_path.read_text().strip() if file_path.exists() else None
        if current == git_version:
            return
        file_path.write_text(f"{git_version}\n")

    def _git_version(self) -> str:
        result = subprocess.run(
            ["git", "log", '--pretty=format:%h %ad', "-n", "1"],
            cwd=self.source_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return "unknown"
        value = result.stdout.strip()
        return value or "unknown"

    def _needs_portable_overrides(self) -> bool:
        return platform.system() == "Darwin" and platform.machine().lower() == "arm64"

    def _portable_make_overrides(
        self,
        *,
        include_dirs: list[Path],
        lib_dirs: list[Path],
        eta: bool,
    ) -> list[str]:
        include_flags = [f"-I{path}" for path in include_dirs]
        lib_search_flags = [f"-L{path}" for path in lib_dirs]
        rpath_flags = [f"-Wl,-rpath,{path}" for path in lib_dirs]

        nc_flags = ["-DUSE_NCF", "-DETA" if eta else "-UETA"]
        fuser_flags = ["-g", "-fopenmp", "-Duseomp"]
        fflags = [*include_flags, "-O3", "-cpp", *nc_flags, *fuser_flags]
        ldflags = [
            *fflags,
            *lib_search_flags,
            *rpath_flags,
            "-leccodes",
            "-leccodes_f90",
            "-lm",
            "-lnetcdff",
        ]
        return [
            f"FUSER={' '.join(fuser_flags)}",
            f"FFLAGS={' '.join(fflags)}",
            f"LDFLAGS={' '.join(ldflags)}",
        ]

    def _merge_path_var(self, current: str | None, paths: list[Path]) -> str:
        merged: list[str] = []
        if current:
            merged.extend([piece for piece in current.split(os.pathsep) if piece])
        merged.extend(str(path) for path in paths)

        deduped: list[str] = []
        seen: set[str] = set()
        for piece in merged:
            if piece in seen:
                continue
            deduped.append(piece)
            seen.add(piece)
        return os.pathsep.join(deduped)

    def _dedupe_paths(self, paths: list[Path]) -> list[Path]:
        deduped: list[Path] = []
        seen: set[Path] = set()
        for path in paths:
            resolved = path.resolve()
            if resolved in seen:
                continue
            deduped.append(resolved)
            seen.add(resolved)
        return deduped
