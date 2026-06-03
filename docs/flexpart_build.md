# FLEXPART Build (macOS + Homebrew)

## Dependencies

```bash
brew install gcc eccodes netcdf netcdf-fortran
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

## Environment variables

```bash
export CPATH="/opt/homebrew/include:$(brew --prefix eccodes)/include:$(brew --prefix netcdf)/include:$(brew --prefix netcdf-fortran)/include"
export LIBRARY_PATH="/opt/homebrew/lib:$(brew --prefix eccodes)/lib:$(brew --prefix netcdf)/lib:$(brew --prefix netcdf-fortran)/lib"
```

## Build

```bash
.venv/bin/python - <<'PY'
from enforceflux.flexpart import FlexpartCompiler

compiler = FlexpartCompiler()
result = compiler.build(jobs=2)
print(result.executable)
PY
```

## Runtime notes

FLEXPART compiled with OpenMP typically requires:

```bash
ulimit -s unlimited
export OMP_PLACES=cores
export OMP_PROC_BIND=true
```

If you run FLEXPART from the Python plugin, set `OMP_*` in the transport config `env` field or export them before launching.

## Apple Silicon notes

The upstream `makefile_gfortran` uses x86-oriented flags such as `-mcmodel=large` and `-march=native`, which do not compile cleanly on macOS arm64. `FlexpartCompiler` automatically swaps those for portable Homebrew-compatible flags and pre-populates `gitversion.txt` so the GNU-specific `sed -i` rule is not needed during the build.

## Verification

Run the repo test suite, including the compiled-binary smoke test:

```bash
.venv/bin/python -m pytest -q
```
