.PHONY: env install install-dev install-flexpart install-microhh figures lint format test clean

VENV ?= .venv
PYTHON ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
PIP ?= $(PYTHON) -m pip

FLEXPART_REPO = https://gitlab.phaidra.org/flexpart/flexpart.git
FLEXPART_DIR  = flexpart
FLEXPART_BIN  = $(FLEXPART_DIR)/src/FLEXPART

ECCODES_PREFIX  ?= $(shell brew --prefix eccodes 2>/dev/null || echo /usr/local)
NETCDF_PREFIX   ?= $(shell brew --prefix netcdf  2>/dev/null || echo /usr/local)
NETCDFF_LIBDIR  ?= $(shell brew --prefix netcdf-fortran 2>/dev/null)/lib
NETCDFF_INCDIR  ?= $(shell brew --prefix netcdf-fortran 2>/dev/null)/include

CPATH_FLEXPART    = $(ECCODES_PREFIX)/include:$(NETCDF_PREFIX)/include:$(NETCDFF_INCDIR)
LIBRARY_PATH_FLEXPART = $(ECCODES_PREFIX)/lib:$(NETCDF_PREFIX)/lib:$(NETCDFF_LIBDIR)

MICROHH_REPO = https://github.com/microhh/microhh.git
MICROHH_DIR  = microhh
MICROHH_BIN  = $(MICROHH_DIR)/build/microhh
# CMake system-config selector (installations/<SYST>.cmake).
# Defaults to the cross-platform config maintained in this repo.
# Override to use an upstream preset, e.g. MICROHH_SYST=ubuntu_20lts.
MICROHH_SYST ?= enforceflux
# MPI-parallel by default: the configs ship with num_workers: 4, which a
# serial binary cannot honour. Needs an MPI toolchain (brew install open-mpi).	
# Set MICROHH_MPI=0 for a serial binary — then every case must use
# num_workers: 1. MicroHH cannot combine MPI with CUDA.
MICROHH_MPI  ?= 1
JOBS         ?= 4
ifeq ($(MICROHH_MPI),1)
MICROHH_CMAKE_FLAGS = -DUSEMPI=TRUE
endif

# Create a local virtual environment (.venv) and upgrade build tooling
env:
	python3 -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip setuptools wheel
	@echo "Activate with: source $(VENV)/bin/activate"

# Full install: Python package (all extras) + FLEXPART binary
install: install-flexpart
	$(PIP) install -e ".[all]"

# Lightweight install for local development and test discovery (entry points)
install-dev:
	$(PIP) install -e ".[dev]"

# Clone (if absent) and compile FLEXPART
install-flexpart:
	@if [ ! -d "$(FLEXPART_DIR)/.git" ]; then \
		echo "Cloning FLEXPART from $(FLEXPART_REPO) ..."; \
		git clone --depth=1 "$(FLEXPART_REPO)" "$(FLEXPART_DIR)"; \
	else \
		echo "FLEXPART repo already present at $(FLEXPART_DIR)/"; \
	fi
	@# Replace the upstream makefile_gfortran with our cross-platform version,
	@# which handles macOS (Intel + Apple Silicon) and Linux (x86_64 + aarch64).
	cp installations/flexpart/makefile_gfortran "$(FLEXPART_DIR)/src/makefile_gfortran"
	@echo "Compiling FLEXPART (eta + NetCDF) ..."
	CPATH="$(CPATH_FLEXPART)" \
	LIBRARY_PATH="$(LIBRARY_PATH_FLEXPART)" \
	FC=gfortran \
	$(MAKE) -f makefile_gfortran -C "$(FLEXPART_DIR)/src" eta=yes -j$(JOBS)
	@echo "FLEXPART binary: $$(ls -lh $(FLEXPART_BIN))"

# Clone (with submodules) and compile MicroHH — plume-scale LES backend.
# Build deps (Homebrew): cmake fftw hdf5 netcdf boost gnu-sed open-mpi. Install:
#   brew install cmake fftw hdf5 netcdf boost gnu-sed open-mpi
# MPI-parallel (MICROHH_MPI=1), double-precision build using the
# installations/$(MICROHH_SYST).cmake system file. For a single-GPU build, pass a
# CUDA-enabled SYST, MICROHH_MPI=0 and add -DUSECUDA=TRUE below (note: MicroHH
# cannot combine -DUSEMPI with -DUSECUDA).
install-microhh:
	@if [ ! -d "$(MICROHH_DIR)/.git" ]; then \
		echo "Cloning MicroHH from $(MICROHH_REPO) ..."; \
		git clone --recurse-submodules "$(MICROHH_REPO)" "$(MICROHH_DIR)"; \
	else \
		echo "MicroHH repo already present at $(MICROHH_DIR)/"; \
		git -C "$(MICROHH_DIR)" submodule update --init --recursive; \
	fi
	@# Copy the cross-platform cmake config so the build can find it.
	cp installations/microhh/enforceflux.cmake "$(MICROHH_DIR)/config/enforceflux.cmake"
	@# USEMPI selects the compilers (mpicxx vs clang++), which CMake refuses to
	@# change in place — drop a cache configured the other way.
	@if [ -f "$(MICROHH_DIR)/build/CMakeCache.txt" ] && \
	   [ "$$(grep -c '^CMAKE_CXX_COMPILER:.*mpi' "$(MICROHH_DIR)/build/CMakeCache.txt")" != "$(MICROHH_MPI)" ]; then \
		echo "USEMPI changed — wiping $(MICROHH_DIR)/build ..."; \
		rm -rf "$(MICROHH_DIR)/build"; \
	fi
	@echo "Configuring MicroHH (SYST=$(MICROHH_SYST), MPI=$(MICROHH_MPI), RELEASE) ..."
	cmake -S "$(MICROHH_DIR)" -B "$(MICROHH_DIR)/build" \
		-DSYST=$(MICROHH_SYST) -DCMAKE_BUILD_TYPE=RELEASE $(MICROHH_CMAKE_FLAGS)
	@echo "Compiling MicroHH ..."
	cmake --build "$(MICROHH_DIR)/build" --target microhh -j$(JOBS)
	@echo "MicroHH binary: $$(ls -lh $(MICROHH_BIN))"

# Regenerate the single-source instrument OSSE figures (Gaussian plume,
# self-contained — no ERA5/FLEXPART required). Needs the .[analysis] extra.
figures:
	$(PYTHON) examples/single_source_instrument_demo.py

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# Skip flexpart_integration tests by default; pass MARKERS='' to run all
MARKERS ?= not flexpart_integration
test: install-dev
	$(PYTHON) -m pytest tests/ -v -m "$(MARKERS)" --cov=src --cov-report=term-missing

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
