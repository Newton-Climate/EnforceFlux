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

CPATH_FLEXPART    = $(ECCODES_PREFIX)/include:$(NETCDF_PREFIX)/include
LIBRARY_PATH_FLEXPART = $(ECCODES_PREFIX)/lib:$(NETCDF_PREFIX)/lib:$(NETCDFF_LIBDIR)

MICROHH_REPO = https://github.com/microhh/microhh.git
MICROHH_DIR  = microhh
MICROHH_BIN  = $(MICROHH_DIR)/build/microhh
# CMake system-config selector (config/<SYST>.cmake). Override for other
# platforms, e.g. SYST=macbook_brew, SYST=ubuntu_20lts, SYST=generic.
MICROHH_SYST ?= macbook_apple_silicon

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
	@echo "Compiling FLEXPART (eta + NetCDF) ..."
	CPATH="$(CPATH_FLEXPART)" \
	LIBRARY_PATH="$(LIBRARY_PATH_FLEXPART)" \
	FC=gfortran \
	$(MAKE) -f makefile_gfortran -C "$(FLEXPART_DIR)/src" eta=yes -j4
	@echo "FLEXPART binary: $$(ls -lh $(FLEXPART_BIN))"

# Clone (with submodules) and compile MicroHH — plume-scale LES backend.
# Build deps (Homebrew): cmake fftw hdf5 netcdf boost gnu-sed. Install with:
#   brew install cmake fftw hdf5 netcdf boost gnu-sed
# Serial, double-precision build using the config/$(MICROHH_SYST).cmake system
# file. For a single-GPU build, pass a CUDA-enabled SYST and add -DUSECUDA=TRUE
# below (note: MicroHH cannot combine -DUSEMPI with -DUSECUDA).
install-microhh:
	@if [ ! -d "$(MICROHH_DIR)/.git" ]; then \
		echo "Cloning MicroHH from $(MICROHH_REPO) ..."; \
		git clone --recurse-submodules "$(MICROHH_REPO)" "$(MICROHH_DIR)"; \
	else \
		echo "MicroHH repo already present at $(MICROHH_DIR)/"; \
		git -C "$(MICROHH_DIR)" submodule update --init --recursive; \
	fi
	@echo "Configuring MicroHH (SYST=$(MICROHH_SYST), RELEASE) ..."
	cmake -S "$(MICROHH_DIR)" -B "$(MICROHH_DIR)/build" \
		-DSYST=$(MICROHH_SYST) -DCMAKE_BUILD_TYPE=RELEASE
	@echo "Compiling MicroHH ..."
	cmake --build "$(MICROHH_DIR)/build" --target microhh -j4
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
