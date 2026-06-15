.PHONY: env install install-flexpart figures lint format test clean

VENV ?= .venv

FLEXPART_REPO = https://gitlab.phaidra.org/flexpart/flexpart.git
FLEXPART_DIR  = flexpart
FLEXPART_BIN  = $(FLEXPART_DIR)/src/FLEXPART

ECCODES_PREFIX  ?= $(shell brew --prefix eccodes 2>/dev/null || echo /usr/local)
NETCDF_PREFIX   ?= $(shell brew --prefix netcdf  2>/dev/null || echo /usr/local)
NETCDFF_LIBDIR  ?= $(shell brew --prefix netcdf-fortran 2>/dev/null)/lib

CPATH_FLEXPART    = $(ECCODES_PREFIX)/include:$(NETCDF_PREFIX)/include
LIBRARY_PATH_FLEXPART = $(ECCODES_PREFIX)/lib:$(NETCDF_PREFIX)/lib:$(NETCDFF_LIBDIR)

# Create a local virtual environment (.venv) and upgrade build tooling
env:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip setuptools wheel
	@echo "Activate with: source $(VENV)/bin/activate"

# Full install: Python package (all extras) + FLEXPART binary
install: install-flexpart
	pip install -e ".[all]"

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

# Regenerate the single-source instrument OSSE figures (Gaussian plume,
# self-contained — no ERA5/FLEXPART required). Needs the .[analysis] extra.
figures:
	python examples/single_source_instrument_demo.py

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# Skip flexpart_integration tests by default; pass MARKERS='' to run all
MARKERS ?= not flexpart_integration
test:
	pytest tests/ -v -m "$(MARKERS)" --cov=src --cov-report=term-missing

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
