# EnforceFlux — Sherlock (Stanford SRCC) module environment.
#
# Source this from an interactive shell or SLURM job BEFORE running any of the
# `make install-*-sherlock` targets. It:
#   1. loads a known-good toolchain via Lmod,
#   2. exports the *_PREFIX / *_DIR variables the top-level Makefile consumes
#      (ECCODES_PREFIX, NETCDF_PREFIX, NETCDFF_LIBDIR, NETCDFF_INCDIR),
#   3. exports SHERLOCK=1 so the Makefile's Sherlock targets can be gated.
#
# Usage:
#   source installations/sherlock/modules.sh
#   make install-sherlock                # FLEXPART + MicroHH + Python package
#
# Every module version can be overridden by exporting *_MOD before sourcing,
# e.g. `GCC_MOD=gcc/12.4.0 source installations/sherlock/modules.sh`. If a
# module name is wrong for your Sherlock generation, `module avail <name>` will
# show what's on the system.

# ── Module defaults (override by exporting before sourcing) ──────────────────
: "${GCC_MOD:=gcc}"
: "${OPENMPI_MOD:=openmpi}"
: "${CMAKE_MOD:=cmake}"
: "${HDF5_MOD:=hdf5}"
: "${NETCDF_C_MOD:=netcdf-c}"
: "${NETCDF_FORTRAN_MOD:=netcdf-fortran}"
: "${FFTW_MOD:=fftw}"
: "${ECCODES_MOD:=eccodes}"          # if missing on your tree: unset ECCODES_MOD
: "${PYTHON_MOD:=python/3.12.1}"
: "${BOOST_MOD:=boost}"              # MicroHH build-dep
: "${GIT_MOD:=git}"

# ── Load Lmod stack ──────────────────────────────────────────────────────────
if ! command -v module >/dev/null 2>&1; then
    echo "ERROR: 'module' command not found — are you on a Sherlock login/compute node?" >&2
    return 1 2>/dev/null || exit 1
fi

module reset >/dev/null 2>&1 || module purge

_load() {
    local m="$1"
    [ -z "$m" ] && return 0
    if ! module load "$m" 2>/dev/null; then
        echo "WARN: could not load module '$m' — check 'module avail ${m%%/*}'" >&2
        return 1
    fi
}

_load "$GCC_MOD"
_load "$OPENMPI_MOD"
_load "$CMAKE_MOD"
_load "$GIT_MOD"
_load "$HDF5_MOD"
_load "$NETCDF_C_MOD"
_load "$NETCDF_FORTRAN_MOD"
_load "$FFTW_MOD"
_load "$BOOST_MOD"
_load "$ECCODES_MOD" || echo "NOTE: eccodes module not loaded — FLEXPART build will fail unless you provide ECCODES_PREFIX manually." >&2
_load "$PYTHON_MOD"

# ── Resolve prefixes for the Makefile ────────────────────────────────────────
# Lmod modules on Sherlock usually export *_ROOT (Spack convention). Fall back
# to deriving from the loaded module's install path if not set.
_prefix_from_module() {
    # $1 = module name (e.g. "netcdf-c"), $2 = ROOT env var name.
    local mod="$1" root_var="$2"
    local root="${!root_var}"
    if [ -n "$root" ]; then
        printf '%s' "$root"
        return
    fi
    # Fall back to `module show` output → look for "prepend_path PATH <prefix>/bin"
    module show "$mod" 2>&1 \
        | awk '/prepend_path\("PATH"/ { gsub(/[",)]/, "", $3); sub(/\/bin$/, "", $3); print $3; exit }'
}

export ECCODES_PREFIX="${ECCODES_PREFIX:-$(_prefix_from_module "$ECCODES_MOD" ECCODES_ROOT)}"
export NETCDF_PREFIX="${NETCDF_PREFIX:-$(_prefix_from_module "$NETCDF_C_MOD" NETCDF_C_ROOT)}"
_NCF_PREFIX="$(_prefix_from_module "$NETCDF_FORTRAN_MOD" NETCDF_FORTRAN_ROOT)"
export NETCDFF_LIBDIR="${NETCDFF_LIBDIR:-${_NCF_PREFIX}/lib}"
export NETCDFF_INCDIR="${NETCDFF_INCDIR:-${_NCF_PREFIX}/include}"
export HDF5_PREFIX="${HDF5_PREFIX:-$(_prefix_from_module "$HDF5_MOD" HDF5_ROOT)}"
export FFTW_PREFIX="${FFTW_PREFIX:-$(_prefix_from_module "$FFTW_MOD" FFTW_ROOT)}"

# So the Sherlock MicroHH cmake config can find libs without hardcoding paths.
export CPATH="${ECCODES_PREFIX:+$ECCODES_PREFIX/include:}${NETCDF_PREFIX:+$NETCDF_PREFIX/include:}${NETCDFF_INCDIR:+$NETCDFF_INCDIR:}${HDF5_PREFIX:+$HDF5_PREFIX/include:}${FFTW_PREFIX:+$FFTW_PREFIX/include:}${CPATH}"
export LIBRARY_PATH="${ECCODES_PREFIX:+$ECCODES_PREFIX/lib:}${NETCDF_PREFIX:+$NETCDF_PREFIX/lib:}${NETCDFF_LIBDIR:+$NETCDFF_LIBDIR:}${HDF5_PREFIX:+$HDF5_PREFIX/lib:}${FFTW_PREFIX:+$FFTW_PREFIX/lib:}${LIBRARY_PATH}"
export LD_LIBRARY_PATH="${ECCODES_PREFIX:+$ECCODES_PREFIX/lib:}${NETCDF_PREFIX:+$NETCDF_PREFIX/lib:}${NETCDFF_LIBDIR:+$NETCDFF_LIBDIR:}${HDF5_PREFIX:+$HDF5_PREFIX/lib:}${FFTW_PREFIX:+$FFTW_PREFIX/lib:}${LD_LIBRARY_PATH}"

export SHERLOCK=1

echo "── EnforceFlux Sherlock env loaded ──"
module list 2>&1 | sed 's/^/  /'
echo "  ECCODES_PREFIX = ${ECCODES_PREFIX:-<unset>}"
echo "  NETCDF_PREFIX  = ${NETCDF_PREFIX:-<unset>}"
echo "  NETCDFF_INCDIR = ${NETCDFF_INCDIR:-<unset>}"
echo "  NETCDFF_LIBDIR = ${NETCDFF_LIBDIR:-<unset>}"
echo "  HDF5_PREFIX    = ${HDF5_PREFIX:-<unset>}"
echo "  FFTW_PREFIX    = ${FFTW_PREFIX:-<unset>}"
echo "Next: make install-sherlock"
