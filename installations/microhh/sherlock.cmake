# EnforceFlux — MicroHH config for Sherlock (Stanford SRCC).
#
# Assumes the module environment from installations/sherlock/modules.sh is
# loaded (gcc, openmpi, cmake, hdf5, netcdf-c, netcdf-fortran, fftw). Libraries
# are resolved through CPATH / LIBRARY_PATH exported there, so we don't
# hardcode Spack install prefixes that change every Sherlock generation.
#
# Copied into microhh/config/sherlock.cmake by `make install-microhh-sherlock`.

# ── Compiler selection ───────────────────────────────────────────────────────
if(USEMPI)
    set(ENV{CC}  mpicc )
    set(ENV{CXX} mpicxx)
    set(ENV{FC}  mpif90)
else()
    set(ENV{CC}  gcc)
    set(ENV{CXX} g++)
    set(ENV{FC}  gfortran)
endif()

# ── Compiler flags ───────────────────────────────────────────────────────────
# Sherlock login nodes and compute nodes have different CPU generations
# (Skylake, Cascade Lake, Icelake, EPYC). `-march=native` on a login node
# produces a binary that may crash on other partitions — override with
# MICROHH_ARCH_FLAG when configuring, e.g.
#   MICROHH_ARCH_FLAG=-march=skylake-avx512 make install-microhh-sherlock
set(NATIVE_ARCH_FLAG "$ENV{MICROHH_ARCH_FLAG}")
if(NOT NATIVE_ARCH_FLAG)
    set(NATIVE_ARCH_FLAG "-march=x86-64-v3")
endif()

set(USER_CXX_FLAGS         "-std=c++17")
set(USER_CXX_FLAGS_RELEASE "-DNDEBUG -O3 ${NATIVE_ARCH_FLAG}")
set(USER_CXX_FLAGS_DEBUG   "-O0 -g -Wall -Wno-unknown-pragmas")

set(USER_FC_FLAGS
    "-fdefault-real-8 -fdefault-double-8 -fPIC -ffixed-line-length-none -fno-range-check")
set(USER_FC_FLAGS_RELEASE  "-DNDEBUG -O3 ${NATIVE_ARCH_FLAG}")
set(USER_FC_FLAGS_DEBUG    "-O0 -g -Wall -Wno-unknown-pragmas")

add_definitions(-DRESTRICTKEYWORD=__restrict__)

# ── Library resolution ───────────────────────────────────────────────────────
# Rely on the CPATH / LIBRARY_PATH from the sourced module env; ask CMake to
# search for HDF5 (Spack builds it as plain libhdf5, not libhdf5_serial).
find_library(HDF5_LIB     NAMES hdf5 hdf5_serial REQUIRED)
find_library(HDF5_HL_LIB  NAMES hdf5_hl hdf5_serial_hl)
find_library(FFTW_LIB     NAMES fftw3   REQUIRED)
find_library(FFTWF_LIB    NAMES fftw3f  REQUIRED)
find_library(NETCDF_LIB_C NAMES netcdf  REQUIRED)

set(LIBS
    ${FFTW_LIB} ${FFTWF_LIB}
    ${NETCDF_LIB_C}
    ${HDF5_HL_LIB} ${HDF5_LIB}
    m z curl)

add_definitions(-DDISABLE_2D_MPIIO=1)
add_definitions(-DRTE_USE_CBOOL)
