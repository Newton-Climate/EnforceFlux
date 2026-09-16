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
    # haswell, not x86-64-v3: openmpi/4.1.2 (pinned in modules.sh) swaps in
    # gcc/10.1.0, which predates the x86-64-v3 name. Same AVX2 baseline.
    set(NATIVE_ARCH_FLAG "-march=haswell")
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
# CMake's find_library() does NOT consult $LIBRARY_PATH / $CPATH, so the module
# environment alone is not enough. Translate them into the variables CMake does
# search, including lib64 (Sherlock/Spack installs land in both).
foreach(_efx_var LIBRARY_PATH LD_LIBRARY_PATH)
    if(DEFINED ENV{${_efx_var}})
        string(REPLACE ":" ";" _efx_dirs "$ENV{${_efx_var}}")
        foreach(_efx_dir ${_efx_dirs})
            if(_efx_dir)
                list(APPEND CMAKE_LIBRARY_PATH "${_efx_dir}")
                string(REGEX REPLACE "/lib$" "/lib64" _efx_alt "${_efx_dir}")
                list(APPEND CMAKE_LIBRARY_PATH "${_efx_alt}")
            endif()
        endforeach()
    endif()
endforeach()
if(DEFINED ENV{CPATH})
    string(REPLACE ":" ";" _efx_incs "$ENV{CPATH}")
    list(APPEND CMAKE_INCLUDE_PATH ${_efx_incs})
endif()
list(REMOVE_DUPLICATES CMAKE_LIBRARY_PATH)

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
