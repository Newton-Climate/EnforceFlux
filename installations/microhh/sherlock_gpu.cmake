# EnforceFlux — MicroHH config for Sherlock (Stanford SRCC), single-GPU build.
#
# Companion to sherlock.cmake. Use this one for CUDA runs:
#   cmake -S microhh -B microhh/build_gpu -DSYST=sherlock_gpu \
#         -DCMAKE_BUILD_TYPE=RELEASE -DUSECUDA=TRUE -DUSEMPI=FALSE -DUSESP=TRUE
#
# MicroHH refuses to combine USEMPI with USECUDA, so a CUDA build is always
# single-rank: every case run against it needs num_workers: 1.
#
# USESP=TRUE gives single-precision (FLOAT_SINGLE). Keep it in step with the
# `precision:` field of the EnforceFlux MicroHH configs — that field tells
# EnforceFlux what dtype MicroHH's raw sbot_2d / cross-section files are in,
# and a mismatch silently misreads them.
#
# Assumes the module environment from installations/sherlock/modules.sh, plus
# the cuda module (loaded there via the toolchain).

# ── Compiler selection ───────────────────────────────────────────────────────
# No MPI wrappers here: CUDA builds are serial.
set(ENV{CC}  gcc)
set(ENV{CXX} g++)
set(ENV{FC}  gfortran)

# ── Host compiler flags ──────────────────────────────────────────────────────
# Sherlock's GPU nodes span several CPU generations, so avoid -march=native.
set(NATIVE_ARCH_FLAG "$ENV{MICROHH_ARCH_FLAG}")
if(NOT NATIVE_ARCH_FLAG)
    set(NATIVE_ARCH_FLAG "-march=x86-64-v3")
endif()

set(USER_CXX_FLAGS         "-std=c++17 -fopenmp")
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

# ── CUDA ─────────────────────────────────────────────────────────────────────
if(USECUDA)
    # Sherlock's GPU pool is heterogeneous (V100 CC 7.0, A100 8.0, RTX/L40S
    # 8.6-8.9, H100/H200 9.0). Emit SASS for all of them so the binary is not
    # pinned to whichever node the build happened to land on. Override with
    # -DCMAKE_CUDA_ARCHITECTURES=80 to build faster for one target.
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70 80 86 90)
    endif()
    set(USER_CUDA_NVCC_FLAGS         "--expt-relaxed-constexpr")
    set(USER_CUDA_NVCC_FLAGS_RELEASE "-Xptxas -O3 -DNDEBUG")
    set(USER_CUDA_NVCC_FLAGS_DEBUG   "-Xptxas -O0 -g -DCUDACHECKS")
    add_definitions(-DRTE_RRTMGP_GPU_MEMPOOL_CUDA)
endif()

add_definitions(-DDISABLE_2D_MPIIO=1)
add_definitions(-DRTE_USE_CBOOL)
