# EnforceFlux cross-platform MicroHH config
# Maintained in EnforceFlux so platform fixes are version-controlled.
# Supports: macOS (Intel x86_64 + Apple Silicon arm64),
#           Linux (x86_64 + aarch64), Windows (MSYS2/Cygwin treated as Linux).
#
# Copied into microhh/config/ during `make install-microhh`.

# ── OS and architecture detection ────────────────────────────────────────────
# This config is included BEFORE project(), so CMAKE_SYSTEM_NAME /
# CMAKE_SYSTEM_PROCESSOR are not yet fully populated. Use uname instead.
if(WIN32)
    set(DETECTED_OS   "Windows")
    set(DETECTED_ARCH "x86_64")
else()
    execute_process(COMMAND uname -s OUTPUT_VARIABLE DETECTED_OS   OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND uname -m OUTPUT_VARIABLE DETECTED_ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(DETECTED_OS STREQUAL "Darwin")
    set(MICROHH_MACOS TRUE)
    # Resolve Homebrew prefix: Apple Silicon = /opt/homebrew, Intel = /usr/local
    execute_process(
        COMMAND brew --prefix
        OUTPUT_VARIABLE HOMEBREW_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT HOMEBREW_PREFIX)
        set(HOMEBREW_PREFIX "/opt/homebrew")
    endif()
    set(GNU_SED "gsed")
else()
    set(MICROHH_MACOS FALSE)
    set(GNU_SED "sed")
endif()

# ARM architecture (Apple Silicon = arm64, Linux ARM = aarch64)
if(DETECTED_ARCH MATCHES "arm64|aarch64")
    set(MICROHH_ARM TRUE)
else()
    set(MICROHH_ARM FALSE)
endif()
# ─────────────────────────────────────────────────────────────────────────────

# ── Compiler selection ────────────────────────────────────────────────────────
if(USEMPI)
    set(ENV{CC}  mpicc )
    set(ENV{CXX} mpicxx)
    set(ENV{FC}  mpif90)
else()
    if(MICROHH_MACOS)
        # Clang ships with Xcode CLT on macOS; more stable than Homebrew GCC
        # for C/C++. Fortran still needs gfortran (brew install gcc).
        set(ENV{CC}  clang  )
        set(ENV{CXX} clang++)
    else()
        set(ENV{CC}  gcc)
        set(ENV{CXX} g++)
    endif()
    set(ENV{FC} gfortran)
endif()
# ─────────────────────────────────────────────────────────────────────────────

# ── CPU architecture flags ────────────────────────────────────────────────────
# -march=native is safe for Clang and GCC on x86.
# For gfortran on AArch64, early toolchain versions reject -march=native when it
# resolves to a CPU name (e.g. apple-m1); -mcpu=native is the correct flag.
# Clang on Apple Silicon handles -march=native fine for C++.
if(MICROHH_ARM)
    set(NATIVE_FC_FLAG "-mcpu=native")
else()
    set(NATIVE_FC_FLAG "-march=native")
endif()
# ─────────────────────────────────────────────────────────────────────────────

# ── Compiler flags ────────────────────────────────────────────────────────────
set(USER_CXX_FLAGS         "-std=c++17")
set(USER_CXX_FLAGS_RELEASE "-DNDEBUG -O3 -march=native")
set(USER_CXX_FLAGS_DEBUG   "-O0 -g -Wall -Wno-unknown-pragmas")

set(USER_FC_FLAGS
    "-fdefault-real-8 -fdefault-double-8 -fPIC -ffixed-line-length-none -fno-range-check")
set(USER_FC_FLAGS_RELEASE  "-DNDEBUG -O3 ${NATIVE_FC_FLAG}")
set(USER_FC_FLAGS_DEBUG    "-O0 -g -Wall -Wno-unknown-pragmas")

add_definitions(-DRESTRICTKEYWORD=__restrict__)
# ─────────────────────────────────────────────────────────────────────────────

# ── Library paths ─────────────────────────────────────────────────────────────
if(MICROHH_MACOS)
    # Use full dylib paths so CMake picks up Homebrew rather than any system
    # frameworks that might shadow the Homebrew libraries.
    set(FFTW_INCLUDE_DIR   "${HOMEBREW_PREFIX}/include")
    set(NETCDF_INCLUDE_DIR "${HOMEBREW_PREFIX}/include")
    set(FFTW_LIB           "${HOMEBREW_PREFIX}/lib/libfftw3.dylib")
    set(FFTWF_LIB          "${HOMEBREW_PREFIX}/lib/libfftw3f.dylib")
    set(NETCDF_LIB_C       "${HOMEBREW_PREFIX}/lib/libnetcdf.dylib")
    set(HDF5_LIB_1         "${HOMEBREW_PREFIX}/lib/libhdf5.dylib")
    set(HDF5_LIB_2         "${HOMEBREW_PREFIX}/lib/libhdf5_hl.dylib")
    set(SZIP_LIB           "${HOMEBREW_PREFIX}/lib/libsz.dylib")
    set(LIBS
        ${FFTW_LIB} ${FFTWF_LIB}
        ${NETCDF_LIB_C}
        ${HDF5_LIB_2} ${HDF5_LIB_1}
        ${SZIP_LIB}
        m z curl)
    set(INCLUDE_DIRS ${FFTW_INCLUDE_DIR} ${NETCDF_INCLUDE_DIR})
else()
    # Linux: let the linker resolve libraries through standard search paths.
    # Ubuntu names HDF5 as hdf5_serial; Fedora/Arch/generic distros use hdf5.
    find_library(HDF5_LIB NAMES hdf5_serial hdf5 REQUIRED)
    set(FFTW_LIB     "fftw3")
    set(FFTWF_LIB    "fftw3f")
    set(NETCDF_LIB_C "netcdf")
    set(LIBS ${FFTW_LIB} ${FFTWF_LIB} ${NETCDF_LIB_C} ${HDF5_LIB})
endif()
# ─────────────────────────────────────────────────────────────────────────────

# ── CUDA (optional, serial-only — MPI+CUDA is not supported by MicroHH) ──────
if(USECUDA)
    set(CMAKE_CUDA_ARCHITECTURES     "70;80;90")
    set(USER_CUDA_NVCC_FLAGS         "--expt-relaxed-constexpr -lineinfo")
    set(USER_CUDA_NVCC_FLAGS_RELEASE "-DNDEBUG")
    set(USER_CUDA_NVCC_FLAGS_DEBUG   "-O0 -g -DCUDACHECKS")
    add_definitions(-DRTE_RRTMGP_GPU_MEMPOOL_CUDA)
    set(LIBS ${LIBS} -rdynamic)
endif()
# ─────────────────────────────────────────────────────────────────────────────

add_definitions(-DDISABLE_2D_MPIIO=1)
add_definitions(-DRTE_USE_CBOOL)
