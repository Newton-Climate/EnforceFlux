#!/bin/bash
# Rebuild the CPU/MPI MicroHH against the openmpi/4.1.2 stack that Sherlock's
# fftw and netcdf-c modules are built with, then check a 4-rank launch works.
#SBATCH -J efx_mhh_build
#SBATCH -p serc,normal
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /home/users/nnewton/efx_logs/mhh_build_%j.log

set -eo pipefail
cd "$HOME/EnforceFlux" || exit 1
source installations/sherlock/modules.sh
echo "mpicxx: $(which mpicxx)"; mpirun --version | head -1

# CMake cannot switch compilers in place; keep the old build for reference.
if [ -d microhh/build ] && grep -q 'openmpi/5' microhh/build/CMakeCache.txt; then
    rm -rf microhh/build_ompi505
    mv microhh/build microhh/build_ompi505
fi
# Always configure fresh: CMake keeps CMAKE_<LANG>_FLAGS_RELEASE in its cache,
# so a changed -march in sherlock.cmake is otherwise silently ignored.
rm -rf microhh/build
make install-microhh-sherlock JOBS=8

echo "=== linkage"
ldd microhh/build/microhh | grep -E "libmpi\.|open-rte|open-pal|pnetcdf"
if ldd microhh/build/microhh | grep -q 'openmpi/5'; then
    echo "FAIL: still links OpenMPI 5"; exit 1
fi

echo "=== 4-rank launch check"
d=$(mktemp -d -p "$SCRATCH")
cd "$d"
# No case files: a working MPI launch reaches MicroHH's own argument/ini
# error within seconds; a broken one hangs until the timeout.
set +e
timeout 60 mpirun -n 4 "$HOME/EnforceFlux/microhh/build/microhh" init nocase > launch.log 2>&1
rc=$?
set -e
tail -5 launch.log
if [ "$rc" = 124 ]; then echo "FAIL: mpirun hung"; exit 1; fi
echo "launch check ok (rc=$rc)"
