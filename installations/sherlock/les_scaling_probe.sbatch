#!/bin/bash
# MPI strong-scaling probe for the 2 km rice-paddy LES on Sherlock.
# Runs each _sherlock_probe_*.yaml (120 s, one scalar) and prints wall time.
#SBATCH -J efx_les_probe
#SBATCH -p serc,normal
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --mem=128G
#SBATCH -t 01:30:00
#SBATCH -o /home/users/nnewton/efx_logs/les_probe_%j.log

set -o pipefail
cd "$HOME/EnforceFlux" || exit 1
source installations/sherlock/modules.sh >/dev/null 2>&1
export MPLBACKEND=Agg OMP_NUM_THREADS=1
echo "node $(hostname)  cpus=$SLURM_CPUS_ON_NODE"

# PROBE_GLOB narrows the set, e.g. PROBE_GLOB='dx40_*'.
for cfg in configs/hetero_rice_paddy_2km/_sherlock_probe_${PROBE_GLOB:-*}.yaml; do
    # Probe outputs are disposable; a stale case dir makes `init` refuse to run.
    # Guarded: only ever delete a sherlock_probe_* dir, never runs/ itself.
    run=$(sed -nE 's/^  name: ([^ #]+).*/\1/p' "$cfg" | head -1)
    case "$run" in
        sherlock_probe_?*) rm -rf "runs/$run" ;;
        *) echo "refusing to clean run dir for '$cfg' (name='$run')"; continue ;;
    esac
    t0=$(date +%s)
    .venv/bin/enforceflux dispersion --config "$cfg" > "${cfg%.yaml}.log" 2>&1
    rc=$?
    echo "PROBE $(basename "$cfg") exit=$rc wall_s=$(( $(date +%s) - t0 ))"
done
