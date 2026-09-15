#!/bin/bash
# 2 km rice-paddy sweep on Sherlock: H.e nature fields, then inversions.
#
# Prerequisites (see configs/hetero_rice_paddy_2km/):
#   - the tagged operator from les2km_pipeline.sbatch (operator.npz),
#   - bls_met.json (derive_bls_met.py); sweep.py renders every config from it,
#   - the 16 bLS donor runs (*_l100_cv0p5_s0_wind3_*_gp), built where R and
#     bLSmodelR exist and copied into runs/.
# Tags are split across NSHARDS parallel processes; stages resume on rerun.
#SBATCH -J efx_sweep2km
#SBATCH -p serc,normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o /home/users/nnewton/efx_logs/sweep2km_%j.log

set -o pipefail
cd "$HOME/EnforceFlux" || exit 1
source installations/sherlock/modules.sh >/dev/null 2>&1
export PATH="$HOME/EnforceFlux/.venv/bin:$PATH" MPLBACKEND=Agg OMP_NUM_THREADS=1
export MPLCONFIGDIR="$SCRATCH/mplconfig"
NSHARDS="${NSHARDS:-${SLURM_CPUS_PER_TASK:-16}}"
MODES="${MODES:---nature --inversions}"
cd configs/hetero_rice_paddy_2km || exit 1
echo "node $(hostname) shards=$NSHARDS modes=[$MODES] $(date)"

# Nature first for every shard (each shard's inversions need its own fields).
# Shard 0 runs the donor tag first; the others only read its bLS runs, which
# must already be present.
rc=0
for mode in $MODES; do
    pids=()
    for k in $(seq 0 $((NSHARDS - 1))); do
        python sweep.py "$mode" --shard "$k" --nshards "$NSHARDS" \
            > "$HOME/efx_logs/sweep2km_${SLURM_JOB_ID}_${mode#--}_$k.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    echo "=== $mode done $(date) rc=$rc"
    [ "$rc" = 0 ] || { grep -l "Traceback" "$HOME"/efx_logs/sweep2km_${SLURM_JOB_ID}_*.log; exit 1; }
done
echo "done $(date)"
