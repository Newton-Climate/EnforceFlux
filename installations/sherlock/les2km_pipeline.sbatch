#!/bin/bash
# 2 km rice-paddy LES pipeline on Sherlock:
#   1. donor spinup run (writes the t=3600 restart)
#   2. one nature run warm-started from it (operator validation truth)
#   3. tagged-tracer operator: build, run, compare
#
# Rank count comes from SLURM_NTASKS; each stage overrides num_workers.
# Submit with e.g.:  sbatch -n 64 installations/sherlock/les2km_pipeline.sbatch
# STAGES (env) selects a subset, e.g. STAGES="tagged_run compare".
#SBATCH -J efx_les2km
#SBATCH -p serc,normal
#SBATCH -N 1
#SBATCH --mem=240G
#SBATCH -t 3-00:00:00
#SBATCH -o /home/users/nnewton/efx_logs/les2km_%j.log

set -eo pipefail
cd "$HOME/EnforceFlux" || exit 1
source installations/sherlock/modules.sh >/dev/null 2>&1
export MPLBACKEND=Agg OMP_NUM_THREADS=1
N="${SLURM_NTASKS:-4}"
CFG_DIR=configs/hetero_rice_paddy_2km
DONOR_CFG="${DONOR_CFG:-$CFG_DIR/les2km_l200_cv2p0_wind3.yaml}"
NATURE_CFG="${NATURE_CFG:-$CFG_DIR/les2km_nature_l500_cv1p0_s1_wind3.yaml}"
STAGES="${STAGES:-donor nature tagged_build tagged_run compare}"
echo "node $(hostname) ranks=$N stages=[$STAGES] $(date)"

# Write a copy of a config with num_workers set to this job's rank count.
with_ranks() {
    local src="$1" dst="${1%.yaml}._n${N}.yaml"
    sed -E "s/^(\s*num_workers:)\s*[0-9]+/\1 ${N}/" "$src" > "$dst"
    echo "$dst"
}

# The tagged operator follows whichever donor config was chosen (80 m or 40 m).
EFX_TAGGED_RUN=$(sed -nE 's/^  name: ([^ #]+).*/\1/p' "$DONOR_CFG" | head -1)
export EFX_TAGGED_RUN
# Template = the nature run, not the donor: the tagged case inherits the
# template's endtime, and the donor stops at 5400 s while the nature run's
# observation window runs 3600-6300 s.
EFX_TAGGED_TEMPLATE=$(sed -nE 's/^  name: ([^ #]+).*/\1/p' "$NATURE_CFG" | head -1)
export EFX_TAGGED_TEMPLATE
export EFX_TAGGED_DONOR_T=3600
export EFX_TAGGED_WORKERS="$N"
# Nature runs share the donor's prefix up to the L/cv label; the donor itself
# is skipped by compare because it starts at t=0.
export EFX_TAGGED_NATURE_GLOB="${EFX_TAGGED_RUN%%_l[0-9]*}_*_surface"
echo "tagged run=$EFX_TAGGED_RUN nature glob=$EFX_TAGGED_NATURE_GLOB"
TAGGED=notebooks/hetero_experiments/run_les_tagged_operator.py

for s in $STAGES; do
    echo "=== $s $(date)"
    case "$s" in
        donor)        .venv/bin/enforceflux dispersion --config "$(with_ranks "$DONOR_CFG")" ;;
        nature)       .venv/bin/enforceflux dispersion --config "$(with_ranks "$NATURE_CFG")" ;;
        tagged_build) .venv/bin/python "$TAGGED" build ;;
        tagged_bench)
            # Time BENCH_S simulated seconds of the real tagged case to project
            # the full run, then discard it so tagged_build starts clean.
            out="runs/${EFX_TAGGED_RUN}_tagged_operator"
            case "$out" in runs/source_heterogeneity_*_tagged_operator) ;;
                *) echo "refusing bench on '$out'"; exit 2 ;; esac
            .venv/bin/python "$TAGGED" build
            ini=$(ls "$out"/microhh_case/*.ini)
            sed -i -E "s/^endtime=.*/endtime=$(( EFX_TAGGED_DONOR_T + ${BENCH_S:-60} ))/" "$ini"
            .venv/bin/python "$TAGGED" run
            echo "BENCH ranks=$N tracers=$(grep -m1 '^slist=' "$ini" | tr ',' '\n' | wc -l)"
            awk 'NR>2 && $1-p==50 {print "BENCH_BLOCK_S", $3, "DT", $4} {p=$1}' "$out"/microhh_case/*.out | tail -5
            rm -rf "$out" ;;
        tagged_run)   .venv/bin/python "$TAGGED" run ;;
        compare)      .venv/bin/python "$TAGGED" compare ;;
        *) echo "unknown stage $s"; exit 2 ;;
    esac
done
echo "done $(date)"
