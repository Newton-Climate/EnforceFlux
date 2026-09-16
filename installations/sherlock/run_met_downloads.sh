#!/usr/bin/env bash
# Run `enforceflux met` for each config in turn, logging per config.
# Meant to run inside a detached screen session on a login node: the work is
# CDS queue time plus network I/O, not CPU.
#
#   screen -dmS era5_met bash installations/sherlock/run_met_downloads.sh CFG...
#   screen -r era5_met            # attach; Ctrl-a d to detach again
#   tail -f runs/met_download_logs/*.log
set -uo pipefail

cd "$(dirname "$0")/../.."
. installations/sherlock/modules.sh >/dev/null 2>&1
. .venv/bin/activate

logdir=runs/met_download_logs
mkdir -p "$logdir"
status=0
for cfg in "$@"; do
    name=$(basename "$cfg" .yaml)
    echo "[$(date '+%F %T')] start $cfg" | tee -a "$logdir/summary.log"
    if enforceflux met --config "$cfg" >"$logdir/$name.log" 2>&1; then
        echo "[$(date '+%F %T')] done  $cfg" | tee -a "$logdir/summary.log"
    else
        echo "[$(date '+%F %T')] FAIL  $cfg (see $logdir/$name.log)" | tee -a "$logdir/summary.log"
        status=1
    fi
done
echo "[$(date '+%F %T')] all finished, status=$status" | tee -a "$logdir/summary.log"
exit $status
