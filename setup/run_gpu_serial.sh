#!/bin/sh
# Run GPU experiments one at a time.
#
# Mandatory flags: OMP_NUM_THREADS=1 --threads=1
# Usage:  setup/run_gpu_serial.sh "phase_diagram 0 1 2" "paired_solver_test"
#   RESUME=1 passes through to the individual experiments.
set -u
cd "$(dirname "$0")/.."

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <job> [<job> ...]   (each job = quoted 'experiment [cells...]')" >&2
    exit 2
fi

total=$#
failed=""
n=0
for job in "$@"; do
    n=$((n + 1))
    set -- $job
    exp=$1
    shift
    script="experiments/${exp}.jl"
    if [ ! -f "$script" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP $exp: $script not found" >&2
        failed="$failed $exp(missing)"
        continue
    fi
    echo "[$(date '+%H:%M:%S')] START job $n/$total : $exp $*"
    start=$(date +%s)
    OMP_NUM_THREADS=1 julia --project=. --threads=1 "$script" "$@"
    rc=$?
    end=$(date +%s)
    dur=$((end - start))
    if [ "$rc" -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] DONE  $exp  (${dur}s)"
    else
        echo "[$(date '+%H:%M:%S')] FAIL  $exp  (rc=$rc, ${dur}s)" >&2
        failed="$failed $exp(rc=$rc)"
    fi
done

echo "----------------------------------------"
if [ -n "$failed" ]; then
    echo "FAILED:$failed"
    exit 1
fi
echo "all $n job(s) succeeded"
