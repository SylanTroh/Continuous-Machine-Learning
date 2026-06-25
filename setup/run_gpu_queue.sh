#!/bin/sh
# Run GPU experiments one at a time from a queue file.
#
# Mandatory flags: OMP_NUM_THREADS=1, threads=1
# Usage:  setup/run_gpu_queue.sh [queue-file]   (default setup/gpu_queue.txt)
set -u
cd "$(dirname "$0")/.."

queue=${1:-setup/gpu_queue.txt}
POLL=${POLL:-10}
log="${queue}.log"

if [ ! -f "$queue" ]; then
    cat > "$queue" <<'EOF'
# GPU experiment queue. One job per line: <experiment> [cells...]
# Edit anytime while run_gpu_queue.sh is running. '#' lines are ignored.
EOF
    echo "[$(date '+%H:%M:%S')] created $queue, add jobs to it; runner is watching"
fi

waiting=0
while :; do
    if [ ! -f "$queue" ]; then
        echo "[$(date '+%H:%M:%S')] $queue removed, stopping"
        break
    fi

    # Pop the first non-blank, non-comment line; write the rest to tmp in one awk pass.
    tmp="${queue}.tmp.$$"
    : > "$tmp"
    job=$(awk -v tmp="$tmp" '
        popped==0 && NF && $1 !~ /^#/ { print; popped=1; next }
        { print >> tmp }
    ' "$queue")

    if [ -z "$job" ]; then
        rm -f "$tmp"
        if [ "$waiting" -eq 0 ]; then
            echo "[$(date '+%H:%M:%S')] queue empty, waiting (add jobs, or rm $queue to stop)"
            waiting=1
        fi
        sleep "$POLL"
        continue
    fi
    mv "$tmp" "$queue"          # commit the pop
    waiting=0

    set -- $job
    exp=$1
    shift
    script="experiments/${exp}.jl"
    if [ ! -f "$script" ]; then
        echo "[$(date '+%H:%M:%S')] SKIP $exp: $script not found" >&2
        echo "$(date '+%Y-%m-%d %H:%M:%S')  SKIP(missing)  $job" >> "$log"
        continue
    fi

    echo "[$(date '+%H:%M:%S')] START $exp $*"
    start=$(date +%s)
    OMP_NUM_THREADS=1 julia --project=. --threads=1 "$script" "$@"
    rc=$?
    dur=$(( $(date +%s) - start ))
    if [ "$rc" -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] DONE  $exp  (${dur}s)"
        echo "$(date '+%Y-%m-%d %H:%M:%S')  DONE  (${dur}s)  $job" >> "$log"
    else
        echo "[$(date '+%H:%M:%S')] FAIL  $exp  (rc=$rc, ${dur}s)" >&2
        echo "$(date '+%Y-%m-%d %H:%M:%S')  FAIL(rc=$rc)  (${dur}s)  $job" >> "$log"
    fi
done
