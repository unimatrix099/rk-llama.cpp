#!/usr/bin/env bash
# Thread-count / core-affinity sweep for CPU-routed decode on RK3588
# (RKNPU2-decode-research.md avenue #5 — micro-tuning, one afternoon).
#
# All shipped benchmarks used default threading. On A55+A76 hybrids the
# scheduler's placement of GEMV threads can swing decode ±10%: little cores
# add bandwidth but can also stall the critical path. This sweeps the
# combinations with llama-bench so the answer lands in one table.
#
# Run from the repo root on the Orange Pi (pin clocks first):
#   ulimit -n 65536
#   bash docs/backend/rknpu2-affinity-sweep.sh 2>&1 | tee affinity-results.log
#
# RK3588 topology: cpu0-3 = Cortex-A55 (little), cpu4-7 = Cortex-A76 (big).

set -u
cd "$(dirname "$0")/../.."

BIN=build/bin
MODELS=${MODELS:-$HOME/models}
BENCH_ARGS=${BENCH_ARGS:--p 128 -n 64 -r 3}

sweep() { # model, env
    local model=$MODELS/$1 env=$2
    [ -f "$model" ] || { echo "SKIP: missing $model"; return; }
    echo "=============================================================="
    echo "== $1   env: ${env:-<none>}"
    echo "=============================================================="
    # taskset restricts every thread (incl. OpenMP prep workers and the
    # rknpu backend's dispatch threads), -t sets llama's compute threads.
    for cfg in \
        "all-8:-t 8:0-7" \
        "big-4:-t 4:4-7" \
        "big-4-on-all:-t 4:0-7" \
        "6-threads:-t 6:0-7" \
        "big+2little:-t 6:2-7" \
        ; do
        IFS=: read -r name targ cores <<< "$cfg"
        echo "--- $name (threads: ${targ#-t }, cores: $cores)"
        env $env taskset -c $cores $BIN/llama-bench -m "$model" $BENCH_ARGS $targ 2>/dev/null \
            | grep -E "pp128|tg64|t/s" | sed 's/^/    /'
        echo
    done
}

echo "affinity sweep: $BENCH_ARGS  $(date -Is)"
echo

# The configs that matter for decode:
# 1. shipped hybrid routing (NPU prefill / CPU decode) — the tg rows are
#    the CPU GEMV path this sweep is really tuning
sweep gemma-4-E4B-it-Q4_0.gguf "RKNPU_CPU_DECODE=32"
sweep gemma-4-E4B-it-Q4_0.gguf "RKNPU_HYBRID=W8A8_STANDARD RKNPU_CPU_DECODE=32"

# 2. NPU decode — sensitive to where the 3 dispatch threads land
sweep gemma-4-E4B-it-Q4_0.gguf ""

# 3. the MoE regression case from the optimization notes (LFM2 NPU decode
#    beat CPU; check whether affinity closes any of the routed-config gap)
sweep LFM2-8B-A1B-Q8_0.gguf ""
sweep LFM2-8B-A1B-Q8_0.gguf "RKNPU_CPU_DECODE=32"

sweep qwen2.5-1.5b-instruct-q8_0.gguf "RKNPU_CPU_DECODE=32"

echo "Done. Re-tune RKNPU_CPU_DECODE thresholds separately once the best"
echo "affinity is known (the two interact)."
