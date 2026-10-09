#!/usr/bin/env bash
# Speculative-decoding experiment matrix for RK3588 (RKNPU2-decode-research.md
# avenue #1 — "one evening of experiments, do this first").
#
# Decode is memory-bandwidth bound: one target-model weight pass verifies
# several draft tokens, so t/s multiplies by the mean accepted length IF the
# draft is cheap enough not to steal the same bus. This script measures that
# directly on the board. Run from the repo root on the Orange Pi:
#
#   ulimit -n 65536
#   sudo scripts/fix_freq_rk3588.sh   # pin clocks first (from rknn-llm repo)
#   bash docs/backend/rknpu2-spec-decode-bench.sh 2>&1 | tee spec-decode-results.log
#
# Draft models this expects in ~/models (skips pairs whose files are missing):
#   qwen2.5-0.5b-instruct-q8_0.gguf   (draft for qwen2.5-1.5b — same family)
#   gemma-4-E2B-it-Q4_0.gguf          (draft for gemma-4-E4B — same family)
# Same-family pairs give the best acceptance; a mismatched tokenizer aborts.
#
# Three decode routings are compared for each pair:
#   cpu   : all matmuls on CPU (RKNPU_CPU_DECODE=999999 routes every M below
#           the threshold to CPU — prefill too, so pp numbers are not
#           meaningful in this config, only tg)
#   route : shipped hybrid — NPU prefill, CPU decode (RKNPU_CPU_DECODE=32)
#   npu   : NPU decode (no routing)
# Both models share one process, so the env applies to draft AND target;
# per-model routing would need backend work — if `route` wins here, that is
# the motivation to build it.
#
# Measured pitfalls baked into this script (2026-08-10 board session):
#   - NEVER taskset a two-model process onto 4 cores: target and draft each
#     spin a ggml threadpool, and 8 spinning threads on 4 cores cost 2.6x
#     (independent of OMP_NUM_THREADS). Leave placement to the scheduler.
#   - llama-cli busy-spins on its interactive "> " prompt at EOF stdin
#     (100% of one core, forever) — baselines use llama-bench instead.
#
# Also runs llama-server's DRAFT-FREE ngram self-speculation (--spec-type
# ngram-simple): zero extra weight traffic, acceptance depends on text
# repetitiveness. Free win if it holds on real workloads.

set -u
cd "$(dirname "$0")/../.."

BIN=build/bin
MODELS=${MODELS:-$HOME/models}
NGEN=${NGEN:-128}
PROMPT=${PROMPT:-"Write a detailed step-by-step tutorial explaining how to set up a small vegetable garden on a balcony, including soil, containers, watering schedule and common mistakes."}

run_one() { # label, extra env as "K=V K=V", cmd...
    local label=$1 env=$2; shift 2
    echo "### $label"
    echo "+ env $env $*"
    env $env "$@" 2>&1 | grep -Ei "draft|accept|decoded|eval time|tokens per second|generation|total time" | sed 's/^/    /'
    echo
}

pair() { # target draft
    local tgt=$MODELS/$1 dft=$MODELS/$2
    [ -f "$tgt" ] || { echo "SKIP: missing $tgt"; return; }
    [ -f "$dft" ] || { echo "SKIP: missing $dft (download the draft model)"; return; }
    echo "=============================================================="
    echo "== target=$1 draft=$2"
    echo "=============================================================="

    for routing in "cpu RKNPU_CPU_DECODE=999999" "route RKNPU_CPU_DECODE=32" "npu IGNORED="; do
        set -- $routing; local rname=$1 renv=$2
        [ "$renv" = "IGNORED=" ] && renv=""

        # no-speculation baseline (llama-bench: llama-cli spins at EOF stdin)
        run_one "$rname / baseline (no draft)" "$renv" \
            $BIN/llama-bench -m "$tgt" -p 0 -n $NGEN -r 2 -t 4

        for dmax in 4 8 16; do
            run_one "$rname / draft-max=$dmax" "$renv" \
                $BIN/llama-speculative -m "$tgt" -md "$dft" -t 4 \
                    -p "$PROMPT" -n $NGEN --draft-max $dmax -s 42
        done
    done
}

ngram_selfspec() { # target — draft-free ngram speculation via llama-server
    local tgt=$MODELS/$1
    [ -f "$tgt" ] || { echo "SKIP: missing $tgt"; return; }
    [ -x $BIN/llama-server ] || { echo "SKIP: llama-server not built"; return; }
    echo "=============================================================="
    echo "== target=$1  ngram self-speculation (no draft model)"
    echo "=============================================================="
    for spec in none ngram-simple ngram-map-k; do
        local args=""
        [ "$spec" != "none" ] && args="--spec-type $spec --draft-max 16"
        echo "### spec=$spec (RKNPU_CPU_DECODE=32)"
        RKNPU_CPU_DECODE=32 $BIN/llama-server -m "$tgt" $args --port 18088 &>/tmp/spec-server.log &
        local pid=$!
        for i in $(seq 60); do curl -fs localhost:18088/health &>/dev/null && break; sleep 1; done
        curl -fs localhost:18088/completion -d "{\"prompt\": $(printf '%s' "$PROMPT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'), \"n_predict\": $NGEN}" \
            | python3 -c 'import json,sys; t=json.load(sys.stdin)["timings"]; print(f"    predicted: {t[\"predicted_per_second\"]:.2f} t/s  prompt: {t[\"prompt_per_second\"]:.2f} t/s")' \
            || echo "    request failed (see /tmp/spec-server.log)"
        kill $pid; wait $pid 2>/dev/null
        echo
    done
}

echo "spec-decode matrix: NGEN=$NGEN  models=$MODELS  $(date -Is)"
echo

pair qwen2.5-1.5b-instruct-q8_0.gguf qwen2.5-0.5b-instruct-q8_0.gguf
pair gemma-4-E4B-it-Q4_0.gguf        gemma-4-E2B-it-Q4_0.gguf

ngram_selfspec qwen2.5-1.5b-instruct-q8_0.gguf
ngram_selfspec gemma-4-E4B-it-Q4_0.gguf

echo "Done. Read acceptance rates alongside t/s: high draft-max with low"
echo "acceptance burns bandwidth on rejected drafts."
