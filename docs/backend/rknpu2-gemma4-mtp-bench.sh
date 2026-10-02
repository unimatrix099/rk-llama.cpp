#!/usr/bin/env bash
# Gemma-4 MTP (assistant drafter) on RK3588: regression + speculative-decode sweep.
#
# Run on the board from the repo root after building the rebased tree:
#   bash docs/backend/rknpu2-gemma4-mtp-bench.sh [E4B|E2B]
#
# Phase 1 re-measures the pre-rebase anchors (llama-bench) so the rebase is
# checked before anything new is believed. Phase 2 serves the model with and
# without the drafter and records decode speed + acceptance on fixed prompts,
# greedy, for each backend configuration.

set -euo pipefail
ulimit -n 65536

VARIANT=${1:-E4B}
MODELS=${MODELS:-$HOME/models}
OUT=${OUT:-$HOME/bench-logs/$(date +%F)-gemma4-mtp-$VARIANT}
BIN=build/bin
PORT=8089
N_PREDICT=${N_PREDICT:-128}
DRAFT_NS=${DRAFT_NS:-"1 2 3 4"}
mkdir -p "$OUT"

case $VARIANT in
    E4B) TGT=$MODELS/gemma-4-E4B-it-Q4_0-ggmlorg.gguf ;;
    E2B) TGT=$MODELS/gemma-4-E2B-it-qat-q4_0.gguf ;;
    *) echo "unknown variant $VARIANT"; exit 1 ;;
esac
DFT=$MODELS/gemma-4-$VARIANT-it-assistant-F16.gguf
for f in "$TGT" "$DFT"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done

# backend configurations: name|env
CONFIGS=(
    "cpu|RKNPU_HYBRID=W8A8_STANDARD RKNPU_CPU_DECODE=999999"
    "routed|RKNPU_HYBRID=W8A8_STANDARD RKNPU_CPU_DECODE=32"
    "npu-default|"
)

echo "== phase 1: regression anchors (llama-bench pp128/tg64)" | tee "$OUT/summary.txt"
for c in "${CONFIGS[@]}"; do
    name=${c%%|*}; envs=${c#*|}
    env $envs taskset -c 4-7 $BIN/llama-bench -m "$TGT" -p 128 -n 64 -r 3 -t 4 -o md \
        > "$OUT/bench-$name.md" 2>&1
    echo "-- $name" >> "$OUT/summary.txt"; grep -E "pp128|tg64" "$OUT/bench-$name.md" >> "$OUT/summary.txt" || true
done

PROMPTS=(
    "Explain how a refrigerator works, step by step."
    "Write a Python function that returns the n-th Fibonacci number, with a docstring."
    "Summarize the causes of the First World War in five bullet points."
    "Translate to French: The meeting is moved to Thursday at ten, please bring the report."
)

serve() { # $1=env  $2=extra args
    env $1 $BIN/llama-server -m "$TGT" -t 4 -c 4096 --port $PORT --temp 0 $2 \
        > "$OUT/server.txt" 2>&1 &
    SPID=$!
    for _ in $(seq 120); do curl -sf localhost:$PORT/health >/dev/null && return 0; sleep 1; done
    echo "server failed to start"; tail -20 "$OUT/server.txt"; kill $SPID; return 1
}

run_prompts() { # $1=label
    for i in "${!PROMPTS[@]}"; do
        python3 - "$PORT" "${PROMPTS[$i]}" "$N_PREDICT" > "$OUT/$1-p$i.json" <<'PY'
import json, sys, urllib.request
port, prompt, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                   "max_tokens": n, "temperature": 0, "seed": 1}).encode()
req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", body,
                             {"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=900))
t = r.get("timings", {})
print(json.dumps({"text": r["choices"][0]["message"]["content"],
                  "predicted_per_second": t.get("predicted_per_second"),
                  "predicted_n": t.get("predicted_n"),
                  "draft_n": t.get("draft_n"),
                  "draft_n_accepted": t.get("draft_n_accepted")}))
PY
    done
}

echo "== phase 2: speculative sweep, $N_PREDICT tokens x ${#PROMPTS[@]} prompts, greedy" | tee -a "$OUT/summary.txt"
for c in "${CONFIGS[@]}"; do
    name=${c%%|*}; envs=${c#*|}
    serve "$envs" "" && { run_prompts "$name-base"; kill $SPID; wait $SPID 2>/dev/null || true; }
    for n in $DRAFT_NS; do
        serve "$envs" "-md $DFT --spec-type draft-mtp --spec-draft-n-max $n" \
            && { run_prompts "$name-mtp$n"; kill $SPID; wait $SPID 2>/dev/null || true; }
    done
done

python3 - "$OUT" >> "$OUT/summary.txt" <<'PY'
import glob, json, os, sys, collections
out = sys.argv[1]
rows = collections.defaultdict(list)
for f in sorted(glob.glob(f"{out}/*-p*.json")):
    label = os.path.basename(f).rsplit("-p", 1)[0]
    rows[label].append(json.load(open(f)))
base_text = {}
print(f"{'config':<22}{'t/s':>8}{'accept':>9}{'same-as-base':>14}")
for label, rs in rows.items():
    tps = sum(r["predicted_per_second"] or 0 for r in rs) / len(rs)
    dn = sum(r["draft_n"] or 0 for r in rs); da = sum(r["draft_n_accepted"] or 0 for r in rs)
    cfg = label.rsplit("-", 1)[0]
    if label.endswith("-base"):
        base_text[cfg] = [r["text"] for r in rs]
    same = "n/a" if cfg not in base_text else \
        f"{sum(a == b['text'] for a, b in zip(base_text[cfg], rs))}/{len(rs)}"
    acc = f"{da / dn:.0%}" if dn else "-"
    print(f"{label:<22}{tps:>8.2f}{acc:>9}{same:>14}")
PY
cat "$OUT/summary.txt"
