#!/usr/bin/env bash
# Production limit tests for Gemma-4 E4B Q4_0 on RK3588 (RKNPU2 W4A4). Run on the board from the repo root.
set -u
ulimit -n 65536
OUT=${OUT:-$HOME/bench-logs/2026-10-09-prod}; mkdir -p "$OUT"
M=$HOME/models/gemma-4-E4B-it-Q4_0-ggmlorg.gguf
DFT=$HOME/models/gemma-4-E4B-it-assistant-centroid-F16.gguf
W=$HOME/wikitext-2-raw/wiki.test.raw
BIN=build/bin
PIN="taskset -c 4-7"
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/progress.txt"; }
STAGES=${STAGES:-"pp tg server ppl"}

# 1. prefill speed vs prompt size (llama-bench, NPU W4A4)
if [[ " $STAGES " == *" pp "* ]]; then
  for p in 128 512 1024 2048 4096 8192 16384 32768; do
    r=2; [ $p -ge 8192 ] && r=1
    log "pp$p"
    timeout 3600 $PIN $BIN/llama-bench -m $M -p $p -n 0 -r $r -t 4 -o json > "$OUT/pp$p.json" 2> "$OUT/pp$p.err"
    echo "rc=$?" >> "$OUT/pp$p.err"; free -m | sed -n 2p >> "$OUT/pp$p.err"; sleep 5
  done
fi

# 2. decode speed vs context depth (no draft)
if [[ " $STAGES " == *" tg "* ]]; then
  for d in 0 1024 2048 4096 8192 16384 32000; do
    log "tg32 @ depth $d"
    timeout 5400 $PIN $BIN/llama-bench -m $M -p 0 -n 32 -d $d -r 1 -t 4 -o json > "$OUT/tg-d$d.json" 2> "$OUT/tg-d$d.err"
    echo "rc=$?" >> "$OUT/tg-d$d.err"; sleep 5
  done
fi

# 3. server: real long prompts with a needle, no draft vs MTP; speed, recall, identity
if [[ " $STAGES " == *" server "* ]]; then
  python3 - "$W" "$OUT" <<'PY'
import sys, json, random
text = open(sys.argv[1], encoding="utf-8").read()
random.seed(7)
for n in (1000, 4000, 8000, 16000, 30000):
    chars = int(n * 3.6)            # ~3.6 chars per token on wikitext with this vocab
    start = random.randrange(0, len(text) - chars - 1)
    body = text[start:start + chars]
    code = "".join(random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(4)) + "-" + str(random.randrange(1000, 9999))
    mid = len(body) // 2
    body = body[:mid] + f"\n\nIMPORTANT NOTE: the secret access code is {code}.\n\n" + body[mid:]
    json.dump({"n": n, "code": code, "text": body}, open(f"{sys.argv[2]}/prompt{n}.json", "w"))
PY
  serve() { # $1=label, rest=args
    local label=$1; shift
    env RKNPU_EXCLUDE_TYPES=f16 $PIN $BIN/llama-server -m $M -t 4 -c 32768 --port 8099 --temp 0 --reasoning off "$@" > "$OUT/server-$label.log" 2>&1 &
    SPID=$!
    for i in $(seq 300); do curl -sf localhost:8099/health >/dev/null && return 0; sleep 1; done
    log "server $label failed to start"; return 1
  }
  ask() { # $1=label
    python3 - "$1" "$OUT" <<'PY'
import sys, json, urllib.request, time
label, out = sys.argv[1], sys.argv[2]
res = []
for n in (1000, 4000, 8000, 16000, 30000):
    p = json.load(open(f"{out}/prompt{n}.json"))
    for task, q, mt in (("needle", "What is the secret access code mentioned in the text above? Answer with the code only.", 32),
                        ("summary", "Summarize the text above in five bullet points.", 256)):
        msg = p["text"] + "\n\n" + q
        body = json.dumps({"messages": [{"role": "user", "content": msg}], "max_tokens": mt, "temperature": 0, "seed": 1, "cache_prompt": False}).encode()
        t0 = time.time()
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:8099/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=7200))
        except Exception as e:
            res.append({"n": n, "task": task, "error": str(e)}); continue
        t = r["timings"]; txt = r["choices"][0]["message"]["content"]
        res.append({"n": n, "task": task, "prompt_n": t["prompt_n"], "prompt_tps": t["prompt_per_second"], "gen_n": t["predicted_n"],
                    "gen_tps": t["predicted_per_second"], "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"),
                    "wall_s": round(time.time() - t0, 1), "needle_ok": (p["code"] in txt) if task == "needle" else None, "text": txt})
        print(label, n, task, res[-1]["prompt_n"], round(res[-1]["prompt_tps"], 1), round(res[-1]["gen_tps"], 2), res[-1]["needle_ok"], flush=True)
json.dump(res, open(f"{out}/server-{label}.json", "w"), indent=1)
PY
  }
  log "server base"; serve base && { ask base >> "$OUT/progress.txt" 2>&1; kill -INT $SPID; wait $SPID; }; sleep 10
  log "server mtp"; serve mtp -md $DFT --spec-type draft-mtp --spec-draft-n-max 5 --spec-draft-p-min 0.6 && { ask mtp >> "$OUT/progress.txt" 2>&1; kill -INT $SPID; wait $SPID; }; sleep 10
fi

# 4. accuracy: NPU vs CPU perplexity at several context sizes; KLD at 2k
if [[ " $STAGES " == *" ppl "* ]]; then
  CPUENV="RKNPU_HYBRID=W8A8_STANDARD RKNPU_CPU_DECODE=999999"
  for cfg in "2048 8" "8192 2" "32768 1"; do
    set -- $cfg; c=$1; ch=$2
    log "ppl c=$c chunks=$ch NPU"
    timeout 7200 $PIN $BIN/llama-perplexity -m $M -f $W -c $c --chunks $ch -t 4 > "$OUT/ppl-npu-c$c.txt" 2>&1
    log "ppl c=$c chunks=$ch CPU"
    if [ $c = 2048 ]; then KB="--kl-divergence-base $OUT/kld-cpu-c2048.bin"; else KB=""; fi
    timeout 10800 env $CPUENV $PIN $BIN/llama-perplexity -m $M -f $W -c $c --chunks $ch -t 4 $KB > "$OUT/ppl-cpu-c$c.txt" 2>&1
  done
  log "kld NPU vs CPU c=2048"
  timeout 7200 $PIN $BIN/llama-perplexity -m $M -f $W -c 2048 --chunks 8 -t 4 --kl-divergence-base $OUT/kld-cpu-c2048.bin --kl-divergence > "$OUT/kld-npu-c2048.txt" 2>&1
  rm -f $OUT/kld-cpu-c2048.bin
fi
log "done"
