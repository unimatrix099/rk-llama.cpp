#!/usr/bin/env bash
# Holdout: MTP draft output vs no-draft output, new prompts, 256 tokens, same build.
# usage: holdout.sh <drafter.gguf> <n_max> [extra server args]
ulimit -n 65536; cd ~/rk-llama.cpp-rebase
DFT=$1; N=$2; shift 2
OUT=/tmp/holdout; rm -rf $OUT; mkdir -p $OUT
run() { # $1=label $2..=server args
    local label=$1; shift
    taskset -c 4-7 build/bin/llama-server -m ~/models/gemma-4-E4B-it-Q4_0-ggmlorg.gguf -t 4 -c 4096 --port 8096 --temp 0 "$@" > $OUT/$label-server.txt 2>&1 & local P=$!
    for i in $(seq 120); do curl -sf localhost:8096/health >/dev/null && break; sleep 1; done
    python3 - $OUT/$label <<'PY'
import json, urllib.request, sys
P=["Write a C function that reverses a singly linked list, with comments.",
   "What is 17 * 23 + 144 / 12? Show the steps.",
   "Erkläre auf Deutsch, wie ein Elektromotor funktioniert.",
   "Return a JSON object describing three fictional books with title, author and year.",
   "Write a short story about a lighthouse keeper who finds a message in a bottle.",
   "List ten tips for improving sleep quality, one line each."]
res=[]
for p in P:
    body=json.dumps({"messages":[{"role":"user","content":p}],"max_tokens":256,"temperature":0,"seed":1}).encode()
    r=json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:8096/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=900))
    t=r["timings"]; res.append({"text":r["choices"][0]["message"]["content"],"tps":t["predicted_per_second"],"n":t["predicted_n"]})
json.dump(res, open(sys.argv[1]+".json","w"))
PY
    kill -INT $P; wait $P 2>/dev/null
}
run base
run mtp -md "$DFT" --spec-type draft-mtp --spec-draft-n-max "$N" "$@"
python3 - <<'PY'
import json
b=json.load(open("/tmp/holdout/base.json")); m=json.load(open("/tmp/holdout/mtp.json"))
same=sum(x["text"]==y["text"] for x,y in zip(b,m))
print("holdout identical %d / %d; base %.2f t/s, mtp %.2f t/s; tokens %s" % (same, len(b), sum(x["tps"] for x in b)/len(b), sum(x["tps"] for x in m)/len(m), [x["n"] for x in m]))
PY
