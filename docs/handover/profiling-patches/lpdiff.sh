#!/usr/bin/env bash
# Per-token top logprobs, no draft vs MTP, one prompt; prints where they first differ
ulimit -n 65536; cd ~/rk-llama.cpp-rebase
PROMPT=${PROMPT:-"Write a short story about a lighthouse keeper who finds a message in a bottle."}
NTOK=${NTOK:-256}
run() { local label=$1; shift
    env $ENVS taskset -c 4-7 build/bin/llama-server -m ~/models/gemma-4-E4B-it-Q4_0-ggmlorg.gguf -t 4 -c 4096 --port 8097 --temp 0 "$@" > /tmp/lp-$label-server.txt 2>&1 & local P=$!
    for i in $(seq 120); do curl -sf localhost:8097/health >/dev/null && break; sleep 1; done
    python3 - "$label" "$PROMPT" "$NTOK" <<'PY'
import json, urllib.request, sys
label, prompt, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
body=json.dumps({"messages":[{"role":"user","content":prompt}],"max_tokens":n,"temperature":0,"seed":1,"logprobs":True,"top_logprobs":3}).encode()
r=json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:8097/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=900))
json.dump(r, open(f"/tmp/lp-{label}.json","w"))
PY
    kill -INT $P; wait $P 2>/dev/null
}
run base
run mtp -md ~/models/gemma-4-E4B-it-assistant-centroid-F16.gguf --spec-type draft-mtp --spec-draft-n-max ${NMAX:-3} $EXTRA
python3 - <<'PY'
import json
def toks(f):
    r=json.load(open(f)); c=r["choices"][0]
    lp=(c.get("logprobs") or {}).get("content") or []
    return [(t["token"], t["logprob"], [(x["token"],x["logprob"]) for x in t.get("top_logprobs",[])]) for t in lp]
a=toks("/tmp/lp-base.json"); b=toks("/tmp/lp-mtp.json")
print("tokens with logprobs: base", len(a), "mtp", len(b))
first_lp=None
for i,(x,y) in enumerate(zip(a,b)):
    if first_lp is None and (x[1]!=y[1] or x[2]!=y[2]): first_lp=i
    if x[0]!=y[0]:
        print("first token difference at", i)
        for j in range(max(0,i-2), i+1):
            print("  base", j, a[j][0].encode(), a[j][2]); print("  mtp ", j, b[j][0].encode(), b[j][2])
        break
print("first logprob difference at", first_lp)
if first_lp is not None:
    print("  base", a[first_lp][2]); print("  mtp ", b[first_lp][2])
PY
