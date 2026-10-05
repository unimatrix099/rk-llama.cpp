#!/usr/bin/env bash
# Greedy decode identity check vs reference texts (pre-loop build 856ddb3ac, 2026-10-02).
ulimit -n 65536; cd ~/rk-llama.cpp-rebase
REF=~/bench-logs/2026-10-02-nthreads/mtp
ulimit -c unlimited; build/bin/llama-server -m ~/models/gemma-4-E4B-it-Q4_0-ggmlorg.gguf -t 4 -c 4096 --port 8095 --temp 0 > /tmp/dc-server.txt 2>&1 & P=$!
for i in $(seq 120); do curl -sf localhost:8095/health >/dev/null && break; sleep 1; done
python3 - <<"PY"
import json, urllib.request, sys
P=["Explain how a refrigerator works, step by step.","Write a Python function that returns the n-th Fibonacci number, with a docstring.","Summarize the causes of the First World War in five bullet points.","Translate to French: The meeting is moved to Thursday at ten, please bring the report."]
bad=0
for i,p in enumerate(P):
    ref=json.load(open(f"/home/pi/bench-logs/2026-10-02-nthreads/mtp/npu-nthr-base-p{i}.json"))["text"]
    body=json.dumps({"messages":[{"role":"user","content":p}],"max_tokens":128,"temperature":0,"seed":1}).encode()
    r=json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:8095/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=900))
    if r["choices"][0]["message"]["content"]!=ref: bad+=1
print("decode-identical", 4-bad, "/ 4"); sys.exit(1 if bad else 0)
PY
rc=$?; kill -INT $P; wait $P; exit $rc
