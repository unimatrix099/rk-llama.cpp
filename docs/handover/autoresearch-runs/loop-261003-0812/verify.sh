#!/usr/bin/env bash
# Verify: sync the current commit to the board, rebuild, print E4B pure-NPU W4A4 tg64 (t/s).
set -euo pipefail
B=pi@192.168.0.178; SSH="ssh -o BatchMode=yes $B"
cd /workspace
BASE=$($SSH 'cd ~/rk-llama.cpp-rebase && git rev-parse HEAD')
git bundle create /tmp/ar.bundle "$BASE"..HEAD >/dev/null 2>&1 || true
if [ "$(git rev-parse HEAD)" != "$BASE" ]; then
  scp -q /tmp/ar.bundle $B:~/ar.bundle
  $SSH 'cd ~/rk-llama.cpp-rebase && git fetch -q ~/ar.bundle HEAD && git checkout -q --detach FETCH_HEAD'
fi
$SSH 'cd ~/rk-llama.cpp-rebase && cmake --build build -j8 --target llama-bench llama-perplexity llama-server >/tmp/ar-build.txt 2>&1' || { echo "build failed" >&2; exit 2; }
$SSH 'ulimit -n 65536; cd ~/rk-llama.cpp-rebase && taskset -c 4-7 build/bin/llama-bench -m ~/models/gemma-4-E4B-it-Q4_0-ggmlorg.gguf -p 0 -n 64 -r 3 -t 4 -o json 2>/dev/null' \
  | python3 -c 'import sys,json; print(round(json.load(sys.stdin)[0]["avg_ts"],3))'
