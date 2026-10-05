#!/usr/bin/env bash
# Tolerant guard (prefill loop): unit tests + 8-chunk PPL within 0.1% of 35.0480
# + llama-server answers all decode prompts, 3 runs, no crash. Identity is reported, not required.
set -euo pipefail
cd /workspace/docs/backend
for t in check-prep check-dispatch check; do make -s -f Makefile.rknpu2-tools $t 2>&1 | grep -q "0 failures" || { echo "unit $t failed"; exit 1; }; done
P=$(ssh -o BatchMode=yes pi@192.168.0.178 'ulimit -n 65536; cd ~/rk-llama.cpp-rebase && build/bin/llama-perplexity -m ~/models/gemma-4-E4B-it-Q4_0-ggmlorg.gguf -f ~/wikitext-2-raw/wiki.test.raw --chunks 8 -t 4 2>&1 | grep -oE "Final estimate: PPL = [0-9.]+" | grep -oE "[0-9.]+$"')
echo "ppl=$P"
python3 -c "import sys; p=float('$P'); sys.exit(0 if abs(p-35.0480)/35.0480 <= 0.001 else 1)" || { echo "ppl out of tolerance"; exit 1; }
for i in 1 2 3; do
  out=$(ssh -o BatchMode=yes pi@192.168.0.178 'bash ~/rk-llama.cpp-rebase/docs/backend/rknpu2-autoresearch/decode-check.sh 2>&1 | tail -1'; true)
  echo "server run $i: $out"
  echo "$out" | grep -q "decode-identical" || { echo "server failed"; exit 1; }
done
