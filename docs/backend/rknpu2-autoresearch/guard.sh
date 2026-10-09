#!/usr/bin/env bash
# Guard: backend unit tests (local) + bit-identical E4B W4A4 8-chunk PPL on the board.
set -euo pipefail
cd /workspace/docs/backend
for t in check-prep check-dispatch check; do make -s -f Makefile.rknpu2-tools $t 2>&1 | grep -q "0 failures" || { echo "unit $t failed"; exit 1; }; done
P=$(ssh -o BatchMode=yes pi@192.168.0.178 'ulimit -n 65536; cd ~/rk-llama.cpp-rebase && build/bin/llama-perplexity -m ~/models/gemma-4-E4B-it-Q4_0-ggmlorg.gguf -f ~/wikitext-2-raw/wiki.test.raw --chunks 8 -t 4 2>&1 | grep -oE "Final estimate: PPL = [0-9.]+" | grep -oE "[0-9.]+$"')
echo "ppl=$P"; [ "$P" = "35.0480" ]
# decode path (M=1) is not exercised by perplexity: greedy text must match the pre-loop reference
ssh -o BatchMode=yes pi@192.168.0.178 'bash ~/rk-llama.cpp-rebase/docs/backend/rknpu2-autoresearch/decode-check.sh'
