#!/usr/bin/env bash
# Prefill-loop guard (agreed 2026-10-05, decode research #1d):
#  1. backend unit tests
#  2. quality: 32-chunk PPL <= +2% over the W4A4 baseline (26.8771 -> 27.41),
#     and KLD vs the CPU reference logits (4 chunks) mean <= 0.65, same-top >= 68%
#     (baseline: 0.591 +- 0.029, 70.4%). Reference: ~/kld-cpu-e4b-4ch.bin, made
#     with RKNPU_HYBRID=W8A8_STANDARD RKNPU_CPU_DECODE=999999 on the baseline build.
#  3. llama-server answers all decode prompts, 3 runs, no crash
set -euo pipefail
B=pi@192.168.0.178
cd /workspace/docs/backend
for t in check-prep check-dispatch check; do make -s -f Makefile.rknpu2-tools $t 2>&1 | grep -q "0 failures" || { echo "unit $t failed"; exit 1; }; done
Q=$(ssh -o BatchMode=yes $B 'ulimit -n 65536; cd ~/rk-llama.cpp-rebase; M=~/models/gemma-4-E4B-it-Q4_0-ggmlorg.gguf; W=~/wikitext-2-raw/wiki.test.raw
  p32=$(build/bin/llama-perplexity -m $M -f $W --chunks 32 -t 4 2>&1 | grep -oE "Final estimate: PPL = [0-9.]+" | grep -oE "[0-9.]+$")
  k=$(build/bin/llama-perplexity -m $M -f $W --chunks 4 -t 4 --kl-divergence-base ~/kld-cpu-e4b-4ch.bin --kl-divergence 2>&1)
  kld=$(echo "$k" | grep -E "^Mean +KLD" | grep -oE "[0-9]+\.[0-9]+" | head -1)
  top=$(echo "$k" | grep -E "^Same top p" | grep -oE "[0-9]+\.[0-9]+" | head -1)
  echo "$p32 $kld $top"')
read -r P32 KLD TOP <<< "$Q"
echo "ppl32=$P32 kld=$KLD same_top=$TOP"
python3 -c "import sys; p,k,t=$P32,$KLD,$TOP; sys.exit(0 if (p <= 26.8771*1.02 and k <= 0.65 and t >= 68.0) else 1)" || { echo "quality gate failed"; exit 1; }
# NPU flash attention vs CPU at Gemma-4 shapes (prefill path the decode check never reaches)
FA=$(ssh -o BatchMode=yes $B 'cd ~/rk-llama.cpp-rebase && g++ -O2 docs/backend/test-rknpu2-flash-attn.cpp -I ggml/include -Lbuild/bin -lggml -lggml-base -lggml-cpu -lggml-rknpu2 -Wl,-rpath,$PWD/build/bin -o /tmp/test-rknpu2-flash-attn && ulimit -n 65536 && /tmp/test-rknpu2-flash-attn 2>&1 | grep -c " OK$"'; true)
echo "flash-attn test: $FA/5 OK"
[ "$FA" = "5" ] || { echo "flash-attn test failed"; exit 1; }
for i in 1 2 3; do
  out=$(ssh -o BatchMode=yes $B 'bash ~/rk-llama.cpp-rebase/docs/backend/rknpu2-autoresearch/decode-check.sh 2>&1 | tail -1'; true)
  echo "server run $i: $out"
  echo "$out" | grep -q "decode-identical" || { echo "server failed"; exit 1; }
done
