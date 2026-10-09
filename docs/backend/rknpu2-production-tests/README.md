# Production limit tests, 2026-10-09 (Orange Pi 5 Ultra, Gemma-4 E4B Q4_0)

Raw results behind `../RKNPU2-production.md`. `prodtest.sh` is the suite
(stages `pp tg server ppl`, run on the board from the repo root).

- `pp*.json`: llama-bench prefill, 128-4096 tokens. pp8192 / 16384 / 32768
  were re-measured after the #1l fixes in separate runs (166.3 / 112.9 /
  28.2 t/s); the first run crashed at 8192 (the bug #1l fixes).
- `tg-d*.json`: llama-bench tg32 at context depths 0-32000.
- `server-base.json`, `server-mtp.json`: llama-server `-c 32768`, needle +
  summary prompts of 0.8k-24.8k tokens, without and with the MTP drafter;
  generated text included.
- `ppl-*.txt`, `kld-npu-c2048.txt`: NPU vs CPU perplexity at 2k and 8k
  context and the 2k KL divergence. A 32k window does not fit
  llama-perplexity's logit buffer (~34 GB).
- `progress.txt`: run log, including the pauses for the fixes.
