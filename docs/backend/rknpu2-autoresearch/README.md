# Autoresearch loop: Gemma-4 E4B NPU decode (2026-10-05)

Harness for the iterative optimization loop recorded in
`../RKNPU2-decode-research.md` #1c. Each iteration is one commit, measured
by `verify.sh` and accepted only if `guard.sh` passes; `results.tsv` is the
log (keep threshold: > +0.04 t/s over the previous keep, ~0.7%, from the
measured baseline noise of 5.795-5.828 t/s).

| File | Runs on | What it does |
|---|---|---|
| `verify.sh` | dev container | bundles HEAD to the board, rebuilds, prints E4B pure-NPU W4A4 `llama-bench -n 64 -r 3 -t 4` tg t/s (taskset 4-7) |
| `guard.sh` | dev container | backend unit tests (check-prep, check-dispatch, check) + E4B 8-chunk PPL must equal 35.0480 + `decode-check.sh` |
| `decode-check.sh` | board | `llama-server` greedy, 4 prompts x 128 tokens; text must be byte-identical to the pre-loop reference (`~/bench-logs/2026-10-02-nthreads/mtp/npu-nthr-base-p*.json`) |
| `stress.sh <tag> <n>` | board | repeats `decode-check.sh` n times with core dumps on, writes a gdb backtrace per crash |

Perplexity alone does not exercise the M=1 decode path (it runs 512-row
batches), which is why the decode check exists; and one decode check per
iteration is not enough to catch an intermittent crash, which is why the
stress loop exists (see #1c, the empty-mul_mat segfault).

Board address and model paths are hard-coded for the Orange Pi 5 Ultra
used here (`pi@192.168.0.178`, `~/models/gemma-4-E4B-it-Q4_0-ggmlorg.gguf`).
