# RKNPU2 production guide: Gemma-4 E4B on the RK3588 NPU

Measured on an Orange Pi 5 Ultra (RK3588, 16 GB LPDDR5, NPU at 1 GHz, DDR at
2400 MHz, big cores pinned at 2.35 GHz). Model:
`gemma-4-E4B-it-Q4_0` (the ggml-org GGUF), pure NPU, default W4A4.
Branch `main` (code identical to the tested `rebase/w4a4-on-upstream`).
Test logs on the board: `~/bench-logs/2026-10-10-prod/` (full suite;
numerically the same code as the final default), `~/bench-logs/2026-10-10-prod-fix/`
(the `RKNPU_FA_ONLINE_MIN=4096` variant) and `~/bench-logs/2026-10-10-ident/`.

## 1. How to run it

Once per boot, pin the clocks (they are not pinned after boot):

    sudo ~/rk3588-scripts/fix_freq_rk3588.sh   # check scaling_governor == userspace

Server, with the MTP drafter (fastest exact setup):

    ulimit -n 65536
    RKNPU_EXCLUDE_TYPES=f16 taskset -c 4-7 build/bin/llama-server \
        -m ~/models/gemma-4-E4B-it-Q4_0-ggmlorg.gguf \
        -md ~/models/gemma-4-E4B-it-assistant-centroid-F16.gguf \
        --spec-type draft-mtp --spec-draft-n-max 5 --spec-draft-p-min 0.6 \
        -t 4 -c 32768 --temp 0

- `taskset -c 4-7 ... -t 4`: run on the four A76 cores. Unpinned, threads
  land on the A55 cores and decode loses ~25%.
- `RKNPU_EXCLUDE_TYPES=f16`: the small F16 drafter runs on the CPU, which is
  slightly faster than the NPU's F16 path (18.19 vs 17.85 t/s).
- The drafter must be the **centroid** conversion (made by this branch's
  `convert_hf_to_gguf.py` from `google/gemma-4-E4B-it-assistant`). An older
  GGUF still works but drafts ~5% slower.
- `--reasoning off` turns Gemma-4's hidden thinking off. Leave it on if the
  answers benefit from it, but count the thinking tokens in `max_tokens`.
- Without a drafter, drop the `-md`/`--spec-*` options: decode is then
  ~8.5-9 t/s at short context.

## 2. Supported sizes

| Limit | Value | Notes |
|---|---|---|
| Context window (`-c`) | **32768 tested** | KV cache ~0.6 GiB at 32k (16 KiB per cell for the 4 full-attention layers + a fixed ~100 MiB sliding-window cache); 16 GB RAM is not the limit |
| Prompt length | **up to 32k** | NPU attention for the whole window (P*V in 2048-position chunks; see decode research #1l/#1m) |
| ubatch (`-ub`) | **512 (default)** | NPU attention is validated for this size only; larger ubatches fall back to CPU attention |
| Parallel slots | 4 (server default), unified KV | all slots share the context window |

Practical guidance (measured 2026-10-10): up to ~8k-token prompts
prefill in under 40 s; 16k takes ~1.5 minutes; 32k takes ~4 minutes.

Memory (server, `-c 32768`, 4 slots, MTP drafter loaded): resident size
8.3 GiB after a short prompt, 10.1 GiB peak after a 24.8k-token prompt;
the board kept >= 9.9 GiB available throughout.

## 3. Speed

### Prompt processing (llama-bench, tokens/s)

| Prompt tokens | 128 | 512 | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|---|---|
| t/s before #1m (2026-10-09) | 191.7 | 296.2 | 275.2 | 261.2 | 239.4 | 166.3 | 112.9 | 28.2 |
| **t/s now (2026-10-10)** | **190** | **288** | **268** | **256** | **237** | **218** | **181** | **137** |
| time now | 0.7 s | 1.8 s | 3.8 s | 8.0 s | 17 s | 38 s | 1.5 min | 4.0 min |
| t/s with `RKNPU_FA_ONLINE_MIN=4096` | 196 | 301 | 280 | 266 | 245 | 219 | 182 | 138 |

(#1m: P*V chunking, FP16 scores, online softmax, early P*V. The online
softmax costs ~3-4% on prompts up to 4k. `RKNPU_FA_ONLINE_MIN=4096` uses it
only above 4096 KV cells and wins that back, but mixing the two softmax
roundings in one prompt broke MTP's output identity on a 12.8k prompt
(#1n); use it only without a drafter.)

### Generation vs context already in the window (no drafter, llama-bench tg32)

| Context depth | 0 | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|---|
| t/s | 8.97 | 8.22 | 7.97 | 7.52 | 6.76 | 5.63 | 4.31 |

### Server with real prompts (needle + 256-token summary, `-c 32768`)

| Prompt tokens | Prompt t/s | Gen t/s, no draft | Gen t/s, MTP | MTP speed-up |
|---|---|---|---|---|
| 784 | 229 | 8.24 | 12.74 | 1.55x |
| 3,428 | 209 | 7.66 | 11.39 | 1.49x |
| 6,637 | 194 | 7.09 | 9.67 | 1.36x |
| 12,796 | 170 (2026-10-09: 115) | 6.20 | 8.36 | 1.35x |
| 24,767 | 139 (2026-10-09: 38) | 4.97 | 5.46 | 1.10x |

Short prompts (the 4 bench prompts, 128-token answers): **18.37 t/s with MTP**
vs 8.5 without (2.2x). MTP gains less on long contexts because every verify
step's CPU attention grows with the context, and acceptance depends on the
text (summaries of long, mixed documents accept less).

## 4. Accuracy

- **No garbage at any size:** the hidden code word was recalled in 10/10
  runs (prompts 0.8k-24.8k, with and without MTP). The 256-token summaries
  are coherent and on-topic up to 24.8k tokens.
- **MTP is exact:** MTP output is byte-identical to no-draft output
  (2026-10-10: all 10 server answers, 0.8k-24.8k prompts; holdout 6/6 at
  256 and 512 tokens). A 6-token verify batch gives bit-identical logits to
  one-at-a-time decoding at 12.8k context (`docs/handover/profiling-patches/longdiff.cpp`).
- **NPU vs CPU perplexity (wikitext-2):** see the table below.

| Context | NPU PPL | CPU PPL (reference) |
|---|---|---|
| 512 (32 chunks, guard) | 27.2322 | 27.01 (#3d; same quantization) |
| 2048 (8 chunks) | 22.64 +/- 0.85 | 24.88 +/- 1.05 |
| 8192 (2 chunks) | 21.89 +/- 0.81 | 20.88 +/- 0.82 |

KL divergence of the NPU output against the CPU reference: 0.610 at 512
context (same top token 70.3%), 0.640 at 2048 (same top 65.9%). The NPU's
W4A4 activations make its token distributions differ from the CPU's, but
perplexity stays level with the CPU (within the error bars), and the
long-context recall and summaries are correct. (llama-perplexity cannot
evaluate a 32k window here: it holds all 32k x 262k logits, ~34 GB.)

**Caveat, determinism across requests:** for long prompts the output can
depend slightly on what the server processed before (the KV cache's cell
layout changes the NPU prefill numerics a little). A freshly started server
is fully reproducible; after other requests a long-prompt answer can differ
in wording at a near-tie, with the same quality. This is related to the
open issue that NPU attention output depends slightly on which NPU core
serves a KV group (decode research #1g).

## 5. What changed, and what it bought

Starting point: the original RKNPU2 backend (invisiofficial/rk-llama.cpp)
ran E4B at pp128 39.8 t/s and tg 3.3 t/s (its W8A8 default); its W4A4
mode ran at pp128 7.7 / tg 3.4 with ~5x the CPU's perplexity. The CPU alone
did pp128 25.2 / tg 4.9.

| Area | Change | Effect (E4B) |
|---|---|---|
| W4A4 quality | block Hadamard (FWHT), per-channel weight scales, clipped INT4, K-padding fix | W4A4 perplexity level with the CPU; -29% NPU memory |
| W4A4 speed | NEON A-prep and dequant, native NPU layouts, MoE/batched-matmul fix | pp128 7.7 -> 37 t/s, tg 3.4 -> 5.5 (decode research #3, #4c) |
| Decode loop | NEON bf16 dot, segments on ggml's OpenMP team, per-thread collect, split A-prep, cached io bindings | tg 5.8 -> 8.24 (#1c), 8.78 after the rebase |
| Prefill loops | pipelined chunks, whole-FFN schedule, NPU flash attention (native A/B/C, pipelined), ggml-cpu norm fusions, fused Q/K/V norms + RoPE + KV-cache writes, range-limited softmax | pp512 68 -> **296 t/s** (4.3x; #1d-#1h) |
| MTP | `set_n_threads` fix, W8A8 native layout for verify batches, centroid drafter, drafter on CPU, `p_min` | decode with drafter 8.5 -> **18.4 t/s** (#1b, #1i) |
| MTP exactness | ggml-cpu flash attention split-KV off by default | MTP output identical to no-draft (#1j) |
| Long context | GQA-grouped ggml-cpu attention for decode/verify | verify at 768 context -17% latency (#1k) |
| Robustness | bounded NPU attention context cache, NPU attention limited to validated shapes, async-runner race fix | prompts >= 8k no longer crash; server no longer segfaults on prefill (#1l) |
| Long prompts | P*V in 2048-position chunks (the NPU FP16 matmul collapses past K 4096), NPU attention to 32k, FP16 scores, online softmax per chunk, early P*V, chunk outputs combined in host memory | pp8k 166 -> ~220, pp16k 113 -> ~181, pp32k 28 -> ~134 (#1m) |

Total for E4B against the original backend's default: **prefill 4.8x**
at pp128 (39.8 -> 190; larger prompts gain more: pp512 288, pp16k 181), **decode 2.7x without a drafter**
(3.3 -> 8.97) and **5.6x with MTP** (3.3 -> 18.4), at CPU-level accuracy.

Every change is described, with its measurements and rejected
alternatives, in `RKNPU2-decode-research.md` (section numbers above) and
`RKNPU2-optimization-notes.md`.

## 6. Branch layout (for upstream PRs)

`main` is a short topic stack on upstream llama.cpp `4ebdf2c74`:

1. The original RKNPU2 backend commits (Invisi, Polarnik, hvalev,
   Gerald Tan, Martino Mensio), unchanged.
2. Generic llama.cpp topics, each building on its own and independent of
   the backend, and so candidates for slim ggml-org PRs:
   - `ggml-cpu: NEON vec_dot_bf16, four rows at a time`
   - `ggml-cpu: NEON tanh-GELU for GEGLU`
   - `ggml-cpu: RMS_NORM + MUL + ADD fusion`
   - `ggml-cpu: flash attention batch invariance + GQA-grouped kernel`
   - `gemma4-assistant: centroid draft logits`
3. `rknpu2:` the backend optimizations, then its tests and probes.
4. `docs:` research notes and this guide (fork only).

Checked on 2026-10-09: the five generic commits cherry-pick cleanly onto
plain upstream `4ebdf2c74`, and each builds there on its own (backend off).
The original backend commits do not build against this newer upstream on
their own (they predate API changes); the tree builds again from the
`rknpu2:` optimization commit on. The full research history stays on
`rebase/w4a4-on-upstream`.
