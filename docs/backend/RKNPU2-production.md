# RKNPU2 production guide: Gemma-4 E4B on the RK3588 NPU

Measured on an Orange Pi 5 Ultra (RK3588, 16 GB LPDDR5, NPU at 1 GHz, DDR at
2400 MHz, big cores pinned at 2.35 GHz). Model:
`gemma-4-E4B-it-Q4_0` (the ggml-org GGUF), pure NPU, default W4A4.
Branch `main` (code identical to the tested `rebase/w4a4-on-upstream`).
Test logs on the board: `~/bench-logs/2026-10-10-prod/` (full suite),
`~/bench-logs/2026-10-10-prod-fix/` (prompt speeds of the final default),
`~/bench-logs/2026-10-10-kvfix-4096/` and `-kvfix-0/` (server runs with the
KV-cache fix), `~/bench-logs/2026-10-10-ident/`, and for the NPU decode
attention (2026-10-11) `~/bench-logs/2026-10-11-k8/` and, for the final build
with the INT4 LM head (2026-10-12), `~/bench-logs/2026-10-12-lmhead/` (server).

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
- Decode attention of the global layers runs on the NPU over a KV mirror
  (`RKNPU_FA_DEC`, default on; decode research #1o; K stored as INT8,
  `RKNPU_FA_DEC_K8`). It costs ~384 MiB of extra memory at 32k.
- The LM head streams INT4 weights and re-scores the near-top logits exactly
  from the original Q8_0 bytes (`RKNPU_LMHEAD_W4`, default on; #1r): the
  argmax is exact (greedy output and MTP acceptance unchanged), the tail of
  the distribution is INT4-quality (PPL 27.0 instead of 26.5), +10% decode.
  It keeps 680 MiB of Q8_0 bytes (+84 MiB of block scales) in host memory and is not compatible with `--context-shift` (off by
  default): with context shift, set `RKNPU_FA_DEC=0`.

## 2. Supported sizes

| Limit | Value | Notes |
|---|---|---|
| Context window (`-c`) | **32768 tested** | KV cache ~0.6 GiB at 32k (16 KiB per cell for the 4 full-attention layers + a fixed ~100 MiB sliding-window cache); 16 GB RAM is not the limit |
| Prompt length | **up to 32k** | NPU attention for the whole window (P*V in 2048-position chunks; see decode research #1l/#1m) |
| ubatch (`-ub`) | **512 (default)** | NPU attention is validated for this size only; larger ubatches fall back to CPU attention |
| Parallel slots | 4 (server default), unified KV | all slots share the context window |

Practical guidance (measured 2026-10-10): up to ~8k-token prompts
prefill in under 40 s; 16k takes ~1.5 minutes; 32k takes ~4 minutes.

Memory (server, `-c 32768`, 4 slots, MTP drafter loaded, NPU decode
attention, INT4 LM head with its Q8_0 copy): resident size 8.8 GiB after a
short prompt, 10.6 GiB peak after a 24.8k-token prompt; the board kept
>= 9.0 GiB available throughout.

## 3. Speed

### Prompt processing (llama-bench, tokens/s)

| Prompt tokens | 128 | 512 | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|---|---|
| t/s before #1m (2026-10-09) | 191.7 | 296.2 | 275.2 | 261.2 | 239.4 | 166.3 | 112.9 | 28.2 |
| **t/s now (2026-10-12)** | **201** | **299** | **280** | **263** | **245** | **219** | **181** | **136** |
| time now | 0.7 s | 1.7 s | 3.7 s | 7.7 s | 17 s | 37 s | 1.5 min | 4.0 min |

(#1m: P*V chunking, FP16 scores, online softmax above 4096 KV cells
(`RKNPU_FA_ONLINE_MIN`; with it on for every size, prompts up to 4k are
3-4% slower), early P*V.)

### Generation vs context already in the window (no drafter, llama-bench tg32, median of 3)

| Context depth | 0 | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|---|
| **t/s, final (NPU decode attention, INT8 K, INT4 LM head)** | **9.76** | **8.46** | **8.38** | **8.23** | **7.99** | **7.34** | **6.50** |
| t/s, before the INT4 LM head (2026-10-11) | 9.00 | 8.18 | 7.84 | 7.92 | 7.55 | 6.96 | 6.30 |
| t/s, CPU attention (`RKNPU_FA_DEC=0`) | 8.99 | 8.25 | 8.03 | 7.63 | 6.70 | 5.72 | 4.43 |

(The first generation after the context grows past a new 2048-position
block creates that block's NPU contexts, a one-time cost per block.)

### Server with real prompts (needle + 256-token summary, `-c 32768`)

| Prompt tokens | Prompt t/s | Gen t/s, no draft | Gen t/s, MTP | MTP speed-up |
|---|---|---|---|---|
| 784 | 235 | 8.41 | 11.64 | 1.38x |
| 3,428 | 209 | 8.51 | 11.66 | 1.37x |
| 6,637 | 193 | 8.25 | 12.35 | 1.50x |
| 12,796 | 170 (2026-10-09: 115) | 7.80 (CPU attention: 6.23) | **11.50** (8.39) | 1.47x |
| 24,767 | 139 (2026-10-09: 38) | 6.99 (4.96) | **9.67** (5.42) | 1.38x |

(`~/bench-logs/2026-10-12-lmhead/`, final build. MTP speed depends on the
text: how many drafts the summary accepts varies by +-10% between texts.)

Short prompts (holdout, 6 prompts x 256 tokens): **18.8 t/s with MTP** vs
9.5 without (2.0x). With the NPU decode attention, MTP's verify steps no
longer slow down much with the context.

## 4. Accuracy

- **No garbage at any size:** the hidden code word was recalled in 10/10
  runs (prompts 0.8k-24.8k, with and without MTP). The 256-token summaries
  are coherent and on-topic up to 24.8k tokens.
- **MTP is exact:** MTP output is byte-identical to no-draft output
  (2026-10-12, final build: all 10 server answers at 0.8k-24.8k prompts;
  holdout 6/6 at 256 and 512 tokens; the 4 bench prompts still produce the
  texts of the 2026-10-02 CPU-attention build). This needs the `llama-kv-cache` fix on this branch (#1n):
  without it, rejected drafts shift the sliding-window cache's cell order,
  the attention rounds differently, and an answer can change at a near-tie.
- **NPU vs CPU perplexity (wikitext-2):** see the table below.

| Context | NPU PPL | CPU PPL (reference) |
|---|---|---|
| 512 (32 chunks, guard) | 26.92 (INT4 LM head; 26.55 with the INT8 LM head) | 27.01 (#3d; same quantization) |
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
| MTP exactness | ggml-cpu flash attention split-KV off by default; KV cache keeps the cell order after rejected drafts | MTP output identical to no-draft (#1j, #1n) |
| Long context | GQA-grouped ggml-cpu attention for decode/verify | verify at 768 context -17% latency (#1k) |
| Robustness | bounded NPU attention context cache, NPU attention limited to validated shapes, async-runner race fix | prompts >= 8k no longer crash; server no longer segfaults on prefill (#1l) |
| Long prompts | P*V in 2048-position chunks (the NPU FP16 matmul collapses past K 4096), NPU attention to 32k, FP16 scores, online softmax per chunk, early P*V, chunk outputs combined in host memory | pp8k 166 -> ~220, pp16k 113 -> ~181, pp32k 28 -> ~134 (#1m) |
| Long-context decode | NPU decode/verify attention for the global layers over a persistent native-layout KV mirror (INT8 K), short rows on a grouped CPU kernel | tg32 @16k 5.72 -> 6.96, @32k 4.43 -> 6.30; MTP on a 24.8k summary 5.42 -> 8.5 (#1o, #1p) |
| LM head bytes | INT4 LM head, near-top logits re-scored exactly from the Q8_0 bytes | tg 8.96 -> 9.8, MTP holdout 17.5 -> 18.8, long-prompt MTP 8.5 -> 9.7 (#1r) |

Total for E4B against the original backend's default: **prefill 4.8x**
at pp128 (39.8 -> 201; larger prompts gain more: pp512 299, pp16k 181), **decode 3.0x without a drafter**
(3.3 -> 9.8) and **5.7x with MTP** (3.3 -> 18.8), at CPU-level accuracy.

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
   - `llama-kv-cache: after a tail seq_rm, continue from the cell of the first removed position`
   - `llama-context: a fused op an accelerator takes from a CPU layer does not disable it`
3. `rknpu2:` the backend optimizations, then its tests and probes.
4. `docs:` research notes and this guide (fork only).

Checked on 2026-10-09: the first five generic commits cherry-pick cleanly
onto plain upstream `4ebdf2c74`, and each builds there on its own (backend
off). The `llama-kv-cache` (2026-10-10) and `llama-context` (2026-10-11)
commits cherry-pick cleanly too.
The original backend commits do not build against this newer upstream on
their own (they predate API changes); the tree builds again from the
`rknpu2:` optimization commit on. The full research history stays on
`rebase/w4a4-on-upstream`.
