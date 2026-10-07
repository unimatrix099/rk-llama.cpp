# RKNPU2: improving 4-bit token-generation (decode) speed — research notes

Research thread with **measured verdicts as of 2026-08-22** (six board
sessions; tooling under "Experiment tooling" below). Where each avenue
landed:

- **#1 speculative decoding** — dead end, every variant measured.
- **#2 cooperative CPU+NPU decode** — dead end (probe gate failed), but
  its measurements re-aimed everything that followed.
- **#3 backend overhead** — root cause found (libgomp team respawn ~1x
  per graph split) and fixed in code (persistent dispatch pool +
  `if(M>1)`): decode +18–28% with no env vars.
- **#3b W4A4 K-padding** — found (1.5–1.6x inflated reads on
  non-pow2 models) and fixed (block-diagonal FWHT): W4A4 tg +28%, NPU
  memory −29%.
- **#3c W4A4 per-channel weight scales** — the quality fix: PPL 5x
  better at equal speed, W4A4 load minutes → seconds; confirmed at 32
  chunks and across models.
- **#3d clipped INT4 scales** — the finish: E4B W4A4 PPL 34.36 → **26.95**,
  i.e. level with the W8A8 (27.29) and pure-CPU (27.01) references.
  W4A4 is no longer a quality compromise on this model.
- **#3e per-channel INT8 scales** — the same fix for W8A8: Qwen 12.24 →
  **9.08** (its CPU ref is 8.88), E4B unchanged within noise. W8A8's
  quality premium is now ~2% on both models instead of 1% on one and
  38% on the other. Also closed INT8 clipping with a reason (it wraps).
- **#5 threading** — shipped guidance (`-t 4` big cores): +19–66% tg.

Net effect on the flagship (Gemma-4 E4B Q4_0), from the state at the
start of these sessions to now: routed pp 41.3/tg 4.9 → **41.4/5.50**;
pure-NPU W4A4 pp 7.7-33/tg 3.4/PPL ~5x-over-CPU → **pp 37.0/tg 5.49/PPL
26.95, level with the CPU (27.01) and W8A8 (27.29) references, at −29%
NPU memory**. Companion documents:
`RKNPU2-optimization-notes.md` (shipped work + dead ends),
`RKNPU2-w4a4-story.md` (narrative account for readers new to the thread),
`RKNPU2-int4-research.md` (INT4 fundamentals), `RKNPU2-neon-prep-plan.md`
(prefill prep). Numbers: Orange Pi 5 Ultra (RK3588, 16 GB), pinned clocks,
`llama-bench -p 128 -n 64` unless noted.

## Where decode stands after the shipped work

> 🧭 This document is the *investigation record* — hypotheses, dead ends
> and the measurements behind them. If you just want the configuration to
> run for a given model, use the model matrix in
> `RKNPU2-deployment-guide.md` ("Known-good models"), which is the
> maintained operational version.

Decode re-reads the active weight set for every generated token, so it is
memory-bandwidth bound. For Gemma-4-E4B Q4_0 (7.46 B dense):

| Decode path (current) | tg64 t/s | PPL (32ch) | notes |
|---|---|---|---|
| Routed CPU decode (`RKNPU_CPU_DECODE=32`) | **5.50** (pp 41.4) | 27.01 | CPU-exact decode; ~25 GB/s at t=4 big-core (the old "13 GB/s wall" was an 8-thread artifact — #5) |
| NPU W4A4 (all defaults: block-FWHT, per-channel, clipped) | **5.49** (pp 37.0) | **26.88** | −29% NPU memory, CPU left free, quality level with the 8-bit tier (#3b/#3c/#3d) |
| NPU W8A8 (per-channel scales) | 4.55 | 27.61 | ~+2% over CPU on both models tested since #3e (was +38% on Qwen) |

Two framing "facts" — **both revised by the 2026-08-10 measurements**:

1. ~~The CPU path is already at its wall.~~ It was a threading wall, not a
   bus wall: 4 big-core threads reach ~25 GB/s effective (avenue #5).
   With Q4_0's 2.4 GB/token that is ~8-9 t/s of headroom for E4B if the
   remaining CPU graph time shrinks; qwen-class models already do 13.5.
2. **The NPU path's overhead is NOT driver dispatch.** The coop probe
   measured the backend's exact per-node M=1 driver sequence at ~28.6
   GB/s aggregate (~14 t/s equivalent for E4B) — the loss to 3.65 t/s
   happens in the backend/CPU side around the matmuls (avenue #2 findings,
   avenue #3 is now the main code lever).

## Avenues, ranked

### 1. Speculative decoding — MEASURED: every variant loses (2026-08-10)

The hypothesis assumed near-free batch verification. That holds on GPUs;
here it fails structurally, and no variant beat its no-spec baseline
(qwen2.5-1.5B routed, t=4, seed 42, 128 tokens):

| Variant | t/s | acceptance | baseline |
|---|---|---|---|
| draft model (0.5B, draft-max 8) | 4.30* | 41.7% | 13.52 (bench) |
| ngram-simple, repetitive rewrite (clean board, OMP fix) | **13.80** | 43% (18/42) | **13.81** (server) |
| ngram-simple + verify on NPU (`RKNPU_CPU_DECODE=16` or `8`) | 11.62* | 60% (48 drafted) | 12.15* |
| any ngram type, creative prompt | 12.17–12.20* | no drafts trigger | 12.15* |
| E4B ngram-simple, repetitive | 3.64* | 2/16 | 4.17* (server) |

*Measured while a stuck background process burned one core (found and
killed 2026-08-11); relative comparisons shared that handicap, so the
verdicts hold. The headline pair was re-verified on a clean board with
`OMP_NUM_THREADS=4`: ngram-simple is exactly **neutral** (13.80 vs
13.81), not the loss the dirty numbers suggested — the drafts that
trigger pay for their verification and no more.

Why each fails:
- **Draft model**: qwen-0.5B is 1/3 of the target's bytes — even at 100%
  acceptance it roughly breaks even, and real acceptance was 42%. The
  E4B+E2B pair was skipped on arithmetic: E2B is 68% of E4B's bytes.
  There is no smaller same-family draft, so this is closed, not tunable.
- **CPU verification**: an M=9–17 batch costs 2–3x a single token on 4
  big cores (compute-bound above M=1), eating the accepted-token savings.
- **NPU verification** (threshold trick, batches to NPU at prefill speed):
  acceptance improved, but every verify batch pays the full multi-split
  graph overhead (~338 splits at bs>1 on qwen), and drafts trigger on
  only ~1/3 of tokens even on a deliberately repetitive rewrite task.
  Identical results at threshold 16 and 8 confirm verify cost is no
  longer the binding constraint — draft trigger rate is.

Operational pitfalls discovered (baked into the scripts):
- `llama-cli` busy-spins on its interactive prompt at EOF stdin — one
  core at 100% forever. Use llama-bench / llama-server for automation.
- Never `taskset` a two-model process onto 4 cores: two spinning ggml
  threadpools oversubscribe and cost 2.6x (independent of OMP settings).

Could be revisited if: a ~150M same-tokenizer draft appears for a target
worth running, or graph-split overhead at bs>1 drops enough (avenue #3)
to make NPU verification cheap.

> **Revisited 2026-10-02 — both conditions met, see #1b.** Gemma-4 ships
> 78.8M MTP drafters that share the target's KV cache, and the rebase onto
> upstream (2026-10-02) brought `--spec-type draft-mtp`. MTP now wins on
> the CPU and on the NPU; the routed path still loses.

### 1b. MTP drafters (Gemma-4) — MEASURED: up to +92% decode (2026-10-02)

Setup: branch `rebase/w4a4-on-upstream` (upstream master 2026-10-02 plus
this fork), drafters converted from `google/gemma-4-{E4B,E2B}-it-assistant`
with `convert_hf_to_gguf.py --outtype f16` (~170 MB each). Targets: ggml-org
E4B Q4_0 (the file every earlier E4B number used) and Google's E2B QAT
Q4_0. `llama-server -md <drafter> --spec-type draft-mtp --spec-draft-n-max
N`, greedy, 4 prompts x 128 tokens, server `predicted_per_second`
averaged. Script: `rknpu2-gemma4-mtp-bench.sh`. Every MTP run produced
output **byte-identical** to its no-draft run (4/4 prompts, all 30 cells).

| decode t/s | no draft | n=1 | n=2 | **n=3** | n=4 |
|---|---|---|---|---|---|
| E4B pure CPU | 6.44 | 6.35 | 6.12 | **10.67** | 7.87 |
| E4B pure NPU W4A4 (default) | 6.89 | **8.49** | 7.85 | 8.32 | 6.45 |
| E4B routed W8A8 + `CPU_DECODE=32` | 5.62 | 5.71 | 5.76 | 4.58 | 4.36 |
| E2B pure CPU | 13.88 | 13.35 | 12.42 | **19.94** | 15.10 |
| E2B pure NPU W8A8 | 6.98 | 12.13 | 12.61 | **13.42** | 10.87 |
| E2B routed W8A8 + `CPU_DECODE=32` | 11.17 | 10.71 | 10.12 | 7.97 | 7.51 |

Acceptance falls with draft length (E4B CPU: 74/65/58/50% for n=1..4;
lower on the W4A4 NPU: 68/57/50/43%, because its numerics differ slightly
from the drafter's training target).

**Why n=3 on the CPU — the 4-row tile.** An n=3 draft is verified as an
M=4 batch, and ggml-cpu's repacked Q4_0 GEMM (`CPU_REPACK` buffer)
processes rows in tiles of four. Measured batch cost, E4B,
`llama-bench -p 1,2,3,4,5,6,8 -n 0`:

| ms per batch | M=1 | M=2 | M=3 | **M=4** | M=5 | M=6 | M=8 |
|---|---|---|---|---|---|---|---|
| pure CPU (`CPU_REPACK`) | 150 | 231 | 293 | **224** | 308 | 369 | 405 |
| routed (RKNPU host copy, plain Q4_0) | 175 | 283 | 359 | 529 | 608 | 642 | 882 |
| pure NPU W4A4 | 172 | 237 | 247 | 262 | 285 | 302 | 338 |

An M=4 batch is *cheaper* than M=2 or M=3 on the repacked path — which is
why n=1 and n=2 lose and n=3 jumps. The cycle arithmetic checks out
against the server numbers (E4B prompt 0: 53 verify cycles for 128
tokens, 9.0 t/s -> 267 ms/cycle = 224 ms verify + ~15 ms per draft step).

**Why routed loses.** In routed mode the weights live in the RKNPU
buffer, and the CPU computes the M < 32 ops from its plain Q4_0 host copy
— not the repacked layout, which only exists for tensors in a CPU buffer.
So routed pays 529 ms for M=4 where pure CPU pays 224. The same effect is
why pure-CPU decode (6.74 t/s llama-bench) now beats routed decode (5.68)
on E4B. **This is the clearest open opportunity on the board:** a routed
mode whose CPU-side copy is repacked would combine NPU prefill (~43 t/s)
with repacked decode, and therefore with MTP at n=3 (~10.7 t/s).

**The NPU win needed a fix — the #3 churn on a new path.** Before it,
MTP *lost* on the NPU (E4B n=1..4: 5.26 / 5.45 / 5.99 / 4.94 against
6.89), although the batch-cost table says an M=2 verify costs only 1.4x
an M=1 step. Two hypotheses, tested in order:

1. *The F16 drafter runs on the NPU's slow W16A16 path.* It does —
   `--device-draft none` cannot move it, because the backend registers as
   an `ACCEL` device and llama prepends ACCEL buffer types to the CPU's
   list (`src/llama-model.cpp`), and `RKNPU_EXCLUDE` cannot target it
   because the drafter's block names (`blk.0`-`blk.3`) collide with the
   target's. Added `RKNPU_EXCLUDE_TYPES=f16` to move it. **Result: no
   change** (n=3: 6.00 vs 5.99). Rejected.
2. *libgomp team respawn.* strace: **40,956 `clone3` for 64 tokens** of
   E4B MTP. Verify batches arrive with M > 1, which re-enables the per-row
   A-prep and C-dequant OpenMP regions (`if(M > 1)`, #3), and their
   default team size differs from ggml-cpu's `-t 4` team. Cheap check
   first: `OMP_NUM_THREADS=4` -> n=1 8.45 t/s. Confirmed.

The fix: the backend now implements `ggml_backend_set_n_threads` (via
`get_proc_address`) and uses that count as the team size for both
regions. n=1 5.26 -> **8.49**, n=3 5.99 -> 8.32; no-draft decode 6.89
unchanged; llama-bench pp128/tg64 unchanged within noise (44.95 / 5.83);
PPL bit-identical (35.0480, 8 chunks). The env workaround is no longer
needed.

**Verdicts.** MTP at n=3 is now the best decode on the board for both
Gemma-4 models (E4B 10.67, E2B 19.94, pure CPU), output-identical, at the
cost of CPU-speed prefill (E4B 24 t/s, E2B 73 t/s). On the NPU it is a
solid win (E4B +23% at n=1, E2B +92% at n=3) with NPU prefill retained.
Routed + MTP should not be used until the routed CPU copy is repacked.

### 1c. NPU decode loop — 5.81 -> 8.24 t/s, bit-identical (2026-10-05)

An iterative measure/keep/revert loop on Gemma-4 E4B, pure NPU W4A4
(defaults), `llama-bench -n 64 -r 3 -t 4`, taskset 4-7. Guard on every
iteration: backend unit tests, 8-chunk PPL exactly 35.0480, and greedy
`llama-server` output byte-identical on 4 prompts. Harness and full log:
`rknpu2-autoresearch/`.

| # | Change | tg t/s | Verdict |
|---|---|---|---|
| 0 | baseline (`rebase/w4a4-on-upstream`) | 5.81 | — |
| 1 | Hadamard sign vector by pointer, not per-node copy | 5.79 | discard (noise) |
| 2 | NEON `ggml_vec_dot_bf16` on ARM — ggml had only a scalar loop there; products in fp32, summed one by one in double in index order | 6.05 | **keep** +4.1% |
| 3 | per-node NPU segments run on ggml's (size-matched, hot) OpenMP team instead of the dispatch pool | 7.78 | **keep** +28.7% |
| 4 | at M=1 each team thread syncs and dequantizes its own segment right after its run | 7.89 | **keep** +1.4% |
| 5 | M=1 int4 A-prep split across the team: Hadamard blocks per thread, amax + quantize in 64-element chunks | 8.02 | **keep** +1.6% |
| 6 | bf16 dot four rows at a time (independent double chains, each in order) | 8.10 | **keep** +1.1% |
| 7 | skip `rknn_matmul_set_io_mem` when the context already holds that A/C buffer | 8.24 | **keep** +1.7% |
| 8 | weights <= 1M elements as one segment on one core | 8.20 | discard (slower) |
| 9 | bf16 x4 with register accumulators (the array spilled) | 8.25 | discard (noise) |
| fix | accept empty mul_mats; abort on packed read-back (see below) | 8.24 | **keep** (bug fix) |

**Where the time went, and why #3 dominates.** perf cannot see the main
thread while it blocks in `rknn_matmul_run`, so the backend got a
`RKNPU_PROFILE` wall-time split. Baseline token (165 ms): 123 ms NPU runs,
16 ms prep/dequant, 26 ms CPU ops. A no-code sweep then showed
`OMP_WAIT_POLICY=passive` alone giving +11.7%: with `-t 4` on four cores
the main thread, three libgomp workers and two dispatch-pool workers
share the cores, and spinning libgomp workers delayed the pool workers'
wake-up after each segment. Since the 2026-10-02 `set_n_threads` fix the
backend's team matches ggml's, so the segments can run on the
already-spinning workers without the #3 respawn churn — #3 turns the
spinning into the dispatch itself, and beats the env var (7.78 vs 6.74).
Prefill is unaffected by #3 and gained from #2 (pp128 45 -> 72 t/s: the
scalar bf16 loop was on the prefill path too).

**Where it stands.** Per token (~121 ms): NPU runs 98 ms, prep 8 ms, CPU
ops ~17 ms. `rknpu2-run-latency-probe.c` measures 32.9 us fixed per
`rknn_matmul_run`, 11.1 GB/s per core and **26.6 GB/s aggregate on three
cores**; streaming E4B's 2.64 GB of W4A4 weights at that rate is ~99 ms,
so the NPU share is at the board's memory-bandwidth ceiling. A context
spanning all three cores (`RKNN_NPU_CORE_0_1_2`) is rejected by the driver
for matmul, so one run per node is unavailable. Raising the 8192 K limit
(ffn_down runs as two K-segments) would cut runs but changes numerics:
each K-segment quantizes its own activation slice. Remaining bit-identical
headroom is the ~25 ms of CPU work, mostly the BF16 `per_layer_model_proj`
(memory-bound now) and attention.

**The guard caught a pre-existing server crash.** Iteration 9's decode
check segfaulted; a 12-run stress loop with core dumps put it at ~50%, in
the pre-loop build too (4/8). Cause: when a ubatch produces no logits,
llama-server builds the output projection with zero rows; `supports_op`
rejected that empty op, the scheduler moved it to the CPU and first
copied the whole weight out of the RKNPU buffer — E4B's tied
`token_embd.weight`, 713 MB requested from a 671 MB NPU-packed allocation.
The read ran 42 MB off the end. The copy was never used (zero rows), so
output was always correct, and llama-bench never builds such a batch; the
only other symptom was the CPU compute buffer silently growing from 118 to
680 MiB, also in perplexity runs. Fix: accept empty mul_mats (a no-op
`graph_compute` already skips), and abort with a message on any
out-of-bounds read-back of a packed weight. After: 0/12 crashes, compute
buffer back to normal, speed unchanged.

### 1d. Prefill loop, part 1 — the quality gate was the wrong instrument (2026-10-05)

Goal: E4B prefill on the NPU, metric `llama-bench -p 512 -n 0 -r 3 -t 4`
(taskset 4-7, default W4A4), baseline **68.4 t/s** (noise 68.2-68.4).
Harness: `rknpu2-autoresearch/prefill/`.

**Prefill is CPU-bound.** `RKNPU_PROFILE` over pp512: NPU runs 11% of wall
time, backend prep/dequant 33%, CPU ops outside the backend 56%. perf:
`ggml_vec_dot_bf16_x4` 33.5% (the BF16 `per_layer_model_proj`, a
2560x10752 GEMM done one column at a time), int16 dequant 19.3%, CPU flash
attention 16.4%, FWHT 6.6%.

**Three ways to take the BF16 tensor off the scalar path — all +32-42%:**

| # | Change | pp512 | 8-chunk PPL (base 35.0480) | 32-chunk PPL (base 26.8771) |
|---|---|---|---|---|
| 1 | BF16 weights on the NPU as W8A8 per-channel | **97.4** | 35.1168 (+0.20%) | — |
| 2 | BF16 weights on the NPU as W16A16 | 97.4 | 34.2145 (−2.38%) | — |
| 3 | 4x4 NEON bf16 GEMM tiles on the CPU, fp32 accumulation | 90.6 | 35.9984 (+2.71%) | 26.8168 (−0.22%) |

All three failed the loop's original quality gate (8-chunk PPL within
0.1%) and were reverted. #3 is the tell: the operator was checked end to
end through ggml against a double reference (worst relative error 5e-8,
1.25 M outputs, real shapes), yet the 8-chunk PPL moved +2.7% — and the
32-chunk PPL moved −0.22%. A 1e-7 perturbation cannot be a real 2.7%
quality change.

**Why: W4A4 output is chaotic.** KL divergence over 4 chunks
(`llama-perplexity --kl-divergence`), first against the W4A4 baseline's
own logits:

| vs W4A4 baseline logits | Mean KLD | Same top token |
|---|---|---|
| pure CPU | 0.738 | 70.4% |
| #3 (1e-7 change in one op) | 0.409 | 75.2% |
| #1 | 0.392 | 74.1% |
| #2 | 0.406 | 74.7% |

A 1e-7 change in one operator flips the top token at a quarter of the
positions: perturbations are amplified through the 4-bit activation
rounding. The baseline's logits are one sample of that noise, not a
reference. Against the **CPU** logits (the true model):

| vs CPU logits | pp512 | Mean KLD | Same top token |
|---|---|---|---|
| W4A4 baseline | 68.4 | 0.591 ± 0.029 | 70.4% |
| #1 BF16 → W8A8 | 97.4 | 0.580 ± 0.025 | 71.0% |
| #2 BF16 → W16A16 | 97.4 | 0.576 ± 0.025 | 69.8% |
| #3 BF16 GEMM | 90.6 | 0.597 ± 0.027 | 70.5% |

All three are as close to the true model as the baseline. Two findings
that outlive this loop:

- **8-chunk PPL cannot gate W4A4 changes.** It scatters ±3% for any
  non-bit-identical change; a tolerance below that passes or fails at
  random. 32 chunks is far steadier.
- **W4A4 agrees with the CPU's top token only ~70% of the time** (KLD
  ≈ 0.59) although its 32-chunk PPL is at parity (26.88 vs 27.01).
  Perplexity parity hid real per-token divergence; this matters when
  choosing W4A4 over W8A8 for E4B.

**New quality gate (agreed 2026-10-05)**, replacing "8-chunk PPL within
0.1%": 32-chunk PPL ≤ +2% over the W4A4 baseline (≤ 27.41; reference
points: Q8_0 ~+0.1%, Q5_K_M ~+1%, Q4_K_M ~+2-3% over FP16), **and** mean
KLD vs the CPU logits ≤ 0.65 (baseline + 2 SE) with same-top ≥ 68%. Unit
tests and the 3-run server stability check stay unchanged.

### 1e. Prefill loop, part 2 — E4B pp512 68.4 -> 161.5 t/s (2026-10-05)

Continuation of #1d under the agreed gate (32-chunk PPL ≤ +2%, KLD vs CPU
≤ 0.65, same-top ≥ 68%, 3 server runs, and from #11 on the NPU
flash-attention hardware test). 25 iterations, 12 kept; full log in
`rknpu2-autoresearch/prefill/results.tsv`.

| # | Change | pp512 | Exact? | Verdict |
|---|---|---|---|---|
| 0 | baseline | 68.4 | — | — |
| 4 | BF16 `per_layer_model_proj` on the NPU (W8A8 per-channel) | 97.8 | no (KLD 0.580 vs 0.591) | keep |
| 5 | native INT16 dequant, four rows per tile | 108.0 | yes | keep |
| 9 | per-thread scratch for A-prep rows, transformed row read in place | 109.6 | yes | keep |
| 10 | **pipelined prefill**: 256-row chunks, A-prep and dequant on the OpenMP team overlap NPU runs on async threads | 120.3 | yes | keep |
| 11 | **flash attention on the NPU** (see below) | 139.1 | no (FP16 attention) | keep |
| 14 | NEON tanh-GELU for geglu (ggml used a scalar fp16-table gather) | 143.7 | no | keep |
| 15 | radix-4 FWHT (two butterfly stages per pass) | 146.8 | yes | keep |
| 16 | NEON softmax for the NPU attention rows | 154.5 | no | keep |
| 17 | pipelined path stores on the first K-segment, no serial M×N memset | 158.2 | ±0 sign only | keep |
| 18 | fused sign multiply + Hadamard transform | 159.4 | yes | keep |
| 21 | NEON rms_norm (four double accumulators) + fused mul | 160.2 | PPL identical | keep |
| 23 | radix-8 FWHT (three stages per pass) | 161.5 | yes | keep |

Discarded, with the reason: store-first dequant *without* the pipeline
(#6, −6.5%: the memset's streaming warm-up beat strided cold stores — the
same idea won inside the pipeline as #17); FlashAttention CPU Q tile
128/32 (#7/#8, 64 is optimal: a 128×256 f32 tile is 128 KB > 64 KB L1);
tile-major dequant over all chunk rows (#12, −10%: strided dst rows thrash
instead); NPU attention with both KV heads' matmuls concurrent (#13, −5%)
and with the next head's Q·Kᵀ on a helper thread overlapping softmax (#20,
−9%: a fifth thread on four cores competes with the spinning OpenMP team);
pipeline chunk 128 (#19, noise); GELU reciprocal estimate (#22, no gain —
and NaN when exp overflows: recpe(inf) = 0, inf·0); 8-row dequant blocks
(#24, −3%); hoisted channel scales alone (#25, +0.2%, below threshold).

**NPU flash attention (#11).** `rknpu2-attention-probe.c` priced the
matmuls at ~135 ms per 512-token prefill on one core against ~1.3 s of CPU
flash attention (go). Per KV head, the rk2 query heads' rows are stacked
(GQA): S = Q·Kᵀ with K rows as a `TP_NORM` B, softmax (scale, softcap,
mask) on the CPU team, O = P·V with V rows as a `NORM` B. Prefill only
(≥ 32 query rows); F16 K/V; no ALiBi/sinks. It took three tries, and the
first two are the instructive part:

1. *+14%, guard passed* — but the guard's perplexity was bit-identical to
   the previous build, which FP16 attention cannot produce. llama-perplexity
   evaluates four sequences per batch (`ne[3] = 4`), which the first version
   rejected, so the guard measured the CPU path. Fixed by supporting
   `ne[3]` with ggml's K/V/mask broadcast.
2. With the NPU path actually exercised, perplexity was **3.36 million**.
   `test-rknpu2-flash-attn.cpp` (RKNPU vs CPU backend at Gemma-4 shapes)
   showed every output row zero: Q·Kᵀ was right, P·V returned zeros. The
   driver converts a non-native B **at `rknn_matmul_set_io_mem` time**, so
   data written into an already-bound B buffer is never seen. Re-binding B
   after each write fixes it; all cases now match the CPU to ≤ 1e-3.
3. Correct version: 139.1 t/s, PPL 26.65 (−0.8%), KLD 0.597.

Enabling it also exposed a latent bug class: `init_tensor`, `set_tensor`,
`get_tensor` and `get_tensor_real_ptr` decided packing from the dtype alone,
so compute-buffer tensors (which reuse offsets) could get or look up packed
NPU allocations, and `get_alloc_size` could size them below `ggml_nbytes`.
All four now treat COMPUTE buffers as plain memory, and the size is never
below `ggml_nbytes`.

**Final state** (E4B, default W4A4, `-t 4`, taskset 4-7): pp512 **161.5**,
pp128 **150.4**, tg64 **8.63** (decode +4.7% as a side effect: the BF16
projection, GELU and RMS norm also run per token), NPU + MTP n=1 **8.90**
(was 8.49; output identical to the no-draft run). Quality: 32-chunk PPL
27.24 (+1.4% vs 26.88), KLD vs CPU 0.587 (baseline 0.591), same-top 72.1%
(baseline 70.4%).

**Where prefill time goes now:** native INT16 dequant ~31% of samples
(memory traffic: int16 C read + fp32 dst write per output element), FWHT
~11%, GELU 8%, sign-multiply+transform 6%, RMS norm 5.5%, residual adds
4%, NPU attention ~3% CPU. The NPU itself is no longer the limit at
prefill; the CPU-side dequant of its INT16 output is.

### 1f. Prefill loop, part 3 — CPU-side work: pp512 161.5 -> 203.4 t/s (2026-10-05)

Goal: shrink the CPU share of the pipelined W4A4 prefill (dequant of the
NPU's INT16 output, transform, the ops between matmuls) or move it into
the backend. Same gate as #1e (32-chunk PPL ≤ +2%, KLD vs CPU ≤ 0.65,
same-top ≥ 68%, NPU flash-attention test, 3 server runs); keep threshold
+0.8 t/s over the best so far. 25 iterations, 6 kept, **every keep
bit-identical** (PPL 27.2432, KLD 0.587297, same-top 72.059% throughout).
Log: `rknpu2-autoresearch/cpu-npu/results.tsv`.

| # | Change | pp512 | Verdict |
|---|---|---|---|
| 0 | baseline (end of #1e) | 161.6 | — |
| 1 | GEGLU in the backend (`RKNPU_GLU`), so [gate, up, GLU, down] stay in one NPU split | 173.4 | keep |
| 6 | up matmul's dequant fused with GEGLU: up rows go to a thread-local buffer, GLU written directly (`RKNPU_FUSE_GLU`); `-falign-functions/loops=64` for the backend | 179.9 | keep |
| 8 | `supports_buft` accepts host buffer types: no scheduler copies of CPU-produced activations into the NPU split | 184.7 | keep |
| 10 | DC ZVA on the dequant store path: destination lines zero-allocated, no read-for-ownership | 189.2 | keep |
| 16 | RKNPU buffer type reports `is_host` (`RKNPU_HOST_BUFFERS`): CPU splits read NPU outputs in place, no copy-back | 195.4 | keep |
| 18 | K-segmented weights (ffn_down, 8192+2048) transform only their own Hadamard blocks | 200.3 | keep |
| 23 | **FFN gate's dequant deferred into up's fused GEGLU** (`RKNPU_DEFER_GATE`): gate's INT16 C is kept per chunk and dequantized into a thread-local row next to up, so gate's FP32 output never goes to DRAM and back | 207.3 | keep |

**Code layout matters (#2–#6).** Fusion variants first measured ~10%
*slower* than the unfused build, and so did adding unrelated dead code.
Pinning the backend's function and loop alignment to 64 bytes
(`-falign-functions=64 -falign-loops=64` in its CMakeLists) removed the
effect, and the fused build then won. Single-digit-percent differences
between builds still carry this noise (#24 vs #25 below): a same-binary
A/B behind an env switch is the reliable comparison.

**Why deferring the gate wins (#23).** A standalone microbenchmark of the
dequant kernel (4 threads, cold caches, `MC = 256`) found that it costs
**~1.0–1.2 ns per element at N = 10240, against 0.27–0.33 at N ≤ 2560**.
That is ~5 GB/s, while reading C alone in the same pattern runs at ~10
GB/s and writing the destination alone at ~28 GB/s. The collapse comes
from mixing the page-hopping C reads (the native layout puts each 8-column
tile 4 KB apart) with many strided destination streams, and it hits the
two N = 10240 FFN matmuls hardest. Things that did *not* fix it in place:
16-row blocks (#22, −6%; a sweep of 4/8/16/32/64 rows gave
200.8/190.9/187.8/191.5/187.9), column-split blocking, a padded stride.
Not writing the gate at all does fix it: about 21 MB less DRAM traffic per
layer per 256-row chunk.

Microbenchmark data (`rknpu2-dequant-microbench.cpp`, ns per output
element, 4 threads, best of 15, 4-row blocks unless noted):

| Case | N=10240 | N=2560 | N=2048 | N=512 |
|---|---|---|---|---|
| store (first K-segment, with DC ZVA) | 1.14–1.21 | 0.31–0.34 | 0.27–0.30 | 0.27 |
| store, ZVA removed | 0.97 | 0.29 | 0.27 | 0.24 |
| store, dst stride N+16 | 0.99 | — | — | — |
| accumulate (read-modify-write dst) | 0.93 | — | — | — |
| read C only, same pattern | 0.21 | — | — | — |
| write dst only, sequential rows | 0.14 | — | — | — |
| store, 2/16-row blocks | 0.93 / 1.13 | 0.37 (2 rows) | — | — |
| store, columns split in 4/8 parts | 1.11 / 0.95 | 0.33 | — | — |

ZVA costs in isolation but won in the pipeline (#10), where it removes
reads that compete with the NPU. No re-blocking gets N=10240 near the
narrow shapes.

Tools added: `rknpu2-dequant-microbench.cpp` (build line in its header) and
`rknpu2-autoresearch/cpu-npu/mtp-check.sh` (MTP n=1 output identity plus
t/s on the board). Profiles came from `perf record -e cycles:u -F 1999
-D 12000` around `llama-bench -p 512 -r 6`: user-mode sampling needs no
sudo, and the delay skips model load. `sudo perf` resolved no symbols here.

Discarded, with the reason. Backend RMS_NORM/ADD/MUL (#7, #9, #19): ggml-cpu's
fused `rms_norm_mul` beats separate passes even with no copies left.
DC ZVA on the GEGLU output (#11). NEON binary ops in ggml-cpu (#12).
ggml-cpu alignment pinning (#13). Fusing FWHT h=1,2 with the sign multiply
(#14): it is off the critical path. Deferring collect by a chunk (#15):
DRAM contention. Gemma-4 Q/K/V emitted back to back (#17). Pipeline chunk
128 (#20): extra B reads. ffn_down's two K-segments merged into one
pipelined pass with a two-source dequant (#21): verified to fire, but
200.7/201.3 off vs 200.7/201.8 on, so noise. Gate+up+GEGLU in one tile
pass with no row buffers (#24): same-binary A/B +0.8%, but below threshold
against the best build. GEGLU divide replaced by reciprocal estimate plus
Newton steps (#25): slower.

**Final state** (E4B, default W4A4, `-t 4`, taskset 4-7): pp512 **203.4**
(+26% over #1e, 2.97× the pre-#1d 68.4), pp128 **169.7**, tg64 **8.65**.
NPU + MTP n=1: **8.65 t/s** (was 8.90 in #1e, 8.49 before #1e), output
identical to the no-draft reference (4/4) with the host-flagged RKNPU
buffers. Quality unchanged from #1e.

**Where prefill time goes now** (perf, user cycles): `dequant_acc` 23%
(q/k/v/o/down collects plus the deferred gate and up rows), GEGLU math 11%,
FWHT 10% + sign/transform 6.6%, ggml-cpu `rms_norm_mul` 7% and `add` 5%,
INT4 quantize 4%, NPU-attention softmax 5%, libgomp waits ~11%. The NPU is
still fully hidden. Possible next steps:
- a narrower C type (the NPU cannot apply dequant scales: INT8→FP32 matmul
  type 9 aborts at run, INT8→INT8 saturates);
- deferring the other single-consumer outputs (q/k → per-head norm);
- the tile-pass GEGLU (#24) re-tested behind an env A/B.

### 1g. Prefill loop, part 4 — whole-block scheduling: pp512 208.3 -> 275 t/s (2026-10-06)

Goal: cut the CPU share named at the end of #1f (dequant, GEGLU, Hadamard,
ggml-cpu norms/adds) and keep the NPU busy. Same gate as #1e/#1f; keep
threshold +0.8 t/s over the best so far, or a same-binary env A/B where
one exists. 30 of the planned 40 iterations, stopped at a plateau (the
last ten gave 3 small keeps). 15 kept. Log:
`rknpu2-autoresearch/cpu-npu-2/results.tsv`.

**What's new in this loop.** The backend no longer runs matmuls strictly
node by node. Two fixed patterns now run as one schedule each, and that
cross-node scheduling is where most of the gain came from:
- The FFN, `[gate, up, GLU, down]`.
- Attention.

| # | Change | pp512 | Verdict |
|---|---|---|---|
| 0 | baseline (end of #1f) | 208.3 | — |
| 2 | ffn_down as a cross-node job: its A-prep runs inside up's fused GEGLU, so the GLU output is never written; down's first chunk starts before up's last collect; both K-segments run per chunk (two-source dequant) | 216.4 | keep |
| 3 | **the whole FFN as one block** (`RKNPU_FFN_BLOCK`): gate and up share one prep pass and one NPU batch per chunk; neither gate, up nor GLU is written | 219.9 | keep |
| 5 | FFN batch c runs gate/up chunk c with down chunk c-2; 128-row chunks (`RKNPU_FFN_MC`) | 222.9 | keep |
| 7 | **NPU attention with native A/C layout** (`RKNPU_FA_NATIVE`): QK+PV runs went from 331 to 186 ms per pp512 because the runtime no longer converts A/C serially; Q/P/S/O are moved in 64-row blocks (`RKNPU_FA_ROWS`) | 229.7 | keep |
| 8 | **NPU attention software-pipelined** over (sequence, KV group) items on rotating cores (`RKNPU_FA_OVERLAP`) | 241.5 | keep |
| 9 | FFN GEGLU straight from the gate/up INT16 tiles, with no row buffers (`RKNPU_GEGLU_TILES`) | 246.7 | keep |
| 11 | ggml-cpu fuses RMS_NORM + MUL + ADD (post-norm plus residual) | 251.2 | keep |
| 12 | ... plus a following MUL by a one-element tensor (Gemma-4 layer scale) | 254.4 | keep |
| 13 | ... plus the next RMS_NORM + MUL of the result, per row (`GGML_CPU_DISABLE_FUSION_CHAIN=1` turns it off) | 257.2 | keep |
| 14 | software prefetch, 8 tiles ahead, in the GEGLU tile pass (`RKNPU_TILE_PF`) | 261.7 | keep |
| 16 | NPU attention with native B (`RKNPU_FA_NATIVE_B`): K/V written in the NPU tiling and B bound once; no per-item driver conversion | 261.9 (A/B +2%) | keep |
| 17 | attention output gathered into a local block, then written as whole rows | 269.2 | keep |
| 20 | FFN preps 2 gate/up chunks during batch 0 (`RKNPU_FFN_PREP_AHEAD`) | 270.1 | keep |
| 21 | attention submits PV(it) before finishing item it-1 | 273.7 | keep |
| 24 | softmax skips leading/trailing -inf mask runs (exact zeros, lane-aligned) | 275.1 | keep |

Every keep except #8 left PPL32, KLD and same-top bit-identical. The
ggml-cpu fusions needed care (see below).

**Final state** (E4B, default W4A4, `-t 4`, taskset 4-7): pp512 **273.7-276.1**
(+32% over #1f, 4.0× the pre-#1d 68.4), pp128 **188.7**, tg64 **8.78**.
NPU + MTP n=1 runs at **9.29 t/s** with output identical to the no-draft
reference (4/4). Quality: PPL32 27.2382 (was 27.2432, see the core note
below), KLD 0.587297, same-top 72.06%.

**Why the FFN wanted cross-node scheduling (#1-#5).**
- Iteration 1 moved down's A-prep into up's collect and lost 2.4%. Down's
  prep had been what hid down's first NPU chunk; with it gone, nothing
  overlapped that chunk.
- A per-node wait profile (`RKNPU_PROFILE` now reports `npu_wait`) then
  showed the imbalance. gate and down spent 870 and 725 ms blocked on the
  NPU while up did nearly all the CPU work.
- The fix is to schedule the three matmuls as one pipeline.

The final FFN schedule, for 4 chunks:

    prep 0,1,2 | G0+U0 || fused 0 (gate+up dequant, GEGLU, down Hadamard+INT4) | G1+U1 ||
    fused 1 + prep 3 | G2+U2+D0 || fused 2 + collect D0 | G3+U3+D1 || ... | D3 || collect D2 | collect D3

A final per-step measurement put the FFN's NPU work (~850 ms per pp512,
about 2/3 of the INT8 peak) roughly equal to its CPU work. The FFN is
now close to balanced. Per-run overhead is small (32.9 us per run); the
128-row chunks re-stream about 250 ms of weights, but larger chunks
lengthen the exposed head and tail and measured slower.

**Attention (#7, #8, #16, #17, #21, #24).** A stage profile per pp512
started at 510 ms:
- QK run 194 ms, PV run 137 ms;
- softmax 73 ms;
- B re-bind 44 ms (the driver converting K/V);
- copies 60 ms.

With `AC_layout = NORM`, the runtime converts A and C on one thread
inside every run. Going native helped only after the moves became
line-friendly:
- Row-by-row 16-byte gathers were 3x slower than the conversion (32 KB
  stride, L1 set aliasing). 64-row blocks fixed it.
- The output stage had 64 scattered destination rows per block, because
  rows interleave by head. Writing whole rows from a local block gave +2.7%
  on its own.
- Pipelining the KV groups gave +5.2%. Groups run on their own core and
  context; items rotate over cores 0-2, so at most items it-1, it and it+1
  are in flight and never share contexts.

Native B removes the remaining `rknn_matmul_set_io_mem` calls, the
driver's serial B conversion. It also stops binding while other runs are
in flight.

**Open issue, pre-existing: NPU attention output depends on the core.**
- With the pipeline, PPL32 went from 27.2432 to 27.2382. It is
  deterministic, run to run.
- The pipeline is not the cause. Running sequentially with rotated cores
  gives 27.2382 as well, and even a static shift of which core serves
  which KV group (old sequential path, `g % 3` → `(g+1) % 3`) moves chunks
  5-8. The first 4-sequence batch is always identical; later batches
  differ.
- The extended `test-rknpu2-flash-attn.cpp` (now 12 cases, two warm
  rounds, a 4-sequence prefill case) matches the CPU within 1e-3 and is
  identical across core assignments. So the effect appears only inside the
  model, after earlier batches have run.
- Something carries state between calls in a core-specific way, or the
  cores differ in some edge case (masked or stale cells). Not explained;
  the magnitude is ±0.02% PPL.

**ggml-cpu fusions (#11-#13): exact only with care.**
- `ggml_cpu_try_fuse_ops` now fuses RMS_NORM + MUL + ADD (+ MUL by a
  scalar) and the following RMS_NORM + MUL into one row pass.
- The first version changed PPL (27.0713, KLD 0.582). GCC contracts the
  intrinsic `vmulq_f32` + `vaddq_f32` into an FMA (`-ffp-contract=fast` is
  the GNU default), which rounds differently from a separate ADD node. An
  empty `asm` register barrier between the multiply and the add (and a
  `volatile` in the scalar tail) restored bit-identity.
- The chained second norm reuses one shared sum-of-squares routine, so
  its accumulation order matches the standalone norm exactly.

**Dequant memory behaviour.** The GEGLU tile pass reads two INT16 C
streams, one 64-byte line per tile, with tiles 2 KB apart. The hardware
prefetcher does not follow that, so a software prefetch 8 tiles ahead was
worth +1.7%. The same prefetch in the plain dequant kernels was slower;
those are bound by their destination writes. Replacing the GELU math with
a plain multiply (a diagnostic build) would gain only 5%, so the tile pass
is mostly memory-bound. An FP16 GELU gained 0.25% and was not worth the
numerics change.

Discarded, with the reason:
- **Q/K/V as one shared-input block** (#6): it fired on every layer with
  no gain; those nodes are CPU-bound and re-reading the row three times
  hits cache.
- **Per-layer-input block** `[inp_gate, GELU, MUL, proj]` (#10): +0.4%.
  The GELU reproduces ggml's fp16 table exactly. Those nodes cost mostly
  their K=2560 prep.
- **FWHT h=1,2 merged** (#18): blocks are L1-resident.
- **Plain pipelined nodes at 128 rows** (#23).
- **Third attention pool worker** (#22).
- **INT4 quantizer single clamp** (#27): exact, no speed change.
- **C invalidation in the runner threads** (#28): +0.6% A/B. The main
  thread spends about 5% of wall time in `rknn_mem_sync`; moving the C
  half off it buys too little.
- **Non-cacheable A buffers** (#29): -5%, because scattered 16-byte
  uncached writes are slow.
- **libgomp spin tuning**: the default is best.
- **Q/K/V sharing one Hadamard sign vector per layer** (#26, quality
  probe): it would let K and V reuse Q's prepped INT4 rows, but PPL32 is
  29.88 (+9.7%), KLD 0.607. The shared rotation gives Q, K and V the same
  quantization error, which correlates the errors in QK^T. This narrows
  #3h's finding: even within one layer's attention, E4B needs per-tensor
  signs.

**Where prefill time goes now** (perf, user cycles):
- GEGLU tile pass 16%;
- FWHT 11%;
- dequant 9% + down's dequant2 7%;
- ggml-cpu norms 10%;
- INT4 quantize 4.5%, softmax 4%;
- libgomp waits about 12%, plus 4-5% system time (cache maintenance).

The FFN is near NPU/CPU balance. Attention and the single-matmul nodes
(Q, K, V, O, inp_gate, proj) are CPU-bound with an idle NPU. The next
gains need structural moves:
- Q/K norm and RoPE inside the backend, so attention becomes one block;
- or W8A8 for the CPU-bound attention projections (no Hadamard prep;
  double NPU work; needs name-based routing and a pipelined W8A8 path).

Process note: `pkill -f <pattern>` matched the shell running it and killed
the command (exit 144). Use `pkill -f "^build/bin/..."` or pgrep with an
anchored pattern.

### 1h. Attention as one block, steps 1-2: Q/K/V norms and RoPE in the backend (2026-10-06)

Plan (from #1g's outlook), measured stage by stage:
1. per-head Q/K/V RMS norms fused into the projection's dequant;
2. RoPE fused after them;
3. KV-cache writes and attention fill straight from registers;
4. a block schedule with block-causal overlap.

Probe first. The ops to absorb are about 5% of CPU cycles (Q/K norm 2.2%,
RoPE 1.6%, cache write 0.85%, V norm 0.1%), and 13% of wall time runs
outside the NPU backend. So steps 1-2 alone were expected to be small;
their job is to make Q/K/V one backend piece.

**Step 1: per-head norms.**
- The backend takes RMS_NORM, and its weight MUL, only when the norm's
  input is a reshape of a matmul output.
- The arithmetic is copied from ggml-cpu: the same double-precision sum of
  squares over four 2-lane accumulators, and `(x*scale)*w`.
- The pipelined collect dequantizes each 4-row block into a thread-local
  buffer, normalizes it per head, and writes the norm (or MUL) output. The
  projection itself is never written. Stand-alone handlers cover decode
  and other non-pipelined cases.

First result: **3% slower**, with 24% more graph splits. llama.cpp pins
every tensor named `"norm"` to the layer's device when the model is fully
offloaded (`llama-context.cpp`, a FIXME scheduler workaround). For this
backend (an ACCEL device) that device is the CPU. `build_norm` names its
intermediate `"norm"`, so each Q/K RMS_NORM was pinned to the CPU while
its weight MUL went to the NPU, adding two backend switches per
projection. V's norm (a bare `ggml_rms_norm`) was unaffected.

Fix: `gemma4.cpp` builds the Q/K norms as `ggml_mul(ggml_rms_norm(..), w)`,
the same ops without the pinned name. Result: bit-identical,
same-binary A/B **+0.6%** (276.5 vs 274.9).

**Step 2: RoPE.**
- ggml-cpu's rotation is plain scalar C++, compiled with fused
  multiply-adds, and GCC contracted the two modes differently.
- `test-rknpu2-rope.cpp` (new; RKNPU vs CPU backend, bit for bit, 5 cases)
  showed NORMAL exact with `y0 = fma(-x1, s, x0*c)`. NEOX differed by one
  rounding in 8-16% of elements.
- Trying all four orders found NEOX = `fma(x0, c, -(x1*s))`,
  `fma(x0, s, x1*c)`.
- The cos/sin table copies `ggml_rope_cache_init` (powf, cosf/sinf,
  frequency factors). Only `ext_factor == 0` (no YaRN) is supported;
  anything else stays on the CPU.
- The guard now runs the RoPE test, since a different compiler could
  contract differently.

Fused version: one cos/sin table per token, applied in place after each
head's norm, while the row is in L1. The projection, normed and pre-RoPE
tensors are never written.

Same-binary A/B, twice:

| RoPE | pp512 |
|---|---|
| on CPU | 273.4 |
| standalone in backend | 275.8 |
| fused | **277.1** (+1.4%) |

Q, K, V, their norms and both RoPEs now form **one NPU-backend split** per
layer, down from five alternating with the CPU; total graph splits fell
21%. Quality is bit-identical (PPL32 27.2382, KLD 0.587297); pp128 and
decode are unchanged.

**Steps 1-2 together: ~+2% pp512**, in line with the probe. What remains
between the projections and attention is the two KV-cache writes
(`SET_ROWS`, CPU), the next step.

**Step 3: KV-cache writes.**
- The backend takes `SET_ROWS` when its source is a view of a fused K (roped)
  or V (normed) output, and does the cache write inside the projection's
  dequant: same F16 conversion as ggml-cpu (round to nearest even), at each
  token's cell index.
- In graph order K's cache write comes *after* V's projection. So it is
  computed early, only if no computing node in between touches that cache,
  and then marked done; the done-set is cleared after every graph, since
  tensor pointers are reused.
- When nothing else reads the FP32 K/V rows they are not written at all
  (normed and roped in place in the thread-local row).
- Decode and other non-pipelined cases use a standalone handler.

Same-binary A/B: **288.1 vs 280.3 (+2.8%)**, verify 287.5. pp128 went from
188.2 to 192.0; decode is unchanged. Bit-identical (PPL32 27.2382, KLD
0.587297), and graph splits fell another 4%.

The gain is larger than the 0.85% of cycles the CPU cache copy cost. It
also removed the CPU piece between V's projection and attention, and the
FP32 K/V write-and-reread. Per layer, attention now runs from the Q/K/V
projections through the output projection in one NPU-backend piece.

Not done: writing K/V straight into attention's native B layouts. The
attention fill reads the whole cache range, not just this batch's rows,
and that copy is about 1%.

**Steps 1-3 together: ~+5% pp512** (about 277 → 288).

**Step 4 (block-causal attention schedule): stopped at the go/no-go probe.**
Per pp512 (~1.78 s), the attention section takes ~680 ms (38%):

| Part | Time | Exposed NPU time |
|---|---|---|
| Q/K/V projections (with fused norm, RoPE, cache write) | ~290 ms | ~0 |
| NPU flash attention (fill 43, softmax 116, output 44, QK wait 22, PV wait 32) | ~261 ms | ~53 ms |
| output projections | ~130 ms | ~0 |

The section is CPU-bound with the NPU mostly idle, so overlap can only hide
the ~55 ms of exposed NPU time: a ceiling of ~2-3%, not the 5-10% guessed
before measuring. A full block-causal scheduler (contexts per chunk shape,
the cache read across chunks, the open core-dependence of NPU attention)
is not worth that.

Open options, cheapest first:
1. **A small cross-node overlap** for the measured waits: attention's first
   QK starts once K is in the cache (during V's projection), and the output
   projection's prep overlaps the last PV. Most of the ~2-3%.
2. **Cheaper softmax** (e.g. FP16 exponentials). A numerics change, so it
   needs the tolerant gate; up to ~3%.
3. **W8A8 for the attention projections:** no Hadamard prep on the CPU, at
   twice the NPU work on an idle NPU. Needs per-name pipeline routing and a
   pipelined INT8 path; unmeasured, maybe ~5%.

**Step 5: where the exposed attention wait sits, and a cheaper softmax
gather (2026-10-07, new dev machine).**

The re-profile (`dbgfa2.py` split by item position, per pp512) located the
~51 ms of exposed NPU time. For pp512 there are only **2 items** per layer
(`n_seq = 1`, `n_kvh = 2`), so nearly all of it is at the pipeline's head
and tail:

| Stage | ms / pp512 |
|---|---|
| fill item 0 / item 1 | 19.3 / 21.7 |
| QK wait, item 0 (head) | **19.6** |
| QK wait, item 1 | 3.1 |
| softmax | 116.5 |
| PV wait, item 0 | 0.3 |
| PV wait, last item (tail) | **31.5** |
| output item 0 / last | 24.6 / 19.5 |

perf (user cycles) splits the softmax stage into the native gather/scatter
loop (3.3%) and the row math (2.8%). Two wastes:
- every 64-row block gathered all `n_kv` S columns, though the causal mask
  limits a block to its union `[lo, hi)`, ~56% of the columns on average
  for pp512;
- every row rescanned its mask for the `-inf` runs, scalar, once per head
  and per group, though the 4 heads and 2 groups share the mask row.

Change (`RKNPU_FA_RANGE`, default on): `[lo, hi)` is found once per mask
row per attention call, and the gather reads only the cells of the
block's union range. The softmax reads only `[lo, hi)` anyway, so the
output is unchanged.

Same-binary A/B, three pairs: **291.9 vs 287.1 (+1.7%)**. Guard
bit-identical (PPL32 27.2382, KLD 0.587297, same-top 72.06%, FA 12/12,
RoPE 5/5, server 3x 4/4).

Discarded: normalizing and converting P to FP16 in one pass instead of
two (`tmp *= inv`, then convert). Exact, but 291.7 vs 291.4, noise.

**Step 6: head-half items — NO-GO (2026-10-07).** This tried option 1's
goal without a cross-node scheduler. Each group item was split into head
subsets (`RKNPU_FA_SPLIT=2`: half-M contexts, 4 items per layer for
pp512), so the exposed first QK and last PV runs are half as long. The
second half copied the first half's native B with a `memcpy` (same K, N,
so the same layout) instead of redoing the K copy and V transpose.
Dynamic-M contexts were no option: one context runs one shape at a time.
- Exact: flash-attention test 12/12 with the split.
- Same-binary A/B, three pairs: **291.3 vs 292.3 (-0.35%)**.
- The stage profile (per pp512, with `RKNPU_FA_RANGE` on in both) shows
  why:

  | ms | split 1 | split 2 | delta |
  |---|---|---|---|
  | fill (Q + B) | 41.7 | 59.9 | +18.2 |
  | QK waits | 27.7 | 9.5 | -18.2 |
  | softmax | 80.3 | 99.7 | +19.4 |
  | PV waits | 26.8 | 10.9 | -15.9 |
  | output | 43.4 | 47.8 | +4.4 |
  | attention total | 236.8 | 246.9 | +10.1 |

  The waits fell as predicted, by 34 ms. But softmax does the same work
  and ran 24% slower once NPU runs overlapped it more. The likely cause is
  the CPU and the NPU's DMA competing for DRAM bandwidth. The B `memcpy`
  is at most ~8 ms of the fill growth, so sharing B through
  `rknn_create_mem_from_fd` cannot recover the loss.

**Lesson: exposed NPU time is not free to hide on this board.** Overlap
slows the memory-heavy CPU stages it overlaps. That makes option 1 (the
cross-node overlap, same ~50 ms) unlikely to pay as well. The remaining
attention gains must cut CPU work or memory traffic, not add overlap.
Code reverted.

### 1i. MTP decode on the NPU: 11.86 -> 17.12 t/s, output-identical (2026-10-07)

Goal: faster token generation for E4B. Start: plain decode tg64 8.78 t/s
(llama-bench), at the board's bandwidth limit. Per token, NPU runs take
~99.6 ms of ~114 ms, ~2.6 GB streamed at ~26.6 GB/s, plus ~213 backend
graph splits.

**Online research, condensed.**
- On RK3588, RKLLM runs LLMs as W8A8 only; this backend's W4A4 already
  reads fewer bytes per token.
- M=1 decode is GEMV-bound, so the NPU's TOPS don't matter; speculative
  decoding is the standard way past that limit on edge NPUs.
- Gemma-4's assistant (MTP drafter) ships **masked-embedding centroids**
  (vLLM `Gemma4MTPMaskedEmbedder`): 2048 centroids, the top 32 chosen, so
  32 x 128 = 4096 candidate tokens are scored instead of 262,144 (the same
  idea as FR-Spec). llama.cpp declared the tensors, but the converter
  dropped them and the graph never used them.

**Measure first.** The NPU verify batch is now nearly free (llama-bench
`-p 1,2,3,4,5,8 -n 0`, ms per batch):

| M | 1 | 2 | 3 | 4 | 5 | 8 |
|---|---|---|---|---|---|---|
| ms | 113.8 | 118.5 | 120.6 | 122.9 | 130.1 | 132.6 |

(It was 172 / 237 / 247 / 262 in #1b, before the #1c decode loop.) The MTP
sweep in #1b ran the server unpinned; pinned (`taskset -c 4-7`) gives:
no draft 8.49, n=1 10.92, n=3 **11.86** t/s, all 4/4 output-identical.

**Where an n=3 cycle went** (~220 ms for 2.27 tokens; trace timings, a
per-node profile, and a timer on the LM-head node):
- drafting: 26 ms per cycle (3 steps), mostly the drafter's full-vocab
  F16 LM head (262,144 x 256, 134 MB per step);
- **the target's LM head at M=4: ~86 ms**, against 21.4 ms at M=1.

**Root cause of the M=4 LM head.** E4B's tied `token_embd` is Q8_0, so the
output projection runs on the W8A8 pipeline (~671 MB of INT8 weight),
whose A/C layout was NORM. With NORM, the RKNN runtime converts C between
row-major and its tiling on one thread inside every run. That is cheap at
M=1 and ~64 ms per core at M=4. Standalone probes show the NPU itself
does not care about M: INT8 K=2560, N=87,392 per core runs in ~20.0 ms at
M=1, 2, 4 and 8, and three INT4 cores at once run in 10.6 ms for M=1, 2
and 4 alike. (The runtime also rejects INT4 contexts with N not a multiple
of 64; INT8 accepts multiples of 32.)

**Changes:**
1. **`RKNPU_W8A8_NATIVE`** (default on): W8A8 nodes with per-channel
   scales and 1 < M <= 32 use NATIVE A/C. INT8 A rows are quantized, then
   scattered into the native tiling. INT32 C is dequantized cell by cell
   with the existing per-channel kernel (`dequant_acc_int32_tiled_perchan`,
   exact by construction; unit test against the row-major kernel). M=1
   and prefill are unchanged. LM head at M=4: 85 -> **21.4 ms** per core
   run; PPL (b=4) identical on and off.
2. **Centroid draft logits** for `gemma4-assistant`:
   - The converter exports `masked_embd_centroids` and
     `masked_embd_ordering` (I32), plus `masked_embedding.centroid_count`
     and `.top_k`.
   - The loader registers the tensors with real ops (upstream had
     `GGML_OP_NONE`, which made the loader skip them).
   - The graph scores only the top-k centroids' tokens and fills the rest
     with -inf.
   - Draft generation went from 26 to 17 ms per cycle, with **identical
     drafts** (same acceptance per position) on the traced prompt. The
     drafter must be reconverted (`gemma-4-E4B-it-assistant-centroid-F16.gguf`
     on the board); an old GGUF falls back to the full LM head.

**Result** (pinned server, 4 prompts x 128 tokens, greedy; every cell
4/4 output-identical to no-draft):

| decode t/s | no draft | n=1 | n=2 | n=3 | n=4 |
|---|---|---|---|---|---|
| before | 8.49 | 10.92 | 10.53 | 11.86 | 8.93 |
| W8A8 native, full drafter | 8.51 | 12.45 | - | 16.44 | - |
| **W8A8 native + centroid drafter** | 8.22 | 12.83 | 15.31 | **17.12** | 16.69 |

MTP n=3 is now **2.0x plain decode**. Acceptance is unchanged (73 / 62 /
57 / 48% for n=1..4). Plain llama-bench tg64 is unchanged, since only
batches of 2-32 rows take the new path. Guard bit-identical (PPL32
27.2382, KLD 0.587297, FA 12/12, RoPE 5/5, server 3x 4/4).

**Draft length (`--spec-draft-p-min`, no code).** Stopping a draft once
the drafter's top probability falls under p_min lets longer drafts pay off:

| t/s (accept) | n=3 | n=4 | n=5 | n=6 |
|---|---|---|---|---|
| p_min 0 | 17.12 (57%) | 16.69 (48%) | - | - |
| p_min 0.3 | 17.21 (61%) | 16.84 (52%) | 17.32 (48%) | 16.64 (41%) |
| p_min 0.6 | 16.46 (79%) | 17.15 (76%) | **17.85 (74%)** | 17.70 (72%) |

Best: `--spec-draft-n-max 5 --spec-draft-p-min 0.6`, **17.85 t/s**, 2.1x
no draft.

**Holdout: MTP output is not strictly identical.** Six new prompts (code,
arithmetic, German, JSON, a story, a list), 256 tokens each, MTP n=5 p0.6
against no draft on the same build: **4/6 identical**. The story splits at
a near-tie ("...below the cliffs." vs "...below."); the arithmetic prompt
splits inside the hidden reasoning. Same with `RKNPU_W8A8_NATIVE=0`, so it
predates this work. Bisect:
- **The target's decode on the NPU path is batch-invariant.** A libllama
  harness (`batchdiff.cpp`, kept with the profiling patches) feeds the
  same tokens one at a time and in batches of 2-8. It also covers batches
  from position 0, a 60-token text, and MTP-like verify batches whose
  wrong drafts are rolled back with `llama_memory_seq_rm`. In every case
  the logits are **bit-identical**, with and without `cb_eval`.
- A first reading of "batch-dependent logits" came from llama-perplexity
  KLD (ub=4 vs ub=1: mean 2e-6, maximum 4.3e-5). That is the precision
  floor of its saved-logits format: ub=1 against itself gives the same.
- Loading the drafter while accepting no drafts (`--spec-draft-p-min 1.0`)
  gives 6/6 identical, so the server's MTP configuration alone changes
  nothing.
- The pure CPU path also diverges under MTP: 5/6 identical, the story
  splitting at char 4. Its decode really is batch-dependent (KLD ub=4 vs
  ub=1: mean 3.8e-4, maximum 0.047).

So on the NPU path the divergence comes from the MTP serving flow (draft,
verify and accept bookkeeping), not from the target's arithmetic. It is
open. Greedy MTP output should be treated as equivalent in quality, not
byte-identical. The earlier "output-identical" claims (#1b and above) hold
for the four 128-token bench prompts only.

**Drafter on the CPU** (`RKNPU_EXCLUDE_TYPES=f16`; the target has no F16
weights): n_max 5, p_min 0.6 gives **18.19 t/s**, 74% acceptance, 4/4,
against 17.85 with the drafter on the NPU's F16 path. This is the best
exact configuration.

**Lossy probe: LM head on W4A4.** `RKNPU_HYBRID=W4A4_HADAMARD` moves the
Q8_0 LM head (~671 MB, ~21 ms of the ~114 ms token) and the BF16
`per_layer_model_proj` to W4A4:

| | exact | W4A4 LM head + PLE proj |
|---|---|---|
| tg64 | 8.78 | **9.76 (+11.2%)** |
| PPL32 | 27.2382 | 27.2524 |
| KLD vs CPU (4 ch) | 0.5873 | 0.5981 |
| same top | 72.06% | 70.98% |
| MTP n=5 p0.6, drafter on CPU | **18.19** (74% accept) | 16.88 (61% accept) |

This is inside the tolerant gate, but it **costs MTP 7%**. The drafter
predicts the exact target, and the coarser LM head disagrees with it more
often. (Without `RKNPU_EXCLUDE_TYPES=f16` the global override also puts
the drafter's F16 weights on W4A4: 38% acceptance, 12.07 t/s.) Worth it
only for plain decode without drafting; not adopted.

### 2. Cooperative CPU+NPU decode — MEASURED: NO-GO (probe, 2026-08-10)

`rknpu2-coop-decode-probe` ran on the board (pinned clocks). Bandwidths do
sum, but the margin at decode-representative shapes fails the >= ~25% gate:

| Shape K x N | CPU solo | NPU solo | best coop (f_npu) | margin |
|---|---|---|---|---|
| 2048 x 2048 | 0.123 ms | 0.123 ms | 0.095 ms (0.375-0.5) | **+29%** |
| 2048 x 16384 | 0.986 | 0.587 | 0.560 (0.875) | +4.7% |
| 8192 x 2048 | 0.510 | 0.325 | 0.297 (0.875) | +9.3% |
| 4096 x 4096 | 0.483 | 0.315 | 0.294 (0.875) | +7.3% |

Only the small square shape clears the gate, and small nodes are a minor
share of per-token time. Aggregate peaks ~29-30 GB/s vs 26-28 NPU-solo:
real but not enough. **Verdict: not worth the backend complexity.**

The probe's real payoff is what it falsified: the NPU's per-node M=1
driver sequence (A set/sync + 3 parallel runs + 3 C syncs, weights beyond
cache) sustains **~28.6 GB/s aggregate** — pure weight reads for E4B would
be ~72 ms/token (~14 t/s), yet the backend decodes at 3.65 t/s (274 ms).
So the "~75% overhead" is NOT driver dispatch: it is backend/CPU-side
work between and around the matmuls (scheduler splits and copies, cache
lookups under mutex, thread oversubscription — the affinity sweep's t=4
gains point the same direction). **This re-ranks avenue #3 (overhead
surgery) from deprioritized to the main code lever**, starting with a
profile of a real decode step (perf record on llama-bench tg).

<details><summary>Original hypothesis (kept for context)</summary>

Bandwidths *sum* when different engines read different bytes
simultaneously. Layer pipelining cannot exploit this for a single stream
(layers are sequential), but **splitting each weight matrix along N** can:
NPU cores compute part of the output columns from the packed INT4 copy
while the CPU computes the rest from the original Q4_0 bytes — which are
*already host-resident* under `RKNPU_CPU_DECODE`'s dual-residency
mechanism (see `feat/mixed-precision-pipelines`).

- Combined streaming ~20+ GB/s → projected ~7–8 t/s on E4B.
- Work: in `ggml_backend_rknpu_graph_compute`, for M=1 nodes, launch the
  NPU segments and a CPU partial-GEMV concurrently, merge partial C, and
  load-balance the N split against measured per-engine rates. All within
  the backend; no scheduler changes.
- Risks: per-node join latency could eat the gain on a 250-node model
  (measure with a 2-node prototype first); CPU threads doing GEMV compete
  with the CPU-side graph ops between nodes.
- To our knowledge no RK3588 stack does this — upstream-worthy if it works.
- The 2-node prototype is built: `rknpu2-coop-decode-probe.c` (see
  "Prepared experiments"). Go/no-go gate: best coop split must beat the
  better solo path by >= ~25% in the probe, since the probe already pays
  the real per-node costs (persistent-pool join, per-node A set/sync +
  3 runs + 3 C syncs) but not the A-prep/dequant/graph-op CPU contention
  a real integration adds.

</details>

### 3. Backend overhead surgery — PROFILED: root cause found (2026-08-11)

The profiling chain (perf cycles → strace -c → gdb breakpoint backtraces,
all on verified decode steady state) attributed the NPU-decode overhead:

- perf cycles: ~95% of on-CPU samples in libgomp spin-wait; the CPU does
  almost no real work during NPU decode. Driver `ioctl`s: ~3% of wall
  time (~75k calls/30s at 13 us) — the NPU driver is NOT the bottleneck,
  confirming the coop-probe finding.
- strace -c: ~33k `clone3` in 30 s of decode — **~550 thread creations
  per token, ~1 per graph split** (E4B: 603 splits/token at bs=1
  NPU-decode; the routed path has 1 split at bs=1, which is why it never
  suffered).
- gdb on `pthread_create`: every spawn is
  `GOMP_parallel <- ggml_backend_rknpu_graph_compute`, with libgomp
  tearing down and re-creating its team around the backend's per-node
  OMP regions. The backend's regions request heterogeneous team sizes
  (default 8 for A-prep/C-dequant, `num_threads(3)` for the matmul runs)
  while ggml-cpu's regions use `-t` (4); libgomp responds by respawning
  workers region-to-region.

**Env-level fix, verified (E4B, W8A8, t=4, clean board):**
`OMP_NUM_THREADS=4` alone lifts NPU decode tg128 3.00 -> 4.27 (+42%).
Combined best config `taskset -c 4-7` + `-t 4` + `OMP_NUM_THREADS=4`:

| E4B config | pp128 | tg64 |
|---|---|---|
| NPU decode, pinned big-4 | 42.01 | 4.55 |
| routed, pinned big-4 | **42.62 ± 0.02** | **5.50** |

This also **resolves the "strict-taskset anomaly"** from the affinity
sweep (was pp 8.75 ± 2.10, now 42.62 ± 0.02): under `taskset -c 4-7`,
libgomp's default team is 4, but without `OMP_NUM_THREADS` the 8-thread
regions still thrash. Pinning without the env var is what was
pathological.

**Code fix, implemented and shipped (2026-08-11):** the backend no longer
forms OMP teams on the M=1 path — `if(M > 1)` on the A-prep and C-dequant
regions (a single row is a few us of NEON, serial), and the
`num_threads(3)` dispatch region replaced by a persistent 2-worker pool
(`rknpu_dispatch_pool`: caller runs segment 0, spin-then-sleep workers run
the rest). clone3 during decode: 33,030 -> 910 per 30 s window (-97%).
Greedy outputs bit-identical (W8A8 and W4A4), 170 prep-kernel checks
green. Measured (E4B, t=4, no env vars):

| Config | before | after |
|---|---|---|
| NPU decode tg128 | 3.00 | **3.84** (+28%) |
| NPU decode tg128, pinned | ~3.4 | **3.94** |
| W4A4 default tg | 3.65 | **4.31** (+18%) |
| routed pinned pp128/tg64 | needed env: 42.62/5.50 | **42.52/5.49 env-free** |
| W4A4 default pp128 | 34.44 | 33.39 (-3%) |

Honest trade-offs, both measured:
- The old code WITH `OMP_NUM_THREADS=4` still beats the pool on the pure
  NPU-decode path (4.27-4.55 vs 3.84-3.97): GOMP's team barrier dispatches
  onto threads that just finished the prep region and are still hot,
  where the pool pays a ~0.1 ms futex wake per node. The same latency
  explains the -3% on W4A4 prefill. Structural fix if NPU decode ever
  matters: have each worker do its own segment's set_io so dispatch
  overlaps the driver calls. Low priority — routed decode (faster on
  every model tested) is unaffected and at its best with no env at all.
- Spinning harder does NOT fix it (tried): longer spins + pre-waking
  workers before the job is ready steal the cores the prep regions and
  CPU graph ops need (routed pp 42.5 -> 37.9, pinned NPU tg halved).
- ~~Residual ~50 clone3/s remains~~ CORRECTED by the clean re-profile
  below: decode-window clone3 is exactly **zero**; the residual was
  load-phase spillover into the measurement window.

#### Re-profile of the churn-free path (2026-08-13): both decode paths are near their floors

Wall-time decomposition via an LD_PRELOAD shim timing every librknnrt
entry point (`rknpu2-driver-shim.c`; cycle sampling cannot see blocked
time), plus decode-only perf/strace windows. E4B, W8A8, t=4.

**NPU decode (3.90 t/s = 256 ms/token):**

| Component | ms/token | share |
|---|---|---|
| `rknn_matmul_run` wall (3 cores concurrent; 1,277 calls at 0.374 ms) | ~159 | 62% |
| `set_io_mem` (2,554 calls) + `mem_sync` (1,704 calls) | ~5 | 2% |
| sequential CPU graph phase | ~92 | 36% |

The 159 ms is 4.1 GB at ~26 GB/s — within 10% of the coop probe's
28.6 GB/s driver ceiling, so the NPU part is essentially at its floor.
The 92 ms CPU phase is real model compute (perf: bf16/f16 attention dots,
GLU, tanh — the AltUp/LAUREL/activation ops), sequentially dependent on
the matmuls, not overlappable within a single stream. ~53% of on-CPU
samples are idle spin (ggml's team barrier while the NPU runs + the
dispatch pool between nodes) — cosmetic, those cores have nothing else
to do. Zero thread creation, futex+ioctl only.

**Routed decode (5.50 t/s = 182 ms/token):** ~92% of cycles are
productive memory-bound kernels (66% Q4_0 weight GEMV, 16% Q8 dot, 8%
bf16, 1.6% f16). No software waste left; purely bandwidth-bound.

**Implications:**
- QKV/gate-up fusion is now bounded: it can only attack the ~5 ms
  driver-misc slice plus per-node fixed costs inside the 159 ms — no
  longer a 40% lever. Deprioritized again, this time with numbers.
- The remaining engineering frontier is BYTES. Concretely: W4A4 NPU
  decode reads ~2.05 GB (floor ~79 ms) yet measures 4.31 t/s
  (232 ms/token) — its CPU-side is ~153 ms vs W8A8's 92 ms. **Closing
  that ~60 ms gap (Hadamard prep + INT16 dequant remnants on the M=1
  path) would put W4A4 decode at ~5.8 t/s — above routed — while
  halving NPU memory.** That is the next investigation.
- Measurement pitfalls fixed along the way: the "decode detector"
  (utime slope + thread count) also fires during load's OMP
  quantization burst, which had contaminated two earlier profile
  windows (the "set_tensor eats 10% of decode" reading and the
  "residual clone3" were both load-tail artifacts — set_tensor at
  decode is zero). Decode windows are now taken at a fixed offset
  validated against the shim's run-call timeline.

Cheap bs>1 verification reviving ngram speculation for servers remains
open (see #1).

### 3b. The W4A4 K-padding find — MEASURED AND FIXED (2026-08-16)

Follow-up to the #3 re-profile's "W4A4 has ~60 ms/token more CPU work"
hypothesis — which turned out to be wrong in an interesting way. The
driver shim on W4A4 decode (4.17 t/s = 240 ms/token) showed the CPU
phase is the SAME as W8A8 (~97 ms); the real problem was on the NPU
side: run wall 137 ms where halved bytes predicted ~80.

Root cause: the Hadamard pipelines padded K to the next power of two
(one full-length FWHT needs pow2). Gemma-4 E4B's dims are NOT powers of
two, and a GGUF census (`rknpu2-gguf-census.py`, headers only) showed the
damage: **216 tensors at K=2560 -> 4096 and the FFN downs at
10240 -> 16384, both 1.60x; in total 5.22 GB stored/read per token
instead of the nominal 3.38 GB (1.54x)**. The padding silently ate most
of INT4's byte advantage — in memory as well as bandwidth.

Fix: **block-diagonal FWHT** (QuaRot-style). Block = largest power of
two dividing K (`K & -K`: 2560 -> 512, 10240 -> 2048, 10752 -> 512;
power-of-two K keeps block = K, bit-identical). K_op == K everywhere: no
padding stored, uploaded, read, or transformed. Dequant divisor becomes
the block length (H*H^T = B*I per block). Blocks need not align with
K-segments — segments only split the summation. `RKNPU_HADAMARD_BLOCK`
selects the mode: 1 = pure block-diagonal (default), 0 = legacy padded
(bit-identical to the pre-change build, verified by greedy diff), pow2 n
= minimum block with padding to a block multiple (measured, rejected).

Measured (E4B Q4_0, default W4A4_HADAMARD, t=4, wikitext-2 8 chunks;
references on the same chunks: **W8A8 NPU 37.78, pure CPU 37.66** — the
Q4_0 file itself is essentially lossless through llama.cpp's native
Q4_0xQ8_0 path with per-32 scales; the W4A4 degradation below is the
price of the NPU's INT4xINT4 scheme — 4-bit activations plus one weight
scale per segment — not of the 4-bit file and not of this backend's
code):

| W4A4 mode | PPL | tg64 | pp128 | NPU buffer |
|---|---|---|---|---|
| legacy (pad to next pow2) | 198.4 | 4.17-4.31 | 33.4-34.4 | 3569 MiB |
| **pure block-diagonal (new default)** | 228.9 (+15%) | **5.51** (+28%) | **37.98** (+11%) | **2521 MiB (-29%)** |
| min-block 1024 (pad to block multiple) | 280.4 | 5.13 | 37.39 | 2765 MiB |

- **W4A4 decode now equals the routed path (5.51 vs 5.50) as a pure-NPU
  mode at -29% NPU memory**, with the CPU left mostly free — exactly
  what the capacity mode is for. Prefill 7.7 -> 38.0 cumulative (4.9x).
- The quality ordering is instructive: full-row padded (198) beats pure
  512-blocks (229) beats mixed half-empty 1024-blocks (280). Spreading
  outliers over MORE lanes helps; a block that is half zero-padding
  hurts worse than a smaller full block — so the intermediate mode is
  dominated and kept only for experiments.
- ~~Quality-sensitive W4A4 users set `RKNPU_HADAMARD_BLOCK=0`~~
  Superseded by #3c: with per-channel weight scales the blocked
  transform wins on quality too (45.4 vs 49.1) — there is no trade-off
  left and no reason to use the legacy mode except as a regression
  anchor.

### 3c. Per-output-channel weight scales — the W4A4 quality fix (2026-08-20)

The PPL post-mortem of #3b blamed the W4A4 quality tier on two suspects:
4-bit activations and the coarse per-segment weight scales. Measured
verdict: **it was mostly the scales.** The channel dimension factors out
of the hardware's K summation, so a per-output-channel scale can be
applied in the existing C dequant pass — finer granularity at zero NPU
cost. INT4 weights now get one amax scale per output channel per
k-segment (grid `[k_idx * N + n]`, applied via `*_perchan` dequant
helpers); the segment-wide entropy search is gone, which also cuts the
W4A4 calibration load from minutes to seconds. W8A8 is untouched
(already at CPU parity). Legacy: `RKNPU_PER_CHANNEL=0`.

E4B Q4_0, same 8 wikitext chunks:

| W4A4 config | PPL | tg64 | pp128 |
|---|---|---|---|
| segment scales + padded FWHT (original) | 198.4 | 4.17-4.31 | 33.4-34.4 |
| segment scales + blocked FWHT | 228.9 | 5.51 | 37.98 |
| **per-channel + blocked (new default)** | **45.35** | **5.54** | 36.87 |
| per-channel + padded (`RKNPU_HADAMARD_BLOCK=0`) | 49.09 | — | — |
| W8A8 / pure CPU reference | 37.78 / 37.66 | — | — |

- **W4A4 PPL 228.9 -> 45.35 (5x)** at unchanged speed: the capacity mode
  now costs ~20% PPL over the 8-bit tier instead of ~5x, at -29% NPU
  memory and NPU-decode speed equal to the routed path.
- With honest weight scales, blocked-FWHT also beats padded on quality
  (45.4 vs 49.1): the padded transform's "advantage" in #3b was partly
  compensating the segment scale with more outlier smearing.
- Bit-identity anchors held: W8A8 unchanged; `RKNPU_HADAMARD_BLOCK=0
  RKNPU_PER_CHANNEL=0` reproduces the original build's greedy output
  exactly. 254 prep-kernel checks (perchan dequant flat + tiled vs
  fused scalar references).
- Remaining gap to 37.8 = the true cost of 4-bit activations at
  per-row scales (+ residual weight coarseness along K). Next candidates
  if it ever matters: finer K-segmentation (speed trade), MSE-clipped
  A-scales.

**Hardened validation (2026-08-21, 32 chunks = 4x the sample; absolute
values shift with the larger text window, compare within the column):**

| Config | PPL (32 chunks) |
|---|---|
| E4B pure CPU | 27.01 ± 1.13 |
| E4B W8A8 | 27.29 ± 1.14 |
| E4B W4A4 (per-channel + blocked) | **34.36 ± 1.37** (+27% vs CPU) |
| Qwen2.5-1.5B pure CPU | 8.88 ± 0.26 |
| Qwen W8A8 | 12.24 ± 0.37 (**+38% vs CPU** — see below) |
| Qwen forced W4A4 (per-channel + blocked) | 26.87 ± 0.92 |
| Qwen forced W4A4 (legacy scales + pad) | 44.31 ± 1.50 |

- The per-channel fix generalizes: Qwen W4A4 44.3 -> 26.9 (1.65x).
  W4A4's relative cost is model-dependent (E4B +27%, Qwen ~3x vs CPU).
- **New finding: W8A8's per-segment scales are NOT free on every model.**
  E4B W8A8 == CPU, but Qwen W8A8 is +38% over CPU. Extending
  per-channel scales to INT8 is now a data-justified follow-up (the C
  INT32 dequant pass is elementwise over n, so the same zero-NPU-cost
  argument applies; it would change W8A8 numerics, moving the
  bit-identity anchor — do it deliberately).
- Chat smoke test (E4B W4A4, chat template): coherent, well-structured
  output. Note E4B is a reasoning model — completions land in
  `reasoning_content` first; short max_tokens can leave `content` empty
  (finish_reason=length), which is model behavior, not a backend bug.

### 3d. Clipped INT4 scales — W4A4 reaches the 8-bit quality tier (2026-08-22)

Follow-up to #3c, and partly a correction of it. The per-channel change
swapped `calculate_entropy_amax` (a KL-divergence search for an optimal
*clipping* point) for a plain per-channel `amax`: it won granularity but
silently dropped clipping. `RKNPU_A_CLIP` / `RKNPU_B_CLIP` restore
clipping as a constant factor (`scale = clip * amax / 7`) — values above
`clip*amax` saturate to ±7, buying finer steps for the bulk. Post-
Hadamard rows are near-Gaussian, so amax is a far-tail sample and
clipping is nearly free in error terms. The two effects compose:
granularity (#3c) + clipping (#3d) beats either alone, at a constant
multiply instead of a 128-step per-segment search.

E4B, 32 chunks, A=0.9 (the B-curve is a smooth U — no knife edge):

| B clip | 1.0 (none) | 0.95 | 0.94 | **0.93** | 0.92 | 0.91 |
|---|---|---|---|---|---|---|
| PPL | 34.36 | 28.83 | 27.76 | **26.95** | 28.72 | 30.02 |

Isolating the two sides at 32 chunks: B-clip alone (A=1.0, B=0.95) gives
30.72, adding A=0.9 gives 28.83 — the weight side dominates, the
activation side is worth ~2 points on top. A is flat over 0.85–0.9
(28.81 vs 28.83), so 0.9 is not a fitted edge.

**Result — the headline of this whole thread:**

| E4B Q4_0, 32 chunks | PPL |
|---|---|
| pure CPU (llama.cpp Q4_0 x Q8_0 kernels) | 27.01 |
| NPU W8A8 | 27.29 |
| **NPU W4A4, defaults (block-FWHT, per-channel, clipped)** | **26.95** |
| NPU W4A4 before this thread's quality work | ~5x worse tier |

**W4A4 is no longer a quality compromise on E4B** — it matches the 8-bit
and CPU references while using 29% less NPU memory and decoding slightly
faster (5.5 t/s). Cross-model check (Qwen2.5-1.5B forced W4A4, 32
chunks): 26.87 -> **21.74** (-19%), and Qwen independently prefers
B=0.93 over 0.95, so the defaults are not overfit to E4B. Qwen's W4A4
still trails its CPU reference (8.88) — how much of the 4-bit tier a
model can absorb remains model-dependent.

Speed is unaffected (paired same-build control: 36.41/5.47 clipped vs
36.48/5.51 unclipped — inside run-to-run drift; an earlier "-1.7%"
reading was board drift, caught by re-measuring both arms in one run).

Defaults `RKNPU_A_CLIP=0.9`, `RKNPU_B_CLIP=0.93`; set both to 1.0 for
plain amax. Caveat: tuned on wikitext-2 across two models — a
quality-critical deployment should re-sweep on its own corpus, which is
now cheap (W4A4 loads in seconds).

### 3e. Per-channel INT8 scales — W8A8 quality becomes model-independent (2026-08-22)

The #3c 32-chunk run exposed that W8A8's per-segment scales are lossless
on E4B but cost +38% PPL on Qwen. Same mechanism as #3c (channel scales
factor out of the K sum, applied in the C dequant pass), now for the
INT32 output path via `dequant_acc_int32_to_fp32_perchan`. Gated by the
same `RKNPU_PER_CHANNEL`; the legacy path is left byte-for-byte intact.

| Config, 32 chunks | per-segment (legacy) | **per-channel (new default)** | CPU ref |
|---|---|---|---|
| Qwen2.5-1.5B W8A8 | 12.24 (+38% vs CPU) | **9.08 (+2.3%)** | 8.88 |
| E4B W8A8 | 27.29 (+1.0%) | 27.61 (+2.2%) | 27.01 |
| E4B W4A4 default (mixed: its non-Q4_0 tensors run W8A8) | 26.95 | **26.88** | 27.01 |

For the same-scale before/after: E4B W4A4 in its **original**
configuration (per-segment entropy scales + padded FWHT + no clipping)
measures **163.01 ± 9.19** at 32 chunks, against 26.88 for the current
defaults and 27.01 for the CPU reference — a 6.1x improvement to CPU
parity, all from #3b/#3c/#3d/#3e.

- **The point is the collapse in variance:** W8A8 went from "+1% on one
  model, +38% on another" to "+2.2%/+2.3% on both". Quality is now
  predictable per pipeline instead of per model, which is what makes the
  pairing guidance trustworthy.
- E4B W8A8 is nominally 0.32 worse, inside its ±1.15 error bar; the
  large Qwen win and the consistency argument carry the default.
- **Not free, corrected 2026-08-23**: on Qwen it is noise (pp 215.1/tg
  9.53 per-channel vs 215.8/9.58 legacy), but on E4B it costs **2.5% of
  prefill** — routed pp 42.47 -> 41.39, A/B'd on a single build with
  `RKNPU_PER_CHANNEL=0`. The per-channel dequant loads and multiplies per
  output element where the segment path used one scalar. Only Qwen was
  measured at the time, which is how "free" got into the docs. The trade
  still clearly favours per-channel (Qwen W8A8 PPL 12.24 -> 9.08), and it
  is why the routed prefill figure across these documents is 41.4 rather
  than the 42.5 measured before #3e.
- Bonus: the E4B *W4A4* default improved 26.95 -> 26.88, because a Q4_0
  GGUF is not all-Q4_0 — its Q8_0/Q6_K tensors map to W8A8 and inherited
  the fix. Worth remembering when reading any "W4A4" number here.

**Closed with a reason: INT8 scale clipping.** The #3d clip trick does
NOT transfer to int8. Measured `RKNPU_B_CLIP_INT8=0.95`: Qwen PPL
12.2 -> **10106**. Cause: `quantize_fp32_to_int8` has no clamp — it is
only ever called with `scale = amax/127`, so `|v/scale| <= 127` holds by
construction, and the NEON narrowing was deliberately allowed to wrap
(pinned by a prep-kernel test, and flagged as "unreachable in the
backend" in `RKNPU2-neon-prep-plan.md`). A clip < 1 makes it reachable:
the extremes land near 134, wrap to about -122, and sign-flip the
largest weights. The knob was removed rather than shipped; re-enabling
would require clamping that kernel first, and int8's 255 levels make the
payoff unlikely. This is the second time the "no clamp" property has
bitten (see the int4 clamp bug in `RKNPU2-neon-prep-plan.md`) — treat
any new scale factor on a quantizer as requiring a clamp audit.

### 3f. What limits W4A4 quality is the MODEL, not the pipeline (2026-08-22)

After #3d put E4B's W4A4 at CPU parity, the obvious question was whether
that generalizes. It does not, and the reason is worth knowing before
promising anything about 4-bit.

| Model / file | CPU | W8A8 | W4A4 | W4A4 penalty |
|---|---|---|---|---|
| Gemma-4 E4B (7.5B), Q4_0 | 27.01 | 27.61 | **26.88** | **−0.5%** |
| Qwen2.5-1.5B (1.8B), Q4_0 | 9.36 | 9.56 | 18.73 | +100% |
| Qwen2.5-1.5B (1.8B), Q8_0 | 8.88 | 9.08 | 21.74 | +145% |

- **Source precision is a real but minor factor.** Requantizing an
  already-4-bit file to int4 costs less than crushing an 8-bit file:
  the same model goes from +145% to +100% purely by starting from Q4_0.
  Worth preferring Q4_0 sources for W4A4, but it explains maybe a third
  of Qwen's gap and none of E4B's parity.
- **QAT was ruled out**: E4B's GGUF metadata gives
  `general.base_model.0.name = Gemma 4 E4B It` — the plain instruct
  model, not Google's `-qat-` variant. E4B's robustness is not trained in.
- **The leading explanation is model capacity** (7.5B vs 1.8B), which
  matches the standard finding that larger models absorb aggressive
  quantization better. A within-family control (Gemma-4 E2B, same
  architecture and recipe, half the size) would settle it, but the
  available E2B GGUF does not load in this fork (`wrong number of
  tensors; expected 601, got 561`).
- **W8A8's penalty, by contrast, is now nearly constant** at +2.2%,
  +2.3%, +2.2% across all three — the payoff of #3e.

**Honest claim to make:** on a ~7B dense model with a Q4_0 source, W4A4
is now quality-free on this backend. On a ~2B model it still costs
roughly 2x perplexity. "4-bit is free" is not a general statement, and
the capacity mode is best suited to exactly the large models it was
built for.

### 4b. MoE models produce wrong output on this backend — PRE-EXISTING BUG

Discovered while looking for a third data point above, and the most
consequential finding of the session: **LFM2-8B-A1B produces garbage on
the NPU and always has.**

| LFM2-8B-A1B Q8_0, 32 chunks | PPL |
|---|---|
| Pure CPU | 15.86 |
| NPU W8A8 | **17402** |
| NPU W4A4 | **20983** |

- **Not a regression.** Legacy per-segment scales give the same garbage
  (20168 vs 19314 at 8 chunks), so this predates every change in this
  thread. It was never noticed because **LFM2 had only ever been
  speed-benchmarked** — every LFM2 number in these documents (9.47 ->
  13.66 t/s and friends) was measured on a model producing nonsense.
  Treat those rows as throughput-only and quality-invalid.
- **Mechanism, partly established.** LFM2 carries 66 three-dimensional
  expert tensors (e.g. `blk.2.ffn_down_exps.weight [1792, 2048, 32]`).
  Every path in the backend reads `ne[0]`/`ne[1]` only:
  `get_tensor_packed_size` sizes a 3D tensor as if it held one expert,
  and `supports_op` never checks `ne[2]`/`ne[3]` while `graph_compute`
  takes `M = src1->ne[1]` and clears just `M*N` of dst. So batched or
  multi-expert work is accepted and only its first slice computed.
- **Two attempted fixes, both reverted.** Rejecting 3D weights in
  `resolve_op_support` changed nothing (identical PPL to the digit — the
  expert tensors evidently never reach it). Additionally rejecting
  non-2D operands in `supports_op` changed which tensors are offloaded
  (LFM2 tg 13.25 -> 12.08, RKNPU buffer 8.3 GB -> 560 MB) and turned the
  wrong numbers into **NaN**. Neither was shipped: a half-understood
  change that swaps one broken behaviour for another is worse than a
  documented defect, and dense models were bit-identical throughout, so
  nothing else was at risk.
- **Scoped follow-up.** Making MoE correct here is its own project:
  establish which ops actually reach the NPU for this architecture
  (instrument `supports_op`), decide whether to implement `MUL_MAT_ID`
  or to cleanly exclude MoE tensors from the RKNPU buffer type
  altogether, and validate against the CPU reference. Until then, run
  MoE models with `RKNPU_CPU_DECODE=999999` (all matmuls on CPU, verified
  correct at 15.86) or on the CPU backend.
- **Method note:** this is what comes of validating only what you
  optimize. LFM2 was in every speed table for weeks; one perplexity run
  would have caught it at any point.

### 3g. The Hadamard transform: mandatory, priced, and already optimally sized (2026-08-22)

The transform was the last unexamined cost in the W4A4 text path. Three
questions, all now answered on E4B (8 chunks; refs W8A8 38.99, CPU 37.66).

**Is it still needed?** The "numerically broken" verdict on
`W4A4_STANDARD` predated the int4 clamp fix, per-channel scales and
clipping, so it was worth re-testing. It is emphatically still needed:

| E4B | pp128 | tg64 | PPL |
|---|---|---|---|
| W4A4_HADAMARD (production) | 37.80 | 5.50 | **35.05** |
| W4A4_STANDARD (no transform) | 41.42 | 6.17 | **26872** |

Per-channel weight scales do nothing for the activation side, and a 0.9
activation clip is nowhere near enough to tame outliers at 15 levels.

**What does it cost?** Exactly the numbers above: **8.7% of prefill and
10.9% of decode**. That is also a hard upper bound on any future
transform optimization. Two useful corollaries:

- The int4 matmul path itself now runs at **41.42 vs W8A8's 42.5** —
  parity. The entire remaining W4A4-vs-W8A8 prefill gap is the transform
  and nothing else. The original "INT4 is 5x slower than INT8" finding
  is fully closed out.
- Transform-free W4A4 would be the fastest decode on the board (6.17),
  which is a tidy statement of what outlier spreading costs.

**Can it be made cheaper by shrinking the blocks?** No — measured. Any
smaller power of two also divides K, so it needs no padding and costs
fewer passes (log2(128)=7 vs log2(512)=9). The trade is bad:

| FWHT block (K=2560) | PPL | pp128 | tg64 |
|---|---|---|---|
| **512 (natural, default)** | **35.05** | 36.98 | 5.48 |
| 256 | 42.32 (+21%) | 38.34 | 5.51 |
| 128 | 55.43 (+58%) | 38.54 | 5.52 |
| 64 | 54.83 (+56%) | 38.75 | 5.55 |

Speed saturates at +4.8% while quality degrades without limit. The
reason speed barely moves: at 512 floats the block is 2 KB and already
L1-resident, so dropping passes saves ALU only — and it also removes the
motivation for the "FWHT stage-fusion" follow-up noted in
`RKNPU2-neon-prep-plan.md`, which was justified by memory traffic that
the block-diagonal change (#3b) had already eliminated. The natural
block (largest power of two dividing K) is the right default, confirmed
rather than assumed.

`RKNPU_HADAMARD_BLOCK` now accepts explicit sizes below the natural
divisor (previously ignored) so this dial stays open for other models;
the default is unchanged and reproduces PPL 35.0480 exactly.

**Verdict: the W4A4 text-speed thread closes here.** The remaining 8.7%
buys the difference between PPL 35 and PPL 26872. It is well spent.

### 3h. Sharing Hadamard sign vectors — reuse blocked, but a quality knob found (2026-08-22)

Follow-up to #3g, chasing the last idea for making the transform cheaper.
Observation from the code: each weight tensor gets its **own** random sign
vector (seeded from its name), and the prepared-A buffer is cached by
geometry only — so its contents are recomputed per node. Q, K and V all
consume the same activations, as do gate and up, yet each runs a full
sign-multiply + FWHT + quantize over that identical input, differing only
because the signs differ. Nothing in the maths requires per-tensor signs:
the rotation only has to be consistent between a matmul's A and B. If
tensors sharing an input shared a sign vector, the transform could be
computed once and reused — roughly 43% fewer transforms, worth ~4%
prefill and ~5% decode.

Tested with `RKNPU_SHARED_SIGNS=1` (seed from K, so all K=2560 tensors —
Q, K, V, gate, up on E4B — share one vector), 32 chunks:

| Model | per-name signs (default) | shared-by-K |
|---|---|---|
| Gemma-4 E4B | **26.88** | 38.36 (+43%) |
| Qwen2.5-1.5B forced W4A4 | 21.74 | **19.94 (−8%)** |

**The reuse optimization is blocked on E4B** — a 43% quality loss for a
4% speed gain is not a trade worth making, so the "redundant"
recomputation stays. On E4B the differing signs are doing real work.

**But the effect is model-dependent and reverses on Qwen**, where sharing
is an 8% quality *improvement* at zero cost. A single mechanism does not
explain both directions, and with two models there is no basis for a
theory; recorded as measured, not explained. The flag is kept as a
per-model quality knob (default 0 = per-name, which is what E4B needs).

Escape route also closed: one could reorder the maths to apply the
Hadamard first and the cheap per-tensor sign flip afterwards (0.8 us vs
10 us), keeping decorrelation while sharing the expensive part. It is a
valid rotation, but useless — sign flips after the transform change no
magnitudes, so the quantization grid becomes identical to having no sign
vector at all, i.e. exactly the shared-signs configuration measured
above.

### 3i. Multimodal: where the time actually goes, and the GPU verdict (2026-08-22)

First measurement of E4B's vision path on this board (mmproj-F16, 990 MB;
768x768 test image; `llama-mtmd-cli --jinja`). One image plus a 48-token
answer takes ~32 s, split as:

| Phase | Time | Backend |
|---|---|---|
| **Vision encode** | **12.5 s** | CPU only |
| 252 image tokens through the LLM | 8.2 s | NPU |
| Text generation (47 tokens) | 7.9 s | NPU + CPU |

**Vision encoding is the dominant cost and had no acceleration at all**
(`clip_ctx: CLIP using CPU backend`). The tower is a plain ViT — 224px,
patch 16, 768-dim, 16 blocks, 12 heads, 478 M — so at 768x768 it is
~440 GFLOP of matmul, which at the ~30 GFLOPS four A76 cores deliver
predicts ~14 s. The measurement matches: it is compute-bound, not a
pathology. For any multimodal use, this dwarfs text t/s.

**Bug fixed on the way (shipped).** `set_tensor` packed tensors on dtype
alone, never checking the alignment `pack_native` asserts on, so CLIP
aborted at load the moment the NPU was offered the vision tower. Dense
LLM dims happen to be aligned, which is why it had never fired. All four
buffer paths now share one `resolve_packable_pipeline` predicate.
Deliberately not folded into `resolve_op_support`, which assigns the
sequence numbers driving cyclic hybrid patterns. Verified: E4B text
reproduces PPL 35.0480 exactly, 269 kernel checks green.

**CLIP on the NPU: measured, not worth it.** With the fix plus
`RKNPU_CPU_DECODE=32` (the RKNPU buffer only advertises `is_host` under
dual residency, without which the CPU cannot read NPU-resident vision
tensors and the scheduler aborts), it runs — and delivers 11923 ms
against 12539 ms, **5%**. The reason is in the split counts:

```
CLIP on NPU:  graph splits = 227,  nodes = 940
CLIP on CPU:  graph splits =   1,  nodes = 940
```

The backend takes only 2D `MUL_MAT`, so every layernorm, GELU, softmax
and reshape bounces back to the CPU, fragmenting a 940-node graph into
227 pieces. Whatever the matmuls gain, the handoffs return. Same lesson
as the text path: fragmentation dominates. Not recommended.

**The GPU: hardware ready, blocked in llama.cpp.** The full stack was
brought up and verified:

- The vendor BSP binds the Mali to ARM's proprietary driver
  (`/dev/mali0`); `panfrost` is loaded but never binds, and the DRM
  nodes belong to `rockchip-drm` and `RKNPU`. So **Vulkan/panvk is not
  available** — mesa enumerates only `llvmpipe`. Changing that means
  replacing the kernel, which would also take the NPU driver with it.
- The OpenCL route works. Rockchip's libmali blob
  (`libmali-valhall-g610-g13p0-gbm.so`, 164 CL symbols) plus an ICD gives
  a working **Mali-G610 r0p0, OpenCL 3.0**, 4 compute units, with
  `cl_khr_fp16`, full subgroup support (shuffle/ballot/clustered reduce)
  and `cl_arm_integer_dot_product_int8`. llama.cpp builds cleanly with
  `-DGGML_OPENCL=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF`.
- **And then ggml's OpenCL backend rejects it by name:**
  `Unsupported GPU: Mali-G610 r0p0 / drop unsupported device`. The
  allowlist covers Adreno, Qualcomm and Intel only.

Enabling Mali is a **backend port, not a flag flip**. `gpu_family` gates
~10 sites, and critically the subgroup size it selects (Adreno 64,
Intel 32) sizes *local memory allocations* —
`clSetKernelArg(..., sizeof(float)*nth/sgs, NULL)`. Mali Valhall is
16-wide, so borrowing the Intel path would under-allocate that scratch
buffer and silently corrupt results. A real port needs a `MALI` family
with the correct subgroup size (or a queried one), every branch handled,
and per-kernel numerical validation. Scoped but not small, and it lives
in upstream ggml rather than this backend.

**Recommendation:** if multimodal matters, the Mali OpenCL port is the
highest-value work available on this board — the GPU is capable, fully
accessible, and would take the whole 940-node ViT in one split instead
of 227. Everything up to the allowlist is already done and documented
here. If it does not, run vision on the CPU and ignore the NPU for CLIP.

### 4c. MoE fixed — it was batched matmuls, not the experts (2026-08-24)

#4b recorded LFM2-8B-A1B producing garbage on the NPU (PPL ~17400 vs
15.86 on CPU) and two failed fix attempts. Diagnosed properly this time
and fixed. **It was never about the expert tensors.**

**Building the instrument first.** `--override-tensor` turned out to be
useless here — `-ot ".*=CPU"` leaves the RKNPU allocation at 560 MiB,
unchanged, and every variant returned PPL identical to seven figures,
which is what exposed the tool as inert rather than the model as
insensitive. So the backend gained `RKNPU_EXCLUDE=<substr>[,...]`: a
diagnostic filter in `resolve_op_support` that keeps matching weights off
the NPU entirely, with their original bytes. With a working instrument
the bisection took one run per class:

| Excluded from the NPU | PPL (8 chunks) |
|---|---|
| nothing | 19313.86 |
| everything | 20.78 ✓ |
| **shortconv only** | **21.90 ✓** |
| attn only | 15909 |
| token_embd only | 18239 |
| dense ffn only | 18726 |

**Root cause.** Instrumenting the accepted ops (`RKNPU_DEBUG_OPS=1`,
also added) showed the shapes immediately:

```
shortconv.in_proj  src1[2048,128,4]  dst[6144,128,4]     <- ne[2] = 4
attn_k             src1[2048,2,1]    dst[512,2,1]        <- ne[2] = 1
```

LFM2's short-convolution projections are **batched** matmuls, and
`graph_compute` took `M = src1->ne[1]` and cleared only `M*N`: it
computed the first of four slices and left the other three as whatever
was in the buffer. Dense models emit `ne[2]==1` exclusively, which is why
six sessions on Gemma and Qwen never tripped it. A stride hypothesis was
checked first and ruled out — every accepted op had a contiguous dst with
the expected row stride.

**Why the earlier attempt produced NaN**, which is the part that had been
missing: rejecting those ops sends them to the CPU, but the host copy
only exists under dual residency — `set_tensor`'s memcpy is conditional
and `get_alloc_size` does not even reserve room for it otherwise. The CPU
then read packed NPU bytes. Rejecting was never going to work without
also making dual residency unconditional, which would have cost every
model a full-size host copy and given back the −29% W4A4 memory win.

**The fix: compute the batches.** `graph_compute` now loops over
`src1->ne[2] * ne[3]`, advancing the activation and destination pointers
by `nb[2]`/`nb[3]` per slice; `supports_op` enforces what the loop
assumes (src0 2D, dst batch shape equal to src1's, no broadcasting). The
change is safe by construction for everything measured before it: with
`nbatch == 1` the loop runs once at zero offset, so dense results must be
bit-identical — and are (E4B 35.0480, Qwen 22.6774, exact).

| LFM2-8B-A1B | before | after | CPU reference |
|---|---|---|---|
| PPL, 8 chunks | 19313.86 | **22.12** | 20.78 |
| PPL, 32 chunks | 17402 | **16.90** | 15.86 |
| pp128 / tg64 | 43.23 / 12.08 (garbage) | 43.21 / 12.13 (correct) | — |

**A bonus that turned out to be spin-wait amplification (chased
2026-08-25).** The same change lifts E4B decode from 5.47 to **6.89 t/s
(+26%)**, reproducibly, with prefill unchanged, perplexity bit-identical
(26.8771), the same 603 graph splits, and byte-identical greedy decode
output over 48 tokens — so no op changed backend and no arithmetic
changed. E4B has no batched matmuls at all (`RKNPU_DEBUG_OPS` over a
full chunk: 3087 accepted mul_mats, every one `ne[2] == 1`), so on this
model the new loop provably runs once at zero offset. The gain is
therefore pure codegen, and it decomposes as follows.

Neither micro-edit inside the change reproduces it on its own — hoisting
`get_tensor_real_ptr(src1)` out of the k-segment loop gives 5.49, making
the destination pointer `const` gives 5.48, against a 5.53 baseline and
6.89 for the full restructuring. So it is the re-scoping of the ~230-line
per-node body as a whole, not any single line.

`perf stat` over an identical 64-token decode shows the pre-fix build
retiring **71.3 G more instructions** (389.7 G vs 318.4 G, +22%) at
identical IPC (2.00 vs 1.98) and identical page-fault counts — more code
executed, not more stalling. Re-running with all spin-waiting disabled
(`GOMP_SPINCOUNT=0 OMP_WAIT_POLICY=passive`, `SPIN_ITERS = 0`) collapses
that gap to **8.1 G (+4.3%)** and the throughput gap from +24.5% to
+10.7% (4.28 → 4.74 t/s). Roughly 89% of the extra instructions were
spin burn.

So the mechanism is a **feedback loop, not a hot spot**: the
restructuring removes a genuine but modest few percent of real work from
the per-node critical path; on an 8-core part already running a libgomp
team and two dispatch-pool workers that spin before sleeping, a longer
critical path means more spin burn, spin burn steals the very cores the
A-prep, dequant and CPU graph ops need, and that lengthens the critical
path further. The amplification is about 2.5x. Two independent
measurements corroborate it: in the decode profile the pre-fix build
sits at 60% of samples in libgomp spin versus 46% after (with useful
`ggml_vec_dot_bf16` work rising 9.5% → 15%), and widening `SPIN_ITERS`
collapses the pre-fix build (5.52 → 2.87 t/s over a 64x range) while
barely moving the fixed one (6.87 → 6.57 over 256x).

The practical reading: the fixed build sits in a far healthier operating
regime — it is nearly insensitive to the spin tuning that the old one
was violently sensitive to. It also means this path amplifies small CPU
savings on the per-node critical path by roughly 2.5x, which raises the
value of avenue #3's remaining per-node CPU work. It equally means
future micro-regressions there will be amplified just as hard, so decode
numbers should be re-measured after any change to the per-node body even
when the arithmetic is untouched.

The shortconv projections now run on the NPU and are computed correctly,
rather than being either wrong or pushed to the CPU. MoE models are no
longer a documented no-go; the earlier warning in
`RKNPU2-optimization-notes.md` is retired.

### 4. Read fewer bytes

- **Hybrid per-layer patterns** (`RKNPU_HYBRID="W8A8_STANDARD,W4A4_HADAMARD"`):
  INT4 on the fat FFN tensors, INT8 on attention. Decode gain tracks byte
  reduction ~linearly; nobody has mapped the accuracy/speed curve. Free to
  explore (no code), and results are now reproducible thanks to the
  name-hash Hadamard seeding.
- **Sub-4-bit CPU formats** (Q3_K, IQ3/IQ2 via the CPU decode path): 25–40%
  fewer bytes, quality falls with them; usually a bad trade at 7B, possibly
  a good one for fitting a 13B.
- **MoE architecture beats all of this**: LFM2-8B-A1B reads ~1.5 GB/token
  → 13.66 t/s measured (routed, t=4). Model choice > engineering.

#### 4d. Quantizing the MoE itself — MEASURED, a clean sweep (2026-08-25)

If MoE decode is bound by the bytes of the *active* experts, halving the
weight precision should halve those bytes. Tested by running the same
model, LFM2-8B-A1B, from LiquidAI's Q4_0 (4.41 GB) instead of the Q8_0
(8.26 GB) the tables above used. Both files SHA-verified against HF.

| Config | pp128 | tg64 | PPL 8ch | PPL 32ch |
|---|---|---|---|---|
| **Q4_0** pure CPU | 55.09 | **23.44** | 18.62 | **14.77** |
| **Q4_0** routed | **62.14** | 21.18 | — | — |
| **Q4_0** NPU W8A8 | 62.15 | 17.95 | 20.30 | 16.01 |
| **Q4_0** NPU W4A4 (file default) | 62.08 | 18.87 | 25.30 | — |
| Q8_0 pure CPU | 37.56 | 13.44 | 20.78 | 15.86 |
| Q8_0 NPU W8A8 | 44.80 | 12.43 | 22.12 | 16.90 |

Decode 13.44 → **23.44 t/s (+74%)**, prefill 44.8 → 62.1 (+39%), file
size −47%. The +74% falls short of the 2x that pure bandwidth scaling
predicts; the shortfall is the routing and attention work that does not
shrink with weight precision, which puts a rough floor on what further
quantization can buy.

**Q4_0 also scores better than Q8_0** — 14.77 vs 15.86 at 32 chunks.
A 4-bit quant cannot beat 8-bit through quantization alone, so this was
checked rather than reported: both files are the identical official
LiquidAI releases (Q8_0 sha256 matches HF), and the in-session Q8_0
references reproduce the previously recorded figures to four significant
figures (20.7758 at 8 chunks, 15.8591 at 32 against the recorded 15.86).
The measurement chain is sound and the effect holds at both sample
sizes.

**Cause: unexplained. The imatrix hypothesis is disproved (2026-08-28).**
This document originally recorded "LiquidAI's Q4_0 is imatrix-calibrated
while the Q8_0 is plain round-to-nearest" as the likely cause. Three
checks killed it:

- **Neither file carries `quantize.imatrix.*` metadata.** ERNIE's GGUF
  does, so the absence here is meaningful — both LFM2 quants are plain
  round-to-nearest.
- **The per-tensor recipes are identical but for one tensor.** Every
  weight is Q4_0 where the other file has Q8_0, except
  `token_embd.weight`: **Q6_K in the Q4_0 file, Q8_0 in the Q8_0 file.**
  That is the only asymmetry, and it points the *wrong* way — Q8_0 is
  ~8.5 bits against Q6_K's ~6.56, so the Q8_0 file has the higher-
  precision embedding table and still scores worse. (A k-quant's
  per-superblock scale structure beating a flat per-32-block scale on a
  65536x2048 table is conceivable, but it is speculation and it is the
  only candidate left.)
- **Both files were uploaded in the same batch** (2025-10-06/07,
  `upload-large-profile` commits), so they are not quantizations of
  different base revisions.

So a 4-bit file beats an 8-bit file from the same checkpoint, same
recipe, no imatrix, and the mechanism is not established. The
*measurement* is solid and reproduces at 8 and 32 chunks; only the
explanation is missing. Do not repeat the imatrix story.

Two operational conclusions:

- **Q4_0 MoE files need `RKNPU_HYBRID=W8A8_STANDARD`.** The file type
  selects W4A4, which costs +25% PPL here (25.30 vs 20.30 at 8 chunks) —
  the same small-dense-tensor trap E2B shows (#4e). MoE dense tensors are
  small even when the model is large, so total parameter count does not
  predict whether W4A4 is safe.
- **Pure CPU is now LFM2's best config outright**: fastest decode (23.44)
  *and* best quality (14.77). The NPU buys +13% prefill (62.1 vs 55.1)
  and costs 10% decode and 8% quality — a direct consequence of
  `MUL_MAT_ID` being unimplemented, so experts never reach the NPU.

#### 4e. Gemma-4 E2B — the small-model trap, and a broken GGUF (2026-08-25)

E2B was benchmarked for the first time. Two findings.

**The unsloth E2B GGUF does not load, and is not corrupt.** It fails with
`done_getting_tensors: wrong number of tensors; expected 601, got 561`,
but its sha256 matches HF exactly — re-downloading cannot fix it. The
export is malformed: it ships `attn_k`/`attn_v` for all 35 layers while
its own metadata says `shared_kv_layers = 20`, so the loader creates K/V
for 35−20 = 15 layers and leaves exactly 20×2 = 40 unclaimed. E4B is
self-consistent (42 blocks, 18 shared, 24 present and expected), which is
why it loads. Google's official QAT build is consistent (541 tensors,
`attn_k` on layers 0–14) and works. Diagnosis cost one range-request of
the file header; the general lesson is that a tensor-count mismatch is a
*publisher* bug far more often than a transfer bug, and the sha256 tells
you which in one command.

Google `gemma-4-E2B_q4_0-it.gguf` (QAT, 3.35 GB), `-t 4`:

| Config | pp128 | tg64 | PPL 8ch |
|---|---|---|---|
| NPU W4A4 (file default) | 113.6 | 4.87 | 74.75 |
| NPU W8A8 | 129.9 | 4.26 | 60.26 |
| **W8A8 + routed** | **135.0** | **10.97** | **60.26** |
| pure CPU | 71.1 | 13.48 | 59.35 |
| *E4B NPU W4A4, same settings* | *36.8* | *6.84* | *—* |

E2B prefills 3.7x faster than E4B and the NPU earns it (135 vs 71 on
CPU). But **decode inverts**: 4.87 on the NPU against 13.48 on CPU, and
slower even than E4B's 6.84 — less compute per byte moved, so the NPU's
advantage disappears. W4A4 costs +26% PPL (74.75 vs 59.35) while W8A8 is
nearly free (60.26) *and* prefills faster, so the Q4_0 default is wrong
on both axes. Recommended: `RKNPU_HYBRID=W8A8_STANDARD
RKNPU_CPU_DECODE=32`. Do not compare E2B's PPL to E4B's 26.88 — different
model scale and tuning; only the CPU column is a valid reference.

#### 4f. ERNIE-4.5-21B-A3B — an MoE that *does* reach CPU parity (2026-08-25)

The open question after #4d was whether E4B's CPU parity was reachable on
a mixture-of-experts at all, or whether LFM2's +36% was what MoE always
costs. Tested on unsloth's ERNIE-4.5-21B-A3B Q4_0 (11.64 GB, imatrix,
sha256-verified), `-t 4`, wikitext-2.

**What the NPU can even see.** Computed from the GGUF tensor table
before downloading: the expert FFNs (`ffn_{gate,up,down}_exps`, 27 layers
x 64 experts) are 3-D, so `MUL_MAT_ID` sends them to the CPU
permanently. Eligible are attention q/k/v/o (28 layers, ~440 M), the two
shared experts per layer (`ffn_*_shexp`, 2560x3072, ~637 M),
`token_embd` (~264 M), the single leading dense block (~94 M) and the
routers (~4 M) — **1.44 B of 21.83 B parameters, 6.6%**, with no
alignment exclusions. 1260 accepted mul_mats per eval. Low parameter
share but high error leverage: the output projection sits on the logits
path and the shared experts fire on every token through all 28 layers.

| Config | PPL 8ch | PPL 32ch | vs CPU (32ch) | pp128 | tg64 |
|---|---|---|---|---|---|
| pure CPU (reference) | 8.2411 | 6.0876 | — | 21.61 | **9.50** |
| **NPU W8A8** | **8.2153** | **6.0793** | **−0.14%** | **25.58** | 7.40 |
| NPU W4A4 (file default) | 9.1317 | 6.7843 | +11.4% | 25.37 | 8.60 |
| routed W8A8 | — | — | — | 25.46 | 9.06 |

**W8A8 reaches parity**, and it holds at the larger sample: −0.3% at 8
chunks, −0.14% at 32. Both are inside noise, so the honest reading is
"indistinguishable from CPU", not "better". That buys **+18% prefill**
(25.58 vs 21.61) at no quality cost, which is a real win on a model where
93% of the weights are untouchable until `MUL_MAT_ID` exists. Decode
still loses (7.40 vs 9.50; routed recovers only to 9.06), the same
pattern as every MoE measured. W4A4's penalty also reproduces across
sample sizes (+10.8% at 8 chunks, +11.4% at 32).

**The W4A4 result independently confirms the tensor-size rule from #4d.**
Ranked by W4A4 damage against CPU: E4B (2560x10240) ~parity, ERNIE
(2560x3072) +10.8%, E2B (1536x~2048) +26%, LFM2 (narrower) +36%,
Qwen-1.5B +145%. That ordering tracks tensor width, not parameter count
— ERNIE is 2.6x LFM2's total size and takes a third of the damage. The
rule was written from four models on 2026-08-25 and ERNIE, measured
afterwards, landed where it predicted.

Recommended: `RKNPU_HYBRID=W8A8_STANDARD` for prefill-heavy work, pure
CPU if decode dominates. The Q4_0 default (W4A4) is strictly worse on
both axes here — 11% quality for slightly *less* prefill.

#### 4g. LFM2-24B-A2B — 24 B of capacity at 15 t/s, and a silent OOM trap (2026-08-27)

Chasing capacity at fixed decode cost: the same LFM2 family as #4d, but
24 B total / 2 B active instead of 8.3 B / 1.5 B. LiquidAI published the
GGUF on 2026-08-24. Q4_0 is 12.54 GB — the tightest fit attempted on this
board's 15 GB. Same `lfm2moe` arch, so the #4c batched-matmul fix applies
unchanged. 40 layers (2 dense + 38 MoE), 64 experts, top-4.

| Config | pp128 | tg64 | PPL 8ch |
|---|---|---|---|
| pure CPU | 32.72 | **15.01** | 89.44 |
| NPU W8A8 | **39.71** | 11.22 | 88.14 |
| routed W8A8 | 39.44 | 14.03 | — |
| NPU W4A4 (file default) | 39.35 | 13.28 | 99.07 |

**15 t/s from a 24 B model**, against the 8 B's 23.44 — 3x the parameters
for 64% of the decode. Backend behaviour is correct and matches ERNIE:
W8A8 at parity with CPU (88.14 vs 89.44), W4A4 +10.8%.

**The trap: it needs an explicit `-c`.** The model's default
`n_ctx = 128000` allocates a **2500 MiB KV cache** on top of 12.54 GB of
weights, which exhausts RAM. Generation then returns *silently empty
output* — no error, no assert, exit code 0. `-c 4096` fixes it
completely. This is the worst failure mode in these docs because nothing
reports it; only the missing text does.

**Perplexity says it is far worse than the 8 B; task output says it is
not.** At matched `-c 512`, 16 chunks, pure CPU: **24 B = 94.71, 8 B =
19.96**. Same gpt2 tokenizer, same 65536 vocab, same Q4_0 type, same
context — so the numbers *are* comparable within this family and the 4.7x
gap is real, not a harness artifact (two wrong diagnoses were eliminated
first: mismatched context, and the `dense_2`/`output` loader warnings,
which are name-formatting noise — the 24 B's tensor inventory is
structurally identical to the 8 B's). Yet on direct prompts the two are
indistinguishable:

| Prompt | 24 B | 8 B |
|---|---|---|
| `17 * 24` | 408 ✓ | 408 ✓ |
| largest planet | Jupiter ✓ | Jupiter ✓ |
| why is the sky blue | correct, cites Rayleigh | correct, cites Rayleigh |

The likeliest reading is heavy instruction/RL post-training, which is
known to inflate raw language-modelling loss — but that is not
established, and wikitext is evidently a poor proxy for this model.

**Verdict: stay on LFM2-8B-A1B.** It is faster (23.44 vs 15.01), a third
of the RAM, and equal on every task tried. Nothing measured here
demonstrates the 24 B's extra capacity. Revisit only with a
task-relevant eval. (LiquidAI's repo now points at a newer
`LFM2.5-8B-A1B` — an untested follow-up candidate.)

#### 4h. LFM2.5-8B-A1B — same shape, doubled vocab, and it thinks (2026-08-28)

LiquidAI's repo now points at LFM2.5-8B-A1B as the successor to #4d's
model. Architecturally it is the *same* model: `lfm2moe`, 24 layers, 32
experts, top-4, hidden 2048, moe_ffn 1792. One parameter changed —
**vocab 128000 vs 65536**. Q4_0 is 4.51 GB, sha256-verified.

| Config | pp128 | tg64 | PPL (`-c 512`, 16ch) |
|---|---|---|---|
| pure CPU | 55.86 | **20.93** | 28.02 |
| **NPU W8A8** | **63.69** | 18.00 | **27.96** |
| routed W8A8 | 63.40 | 18.86 | — |
| NPU W4A4 (file default) | 63.38 | 19.40 | 33.90 |
| *LFM2-8B-A1B pure CPU (ref)* | *53.90* | *23.17* | *—* |

**W8A8 at CPU parity again** (27.96 vs 28.02) — the third model to reach
it after E4B and ERNIE, giving +14% prefill for free.

**Prefill up 3.6%, decode down 10%** against LFM2 (20.93 vs 23.17). The
doubled vocabulary doubles the embedding and output projection, and that
is per-token work on the decode path — the cost lands exactly where the
bandwidth model says it should.

**It is a reasoning model.** It emits `<think>` blocks before answering
(correctly: "Straight multiplication: 17*24 = 408"). Benchmark tg64
therefore overstates *usable* throughput, because a real request pays for
reasoning tokens before the answer starts. This is the main practical
difference from LFM2 and it is a workload question, not a speed one.

**Do not compare its PPL to LFM2's.** 28.02 against 19.96 is a tokenizer
artifact — 128000 vs 65536 vocab puts perplexity on a different scale.
Only the within-model NPU-vs-CPU column is meaningful.

**Fourth confirmation of the tensor-area rule.** W4A4 costs +21%, placing
it between ERNIE (7.9 M area, +11%) and E2B (3.1 M, +26%) — its own area
is 3.7 M. Predicted before measuring, landed where predicted. The full
ordering is in the deployment guide.

**Verdict:** for raw speed stay on LFM2-8B-A1B (23.17 vs 20.93, no
thinking-token overhead). For reasoning quality, LFM2.5 with
`RKNPU_HYBRID=W8A8_STANDARD` gives 63.7 prefill at parity. Neither should
run the Q4_0 default.

### 5. Micro-tuning — MEASURED: far bigger than expected (sweep, 2026-08-10)

`rknpu2-affinity-sweep.sh` (trimmed variant), pinned clocks, llama-bench
pp128/tg64 r=3. **`-t 4` on the A76 big cores is a free +19-51% on decode
for every model tested** — the previous "±10%" guess was a big
underestimate, and it moves every baseline in these docs:

| Model / config | tg64 all-8 (old default) | tg64 t=4 big cores | gain |
|---|---|---|---|
| Qwen2.5-1.5B Q8, routed | 8.93 | **13.52** (pinned 4-7) | +51% |
| E4B Q4_0, W8A8+routed | 4.64 | **5.50** (t=4 floating) | +19% |
| E4B Q4_0, W8A8 NPU decode | 3.43 | **4.54** (pinned) | +32% |
| LFM2-8B-A1B NPU decode | 9.47 | **13.25** (pinned) | +40% |
| LFM2-8B-A1B routed | 8.22 | **13.66** (pinned) | +66% |

- The old "CPU decode wall" of ~13 GB/s was an 8-thread artifact: little
  cores on the critical path drag the A76 cluster down. 4 big-core
  threads reach ~25 GB/s effective on the same models.
- **LFM2's routing regression flips at t=4**: routed 13.66 > NPU 13.25,
  so `RKNPU_CPU_DECODE=32` stops being model-conditional once threading
  is right (revalidate per model, but the known counterexample is gone).
- Qwen prefill also gains from pinning: pp128 215 -> 280 t/s.
- NPU decode gains too (+32-40%). Root cause found the next day: under
  `taskset -c 4-7` libgomp defaults to 4 threads, which damps the OMP
  team-respawn churn diagnosed in avenue #3 — much of the "affinity"
  gain on NPU-heavy configs is really the OMP effect.
- ~~Anomaly to recheck: E4B routed with strict taskset was noisy/slow~~
  RESOLVED (see #3): pinning without `OMP_NUM_THREADS=4` thrashes the
  8-thread backend regions on 4 cores. With the env var set, strict
  pinning is clean and fastest (pp 42.62 ± 0.02).
- **Recommended config (RK3588, 2026-08-11):**
  `OMP_NUM_THREADS=4 taskset -c 4-7 ... -t 4` for CLI/bench; servers add
  `--threads 4 --threads-batch 8`.
- Still to do: re-tune `RKNPU_CPU_DECODE` threshold per model at t=4.

### 6. Batch serving (throughput only)

Multiple concurrent streams share each weight read — large aggregate
gains, zero single-stream latency gain. Relevant only if the workload
becomes a server.

## Backend environment variables added by this thread

| Variable | Default | Meaning |
|---|---|---|
| `RKNPU_HADAMARD_BLOCK` | 1 | 1 = natural block-diagonal FWHT, largest pow2 dividing K, no padding (measured optimal, #3g); 0 = legacy pad-to-pow2; pow2 n = explicit block — below the natural divisor it costs no padding but degrades quality fast, above it pads (both measured worse, #3b/#3g) |
| `RKNPU_PER_CHANNEL` | 1 | 1 = per-output-channel weight scales for INT4 (#3c) and INT8 (#3e); 0 = legacy per-segment scales (entropy search for INT4, segment amax for INT8) |
| `RKNPU_A_CLIP` | 0.9 | INT4 activation scale = clip * amax / 7; 1.0 = plain amax — #3d |
| `RKNPU_B_CLIP` | 0.93 | same for per-channel INT4 weight scales (no effect when `RKNPU_PER_CHANNEL=0`) — #3d |
| (no INT8 clip knob) | — | measured and removed: int8's packer has no clamp, so any clip < 1 wraps and sign-flips extremes (PPL 10106) — #3e |
| `RKNPU_EXCLUDE` | unset | Diagnostic: comma-separated name substrings; matching weights are never offloaded and keep their original bytes. Bisects which tensor class causes a wrong-output bug — `--override-tensor` cannot do this, it leaves the RKNPU allocation untouched (#4c) |
| `RKNPU_DEBUG_OPS` | unset | Diagnostic: log the geometry of every accepted mul_mat (dims, dst contiguity, row stride) (#4c) |
| `RKNPU_SHARED_SIGNS` | 0 | 1 = one Hadamard sign vector per K instead of per tensor. Model-dependent: E4B +43% PPL (bad), Qwen −8% (good). Blocks/enables transform reuse — #3h |
| `RKNPU_DOMAINS` | unset | Restrict NPU allocations to the listed IOMMU domains (`0,2` or `0-3`). Unset = the allocator uses domains 0-15 freely. **Setting it makes concurrent NPU access from multiple processes panic the kernel** — the backend prints a warning saying so. Diagnostic/experimental only; see the domain note below |
| `RKNPU_EXCLUDE_TYPES` | unset | Diagnostic: comma-separated ggml type names (`f16`, ...); weights of those types are never offloaded. Keeps an F16 MTP drafter on the CPU where `RKNPU_EXCLUDE` (name collision) and `--device-draft` (ACCEL buffer type) cannot — #1b. Measured no speed effect for Gemma-4 drafters |
| `RKNPU_PIPELINE` | 1 | 0 = disable the pipelined W4A4 prefill path (256-row chunks overlapping CPU prep/dequant with NPU runs, #1e); for A/B only, results are identical |
| `RKNPU_FLASH_ATTN` | 1 | 0 = keep prefill flash attention on the CPU (#1e) |
| `RKNPU_GLU` | 1 | 0 = leave GEGLU on ggml-cpu instead of the backend's NEON copy (#1f) |
| `RKNPU_FUSE_GLU` | 1 | 0 = do not fuse GEGLU into the up matmul's dequant (#1f); results identical |
| `RKNPU_DEFER_GATE` | 1 | 0 = dequantize the FFN gate into its own output instead of deferring it into up's fused GEGLU (#1f); results identical |
| `RKNPU_HOST_BUFFERS` | 1 | 0 = RKNPU buffer type not reported as host memory, so the scheduler copies NPU outputs back for CPU splits (#1f) |
| `RKNPU_DOWN_JOB` | 1 | 0 = no cross-node ffn_down job (used when the FFN block does not apply) (#1g) |
| `RKNPU_FFN_BLOCK` | 1 | 0 = run `[gate, up, GLU, down]` node by node (deferred gate / down job / fused GEGLU paths) (#1g); results identical |
| `RKNPU_FFN_MC` | 128 | FFN block chunk rows (32..512, multiple of 32) (#1g) |
| `RKNPU_FFN_PREP_AHEAD` | 2 | gate/up chunks prepped during the FFN block's first NPU batch (#1g) |
| `RKNPU_GEGLU_TILES` | 1 | 0 = FFN GEGLU via dequantized row buffers instead of straight from the INT16 tiles (#1g); identical |
| `RKNPU_TILE_PF` | 8 | prefetch distance (tiles) in the GEGLU tile pass, 0 = off (#1g) |
| `RKNPU_FA_NATIVE` | 1 | 0 = NPU attention with plain A/C layouts (runtime converts per run) (#1g) |
| `RKNPU_FA_NATIVE_B` | 1 | 0 = plain K/V B layout re-bound per item (driver converts) (#1g) |
| `RKNPU_FA_ROWS` | 64 | row block for the native-layout attention moves (#1g) |
| `RKNPU_FA_OVERLAP` | 1 | 0 = attention items run strictly in sequence (#1g); note the core-dependence (±0.02% PPL) in #1g |
| `GGML_CPU_DISABLE_FUSION_CHAIN` | unset | 1 = ggml-cpu's fused post-norm pass does not also compute the following RMS_NORM + MUL (#1g); identical |
| `RKNPU_HEAD_NORM` | 1 | 0 = per-head Q/K/V RMS norms (and weight MUL) stay on ggml-cpu (#1h) |
| `RKNPU_ROPE` | 1 | 0 = RoPE stays on ggml-cpu (#1h) |
| `RKNPU_ROPE_FUSE` | 1 | 0 = RoPE runs as a separate backend op instead of inside the Q/K dequant (#1h) |
| `RKNPU_ROPE_ANY` | unset | 1 = backend takes any F32 NORMAL/NEOX RoPE (exactness test hook) (#1h) |
| `RKNPU_KV_WRITE` | 1 | 0 = KV-cache writes (`SET_ROWS`) of K/V stay on ggml-cpu (#1h) |
| `RKNPU_W8A8_NATIVE` | 1 | 0 = W8A8 nodes keep NORM A/C at 1 < M <= 32 (the runtime then converts C on one thread per run; #1i) |
| `RKNPU_FA_RANGE` | 1 | 0 = NPU attention softmax gathers all S columns and scans each row's mask (#1h step 5) |
| `RKNPU_PROFILE` | unset | Diagnostic: prints cumulative wall time in the backend (graph / per-node / NPU run) every ~5 s to stderr; take the slope over a decode window and divide by the token rate (#1c) |
| `RKNPU_DISPATCH_POOL` | unset | 1 = old dispatch path: NPU segments on the persistent pool and serial M=1 A-prep instead of ggml's OpenMP team. For A/B comparison only (#1c) |
| `OMP_NUM_THREADS=4` | unset | no longer required: the #3 fix covers M=1, and since 2026-10-02 the backend takes ggml's thread count for M > 1 too (#1b); still harmless |

`RKNPU_HADAMARD_BLOCK=0 RKNPU_PER_CHANNEL=0 RKNPU_A_CLIP=1.0
RKNPU_B_CLIP=1.0` together reproduce the pre-2026-08-16 W4A4 numerics
bit-exactly (regression anchor — verified after every change since).
W8A8 and the routed path are unaffected by all of the above.

### The IOMMU domain limit is ~2 GiB, not 4 GB (measured 2026-08-26)

Two older documents state a "4 GB per-IOMMU-domain limit" and use it to
argue that W4A4's value is fitting larger models underneath it. The
number is wrong and the framing is misleading, so both are corrected
here and in place.

`IOMMUDomainManager::max_domain_size` is
`std::numeric_limits<int32_t>::max() - 65536` = **2,147,418,111 bytes,
just under 2 GiB** — the *signed* 32-bit limit, not the 4 GiB a 32-bit
device address space would allow. (The signed half is the classic
signature of int32 offset arithmetic inside `librknnrt`; the mechanism is
inference, the cap is not.) Every NPU allocation goes through
`assign_domain_memory`, which walks domains **0-15** and takes the first
with room.

Confirmed by forcing a model that exceeds one domain into one domain —
E4B's W4A4 footprint is ~2.5 GB:

```sh
RKNPU_DOMAINS=0 build/bin/llama-bench -m gemma-4-E4B-it-Q4_0.gguf ...
# RKNPU ERROR: Out of memory in allowed IOMMU domains!
# GGML_ASSERT(alloc.mem != nullptr ...) failed
```

The same model loads fine on the default path, which spreads it over two
domains. So the practical ceilings are:

| Level | Limit | Binding? |
|---|---|---|
| one IOMMU domain | ~2 GiB | yes, but routed around automatically |
| all 16 domains | ~32 GiB | no |
| board RAM (CPU+NPU shared) | 15 GB | **yes — the real ceiling** |

**There is no practical 4 GB NPU cap.** What limits model size on this
board is system RAM, and what makes W4A4 valuable is bandwidth and RAM
footprint — not domain space. This also corrects the reasoning in #4d/#4f
about MoE experts: 10.2 GB of int4 experts could be spread across six
domains without trouble; it is the 15 GB of physical RAM that blocks it.

## Experiment tooling (2026-08-10..21; reusable)

Everything compiles/links on any aarch64 Linux; running needs the board.
Pin clocks (`scripts/fix_freq_rk3588.sh`) and `ulimit -n 65536` first.

1. **`rknpu2-coop-decode-probe`** (produced the #2 verdict and the #3
   re-ranking) — `make -f Makefile.rknpu2-tools rknpu2-coop-decode-probe`,
   run with `LD_LIBRARY_PATH=../../ggml/src/ggml-rknpu2/libs`. Per shape:
   CPU-solo (Q4_0 GEMV, NEON sdot, big cores), NPU-solo (the backend's
   exact per-node M=1 driver sequence, 3-core N split), sweep of
   cooperative N-split fractions, 4 weight sets cycled to model
   consecutive layers. GEMV kernel self-checks against a scalar reference
   at startup. NPU dispatch threads and the main thread stay on little
   cores by design — keep that in any backend work. `COOP_CPU_ONLY=1`
   smoke-tests the CPU half on machines without the NPU.
2. **`rknpu2-spec-decode-bench.sh`** (produced the #1 verdict) — draft
   pairs x routing x draft-max, plus llama-server ngram self-speculation.
   Draft GGUFs now in `~/models` on the board.
3. **`rknpu2-affinity-sweep.sh`** (produced the #5 verdict) — llama-bench
   across thread count x taskset for routed, NPU-only and MoE configs.
4. **`rknpu2-driver-shim.c`** (produced the #3 re-profile and the #3b
   wall-time decomposition) — LD_PRELOAD shim timing every librknnrt
   entry point, cumulative counters printed every 5 s; the steady-window
   slope over the token rate gives ms/token per driver call.
   `make -f Makefile.rknpu2-tools rknpu2-driver-shim.so`, then
   `LD_PRELOAD=./rknpu2-driver-shim.so llama-bench ... 2>shim.log`.
   Also the reference for decode-window timing: take profile windows at a
   fixed offset validated against the shim's run-call slope — utime-based
   "decode detectors" fire during load's quantization burst too.
5. **`rknpu2-gguf-census.py`** (produced the #3b padding find) — parses
   GGUF headers only; per-tensor dims/types and the K-padding byte
   inflation table.
6. **Quality methodology (#3b/#3c):** wikitext-2 (`~/wikitext-2-raw/` on
   the board) via `llama-perplexity --chunks 8` for quick gates, 32 for
   hardened claims; always compare configs on the same chunk count, and
   read relative gaps, not absolute values (instruct model on raw
   encyclopedia text inflates absolutes). Greedy bit-identity via
   llama-server temp-0 completions anchors refactors.

## Handover notes (continuing on another machine)

> **Current handover (2026-10-07): see [`docs/handover/README.md`](../handover/README.md).**
> It covers state, setup on a new machine, restoring Claude's memory, the
> profiling patches and the loop harness. The notes below are older.

### State as of 2026-10-02 (supersedes the board details below)

- **Branch:** `rebase/w4a4-on-upstream`, local in the dev container, **not pushed**.
  It is the whole fork rebased onto ggml-org master `4ebdf2c74` (2026-10-02), which
  had moved 2,707 commits past the fork base `650bf14eb`. On top sit this session's
  commits: vtable NULLs, the MTP bench script, the OMP team fix,
  `RKNPU_EXCLUDE_TYPES`, and docs. `feat/w4a4-neon-prep` is unchanged at `7ac58d21c`.
- **How the rebase was done** (repeat this for the next one):
  - `feat/w4a4-neon-prep` has a *parentless* root commit `cb909e64f`, a copy of the
    `rknpu2` tip `81eff6a45` with an identical tree. A one-shot rebase therefore
    replays the whole tree as add/add conflicts, so do it in two stages:
    1. `git rebase --onto upstream/master 650bf14eb` on a branch made at `81eff6a45`.
       This carries the 12 base RKNPU2 commits; the only conflicts were in
       `ggml/CMakeLists.txt` and `ggml-backend-reg.cpp`.
    2. `git rebase --onto <stage-1 branch> cb909e64f` on a branch made at the feature
       tip.
  - `8c8f7ae5f` (the Gemma-4 QAT / `gemma4-assistant` workaround) was skipped, because
    upstream's `src/models/gemma4*.cpp` supersedes it.
  - Upstream's backend API v3 added optional vtable slots: `set/get_tensor_2d`,
    `set/get_tensor_2d_async`, `alloc_buffer_n` and `get_alloc_size_n`. All are NULL
    here, and upstream falls back to its old per-tensor paths.
  - The fork's root `CMakeLists.txt` still carries a sanitizer block that upstream
    moved to `cmake/common.cmake`. It is harmless while the options stay OFF.
- **Validation:** a paired A/B against the pre-rebase build, run on the same file in
  the same session. E4B W4A4 PPL was 26.8771 on both builds (32 chunks). Speed was
  equal or 1–3% faster in both configs. Unit tests: 269 + 1,420 + 341 checks, all
  green.
- **Board sync without pushing:** the container makes a `git bundle create x.bundle
  <base>..<branch>`, then `scp` copies it to the board. The board fetches upstream by
  full SHA from GitHub, then fetches the branch from the bundle. The rebased tree is a
  worktree at `~/rk-llama.cpp-rebase`. The pre-rebase tree and build at
  `~/rk-llama.cpp` are kept as the A/B baseline.
- **Board contents** (2026-10-02):
  - In `~/models/`:
    - `gemma-4-E4B-it-Q4_0-ggmlorg.gguf`: the measured E4B, sha `a555b900…`.
    - `gemma-4-E4B-it-Q4_0.gguf`: the current unsloth recipe, 720 tensors. Don't use it
      for comparisons.
    - `gemma-4-E2B-it-qat-q4_0.gguf` (Google).
    - `gemma-4-{E4B,E2B}-it-assistant-F16.gguf`: the MTP drafters, converted from the
      safetensors in `~/hf/`.
    - `LFM2.5-8B-A1B-Q4_0.gguf`.
  - Deleted to free disk (91% full before): ERNIE and `~/Bonsai-demo`.
  - The converter's Python env (with torch) is `~/venv-convert`.
  - The clock-pin script is now `~/rk3588-scripts/fix_freq_rk3588.sh`; the old
    `~/rknn-llm` path is gone. Clocks are **not** pinned after boot, so run it and check
    for `userspace`.
- **Logs:** `~/bench-logs/2026-10-02-rebase-ab/` holds the A/B, including the
  `ggmlorg/` subfolder. `2026-10-02-mtp-E4B{,-dftcpu,-omp4}/`, `2026-10-02-nthreads/`
  and `2026-10-02-mtp-E2B/` hold the MTP sweeps, batch-cost sweeps and the OMP-fix
  validation.
- **The board drifted after its 2026-09-24 reflash.** The pre-rebase build no longer
  reproduces some August speeds on the same file. E4B W4A4 llama-bench tg64 is now
  5.69 (was 6.89; `llama-server` still shows 6.89). Pure-CPU decode is now 6.74, which
  beats routed at 5.68 (it was 4.9 vs 5.50). Quality anchors reproduce exactly.
  Re-baseline before comparing against any pre-September number.
- **Operational lessons from this session:**
  - **"No route to host" to 192.168.0.x from the container is the sandbox firewall.**
    The user restarts the sandbox without it; credentials are not the issue.
  - **The container cannot reach huggingface.co.** Download on the board instead.
  - **The container has no cmake.** A portable cmake tarball from GitHub releases
    works. It also has only ~2 GB RAM: build with `-j1`, and use
    `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16` because native
    detection turns off fp16.
  - **Device selection can't keep anything off the NPU.** `--device none` and
    `--device-draft none` don't work because the backend is an ACCEL device; use
    `RKNPU_EXCLUDE` or `RKNPU_EXCLUDE_TYPES`.
  - **Stop servers with a single SIGINT.** Two signals in a row (`timeout` plus
    `kill`) trigger "Received second interrupt" and a segfault in librknnrt teardown.
    That is a test-harness artifact, not a bug.
- **Next opportunity:** a routed mode whose CPU-side copy uses the CPU_REPACK layout.
  It would put NPU prefill and the M=4 repack tile in one process: E4B ~43 t/s prefill
  plus ~10.7 t/s decode with MTP (#1b). Model candidates: `RKNPU2-model-landscape.md`.

- Everything lives on `github.com/unimatrix099/rk-llama.cpp`; the current
  tip branch is `feat/w4a4-neon-prep` (stacked on
  `feat/int4-native-layout` ← `feat/mixed-precision-pipelines` ←
  `fix/w4a4-calibration-crashes` ← upstream `rknpu2` + PR #21). The
  2026-08-10..21 sessions added the commits from "measured decode
  verdicts" through "32-chunk validation": tooling, profiling, the
  churn-free dispatch pool, block-diagonal Hadamard, and per-channel
  INT4 scales. NOTE: as of 2026-08-21 the stack is committed locally in
  the dev container and mirrored file-for-file on the board, but NOT yet
  pushed (no GitHub credentials on either machine) — push before
  anything else. The board's `libggml-rknpu2.so` is rebuilt from the
  same sources.
- The board (Orange Pi 5 Ultra) has the repo at `~/rk-llama.cpp`, models
  in `~/models/` (`gemma-4-E4B-it-Q4_0.gguf`, `gemma-4-E2B-it-Q4_0.gguf`,
  `qwen2.5-1.5b-instruct-q8_0.gguf`, `qwen2.5-0.5b-instruct-q8_0.gguf`,
  `LFM2-8B-A1B-Q8_0.gguf`), binaries in `build/bin/` incl.
  `llama-speculative`. Remember `ulimit -n 65536` and
  `scripts/fix_freq_rk3588.sh` (in `~/rknn-llm/scripts/`) before benching.
- Raw logs are archived on the board under `~/bench-logs/<date>/`:
  2026-08-10 (probe, sweeps, speculative), 2026-08-11 (perf/strace/gdb
  profiling, OMP experiments), 2026-08-13 (driver-shim re-profile),
  2026-08-16 (padding census, block-FWHT PPL matrix), 2026-08-20
  (per-channel validation, CPU PPL reference), 2026-08-21 (32-chunk
  hardening, chat smoke), 2026-08-22 (clip sweeps, B-curve, final
  validation).
- Measurement hygiene, learned the hard way: before benching, check for
  stale processes (`pgrep -af llama`) — a stuck llama-cli burned one core
  through part of the 2026-08-10 evening (numbers marked * in #1).
- Diagnostic tools build from `docs/backend/Makefile.rknpu2-tools`
  (`make all`, `check`, `check-prep`, `check-hw`, `coverage`).
- Reference numbers to beat: decode table above; prefill state in
  `RKNPU2-neon-prep-plan.md`.
