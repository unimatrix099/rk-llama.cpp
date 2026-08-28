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
| `OMP_NUM_THREADS=4` | unset | no longer required (the #3 code fix); still harmless |

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
