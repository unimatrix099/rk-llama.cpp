# RKNPU2 backend: optimization notes (RK3588)

A record of every performance optimization attempted for this backend on
RK3588 — what shipped, what failed, and the measurements behind each verdict.
Failed attempts are documented deliberately: each one closes off a
plausible-sounding path so future contributors don't re-walk it.

Companion document: `RKNPU2-W4A4-investigation.md` (crash root-causes, bug
fixes, environment details). All numbers: Orange Pi 5 Ultra (RK3588, 16 GB),
`llama-bench -p 128 -n 64` unless noted.

## The mental model that explains every result

- **Token generation (M=1) is memory-bandwidth bound.** Every token re-reads
  the full active weight set. Whoever reads fewer bytes wins; compute is
  nearly irrelevant.
- **Prompt processing (M large) is compute-amortized.** One weight read
  serves the whole batch, so the NPU's matmul throughput dominates and it
  beats the CPU.
- The backend upconverts weights to the pipeline's precision: a Q4_0 GGUF on
  `W8A8` doubles the NPU's per-token read vs the CPU's native Q4_0 read.
  This single fact is why NPU decode loses on dense models, and it framed
  every attempt below.

## Baselines

### Gemma 4 E4B Q4_0 (7.46 B dense — "effective 4B" applies only to
Google's edge stack, not llama.cpp; see investigation doc)

| Config | pp128 t/s | tg64 t/s |
|---|---|---|
| CPU only | 25.2 | 4.9 |
| NPU `W8A8_STANDARD` | 39.8 | 3.3 |
| NPU `W16A16_STANDARD` | 28.2 | 1.9 |
| NPU `W4A4_HADAMARD` (default for Q4_0) | 7.7 | 3.4 |
| NPU `W4A4_STANDARD` (control, broken accuracy) | 10.8 | n/a |
| NPU + `RKNPU_CPU_DECODE=32` (shipped) | 41.3 | 4.9 |
| **Current best, prefill (2026-08-23): W8A8 + `CPU_DECODE=32`, pinned t=4** | **41.4** | **5.50** |
| **Current best, memory (2026-08-22): default W4A4 (block-FWHT + per-channel + clipped)** | **37.0** | **5.49** |

(The default-threading rows above predate the threading guidance and the
churn fix; they remain as the historical baseline each entry below built
on. Cumulative: routed pp 39.8→41.4 / tg 3.3→5.50 at identical accuracy;
W4A4 pp 7.7→37.0 / tg 3.4→5.49 with PPL going from ~5x-over-CPU to level
with it — 26.95 vs 27.01 — at −29% NPU memory and seconds-long loads.)

### Others

> ⚠ **All LFM2 rows below are throughput-only and quality-INVALID.** As of
> 2026-08-22, MoE models produce garbage on the NPU (LFM2 PPL ~17400 vs
> 15.86 on CPU) — a pre-existing bug, see decode research #4b. The speed
> numbers are real; the outputs were not. Run MoE on CPU until fixed.

| Model | Config | pp128 | tg64 |
|---|---|---|---|
| LFM2-8B-A1B Q8_0 (MoE, ~1.5B active) | NPU `W8A8` | 48.3 | 9.1 |
| LFM2-8B-A1B Q8_0 | CPU only | 46.2 | 7.6 |
| LFM2-8B-A1B Q8_0 | NPU + `CPU_DECODE=32` | 47.8 | 7.7 ← regression at default threads; **13.66 at t=4 pinned** |
| Qwen2.5-1.5B Q8_0 | NPU `W8A8` (interactive) | ~50 | ~5.0 |
| Qwen2.5-1.5B Q8_0 | NPU + `CPU_DECODE=32`, t=4 pinned | 280.8 | **13.52** |

## ✅ Shipped: M-dependent routing (`RKNPU_CPU_DECODE=<M>`)

**Hypothesis:** since NPU wins prefill and CPU wins decode (dense models),
route per-op by batch size and get both.

**Implementation:** opt-in env var. `supports_op` rejects MUL_MAT with
M < threshold, so the scheduler runs decode on the CPU backend; the buffer
keeps the original GGUF bytes host-resident next to the packed NPU copy
(dual residency, `is_host` advertised) so the CPU computes in place with no
per-graph copies.

**Result:** E4B pp 41.3 / tg 4.9 — matches the better column of both
baselines simultaneously; +49% tg over NPU-only.

**Caveats (all measured or verified):**
- Opt-in on purpose: models whose NPU decode already beats CPU regress
  (LFM2 tg 9.1 → 7.7). Enable only when CPU-only tg > NPU tg for the model.
- One extra model-size resident copy in RAM.
- Requires mmap loading (default); `--no-mmap` would bypass `set_tensor`
  for host-visible buffers and the NPU copy would never be built.
- Prompts shorter than the threshold also run on CPU (M = prompt length at
  prefill), visible as low pp on very short prompts.

```sh
RKNPU_HYBRID=W8A8_STANDARD RKNPU_CPU_DECODE=32 \
  ./build/bin/llama-cli -m gemma-4-E4B-it-Q4_0.gguf ...
```

## ✅ Shipped: native A/C layout for INT4 pipelines (default on)

**Hypothesis (from the INT4 research doc):** the NORM-layout A/C matrices
force a serial per-run repack inside librknnrt that caps INT4 matmuls at
~44 GOPS; producing A in the NPU's native tiling and untiling C in the
existing dequant pass should recover the hardware's 3.6-3.9 TOPS.

**Result (phases 1-4 in `RKNPU2-native-layout-plan.md`):** E4B Q4_0
prefill 7.7 -> **31.9 t/s** (4.15x); Qwen2.5-1.5B forced W4A4 25.7 ->
**93.9 t/s** (3.7x); decode unchanged (layout-neutral at M=1, as
measured); results bit-identical to the NORM path; W8A8/W16A16
unaffected. Composes with `RKNPU_CPU_DECODE` (E4B: pp 31.9 / tg 5.35 at
half the NPU memory of W8A8). Opt out with `RKNPU_AC_NATIVE=0`.
W4A4 is thereby no longer the slowest NPU pipeline; the remaining gap to
W8A8 prefill is CPU-side activation prep (scalar Hadamard + INT4 pack).

## ✅ Shipped: NEON vectorization of the W4A4 CPU prep

**Hypothesis (RKNPU2-neon-prep-plan.md):** after the native-layout fix,
the W4A4 prefill gap to W8A8 is scalar CPU prep (FWHT, sign multiply,
amax, int4 pack, C dequant); NEON should recover much of it.

**Result:** kernel-level 3.0-4.0x (measured, `rknpu2-bench-prep`);
end-to-end E4B W4A4 pp 31.95 -> **34.44** (+8%), Qwen forced-W4A4 pp
93.88 -> **105.69** (+13%), W8A8 pp 41.10 -> 41.53 (free int8-quant
bonus). Gains dilute across OpenMP workers, hence less than raw kernel
speedups. Along the way the validation pipeline caught and fixed a
pre-existing int4 clamp bug (extreme outliers sign-flipped via int8 wrap
before clamping; only the broken W4A4_STANDARD pipeline was affected)
and replaced pointer-seeded Hadamard sign vectors with name-hash seeding,
making W4A4_HADAMARD reproducible across runs and builds for the first
time. 170 exactness checks (`check-prep`), touched functions at 100%
line coverage.

## ✅ Shipped as guidance: threading config (`-t 4` + `OMP_NUM_THREADS=4` + big cores)

**Hypothesis (decode research #5):** default threading (8 threads across
A55+A76) drags the critical decode path; big-cores-only could swing ±10%.

**Result (measured 2026-08-10/11, `rknpu2-affinity-sweep.sh` + profiling):**
far larger — tg64 +19% to +66% across every model tested, no code changes:
Qwen routed 8.93→13.52, E4B W8A8+routed 4.64→5.50, E4B NPU decode
3.43→4.55, LFM2 NPU 9.47→13.25, LFM2 routed 8.22→13.66. The old "~13 GB/s
CPU decode ceiling" was an 8-thread artifact; t=4 big-core reaches ~25
GB/s effective. LFM2's routing regression (see `RKNPU_CPU_DECODE` caveats
above) disappears at t=4.

Profiling then found the second half of the story (decode research #3):
the backend's per-node OMP regions request heterogeneous team sizes and
libgomp respawns its workers ~once per graph split (~550 threads/token on
E4B NPU decode). `OMP_NUM_THREADS=4` stops the churn (+42% on NPU decode
by itself) and fixes the strict-taskset pathology the sweep had flagged.

**Recommended:** `taskset -c 4-7 ... -t 4` (servers:
`--threads 4 --threads-batch 8`). E4B best measured: routed pp 41.4 /
tg 5.50. The backend code fix below made `OMP_NUM_THREADS=4` unnecessary
for the recommended routed config; it remains an optional micro-tune for
NPU-decode-only setups.

## ✅ Shipped: per-channel INT8 scales — W8A8 quality stops depending on the model

**Hypothesis (decode research #3e):** the +38% PPL that W8A8 costs on
Qwen (but not on E4B) is the same per-segment scale coarseness fixed for
INT4 in #3c, and the same zero-NPU-cost mechanism applies to the INT32
output path.

**Result (2026-08-22):** Qwen W8A8 PPL **12.24 -> 9.08** against a CPU
reference of 8.88; E4B 27.29 -> 27.61 (inside its ±1.15 bar). W8A8's
premium over CPU is now ~2% on both models instead of 1%/38% — the
variance across models collapsed, which is what makes pipeline guidance
trustworthy. **Costs 2.5% of E4B prefill** (routed pp 42.47 -> 41.39,
A/B'd on one build via `RKNPU_PER_CHANNEL=0`): the per-channel dequant
loads and multiplies per output element where the segment path used one
scalar. The original "speed unchanged" claim was measured on Qwen only,
where it genuinely is noise (pp 215.1 vs 215.8) — corrected 2026-08-23.
Still a clear win (Qwen W8A8 PPL 12.24 -> 9.08), but a trade, not free.
Bonus: E4B's *W4A4* default also improved (26.95 -> 26.88) because a
Q4_0 GGUF's non-Q4_0 tensors run W8A8. Same `RKNPU_PER_CHANNEL=0`
opt-out; legacy paths byte-for-byte intact. 263 kernel checks.

**Closed with a reason:** INT8 scale clipping (the #3d trick) is
catastrophic — `RKNPU_B_CLIP_INT8=0.95` gave PPL 10106, because
`quantize_fp32_to_int8` has no clamp and relies on `scale = amax/127`.
The knob was removed, not shipped. Second time this property has bitten;
audit any new scale factor against the quantizer's clamping.

## ✅ Shipped: clipped INT4 scales — W4A4 reaches the 8-bit quality tier

**Hypothesis (decode research #3d):** the per-channel change (below)
replaced an entropy search that was really a *clipping* search with a
plain amax, so granularity was won but clipping was lost; restoring
clipping as a constant factor should compose with granularity.

**Result (2026-08-22):** `scale = clip * amax / 7` with defaults
`RKNPU_A_CLIP=0.9`, `RKNPU_B_CLIP=0.93` (32-chunk sweep, two models;
the B-curve is a smooth U with a 0.93 minimum). E4B W4A4 PPL
**34.36 -> 26.95**, i.e. level with W8A8 (27.29) and pure CPU (27.01):
**the 4-bit NPU pipeline is no longer a quality compromise on this
model**, at -29% NPU memory and tg 5.5. Qwen forced-W4A4 26.87 ->
21.74 and independently prefers 0.93, so the defaults are not overfit.
Speed unaffected (paired same-build control). Set both to 1.0 for plain
amax; the four-var legacy combo still reproduces the original numerics
bit-exactly.

## ✅ Shipped: per-output-channel W4A4 weight scales (the quality fix)

**Hypothesis (decode research #3c):** most of W4A4's quality tier is the
one-scale-per-segment weight quantization, not the 4-bit activations —
and the channel dimension factors out of the hardware K-sum, so
per-channel scales are free in the existing C dequant pass.

**Result (2026-08-20):** wikitext PPL **228.9 -> 45.35 (5x)** at
unchanged speed (tg 5.54 / pp 36.9) and -29% NPU memory; W4A4 now costs
~20% PPL over W8A8 (37.8) instead of ~5x. Per-channel amax replaces the
segment entropy search, cutting W4A4 load from minutes to seconds. With
honest scales, blocked-FWHT also beats padded on quality (45.4 vs 49.1)
— the #3b trade-off is gone. W8A8 untouched (bit-identical); full-legacy
env combo reproduces the original build exactly; 254 kernel checks.
Opt-out: `RKNPU_PER_CHANNEL=0`.

## ✅ Shipped: block-diagonal Hadamard (W4A4 K-padding removed)

**Hypothesis (decode research #3b):** the Hadamard pipelines' pad-K-to-
next-pow2 inflates every W4A4 weight read and allocation ~1.5-1.6x on
non-power-of-two models (E4B: K=2560→4096, 10240→16384; GGUF census:
5.22 GB read/token vs 3.38 nominal).

**Result (2026-08-16):** FWHT applied per largest-pow2-divisor block
(2560→5x512, 10240→5x2048), K_op = K, divisor = block; legacy behavior
verified bit-identical behind `RKNPU_HADAMARD_BLOCK=0`. E4B W4A4:
tg 4.31→**5.51** (equals the routed path as a pure-NPU mode), pp
34.4→**38.0**, NPU model buffer 3569→**2521 MiB (−29%)**. Measured
quality cost on the already-degraded capacity mode: wikitext PPL
198→229 (W8A8 reference: 37.8); a min-block-1024 middle variant was
measured strictly worse (PPL 280, tg 5.13) and rejected. 203 kernel
checks. Cumulative W4A4 prefill since the start: 7.7 → 38.0 t/s (4.9x).

## ✅ Shipped: churn-free M=1 path in the backend (dispatch pool + if(M>1))

**Hypothesis (decode research #3):** the per-node OMP team respawn can be
eliminated in code so the env workaround isn't needed.

**Result (2026-08-11):** `if(M > 1)` on the A-prep/C-dequant regions and a
persistent 2-worker dispatch pool replacing the `num_threads(3)` run
region. Thread creations during decode -97%; E4B NPU decode tg +28%
(3.00→3.84), W4A4 decode tg +18% (3.65→4.31), routed at its best with no
env vars. Bit-identical outputs (W8A8 + W4A4 greedy), 170 kernel checks
green. Known trade-offs: W4A4 prefill -3%, and pure NPU decode stays
~0.4 t/s below the old-code+env optimum (pool wake latency vs GOMP's
hot-team dispatch — see decode research #3 for the structural fix if it
ever matters). Spin-harder/pre-wake variants measured and rejected.

## ✅ Shipped as guidance: pipeline pairing (rewritten 2026-08-21)

The original guidance ("W4A4 is 5x slower at prefill, capacity-only")
predates the native-layout, NEON-prep, dispatch-pool, block-FWHT and
per-channel-scale work. Current state on E4B Q4_0:

| Pipeline choice (E4B Q4_0) | pp128 | tg64 | PPL (32ch) | NPU memory |
|---|---|---|---|---|
| routed: `RKNPU_HYBRID=W8A8_STANDARD RKNPU_CPU_DECODE=32` | **41.4** | 5.50 | 27.01 (CPU-exact decode) | high (dual residency) |
| default W4A4 (no env needed) | 37.0 | 5.49 | **26.95** | **2.5 GB (−29%)**, CPU mostly free |
| `RKNPU_HYBRID=W8A8_STANDARD` (NPU decode) | 41.4 | 4.55 | 27.61 | high |

Pick **routed** for maximum prefill (41.4) and CPU-exact decode; pick
**default W4A4** for minimum NPU memory and a free CPU, now at the same
quality tier (the historical "W4A4 is a capacity mode, not a speed or
quality mode" guidance is obsolete as of #3d). Two caveats: how much of
the 4-bit tier a model absorbs is model-dependent (Qwen's W4A4 still
trails its CPU reference), and W8A8's per-segment scales are lossless on
E4B but cost +38% PPL on Qwen — check per model until per-channel INT8
scales ship (decode research #3c).

## ❌ Dead end: mixed-precision pipelines (W4A16 / W8A16)

**Hypothesis:** weight-only quantized NPU matmuls
(`RKNN_FLOAT16_MM_INT4_TO_FLOAT32` etc.) would halve decode weight traffic
vs W8A8, eliminate activation quantization error, and make the Hadamard
transform unnecessary — the same semantics that make GGUF Q4_0 work on CPU.
The types exist in the vendored `rknn_matmul_api.h`, and the fork already
has every building block (FP16 A-path, INT4 B-path, CPU-side scaling).

**Method:** standalone probe program creating matmul contexts for types
1/2/10 (symmetric) and 5/7/11 (mixed) at aligned shapes, NATIVE B layout.

**Result:** all symmetric types create successfully; **every mixed type
fails with `unsupported matmul dtype in this platform`** — a runtime
platform-capability rejection, not an alignment or layout issue. The
vendored librknnrt 2.3.2 is the latest Rockchip has released (checked
against airockchip/rknn-toolkit2), so there is no newer blob to swap in.
The mixed types evidently target other Rockchip silicon.

**Verdict:** not implementable on RK3588 until Rockchip enables these
dtypes for it. Revisit if a librknnrt newer than 2.3.2 ships — the probe
program takes minutes to re-run.

## ❌ Dead end: tensor exclusion / offload heuristics

**Hypothesis:** the giant vocab tensor (262144x2048, ~0.5 GB as INT8) or
the many tiny AltUp/LAUREL/per-layer matmuls (NPU dispatch overhead) drag
down E4B; excluding them from the NPU should help.

**Method:** `--override-tensor` with three configs on E4B `W8A8`:
big embedding/LM-head → CPU; small arch tensors → CPU; both.

**Result:** pp 40.2–40.5, tg 3.29–3.31 — all within noise of the 39.8/3.28
baseline.

**Verdict:** no individual tensor class matters; the cost is the aggregate
weight-byte volume. Size- or name-based offload filtering in
`resolve_op_support` is not worth building. (This null result is also what
pointed at bandwidth and motivated M-dependent routing.)

## ❌ Dead end: offloading the Hadamard transform to the NPU

**Hypothesis:** W4A4's per-token CPU Hadamard transform (needed to spread
activation outliers before 4-bit quantization) is the reason W4A4 prefill
is 5x slower than W8A8; the transform folds into a constant ±1 matrix and
could run as an NPU matmul (QuaRot-style, block-diagonal to keep the M=1
constant-matrix re-read affordable).

**Method (control experiment):** benchmark `W4A4_STANDARD`, which runs the
identical INT4 NPU path with zero transform cost (its output is numerically
broken, but the timing isolates the transform).

**Result:** pp only improves 7.7 → 10.8 t/s — still 3.7x slower than W8A8's
39.8. The dominant cost is the INT4 matmul path inside the closed runtime
itself, consistent with upstream's own dense-model benchmarks (Gemma3 1B:
INT4 NPU pp 51 vs INT8 NPU pp 378).

**Verdict:** even a perfect, free transform leaves W4A4 far behind W8A8.
Research-grade surgery (block-Hadamard weight+compute changes, accuracy
revalidation) to optimize the minor term of the slowdown is not worth it.

## ⏸ Assessed and deprioritized

- **QKV / gate-up matmul fusion** (fewer NPU dispatches, fewer of E4B's 603
  graph splits/token): the exclusion experiments showed dispatch/split
  overhead is not the decode bottleneck, so expected payoff is a minor
  prefill gain at moderate complexity. Revisit only if profiling shows
  dispatch overhead after other wins.
- **Calibration caching to disk**: W4A4's entropy/KL scale search costs
  minutes per load on big tensors; caching per-tensor scales (keyed by
  tensor hash) would fix reload times. Load-time QoL only — and less
  relevant while W4A4 remains a niche mode.
- **Speculative decoding** (Gemma 4 E2B/270M draft for E4B via
  `llama-speculative`): plausible tg multiplier, but draft and target share
  the same bandwidth-starved memory bus on this board, so gains are
  uncertain. Untested.

## Summary table

| Attempt | Verdict | Evidence |
|---|---|---|
| M-dependent routing (`RKNPU_CPU_DECODE`) | ✅ shipped | E4B tg +49%, pp preserved |
| W8A8 pairing guidance | ✅ shipped (docs) | fastest pipeline on all models tested |
| Native A/C layout for INT4 | ✅ shipped | E4B W4A4 pp 4.15x, bit-identical |
| NEON W4A4 prep | ✅ shipped | pp +8-13%, 170 exactness checks |
| Threading config (`-t 4` + `OMP_NUM_THREADS=4` + big cores) | ✅ shipped (docs) | tg +19-66%; +42% NPU decode from OMP fix alone |
| Mixed W4A16/W8A16 pipelines | ❌ platform-blocked | runtime rejects dtypes on RK3588 |
| Tensor exclusion heuristics | ❌ no effect | 3 configs within noise |
| Hadamard transform on NPU | ❌ not worth it | transform-free control still 3.7x slower |
| Cooperative CPU+NPU decode | ❌ probe: gate failed | +5-9% at fat shapes, <25% gate (decode research #2) |
| Speculative decoding (draft, ngram, NPU-verify) | ❌ all variants lose | decode research #1 tables |
| Backend overhead surgery (OMP churn) | ✅ shipped | dispatch pool + if(M>1): decode clone3 = 0, NPU tg +28%, W4A4 tg +18%, env-free (decode research #3) |
| QKV fusion | ❌ bounded out | re-profile: can only attack ~5 ms/token of driver misc (decode research #3 re-profile) |
| W4A4 K-padding (block-diagonal FWHT) | ✅ shipped | tg 4.31→5.51, pp 34.4→38.0, NPU mem −29% — decode research #3b |
| W4A4 per-channel weight scales | ✅ shipped | PPL 228.9→45.35 (5x, W8A8=37.8) at equal speed; load minutes→seconds; confirmed at 32 chunks + on Qwen — decode research #3c |
| Clipped INT4 scales (A/B clip) | ✅ shipped | E4B W4A4 PPL 34.36→26.95 = level with W8A8 (27.29) and CPU (27.01), free — decode research #3d |
| W8A8 per-channel scales (INT8) | ✅ shipped | Qwen W8A8 PPL 12.24→9.08 (CPU 8.88); W8A8 premium now ~2% on both models — decode research #3e |
| INT8 scale clipping | ❌ removed | wraps without a clamp: PPL 12.2→10106 — decode research #3e |
| **MoE models on the NPU** | 🐛 **broken, pre-existing** | LFM2 PPL ~17400 vs 15.86 CPU; two fix attempts reverted — decode research #4b |
| W4A4 quality generalization | ⚠ model-dependent | free on 7.5B dense, ~2x on 1.8B; source precision a minor factor — decode research #3f |
| Dropping the Hadamard transform | ❌ mandatory | without it PPL 35→26872; it costs 8.7% pp / 10.9% tg, the whole remaining gap to W8A8 — decode research #3g |
| Smaller FWHT blocks | ❌ bad trade | +4.8% speed for +58% PPL; natural block confirmed optimal — decode research #3g |
| FWHT stage-fusion | ❌ no longer motivated | blocks are L1-resident since #3b; the memory-traffic premise is gone — decode research #3g |
| Transform reuse via shared sign vectors | ❌ blocked on E4B | +43% PPL for ~4% speed; per-tensor signs are load-bearing there — decode research #3h |
| `RKNPU_SHARED_SIGNS` as a quality knob | ⚠ model-dependent | −8% PPL on Qwen, +43% on E4B; default off — decode research #3h |
| Hadamard transform on the GPU | ❌ not viable | ~600 dispatches/token at ~50us each exceeds the ~20ms of work it would replace; marginal at prefill only, needs a GPU backend plus cross-device plumbing |
| set_tensor packing unalignable tensors | ✅ fixed | crashed at load whenever the NPU was offered CLIP; dense dims happen to be aligned so it never fired — decode research #3i |
| CLIP vision encoder on the NPU | ❌ 5% only | 227 graph splits vs 1: the backend takes only 2D MUL_MAT, so the ViT fragments — decode research #3i |
| Mali GPU via Vulkan/panvk | ❌ unavailable | vendor BSP binds the GPU to the proprietary driver; no panfrost DRM node, mesa sees only llvmpipe — decode research #3i |
| Mali GPU via OpenCL | ⏭ blocked by an allowlist | stack verified working (OpenCL 3.0, fp16, subgroups); ggml-opencl supports Adreno/Intel only, and its subgroup size sizes local memory — needs a real port — decode research #3i |
| Calib cache | ⏸ load-time QoL | unchanged |
