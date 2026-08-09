# Implementation plan: native A/C layout for the INT4 pipelines

Target: `invisiofficial/rk-llama.cpp`, RKNPU2 backend
(`ggml/src/ggml-rknpu2/`). Companion to `RKNPU2-int4-research.md`, which
holds the measurements this plan is built on.

## 1. Summary

The backend creates every matmul context with
`info.AC_layout = RKNN_MM_LAYOUT_NORM` (`ggml-rknpu2.cpp`,
`rknpu_matmul_context` constructor). For INT4 this routes every
`rknn_matmul_run` through a serial per-row repack inside the closed runtime
that caps throughput at ~44 GOPS regardless of shape. With
`RKNN_MM_LAYOUT_NATIVE` for A/C the same matmuls run at **3.6–3.9 TOPS —
an ~80x speedup, and ~2x faster than the INT8 path the backend prefers
today** (measured; see research doc, `rknpu2-matmul-layout-bench.c`).

Plan: produce the A matrix directly in the NPU's native tiling during the
existing activation-quantization pass, and unpack C inside the existing
dequantization pass, for the INT4 pipelines only, behind an env kill-switch
until validated.

## 2. Research: the native formats (probed, RK3588, librknnrt 2.3.2)

`layout_dims_probe.c` results, all with B_layout=NATIVE:

```
type  AC      M    K     N     A dims        C dims       sizes
INT4  NATIVE  1    2048  2048  [64,1,32]     [256,1,8]    A 1024B  C 4096B
INT4  NATIVE  2    2048  2048  [64,2,32]     [256,2,8]
INT4  NATIVE  5    2048  2048  [64,5,32]     [256,5,8]
INT4  NATIVE  32   2048  2048  [64,32,32]    [256,32,8]
INT4  NATIVE  128  2048  2048  [64,128,32]   [256,128,8]
INT4  NATIVE  32   1024  2048  [32,32,32]    [256,32,8]   (K scales dim0)
INT4  NATIVE  32   2048  4096  [64,32,32]    [512,32,8]   (N scales C dim0)
INT8  NATIVE  M    2048  2048  [128,M,16]    [512,M,4]
```

Established facts:

- **A native (INT4): `[K/32, M, 32]`** — element `(m, k)` lives in K-tile
  `k/32`, row-interleaved. One `(tile, m)` cell is 32 int4 = **16 bytes,
  contiguous**. Byte address of element (m,k), nibble-packed:
  `((k/32) * M + m) * 16 + (k % 32) / 2`, low nibble first (matching the
  existing `quantize_fp32_to_int4_packed` pairing; to be confirmed in the
  phase-1 equivalence test).
- **C native (INT4→INT16): `[N/8, M, 8]`** — element `(m, n)` at int16
  index `((n/8) * M + m) * 8 + (n % 8)`.
- **INT8 native**: A `[K/16, M, 16]`, C(INT32) `[N/4, M, 4]` — same
  pattern, different granules. (Deferred; see §4.)
- **No M padding**: arbitrary M (1, 2, 5) creates fine; buffer sizes equal
  NORM sizes exactly. No new alignment constraints appear.
- **Same total bytes** — memory pressure and existing size bookkeeping are
  unaffected.

Where today's time goes: our timing loop contains only `rknn_matmul_run`,
so the NORM penalty is paid **inside every run call** (per-run repack), not
in `set_io_mem`. Producing native A directly is therefore pure win — the
backend writes the same number of bytes it already writes, just at tiled
addresses, and the runtime's per-run repack disappears.

## 3. Design decisions

1. **Per-pipeline layout, not global.** Add
   `rknn_matmul_layout ac_layout;` to `Rknpu2HardwarePipeline`
   (`rknpu2-configuration.h`). Set `RKNN_MM_LAYOUT_NATIVE` for
   `W4A4_STANDARD` and `W4A4_HADAMARD`; keep `RKNN_MM_LAYOUT_NORM` for all
   others.
2. **INT4 first, INT8 deferred.** The INT8 NORM penalty is shape-dependent
   and can invert: at K=1024/N=8192 native wins 554→1877 GOPS, but at
   K=2048/N=2048 native *loses* 1693→1347. Enabling native INT8 needs a
   per-shape decision layer (or measurement across the target model's real
   shapes) — out of scope here. The INT4 case has no such ambiguity:
   native wins ~80x at every shape probed.
3. **Kill-switch, then default.** Phase 2 ships behind
   `RKNPU_AC_NATIVE=0|1` (default on for INT4 after phase 3 validation);
   the flag stays as a diagnostic.
4. **No format translation helpers from the SDK are assumed.** The header
   offers `rknn_B_normal_layout_to_native_layout` for B only; A/C tiling is
   done by the backend itself, mirroring how `pack_native()` already
   hand-tiles B.

## 4. Code changes, by anchor

All in `ggml/src/ggml-rknpu2/` unless noted. Line anchors refer to the
`research/int4-acceleration` branch.

### 4.1 `rknpu2-configuration.h` / `.cpp`
- Add `rknn_matmul_layout ac_layout;` to `Rknpu2HardwarePipeline` (after
  `mm_type`).
- Update the six RK3588 pipeline initializers: NATIVE for the two W4A4
  entries, NORM elsewhere. (Aggregate initializers — all six entries must
  gain the field.)

### 4.2 `ggml-rknpu2.cpp` — context creation (~line 331,
`rknpu_matmul_context` ctor)
- Ctor gains a layout parameter (or takes the pipeline pointer);
  `info.AC_layout = layout;`.
- `get_matmul_ctx` cache: its key already includes `matmul_type`, and
  layout is a pure function of pipeline (hence of type) in this design, so
  **no key change is required** — but add the layout to the key anyway if
  the env kill-switch can flip it at runtime mid-process (it cannot if the
  flag is read once into a static; do that).

### 4.3 A-matrix fill (graph_compute step 2, ~lines 628–690)
Current: per row `m`, quantize `ready_row[K_seg_op]` into
`dst_row = dst_ptr + m * (K_seg_op / 2)` (row-major nibbles).

New, when the pipeline's layout is NATIVE and type is INT4:
- Quantize the row into a small stack/scratch buffer (`K_seg_op/2` bytes)
  exactly as today (reuses `quantize_fp32_to_int4_packed`, including its
  scale handling), then scatter it as `K_seg_op/32` contiguous 16-byte
  cells to `dst_base + ((t * M_op + m) * 16)` for tile `t`.
  A 16-byte `memcpy`/NEON store per cell; the scatter adds one extra pass
  over K_seg_op/2 bytes — negligible against the matmul.
- Alternative (skip the scratch): quantize directly into the tiled
  destination; consecutive `k` share a cell until the 32-boundary, so the
  inner loop only changes its output pointer every 16 nibble-pairs. Choose
  whichever benches better; start with the scratch version for clarity.
- `M_op` (the padded M used at context creation) is the interleave stride —
  **not** the actual row count M. Rows `M..M_op-1` are never written and
  never read back (C unpack iterates real M only), matching today's
  behaviour with NORM.
- The FP16 (W16A16) and INT8 (W8A8) branches are untouched.

### 4.4 C-matrix read (graph_compute step 5, ~lines 725–770)
Current: per row `m`, per segment, walk `n` linearly over
`src_segment_base + m * N_segment`, applying
`dst_ptr[n] += src_ptr[n] * dequant_scale`.

New, NATIVE + INT16 C: index `src16[((n/8) * M_op + m) * 8 + (n % 8)]`.
Restructure the inner loop as: for each 8-wide N-tile, load 8 consecutive
int16 (one cell — contiguous), dequant-accumulate into `dst_ptr[n..n+7]`.
This *fuses* the untiling into the pass that already exists, so the added
cost is address arithmetic only. NEON: `vld1q_s16` per cell → widen →
`vcvtq/vmla` — same operation count as today.

### 4.5 A/C buffer cache (`get_tensor_buffer`, backend ctx caches)
- Buffer sizes come from `io_attr` (unchanged values, per §2), so the
  existing cache keys `(M_op, K_seg_op, type, domain)` for A and
  `(M_op, N_seg, core, type, domain)` for C remain valid. No change beyond
  reading `io_attr` from a context created with the new layout.

### 4.6 Documentation
- `RKNPU2-optimization-notes.md`: move this from "research" to "shipped"
  with before/after numbers once validated.
- README section for the backend if upstreaming (`RKNPU_AC_NATIVE` flag).

## 5. Validation plan (phased)

**Phase 1 — standalone numerical equivalence (no backend changes).**
Extend `rknpu2-matmul-layout-bench.c` into a correctness harness:
random A/B (INT4), run the same multiplication twice — NORM layout with
row-major A vs NATIVE layout with hand-tiled A (formulas of §2) — and
compare C element-wise after untiling (must be bit-identical: same
integers in, same MACs). This pins the exact tiling/nibble order **before**
touching the backend and is the cheapest place to discover, e.g., a nibble
endianness surprise. Half a day.

**Phase 2 — backend implementation behind `RKNPU_AC_NATIVE`.**
Changes of §4. Smoke: Qwen2.5-1.5B-Q4_0 forced through
`RKNPU_HYBRID=W4A4_HADAMARD` — token-identical output vs flag-off at
temperature 0 (the transform and quantization are unchanged; only layouts
move). One day including debugging.

**Phase 3 — performance + quality matrix.** On Gemma-4-E4B-Q4_0 and
Qwen2.5-1.5B-Q4_0:
- `llama-bench -p 128 -n 64`: W4A4_HADAMARD flag-off vs flag-on; W8A8
  reference; `RKNPU_CPU_DECODE` combinations.
- `llama-perplexity -f wiki.test.raw --chunks 32`: flag-on must equal
  flag-off exactly (same math). Any drift = phase 1 missed something.
- 5x stress loads (the W4A4 load path has history; see investigation doc).

**Phase 4 — flip default for INT4 pipelines, upstream PR** with the
research doc + benchmarks attached. Keep the flag as an opt-out.

## 6. Expected results (honest projection)

- Matmul-level: ~80x on the INT4 matmuls themselves (measured ceiling).
- End-to-end W4A4 prefill: the NORM-layout matmul cost dominates today's
  7.7 t/s (a K=2048/N=2048 matmul: 24.2 ms NORM vs 0.30 ms NATIVE). With it
  gone, prefill should land in the **W8A8 neighbourhood (30–40 t/s), with
  the CPU-side Hadamard + scalar INT4 packing as the new limiter** —
  that is when NEON-vectorising the prep (and possibly the GPU idea from
  the research doc) becomes the next lever. A conservative floor is the
  W4A4_STANDARD control (10.8) times whatever fraction of its time was
  matmul — expect well above 20.
- Decode: unchanged (~3.4 t/s; layout is timing-neutral at M=1, measured).
  `RKNPU_CPU_DECODE` remains the decode answer.
- Quality: identical to current W4A4 (same quantized values, same MACs) —
  which means still perplexity-degraded vs W8A8. The speed win makes the
  quality trade *available*, not free.
- Memory: unchanged.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Nibble order inside the 16-byte cell differs from `quantize_fp32_to_int4_packed`'s pairing | Phase 1 harness pins it before backend work |
| Native A interleave uses padded M_op stride vs actual M in ways the probe didn't reveal (probe showed dims `[K/32, M, 32]` with *exact* M) | Contexts are already created with M_op, so dims report M_op consistently; harness tests M != pow2 via a ctx created at that exact M... and the backend always creates at M_op — covered |
| Runtime version changes tiling silently | Dims are read from `io_attr` at ctx creation — derive tile geometry from `A.dims`/`C.dims` at runtime instead of hard-coding 32/8 (cheap, robust; do this) |
| INT8 temptation | Explicitly out of scope; the anomaly table in the research doc says measure per-shape first |
| Hybrid patterns mixing NORM and NATIVE contexts | Each `rknpu_matmul_context` carries its own layout; A/C buffers are cached per type — no cross-contamination. Covered by a hybrid-pattern smoke test in phase 3 |

## 8. Effort estimate

- Phase 1: 0.5 day
- Phase 2: 1 day
- Phase 3: 0.5 day (mostly machine time)
- Phase 4 (PR write-up): 0.5 day

Total: ~2.5 days of focused work, of which the genuinely novel part —
the tiling formulas — is already established by the probes above.

## Phase 1 results (complete)

Delivered on branch `feat/int4-native-layout`:

- `ggml/src/ggml-rknpu2/rknpu2-native-layout.{h,c}` — the tiling module
  Phase 2 will reuse (geometry parse from io_attr dims, scatter/gather).
  **Coverage: 100% lines, 100% branches** (make -f Makefile.rknpu2-tools
  coverage), 341 unit checks.
- `docs/backend/test-rknpu2-native-layout.c` — CPU unit tests (TDD:
  written before the implementation).
- `docs/backend/test-rknpu2-native-equivalence.c` — on-NPU test:
  **bit-identical NORM vs NATIVE results and 100% agreement with a CPU
  int32 reference** at M=1/K=256/N=256, M=5/K=320/N=256 (non-pow2 M),
  M=32/K=2048/N=2048 and M=128/K=1024/N=512.

Contract confirmations and findings for Phase 2:

1. A native tiling `[K/32, M, 32]` with plain byte-copy of 16-byte cells is
   exactly right — nibble order inside cells matches
   `quantize_fp32_to_int4_packed` (low nibble = even element); the
   nibble-swap hypothesis is refuted.
2. C native `[N/8, M, 8]` int16 gathers back with the same generic cell
   copy.
3. `rknn_mem_sync` TO_DEVICE after CPU writes / FROM_DEVICE before CPU
   reads is mandatory for correctness on cached mappings (the backend
   already does this; standalone tools must too).
4. `rknn_B_normal_layout_to_native_layout` takes **unpacked** int4 input
   (one value per byte, K*N bytes). Feeding nibble-packed data makes it
   read K*N/2 bytes past the buffer — silent heap overread producing
   nondeterministic results. Worth remembering for any future tool.

## Phase 2 results (complete)

Implementation as specified in §4, gated behind `RKNPU_AC_NATIVE=1`
(read once at config construction; contexts and cached buffers depend on
it, so it must not change mid-process).

Measured (llama-bench pp128/tg64, default W4A4_HADAMARD):

| Model | flag off | flag on | speedup |
|---|---|---|---|
| Gemma-4-E4B Q4_0 | pp 7.7 / tg 3.4 | **pp 31.95 / tg 3.54** | 4.15x pp |
| Qwen2.5-1.5B Q8 (forced W4A4, llama-cli) | pp 5.6 | **pp 44.8** | 8x pp |

The remaining gap to W8A8 (39.8 pp on E4B) is CPU-side activation prep
(Hadamard transform + scalar INT4 packing), as predicted — that is the
next lever, not the matmul.

Correctness validation:
- Hardware equivalence extended to prefill-scale shapes
  (M=512, K=8192/N=1536, K=2048/N=8960): still bit-identical, still 100%
  vs CPU reference.
- End-to-end identity: `W4A4_STANDARD` perplexity (chunks to 4 decimal
  places) is bit-identical flag-on vs flag-off, and deterministic across
  runs for both.
- `dims` in `rknn_matmul_tensor_attr` is `uint32_t[]`; the tiling module
  signature was aligned accordingly (C++ would reject the implicit cast).

**Incidental finding (pre-existing, worth an upstream issue):**
`W4A4_HADAMARD` results are not reproducible across process runs — the
per-tensor Hadamard sign vectors are seeded from the tensor's memory
address (`std::mt19937 gen(reinterpret_cast<uintptr_t>(tensor))`,
ggml-rknpu2.cpp), which changes with ASLR. Perplexity on the same model
and text varied 19.7 vs 58.6 across identical flag-off runs. Seeding from
a stable key (e.g. tensor name hash) would make results reproducible.
Not addressed by this change; both flag states inherit it equally.
