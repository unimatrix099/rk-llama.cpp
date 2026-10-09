# RKNPU2: making INT4 (W4A4) useful on RK3588 — research notes

Open research thread. Unlike the companion documents
(`RKNPU2-W4A4-investigation.md` = bug fixes, `RKNPU2-optimization-notes.md`
= shipped/failed optimizations), **nothing here is implemented**. This
records the problem framing, candidate approaches, and the experiment that
should be run first because its outcome decides whether the rest is worth
building.

Every number labelled *measured* comes from this board (Orange Pi 5 Ultra,
RK3588, `llama-bench -p 128 -n 64`); everything else is explicitly a
hypothesis.

## Why bother with INT4 at all

Token generation on RK3588 is memory-bandwidth bound: each token re-reads
the whole active weight set. On the NPU the floor is 8-bit weights (see
"hardware constraints" below), so a 7.46 B model like Gemma-4-E4B drags
~7.5 GB across the bus per token. INT4 weights would halve that — the single
biggest lever available on this chip for decode speed.

The catch, measured: today's INT4 pipeline is the *slowest* NPU option we
have, not the fastest.

| E4B Q4_0 pipeline | pp128 t/s | tg64 t/s | note |
|---|---|---|---|
| `W8A8_STANDARD` | 39.8 | 3.3 | best NPU prefill |
| `W16A16_STANDARD` | 28.2 | 1.9 | fattest reads |
| `W4A4_HADAMARD` (default for Q4_0) | 7.7 | 3.4 | *slowest prefill* |
| `W4A4_STANDARD` | 10.8 | n/a | accuracy broken; timing control only |
| CPU only | 25.2 | 4.9 | reads Q4_0 natively |
| NPU + `RKNPU_CPU_DECODE=32` (shipped) | 41.3 | 4.9 | current best |

So INT4 currently buys memory capacity (halves NPU residency) and costs
speed. The research question: is that inherent, or fixable?

## Hardware constraints (established, not negotiable)

- The RK3588 matmul API accepts **symmetric dtypes only**: FP16xFP16,
  INT8xINT8, INT4xINT4. Probed directly (see
  `rknpu2-matmul-probe.c` next to this file): every mixed combination
  (`FLOAT16_MM_INT4`, `FLOAT16_MM_INT8`, `INT8_MM_INT4`) is rejected with
  `unsupported matmul dtype in this platform`. Rockchip's own RKLLM stack
  agrees from the other side: w8a8 only on RK3588, w4a16 from RK3576 up.
- Consequence: **there is no "4-bit weights, 16-bit activations" mode.**
  Using 4-bit weights on the NPU forces 4-bit activations too — and 4-bit
  activations are where the accuracy collapses (upstream README, measured on
  Granite-4.0-350M: `W4A4_STANDARD` perplexity 240048 vs `W8A8_STANDARD`
  22.5). That is the entire reason the Hadamard transform exists in this
  backend: it smooths activation outliers so INT4 activations are merely bad
  (86.16) instead of catastrophic.
- The NPU is **fixed-function**: a MAC array fed by descriptors, not a
  programmable core. A fused "read 4-bit, widen on-chip, multiply at 8-bit"
  stage cannot be added in software — that widening front-end is exactly
  what RK3576's newer NPU added and RK3588 lacks.

Engine map for this board, for anyone reasoning about where work can move:

| Engine | Programmable | Can fuse dequant into the matmul read? |
|---|---|---|
| CPU (4xA76 + 4xA55) | yes | **yes — does it today**; that is why CPU decode (4.9) beats NPU decode (3.3) |
| NPU (3 cores) | no (descriptor menu) | no — missing silicon stage |
| GPU (Mali-G610) | yes (OpenCL) | yes in principle; llama.cpp's OpenCL backend is tuned for Adreno, community reports Mali ≈ CPU or worse, and it shares the same LPDDR5 bus |

## Idea 1: prepare activations on the GPU instead of the CPU

**Proposal.** The per-token activation prep for W4A4 (sign flip, fast
Walsh-Hadamard transform, amax, pack to INT4) runs on the CPU today
(`rknpu2-calibration.cpp`, `ggml-rknpu2.cpp` A-matrix path). Move it to a
Mali OpenCL kernel: the FWHT is a butterfly pattern, embarrassingly
parallel, and the SoC allows zero-copy sharing of the buffers.

**Assessment: technically sound, but capped low by measurement.**
`W4A4_STANDARD` is the perfect control for this idea — same INT4 NPU path,
*zero* transform cost, i.e. the ceiling an infinitely fast prep engine could
reach. It gets 10.8 t/s prefill, versus 39.8 for W8A8. So eliminating 100%
of the prep cost recovers ~3 t/s and still leaves INT4 ~3.7x behind INT8.
At decode the prep is already negligible (W4A4 tg 3.4 ≈ W8A8 tg 3.3 despite
running the transform).

Additional risk: E4B issues ~250 MUL_MAT nodes per token; a GPU hop per node
adds OpenCL enqueue/sync latency that could exceed the ~3 t/s prize.

**Verdict: back on the table, but only after the layout fix.** Idea 0 (below)
found that the 10.8 t/s ceiling this assessment was built on is an artefact
of the backend's A-matrix layout, not a hardware limit. Once INT4 matmuls run
at native-layout speed, CPU-side activation prep becomes the dominant
per-token cost and moving it to the GPU — or, more likely, simply
NEON-vectorising it, since it is currently scalar — becomes worthwhile.
Note the GPU kernel would then also have to emit A in the NPU's native
layout, which is arguably a better fit for a GPU than for scalar CPU code.

## Idea 0 — RESOLVED: INT4 is fast, but only with a native-layout A matrix

**Headline: the backend leaves ~80x of INT4 matmul performance on the table
by using `RKNN_MM_LAYOUT_NORM` for the A/C matrices.** This is an SDK
behaviour, not a silicon limit.

`ggml-rknpu2.cpp:357` sets `info.AC_layout = RKNN_MM_LAYOUT_NORM`. Sweeping
that one flag (B stays NATIVE, single core, pinned clocks,
`rknpu2-matmul-layout-bench.c`):

```
M=128 K=1024 N=8192        AC=NORM              AC=NATIVE
  INT8xINT8            3.875 ms   554 GOPS    1.144 ms  1877 GOPS
  INT4xINT4           48.186 ms    45 GOPS    0.593 ms  3623 GOPS   (81x)

M=128 K=2048 N=2048        AC=NORM              AC=NATIVE
  INT8xINT8            0.634 ms  1693 GOPS    0.797 ms  1347 GOPS
  INT4xINT4           24.171 ms    44 GOPS    0.300 ms  3578 GOPS   (81x)

M=128 K=2048 N=8192        AC=NORM              AC=NATIVE
  INT8xINT8            3.974 ms  1081 GOPS    3.102 ms  1384 GOPS
  INT4xINT4           96.171 ms    45 GOPS    1.110 ms  3868 GOPS   (87x)

M=1   K=2048 N=2048    (layout makes no difference at M=1)
  INT8xINT8            0.390 ms    21 GOPS    0.390 ms    21 GOPS
  INT4xINT4            0.203 ms    41 GOPS    0.203 ms    41 GOPS
```

With native A/C layout INT4 reaches **3.6-3.9 TOPS**, is consistently ~2x
faster than INT8, and matches the ~4 TOPS independently measured by
marty1885 (see references). With NORM layout INT4 is pinned at ~44 GOPS
regardless of shape — a flat ~189 us per row of A, i.e. a serial
per-row repacking path, presumably nibble-level reordering that the runtime
performs on the CPU. INT8 pays a much smaller and shape-dependent NORM
penalty (sometimes none), which is why this went unnoticed: **only INT4 is
catastrophically penalised.**

This fully explains the end-to-end W4A4 prefill deficit (7.7 vs 39.8 t/s):
the backend is running INT4 matmuls at 2.6% of their achievable speed.

### Corrected consequences

- **INT4 prefill is not hopeless — it is currently mis-configured.** With
  native A/C, INT4 matmul is ~2x *faster* than the INT8 path the backend
  uses today, on top of halving weight bandwidth.
- **The fix is real work, not a one-line flag.** `AC_layout` governs A *and*
  C together, so the backend would have to (a) produce activations directly
  in the NPU's native A layout instead of row-major, and (b) unpack C from
  native layout after each run. The backend already does exactly this kind
  of tiling for B (`pack_native()`), so the machinery exists.
- **INT8 may gain too**, shape-dependently (554 -> 1877 GOPS at
  K=1024/N=8192; roughly neutral at K=N=2048). Worth measuring against real
  model shapes before changing the default.
- **Decode (M=1) is unaffected** — layout is irrelevant for a single row,
  and both dtypes remain purely bandwidth-bound at ~10 GB/s per core, with
  INT4 1.9x faster purely by reading half the bytes. The
  `RKNPU_CPU_DECODE` routing remains the best decode option.
- **Accuracy is untouched by any of this**: W4A4 would still carry its
  4-bit-activation quality penalty (perplexity 86 vs 22.5 for W8A8 on
  Granite-350M). A fast W4A4 is a speed/quality trade, not a free win.

### Correction notice

An earlier revision of this document concluded "INT4 compute is not
accelerated on RK3588, ~2.6% of INT8 peak, not fixable in a backend". That
conclusion was **wrong**: it was measured only with `AC_layout=NORM`,
inherited from the backend's own configuration, and mistook an SDK layout
penalty for a hardware limit. The error was caught by checking published
third-party benchmarks, which reported ~4 TOPS INT4 and prompted the layout
sweep. Retained here rather than deleted, as a caution: when probing a
closed runtime, sweep the configuration flags before concluding anything
about the silicon.

## Superseded initial measurement (kept for the record)

The first pass measured only `AC_layout=NORM` and produced the flat
~44 GOPS INT4 numbers reproduced in the table above (right-hand column of
the NORM columns). Its conclusions — "INT4 is serial per-row", "2.6% of
INT8 peak", "not fixable in a backend", and the resulting dismissal of
ideas 1/2/4 — are **withdrawn**. See the correction notice above.

## Idea 0 (original framing, kept for context): where does INT4's deficit live?

The measurements above say the cost is *inside the INT4 path*, but not
whether that is the silicon or this backend's INT4 plumbing. Suspects on the
software side, all visible in the source:

- `rknpu2_quantization::quantize_fp32_to_int4_packed()` is a **scalar**
  loop with per-element rounding and nibble packing, while the FP16 path
  (`convert_fp32_to_fp16`) and the dequant paths are NEON-vectorised. The
  INT8 quantizer is scalar too.
- `fwht_iterative()` is scalar, and the Hadamard path pads K to the next
  power of two (`next_power_of_two`), so tensors with non-power-of-2 K do
  extra work on padded data.
- INT4 requires `n_align = 64` (vs 32 for INT8), producing different
  segmentation, and the INT4 matmul returns INT16 (`INT4_MM_INT4_TO_INT16`)
  needing a separate dequant pass.

**Experiment.** Extend the probe next to this file to *time* raw
`rknn_matmul_run` for INT8 vs INT4 at identical M/K/N, with no llama.cpp,
no packing, no scales in the loop — just steady-state NPU throughput.

Outcomes and consequences:

- **INT4 raw ≈ INT8 raw (or faster).** The deficit is in this backend's
  plumbing → vectorise the INT4 quantiser/packer and the FWHT, revisit
  segmentation. INT4 prefill could approach INT8, decode could finally
  exploit halved weight traffic, **and Idea 1 becomes worthwhile**.
- **INT4 raw ≈ 4x slower than INT8.** The silicon's INT4 mode saves
  bandwidth but not compute (plausibly unpacked internally to the same MAC
  width). Then W4A4 can never win on speed, and its only honest use on
  RK3588 is fitting larger models under the 4 GB/IOMMU-domain limit. Ideas
  1-3 should be dropped.
  *(Outcome: this branch was not taken. The slowness was the runtime's
  serial per-run A repack, not the silicon — the native A/C layout took
  E4B prefill 7.7 → 31.9 t/s, so W4A4 does win on speed. The "4 GB"
  figure is also wrong: the cap is ~2 GiB per domain and the allocator
  uses 16 of them, so it was never a model-size ceiling — see decode
  research "The IOMMU domain limit".)*

Cost: ~15 minutes. Everything else in this document is gated on it.
*(Done — see the resolved section above.)*

## Idea 4 (new, post-measurement): M-dependent *pipeline* selection

The measurement says INT4 is good at exactly one thing: M=1 matmul latency,
where it is 1.9x faster than INT8 because it reads half the bytes. That is
the mirror image of prefill, where it is 38x slower.

So rather than choosing one pipeline per tensor for the whole run, choose
per *call*: **INT8 when M is large, INT4 when M == 1.** The backend already
has the plumbing — `RKNPU_CPU_DECODE` demonstrates M-dependent dispatch in
`supports_op`, and `resolve_op_support` already selects pipelines per
tensor; this would extend selection to depend on the activation batch size,
keeping both a packed INT8 and a packed INT4 copy of each weight resident
(1.5x weight memory).

Honest expectations, from the numbers above: the ceiling is the ~0.19 ms
saved per matmul at M=1, against a token time dominated by non-matmul work,
and the result would still have to beat CPU decode at 4.9 t/s — which reads
4-bit weights natively with zero dispatch overhead. **Probably not worth it
on RK3588**; recorded because it is the only remaining shape in which INT4
is not strictly worse, and because on a chip with a real INT4 datapath (or
w4a16, i.e. RK3576+) the same design would be clearly correct.

## Idea 2: remove the runtime transform instead of accelerating it

The Hadamard exists to tame activation outliers *at runtime*. Two families
of offline alternatives could remove or shrink that per-token work:

- **SmoothQuant-style migration** ([2211.10438](https://arxiv.org/abs/2211.10438)):
  scale activation outliers down and fold the inverse scale into the weights
  once at load time. Zero per-token cost, no transform at all. Unknown
  whether it is sufficient at 4-bit activations (it is usually applied at
  8-bit), but it is a load-time experiment, not an architecture change.
- **Fold the rotation into adjacent weights** (QuaRot
  [2404.00456](https://arxiv.org/abs/2404.00456), SpinQuant
  [2405.16406](https://arxiv.org/abs/2405.16406)): most rotations are
  mathematically absorbable into the neighbouring weight matrices offline;
  only a couple of positions per layer genuinely need an online transform.
  That would cut runtime transforms from ~7 per layer to ~2 — and *those*
  would be the right target for Idea 1's GPU kernel.

Both change accuracy characteristics and need perplexity validation
(`llama-perplexity -f wiki.test.raw --chunks 32`, the upstream methodology).

## Idea 3: spend INT4 only where it pays

`RKNPU_HYBRID` already applies a pipeline pattern cyclically per tensor, so
mixed strategies are testable **today with no code changes** — e.g. INT4 on
the large FFN weights (bandwidth relief where the bytes are) and INT8 on
attention (accuracy where activations are most outlier-prone):

```sh
RKNPU_HYBRID="W8A8_STANDARD,W4A4_HADAMARD" ./build/bin/llama-bench -m model.gguf
```

Nobody has mapped this trade-off curve on a real model. Cheap to explore
(pure benchmarking + perplexity), independent of Idea 0's outcome, and it is
the only listed idea that could improve *decode* on the NPU without new
hardware capability.

## Priority (revised after the layout finding)

1. ~~**Idea 0**~~ — **done**, and it changed everything: INT4 matmul is ~2x
   *faster* than INT8 (3.6-3.9 TOPS) once the A/C matrices use native
   layout. The backend's `AC_layout = RKNN_MM_LAYOUT_NORM` costs ~80x.
2. **Idea 5 (new, now the main thread): native A/C layout for INT4.**
   Produce activations directly in the NPU's native A layout and unpack C
   after the run, at least for the W4A4 pipelines. This is the single
   change with a measured ~80x matmul-level headroom behind it. Scope:
   derive the native A tiling from `io_attr.A.dims`, mirror the existing
   `pack_native()` approach used for B, add the C-side unpack. Validate
   with `llama-bench` and `llama-perplexity`.
3. **Idea 1 / vectorisation** — once matmuls are fast, the scalar CPU-side
   activation prep (INT4 packer, FWHT) becomes the bottleneck: NEON first,
   GPU only if that is not enough.
4. **Idea 3** — hybrid per-layer patterns; free to explore via
   `RKNPU_HYBRID`, and far more interesting if W4A4 becomes fast.
5. **Idea 2** — offline transform elimination; unchanged, still the
   principled way to cut per-token work.
6. **Idea 4** — M-dependent pipeline selection; unchanged, small prize.

## Reality check

The picture after the layout finding:

- **Prefill**: INT4 has real headroom on this chip — ~2x INT8 at the matmul
  level — but the backend cannot reach it without the native-layout work.
  Whether that translates end-to-end depends on how much of prefill time is
  NPU matmul versus CPU-side graph work (for Gemma-4-E4B, with ~600 backend
  splits per token, a meaningful share is the latter).
- **Decode**: unchanged and layout-independent. Purely bandwidth-bound at
  ~10 GB/s per core; INT4 is 1.9x faster than INT8 simply by reading half
  the bytes, but the shipped `RKNPU_CPU_DECODE` routing (41.3 / 4.9 on E4B)
  still wins by letting the CPU read Q4_0 natively with no dispatch cost.
- **Accuracy**: the reason to be cautious about W4A4 regardless. 4-bit
  activations cost real quality (perplexity 86 vs 22.5 on Granite-350M),
  so even a fast W4A4 is a trade, and hybrid patterns (Idea 3) are the
  natural way to spend it selectively.

## References

- marty1885, "Benchmarking RK3588 NPU matrix multiplication performance"
  (EP1/EP3) — independent measurements reporting ~900 GFLOPS FP16,
  ~1.8 TOPS INT8, ~4 TOPS INT4, and the A-layout/GEMV observations that
  prompted the correction above:
  https://clehaxze.tw/gemlog/2024/02-14-benchmarking-rk3588-npu-matrix-multiplcation-performance-ep2.gmi
- marty1885/rk3588-matmul-bench — the reference benchmark tool (sweeps
  ac_native/b_native across shapes): https://github.com/marty1885/rk3588-matmul-bench
