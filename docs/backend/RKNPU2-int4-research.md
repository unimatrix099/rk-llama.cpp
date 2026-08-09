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

**Verdict: not worth building _until_ Idea 0 shows the INT4 ceiling can be
lifted.** If the ceiling does lift, this becomes valuable again.

## Idea 0 — RESOLVED by measurement: INT4 compute is not accelerated

**Run it yourself:** `rknpu2-matmul-bench.c` next to this file. Raw
`rknn_matmul_run` only, single NPU core (`RKNN_NPU_CORE_0`), K=N=2048,
clocks pinned via `scripts/fix_freq_rk3588.sh` from the rknn-llm repo.

```
M     FP16xFP16->FP32      INT8xINT8->INT32     INT4xINT4->INT16
1       0.764 ms   11 GOPS   0.390 ms   22 GOPS   0.203 ms   41 GOPS
32      0.797 ms  337 GOPS   0.400 ms  670 GOPS   6.053 ms   44 GOPS
128     1.688 ms  636 GOPS   0.638 ms 1683 GOPS  24.172 ms   44 GOPS
512     6.740 ms  637 GOPS   2.494 ms 1722 GOPS  96.632 ms   44 GOPS
```

Two clean facts fall out.

**1. INT4 has no parallel compute path.** Its throughput is pinned at
~44 GOPS at every batch size, and its cost is a flat ~189 us per row of A
(203/189/189/189 us at M=1/32/128/512) — the signature of serial per-row
processing, not a MAC array. For scale: INT8 reaches 1722 GOPS, which is
86% of the RK3588's theoretical 2 TOPS per core, so the harness is sound
and the chip is delivering its rated INT8 performance. **INT4 runs at 2.6%
of the INT8 peak** and is 38x slower at M=512. Whether this is emulation in
the runtime or an unaccelerated hardware mode, it is not something a backend
can fix.

**2. At M=1 everything is purely memory-bandwidth bound, and INT4 wins
exactly as predicted.** Dividing B-matrix bytes by time gives 10.98 / 10.75 /
10.33 GB/s per core for FP16 / INT8 / INT4 — the same bandwidth for all
three, so the only thing that matters at decode is how many bytes the weights
occupy. INT4 reads half of INT8 and takes half the time (0.203 vs 0.390 ms,
1.92x). This is the bandwidth model of this whole investigation confirmed
directly at the hardware level.

### Consequences

- **INT4 can never help prefill on RK3588.** The 44 GOPS ceiling is a hard
  wall. This also explains the end-to-end W4A4 prefill deficit measured in
  the fork (7.7 vs 39.8 t/s) — and the fork actually does *better* than the
  raw 38x ratio, because it spreads work over 3 cores and much of prefill
  time is spent outside the NPU.
- **INT4's only real advantage is decode latency**, and it is genuine at the
  matmul level (1.9x). But end-to-end the fork measures W4A4 tg 3.4 vs W8A8
  3.3 — the win is largely eaten by per-token CPU work (Hadamard transform,
  scalar INT4 activation packing, INT16 output dequant) and by the fact that
  decode time on a model like E4B is dominated by non-matmul work anyway
  (~600 backend splits per token).
- **Even fully captured, it would lose to the shipped routing.** CPU decode
  already reads Q4_0 weights natively at 4 bits with no dispatch overhead and
  delivers 4.9 t/s (`RKNPU_CPU_DECODE=32`), above any plausible INT4-on-NPU
  decode projection.

### Verdicts this settles

- **Idea 1 (GPU activation prep): dead.** It targets prefill, where INT4 is
  capped at 2.6% of INT8 regardless of how fast activations are prepared.
- **Idea 2 (offline transform elimination): downgraded.** Still the right
  way to make INT4 cheaper, but the prize is now known to be small — it
  cannot lift the 44 GOPS ceiling.
- **A new, narrower idea replaces them** (see Idea 4 below).

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

## Priority (revised after Idea 0)

1. ~~**Idea 0**~~ — **done**; INT4 compute is capped at 2.6% of INT8.
2. **Idea 3** — hybrid per-layer patterns; still free to explore
   (`RKNPU_HYBRID`, no code changes) and now clearly framed: any layer put
   on W4A4 pays a 38x prefill penalty for that layer, so useful only for
   fitting larger models, not for speed.
3. **Idea 4** — M-dependent pipeline selection; small expected prize,
   documented mainly for RK3576+ where it would pay off properly.
4. ~~**Idea 2**~~ — downgraded; cannot lift the ceiling.
5. ~~**Idea 1**~~ — dead; targets a phase where INT4 cannot compete.

## Reality check

INT4 on RK3588 is a *bandwidth* play, not a compute play — and now that is
measured rather than inferred: 44 GOPS flat versus INT8's 1722 GOPS, with
equal ~10 GB/s memory bandwidth for both at M=1. The practical consequence
is that W4A4's honest role on this chip is **fitting larger models under the
4 GB per-IOMMU-domain limit**, at a large speed cost, and the shipped
`RKNPU_CPU_DECODE` routing (41.3 / 4.9 on E4B) already captures the decode
bandwidth win in the only place this hardware makes it available: the CPU,
which reads 4-bit weights natively and unpacks them in registers.

The path to real 4-bit LLM acceleration on Rockchip runs through silicon
with a w4a16 datapath (RK3576 and later), not through this backend.
