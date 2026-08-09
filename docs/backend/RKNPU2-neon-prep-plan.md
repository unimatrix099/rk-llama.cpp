# NEON vectorization of the W4A4 CPU-side prep: research + plan

Branch: `feat/w4a4-neon-prep`, on top of the native-layout work
(`RKNPU2-native-layout-plan.md`). Status: **research done, measured;
implementation not started.**

## Why this is the next lever

After the native A/C layout change, E4B Q4_0 prefill is 31.9 t/s vs 41.1
for W8A8 — a gap of ~7 ms per token that is no longer the NPU's fault. The
per-activation-row prep around every W4A4 matmul runs on the CPU and is
**entirely scalar** today, while the FP16 conversion and INT16/INT32
dequant paths in the same files are NEON-vectorized:

| Step | Where | Today |
|---|---|---|
| sign multiply (`row * s_vec`) | inline, `ggml-rknpu2.cpp` A-pass | scalar |
| FWHT (`fwht_iterative`) | `rknpu2-calibration.cpp` (static) | scalar |
| amax scan | inline, A-pass | scalar |
| INT4 quantize+pack | `rknpu2-quantization.cpp` | scalar |
| INT8 quantize (W8A8 A-pass) | `rknpu2-quantization.cpp` | scalar (bonus target) |
| C INT16 dequant-accumulate | `ggml-rknpu2.cpp` C-pass | scalar |

## Measured: scalar vs NEON prototypes (`rknpu2-bench-prep.c`)

RK3588 big core (taskset 4-7), pinned clocks, element-exact vs scalar
(including rounding ties — see below). Per call, microseconds:

| Step | K=2048 | K=4096 | K=16384 | speedup |
|---|---|---|---|---|
| fwht | 14.3 → 4.7 | 30.6 → 10.0 | 139.1 → 45.3 | **3.0–3.1x** |
| signmul | 1.6 → 0.4 | 3.2 → 0.8 | 12.9 → 3.4 | 3.8–4.0x |
| amax | 1.8 → 0.5 | 3.7 → 0.9 | 14.6 → 3.7 | 4.0x |
| q4pack | 3.5 → 1.1 | 7.0 → 2.1 | 28.0 → 8.5 | 3.3x |
| c16dequant | 2.3 → 0.7 | 4.6 → 1.5 | 18.3 → 6.0 | 3.1x |

Per-row prep total at K_op=4096 (E4B-representative): **44.5 µs → 13.8 µs
(3.2x)**. The FWHT dominates the prep at every size and is memory-bound in
its later stages (log2(K) full passes over the row), which is why it caps
near 3x; stage-fusion (two butterfly stages per pass) could push it
further and is noted as a follow-up, not required.

## Correctness notes discovered during prototyping

1. **Rounding ties**: the scalar packer uses `roundf` = round-half-away-
   from-zero. `vcvtnq_s32_f32` rounds half-to-even and silently differs on
   exact ties (random-data tests pass by luck). **`vcvtaq_s32_f32` is the
   exact match** — the harness includes a tie-vector test
   (`..., -0.5, 0.5, 1.5, ...`) that fails with vcvtn and passes with
   vcvta.
2. FWHT stage h=1 needs a lane shuffle (`vrev64q` + `vtrn1q(sum, dif)`);
   getting the trn operands wrong produces sign-flipped odd lanes — the
   harness's exactness check catches it instantly. Stages h>=4 vectorize
   trivially; h=2 uses 2-lane halves.
3. Everything is element-exact vs scalar, so backend integration must be
   validated the same way as the layout work: identical perplexity to the
   scalar build (no ASLR caveat needed here — same values in, same values
   out, per function).

## Implementation plan (next step, TDD like the layout work)

1. Add NEON paths behind `#ifdef __ARM_NEON` next to the existing scalar
   code (same pattern the file already uses for FP16/dequant):
   - `rknpu2-quantization.cpp`: `quantize_fp32_to_int4_packed`,
     `quantize_fp32_to_int8` (bonus for W8A8).
   - `rknpu2-calibration.cpp`: `fwht_iterative`.
   - `ggml-rknpu2.cpp`: sign-multiply + amax fuse into one pass over the
     row (they read the same data — one load serves both); C-pass INT16
     dequant (both NORM and native-tiled variants; the native one
     processes 8-wide cells, a perfect NEON fit).
2. Unit tests first: extend the docs/backend test suite with exactness
   tests per function (including the tie vector) against the scalar
   reference, plus odd-size tails (K not multiple of 16).
3. Validate end-to-end: perplexity identical to the scalar build;
   llama-bench E4B + Qwen W4A4 before/after.
4. No behavioural flags needed — this is a pure same-results optimization.

## Honest projection

Prep is ~44.5 µs/row scalar at E4B shapes across ~7 W4A4 matmuls per layer,
executed under OpenMP across cores; NEON cuts it ~3.2x. That should
recover **a substantial fraction of the 7 ms/token gap to W8A8** — a
reasonable expectation is E4B W4A4 prefill in the **~36–39 t/s** range,
with Qwen-class models (smaller K, prep-heavier mix) gaining
proportionally more. The C-dequant and INT8-quantize bonuses also shave a
little off every pipeline, including W8A8 itself. To be confirmed by
benchmark, not asserted.
