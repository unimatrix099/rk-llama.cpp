---
name: bonsai2-tgspeed-optimization
description: "Work to speed up Bonsai 2 27B token generation on RK3588 - profiling, NEON PTQ1_0 kernel, TDD"
metadata: 
  node_type: memory
  type: project
  originSessionId: 99d50116-a329-4a80-bf79-9e91065d77ff
  modified: 2026-09-25T13:50:28.572Z
---

Goal (2026-09-25): improve Bonsai 2 27B decode tok/s on the Orange Pi 5 Ultra. Builds on [[bonsai2-eval-plan]] (measured 0.38 tg PTQ1_0 / 0.32 PQ2_0). See [[rk3588-board-access]].

**PROFILE (measure-first, go/no-go) - perf flat profile of PTQ1_0 tg128 on board:**
- `ggml_vec_dot_ptq1_0_q8_0` = **86.15%** self-time (the hot path).
- ~11% threadpool spin (0x21d5x offsets). `ggml_compute_forward_fwht` = **0.10% (Hadamard NOT the bottleneck - my earlier hypothesis was WRONG; profiling saved the effort).**
- Key: PTQ1_0 vec_dot is SCALAR-only on ARM (arch-fallback.h aliased it to generic; no NEON). PQ2_0 has native ARM but PTQ1_0 did not.

**PTQ1_0 format:** block = qs[24] (5 trits/byte, base-3 packed) + qh[2] (4 trits/byte) + d(fp16) = 128 trits in 28 B (1.75bpw), dotted vs 4x block_q8_0. Unpack per trit: `v=byte*pow3[nn] (u8 wrap); xi=(u16(v)*3)>>8 in {0,1,2}; q=xi-1`.

**TDD NEON kernel (Option C) - DONE, bit-exact:**
- Standalone test `scratchpad/ptq1_test.c` (container is aarch64+dotprod, runs it): random blocks, compare vs verbatim generic ref, 200/200 BIT-EXACT. Microbench: dot-only vectorization = 1.03x (dot is NOT the cost); **unpack+dot vectorized = 2.05x**.
- Unpack vectorized with vmulq_u8/vmovl/vshrq; dot via `ggml_vdotq_s32` (portable, SDOT when available). Accumulation order kept identical to ref -> bit-exact.
- Integrated into fork: added native `ggml_vec_dot_ptq1_0_q8_0` in `ggml/src/ggml-cpu/arch/arm/quants.c` (guarded `#if __ARM_NEON`, else calls _generic), removed the ptq1_0 alias in `arch-fallback.h` aarch64 block. Patch saved: `scratchpad/ptq1_neon.patch`.
- Projected end-to-end: 86% hot * 2.05x -> ~1.8x decode, ~0.38 -> ~0.68 tg (TO BE CONFIRMED on board).

**Build/test method:** source tar'd to board `~/bonsai2/src/PrismML-llama.cpp` (exclude ONLY .git; `build*` exclude wrongly drops common/build-info.cpp.in). cmake `-DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS/EXAMPLES/SERVER=OFF`, build targets `llama-bench llama-cli`. A/B: build native BASELINE (patch reversed) then `patch -p1` re-apply + incremental rebuild = OPTIMIZED. baseline-native vs prebuilt b10735 = Option B (native flags); baseline-native vs optimized = Option C (my kernel). Detach builds (nohup, log, poll) - full build ~15-20min on 8 cores.

**Options - ALL TESTED (2026-09-25):**
- A (runtime tuning): t=8 all cores 0.67 (best), t=6 0.56, t=4 A76-pinned 0.57. Default already optimal. NO WIN.
- B (native -mcpu flags): 0.38 == prebuilt. NO WIN.
- C (NEON PTQ1_0 kernel): 0.38 -> 0.67 t/s = **1.76x**. THE WIN.
Net result: PTQ1_0 decode 0.38 -> **0.86 t/s (2.26x)** from the kernel alone (best). Governor reset to schedutil.

**AUTORESEARCH LOOP (2026-09-25, autoresearch plugin, 6-iter bounded):** iter1 KEEP = vectorized the qh 8-trit scalar tail (threshold), board tg 0.80 -> **0.86** (+7.5%), bit-exact, committed 21106d58e. iters 2-4 all WASH on the A76 STANDALONE microbench pre-filter (fast, faithful A76 predictor) so not board-verified: dot fp-reduction (vpaddq combine + 1 f32 reduce), skip pow3[0]=1 multiply, 2-block unroll (2.10 vs 2.11x - OoO already covers single-block ILP). Plateau -> early stop. Kernel at practical throughput floor: unpack ~5 NEON ops/16-trit near-minimal. Loop artifacts in fork `autoresearch/loop-*/` (results.tsv, handoff.json), doc table updated, pushed (02e559a3f); demo note -> 0.86/2.26x (8c14d20). KEY method: use the board STANDALONE harness microbench (compiles in secs on A76) as a fast pre-filter before the ~10min full llama.cpp board build+bench.

**AUTORESEARCH LOOP 2 (2026-09-25, build/ops/config avenues) - NO further gain, 0.86 is the ceiling:** (1) GGML_LTO=ON: 0.86, no gain (O3 hot loop already optimal; reverted). (2) config KV q8_0 / flash-attn: 0.85-0.86, no gain (don't affect short-ctx tg). (3) op-reduction: re-profile of 0.86 build = vec_dot **81%**, **~13% threadpool-barrier spin** (0x21d5x) is the ONLY remaining headroom, FWHT 0.19%, rest <1%. Capturing the barrier needs invasive ggml op-fusion (fold the sign-flip `ggml_mul(cur_mm,t.signs)` into the FWHT across ~300 sites/token) - high whole-model correctness risk, ~4-5% realistic - NOT pursued. Artifacts in fork `autoresearch/loop2-*/` (3e106ca7f). **FINAL: PTQ1_0 tg = 0.86 t/s (2.26x over scalar) is the practical CPU ceiling for this kernel/format on RK3588.**

**THRESHOLD UNPACK WIN (2026-09-25) - BEST/PRODUCTION KERNEL, tg 0.80 t/s (2.1x):** investigated WITHIN the vec_dot (long pole = the UNPACK, not the dot; confirmed by dot-only 1.03x vs unpack+dot 2.05x). Key insight: base-3 decode `(v*3)>>8` for byte v in [0,255] is a 2-threshold STEP fn (0 if v<86, 1 if v<171, else 2), so the widen(vmovl)/mul(vmulq_n_u16)/shift(vshrq)/narrow(vmovn) chain (~9 ops/16 trits, in u16) collapses to two `vcgeq_u8` compares staying in u8: `q = -1 - (v>=86) - (v>=171)`. Bit-exact 200/200. Microbench 2.28->2.79x (container), 1.26x over fused on A76. **Board tg 0.70 -> 0.80 t/s (+14%)**, coherent output. Committed to production branch `opt/ptq1_0-arm-neon` (37b0bab58); demo note -> 2.1x (1505ea0). Also ~throughput-bound estimate: ~128 NEON ops/128-trit block; threshold cuts the unpack portion ~2x.

**PQ2_0 vs PTQ1_0 on CPU (2026-09-25):** measured PQ2_0 tg = **0.32 t/s** (6.70 GiB) vs threshold PTQ1_0 **0.80** (5.53 GiB). PTQ1_0 is 2.5x FASTER and smaller. PQ2_0 does NOT use the repack path on neon+dotprod (get_optimal returns nullptr - repack only for x86 avx512+vnni), so it runs its NATIVE per-row dotprod vec_dot (arch/arm/quants.c:297) and still loses - its vqtbl 2-bit unpack + per-k fp (vcvtq+vmlaq_n) is less efficient than the threshold ternary kernel, plus 21% more bytes. **VERDICT: PTQ1_0 is the clear CPU pick (faster + smaller); PQ2_0's only edge is precision (2.16 vs 1.75 bpw). PQ2_0's native kernel likely has similar optimization headroom (not pursued).**

**ROUND 2 research (2026-09-25, "still far from theoretical ~3.8 t/s BW roofline"):**
- Re-profile of OPTIMIZED build: `ggml_vec_dot_ptq1_0_q8_0` STILL **86.4%** (unchanged share), ~9% threadpool barrier spin (0x21d5x), everything else <1%. Effective BW ~4 GB/s of ~22 -> NOT bandwidth-bound; compute+latency-bound on the ternary unpack. The 1.76x-decode vs 2.05x-microbench gap = real decode streams weights cold from DRAM (microbench was cache-resident).
- **Speculative decoding: NOT viable.** Bonsai 2 has NO official DSpark drafter (`BONSAI_SPECULATIVE=1` warns + runs without it for bonsai2 family; only prev-gen ternary/bonsai have drafters). Gains are CUDA/Metal only; on compute-bound CPU verifying K tokens costs ~Kx compute (draft competes for same cores) -> poor fit. Off the table.
- ggml mul_mat already uses DYNAMIC chunk work-stealing (atomic current_chunk, 16-row chunks) so big.LITTLE imbalance is auto-handled; the ~9% spin is per-op barrier overhead across ~400 ops/token, not row imbalance -> weak lever.
- `perf stat` optimized decode: IPC=1.77, **backend stalls 29.5% cycles idle**, frontend ~0, cache-miss 0.42%. => ~30% headroom from memory-latency + dep-chain stalls, NOT compute-saturated. Lever = software prefetch of streamed weights + block-interleave ILP. Est ~1.2-1.3x -> ~0.8 t/s.
- **Honest ceiling:** the 3.8 t/s BW roofline assumes FREE compute; ternary unpack is not free, so 3.8 is unreachable on this CPU/format. Pre-unpacking weights to int8 (no per-token unpack) would need ~24GB RAM (>16GB) - doesn't fit. Biggest untried pure-kernel lever = PTQ1_0 4-row repack GEMV (complex, uncertain payoff given ~9% barrier floor).

**ROUND 2 IMPLEMENTED (2026-09-25):**
- Prefetch (`__builtin_prefetch`): NO board gain (HW prefetch already covers sequential weight stream; not memory-latency bound). Kept (harmless).
- **Fused unpack->dot (no int8 q[128] temp)**: microbench 2.05x->2.28x; board tg 0.67->**0.70 t/s**. Removes store-to-load latency (part of the 30% backend stall). Bit-exact 200/200 kept.
- **Cumulative: 0.38 -> 0.70 t/s = 1.84x.** Near practical kernel ceiling (unpack arithmetic near-minimal; further micro-opt translates weakly - board realized ~81% of microbench speedup, Amdahl-bounded by the 86% vec_dot share + ~9% barrier spin).
- Committed+pushed: `opt/ptq1_0-arm-neon` commit 8148feba7 (fused kernel + updated docs/test); demo `benchmark/rk3588-cpu-bonsai2` commit c46df61 (note updated to 1.84x). Governor reset to schedutil.

**KERNEL-IMPROVEMENT INVESTIGATION (2026-09-25):**
- nrows=2 multi-row path is i8mm-only (RK3588 has NO i8mm) AND decode-disabled (forced to 1 row when ne11/tokens is odd; decode=1 token). Useless for decode. Only helps prefill w/ i8mm.
- **Decode multi-row ILP = REPACK GEMV path** (`ggml_gemv_*`, N rows/call at nr1=1). Fork ALREADY has a full `block_q1_0x4` repack template: make_block_q1_0x4, repack_q1_0_to_q1_0_4_bl, ggml_gemv/gemm_q1_0_4x4+4x8 (generic in repack.cpp:1462/1525/2672/2757), registration at repack.cpp:4604/4608. => a PTQ1_0 4x8 repack GEMV is very tractable (mirror Q1_0). Processes 4 rows -> 4 SDOT accum streams -> fills the 29.5% backend idle. Est 1.2-1.5x -> ~0.9-1.0 t/s. Effort moderate-high; write NEON ptq1_0 gemv (q4_0_4x8 arm is the SIMD template). RECOMMENDED next lever.
- Quick win: drop the `-1` sub via q=xi-1 -> dot = sum(xi*y) - sum(y); sum(y) per q8 block computed ONCE per activation (shared across all weight rows) -> removes ~10 vsubq/block from unpack, xi{0,1,2} feeds vdot directly. Low effort, ~5-10%. TDD-able.
- Minor: vectorize qh 8-trit scalar tail (~2-3%).

**REPACK GEMV PROTOTYPED + VALIDATED ON A76 (2026-09-25):** core 4-row gemv written in scratchpad (`ptq1_gemv.c`/`gemv_bench.c`), bit-exact 200/200. IMPORTANT bench pitfall: comparing vs a single-row loop with identical inputs gets the baseline CSE-hoisted -> false 0.88x. With anti-hoist (perturb 1 input byte/iter) and PRODUCTION fused kernel as baseline, on A76 (taskset -c 6, -mcpu=native): **gemv4 = 1.24x, gemv2 = 1.14x** over production single-row. Win = sum(y) amortized across 4 rows + row-level ILP. 4-row beats 2-row despite register spills. Projected end-to-end ~1.2x -> ~0.70 -> ~0.83 t/s. Bug found+fixed in prototype: qh scalar unpack must use `uint8_t v = qh*pow3` (wrap mod 256), NOT `(uint16_t)(...)`. **FULL REPACK INTEGRATION DONE + MEASURED = REGRESSION (2026-09-25).** Implemented complete ggml repack path for PTQ1_0: block_ptq1_0x4 (concatenated 4-row), repack_ptq1_0_to_ptq1_0_4_bl, NEON ggml_gemv_ptq1_0_4x8_q8_0 (sum(y) amortized, hoisted once/activation), scalar ggml_gemm (prefill), template specializations, registration, arch-fallback aliases. Bit-exact + coherent output (EXIT=0, "capital of France"). BUT board tg = **0.61-0.62 t/s, SLOWER than fused 0.70**. The single-core microbench 1.24x did NOT survive 8-thread decode: repack path's own chunking loses to ggml dynamic work-stealing on small GEMV across big.LITTLE; hoisting sum(y) didn't help (0.62->0.61). Also repack DOUBLES resident weight mem at load -> OOMs at default 262K ctx on 16GB (needs small -c). CONCLUSION: repack GEMV not beneficial on this CPU. Committed to side branch `opt/ptq1_0-repack-gemv` (commit 3b7d8cc9b, NOT merged) + doc `docs/development/rk3588-ptq1_0-repack-gemv.md`. **PRODUCTION stays on fused kernel, branch `opt/ptq1_0-arm-neon`, 0.70 t/s / 1.84x.** Lesson: single-core cache-resident microbench is not a reliable predictor of threaded cold-weight decode - always measure end-to-end on the target.
- Hard walls: ternary unpack irreducible; ~9% per-op barrier floor (400 ops/token, needs graph fusion); int8 pre-unpack = 24GB > 16GB. 3.8 t/s roofline unreachable (assumes free compute).

**BOARD A/B RESULTS (2026-09-25, tg-only bench `-p 0 -n 64 -r 3`, gov=performance):**
- Baseline native build (scalar ptq1): tg64 = **0.38 t/s** (== generic prebuilt -> Option B native flags give nothing).
- Optimized (my NEON kernel): tg64 = **0.67 t/s** = **1.76x** end-to-end speedup. Matches ~1.8x projection (86% hot-path * 2.05x kernel).
- Correctness confirmed end-to-end: `llama-completion` (the buildable CLI - NOTE: `llama-cli` target doesn't exist in this fork; CLI is `llama-app` which needs cli-impl+server-impl libs; `llama-completion` builds standalone) produced coherent "capital of France" reasoning output on the optimized build.
- Bench pitfall: prompt processing (pp) is ALSO ~0.49 t/s on this model, so use `-p 0` to measure tg only, else a bench takes ~50 min. `llama-bench` only prints at the very end (no partial output).
- Build pitfall: `cmake --build build --target llama-bench llama-cli` only builds the FIRST target; build llama-cli separately.

**COMMITTED + PUSHED + DOCUMENTED (2026-09-25, user approved):**
- `unimatrix099/PrismML-llama.cpp` branch `opt/ptq1_0-arm-neon` (commit 59c2648d4): kernel in arch/arm/quants.c + arch-fallback.h, plus `docs/development/rk3588-ptq1_0-neon.md` (writeup) and `rk3588-ptq1_0-neon-test.c` (standalone TDD harness).
- `unimatrix099/Bonsai-demo` branch `benchmark/rk3588-cpu-bonsai2` (commit 8a091bf): added custom-kernel note to the CPU benchmark file (clearly marked as not the stock build).
- Both committed as unimatrix099 with `Co-Authored-By: Claude Opus 4.8` trailer. No PRs opened upstream (not requested).
- Board build at `~/bonsai2/src/PrismML-llama.cpp/build` has the OPTIMIZED kernel; container fork working tree on branch `opt/ptq1_0-arm-neon`.
