---
name: bonsai2-pq2-optimization
description: "PQ2_0 kernel optimization on RK3588 - NEON q8_K kernel gave 4.2x, now fastest Bonsai 2 band"
metadata: 
  node_type: memory
  type: project
  originSessionId: 99d50116-a329-4a80-bf79-9e91065d77ff
  modified: 2026-09-25T18:19:35.752Z
---

Autoresearch loop 3 (2026-09-25): optimized Bonsai 2 **PQ2_0** tg on the Orange Pi 5 Ultra. See [[bonsai2-tgspeed-optimization]] (PTQ1_0 work) and [[rk3588-board-access]].

**RESULT: PQ2_0 tg 0.32 -> 1.35 t/s (4.2x), bit-exact, coherent output. PQ2_0 is now the FASTEST Bonsai 2 band on RK3588** (ahead of PTQ1_0 0.86) AND higher precision (2.16 vs 1.75 bpw). This FLIPS the earlier "PTQ1_0 is the CPU pick" conclusion in [[bonsai2-eval-plan]].

**Root cause (found by profiling, not assumption):** PQ2_0 weight matrices with ne[0] % QK_K(256) == 0 dot against **Q8_K** activations (not Q8_0). `ggml_vec_dot_pq2_0_q8_K` was SCALAR-only on ARM (arch-fallback aliased it to generic; "only generic outside x86") = 88% of PQ2_0 decode. The pq2_0 x q8_0 kernel (which HAS native NEON) is NOT the hot path for the model - it's only used for tensors whose ne[0] is not a multiple of 256. My initial harness tested the wrong (q8_0) variant.

**Fix:** NEON `ggml_vec_dot_pq2_0_q8_K` in `ggml/src/ggml-cpu/arch/arm/quants.c` + removed the aarch64 arch-fallback alias. Same 2-bit vqtbl codec as pq2_0 x q8_0, BUT Q8_K has ONE float scale per 256, so all 4 sub-blocks accumulate into a single int32 acc, reduced+scaled once per 128-block (simpler than q8_0's per-32 d1). Microbench: 5.36x over scalar on A76. Committed 0e07eef3d on `opt/ptq1_0-arm-neon`; standalone test `docs/development/rk3588-pq2_0-q8k-test.c`.

**Re-profile @1.35:** q8_K kernel 73%, ~19% barrier spin. 2-bit vqtbl codec near-minimal; remaining headroom = the same per-op barrier floor as PTQ1_0. Stopped (diminishing returns).

**KEY LESSON:** profile to find the ACTUAL hot function before optimizing - PQ2_0 used a different (q8_K, scalar) kernel than the (q8_0, NEON) one visible in the code. Loop artifacts: fork `autoresearch/loop3-pq2-*/` (43ce64f4c). Demo note updated (79c6c4a).

**Updated recommendation for Bonsai 2 on RK3588:** PQ2_0 (1.35 t/s, 2.16bpw) > PTQ1_0 (0.86, 1.75bpw). PQ2_0 is both faster and higher quality now.

**LOOP 4 (2026-09-25, more PQ2_0 + NPU/Vulkan/GPU): no further gain, 1.35 stands.**
- OFFLOAD verified non-viable on the (reflashed) board which DOES now have GPU stack: Mali-G610 present; **Vulkan via Mesa Panfrost only exposes `llvmpipe`** (CPU software rasterizer, deviceType=CPU - not the real GPU); **OpenCL via `libmali-g13p0.so` exposes the REAL Mali-G610** (has `cl_arm_integer_dot_product_int8`), and the fork HAS a ggml-opencl backend, BUT it has NO PQ2_0/Hadamard kernels (Adreno-focused) - would need writing them; decode is memory-bound on shared LPDDR5 (~22GB/s) so GPU payoff capped ~2.4x. **NPU** (fdab0000.npu @1GHz) has no backend in the Bonsai fork + INT8-only. All three: not tractable.
- Kernel micro-opt: bsums-offset (use q8_K free `bsums` to drop the per-weight -1) = bit-exact but A76 WASH (0.129 vs 0.131s; vsub hidden by vdot chain). PQ2_0 q8_K kernel at practical optimum (73% decode, ~19% barrier).
- Loop artifacts: fork `autoresearch/loop4-pq2gpu-*/` (22a809d04).

**OpenCL/Mali EXPERIMENT (2026-09-25, branch `opt/pq2-opencl-mali`): BLOCKED, do not retry.** Pushed the Adreno/Intel-focused `ggml-opencl` backend to actually run on Mali-G610. Cleared 5 gates (device accept + `GPU_FAMILY::MALI`; `-DGGML_OPENCL_USE_ADRENO_KERNELS=OFF`; kernels self-detect INTEL_GPU/ADRENO_GPU via driver exts Mali lacks → host-inject `-DINTEL_GPU=1` + empty-def `REQD_SUBGROUP_SIZE_*`; ~5 `sgs` switches → Mali=16; ~39 `if INTEL/else if ADRENO/else ASSERT` matmul-dispatch sites → INTEL branch accepts MALI). Got matmul/norm kernels dispatching on GPU. **Gate 6 = runtime `clEnqueueNDRangeKernel` failure (ggml-opencl.cpp:1049)**: Intel work-group/local-size invalid for Mali → per-kernel work-group tuning = open-ended, NOT a bounded patch. Backend is bifurcated Intel-vs-Adreno across ~60 sites. Driver itself is fine (Mali-G610 has `cl_khr_subgroups`+shuffle/ballot + `cl_arm_integer_dot_product_accumulate_int8`; natural subgroup=16). NOT worth finishing: (a) only lights up standard quants — Bonsai PQ2_0/PTQ1_0 have NO OpenCL kernels + need Hadamard kernel from scratch; (b) decode memory-bound on shared LPDDR5, Mali has no bandwidth edge over 8 CPU cores → best case parity, likely regression. **CPU NEON (PQ2_0 1.35 t/s) stays the answer.** Writeup: `docs/development/opencl-mali-experiment.md`. Committed WIP f239116d9.

**FROM-SCRATCH GPU feasibility (2026-09-25, measure-first standalone OpenCL probes `scratchpad/mali_bw*.c`,`mali_gemv*.c`): DEFINITIVE NO-GO.** Measured on real Mali-G610: peak coalesced read BW **19.7 GB/s** (tuned uint16); int8 GEMV (no unpack, memory-bound @115% read-peak) **22.6 Gw/s = 0.84 t/s**; vectorized 2-bit GEMV **9.8 Gw/s = 0.37 t/s**; scalar 2-bit 5.9 Gw/s. **CPU NEON PQ2_0 = 36.3 Gw/s = 1.35 t/s.** Key: the GPU's ABSOLUTE ceiling (trivial int8, memory-bound) 22.6 Gw/s is ALREADY below CPU 36.3 — any real low-bit kernel adds unpack ALU and is slower. Mali local-mem type=Global (no SRAM scratchpad, tiling useless); 4 CUs@1GHz share LPDDR5 w/ CPU (no VRAM, no parallel BW). 2-bit unpack is ALU-bound (2.5 of 19.7 GB/s used). Projected 0.4-0.8 t/s = 2-3x REGRESSION vs CPU for huge effort. **The 4xA76 SDOT/i8mm CPU is genuinely the faster engine for low-bit GEMV decode on RK3588.** GPU only viable for compute-bound prefill/GEMM, not 1-token decode. DO NOT re-investigate GPU decode.

**NPU + HETEROGENEOUS study (2026-09-25, probes `mali_hog.c`,`mali_gemvloop.c`):** NPU = NO-GO: RKNPU kernel driver loaded but NO userspace runtime (librknnrt absent), no rknn-toolkit, no ggml rknpu backend; RKNN op set doesn't support autoregressive LLM decode (KV/Hadamard/dynamic shapes); INT8-only so 27B=27GB won't fit 16GB. HETEROGENEOUS CPU+GPU = the ONLY path with positive headroom: measured aggregate LPDDR5 ceiling ~25 GB/s (CPU decode uses only ~9.8, it's compute/barrier-bound). CPU PQ2 decode 1.33 t/s alone; drops to 0.82 (-38%) under a 19 GB/s GPU hog (contention) BUT only to 1.30 (-2%) under a *useful* light GPU 2-bit GEMV (2.5 GB/s, 9.9 Gw/s). So a bandwidth-throttled GPU co-processor coexists nearly free → pipelined ceiling ~1.68 t/s (1.26x), realistic ~1.1-1.2x after tensor-parallel per-layer sync. NOT worth building: needs working Mali kernels (broken port) + TP-decode infra (absent) for ~15% gain vs CPU NEON's already-delivered 2.3-4.2x. Full writeup appended to `docs/development/opencl-mali-experiment.md`.
