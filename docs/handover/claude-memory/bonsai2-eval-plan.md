---
name: bonsai2-eval-plan
description: "Plan to evaluate Bonsai 2 27B on the RK3588 board and benchmark it against LFM2.5, ERNIE, and Bonsai v1"
metadata: 
  node_type: memory
  type: project
  originSessionId: 99d50116-a329-4a80-bf79-9e91065d77ff
  modified: 2026-09-25T06:20:02.651Z
---

Goal (set 2026-09-24): get PrismML **Bonsai 2 27B** running on the Orange Pi 5 Ultra and compare it head-to-head with the other models kept on the board. See [[bonsai-27b-on-board]] for v1 setup and [[rk3588-board-access]] for board access.

**Bonsai 2 27B quants** (docs.prismml.com/bonsai-2-27b): GGUF `PTQ1_0` (1.76 bpw, 5.93 GB) and `PQ2_0` (2.16 bpw, 7.25 GB); MLX is Apple-only. New arch = hybrid ~75% linear / ~25% full attention, 262k ctx, multimodal, reasoning-on-by-default. Both GGUFs need the **PrismML llama.cpp fork** (stock refuses them).

**ARCHITECTURE SUPPORT — investigated 2026-09-24 (source audit of the forked `prism` branch, HEAD = b10735-842b188):**
- Bonsai 2 27B declares GGUF arch **`qwen35` (DENSE, no experts)** — NOT qwen3next/qwen35moe, despite the marketing "hybrid ~75% linear/25% full attention" wording. Handled by `src/models/qwen35.cpp` (`LLM_ARCH_QWEN35` = "qwen35"). Mainline llama.cpp ALSO knows `qwen35`+`Q2_0` but omits Bonsai 2's Hadamard/sign-flip transforms → loads without error and outputs **gibberish**. Must use the fork.
- **CPU/ARM path (what the board uses — no usable Vulkan) is fully supported:**
  - **PQ2_0**: has a NATIVE ARM NEON/**dotprod** kernel (`ggml/src/ggml-cpu/arch/arm/quants.c:297`, 134 vdotq uses; board HAS dotprod). → fastest low-bit path on RK3588.
  - **PTQ1_0**: **scalar generic only on ARM** (no NEON kernel; aliased in `arch-fallback.h`). Correct but slower per-weight unpack. This flips the earlier "smaller=faster" guess.
  - Q1_0 / Q2_0: native ARM kernels present.
  - **Hadamard/FWHT + sign flips**: scalar, arch-independent (`ops.cpp` `ggml_compute_forward_fwht_impl<float/fp16>`), CPU ✅. The Bonsai-2-critical transform IS present.
- Per `Bonsai-demo/BACKEND-SUPPORT.md` (baseline b10709): CPU row = ✅ for Q1_0/PQ2_0/PTQ1_0/Q2_0 AND ✅ Hadamard.
- **Min release: b10709** (board has b10660 = too old). Prebuilt aarch64 CPU asset exists: **`llama-prism-b10735-842b188-bin-ubuntu-arm64.tar.gz`** (also `-vulkan-arm64`, but board has no libvulkan → use plain CPU one). No need to compile.
- ~~Recommendation: PQ2_0 (native ARM dotprod kernel)~~ **OVERTURNED BY MEASUREMENT — see results below. PTQ1_0 wins on the board.**
- Formats/quants for Bonsai 2 27B: main HF repo `prism-ml/Ternary-Bonsai-2-27B-gguf` has PQ2_0 (default) + PTQ1_0 (1.75bpw/5.9GB); Q2_0 (2.25bpw/7.6GB) is testing-only in `-gguf-dev`. **Model is private → needs HF token (`BONSAI_TOKEN=hf_...`).** No ARM/RK3588 CPU benchmark exists yet (all community benchmarks are GPU); we'd be first.

**ON-BOARD RESULTS (2026-09-24):** Bonsai 2 27B **RUNS CORRECTLY on the RK3588** — PQ2_0 answered "The capital of France is Paris." (coherent, not gibberish → fork Hadamard/sign-flip transforms work on ARM CPU). Setup on board:
- Binary: `~/bonsai2/bin/llama-prism-b10735-842b188/` (prebuilt `ubuntu-arm64` CPU asset, b10735; auto-loads `libggml-cpu-armv8.2_2.so`). Run with `LD_LIBRARY_PATH=$BIN`, `ulimit -n 65536`, `-ngl 0`.
- Models: `~/bonsai2/models/Ternary-Bonsai-2-27B-{PQ2_0,PTQ1_0}.gguf` (6.8G / 5.6G). PQ2_0 ftype reports "2.13 bpw (group 128)".
- **llama-bench flag gotcha (this build): `-fa on|off|auto`, NOT `-fa 1`** (a bad `-fa` value makes llama-bench exit silently with no table). Dense 24B is slow to load from SD + slow compute; benchmark runs take many minutes.

**MEASURED (llama-bench, -ngl 0 -fa on -t 8, governor=performance, 48°C no throttle, 2 reps):**
| Bonsai 2 27B quant | size | pp512 t/s | tg128 t/s |
|---|---|---|---|
| PQ2_0 (2.13 bpw) | 6.70 GiB | 0.40 | 0.32 |
| PTQ1_0 (1.75 bpw) | 5.53 GiB | **0.49** | **0.38** |
arch reported = `qwen35`, params 26.90 B **DENSE** (all active/token).

**Two key findings:**
1. **PTQ1_0 is faster AND smaller on the board** (+19% tg, +23% pp) — this OVERTURNS the source-analysis prediction that PQ2_0's native ARM dotprod kernel would win. On this board the smaller footprint + less Hadamard/BW cost dominates the vec_dot kernel advantage. Same "smaller=faster" pattern as v1. **Recommend PTQ1_0 for the RK3588** (also the production/main-repo format).
2. **Bonsai 2 27B is ~3-4× SLOWER than Bonsai v1 27B** (v1 PQ2_0 was 1.13 tg at similar 6.66 GiB) despite near-identical size — the extra cost is Bonsai 2's per-token **Hadamard rotation + sign-flip** transforms (scalar F32 FWHT on CPU). Net: ~0.3-0.4 tok/s = "ask and wait", marginal for interactive use. Governor reset to schedutil after.

**FULL COMPARISON (all on b10735 binary, -ngl 0 -fa on -t 8, perf gov, 2 reps) — done 2026-09-25:**
| Model | arch | active | size | pp512 t/s | tg128 t/s |
|---|---|---|---|---|---|
| Bonsai 2 27B PQ2_0 | qwen35 dense | 26.9B (all) | 6.70 GiB | 0.40 | 0.32 |
| Bonsai 2 27B PTQ1_0 | qwen35 dense | 26.9B (all) | 5.53 GiB | 0.49 | 0.38 |
| ERNIE 4.5 21B-A3B Q4_0 | ernie4_5-moe | ~3B | 11.64 GiB | 25.98 | 5.59 |
| LFM2.5-8B-A1B Q4_0 | lfm2moe | ~1B | 4.50 GiB | 63.26 | 7.96 |

**Takeaway:** the MoE models are **15-25× faster at generation** and **65-130× faster at prefill** than dense Bonsai 2 27B. LFM2.5 (7.96 tg) and ERNIE (5.59 tg) are genuinely interactive; Bonsai 2 (0.3-0.4 tg) is "ask and wait". The gap = sparse activation (few active params/token) + standard Q4_0 (no per-token Hadamard). tg tracks ACTIVE params, not file size (ERNIE 11.6 GiB but only ~3B active → fast). The single b10735 binary runs all archs. Trade-off is capability, not just speed: Bonsai 2 is a strong reasoning+vision model, ERNIE/LFM2.5 are lighter MoEs — tok/s comparable, quality tiers are not. Governor reset to schedutil.

**Comparison set to benchmark (llama-bench, -ngl 0 -fa 1 -t 8):**
- Bonsai 2 27B PQ2_0 and PTQ1_0 (once runnable)
- Bonsai v1 27B PQ2_0 / Q1_0 (already measured: 1.13 / 1.87 tok/s gen)
- **ERNIE 4.5 21B-A3B** Q4_0 (Unsloth imatrix, 12 GB, `~/models/ERNIE-Q4_0.gguf`) — MoE, ~3B active/token, standard Q4_0 on stock llama.cpp. **Never yet run on the board — no logs.** Expected faster than the 27B ternary models.
- **LFM2.5-8B-A1B** Q4_0 (4.6 GB, `~/models/LFM2.5-8B-A1B-Q4_0.gguf`) — LiquidAI MoE. (Older LFM2-8B and the gemma/qwen models were deleted 2026-09-24 to free disk.)

Board after cleanup: 25 GB free (was 11).

**Forks (done 2026-09-24, gh authed as unimatrix099, workflow scope added, `gh auth setup-git` credential helper active):**
- `unimatrix099/Bonsai-demo` — real network fork of `PrismML-Eng/Bonsai-demo`. Local: `/workspace/Bonsai-demo` (origin=fork, upstream=PrismML-Eng, branch main). Note upstream branch `update-runtime-b10735`.
- `unimatrix099/PrismML-llama.cpp` — **standalone repo, NOT a network fork** (couldn't network-fork PrismML-Eng/llama.cpp because `unimatrix099/rk-llama.cpp` already occupies the ggml-org/llama.cpp fork network). Default branch `prism` pushed. Local: `/workspace/PrismML-llama.cpp` (origin=standalone, upstream=PrismML-Eng/llama.cpp). The `prism` branch already has Bonsai-2 quant kernels (PTQ1_0/PQ2_0, tied Hadamard weights); v1 model file is `src/models/qwen35moe.cpp` — still need to confirm a hybrid-linear-attention model file exists for Bonsai 2.
