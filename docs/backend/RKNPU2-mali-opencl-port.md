# Mali GPU via OpenCL on RK3588: how far it goes, and why it stops

Branch: `feat/opencl-mali`, stacked on `feat/w4a4-neon-prep`.
**Status: abandoned deliberately — the port compiles and runs, but is
numerically incorrect AND ~15x slower than the CPU. Fixing correctness
would not make it useful.** Kept as a record so nobody re-walks it, and
as working groundwork if a future Mali-tuned kernel set appears.

Companion: `RKNPU2-decode-research.md` #3i (the multimodal measurements
that motivated this).

## Why we tried

E4B's vision encoder costs **12.5 s per image on the CPU** with no
acceleration, dwarfing everything else in a multimodal turn (~32 s
total). The NPU only helps 5% because it takes just 2D `MUL_MAT` and
fragments the 940-node ViT into 227 graph splits. A GPU backend supports
the whole op set, so it would take the graph in one split — the
structural argument for trying.

## What works (all verified on the board)

| Layer | State |
|---|---|
| GPU hardware | Mali-G610 MP4 @ 1 GHz, `/dev/mali0` present |
| Kernel driver | ARM proprietary (vendor BSP) |
| Userspace | Rockchip libmali blob `g13p0`, 164 OpenCL symbols |
| OpenCL | **works** — OpenCL 3.0, 4 CUs, fp16, subgroups, int8 dot product |
| llama.cpp build | builds clean with `-DGGML_OPENCL=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF` |
| Device enumeration | **works** after this port — `GPUOpenCL: Mali-G610 r0p0` |
| All 47 kernels compile | **yes**, after the fixes below |
| Numerical correctness | **NO** — PPL 104.4 vs 22.7 on CPU |
| Speed | **NO** — 15x slower than CPU at prefill |

**Vulkan is not an option** on this board: the vendor BSP binds the GPU
to ARM's proprietary driver, so `panfrost` never binds and there is no
DRM render node for it (`card0` = rockchip-drm, `card1` = RKNPU). Mesa
enumerates only `llvmpipe`. Changing that means replacing the kernel,
which would take the NPU driver with it.

## What this port changes

Host side (`ggml-opencl.cpp`):
- `GPU_FAMILY::MALI` added; detection by device name, restricted to
  Valhall and newer (`-G5`/`-G6`/`-G7`).
- A single `subgroup_size` field replaces five copies of the
  `if ADRENO 64 / else INTEL 32 / else assert` chain. This is not just
  tidying: that value sizes local-memory kernel arguments
  (`sizeof(float)*nth/sgs`), so getting it wrong silently corrupts
  reductions rather than merely mistuning them.
- Mali follows the generic (Intel) kernel selection at 22 dispatch sites.

Kernel side (37 `.cl` files, one uniform preamble):
- The kernels self-detect the vendor from extension macros
  (`cl_intel_required_subgroup_size` / `cl_qcom_reqd_sub_group_size`).
  Mali advertises neither, so no family was defined and `N_DST` /
  `N_SIMDGROUP` / `N_SIMDWIDTH` never existed — the original
  `error code 63` compile failure. Added an `#else` branch.
- It reuses `INTEL_GPU`'s tuning constants, which is safe and checked:
  `INTEL_GPU` guards **only** tuning constants (no Intel intrinsics), and
  **no kernel sets `N_SIMDWIDTH 32` under it** while 20 set 16 — matching
  Valhall's 16-wide warps.
- Subgroup width is pinned with the core
  `__attribute__((reqd_sub_group_size(16)))`. Leaving it unpinned is not
  an option — see the measurement below.
- `l2_norm.cl` and `mul_mv_q4_k_f32.cl` use `half` without enabling
  `cl_khr_fp16`. Adreno and Intel compilers accept that implicitly; Mali
  rejects it per spec. Added the pragma. **These two are genuine upstream
  bugs, portable-correctness fixes independent of Mali.**

## Why it stops

**1. Numerically wrong.** Qwen2.5-1.5B Q4_0, 8 chunks of wikitext:

| Config | PPL |
|---|---|
| CPU reference (`RKNPU_CPU_DECODE=999999`) | **11.38** |
| NPU W4A4, the default path for this file | 22.68 |
| Mali OpenCL, subgroup width unpinned | 107.75 |
| Mali OpenCL, width pinned to 16 | 104.37 |

> Corrected 2026-08-23. The first version of this table labelled 22.68 as
> the CPU reference. It is not: `--device none -ngl 0` does **not**
> disable the RKNPU backend (llama-bench still prints `RKNPU` in the
> backend column), so that run was the NPU's own 4-bit path. The only
> reliable way to get a CPU-only baseline on this fork is
> `RKNPU_CPU_DECODE=999999`, which routes every matmul to the CPU. The
> conclusion is unaffected and in fact stronger — Mali is ~9x worse than
> CPU, not ~4.6x.

Pinning the width was necessary but nowhere near sufficient — it moved
PPL by 3%. Something deeper in the Adreno/Intel-tuned kernels does not
hold on Mali. Localizing it means per-op comparison via
`test-backend-ops` (build with `-DLLAMA_BUILD_TESTS=ON`), which is the
correct next step for anyone resuming this.

**2. And it would not be worth fixing.** Same model, same build:

| Backend | pp128 | tg64 |
|---|---|---|
| CPU, 4 threads (`RKNPU_CPU_DECODE=999999`) | **107.85** | **22.25** |
| NPU W4A4, the default path | 101.50 | 6.38 |
| Mali OpenCL, all layers | 7.14 | 3.36 |

**15x slower at prefill and 6.6x slower at decode than the CPU** — and
still 14x/1.9x slower than the NPU path it would have to beat. (These
ratios were understated in the first version of this document, which
compared against a baseline mislabelled as CPU; see the note above.)
A 4-core Mali-G610 at
1 GHz is simply not competitive with four Cortex-A76 cores running
llama.cpp's NEON kernels — at least not through kernels written and
tuned for Adreno. Since the whole motivation was accelerating a
compute-bound FP16 vision encoder, and prefill matmuls are the same kind
of work, the vision tower would lose too.

**Verdict: the GPU is not a useful compute resource for llama.cpp on
RK3588.** That closes the "use the GPU" avenue with data rather than
speculation. The NPU remains the only worthwhile accelerator here, and
the vision encoder stays on the CPU.

## If someone resumes this

The groundwork above is reusable. What would have to change for the
conclusion to flip:

1. **Correctness**: run `test-backend-ops` on the OpenCL backend to list
   the failing ops, then fix or disable them individually.
2. **Performance**: the generic kernels are Adreno-shaped. Mali would
   need its own tuning — different tile sizes, vector widths, and use of
   `cl_arm_integer_dot_product_int8`, which the device advertises and no
   kernel currently uses. That is a kernel-authoring project, not a
   porting one.
3. Only then re-measure against the CPU baselines above.

The two `cl_khr_fp16` pragma fixes are worth upstreaming regardless —
they are portability bugs that happen to be invisible on Adreno and
Intel.
