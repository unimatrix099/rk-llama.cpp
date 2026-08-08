# RKNPU2 backend: Gemma 4 E4B / W4A4 investigation

Findings from debugging and benchmarking the RKNPU2 backend with Gemma 4 QAT
models on an RK3588 board. Two bugs were found and fixed on this branch, one
upstream limitation was measured and documented.

## Environment

| Component | Version |
|---|---|
| Board | Orange Pi 5 Ultra (RK3588, 16 GB RAM, 3 NPU cores) |
| OS | Armbian community 26.8.0 (Debian 13 trixie) |
| Kernel | 6.1.115-vendor-rk35xx (BSP, built-in RKNPU driver v0.9.8) |
| librknnrt.so | vendored in repo (`ggml/src/ggml-rknpu2/libs/`) |
| Toolchain | GCC 14.2, CMake 3.31, `-DLLAMA_RKNPU2=ON` |
| Base branch | `rknpu2` + PR #21 (`gemma4-assistant` ISWA support) |
| Models | `gemma-4-E4B-it-Q4_0.gguf` (4.3 GB), `qwen2.5-1.5b-instruct-q8_0.gguf` |

## Summary

| # | Problem | Root cause | Status |
|---|---|---|---|
| 1 | Gemma 4 QAT fails to load (`missing tensor blk.N.attn_k.weight`) | ISWA layers share KV cache and omit K tensors | Fixed by PR #21 |
| 2 | Deterministic crash in `calculate_entropy_amax` during W4A4 weight upload | Out-of-bounds negative histogram index | Fixed: `815e1e9` |
| 3 | Intermittent SIGSEGV/heap corruption during W4A4 weight upload (0/6 native runs survived) | librknnrt unmaps VA ranges aliasing glibc's mmap-served large mallocs | Fixed: `23c7bb5` |
| 4 | W4A4 prompt processing ~5x slower than W8A8 | RK3588 INT4 matmul path itself; CPU Hadamard transform is secondary | Hardware/blob limitation, measured below |

With #2 and #3 fixed, the default `W4A4_HADAMARD` pipeline loads and runs
Gemma 4 E4B Q4_0 reliably: 5/5 consecutive native runs (previously 0/6).

## Issue 2: out-of-bounds histogram write in entropy calibration

`calculate_entropy_amax()` (rknpu2-calibration.cpp) clamped only the upper
bound of the histogram bin index when building the reference distribution:

```cpp
int bin_index = static_cast<int>(((data[i] + abs_max_val) / bin_width));
bin_index = std::min(num_bins - 1, bin_index);   // no lower clamp
p_dist[bin_index]++;                              // OOB write if negative
```

Non-finite or out-of-range inputs produce a negative index and a heap write
far outside the vector. The identical re-binning loop ~40 lines below already
clamped both bounds; the fix applies the same `std::max(0, ...)` here.

This function only runs for INT4 pipelines (`W4A4_*`), which is why Q8_0/F16
models never crashed: the entropy/KL-divergence scale search is the INT4-only
calibration step, invoked per weight segment at load time.

## Issue 3: librknnrt address-space collision with glibc malloc

### Symptom

After fixing issue 2, loading Gemma 4 E4B Q4_0 through W4A4 still crashed on
most native runs, each time somewhere else: inside `calculate_entropy_amax`,
inside `calculate_percentile_amax`, inside libgomp, inside
`quantize_fp32_to_int4_packed`. With debug symbols the picture became
consistent: a freshly allocated 50+ MB calibration buffer (one segment of the
262144x2048 embedding tensor = 13.9M floats) faulted *while being filled or
read by ordinary single-threaded code*. The backing pages were disappearing
mid-use.

### Evidence chain

| Experiment | Result | Eliminates |
|---|---|---|
| Native, default | 6/6 crash (varied locations) | — |
| Under gdb (ASLR off) | pass | — |
| Under gdb (ASLR re-enabled) | pass | ASLR-seed theory |
| AddressSanitizer build | 4/4 pass, zero reports | OOB/UAF in fork code |
| Native, `setarch -R` (ASLR off) | 3/3 crash | pure timing race |
| Native, `MALLOC_MMAP_MAX_=0` | **3/3 pass** | — confirms VA collision |

The only intervention that fixed it at full native speed changes *where*
glibc places large mallocs (>=128 KB are served by `mmap` by default, landing
in the same VA region as librknnrt's DMA/device mappings; with the tunable
they come from the brk heap instead). strace additionally shows librknnrt
worker threads issuing large `munmap` calls in that contested region during
weight upload.

Conclusion: librknnrt (closed source) unmaps address ranges that can alias
live glibc mmap chunks. Instrumented environments (gdb, ASan) rearrange the
address space and/or timing enough that the collision never occurs, which is
why the bug resists debuggers.

### Fix

`ggml_backend_rknpu2_reg()` now calls `mallopt(M_MMAP_MAX, 0)` once (glibc
only), keeping all large allocations on the brk heap. Trade-off: somewhat
higher peak RSS, since the brk heap returns memory to the OS less eagerly.
This protects the whole process, not just the backend's own buffers.

Note for issue triage: this likely explains other "random crash during
load" reports against this backend on models with large tensors, and is
worth checking against issue #15 (Qwen3-Coder-Next Q6_K load failure —
Q6_K also routes through `W4A4_HADAMARD` for half its layers).

## Benchmarks (RK3588, `llama-bench -p 128 -n 64`)

### Gemma 4 E4B Q4_0 (7.46 B params, MoE-style with per-layer embeddings)

| Pipeline | pp128 t/s | tg64 t/s | NPU memory |
|---|---|---|---|
| CPU only (`RKNPU_HYBRID=CPU_STANDARD`) | 25.2 | **4.9** | 0 |
| NPU `W8A8_STANDARD` | **39.8** | 3.3 | 4.9 GB |
| NPU `W16A16_STANDARD` | 28.2 | 1.9 | 9.3 GB |
| NPU `W4A4_HADAMARD` (default for Q4_0) | 7.7 | 3.4 | 4.1 GB |
| NPU `W4A4_STANDARD` (broken accuracy; control) | 10.8 | n/a | 4.1 GB |

### Qwen2.5 1.5B Q8_0 (dense), default `W8A8_STANDARD`

Prompt ~47-54 t/s, generation ~5 t/s, NPU 1.8 GB. Works out of the box.

## Why W4A4 is slow (and why offloading the Hadamard transform won't fix it)

The INT4 pipelines require a randomized Hadamard transform on every
activation row (CPU-side, per token, per layer) to spread outliers before
4-bit quantization — weights get the matching transform once at load time.
The natural question is whether that per-token CPU work is the bottleneck
and could be moved to the NPU.

Measured answer: **no.** `W4A4_STANDARD` runs the identical INT4 NPU path
with zero transform cost and reaches only 10.8 t/s prompt processing vs
39.8 t/s for W8A8. So even a free transform leaves INT4 ~3.7x slower than
INT8; the dominant cost is the INT4 matmul path itself (rknn INT4 mode,
nibble packing, INT16 output handling), inside the closed runtime. The
upstream README's own dense-model numbers show the same fingerprint
(Gemma3 1B: Q4_0 NPU pp 51 t/s vs Q8_0 NPU pp 378 t/s).

For reference, offloading the transform is *technically* feasible — the
sign-flip vector and Hadamard matrix fold into one constant +-1 matrix,
executable as an FP16 NPU matmul; for single-token generation the full
K x K constant would be bandwidth-bound (megabytes re-read per layer per
token), so a practical design would use a block-diagonal transform
(QuaRot-style) to cut that read ~16x. Given the measured ceiling above,
this is not worth the complexity on RK3588.

**Practical guidance:** treat W4A4 as a capacity mode (halves NPU memory vs
INT8, fits larger models under the 4 GB per-IOMMU-domain limit), not a speed
mode. For throughput, use `W8A8_STANDARD` — for Q4_0 GGUFs:

```sh
RKNPU_HYBRID="W8A8_STANDARD" ./build/bin/llama-cli -m model-q4_0.gguf ...
```

Note this upconverts 4-bit weights to INT8 (no accuracy gain over the source
Q4_0, per the README's "terrible case" pairing guidance) — prefer a Q8_0 GGUF
when disk/bandwidth allows.

## Operational notes

- `ulimit -n 65536` before running is mandatory: the runtime opens one fd
  per NPU memory allocation. With the default 1024 limit, loading fails with
  `failed to convert handle to fd, errno 24` followed by a segfault instead
  of a clean error. Consider `pi soft/hard nofile 65536` in
  `/etc/security/limits.conf` for persistence.
- This fork's `llama-cli` defaults to interactive conversation mode even with
  `-p`; use `-st` (single turn) for scripted runs, or `llama-completion`.
- A crash inside weight upload frequently leaves the process wedged in a
  secondary failure (`std::system_error: Resource deadlock avoided` or a
  hung gdb attach from the ggml crash handler) — the *first* fault in the
  log is the meaningful one.
