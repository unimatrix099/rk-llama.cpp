# RKNPU2: improving 4-bit token-generation (decode) speed — research notes

Open research thread; **nothing here is implemented**. Companion documents:
`RKNPU2-optimization-notes.md` (shipped work + dead ends),
`RKNPU2-int4-research.md` (INT4 fundamentals), `RKNPU2-neon-prep-plan.md`
(prefill prep). Numbers: Orange Pi 5 Ultra (RK3588, 16 GB), pinned clocks,
`llama-bench -p 128 -n 64` unless noted.

## Where decode stands after the shipped work

Decode re-reads the active weight set for every generated token, so it is
memory-bandwidth bound. For Gemma-4-E4B Q4_0 (7.46 B dense):

| Decode path | tg64 t/s | effective read rate |
|---|---|---|
| CPU (`RKNPU_CPU_DECODE=32`, native Q4_0 reads, ~2.4 GB/token) | **5.35** | ~13 GB/s — near the CPU's bus ceiling |
| NPU W4A4 native layout (~2.05 GB/token INT4) | 3.65 | ~7.5 GB/s aggregate |

Two hard facts frame everything below:

1. **The CPU path is already at its wall.** 13 GB/s effective means further
   CPU-side code optimization cannot raise single-stream decode; only
   reading fewer bytes per token, or reading with more engines at once,
   can.
2. **The NPU path is mostly overhead, not reading.** Measured single-core
   M=1 streaming is 10.3 GB/s (`rknpu2-matmul-bench.c`), and the backend
   splits each weight across 3 cores. Pure-read arithmetic puts the NPU
   decode ceiling near ~14 t/s — actual is 3.65, so **~75% of NPU decode
   time is per-node dispatch/sync (~600 graph splits/token on E4B) and the
   CPU-side ops between matmuls (attention, norms, AltUp/LAUREL)**, not
   weight traffic.

## Avenues, ranked

### 1. Speculative decoding — the only cheap idea that cheats the wall

One weight-read pass verifies several draft-proposed tokens at once, so
bytes-per-accepted-token divides by the acceptance count. With 2–3 tokens
accepted per pass, 5.35 → 8–12 t/s is plausible.

- Draft must be tiny (it shares the same bus): Qwen2.5-0.5B for
  Qwen2.5-1.5B, or Gemma-4-E2B (Q4) for E4B. Same-family pairs give the
  best acceptance.
- `llama-speculative` is already built in the board's `build/bin/`.
- Unknowns: acceptance rates on heavily quantized models; interaction with
  `RKNPU_CPU_DECODE` (both target and draft decode on CPU — they contend;
  it may be better to leave the draft on pure CPU and the target routed).
- **Effort: one evening of experiments. Do this first.**

### 2. Cooperative CPU+NPU decode — novel, physically sound, half-built

Bandwidths *sum* when different engines read different bytes
simultaneously. Layer pipelining cannot exploit this for a single stream
(layers are sequential), but **splitting each weight matrix along N** can:
NPU cores compute part of the output columns from the packed INT4 copy
while the CPU computes the rest from the original Q4_0 bytes — which are
*already host-resident* under `RKNPU_CPU_DECODE`'s dual-residency
mechanism (see `feat/mixed-precision-pipelines`).

- Combined streaming ~20+ GB/s → projected ~7–8 t/s on E4B.
- Work: in `ggml_backend_rknpu_graph_compute`, for M=1 nodes, launch the
  NPU segments and a CPU partial-GEMV concurrently, merge partial C, and
  load-balance the N split against measured per-engine rates. All within
  the backend; no scheduler changes.
- Risks: per-node join latency could eat the gain on a 250-node model
  (measure with a 2-node prototype first); CPU threads doing GEMV compete
  with the CPU-side graph ops between nodes.
- To our knowledge no RK3588 stack does this — upstream-worthy if it works.

### 3. NPU overhead surgery — attack the 75%

- QKV / gate-up matmul fusion: ~40% fewer NPU dispatches per token.
- Batch the per-node segment syncs (3 cores currently sync'd via one
  `rknn_mem_sync` each per matmul).
- Even perfect surgery only approaches the CPU's 5.35 before hitting the
  same bus — the earlier finding that graph splits are *not* the prefill
  bottleneck does not carry to decode, where fixed per-node costs dominate.
  Worth profiling after #1/#2 conclusions, not before.

### 4. Read fewer bytes

- **Hybrid per-layer patterns** (`RKNPU_HYBRID="W8A8_STANDARD,W4A4_HADAMARD"`):
  INT4 on the fat FFN tensors, INT8 on attention. Decode gain tracks byte
  reduction ~linearly; nobody has mapped the accuracy/speed curve. Free to
  explore (no code), and results are now reproducible thanks to the
  name-hash Hadamard seeding.
- **Sub-4-bit CPU formats** (Q3_K, IQ3/IQ2 via the CPU decode path): 25–40%
  fewer bytes, quality falls with them; usually a bad trade at 7B, possibly
  a good one for fitting a 13B.
- **MoE architecture beats all of this**: LFM2-8B-A1B reads ~1.5 GB/token
  → 7.6–9.1 t/s measured. Model choice > engineering.

### 5. Micro-tuning (one afternoon, do alongside #1)

- Thread count / affinity for CPU decode: benchmarks so far used defaults;
  `taskset -c 4-7 -t 4` (big cores only) vs all-8 can swing ±10% on
  A76+A55 hybrids.
- Re-tune the `RKNPU_CPU_DECODE` threshold per model.

### 6. Batch serving (throughput only)

Multiple concurrent streams share each weight read — large aggregate
gains, zero single-stream latency gain. Relevant only if the workload
becomes a server.

## Handover notes (continuing on another machine)

- Everything lives on `github.com/unimatrix099/rk-llama.cpp`; the current
  tip branch is `feat/w4a4-neon-prep` (stacked on
  `feat/int4-native-layout` ← `feat/mixed-precision-pipelines` ←
  `fix/w4a4-calibration-crashes` ← upstream `rknpu2` + PR #21).
- The board (Orange Pi 5 Ultra) has the repo at `~/rk-llama.cpp`, models
  in `~/models/` (`gemma-4-E4B-it-Q4_0.gguf`,
  `qwen2.5-1.5b-instruct-q8_0.gguf`, `LFM2-8B-A1B-Q8_0.gguf`), binaries in
  `build/bin/` incl. `llama-speculative`. Remember `ulimit -n 65536` and
  `scripts/fix_freq_rk3588.sh` (from the rknn-llm repo) before benching.
- Diagnostic tools build from `docs/backend/Makefile.rknpu2-tools`
  (`make all`, `check`, `check-prep`, `check-hw`, `coverage`).
- Reference numbers to beat: decode table above; prefill state in
  `RKNPU2-neon-prep-plan.md`.
