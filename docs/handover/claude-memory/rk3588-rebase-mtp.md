---
name: rk3588-rebase-mtp
description: "Fork rebased onto upstream 2026-10-02 (branch rebase/w4a4-on-upstream); Gemma-4 MTP drafters measured, best decode E4B 10.67 / E2B 19.94 t/s; next opportunity = repacked routed copy"
metadata:
  node_type: memory
  type: project
  originSessionId: 07117338-6dde-411f-94d9-7e827590afe8
  modified: 2026-10-02T22:49:04.349Z
---

On 2026-10-02 the RKNPU2 fork was rebased onto ggml-org master (4ebdf2c74) as local branch `rebase/w4a4-on-upstream` (not pushed). The feature branch `feat/w4a4-neon-prep` has a parentless root commit (cb909e64f = copy of rknpu2 tip 81eff6a45), so the rebase is two-stage: rknpu2's 12 commits onto upstream, then the feature commits onto that. Fork base was upstream 650bf14eb. The fork's Gemma-4 QAT workaround commit was dropped (upstream supersedes it).

Validated with a paired A/B on the board: bit-identical PPL (26.8771), speed equal or better. Gemma-4 MTP results, all output-identical: pure CPU with `--spec-draft-n-max 3` is best (E4B 10.67, E2B 19.94 t/s), because the M=4 verify batch fits the CPU_REPACK 4-row tile. Routed mode loses with MTP because its CPU path reads the plain (not repacked) RKNPU host copy. The backend now implements `ggml_backend_set_n_threads`, which fixed libgomp churn on M>1 verify batches (NPU MTP 5.26→8.49).

**Why:** this is the new baseline. Older docs' E4B numbers came from ggml-org's E4B Q4_0, not unsloth's (the guide is now fixed). The board also changed after the 2026-09-24 reflash: pure-CPU decode is now faster than routed.

**How to apply:** build on the rebased branch. The top open opportunity is a routed mode whose CPU copy is repacked, combining NPU prefill with repacked decode and MTP. Details are in decode research #1b. See [[rk3588-board-access]] and [[rk3588-project-workflow]].

**Update 2026-10-05 (NPU decode loop):** on `rebase/w4a4-on-upstream`, E4B pure-NPU W4A4 decode went from 5.81 to 8.24 t/s (+42%), bit-identical. The biggest single gain was running NPU segments on ggml's hot OpenMP team (+28.7%). The NPU share of each token is now at the board's ~26.6 GB/s bandwidth ceiling. A pre-existing llama-server segfault (zero-row output mul_mat → 713 MB packed read-back) was fixed in 52647f68e. Write-up is decode research #1c; the loop harness is in `docs/backend/rknpu2-autoresearch/`. The board is now at **192.168.0.178** (the IP changed).

**Update 2026-10-05 (prefill loop):** E4B pp512 went from 68.4 to 161.5 t/s (2.36×), pp128 to 150.4, and decode tg64 to 8.63, on `rebase/w4a4-on-upstream`. The NPU now also runs prefill flash attention and the BF16 per_layer_model_proj, and prefill is pipelined. Driver gotcha: the RKNN driver converts a non-native runtime B matrix at `rknn_matmul_set_io_mem` time, so you must re-bind after writing it. W4A4 output is chaotic and 8-chunk perplexity is useless as a gate; use 32-chunk PPL plus KLD vs the CPU logits (`~/kld-cpu-e4b-4ch.bin` on the board). The remaining prefill limit is the CPU dequant of the NPU's INT16 output (~31%). Details are in decode research #1d/#1e.

**Update 2026-10-05 (prefill loop 3, CPU-side work):** E4B pp512 went from 161.5 to 203.4 (pp128 169.7, tg64 8.65), and all keeps were bit-identical. The gains came from GEGLU in the backend fused into up's dequant, deferring the FFN gate's dequant into that fused GEGLU (`RKNPU_DEFER_GATE`), host buffer types, ZVA stores, and K-segment-local Hadamard. Lessons:
- Pin code alignment with `-falign-*=64` and judge small effects with a same-binary env A/B. Rebuilt binaries swing ±5%.
- Dequant at N=10240 is ~3.5× slower per element than at N≤2560, so cutting bytes wins and re-blocking does not.
- `perf record -e cycles:u -D 12000` works without sudo on the board.

Details are in decode research #1f.

**Update 2026-10-06 (prefill loop 4, whole-block scheduling):** E4B pp512 went from 208 to 275 (pp128 188.7, tg64 8.78, NPU+MTP n=1 9.29), stopping at iteration 30/40 on a plateau.
- The FFN `[gate, up, GLU, down]` and the NPU attention now run as scheduled blocks across nodes.
- ggml-cpu fuses norm+mul+add(+scale)(+next norm).
- Lesson: GCC contracts `vmulq`+`vaddq` intrinsics into an FMA. Fused float ops need an `asm("" : "+w"(p))` barrier to stay bit-identical.
- Open issue: NPU attention output depends on which core serves a KV group (±0.02% PPL).
- Q/K/V sharing a sign vector per layer costs +9.7% PPL.
- The container's cmake lived in the scratchpad and is gone, so build on the board (`verify.sh`). `pkill -f pattern` can kill its own shell.

Details are in decode research #1g.

**Update 2026-10-06 (attention block, steps 1-3):** Q/K/V head norms, RoPE and the KV-cache writes are now fused into the projection dequant, and all of it is bit-identical. pp512 went from ~277 to ~288, and pp128 to 192.
- Trap: llama pins tensors named "norm" to the CPU, so gemma4.cpp builds the Q/K norms without that name.
- RoPE exactness depends on GCC's FMA contraction; `test-rknpu2-rope.cpp` checks it in the guard.
- Step 4 (block-causal scheduler) was stopped at the probe. Attention is CPU-bound and only ~55 ms of NPU time is exposed, so the ceiling is ~2-3%.
- Open options, already discussed with the user: (1) a small cross-node overlap, (2) FP16 softmax, (3) W8A8 attention projections. They are in decode research #1h.

The board was shut down on 2026-10-06 at the user's request and must be powered on again before any board work.
