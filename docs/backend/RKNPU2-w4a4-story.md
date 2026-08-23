# Making 4-bit NPU inference actually work on the RK3588

A narrative account of six board sessions spent on llama.cpp's RKNPU2
backend, written for readers who did not live through it. The two
companion documents are the technical record — `RKNPU2-decode-research.md`
(every avenue, ranked, with verdicts) and `RKNPU2-optimization-notes.md`
(the ledger of what shipped and what died). This one tells the story,
including the wrong turns, because most of the useful information is in
those.

All numbers: Orange Pi 5 Ultra (RK3588, 16 GB), pinned clocks,
`llama-bench -p 128 -n 64`, perplexity on wikitext-2 via
`llama-perplexity --chunks 32` unless stated. Raw logs are archived on
the board under `~/bench-logs/<date>/`.

## The setup

The RK3588 has a 3-core NPU rated ~6 TOPS. llama.cpp's RKNPU2 backend
offloads matrix multiplies to it through Rockchip's closed
`librknnrt.so`, which accepts a few fixed operand type combinations. The
two that matter here:

- **W8A8** — int8 weights, int8 activations.
- **W4A4** — int4 weights, int4 activations. Half the weight bytes, and
  the NPU's fastest mode on paper.

Note what W4A4 is *not*: it is not the same thing as a "4-bit model". A
Q4_0 GGUF stores 4-bit weights with a scale for every 32 of them, and
computes against 8-bit activations. The NPU's int4 mode demands 4-bit on
**both** sides, with far coarser scales. That distinction turns out to be
the whole story.

The target model is Gemma-4 E4B (7.46 B dense) as a Q4_0 GGUF, with
Qwen2.5-1.5B and LFM2-8B-A1B as cross-checks.

### Where we started

The fork already had substantial work behind it (native A/C tiling, NEON
prep vectorization). At the start of these sessions:

| E4B Q4_0 | pp128 | tg64 | quality |
|---|---|---|---|
| Routed (NPU prefill, CPU decode) | 41.3 | 4.9 | CPU-exact |
| Pure NPU W8A8 | 39.8 | 3.3 | ≈ CPU |
| Pure NPU W4A4 | 34.4 | 3.65 | badly degraded, unmeasured |

W4A4 was documented as "a capacity mode, not a speed or quality mode":
you used it to fit a model in NPU memory, and you paid for it. Nobody had
put a number on what you paid.

## Act 1: Two ideas that failed, and why that was worth it

The decode path is memory-bandwidth bound — every generated token
re-reads the whole active weight set — so the research doc's top two
ideas both attacked bandwidth.

**Speculative decoding.** One weight-read pass verifies several
draft-proposed tokens, so bytes-per-accepted-token divides by the
acceptance rate. Standard technique, big wins on GPUs. Here every variant
lost: a Qwen2.5-0.5B draft gave 4.30 t/s against a 13.52 baseline;
draft-free ngram self-speculation was exactly neutral (13.80 vs 13.81);
routing the verification batches to the NPU raised acceptance to 60% and
still lost. The reason is structural — this stack has no cheap batch
verification. On 4 big cores an M=9–17 batch costs 2–3× a single token,
and on the NPU each verify batch pays the full multi-split graph
overhead. Closed with data.

**Cooperative CPU+NPU decode.** If the CPU reads the original Q4_0 bytes
while the NPU reads its packed int4 copy, the two engines' bandwidths
should sum. This one was novel enough to be worth a purpose-built probe
before any backend work: a standalone tool that replicates the backend's
exact per-node driver sequence and sweeps the CPU/NPU split fraction.
Verdict: bandwidths *do* sum (~30 GB/s combined vs 26–28 NPU-solo), but
at realistic layer shapes the margin was only **+4.7% to +9.3%** — under
the ≥25% gate we had set in advance. Not worth the complexity.

The probe's real value was a number nobody expected: **the NPU's per-node
driver path sustains ~28.6 GB/s at M=1**. Pure weight traffic for E4B
would then be ~72 ms/token, about 14 t/s — but the backend was decoding
at 3.65. Roughly 200 ms per token was going somewhere that was neither
weight reads nor driver dispatch. That killed the prevailing theory
("NPU decode is dominated by dispatch overhead") and pointed the next
session at the backend's own CPU-side code.

**Lesson worth keeping:** a failed experiment that produces a *number*
is not a failure. Both dead ends re-aimed everything that followed.

## Act 2: 19–66% for free, from a `taskset`

Before the deep work, a mundane sweep: thread count and core affinity.
The docs had guessed "±10%". The measurement:

| Model / config | tg64, 8 threads | tg64, 4 big cores | gain |
|---|---|---|---|
| Qwen2.5-1.5B routed | 8.93 | 13.52 | +51% |
| E4B W8A8 + routed | 4.64 | 5.50 | +19% |
| LFM2-8B-A1B routed | 8.22 | 13.66 | +66% |

The RK3588 is a big.LITTLE part (4× A76 + 4× A55). Putting the little
cores on the critical path drags the whole cluster down; four big-core
threads reach ~25 GB/s effective where eight mixed threads reached ~13.
A documented "CPU decode wall" turned out to be a threading artifact.

An oddity surfaced here and got parked: E4B under a strict `taskset`
was slow *and* noisy (pp 8.75 ± 2.10). It resolved itself in the next
act.

## Act 3: A profiling detective story

The question: where do ~200 ms/token go in NPU decode?

**perf** said ~95% of on-CPU samples sat in libgomp spin-wait — the CPU
was doing almost nothing while the NPU worked. Suspicious, but not an
answer.

**strace** gave the answer's shape: **33,030 `clone3` calls in 30 s of
decoding.** That is ~550 thread *creations* per generated token — and
E4B's graph has 603 NPU splits per token. One thread creation per split.

**gdb**, breaking on `pthread_create`, named the culprit:
`GOMP_parallel ← ggml_backend_rknpu_graph_compute`. The backend's
per-node OpenMP regions requested different team sizes (8 for the
activation prep, `num_threads(3)` for the matmul dispatch) than the
surrounding ggml regions (4), and libgomp responded by tearing down and
respawning its worker team around every single node.

`OMP_NUM_THREADS=4` alone recovered +42% on NPU decode — and explained
Act 2's parked anomaly, since `taskset` to 4 cores implicitly caps
libgomp's team too.

The code fix removed the env workaround: no OpenMP teams at all on the
M=1 path (`if(M > 1)` on the prep and dequant regions — a single row is a
few microseconds of NEON, cheaper serial than any team formation), and a
persistent 2-worker dispatch pool replacing the `num_threads(3)` region.
Thread creations during decode: **33,030 → 0**. NPU decode +28%, W4A4
decode +18%, and the best routed config no longer needed any environment
variables.

Two variants were tried and rejected with measurements: spinning the pool
workers longer, and pre-waking them before the job was ready. Both *lose*
— the spinners steal exactly the cores the prep work needs.

Re-profiling the fixed path with an `LD_PRELOAD` shim that wall-clocks
every `librknnrt` entry point decomposed the token cleanly: 62% NPU
matmul wall (within 10% of the probe's bandwidth ceiling), 2% driver
calls, 36% sequential CPU graph work (attention, GLU, activations). Both
decode paths were within ~15% of physical limits. The software-waste era
was over — which also bounded QKV fusion, a long-standing backlog idea,
to ~5 ms/token of addressable time. Deprioritized with numbers.

## Act 4: The model's dimensions were not powers of two

With speed near its floor, attention turned to W4A4's quality, and the
first finding was not about quality at all.

W4A4 uses a Hadamard transform to spread activation outliers before
4-bit quantization. The fast Walsh–Hadamard transform needs a
power-of-two length, so the code padded K up to the next one. Gemma-4
E4B's dimensions are 2560, 10240, 10752 — **none** are powers of two. A
GGUF header census made the damage concrete:

```
K= 2560 -> K_op= 4096  x216 tensors   (1.60x)
K=10240 -> K_op=16384  x42            (1.60x)
total: 3.38 GB nominal -> 5.22 GB stored and read per token (1.54x)
```

The padding was silently eating most of int4's byte advantage, in memory
and bandwidth alike.

The fix is standard in the QuaRot literature but had not been applied
here: **block-diagonal FWHT**. Transform in blocks of the largest power
of two dividing K (2560 = 5×512, 10240 = 5×2048), so no padding is
stored, uploaded, read, or transformed. Power-of-two dimensions are
bit-identical to before.

W4A4 decode 4.31 → 5.51 t/s, prefill 34.4 → 38.0, NPU memory **3569 →
2521 MiB (−29%)**. A middle variant (pad to a minimum block size) was
measured strictly worse and rejected.

## Act 5: The quality question, and what it turned out to be

Only now was the obvious question asked properly: *how bad is W4A4,
really, and whose fault is it?* Three configurations, same model file,
same text:

| E4B Q4_0, 32 chunks | PPL |
|---|---|
| Pure CPU (llama.cpp Q4_0×Q8_0 kernels) | 27.01 |
| NPU W8A8 | 27.29 |
| NPU W4A4 (as it then stood) | 163.01 |

This is the pivotal measurement of the whole effort. The 4-bit *file* is
essentially lossless — the CPU runs it at 27.01, indistinguishable from
8-bit. The 6× degradation belonged entirely to the NPU's int4 *scheme*.
Two suspects:

1. **4-bit activations.** 15 usable levels, with the scale set by the
   largest value in the row. LLM activations have outliers; one of them
   stretches the scale until ordinary values collapse onto 2–3 levels.
   Errors also compound — each layer's quantization noise becomes the
   next layer's input.
2. **Scale granularity.** Q4_0 gives every 32 weights their own scale.
   The RKNN API takes roughly *one scale per matmul segment* — millions
   of weights sharing one number.

Suspect 2 was testable cheaply, thanks to a small piece of algebra:
since `C[m,n] = Σₖ A[m,k]·B[k,n]`, any scale that varies along the
**output channel** `n` factors straight out of the hardware's summation.
It can be applied afterwards, in the dequantization pass the backend
already runs on the CPU. Finer granularity, zero NPU cost.

Per-output-channel weight scales took W4A4 from **163.01 → 34.36** at
unchanged speed. They also deleted the entropy-search calibration,
cutting W4A4 model load from minutes to seconds.

### The self-correction

That change had a flaw its author (me) missed: the entropy search it
replaced was not merely computing a scale, it was searching for an
optimal **clipping** point via KL divergence. Swapping it for a plain
`amax` won granularity and silently lost clipping.

Restoring clipping as a constant factor — `scale = clip · amax / 7`,
where values above the clip saturate in exchange for finer steps on the
bulk — is one multiply instead of a 128-step search, and the two effects
compose. A 32-chunk sweep on two models put the optimum at 0.93 for
weights and 0.9 for activations, on a smooth curve rather than a knife
edge:

| B clip | 1.0 | 0.95 | 0.94 | **0.93** | 0.92 | 0.91 |
|---|---|---|---|---|---|---|
| E4B PPL | 34.36 | 28.83 | 27.76 | **26.95** | 28.72 | 30.02 |

**W4A4 landed at 26.95 — level with W8A8 (27.29) and the pure-CPU
reference (27.01).** The mode that had been "capacity only, quality be
damned" was now indistinguishable from 8-bit, at 29% less NPU memory.

## Act 6: The same fix for 8-bit, and a bug caught in the act

Hardened 32-chunk validation across models revealed that W8A8 was not
uniformly innocent either: lossless on E4B, but **+38% PPL on Qwen**
(12.24 vs its CPU reference of 8.88). Same coarse-scale mechanism, same
fix — per-channel scales for the int8 path took Qwen to **9.08**, within
2.3% of CPU.

The valuable part is the variance: W8A8 went from "+1% on one model, +38%
on another" to "+2.2% and +2.3%". Predictable per-pipeline behaviour is
what makes deployment guidance trustworthy.

While implementing it I reached for the clipping trick again, this time
on int8. It produced **PPL 10106**. The cause was in our own notes:
`quantize_fp32_to_int8` has no clamp, because it is only ever called with
`scale = amax/127`, making overflow unreachable *by construction* — the
NEON narrowing was deliberately allowed to wrap. A clip below 1 makes it
reachable: extremes land near 134, wrap to −122, and sign-flip the
largest weights. The knob was deleted rather than shipped, and the docs
now carry a standing rule: **any new scale factor requires a clamp audit
of the quantizer it feeds.** This was the second time that property bit
this project.

## Act 7: How far does "4-bit is free" actually go?

E4B reaching parity invited the obvious question: does it generalize? It
does not, and the shape of the answer matters more than the answer.

| Model / file | CPU | W8A8 | W4A4 | W4A4 penalty |
|---|---|---|---|---|
| Gemma-4 E4B (7.5B), Q4_0 source | 27.01 | 27.61 | **26.88** | **−0.5%** |
| Qwen2.5-1.5B (1.8B), Q4_0 source | 9.36 | 9.56 | 18.73 | +100% |
| Qwen2.5-1.5B (1.8B), Q8_0 source | 8.88 | 9.08 | 21.74 | +145% |

The hypothesis going in was **source precision**: requantizing an
already-4-bit file to int4 should cost less than crushing an 8-bit one.
That is true — the same model improves from +145% to +100% purely by
starting from a Q4_0 file — but it explains perhaps a third of Qwen's
gap and none of E4B's parity. A backup hypothesis, that E4B might be one
of Google's quantization-aware-trained checkpoints, was ruled out by the
GGUF metadata: it is the plain instruct model.

That leaves **model capacity** (7.5 B against 1.8 B) as the leading
explanation, consistent with the standard finding that larger models
absorb aggressive quantization better. The clean within-family control —
Gemma-4 E2B, same architecture and recipe at half the size — could not
be run, because that GGUF does not load in this fork.

Meanwhile W8A8's penalty is now +2.2%, +2.3%, +2.2% across all three.
That constancy is the real payoff of the per-channel work: quality you
can predict from the pipeline, rather than discover per model.

So the honest claim is narrower than the headline. On a ~7 B dense model
from a 4-bit source, W4A4 is quality-free here; on a ~2 B model it still
costs roughly double. Which suits the capacity mode fine — large models
are exactly what it exists for.

## Act 8: The bug hiding in the speed tables

Hunting for a third data point, I ran a perplexity measurement on
LFM2-8B-A1B — something nobody had ever done. It came back at
**17,402, against 15.86 on the CPU.**

Not a regression: the pre-existing per-segment path produces the same
garbage, so this predates every change described above. It had gone
unnoticed for a simple reason — **LFM2 was only ever speed-benchmarked.**
It appears in table after table at 9.47, 13.25, 13.66 tokens per second,
and every one of those runs was producing nonsense at speed.

The mechanism is partly established. LFM2 is a mixture-of-experts model
whose expert weights are three-dimensional — `[1792, 2048, 32]`, one
slice per expert. The backend reads `ne[0]` and `ne[1]` everywhere it
inspects a tensor, so it sizes such a tensor as though it held a single
expert; and `supports_op` never checks the higher dimensions, while
`graph_compute` takes `M` from `src1->ne[1]` and clears only `M×N` of the
destination. Batched or multi-expert work is accepted and only its first
slice computed. Dense models emit two-dimensional matmuls, which is
exactly why six sessions on Gemma and Qwen never tripped over it.

I attempted two fixes and shipped neither. Rejecting three-dimensional
weights changed nothing — perplexity came back identical to the digit,
so those tensors never reach that path. Additionally rejecting non-2D
operands did change what gets offloaded, and turned the wrong numbers
into NaN. At that point I had guessed twice at a bug I did not
understand, so I reverted, verified the restore against the dense
models' bit-identity anchors, and wrote the defect down instead.
Swapping one broken behaviour for another is not a fix.

The uncomfortable lesson is the transferable one: **we validated only
what we optimized.** Every quality gate in this project pointed at the
thing being changed. LFM2 sat in the speed tables for weeks, and one
perplexity run at any point would have caught it.

## Where it ended up

Gemma-4 E4B Q4_0, RK3588, `-t 4` on the big cores:

| Config | pp128 | tg64 | PPL (32ch) | NPU memory |
|---|---|---|---|---|
| Routed (NPU prefill + CPU decode) | **41.4** | 5.50 | 27.01 | high |
| **Pure NPU W4A4 (all defaults)** | 37.0 | **5.49** | **26.88** | **2.5 GB** |
| Pure NPU W8A8 | 41.4 | 4.55 | 27.61 | high |
| Pure CPU | 25.2 | 4.9 | 27.01 | — |

And the W4A4 arc across the whole fork's history:

| | prefill | decode | PPL | load |
|---|---|---|---|---|
| Originally | 7.7 | 3.4 | 163.01 | minutes |
| **Now** | **37.0** | **5.49** | **26.88** | **seconds** |

Nearly 5× prefill, 1.6× decode, 6× quality, −29% memory. Every step
verified against bit-identity anchors: setting four environment
variables still reproduces the original numerics exactly, so every
change remains auditable and reversible.

Cross-model: Qwen2.5-1.5B W4A4 44.31 → 21.74 and W8A8 12.24 → 9.08.
(LFM2's decode numbers, 9.47 → 13.66 t/s, are withdrawn as
quality-invalid — see Act 8.)

Two caveats belong next to those numbers rather than in a footnote: the
4-bit parity is demonstrated on one 7.5 B model and does not hold at
~2 B, and mixture-of-experts models are currently broken on this backend
regardless of pipeline.

## What did not work, so nobody repeats it

- **Speculative decoding** — draft-model, ngram self-speculation, and
  NPU-verified variants all lose; no cheap batch verification exists here.
- **Cooperative CPU+NPU decode** — bandwidths sum, but only +5–9% at real
  shapes; below the pre-registered gate.
- **QKV/gate-up fusion** — bounded by profiling to ~5 ms/token.
- **Mixed W4A16/W8A16 dtypes** — the runtime rejects them on RK3588;
  revisit only if Rockchip ships a newer librknnrt than 2.3.2.
- **int8 scale clipping** — wraps without a clamp (PPL 10106).
- **Minimum-block Hadamard** — dominated by pure block-diagonal.
- **Spinning/pre-waking the dispatch pool** — steals cores from prep.
- **Two MoE fix attempts** — rejecting 3D weights had no effect at all;
  additionally rejecting non-2D operands turned wrong output into NaN.
  Both reverted; the defect is documented rather than half-fixed.

## Method notes that mattered more than any single fix

1. **Gate expensive builds behind cheap probes, with the threshold
   written down first.** The cooperative-decode probe cost a day and
   saved a week — and produced the bandwidth number that redirected
   everything.
2. **Paired controls, same build, same session.** Two "regressions"
   (−1.7% prefill, −0.6 t/s) evaporated when both arms were measured in
   one run. Board state drifts; only paired comparisons survive it.
3. **Bit-identity anchors.** Every semantic change kept an environment
   flag that reproduces the previous numerics exactly, verified by
   greedy-decode diffs. Refactors then prove themselves: after the
   clipping work, perplexity reproduced to all seven digits.
4. **Beware phase-mixed profiling windows.** A "decode detector" based on
   CPU-time slope also fires during the load-time quantization burst.
   Two early readings ("set_tensor eats 10% of decode", "residual thread
   churn") were load-phase artifacts, corrected later.
5. **Check for stale processes before benchmarking.** A stuck
   `llama-cli` — busy-spinning on its interactive prompt at EOF stdin —
   burned one core through an entire evening of measurements.
6. **Read your own documentation.** Both the int4 clamp bug and the int8
   wrap were described in the project's notes before they bit.
7. **Write down the failures with their numbers.** Half of this document
   is dead ends, and that half is what stops the next person from
   spending a week on them.
