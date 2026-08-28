# Making 4-bit NPU inference actually work on the RK3588

A narrative account of a run of board sessions spent on llama.cpp's
RKNPU2 backend, written for readers who did not live through it. The two
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

(The mechanism sketched above is the one I believed at the time, and it
is wrong in its particulars — the 3-D expert weights are a red herring.
Act 12 has the real cause, which is batched *activations*. I have left
the mistaken reasoning here rather than quietly correcting it, because
the two failed fixes only make sense against it.)

## Act 9: Pricing the transform that makes 4-bit possible

With quality solved, the last unexamined cost was the Hadamard transform
itself — the reversible shuffle that spreads activation outliers so 15
quantization levels are enough. It is expensive: about 72% of all
per-row preparation work. And the project had written off the
transform-free pipeline as "numerically broken" *before* the clamp fix,
per-channel scales and clipping all landed, so the verdict was worth
re-testing.

It survived, emphatically. Without the transform, E4B goes to **PPL
26,872** against 35.05 with it. Per-channel weight scales do nothing for
the activation side, and a 0.9 activation clip cannot tame outliers
across 15 levels.

But the failed experiment priced the thing precisely: **8.7% of prefill
and 10.9% of decode.** Two corollaries fell out. The int4 matmul path
itself now runs at 41.42 against int8's ~41.4 — parity — so the
transform is the *entire* remaining gap between the two modes, which
finally closes out the oldest claim in this project, that "INT4 is 5×
slower than INT8." And transform-free W4A4 would be the fastest decode
on the board at 6.17 t/s, which is a tidy statement of what outlier
spreading costs.

Then two attempts to make it cheaper, both refused by measurement.
Smaller FWHT blocks divide K just as evenly and need fewer passes — but
they buy 4.8% speed for 58% worse perplexity, and speed saturates almost
immediately because a 512-float block is already resident in L1 cache.
That also retired a *planned* optimization: "FWHT stage-fusion" was
sitting in the backlog justified by memory traffic that the
block-diagonal change had already eliminated. Deleting a bad plan before
someone spends a week on it counts as a result.

The second attempt was subtler and taught me something. Reading the
code, Q/K/V all consume the same activations yet each runs a full
transform over them, differing only because each weight tensor carries
its own random sign vector. Nothing in the maths requires that, so
sharing the signs would let one transform serve three matmuls — about
43% less work. Measured: E4B got **43% worse**. The "redundant"
recomputation is not redundant; the differing signs keep each tensor's
quantization error independent. What looked like waste was load-bearing.

The consolation prize is real though: the same switch *improves* Qwen by
8%. It ships as an off-by-default knob, recorded as measured rather than
explained — one mechanism does not fit both directions, and two models
are no basis for a theory.

## Act 10: The other 40 seconds — vision, and the GPU

Everything so far was text. Pointing the same rig at an image was
sobering: one picture plus a 48-token answer takes ~32 seconds, and
**12.5 of those are the vision encoder running on the CPU with no
acceleration at all.** All the text work is a rounding error beside it.

The encoder is a plain ViT, so at 768×768 it is ~440 GFLOP of matmul —
and four A76 cores at ~30 GFLOPS predict ~14 s. The measurement matches;
this is honest compute, not a pathology.

The NPU barely helps. Getting it to accept the vision tower first
required fixing a genuine crash (the tensor packer decided what to pack
from dtype alone, never checking the alignment it would later assert
on — dense model dimensions happen to be aligned, so it had never
fired). After that it runs, and gains **5%**. The split counts say why:
227 graph splits against the CPU's 1, because the backend takes only 2-D
matmuls and every layernorm, GELU and softmax bounces back. Same lesson
as the text path — fragmentation dominates.

Which pointed at the GPU, and this is where the argument was structural
rather than hopeful: a GPU backend supports the *whole* op set, so it
would take all 940 nodes in one split. So I brought the stack up
properly. Vulkan turned out to be impossible — the vendor kernel binds
the Mali to ARM's proprietary driver, `panfrost` never binds, and mesa
sees only a software rasterizer. OpenCL did work: Rockchip's libmali
blob gives a genuine Mali-G610, OpenCL 3.0, fp16, subgroups, even an
int8 dot-product extension.

And then ggml rejected the device by name — its OpenCL backend
allowlists Adreno and Intel. So I wrote the port: a `MALI` family, a
single `subgroup_size` field replacing five copies of a per-vendor
branch, Mali routed through the generic kernel selection at 22 dispatch
sites, and a fallback added to 37 kernel preambles that self-detect the
vendor through extension macros Mali advertises neither of. Along the
way it exposed two real upstream portability bugs — two kernels use
`half` without enabling `cl_khr_fp16`, which Adreno and Intel compilers
forgive and Mali correctly does not.

All 47 kernels compiled. The device enumerated. And the measurements
killed it anyway: **PPL 104 against 11 on the CPU**, and **15× slower at
prefill**. Pinning the subgroup width was necessary but moved quality
only 3%, so something deeper in these Adreno-shaped kernels does not
hold on Mali. Fixing that would have bought a correct backend that is an
order of magnitude slower than doing nothing.

So the GPU is not a useful compute resource for llama.cpp on this chip —
closed with data rather than speculation, on a branch of its own so the
groundwork survives if Mali-tuned kernels ever appear.

## Act 11: Deploying it, which promptly caught two of my own errors

Setting the work up on a second board meant writing down what a fresh
install actually requires, and verifying the instructions rather than
recalling them. Two things fell out immediately.

Cloning the repo *from the first board* produced code that built fine,
passed its own tests, and was months out of date — because development
happened elsewhere and the board had only ever been kept in sync by
copying files into its working tree. Its git history had never moved.
The unit-test count now doubles as a version check for exactly this
reason: 269 means current, 170 means you have the old tree.

Then the verification failed on its own documentation. Routed prefill
came back at 41.4 where the docs promised 42.5. Not thermal — 36 °C with
clocks pinned, and it reproduced. An A/B on a single build found it:
per-channel INT8 scales, which I had documented as costing nothing, cost
**2.5% of E4B prefill.** I had measured that claim on Qwen alone, where
it genuinely is noise, and never re-measured E4B afterwards. Still a
clear trade — Qwen's 8-bit perplexity went 12.24 → 9.08 for it — but a
trade, not free, and every inherited figure had to be corrected.

A third error surfaced while writing the guidance for other models.
Checking a "measure your own model" recipe, I noticed a CPU baseline
number that looked suspiciously familiar. **`--device none -ngl 0` does
not disable this backend** — llama-bench still prints `RKNPU` in the
backend column. What I had published as the Mali port's "CPU reference"
was the NPU's own 4-bit path. The verdict there did not change, and in
fact got stronger, but the ratios were wrong and are now right.

The guidance itself is worth stating plainly, because it is the one
thing a new user will get wrong: **Q4_0 files default to the 4-bit
pipeline.** That is correct for a 7.5 B model and wrong for a small one,
and it fails silently and at full speed. Run Qwen-1.5B from a Q4_0 file
with no environment variables and perplexity doubles with nothing to
warn you.

## Act 12: Finishing Act 8's bug, and a 26% surprise I did not order

The MoE defect had been written down rather than fixed, which is a
tolerable place to leave something for a week and a bad place to leave
it permanently. Coming back to it with the diagnostic tooling built in
the meantime, the actual mechanism turned out to be neither of the
things I had guessed.

The two failed attempts had both tried to *reject* the offending
tensors. The reason that made things worse is now obvious: rejecting an
op sends it to the CPU backend, but the CPU can only compute it if a
host copy of the weights exists — and under W4A4 it does not, because
dropping the host copy is exactly where the −29% memory win comes from.
The first attempt was inert, the second produced NaN, and both were
me steering around a bug I had not located.

Locating it needed a tool. `--override-tensor` looked like the obvious
way to bisect which tensor class was responsible, and it silently does
nothing here — the RKNPU allocation is unchanged by `-ot ".*=CPU"`. I
only noticed because every variant returned perplexity identical to
seven figures, which is not a result, it is an instrument reading zero.
So I added `RKNPU_EXCLUDE`, which genuinely keeps a weight out of the
backend, and `RKNPU_DEBUG_OPS`, which logs the geometry of every matmul
the backend accepts.

With those, the answer was immediate and it was not the experts at all.
LFM2's *short-convolution projections* arrive as `src1[2048,128,4]` —
batched activations against ordinary 2-D weights. The backend computed
the first slice of four and left the rest as the zeros the memset had
written. The fix is what the bug description implies once you can see
it: loop over `ne[2] × ne[3]`, advancing the activation and destination
pointers by the real `nb[]` strides, and make `supports_op` enforce what
the loop assumes. LFM2 perplexity went from 17,402 to **16.90**, against
15.86 on the CPU, at unchanged speed. The dense models stayed
bit-identical, as a loop that runs once at zero offset must.

Then the odd part. The same change lifted E4B decode from 5.47 to
**6.89 t/s, a 26% gain**, on a model that has no batched matmuls
whatsoever — 3,087 accepted matmuls in a perplexity chunk, every one of
them `ne[2] == 1`. Identical perplexity, identical 603 graph splits,
byte-identical greedy output over 48 tokens. A provable no-op was worth
a quarter of decode throughput, which is the kind of result you should
refuse to publish until you can explain it.

Chasing it produced the most interesting systems finding of the project.
Neither micro-edit inside the change reproduces it alone. `perf stat`
showed the old build retiring 71 billion more instructions at identical
IPC — more code executed, not more waiting on memory. But disabling
every spin-wait in the system collapsed that gap from 71 G to 8 G and
the speed gap from +26% to +11%. Nearly 90% of those "extra
instructions" were threads spinning.

So the mechanism is a feedback loop rather than a hot spot. The
restructuring removes a few percent of real work from the per-node
critical path. On eight cores already hosting a libgomp team and two
dispatch-pool workers that spin before they sleep, a longer critical
path means more spin burn; spin burn steals precisely the cores that the
activation prep and dequantization need; and that lengthens the critical
path again. The measured amplification is about 2.5×. Two independent
observations confirm it: the old build spends 60% of decode samples in
libgomp spin against 46% for the new one, and widening the spin window
collapses the old build (5.52 → 2.87 t/s) while barely moving the new
one (6.87 → 6.57 across a 256× range).

That last asymmetry is the part worth keeping. The fixed build is nearly
insensitive to a tuning parameter the old one was violently sensitive
to, which is a much better place for a system to sit. It also sets the
exchange rate for future work: on this path, a millisecond saved per
node is worth about two and a half, and a millisecond lost costs the
same.

## Act 13: The 4-bit file that beat the 8-bit file

With MoE working, the obvious question was whether the mixture-of-experts
models should be quantized harder. The reasoning is simple enough to
state in one line: MoE decode is bound by the bytes of the *active*
experts, so halving the weight precision should halve those bytes.

LFM2-8B-A1B had only ever been run from its Q8_0 file. Running the same
model from LiquidAI's Q4_0 instead moved decode from 13.44 to **23.44
tokens per second**, prefill from 44.8 to 62.1, and halved the file to
4.41 GB. The 74% is short of the 2x that pure bandwidth scaling
predicts, and the shortfall is informative: it is the routing and
attention work that does not shrink when the weights do, which sets a
rough floor on what further quantization can buy.

Then the perplexity came back at **14.77 against the Q8_0's 15.86**. The
4-bit file was not merely as good as the 8-bit one, it was measurably
better, and that is not something quantization can do. A number that
good is a bug report about your own methodology until proven otherwise,
so I went looking for the mistake rather than writing it up.

There wasn't one. Both files are the official LiquidAI releases and the
Q8_0's sha256 matches Hugging Face exactly, so it is not a publisher
mismatch. The Q8_0 references I was comparing against came from earlier
sessions, so I re-ran them in the same session, and they reproduced the
recorded figures to four significant figures — 20.7758 at eight chunks,
15.8591 at thirty-two against a recorded 15.86. The effect survived
going from 8 chunks to 32.

I reached for the obvious explanation — that LiquidAI's Q4_0 is
imatrix-calibrated while the Q8_0 is plain round-to-nearest, and that
importance-matrix quantization can genuinely beat naive higher precision.
I wrote it down as inference rather than fact, which turned out to be the
right amount of hedging, because a few days later it was dead.

Neither file carries `quantize.imatrix.*` metadata; ERNIE's does, so the
absence means something. Both were uploaded in the same batch, so they
are not built from different checkpoints. And their per-tensor recipes
are identical except for a single tensor — the token embedding table,
which is Q6_K in the 4-bit file and Q8_0 in the 8-bit one. That is the
only asymmetry in the entire model, and it points the wrong way: 8.5 bits
against 6.56, so the file that scores *worse* is the one with the
higher-precision embeddings.

So the finding stands and the explanation does not. A 4-bit file beats an
8-bit file from the same checkpoint with the same recipe and no
calibration, reproducibly, at two sample sizes, and I cannot say why. It
is worth being clear that this is the honest state rather than quietly
leaving the tidy story in place, because the tidy story was wrong and
someone would have repeated it.

The practical finding underneath is sharper than the curiosity. W4A4 cost
25% perplexity on this model — and LFM2 has eight billion parameters. The
rule I had been carrying, "4-bit is free on big models and expensive on
small ones", was the wrong rule. It is not about the model, it is about
the *tensors*: in a mixture-of-experts nearly all the parameters live in
the experts, which run on the CPU, and the dense tensors the NPU actually
quantizes stay small no matter how large the model gets. An 8 B MoE
behaves, for this purpose, like a 2 B dense model.

So the two 4-bit decisions are independent, and worth separating
explicitly: take the small *file*, then force the 8-bit *pipeline*.

The rule earned its keep almost immediately. The next model tried was
ERNIE-4.5-21B-A3B, a 21-billion-parameter mixture of 64 experts, and the
question was whether the CPU parity E4B had reached was available to an
MoE at all or whether LFM2's 36% was simply what these models cost.

Reading the tensor table before downloading said the NPU would barely
participate: the expert FFNs are three-dimensional, so they belong to
`MUL_MAT_ID` and stay on the CPU forever, leaving attention, the two
shared experts per layer, the output projection and a single dense block
— 1.44 billion parameters out of 21.83, **6.6% of the model**. A small
share, but not a harmless one: the output projection sits directly on the
logits path and the shared experts fire on every token through all
twenty-eight layers.

It reached parity. W8A8 scored **8.2153 against the CPU's 8.2411**, and
because a third of a percent at eight chunks is well inside noise, I ran
it again at thirty-two: **6.0793 against 6.0876**. Both say the same
thing, which is not that the NPU is better but that it is
indistinguishable — the right claim to make and the right one to stop at.
That came with 18% more prefill throughput, a genuine win on a model
where 93% of the weights cannot be touched at all until someone
implements `MUL_MAT_ID`.

The satisfying part was the four-bit number. W4A4 cost 10.8% here,
against LFM2's 36% — and ERNIE is 2.6 times LFM2's size. Under the old
"big models are safe" rule that is backwards. Under the tensor-width
rule it is exactly right: ERNIE's shared experts are 2560×3072 where
LFM2's dense tensors are narrower, and sorting every model measured by
W4A4 damage sorts them by tensor width, from E4B's 2560×10240 at parity
down to Qwen-1.5B at +145%. The rule had been written that morning from
four models. ERNIE was measured afterwards and landed where it predicted,
which is the only kind of confirmation worth much.

Two more models tested the rule rather than illustrating it. LFM2-24B-A2B
— three times the capacity for a third more active parameters — ran at 15
tokens per second and taught a lesson that had nothing to do with
quantization: its default context of 128,000 tokens asks for a 2.5 GB KV
cache, which on top of 12.5 GB of weights exhausts the board and makes
llama.cpp return **completely empty output, no error, exit code zero**.
The only symptom is missing text. Capping the context fixes it entirely.
Its perplexity also came in nearly five times worse than the 8B's while
its answers to actual questions were indistinguishable, which is a useful
reminder that perplexity is a proxy and proxies fail.

LFM2.5-8B-A1B was the more interesting one. Same architecture, same
shape, one changed number — the vocabulary doubled from 65,536 to
128,000 — and the effects were exactly what the bandwidth model predicts:
prefill up slightly, decode down 10%, because a doubled embedding table
is per-token work on the decode path. Its W4A4 penalty came in at 21%,
which placed it between ERNIE's 11% and E2B's 26%, matching its tensor
area of 3.7 M against their 7.9 M and 3.1 M. Four models predicted
correctly now.

It is also a reasoning model, which the benchmark cannot see: it emits
`<think>` blocks before answering, so its 20.9 tokens per second buys
fewer *useful* tokens than LFM2's 23.2. And its perplexity of 28.02
against LFM2's 19.96 means nothing at all, because the tokenizers differ
— the same trap that makes every cross-model perplexity in these
documents a comparison you must not make.

The same week's Gemma-4 E2B run made the point from the other direction.
Its prefill is 3.7x E4B's and the NPU clearly earns its place there — 135
against 71 on the CPU. Its decode is 4.87 on the NPU against 13.48 on the
CPU, slower even than the model twice its size. Less compute per byte
moved, and the accelerator becomes a liability. On this board the NPU is
a prefill engine, and the smaller the model the more sharply that is
true.

## Where it ended up

Gemma-4 E4B Q4_0, RK3588, `-t 4` on the big cores:

| Config | pp128 | tg64 | PPL (32ch) | NPU memory |
|---|---|---|---|---|
| Routed (NPU prefill + CPU decode) | **41.4** | 5.50 | 27.01 | high |
| **Pure NPU W4A4 (all defaults)** | 37.0 | **6.89** | **26.88** | **2.5 GB** |
| Pure NPU W8A8 | 41.4 | 4.55 | 27.61 | high |
| Pure CPU | 25.2 | 4.9 | 27.01 | — |

And the W4A4 arc across the whole fork's history:

| | prefill | decode | PPL | load |
|---|---|---|---|---|
| Originally | 7.7 | 3.4 | 163.01 | minutes |
| **Now** | **37.0** | **6.89** | **26.88** | **seconds** |

Nearly 5× prefill, 2× decode, 6× quality, −29% memory. Every step
verified against bit-identity anchors: setting four environment
variables still reproduces the original numerics exactly, so every
change remains auditable and reversible.

Cross-model: Qwen2.5-1.5B W4A4 44.31 → 21.74 and W8A8 12.24 → 9.08.
LFM2's decode numbers, 9.47 → 13.66 t/s, were withdrawn as
quality-invalid in Act 8 and are reinstated by the Act 12 fix: the
speeds always stood, and the outputs behind them are now correct. Act 13
then takes LFM2 to **23.44 t/s** by switching to its Q4_0 file, at
better perplexity than the Q8_0 it replaces.

One caveat belongs next to those numbers rather than in a footnote: the
4-bit parity is demonstrated on one 7.5 B model and does not hold at
~2 B.

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
  Both reverted, and both were guesses at an unlocated bug. Fixed
  properly in Act 12 once `RKNPU_EXCLUDE` and `RKNPU_DEBUG_OPS` existed
  to locate it — it was batched activations, not the experts.
- **`--override-tensor` as a bisection tool** — inert for this backend;
  it leaves the RKNPU allocation untouched. Cost a round of experiments
  that all returned the same number, which is how it was caught.
- **Widening the dispatch-pool spin window** — still loses, and now
  quantified: it collapses the pre-#4c build 5.52 → 2.87 t/s.
- **Removing the Hadamard transform** — mandatory: PPL 35 → 26,872. It
  costs 8.7% prefill / 10.9% decode and is worth every bit of it.
- **Smaller FWHT blocks** — +4.8% speed for +58% perplexity; the natural
  block was already optimal.
- **FWHT stage-fusion** — a *planned* optimization, retired unbuilt: its
  memory-traffic premise vanished when blocks became L1-resident.
- **Sharing Hadamard sign vectors to reuse transforms** — +43% PPL on
  E4B. The apparent redundancy is what keeps tensor errors independent.
- **The vision encoder on the NPU** — 5%, because 940 nodes fragment
  into 227 splits.
- **The Mali GPU** — Vulkan structurally unavailable; the OpenCL port
  compiles and runs but is numerically wrong *and* 15× slower than CPU.

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
7. **Verify the baseline you are claiming, not the one you assume.**
   Two published numbers were wrong in the same way: a "CPU reference"
   that was actually the NPU (`--device none -ngl 0` does not disable
   this backend), and a "costs nothing" measured on one model and never
   rechecked on the other. Both were caught by writing a deployment
   guide and then following it.
8. **A claim measured on one model is a claim about one model.**
   Per-channel INT8 is free on Qwen and costs 2.5% on E4B; sign-sharing
   helps Qwen and ruins E4B; 4-bit is free at 7.5 B and doubles error at
   1.8 B. Nothing here generalized as reliably as it first appeared.
9. **Write down the failures with their numbers.** Well over half of
   this document is dead ends, and that half is what stops the next
   person from spending a week on them.
