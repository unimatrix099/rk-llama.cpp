# Deploying this fork on a second RK3588 board

Step-by-step setup for a fresh board, with the verification numbers to
prove it worked. Everything here was derived from the working install on
an Orange Pi 5 Ultra; the numbers in "Verify" are what that board
produces and are what a correct build should reproduce within a few
percent.

Research background lives in `RKNPU2-decode-research.md` (what was tried
and why) and `RKNPU2-optimization-notes.md` (what shipped). This document
assumes you just want it running.

## What you get

Gemma-4 E4B Q4_0, four A76 threads, pinned clocks:

| Configuration | prefill | decode | quality (wikitext PPL) | NPU memory |
|---|---|---|---|---|
| **Pure NPU W4A4 (all defaults)** | 37.0 | **6.89** | **26.88** | **2.5 GB** |
| Routed — NPU prefill, CPU decode | **41.4** | 5.50 | 27.01 (CPU-exact) | high |
| Pure NPU W8A8 | 41.4 | 4.55 | 27.61 | high |
| Pure CPU (no NPU) | 25.2 | 4.9 | 27.01 | — |

Both NPU paths now sit at CPU-parity quality; W4A4 additionally uses 29%
less NPU memory, loads in seconds, and since the batched-matmul work
(decode research #4c) is also the fastest decode path — so on E4B the
defaults are simply the right answer.

For other models see **[the model matrix](#known-good-models)** below;
E4B is the only one measured where W4A4 is free and the NPU wins both
phases.

## 0. Prerequisites

**Hardware:** any RK3588 / RK3588S board. 16 GB is recommended — an
8 GB board will run smaller models but not E4B comfortably alongside the
dual-residency copy.

**Kernel — this is the one thing you cannot work around.** You need a
**vendor BSP kernel with the RKNPU driver**, not a mainline kernel.
Verify:

```sh
ls /dev/dri/card*            # one of these must be the NPU
for c in /sys/class/drm/card*/device/driver; do
    echo "$c -> $(basename $(readlink -f $c))"
done
# expect a line ending in '-> RKNPU'

cat /sys/kernel/debug/rknpu/version   # needs root on some images
# reference board: RKNPU driver: v0.9.8
```

If no `RKNPU` node appears, stop here — the backend cannot work. Armbian
"vendor" images (e.g. `6.1.115-vendor-rk35xx`) have it; mainline and
`edge` images generally do not.

The reference board runs Armbian community 26.8.0 (Debian trixie),
kernel 6.1.115-vendor-rk35xx. Driver v0.9.8 pairs with the vendored
runtime `librknnrt 2.3.2`, which ships **inside this repo** — you do not
install it separately.

## 1. System packages

```sh
sudo apt update
sudo apt install -y build-essential cmake git python3 wget
# optional, only for the diagnostic tools and profiling:
sudo apt install -y linux-perf gdb strace
```

## 2. Get the code

```sh
git clone https://github.com/unimatrix099/rk-llama.cpp.git ~/rk-llama.cpp
cd ~/rk-llama.cpp
git checkout feat/w4a4-neon-prep
```

> The optimizations live on `feat/w4a4-neon-prep`, stacked on
> `feat/int4-native-layout` ← `feat/mixed-precision-pipelines` ←
> `fix/w4a4-calibration-crashes` ← upstream `rknpu2`.
> The `feat/opencl-mali` branch is a **documented dead end** — do not
> deploy it (see `RKNPU2-mali-opencl-port.md`).

> **Trap worth knowing about.** Do not clone from the first board unless
> you have verified its git history is current. Development happened in
> a separate environment and the board was kept in sync by copying files
> into its *working tree*, so its branch can point at a much older
> commit while the checked-out files are new. A clone from it then
> yields stale code that still builds and still passes its own tests —
> just with the old behaviour. Clone from the remote, and confirm you
> got the right code with step 6a: **269 checks means current, 170 means
> you have the pre-optimization tree.**

## 3. Build

```sh
cd ~/rk-llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_RKNPU2=ON -DLLAMA_CURL=OFF
cmake --build build -j$(nproc)
```

`-DGGML_RKNPU2=ON` is the only non-default flag that matters.
`-DLLAMA_CURL=OFF` avoids needing `libcurl4-openssl-dev`; drop it if you
want llama.cpp's built-in model downloader.

Confirm the backend was built:

```sh
ls build/bin/libggml-rknpu2.so    # must exist
build/bin/llama-cli --list-devices
# expect: RKNPU: Rockchip NPU
```

**Do not move or delete the source tree after building.** The binaries
find `librknnrt.so` through a RUNPATH pointing at
`ggml/src/ggml-rknpu2/libs/`. If you relocate things, set
`LD_LIBRARY_PATH=<repo>/ggml/src/ggml-rknpu2/libs` instead.

## 4. Models

```sh
mkdir -p ~/models && cd ~/models
wget https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_0.gguf
# optional, for multimodal (vision runs on CPU — see limitations):
wget https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/mmproj-F16.gguf \
     -O mmproj-E4B-F16.gguf
```

### Picking a pipeline for your model — read this before running

Two rules, and the second one is a trap if you skip it.

**Rule 1: match the file to the pipeline.** Use Q4_0 files with W4A4 and
Q8_0 files with W8A8. Squeezing an 8-bit file down to 4 bits at load time
costs noticeably more quality than starting from a file that is already
4-bit.

**Rule 2: only use 4-bit on big models.** Roughly 7 B and up. Below that,
use 8-bit.

The reason is simple. Quantizing to 4 bits throws away detail, and a big
model has enough redundancy to absorb the loss — a small one does not.
Same backend, same settings, measured on the same text:

| Model | quality on CPU | on 8-bit NPU | on 4-bit NPU |
|---|---|---|---|
| Gemma-4 E4B (7.5 B) | 27.01 | 27.61 | **26.88 — as good as CPU** |
| Qwen2.5-1.5B (1.8 B) | 9.36 | 9.56 | **18.73 — twice as bad** |

(Lower is better. These are perplexity scores; only compare down a
column, never across, since different models score on different scales.)

So on E4B, 4-bit is free — same quality, 29% less NPU memory. On a 1.5 B
model it roughly doubles the error, which you would notice.

**Where the trap is:** the backend picks the pipeline from the file type,
and **a Q4_0 file defaults to 4-bit**. That default is right for E4B and
wrong for a small model. If you run a small Q4_0 model with no
environment variables, you silently get the bad path — it will not warn
you, and it will still be fast.

```sh
# Small model (~2 B) in a Q4_0 file: force 8-bit, or quality halves
RKNPU_HYBRID=W8A8_STANDARD build/bin/llama-cli -m qwen2.5-1.5b-instruct-q4_0.gguf ...

# Small model in a Q8_0 file: the default is already 8-bit, nothing to do
build/bin/llama-cli -m qwen2.5-1.5b-instruct-q8_0.gguf ...

# Big model (7 B+) in a Q4_0 file: the default is already right
build/bin/llama-cli -m gemma-4-E4B-it-Q4_0.gguf ...
```

If you must run 4-bit on a small model anyway — to fit it in NPU memory —
`RKNPU_SHARED_SIGNS=1` recovers about 8% of the loss on Qwen-class
models. It makes E4B considerably worse, so only use it on small models,
and measure.

**It is not really about model size — it is about tensor size.** Six
models now show the same trap, and sorting them by the *area of the
tensors the NPU quantizes* sorts them by W4A4 damage — parameter count
does not:

| Model | NPU-quantized shape | area | W4A4 | W8A8 | CPU | W4A4 vs CPU |
|---|---|---|---|---|---|---|
| Gemma-4 E4B Q4_0 (7.5 B dense) | 2560 × 10240 | 26.2 M | 26.88 | 27.61 | 27.01 | **~parity** |
| ERNIE-4.5-21B-A3B Q4_0 (MoE) | 2560 × 3072 | 7.9 M | 6.78 | 6.08 | 6.09 | +11% |
| LFM2.5-8B-A1B Q4_0 (MoE) | 2048 × 1792 | 3.7 M | 33.90 | 27.96 | 28.02 | +21% |
| Gemma-4 E2B Q4_0 (QAT, ~2 B) | 1536 × ~2048 | 3.1 M | 74.75 | 60.26 | 59.35 | +26% |
| LFM2-8B-A1B Q4_0 (MoE, 8 B) | 2048 × 1792 | 3.7 M | 25.30 | 20.30 | 18.62 | +36% |
| Qwen2.5-1.5B Q4_0 | small | — | 21.74 | 9.08 | 8.88 | +145% |

(Compare only across a row, never down a column — different models score
on different scales, and several of these use different tokenizers.
Sample sizes differ too: ERNIE and LFM2.5 are 32/16-chunk, the rest are
as originally measured.)

The ordering is monotonic in tensor area except for the two LFM2 rows,
which share a shape but were measured at different chunk counts and
contexts (LFM2 at 8 chunks/default ctx, LFM2.5 at 16 chunks/`-c 512`) —
so treat +21% and +36% as "both well off parity at ~3.7 M" rather than
as a real ordering between them.

ERNIE is 2.6x LFM2's total size yet takes a third of the W4A4 damage,
because its shared-expert tensors are 2560×3072 while LFM2's dense
tensors are narrower. In a mixture-of-experts almost all the parameters
sit in the experts — which run on the CPU — so the model's headline size
tells you nothing about the tensors the NPU actually sees. On ERNIE that
is 1.44 B of 21.83 B parameters, 6.6%.

**So parameter count is a guess; measure.** For any MoE in a Q4_0 file,
start with `RKNPU_HYBRID=W8A8_STANDARD`. On ERNIE that is free in both
directions — W8A8 reaches CPU parity (6.0793 vs 6.0876 at 32 chunks)
*and* prefills slightly faster than W4A4 (25.58 vs 25.37), so the Q4_0
default is strictly worse on both axes.

(ERNIE's row is 32-chunk; the others are the sample size each was
originally measured at. Compare the W4A4-vs-CPU ratio, not the raw
numbers.)

**Prefer Q4_0 files for MoE models.** On LFM2, moving from the Q8_0 to
the Q4_0 release gained 74% decode, 39% prefill, half the disk, *and*
better perplexity (14.77 vs 15.86 at 32 chunks). Why the 4-bit file
scores better is **unexplained** — neither file is imatrix-calibrated and
both came from the same upload batch, so the obvious explanations are
ruled out (decode research #4d). The measurement is reproducible; the
mechanism is not known. The 4-bit *file* and the 4-bit *NPU pipeline* are
independent choices: take the small file, then force W8A8.

**Rule 2b: cap the context on large models, or you get silent nonsense.**
Some models declare a huge default context — LFM2-24B-A2B asks for
`n_ctx = 128000`, which allocates a **2500 MiB KV cache**. On top of a
12.5 GB model that exhausts this board's 15 GB, and llama.cpp then
produces **completely empty output with no error and exit code 0**. The
only symptom is missing text.

```sh
# 12 GB+ model: always pass -c, or generation silently returns nothing
build/bin/llama-completion -m LFM2-24B-A2B-Q4_0.gguf -c 4096 -p "..." -n 40
```

Budget roughly: model file + KV cache + ~1 GB must stay under 14 GB. If a
big model produces no output, suspect the KV cache before anything else.

<a id="known-good-models"></a>
### Known-good models and their measured best configuration

Everything below was measured on this board at `-t 4` on the A76 cores,
pp128/tg64, `llama-bench -r 3`. **Copy the command; it is the config that
won.** Full data and reasoning in decode research #4d–#4h.

| Model | Best command prefix | pp128 | tg64 | Quality vs its own CPU |
|---|---|---|---|---|
| **Gemma-4 E4B** Q4_0 (7.5 B dense, multimodal) | *(none — defaults)* | 37.0 | 6.89 | **parity** (26.88 / 27.01) |
| **LFM2-8B-A1B** Q4_0 (MoE 8.3 B / 1.5 B) | `RKNPU_CPU_DECODE=999999` | 55.1 | **23.4** | best on CPU (14.77) |
| **LFM2.5-8B-A1B** Q4_0 (MoE, reasoning) | `RKNPU_HYBRID=W8A8_STANDARD` | **63.7** | 18.0 | **parity** (27.96 / 28.02) |
| **ERNIE-4.5-21B-A3B** Q4_0 (MoE 21.8 B / 3 B) | `RKNPU_HYBRID=W8A8_STANDARD` | 25.6 | 7.4 | **parity** (6.079 / 6.088) |
| **Gemma-4 E2B** QAT Q4_0 (~2 B dense) | `RKNPU_HYBRID=W8A8_STANDARD RKNPU_CPU_DECODE=32` | **135.0** | 11.0 | −1.5% (60.26 / 59.35) |
| **Qwen2.5-1.5B** Q8_0 | `RKNPU_HYBRID=W8A8_STANDARD RKNPU_CPU_DECODE=32`, pinned `taskset -c 4-7` | **280.8** | 13.5 | −2.3% (9.08 / 8.88) |
| LFM2-24B-A2B Q4_0 (MoE 24 B / 2 B) | `RKNPU_CPU_DECODE=999999 -c 4096` | 32.7 | 15.0 | parity, but see note |

Pick by what you actually need:

| Goal | Model | Why |
|---|---|---|
| Fastest decode | LFM2-8B-A1B, pure CPU | 23.4 t/s; nothing measured beats it |
| Fastest prefill | Qwen2.5-1.5B / E2B | 280 / 135 — the NPU's strongest showing |
| Multimodal | Gemma-4 E4B + mmproj | only vision option; vision runs on CPU |
| Reasoning | LFM2.5-8B-A1B, W8A8 | emits `<think>`, so usable t/s is below 18.0 |
| Most capacity at parity quality | ERNIE-4.5-21B-A3B, W8A8 | 21.8 B at CPU-exact quality |
| Best 4-bit behaviour | Gemma-4 E4B | the only model where W4A4 is free |

Notes that will save you a day:

- **Only E4B should run the W4A4 default.** Every other model here needs
  `RKNPU_HYBRID=W8A8_STANDARD`; the Q4_0 file selects W4A4 silently and
  it costs 11–145% quality depending on tensor size (table above).
- **On MoE models the NPU only helps prefill.** Expert weights are 3-D
  and `MUL_MAT_ID` is unimplemented, so they always run on the CPU — on
  ERNIE only 6.6% of parameters ever reach the NPU. Pure CPU is often the
  best decode config for these.
- **LFM2-24B-A2B is not recommended** despite working: slower than the
  8 B (15.0 vs 23.4), three times the RAM, and identical answers on every
  task tried. Revisit only with a task-relevant eval.
- **Do not compare perplexity between rows.** Different models, several
  with different tokenizers. Each "quality" figure is only against that
  model's own CPU baseline.
- **The unsloth `gemma-4-E2B-it-Q4_0.gguf` does not load** — a malformed
  export, not a bad download (its sha256 matches). Use Google's QAT build
  (`google/gemma-4-E2B-it-qat-q4_0-gguf`).

**When in doubt, measure your own model** with step 6c below: run
perplexity once with your settings and once with
`RKNPU_HYBRID=W8A8_STANDARD RKNPU_CPU_DECODE=999999` (everything on the
CPU, which is the quality reference). If the two are close, your
configuration is fine. The threshold is not exactly 7 B — it depends on
the model — so the measurement beats the rule of thumb.

## 5. System tuning (per boot)

Two things, both required for reproducible performance:

```sh
ulimit -n 65536      # librknnrt opens many fds; the default 1024 fails
```

and pin the clocks — otherwise DVFS makes every measurement noise:

```sh
git clone https://github.com/airockchip/rknn-llm.git ~/rknn-llm
sudo bash ~/rknn-llm/scripts/fix_freq_rk3588.sh

# verify (expect ~1.8 GHz little, ~2.35 GHz big, 1 GHz NPU):
cat /sys/devices/system/cpu/cpufreq/policy{0,4,6}/scaling_cur_freq
cat /sys/class/devfreq/fdab0000.npu/cur_freq
```

To make `ulimit` permanent, add `* soft nofile 65536` /
`* hard nofile 65536` to `/etc/security/limits.conf`.

## 6. Verify the install

**a. Unit tests** — pure CPU, no NPU needed. Catches a bad toolchain
*and* doubles as a version check:

```sh
cd docs/backend && make -f Makefile.rknpu2-tools check-prep && cd ../..
# expect: PASS: 269 checks, 0 failures
#   269 = current tree
#   170 = you are on the pre-optimization code (see the trap in step 2)
```

**b. Speed** — the headline configurations:

```sh
ulimit -n 65536
# routed (best prefill, CPU-exact decode)
RKNPU_HYBRID=W8A8_STANDARD RKNPU_CPU_DECODE=32 taskset -c 4-7 \
  build/bin/llama-bench -m ~/models/gemma-4-E4B-it-Q4_0.gguf -p 128 -n 64 -r 3 -t 4
# expect ~pp 41.4 / tg 5.50

# pure NPU 4-bit (lowest memory, best decode — the recommended default)
build/bin/llama-bench -m ~/models/gemma-4-E4B-it-Q4_0.gguf -p 128 -n 64 -r 3 -t 4
# expect ~pp 37.0 / tg 6.89
```

If decode comes back near 5.5 rather than 6.9 on the second command, you
are on a build from before the batched-matmul work (decode research #4c)
— that change is worth +26% here.

**c. Quality** — the check that actually proves correctness. A build
that is fast but wrong will pass (b) and fail this:

```sh
wget https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip -O /tmp/wt.zip
python3 -c "import zipfile,os; zipfile.ZipFile('/tmp/wt.zip').extractall(os.path.expanduser('~'))"

build/bin/llama-perplexity -m ~/models/gemma-4-E4B-it-Q4_0.gguf \
  -f ~/wikitext-2-raw/wiki.test.raw --chunks 32 -t 4
# expect PPL ~26.9  (CPU reference is 27.01 — parity is the point)
```

If PPL comes out in the hundreds or thousands, something is wrong with
the build; do not use it. See Troubleshooting.

## 7. How to run it

Pick a configuration:

```sh
# A. Best all-round: NPU prefill + CPU-exact decode
RKNPU_HYBRID=W8A8_STANDARD RKNPU_CPU_DECODE=32 taskset -c 4-7 \
  build/bin/llama-cli -m ~/models/gemma-4-E4B-it-Q4_0.gguf -t 4 -p "..." -no-cnv

# B. Lowest NPU memory, CPU left free (same decode speed, ~13% less prefill)
build/bin/llama-cli -m ~/models/gemma-4-E4B-it-Q4_0.gguf -t 4 -p "..." -no-cnv

# C. Server
RKNPU_HYBRID=W8A8_STANDARD RKNPU_CPU_DECODE=32 \
  build/bin/llama-server -m ~/models/gemma-4-E4B-it-Q4_0.gguf \
  --threads 4 --threads-batch 8 --jinja
```

**Threading is the single biggest free win: use `-t 4`.** Four A76
threads beat all eight cores by 19–66% on decode, because the A55s on
the critical path drag the cluster down. Pin with `taskset -c 4-7` for
smaller models. Servers want `--threads 4 --threads-batch 8` (decode
prefers 4 big cores, prefill prep still benefits from 8).

### Environment variables

Every default is already the measured-best value; you only need these to
deviate or to debug.

| Variable | Default | Meaning |
|---|---|---|
| `RKNPU_HYBRID` | per-dtype | Force a pipeline, e.g. `W8A8_STANDARD`. Accepts a comma list applied cyclically. |
| `RKNPU_CPU_DECODE` | 0 (off) | Route matmuls with M below this to the CPU. `32` enables the routed mode; `999999` puts everything on the CPU. |
| `RKNPU_CORES` | `0,1,2` | Which NPU cores to use. |
| `RKNPU_AC_NATIVE` | 1 | Native A/C tiling for INT4. `0` restores the slow NORM layout. |
| `RKNPU_HADAMARD_BLOCK` | 1 | Block-diagonal FWHT. `0` = legacy padded. Measured optimal at 1. |
| `RKNPU_PER_CHANNEL` | 1 | Per-output-channel weight scales (INT4 and INT8). |
| `RKNPU_A_CLIP` | 0.9 | INT4 activation scale clip. |
| `RKNPU_B_CLIP` | 0.93 | INT4 weight scale clip. |
| `RKNPU_EXCLUDE` | unset | Diagnostic. Comma-separated name substrings kept off the NPU, e.g. `RKNPU_EXCLUDE=shortconv`. Use it to bisect a wrong-output bug to a tensor class. |
| `RKNPU_DEBUG_OPS` | unset | Diagnostic. Logs the shape of every mul_mat the NPU accepts. |
| `RKNPU_SHARED_SIGNS` | 0 | Share Hadamard signs across tensors with equal K. Model-dependent: helps Qwen (−8% PPL), hurts E4B (+43%). |

Regression anchor: `RKNPU_HADAMARD_BLOCK=0 RKNPU_PER_CHANNEL=0
RKNPU_A_CLIP=1.0 RKNPU_B_CLIP=1.0` reproduces the pre-optimization
numerics bit-exactly. Useful for bisecting if you suspect a quality
regression.

## 8. Known limitations

- ~~MoE models produce wrong output on the NPU.~~ **Fixed 2026-08-24**
  (decode research #4c): the backend computed only the first slice of a
  batched mul_mat, which is how LFM2's short-convolution projections are
  shaped. LFM2-8B-A1B now measures PPL 16.90 against 15.86 on CPU, at
  unchanged speed. If you are on an older commit, run MoE models with
  `RKNPU_CPU_DECODE=999999`.
- **Multimodal vision runs on the CPU**, ~12.5 s per image for E4B, and
  that dominates any image interaction. Pointing the NPU at it gains
  only 5%. The GPU cannot help either — see
  `RKNPU2-mali-opencl-port.md`. Multimodal also needs `--jinja`.
- **`--no-mmap` breaks the NPU path** — it bypasses `set_tensor` for
  host-visible buffers, so the packed NPU copy is never built. Use the
  default mmap loading.
- **W4A4 quality depends on model size, and the default can pick it for
  you.** It reaches CPU parity on a 7.5 B dense model and roughly doubles
  perplexity on a ~2 B one — and since Q4_0 files default to W4A4, a
  small Q4_0 model silently takes the bad path. See "Picking a pipeline
  for your model" in step 4; force `RKNPU_HYBRID=W8A8_STANDARD` below
  ~7 B.

## 9. Troubleshooting

**`Too many open files` / driver errors at load** — you forgot
`ulimit -n 65536`.

**Wildly wrong output, PPL in the hundreds** — check the model is not
MoE (see limitations), then bisect with the regression anchor above.

**Benchmarks noisy or slower than the table** — check the clocks are
pinned, and check for stale processes first:
`pgrep -af llama`. A stuck `llama-cli` busy-spins on its interactive
prompt at EOF stdin and will silently eat a core; use
`llama-bench`/`llama-server` for automation, and never `taskset` a
two-model process (e.g. speculative decoding) onto 4 cores — two
spinning threadpools oversubscribe and cost 2.6x.

**`GGML_ASSERT ... k_segment % k_align` at load** — a tensor whose dims
are not NPU-alignable reached the packer. Fixed on this branch; if you
see it, you are on an older commit.

**Rebuilding after pulling** — the NPU backend alone:
`cmake --build build --target ggml-rknpu2 -j8`.

## 10. Diagnostic tools

Built from `docs/backend/Makefile.rknpu2-tools`:

```sh
cd docs/backend
make -f Makefile.rknpu2-tools all          # probes and benchmarks
make -f Makefile.rknpu2-tools check-prep   # prep-kernel unit tests
make -f Makefile.rknpu2-tools check-hw     # on-NPU equivalence test
```

`rknpu2-driver-shim.so` (LD_PRELOAD, wall-clocks every librknnrt call)
and `rknpu2-coop-decode-probe` are the two most useful for performance
work; `rknpu2-gguf-census.py` reports per-tensor dims and padding
inflation for a model.
