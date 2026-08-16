# RKNPU2: improving 4-bit token-generation (decode) speed — research notes

Research thread with **measured verdicts as of 2026-08-11** (board
sessions: probe + sweeps + perf/strace/gdb profiling; tooling under
"Experiment tooling" below). Summary: #5 threading is a shipped big win
(+19–66% tg, no code); #1 speculative and #2 cooperative decode are
measured dead ends; **#3 found its root cause and shipped the fix** —
the backend's per-node OMP regions made libgomp respawn its thread team
~once per graph split; the backend now runs the M=1 path without OMP
teams (persistent dispatch pool + `if(M>1)`), making the decode paths
+18–28% faster with no env vars. Companion documents:
`RKNPU2-optimization-notes.md` (shipped work + dead ends),
`RKNPU2-int4-research.md` (INT4 fundamentals), `RKNPU2-neon-prep-plan.md`
(prefill prep). Numbers: Orange Pi 5 Ultra (RK3588, 16 GB), pinned clocks,
`llama-bench -p 128 -n 64` unless noted.

## Where decode stands after the shipped work

Decode re-reads the active weight set for every generated token, so it is
memory-bandwidth bound. For Gemma-4-E4B Q4_0 (7.46 B dense):

| Decode path | tg64 t/s | effective read rate |
|---|---|---|
| CPU (`RKNPU_CPU_DECODE=32`, native Q4_0 reads, ~2.4 GB/token) | ~~5.35~~ **5.50** (t=4 + OMP fix; pp 42.62) | ~~13~~ **~25 GB/s at t=4 big-core** (13 was an 8-thread artifact — see #5) |
| NPU W4A4 native layout (~2.05 GB/token INT4) | 3.65 (W8A8: 4.55 with t=4 + OMP fix) | ~7.5 GB/s aggregate |

Two framing "facts" — **both revised by the 2026-08-10 measurements**:

1. ~~The CPU path is already at its wall.~~ It was a threading wall, not a
   bus wall: 4 big-core threads reach ~25 GB/s effective (avenue #5).
   With Q4_0's 2.4 GB/token that is ~8-9 t/s of headroom for E4B if the
   remaining CPU graph time shrinks; qwen-class models already do 13.5.
2. **The NPU path's overhead is NOT driver dispatch.** The coop probe
   measured the backend's exact per-node M=1 driver sequence at ~28.6
   GB/s aggregate (~14 t/s equivalent for E4B) — the loss to 3.65 t/s
   happens in the backend/CPU side around the matmuls (avenue #2 findings,
   avenue #3 is now the main code lever).

## Avenues, ranked

### 1. Speculative decoding — MEASURED: every variant loses (2026-08-10)

The hypothesis assumed near-free batch verification. That holds on GPUs;
here it fails structurally, and no variant beat its no-spec baseline
(qwen2.5-1.5B routed, t=4, seed 42, 128 tokens):

| Variant | t/s | acceptance | baseline |
|---|---|---|---|
| draft model (0.5B, draft-max 8) | 4.30* | 41.7% | 13.52 (bench) |
| ngram-simple, repetitive rewrite (clean board, OMP fix) | **13.80** | 43% (18/42) | **13.81** (server) |
| ngram-simple + verify on NPU (`RKNPU_CPU_DECODE=16` or `8`) | 11.62* | 60% (48 drafted) | 12.15* |
| any ngram type, creative prompt | 12.17–12.20* | no drafts trigger | 12.15* |
| E4B ngram-simple, repetitive | 3.64* | 2/16 | 4.17* (server) |

*Measured while a stuck background process burned one core (found and
killed 2026-08-11); relative comparisons shared that handicap, so the
verdicts hold. The headline pair was re-verified on a clean board with
`OMP_NUM_THREADS=4`: ngram-simple is exactly **neutral** (13.80 vs
13.81), not the loss the dirty numbers suggested — the drafts that
trigger pay for their verification and no more.

Why each fails:
- **Draft model**: qwen-0.5B is 1/3 of the target's bytes — even at 100%
  acceptance it roughly breaks even, and real acceptance was 42%. The
  E4B+E2B pair was skipped on arithmetic: E2B is 68% of E4B's bytes.
  There is no smaller same-family draft, so this is closed, not tunable.
- **CPU verification**: an M=9–17 batch costs 2–3x a single token on 4
  big cores (compute-bound above M=1), eating the accepted-token savings.
- **NPU verification** (threshold trick, batches to NPU at prefill speed):
  acceptance improved, but every verify batch pays the full multi-split
  graph overhead (~338 splits at bs>1 on qwen), and drafts trigger on
  only ~1/3 of tokens even on a deliberately repetitive rewrite task.
  Identical results at threshold 16 and 8 confirm verify cost is no
  longer the binding constraint — draft trigger rate is.

Operational pitfalls discovered (baked into the scripts):
- `llama-cli` busy-spins on its interactive prompt at EOF stdin — one
  core at 100% forever. Use llama-bench / llama-server for automation.
- Never `taskset` a two-model process onto 4 cores: two spinning ggml
  threadpools oversubscribe and cost 2.6x (independent of OMP settings).

Could be revisited if: a ~150M same-tokenizer draft appears for a target
worth running, or graph-split overhead at bs>1 drops enough (avenue #3)
to make NPU verification cheap.

### 2. Cooperative CPU+NPU decode — MEASURED: NO-GO (probe, 2026-08-10)

`rknpu2-coop-decode-probe` ran on the board (pinned clocks). Bandwidths do
sum, but the margin at decode-representative shapes fails the >= ~25% gate:

| Shape K x N | CPU solo | NPU solo | best coop (f_npu) | margin |
|---|---|---|---|---|
| 2048 x 2048 | 0.123 ms | 0.123 ms | 0.095 ms (0.375-0.5) | **+29%** |
| 2048 x 16384 | 0.986 | 0.587 | 0.560 (0.875) | +4.7% |
| 8192 x 2048 | 0.510 | 0.325 | 0.297 (0.875) | +9.3% |
| 4096 x 4096 | 0.483 | 0.315 | 0.294 (0.875) | +7.3% |

Only the small square shape clears the gate, and small nodes are a minor
share of per-token time. Aggregate peaks ~29-30 GB/s vs 26-28 NPU-solo:
real but not enough. **Verdict: not worth the backend complexity.**

The probe's real payoff is what it falsified: the NPU's per-node M=1
driver sequence (A set/sync + 3 parallel runs + 3 C syncs, weights beyond
cache) sustains **~28.6 GB/s aggregate** — pure weight reads for E4B would
be ~72 ms/token (~14 t/s), yet the backend decodes at 3.65 t/s (274 ms).
So the "~75% overhead" is NOT driver dispatch: it is backend/CPU-side
work between and around the matmuls (scheduler splits and copies, cache
lookups under mutex, thread oversubscription — the affinity sweep's t=4
gains point the same direction). **This re-ranks avenue #3 (overhead
surgery) from deprioritized to the main code lever**, starting with a
profile of a real decode step (perf record on llama-bench tg).

<details><summary>Original hypothesis (kept for context)</summary>

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
- The 2-node prototype is built: `rknpu2-coop-decode-probe.c` (see
  "Prepared experiments"). Go/no-go gate: best coop split must beat the
  better solo path by >= ~25% in the probe, since the probe already pays
  the real per-node costs (persistent-pool join, per-node A set/sync +
  3 runs + 3 C syncs) but not the A-prep/dequant/graph-op CPU contention
  a real integration adds.

</details>

### 3. Backend overhead surgery — PROFILED: root cause found (2026-08-11)

The profiling chain (perf cycles → strace -c → gdb breakpoint backtraces,
all on verified decode steady state) attributed the NPU-decode overhead:

- perf cycles: ~95% of on-CPU samples in libgomp spin-wait; the CPU does
  almost no real work during NPU decode. Driver `ioctl`s: ~3% of wall
  time (~75k calls/30s at 13 us) — the NPU driver is NOT the bottleneck,
  confirming the coop-probe finding.
- strace -c: ~33k `clone3` in 30 s of decode — **~550 thread creations
  per token, ~1 per graph split** (E4B: 603 splits/token at bs=1
  NPU-decode; the routed path has 1 split at bs=1, which is why it never
  suffered).
- gdb on `pthread_create`: every spawn is
  `GOMP_parallel <- ggml_backend_rknpu_graph_compute`, with libgomp
  tearing down and re-creating its team around the backend's per-node
  OMP regions. The backend's regions request heterogeneous team sizes
  (default 8 for A-prep/C-dequant, `num_threads(3)` for the matmul runs)
  while ggml-cpu's regions use `-t` (4); libgomp responds by respawning
  workers region-to-region.

**Env-level fix, verified (E4B, W8A8, t=4, clean board):**
`OMP_NUM_THREADS=4` alone lifts NPU decode tg128 3.00 -> 4.27 (+42%).
Combined best config `taskset -c 4-7` + `-t 4` + `OMP_NUM_THREADS=4`:

| E4B config | pp128 | tg64 |
|---|---|---|
| NPU decode, pinned big-4 | 42.01 | 4.55 |
| routed, pinned big-4 | **42.62 ± 0.02** | **5.50** |

This also **resolves the "strict-taskset anomaly"** from the affinity
sweep (was pp 8.75 ± 2.10, now 42.62 ± 0.02): under `taskset -c 4-7`,
libgomp's default team is 4, but without `OMP_NUM_THREADS` the 8-thread
regions still thrash. Pinning without the env var is what was
pathological.

**Code fix, implemented and shipped (2026-08-11):** the backend no longer
forms OMP teams on the M=1 path — `if(M > 1)` on the A-prep and C-dequant
regions (a single row is a few us of NEON, serial), and the
`num_threads(3)` dispatch region replaced by a persistent 2-worker pool
(`rknpu_dispatch_pool`: caller runs segment 0, spin-then-sleep workers run
the rest). clone3 during decode: 33,030 -> 910 per 30 s window (-97%).
Greedy outputs bit-identical (W8A8 and W4A4), 170 prep-kernel checks
green. Measured (E4B, t=4, no env vars):

| Config | before | after |
|---|---|---|
| NPU decode tg128 | 3.00 | **3.84** (+28%) |
| NPU decode tg128, pinned | ~3.4 | **3.94** |
| W4A4 default tg | 3.65 | **4.31** (+18%) |
| routed pinned pp128/tg64 | needed env: 42.62/5.50 | **42.52/5.49 env-free** |
| W4A4 default pp128 | 34.44 | 33.39 (-3%) |

Honest trade-offs, both measured:
- The old code WITH `OMP_NUM_THREADS=4` still beats the pool on the pure
  NPU-decode path (4.27-4.55 vs 3.84-3.97): GOMP's team barrier dispatches
  onto threads that just finished the prep region and are still hot,
  where the pool pays a ~0.1 ms futex wake per node. The same latency
  explains the -3% on W4A4 prefill. Structural fix if NPU decode ever
  matters: have each worker do its own segment's set_io so dispatch
  overlaps the driver calls. Low priority — routed decode (faster on
  every model tested) is unaffected and at its best with no env at all.
- Spinning harder does NOT fix it (tried): longer spins + pre-waking
  workers before the job is ready steal the cores the prep regions and
  CPU graph ops need (routed pp 42.5 -> 37.9, pinned NPU tg halved).
- ~~Residual ~50 clone3/s remains~~ CORRECTED by the clean re-profile
  below: decode-window clone3 is exactly **zero**; the residual was
  load-phase spillover into the measurement window.

#### Re-profile of the churn-free path (2026-08-13): both decode paths are near their floors

Wall-time decomposition via an LD_PRELOAD shim timing every librknnrt
entry point (`rknpu2-driver-shim.c`; cycle sampling cannot see blocked
time), plus decode-only perf/strace windows. E4B, W8A8, t=4.

**NPU decode (3.90 t/s = 256 ms/token):**

| Component | ms/token | share |
|---|---|---|
| `rknn_matmul_run` wall (3 cores concurrent; 1,277 calls at 0.374 ms) | ~159 | 62% |
| `set_io_mem` (2,554 calls) + `mem_sync` (1,704 calls) | ~5 | 2% |
| sequential CPU graph phase | ~92 | 36% |

The 159 ms is 4.1 GB at ~26 GB/s — within 10% of the coop probe's
28.6 GB/s driver ceiling, so the NPU part is essentially at its floor.
The 92 ms CPU phase is real model compute (perf: bf16/f16 attention dots,
GLU, tanh — the AltUp/LAUREL/activation ops), sequentially dependent on
the matmuls, not overlappable within a single stream. ~53% of on-CPU
samples are idle spin (ggml's team barrier while the NPU runs + the
dispatch pool between nodes) — cosmetic, those cores have nothing else
to do. Zero thread creation, futex+ioctl only.

**Routed decode (5.50 t/s = 182 ms/token):** ~92% of cycles are
productive memory-bound kernels (66% Q4_0 weight GEMV, 16% Q8 dot, 8%
bf16, 1.6% f16). No software waste left; purely bandwidth-bound.

**Implications:**
- QKV/gate-up fusion is now bounded: it can only attack the ~5 ms
  driver-misc slice plus per-node fixed costs inside the 159 ms — no
  longer a 40% lever. Deprioritized again, this time with numbers.
- The remaining engineering frontier is BYTES. Concretely: W4A4 NPU
  decode reads ~2.05 GB (floor ~79 ms) yet measures 4.31 t/s
  (232 ms/token) — its CPU-side is ~153 ms vs W8A8's 92 ms. **Closing
  that ~60 ms gap (Hadamard prep + INT16 dequant remnants on the M=1
  path) would put W4A4 decode at ~5.8 t/s — above routed — while
  halving NPU memory.** That is the next investigation.
- Measurement pitfalls fixed along the way: the "decode detector"
  (utime slope + thread count) also fires during load's OMP
  quantization burst, which had contaminated two earlier profile
  windows (the "set_tensor eats 10% of decode" reading and the
  "residual clone3" were both load-tail artifacts — set_tensor at
  decode is zero). Decode windows are now taken at a fixed offset
  validated against the shim's run-call timeline.

Cheap bs>1 verification reviving ngram speculation for servers remains
open (see #1).

### 3b. The W4A4 K-padding find — MEASURED AND FIXED (2026-08-16)

Follow-up to the #3 re-profile's "W4A4 has ~60 ms/token more CPU work"
hypothesis — which turned out to be wrong in an interesting way. The
driver shim on W4A4 decode (4.17 t/s = 240 ms/token) showed the CPU
phase is the SAME as W8A8 (~97 ms); the real problem was on the NPU
side: run wall 137 ms where halved bytes predicted ~80.

Root cause: the Hadamard pipelines padded K to the next power of two
(one full-length FWHT needs pow2). Gemma-4 E4B's dims are NOT powers of
two, and a GGUF census (`rknpu2-gguf-census.py`, headers only) showed the
damage: **216 tensors at K=2560 -> 4096 and the FFN downs at
10240 -> 16384, both 1.60x; in total 5.22 GB stored/read per token
instead of the nominal 3.38 GB (1.54x)**. The padding silently ate most
of INT4's byte advantage — in memory as well as bandwidth.

Fix: **block-diagonal FWHT** (QuaRot-style). Block = largest power of
two dividing K (`K & -K`: 2560 -> 512, 10240 -> 2048, 10752 -> 512;
power-of-two K keeps block = K, bit-identical). K_op == K everywhere: no
padding stored, uploaded, read, or transformed. Dequant divisor becomes
the block length (H*H^T = B*I per block). Blocks need not align with
K-segments — segments only split the summation. `RKNPU_HADAMARD_BLOCK`
selects the mode: 1 = pure block-diagonal (default), 0 = legacy padded
(bit-identical to the pre-change build, verified by greedy diff), pow2 n
= minimum block with padding to a block multiple (measured, rejected).

Measured (E4B Q4_0, default W4A4_HADAMARD, t=4, wikitext-2 8 chunks;
W8A8 reference PPL 37.78 — W4A4 is a capacity mode, heavily degraded in
every variant):

| W4A4 mode | PPL | tg64 | pp128 | NPU buffer |
|---|---|---|---|---|
| legacy (pad to next pow2) | 198.4 | 4.17-4.31 | 33.4-34.4 | 3569 MiB |
| **pure block-diagonal (new default)** | 228.9 (+15%) | **5.51** (+28%) | **37.98** (+11%) | **2521 MiB (-29%)** |
| min-block 1024 (pad to block multiple) | 280.4 | 5.13 | 37.39 | 2765 MiB |

- **W4A4 decode now equals the routed path (5.51 vs 5.50) as a pure-NPU
  mode at -29% NPU memory**, with the CPU left mostly free — exactly
  what the capacity mode is for. Prefill 7.7 -> 38.0 cumulative (4.9x).
- The quality ordering is instructive: full-row padded (198) beats pure
  512-blocks (229) beats mixed half-empty 1024-blocks (280). Spreading
  outliers over MORE lanes helps; a block that is half zero-padding
  hurts worse than a smaller full block — so the intermediate mode is
  dominated and kept only for experiments.
- Quality-sensitive W4A4 users set `RKNPU_HADAMARD_BLOCK=0` (legacy
  speed/memory, old numerics). 203 prep-kernel checks green (blocked
  exactness vs per-block scalar reference, involution, helper
  semantics, both-mode degenerate cases).

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
  → 13.66 t/s measured (routed, t=4). Model choice > engineering.

### 5. Micro-tuning — MEASURED: far bigger than expected (sweep, 2026-08-10)

`rknpu2-affinity-sweep.sh` (trimmed variant), pinned clocks, llama-bench
pp128/tg64 r=3. **`-t 4` on the A76 big cores is a free +19-51% on decode
for every model tested** — the previous "±10%" guess was a big
underestimate, and it moves every baseline in these docs:

| Model / config | tg64 all-8 (old default) | tg64 t=4 big cores | gain |
|---|---|---|---|
| Qwen2.5-1.5B Q8, routed | 8.93 | **13.52** (pinned 4-7) | +51% |
| E4B Q4_0, W8A8+routed | 4.64 | **5.50** (t=4 floating) | +19% |
| E4B Q4_0, W8A8 NPU decode | 3.43 | **4.54** (pinned) | +32% |
| LFM2-8B-A1B NPU decode | 9.47 | **13.25** (pinned) | +40% |
| LFM2-8B-A1B routed | 8.22 | **13.66** (pinned) | +66% |

- The old "CPU decode wall" of ~13 GB/s was an 8-thread artifact: little
  cores on the critical path drag the A76 cluster down. 4 big-core
  threads reach ~25 GB/s effective on the same models.
- **LFM2's routing regression flips at t=4**: routed 13.66 > NPU 13.25,
  so `RKNPU_CPU_DECODE=32` stops being model-conditional once threading
  is right (revalidate per model, but the known counterexample is gone).
- Qwen prefill also gains from pinning: pp128 215 -> 280 t/s.
- NPU decode gains too (+32-40%). Root cause found the next day: under
  `taskset -c 4-7` libgomp defaults to 4 threads, which damps the OMP
  team-respawn churn diagnosed in avenue #3 — much of the "affinity"
  gain on NPU-heavy configs is really the OMP effect.
- ~~Anomaly to recheck: E4B routed with strict taskset was noisy/slow~~
  RESOLVED (see #3): pinning without `OMP_NUM_THREADS=4` thrashes the
  8-thread backend regions on 4 cores. With the env var set, strict
  pinning is clean and fastest (pp 42.62 ± 0.02).
- **Recommended config (RK3588, 2026-08-11):**
  `OMP_NUM_THREADS=4 taskset -c 4-7 ... -t 4` for CLI/bench; servers add
  `--threads 4 --threads-batch 8`.
- Still to do: re-tune `RKNPU_CPU_DECODE` threshold per model at t=4.

### 6. Batch serving (throughput only)

Multiple concurrent streams share each weight read — large aggregate
gains, zero single-stream latency gain. Relevant only if the workload
becomes a server.

## Experiment tooling (all run on the board 2026-08-10; reusable)

Everything compiles/links on any aarch64 Linux; running needs the board.
Pin clocks (`scripts/fix_freq_rk3588.sh`) and `ulimit -n 65536` first.

1. **`rknpu2-coop-decode-probe`** (produced the #2 verdict and the #3
   re-ranking) — `make -f Makefile.rknpu2-tools rknpu2-coop-decode-probe`,
   run with `LD_LIBRARY_PATH=../../ggml/src/ggml-rknpu2/libs`. Per shape:
   CPU-solo (Q4_0 GEMV, NEON sdot, big cores), NPU-solo (the backend's
   exact per-node M=1 driver sequence, 3-core N split), sweep of
   cooperative N-split fractions, 4 weight sets cycled to model
   consecutive layers. GEMV kernel self-checks against a scalar reference
   at startup. NPU dispatch threads and the main thread stay on little
   cores by design — keep that in any backend work. `COOP_CPU_ONLY=1`
   smoke-tests the CPU half on machines without the NPU.
2. **`rknpu2-spec-decode-bench.sh`** (produced the #1 verdict) — draft
   pairs x routing x draft-max, plus llama-server ngram self-speculation.
   Draft GGUFs now in `~/models` on the board.
3. **`rknpu2-affinity-sweep.sh`** (produced the #5 verdict) — llama-bench
   across thread count x taskset for routed, NPU-only and MoE configs.
4. **`rknpu2-driver-shim.c`** (produced the #3 re-profile) — LD_PRELOAD
   shim timing every librknnrt entry point, cumulative counters printed
   every 5 s; the steady-window slope over the token rate gives ms/token
   per driver call. `make -f Makefile.rknpu2-tools rknpu2-driver-shim.so`,
   then `LD_PRELOAD=./rknpu2-driver-shim.so llama-bench ... 2>shim.log`.
   Also the reference for decode-window timing: take profile windows at a
   fixed offset validated against the shim's run-call slope — utime-based
   "decode detectors" fire during load's quantization burst too.

## Handover notes (continuing on another machine)

- Everything lives on `github.com/unimatrix099/rk-llama.cpp`; the current
  tip branch is `feat/w4a4-neon-prep` (stacked on
  `feat/int4-native-layout` ← `feat/mixed-precision-pipelines` ←
  `fix/w4a4-calibration-crashes` ← upstream `rknpu2` + PR #21). The
  2026-08-10/11 sessions added three commits on that branch: measured
  verdicts + tooling, profiling results, and the churn-free backend
  decode path (`rknpu_dispatch_pool`); the board's `~/rk-llama.cpp` has
  the same files applied and `libggml-rknpu2.so` rebuilt from them.
- The board (Orange Pi 5 Ultra) has the repo at `~/rk-llama.cpp`, models
  in `~/models/` (`gemma-4-E4B-it-Q4_0.gguf`, `gemma-4-E2B-it-Q4_0.gguf`,
  `qwen2.5-1.5b-instruct-q8_0.gguf`, `qwen2.5-0.5b-instruct-q8_0.gguf`,
  `LFM2-8B-A1B-Q8_0.gguf`), binaries in `build/bin/` incl.
  `llama-speculative`. Remember `ulimit -n 65536` and
  `scripts/fix_freq_rk3588.sh` (in `~/rknn-llm/scripts/`) before benching.
- Raw logs are archived on the board: `~/bench-logs/2026-08-10/` (probe,
  sweeps, speculative) and `~/bench-logs/2026-08-11/` (perf/strace/gdb
  profiling, OMP experiments, clean re-verification).
- Measurement hygiene, learned the hard way: before benching, check for
  stale processes (`pgrep -af llama`) — a stuck llama-cli burned one core
  through part of the 2026-08-10 evening (numbers marked * in #1).
- Diagnostic tools build from `docs/backend/Makefile.rknpu2-tools`
  (`make all`, `check`, `check-prep`, `check-hw`, `coverage`).
- Reference numbers to beat: decode table above; prefill state in
  `RKNPU2-neon-prep-plan.md`.
