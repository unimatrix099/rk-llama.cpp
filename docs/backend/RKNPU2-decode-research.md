# RKNPU2: improving 4-bit token-generation (decode) speed — research notes

Research thread with **measured verdicts as of 2026-08-10** (board session:
probe + sweeps, tooling under "Experiment tooling" below). Summary: #5
threading is a shipped big win (+19–66% tg, no code); #1 speculative and
#2 cooperative decode are measured dead ends; **#3 backend overhead
surgery is now the main code lever** — the probe showed the NPU driver
path itself sustains ~28.6 GB/s at M=1, so the decode gap lives in the
backend/CPU side. No backend code changes yet. Companion documents:
`RKNPU2-optimization-notes.md` (shipped work + dead ends),
`RKNPU2-int4-research.md` (INT4 fundamentals), `RKNPU2-neon-prep-plan.md`
(prefill prep). Numbers: Orange Pi 5 Ultra (RK3588, 16 GB), pinned clocks,
`llama-bench -p 128 -n 64` unless noted.

## Where decode stands after the shipped work

Decode re-reads the active weight set for every generated token, so it is
memory-bandwidth bound. For Gemma-4-E4B Q4_0 (7.46 B dense):

| Decode path | tg64 t/s | effective read rate |
|---|---|---|
| CPU (`RKNPU_CPU_DECODE=32`, native Q4_0 reads, ~2.4 GB/token) | ~~5.35~~ **5.50** at t=4 | ~~13~~ **~25 GB/s at t=4 big-core** (13 was an 8-thread artifact — see #5) |
| NPU W4A4 native layout (~2.05 GB/token INT4) | 3.65 (4.54 W8A8 at t=4) | ~7.5 GB/s aggregate |

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
| draft model (0.5B, draft-max 8) | 4.30 | 41.7% | 13.52 (bench) |
| ngram-simple, repetitive rewrite prompt | 10.76 | 43% (42 drafted) | 12.15 (server) |
| ngram-map-k, repetitive rewrite | 11.86 | 4/7 | 12.15 |
| ngram-simple + verify on NPU (`RKNPU_CPU_DECODE=16` or `8`) | 11.62 | 60% (48 drafted) | 12.15 |
| any ngram type, creative prompt | 12.17–12.20 | no drafts trigger | 12.15 |
| E4B ngram-simple, repetitive | 3.64 | 2/16 | 4.17 (server) |

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

### 3. Backend overhead surgery — NOW THE MAIN CODE LEVER (re-ranked 2026-08-10)

The coop probe changed this avenue's status and its target. Measured: the
per-node M=1 driver sequence (A set/sync, 3 parallel runs, 3 C syncs)
sustains ~28.6 GB/s — an E4B NPU-decode read floor of ~72 ms/token
(~14 t/s), and the t=4 CPU path reads at ~25 GB/s (~8-9 t/s potential).
Actual: 3.65-5.50. The missing 100-200 ms/token is backend/CPU-side and
unprofiled. Plan:

1. `perf record` a real decode step (llama-bench tg, routed and NPU-only)
   and attribute time: ggml scheduler splits/copies, backend cache
   lookups under mutex, A-prep, C dequant, thread-pool wake/sync,
   between-node CPU ops.
2. Fix in measured order. Candidates from earlier analysis: per-split
   scheduler cost (~600 splits/token on E4B), QKV/gate-up fusion (~40%
   fewer dispatches), batching the 3 per-node `rknn_mem_sync` calls.
- The old caveat ("splits are not the prefill bottleneck") measured
  aggregate byte volume, not fixed per-node costs at M=1 — it does not
  contradict this. The speculative results add a second motivation:
  cheap bs>1 verification would revive ngram speculation for servers.

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
- NPU decode gains too (+32-40%): the between-node CPU work contends
  with idle-thread scheduling — more evidence for the avenue #3 finding.
- Anomaly to recheck: E4B routed with strict `taskset -c 4-7` was noisy
  and slow (pp 8.75 +/- 2.10); t=4 WITHOUT taskset was clean and best.
  Until understood, recommend `-t 4` unpinned for E4B-class models and
  pinning only smaller models.
- Deployment note: for servers, use `--threads 4 --threads-batch 8`
  (decode wants 4 big; prefill prep still profits from 8).
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

## Handover notes (continuing on another machine)

- Everything lives on `github.com/unimatrix099/rk-llama.cpp`; the current
  tip branch is `feat/w4a4-neon-prep` (stacked on
  `feat/int4-native-layout` ← `feat/mixed-precision-pipelines` ←
  `fix/w4a4-calibration-crashes` ← upstream `rknpu2` + PR #21).
- The board (Orange Pi 5 Ultra) has the repo at `~/rk-llama.cpp`, models
  in `~/models/` (`gemma-4-E4B-it-Q4_0.gguf`, `gemma-4-E2B-it-Q4_0.gguf`,
  `qwen2.5-1.5b-instruct-q8_0.gguf`, `qwen2.5-0.5b-instruct-q8_0.gguf`,
  `LFM2-8B-A1B-Q8_0.gguf`), binaries in `build/bin/` incl.
  `llama-speculative`. Remember `ulimit -n 65536` and
  `scripts/fix_freq_rk3588.sh` (in `~/rknn-llm/scripts/`) before benching.
- Raw logs from the 2026-08-10 session are archived on the board in
  `~/bench-logs/2026-08-10/`.
- Diagnostic tools build from `docs/backend/Makefile.rknpu2-tools`
  (`make all`, `check`, `check-prep`, `check-hw`, `coverage`).
- Reference numbers to beat: decode table above; prefill state in
  `RKNPU2-neon-prep-plan.md`.
