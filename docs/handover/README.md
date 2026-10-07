# Handover: RKNPU2 / Gemma-4 work, moving to another machine (2026-10-07)

This folder holds what lived only on the old dev machine: Claude's memory
notes, the board-side profiling patches, and the git-ignored autoresearch run
directories. Everything else is in the git history and in
`docs/backend/RKNPU2-decode-research.md` (§1c-§1h are the recent work). This
supersedes the older "Handover notes" section at the end of that file.

## Where things stand

- **Branch:** `rebase/w4a4-on-upstream` on `github.com/unimatrix099/rk-llama.cpp`
  (pushed). It is the RKNPU2 fork rebased onto ggml-org master `4ebdf2c74`
  (2026-10-02), plus all optimization work since. `upstream` =
  `github.com/ggml-org/llama.cpp`.
- **Board:** Orange Pi 5 Ultra (RK3588) at `pi@192.168.0.178`, **shut down**
  2026-10-06. Its checkout `~/rk-llama.cpp-rebase` is at `ee331dcbe` (the last
  code commit; later commits are docs only) and its `build/` matches it.
- **E4B (gemma-4-E4B-it Q4_0, ggml-org file), pure NPU W4A4, `-t 4`, taskset 4-7:**

  | | Value |
  |---|---|
  | pp512 | ~284-288 t/s (68.4 before the prefill loops) |
  | pp128 | ~192 t/s |
  | tg64 | 8.78 t/s |
  | NPU decode with MTP drafter (n=1) | 9.29 t/s |
  | PPL32 | 27.2382 |
  | KLD vs CPU (4 chunks) | 0.587297, same-top 72.06% |

  Every recent change is bit-identical against those quality numbers.
- **Work done, by loop:**

  | Doc section | Work |
  |---|---|
  | §1c | NPU decode loop |
  | §1d/§1e | Prefill loops 1-2 |
  | §1f | Prefill loop 3 (CPU-side work) |
  | §1g | Prefill loop 4 (FFN/attention block scheduling, ggml-cpu fusions) |
  | §1h | Attention block steps 1-3 (Q/K/V norms, RoPE, KV-cache writes fused into the projection dequant) |

- **Stopped at:** §1h step 4 (block-causal attention scheduler). It ended at its
  go/no-go probe: attention is CPU-bound and only ~55 ms of NPU time per pp512
  is exposed, so the ceiling is ~2-3%. Open options, cheapest first:
  1. a small cross-node overlap (attention's first QK during V's projection;
     the output projection's prep during the last PV), ~2-3%;
  2. FP16 softmax (numerics change, needs the tolerant gate), up to ~3%;
  3. W8A8 for the attention projections (no Hadamard prep on the CPU; needs
     per-name pipeline routing and a pipelined INT8 path), maybe ~5%, unmeasured.
- **Open issue:** NPU attention output depends slightly on which NPU core
  serves a KV group (±0.02% PPL, first batch always identical); see §1g.

## Setting up the new machine

1. **Clone and check out the branch:**

       git clone https://github.com/unimatrix099/rk-llama.cpp.git
       cd rk-llama.cpp
       git checkout rebase/w4a4-on-upstream
       git remote add upstream https://github.com/ggml-org/llama.cpp.git
       git config user.name unimatrix099
       git config user.email unimatrix099@github.com

   Commits must use that identity, never a profile email. Use the
   `Co-Authored-By` line the session asks for. **Never push or commit without
   the user's approval.**

2. **Local tools:** git, ssh/scp, python3, make and g++ (for the unit tests in
   `docs/backend/Makefile.rknpu2-tools`). No local cmake build is needed:
   everything is built and run **on the board**. The scripts send the local
   commits there as a git bundle.

3. **Restore Claude's memory** (optional but recommended). Copy
   `docs/handover/claude-memory/*.md` into Claude Code's project memory
   directory on the new machine:
   `~/.claude/projects/<slug>/memory/`, where `<slug>` is the repo path with
   `/` replaced by `-` (for `/workspace` it is `-workspace`; for
   `/home/me/rk-llama.cpp` it is `-home-me-rk-llama-cpp`). Then put the board
   password back into `rk3588-board-access.md` locally. It was redacted
   because this fork is public; the user has it.

4. **Network and SSH to the board.** The new machine must reach
   192.168.0.178 on the same LAN. If a sandbox reports "No route to host" to
   192.168.0.x, its firewall is blocking the LAN.

       ssh-keygen -t ed25519            # if there is no key yet
       ssh-copy-id pi@192.168.0.178     # password from the user
       ssh -o BatchMode=yes pi@192.168.0.178 true && echo ok

5. **Power the board on**, then once per boot:

       ssh pi@192.168.0.178
       sudo ~/rk3588-scripts/fix_freq_rk3588.sh   # pins clocks; check scaling_governor == userspace
       ulimit -n 65536                            # needed for every NPU process

6. **Recreate the loop harness** (`autoresearch/` is git-ignored):

       mkdir -p autoresearch/loop-next
       cp docs/backend/rknpu2-autoresearch/cpu-npu-2/{verify.sh,guard.sh} autoresearch/loop-next/

   The scripts find the repo from their own location; set
   `RKNPU_BOARD=user@host` if the board address changed.
   - `verify.sh`: bundles local commits to the board, rebuilds there, and
     prints E4B pp512 t/s.
   - `guard.sh` runs:
     - the unit tests (locally);
     - 32-chunk PPL ≤ 27.41;
     - KLD ≤ 0.65 vs `~/kld-cpu-e4b-4ch.bin`, with same-top ≥ 68%;
     - the NPU flash-attention test (12/12) and the RoPE exactness test (5/5);
     - three `llama-server` decode-identity runs.

7. **Sanity check:** run `verify.sh`. It should print about 284-288. Then
   `guard.sh` should print PPL 27.2382 and KLD 0.587297.

8. **Resuming with Claude:** open the repo and say something like: *"Read
   docs/handover/README.md and decode research §1h, then continue with option
   1 (small attention overlap)."*

## Folder contents

- `claude-memory/`: Claude's memory notes, board password redacted.
  `MEMORY.md` is the index.
- `profiling-patches/`: board-side instrumentation used for every profile in
  §1g/§1h. They are Python scripts that patch
  `ggml/src/ggml-rknpu2/ggml-rknpu2.cpp` by text anchors. Use them on the
  board, then revert:

      scp docs/handover/profiling-patches/dbgwait.py pi@192.168.0.178:/tmp/
      ssh pi@192.168.0.178 'cd ~/rk-llama.cpp-rebase && python3 /tmp/dbgwait.py && \
        cmake --build build -j8 --target llama-bench && ulimit -n 65536 && \
        RKNPU_PROFILE=1 taskset -c 4-7 build/bin/llama-bench -m ~/models/gemma-4-E4B-it-Q4_0-ggmlorg.gguf \
          -p 512 -n 0 -r 8 -t 4 2>&1 | grep -E "DBGW|RKNPU_PROF"; \
        git checkout ggml/src/ggml-rknpu2/ggml-rknpu2.cpp'

  | Script | What it measures |
  |---|---|
  | `dbgwait.py` | per-node time and NPU-wait (by name) |
  | `dbgfa2.py` | NPU flash-attention stages |
  | `dbgffn.py` / `dbgffnc.py` | FFN block phases / per-step waits |
  | `dbgspan.py` | runner NPU busy span |
  | `dbgsync.py` | time in `rknn_mem_sync` |
  | `dbgnomath.py` | GEGLU math cost (diagnostic, wrong output) |

  Anchors can drift as the code changes. A script asserts or silently fails
  to patch, so check that its `DBG` output appears.
  `RKNPU_PROFILE=1` (built in) prints graph, node, NPU-run and NPU-wait totals.
  User-space perf works without sudo:
  `perf record -e cycles:u -F 1999 -D 12000 ...`.
- `autoresearch-runs/`: the four loops' `verify.sh`, `guard.sh`,
  `results.tsv` and `handoff.json`, as they ran. The originals use
  `/workspace` paths.

## Gotchas learned the hard way

- **Code layout moves results by a few percent between builds.** Judge
  changes with a same-binary A/B behind an env switch; every new path in the
  backend has one.
- **GCC contracts multiply+add (including NEON intrinsics) into FMA.**
  - Fused ops that must match an unfused path need an
    `__asm__("" : "+w"(x))` barrier.
  - RoPE exactness depends on how ggml-cpu was compiled (§1h);
    `test-rknpu2-rope.cpp` guards it.
- **llama.cpp pins tensors named `"norm"` to the layer device (the CPU here)**
  when fully offloaded. Name new norms differently if the backend should take
  them (§1h).
- **`pkill -f <pattern>` can kill the shell that runs it.** Anchor the
  pattern (`"^build/bin/llama-perplexity"`). A killed NPU process frees its
  memory slowly, so the next run may fail at model load. Wait or retry.
- **huggingface.co may be unreachable from the dev machine.** Download on the
  board instead.
- **The board's models** are in `~/models/`. The measured E4B is
  `gemma-4-E4B-it-Q4_0-ggmlorg.gguf`; the MTP drafter is
  `gemma-4-E4B-it-assistant-F16.gguf`.
