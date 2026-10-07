---
name: bonsai-27b-on-board
description: PrismML Bonsai Ternary-27B is installed and runs CPU-only on the Orange Pi 5 Ultra (~1.1 tok/s)
metadata: 
  node_type: memory
  type: project
  originSessionId: ffc99c7f-1775-4498-b753-c29d2fb6fc47
  modified: 2026-08-31T13:02:45.720Z
---

PrismML Bonsai **Ternary-27B** (their Qwen3.5-arch 27B, a reasoning model) is installed on the board and verified working, set up 2026-08-31 following https://docs.prismml.com/get-started/quickstart. See [[rk3588-board-access]] for how to connect.

Layout on the board:
- Repo: `~/Bonsai-demo` (cloned from `PrismML-Eng/Bonsai-demo`).
- Binary: `~/Bonsai-demo/bin/vulkan/llama-cli` (+ `llama-server`, `llama-bench`) — the prebuilt `linux arm64` asset from the Prism llama.cpp fork release `prism-b10660-e311ed3`. Despite the `vulkan/` dir name it runs CPU: board has **no libvulkan** ("ggml_vulkan: No devices found"), falls back to the `armv8.2` dotprod CPU backend. Always pass `-ngl 0`.
Models installed (downloaded one GGUF each via HF resolve URL, NOT `setup.sh` — it also pulls a bf16 drafter + mmproj + Open WebUI/Jupyter, too much for the root disk; `~/models` already holds 31 GB of other GGUFs). Ternary = `models/ternary-gguf/{S}B/Ternary-Bonsai-{S}B-PQ2_0.gguf` (repo `prism-ml/Ternary-Bonsai-{S}B-gguf`); 1-bit = `models/1bit-gguf/{S}B/Bonsai-{S}B-Q1_0.gguf` (repo `prism-ml/Bonsai-{S}B-gguf`). Have 27B + 8B of each.

Convenience launcher: `~/bonsai.sh {chat|ask "prompt"|server}`, env `SIZE=27|8` / `BITS=0|1` / `THREADS` / `CTX` / `THINK=0`.

**Measured (llama-bench, -ngl 0, -fa 1, best at `-t 8`):**
| Model | Size | prompt t/s | gen t/s |
|---|---|---|---|
| 27B ternary PQ2_0 | 6.66 GiB | 1.41 | 1.13 |
| 27B 1-bit Q1_0 | 3.53 GiB | 5.40 | 1.87 |
| 8B ternary PQ2_0 | 2.03 GiB | 5.05 | 3.64 |
| 8B 1-bit Q1_0 | 1.07 GiB | 19.2 | 6.41 |

**Roofline (measured 2026-08-31):** sustained CPU read BW (STREAM-style OpenMP sum, `~/bw.c`/`~/bw`) = ~25 GB/s peak at 2 threads, ~22.5 GB/s at 8 (BW *drops* as threads rise). Roofline tg = BW/size. Measured tg is only **~33% of that ceiling** for all four models. It's **compute-bound on low-bit unpack, NOT bandwidth-bound**: tg rises 4→8 threads even though DRAM BW falls; A76-only pinned (`taskset -c 4-7`, tg 5.36) is *slower* than all-8 (6.41). Board already flat-out (userspace gov @ max: A55 1.8/A76 2.35 GHz/DDR 2.4 GHz, 42 °C, no throttle). So the ~3× gap to roofline is unreclaimable by tuning — it's the cost of unpacking 2.13-bpw/1-bit weights on a CPU with dotprod but no i8mm/bf16. Lever that moves tok/s is model size, not knobs. gen t/s roughly tracks 1/size; barely scales past `-t 4`. CPU has dotprod but no i8mm/bf16/SVE. 27B models are arch `qwen35`, 8B is `qwen3` (8.19B). The 27B chat template leaves reasoning/thinking ON (wastes tokens at ~1 tok/s — use `THINK=0`); the **8B template defaults thinking OFF** (answers directly). 8B 1-bit (~6.4 tok/s) is the only genuinely interactive option; 27B is "ask and wait".

**Note:** the 27B chat template leaves reasoning/thinking ON, which wastes tokens on short prompts at ~1 tok/s. Pass `THINK=0` (`--reasoning-budget 0 --chat-template-kwargs {"enable_thinking":false}`) to suppress it.

**Update 2026-10-02:** `~/Bonsai-demo` and `~/bonsai.sh` were deleted from the board at the user's request to free disk; Bonsai 2 (`~/bonsai2`) is unaffected.
