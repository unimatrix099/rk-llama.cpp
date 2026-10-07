---
name: rk3588-dev-environment
description: This /workspace container has no NPU or board access — dev/compile only; the Orange Pi 5 Ultra board is reached outside this session
metadata: 
  node_type: memory
  type: project
  originSessionId: ba415b22-8ecc-40fd-b8a4-dd7d353ed4e9
  modified: 2026-08-10T21:02:33.141Z
---

The `/workspace` rk-llama.cpp checkout runs in a 2-core aarch64 Docker container (Apple-silicon host, ~2 GB RAM). No `/dev/rknpu*`, no SSH config to the board. The vendored `ggml/src/ggml-rknpu2/libs/librknnrt.so` is aarch64-Linux, so board tools in `docs/backend/` compile AND link here — they just can't execute NPU calls. CPU-side logic can be smoke-tested here (e.g. `COOP_CPU_ONLY=1` for the coop-decode probe; NEON incl. dotprod works).

**Why:** every measurement requires the Orange Pi 5 Ultra — now reachable directly over SSH, see [[rk3588-board-access]].

**How to apply:** develop and compile-verify here, measure on the board over SSH; don't trust the container's timing numbers. Follow the project's measure-first methodology ([[rk3588-project-workflow]]).
