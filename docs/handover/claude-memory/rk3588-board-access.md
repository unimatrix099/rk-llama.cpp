---
name: rk3588-board-access
description: SSH access to the Orange Pi 5 Ultra board (192.168.0.178) for running RK3588 benchmarks
metadata: 
  node_type: memory
  type: project
  originSessionId: ba415b22-8ecc-40fd-b8a4-dd7d353ed4e9
  modified: 2026-10-02T19:59:09.221Z
---

The Orange Pi 5 Ultra is reachable at `pi@192.168.0.178`. Password: ask the user (not stored in git) (lowercase, verified 2026-09-24 after a reflash — kernel `6.1.115-vendor-rk35xx`, hostname `orangepi5-ultra`; the earlier capital-`O` note is now stale). Passwordless-ish sudo works via `echo <board-password> | sudo -S`. Containers are ephemeral: a fresh container has no `~/.ssh`, so the previously-installed key is gone and you must re-bootstrap. Bootstrap once: `ssh-keyscan` the host into known_hosts, then drive the password prompt with a pty (no sshpass/root here) to append this container's `~/.ssh/id_ed25519.pub` to the board's `authorized_keys`; after that `ssh -o BatchMode=yes pi@192.168.0.178` is key-based. A working pty password helper (`pssh.py`, reads `BOARD_PW`) was written to scratchpad on 2026-08-31.

Board facts (verified 2026-10-02): repo at `~/rk-llama.cpp` (pre-rebase tree + build, kept as A/B baseline), rebased tree as a git worktree at `~/rk-llama.cpp-rebase` (branch `rebase/w4a4-on-upstream`), binaries in `build/bin/`. Models in `~/models/`: gemma-4-E4B-it-Q4_0 (unsloth, sha 4a403d2e…), gemma-4-E2B-it-qat-q4_0 (Google), gemma-4-{E4B,E2B}-it-assistant-F16 (MTP drafters, converted from Google safetensors in `~/hf/`), LFM2.5-8B-A1B-Q4_0. ERNIE and `~/Bonsai-demo` deleted 2026-10-02 at user's request (disk was 91% full). Python venv with torch for `convert_hf_to_gguf.py`: `~/venv-convert`. Clock-pin script is `~/rk3588-scripts/fix_freq_rk3588.sh` (the old `~/rknn-llm/` path is gone); run with sudo and check `scaling_governor` == userspace — it is NOT pinned after boot. Bench logs in `~/bench-logs/`. Always `ulimit -n 65536` before NPU work.

Network: if the container gets "No route to host" for 192.168.0.x (even the router), it's the sandbox firewall — the user restarts the sandbox without the firewall; credentials are not the issue. huggingface.co is blocked from the container but reachable from the board — download on the board.

**Why:** this supersedes the "no board access" limitation in [[rk3588-dev-environment]] — measurements can now be driven directly from this container.

**How to apply:** long benchmarks: launch with `ssh -f ... 'nohup ... &'` and poll the log (a plain `&` keeps the ssh session hanging). Pitfalls measured on-board: llama-cli busy-spins at EOF stdin (use llama-bench/llama-server); never `taskset` a two-model process onto 4 cores (spinning threadpools, 2.6x cost); heredoc-quoted inline python mangles `\"` escapes — ship parser scripts as files.
