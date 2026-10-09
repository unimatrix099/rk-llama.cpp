# Memory index

- [RK3588 board access](rk3588-board-access.md) — ssh pi@192.168.0.178 (key installed), board layout, benchmark pitfalls
- [RK3588 dev environment](rk3588-dev-environment.md) — this container compiles/links board tools; board experiments now possible via SSH (see board access)
- [RK3588 project workflow](rk3588-project-workflow.md) — measure-first probes with go/no-go gates, TDD exactness tests, document all attempts, no commits without approval
- [Bonsai 27B on board](bonsai-27b-on-board.md) — v1 demo (~/Bonsai-demo, ~/bonsai.sh) DELETED from board 2026-10-02 at user request; Bonsai 2 in ~/bonsai2 remains
- [Bonsai 2 eval plan](bonsai2-eval-plan.md) — get Bonsai 2 27B running on board; benchmark vs LFM2.5, ERNIE, Bonsai v1; repos forked to /workspace
- [Bonsai 2 tg-speed optimization](bonsai2-tgspeed-optimization.md) — NEON PTQ1_0 kernel, 0.38→0.86 t/s (2.26x), bit-exact; repack GEMV regressed
- [Bonsai 2 PQ2 optimization](bonsai2-pq2-optimization.md) — NEON pq2_0_q8_K kernel, 0.32→1.35 t/s (4.2x); PQ2_0 now FASTEST band (beats PTQ1_0); + GPU/NPU decode all NO-GO (measured)
- [RK3588 24/7 VLM candidates](rk3588-247-vlm-candidates.md) — always-on image model: use NPU/RKLLM; Qwen2.5-VL-3B 7t/s, Qwen3-VL-2B 11.5t/s; board driver 0.9.8 ready
- [Git commit identity](git-commit-identity.md) — always commit as unimatrix099 <unimatrix099@github.com>, never the profile email
- [RK3588 rebase + Gemma-4 MTP](rk3588-rebase-mtp.md) — NPU loops: decode 8.78, prefill pp512 ~288 (#1c-#1h); attention-block steps 1-3 done, step 4 stopped at probe; board shut down 2026-10-06
