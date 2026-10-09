---
name: rk3588-project-workflow
description: "RKNPU2 fork working conventions — measure-first, TDD, every attempt documented (incl. failures), no AI commits without approval"
metadata: 
  node_type: memory
  type: project
  originSessionId: ba415b22-8ecc-40fd-b8a4-dd7d353ed4e9
  modified: 2026-08-10T19:17:09.023Z
---

Working conventions on the rk-llama.cpp RKNPU2 fork (established across the native-layout and NEON-prep work, visible in `docs/backend/RKNPU2-*.md`):

- Measure before building: standalone probe in `docs/backend/` (wired into `Makefile.rknpu2-tools`) gates any backend change; explicit go/no-go thresholds written into the research doc.
- TDD with element-exact scalar references for every kernel; validation = bit-identical perplexity vs the previous build.
- Every optimization attempt — shipped, failed, deprioritized — is recorded in `RKNPU2-optimization-notes.md` with the measurements; docs are kept handover-ready.
- AGENTS.md (llama.cpp AI policy): never commit/push without explicit human approval; the user authors commit messages and PR text. Private fork, so AI assistance on code is fine, but the user must be able to defend all of it.

**Why:** the fork's history shows failed-path documentation deliberately closing off dead ends, and the divergence investigation proved the exactness culture catches real pre-existing bugs.

**How to apply:** for any new optimization: research-doc section → probe/measurement → plan doc → TDD implementation → board validation → results recorded. See [[rk3588-dev-environment]] for what can run where.
