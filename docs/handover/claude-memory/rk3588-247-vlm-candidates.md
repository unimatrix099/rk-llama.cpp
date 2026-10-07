---
name: rk3588-247-vlm-candidates
description: Candidate 24/7 vision-LLMs for the RK3588 board — NPU via RKLLM is the runtime; measured speeds
metadata: 
  node_type: memory
  type: project
  originSessionId: 99d50116-a329-4a80-bf79-9e91065d77ff
  modified: 2026-09-29T14:25:00.273Z
---

Goal (2026-09-29): find an always-on (24/7) model for the Orange Pi 5 Ultra (RK3588) that does **images + decent intelligence + decent speed**, separate from Bonsai. See [[rk3588-board-access]], [[bonsai2-pq2-optimization]].

**KEY INSIGHT: use the NPU via RKLLM, not CPU/llama.cpp.** Bonsai was CPU-only because of its exotic ternary quant + Hadamard (no NPU op support). Mainstream VLMs DO have RK3588 NPU ports (RKLLM) and run 5-11 t/s (vs Bonsai 1.35), lower power, freeing the 8 CPU cores — ideal for 24/7.

**Board is READY:** RKNPU driver **v0.9.8** (matches RKLLM 1.2.3 ports), 15 GB RAM free. Only missing `librkllmrt` 1.2.3 userspace runtime (single .so drop-in) + a pre-converted `.rkllm` model.

**Measured on-device NPU speeds (Qengineering ports, w8a8 LLM + fp16 vision encoder):**
| Model | decode | RAM | notes |
|-------|-------:|----:|-------|
| Qwen3-VL-2B | 11.5 t/s | 3.1 GB | fastest; weaker dense-doc OCR; 448x448 |
| Qwen2.5-VL-3B | 7.0 t/s | 4.8 GB | **balanced default**; strong OCR/charts; 392x392 |
| Qwen3-VL-4B | 5.7 t/s | 8.7 GB | most intelligence that fits; DocVQA 95.3, MMMU 67.4 (beats Gemma3-27B on docs); 448x448 |

Intelligence: InternVL3.5-4B highest reasoning (MMMU 82.6) but NPU port less turnkey (InternVL2-1B/InternVL3-1B are the supported ones). Gemma 3 4B = good general + 128K ctx + best llama.cpp/GGUF support but Qwen3-VL beats it on OCR. **Qwen3-VL family = standout for OCR/documents/screenshots.**

**REFINED GOAL (2026-09-29): user wants an LLM for AGENTS (tool/function calling) that ALSO reads images.** => Qwen3-VL family is the pick (strong agentic tool-calling + MCP via Qwen-Agent, AND vision). Tool-calling reliability scales with size: **4B is the practical floor** for real multi-step agent use; 2B only for trivial single-step tool use; 8B if workflows are complex (~3-4 t/s est, ~10-12GB, still fits 15GB). InternVL3.5-8B (NPU port exists, Qengineering) = alt for max reasoning but Qwen wins on tool-calling ecosystem.

**RECOMMENDATION: Qwen3-VL-4B on NPU (5.7 t/s, 8.7GB) via an OpenAI-compatible RKLLM tool-calling server.** NPU path IS agent-ready: `rkllama` (tools + OpenAI/Ollama API + VLM), Luna-Inference `rkllm-server` (OpenAI tool-calling w/ Qwen3, verified rk3588), `huangyajie/rkllm-openai`. Gives agents standard /v1/chat/completions + function-calling + image input. Deployment guide: Sngular "Deploying Qwen3 on Orange Pi 5 NPU using RKLLama + MicroK8s".

**Tooling:** `airockchip/rknn-llm` (official RKLLM 1.2.3), `rkllama` (Ollama-like, tools+VLM), Qengineering repos (ready-to-run RK3588 .rkllm builds w/ measured speeds).

**GOOGLE / GEMMA 4 (clarified 2026-09-29):** Gemma 4 (Apr 2026): E2B, E4B (image+video+audio, natively agentic "structured tool use", 128K), + **26B-MoE** + 31B dense. Gemma 4 E4B = 4.5B effective (8B w/ embeddings), DENSE hybrid-attn. **Vision on the board: only via llama.cpp CPU path** — llama.cpp supports Gemma 4 images from day one via mmproj/libmtmd. **RKLLM/NPU Gemma builds are TEXT-ONLY** (vision encoder not converted; in progress, jaylfc issue #197). So Gemma-with-vision = CPU (slower prefill, CPU busy 24/7). MoE note: Gemma-4-26B-MoE & Qwen3-VL-30B-A3B don't fit 16GB well (Q4 ~14-18GB, tight/over) + MoE-on-NPU not turnkey (needs expert-parallelism); board RAM caps TOTAL params so MoE speed advantage doesn't pay off here.

**DECISION AXIS: NPU speed vs Gemma feature set.** Qwen3-VL-4B (NPU) = vision NPU-accelerated, fast, CPU-free, best for 24/7. Gemma 4 E4B (llama.cpp CPU) = Google + audio+video + 128K + strong agentic, but CPU-bound vision. NEXT: head-to-head on board (measure-first) — Qwen3-VL-4B NPU vs Gemma 4 E4B CPU: same image+tool-call test, compare t/s, image-prefill latency, tool-call reliability. Not yet run.
