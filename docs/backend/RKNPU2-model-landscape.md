# Model landscape for the RK3588 board (snapshot 2026-10-02)

This file records what we know about which models to run on the Orange Pi 5 Ultra
(RK3588, 16 GB). It sets public "intelligence" benchmark scores beside the speeds
measured on the board, lists which models ship MTP or drafter support, and names the
new models still to test.

How this relates to the other docs:

- **Board speeds** come from `RKNPU2-deployment-guide.md` ("Known-good models") and
  `RKNPU2-decode-research.md` (#1b for MTP). That guide is the maintained source for
  run configurations.
- **Benchmark scores** come from vendor model cards and third-party trackers, gathered
  by web search on 2026-10-02. Vendors run their own evaluations with their own
  settings, so scores are only roughly comparable across vendors.
- **Every number here is dated.** Re-check the scores before relying on them.

## Tested models: intelligence vs speed

Board speeds use each model's best configuration with `-t 4` on the A76 cores. They
come from llama-bench pp128 / tg64, except where marked "MTP": those are
`llama-server` runs, greedy, 4 prompts x 128 tokens.

| Model | Released | MMLU-Pro | GPQA | Math | IFEval | Board prefill t/s | **Board decode t/s** |
|---|---|---|---|---|---|---|---|
| LFM2.5-8B-A1B (reasoning) | 2026-05-28 | ~63 (3rd-party) | ~51 (3rd-party) | AIME26 50.0, MATH500 88.8 | **91.8** | 63.7 | 18.0 (part of it `<think>`) |
| **Gemma-4 E4B** | 2026-04-02 | **69.4** | **58.6** (Diamond) | AIME26 42.5 | — | 37.0–45 | 6.9; **10.67 with MTP** (CPU) |
| **Gemma-4 E2B** | 2026-04-02 | 60.0 | 43.4 (Diamond) | AIME26 37.5 | — | 135–174 | 11.0–13.9; **19.94 with MTP** (CPU) |
| LFM2-24B-A2B | 2026-02-24 | not published as numbers | 47.4 (Diamond, AA) | — | — | 32.7 | 15.0 |
| LFM2-8B-A1B (retired by Liquid) | 2025-10-07 | 37.4 | 29.3 (full set) | MATH500 74.2 | 77.6 | 55.1 | **23.4** |
| ERNIE-4.5-21B-A3B | 2025-06-30 | ~66 (Baidu table, column uncertain) | not found | — | 80.0 | 25.6 | 7.4 |
| Qwen2.5-1.5B | 2024-09-19 | 32.4 | 29.8 (Diamond) | MATH 55.2 | — | **280.8** | 13.5 |

Gemma-4 scores are Google's own, probably measured with thinking enabled. LiveCodeBench
v6 scores, where published: E4B 52, E2B 44, LFM2-8B 21. "AA" means Artificial Analysis.
GPQA "full set" is easier than "Diamond", so those two columns don't compare directly.

Perplexity cannot be compared across rows. The tokenizers differ, so a model's
perplexity only means something against its own CPU baseline.

### Reading the trade-off (2026-10-02)

- **Gemma-4 E4B is the smartest model that fits.** It is the only one with vision, and
  W4A4 costs it no quality. With MTP it reaches 10.67 t/s, so it is no longer a slow
  model.
- **Gemma-4 E2B with MTP is the best balance.** It decodes at 19.94 t/s, sits in the
  middle on intelligence, has vision, and gets the NPU's second-best prefill.
- **LFM2.5-8B-A1B leads on instruction following and math.** Its knowledge is weak (AA
  Omniscience accuracy 8.7%), and its reasoning tokens reduce the speed you actually get.
- **LFM2-8B-A1B is still the fastest at 23.4 t/s**, but it has been superseded.
- **ERNIE, LFM2-24B and Qwen2.5-1.5B are dominated** on the speed/intelligence trade.
  ERNIE and the Bonsai v1 demo were deleted from the board on 2026-10-02 to free space.

## MTP / drafter availability

| Model | Drafter | Usable on the rebased fork | Measured |
|---|---|---|---|
| Gemma-4 E2B / E4B | Google `gemma-4-*-it-assistant` (78.8M, shares the target's KV) | yes, `--spec-type draft-mtp` | **yes, decode research #1b** |
| Gemma-4 26B-A4B / 12B Unified | Google assistant drafters; the 12B's ships in the QAT repo | yes | not measured |
| Qwen3.5 0.8B–9B | built-in MTP heads (Unsloth `*-MTP-GGUF`) | yes. Upstream disables `--mmproj` while MTP is on | not measured |
| ERNIE-4.5-21B-A3B | MTP head in the checkpoint (`model.mtp_*`) | no llama.cpp path (vLLM/FastDeploy only) | — |
| LFM2.5-8B-A1B | DSpark drafter, ~330M (2026-08) | no: upstream DSpark supports Qwen3 backbones only | — |
| MiniCPM5-2B | DSpark drafter, ~324M; no MTP heads | probably no (Qwen3-backbone limit); unverified | — |
| LFM2-8B-A1B, LFM2-24B-A2B, Qwen2.5-1.5B | none | — | — |

## Candidates not yet tested

Ranked by expected value for this board. RAM budget: keep the file under ~13 GB.

| # | Model | Released | Q4_0 size | Why | Watch out for |
|---|---|---|---|---|---|
| 1 | **Qwen3.5-4B** | 2026-03-02 | ~2.5 GB | strongest model under 5B (AA index 27 vs E4B's 9); vision; IFEval 89.8; built-in MTP | hallucination ~80% on AA Omniscience; very token-hungry reasoning |
| 2 | **MiniCPM5-2B** | 2026-09-07 | ~1.5 GB | MMLU-Pro 70.8, AIME 86.5, LiveCodeBench 69, strong tool use; Llama architecture | text only; quantized build runs away 92% of the time on default settings (one-flag fix) |
| 3 | **Qwen3.5-9B** | 2026-03-02 | ~5 GB | smartest model under 10B (AA 32, GPQA-Diamond 81.7); vision; MTP | expect E4B-class speed before MTP |
| 4 | Gemma-4 26B-A4B (Google QAT Q4_0) | 2026-04-02 | 14.4 GB | MMLU-Pro 82.6, GPQA 79; MoE, 3.8B active | likely too tight for RAM; may need a ~11 GB 3-bit file |
| 5 | Gemma-4 12B Unified | 2026-06-03 | ~8 GB | unified text, image and audio; ships an MTP drafter | needs `gemma4_unified`, which the pre-rebase fork lacks; the rebased tree should load it (untested) |

Doesn't fit or not available: Qwen3.6-35B-A3B and Qwen3.8-27B (16–20 GB at Q4;
a dense 27B would decode at ~1 t/s here). LFM2.5-24B-A2B is announced but not released.
Qwen3.6 has no small sizes.

Two rules from the backend work apply to every candidate:

1. **W4A4 damage tracks the area of the tensors the NPU quantizes**, not the parameter
   count (decode research #4f/#4h). Small and MoE models need
   `RKNPU_HYBRID=W8A8_STANDARD` or pure CPU. Check with one perplexity run against the
   CPU.
2. **On the CPU, MTP wins at `--spec-draft-n-max 3`**, because the 4-row repack tile
   applies to any Q4_0 target. Routed mode loses with MTP (decode research #1b).

## Sources (retrieved 2026-10-02)

- Gemma 4: [HF blog](https://huggingface.co/blog/gemma4), [technical report](https://arxiv.org/pdf/2607.02770), [AA E4B](https://artificialanalysis.ai/models/gemma-4-e4b), [AA 26B-A4B](https://artificialanalysis.ai/models/gemma-4-26b-a4b), [Gemma MTP docs](https://ai.google.dev/gemma/docs/mtp/mtp), [Unsloth Gemma 4](https://unsloth.ai/docs/models/gemma-4)
- LFM2 / LFM2.5: [LFM2-8B-A1B card](https://huggingface.co/LiquidAI/LFM2-8B-A1B), [LFM2 tech report](https://arxiv.org/html/2511.23404v1), [LFM2.5-8B-A1B blog](https://www.liquid.ai/blog/lfm2-5-8b-a1b), [AA LFM2.5](https://artificialanalysis.ai/models/lfm2-5-8b-a1b), [LFM2-24B-A2B blog](https://www.liquid.ai/blog/lfm2-24b-a2b), [LFM2.5-8B-A1B-DSpark](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-DSpark)
- ERNIE: [ERNIE 4.5 tech report](https://ernie.baidu.com/blog/publication/ERNIE_Technical_Report.pdf), [MTP tensors issue](https://github.com/pjordanandrsn/experts4bit-qlora/issues/529)
- Qwen: [Qwen2.5-LLM blog](https://qwenlm.github.io/blog/qwen2.5-llm/), [Qwen3 tech report](https://arxiv.org/pdf/2505.09388), [AA Qwen3.5 small models](https://artificialanalysis.ai/articles/qwen3-5-small-models), [Unsloth Qwen3.5-9B-MTP](https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF)
- MiniCPM5: [MarkTechPost](https://www.marktechpost.com/2026/09/07/openbmb-releases-minicpm5-2b-a-2-52b-dense-model-averaging-53-9-across-34-benchmarks-and-built-to-run-on-device/), [AA](https://artificialanalysis.ai/articles/openbmb-releases-minicpm5-2b), [DSpark GGUF](https://huggingface.co/aj9o9/MiniCPM5-2B-DSpark-GGUF)
- llama.cpp DSpark: [PR #25173](https://github.com/ggml-org/llama.cpp/pull/25173)
