---
type: analysis
status: complete
created: 2026-08-26
purpose: test whether the reproducibility floor groups by how the backbone is served rather than by which backbone it is
producer: scripts/analysis/serving_mode_floor.py
---

# Is the reproducibility floor a property of the model, or of the serving path?

Regenerate: `.venv/bin/python3 scripts/analysis/serving_mode_floor.py`

Until a second API-served backbone landed (2026-08-21) this question could not be asked: the project held one API model and one local one, so *model* and *serving path* were the same variable. Every floor below is the same functional — per-task discordance between two runs of an identical condition.

## 1. The two groups

| serving | arms | families | sites | floor range | powered arms (d≥10) |
|---|---|---|---|---|---|
| **API** | 13 | OpenAI, Qwen | 2 | **4.93–14.29%** | 12 (7.39–14.29%) |
| **local** | 14 | Gemma, Qwen | 4 | **0.00–9.62%** | 6 (0.00–9.62%) |

The groups **overlap**.

Exact one-sided rank test on a perfect split: **p = 0.0000** (680/20058300 assignments at least this extreme). ⚠️ arms within a cell are not independent (shared site, backbone, task universe); descriptive separation statistic, NOT a gateable test.

Restricted to arms carrying an interval (d≥10): separated=**False**, p = 0.0015, gap -2.23pp.

## 2. Every arm, with its power

| serving | backbone | arch | site | arm | n | SR | floor | d | interval? |
|---|---|---|---|---|---|---|---|---|---|
| API | `Qwen3-VL-235B-A22B` | MoE 235B-A22B | VWA-classifieds | `vision` | 224 | 24.55% | **14.29%** | 32.4 | yes |
| API | `Qwen3-VL-235B-A22B` | MoE 235B-A22B | VWA-classifieds | `som` | 224 | 28.35% | **12.95%** | 37.5 | yes |
| API | `GPT-5.6-terra` | undisclosed | VWA-classifieds | `dom` | 224 | 24.33% | **12.95%** | 32.2 | yes |
| API | `Qwen3-VL-235B-A22B` | MoE 235B-A22B | VWA-classifieds | `pprompt` | 224 | 18.30% | **12.50%** | 24.2 | yes |
| API | `Qwen3-VL-235B-A22B` | MoE 235B-A22B | VWA-classifieds | `dom` | 224 | 16.29% | **12.05%** | 21.5 | yes |
| API | `Qwen3-VL-235B-A22B` | MoE 235B-A22B | VWA-classifieds | `psom` | 224 | 14.96% | **12.05%** | 19.8 | yes |
| API | `Qwen3-VL-235B-A22B` | MoE 235B-A22B | VWA-reddit | `pprompt` | 203 | 11.08% | **11.33%** | 13.3 | yes |
| API | `Qwen3-VL-235B-A22B` | MoE 235B-A22B | VWA-reddit | `psom` | 203 | 12.56% | **10.34%** | 15.0 | yes |
| API | `Qwen3-VL-235B-A22B` | MoE 235B-A22B | VWA-classifieds | `ptext` | 224 | 14.96% | **10.27%** | 19.8 | yes |
| API | `Qwen3-VL-235B-A22B` | MoE 235B-A22B | VWA-reddit | `dom` | 203 | 12.81% | **9.85%** | 15.3 | yes |
| API | `Qwen3-VL-235B-A22B` | MoE 235B-A22B | VWA-reddit | `som` | 203 | 13.55% | **8.37%** | 16.2 | yes |
| API | `Qwen3-VL-235B-A22B` | MoE 235B-A22B | VWA-reddit | `ptext` | 203 | 11.58% | **7.39%** | 13.9 | yes |
| API | `Qwen3-VL-235B-A22B` | MoE 235B-A22B | VWA-reddit | `vision` | 203 | 7.39% | **4.93%** | 8.9 | **no — inventory only** |
| local | `Qwen3-VL-4B` | dense 4B | WA-reddit | `dom` | 104 | 19.23% | **9.62%** | 11.8 | yes |
| local | `Qwen3-VL-4B` | dense 4B | WA-reddit | `pprompt` | 104 | 16.83% | **8.65%** | 10.3 | yes |
| local | `Qwen3-VL-4B` | dense 4B | WA-reddit | `som` | 104 | 12.02% | **6.73%** | 7.4 | **no — inventory only** |
| local | `Qwen3-VL-4B` | dense 4B | WA-reddit | `ptext` | 104 | 18.27% | **5.77%** | 11.2 | yes |
| local | `Qwen3-VL-4B` | dense 4B | VWA-shopping | `som` | 432 | 7.99% | **5.32%** | 20.4 | yes |
| local | `Qwen3-VL-4B` | dense 4B | WA-reddit | `psom` | 104 | 12.98% | **4.81%** | 8.0 | **no — inventory only** |
| local | `Qwen3-VL-4B` | dense 4B | VWA-reddit | `dom` | 203 | 5.67% | **3.45%** | 6.8 | **no — inventory only** |
| local | `Qwen3-VL-4B` | dense 4B | VWA-classifieds | `dom` | 224 | 6.47% | **3.12%** | 8.6 | **no — inventory only** |
| local | `Qwen3-VL-4B` | dense 4B | VWA-reddit | `som` | 203 | 6.90% | **1.97%** | 8.3 | **no — inventory only** |
| local | `Qwen3-VL-4B` | dense 4B | WA-reddit | `vision` | 104 | 10.58% | **1.92%** | 6.5 | **no — inventory only** |
| local | `google/gemma-3-4b-it` | dense 4B | VWA-classifieds | `vision` | 224 | 2.01% | **0.45%** | 2.7 | **no — inventory only** |
| local | `Qwen3-VL-4B` | dense 4B | VWA-classifieds | `vision` | 224 | 12.50% | **0.00%** | 16.5 | yes |
| local | `Qwen3-VL-4B` | dense 4B | VWA-classifieds | `som` | 224 | 14.29% | **0.00%** | 18.9 | yes |
| local | `google/gemma-3-4b-it` | dense 4B | VWA-classifieds | `som` | 224 | 2.23% | **0.00%** | 2.9 | **no — inventory only** |

## 3. What this does and does not license

**Retracted — the serving-path grouping does not hold.** 5 local arm(s) sit at or above the lowest API floor (4.93%): WA-reddit `dom` 9.62%, WA-reddit `pprompt` 8.65%, WA-reddit `som` 6.73% (inventory), WA-reddit `ptext` 5.77%, VWA-shopping `som` 5.32%. The rule was declared before these replicates ran (`docs/checkpoints/pre_run/local_replicate_chain_launch_intent_20260915.md`, Reading 1: any powered local arm ≥ the API lower edge ⇒ the claim is dead as stated, retracted rather than hedged). What survives is the per-arm table in §2: floors differ by cell — benchmark/workload × backbone — and a two-group summary by serving path is not supported.

**Still true, narrower.** 实验笔记 §298.2: a controlled step-level probe on B1 (dense, local, temp=0) returned determinism 133/133 OK. The local group's near-zero floor is therefore not only a replicate-pair inference. That probe is on one VWA cell; it does not extend to the WA-reddit arms above.

**No mechanism.** none offered. 实验笔记 §302.5: the claim stops at an observable provider-dependent floor; 'MoE is the cause', 'switch provider', and 'provider bug' are all named unusable without a server-side audit artifact.

**Coverage gaps.**
- B2 (local, Gemma) carries no replicate: at its SR (0.45-2.23%) d~1.8, far below the bar — the local group cannot be given a second family by measuring B2, which is a power limit, not a scheduling one
- B5 has no reddit replicate yet (_b5_reddit_chain.sh is armed for it)
- the local group spans 4 site(s) only at INVENTORY grade: restricted to arms carrying an interval (d>=10.0) it covers 3 — VWA-classifieds, VWA-shopping, WA-reddit. Dropped at the bar: VWA-reddit. A cross-site claim about this group therefore rests on arms that were declared underpowered before they ran, and cannot be upgraded by pointing at the site count alone

## 4. Why it matters beyond this project

Web-agent benchmarks report success rates as point estimates. If a condition rerun through an API disagrees with itself on a tenth of its tasks, then any reported difference smaller than that is not distinguishable from repetition — and the overwhelming majority of agent evaluations are run through exactly such an API, once. With the serving-path grouping retracted, the statement is the plainer one: a rerun floor has to be measured per cell — a local backbone is not exempt.

