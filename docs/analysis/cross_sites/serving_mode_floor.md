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
| **local** | 5 | Qwen | 2 | **0.00–3.45%** | 2 (0.00–0.00%) |

The groups **do not overlap**: the lowest API floor (4.93%) is 1.48pp above the highest local one (3.45%).

Exact one-sided rank test on a perfect split: **p = 0.0001** (1/8568 assignments at least this extreme). ⚠️ arms within a cell are not independent (shared site, backbone, task universe); descriptive separation statistic, NOT a gateable test.

Restricted to arms carrying an interval (d≥10): separated=**True**, p = 0.0110, gap 7.39pp.

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
| local | `Qwen3-VL-4B` | dense 4B | VWA-reddit | `dom` | 203 | 5.67% | **3.45%** | 6.8 | **no — inventory only** |
| local | `Qwen3-VL-4B` | dense 4B | VWA-classifieds | `dom` | 224 | 6.47% | **3.12%** | 8.6 | **no — inventory only** |
| local | `Qwen3-VL-4B` | dense 4B | VWA-reddit | `som` | 203 | 6.90% | **1.97%** | 8.3 | **no — inventory only** |
| local | `Qwen3-VL-4B` | dense 4B | VWA-classifieds | `vision` | 224 | 12.50% | **0.00%** | 16.5 | yes |
| local | `Qwen3-VL-4B` | dense 4B | VWA-classifieds | `som` | 224 | 14.29% | **0.00%** | 18.9 | yes |

## 3. What this does and does not license

**Licensed.** The floor groups by serving path across two unrelated model families (OpenAI, Qwen) and 2 site(s) on the API side. Before this, one API model meant *model* and *serving* were the same variable; the second family removes the reading that the floor is a quirk of one architecture.

**Not removed: scale.** serving mode covaries with scale (235B/undisclosed vs 4B). A second API family removes the FAMILY explanation, not the SCALE one. The experiment that would settle it is the same checkpoint served both ways — e.g. Qwen3-VL-4B through an API endpoint, or a 235B-class model self-hosted. Neither is in this project's compute envelope; naming it is the honest substitute.

**No mechanism.** none offered. 实验笔记 §302.5: the claim stops at an observable provider-dependent floor; 'MoE is the cause', 'switch provider', and 'provider bug' are all named unusable without a server-side audit artifact.

**Local side, independent corroboration.** 实验笔记 §298.2: a controlled step-level probe on B1 (dense, local, temp=0) returned determinism 133/133 OK. The local group's near-zero floor is therefore not only a replicate-pair inference.

**Coverage gaps.**
- B2 (local, Gemma) carries no replicate: at its SR (0.45-2.23%) d~1.8, far below the bar — the local group cannot be given a second family by measuring B2, which is a power limit, not a scheduling one
- B5 has no reddit replicate yet (_b5_reddit_chain.sh is armed for it)
- the local group spans 2 site(s) only at INVENTORY grade: restricted to arms carrying an interval (d>=10.0) it covers 1 — VWA-classifieds. Dropped at the bar: VWA-reddit. A cross-site claim about this group therefore rests on arms that were declared underpowered before they ran, and cannot be upgraded by pointing at the site count alone

## 4. Why it matters beyond this project

Web-agent benchmarks report success rates as point estimates. If a condition rerun through an API disagrees with itself on a tenth of its tasks, then any reported difference smaller than that is not distinguishable from repetition — and the overwhelming majority of agent evaluations are run through exactly such an API, once. The local column is what makes this a statement about the serving path rather than about benchmarks being noisy in general.

