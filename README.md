<div align="center">

# P79: Cost-Aware Routing for Web Usage Agents

**A new frontier in web agent efficiency: discovering the phantom routing space.**

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Paper Status](https://img.shields.io/badge/paper-in_preparation-orange.svg)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
![arXiv](https://img.shields.io/badge/arXiv-TBD-b31b1b.svg)

</div>

## 👻 The Phantom Routing Space

We discovered a **phantom routing space** hiding on the boundary of *"agents that skip the annotated screenshot."* This space contains three sibling routing arms: `P-text`, `P-prompt`, and `P-SoM`. 

Our hero arm, **P-SoM** (deployment representative), exhibits a stunning **4-fold drop-in property**. The implication? You can route web agents to a representation that costs like DOM, runs like a sprinter, and signals like SoM — without ever rendering an annotated screenshot.

| Property | Claim | Evidence |
|---|---|---|
| **(a) Cost** | ≈ DOM | regex-filter the same AXTree; no bbox, no image |
| **(b) Latency** | ~50% lower than full SoM | cls SoM p95 `74s` → P-SoM `18.2s` (≈ 4× speedup) |
| **(c) Signal** | AUROC ≥ baseline | All 6 modes `overall_usable=True` |
| **(d) Drop-one oracle** | 1.7-3.3 pp | red P-SoM `3.33 pp` ≥ SoM `1.90 pp` |

## ✨ Highlights

- **Zero Image Overhead**: Achieve Set-of-Marks (SoM) level signaling purely through text prompts and AXTree parsing. We strip the visual dependency while maintaining the dense positional cues agents rely on.
- **Sub-20s Latency**: Dramatically accelerate your agent decision loops. By bypassing the heavy rendering and vision-encoder inference costs of traditional visual modes, P-SoM slices P95 latency by nearly 4×.
- **Robust Baselines**: Benchmarked rigorously across major model families and scales, including Qwen3-VL-235B (via proxy), Qwen3-VL-4B, and Gemma3-VL-4B.
- **Exhaustive Testing**: Built on an extensive test suite covering runner invariants, schema migrations, and analysis pipelines to guarantee paper-grade reproducibility.

## 🚀 Quick Start

Get the project running locally in under a minute.

```bash
# 1. Clone the repository and initialize the VisualWebArena fork
git clone --recursive <repo-url> Cost-Aware-Routing-for-Web-Usage-Agents
cd Cost-Aware-Routing-for-Web-Usage-Agents

# 2. Install dependencies (test suite profile recommended for full analysis)
pip install -e ".[test]"

# 3. Run the preflight check (CUDA, VWA endpoints, Auth state)
bash scripts/preflight_v2.sh

# 4. Launch an experiment (race-safe, watchdog-enabled)
make launch BASELINE=B0 SITE=reddit MODE=som
```

## 🧪 Experimental Setup

We evaluate web agents across **3 baselines** and **6 observation modes**, structuring our findings through rigorous statistical testing and multi-phase execution.

### Baselines (Cross-Family & Cross-Scale Control)
- **`B0`**: Qwen3-VL-235B-A22B (AWS proxy via Anthropic-style URL + OpenAI-style tools schema)
- **`B1`**: Qwen3-VL-4B (Local, bf16, ~10 GB VRAM)
- **`B2`**: `google/gemma-3-4b-it` (Gemma3-VL — cross-family 4B match for B1)

### Observation Modes
- `dom`: Viewport-only AXTree.
- `som`: `[SOM_MARKS]` text with bounding box screenshots.
- `vision`: Raw screenshots only.
- `phantom_text`: DOM prompt combined with `[SOM_MARKS]` text.
- `phantom_prompt`: SoM prompt combined with AXTree text.
- `phantom_som` (Hero arm): SoM prompt with `[SOM_MARKS]` text, bypassing the visual encoder entirely.

### Execution Phases
- **Phase 1a (Workshop Target)**: 42 conditions across 6 statistical cells. Includes Pass-1 baselines (36) and Pass-2 learned routers (6).
- **Phase 1b (Main Paper)**: Deferred expansion adding 18 conditions on the shopping site.

*Our primary statistical gate is a one-sided FE inverse-variance pooled superiority test (H₀: θ_FE ≤ +1.0 pp, α=0.05), projecting 97% power at k=6 cells.*

## 📚 Documentation Map

The repository operates on a 6-document architecture powered by an Obsidian data layer. Skip the noise and jump straight to the signal:

- [`docs/checkpoints/next_steps.md`](docs/checkpoints/next_steps.md) — The active daily action ledger.
- [`docs/checkpoints/paper_planning.md`](docs/checkpoints/paper_planning.md) — Core strategy, theory framework, and router design.
- [`docs/checkpoints/phase1_plan.md`](docs/checkpoints/phase1_plan.md) — Canonical execution plan for Phase 1.
- [`docs/checkpoints/PLAYBOOK.md`](docs/checkpoints/PLAYBOOK.md) — Operating manual and live status management.
- [`docs/checkpoints/paper_drafts/`](docs/checkpoints/paper_drafts/) — Final prose, sections 1-8, and bibliography.
- [`docs/checkpoints/实验笔记.md`](docs/checkpoints/实验笔记.md) — Append-only chronicle of the research journey.

*Tip: Check current Phase 1a active cells and GPU status by running `make active` or viewing `cells.base` directly. Do not hardcode state here.*

## 🖥 Runtime Tiers

Our compute architecture spans three specialized infrastructure tiers:

| Tier | Host Identity | Features | Target Workload |
|---|---|---|---|
| **DGX Spark** | `spark-9ea3` | aarch64 GB10 (~128 GB), shared GPU | Dev sessions, curation, archived data sources |
| **Condenser A100** | `a100-jiaming-test` | Dedicated A100-40GB, self-hosted VWA | **Paper-grade fire (Phase 1a/1b)** |
| **Myriad HPC** | `myriad.rc.ucl.ac.uk` | 4×A100 80GB, SGE qsub, wallclocks | Large-model batch processing, CPU analysis |

## 🛡 Hard Rules (Paper-Grade Hygiene)

To maintain pristine experimental conditions and prevent cross-contamination, you **must** strictly follow these load-bearing rules:

1. **Never bypass `make launch` or queue scripts.** Do not run `python scripts/run_experiment.py` directly. Instead, rely on `queue_baseline.sh`, `queue_phantom_*.sh`, or `queue_phase1_paper_grade.sh`. Our queues handle race-safe resets, environment setups, idempotent skips, and watchdogs. Bare runners have historically caused paper-grade contamination.
2. **One baseline per site at a time.** (`B0` XOR `B1` XOR `B2`). Accounts and docker containers are shared across the pipeline. Running concurrent baselines on the same site causes session races, cart pollution, and watchdog authentication collisions. Use `queue_chain.sh` for safe sequential execution.
3. **One site-chain per host for paper-grade fire.** (`cls` XOR `red` XOR `shop`). Running multiple sites concurrently risks docker bridge contention and busy-wait timeouts (e.g., the documented B-1581 asyncio race).

## 📜 Citation & License

**Paper in preparation, targeting EMNLP main + workshop. Stay tuned.**

This project is part of active undergraduate thesis research. 
Released under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).