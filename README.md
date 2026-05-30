<div align="center">

# Cost-Aware Routing for Web-Usage Agents

**Can we route a vision-language web agent to a cheaper, faster *representation* of the page — without losing the signal it needs to act correctly?**

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![Tests](https://img.shields.io/badge/tests-1.1k_functions-blue.svg)
![Domain](https://img.shields.io/badge/domain-LLM_agents_·_VisualWebArena-8a2be2.svg)
![Status](https://img.shields.io/badge/research-in_progress-orange.svg)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

*Undergraduate thesis research · solo · ~117K LOC Python · 1,088 commits over 4 months*

</div>

---

## TL;DR

A web agent built on a vision-language model can perceive a page in several ways: raw screenshot (`vision`), accessibility tree (`dom`), or **Set-of-Marks** — a screenshot annotated with numbered boxes plus a matching text list (`som`). The annotated screenshot is the expensive part: it costs a full image's worth of vision-encoder tokens and the latency of rendering and encoding it.

This project asks whether you can keep the *signal* of Set-of-Marks while dropping the *image* — and whether an agent should **route** between these representations per task to trade off success rate against cost and latency. To answer it rigorously I built a **preregistered, reproducible evaluation harness** that runs three model families × six observation modes across three [VisualWebArena](https://github.com/web-arena-x/visualwebarena) sites, on a three-tier compute fleet, with mechanistic-interpretability probes to explain *why* a representation works rather than only *that* it does.

> **Honest status.** The central hypothesis (below) is **under active evaluation** — `classifieds` is complete across all six modes for the largest model; other cells are still firing. I report the harness, methodology, and preliminary signal here; confirmed effect sizes belong in the paper, not the README.

---

## The research question: a "phantom" routing space

Set-of-Marks sits at a boundary. On one side, the agent reads the annotated **image**; on the other, it reads only **text**. Right on that boundary lives a family of representations I call the **phantom routing space** — they carry Set-of-Marks-style cues but render **no annotated screenshot at all**:

| Mode | Text channel | Prompt style | Renders image? |
|---|---|---|---|
| `dom` | viewport AXTree | DOM | no |
| `som` (full Set-of-Marks) | `[SOM_MARKS]` list | SoM | **yes** |
| `vision` | — | vision | raw screenshot |
| `phantom_text` | `[SOM_MARKS]` list | DOM | **no** |
| `phantom_prompt` | AXTree | SoM | **no** |
| **`phantom_som`** (hero) | `[SOM_MARKS]` list | SoM | **no** |

**The hypothesis.** `phantom_som` is *designed* to be a drop-in for full Set-of-Marks that:

| Designed property | Why, by construction | How it's tested |
|---|---|---|
| **Representation cost** stays in the DOM class | no rendered image ⟹ zero vision-encoder image tokens; the text channel is a regex-filtered AXTree | image-token accounting; end-to-end *billed* dollars sit in a near-flat **~$0.064–$0.073 / episode** band across all six modes (`classifieds`/B0), so cost is a floor — latency and signal are where the modes actually diverge |
| **Latency** drops | skips screenshot render + vision-encoder forward pass | per-step wall-clock timing |
| **Signal** is preserved | Set-of-Marks positional cues are retained *as text* | confidence-signal AUROC vs. baseline modes |
| **Routing value** is real | adding it to a mode portfolio raises an oracle's ceiling | drop-one oracle lift, evaluated across 6 `(site × model)` cells |

This is the kind of claim that is easy to *assert* and hard to *earn*: the entire harness below exists to either confirm it across models and sites under a preregistered test, or to falsify it honestly.

---

## What this project demonstrates

A compact map from the work to transferable AI/ML research-engineering skills.

| Competency | Where it shows up here |
|---|---|
| **Experimental design** | Factorial design that decomposes `dom → phantom_som → som` into independent *text-axis* and *prompt-axis* sub-effects; selection-bias-controlled 2×2; content-specific (random-injection) negative control. |
| **Statistical rigor** | Preregistered primary gate = one-sided **fixed-effect inverse-variance pooled superiority test** (H₀: θ ≤ +1.0 pp); DerSimonian–Laird random-effects, HKSJ, and TOST equivalence as sensitivity analyses; paired-bootstrap SEs; explicit power analysis. |
| **LLM experimentation** | Three model families benchmarked head-to-head: a 235B MoE via API, a 4B VLM run locally in bf16, and a cross-family 4B (Gemma-3) as a matched-capability control. |
| **Mechanistic interpretability** | Activation patching, linear probes on hidden states, and layer-localization to find *where* a representation's effect lives inside the network — not just black-box outcomes. |
| **ML systems engineering** | Race-safe distributed experiment orchestration across three heterogeneous compute tiers, with watchdog auto-recovery, structured logging, and schema-versioned data. |
| **Reproducibility discipline** | Preregistration + OSF lock manifest, a software bill-of-materials for the benchmark fork, deterministic seeding, and 1,121 test functions guarding the analysis pipeline. |

---

## System architecture

The science is only as trustworthy as the harness. The repo is a ~117K-LOC Python package plus ~230 analysis/orchestration scripts.

```
p79/
├── agents/        # VLM inference loops (local Qwen3-VL, Gemma-3; API-proxy agent)   2.4K LOC
├── backends/      # backend abstraction: local transformers / API proxy / heuristic 1.8K LOC
├── envs/          # VisualWebArena wrappers, observation standardization             2.5K LOC
├── experiment/    # core engine: runner orchestration · routing · cost/energy/latency
│                  #   metrics · structured JSONL logging · schema migrations         15K LOC
├── mechanistic/   # activation patching · linear probes · hidden-state extraction
├── policies/      # rule-based + learned routers (feature extraction, manifest)
└── utils/         # CUDA workarounds, auth refresh, async helpers

scripts/
├── analysis/      # 99 scripts, incl. 28 paper-grade figure generators
├── queues/        # 53 scripts: race-safe launch queues + HPC qsub submission
├── maintenance/   # 67 scripts: watchdog, contamination cleanup, live-status sidecars
└── mechanistic/   # interpretability pilots (patching, probing, steering sweeps)
```

**Three-tier compute fleet.** Workloads are matched to hardware: an aarch64 GB10 dev box (Tailscale-tunneled), a dedicated A100 VM that self-hosts the benchmark Docker stack for clean paper-grade runs, and an SGE-batch HPC cluster for large-model and CPU-heavy analysis.

**Reliability under contention.** Web-agent runs share live Docker sites and authenticated accounts, so concurrency is a correctness hazard, not just a speed one. The harness enforces single-baseline-per-site locking, resets site state before every condition, and runs a **6-layer watchdog** (detect contamination → alert → refresh auth → clean the poisoned episode → re-run → verify) so that every episode that lands is clean by construction rather than by post-hoc filtering.

**Data integrity.** Every step is written as schema-versioned JSONL with `fsync` durability and restart-deduplication; a single canonical reader handles corrupt-line recovery; a migration framework lets the schema evolve without orphaning historical runs.

---

## Methodology & reproducibility

- **Preregistration first.** The primary hypothesis, gate, and analysis plan are locked in `docs/checkpoints/pre_run/preregistration.md` (with an OSF lock manifest) *before* the confirmatory runs — so the headline test cannot be chosen after seeing the data.
- **One primary gate, transparent sensitivity.** A one-sided FE inverse-variance pooled superiority test is the single confirmatory gate; random-effects, HKSJ, and TOST equivalence are reported as sensitivity analyses, and a K-of-N count is kept as transparency only (not a gate).
- **Controls against the obvious confounds.** A selection-bias-controlled 2×2 separates *format* from *prompt style*; a random-injection negative control checks that gains are content-specific rather than artifacts of longer prompts.
- **Reproducible by command.** The whole pipeline is one-command via a `Makefile` (`make analysis`, `make test`, `make launch …`); the benchmark itself is pinned as a git submodule fork with a documented software bill-of-materials.

---

## Quick start

```bash
# Clone with the VisualWebArena submodule fork
git clone --recursive <repo-url> Cost-Aware-Routing-for-Web-Usage-Agents
cd Cost-Aware-Routing-for-Web-Usage-Agents

# Install (analysis + dev extras pull in pandas/scipy/matplotlib + pytest)
pip install -e ".[analysis,dev]"

# Run the test suite (no GPU or live sites required — these tests are skipped by default)
make test

# Explore the analysis pipeline on shipped result artifacts
make help
```

Reproducing a full experiment additionally needs GPU(s), a running VisualWebArena Docker stack, and API credentials for the largest model; launches always go through the race-safe queue layer (`make launch BASELINE=… SITE=… MODE=…`), never the bare runner, to preserve clean-run guarantees.

**Tech stack:** Python · PyTorch · Hugging Face `transformers` / `accelerate` · `bitsandbytes` (bf16/quantized local inference) · OpenAI-style tool-calling over an API proxy · NumPy / pandas / SciPy / Matplotlib · Playwright-driven browser env · pytest · SGE/qsub.

---

## Repository tour

| Looking for… | Path |
|---|---|
| Core experiment engine | [`p79/experiment/`](p79/experiment/) |
| Routing policies (rule-based + learned) | [`p79/policies/`](p79/policies/) |
| Interpretability probes | [`p79/mechanistic/`](p79/mechanistic/) |
| Analysis + figure scripts | [`scripts/analysis/`](scripts/analysis/) |
| Preregistration & statistical plan | [`docs/checkpoints/pre_run/`](docs/checkpoints/pre_run/) |
| Paper drafts (sections 1–8) | [`docs/checkpoints/paper_drafts/`](docs/checkpoints/paper_drafts/) |
| Test suite | [`tests/`](tests/) |

---

## Status, author & license

- **Status:** active undergraduate thesis research; paper in preparation. The phantom-routing-space hypothesis is under evaluation across the six `(site × model)` cells.
- **Author:** `<your name>` · `<email>` · `<LinkedIn / portfolio>`  *(fill in before sharing)*
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Built on [VisualWebArena](https://github.com/web-arena-x/visualwebarena).
