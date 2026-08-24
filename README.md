<div align="center">

# Cost-Aware Routing for Web-Usage Agents

**When does a web agent actually need expensive multimodal context — and can that need be predicted cheaply enough to route representations per task?**

![Python](https://img.shields.io/badge/Python-research%20stack-3776AB?logo=python&logoColor=white)
![Domain](https://img.shields.io/badge/domain-Web%20Agents%20%C2%B7%20Evaluation-7c3aed)
![Method](https://img.shields.io/badge/method-preregistered%20evaluation-0f766e)
![Result](https://img.shields.io/badge/result-negative%20router%20result%2C%20mechanism%20identified-b45309)
[![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

*MSc dissertation research · UCL · Web agents · representation routing · evaluation reliability*

**[Interactive project walkthrough](https://quarkgluonmixture.github.io/Cost-Aware-Routing-for-Web-Usage-Agents/portfolio/)**

</div>

---

## The result in one paragraph

Web-agent representation needs are **state-dependent**: an oracle that can choose the right observation mode per task has a real upper bound over any fixed representation. But under the measured regime, a learned router does **not** recover that value cheaply enough to beat a trivial fixed policy: in nested cross-validation, **0/6 evaluated cells** Pareto-dominate `always-cheapest`.

The failure is not simply "the classifier was weak." The bottleneck is **label supply**. A useful routing label only appears once the underlying task has been solved well enough to reveal which representation mattered, while base task success in the studied cells is only **2–27%**. The project therefore ends with a boundary result: the routing opportunity is real, but the supervision needed to learn it is structurally scarce in the regime measured here.

> **Thesis headline:** expensive multimodal context is sometimes necessary, but predicting *when* it is necessary can fail for structural reasons even when the oracle ceiling is demonstrably real.

Canonical writing anchor: [`final_dissertation/THESIS_ONE_SENTENCE.md`](final_dissertation/THESIS_ONE_SENTENCE.md).

---

## Why this is interesting

A web agent can observe the same page through very different representations:

| Mode | What the model receives | Image required? |
| --- | --- | ---: |
| `dom` | accessibility / DOM-derived text | No |
| `vision` | raw screenshot | Yes |
| `som` | Set-of-Marks screenshot + matching text | Yes |
| `phantom_text` | SoM-style text under DOM prompting | No |
| `phantom_prompt` | DOM text under SoM prompting | No |
| `phantom_som` | SoM positional cues as text, without the annotated image | No |

The last three form a **phantom routing space**: representations that keep parts of the Set-of-Marks signal while removing the rendered annotation image. That factorization lets the experiments ask a cleaner question than "vision or no vision?": **which part of the representation is helping, on which tasks, and is that variation learnable?**

---

## What the experiments establish

### 1. There is something worth routing

Across the preregistered evaluation cells, the oracle representation portfolio improves success rate by roughly **+3.45 to +16.07 percentage points** while using **13.7–35.3% less cost** than the strongest fixed reference in the corresponding comparison.

That matters because a failed learned router is only scientifically interesting if the oracle first proves that a routing opportunity exists.

### 2. The learned router does not recover the ceiling

Under true nested cross-validation, **0/6 cells** produce a learned policy that Pareto-dominates the trivial `always-cheapest` fixed strategy.

This README intentionally does **not** turn that into a claim that representation routing is impossible in general. The result is scoped to the observed benchmark/model regime.

### 3. The bottleneck is supervision availability

The strongest diagnosis is not "pick a better classifier." Routing labels are generated only by informative solved tasks. At low base success rates, the routing problem becomes label-starved before model selection becomes the main issue.

### 4. The representations are complementary, not substitutes

`phantom_som` is **not** claimed as a cheaper drop-in replacement for full Set-of-Marks. Its value is that it exposes a distinct signal axis and can add portfolio value in tasks where other modes fail.

---

## Research design

The repository is deliberately built around **claim → evidence → failure-mode** discipline rather than a single leaderboard number.

### Factorized representation study

```text
                 prompt style
              DOM            SoM
           ┌──────────┬──────────┐
DOM text   │   dom    │ phantom_ │
           │          │  prompt  │
           ├──────────┼──────────┤
SoM text   │ phantom_ │ phantom_ │
           │  text    │   som    │
           └──────────┴──────────┘

+ full SoM image
+ raw vision
```

This separates text content, prompting style, and image rendering instead of confounding them into one "multimodal" switch.

### Preregistered evaluation

The confirmatory analysis was locked before the main run. Sensitivity analyses are kept separate from the primary gate, and negative controls are used to distinguish content-specific gains from prompt-length or formatting artifacts.

### Cross-benchmark / OOD checks

VisualWebArena is the primary environment; Online-Mind2Web-style / WA-derived cells are used to test whether the observed structure survives beyond one site/task distribution. Cross-benchmark comparisons are treated as distribution shift, not casually pooled as if the feature spaces were identical.

### Mechanistic and diagnostic probes

The project includes hidden-state probes, activation-patching experiments, benchmark EDA, label-supply analysis, pass@k-style noise-floor checks, and targeted stress tests. Probes that turned out to be trivial or in-sample artifacts are explicitly retracted rather than recycled as supporting evidence.

---

## Engineering the experiment

The science sits on top of a fairly large reliability problem: live web environments are stateful, authenticated, slow, and easy to contaminate.

```text
configs / preregistration
          ↓
race-safe experiment queues
          ↓
VisualWebArena / web-agent runner
          ↓
schema-versioned episode records
          ↓
validation + contamination checks
          ↓
analysis / oracle / router / diagnostics
          ↓
claim-evidence matrix
          ↓
final dissertation
```

The harness includes:

- heterogeneous local/API model backends;
- per-site serialization and reset discipline;
- restart-safe JSONL logging and migrations;
- watchdogs for auth, quota, contaminated episodes, and failed runs;
- deterministic analysis entry points;
- explicit run-set vs scored-set accounting;
- mutation/regression tests for failure modes that previously produced silent false confidence.

The operational lessons are summarized in [`GOTCHAS.md`](GOTCHAS.md): many of the expensive failures in this project came from checks that *passed while checking the wrong thing*.

---

## Repository map

```text
p79/
├── agents/          web/VLM agent loops
├── backends/        local transformers + API backends
├── envs/            benchmark/environment wrappers
├── experiment/      runner, routing, metrics, logging, migrations
├── mechanistic/     hidden-state extraction and probes
├── policies/        fixed and learned routing policies
└── utils/           auth, CUDA, async and reliability helpers

scripts/
├── analysis/        paper-grade analysis and figure generation
├── queues/          launch / HPC / serialized run orchestration
├── maintenance/     watchdogs, cleanup, integrity checks
└── mechanistic/     probing / patching / localization experiments

final_dissertation/  thesis-level claim and writing control files
tests/               unit, regression, analysis and integrity tests
```

---

## Start here

If you are reading the repository as a researcher or reviewer:

1. [`final_dissertation/THESIS_ONE_SENTENCE.md`](final_dissertation/THESIS_ONE_SENTENCE.md) — problem, research question, headline answer.
2. `final_dissertation/CLAIM_EVIDENCE_MATRIX.md` — what is claimed, what supports it, and which old claims were retired.
3. [`GOTCHAS.md`](GOTCHAS.md) — reusable experiment / infrastructure failure modes.
4. `docs/checkpoints/pre_run/` — preregistration and locked analysis plan.
5. `scripts/analysis/` — executable evidence behind the paper figures and tables.

The README deliberately does **not** mirror the current run cursor, queue state, or temporary machine status. Those belong in the repository's handoff/checkpoint material, not in the public landing page.

---

## Quick start

```bash
git clone --recursive https://github.com/Quarkgluonmixture/Cost-Aware-Routing-for-Web-Usage-Agents.git
cd Cost-Aware-Routing-for-Web-Usage-Agents

pip install -e ".[analysis,dev]"
make test
make help
```

Reproducing the live benchmark requires the pinned VisualWebArena environment, model access, and the queue/orchestration layer. Do not launch paper-grade conditions through a bare runner: serialization and reset rules are part of the experiment contract.

---

## What this project does **not** claim

- `phantom_som` universally replaces full SoM;
- removing images is automatically cheaper in end-to-end dollars for every provider/model;
- the observed routing failure generalizes to every model or benchmark;
- token cost is equivalent to energy use or carbon emissions;
- a high in-sample probe score is evidence of a deployable router.

Keeping those boundaries explicit is part of the result, not a disclaimer bolted on afterwards.

---

## Author & license

**Jiaming Wei** · UCL MSc Artificial Intelligence for Sustainable Development

- Portfolio: [quarkspace.top](https://quarkspace.top)
- Project walkthrough: [interactive demo](https://quarkgluonmixture.github.io/Cost-Aware-Routing-for-Web-Usage-Agents/portfolio/)
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Built on [VisualWebArena](https://github.com/web-arena-x/visualwebarena) and related web-agent evaluation infrastructure.
