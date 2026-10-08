# Evidence handoff — observation representation and routing for web agents (P79)

Snapshot: **2026-10-08**. Audience: another AI (or person) reading this cold, with no access to the
repository. Everything needed is either written here or shipped in this package (§12 lists the files).

---

## 0. How to read this package

**This is a zero-preset handoff.** It states what was built, what was run, what was measured, how,
and what went wrong — and nothing about how the results *should* be framed. Specifically withheld,
on purpose:

- the current conference draft and its framing documents;
- the two workshop papers built on earlier subsets of this data, and the frame proposals and
  cross-AI frame reviews written along the way;
- two research-direction audits (2026-09-22) whose text delivers a framing verdict.

The lab notebook (`05_experiment_log/`) and the adjudicated conclusion files (`03_conclusion_layer/adjudicated_*.md`)
are included in full because they are the record of what was run and decided; they also record past framing
discussions and choices. Read those parts as history, not as guidance.

Reviewer comments on those workshop papers *are* included (`08_external_feedback/`), because they
are external evidence of what readers asked for, not our interpretation. If you are asked to
propose a framing, form it from §3–§8 and the products, not from any narrative implied by the order
of this document. The order below follows the measurement pipeline, not an argument.

**Rules the numbers follow.**

1. Every number names the analysis product it was read from (`products/<name>.md|json`, in
   `01_products/`). The product's own header states its estimator, data scope and caveats; read it
   before reusing a number.
2. Almost every analysis after the preregistered primary test is flagged
   `post_hoc_exploratory=True` in its product. Treat them as exploratory.
3. Before reusing any number from the conclusion layer or the experiment log, check the retraction
   lists (§10, `03_conclusion_layer/retracted.md`, `retracted_2.md`). Roughly 330 earlier statements
   were retracted or corrected over the project.
4. Cost is **comparable within a cell only** (API dollars and local-serving proxy dollars are
   different quantities; §2.6).
5. "Cell" = one site × one backbone. "Arm" or "mode" = one observation representation (§2.3).
   "Run a" = the run every analysis reads; "run b" = a registered same-condition rerun.

---

## 1. The study in one paragraph

A single, deliberately plain web agent (one model call per step, no planner, memory, reflection or
retry policy) is run on self-hosted copies of VisualWebArena (VWA: classifieds, reddit, shopping)
and WebArena (WA: reddit), with four vision-language backbones served either through an API or
locally. The independent variable is the **observation representation** the agent receives each
step: six modes that cross *what text the page is given as* (accessibility tree vs a flattened list of
set-of-marks elements), *which prompt family is used* (DOM vs SoM), and *whether a page screenshot is
attached* (none, raw, or set-of-marks annotated). Outcomes are binary task success from the
benchmarks' evaluators, per-episode cost, latency, energy, step-level behaviour, and failure
attribution. Identical conditions were rerun to measure run-to-run variation. On top of the outcome
matrix, many per-task decision policies were evaluated offline: learned and rule-based routers that
choose a representation per task, cascades, abstention, early abort, and step-budget control,
including one prospective (pre-frozen) test.

---

## 2. Apparatus

### 2.1 Benchmarks, sites, task sets

Source: `01_products/benchmark_eda/corpus_eda.md`.

| corpus | corpus size | − N/A (removed at load) | run set | − protocol exclusions | **scored set** | distinct templates | tasks with reference image |
|---|---:|---:|---:|---:|---:|---:|---:|
| VWA classifieds | 234 | 10 | 224 | 0 | **224** | 75 | 68 (29.1%) |
| VWA reddit | 210 | 5 | 205 | 2 (tasks 160, 58) | **203** | 87 | 84 (40.0%) |
| VWA shopping | 466 | 31 | 435 | 3 (463, 465, 345) | **432** | 152 | 169 (36.3%) |
| WA reddit | 106 | 2 | 104 | 0 | **104** | 21 | 0 |

- N/A tasks are `string_match` tasks whose reference answer is "N/A"; the agent prompt has no N/A
  exit, so they are unpassable and removed at load (preregistered).
- Protocol exclusions are applied at analysis time only (AMENDMENT_08, AMENDMENT_10; see `06_preregistration/`):
  one reddit task whose evaluator never checks the requested action, one cross-site task answerable from
  memory, three shopping tasks with substrate defects.
- Templates: a template is one question shape with different slot fills. Task diversity is well below
  task count (e.g. WA reddit: 104 tasks, 21 templates).
- WA ships no reference images and no difficulty annotations; VWA ships human difficulty labels
  (reasoning / visual / overall) and reference images for 29–40% of tasks.
- Evaluators are binary (`evaluator_score_granularity`: over 36 conditions / 7,686 scored episodes the only
  observed scores are 0 and 1). Evaluator families by site are in `corpus_eda.md` §3.
- VWA-reddit and WA-reddit are different task sets on the same application (Postmill).

### 2.2 Backbones

| id | model | serving | where run |
|---|---|---|---|
| B0 | Qwen3-VL-235B-A22B | hosted API (proxy), temperature 0 | classifieds, reddit, shopping, WA reddit |
| B1 | Qwen3-VL-4B-Instruct | local, pinned revision, greedy | classifieds, reddit, shopping, WA reddit |
| B2 | Gemma-3-4B-it | local, pinned revision, greedy | classifieds, reddit |
| B5 | GPT-5.6 (mid-price "terra" tier) | hosted API, structured output | classifieds only |
| (B3/B4, MiMo-VL) | pilots only | — | abandoned; not in any product |

Local paper-grade serving was on one A100 40 GB VM. B2 solves 0–4% of tasks per mode on every cell (a
measured capability floor, not a bug: experiment log §338, §346).

### 2.3 The six observation modes

| mode | text payload | prompt family | page screenshot |
|---|---|---|---|
| **DOM** | accessibility tree | DOM | none |
| **P-prompt** | accessibility tree | SoM | none |
| **P-text** | flattened set-of-marks element list | DOM | none |
| **P-SoM** | flattened set-of-marks element list | SoM | none |
| **SoM** | flattened set-of-marks element list | SoM | set-of-marks annotated screenshot |
| **Vision** | empty | Vision | raw screenshot |

("P-" = "phantom": a screenshot-free control. `phantom_dom` in configs is a legacy alias of P-text.)
SoM − P-SoM differs only in the annotated screenshot (same marks, same prompt). A task's own reference
image, when it has one, is given in **every** mode. System prompts for each mode are in `02_data/prompts/`.

### 2.4 Agent and run contract

From the run configurations (`02_data/configs/`) and code:

- One model call per step: thought + action + arguments. Prompt = task instruction, mode system prompt,
  last 8 steps (thought, action, result), current observation.
- Greedy decoding (temperature 0), ≤ 4,096 new tokens per step. Viewport 1280×720; observation captured
  0.5 s after each action. Text observation never truncated.
- Nominal step budget 30; three consecutive unparsable outputs end an episode.
- Seed 42. Sites are reset before every run. **Within a run, only classifieds resets between tasks**
  (`require_reset` is a no-op on reddit and shopping in the upstream environment), so reddit and
  shopping episodes can see state left by earlier episodes (§8.3).
- GPT-5.6 receives one extra prompt sentence ("Emit exactly ONE JSON object for the single next action …").
- Every episode records code version, configuration and environment snapshot.

### 2.5 What is recorded per episode and per step

Per episode: success, steps, billed cost, token counts, latency (total, canonical — minus retries and
busy-waits — and model-call share), energy (local), finish/abort reason, failure bucket.
Per step: action, parse validity, dispatch path, page-changed flag, URL, latency split, energy.
Screenshots and page snapshots ("artifacts") exist for the A100 runs.
Field-level reliability was audited (`_data_quality_audit.md`): 10 fields are dead (always None),
some are conditionally missing per mode, and `page_changed` has false positives.

### 2.6 Cost, latency, energy

- **Billed cost** (`total_billed_cost_usd`) = API charge for B0 and B5. For B1/B2 it is a token-priced
  proxy for local serving whose constant was derived for a different accelerator (DGX Spark: hardware
  amortisation + electricity) while the runs ran on an A100 (`local_cost_estimand_audit.md`); the
  audit shows it is not a single multiplier — within-cell cost orders change under GPU time.
- **Wall-clock** (`total_latency_canonical_ms`) and **GPU time** (Σ `latency_ms.backend_infer`, local
  backbones only) are the two alternative cost bases used in sensitivity analyses.
- Energy is wall-clock in other units: per-step energy vs latency r = 0.966–0.9998 over 24 conditions,
  power ≈ 66.3 W (`energy_carbon_audit.md`).
- Model call share of latency: 22–28% for B0, 32–58% for B1 (`latency_decomposition.md`); the rest is
  environment (page loads, waits).
- Within a cell the three deployment modes' mean costs differ by 12–78% on the eight core cells (e.g.
  cls_B0 6.48–7.24 US cents; WA-reddit B1 4.47–7.94) and by up to 126% on shopping B1 (per-cell values:
  `02_data/episodes.csv`, `routing_three_arm.json` → `cells[].frontier.fixed_modes`). On shopping, SoM is cheaper than DOM (product-dense pages make the
  accessibility tree longer than the marks list).

---

## 3. Data volume

Sources: `04_run_inventory/` (`README.md`, `run_matrix.md`, `run_inventory.json`), merge summary.

| quantity | value |
|---|---|
| unique (run, condition) entries on disk | 131 |
| full runs (complete episode set) | 88 (87 in the site × backbone × mode matrix) |
| episodes across all inventoried runs | 20,199 (18,957 in full runs) |
| steps in the 8 core cells' canonical runs (6 modes each) | 200,975 over 9,984 episodes |
| registered same-condition rerun pairs | 24 (+1 pilot draw) |
| merged run store | 1,340,115 files, 105 GB (the A100 store with screenshots/DOM artifacts is 83.7 GB of it) |
| per-episode table shipped here (`02_data/episodes.csv`) | 18,446 rows (13,942 run-a rows + 4,504 rerun rows) |

**Run matrix — success rate (%) of every full run over the scored set.** Tags: P = core manifest,
X = extension manifest, R = registered rerun, U = unregistered (WA/shopping are discovered by glob, so U
there is not a defect). Two values in one cell = a same-condition pair exists.

| site | backbone | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|---|
| VWA·classifieds | B0 | 17.4 P / 15.2 R | 27.2 P / 29.5 R | 25.0 P / 24.1 R | 15.6 P / 14.3 R | 19.6 P / 17.0 R | 15.6 P / 14.3 R |
| VWA·classifieds | B1 | 6.2 P / 6.7 R | 14.3 P / 14.3 R | 12.5 P / 12.5 R | 7.6 P | 6.7 P | 6.7 P |
| VWA·classifieds | B2 | 1.3 P | 2.2 P | 2.2 P | 0.4 P | 1.8 P | 0.9 P |
| VWA·classifieds | B5 | 23.7 X / 25.0 R | 37.0 X | 12.0 U (broken run) | 24.1 X | 21.9 X | 22.8 X |
| VWA·reddit | B0 | 14.3 P / 11.3 R | 14.8 P / 12.3 R | 7.4 P / 7.4 R | 13.3 P / 9.8 R | 12.3 P / 9.8 R | 10.8 P / 14.3 R |
| VWA·reddit | B1 | 5.9 P / 5.4 R | 7.4 P / 6.4 R | 2.5 P | 5.9 P | 5.4 P | 5.9 P |
| VWA·reddit | B2 | 3.9 P | 1.0 P | 2.0 P | 2.0 P | 0.0 P | 0.5 P |
| VWA·shopping | B0 | 11.3 X | 14.8 X | 14.3 X | — | — | — |
| VWA·shopping | B1 | 4.9 X | 7.6 X | 5.6 X | 6.7 X | 5.3 X | 4.6 X |
| WA·reddit | B0 | 26.9 U | 22.1 U | 19.2 U | 35.6 U | 26.0 U | 25.0 U |
| WA·reddit | B1 | 16.4 R / 22.1 R | 13.5 R / 10.6 R | 9.6 R / 11.5 R | 16.4 R / 20.2 R | 16.4 R / 17.3 R | 11.5 R / 14.4 R |

**Eleven cells.** The eight with every mode in one run and used by the cross-site suite: classifieds and
VWA-reddit × {B0, B1, B2}, WA-reddit × {B0, B1}. Extensions, never pooled with the eight: classifieds B5
(five modes; its Vision run is the broken coordinate-contract run, §8.2), shopping B0 (three modes, no
screenshot-free arms: paid API budget) and shopping B1 (six modes).

**Rerun coverage.** Every mode rerun: cls_B0, red_B0, WA-red B1. Partial: cls_B1 (3/6), red_B1 (2/6),
cls_B5 (DOM). None: cls_B2, red_B2, shop_B0, shop_B1, WA-red B0. Five of eleven cells therefore have no
run-to-run reference at all.

**Not on disk / unavailable.** Several pre-protocol-reset replicate directories referenced in May
planning are empty (`phase0b_noise_floor.md`); per-step artifacts exist only for A100 runs; the B1
shopping SoM rerun is partial (283/435 episodes as of 2026-09-24, partial, unregistered).

**In flight (2026-10-08, A100).** A chain relaunched 2026-10-07 17:37 UTC: B2 classifieds SoM (≈134/233
done at 07:24 UTC 10-08, SR ≈3%, matching the canonical 2.2%) → B2 classifieds Vision → B1 shopping SoM
rerun (resume) → B2 reddit DOM. These are reruns that would add rerun coverage to cls_B2 and shop_B1.

---

## 4. What was done, in order

Full chronology: `05_experiment_log/chronology_index.md` (all 561 numbered section titles of the lab notebook,
with dates and tags) and the notebook itself (`05_experiment_log/experiment_notebook.md`, Chinese).

| period | what happened |
|---|---|
| 2026-04 | Agent scaffold, six-mode observation pipeline, VWA/WA hosting, B0 API agent; dozens of harness bug fixes (focus loss, confirm dialogs, set-of-marks option injection, state-change detection). First three-mode classifieds analyses. |
| 2026-04 → 05 | Internal mechanism study (linear probes and activation patching on B1 hidden states across modes; on a university cluster). **Shelved 2026-05-14** (§9.1). |
| 2026-05-13 → 05-18 | Preregistration written and deposited (OSF; `06_preregistration/preregistration.md`), hypotheses H1–H11 (§6). Very large multi-AI code audit before the first paper-grade fire (bug IDs B-6xx–B-17xx). |
| 2026-05-20 → 05-25 | "Protocol Reset" (AMENDMENT_01): accounting, schema, coordinate contract (AMENDMENT_05), set-of-marks identifier contract (AMENDMENT_07). Phase 1a fired on the A100. |
| 2026-05-25 → 07 | 36 core conditions (2 sites × 3 backbones × 6 modes) landed; per-condition failure diagnosis (`/diag`: deterministic rules + model-assisted deep dives); first replicate pairs; learned-router experiments (v1→v7 designs, all negative against fixed modes); WA reddit added (six modes, two backbones). |
| 2026-07-27 → 08-06 | Ledger/conclusion-layer rebuild; evidence-layer products (most of `01_products/cross_sites/`); scored-set amendment (AMENDMENT_08); a workshop submission (REALM). |
| 2026-08 | More replicates (B0 classifieds all six modes, B0 reddit, B1 partial), GPT-5.6 on classifieds (five modes), shopping (B0 three modes, B1 six), abstention / early abort / cascade analyses, behaviour-noise references. AMENDMENT_10 (shopping substrate). |
| 2026-09-09 | 11-cell router pilot ("§505", `07_router_pilot/` + `01_products/cross_sites/one_step_lookahead_2026-09-09.md`): LLM-router pilot, one-step lookahead, contextual bandit, three-arm collapse, budget routers. Budget router frozen and tested prospectively on shopping B1. |
| 2026-09 | MSc dissertation and showcase built on the data; second workshop paper accepted (VLM4RWD); WA-reddit B1 six-mode reruns (09-15 chain). |
| 2026-10-05 → 10-08 | All run copies merged and inventoried; every cross-site product checked for cell coverage (0 gaps); new products: representation-routing frontier against random mixtures, deployable-router bounds, task×mode interaction null + decision study, cross-run/template validation, extension cells (B5, shopping), three-arm versions, cost-basis sensitivity. |

---

## 5. Outcome layer

### 5.1 Success by mode

Run-a success rates are the P/X/U rows of the matrix in §3. Paired contrasts with task-bootstrap 95% CIs
(`fusion_premium.md`, comparators fixed a priori):

| cell | n | SoM − Vision | SoM − DOM |
|---|---|---|---|
| cls_B0 | 224 | +2.23 [−2.68, +7.59] | **+9.82 [+3.57, +16.07]** |
| cls_B1 | 224 | +1.79 [−2.68, +6.25] | **+8.04 [+3.57, +12.95]** |
| cls_B2 | 224 | +0.00 [−2.68, +2.68] | +0.89 [−1.34, +3.57] |
| red_B0 | 203 | **+7.39 [+2.46, +12.32]** | +0.49 [−3.94, +4.93] |
| red_B1 | 203 | **+4.93 [+1.48, +8.87]** | +1.48 [−1.48, +4.43] |
| red_B2 | 203 | −0.99 [−3.45, +1.48] | −2.96 [−5.91, −0.49] (−1.48 [−3.45, +0.49] after removing 3 inherited-state successes, `leakage_sensitivity.md`) |
| WA-red B1 | 104 | +3.85 [−1.92, +9.62] | −2.88 [−9.62, +2.88] |
| WA-red B0 | 104 | +2.88 [−5.77, +11.54] | −4.81 [−12.50, +2.88] |

- Best single mode per cell: SoM on most VWA cells (tied with Vision on cls_B2); DOM on VWA-red B2;
  DOM, P-text and P-prompt tied on WA-red B1 (16.4%); **P-text (35.6%) on WA-red B0**; SoM on cls_B5 (≈37%). Shopping: SoM ≥ Vision > DOM on both
  backbones, filtered or not.
- Representation classes (`representation_class_comparison.md`): no-image (four modes) / vision-only /
  hybrid (SoM). The four no-image modes are not behaviourally homogeneous: they meet an 83% consistency
  bar on 0 of 26 behaviour metrics over 8 cells (Vision 9, SoM 5).
- Per mode × 26 metrics × 4 dimensions (outcome, macro behaviour, micro decisions, efficiency):
  `per_mode_four_dimension_profile(_with_wa).md`.
- Universal-fail share (no mode solves): 56.7% (cls_B0) to 92.9% (cls_B2) (`abstention_learnability.md`).

### 5.2 The screenshot alone (matched contrast)

SoM − P-SoM isolates the annotated screenshot. A zero-cost regex over the task intent flags tasks
asking about an image, photo, colour or a count of things shown, excluding tasks that carry a reference
image (`visual_intent_routing.md`; regex in `02_data/tasks.csv` flags and the product):

```
\b(image|picture|photo|screenshot)\b|\bcolou?r of\b|\bhow many\b[^.]{0,40}\bin (?:the|this)\b
```

Flags 71/224 classifieds, 63/203 VWA-reddit, 56/432 shopping, 5/104 WA-reddit tasks.
Provenance: committed 2026-07-27 as a reddit failure-diagnosis pattern, first used as a partition
2026-08-03; classifieds hits were never inspected (out-of-sample on classifieds).

| cell | flagged: SoM − P-SoM | other tasks |
|---|---|---|
| cls_B0 | **+25.35 [+14.08, +36.62]** (n=71) | +5.23 (n=153) |
| cls_B0 rerun (both modes rerun) | **+26.76 [+14.08, +39.44]** (23 vs 4 successes) | — |
| cls_B1 | +11.27 | — |
| cls_B2 | +2.82 | — |
| cls_B5 (GPT-5.6) | **+29.58 [+16.90, +42.25]** | — |
| red_B0 | +3.17 | +4.29 |
| red_B0 rerun | +1.59 | −3.57 |
| red_B2 | SoM and P-SoM solve 0 flagged tasks (Vision 4, DOM 1) | — |
| shopping | ≤ 5 successes per mode on flagged tasks | — |

Routing on this rule (`rule_routing_pareto.md`): rules such as "flagged → Vision, else DOM" land on the
(success, cost, latency) frontier on cls_B0 but do not beat always-SoM on success.

### 5.3 Oracle / ceiling quantities

- Six-mode "any mode solves" ceiling vs best single mode (`routing_ceiling.md`, inherited-state
  successes zeroed): cls_B0 27.23 → 43.30; WA-red B0 35.58 → 51.92; WA-red B1 16.35 → 30.77;
  red_B0 14.78 → 26.11; cls_B1 14.29 → 24.55; red_B1 6.90 → 11.82; cls_B2 2.23 → 7.14; red_B2 2.46 → 5.91.
  The ceiling is a six-arm union against a one-arm baseline; arm-matched columns ("+1 arm", "rerun once")
  are printed beside it.
- Six reruns of one mode vs six distinct modes on cls_B0 (`rerun_union_extrapolation.md`): six-mode oracle
  43.30%; modelled six-rerun unions 25.6–39.8% depending on the mode.
- Supply and value (`supply_value_coupling.md`): across the 8 cells the share of tasks solved by >1 mode
  tracks the best single-mode SR (Spearman ρ = 0.952).

---

## 6. Preregistration and its outcomes

Document: `06_preregistration/preregistration.md` (substance-locked 2026-05-18, OSF), amendments 01–10
and protocol notes 01–06 in the same folder.

| hypothesis | what it tested | outcome |
|---|---|---|
| H1 (single primary gate) | P-SoM as a "hidden routing arm": fixed-effects pooled drop-one effect > +1.0 pp over the six core cells | **Failed** at k = 6 cells: θ_FE = 0.79 pp, one-sided p = 0.807 (drop-one 0.0–1.3 pp per cell) |
| H2(a) | P-SoM cost within 1.20× of comparators (by-construction property with falsification check) | not falsified (5/5 within band at the time read) |
| H3(i)/(ii) | unique-solve structure along the text and prompt axes | below the run-to-run noise floor |
| H10 | learned classifier router Pareto non-dominance in ≥ 5 of 6 cells | not met; later learned routers dominated always-cheapest in 0/6 cells (§7.2) |
| H4–H8, H9, H11 | exploratory / deferred to a second paper | not gating |

The preregistered routing gate (H10) and the later work in §7 use different estimands; §7 products are
all post hoc.

---

## 7. Decision-policy layer (routing, cascades, abstention, budget)

### 7.1 Is there per-task structure beyond difficulty, and can a single run measure it?

`task_mode_interaction_null.md` (six modes) and `routing_three_arm.md` (DOM/Vision/SoM only), on the three
fully rerun cells. Null: logit P(success) = task difficulty + (mode, run) easiness, no persistent
task×mode interaction; approximated by curveball MCMC conditioned on all row and column totals.
Statistic: σ²_int = (MS_int − MS_err)/2 from the task×mode ANOVA with two runs.

| cell | six modes: σ²_int (null q95), p | three modes: σ²_int (null q95), p | single-run reliability (6 / 3 modes) | runs to reach 0.5 (6 / 3; 3-mode 95% range) |
|---|---|---|---|---|
| cls_B0 | 0.0252 (0.0097), 0.0005 | 0.033 (0.015), ≤0.002 | 0.29 / 0.34 | 3 / 2 (2–4) |
| red_B0 | 0.0073 (0.0055), 0.0165 | 0.018 (0.009), ≤0.002 | 0.14 / 0.31 | 6 / 3 (2–7) |
| WA-red B1 | 0.0267 (0.0090), 0.0005 | 0.030 (0.014), ≤0.002 | 0.46 / 0.50 | 2 / 2 (1–3) |

Holm across cells: all pass. Caveat (not resolved): rejecting this additive (Rasch-type) null does not
distinguish a task-specific preference from a mode whose success responds more steeply to difficulty.
Reliability is that of the averaged interaction profile, not of a router's chosen arm.
Secondary readings: tasks with a stable strict preference (some mode 2/2, another 0/2) exceed the null
(cls_B0 54 vs 41.5; red_B0 24 vs 17.1; WA-red B1 19 vs 11.0); the high flip rate on contested tasks
does **not** exceed the null.

Rerun label instability (`label_instability.md`, cls_B0): 86/224 tasks change outcome in at least one
rerun arm; 81.8% of tasks whose arms disagree flip, vs 10.3% of tasks no or every mode solved.
That enrichment is about what difficulty arithmetic predicts (an earlier "2.20× above the floor" claim was retracted).

### 7.2 Learned routers on pre-run features

Features (18): 14 intent-keyword indicators, intent length, 3 first-page statistics (VWA-only
annotations — reasoning difficulty, has-reference-image — are dropped for cross-site comparability; the
20-feature VWA versions are reported separately).

- **Label supply** (`router_label_supply_diagnosis.md`): a which-mode label exists only on tasks some
  mode solved: 97 (cls_B0), 53, 55, 24, 16, 15 labelled tasks in the six VWA cells; pooled 260.
- **Triage** (solvable vs hopeless; `router_triage_learnability(_with_wa).md`): out-of-fold AUROC 0.73
  (cls_B0), 0.78 (red_B0), 0.73 (cls_B1), 0.86 (red_B1), 0.64, 0.62 (B2) with the 20 VWA features; the
  single best feature (usually the benchmark's reasoning-difficulty annotation) is within 0.02 of the full
  model on classifieds B0/B1, 0.09–0.13 behind it on reddit B0/B1, and ahead of it on both B2 cells. More training data does
  little (`router_undersampling_control.md`: ≈ +0.05 AUROC from 25% → 100% of training rows on the cells
  shown). Adding the visual-difficulty annotation changes AUROC by −0.013 to −0.000 (`visual_difficulty_router.md`).
  The `has_reference_image` feature has the opposite sign to the intuitive one (`routing_feature_diagnostics.md`).
- **Two-arm action label** (`two_arm_action_learnability.md`): per-task oracle over {best, cheap} beats the
  per-class assignment by +6.70 pp (cls_B0), +3.45 pp (red_B0); its positive class is 2.2–14.4% of tasks.
- **Pooled cross-backbone, cost-tier labels** (`router_pooled_tier_learnability.md`): dominates
  always-cheapest in 0/6 cells; dominated by the fixed-mode menu in every cell. Earlier offline chain: 0/6
  Pareto wins; most-favourable-corner retest 0/26 dominating, 0/26 on the frontier.
- **Frontier against fixed modes and their random mixtures** (`representation_routing_frontier*.md`,
  `routing_three_arm*.md`, `routing_gain_upper_bounds.md`, `routing_crossrun_template_validation.md`,
  `routing_extension_cells.md`). Two router constructions: per-arm success heads (cheapest mode whose
  predicted success ≥ τ) and triage (predicted hopeless → cheapest mode, else best mode). Frontier = upper
  concave envelope of the fixed modes (random mixtures). "Curve max" picks the best threshold after seeing
  test outcomes and is tested against a label-shuffle null that refits the whole curve; "deployable" picks
  the threshold on training folds only.

  | setting | curve max, per-arm (p) | curve max, triage (p) | deployable per-arm | deployable triage |
  |---|---|---|---|---|
  | 3 modes, 8 cells, billed cost | +0.86 (0.072) | **+1.69 (0.001)** | +0.23 | +0.59 |
  | 3 modes, 6 cells without B2, billed | +1.15 (0.071) | **+1.85 (0.001)** | +0.50 | +0.39 |
  | 3 modes, 8 cells, wall-clock | +1.28 (0.077) | +0.68 (0.418) | +0.10 | −0.48 |
  | 3 modes, 5 local cells, GPU time | +1.61 (0.076) | +1.24 (0.387) | +1.47 | +0.05 |
  | 6 modes, 8 cells, billed | +0.67 (0.252) | +1.24 (0.043) | −0.38 | −0.39 |
  | GPT-5.6 classifieds (5 modes) | +0.45 (0.768) | +3.86 (0.127) | −2.45 | −1.63 |
  | shopping B0 clean / all (3 modes) | +1.14 (0.062) / +1.09 (0.049) | +0.20 / +0.18 | −0.18 / +1.30 | −0.87 / −0.05 |
  | shopping B1 clean / all (6 modes) | +1.11 (0.143) / +0.48 (0.169) | +1.63 / +0.69 | +0.56 / −0.43 | +1.19 / +0.40 |

  Units: success-rate points above the frontier; pooled rows average cells on a cost axis normalised per
  cell (cheapest mode = 0, dearest = 1). Deployable intervals come from a task bootstrap that re-selects
  the threshold each draw; in two rows the 95th percentile falls below the point estimate (e.g. 3-mode
  per-arm +0.23 with upper +0.22), so the intervals are treated as magnitudes, not tests. Template-grouped
  folds and fit-on-run-a / score-on-run-b give point estimates of both signs (`routing_crossrun_template_validation.md`).
  Persistence diagnostic: using each task's cheapest succeeding mode from run a and scoring it on run b
  beats the six-mode frontier by 2.4–10.5 pp on red_B0 and WA-red B1 (uses the task's own label; not a
  deployable policy).
- **Earlier baselines** (`router_covariate_baseline_2026-07-05.md`, `router_objective_ordering.md`,
  `confidence_cascade(_with_wa).md`): kNN beats the locked logistic regression but stays negative;
  only 2 of 6 cells were trainable under the preregistered minimum-class rule in July.

### 7.3 Cascades, abstention, early abort

- **Confidence cascade** Vision → SoM (`confidence_cascade_with_wa.md`): an oracle escalation buys +2.2 to
  +10.8 pp for +2–12% cost (escalating 2–22 tasks); the product then measures how much of that each
  deployable confidence signal recovers, against random and oracle escalation. All cascade outcomes are offline splices (a real cascade would start the rich episode on a site the cheap
  episode already changed).
- **Abstention** (`abstention_learnability.md`, `abstention_site_transfer.md`): the "did any mode solve
  it" label exists on every task (224 vs 97 labelled rows on cls_B0); cross-site transfer judged against a
  200-permutation null.
- **Early abort from the first k steps** (`early_abort_B0_classifieds.md`, B0 classifieds): AUROC 0.50–0.67
  at k = 3–10 for the DOM and SoM rows;
  the 0.877 "routing AUROC" in older summaries aggregates the whole episode and is not available at step k.

### 7.4 The 11-cell router pilot (2026-09-09, "§505")

`01_products/cross_sites/one_step_lookahead_2026-09-09.md` (25 sections) + `07_router_pilot/`. Section
headings carry the readings; highlights as written there:

- the run-to-run flip is not visible at step 0; the model's first output adds ≈ 0 AUROC over model-free
  pre-flight features;
- a contextual bandit replay does not reach a fixed arm; collapsing to three arms changes no verdict;
- richer labels (an upper-bound proxy for an LLM grader) help a little on classifieds, nothing on reddit;
- the failure *cause* is stable across reruns but does not say whether another mode would succeed;
- on GPT-5.6, capability raised the mode main effect, not the interaction; the noise floor did not move;
- hand-written rules forward and mined rules backward both land within ±1 pp;
- task difficulty pools across backbones and sites; mode fit does not;
- switching the decision variable from "what to observe" to "how long to run" (step budget) is where the
  savings are: across 18 replicate pairs a learned pre-flight budget lost −2.15 pp vs −3.58 (fixed cap) and
  −4.06 (random) at −41% cost (14/18 pairs in the expected direction).

### 7.5 Prospective test: pre-run step-budget router (shopping B1)

Frozen 2026-09-09 before the held-out arms' outcomes were read
(`06_preregistration/budget_router_prospective_shop_B1_20260909.{md,json}`); identification interval
`01_products/vwa_shopping/budget_router_identification.md`. A capped success whose final state was reached
at an unknown step is treated as an interval.

| held-out arm | policy | learned − fixed cap | learned − random | holds for every completion? | cost saved |
|---|---|---|---|---|---|
| P-text (n=216) | two-tier (primary) | [−0.46, 0.00] | [−0.71, −0.41] | no | 53% |
| P-prompt (n=432) | two-tier (primary) | [−2.08, −0.93] | [−1.51, −1.05] | yes | 44% |

(Negative = learned loses less success than the comparator.) No rerun band exists for shopping B1.

---

## 8. Validity layer: noise, harness defects, contamination

### 8.1 Run-to-run variation (24 registered pairs)

`noise_floor_inventory.md`; per-task discordance between two runs of one (backbone, site, mode):

- B0 (API), classifieds, all six modes: 10.3–14.3%. B0 reddit: 4.9–11.3%. B5 classifieds DOM: 12.9%.
- B1 (local, greedy), classifieds: Vision 0.0%, SoM 0.0%, DOM 3.1%; reddit SoM 2.0%, DOM 3.5%;
  **WA-reddit all six modes 1.9–9.6%**.
- Overall 0–14.3%. Under exchangeability, SD(ΔSR) between two runs = √d / n (0.0–3.0 pp per pair).
- Serving path vs model (`serving_mode_floor.md`): API arms 4.9–14.3%, local 0–9.6%; the groups
  overlap. A preregistered reading that the floor is set by the serving path (C1) was **retracted** on
  2026-10-06 when WA-reddit B1 local arms landed at 9.6/8.7/5.8% (≥ the 4.93% kill line in the intent file).
- Cited cause of API variation: inference nondeterminism even at temperature 0 (batch-invariance
  literature). Local greedy B1 is deterministic on classifieds Vision/SoM but not on WA-reddit.
- Rerun flips (`rerun_flip_failure_anatomy.md`): failure bucket of tasks failing in both runs agrees
  72.4% (κ = 0.62 pooled; 0.36–0.63 on stochastic arms); flipping runs share a median ~4% of their
  prefix (part from early steps).
- 20 of 25 behaviour metrics have a cross-mode spread larger than their rerun band on cls_B0
  (`replicate_metric_noise.md`).
- "Retry vs switch" (`retry_vs_switch_label_supply.md`): at the one-arm margin on cls_B0, adding a
  distinct representation buys 0.6–4.3× what one rerun buys, depending on the base arm.

### 8.2 Harness defects that touch the data

Full catalog: `09_harness_defects/master_bug_catalog.md` (678 numbered entries; index in `bug_index.md`).
Ones that bear on outcomes:

- **B-2002 (shopping search box)**: typing into a box that already holds a query submits old + new text;
  43% of such submissions in shopping (< 1% on classifieds/reddit), far more in text modes than Vision.
  "Clean" shopping cells drop every task affected in **any** mode (B0 keeps 368/432, B1 192/432) — a filter
  that depends on agent behaviour under the compared conditions; results are reported filtered and unfiltered.
- **Shopping state**: category grid order differs between runs (9 runs in 4 states; 42 order-sensitive
  tasks), wishlist not reset between tasks (B-2003; 4 tasks are 0 in all conditions).
- **B-1997**: GPT-5.6 Vision run used a broken coordinate contract (SR 12.0%); excluded.
- **B-1998**: when a model emitted several valid actions, the first was executed but the step was booked as
  a free wait. GPT-5.6: 49–89 such steps per condition (1.2–2.4% of steps; some episodes reached 31–32
  counted steps); B2: 2–24; B0/B1: 0.
- **B-1999**: a failure-diagnosis rule (P31) silently exempts half of the budget-exhausted classifieds
  failures (URL path comparison); affects diagnosis counts, not success.
- Element-identifier contract (AMENDMENT_07) and coordinate contract (AMENDMENT_05) changes split runs into
  pre/post generations; only post-fix runs are canonical.
- Dispatch path (`dispatch_path_audit.md`): action success 88.9% via element-id locator, 38.6% via
  coordinate click, 16.1% via framework fallback; the mix differs by mode, so representation affects how
  many budgeted steps actually act.
- Off-site navigation (`offsite_navigation_audit.md`): reddit posts link to the live internet; 1–2% of
  reddit steps and 3–6% of reddit episodes leave localhost; 0 on classifieds.

### 8.3 Inherited state (reddit)

- Sidebar-scored tasks (`reddit_sidebar_leakage_audit(_with_wa).md`, `leakage_sensitivity.md`): 6 scored
  successes were credited without the episode visiting the target forum (B0 DOM 1, B0 Vision 1, B1 SoM 1,
  B2 DOM 3). Zeroing them changes one paired contrast (red_B2 SoM − DOM, above).
- Broader persistent-state audit (`persistent_state_leakage_audit.md`): among VWA-reddit successes scored on
  persistent state, **leaked** = credited on state the episode did not create: red_B0 11/18, red_B1 8/15,
  red_B2 3/4; WA-reddit B0 0/39, B1 0/22. The WA zero is a lower bound (the check is "visited the forum",
  and a confirmed case exists of arriving at an already-subscribed forum).
- Persistent state can also reproduce across reruns of the same task order, so reruns do not certify that
  a stable outcome is the agent's own.

### 8.4 Other data-validity checks

- Scored-set amendment sensitivity (`amendment08_sensitivity.md`): classifieds untouched by construction;
  reddit SR moves ≤ 0.4 pp in the rows shown there.
- Steps file ↔ summary identity (`steps_summary_identity_audit.md`) and page-change-corrected metrics
  (`page_change_corrected.md`).
- Power (`power_analysis.md`): per-cell minimum detectable effect ≈ 7.3 pp (classifieds), 7.7 (reddit),
  5.2 (shopping) at 80% power for a paired McNemar design with conservative discordance.

---

## 9. Behaviour and failure layer

- Failure buckets per cell × mode (`failure_modes_per_cell.md`): early-finish / wrong-commit dominates
  (e.g. 63–70% of B0 classifieds failures in DOM, P-SoM, P-prompt), then search loops, click loops, budget exhaustion.
- Cross-mode failure signatures (`cross_mode_failure_signatures.md`, 7,686 episodes, ruleset v11):
  budget exhausted 49.8%, degenerate walk-fail 43.9%, perception-missing loops 43.5% of episodes, with
  mode spreads up to 37 pp.
- Conditional attribution (`conditional_failure_attribution.md`): tasks only a text arm solves vs only an
  image arm solves — pooled 109 vs 101 (text has four arms, image two).
- Per-condition diagnosis digests: 58 (`diag_digest_index.md`; classes agent-limit / scaffold-bug /
  benchmark false positive).
- Behaviour axes (`axis_effect_size*`, `axis1_microbehavior*`, `mechanism_per_task_report.md`): text-axis and
  prompt-axis decompositions of DOM → P-SoM, URL-trajectory and click-transition Jaccard (≈0.3), first-action
  divergence; decision-quality differences exceed macro-frequency differences on most cells.
- Deployment properties (`representation_deployment_profile.md`): share of failures with a named
  mechanism 87–93% on B0 classifieds.
- Per-attempt vs per-success efficiency (`outcome_efficiency.md`): the cheapest and the fastest mode per
  *success* differ from those per *attempt* on 3 of the 6 VWA cells (all three reddit cells).
- Is latency a separate axis from cost (`multimetric_pareto(_with_wa).md`): cheapest ≠ fastest on 3 of 6
  VWA cells; ρ(cost, latency) ranges −0.60 to +0.77.

### 9.1 Internal-mechanism results (shelved)

`mechanism_evidence.md` (frozen; work shelved 2026-05-14): linear probes separate all 15 mode pairs at
AUROC 1.000 on B1 hidden states (both sites) — judged an uninformative test (linear readability vs
magnitude); activation patching along the prompt-family axis with random-injection and task-shuffled
controls shows real displacement 0.23–0.34 vs random 0.99, convergence 0.19 vs controls 0.09–0.16.

---

## 10. Retracted and corrected claims (selection)

The full lists are `03_conclusion_layer/retracted.md` (§1–§397) and `retracted_2.md` (§398–§527); the
recurring error patterns (M1–M23) are catalogued there. Items an analyst is most likely to rediscover:

- **H1 / "P-SoM is a hidden routing arm"** — failed its preregistered test (§6).
- **Serving-path noise floor (C1)** — retracted 2026-10-06 (§8.1).
- **"Label flips are 2.20× above the floor"** — unit mismatch (a six-arm union against a one-arm floor);
  corrected ratio 0.83–0.94×.
- **High routing AUROC (0.877)** — whole-episode aggregation; not available before or early in an episode.
- **Names that over-claimed in-sample quantities** ("Bayes ceiling", "interaction", "mode-invariant") —
  retracted (pattern M7).
- **A max-over-curve percentile bootstrap** gave invalid 8–12 pp "upper bounds" — replaced by the
  cross-fitted deployable estimand.
- **Template bootstrap** in the cross-run product labelled resampled templates by draw position — fixed;
  intervals moved, points unchanged.
- **Gemma fusion disadvantage on reddit** — depended on three inherited-state successes (§5.1).
- Several earlier "frames" were adopted and abandoned (2026-08-02, 08-03); the evidence summary written then
  (`01_products/cross_sites/EVIDENCE_LAYER_SUMMARY.md`) records why, but its §5b frames are dead by its own banner.

---

## 11. Known gaps and constraints

- No rerun at all for 5 of 11 cells; B2 and shopping cells can only be read without a run-to-run
  reference (pending A100 chain may add cls_B2 SoM/Vision and shop_B1 SoM).
- GPT-5.6 on one site only; its Vision run unusable; paid API budget prevents more.
- Shopping B0 lacks the screenshot-free modes (paid budget).
- No sequential cascade was ever run; all cascade numbers are offline splices.
- No 2PL-type (mode-specific slope) null for the interaction test; deployable-policy intervals are
  bootstrap percentiles with known bias.
- The project's conclusion layer is transcribed from the notebook; it was never recomputed from artifacts
  as a whole. Products in `01_products/` are recomputed from runs.
- Venue context (fact only): a long-paper submission to an ACL-family venue via ACL Rolling Review is due
  2026-10-12 (8 pages body, unlimited appendix).

---

## 12. Package map

| folder | contents |
|---|---|
| `00_START_HERE.md` | this document |
| `01_products/` | the analysis reports (`.md`, 150) and result files (`.json`, 76) under the repository's `docs/analysis/`, minus the two withheld audits; `products_index.md` lists each report with its title and opening lines |
| `02_data/` | `episodes.csv` (one row per cell × mode × run × task: success, cost, wall-clock, GPU seconds), `features.csv` (18 router features per cell × task), `tasks.csv` (template id, intent-rule flag, shopping clean-set membership), `prompts/` (per-mode system prompts), `configs/` (run configurations, comments stripped, endpoints redacted), `DATA_README.md` |
| `03_conclusion_layer/` | the project's adjudicated / measured / retracted conclusion files (Chinese) and the atomic ledger (`ledger.jsonl`) |
| `04_run_inventory/` | run inventory (`run_inventory.json`, `run_matrix.md`, `README.md`), product coverage and scope registry |
| `05_experiment_log/` | the full lab notebook (Chinese, ≈3.5 MB) and `chronology_index.md` |
| `06_preregistration/` | preregistration, amendments 01–10, protocol notes, the frozen budget-router prospective test and intent files |
| `07_router_pilot/` | scripts and outputs of the 2026-09-09 11-cell router pilot |
| `08_external_feedback/` | reviews received by the two earlier workshop submissions |
| `09_harness_defects/` | the master bug catalog and its index |

Columns of `02_data/episodes.csv` are documented in `02_data/DATA_README.md`; recomputing any cell's
per-mode success rate and mean cost from it reproduces the products exactly (checked 2026-10-08: 0
mismatches over 11 cells × 3 deployment modes).
