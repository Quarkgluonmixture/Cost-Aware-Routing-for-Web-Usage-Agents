<!--
=====================================================================
AAAI-27 MAIN TRACK CONSOLIDATED DRAFT v0.1 (2026-07-01)
=====================================================================
Target: AAAI-27 Main Technical Track
  - Abstract deadline:      2026-07-21 (UTC-12)
  - Full paper deadline:    2026-07-28 (UTC-12)
  - Supplementary deadline: 2026-07-31 (UTC-12)
  - Format: 7 pages technical content + references + reproducibility
    checklist; AAAI two-column camera-ready style; double-blind.

Source corpus: docs/checkpoints/paper_drafts/section{1..8}_*.md
(working drafts, ~400KB) compressed to conference length. This file
is the AAAI submission master; the section files remain the working
corpus and are NOT superseded.

NUMBER PROVENANCE TAGS (used throughout; strip before submission):
  [A]   A100 canonical Phase 1a aggregate (paper-grade, pinned-HEAD)
  [P]   provisional — run complete, cron frontmatter SR only; NOT yet
        in cross-condition aggregates (rerun `make analysis` to lift
        to [A])
  [V]   archive-vintage (pre-fix DGX substrate; vintage denominators
        N=234/210) — motivation / Appendix-D sensitivity ONLY
  <TBD> pending data or pending gate verdict — do NOT paraphrase as
        if resolved

VERDICT SLOTS (data-conditional per prereg R1–R5; do not pre-commit):
  <H1-VERDICT>  FE pooled P-SoM drop-one vs +1.0pp — currently
                INSUFFICIENT_DATA (needs ≥2 complete 6-mode cells)
  <H3-VERDICT>  axis-1/axis-2 structural non-overlap FE gates
  <H10-VERDICT> router operational deployment gate — currently
                no_pass2_runs, gate=False
  Intro/abstract sentences marked (R-CONDITIONAL) must be rewritten
  per the realized R-tier before submission.
=====================================================================
-->

---
title: "The Phantom Routing Space: Cost-Aware Representation Routing for Multimodal Web Agents"
bibliography: ../paper.bib
---

# Abstract

Multimodal web agents built on Set-of-Marks (SoM) scaffolds consume a bundled observation: a screenshot annotated with numbered marks, a textual legend of those marks, and a prompt that describes the annotated image. This bundling convention hides a family of agent configurations that has never been evaluated in isolation. We study the boundary *skip the annotated screenshot*: holding the model, the action space, and the evaluator fixed, we vary only the text-payload format (hierarchical accessibility tree vs. flattened `[SOM_MARKS]` legend) and the prompt family (DOM-style vs. SoM-style), producing three screenshot-free arms — P-text, P-prompt, and their compound P-SoM — that we call the **phantom routing space**. On VisualWebArena Classifieds and Reddit, across three backbones (a 235B API model, a 4B local model of the same lineage, and a cross-family 4B model), we run a pre-registered six-mode evaluation against DOM, full SoM, and Vision baselines, asking whether these arms are degraded SoM or independent routing structure. On landed cells each arm opens a low-overlap success pool with non-empty unique coverage; the pre-registered fixed-effects pooled drop-one gate over the six (site, model) cells (<H1-VERDICT>: gated against a +1.0pp threshold) and per-axis structural gates (<H3-VERDICT>) determine the final claim tier. By construction the arms inherit DOM-level cost and skip the per-step screenshot-encoding stage; we verify the cost property with a pre-registered falsification check. Finally, we train a per-cell learned router that selects a representation per task and evaluate it under a pre-registered Pareto non-dominance deployment gate (<H10-VERDICT>). We do not claim the phantom arms replace full SoM; their value is complementarity — they widen the routing menu at DOM-equivalent cost.

<!-- Abstract ~230 words; AAAI CMT limit typically 250 -->

# 1 Introduction

A web agent's observation is a design decision. The same page can reach the model as a hierarchical accessibility tree (DOM-style text), as a raw screenshot (vision), or — in the Set-of-Marks (SoM) convention that dominates multimodal web-agent scaffolds [@yang2023som; @koh2024visualwebarena; @zheng2024seeact] — as a *bundle*: a screenshot annotated with numbered marks, a textual legend indexing those marks, and a system prompt that instructs the model to ground its actions in the annotated image. The bundle is treated as one mode. Benchmarks compare DOM vs. SoM vs. Vision as atomic alternatives; deployments pick one and pay its cost profile. The legend text inside the SoM bundle — the part that costs no screenshot tokens — is treated as an auxiliary index for the marked screenshot, not as a controlled variable in its own right.

This paper evaluates what the bundling convention hides. Starting from the DOM baseline, we vary two textual knobs while *removing the per-step annotated screenshot entirely*: (i) the **text-payload format** — the same accessibility-tree content, either in its native hierarchical serialization or flattened into the `[SOM_MARKS]` legend format; and (ii) the **prompt family** — the DOM-style prompt or the SoM-style prompt (which still describes an annotated screenshot the model never receives). The 2×2 ablation diamond yields three screenshot-free arms beyond DOM: **P-text** (`[SOM_MARKS]` under DOM-prompt), **P-prompt** (SoM-prompt over AXTree), and their compound **P-SoM** (`[SOM_MARKS]` under SoM-prompt) — the deployment-relevant representative that receives everything full SoM receives *except* the marked screenshot. We call the set of configurations on this *skip-annotated-screenshot* boundary the **phantom routing space**. Throughout, "screenshot-free" means no per-step page screenshot; task-supplied reference images, which are part of the task specification rather than the observation pipeline, are preserved identically in every mode.

At the start of this project, P-SoM looked like a broken ablation: a prompt that promises an image the model never sees, over a legend meant to index that image. The data reject that expectation. (R-CONDITIONAL) On both sites and within each backbone, the phantom arms behave as *distinct routing arms*: they solve tasks that DOM, full SoM, and Vision all miss, and they fail characteristic task sets of their own. Cross-mode success-pool overlap is far from complete (same-task Jaccard 0.29–0.49 on the archive substrate [V] — *above* the independence baseline of E[J] ≈ 0.06–0.10, as shared task difficulty predicts, yet leaving a non-empty unique-pass residue per arm); the unique-pass sets survive per-task inspection rather than collapsing into a single task family. We emphasize what we do **not** claim: P-SoM is not the best single arm on every site, and we do not claim it replaces full SoM. **Its value is complementarity.**

Because "solves tasks others miss" is a portfolio property, our principal metric is the **drop-one oracle**: the loss in oracle-ceiling success rate when one arm is removed from the six-mode portfolio, estimated per (site, model) cell with task-paired bootstrap and pooled across the six pre-registered cells by fixed-effects inverse variance. The pooled P-SoM contribution is the pre-registered hero gate of the paper (H1: one-sided superiority over a +1.0pp substantive threshold; <H1-VERDICT>). The drop-one number is an *oracle ceiling*, not a realized router gain; §6 reports how much of it a learned router recovers.

Two further properties make the space *deployable* rather than merely curious — and both are architectural consequences of the screenshot-off boundary, not empirical discoveries. First, **cost ≈ DOM by construction**: `[SOM_MARKS]` is a regex filter plus deterministic renumbering over the *same* accessibility-tree text the DOM baseline already consumes, with no page-screenshot tokens (and identical reference-image payloads on both sides of the comparison); we pre-registered a falsification check (per-task median cost ratio vs. DOM ≤ 1.20× per cell) rather than a category-error equivalence test. Second, **latency below full SoM**: the arms skip the per-step marked-screenshot encoding stage. On the archive substrate the classifieds P95 step latency was 18.2s for P-SoM vs. 74.0s for full SoM [V]; Phase 1a canonical recomputation is reported in §5.4. These properties position the phantom arms as additions to the routing *menu* at DOM-equivalent cost — which is exactly what a representation router needs.

We are not the first to deploy text-only, marked, or SoM-style observations; industrial agents mix such configurations freely for token economy. The gap we target is a **characterization gap, not a first-deployment gap**: no prior work isolates the annotated screenshot from its textual scaffold under a fixed model, fixed action space, and fixed evaluator, and asks whether the screenshot-free remainder constitutes independent routing structure.

**Contributions.**

1. **A controlled, pre-registered characterization of the phantom boundary** (§3–§5). Six observation modes — DOM, SoM, Vision, P-text, P-prompt, P-SoM — evaluated on VisualWebArena Classifieds (224 scored tasks) and Reddit (205 scored tasks) across three backbones: Qwen3-VL-235B (API), Qwen3-VL-4B (local), and Gemma-3-4B (local, cross-family). P-text and P-prompt serve as specificity controls that test whether the effect collapses into a single prompt trick or a single format swap (pre-registered structural gates H3(i)/(ii); <H3-VERDICT>). Hypotheses, gates, exclusion rules, and the data-conditional framing ladder were locked before the canonical data fire, with public witness.
2. **A behavioural hypothesis-generation observation** (§5.5): in a matched-task 2×2 ablation, text representation appears to shape *exploration* (flattened marks cut search-loop rate) while prompt wording appears to modulate *commitment timing* (SoM-prompt arms show smaller false-positive finish gaps). We advance this as behavioural characterization only; mechanism analysis is deferred to follow-up work.
3. **A representation-axis cost-aware router** (§6). In contrast to model-routing systems [@chen2023frugalgpt; @ong2025routellm], we hold the model fixed and route the *observation representation* per task: a per-cell logistic-regression router over 18 task features, trained with task-held-out five-fold cross-validation, evaluated against an intelligent-baseline ladder under a pre-registered two-layer Pareto non-dominance deployment gate (H10; <H10-VERDICT>). The router's label space is exactly the six-mode menu that the phantom space widens.

As a sister observation (exploratory, not a pre-registered hypothesis), the *utility* of the space is site-modulated: full SoM leads on the visually dense marketplace while screenshot-free arms are competitive on the text-dominated forum — a directional pattern on two sites, not a statistical correlation.

# 2 Related Work

**Observation modes for web agents.** Text-only agents consume DOM or accessibility-tree serializations [@zhou2024webarena; @deng2023mind2web]; multimodal agents add screenshots, most effectively with Set-of-Marks annotation [@yang2023som], which VisualWebArena adopts as its standard multimodal scaffold [@koh2024visualwebarena] and SeeAct examines for grounding [@zheng2024seeact]; successor systems inherit the bundle [@yang2025magma; @li2025ferretui2]. Across this literature DOM, SoM, and Vision are compared as atomic modes, and the SoM text legend exists only as the image's index. That convention — never isolating the legend from the image — is the gap this paper targets.

**Routing in LLM systems.** Existing efficiency routing operates on the *compute* axis: which model [@chen2023frugalgpt; @ong2025routellm; @webrouter2025], which capacity within an agent [@li2026pando], which numeric precision [@li2026dmr], which grounding expert [@li2026avenirweb], or how to schedule modality encoding in serving [@qiu2025modserve]. Agent-E selects among DOM-variant text views, but within one text modality and not learned per task [@dhondt2024agente]. Our router is, to our knowledge, the first *learned, per-task representation router within a single model* that crosses the screenshot-on/screenshot-off boundary. These axes compose rather than compete: a production stack can route representation first and escalate models on stall.

**Prompt-format sensitivity.** LLM performance is known to vary with semantically equivalent prompt formats [@sclar2024promptformat; @mishra2022reframing]; recent circuit-level work localizes format-conditional sub-computation and finds its separability is family- and scale-dependent [@feldhus2026judgecircuits]. This motivates treating prompt family as a first-class axis, and pre-registers a risk: our cross-family 4B backbone may not replicate the Qwen-lineage pattern (§8).

**Efficient observations and language priors.** FocusAgent prunes accessibility-tree text with a retriever [@kerboua2025focusagent]; ReVision prunes visual tokens with a learned keep-mask [@abaskohi2026revision]. Phantom arms are neither text pruning nor image scheduling: they *re-format the same text content* (within ±7% of DOM token length [V]) and drop the per-step screenshot entirely. Their viability is made plausible by evidence that VLMs often answer from language priors — high no-image accuracy [@asadi2026mirageillusionvisualunderstanding], modality-mention effects [@vu2026scaffold], image-occlusion invariance on 20–40% of VQA samples [@zhou2026visualignorance] — and that the bottleneck is often integration rather than perception fidelity [@liu2025seeing]. We use these as motivational anchors only; none is web-agent evidence, and we import no mechanism claim from them.

# 3 The Phantom Routing Space

## 3.1 Six modes, one substrate

All modes share the same VWA page state, action space, step budget, and evaluator; they differ only in the observation triple (prompt family, text payload, image). Table 1 fixes the construction.

| Mode | Prompt | Text payload | Per-step screenshot |
|---|---|---|---|
| DOM | DOM-prompt | AXTree (hierarchical) | — |
| P-text | DOM-prompt | `[SOM_MARKS]` (flattened) | — |
| P-prompt | SoM-prompt | AXTree (hierarchical) | — |
| **P-SoM** | SoM-prompt | `[SOM_MARKS]` (flattened) | — |
| SoM | SoM-prompt | `[SOM_MARKS]` (flattened) | marked screenshot |
| Vision | Vision-prompt | — | raw screenshot |

*Table 1: Observation construction. The four screenshot-free rows form the 2×2 ablation diamond over (text format × prompt family); P-SoM is the compound corner and full SoM is its screenshot-on endpoint. Vision anchors the screenshot-only extreme. Task-supplied reference images are preserved in all six rows.*

The `[SOM_MARKS]` payload is produced by a single anchored regex pass over the *same* accessibility-tree serialization the DOM baseline consumes: lines matching `^\s*\[(\d+)\]\s+\w` are kept, renumbered deterministically 1..K, and wrapped in `[SOM_MARKS]…[/SOM_MARKS]`; a seq→native-id map is retained solely for action dispatch. No bounding boxes are computed and no image work is done. The SoM-family text uses deterministic sequential selection ids, while the AXTree payload retains native node ids; P-prompt therefore differs from P-SoM in *both* structural format and identifier contract, and we report the P-prompt axis as a structural-region test rather than a clean prompt-only isolation.

The SoM-prompt in the screenshot-free arms still instructs the model to act on "an annotated screenshot with numbered boxes" — a *mirage prompt*. P-text and P-prompt exist precisely to decompose this compound: if P-SoM's behaviour were purely a prompt trick, P-prompt would reproduce it; if purely a format effect, P-text would.

## 3.2 By-construction deployment profile

Two properties follow from the construction, prior to any experiment. **(a) Cost ≈ DOM.** The token count of `[SOM_MARKS]` is bounded by the AXTree it filters (measured within ±7% on both sites [V]) and the arms emit no page-screenshot tokens; task-supplied reference images are encoded identically in every mode (they are task specification, not observation), so their cost cancels in each within-baseline comparison. Because this is a near-deterministic property of the input pipeline, we pre-registered a *falsification check* rather than an equivalence test: if any (site, model) cell shows a per-task median ratio cost(P-SoM)/cost(DOM) above 1.20×, the claim is contradicted and must be investigated. **(b) No per-step screenshot-encoding stage**, hence lower step latency than full SoM; this is characterized descriptively (§5.4), not gated.

**Scope.** The construct presupposes a screenshot-based SoM-style scaffold — an agent whose observation pipeline can produce the marks legend and whose cost profile contains a non-trivial image-encoding stage. Text-only, coordinate-only, and raw-pixel agents fall outside the class. Operationally, a target harness is drop-in compatible iff its accessibility-tree dump passes the anchored-regex check on ≥95% of first-step rows (VWA and WebArena pass; Mind2Web-style raw DOM requires an adapter).

**Construct-validity note.** Our pipeline numbers *all* valid accessibility-tree nodes (including static text), unlike the upstream VWA convention of numbering interactables only. This keeps all six modes on one element universe, so every within-paper differential cancels the convention; the cost is that *absolute* success rates are not comparable to external VWA leaderboards (§8).

# 4 Experimental Setup

**Benchmark and tasks.** VisualWebArena [@koh2024visualwebarena] Classifieds and Reddit, self-hosted with fresh containers. We exclude tasks whose gold answer is "N/A" at task load (73 across the suite), because the LLM judge's handling of unachievable tasks is a known false-positive source; the scored sets are 224 (Classifieds) and 205 (Reddit) tasks. The third VWA site (Shopping) is deferred to a follow-up expansion and is not part of any claim here.

**Backbones.** B0 = Qwen3-VL-235B-A22B via a commercial API (high-capability anchor); B1 = Qwen3-VL-4B, local (within-family capability contrast); B2 = Gemma-3-4B-it, local (cross-family direction test at 4B parameter parity — *parameter* parity, not matched capability). With three backbones the deployment-tier axis ({API} vs {local}) and the family axis ({Qwen} vs {Gemma}) are not mathematically separable; we scope B0-vs-B1 as a deployment-tier + capability contrast and B1-vs-B2 as a cross-family robustness direction test, and we pre-registered a claim-tier gate on the lineage cell taxonomy (Qwen-lineage = 4 statistical cells, {B0, B1} × {cls, red}; Gemma-lineage = 2 cells, B2 × {cls, red}): if either B2 cell fails the per-cell drop-one criterion while all four Qwen cells pass, the paper reports "Qwen-lineage verified; Gemma-lineage did not replicate" and the cross-family claim tier downgrades one step without collapsing the construct.

**Conditions and cells.** Pass-1: 2 sites × 3 backbones × 6 modes = 36 conditions. Pass-2: one learned-router condition per (site, backbone) cell = 6 conditions. Statistics are stratified at the 6 (site, backbone) cells. All conditions run sequentially per site with full container resets between conditions, per-task identity restoration on Reddit, fixed canonical task order, and a fail-closed quarantine protocol for transient infrastructure faults; the protocol and all deviations are witnessed in a public pre-registration with amendment log.

**Decoding and action interface.** Greedy decoding throughout (B0: temperature 0; B1/B2: `do_sample=False`); `max_new_tokens=4096` for all three. Actions use one shared semantic schema (8 actions) with backbone-appropriate serialization: native tool-calling for B0 (`tool_choice="required"`, schema programmatically identical to the validator) and structured JSON for B1/B2. This elicitation asymmetry is backbone-capability-driven and disclosed, not hidden (§8).

**Evaluation.** Upstream VWA evaluators with two audited patches, applied uniformly across all modes: a judge-polarity fix (`"incorrect" ⊄ "correct"` substring bug) and a deterministic-zero guard for empty predictions. The LLM judge is pinned (`gpt-4o-mini`, temperature 0). Canonical success is VWA-Success on the N/A-excluded scored set; no post-hoc adjustment layer is applied.

**Cost and latency estimands.** Cost is total billed cost per episode, *within-baseline only*: B0 cost is commercial-API USD while B1/B2 cost is electricity-derived USD, and the two bases differ by ~3 orders of magnitude per token — cross-baseline absolute-cost comparison would be a unit artifact, so all headline cost claims are within-baseline cross-mode ratios. Latency is retry-adjusted (network-retry backoff subtracted); raw latency is reported as a sensitivity column.

**Pre-registered gates.** H1 (hero): the fixed-effects inverse-variance pooled P-SoM drop-one contribution over the six planned cells exceeds +1.0pp, one-sided bootstrap-percentile test at α=0.05 (task-paired bootstrap, B=1000, per-cell; the six cells are the complete design, not a sample — no between-cell τ² is estimated; HKSJ random-effects reported as appendix sensitivity). H2(a): the cost falsification check above. H3(i)/(ii) (structural): FE-pooled unique-task contribution of the P-text axis and P-prompt axis (tasks solved by the axis arm but not by P-SoM) excludes zero. H10 (router, §6): a two-layer operational deployment gate. H10 is decoupled from the paper's framing ladder: its failure caps only the router section's deployment claim, never the §5 phenomenon claim.

# 5 Results I: The Phenomenon

<!-- Data status 2026-07-02 (post registry-promotion): [A] = 22-cell aggregate
     (B0/B1/B2 cls ×6 + B0-red dom/som/vision/P-text) — run_manifest.yaml promoted
     from A100 fire_manifest.json (22 bound conditions; P-text R32139 promoted
     2026-07-02), `make analysis` ingested;
     <TBD> = B0-red psom/pprompt (in flight), B1/B2-red (queued), all final
     verdicts. Interim pools (k=3 cls) recorded in the H1/H3 slots — do NOT
     promote them to verdicts. Sync recipe when new cells land: rsync A100
     fire_manifest.json → promote into run_manifest.yaml cells: → make analysis
     (promotion-gap watch now cron-automated: check_manifest_promotion_gap.py). -->

## 5.1 Single-mode success rates

*Table 2: Success rate (%) per mode and cell on the scored sets (Classifieds N=224, Reddit N=205). Provenance: [A] canonical aggregate; [P] provisional pending aggregation; <TBD> condition in flight.*

| Cell | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| cls·B0 [A] | 17.4 | **27.2** | 25.0 | 15.6 | 19.6 | 15.6 |
| cls·B1 [A] | 6.3 | **14.3** | 12.5 | 7.6 | 6.7 | 6.7 |
| cls·B2 [A] | 1.3 | 2.2 | 2.2 | 0.4 | 1.8 | 0.9 |
| red·B0 | 14.6 [A] | 14.6 [A] | 7.8 [A] | 13.7 [A] | \<TBD\> | \<TBD\> |
| red·B1 | \<TBD\> | \<TBD\> | \<TBD\> | \<TBD\> | \<TBD\> | \<TBD\> |
| red·B2 | \<TBD\> | \<TBD\> | \<TBD\> | \<TBD\> | \<TBD\> | \<TBD\> |

Three patterns are stable across the landed cells. First, on the visually dense marketplace, full SoM is the strongest single arm at every capability tier (B0: 27.2% vs. 17.4% DOM) — the phantom arms do *not* dominate, and single-mode SR is not the claim. Second, on the text-dominated forum the screenshot-free arms are competitive with the bundle: B0 Reddit DOM 14.6%, SoM 14.6%, P-text 13.7%, while Vision drops to 7.8% — differences among the leading arms sit within the B0 serving-noise floor (~±2.5pp net SR, measured on classifieds same-payload replay; §8) and we describe them as *comparable within sampling error*, not as ranking. Third, capability modulates everything: B1 preserves the top of the cls ordering (SoM strongest, Vision second) at roughly half the SR, though the phantom-arm tail reorders; B2 sits at a ~1–2% floor on cls — a genuine capability floor on this benchmark (six independent diagnostics; §8) — which mutes its within-cell contrasts and is why the cross-family claim runs through the pre-registered claim-tier gate rather than through pooling sleight-of-hand.

## 5.2 Drop-one oracle: each arm's irreplaceable coverage

For each cell, the six-mode oracle ceiling is the SR of the best arm per task; the drop-one loss of arm *m* is the ceiling drop when *m* is removed. On the landed canonical cell (cls·B0 [A]; ceiling 43.3%):

*Table 3: Drop-one loss (pp) with 95% task-paired bootstrap CIs, cls·B0 [A].*

| Arm | Drop-one loss | 95% CI |
|---|---|---|
| Vision | 4.02 | [1.79, 6.70] |
| SoM | 2.68 | [0.89, 5.36] |
| P-prompt | 2.68 | [0.89, 4.91] |
| DOM | 1.79 | [0.45, 3.57] |
| P-text | 0.89 | [0.00, 2.23] |
| P-SoM | 0.89 | [0.00, 2.23] |

Two readings matter. (i) The six-mode ceiling (43.3%) is far above the best single arm (27.2%): over a third of the benchmark's oracle-solvable mass (36 of 97 tasks) lies outside the best single arm, which is the existence proof for representation routing on this cell. (ii) Every arm — including all three phantom arms — carries a positive point-estimate of irreplaceable coverage, with P-prompt's CI excluding zero on this cell. On the other two complete cells the pattern repeats at lower amplitude: P-SoM drop-one is +1.34pp [0.00, 3.12] on cls·B1 and +0.45pp [0.00, 1.34] on the cls·B2 floor [A]. The pre-registered hero gate is *not* a per-cell statement: H1 pools the P-SoM drop-one across all six cells by fixed-effects inverse variance and tests it against +1.0pp. <H1-VERDICT: PARTIAL_DATA as of 2026-07-01 — interim pool over the k=3 complete cells (all classifieds) gives θ_FE = +0.98pp, 95% CI [−0.05, 2.00], *below* the +1.0pp bar on classifieds alone; the k=6 verdict hinges on the three pending Reddit cells (archive calibration [V]: pooled +2.34pp, I²=0%, with Reddit the strongest P-SoM site). Populate from `phase1_prereg_gate.md` when Reddit completes; do NOT quote the interim pool as the verdict.>

The structural gates ask whether the space is genuinely two-dimensional: H3(i) tests whether P-text solves tasks P-SoM does not; H3(ii) the same for P-prompt. <H3-VERDICT: PARTIAL_DATA — interim FE pools over landed complete cells: axis-1 +3.20pp [1.58, 4.82] (k=2; cls·B2 excluded by the ≥2-task per-cell noise floor) and axis-2 +2.26pp [1.14, 3.38] (k=3), both interim CIs excluding 0; final verdict at k=6.> On cls·B0 [A], both axis unique-sets are non-empty under the pre-registered estimand: P-text solves 9 tasks P-SoM misses (4.02pp) and P-prompt 16 tasks (7.14pp) — pairwise non-redundancy against the compound arm, a distinct and looser quantity than Table 3's drop-one irreplaceability against the full portfolio.

## 5.3 Complementarity, not average superiority

Cross-mode success-pool overlap is far from complete: same-task Jaccard between mode pairs spans 0.29–0.49 on the archive substrate [V]. This is *above* the independence baseline (E[J] ≈ 0.06–0.10 at the observed SRs) — modes agree more than chance, exactly as shared task difficulty predicts — so the complementarity claim rests not on low overlap but on the unique-pass residue that survives it, and those unique-pass sets are distributed across task categories (search, comparison, navigation) rather than concentrating in one family. Concretely, on archive Reddit, P-SoM uniquely solved 7 tasks no other mode touched [V]; on canonical cls·B0 the P-SoM unique set is 2 tasks [A] — smaller, which is precisely why the paper's claim is gated on the *pooled* estimate rather than any single cell, and why we report per-cell unique counts as transparency, never as a K-of-N consistency argument. Modes with similar average SR can still visit different pages and fail in different basins; the routing value lives in the residual unique coverage, not in the mean.

## 5.4 Cost and latency: the constructed substrate holds

On the canonical B0-cls cell [A], per-episode billed cost is descriptively flat across arms: DOM $0.0696, P-text $0.0692, P-prompt $0.0685, P-SoM $0.0721, SoM $0.0724. The pre-registered H2(a) check is stricter than this marginal comparison — the per-task median of *paired* cost ratios cost(P-SoM)/cost(DOM) against a 1.20× falsification bound — and on the three landed complete cells it stands unfalsified: per-task median ratios 1.01 (cls·B0), 1.04 (cls·B1), 1.08 (cls·B2) [A]; remaining cells <TBD>. Note the *absolute* flatness of B0 cost across screenshot-on/off arms is an artifact of the API's pricing bundle; the by-construction claim is about the input pipeline (no page-screenshot tokens, same filtered text), which the falsification check operationalizes. On local backbones, where the image channel is directly measurable, removing the marked screenshot saves 733 (Reddit) and 1064 (cls) image tokens per step [V, B1]. Latency: on the archive substrate, cls P95 step latency was 18.2s (P-SoM) vs. 74.0s (SoM) [V]; canonical retry-adjusted recomputation <TBD>. These are architectural consequences being *verified*, not discoveries being made.

## 5.5 A behavioural two-knob observation (hypothesis-generating)

On a matched-task Reddit subset (N=48 [V]), the two knobs separate behaviourally: flattening the text payload cuts the search-loop rate roughly in half (DOM 22.7% → P-text/P-SoM 10.8%), while the SoM-prompt arms show a smaller false-positive finish gap than DOM-prompt arms (2.1pp vs. 6.3pp, measured under a since-retired success-adjustment layer — direction-only evidence) — *text representation appears to shape exploration; prompt wording appears to modulate commitment timing*. We advance this strictly as behavioural characterization on an archive substrate; it generates the hypotheses a mechanism study would test, and no mechanism claim is made here.

## 5.6 Site-modulated utility (exploratory)

Across the landed cells the *existence* of phantom-arm coverage appears on both sites, but its *utility* is site-modulated: the marketplace rewards the screenshot (SoM and Vision lead cls at every tier), while the forum is text-dominated (on red·B0 the landed screenshot-free arms match the bundle — DOM 14.6% vs. SoM 14.6% — with the remaining phantom arms <TBD>). With two sites this is a directional pattern — it maps *where* the space pays, not *why* — and it is exactly the situation a site-aware representation router exploits.

# 6 Results II: A Learned Representation Router

**Design.** One logistic-regression head per (site, backbone) cell over 18 features (selected per fold by mutual information from a 50-feature pool: 30 TF-IDF over the task intent + 5 numeric + 15 binary features such as reference-image presence and intent-keyword banks). Task-held-out 5-fold CV with a per-site fold map shared across the three backbone cells (twin tasks share folds; no intent leaks across folds); a confidence threshold τ tuned by inner CV, falling back to P-SoM below threshold. Cell identity is carried by head selection, not runtime features. The training label is the cheapest successful mode under a fixed prior cost ordering — a disclosed approximation, since measured per-mode costs are nearly flat (§5.4) and the prior ordering can mis-rank the frontier; measured-cost relabeling is future work.

**Evaluation.** H10 is a two-layer *operational deployment gate*, not a significance test: a cell passes if the router's (Cost, SR) point is Pareto non-dominated by all five pre-registered single-mode baselines in ≥95% of task-paired bootstrap replicates; the router is *deployable* if ≥5 of 6 cells pass. Alongside, an intelligent-baseline ladder bounds the router from both sides: always-cheapest, single-feature decision stump, DOM-features-only LR, and a per-task lookup table (an ∞-capacity reductio that bounds generalization headroom from above). On cls·B0 the ladder spans always-DOM 17.4% → stump 19.6% → an 8-feature LR proxy 25.0% → oracle ceiling 43.3% (disclosure-only: the proxy is an archive-sanity stand-in for the Pass-2 router and enters no gate); on this one cell it brackets the plausible value of task features (+5.4pp over the stump) and the headroom no 18-feature router can close (+18.3pp to oracle).

*Table 4: H10 verdict per cell — router (Cost, SR) vs. five single-mode baselines. <TBD: populate from `aggregate_h10_pareto.py` after Pass-2 fire. Current artifact state: no Pass-2 runs in any cell; Pass-1 still missing for red·B1/red·B2; entropy-gate artifact absent → `h10_status=entropy_unavailable`, deployability fail-closed per prereg (no entropy verdict ⇒ no deployability claim).>*

| Cell | n common | Router SR | Router cost | frac non-dominated | Pass |
|---|---|---|---|---|---|
| cls·B0 | \<TBD\> | \<TBD\> | \<TBD\> | \<TBD\> | \<TBD\> |
| cls·B1 | \<TBD\> | \<TBD\> | \<TBD\> | \<TBD\> | \<TBD\> |
| cls·B2 | \<TBD\> | \<TBD\> | \<TBD\> | \<TBD\> | \<TBD\> |
| red·B0 | \<TBD\> | \<TBD\> | \<TBD\> | \<TBD\> | \<TBD\> |
| red·B1 | \<TBD\> | \<TBD\> | \<TBD\> | \<TBD\> | \<TBD\> |
| red·B2 | \<TBD\> | \<TBD\> | \<TBD\> | \<TBD\> | \<TBD\> |

<!-- Router prose below is verdict-conditional; keep both branches drafted
     until Table 4 lands, then delete the non-realized branch. -->
(If H10 passes) The router turns a slice of the §5.2 oracle ceiling into realized gain at text-dominated cost, and non-dominance holds across ≥5 cells including at least one low-SR cell — evidence that per-task representation choice is learnable from static task features alone. (If H10 fails or defers) We report per-cell descriptive operating points: the router's value claim is then bounded to the oracle-ceiling existence proof of §5.2, and the deployment claim is explicitly *not* made — the phenomenon claim of §5 is unaffected by construction of the gate. In either branch, two honesty constraints apply: non-dominance alone shows "not worse," so any deployability language additionally cites the strictly-better diagnostics; and B0 cells with non-dominance margins under the ~13% per-task serving-discordance floor are reported descriptively only. Routing signals measured on the phantom arms are as usable as on baseline modes (exploratory, in-condition signal AUROC: top signal 0.766 for P-SoM vs. 0.673 for DOM on cls·B0 [A]), so the arms drop into existing router feature stacks without new instrumentation.

# 7 Discussion

**Representation as a deployable control surface.** Three properties make the phantom arms deployable levers rather than benchmark curiosities: they are model-fixed (no second model to host), training-free at inference (a regex over text the agent already consumes), and cost-anchored to DOM by construction. The empirical content of the paper sits on top: the arms open low-overlap success pools (§5.3) and carry irreplaceable oracle coverage (§5.2), which is what a router can monetize.

**Composition with compute-axis routing.** Representation routing is orthogonal to model/capacity/precision routing and to token pruning; a production stack can route representation per task, prune within the chosen representation, and escalate models on stall. We resist the stronger reading that representation routing *replaces* model routing. We also register a boundary condition: the lever's magnitude tracks the per-step cost of image encoding, and cheaper vision encoders will shrink (not erase) it.

**Existence vs. utility across settings.** On all three complete canonical cells, screenshot-free arms carry unique coverage even though full SoM leads on average — P-SoM uniquely solves 2, 3, and 1 tasks on cls·B0/B1/B2 respectively [A], the last at the pre-registered per-cell noise floor; whether this pattern survives pooling across all six cells is precisely what the gates test (<TBD>). The space's *utility*, by contrast, is already visibly site- and capability-modulated (§5.6). Deployments should expect the routing menu to be worth widening precisely where observations are text-dominated or image encoding is expensive, and the router to be worth training only where per-task signal exists.

# 8 Limitations

**Scope.** One benchmark family (VWA), two sites; the third site and cross-benchmark (WebArena) validation are explicit future work with a pre-registered portability check. The construct presupposes SoM-style screenshot scaffolds (§3.2). Absolute SRs are not comparable to external leaderboards (all-node numbering; pinned patched judge); only within-paper paired contrasts are supported.

**Models.** Three backbones cannot separate deployment-tier from family effects (no fourth matched corner); B1-vs-B2 is a 4B parameter-parity direction test, not a matched-capability control. B2 (Gemma-3-4B) sits at a diagnosed capability floor on cls (~1–2% SR), which mutes its within-cell contrasts; per the pre-registered claim-tier gate, failure of either B2 cell at the per-cell drop-one criterion downgrades the cross-family claim without collapsing the construct — and format-circuit separability is known to be family/scale-dependent [@feldhus2026judgecircuits], so this outcome was registered as a live risk, not assumed away.

**Measurement.** B0 is served through a commercial API with a measured same-payload nondeterminism floor (~±2.5pp net SR; ~13–14% per-task discordance on replay) [@he2025nondeterminism; @yuan2025numerical]; all B0 contrasts smaller than this floor are treated as noise, and the pooled H1 gate — not any single B0 cell — carries the hero claim. Cost bases differ across backbones (API-USD vs. electricity-USD), so cost claims are within-baseline only. The judge is an LLM with known drift sensitivity (~±2pp across snapshots); it is pinned, patched, and applied uniformly across modes, so mode differentials are protected even where absolute levels are not. The sequential single-account protocol pairs tasks by identity and position, not by byte-identical substrate state; per-mode environmental-footprint differences remain a disclosed sensitivity (order-position and footprint diagnostics in the supplement).

**Statistics.** The design is powered for medium pooled effects but underpowered for small per-cell effects; the pre-registration's archive power projection used a 4-mode additive estimand that upper-bounds the strict 6-mode gate effect (Amendment 02), so strict-gate power is reported as <TBD> rather than that projection. Per-cell CIs are reported as transparency, never as gates. Our fixed-effects pooling over the six planned cells deliberately diverges from small-k HKSJ recommendations *on design grounds* (the cells are the population); HKSJ sensitivity is in the appendix, and a degenerate low-variance cell cannot silently dominate the pool — per-cell weights use bootstrap SEs subject to a pre-registered 1.0pp lower floor (Agresti-Coull-anchored). H1 is the single primary statistical test; H10 is an operational gate with no across-cell α. The pre-registration disclosed a joint Type-I exposure of 0.0975 over {H1, H10} prior to H10's reclassification as an operational criterion; we retain that disclosure for transparency while noting that no joint FWER is defined under the current H10 semantics.

**Mechanism.** All mechanism-level analysis (activation patching, probes) is deferred to follow-up work; this paper's behavioural account (§5.5) is hypothesis-generating characterization on an archive substrate.

# Reproducibility Statement

Code (agent scaffold, patched-evaluator submodule with pinned tree hash, analysis pipeline), the pre-registration with time-stamped amendment log (anonymized OSF view-only registration for review; DOIs de-anonymized at camera-ready), per-condition run manifests, quarantine registry, and per-task JSONL trajectories sufficient to recompute every table are released with the supplement. The reproducibility checklist is filed per AAAI-27 requirements. <TODO: de-anonymized artifact URLs at camera-ready; anonymized OSF view for review.>

<!--
=====================================================================
PRE-SUBMISSION CHECKLIST (delete before compile)
 1. Rerun `make analysis` after: B0-red psom/pprompt land; B1/B2-red
    land; Pass-2 router fire. Lift [P] → [A], fill <TBD>.
 2. Resolve <H1-VERDICT>, <H3-VERDICT>, <H10-VERDICT>; rewrite all
    (R-CONDITIONAL) sentences per realized R-tier (prereg §2.5).
 3. Latency canonical recomputation (retry-adjusted) → §5.4.
 4. Figures: (F1) 2×2 diamond schematic; (F2) drop-one forest
    (fig_forest_drop_one.py, 6-cell); (F3) per-cell Pareto scatter
    (aggregate_h10_pareto.py); optional (F4) unique-pass Venn
    (fig_phantom_structure_venn.py).
 5. /stress + /codex-stress + /gemini-stress before any commit of
    this prose (CLAUDE.md auto-trigger).
 6. Verify all bib keys compile against paper.bib; nikankin2025sametask
    reserved for rebuttal use.
 7. Word budget (measure by stripping HTML comments, then wc -w; do
    NOT paste the strip-regex here — a literal close-comment token
    inside this block truncates it, empirically caught 2026-07-01):
    body = 5275 words incl. 4 tables ≈ 6.7-7.0 pages two-column;
    figures F1-F3 add ~1.0-1.2 pages → currently ~1 page OVER.
    Plan to cut ~800-1000 words at verdict-landing time (candidates:
    §2 para 4 anchors, §5.5 compression, §8 statistics para).
 8. Anonymity pass: strip host names, usernames, grant info; OSF must
    be anonymous view-only link for review (DOIs camera-ready only).
 9. Regression grep before every commit of this file (2026-07-01
    stress lesson — compression re-introduces banned phrasing):
    grep -nE "image-free|image-off|no image tokens|text-only cost|both Qwen cells|most of the.*mass" aaai27_main.md
    must return 0 hits.
=====================================================================
-->
