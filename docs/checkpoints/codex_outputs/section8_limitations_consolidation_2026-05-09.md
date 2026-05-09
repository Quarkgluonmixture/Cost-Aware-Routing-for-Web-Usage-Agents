Reading prompt from stdin...
OpenAI Codex v0.128.0 (research preview)
--------
workdir: /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
model: gpt-5.5
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: high
reasoning summaries: none
session id: 019e0cd7-fb4e-7561-8f6c-23e3e35f25f2
--------
user
# Codex Prompt — Paper §8 Limitations Consolidation (D6 + H1)

## Goal

Write the **final consolidated §8 Limitations** for the paper as a new file `docs/checkpoints/paper_drafts/section8_limitations.md` (do NOT overwrite the existing piecewise `section4_limitations_disclosure.md` — that stays as appendix-level long-form prose).

§8 is the **camera-ready limitations section** that goes in the body of the paper (~1.5 pages, ~900-1100 words). It should be reviewer-rebuttal-ready: every concession is paired with a *blast-radius bound* (how the limitation affects which claim, and why the affected claim is still defensible).

Addresses audit constraints **D6** ("State failed assumptions and limitations in a dedicated section") and **H1** ("Maintain a dedicated limitations section with strong assumptions and robustness failures"). Both ⚠️ → ✓ on completion.

## Inputs to read FIRST (in order, ~15 min)

1. `docs/checkpoints/pre_run/topvenue_constraints.md` — locate D6, H1, F2 (construct), F3 (external), F4 (internal validity), C9, A14 (external validity bound), G5 (post-hoc disclosure), B6 (missing-data), and read the "reviewer one-liner" column for each. These one-liners are the *spine* of §8.
2. `docs/checkpoints/paper_drafts/section4_limitations_disclosure.md` — 10 sub-disclosures (§4.X.1-§4.X.10) on specific bugs (ua_match drift, string_match fuzzy, program_html brittleness, finish_wrong_state, viewport bug, scroll direction, B0/B1 asymmetry, cross-machine drift, pre/post Phase A asymmetry, Stage 2B input vintage). These are inputs but §8 should *summarize and cluster*, not duplicate. §4_limitations_disclosure stays as appendix.
3. `docs/checkpoints/pre_run/pre_rerun_audit.md` §4.4 — final enumeration list for §8 prose.
4. `docs/checkpoints/pre_run/negative_results_registry.md` — 12 retracted framings + 2 confirmed framings table; the 5 paper-§ action items at the end. §8 should reference this registry, not duplicate it.
5. `docs/checkpoints/pre_run/preregistration.md` — H1-H8 hypothesis lock, §5.X post-hoc layer disclosure, §7 reproducibility scope. §8 references the H4 (exploratory) and H5-H6 (post-hoc) labels.
6. `docs/checkpoints/paper_drafts/section1_intro.md` — read first 100 lines for the R1-R5 outcome-conditional framing rules (H4 audit).
7. `docs/checkpoints/paper_drafts/section5_mechanism_reddit.md` §5.1 method box (just written, addresses G1) — §8 should cite §5.1 when discussing mechanism limitations (e.g., 36-layer Qwen3-VL only, no cross-architecture).
8. `docs/checkpoints/实验笔记.md` §111 (task-0 over-interpretation retracted), §111.5b (reverse-null overturn), §117 (Stage 2 cell A-E), §118 (selection-bias control rejected). Quote-paraphrase, do NOT bulk-import.
9. `docs/checkpoints/paper_drafts/paper.bib` — check existing bibkeys; flag NEEDS_BIB_ENTRY for any new cites (likely Lipton & Steinhardt 2018 "Troubling Trends", Pineau 2021 reproducibility, NeurIPS checklist).

## Required structure for §8 Limitations (camera-ready prose)

Use these subsection headers (LaTeX `\subsection*{...}` style at section level, no auto-numbering since §8 is "Limitations"):

### 8.1 Scope & external-validity bounds (~200 words)
- 3 sites (cls/red/shop), 1 benchmark family (VWA + WA-mini), 2 model classes (Qwen3-VL-4B / Qwen3-Omni-235B-Thinking proxy). Mechanism §5 evidence is from **B1 only** (open-weight); B0 has no activation access.
- Cross-architecture (e.g., GPT-4o family) untested → claim is "phantom routing space exists in the Qwen3-VL family on VWA-style tasks", not "in all VLMs everywhere".
- Cite A14 + F3 reviewer-rebuttal language from constraint table.

### 8.2 Construct validity & evaluator threats (~200 words)
- ua_match GPT-judge drift (B-20), string_match fuzzy_threshold misnomer, program_html selector brittleness, finish_wrong_state. Cluster as 4 evaluator-class threats; cite §4_limitations_disclosure §4.X.1-§4.X.4 for full prose.
- FP filter (§95 eval_fp/visual_fp + §78a na_fp) addresses these *measurement-side*, not *task-side*; report raw + adjusted SR.
- Cite F2 reviewer-rebuttal: "We report raw and adjusted success and isolate evaluator-class threats in limitations."

### 8.3 Internal-validity threats: known scaffold bugs (~150 words)
- in_viewport_ratio operator precedence (B-26 — affects all DOM modes uniformly, doesn't bias inter-mode contrast), scroll direction confusion (B-28), Stage 2B input vintage independence (笔记 §116). All three: blast radius is **uniform across modes**, so does not explain inter-mode SR gaps.
- Cite C9 reviewer-rebuttal: "We treat environment failures as measurement threats and disclose their blast radius rather than folding them into cognitive claims."

### 8.4 Pre-vs-post-hoc analyses & retracted framings (~200 words)
- Per preregistration §5.X, the L11/L17 mechanism layer choice was **not pre-registered**; it emerged from Stage 2A pilot (hypothesis-generating). Confirmed by 4 independent paths (logit_shift, forward overlap, reverse overlap, cross-tier).
- Reference negative_results_registry: 12 framings retracted (e.g., "task-0 over-interpretation" §111, "reverse-null" §111.5b, "selection bias" §118 Welch p≥0.5 NS), 2 confirmed framings (4-fold drop-in property, sparse mechanism).
- Cite G5 reviewer-rebuttal: "Layer-set choice is disclosed as exploratory in §5 and pre-registered in the deposited registry."

### 8.5 Statistical & methodological limits (~150 words)
- Holm-Bonferroni across 6 layers chosen post-hoc (not 36); disclose. Bootstrap clustering policy locked at task-pair (B2). Random-effects meta-analysis only on cells with N≥10 to avoid τ² instability (per B8 lock).
- Power analysis (B9) shows minimum-detectable effect for cells with N=15 ~0.65 Cohen's d at α=0.05 — not powered for small mid-layer effects.
- Missing-data policy: complete-case (B6 lock); ≤5% per cell, no multiple-imputation needed.

### 8.6 Sparse-mechanism caveat (~100 words)
- G8 finding: median Levenshtein-distance shift = 0 for 4 of 5 cells at L17, IQR includes zero. Mechanism activates on *task subsets* (~25% of strong-tier), not uniformly — reframes the §5 claim as "task-conditional sparse mechanism" rather than "universal mid-layer circuit". Cite §5 fig and §117 笔记 entry.

### 8.7 Compute, cost, and sustainability bounds (~80 words)
- Per-cell GPU-hours / USD / kg-CO₂ table per A10 reviewer-rebuttal. Cross-machine power profile (DGX Spark vs Myriad V100) is one limitation source for absolute energy numbers; relative comparisons within the same cell are unaffected.

## Constraints

- One file created: `docs/checkpoints/paper_drafts/section8_limitations.md`
- Plain markdown (compatible with codex / pandoc), use `\citep{...}` for bibkeys
- Length: 900-1100 words total (sum of 8.1-8.7). Do NOT exceed 1200.
- Each subsection: lead with the *concession* (1 sentence), then the *blast-radius bound* (1-2 sentences), then *paper-impact* (1 sentence: which §/claim this affects, why still defensible).
- Reference but do NOT duplicate `section4_limitations_disclosure.md`. §4 stays as long-form appendix; §8 is paper-body summary.
- Flag NEEDS_BIB_ENTRY for any cite not in `paper.bib`.
- Do NOT recommend new experiments; this is a *bounding* document, not an action list.
- Use \citep{} not [X] format throughout (per D8 audit upgrade earlier this session).

## Output

Print exactly:
```
DONE: wrote section8_limitations.md (W words across 7 subsections)
```
plus 5-bullet diff summary listing the new file's subsection headers + word counts + any NEEDS_BIB_ENTRY flags.

Stop. Do not modify any other file.

codex
I’ll read the audit and draft inputs first, then write only the new §8 file. I’ll keep the long-form §4 appendix untouched and verify the final word count before reporting.
exec
/bin/bash -lc "sed -n '1,260p' docs/checkpoints/paper_drafts/section4_limitations_disclosure.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
# Section 4 — Known Evaluator Limitations & Disclosure (Draft)

**Status**: 🟡 Draft prose for paper §4 / §3 limitation table. Each subsection cites the
master_bug_catalog.md entry by ID. Reviewer-defensible: bugs are CONFIRMED but blast radius
bounded; mitigations or paper-§3 disclosure rather than retraction.

**Source**: `docs/reference/master_bug_catalog.md` B-15 / B-20 / B-21 / B-22 / B-26 / B-28 +
`docs/checkpoints/pre_run/preregistration.md` §A1/A3 design asymmetries.

---

## §4.X.1 ua_match GPT-judge drift (B-20)

VWA's `ua_match` evaluator uses a GPT-4o-mini judge to rate the agent's terminating answer
against the task's reference answer. The judge prompt template is fixed in
`evaluation_harness/helper_functions.py` (`llm_fuzzy_match`) and not modified in this work.
However, GPT-4o-mini is a stochastic API: the judge's output drifts across re-evaluations
in 4 distinct modes (semantic equivalence vs strict literal match, spurious "partial credit",
hallucinated rationale, and length-dependent confidence). Static audit of 87 N/A-task FP
episodes (笔记 §95) showed the judge's binary verdict varies on ~12% of borderline cases when
re-queried with temperature ≥0.

**Mitigation in this work**: We pin judge `temperature=0` for all evaluations and report all
ua_match-affected tasks as part of the `na_fp` exclusion class (preregistration.md §3 FP filter).
Sensitivity analysis (Appendix D) shows our H1/H3 conclusions hold under three FP filter
variants (raw / +na_fp / +na_fp+eval_fp), so judge drift cannot flip the paper's hero claim.

**Residual concern**: If a future reviewer re-runs the evaluator with a newer GPT-4o-mini
snapshot, single-task labels may flip. The aggregate per-cell SR is robust to this within
±2pp by simulation. We make this explicit in our reproducibility statement (§3.X) rather
than retract the SR claim.

---

## §4.X.2 string_match fuzzy_threshold misnomer (B-21)

VWA's `string_match` evaluator exposes a `fuzzy_threshold` parameter that suggests a
numerical similarity cutoff for string matching. In practice (catalog B-21 static audit),
the parameter is **only honored when fuzzy_threshold=1.0** — under which the evaluator
falls through to the same GPT-4o-mini fuzzy_match judge as `ua_match`. Threshold values
strictly below 1.0 trigger a brittle exact-token-overlap path with no judge involvement.
This is effectively binary GPT-judged matching, not a tunable similarity metric.

**Mitigation**: We use `fuzzy_threshold=1.0` consistently across all conditions (verified
via condition_meta.json `evaluator_config.fuzzy_threshold`), so the variability source is
the same as B-20 ua_match drift and is jointly bounded by the same FP filter robustness.
The mis-naming does not affect our results, but we flag it for readers attempting to
interpret raw VWA evaluator parameters.

---

## §4.X.3 program_html selector brittleness (B-22)

VWA's `program_html` evaluator scores tasks by goto'ing a target URL and querying DOM with
CSS/XPath selectors authored in each task's reference config. Static audit (笔记 §107
Tier 5) found 562 of 1598 (35.2%) selectors are class-only or attribute-only patterns
(e.g., Magento's `.order-details-items.ordered`, classifieds' `.price` / `.desc`) that
match site-skin-dependent layout. When the site's CSS skin updates between evaluator
authoring time (2024) and our experimental deployment (2026), selectors can match the
wrong DOM node or miss the intended element entirely.

**Per-cell quantification**: We measure selector hit-rate parity in our archive — for each
program_html task, we count post-action DOM nodes matching the reference selector. A pre/post
ratio outside 0.95-1.05 across modes within the same task is flagged (~3% of program_html
tasks); these are excluded from H1/H3 per the preregistered FP filter `eval_fp` rule.

**Cannot-fix scope**: Patching all 562 brittle selectors requires authoring a parallel
evaluator harness, which is out of scope for this paper. We retain VWA's evaluator unchanged
(reviewer-defensible upstream parity per §3 evaluator independence) and bound the impact
via the FP filter sensitivity ladder (Appendix D).

---

## §4.X.4 finish_wrong_state — agent error not scaffold (B-15)

In Tier 2 silent-failure analysis (笔记 §107), 1552 of 4501 episodes (34.5%) had the agent
emit `finish` while the page state did not match the task goal. Initial framing classified
this as a scaffold bug; subsequent self-replay (笔记 §95 reform) showed it is an **agent
reasoning error** — the agent decides to terminate prematurely or with partial completion,
not a runner / dispatch / observation failure.

**Treatment**: This is captured in our `eval_fp` filter rule (preregistration.md §3): if
`agent_finished=True` but evaluator returns success and the agent has no effective action
in the trajectory, we mark the episode as `eval_fp`. The agent error itself is not a paper
limitation — different baselines and modes can succeed or fail at terminating decisions, and
our paired-design comparison absorbs this into per-task variance.

---

## §4.X.5 in_viewport_ratio operator precedence (B-26)

In `external/visualwebarena/browser_env/processors.py:218`, the `in_viewport_ratio`
calculation `overlap_w * overlap_h / w * h` is parsed by Python as
`((overlap_w * overlap_h) / w) * h` — multiplication-first then division — instead of the
intended ratio `(overlap_w * overlap_h) / (w * h)`. The result is that the 0.6 viewport-overlap
threshold (`current_viewport_only=True`) is effectively bypassed, allowing partially-visible
elements to remain in the AXTree with their full text content even when they are visually
truncated.

**Implication for our claims**: This bug exists in upstream VWA and is documented in our
CLAUDE.md as "DOM has structural information advantage." It systematically helps DOM mode
relative to Vision/SoM modes by exposing element text that is visually clipped. We do **not**
fix this bug because: (a) it's upstream code; (b) any threshold value would be debatable;
(c) it does not affect our **paired** comparisons (P-SoM uses the same DOM-derived
`[SOM_MARKS]` text), so our hero claims (P-SoM ≥ best of DOM/SoM/Vision) are invariant to
this asymmetry. We disclose the asymmetry source for cross-mode interpretation.

---

## §4.X.6 scroll direction confusion (B-28)

Early experiments (B0 cls/red, 笔记 §50) revealed inconsistent agent behavior for scroll
direction conventions: Web CSS uses `dy>0 = scroll DOWN` (content moves up), but Win32 and
macOS natural scrolling invert this convention. The 235B model occasionally chose the wrong
direction sign, producing scroll-up-when-needed-down patterns counted as no-progress.

**Mitigation**: §67 schema reform replaced `delta: [dx, dy]` with explicit
`scroll_direction: enum("up", "down")` in the action schema (B0 only via tool-calling
schema; B1 still uses delta in greedy decoding). This eliminates the symbol convention
confound for B0 going forward but does not retroactively fix archived B0 data. We disclose
this asymmetry in §3 evaluator-side fairness discussion.

---

## §4.X.7 A1/A3 baseline-design asymmetries (B-56)

This work compares B0 (Qwen3-VL-235B-A22B via proxy API) against B1 (Qwen3-VL-4B-Instruct
local). Two configuration asymmetries are intentional and documented:

**A1 — Decoding strategy**: B0 uses `temperature=0.0` with `top_p=1.0` (B-37 fix
post-§107); B1 uses `do_sample=False` (greedy top-1). Both target deterministic outputs,
but B0 still inherits proxy-side stochasticity for which the API has no `seed` parameter.
Cross-run trajectory variance for B0 is bounded by single-step branching at ties; aggregate
SR is stable (laughs at our N=234+210+466 sampling).

**A3 — Token budget**: B0 has `max_new_tokens=4096`; B1 has `max_new_tokens=384`. The
asymmetry stems from B0's verbose thought + JSON output requirement; B1's parser is more
robust to compact outputs. In rare cases (~0.15%), B1's compact budget causes truncated JSON →
parse_failure → `wait` action. We retain this asymmetry as a B1-specific structural
limitation rather than artificially inflate B1's budget; the impact is bounded and disclosed
in §3 baseline configuration table.

---

## §4.X.8 Cross-machine numerical drift (笔记 §114 Gap 5)

Our work runs across three GPU architectures: DGX Spark (NVIDIA GB10, sm_121), UCL Condense
A100 (sm_80), and UCL Myriad (sm_70 V100 / sm_80 A100). Mechanistic Stage 2B/2C activation
patching outputs are sensitive to floating-point matmul precision differences across CUDA
generations (sm_70 vs sm_80 vs sm_121). We run `numerical_determinism_check.py` to quantify
maximum absolute hidden-state drift |Δh| across machines on a fixed input.

**Reproducibility statement**: Cross-machine numerical agreement on Qwen3-VL-4B between
{DGX, A100, Myriad} layers L0-L35: max |Δh| < [TBD post-rerun, target <1e-2] at L11 (the
mirage causal layer per §5). This bounds inter-machine reproducibility drift to a level that
does not flip top-1 logit comparisons; aggregate SR claims are unaffected.

---

## §4.X.9 Pre-Phase-A vs post-Phase-A asymmetry (B-01 to B-37 family)

The 16-cell rerun (preregistration.md §4 cell inclusion) uses post-Phase-A code only
(commit ≥ `3c15cd7`, dispatch + page_changed + cycle + RNG fixes deployed). Pre-Phase-A
data is retained as Appendix D robustness check (preregistration.md `Cell inclusion (Appendix D)`).
For mechanistic Stage 2B/2C input artifacts, we use pre-Phase-A archived observations
(`results/mechanistic/archive_subset_b1_cls/`); per 笔记 §116 user-prompt analysis, agent
trajectory bugs (Phase A scaffold issues) do **not** affect the model's forward-pass
input→output mapping at any frozen step. Mechanism findings (L11 causal layer, forward-vs-reverse
asymmetry) are therefore unaffected by Phase A vintage; we make this independence explicit
in §5.

---

## §4.X.10 Stage 2B input vintage independence (笔记 §116 user Q)

Mechanistic Stage 2B (forward L11 mirage causal layer) and Stage 2C (reverse direction
asymmetry) use frozen `observation_dom.txt` + `screenshot_annotated.png` artifacts from
`B1_phantom_som_classifieds_20260428` archive (pre-Phase-A). Per 笔记 §116 user analysis:
the mechanistic claim is about model forward-pass behavior given a fixed input, not about
agent trajectory soundness. Phase A bugs in dispatch / cycle / RNG affect *which step* the
agent reaches, not *what the model thinks* given a frozen step's observation. The L11
mirage finding is therefore Phase-A-vintage-independent.

For full robustness, we pre-specify a post-Phase-A spot-check (5-10 tasks from a clean
post-`3c15cd7` cell) where we re-run Stage 2B and verify L11 causal layer holds. This
sensitivity check is in §5 Appendix and does not gate the main mechanism claim.

---

## References

- `docs/reference/master_bug_catalog.md` — full bug catalog (~80 entries)
- `docs/checkpoints/pre_run/preregistration.md` §3-§4 — locked analysis choices including FP filter
- `docs/checkpoints/pre_run/evaluator_change_protocol.md` — Protocol A Tier classification
- 笔记 §95 (FP reform) / §107 (Phase A wave) / §114 (provenance) / §116 (audit) / §116.X user prompts

exec
/bin/bash -lc "rg -n \"\\b(D6|H1|F2|F3|F4|C9|A14|G5|B6)\\b|reviewer one-liner|one-liner\" docs/checkpoints/pre_run/topvenue_constraints.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
9:- ✓ Done: 50 constraints (post-2026-05-09: G6 + G7 cell E + 2x2; B2 bootstrap policy lock; A14 + F3 reproducibility / external-validity scope; H2 negative-results registry; F7 + H3 ethics/license/COI/misuse-safety; G5 post-hoc layer disclosure; D8 + H5 section1 intro rewrite; B6 + B7 + B8 pre-spec policy lock; A5 + F8 version pinning; A11 + A12 + A13 release cards; G9 token alignment; G8 sparse-mechanism heterogeneity; C10 pre-launch + validate-strict Make targets; G1 §5.1 method box codex-delegated)
26:| ~~9~~ | ~~G5 post-hoc layer selection~~ | ✓ disclosed 2026-05-09 (preregistration §5.X — 4-path convergence + Holm) | — |
33:| A1 | Preregister primary hypotheses, decision rules, and analysis families before post-rerun data are inspected | NEEDS_BIB_ENTRY: Pineau et al. 2021; NeurIPS checklist Q4/Q6 | ⚠️ | `docs/checkpoints/pre_run/preregistration.md` has H1-H8, Holm families, R1-R5, but frontmatter is `status: draft`, `registered_at`, `registered_git_sha`, `witnessed_by`, and OSF DOI are pending. Remediation: lock after advisor email, tag git, deposit OSF; cost 2-4h. | "The hypotheses and decision rules were written before the 16-cell rerun; the camera-ready will cite the lock SHA/OSF DOI once advisor witness is received." |
46:| A14 | State exactly which claims are reproducible under released artifacts and which depend on closed APIs | NeurIPS checklist Q4-Q5 | ✓ | `preregistration.md §7` (added 2026-05-09) gives 6-tier reproducibility scope table (B1 / B1 mechanistic / B0 / VWA env / Evaluator / Stage 2 analysis) + paper §3 quotable scope-claim paragraph + external validity scope (audit F3). | "B1 (local Qwen3-VL-4B) and mechanistic Stage 2 claims are byte-identical reproducible; B0 (proxy API Qwen3-Omni-235B) is verifiable from released traces and replayable subject to API access. Full scope statement in preregistration §7." |
57:| B6 | Predefine missing-data/crashed-episode handling | CONSORT-AI/Liu et al. 2020; NeurIPS checklist Q7 | ✓ | Locked 2026-05-09 in `preregistration.md §4`: crashed episodes (uncaught exception / OOM / >30min timeout / browser crash) excluded from paired-N denominators, NOT imputed; not_logged_in / auth_drift after 3 retries excluded; missing artifacts excluded from per-step (not full-task); per-cell exclusion count + reason histogram in Appendix C. | "Listwise deletion only — crashes excluded from paired N, no success/failure imputation, exclusion histograms reported per cell." |
74:| C9 | Separate benchmark bugs/scaffold failures from agent failures | `zhou2024webarena`; NEEDS_BIB_ENTRY: Breck et al. 2017; NEEDS_BIB_ENTRY: Sculley et al. 2014 | ✓ | `docs/reference/master_bug_catalog.md` has 80+ entries with confirmed/fixed/not-a-bug status; `section4_limitations_disclosure.md` is referenced by `pre_rerun_audit.md §4.4`. | "We treat environment failures as measurement threats and disclose their blast radius rather than folding them into cognitive claims." |
86:| D2 | Distinguish confirmatory, exploratory, and post-hoc analyses | NEEDS_BIB_ENTRY: Lipton & Steinhardt 2018; NeurIPS checklist Q7 | ✓ | `preregistration.md §2/§5` labels H1-H3 primary/structural, H4 exploratory, H5-H6 post-hoc; `paper_planning.md` repeatedly marks provisional mechanisms. | "The paper separates preregistered gatekeeping tests from post-hoc explanatory analyses." |
90:| D6 | State failed assumptions and limitations in a dedicated section | NeurIPS checklist Q2; NEEDS_BIB_ENTRY: Lipton & Steinhardt 2018 | ⚠️ | `section4_limitations_disclosure.md` exists and `pre_rerun_audit.md §4.4` lists coverage, but final Section 8 prose is still TBD. Remediation: consolidate limitations in final paper; cost 0.5 day. | "The limitations appendix already enumerates evaluator, scaffold, numerical, and pre/post-fix threats." |
113:| F2 | Construct validity: ensure metrics measure task success and not evaluator artifacts | `zhou2024webarena`; `koh2024visualwebarena`; NEEDS_BIB_ENTRY: Lipton & Steinhardt 2018 | ⚠️ | Adjusted SR and FP filters address evaluator artifacts; `pre_rerun_audit.md §4.5.4` LLM-judge reproducibility is TBD. Remediation: judge spot-check and per-evaluator-class sensitivity; cost 0.5 day. | "We report raw and adjusted success and isolate evaluator-class threats in limitations." |
114:| F3 | External validity: scope across sites, tasks, and models | `deng2023mind2web`; `drouin2024workarena`; `koh2024visualwebarena` | ✓ | `preregistration.md §7` (added 2026-05-09) explicitly scopes empirical claims to "Qwen-family VWA characterization" — Qwen3-VL-4B (B1) + Qwen3-Omni-235B-Thinking (B0) on VWA classifieds / reddit / shopping. Cross-benchmark (WA) and cross-model-family (Llama-VL, GPT-4o-V, Gemini-Pro-VL) explicitly future work. Mechanistic findings scoped to curated mirage-disagreement tiers, broader universality conditional on cells F/G cross-site results. | "All empirical claims are scoped to Qwen-family VWA; cross-bench and cross-model are explicit future work — see preregistration §7." |
115:| F4 | Statistical conclusion validity: report uncertainty and sensitivity to thresholds | NeurIPS checklist Q7; NEEDS_BIB_ENTRY: Cook & Campbell 1979 | ⚠️ | CIs/meta-analysis exist; `pre_rerun_audit.md §4.8.1` K±1 and `§4.8.2` leave-one-cell-out are TBD. Remediation: add threshold-gradient and cell-removal table; cost 0.5 day. | "The decision rule is locked; sensitivity tables show how close the conclusion is to the threshold." |
129:| G5 | Avoid post-hoc layer selection by using independent confirmation or disclosure | NEEDS_BIB_ENTRY: Zhang et al. 2024; NEEDS_BIB_ENTRY: Wang et al. 2023 | ✓ | `preregistration.md §5.X` (added 2026-05-09) explicitly maps L11/L17 selection chain: Stage 2A logit_shift (hypothesis-generating) → §111 task-0 (single-task outlier, acknowledged) → Stage 2B 24-task / 2C 15-task / Cell D 24-task (3 confirmatory cells, all Holm-significant L17 + cell D L11). Holm-Bonferroni across 6 canonical layers handles multiple-comparison; convergence of 4 independent analysis paths (logit_shift, forward overlap, reverse overlap, cross-tier) constitutes confirmation. Cell E random-injection rules out generic-injection alternative. | "Layers L11/L17 are not pre-registered but emerged from Stage 2A pilot and are confirmed by 3 independent Stage 2B/2C/Cell-D tests with Holm correction — see preregistration §5.X." |
141:| H1 | Maintain a dedicated limitations section with strong assumptions and robustness failures | NeurIPS checklist Q2; NEEDS_BIB_ENTRY: Lipton & Steinhardt 2018 | ⚠️ | `section4_limitations_disclosure.md` exists and `pre_rerun_audit.md §4.4` enumerates limitations; final Section 8 prose is not complete. Remediation: consolidate into final limitations section; cost 0.5 day. | "Limitations are not hidden in appendices; they are surfaced alongside the main empirical claims." |
144:| H4 | Make data-conditional framing rules visible so weak outcomes are not spun | NeurIPS checklist Q1-Q2; NEEDS_BIB_ENTRY: Nosek et al. 2015 | ✓ | `preregistration.md R1-R5` explicitly maps outcomes to strongest/moderate/weak/death-scenario framings. | "If H1/H2/H3 fail, the paper framing is precommitted to weaken or pivot." |
161:13. ⚠️ B8/F4 — Add K±1 and leave-one-cell-out sensitivity tables (0.5 day).

exec
/bin/bash -lc 'rg -n "''^##? 4'"\\.4|"'^###? 4'"\\.4|D6|H1|limitations|Final enumeration|enumeration\" docs/checkpoints/pre_run/pre_rerun_audit.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
133:| 1.4.1 | Multiple comparison correction for H1+H3+TOST family | ✅ in preregistration.md §3 | Holm-Bonferroni step-down per H-sub-family |
134:| 1.4.2 | Bootstrap CI procedure spec (N resamples, RNG seed, BCa vs percentile) | 🟡 partial | Add for H1/H3 oracle lift CI |
159:| 1.4.16 | **Heterogeneity analysis pre-spec** — site differences in H1/H3 | 🔴 TBD | Pre-spec: report per-site SR + 95% CI; if >5pp site-difference, note as "site-modulated" rather than retract |
161:| 1.4.18 | **Falsification criteria pre-spec** — what data outcome retracts hero claim | 🔴 TBD | Per paper_planning R5: <X cells pass H1+H3 → pivot to VWA bug paper; formalize threshold here |
291:> Every CONFIRMED catalog entry has a **probe script** that re-validates the bug exists before/after a fix. Reviewer-defensible: "we don't just claim B-XX is fixed, we re-run the probe and show the symptom is gone." Probes are paper §4 limitations cite-anchors.
416:| 4.1.3 | Bootstrap CI (BCa) on H1/H3 oracle lift | 🟡 partial | Spec N=1000 resamples + RNG seed=42 |
451:**Source**: `docs/checkpoints/paper_drafts/section4_limitations_disclosure.md` (created today, ~10 subsections)
490:| 4.4.c.5 | §94 max_marks reversal as B1 capability-modulated finding (paper §7) | ✅ documented | section 4 limitations or section 7 cross-capability |
521:| 4.8.4 | **Falsification hierarchy** (per paper_planning §5 R-rules R1-R5) | 🟡 partial | Formalize threshold: if H1<10/16 AND H3<8/16 → R5 retract+pivot to VWA bug paper |
573:| 5.1.9 | section4_limitations_disclosure.md | Paper §4 + OSF | ✅ |
644:  Fill §4.4 limitations prose with post-rerun numbers
658:- `docs/checkpoints/paper_drafts/section4_limitations_disclosure.md` (created 5/8 — 10 prose drafts)

codex
I found the constraint-table one-liners and the existing appendix-level bug prose. Next I’m pulling the preregistration and negative-results context so §8 can name the pre/post-hoc boundaries without turning into another appendix.
exec
/bin/bash -lc "sed -n '430,505p' docs/checkpoints/pre_run/pre_rerun_audit.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
| 4.2.3 | FP filter sensitivity (raw / +na_fp / +na_fp+eval_fp) | ✅ pre-spec'd | `aggregate_sr_fp_per_mode.py` 3 variants |
| 4.2.4 | K_h1 / K_h3 ±1 threshold gradient | 🔴 TBD | re-run preregistration_decision_test with K±1 |
| 4.2.5 | Per-difficulty bucket (intent length / N actions / has_ref_image) | 🔴 TBD | Add bucketing in aggregate scripts |
| 4.2.6 | Hold-out site validation (LOSO if locked) | 🟡 advisor email | `router_split.py` LOSO mode |
| 4.2.7 | Cross-machine numerical agreement (DGX/A100/Myriad) | 🟡 needs A100/Myriad SSH | `numerical_determinism_check.py compare` |
| 4.2.8 | **`analyze_cross_representation.py`** — cross-rep behavior diagnostic (DOM↔SoM↔Vision token attention) | ✅ exists | post-rerun |
| 4.2.9 | **`analyze_search_over_browse.py`** — search vs browse behavior diagnostic (Reddit) | ✅ exists | post-rerun |
| 4.2.10 | **`analyze_comment_selflink_loop.py`** + **`analyze_reddit_selflink_cycle.py`** — Reddit cycle / self-link diagnostics | ✅ exists | post-rerun on red cells |
| 4.2.11 | **`analyze_noninteractive_click_earlystop.py`** — non-interactive click + early-stop pattern (paper §4) | ✅ exists | post-rerun |
| 4.2.12 | **`analyze_confidence_calibration.py`** — model confidence calibration diagnostic (logprob/entropy/margin) | ✅ exists | feeds Phase 2 router signal selection (per `routing_signals.md` lit) |

## §4.3 Inter-Rater Reliability Execution (κ ≥ 0.7)

| # | Item | Status | Output |
|---|---|---|---|
| 4.3.1 | FP labeling 30-task pilot (2 raters) | 🔴 TBD pre-rerun | Cohen κ report |
| 4.3.2 | Failure-mode 5-bucket rubric reliability | 🔴 TBD | κ report |
| 4.3.3 | Codex-as-rater calibration spot-check | 🔴 TBD | Disagreement <30% threshold |

## §4.4 Section 4 Limitations Disclosure Prose

**Source**: `docs/checkpoints/paper_drafts/section4_limitations_disclosure.md` (created today, ~10 subsections)

| # | Item | Status | Source |
|---|---|---|---|
| 4.4.1 | B-20 ua_match GPT-judge drift prose | ✅ today | Section 4.X.1 |
| 4.4.2 | B-21 string_match fuzzy_threshold misnomer | ✅ today | Section 4.X.2 |
| 4.4.3 | B-22 program_html selector brittleness | ✅ today | Section 4.X.3 |
| 4.4.4 | B-15 finish_wrong_state (handled by §95 FP) | ✅ today | Section 4.X.4 |
| 4.4.5 | B-26 in_viewport_ratio operator precedence | ✅ today | Section 4.X.5 |
| 4.4.6 | B-28 scroll direction (mitigated via §67) | ✅ today | Section 4.X.6 |
| 4.4.7 | A1/A3 baseline-design asymmetries (B-56) | ✅ today | Section 4.X.7 |
| 4.4.8 | Cross-machine numerical drift | ✅ template | Section 4.X.8 (post-rerun fill numbers) |
| 4.4.9 | Pre-Phase-A vs post-Phase-A asymmetry | ✅ today | Section 4.X.9 |
| 4.4.10 | Stage 2B input vintage independence | ✅ today | Section 4.X.10 |

## §4.4.b Failure-mode pattern detection (`diag_pattern_match.py` + analysis pipeline)

**Source**: 笔记 §92 (P1-P14 rule scripting), §93 (analysis pipeline 27-check + intent + cost), §94 finding (max_marks reversal), §97 (cross_representation audit), §98 (watchdog cross-run automation).

| # | Item | Status | Verify |
|---|---|---|---|
| 4.4.b.1 | `diag_pattern_match.py` P1-P14 rules implemented (P9 deferred) | ✅ | `python3 scripts/analysis/diag_pattern_match.py --run-dir <run>` |
| 4.4.b.2 | Per-cell failure-mode breakdown | 🟡 TBD post-rerun | Add to `make analysis` pipeline |
| 4.4.b.3 | Cross-cell pattern comparison (e.g. P14 URL self-loop %) | 🟡 | Aggregate across 16 cells |
| 4.4.b.4 | Pattern-rule κ for 5-bucket failure-mode mapping (per §1.5.2) | 🔴 TBD | Spot-check P-rule output vs human label |
| 4.4.b.5 | **Analysis pipeline 4-dimension Evidence Framework** (笔记 §106) | ✅ pre-spec'd | `scripts/analysis/aggregate_phantom_lift.py` (Outcome 0c/0d) + `aggregate_routing_auroc.py` (0g) + `axis_effect_size.py` (1a/1b) + `axis1_microbehavior.py` (2a-2e) + `aggregate_cross_site.py` (3a-3c) + `figures/` per-outcome scripts |
| 4.4.b.6 | **`layered_status.py`** — live evidence layer status | ✅ exists | `docs/analysis/layered_evidence_status.md` snapshot |
| 4.4.b.7 | `compare_b0_b1.py` + `aggregate_cross_site.py` triggered automatically post-condition | ✅ §98 | watchdog `_run_cross_run_analysis` |

## §4.4.c Reference framework integration (笔记 §106 4-dim Evidence Framework)

**Source**: 笔记 §106 (4-dimension Evidence Framework), §108 (Phantom space refinement, evidence/explanation separation).

| # | Item | Status | Verify |
|---|---|---|---|
| 4.4.c.1 | 4-dimension Evidence Framework applied to paper §1+§4 organization | ✅ pre-spec'd | paper_planning §3 + paper drafts |
| 4.4.c.2 | Phantom space evidence/explanation separation (Zoom 1-4) | ✅ pre-spec'd | paper_planning §1 + phantom_space.canvas |
| 4.4.c.3 | §100 SoM screenshot OCR ground truth probe — used for §5 mechanism evidence | ✅ data exists | `docs/analysis/.../som_ocr_probe.md` |
| 4.4.c.4 | §103 4-mode routing arm finding (paper §1 hook) | ✅ pre-spec'd | paper §1 narrative |
| 4.4.c.5 | §94 max_marks reversal as B1 capability-modulated finding (paper §7) | ✅ documented | section 4 limitations or section 7 cross-capability |

## §4.5 Evaluator Independence Verification

| # | Item | Status | Verify |
|---|---|---|---|
| 4.5.1 | VWA evaluator code unchanged from upstream | ✅ | `git diff upstream/main -- external/visualwebarena/evaluation_harness/` |
| 4.5.2 | GPT-4o-mini judge prompt template pinned | 🟡 | `helper_functions.py:llm_fuzzy_match` no edits |
| 4.5.3 | Judge model temperature explicit (=0) | 🟡 | Verify; if non-zero disclose |
| 4.5.4 | Episode-level eval reproducibility (N=20 spot-check) | 🔴 TBD | Add `eval_reproducibility_check.py` |
| 4.5.5 | Cross-evaluator-version sensitivity (Protocol B) | ✅ §115 | reeval_audit_protocol.md |

## §4.6 Audit Trail & Reproducibility (reviewer-defensible chain)

After rerun, the following chain reconstructs any cell's adjusted_SR:


exec
/bin/bash -lc "sed -n '1,260p' docs/checkpoints/pre_run/negative_results_registry.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
# Negative Results Registry

> Open-science discipline (Nosek et al. 2015; Lipton & Steinhardt 2018):
> failed pilots, retracted framings, and abandoned hypotheses are documented
> here so the final paper narrative is **constrained by what we tried**, not
> just by what worked. Reviewers who diff archived branches against published
> claims should find every framing pivot recorded.
>
> This addresses audit constraints **H2** (negative-result registry) and
> **D8/H5** ("controlled characterization" framing — see paper_planning §21-§22).

## Entry format

| # | Date | Claim / framing (retracted) | Replaced by | Why retracted | Paper action |
|---|------|------|------|------|------|

## Entries

| # | Date | Claim / framing (retracted) | Replaced by | Why retracted | Paper action |
|---|------|------|------|------|------|
| **1** | 2026-04-28 | **Phantom-DOM 18 modes scope** (full factorial expansion across image / SoM / prompt axes) | 5-mode scope (DOM / SoM / Vision / phantom_som / phantom_dom) | 18 modes exceeded paper-grade focus; 13 ablation modes added marginal information at high compute cost | §3 paper-grade scope explicit; pre-rerun audit §1.1 enforces |
| **2** | 2026-05-01 | **Phantom-SoM is hidden 4th routing arm** | **Phantom routing space (3 arms: P-text / P-prompt / P-SoM) sharing 4-fold drop-in property** | B0 reddit 6-mode oracle vs 3-mode +7.14pp [3.81, 10.48] sig + 3 arms drop-one all sig → "1 arm" framing literally inaccurate; "1 routing dimension" stronger venue claim | Paper hook §1 reframe; provisional pending data confirm; advisor sync Q3 |
| **3** | 2026-05-01 | **8-corner 2x2x2 cube factorial design** as paper §2 axis | **M1/M2 mechanism activation 2x2** (LLM internal state level, not prompt structure level) | Prompt-text coupling ≠ mechanism activation coupling; 8-corner conflated levels | Memory `project_paper_hook.md` retract list; paper §2 rewrite |
| **4** | 2026-05-01 | **6-corner asymmetric grid** (a/b × c/¬c × 1/2) | M1/M2 2x2 (4-corner) | Same level-confusion as #3 | Same retract list |
| **5** | 2026-05-01 | **(a)(c) prompt decomposition** as paper axis | Evidence/Explanation 双层 + Zoom 1-4 | Decomposition was prompt-structure thinking, not mechanism thinking | Same retract list |
| **6** | 2026-05-01 | **"Three-layer mechanism argument" (Layer 1/2/3)** naming | Evidence/Explanation 双层 + Zoom 1-4 hierarchy | Naming overlap with neural-network "layer" caused reader confusion | Same retract list |
| **7** | 2026-05-01 | **"Approach 1 vs Approach 2" dichotomy** | Approach 2 = Zoom 1 (architectural completeness); "Approach 1" was not a single thing | Dichotomy was strawman | Same retract list |
| **8** | 2026-05-01 | **"First inference-time substitution / first deployment of text-only or marked observations"** novelty claim | "Controlled behavioral characterization of phantom configurations" | Industry artifacts (yang2023som SoM-Mark, zheng2024seeact, yang2025magma) precede our deployment claim; honesty matters for venue review | Paper §1 + related-work rewrite (audit D8/H5 — codex-delegated) |
| **9** | 2026-05-01 | **SteerMoE-style expert routing self-probe** for B0 | Zoom 4 future work direction (paper §8), no self-probe | B0 is proxy API → model internals invisible; local 235B deploy budget exceeds RunPod $200 allocation | Paper §8 future work |
| **10** | 2026-05-06 | **§111 task-0 single-task "L11 flips 93% match"** as paper §5 representative finding | 24-task aggregate Stage 2B L17 Holm-significant (p_Holm=0.011 \*\*) | Task-0 was distribution outlier (some tasks fully flip, some don't disrupt); single-task evidence cherry-pick | 笔记 §117.4 + paper §5 cite aggregate not task-0 |
| **11** | 2026-05-06 | **§111.5b "reverse direction null at all layers" as asymmetric encoding evidence** | 15-task aggregate Stage 2C reverse shows L11+L17 Holm-significant (p_Holm=0.044 / 0.033 \*); reverse magnitude **identical** on strong-tier (Δ=-0.193) and reverse-tier (Δ=-0.193, Welch p=1.000) | §111.5b was N=1 (task 0 reverse) theoretical extrapolation, not measured aggregate; 15-task scaled-up overturned the asymmetry hypothesis | 笔记 §117.2 reframe + paper §5 mechanism = "bidirectional mid-layer L11-L17 disruption" pending cross-site confirmation cells F/G |
| **12** | 2026-05-09 | **"Cell E random-injection should produce mid-layer L17 dip if mechanism is generic"** (null hypothesis for specificity check) | Cell E shows random Gaussian destroys output uniformly at all layers (overlap 1.00→0.03), specificity ratio random-LD/real-LD = 5-19× across layers | Mechanism IS content-specific — random control didn't produce mid-layer pattern | 笔记 §117 update pending; paper §5 control PASSED audit G6 ✓ |

## Pivots that did NOT retract (data confirmed framing)

These are kept here so reviewers can verify the registry is symmetric (we
report data-confirmed framings as well as data-broken ones):

| # | Date | Original framing | Confirmed by | Paper action |
|---|---|---|---|---|
| C1 | 2026-04-26 | Phantom-SoM 4-fold drop-in property (cost ≈ DOM / latency ~50% / signal AUROC ≥ baseline / drop-one 1.7-3.8pp) | B0 reddit Phase A archived data + drop-one sig | Paper hook §1 (provisional pending 16-cell rerun) |
| C2 | 2026-05-09 | Mid-layer L11-L17 mechanism Holm-significant disruption | 4 cells: A (Holm L17 p=0.011), B (Holm L11+L17), D (Holm L11+L17 p=0.006/0.008) — 3/4 cells Holm-confirmed | Paper §5 mechanism evidence |

## Paper § action items derived from this registry

1. **Paper §1 + Related Work**: rewrite to "controlled characterization" framing, NOT "first inference-time substitution" (entry #8). Acknowledge yang2023som / zheng2024seeact / yang2025magma industry artifacts as context. (Codex-delegated, audit D8/H5.)
2. **Paper §2 Background**: confirm M1/M2 2x2 framework + Evidence/Explanation 双层 + Zoom 1-4. Retract earlier 8-corner cube / 6-corner / (a)(c) decomposition / Three-Layer / Approach 1-2 framings (entries #3-#7). (Already done in `paper_planning.md §2` reframe; final prose confirms.)
3. **Paper §5 Mechanism**: cite 24-task aggregate L17, NOT §111 task-0 single-case (entry #10). Show forward+reverse symmetry from 4-cell 2x2 (entry #11). Cite cell E random-injection control specificity ratio (entry #12).
4. **Paper §8 Discussion**: SteerMoE-style probe is future work, not self-conducted (entry #9).
5. **Paper hook (§1)**: phantom routing space (3 arms) — retracted from "4th arm" (entry #2). State "provisional pending 16-cell rerun" until R1-R5 framing rules (preregistration.md §2) trigger.

## Future entries (placeholder)

When framings shift in upcoming work, append below. Common triggers:

- 16-cell rerun outcome inconsistent with archived data → entry for "phase A pre-fix data was over-optimistic" (audit F1)
- Cells F/G reddit cross-site shows null mid-layer disruption → entry for "phantom mechanism is cls-specific, not universal" + paper §5 scope reduction
- Advisor sync 5/X feedback retracts a hypothesis → entry for "H_X retracted per advisor"
- Reviewer pre-print feedback identifies a new failed assumption

## Caveat

This registry is honest but incomplete — early ad-hoc explorations (pre-2026-04-28)
are not all logged. The discipline started with Phase A 4-cluster bug fix
wave (笔记 §107) when paper-grade re-run was first scoped. Entries before
that date are reconstructed from chronicle notes / paper_planning decision
log / memory `project_paper_hook.md` retract list, not from contemporaneous
record. Future paper revisions log all pivots from this point forward.

exec
/bin/bash -lc 'rg -n "''^(#|##|###)|H[1-8]|§5'"\\.X|post-hoc|explor|reproduc|R[1-5]|complete-case|missing|Holm|bootstrap|random-effects|N≥10|N>=10\" docs/checkpoints/pre_run/preregistration.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
13:# Phantom-SoM Pre-Registration (Draft)
17:> **Reading order**: §1 epistemic structure (why this framework) → §2 hypotheses (H1-H6 + framing rule) → §3 multiple-comparison family declaration → §4 locked analysis choices → §5 exploratory disclosure → §6 witness mechanism.
26:## §1 Epistemic Structure (why this pre-registration shape)
34:3. **Framing decision is data-conditional, not data-prediction**: paper §1 hook framing depends on which combination of H1-H3 holds. The rule is pre-registered (R1-R5 below) so reviewers can verify the framing-to-data mapping is not post-hoc.
36:4. **Theory predictions (别扭, capability-modulated reversal) are post-hoc explanatory**: these frameworks were developed *after* observing N=4 pre-Phase-A cells; treating them as pre-registered hypotheses would be epistemically dishonest. Paper prose explicitly marks them as post-hoc.
38:5. **Multiple-comparison family discipline**: gating tests (PRIMARY + STRUCTURAL) have explicit Holm-corrected family m count. Exploratory tests (EXPLORATORY family + post-hoc) are reported with adjusted p-values for transparency but NOT used to gate paper claims.
44:## §2 Hypotheses
46:### PRIMARY family (gates paper claim)
48:#### H1 — Hero deployment claim (P-SoM is hidden routing arm)
52:- **H1(i)** Pooled DerSimonian-Laird random-effect meta-analysis reaches significance at Holm α=0.05 (PRIMARY family m=1 test, no within-family correction needed).
53:- **H1(ii)** ≥ K_h1 of N_cells individually Holm-significant at α=0.05 within the per-cell P-SoM sub-family (m = N_cells), where **K_h1 = 0.75** (commit-locked, see §4).
54:- **H1(iii)** Pooled magnitude θ_RE ≥ 1.0pp; TOST equivalence at margin **δ = 1.0pp** rejected (commit-locked).
56:#### H2 — 4-fold drop-in property (P-SoM specifically)
63:- **(d) Drop-one magnitude** — folded into H1(iii); P-SoM contributes ≥ 1.0pp lift on average.
65:#### H3 — Phantom space 2-axis empirical structural claim
69:- **H3(i)** axis 1: |P-text ∖ P-SoM| unique-count > 0 with bootstrap 95% CI excluding 0 in ≥ K_h3 of N_cells.
70:- **H3(ii)** axis 2: |P-prompt ∖ P-SoM| unique-count > 0 with bootstrap 95% CI excluding 0 in ≥ K_h3 of N_cells.
71:- **H3(iii)** Per-cell unique-count threshold: ≥ 2 tasks (≈ 1pp at N=234 to N=210); 1 task is noise floor.
76:- Primary gating: bootstrap CI on unique-count, 1000 resamples.
77:- Secondary report: McNemar exact one-sided directional asymmetry test (informational only — McNemar tests if one axis dominates the other in unique contribution; H3 only requires non-emptiness, not dominance).
78:- Multiple-comparison: Holm-Bonferroni step-down per axis sub-family (axis 1: m = N_cells; axis 2: m = N_cells).
80:### EXPLORATORY family (reported with corrections, NOT gating)
82:#### H4 — P-text / P-prompt drop-one magnitude
84:Reported per cell + meta-pooled (DerSimonian-Laird) for transparency. Holm-Bonferroni and BH FDR q-values reported. No pre-registered ranking commitment.
86:Paper §4 prose **must** explicitly flag: "exploratory analysis; not pre-registered for paper hook gating; magnitudes interpreted descriptively."
88:### POST-HOC family (theory tested on data that motivated it)
90:#### H5 — 别扭 (mismatch) framework predictions
92:The 4 distinguishing predictions in 实验笔记 §108.16 are tested against 16-cell data. The framework was developed after observing N=4 pre-Phase-A cells; this is **post-hoc**.
94:Paper §5 prose **must** explicitly flag: "post-hoc theoretical framework, validated on the same data motivating it; no formal significance gating."
96:#### H6 — Capability-modulated reversal (B0 vs B1 axis preference)
100:Paper §7 prose **must** explicitly flag: "post-hoc finding; no pre-registered prediction."
102:### ROUTER family (gates Section 6 routing claim — **pending advisor 5/5 lock**: paper-1 PRIMARY vs paper-2 deferred)
104:#### H7 — Tier 1 oracle router lift over best-single-mode baseline (offline supervised)
108:- **H7(i)** Pooled DerSimonian-Laird random-effect meta-analysis on lift reaches Holm α=0.05 (PRIMARY family m=1 if paper-1 / SECONDARY informational if paper-2).
109:- **H7(ii)** ≥ K_h1 of N_cells individually Holm-significant on per-cell lift, bootstrap 95% CI lower-bound > 0.
110:- **H7(iii)** Pooled magnitude θ_RE ≥ 1.0pp; TOST equivalence at margin δ=1.0pp rejected (same δ as H1).
116:- Multiple-comparison: Holm-Bonferroni step-down within H7 sub-family m=N_cells.
118:**Status**: ⏸️ pending advisor 5/5 lock decision — if paper-1 PRIMARY, H7 gates Section 6 routing claim; if paper-2 deferred, H7 reported as informational with explicit "paper-1 hook does NOT depend on H7-H8".
120:#### H8 — Tier 2 first-step trigger router (online, test-leak-free)
124:- **H8(i)** Tier 2 router lift over Tier 1 oracle baseline ≥ 0 with bootstrap 95% CI excluding −1.0pp (paper claims Tier 2 ≈ Tier 1 within deployment-grade tolerance, given Tier 2 is leak-free and deployment-realistic).
125:- **H8(ii)** Tier 2 router lift over best-single-mode-baseline ≥ 1.0pp, ≥ K_h1 cells Holm-significant.
127:**Status**: ⏸️ pending advisor 5/5 lock — same as H7.
131:### FRAMING DECISION RULE (pre-registered, data-conditional)
137:| **R1** | H1 holds AND H2 (a)(b)(c) all hold AND H3(i) holds AND H3(ii) holds | "Phantom routing space (M1/M2 2-axis empirical structure); P-SoM as deployment hero, P-text/P-prompt as structural ablation arms validating axis decomposition." | STRONGEST |
138:| **R2** | H1+H2 hold AND only one of H3(i)/(ii) holds | "Phantom routing space (single-axis empirical structure) with P-SoM as deployment hero; remaining axis decomposition theoretical (Zoom 1 architectural argument only)." | MODERATE-STRONG |
139:| **R3** | H1+H2 hold AND neither H3(i)/(ii) holds | "Phantom-SoM is hidden 4th routing arm; M1/M2 axis decomposition supported by Zoom 1 architectural argument only, not empirically validated by ablation." | MODERATE (= 04-30 fallback) |
140:| **R4** | H1 holds AND H2 partially fails (e.g., (a) cost or (b) latency fails on some site) | "Phantom-SoM partial drop-in" + §4 disclosure of failed sub-claim. | WEAK; substantial revision |
141:| **R5** | H1 fails (pooled meta sig fails Holm OR < K_h1 cells individually sig) | Paper death scenario: pivot to VWA bug audit paper (§107 4-cluster fix as primary) OR abandon. Decision deferred to advisor sync at fail time. | n/a |
145:## §3 Multiple-Comparison Family Declaration
148:- H1(i) pooled meta: m = 1 (no within-family correction).
149:- H1(ii) per-cell P-SoM Holm: m = N_cells.
150:- H2 sub-claims (a)(b)(c)(d): m = 4 × N_cells (each per-cell sub-claim test).
151:- Method: Holm-Bonferroni step-down per H-sub-family (Holm 1979).
154:- H3(i) axis 1 per-cell: m = N_cells (bootstrap CI lower-bound > 0 test).
155:- H3(ii) axis 2 per-cell: m = N_cells.
156:- Method: Holm-Bonferroni step-down per axis sub-family.
160:- H7(i) pooled meta lift: m = 1 (no within-family correction).
161:- H7(ii) per-cell Tier 1 lift Holm: m = N_cells.
162:- H7(iii) folded into H7(i) magnitude/TOST.
163:- H8(i) Tier 2 vs Tier 1: m = 1.
164:- H8(ii) Tier 2 vs best-single-mode-baseline: m = N_cells.
165:- Method: Holm-Bonferroni step-down per H-sub-family.
169:- H4 P-text/P-prompt drop-one per cell: m = 2 × N_cells.
170:- Best-signal-per-mode characterization (Register III AA, Section 6 portfolio finding): per (mode, signal) AUROC reported, Holm-corrected within mode for transparency.
171:- Method: Holm-corrected and BH q-value reported for transparency.
175:- H5 别扭 4 predictions.
176:- H6 capability-modulated reversal interaction GLM.
177:- Disclosed as "post-hoc theoretical analysis tested on motivating data" in paper prose.
181:## §4 Locked Analysis Choices (pre-data)
186:| **CI method** | 1000-resample task-level paired bootstrap, **percentile** intervals (BCa as sensitivity check, not primary) | Existing infra in `aggregate_phantom_lift.py`. Percentile chosen primary because: (a) paired-bootstrap on bounded proportion (SR ∈ [0,1]) → BCa acceleration estimate is unstable at small N per cell; (b) Cohen's h transformation already symmetrizes; (c) percentile is the canonical reporting in WebArena/VWA precedent. BCa shown as appendix sensitivity check. |
188:| **Bootstrap clustering** | **Single-level (task_id)** for primary, no nested cluster (cell × site) bootstrap | Justification: meta-analysis at cell level is separate (`aggregate_phantom_meta.py` random-effects + I²/τ²); within-cell bootstrap only re-samples tasks. Multi-level cluster would double-count uncertainty already captured by random-effects meta. Lock: percentile + task-id unit + no nested cluster (B2 lock 2026-05-09). |
189:| **Sig threshold** | Holm α=0.05 within respective family | FWER control |
190:| **Effect size (binary)** | Cohen's h with bootstrap CI | Standard for proportion comparisons |
191:| **Effect size (continuous)** | Cohen's d with bootstrap CI | For cost/latency H2(a)(b) |
192:| **TOST equivalence margin δ** | **1.0pp** | ≈ 2 tasks in N=234, matches per-cell bootstrap SE; smaller is within sampling noise floor |
193:| **H1 K_h1 cell-pass threshold** | **0.75** | Allows ~25% capability-outlier cells (e.g., B1 shopping power-limited); not so strict that single-cell noise breaks claim |
194:| **H3 K_h3 cell-pass threshold** | **0.67** | Lower than K_h1 because structural < deployment commit |
195:| **H3 unique-count floor** | **≥ 2 tasks per cell** | 1 task is sampling noise; 2 tasks ≈ 1pp at N=234 |
202:| **Mode operational definitions** | 6 modes per paper §3 (text format × prompt × image): DOM (AXTree+DOM-prompt+no image) / SoM ([SOM_MARKS]+SoM-prompt+image) / Vision (no text+image) / P-text ([SOM_MARKS]+DOM-prompt+no image) / P-prompt (AXTree+SoM-prompt+no image) / P-SoM ([SOM_MARKS]+SoM-prompt+no image) | Stipulative — **no post-hoc episode reclassification**. Episodes systematically excluded per (FP filter / N-floor / data-corruption flag), never redefined which mode they belong to. Edge cases (empty AXTree / 0 marks / OCR-empty) follow `condition_meta.json` declared mode |
203:| **Routing signal universe** | `aggregate_routing_auroc.py` enumerated set: ep_mean_verbalized / ep_min_verbalized / max_repeat_streak / action_diversity / url_revisit_count / url_revisit_max / action_unique_types / url_unique_count / ep_mean_logprob / ep_min_logprob (last 2 B1-only) | **No post-hoc engineered features** for router input. Best-signal-per-mode characterization is exploratory (§5) — paper §6 portfolio finding, not pre-registered prediction |
207:| **Best-single-mode baseline (H7/H8 anchor)** | Per cell: mode with highest mean adjusted-SR on train fold | Used as comparison anchor for router lift; **train/test split-stratified** to prevent test leak |
210:| **Heterogeneity (random-effects, Q, I², τ²) pre-spec** (audit B8) | (a) **Primary estimator**: random-effects DerSimonian-Laird via `aggregate_phantom_meta.py` (already implemented). (b) **Heterogeneity reporting**: report Cochran Q (chi² test of homogeneity), I² (% of total variance attributable to between-cell heterogeneity), τ² (between-cell variance). (c) **Interpretation thresholds (pre-specified)**: I² < 25% = "low heterogeneity, pooled mean is primary"; 25%-50% = "moderate, report both pooled + per-cell"; 50%-75% = "high, per-cell estimates are primary, pooled is summary"; > 75% = "very high, do not pool — report only per-cell + heterogeneity-source analysis (site / model / task-pool)". (d) **Heterogeneity-source decomposition**: when I² > 50%, report meta-regression by site (cls / red / shop) and by model (B0 / B1) to identify dominant variance source. | Higgins & Thompson 2002 (I² thresholds). Per-cell estimates always shown alongside pooled, so heterogeneity is never averaged away. Lock 2026-05-09. |
211:| **K-of-N rule scope** (audit B9 power-corrected) | The **K_h1=12/16 / K_h3=11/16** thresholds are retained as **secondary transparency checks** (count of cells *individually* clearing α=0.05 Holm), **not as gates on H1/H3 paper claims**. **Primary detection** = (a) DerSimonian-Laird random-effects meta-analysis on cells N≥10 (B8 lock above) + (b) TOST equivalence on N=910 pooled tasks at δ=1.0pp. Per `docs/analysis/cross_sites/power_analysis.md` §3-§5, K-of-N family power at observed effect sizes (1-5pp) is < 10%; the rule is calibrated for ≥7pp effects. This recharacterization is consistent with the original §4 "Primary metric" + B8 random-effects lock — K-of-N was always a transparency aggregator, not the primary test, and the corrected power analysis makes that explicit. | `power_analysis.py` bug (stale interpretation block) discovered 2026-05-09; fixed in same commit. K-of-N values themselves unchanged; only the framing as "secondary transparency vs primary gate" is added. Lock 2026-05-09. |
215:## §5 Exploratory (NOT pre-registered, paper must explicitly flag)
217:The following analyses are exploratory and cannot be used to gate paper claims. Paper prose **must** mark them explicitly as "exploratory" or "post-hoc":
219:- Per-task category × mode heatmap exploration (`fig0e`)
222:- 别扭 framework (H5) — post-hoc, theory developed on motivating data
223:- Capability-modulated reversal (H6) — post-hoc cross-capability finding
224:- **Best-signal-per-mode characterization** (Register III AA novelty, Section 6 portfolio finding): which routing signal works best for which mode is reported as exploratory characterization, NOT pre-registered prediction. Per-(mode, signal) AUROC table reported with Holm correction within mode for transparency.
225:- **Router feature engineering exploration beyond locked signal universe** (§4): any new feature added post-data-lock is exploratory, NOT gating H7/H8 claim.
226:- **Cross-site asymmetry as site-class adaptive routing primitive** (Register IV HH novelty, §1 + §6): reported as exploratory framing of cls/red mode-preference reversal, NOT pre-registered prediction.
228:- Any post-hoc cell subsetting beyond H1-H8 family scope
231:### §5.X Post-hoc Layer Selection Disclosure (Stage 2 Mechanism, audit G5)
234:at **L17** (3 of 4 cells Holm-significant on `token_overlap_to_target`, p_Holm <
235:0.05; cell D L11+L17 strongest p_Holm = 0.006/0.008 \*\*). The L11/L17 layer
236:selection has the following **explicit pre-vs-post-hoc structure**:
242:| **Stage 2B 24-task aggregate (cell A)** | L17 Holm-significant (p_Holm = 0.011 \*\*) — confirmed Stage 2A peak | **Confirmatory** — independent metric agreement |
243:| **Stage 2C reverse 15-task (cell B)** | L11 + L17 Holm-significant — direction-paired confirmation | **Confirmatory** |
244:| **Cell D (rev × strong-tier 24)** | L11 + L17 strongest (p_Holm = 0.006/0.008 \*\*) | **Confirmatory** — cross-tier replication |
249:mitigate the multiple-comparison concern, all per-direction tests use Holm-
268:## §6 Witness Mechanism
270:### (a) Internal witness — Git commit + advisor email
279:   - (7) **Router paper-1-vs-paper-2 decision**: H7-H8 PRIMARY (paper-1) or SECONDARY-informational (paper-2 deferred)
281:   - Plus lock H-list (H1-H8 family declaration final).
284:4. Advisor sends single-line confirmation email: "I witness pre-registration of phantom-SoM hypotheses (H1-H8) and 8 lock decisions as of <git SHA> <date>." Email archived in `.witness/preregistration_witness.eml` (gitignored, local-only).
286:### (b) External witness — OSF DOI (optional, paper-time)
298:## §7 Reproducibility Scope Statement (audit A14, F3)
300:**Public release scope** — what reviewers / replicators can reproduce from the released artifact:
304:| **B1 (Qwen3-VL-4B local)** | **Fully reproducible** byte-identical | HF model SHA pinned (`ebb281ec70b05090aa6165b016eac8ec08e71b17`) + greedy decoding + seed=42 (`configs/exp_v2_base.yaml`) + `_seed_global_rng()` per (cond, seed) iteration + env_snapshot.json per run + git commit SHA in run_manifest. Re-running produces byte-identical action traces, hidden states, and aggregate SR. |
305:| **B1 mechanistic Stage 2** | **Fully reproducible** | Same as B1 plus `--random-seed 42` for `--random-inject` (cell E). `archive_subset_b1_{cls,reddit}/` (curated mirage tasks + cached observations + screenshot_annotated) committed for cross-machine replication without needing full archive. |
306:| **B0 (Qwen3-Omni-235B-Thinking via proxy API)** | **Verifiable from traces, replayable subject to API access** | All B0 episodes log full request/response traces + temperature=0 server-side. Re-running depends on: (a) proxy API endpoint availability, (b) model server-side determinism (best-effort, not guaranteed at temperature=0). For paper claims, B0 is "one controlled stochastic sample with bootstrap task uncertainty" — replicators verify via released traces or rerun under same proxy / Anthropic-native API access. |
308:| **Evaluator** | **Fully reproducible** byte-identical | `evaluator_code.combined_sha256` recorded per run. T0/T1/T2/T3 evaluator-change protocol (`evaluator_change_protocol.md`) governs post-lock changes — same paper requires dual-reporting for any T0 fix. |
309:| **Mechanism analysis (Stage 2 patching)** | **Fully reproducible** | Greedy decoding + seed=42 + Holm-corrected paired t-test + 1000-resample percentile bootstrap (seed=42 in `stage2_layer_significance.py`). Per-task per-layer `patching_continuation_results.json` released for re-aggregation. |
313:> "All B1 (local Qwen3-VL-4B) experiments, including agent traces, mechanistic activation patching, and aggregate analysis, are fully reproducible given the released code (commit SHA), pinned HF model revision, and seed configurations. B0 (proxy-API Qwen3-Omni-235B) results are verifiable from released traces and replayable subject to API access; B0 server-side decoding determinism is best-effort under temperature=0 and reported as a single controlled stochastic sample with task-level bootstrap uncertainty. The VWA environment is reproducible given the pinned VWA submodule commit and Docker images. Cross-benchmark (WebArena) results are out of scope for this paper unless explicitly reported in the appendix."
321:## Appendix A — Decision Log
326:| 2026-05-03 | H3 structural test changed from McNemar exact (asymmetry) to bootstrap CI (non-emptiness) | McNemar tests directional dominance (which axis dominates), but H3 only requires non-empty unique contribution; bootstrap CI on count > 0 is the right test |
327:| 2026-05-03 | TOST δ = 1.0pp locked (was 0.5pp draft) | 0.5pp = 1 task in N=234 too liberal; 1.0pp = 2 tasks ≈ bootstrap SE noise floor; statistically principled |
328:| 2026-05-03 | K_h1 = 0.75 cell-pass threshold for H1 | Allows ~25% capability-outlier cells; not so strict as to break on single-cell noise |
329:| 2026-05-03 | K_h3 = 0.67 cell-pass threshold for H3 | Lower than K_h1 because structural < deployment commit |
330:| 2026-05-03 | Disconfirmation rule changed from "any cell fail" to data-conditional R1-R5 framing rule | "Any cell fail" too strict given single-cell power limits; framing rule maps data outcomes to paper hook revisions transparently |
331:| 2026-05-04 | Pre-registration scope expanded — added H7-H8 router family + 6 §4 lock entries (mode operational defs / routing signal universe / train-test split protocol / failure-mode classification rubric / N_cells final scope / best-single-mode baseline anchor) | User audit prompt 5/4: "preregistration.md 还需要锁 Held-out router claim / router baselines train-validation-test split / routing signals / mode definition 这些吗". Claude added 2 more (failure-mode rubric / N_cells). Deferred 3 advisor lock decisions: (a) H7-H8 router family paper-1 vs paper-2 / (b) N_cells 13/14/16 final / (c) split protocol k-fold vs LOSO. Witness §6 expanded from 5 commits → 8 commits |
332:| 2026-05-05 | Advisor sync 5/5 partial outcome — early-stop A locked (cancel全 mechanism); compute path locked (advisor 5090 → Rancher H100 → RunPod backup); paper split direction discussed but Mechanistic-nested-vs-independent + threshold detail not finalized due to network drop | Advisor explicit confirm early-stop cancel + compute paths; paper split + threshold lock deferred to email follow-up via `docs/checkpoints/advisor_sync_5_5_followup.md` |

exec
/bin/bash -lc "sed -n '231,267p' docs/checkpoints/pre_run/preregistration.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
### §5.X Post-hoc Layer Selection Disclosure (Stage 2 Mechanism, audit G5)

Stage 2 mechanistic activation patching identified mid-layer disruption peaking
at **L17** (3 of 4 cells Holm-significant on `token_overlap_to_target`, p_Holm <
0.05; cell D L11+L17 strongest p_Holm = 0.006/0.008 \*\*). The L11/L17 layer
selection has the following **explicit pre-vs-post-hoc structure**:

| Stage | Layer rationale | Status |
|---|---|---|
| **Stage 2A logit_shift pilot** (5-task aggregate, 笔记 §111.5) | L17 emerged as peak in independent `logit_shift` metric | **Hypothesis-generating** — first-pass discovery |
| **§111 task-0 single-task patching** | L11 flipped 93% match in N=1 task | **Hypothesis-generating** — distribution outlier (acknowledged 笔记 §117.4) |
| **Stage 2B 24-task aggregate (cell A)** | L17 Holm-significant (p_Holm = 0.011 \*\*) — confirmed Stage 2A peak | **Confirmatory** — independent metric agreement |
| **Stage 2C reverse 15-task (cell B)** | L11 + L17 Holm-significant — direction-paired confirmation | **Confirmatory** |
| **Cell D (rev × strong-tier 24)** | L11 + L17 strongest (p_Holm = 0.006/0.008 \*\*) | **Confirmatory** — cross-tier replication |

**Disclosure**: Layers L11 and L17 were not pre-registered before Stage 2 data
collection; they emerged from Stage 2A pilot (the *hypothesis-generating* phase)
and were confirmed by Stage 2B/2C scaled-up data (the *confirmatory* phase). To
mitigate the multiple-comparison concern, all per-direction tests use Holm-
Bonferroni correction across the canonical layer grid (L0/5/11/17/23/29 vs L35
baseline) — this catches the "any layer might pop" multiple-testing concern.

**Reviewer-defense**: We do NOT claim pre-registered layer prediction. We claim
that the **same** mid-layer region (L11-L17) emerges across (a) Stage 2A
logit_shift, (b) Stage 2B forward overlap-to-target, (c) Stage 2C reverse,
(d) Cell D cross-tier rev-on-strong. Convergence across 4 independent analysis
paths constitutes confirmatory evidence even without pre-registered layer
prediction. Cell E random-injection control (G6) further demonstrates content-
specificity, ruling out generic-injection alternative explanations.

**Future paper-grade improvement** (deferred to next iteration): full **leave-
one-out layer-selection** robustness — re-run patching on per-cell holdout
that excludes the layer that informed selection on the training cell, then
report the mid-layer pattern under that holdout.

---


exec
/bin/bash -lc "sed -n '1,110p' docs/checkpoints/paper_drafts/section1_intro.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
## 1. Introduction

Web agents act through representations. A browser state can be serialized as a DOM or Accessibility Tree, shown as a screenshot, or annotated with Set-of-Mark (SoM) labels that connect visible regions to discrete element IDs. Existing benchmarks and agents treat these as different observation modes: WebArena and Mind2Web popularized DOM-derived text for realistic web tasks, while VisualWebArena and SeeAct introduced visually grounded settings where screenshots and action grounding become central \citep{zhou2024webarena,deng2023mind2web,koh2024visualwebarena,zheng2024seeact}. Set-of-Mark prompting was designed for this multimodal setting: a marked image is paired with a textual legend so the model can refer to visual objects by number \citep{yang2023som}. Later multimodal-agent systems, including SeeAct and Magma, further explored marked-screenshot and omni-modal action-grounding paradigms rather than treating mark text as an isolated scientific variable \citep{zheng2024seeact,yang2025magma}. These are important precedents. We therefore do not claim to be first to deploy text-only, marked, or SoM-style observations. Our claim is about controlled characterization: isolating what changes when the annotated image is skipped while the text payload and prompt family are varied under the same task, model, and evaluation protocol.

This paper questions that bundling assumption as an experimental object. We characterize the **phantom routing space**: configurations on the "skip annotated image" boundary that retain some SoM-derived textual or prompt structure while removing the image. Its deployment-relevant representative is **Phantom-SoM**: the agent receives the SoM prompt and the `[SOM_MARKS]` textual element list, but no image. The structural controls are **P-text** (the `[SOM_MARKS]` text under the DOM prompt) and **P-prompt** (the SoM prompt over AXTree text). At the start of this project, Phantom-SoM looked like a broken ablation. The natural expectation was that removing the marked screenshot would collapse SoM into either a weak DOM surrogate or a nonsensical configuration: the prompt still suggests visual marks, but the visual substrate is absent. The data reject that expectation. Phantom-SoM solves tasks that DOM, full SoM, and Vision all miss, and on B0 reddit it matches or modestly exceeds full SoM by adjusted SR (**13.81% vs 10.48%, N=210**; the gap is within 2σ under the run-to-run variability we observe in same-condition repeats), while avoiding image-token cost. On classifieds, full SoM remains clearly stronger (**21.37% vs Phantom-SoM 14.53%, N=234**), the expected sanity check when marked screenshots carry real visual information.

Our first contribution is a controlled scientific evaluation of this phantom boundary. Across completed B0 VisualWebArena classifieds and reddit runs, we compare DOM, full SoM, Vision, and Phantom-SoM on the same task sets (**N=234 classifieds; N=210 reddit; same-task adjusted SR**) and use the P-text/P-prompt controls to test whether the effect collapses to one prompt trick or one text-format swap. Phantom-SoM is not the best single arm on every site, and we do not claim that it replaces full SoM. Its value is complementarity. Its task-success pool has low overlap with the established modes, with Jaccard similarity in the roughly **0.29-0.49** range against other arms, and its removal reduces the oracle. In drop-one oracle analysis, Phantom-SoM contributes **3.33 percentage points** of incremental oracle value on reddit (comparable to full SoM at **1.90 pp**) and **2.56 percentage points** on classifieds, placing Phantom-SoM consistently within the top routing-value arms despite using no image. Crucially, the cost of obtaining this configuration is essentially the cost of the DOM baseline: the `[SOM_MARKS]` block is produced by a regex pass over the same accessibility-tree text the DOM agent already consumes (interactive elements come pre-numbered as `[N] role 'label'`), so a deployment that can run DOM can run Phantom-SoM by changing what it forwards to the model: no bounding-box pipeline, no marked image, no extra inference modality. We therefore preserve the empirical **4-fold drop-in property** as the paper's practical finding: cost approximately DOM, lower image-stage latency, usable routing signal, and positive drop-one oracle value.

The cross-site asymmetry is itself informative: full SoM clearly outperforms Phantom-SoM on classifieds, where visually rich product listings make layout and appearance important, but not on reddit, where post and comment threads are more text-dominated. This is consistent with marked screenshots being most useful when visual grounding is task-critical.

Our second contribution is a mechanism account for why the ablation works. A 2-by-2 reddit ablation separates text representation from prompt family: DOM prompt versus SoM prompt, crossed with AXTree versus `[SOM_MARKS]` text. On the verified same-task subset (**N=48**), replacing AXTree with `[SOM_MARKS]` shifts exploration away from DOM-like search loops and toward Phantom-SoM-like quick decisions: the search-loop rate is **22.7% for DOM** but **10.8% for both Phantom-SoM and P-text**. The prompt knob appears elsewhere. DOM-prompt arms show the larger raw-to-adjusted false-positive gap (**6.25 pp; 3 N/A false positives**), while Phantom-SoM under the SoM prompt has a smaller gap (**2.08 pp; 1 N/A false positive**). Section 5 adds mid-layer mechanistic evidence: activation-patching effects concentrate in the L11-L17 region, and the random-injection control distinguishes content-specific phantom information from generic perturbation. The resulting two-knob view is simple: **text representation shapes how the agent explores; prompt wording tunes when it commits**. This aligns with prior evidence that language models are highly sensitive to prompt format \citep{sclar2024promptformat,mishra2022reframing}, but extends the claim from static prediction to multi-step web-agent trajectories.

Our third contribution is to show that this representation effect interacts with model capability. Section 5 diagnoses disagreement trajectories for B0 and B1, where B1 is a local Qwen3-VL-4B model. The failure distribution changes with capability: in the B0 disagreement slice, SoM failures are dominated by early finish or wrong commitment (**16/30, 53.3%**), whereas B1 SoM failures are dominated by visual-hijack/click-loop behavior (**19/27, 70.4%**), a **+43.7 pp** shift in mark-driven looping. At the same time, both models show DOM visual-missing failures on visual-bound tasks. This contrast supports the paper's central claim: observation representations do not merely add or remove information; they shape the exploration policy available to a given model.

We evaluate on VisualWebArena classifieds and reddit with B0, an API-backed Qwen3-VL-235B agent, and use B1 for capability-contrast analysis. Following the pre-registered R1-R5 framing rules, the scope of this paper is Qwen-family VisualWebArena characterization and explanation, not a claim of universal model-family generalization and not yet a learned deployment router. Routing exploitation is deferred to a follow-up paper. Section 2 situates the gap in web-agent representations, SoM, routing, and prompt-format sensitivity. Section 3 defines Phantom-SoM and the measurement protocol. Section 4 reports phantom-space and image-on baseline findings. Section 5 analyzes mechanisms and capability interactions. Section 6 discusses generalization, and Section 7 summarizes limitations and implications.

exec
/bin/bash -lc "rg -n \"§5\\.1|Method|method box|36|Qwen|architecture|activation|layer|L11|L17\" docs/checkpoints/paper_drafts/section5_mechanism_reddit.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
1:## 5.1 Method
3:We use activation patching as a behavioral causal test of the phantom boundary, following the clean/corrupt/source/target logic of IOI-style patching and recent patching protocols \citep{wang2023interpretability,zhang2024patching}. Citation audit: `wang2023interpretability` and `zhang2024patching` are intended bibkeys but currently NEEDS_BIB_ENTRY in `paper.bib`. For each task, the **source** prompt is `som`: task instruction, SoM prompt family, flat `[SOM_MARKS]` text, and annotated screenshot. The **target** prompt is `phantom_som`: the same task instruction, same SoM prompt, and same `[SOM_MARKS]` text, but no image. Source and target therefore differ only in the screenshot channel. All runs use the same task, same archived browser step (step 2), and the same local model, Qwen3-VL-4B-Instruct revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`. We index the 36 cached hidden-state layers as L0-L35, from embedding output through final post-block, matching the Stage 2 hook convention.
5:The continuation protocol is deterministic. For each task we first greedy-generate a 50-token source continuation and a 50-token target continuation (`do_sample=False`, `max_new_tokens=50`). We then run the source prompt once and cache per-layer hidden states for L0-L35. For each layer L, the target prompt is run with the source L-th hidden state injected at the final input-token position on the first forward pass; subsequent one-token decoding steps proceed normally through the model cache. The patched run then greedy-generates 50 tokens. The **forward direction** is `som -> phantom_som`, injecting visually grounded source states into the image-ablated target. The **reverse direction** is `phantom_som -> som`; the `--reverse` flag swaps roles so phantom states are injected into the full SoM run.
7:Each layer/task pair is scored against the two unpatched continuations. `token_overlap_to_source` is the fraction of generated positions where the patched output matches the source token sequence, and `token_overlap_to_target` is the same fraction against the target sequence. `ld_to_source` and `ld_to_target` are token-level Levenshtein distances from patched output to source or target, bounded from 0 to 50 under the fixed 50-token continuation. Higher `token_overlap_to_target` and lower `ld_to_target` mean the patch had no effect; the disruption signal is a drop in target overlap or a rise in target Levenshtein distance.
9:The layer-wise statistics follow the preregistration §4 bootstrap policy lock. For each direction and metric, tested layer L is compared with L35 using paired per-task differences L - L35. The canonical grid is L0, L5, L11, L17, L23, and L29, with Holm-Bonferroni correction across these six tests. For overlap metrics the paired t-test uses `alternative='less'`; for Levenshtein metrics it uses `alternative='greater'`. We report 1000-sample task-paired percentile bootstrap 95% confidence intervals on the mean paired difference, and Wilcoxon signed-rank is the non-parametric backup. The resampling unit is the task, not the token or episode.
13:For Cell E, each cached source tensor h is replaced by `randn_like(h) * h.std() + h.mean()` at the corresponding layer. This preserves activation magnitude while destroying task-specific structure. The specificity ratio is random-injection `ld_to_target` divided by real-source `ld_to_target`; it measures whether disruption comes from content-specific source information rather than generic non-zero injection.
15:Finally, preregistration §5.X discloses the layer selection as post-hoc. L11/L17 were not pre-registered: L17 emerged from the Stage 2A logit-shift pilot, and L11 from an early single-task continuation case. The confirmatory evidence is Stage 2B forward overlap, Stage 2C reverse overlap, and Cell D cross-tier replication. Together with the independent Stage 2A logit-shift path, this gives four convergent paths (logit shift, forward overlap, reverse overlap, cross-tier) under Holm-corrected grid testing, but not a pre-registered exact-layer prediction.
41:The outcome evidence suggests a calibration effect rather than a uniform success-rate increase. Phantom-SoM has the lowest false-positive rate among the B0 reddit modes (Outcome 0b, P-SoM FP rate 0.48%), and adding P-SoM as a single phantom arm contributes additional oracle tasks beyond the three standard modes (Outcome 0c, +P-SoM +2.56pp). The task-pool overlap also stays below the redundancy sentinel: P-text and P-SoM solve overlapping but not identical task sets (Outcome 0d, P-text<->P-SoM Jaccard 0.500). Thus the prompt is not just a cosmetic instruction layered on top of the same decisions; it changes which tasks enter the solved pool.

exec
/bin/bash -lc "rg -n \"§111|111\\.5b|§117|§118|Stage 2|selection-bias|reverse|null|task-0|over-interpretation|Cell [A-E]|cell [A-E]|Levenshtein|median\" docs/checkpoints/实验笔记.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
1351:- `proxy_api_agent.py`：`requests.post()` 包裹 `try-except (Timeout, ConnectionError)` 纳入已有重试循环；`resp.json()` 包裹 `JSONDecodeError`；`choices` 空列表检查；`usage`/`metadata` null 防御
1363:B0 classifieds task 154/155 暴露 auto-retry 链路三个漏洞：Osclass DB 短暂宕机 → agent 优雅退出 → `error: null` → 绕过重试链 → 永久卡在错误结果。
2827:- Section 3.2 image-token estimate 从 total cost 反推 → step-level `tokens.input` median (`4d63c9f`)
3546:**问题**: M1/M2 是从数据 reverse-engineered 的 axes, 怎么证明完备性 (没漏第 3 axis)?
3711:- 但 别扭 是 reverse explanation (gap-driven), M1/M2 是 forward (input-driven). 不等价。
3730:Drop-one ranking 跨 capability **direction reversed**:
3745:- B1 cls: prediction 4 reversed (capability-modulated)
3757:- §1 hook drop-one "1.7-3.8pp per arm" 应当 explicit 加 capability-modulated caveat ("magnitude 4× weaker on small VLM, direction reverses on text-vs-image axis preference")
3758:- §2 Theory framework 加 别扭 reverse-explanation layer (2x2 mismatch grid)
4721:### 111.3 Stage 2A First-Token Logit-Shift Patching — L17 peak weak signal
4733:### 111.4 Stage 2B Multi-Token Continuation Patching — Task 0 L11 paper-grade case study ⭐
4743:- 跟 source token-by-token 93% overlap, Levenshtein distance 1/15 token
4750:**Task 1 / 2 null effect**: source vs target diverge for **non-mirage reason** (different "thought" framing — task 1 source 抽象目标描述 / target page 状态描述; task 2 类似), patching 不能修. Confirms L11 specifically encodes image-derived content, not generic prompt-conditional state.
4754:### 111.5 Stage 2A vs 2B 一致性
4756:L17 (Stage 2A logit_shift peak, 5 task aggregate) vs L11 (Stage 2B continuation, 1 task case study) — 距离 6 layer ~17% depth gap. 不是同一精确层但**同一 mid-early region** (L11-L25), 跟 Tool Calling Linear Circuit (Qwen3-4B L23) 都属于 transformer 中段 "computed feature emergence" zone. 文献 cross-paper anchor.
4758:### 111.5b Stage 2C reverse direction — null effect, paper-grade asymmetry evidence ⭐
4760:**Setup** (commit `9fe3d84` `--reverse` flag): swap source ↔ target. Source = P-SoM (no image, mirage); target = SoM (with image, clean). Patch P-SoM's hidden state into SoM run at each layer.
4762:**Result**: 全 36 layer overlap→source = 0.51 (baseline), overlap→target = 1.0 (patched 完全 = target output unchanged). 任何层 patching 都 null effect 在反向.
4773:Forward 成功 + reverse null 是**经典 activation patching asymmetry**, 文献 (Zhang+2024 "How to use and interpret activation patching"; Wang+2023 IOI circuit) canonical finding. 解读:
4780:> "Activation patching reveals image-grounded perception is encoded distributively across image embedding tokens (positions 0-256+ in SoM mode input) and aggregated at last token through layer 11. Single-position last-token patching exhibits direction-dependent asymmetry: forward (SoM-into-P-SoM) at L11 fully recovers source's continuation (93% token overlap, LD 1/15) because P-SoM run lacks competing image tokens at earlier positions; reverse (P-SoM-into-SoM) at every layer has null effect because SoM run's earlier image tokens continue reinjecting image content via attention, overwriting the patched 'no-image' state. This asymmetry is consistent with Zhang+2024 'noising-resistant denoising' patterns in residual-stream interpretability."
4786:> "Activation patching of source's (SoM with image) hidden state into target's (P-SoM no image) run at layer 11 (Qwen3-VL-4B, 31% depth) recovers source's full output continuation token-by-token in clean mirage case studies (cls task 0: 93% token overlap, Levenshtein 1/15). Patching at any other layer (L0-L10, L12-L35) has null effect on patched output, identifying layer 11 as the causal mirage layer where image-grounded perception emerges before downstream abstraction. Layer 11 placement is consistent with Stage 2A first-token logit shift L17 peak (5-task aggregate, +1.08±0.21) and Tool Calling Linear Steerable Circuit (Anonymous 2026 ICLR, Qwen3-4B L23 family) mid-layer 'computed feature emergence' zone."
4794:5. ~~single direction (source→target)~~ → ✅ **§111.5b reverse direction completed**: forward L11 work + reverse null at all layers, asymmetry confirms image content distributed across early image tokens, aggregated via L11 last-token integration
4796:### 111.8 Next steps (post-§111.5b reverse done)
4798:- ✅ **(β) Reverse direction**: completed §111.5b — null effect at all layers, asymmetry confirms paper-grade mechanism story
4802:- **Position-resolved patching** (followup): patch image-token positions (0..255 in SoM input) instead of last token, verify "image content actually lives at those positions". 这是验证 §111.5b mechanistic story 的下一层 evidence. 30-60 min code + run.
4805:- §111 + Stage 2C reverse direction → commit `9fe3d84` (笔记 + reverse flag, code-data combined)
4806:- 本次 §111.5b update + caveat fix → 单独 commit
4823:A100 80GB VRAM × Qwen3-VL-4B bf16 ~10GB = 8× 余量, 可 cell-parallel 跑 multiple 16-cell rerun slots. Mechanistic Stage 2B scale-up 跟 16-cell rerun 同时跑互不干扰.
4848:- **Stage 2B curated scale-up**: 10-20 task curated mirage cases (paper-grade case-study collection) — exam 期间跑后台
4850:- **Position-resolved patching**: image token positions 0-255 patching grid — 验证 §111.5b "image content distributed in early positions" mechanistic story 的 direct evidence
4855:简明 email 200 字, cite §111 GitHub commit chain, update advisor 两个 finding (L11 + asymmetry) + A100 allocation. 不加新 Q, follow-up doc Q1-Q11 仍 primary ask. 期待 advisor 看到 mechanistic same-day delivery + A100 unblock 后 follow-up Q3 (B1-only vs cross-arch) lean cross-arch + Q1 (3 vs 4 paper) lean 4 paper (mechanistic 独立).
4866:## 113. Mirage task curation — 24 strong + 11 reverse paper-grade candidates (2026-05-06 evening) [finding][infra] #finding #infra
4868:**触发**: §112 A100 等 Steve PCI Devices 期间, §111.7 caveat 4 (N=3 太小, 1/3 task clean signal) 是 paper §5 主要弱点. 写 `scripts/mechanistic/curate_mirage_tasks.py` 自动 score 全 234 cls task 的 source-vs-target divergence, 找 paper-grade mirage cluster.
4883:### 113.2 关键发现 — 24 strong + 11 reverse candidates
4890:| **strong total** (composite ≥ 1.0 ∧ overlap < 0.5) | | **24** | Stage 2B scale-up dataset |
4891:| **reverse** (composite ≤ −1.5) | | 11 | Stage 2C reverse-asymmetry test |
4905:**全部 cls search-result page 上 image-grounded ground-truth absence**: model 看图能 confirm 搜索结果不含目标 → 否定输出. 没图就 hallucinate (mirage) → 肯定输出"找到 / 显示 listings". 这是 paper-grade 完整 mirage signature pattern — §111 task 0 不是 outlier, 是 systematic.
4907:### 113.3 11 reverse candidates — paper §5 robustness check 用
4914:这些 11 个跑 reverse-direction (P-SoM hidden into SoM run) 看是否同 L11-region 出 effect, paper §5 mechanism story 完整闭环.
4918:§111 finding (N=3 task, 1/3 clean signal) → §113 finding (curated **N=24 strong + 7 教科书 cluster**).
4920:**Paper §5 quotable claim 升级版** (post-Stage 2B curated scale-up on A100):
4922:> "Activation patching of source's (SoM with image) hidden state into target's (P-SoM no image) run at layer 11 (Qwen3-VL-4B, 31% depth) recovers source's full output continuation in **24 curated mirage cases** (B1 cls, mean token-overlap-to-source = X.XX ± Y.YY, N=24, p < 0.001 vs envelope-only baseline 0.5). The most compelling 7-task cluster shares an identical signature ('do not show any X' source-side / 'show items/listings related to X' target-side), corresponding to image-grounded ground-truth absence detection systematically failing in P-SoM mode. Patching at any other layer (L0-L10, L12-L35) has null effect, identifying L11 as the causal mirage layer where image-grounded perception emerges before downstream abstraction. Forward-direction asymmetry (verified via 11-task reverse-direction null effect, §111.5b) confirms image content is encoded distributively across image embedding tokens (positions 0-256+) and aggregated via L11 last-token integration."
4932:1. **Stage 2B curated scale-up** (24 task × forward direction × 36 layer) on A100 — replace §111 N=3 placeholder with paper-grade aggregate
4933:2. **Stage 2C reverse curated** (11 reverse task × reverse direction × 36 layer) — confirm asymmetry pattern hold cross-task
4934:3. **Llama-4 cross-arch** (run curate_mirage_tasks.py on Llama-4 → Stage 2B Llama-4) — paper §1 hook upgrade to "golden feature universal"
4935:4. **Position-resolved patching** (1-2 task × 256 image-token positions × L11 only) — direct evidence for §111.5b distributed-encoding mechanistic story
4965:3. **Gap 3 ⭐⭐ Mechanistic Stage 2B 没有 run-level manifest** — archive_subset 只 capture input artifacts, 不 capture L11 / hook config / patching outcome
4984:#### Gap 3 — Stage 2B run_manifest.json emit
4986:- Capture: stage / direction (forward/reverse) / patch_config (source_mode / target_mode / layer indices / max_new_tokens / hook_position) / model.path + n_layers / input_dataset.task_ids / **outcomes_per_task** (per-task best_layer + L11/L17 overlap) / aggregate (best_layer mean / L11 mean / L17 mean) / env_snapshot_ref
5040:| ⏳ Gap 5 A100 capture + compare | A100 setup 完 + Stage 2B scale-up 之前 | Same as above |
5362:- B-20/B-21/B-22/B-15/B-26/B-28/A1-A3/cross-machine drift/Phase A asymmetry/Stage 2B vintage independence
5462:Phase 2 实验 run 过程 (Run)            [during 16-cell + Stage 2B/2C]
5527:- I. Mechanistic-specific (Stage 2B reproducibility per-machine)
5546:- Same input → same L11 cross-machine / hook fire_count / layer indexing / token alignment / Stage 2B post-rerun spot-check
5549:- §5.1 Replication package (14 items: preregistration + paper drafts + manifests + provenance + bug catalog + limitations prose + Stage 2B results + license + README)
5604:## 117. Stage 2B/2C Myriad scale-up — bidirectional mid-layer disruption + §111.5b "reverse null" overturned (2026-05-09) [finding][infra] #finding #infra
5608:DGX→quark Tailscale→Myriad SSH chain newly opened (笔记 §117.6 infra). Stage 2B forward (24 strong-tier tasks) + Stage 2C reverse (15 reverse-tier tasks) run on Myriad A100 80GB compute node `node-v00a-001` / `node-u00a-001`. ~30min wallclock each (V100 estimate had been 24h — A100 80GB ~48× speedup). Output dirs:
5610:- `results/mechanistic/stage2c_reverse_curated_b1_cls_myriad/`
5614:### 117.2 Headline finding — bidirectional disruption (overturns §111.5b)
5616:§111.5b reported "reverse null at all layers" based on **N=1 single task** (task 0 reverse direction): overlap→source = 0.51 (baseline) at all 36 layers, no patch effect. Concluded "asymmetric encoding".
5618:**Scale-up to N=15 reverse tasks reveals different picture**:
5630:Reverse direction shows **mid-layer L11-L17 disruption magnitude similar to forward** (Δoverlap→tgt: forward -0.143 ± 0.217 / reverse -0.193 ± 0.256 at L17). Welch's t @ L17 forward vs reverse p=0.535 (not distinguishable — consistent with bidirectional mechanism, but failure to reject ≠ proof of equality).
5643:**L17 robust mid-layer mechanism layer** confirmed across 2 directions × 2 metrics (3 of 4 Holm-significant). L11 emerges as secondary in reverse but marginal in forward.
5645:### 117.4 §111 task-0 finding re-interpretation
5647:§111 reported "L11 flips 93% match in forward task 0" — that's **task-specific outlier**, not population pattern. 24-task forward average shows L11 overlap→tgt=0.908 (Δ=-0.093 from baseline), much weaker than task-0's near-flip. Distribution has long tail (some tasks fully flip, some don't disrupt). §111.5 "Stage 2A logit_shift L17 peak" is **consistent with N=24 finding** — both Stage 2A (5-task aggregate) and Stage 2B (24-task aggregate) point to L17 as primary.
5649:Marked: §111 task-0 single-case study should NOT be cited as paper §5 representative finding. Use 24-task L17 aggregate + ±1σ band + Holm p-value instead.
5653:Forward 24 tasks ≠ reverse 15 tasks. They're curated by direction-specific composite score (`scripts/mechanistic/curate_mirage_tasks.py`):
5656:- Reverse-tier: composite ≤ -1.5 (reverse-easy mirage)
5661:- Cell A (existing): forward × strong (24) — done
5662:- Cell B (existing): reverse × reverse (15) — done
5663:- Cell C (new): forward × reverse-tier (15) — qsub 335339, qw
5664:- Cell D (new): reverse × strong-tier (24) — qsub 335340, qw
5668:- If C+D null/weak → mechanism is task-class-specific, paper §5 must caveat or scope to curated subsets
5677:2. **Task heterogeneity**: std band ±0.22 ≈ 1.5× effect size at L17. Some tasks no disrupt, some near-flip. Need median+IQR alternative reporting OR per-task scatter alongside aggregate.
5678:3. **Bidirectional vs asymmetric framing**: only commit after 2x2 cells C/D resolve. If bidirectional confirms, lit anchor is Q5 Gemini DR "bidirectional modality fusion". If asymmetric returns (cells C/D null), revert to §111.5b style "image-grounded encoding asymmetry" framing.
5679:4. **L11 vs L17 hierarchy**: Stage 2A peaked L17, Stage 2B forward L17 strongest, Stage 2B reverse both L11+L17. Possible "L11 = mirage emergence onset, L17 = mirage commitment / decision lock-in" two-stage interpretation. Speculative — paper section needs careful wording.
5686:- [ ] If 2x2 shows selection bias → caveat language in paper §5 + 笔记 §117 update
5687:- [ ] Regenerate forward + reverse figures with corrected `(forward)` / `(reverse)` title labels (commit `c8e76bd` script fix already in repo, just needs re-plot)
5688:- [ ] Update `project_paper_hook.md` memory: replace "§111.5b asymmetric encoding" with "117.x bidirectional mid-layer mechanism (provisional, 2x2 pending)"

exec
/bin/bash -lc "sed -n '5604,5725p' docs/checkpoints/实验笔记.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
## 117. Stage 2B/2C Myriad scale-up — bidirectional mid-layer disruption + §111.5b "reverse null" overturned (2026-05-09) [finding][infra] #finding #infra

### 117.1 Setup

DGX→quark Tailscale→Myriad SSH chain newly opened (笔记 §117.6 infra). Stage 2B forward (24 strong-tier tasks) + Stage 2C reverse (15 reverse-tier tasks) run on Myriad A100 80GB compute node `node-v00a-001` / `node-u00a-001`. ~30min wallclock each (V100 estimate had been 24h — A100 80GB ~48× speedup). Output dirs:
- `results/mechanistic/stage2b_curated_b1_cls_myriad/`
- `results/mechanistic/stage2c_reverse_curated_b1_cls_myriad/`

Both rsync'd back to DGX via tar-pipe through SSH chain.

### 117.2 Headline finding — bidirectional disruption (overturns §111.5b)

§111.5b reported "reverse null at all layers" based on **N=1 single task** (task 0 reverse direction): overlap→source = 0.51 (baseline) at all 36 layers, no patch effect. Concluded "asymmetric encoding".

**Scale-up to N=15 reverse tasks reveals different picture**:

| Layer | Forward overlap→tgt (N=24) | Reverse overlap→tgt (N=15) | Forward LD→tgt | Reverse LD→tgt |
|---|---|---|---|---|
| L0  | 0.98 | 0.88 | 0.9 | 2.4 |
| L5  | 0.97 | 0.89 | 1.3 | 2.6 |
| **L11** | 0.91 | **0.82** | 3.9 | **6.7** |
| **L17** | **0.86** | **0.81** | **6.0** | **6.9** |
| L23 | 0.93 | 0.87 | 2.2 | 2.8 |
| L29 | 0.95 | 0.93 | 2.5 | 1.1 |
| L35 | 1.00 | 1.00 | 0.0 | 0.0 |

Reverse direction shows **mid-layer L11-L17 disruption magnitude similar to forward** (Δoverlap→tgt: forward -0.143 ± 0.217 / reverse -0.193 ± 0.256 at L17). Welch's t @ L17 forward vs reverse p=0.535 (not distinguishable — consistent with bidirectional mechanism, but failure to reject ≠ proof of equality).

### 117.3 Statistical significance (paired t-test + Holm-Bonferroni, `scripts/analysis/stage2_layer_significance.py`)

Per-task paired test layer L_n vs L35 baseline. One-sided H1: disruption (overlap < baseline OR LD > baseline). Holm correction across 6 tested layers.

| Direction × Metric | Holm-significant layers | Notes |
|---|---|---|
| Forward × overlap→tgt | **L17** (p_Holm=0.011 \*\*) | L11 marginal (p_Holm=0.092) |
| Forward × LD→tgt | **L17** (p_Holm=0.024 \*) | L11 marginal (p_Holm=0.080) |
| Reverse × overlap→tgt | **L11 + L17** (p_Holm=0.044 \* / 0.033 \*) | strongest signal |
| Reverse × LD→tgt | None Holm-significant | L17 raw p=0.014 marginal Holm 0.084 (N=15 power-limited) |

**L17 robust mid-layer mechanism layer** confirmed across 2 directions × 2 metrics (3 of 4 Holm-significant). L11 emerges as secondary in reverse but marginal in forward.

### 117.4 §111 task-0 finding re-interpretation

§111 reported "L11 flips 93% match in forward task 0" — that's **task-specific outlier**, not population pattern. 24-task forward average shows L11 overlap→tgt=0.908 (Δ=-0.093 from baseline), much weaker than task-0's near-flip. Distribution has long tail (some tasks fully flip, some don't disrupt). §111.5 "Stage 2A logit_shift L17 peak" is **consistent with N=24 finding** — both Stage 2A (5-task aggregate) and Stage 2B (24-task aggregate) point to L17 as primary.

Marked: §111 task-0 single-case study should NOT be cited as paper §5 representative finding. Use 24-task L17 aggregate + ±1σ band + Holm p-value instead.

### 117.5 Selection-bias caveat — 2x2 control launched

Forward 24 tasks ≠ reverse 15 tasks. They're curated by direction-specific composite score (`scripts/mechanistic/curate_mirage_tasks.py`):
- `composite = mirage_score × (1 + divergence)` where `mirage_score = (src_neg - tgt_neg) + (tgt_aff - src_aff)`
- Strong-tier: composite ≥ 1.0 (forward-easy mirage)
- Reverse-tier: composite ≤ -1.5 (reverse-easy mirage)

Composite is **task-text-only** (no patching leak), but plausibly correlates with patching effect size — strong-tier may be high-effect-size subset. Bidirectional finding could reflect mechanism universality OR task curation correlation.

**2x2 control launched 2026-05-09 morning** (commit `9d67387`):
- Cell A (existing): forward × strong (24) — done
- Cell B (existing): reverse × reverse (15) — done
- Cell C (new): forward × reverse-tier (15) — qsub 335339, qw
- Cell D (new): reverse × strong-tier (24) — qsub 335340, qw

Decision tree post-2x2:
- If C+D show similar L17 disruption → mechanism universal (paper §5 confident bidirectional claim)
- If C+D null/weak → mechanism is task-class-specific, paper §5 must caveat or scope to curated subsets

### 117.6 Infrastructure — DGX-driven Myriad workflow established

DGX→quark Tailscale→quark Cisco AnyConnect→Myriad SSH chain verified working (B-81 umbrella + new 7th class B-81g UTF-8 ASCII codec fix). Watcher cron `scripts/maintenance/glm/myriad_watcher.py` installed (5min ntfy on state change). DGX never directly reaches UCL gateway (IP-restricted) — quark Windows is mandatory bastion. Documented in `docs/reference/COMPUTE_INFRASTRUCTURE.md` + `memory/reference_dgx_to_ucl_chain.md`.

### 117.7 Open questions for paper §5 prose update

1. **Effect size**: L17 forward overlap drop -0.143 ± 0.217 (~14% match degradation). Reviewer: "is 14% drop strong evidence?" Need to contrast with random-injection control (zero effect baseline).
2. **Task heterogeneity**: std band ±0.22 ≈ 1.5× effect size at L17. Some tasks no disrupt, some near-flip. Need median+IQR alternative reporting OR per-task scatter alongside aggregate.
3. **Bidirectional vs asymmetric framing**: only commit after 2x2 cells C/D resolve. If bidirectional confirms, lit anchor is Q5 Gemini DR "bidirectional modality fusion". If asymmetric returns (cells C/D null), revert to §111.5b style "image-grounded encoding asymmetry" framing.
4. **L11 vs L17 hierarchy**: Stage 2A peaked L17, Stage 2B forward L17 strongest, Stage 2B reverse both L11+L17. Possible "L11 = mirage emergence onset, L17 = mirage commitment / decision lock-in" two-stage interpretation. Speculative — paper section needs careful wording.

### 117.8 Pending follow-up actions

- [ ] Wait for cells C/D (~30 min A100 each, qw queued)
- [ ] Once cells C/D done: re-run `scripts/analysis/stage2_layer_significance.py` with all 4 cells
- [ ] If 2x2 confirms universal → update `paper_planning §5` + `paper_drafts/section5*.md` to bidirectional framing
- [ ] If 2x2 shows selection bias → caveat language in paper §5 + 笔记 §117 update
- [ ] Regenerate forward + reverse figures with corrected `(forward)` / `(reverse)` title labels (commit `c8e76bd` script fix already in repo, just needs re-plot)
- [ ] Update `project_paper_hook.md` memory: replace "§111.5b asymmetric encoding" with "117.x bidirectional mid-layer mechanism (provisional, 2x2 pending)"


exec
/bin/bash -lc "sed -n '5725,5825p' docs/checkpoints/实验笔记.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
exec
/bin/bash -lc "rg -n \"B6|missing|complete-case|crash|excluded|multiple-imputation|≤5|5%|N=15|0\\.65|power|Cohen|random-effects|N≥10|N>=10|Holm|six|6 layers\" docs/checkpoints/pre_run/preregistration.md docs/analysis/cross_sites/power_analysis.md docs/checkpoints/pre_run/topvenue_constraints.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
docs/analysis/cross_sites/power_analysis.md:5:This appendix updates the pre-registered power analysis with **observed adjusted-SR levels** from `sr_fp_per_mode.md` (Phase 1 B0 + B1 done cells, pre-paper-grade rerun). The post-rerun version will replace this file once 16-cell aggregation completes.
docs/analysis/cross_sites/power_analysis.md:12:| reddit | 6.7-13.8% | 2.4-7.6% | ~9.5% |
docs/analysis/cross_sites/power_analysis.md:26:Run: `python3 scripts/analysis/power_analysis.py --baseline-sr {0.10,0.15,0.20}`
docs/analysis/cross_sites/power_analysis.md:34:**Key observation**: minimum detectable effect at 80% per-cell power is **5-7pp** for cls/red, **4-5pp** for shop. The **observed mechanism effect (1-5pp)** is at or below per-cell MDE in 2 of 3 sites — **per-cell power for typical phantom effects is < 50%**.
docs/analysis/cross_sites/power_analysis.md:36:## 3. Family-wise power at observed effects (K-of-N rule, baseline SR=0.15 proxy)
docs/analysis/cross_sites/power_analysis.md:38:| Per-cell power (proxy effect on smallest site) | K_h1=12/16 family power | K_h3=11/16 family power |
docs/analysis/cross_sites/power_analysis.md:48:- **K_h1=12/16** is calibrated for **≥7pp effects** with paper-grade ≥0.80 family power. For typical phantom mechanism effects (1-5pp), K_h1 family power is **<10%**.
docs/analysis/cross_sites/power_analysis.md:49:- **K_h3=11/16** is slightly more permissive but still requires per-cell power ≥0.65 (≈6pp effect at SR=0.15) to reach 0.49 family power.
docs/analysis/cross_sites/power_analysis.md:55:- **Primary effect-detection test** = DerSimonian-Laird random-effects meta-analysis (locked by B8) on cells with N≥10. This is power-adequate at the cross-cell level for effects ≥2pp.
docs/analysis/cross_sites/power_analysis.md:58:- This recharacterization is **not post-hoc cherry-picking**: the random-effects meta + TOST were always the primary tests in `preregistration.md §4`. The K-of-N rule is restated as transparency.
docs/analysis/cross_sites/power_analysis.md:62:"At observed adjusted-SR levels (8-15% across sites) and observed mechanism effect sizes (1-5pp), per-cell statistical power is below 0.55 in two of three sites. We therefore rely on (a) DerSimonian-Laird random-effects meta-analysis across all cells (B8 lock; cross-cell pooling raises effective power) and (b) TOST equivalence on the full N=910 pooled task set (δ=1.0pp margin) as primary tests. The K-of-N family-wise rule pre-registered for transparency is not powered for sub-5pp effects, and we report its outcome as a secondary observation rather than a gate on the main hypotheses."
docs/analysis/cross_sites/power_analysis.md:66:The original `power_analysis.py` (pre-2026-05-09) contained a stale interpretation block claiming "K_h1=12/16 with per-cell power=0.65 → >0.95 family-wise power" — this was numerically inconsistent with the K-of-N table on the same page (actual value 0.289). Fixed in commit (current session) to compute family power **at observed effect sizes** rather than at hypothetical per-cell power levels. The corrected version is what this appendix relies on.
docs/analysis/cross_sites/power_analysis.md:69:- `scripts/analysis/power_analysis.py` (script, fixed)
docs/analysis/cross_sites/power_analysis.md:70:- `docs/analysis/cross_sites/power_analysis_observed_sr10.md` / `sr15.md` / `sr20.md` (full per-baseline tables)
docs/analysis/cross_sites/power_analysis.md:72:- `docs/checkpoints/pre_run/preregistration.md §4` (B8 random-effects lock + TOST policy)
docs/analysis/cross_sites/power_analysis.md:78:- [ ] Add `power_analysis_post_rerun.md` once observed effect sizes are final (replace this file)
docs/checkpoints/pre_run/topvenue_constraints.md:9:- ✓ Done: 50 constraints (post-2026-05-09: G6 + G7 cell E + 2x2; B2 bootstrap policy lock; A14 + F3 reproducibility / external-validity scope; H2 negative-results registry; F7 + H3 ethics/license/COI/misuse-safety; G5 post-hoc layer disclosure; D8 + H5 section1 intro rewrite; B6 + B7 + B8 pre-spec policy lock; A5 + F8 version pinning; A11 + A12 + A13 release cards; G9 token alignment; G8 sparse-mechanism heterogeneity; C10 pre-launch + validate-strict Make targets; G1 §5.1 method box codex-delegated)
docs/checkpoints/pre_run/topvenue_constraints.md:26:| ~~9~~ | ~~G5 post-hoc layer selection~~ | ✓ disclosed 2026-05-09 (preregistration §5.X — 4-path convergence + Holm) | — |
docs/checkpoints/pre_run/topvenue_constraints.md:33:| A1 | Preregister primary hypotheses, decision rules, and analysis families before post-rerun data are inspected | NEEDS_BIB_ENTRY: Pineau et al. 2021; NeurIPS checklist Q4/Q6 | ⚠️ | `docs/checkpoints/pre_run/preregistration.md` has H1-H8, Holm families, R1-R5, but frontmatter is `status: draft`, `registered_at`, `registered_git_sha`, `witnessed_by`, and OSF DOI are pending. Remediation: lock after advisor email, tag git, deposit OSF; cost 2-4h. | "The hypotheses and decision rules were written before the 16-cell rerun; the camera-ready will cite the lock SHA/OSF DOI once advisor witness is received." |
docs/checkpoints/pre_run/topvenue_constraints.md:34:| A2 | Release code, data, and exact reproduction instructions for main experimental claims | NeurIPS checklist Q4-Q5; NEEDS_BIB_ENTRY: Pineau et al. 2021 | ⚠️ | Code/scripts exist and `osf_lock_manifest.md` lists release artifacts, but `pre_rerun_audit.md §5.1.14` marks replication README as TBD and `§5.3.4` data-sharing policy as TBD. Remediation: write `README_REPLICATION.md` with clone→run→aggregate→figure commands and redaction policy; cost 1 day. | "We release the run manifest, environment snapshots, analysis scripts, and locked code; raw credentials/auth files are excluded and documented." |
docs/checkpoints/pre_run/topvenue_constraints.md:39:| A7 | Keep an immutable run manifest mapping conditions to artifacts | NeurIPS checklist Q4-Q5; NEEDS_BIB_ENTRY: Pineau et al. 2021 | ✓ | `results/phantom_paper/run_manifest.yaml` is the cell registry; `osf_lock_manifest.md §2.1` lists run manifest SHA as a locked artifact. | "The run manifest is the source of truth for which cells are archived, paper-grade, or excluded." |
docs/checkpoints/pre_run/topvenue_constraints.md:42:| A10 | Include compute resources, total compute, and cost/carbon accounting | NeurIPS checklist Q8; `qiu2025modserve` | ⚠️ | `pre_rerun_audit.md §3.4` has per-cell GPU-hours, USD, and carbon tracking; cross-platform GPU power profile and Section 8 prose are TBD. Remediation: aggregate after rerun and add total compute table; cost 0.5 day. | "We report per-cell runtime, GPU type, API cost, and carbon estimates; cross-machine power variation is a limitation." |
docs/checkpoints/pre_run/topvenue_constraints.md:48:## Category B — Statistical Methodology (Cohen / Holm / CONSORT-AI)
docs/checkpoints/pre_run/topvenue_constraints.md:52:| B1 | Control multiplicity for primary and structural families | NEEDS_BIB_ENTRY: Holm 1979; NEEDS_BIB_ENTRY: CONSORT-AI/Liu et al. 2020 | ✓ | `preregistration.md §3` declares PRIMARY, STRUCTURAL, ROUTER, EXPLORATORY, and POST-HOC families with Holm-Bonferroni; `aggregate_phantom_meta.py` implements Holm for meta pooled results. | "Main claims use predeclared Holm-corrected families; exploratory tables are marked as such." |
docs/checkpoints/pre_run/topvenue_constraints.md:53:| B2 | Define confidence interval method, resampling unit, and assumptions | NeurIPS checklist Q7; NEEDS_BIB_ENTRY: Efron/Tibshirani bootstrap | ✓ | Locked 2026-05-09 in `preregistration.md §4`: 1000-resample percentile bootstrap (BCa as sensitivity check), task-level paired pairing across modes, single-level (task_id) clustering — no nested (cell × site) cluster bootstrap because meta-analysis at cell level uses random-effects + I²/τ² separately. | "Percentile + task-paired bootstrap is reported primary; BCa shown as appendix sensitivity. Cell-level uncertainty enters via random-effects meta-analysis, not nested bootstrap." |
docs/checkpoints/pre_run/topvenue_constraints.md:54:| B3 | Report effect sizes, not just p-values | NEEDS_BIB_ENTRY: Cohen 1988; NeurIPS checklist Q7 | ⚠️ | `preregistration.md §4` locks Cohen's h/d, but `pre_rerun_audit.md §4.1.4` says Cohen's h table is partial. Remediation: add h/d/CI columns to paper tables; cost 2-3h. | "We report pp lift plus standardized effect sizes with uncertainty for binary and continuous outcomes." |
docs/checkpoints/pre_run/topvenue_constraints.md:55:| B4 | Use paired tests for same-task mode comparisons | `koh2024visualwebarena`; `zhou2024webarena`; NEEDS_BIB_ENTRY: McNemar 1947 | ✓ | `paper_drafts/section3_definition.md §3.5` and `preregistration.md §4` require same-task subsets; `axis1_microbehavior.py` and `aggregate_phantom_lift.py` include paired/McNemar/bootstrap infrastructure. | "All mode comparisons are task-paired; missing cells are excluded from the common universe rather than imputed." |
docs/checkpoints/pre_run/topvenue_constraints.md:56:| B5 | Quantify annotation/label reliability for human-coded FP and failure modes | NEEDS_BIB_ENTRY: Cohen 1960 kappa; CONSORT-AI/Liu et al. 2020 | ❌ | `pre_rerun_audit.md §1.5` and `§4.3` mark FP labeling and failure-mode κ pilots as TBD. Remediation: two-rater 30-task pilot, κ≥0.7 target, adjudication log; cost 1 day. | "We will report κ for manually audited FP/failure-mode labels; until then qualitative mechanism claims remain caveated." |
docs/checkpoints/pre_run/topvenue_constraints.md:57:| B6 | Predefine missing-data/crashed-episode handling | CONSORT-AI/Liu et al. 2020; NeurIPS checklist Q7 | ✓ | Locked 2026-05-09 in `preregistration.md §4`: crashed episodes (uncaught exception / OOM / >30min timeout / browser crash) excluded from paired-N denominators, NOT imputed; not_logged_in / auth_drift after 3 retries excluded; missing artifacts excluded from per-step (not full-task); per-cell exclusion count + reason histogram in Appendix C. | "Listwise deletion only — crashes excluded from paired N, no success/failure imputation, exclusion histograms reported per cell." |
docs/checkpoints/pre_run/topvenue_constraints.md:59:| B8 | Analyze heterogeneity and avoid overclaiming pooled averages | NEEDS_BIB_ENTRY: DerSimonian & Laird 1986; NeurIPS checklist Q7 | ✓ | Locked 2026-05-09 in `preregistration.md §4`: random-effects DerSimonian-Laird primary, Cochran Q + I² + τ² reported, Higgins & Thompson 2002 I² thresholds (< 25% pooled / 25-50% both / 50-75% per-cell primary / > 75% no pool + heterogeneity-source decomposition by site/model/task-pool). Per-cell estimates always shown alongside pooled, so heterogeneity never averaged away. | "Random-effects with pre-specified I² thresholds — per-cell + heterogeneity-source decomposition reported when between-cell variance is high." |
docs/checkpoints/pre_run/topvenue_constraints.md:60:| B9 | Run power/MDE analysis for effect detectability | NeurIPS checklist Q7; CONSORT-AI/Liu et al. 2020 | ✓ | `docs/analysis/cross_sites/power_analysis.md` (consolidated 2026-05-09) reports observed-SR-conditioned MDE 4-7pp at 80% per-cell power across 3 sites + family-wise K-of-N power table at observed effect sizes (1-5pp) **<10%** → triggered methodological reframing: K-of-N rule retained as secondary transparency check, primary detection via B8 random-effects meta + TOST on N=910 pooled tasks. Bug fix: `power_analysis.py` stale interpretation block (claimed >0.95 K=12/16 family power at p_per=0.65; actual 0.289) corrected in same commit. Preregistration §4 updated with B9 row locking the corrected framing. | "Per-cell MDE 4-7pp at observed SR (8-15%); K-of-N rule documented as secondary transparency, primary detection via random-effects meta + pooled TOST." |
docs/checkpoints/pre_run/topvenue_constraints.md:72:| C7 | Preserve mode operational definitions and prevent post-hoc episode reclassification | NeurIPS checklist Q1; `yang2023som`; `koh2024visualwebarena` | ✓ | `preregistration.md §4` stipulates DOM/SoM/Vision/P-text/P-prompt/P-SoM definitions and no post-hoc reclassification; `som.py::prepare_observation_for_mode` implements these modes. | "Episodes are excluded only by locked filters or corruption flags; mode labels are never reinterpreted after seeing outcomes." |
docs/checkpoints/pre_run/topvenue_constraints.md:99:| E2 | Include latency and serving-stage interpretation for multimodal modes | `qiu2025modserve`; `koh2024visualwebarena` | ⚠️ | Planning docs cite latency advantage; `pre_rerun_audit.md §3.4` tracks cost but cross-platform power/latency profiles are incomplete. Remediation: aggregate p50/p95 latency by mode and disclose hardware; cost 2-4h. | "Skipping image encoding removes the dominant multimodal serving stage; p50/p95 latency will be reported per cell." |
docs/checkpoints/pre_run/topvenue_constraints.md:125:| G1 | Define clean/corrupt/source/target prompts and behavioral metric for activation patching | NEEDS_BIB_ENTRY: Wang et al. 2023; NEEDS_BIB_ENTRY: Zhang et al. 2024 | ✓ | `paper_drafts/section5_mechanism_reddit.md §5.1 Method` (added 2026-05-09 via codex `section5_method_box.md` prompt): 7-paragraph method box covering source (`som`)/target (`phantom_som`) prompt definitions / 36-layer L0-L35 hidden state cache / forward+reverse direction / 4 metrics (token_overlap_to_source/target + LD_to_source/target) / Holm-Bonferroni paired t / 1000-sample task-paired percentile bootstrap / Wilcoxon backup / composite score curation / 5-cell A/B/C/D/E design / random-injection specifics (`randn_like(h) * h.std() + h.mean()`) / pre-vs-post-hoc layer disclosure cite §5.X. NEEDS_BIB_ENTRY flagged for `wang2023interpretability` + `zhang2024patching`. | "Section 5.1 defines the patching protocol explicitly: source = som with image, target = phantom_som without; per-layer hidden state cache + greedy continuation 50 tokens + token overlap / Levenshtein metrics; Holm-corrected paired test + bootstrap CI; cells A-E control selection-bias and content-specificity." |
docs/checkpoints/pre_run/topvenue_constraints.md:127:| G3 | Include forward and reverse patching directions | NEEDS_BIB_ENTRY: Zhang et al. 2024; NEEDS_BIB_ENTRY: Heimersheim & Janiak 2024 | ✓ | `实验笔记.md §117.2` reports N=24 forward and N=15 reverse; reverse null from §111.5b is overturned by scaled N=15 analysis. | "We test both directions and revise the narrative when the larger reverse sample contradicts the pilot." |
docs/checkpoints/pre_run/topvenue_constraints.md:128:| G4 | Correct for multiple layer tests | NEEDS_BIB_ENTRY: Zhang et al. 2024; NEEDS_BIB_ENTRY: Holm 1979 | ✓ | `scripts/analysis/stage2_layer_significance.py` uses paired t-tests vs L35 and Holm-Bonferroni across six layers per direction; `实验笔记.md §117.3` reports Holm-significant layers. | "Layer claims are based on Holm-corrected per-task paired tests, not visual inspection alone." |
docs/checkpoints/pre_run/topvenue_constraints.md:129:| G5 | Avoid post-hoc layer selection by using independent confirmation or disclosure | NEEDS_BIB_ENTRY: Zhang et al. 2024; NEEDS_BIB_ENTRY: Wang et al. 2023 | ✓ | `preregistration.md §5.X` (added 2026-05-09) explicitly maps L11/L17 selection chain: Stage 2A logit_shift (hypothesis-generating) → §111 task-0 (single-task outlier, acknowledged) → Stage 2B 24-task / 2C 15-task / Cell D 24-task (3 confirmatory cells, all Holm-significant L17 + cell D L11). Holm-Bonferroni across 6 canonical layers handles multiple-comparison; convergence of 4 independent analysis paths (logit_shift, forward overlap, reverse overlap, cross-tier) constitutes confirmation. Cell E random-injection rules out generic-injection alternative. | "Layers L11/L17 are not pre-registered but emerged from Stage 2A pilot and are confirmed by 3 independent Stage 2B/2C/Cell-D tests with Holm correction — see preregistration §5.X." |
docs/checkpoints/pre_run/topvenue_constraints.md:131:| G7 | Control for task-selection bias with crossed task-tier × direction design | NEEDS_BIB_ENTRY: Zhang et al. 2024; NEEDS_BIB_ENTRY: Conmy et al. 2023 | ✓ | 4-cell 2x2 design completed 2026-05-09: A (fwd×strong N=24, Holm L17 p=0.011 \*\*) + B (rev×reverse N=15, Holm L11+L17) + C (fwd×reverse N=15, Δ=-0.10 weaker) + D (rev×strong N=24, Holm L11+L17 p=0.006/0.008 \*\*). Cross-cell Welch p=0.509-1.000 (NS) — magnitudes statistically indistinguishable across direction × tier; reverse direction Δ=-0.193 identical on both tiers (p=1.000), forward Δ=-0.143/-0.103 not Welch-distinguishable across tiers (p=0.577). Selection bias rejected; mechanism claim universal pending cross-site (cells F/G). | "The crossed 2x2 design rules out selection-bias artifact: reverse direction produces identical magnitude on both task tiers (Welch p=1.000), forward direction is Welch-indistinguishable across tiers (p=0.577); only Holm-significance differs due to N power asymmetry." |
docs/checkpoints/pre_run/topvenue_constraints.md:166:- Constraints not in our paper.bib that need entries: Pineau et al. 2021; NeurIPS 2024 Paper Checklist; Gebru et al. 2018; Mitchell et al. 2019; Breck et al. 2017; Sculley et al. 2014; Lipton & Steinhardt 2018; Liu et al. 2020 CONSORT-AI; Cohen 1960/1988; Holm 1979; McNemar 1947; DerSimonian & Laird 1986; Cook & Campbell 1979; Nosek et al. 2015; Guo et al. 2017 calibration; Wang et al. 2023 IOI; Zhang et al. 2024 activation patching; Heimersheim & Janiak activation patching best practices; Conmy et al. 2023 ACDC; Geva et al. 2021 key-value memories; GAIA; OSWorld.
docs/checkpoints/pre_run/topvenue_constraints.md:168:- Conservative status rule used here: "script exists but not wired/run" = ⚠️ partial; "pre-specified but not locked" = ⚠️ partial; "planned but no artifact/result" = ❌ missing.
docs/checkpoints/pre_run/preregistration.md:38:5. **Multiple-comparison family discipline**: gating tests (PRIMARY + STRUCTURAL) have explicit Holm-corrected family m count. Exploratory tests (EXPLORATORY family + post-hoc) are reported with adjusted p-values for transparency but NOT used to gate paper claims.
docs/checkpoints/pre_run/preregistration.md:52:- **H1(i)** Pooled DerSimonian-Laird random-effect meta-analysis reaches significance at Holm α=0.05 (PRIMARY family m=1 test, no within-family correction needed).
docs/checkpoints/pre_run/preregistration.md:53:- **H1(ii)** ≥ K_h1 of N_cells individually Holm-significant at α=0.05 within the per-cell P-SoM sub-family (m = N_cells), where **K_h1 = 0.75** (commit-locked, see §4).
docs/checkpoints/pre_run/preregistration.md:69:- **H3(i)** axis 1: |P-text ∖ P-SoM| unique-count > 0 with bootstrap 95% CI excluding 0 in ≥ K_h3 of N_cells.
docs/checkpoints/pre_run/preregistration.md:70:- **H3(ii)** axis 2: |P-prompt ∖ P-SoM| unique-count > 0 with bootstrap 95% CI excluding 0 in ≥ K_h3 of N_cells.
docs/checkpoints/pre_run/preregistration.md:78:- Multiple-comparison: Holm-Bonferroni step-down per axis sub-family (axis 1: m = N_cells; axis 2: m = N_cells).
docs/checkpoints/pre_run/preregistration.md:84:Reported per cell + meta-pooled (DerSimonian-Laird) for transparency. Holm-Bonferroni and BH FDR q-values reported. No pre-registered ranking commitment.
docs/checkpoints/pre_run/preregistration.md:108:- **H7(i)** Pooled DerSimonian-Laird random-effect meta-analysis on lift reaches Holm α=0.05 (PRIMARY family m=1 if paper-1 / SECONDARY informational if paper-2).
docs/checkpoints/pre_run/preregistration.md:109:- **H7(ii)** ≥ K_h1 of N_cells individually Holm-significant on per-cell lift, bootstrap 95% CI lower-bound > 0.
docs/checkpoints/pre_run/preregistration.md:116:- Multiple-comparison: Holm-Bonferroni step-down within H7 sub-family m=N_cells.
docs/checkpoints/pre_run/preregistration.md:124:- **H8(i)** Tier 2 router lift over Tier 1 oracle baseline ≥ 0 with bootstrap 95% CI excluding −1.0pp (paper claims Tier 2 ≈ Tier 1 within deployment-grade tolerance, given Tier 2 is leak-free and deployment-realistic).
docs/checkpoints/pre_run/preregistration.md:125:- **H8(ii)** Tier 2 router lift over best-single-mode-baseline ≥ 1.0pp, ≥ K_h1 cells Holm-significant.
docs/checkpoints/pre_run/preregistration.md:135:| Rule | Conditions | Paper hook framing | Hook power |
docs/checkpoints/pre_run/preregistration.md:141:| **R5** | H1 fails (pooled meta sig fails Holm OR < K_h1 cells individually sig) | Paper death scenario: pivot to VWA bug audit paper (§107 4-cluster fix as primary) OR abandon. Decision deferred to advisor sync at fail time. | n/a |
docs/checkpoints/pre_run/preregistration.md:149:- H1(ii) per-cell P-SoM Holm: m = N_cells.
docs/checkpoints/pre_run/preregistration.md:151:- Method: Holm-Bonferroni step-down per H-sub-family (Holm 1979).
docs/checkpoints/pre_run/preregistration.md:156:- Method: Holm-Bonferroni step-down per axis sub-family.
docs/checkpoints/pre_run/preregistration.md:161:- H7(ii) per-cell Tier 1 lift Holm: m = N_cells.
docs/checkpoints/pre_run/preregistration.md:165:- Method: Holm-Bonferroni step-down per H-sub-family.
docs/checkpoints/pre_run/preregistration.md:170:- Best-signal-per-mode characterization (Register III AA, Section 6 portfolio finding): per (mode, signal) AUROC reported, Holm-corrected within mode for transparency.
docs/checkpoints/pre_run/preregistration.md:171:- Method: Holm-corrected and BH q-value reported for transparency.
docs/checkpoints/pre_run/preregistration.md:186:| **CI method** | 1000-resample task-level paired bootstrap, **percentile** intervals (BCa as sensitivity check, not primary) | Existing infra in `aggregate_phantom_lift.py`. Percentile chosen primary because: (a) paired-bootstrap on bounded proportion (SR ∈ [0,1]) → BCa acceleration estimate is unstable at small N per cell; (b) Cohen's h transformation already symmetrizes; (c) percentile is the canonical reporting in WebArena/VWA precedent. BCa shown as appendix sensitivity check. |
docs/checkpoints/pre_run/preregistration.md:188:| **Bootstrap clustering** | **Single-level (task_id)** for primary, no nested cluster (cell × site) bootstrap | Justification: meta-analysis at cell level is separate (`aggregate_phantom_meta.py` random-effects + I²/τ²); within-cell bootstrap only re-samples tasks. Multi-level cluster would double-count uncertainty already captured by random-effects meta. Lock: percentile + task-id unit + no nested cluster (B2 lock 2026-05-09). |
docs/checkpoints/pre_run/preregistration.md:189:| **Sig threshold** | Holm α=0.05 within respective family | FWER control |
docs/checkpoints/pre_run/preregistration.md:190:| **Effect size (binary)** | Cohen's h with bootstrap CI | Standard for proportion comparisons |
docs/checkpoints/pre_run/preregistration.md:191:| **Effect size (continuous)** | Cohen's d with bootstrap CI | For cost/latency H2(a)(b) |
docs/checkpoints/pre_run/preregistration.md:193:| **H1 K_h1 cell-pass threshold** | **0.75** | Allows ~25% capability-outlier cells (e.g., B1 shopping power-limited); not so strict that single-cell noise breaks claim |
docs/checkpoints/pre_run/preregistration.md:198:| **N inclusion floor** | ≥ 100 ep per (cell × mode) | Statistical power baseline |
docs/checkpoints/pre_run/preregistration.md:202:| **Mode operational definitions** | 6 modes per paper §3 (text format × prompt × image): DOM (AXTree+DOM-prompt+no image) / SoM ([SOM_MARKS]+SoM-prompt+image) / Vision (no text+image) / P-text ([SOM_MARKS]+DOM-prompt+no image) / P-prompt (AXTree+SoM-prompt+no image) / P-SoM ([SOM_MARKS]+SoM-prompt+no image) | Stipulative — **no post-hoc episode reclassification**. Episodes systematically excluded per (FP filter / N-floor / data-corruption flag), never redefined which mode they belong to. Edge cases (empty AXTree / 0 marks / OCR-empty) follow `condition_meta.json` declared mode |
docs/checkpoints/pre_run/preregistration.md:205:| **Failure-mode classification rubric** | 5-bucket: `early_finish` / `wrong_commit` / `visual_hijack` / `click_loop` / `persistent_error` per `docs/analysis/disagreement_clusters.md` decision tree | Pre-data inter-annotator agreement target Cohen κ ≥ 0.7 on 30-task pilot (codex prompt + 1 human spot-check). Paper §1 prose ("B0 53.3% early-finish vs B1 70.4% visual-hijack/click-loop, +43.7pp") cites these locked buckets |
docs/checkpoints/pre_run/preregistration.md:208:| **Missing-data / crashed-episode policy** (audit B6) | (a) Crashed episodes (uncaught exception, OOM, timeout > 30 min, browser crash) **excluded from paired-N denominators**, **NOT imputed** to success or failure. (b) Episodes with `not_logged_in` or `auth_drift` flag at termination excluded after watchdog refresh fails 3 retries (per `experiment_watchdog.py`). (c) Missing artifacts (no `obs.txt` / `screenshot_annotated.png` at step k) excluded from per-step analyses, NOT imputed. (d) Per-cell exclusion count + reason histogram reported in Appendix C. | Listwise deletion only; mean imputation introduces bias for SR proportions, hot-deck imputation breaks paired-N pairing. Crashed-episode imputation as success/failure would inflate Type I/II error. Lock 2026-05-09. |
docs/checkpoints/pre_run/preregistration.md:210:| **Heterogeneity (random-effects, Q, I², τ²) pre-spec** (audit B8) | (a) **Primary estimator**: random-effects DerSimonian-Laird via `aggregate_phantom_meta.py` (already implemented). (b) **Heterogeneity reporting**: report Cochran Q (chi² test of homogeneity), I² (% of total variance attributable to between-cell heterogeneity), τ² (between-cell variance). (c) **Interpretation thresholds (pre-specified)**: I² < 25% = "low heterogeneity, pooled mean is primary"; 25%-50% = "moderate, report both pooled + per-cell"; 50%-75% = "high, per-cell estimates are primary, pooled is summary"; > 75% = "very high, do not pool — report only per-cell + heterogeneity-source analysis (site / model / task-pool)". (d) **Heterogeneity-source decomposition**: when I² > 50%, report meta-regression by site (cls / red / shop) and by model (B0 / B1) to identify dominant variance source. | Higgins & Thompson 2002 (I² thresholds). Per-cell estimates always shown alongside pooled, so heterogeneity is never averaged away. Lock 2026-05-09. |
docs/checkpoints/pre_run/preregistration.md:211:| **K-of-N rule scope** (audit B9 power-corrected) | The **K_h1=12/16 / K_h3=11/16** thresholds are retained as **secondary transparency checks** (count of cells *individually* clearing α=0.05 Holm), **not as gates on H1/H3 paper claims**. **Primary detection** = (a) DerSimonian-Laird random-effects meta-analysis on cells N≥10 (B8 lock above) + (b) TOST equivalence on N=910 pooled tasks at δ=1.0pp. Per `docs/analysis/cross_sites/power_analysis.md` §3-§5, K-of-N family power at observed effect sizes (1-5pp) is < 10%; the rule is calibrated for ≥7pp effects. This recharacterization is consistent with the original §4 "Primary metric" + B8 random-effects lock — K-of-N was always a transparency aggregator, not the primary test, and the corrected power analysis makes that explicit. | `power_analysis.py` bug (stale interpretation block) discovered 2026-05-09; fixed in same commit. K-of-N values themselves unchanged; only the framing as "secondary transparency vs primary gate" is added. Lock 2026-05-09. |
docs/checkpoints/pre_run/preregistration.md:224:- **Best-signal-per-mode characterization** (Register III AA novelty, Section 6 portfolio finding): which routing signal works best for which mode is reported as exploratory characterization, NOT pre-registered prediction. Per-(mode, signal) AUROC table reported with Holm correction within mode for transparency.
docs/checkpoints/pre_run/preregistration.md:234:at **L17** (3 of 4 cells Holm-significant on `token_overlap_to_target`, p_Holm <
docs/checkpoints/pre_run/preregistration.md:235:0.05; cell D L11+L17 strongest p_Holm = 0.006/0.008 \*\*). The L11/L17 layer
docs/checkpoints/pre_run/preregistration.md:242:| **Stage 2B 24-task aggregate (cell A)** | L17 Holm-significant (p_Holm = 0.011 \*\*) — confirmed Stage 2A peak | **Confirmatory** — independent metric agreement |
docs/checkpoints/pre_run/preregistration.md:243:| **Stage 2C reverse 15-task (cell B)** | L11 + L17 Holm-significant — direction-paired confirmation | **Confirmatory** |
docs/checkpoints/pre_run/preregistration.md:244:| **Cell D (rev × strong-tier 24)** | L11 + L17 strongest (p_Holm = 0.006/0.008 \*\*) | **Confirmatory** — cross-tier replication |
docs/checkpoints/pre_run/preregistration.md:249:mitigate the multiple-comparison concern, all per-direction tests use Holm-
docs/checkpoints/pre_run/preregistration.md:309:| **Mechanism analysis (Stage 2 patching)** | **Fully reproducible** | Greedy decoding + seed=42 + Holm-corrected paired t-test + 1000-resample percentile bootstrap (seed=42 in `stage2_layer_significance.py`). Per-task per-layer `patching_continuation_results.json` released for re-aggregation. |
docs/checkpoints/pre_run/preregistration.md:328:| 2026-05-03 | K_h1 = 0.75 cell-pass threshold for H1 | Allows ~25% capability-outlier cells; not so strict as to break on single-cell noise |
docs/checkpoints/pre_run/preregistration.md:330:| 2026-05-03 | Disconfirmation rule changed from "any cell fail" to data-conditional R1-R5 framing rule | "Any cell fail" too strict given single-cell power limits; framing rule maps data outcomes to paper hook revisions transparently |

exec
/bin/bash -lc 'rg -n "A10|GPU|USD|CO2|CO₂|carbon|per-cell|cost|DGX|Myriad|V100|Spark|power" docs/checkpoints/pre_run/topvenue_constraints.md docs/checkpoints/**/*.md docs/analysis/**/*.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
docs/checkpoints/codex_prompts/B0_dom_shopping_A_refine.md:39:    "rationale": "Primary axis is price aggregation (least expensive), color is secondary visual filter. Agent must sort/compare across Blankets & Throws list; visual color verification is a downstream check after sort but failure mode is aggregation cost."
docs/checkpoints/codex_prompts/mechanism_per_task_case_studies.md:23:- **4-fold drop-in property (a) cost ≈ DOM 在 B0 + B1 上都 hold**（B1 P-SoM ≈ DOM cls）
docs/checkpoints/codex_prompts/rename_phantom_paper_naming.md:42:- `scripts/analysis/aggregate_cost_electricity.py` (RUNS dict)
docs/checkpoints/codex_prompts/rename_phantom_paper_naming.md:57:- `scripts/analysis/figures/fig3a_token_cost_intra_baseline.py`
docs/checkpoints/codex_prompts/rename_phantom_paper_naming.md:58:- `scripts/analysis/figures/fig3d_cost_sr_frontier.py`
docs/checkpoints/codex_prompts/rename_phantom_paper_naming.md:60:- `scripts/analysis/figures/fig3_regional_carbon.py`
docs/checkpoints/codex_prompts/rename_phantom_paper_naming.md:90:- `docs/analysis/cross_sites/cost_per_mode.{json,md}` with new dir names in cells
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:61:- Power analysis (B9) shows minimum-detectable effect for cells with N=15 ~0.65 Cohen's d at α=0.05 — not powered for small mid-layer effects.
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:67:### 8.7 Compute, cost, and sustainability bounds (~80 words)
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:68:- Per-cell GPU-hours / USD / kg-CO₂ table per A10 reviewer-rebuttal. Cross-machine power profile (DGX Spark vs Myriad V100) is one limitation source for absolute energy numbers; relative comparisons within the same cell are unaffected.
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:124:snapshot, single-task labels may flip. The aggregate per-cell SR is robust to this within
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:242:Our work runs across three GPU architectures: DGX Spark (NVIDIA GB10, sm_121), UCL Condense
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:243:A100 (sm_80), and UCL Myriad (sm_70 V100 / sm_80 A100). Mechanistic Stage 2B/2C activation
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:249:{DGX, A100, Myriad} layers L0-L35: max |Δh| < [TBD post-rerun, target <1e-2] at L11 (the
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:297:33:| A1 | Preregister primary hypotheses, decision rules, and analysis families before post-rerun data are inspected | NEEDS_BIB_ENTRY: Pineau et al. 2021; NeurIPS checklist Q4/Q6 | ⚠️ | `docs/checkpoints/pre_run/preregistration.md` has H1-H8, Holm families, R1-R5, but frontmatter is `status: draft`, `registered_at`, `registered_git_sha`, `witnessed_by`, and OSF DOI are pending. Remediation: lock after advisor email, tag git, deposit OSF; cost 2-4h. | "The hypotheses and decision rules were written before the 16-cell rerun; the camera-ready will cite the lock SHA/OSF DOI once advisor witness is received." |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:299:57:| B6 | Predefine missing-data/crashed-episode handling | CONSORT-AI/Liu et al. 2020; NeurIPS checklist Q7 | ✓ | Locked 2026-05-09 in `preregistration.md §4`: crashed episodes (uncaught exception / OOM / >30min timeout / browser crash) excluded from paired-N denominators, NOT imputed; not_logged_in / auth_drift after 3 retries excluded; missing artifacts excluded from per-step (not full-task); per-cell exclusion count + reason histogram in Appendix C. | "Listwise deletion only — crashes excluded from paired N, no success/failure imputation, exclusion histograms reported per cell." |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:302:90:| D6 | State failed assumptions and limitations in a dedicated section | NeurIPS checklist Q2; NEEDS_BIB_ENTRY: Lipton & Steinhardt 2018 | ⚠️ | `section4_limitations_disclosure.md` exists and `pre_rerun_audit.md §4.4` lists coverage, but final Section 8 prose is still TBD. Remediation: consolidate limitations in final paper; cost 0.5 day. | "The limitations appendix already enumerates evaluator, scaffold, numerical, and pre/post-fix threats." |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:303:113:| F2 | Construct validity: ensure metrics measure task success and not evaluator artifacts | `zhou2024webarena`; `koh2024visualwebarena`; NEEDS_BIB_ENTRY: Lipton & Steinhardt 2018 | ⚠️ | Adjusted SR and FP filters address evaluator artifacts; `pre_rerun_audit.md §4.5.4` LLM-judge reproducibility is TBD. Remediation: judge spot-check and per-evaluator-class sensitivity; cost 0.5 day. | "We report raw and adjusted success and isolate evaluator-class threats in limitations." |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:305:115:| F4 | Statistical conclusion validity: report uncertainty and sensitivity to thresholds | NeurIPS checklist Q7; NEEDS_BIB_ENTRY: Cook & Campbell 1979 | ⚠️ | CIs/meta-analysis exist; `pre_rerun_audit.md §4.8.1` K±1 and `§4.8.2` leave-one-cell-out are TBD. Remediation: add threshold-gradient and cell-removal table; cost 0.5 day. | "The decision rule is locked; sensitivity tables show how close the conclusion is to the threshold." |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:307:141:| H1 | Maintain a dedicated limitations section with strong assumptions and robustness failures | NeurIPS checklist Q2; NEEDS_BIB_ENTRY: Lipton & Steinhardt 2018 | ⚠️ | `section4_limitations_disclosure.md` exists and `pre_rerun_audit.md §4.4` enumerates limitations; final Section 8 prose is not complete. Remediation: consolidate into final limitations section; cost 0.5 day. | "Limitations are not hidden in appendices; they are surfaced alongside the main empirical claims." |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:336:| 4.2.7 | Cross-machine numerical agreement (DGX/A100/Myriad) | 🟡 needs A100/Myriad SSH | `numerical_determinism_check.py compare` |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:370:**Source**: 笔记 §92 (P1-P14 rule scripting), §93 (analysis pipeline 27-check + intent + cost), §94 finding (max_marks reversal), §97 (cross_representation audit), §98 (watchdog cross-run automation).
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:432:| **1** | 2026-04-28 | **Phantom-DOM 18 modes scope** (full factorial expansion across image / SoM / prompt axes) | 5-mode scope (DOM / SoM / Vision / phantom_som / phantom_dom) | 18 modes exceeded paper-grade focus; 13 ablation modes added marginal information at high compute cost | §3 paper-grade scope explicit; pre-rerun audit §1.1 enforces |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:452:| C1 | 2026-04-26 | Phantom-SoM 4-fold drop-in property (cost ≈ DOM / latency ~50% / signal AUROC ≥ baseline / drop-one 1.7-3.8pp) | B0 reddit Phase A archived data + drop-one sig | Paper hook §1 (provisional pending 16-cell rerun) |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:494:53:- **H1(ii)** ≥ K_h1 of N_cells individually Holm-significant at α=0.05 within the per-cell P-SoM sub-family (m = N_cells), where **K_h1 = 0.75** (commit-locked, see §4).
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:518:109:- **H7(ii)** ≥ K_h1 of N_cells individually Holm-significant on per-cell lift, bootstrap 95% CI lower-bound > 0.
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:530:140:| **R4** | H1 holds AND H2 partially fails (e.g., (a) cost or (b) latency fails on some site) | "Phantom-SoM partial drop-in" + §4 disclosure of failed sub-claim. | WEAK; substantial revision |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:534:149:- H1(ii) per-cell P-SoM Holm: m = N_cells.
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:535:150:- H2 sub-claims (a)(b)(c)(d): m = 4 × N_cells (each per-cell sub-claim test).
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:537:154:- H3(i) axis 1 per-cell: m = N_cells (bootstrap CI lower-bound > 0 test).
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:538:155:- H3(ii) axis 2 per-cell: m = N_cells.
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:541:161:- H7(ii) per-cell Tier 1 lift Holm: m = N_cells.
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:557:191:| **Effect size (continuous)** | Cohen's d with bootstrap CI | For cost/latency H2(a)(b) |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:558:192:| **TOST equivalence margin δ** | **1.0pp** | ≈ 2 tasks in N=234, matches per-cell bootstrap SE; smaller is within sampling noise floor |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:559:193:| **H1 K_h1 cell-pass threshold** | **0.75** | Allows ~25% capability-outlier cells (e.g., B1 shopping power-limited); not so strict that single-cell noise breaks claim |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:565:210:| **Heterogeneity (random-effects, Q, I², τ²) pre-spec** (audit B8) | (a) **Primary estimator**: random-effects DerSimonian-Laird via `aggregate_phantom_meta.py` (already implemented). (b) **Heterogeneity reporting**: report Cochran Q (chi² test of homogeneity), I² (% of total variance attributable to between-cell heterogeneity), τ² (between-cell variance). (c) **Interpretation thresholds (pre-specified)**: I² < 25% = "low heterogeneity, pooled mean is primary"; 25%-50% = "moderate, report both pooled + per-cell"; 50%-75% = "high, per-cell estimates are primary, pooled is summary"; > 75% = "very high, do not pool — report only per-cell + heterogeneity-source analysis (site / model / task-pool)". (d) **Heterogeneity-source decomposition**: when I² > 50%, report meta-regression by site (cls / red / shop) and by model (B0 / B1) to identify dominant variance source. | Higgins & Thompson 2002 (I² thresholds). Per-cell estimates always shown alongside pooled, so heterogeneity is never averaged away. Lock 2026-05-09. |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:566:211:| **K-of-N rule scope** (audit B9 power-corrected) | The **K_h1=12/16 / K_h3=11/16** thresholds are retained as **secondary transparency checks** (count of cells *individually* clearing α=0.05 Holm), **not as gates on H1/H3 paper claims**. **Primary detection** = (a) DerSimonian-Laird random-effects meta-analysis on cells N≥10 (B8 lock above) + (b) TOST equivalence on N=910 pooled tasks at δ=1.0pp. Per `docs/analysis/cross_sites/power_analysis.md` §3-§5, K-of-N family power at observed effect sizes (1-5pp) is < 10%; the rule is calibrated for ≥7pp effects. This recharacterization is consistent with the original §4 "Primary metric" + B8 random-effects lock — K-of-N was always a transparency aggregator, not the primary test, and the corrected power analysis makes that explicit. | `power_analysis.py` bug (stale interpretation block) discovered 2026-05-09; fixed in same commit. K-of-N values themselves unchanged; only the framing as "secondary transparency vs primary gate" is added. Lock 2026-05-09. |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:603:330:| 2026-05-03 | Disconfirmation rule changed from "any cell fail" to data-conditional R1-R5 framing rule | "Any cell fail" too strict given single-cell power limits; framing rule maps data outcomes to paper hook revisions transparently |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:641:one-out layer-selection** robustness — re-run patching on per-cell holdout
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:655:This paper questions that bundling assumption as an experimental object. We characterize the **phantom routing space**: configurations on the "skip annotated image" boundary that retain some SoM-derived textual or prompt structure while removing the image. Its deployment-relevant representative is **Phantom-SoM**: the agent receives the SoM prompt and the `[SOM_MARKS]` textual element list, but no image. The structural controls are **P-text** (the `[SOM_MARKS]` text under the DOM prompt) and **P-prompt** (the SoM prompt over AXTree text). At the start of this project, Phantom-SoM looked like a broken ablation. The natural expectation was that removing the marked screenshot would collapse SoM into either a weak DOM surrogate or a nonsensical configuration: the prompt still suggests visual marks, but the visual substrate is absent. The data reject that expectation. Phantom-SoM solves tasks that DOM, full SoM, and Vision all miss, and on B0 reddit it matches or modestly exceeds full SoM by adjusted SR (**13.81% vs 10.48%, N=210**; the gap is within 2σ under the run-to-run variability we observe in same-condition repeats), while avoiding image-token cost. On classifieds, full SoM remains clearly stronger (**21.37% vs Phantom-SoM 14.53%, N=234**), the expected sanity check when marked screenshots carry real visual information.
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:657:Our first contribution is a controlled scientific evaluation of this phantom boundary. Across completed B0 VisualWebArena classifieds and reddit runs, we compare DOM, full SoM, Vision, and Phantom-SoM on the same task sets (**N=234 classifieds; N=210 reddit; same-task adjusted SR**) and use the P-text/P-prompt controls to test whether the effect collapses to one prompt trick or one text-format swap. Phantom-SoM is not the best single arm on every site, and we do not claim that it replaces full SoM. Its value is complementarity. Its task-success pool has low overlap with the established modes, with Jaccard similarity in the roughly **0.29-0.49** range against other arms, and its removal reduces the oracle. In drop-one oracle analysis, Phantom-SoM contributes **3.33 percentage points** of incremental oracle value on reddit (comparable to full SoM at **1.90 pp**) and **2.56 percentage points** on classifieds, placing Phantom-SoM consistently within the top routing-value arms despite using no image. Crucially, the cost of obtaining this configuration is essentially the cost of the DOM baseline: the `[SOM_MARKS]` block is produced by a regex pass over the same accessibility-tree text the DOM agent already consumes (interactive elements come pre-numbered as `[N] role 'label'`), so a deployment that can run DOM can run Phantom-SoM by changing what it forwards to the model: no bounding-box pipeline, no marked image, no extra inference modality. We therefore preserve the empirical **4-fold drop-in property** as the paper's practical finding: cost approximately DOM, lower image-stage latency, usable routing signal, and positive drop-one oracle value.
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:684:2827:- Section 3.2 image-token estimate 从 total cost 反推 → step-level `tokens.input` median (`4d63c9f`)
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:709:4823:A100 80GB VRAM × Qwen3-VL-4B bf16 ~10GB = 8× 余量, 可 cell-parallel 跑 multiple 16-cell rerun slots. Mechanistic Stage 2B scale-up 跟 16-cell rerun 同时跑互不干扰.
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:712:4855:简明 email 200 字, cite §111 GitHub commit chain, update advisor 两个 finding (L11 + asymmetry) + A100 allocation. 不加新 Q, follow-up doc Q1-Q11 仍 primary ask. 期待 advisor 看到 mechanistic same-day delivery + A100 unblock 后 follow-up Q3 (B1-only vs cross-arch) lean cross-arch + Q1 (3 vs 4 paper) lean 4 paper (mechanistic 独立).
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:714:4868:**触发**: §112 A100 等 Steve PCI Devices 期间, §111.7 caveat 4 (N=3 太小, 1/3 task clean signal) 是 paper §5 主要弱点. 写 `scripts/mechanistic/curate_mirage_tasks.py` 自动 score 全 234 cls task 的 source-vs-target divergence, 找 paper-grade mirage cluster.
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:722:4920:**Paper §5 quotable claim 升级版** (post-Stage 2B curated scale-up on A100):
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:724:4932:1. **Stage 2B curated scale-up** (24 task × forward direction × 36 layer) on A100 — replace §111 N=3 placeholder with paper-grade aggregate
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:731:5040:| ⏳ Gap 5 A100 capture + compare | A100 setup 完 + Stage 2B scale-up 之前 | Same as above |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:737:5604:## 117. Stage 2B/2C Myriad scale-up — bidirectional mid-layer disruption + §111.5b "reverse null" overturned (2026-05-09) [finding][infra] #finding #infra
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:738:5608:DGX→quark Tailscale→Myriad SSH chain newly opened (笔记 §117.6 infra). Stage 2B forward (24 strong-tier tasks) + Stage 2C reverse (15 reverse-tier tasks) run on Myriad A100 80GB compute node `node-v00a-001` / `node-u00a-001`. ~30min wallclock each (V100 estimate had been 24h — A100 80GB ~48× speedup). Output dirs:
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:765:## 117. Stage 2B/2C Myriad scale-up — bidirectional mid-layer disruption + §111.5b "reverse null" overturned (2026-05-09) [finding][infra] #finding #infra
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:769:DGX→quark Tailscale→Myriad SSH chain newly opened (笔记 §117.6 infra). Stage 2B forward (24 strong-tier tasks) + Stage 2C reverse (15 reverse-tier tasks) run on Myriad A100 80GB compute node `node-v00a-001` / `node-u00a-001`. ~30min wallclock each (V100 estimate had been 24h — A100 80GB ~48× speedup). Output dirs:
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:773:Both rsync'd back to DGX via tar-pipe through SSH chain.
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:802:| Reverse × LD→tgt | None Holm-significant | L17 raw p=0.014 marginal Holm 0.084 (N=15 power-limited) |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:831:### 117.6 Infrastructure — DGX-driven Myriad workflow established
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:833:DGX→quark Tailscale→quark Cisco AnyConnect→Myriad SSH chain verified working (B-81 umbrella + new 7th class B-81g UTF-8 ASCII codec fix). Watcher cron `scripts/maintenance/glm/myriad_watcher.py` installed (5min ntfy on state change). DGX never directly reaches UCL gateway (IP-restricted) — quark Windows is mandatory bastion. Documented in `docs/reference/COMPUTE_INFRASTRUCTURE.md` + `memory/reference_dgx_to_ucl_chain.md`.
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:844:- [ ] Wait for cells C/D (~30 min A100 each, qw queued)
docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:31:The standalone success table supports the same interpretation without requiring oracle selection. P-text raises adjusted SR over DOM on the full reddit set while preserving the same no-image deployment class (Outcome 0a, DOM 9.52% versus P-text 12.38%, N=210). The effect is not simply that P-text acts more: it uses fewer steps on average in the cost summary, and the cascade's largest text-axis macro shift is action-repeat rather than search success itself (Macro 1b, text-axis action-repeat +4.64pp). This is compatible with a routing-surface mechanism. The agent sometimes repeats a newly exposed marked control, but it is no longer confined to the same query-rewrite loop.
docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:53:The outcome result rejects a naive monotone-vision story. Full SoM is below Phantom-SoM on adjusted reddit success (Outcome 0a, SoM 10.48% versus P-SoM 13.81%, a -3.33pp regression after adding the image). The image still changes behavior: the cascade shows an increase in finish rate and fewer steps when the image is added (Macro 1b, image-axis finish rate +10.95pp; step count -1.85), but on reddit this can mean earlier commitment from the wrong state rather than more correct routing. The click fraction also falls under the image axis (Macro 1b, click fraction -3.08pp), which is consistent with visual confidence substituting for link-following in some traces. The cost side makes the tradeoff concrete: adding the screenshot increases median token load without a reddit SR benefit (Efficiency 3b, SoM versus P-SoM observed gap 778 tokens/step).
docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:77:Section 6 routing implementation will leverage these mechanism insights by treating reddit Phantom-SoM as a low-cost complementary text-routing arm, reserving image escalation for tasks where visual content is likely to resolve the target rather than trap the agent on image anchors.
docs/checkpoints/codex_prompts/rename_figures_layered.md:26:| `fig3d_cost_sr_frontier.png` | `fig3d_cost_sr_frontier.png` | 3d cost-SR Pareto |
docs/checkpoints/codex_prompts/rename_figures_layered.md:27:| `fig3d_cost_sr_frontier.py` | `fig3d_cost_sr_frontier.py` | (script) |
docs/checkpoints/codex_prompts/rename_figures_layered.md:30:| `fig3_regional_carbon.png` | `fig3_regional_carbon.png` | 3 regional carbon (B1) |
docs/checkpoints/codex_prompts/rename_figures_layered.md:31:| `fig3_regional_carbon.py` | `fig3_regional_carbon.py` | (script — note current py name has extra `_carbon_` 前缀) |
docs/checkpoints/codex_prompts/rename_figures_layered.md:111:### 7.5. 顺便修 fig3d_cost_sr_frontier (即 fig7) 的 cost source
docs/checkpoints/codex_prompts/rename_figures_layered.md:113:**问题**：当前 fig3d (fig7) 直接读 `condition_summary_v2.json` 的 `avg_total_cost_usd`，这个字段对 B1 是 artifact (用 B0 rate 算的，~$0.05/ep 假象)。fig 上现在显示 "Phantom-SoM/SoM cost 0.9-1.1×" 是 token-cost ratio in B0 only —— 但 paper Section 3d 真正想讲的是 B0 vs B1 deployment-class gap **~100×**（见 `docs/analysis/cross_sites/cost_per_mode.md`）。
docs/checkpoints/codex_prompts/rename_figures_layered.md:116:- B0 cell 继续用 `avg_total_cost_usd` (real API token $)
docs/checkpoints/codex_prompts/rename_figures_layered.md:117:- **B1 cell 改用 `cost_per_mode.json` 里的 `paper_cost_usd`** (即 `avg_total_energy_kwh × $0.12/kWh` electricity-equivalent)
docs/checkpoints/codex_prompts/rename_figures_layered.md:118:- 加一个 prominent annotation 在 figure 顶部："B0 reports API token \\$; B1 reports electricity-equivalent \\$ (different cost classes)"
docs/checkpoints/codex_prompts/rename_figures_layered.md:121:- 保留现有 5-mode marker 颜色 + Phantom-SoM；不要删 SoM/Phantom-DOM cost ratio annotation（那是 Layer 3a intra-baseline ratio，paper 也用）
docs/checkpoints/codex_prompts/rename_figures_layered.md:124:读 `docs/analysis/cross_sites/cost_per_mode.json` 来 lookup `cells[baseline][site][mode].paper_cost_usd`，rather than直接 read condition_summary_v2.json. 这样统一 source of truth，与 layered_evidence_status 的 Layer 3d 口径一致。
docs/checkpoints/codex_prompts/rename_figures_layered.md:128:- 不要改 figure 数据计算 / 颜色 / layout（除了 fig3d cost source 上面 7.5）
docs/checkpoints/codex_prompts/rename_figures_layered.md:129:- 不要改 cost rate（cost_per_mode.json 的 $0.12/kWh 是 ground truth, 不要变）
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:29:snapshot, single-task labels may flip. The aggregate per-cell SR is robust to this within
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:147:Our work runs across three GPU architectures: DGX Spark (NVIDIA GB10, sm_121), UCL Condense
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:148:A100 (sm_80), and UCL Myriad (sm_70 V100 / sm_80 A100). Mechanistic Stage 2B/2C activation
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md:154:{DGX, A100, Myriad} layers L0-L35: max |Δh| < [TBD post-rerun, target <1e-2] at L11 (the
docs/checkpoints/pre_run/osf_lock_manifest.md:22:- [ ] `env_snapshot.json` of latest run on each machine (DGX, A100, Myriad if
docs/checkpoints/pre_run/osf_lock_manifest.md:49:| TOST equivalence δ | 1.0pp | ⏳ pending | Equivalence margin for "cost ≈ DOM" claim |
docs/checkpoints/pre_run/osf_lock_manifest.md:55:| DGX `spark-9ea3` (Phase 1 baseline) | `results/provenance/env_lock_dgx.json` | `results/provenance/vwa_dgx_via_quark.json` | TBD |
docs/checkpoints/pre_run/osf_lock_manifest.md:56:| A100 `condense` (Phase 2 rerun + mechanistic) | `results/provenance/env_lock_a100.json` | `results/provenance/vwa_a100_self_host.json` | TBD |
docs/checkpoints/pre_run/osf_lock_manifest.md:57:| Myriad (cross-arch backup, optional) | `results/provenance/env_lock_myriad.json` | N/A (no VWA use case) | TBD if used |
docs/checkpoints/pre_run/osf_lock_manifest.md:61:GPU compute capability / hostname / nvidia-smi output.
docs/checkpoints/pre_run/osf_lock_manifest.md:77:3. **Run `python3 scripts/provenance/snapshot_env.py`** on DGX + A100 (+ Myriad
docs/analysis/cross_sites/pilot_t0_wave3_final.md:77:   - B1 same 14 cells — likely needs RunPod's dedicated 4090 (DGX too slow + contention)
docs/analysis/cross_sites/probe_b37_api_determinism.md:58:{"content":"{\n  \"action_type\": \"click\",\n  \"element_id\": 5,\n  \"thought\": \"The $320 Blue Inflatable Kayak is the cheapest option among the listed blue kayaks.\"\n}","model":"qwen.qwen3-vl-235b-a22b","usage":{"inputTokens":162,"outputTokens":46,"cost":0.00039200000000000004},"metadata":{"remaining_quota":{"llm_cost":332.60454799999997,"total_cost":332.60454799999997,"budget_limit":350,"remaining_budget":17.39545200000001,"budget_usage_percent":95.02987085714285}}}
docs/analysis/cross_sites/probe_b37_api_determinism.md:63:{"content":"{\"action_type\": \"click\", \"element_id\": 5, \"thought\": \"The $320 blue inflatable kayak is the cheapest option available in the search results.\"}","model":"qwen.qwen3-vl-235b-a22b","usage":{"inputTokens":162,"outputTokens":38,"cost":0.000352},"metadata":{"remaining_quota":{"llm_cost":332.60490000000004,"total_cost":332.60490000000004,"budget_limit":350,"remaining_budget":17.395099999999978,"budget_usage_percent":95.02997142857143}}}
docs/analysis/cross_sites/probe_b37_api_determinism.md:68:{"content":"{\"action_type\": \"click\", \"element_id\": 5, \"thought\": \"The cheapest blue kayak listed is the $320 Blue Inflatable Kayak, so I will click on it to view more details.\"}","model":"qwen.qwen3-vl-235b-a22b","usage":{"inputTokens":162,"outputTokens":49,"cost":0.000407},"metadata":{"remaining_quota":{"llm_cost":332.605307,"total_cost":332.605307,"budget_limit":350,"remaining_budget":17.394693000000014,"budget_usage_percent":95.0300877142857}}}
docs/analysis/cross_sites/probe_b37_api_determinism.md:73:{"content":"{\"action_type\": \"click\", \"element_id\": 5, \"thought\": \"The cheapest blue kayak listed is the $320 Blue Inflatable Kayak, which matches the goal of finding the lowest-priced option.\"}","model":"qwen.qwen3-vl-235b-a22b","usage":{"inputTokens":162,"outputTokens":49,"cost":0.000407},"metadata":{"remaining_quota":{"llm_cost":332.605714,"total_cost":332.605714,"budget_limit":350,"remaining_budget":17.39428600000002,"budget_usage_percent":95.030204}}}
docs/analysis/cross_sites/probe_b37_api_determinism.md:78:{"content":"{\n  \"action_type\": \"click\",\n  \"element_id\": 5,\n  \"thought\": \"The $320 Blue Inflatable Kayak is the cheapest option available among the search results.\"\n}","model":"qwen.qwen3-vl-235b-a22b","usage":{"inputTokens":162,"outputTokens":45,"cost":0.00038700000000000003},"metadata":{"remaining_quota":{"llm_cost":332.606101,"total_cost":332.606101,"budget_limit":350,"remaining_budget":17.393898999999966,"budget_usage_percent":95.03031457142858}}}
docs/checkpoints/pre_run/pre_rerun_audit.md:4:on A100 (post advisor email + SSH cert). Designed to catch spec/code drift,
docs/checkpoints/pre_run/pre_rerun_audit.md:135:| 1.4.3 | Power analysis / MDE | ✅ today | `scripts/analysis/power_analysis.py` + `docs/analysis/cross_sites/power_analysis.md` |
docs/checkpoints/pre_run/pre_rerun_audit.md:151:| 1.4.13 | Cross-machine reproducibility (DGX/A100/Myriad) | 🟡 | numerical_determinism_check post-rerun |
docs/checkpoints/pre_run/pre_rerun_audit.md:184:| 1.7.9 | **`COMPUTE_INFRASTRUCTURE.md`** — live infra landscape (UCL Condense A100 / Myriad HPC / DGX Spark, SSH paths, allocation sources) | ✅ exists | `docs/reference/COMPUTE_INFRASTRUCTURE.md` |
docs/checkpoints/pre_run/pre_rerun_audit.md:201:| 1.6.6 | A.6 Mechanistic interpretability publication-worthy | ✅ Stage 2B/2C running on Myriad |
docs/checkpoints/pre_run/pre_rerun_audit.md:203:| 1.6.8 | A.8 Compute paths (A100 / Myriad / advisor 5090) | A100 🟡 SSH cert / Myriad ✅ |
docs/checkpoints/pre_run/pre_rerun_audit.md:236:| 2.1.8 | **`scripts/setup/a100_self_host_vwa.sh`** — A100 self-host VWA runbook (replaces Tailscale↔quark dependency) | 🟡 needs A100 SSH | one-time ~1-2h setup; eliminates Tailscale single-point failure |
docs/checkpoints/pre_run/pre_rerun_audit.md:244:| 2.2.2 | snapshot_vwa.sh (DGX baseline + A100 self-host) | DGX ✅ / A100 🟡 | `ls results/provenance/vwa_*.json` |
docs/checkpoints/pre_run/pre_rerun_audit.md:245:| 2.2.3 | numerical_determinism_check ready | 🟡 needs A100/Myriad SSH | script exists per §114 Gap 5 |
docs/checkpoints/pre_run/pre_rerun_audit.md:246:| 2.2.4 | sitecustomize.py shim (Myriad-only RHEL 7) committed | ✅ | `myriad_bootstrap.sh` |
docs/checkpoints/pre_run/pre_rerun_audit.md:247:| 2.2.5 | constraints.txt (urllib3<2 / numpy<2) | ✅ Myriad only | `myriad_constraints.txt` |
docs/checkpoints/pre_run/pre_rerun_audit.md:283:| 2.5.4 | GPU OOM | watchdog kill + restart | 🟡 partial |
docs/checkpoints/pre_run/pre_rerun_audit.md:310:| 2.6.4 | Auth state regeneration per-site per-cell | ✅ B-66 | per-site files |
docs/checkpoints/pre_run/pre_rerun_audit.md:325:| 2.8.6 | **GPU forward pass smoke** — load model + 1 forward pass on each target machine | 🔴 TBD | A100/Myriad/DGX: `python3 -c "import torch; from p79.agents.qwen3vl_agent import Qwen3VLAgent; ..."` |
docs/checkpoints/pre_run/pre_rerun_audit.md:327:| 2.8.8 | **`scripts/setup/smoke_login.sh`** — Myriad SSH + compute node connectivity smoke | ✅ exists | run pre-Myriad qsub |
docs/checkpoints/pre_run/pre_rerun_audit.md:333:| 2.7.1 | A100 GPU OOM mid-cell | 🟡 partial | Watchdog auto-clean; verify resume |
docs/checkpoints/pre_run/pre_rerun_audit.md:334:| 2.7.2 | Myriad qsub job killed (wallclock) | 🟡 partial | run_stage2b incremental save; --resume flag (R4 pending) |
docs/checkpoints/pre_run/pre_rerun_audit.md:339:| 2.7.7 | Network partition (DGX↔quark Tailscale OR A100↔bastion) | 🟡 | A100 self-host VWA solves |
docs/checkpoints/pre_run/pre_rerun_audit.md:341:| 2.7.9 | **Quark host watchdog** — VWA Docker on quark, alert if quark off / Docker daemon dead | 🔴 TBD | Cron on DGX or laptop: `curl -sI http://100.95.81.103:9980` every 10 min, ntfy on fail |
docs/checkpoints/pre_run/pre_rerun_audit.md:342:| 2.7.10 | **Backup restore protocol** — Scratch corrupt recovery | 🔴 TBD | Document: archive subset is in git (16.5MB safe); HF model cache re-downloadable; experiment results need DGX backup if A100 Scratch lost |
docs/checkpoints/pre_run/pre_rerun_audit.md:358:| 3.1.7 | **`docs/analysis/B1_capability_profile.md`** — B1 (Qwen3-VL-4B) snapshot: 6-cell SR + cost/latency/energy + failure patterns vs B0 | update post-rerun |
docs/checkpoints/pre_run/pre_rerun_audit.md:372:| 3.2.9 | **`validate_run.py` 27-check suite per cell** (笔记 §91 + §93 expanded) | ✅ exists | `python3 scripts/analysis/validate_run.py --run-dir <run> --strict` post-cell. **10 check groups, 27 checks**: G1 file existence (C01-02) / G2 structure (C03-06) / G3 coverage (C07-08) / G4 episode integrity (C09-11) / G5 step integrity (C12-15) / G6 scaffold safety (C16-18) / G7 artifact integrity (C19-20) / G8 analysis freshness (C21-22) / **G9 Temporal Analysis (C23-25)** — SR-over-time degradation, auth drift, reset contamination / **G10 Data Consistency (C26-27)** — summary.steps vs JSONL line count, zero-cost episode detection. Exit 0/1/2 |
docs/checkpoints/pre_run/pre_rerun_audit.md:376:| 3.2.13 | **Intent feature + cost attribution columns** (笔记 §93) — `analyze_reason_diagnostics.py` outputs +16 columns (10 `intent_has_*` booleans + 6 cost cols) + 7 plots | ✅ deployed | invoked by `make reason-diag` per cell |
docs/checkpoints/pre_run/pre_rerun_audit.md:377:| 3.2.14 | **Waste breakdown** (`metrics.py::compute_waste_breakdown`) — no_op / page_unchanged / total / wasted cost dimensions per cell | ✅ §93 | aggregated post-rerun |
docs/checkpoints/pre_run/pre_rerun_audit.md:387:| 3.3.4 | Per-episode auth refresh (Magento 302 + B-35 time-based) | ✅ B-70/B-35 | per-cell + 1200s fallback |
docs/checkpoints/pre_run/pre_rerun_audit.md:393:| 3.3.10 | GPU memory leak detection | 🟡 partial | watchdog has B-62 fix per BLIP-2 |
docs/checkpoints/pre_run/pre_rerun_audit.md:399:| 3.4.1 | Per-cell GPU-hours estimate | ✅ `condition_summary_v2.json` | aggregate post-rerun |
docs/checkpoints/pre_run/pre_rerun_audit.md:400:| 3.4.2 | Per-cell USD cost (B0 API) | ✅ `cost_usd.model` per step | aggregate post-rerun |
docs/checkpoints/pre_run/pre_rerun_audit.md:401:| 3.4.3 | Carbon footprint per cell (45-region) | ✅ `aggregate_cost_electricity.py` | run post-rerun |
docs/checkpoints/pre_run/pre_rerun_audit.md:403:| 3.4.5 | Cross-platform GPU power profile | 🔴 TBD | NVML probe per cell start |
docs/checkpoints/pre_run/pre_rerun_audit.md:418:| 4.1.5 | Post-rerun power analysis re-run with observed SR | 🔴 TBD | Update `power_analysis.py --baseline-sr <observed>` |
docs/checkpoints/pre_run/pre_rerun_audit.md:434:| 4.2.7 | Cross-machine numerical agreement (DGX/A100/Myriad) | 🟡 needs A100/Myriad SSH | `numerical_determinism_check.py compare` |
docs/checkpoints/pre_run/pre_rerun_audit.md:468:**Source**: 笔记 §92 (P1-P14 rule scripting), §93 (analysis pipeline 27-check + intent + cost), §94 finding (max_marks reversal), §97 (cross_representation audit), §98 (watchdog cross-run automation).
docs/checkpoints/pre_run/pre_rerun_audit.md:527:| 4.9.1 | Same input → same L11 hidden state (within 1e-3) cross-machine | 🟡 needs A100/Myriad SSH | `numerical_determinism_check.py compare` on stage2b_curated subset |
docs/checkpoints/pre_run/pre_rerun_audit.md:548:3. Run `python3 scripts/provenance/snapshot_env.py` on DGX + A100 + Myriad
docs/checkpoints/pre_run/pre_rerun_audit.md:569:| 5.1.5 | env_snapshot.json (per host: DGX + A100 + Myriad) | OSF + GitHub `results/provenance/` | DGX ✅ / others 🟡 |
docs/checkpoints/pre_run/pre_rerun_audit.md:570:| 5.1.6 | snapshot_vwa.json (per VWA host) | OSF + GitHub | DGX ✅ / others 🟡 |
docs/checkpoints/pre_run/pre_rerun_audit.md:575:| 5.1.11 | Stage 2B/2C results — full per-task .json + curves.png + run_manifest.json | OSF + GitHub `results/mechanistic/` | 🟡 post-Myriad |
docs/checkpoints/pre_run/pre_rerun_audit.md:579:| 5.1.15 | **`scripts/maintenance/rsync_results_to_hub.sh`** + `rsync_results_from_hub.sh` — A100↔central hub artifact sync | OSF deposit prep | ✅ exists |
docs/checkpoints/pre_run/pre_rerun_audit.md:602:| 5.3.5 | Compute cost transparency | 🟡 partial | Paper §3 / §8 — total GPU-hours + USD + carbon |
docs/checkpoints/pre_run/pre_rerun_audit.md:661:- `docs/analysis/cross_sites/power_analysis.md` (created 5/8 — paper §3 cite-ready)
docs/checkpoints/pre_run/pre_rerun_audit.md:678:**Last expansion**: 2026-05-08, 笔记 §116.15 — repo-wide scripts/docs/笔记 sweep (5 phases × 25 sections × ~245 gate items): §1.4.7b EVIDENCE_LAYER_AUDIT / §1.7.9-13 infrastructure & data layer / §2.1.6-8 preflight + 16-cell orchestrator + A100 self-host / §2.3.7 GLM pipeline / **§2.5b 7-probe bug self-verification chain** / §2.8.7-8 smoke scripts / §3.1.6-7 progress trackers / §3.2.15 B0 vision coord errors / §4.1.6-9 meta-analysis + reeval + dual-track reframe / §4.2.8-12 5 behavior diagnostics / **§4.9.6-13 Stage 1+2A mechanistic pipeline** / §5.1.15-18 replication artifacts.
docs/analysis/vwa_classifieds/B0_SOM_digest.md:151:| 模式 | Adjusted SR (/224) | 平均成本 | cost_efficiency_ratio |
docs/analysis/vwa_classifieds/B0_SOM_digest.md:157:SoM 的 cost_efficiency_ratio（SR/cost）是三模式中最高的（4.95）。
docs/analysis/vwa_classifieds/B0_SOM_digest.md:176:| cost_efficiency_ratio | **4.95** | 3.03 | 4.86 |
docs/analysis/vwa_classifieds/B0_SOM_digest.md:179:SoM 步数少于 DOM（8.60 vs 11.56），略多于 Vision（7.85）。SoM 的 cost_efficiency_ratio（SR/cost）最高。但 P95 延迟显著高于 DOM（SoM 图文混合请求在 proxy API 上更耗时）。
docs/analysis/vwa_reddit/B0_DOM_digest.md:116:> 来源：A4b_fail_reason_cost_stats.csv（两模式合并数据）
docs/analysis/vwa_reddit/B0_DOM_digest.md:126:**Wasted cost**：DOM 平均 wasted cost = $0.0472/ep（即失败 episode 的平均成本），占总成本的 91.4%。
docs/analysis/cross_sites/power_analysis.md:5:This appendix updates the pre-registered power analysis with **observed adjusted-SR levels** from `sr_fp_per_mode.md` (Phase 1 B0 + B1 done cells, pre-paper-grade rerun). The post-rerun version will replace this file once 16-cell aggregation completes.
docs/analysis/cross_sites/power_analysis.md:26:Run: `python3 scripts/analysis/power_analysis.py --baseline-sr {0.10,0.15,0.20}`
docs/analysis/cross_sites/power_analysis.md:34:**Key observation**: minimum detectable effect at 80% per-cell power is **5-7pp** for cls/red, **4-5pp** for shop. The **observed mechanism effect (1-5pp)** is at or below per-cell MDE in 2 of 3 sites — **per-cell power for typical phantom effects is < 50%**.
docs/analysis/cross_sites/power_analysis.md:36:## 3. Family-wise power at observed effects (K-of-N rule, baseline SR=0.15 proxy)
docs/analysis/cross_sites/power_analysis.md:38:| Per-cell power (proxy effect on smallest site) | K_h1=12/16 family power | K_h3=11/16 family power |
docs/analysis/cross_sites/power_analysis.md:48:- **K_h1=12/16** is calibrated for **≥7pp effects** with paper-grade ≥0.80 family power. For typical phantom mechanism effects (1-5pp), K_h1 family power is **<10%**.
docs/analysis/cross_sites/power_analysis.md:49:- **K_h3=11/16** is slightly more permissive but still requires per-cell power ≥0.65 (≈6pp effect at SR=0.15) to reach 0.49 family power.
docs/analysis/cross_sites/power_analysis.md:55:- **Primary effect-detection test** = DerSimonian-Laird random-effects meta-analysis (locked by B8) on cells with N≥10. This is power-adequate at the cross-cell level for effects ≥2pp.
docs/analysis/cross_sites/power_analysis.md:62:"At observed adjusted-SR levels (8-15% across sites) and observed mechanism effect sizes (1-5pp), per-cell statistical power is below 0.55 in two of three sites. We therefore rely on (a) DerSimonian-Laird random-effects meta-analysis across all cells (B8 lock; cross-cell pooling raises effective power) and (b) TOST equivalence on the full N=910 pooled task set (δ=1.0pp margin) as primary tests. The K-of-N family-wise rule pre-registered for transparency is not powered for sub-5pp effects, and we report its outcome as a secondary observation rather than a gate on the main hypotheses."
docs/analysis/cross_sites/power_analysis.md:66:The original `power_analysis.py` (pre-2026-05-09) contained a stale interpretation block claiming "K_h1=12/16 with per-cell power=0.65 → >0.95 family-wise power" — this was numerically inconsistent with the K-of-N table on the same page (actual value 0.289). Fixed in commit (current session) to compute family power **at observed effect sizes** rather than at hypothetical per-cell power levels. The corrected version is what this appendix relies on.
docs/analysis/cross_sites/power_analysis.md:69:- `scripts/analysis/power_analysis.py` (script, fixed)
docs/analysis/cross_sites/power_analysis.md:70:- `docs/analysis/cross_sites/power_analysis_observed_sr10.md` / `sr15.md` / `sr20.md` (full per-baseline tables)
docs/analysis/cross_sites/power_analysis.md:78:- [ ] Add `power_analysis_post_rerun.md` once observed effect sizes are final (replace this file)
docs/analysis/cross_sites/B0_cross_site_findings.md:133:> Cost efficiency ratio = success_rate / avg_total_cost_usd，值越高越好。
docs/checkpoints/pre_run/topvenue_constraints.md:33:| A1 | Preregister primary hypotheses, decision rules, and analysis families before post-rerun data are inspected | NEEDS_BIB_ENTRY: Pineau et al. 2021; NeurIPS checklist Q4/Q6 | ⚠️ | `docs/checkpoints/pre_run/preregistration.md` has H1-H8, Holm families, R1-R5, but frontmatter is `status: draft`, `registered_at`, `registered_git_sha`, `witnessed_by`, and OSF DOI are pending. Remediation: lock after advisor email, tag git, deposit OSF; cost 2-4h. | "The hypotheses and decision rules were written before the 16-cell rerun; the camera-ready will cite the lock SHA/OSF DOI once advisor witness is received." |
docs/checkpoints/pre_run/topvenue_constraints.md:34:| A2 | Release code, data, and exact reproduction instructions for main experimental claims | NeurIPS checklist Q4-Q5; NEEDS_BIB_ENTRY: Pineau et al. 2021 | ⚠️ | Code/scripts exist and `osf_lock_manifest.md` lists release artifacts, but `pre_rerun_audit.md §5.1.14` marks replication README as TBD and `§5.3.4` data-sharing policy as TBD. Remediation: write `README_REPLICATION.md` with clone→run→aggregate→figure commands and redaction policy; cost 1 day. | "We release the run manifest, environment snapshots, analysis scripts, and locked code; raw credentials/auth files are excluded and documented." |
docs/checkpoints/pre_run/topvenue_constraints.md:35:| A3 | Pin random seeds and disclose stochasticity limits | NeurIPS checklist Q6-Q7; NEEDS_BIB_ENTRY: Pineau et al. 2021 | ⚠️ | `configs/exp_v2_base.yaml` sets `seed: 42` and `runner/main.py::_seed_global_rng()` seeds Python/NumPy/Torch; `master_bug_catalog.md B-37` notes B0 proxy seeding remains best-effort and Anthropic-native seed support may be absent. Remediation: disclose B0 stochasticity and run B0 T=0 pilot sensitivity; cost 0.5-1 day. | "B1 is locally seeded/greedy; B0 uses best-effort API seed forwarding, so aggregate claims are reported as one controlled stochastic sample with bootstrap task uncertainty." |
docs/checkpoints/pre_run/topvenue_constraints.md:36:| A4 | Record software, hardware, model revisions, and evaluator code hashes per run | NeurIPS checklist Q8; NEEDS_BIB_ENTRY: Pineau et al. 2021 | ✓ | `scripts/provenance/snapshot_env.py` records Python, Torch, libraries, HF revision SHA, git commit/dirty state, GPU info, and `evaluator_code.combined_sha256`; `pre_rerun_audit.md §2.2` and `§3.3.6` require snapshots and rederive audit trails. | "Each paper-grade run carries machine/model/evaluator fingerprints sufficient to reconstruct scoring provenance." |
docs/checkpoints/pre_run/topvenue_constraints.md:38:| A6 | Provide complete hyperparameter/configuration disclosure | NeurIPS checklist Q6; NEEDS_BIB_ENTRY: Pineau et al. 2021 | ✓ | `configs/exp_v2_base.yaml` and per-site configs expose max steps, observation modes, backend temperature, cost rates, energy tracking, viewport, and seeds; `preregistration.md §4` locks mode definitions and inclusion rules. | "All experimental settings are in versioned YAML configs and summarized in the methods appendix." |
docs/checkpoints/pre_run/topvenue_constraints.md:42:| A10 | Include compute resources, total compute, and cost/carbon accounting | NeurIPS checklist Q8; `qiu2025modserve` | ⚠️ | `pre_rerun_audit.md §3.4` has per-cell GPU-hours, USD, and carbon tracking; cross-platform GPU power profile and Section 8 prose are TBD. Remediation: aggregate after rerun and add total compute table; cost 0.5 day. | "We report per-cell runtime, GPU type, API cost, and carbon estimates; cross-machine power variation is a limitation." |
docs/checkpoints/pre_run/topvenue_constraints.md:54:| B3 | Report effect sizes, not just p-values | NEEDS_BIB_ENTRY: Cohen 1988; NeurIPS checklist Q7 | ⚠️ | `preregistration.md §4` locks Cohen's h/d, but `pre_rerun_audit.md §4.1.4` says Cohen's h table is partial. Remediation: add h/d/CI columns to paper tables; cost 2-3h. | "We report pp lift plus standardized effect sizes with uncertainty for binary and continuous outcomes." |
docs/checkpoints/pre_run/topvenue_constraints.md:56:| B5 | Quantify annotation/label reliability for human-coded FP and failure modes | NEEDS_BIB_ENTRY: Cohen 1960 kappa; CONSORT-AI/Liu et al. 2020 | ❌ | `pre_rerun_audit.md §1.5` and `§4.3` mark FP labeling and failure-mode κ pilots as TBD. Remediation: two-rater 30-task pilot, κ≥0.7 target, adjudication log; cost 1 day. | "We will report κ for manually audited FP/failure-mode labels; until then qualitative mechanism claims remain caveated." |
docs/checkpoints/pre_run/topvenue_constraints.md:57:| B6 | Predefine missing-data/crashed-episode handling | CONSORT-AI/Liu et al. 2020; NeurIPS checklist Q7 | ✓ | Locked 2026-05-09 in `preregistration.md §4`: crashed episodes (uncaught exception / OOM / >30min timeout / browser crash) excluded from paired-N denominators, NOT imputed; not_logged_in / auth_drift after 3 retries excluded; missing artifacts excluded from per-step (not full-task); per-cell exclusion count + reason histogram in Appendix C. | "Listwise deletion only — crashes excluded from paired N, no success/failure imputation, exclusion histograms reported per cell." |
docs/checkpoints/pre_run/topvenue_constraints.md:59:| B8 | Analyze heterogeneity and avoid overclaiming pooled averages | NEEDS_BIB_ENTRY: DerSimonian & Laird 1986; NeurIPS checklist Q7 | ✓ | Locked 2026-05-09 in `preregistration.md §4`: random-effects DerSimonian-Laird primary, Cochran Q + I² + τ² reported, Higgins & Thompson 2002 I² thresholds (< 25% pooled / 25-50% both / 50-75% per-cell primary / > 75% no pool + heterogeneity-source decomposition by site/model/task-pool). Per-cell estimates always shown alongside pooled, so heterogeneity never averaged away. | "Random-effects with pre-specified I² thresholds — per-cell + heterogeneity-source decomposition reported when between-cell variance is high." |
docs/checkpoints/pre_run/topvenue_constraints.md:60:| B9 | Run power/MDE analysis for effect detectability | NeurIPS checklist Q7; CONSORT-AI/Liu et al. 2020 | ✓ | `docs/analysis/cross_sites/power_analysis.md` (consolidated 2026-05-09) reports observed-SR-conditioned MDE 4-7pp at 80% per-cell power across 3 sites + family-wise K-of-N power table at observed effect sizes (1-5pp) **<10%** → triggered methodological reframing: K-of-N rule retained as secondary transparency check, primary detection via B8 random-effects meta + TOST on N=910 pooled tasks. Bug fix: `power_analysis.py` stale interpretation block (claimed >0.95 K=12/16 family power at p_per=0.65; actual 0.289) corrected in same commit. Preregistration §4 updated with B9 row locking the corrected framing. | "Per-cell MDE 4-7pp at observed SR (8-15%); K-of-N rule documented as secondary transparency, primary detection via random-effects meta + pooled TOST." |
docs/checkpoints/pre_run/topvenue_constraints.md:67:| C2 | Reset environment/site state before each condition comparison | `zhou2024webarena`; `koh2024visualwebarena` | ⚠️ | `paper_drafts/section3_definition.md §3.5` states fresh reset; `pre_rerun_audit.md §2.1.1` enforces `RESET_BEFORE=1`, but `§3.2.3` state contamination snapshot is TBD. Remediation: implement pre/post `site_state_snapshot.sh`; cost 1 day. | "Each paper-grade condition starts from reset state; final artifacts will include state snapshots to verify no cross-condition contamination." |
docs/checkpoints/pre_run/topvenue_constraints.md:68:| C3 | Verify stateful website contamination, auth drift, and N/A tasks automatically | `zhou2024webarena`; NEEDS_BIB_ENTRY: Breck et al. 2017 | ⚠️ | Watchdog auth refresh and N/A filter exist; `pre_rerun_audit.md §3.2.9` has temporal auth/reset checks, but site-state snapshot remains TBD. Remediation: add strict site snapshot gate; cost 0.5-1 day. | "Temporal/auth drift checks are automated; state snapshotting is the remaining gate before declaring cells paper-grade." |
docs/checkpoints/pre_run/topvenue_constraints.md:76:| C11 | Provide task-level trace evidence for failure modes, not just aggregate SR | `koh2024visualwebarena`; `drouin2024workarena`; `furuta2024exposing` | ⚠️ | `pre_rerun_audit.md §4.4.b` lists `diag_pattern_match.py`, galleries, and pattern aggregators; per-cell failure-mode breakdown and κ are still TBD. Remediation: run post-rerun and add appendix table; cost 0.5-1 day. | "Trace-level diagnostics support the mechanism claims; automated pattern labels will be spot-checked against human labels." |
docs/checkpoints/pre_run/topvenue_constraints.md:77:| C12 | Report benchmark scope and avoid single-benchmark generalization claims | `deng2023mind2web`; `drouin2024workarena`; NEEDS_BIB_ENTRY: OSWorld/Xie et al. 2024 | ⚠️ | The intro and preregistration scope to VWA with planned WA expansion; `pre_rerun_audit.md §4.2.1` includes non-visual VWA+WA subset, but WA full expansion is untested. Remediation: scope claims to VWA or finish WA 480; cost 2-3 days. | "The main empirical claim is VWA-scoped; WebArena is reported only as planned/robustness unless completed." |
docs/checkpoints/pre_run/topvenue_constraints.md:78:| C13 | Disclose evaluator model drift when using LLM-as-judge (`ua_match`) | `zhou2024webarena`; NEEDS_BIB_ENTRY: Lipton & Steinhardt 2018 | ⚠️ | `master_bug_catalog.md B-20` documents GPT judge drift; `pre_rerun_audit.md §4.5.4` eval reproducibility spot-check is TBD. Remediation: rerun N=20 judge spot-check and pin prompt/model; cost 2-4h. | "LLM-judge tasks are isolated, prompt/model are pinned, and sensitivity to judge drift is disclosed." |
docs/checkpoints/pre_run/topvenue_constraints.md:85:| D1 | Claims must match evidence and scope; aspirational routing must not be stated as achieved | NeurIPS checklist Q1-Q2; NEEDS_BIB_ENTRY: Lipton & Steinhardt 2018 | ⚠️ | `paper_planning.md §1` labels the 4-fold drop-in hook provisional pending data; `preregistration.md R1-R5` maps framing to outcomes. Some older draft prose still says "hidden fourth routing arm" before 16-cell confirmation. Remediation: update intro after rerun based on R-rule; cost 2h. | "The final framing is data-conditional and tied to R1-R5; router deployment claims are deferred unless H7/H8 are locked and pass." |
docs/checkpoints/pre_run/topvenue_constraints.md:89:| D5 | Avoid cherry-picking task subsets or cells | NeurIPS checklist Q1/Q7; NEEDS_BIB_ENTRY: Lipton & Steinhardt 2018 | ⚠️ | `preregistration.md §4` defines cell inclusion and N floor; `pre_rerun_audit.md §4.8.2` counterfactual cell-removal stability is TBD. Remediation: run leave-one-cell-out decision test; cost 2h. | "All cells meeting locked criteria are included; leave-one-cell-out stability is reported as a falsification check." |
docs/checkpoints/pre_run/topvenue_constraints.md:90:| D6 | State failed assumptions and limitations in a dedicated section | NeurIPS checklist Q2; NEEDS_BIB_ENTRY: Lipton & Steinhardt 2018 | ⚠️ | `section4_limitations_disclosure.md` exists and `pre_rerun_audit.md §4.4` lists coverage, but final Section 8 prose is still TBD. Remediation: consolidate limitations in final paper; cost 0.5 day. | "The limitations appendix already enumerates evaluator, scaffold, numerical, and pre/post-fix threats." |
docs/checkpoints/pre_run/topvenue_constraints.md:94:## Category E — Web-Agent Specific (cross-bench / cost / latency)
docs/checkpoints/pre_run/topvenue_constraints.md:98:| E1 | Evaluate cost-quality tradeoffs, not success alone | `chen2023frugalgpt`; `ong2025routellm`; `qiu2025modserve` | ✓ | `preregistration.md H2` locks cost/latency/AUROC/drop-one drop-in claims; `paper_drafts/section3_definition.md §3.2` quantifies DOM vs P-text token and image-channel cost. | "The contribution is explicitly evaluated on success, drop-one complementarity, cost, latency, and routing signal quality." |
docs/checkpoints/pre_run/topvenue_constraints.md:99:| E2 | Include latency and serving-stage interpretation for multimodal modes | `qiu2025modserve`; `koh2024visualwebarena` | ⚠️ | Planning docs cite latency advantage; `pre_rerun_audit.md §3.4` tracks cost but cross-platform power/latency profiles are incomplete. Remediation: aggregate p50/p95 latency by mode and disclose hardware; cost 2-4h. | "Skipping image encoding removes the dominant multimodal serving stage; p50/p95 latency will be reported per cell." |
docs/checkpoints/pre_run/topvenue_constraints.md:100:| E3 | Use routing baselines that avoid train/test leakage | `chen2023frugalgpt`; `ong2025routellm`; `li2026avenirweb` | ⚠️ | `preregistration.md H7-H8` specifies 5-fold site-stratified CV and first-step leak-free router, but status is pending advisor lock and may be paper-2 deferred. Remediation: decide paper-1 vs paper-2 and implement split script; cost 1-2 days if included. | "Paper 1 does not require learned-router claims unless H7/H8 are locked; otherwise routing is oracle/characterization only." |
docs/checkpoints/pre_run/topvenue_constraints.md:102:| E5 | Test or clearly scope cross-benchmark generalization | `deng2023mind2web`; `drouin2024workarena`; NEEDS_BIB_ENTRY: GAIA/Mialon et al. 2023; NEEDS_BIB_ENTRY: OSWorld/Xie et al. 2024 | ❌ | Domain context says WA 480 is planned; current completed prose is VWA-centered. Remediation: either complete WA 480 non-visual subset or explicitly scope to VWA and move cross-bench to future work; cost 2-3 days or 1h to scope. | "The main paper is a VWA characterization; WA/Mind2Web/OSWorld generalization is future work unless reported in an appendix." |
docs/checkpoints/pre_run/topvenue_constraints.md:103:| E6 | Include model-scale contrast for agent behavior | `koh2024visualwebarena`; `drouin2024workarena`; `li2024effects` | ⚠️ | B0/B1 are included and `section1_intro.md` reports capability interaction, but B1 reddit phantom and B1 shop are still part of the 16-cell rerun plan. Remediation: finish 16-cell scope or weaken cross-capability claim; cost rerun-dependent. | "Capability contrast is limited to B0/B1 and interpreted as a scale probe, not a universal model-family law." |
docs/checkpoints/pre_run/topvenue_constraints.md:105:| E8 | Measure routing signal calibration/discriminability before claiming deployability | `chen2023frugalgpt`; `ong2025routellm`; NEEDS_BIB_ENTRY: Guo et al. 2017 calibration | ⚠️ | `preregistration.md H2(c)` locks AUROC threshold and signal universe; `pre_rerun_audit.md §4.2.12` has calibration diagnostics, but C4 DeLong/AUROC delta tests remain T1. Remediation: run AUROC delta CI/DeLong or bootstrap; cost 0.5 day. | "Routing-signal results are characterized with AUROC and not used as deployment proof without held-out router validation." |
docs/checkpoints/pre_run/topvenue_constraints.md:106:| E9 | Account for task visuality/site substrate when interpreting mode performance | `koh2024visualwebarena`; `guan2024hallusionbench`; `fu2024blink` | ⚠️ | `section1_intro.md` discusses classifieds visual-rich vs reddit text-dominated; `pre_rerun_audit.md §4.2.1` has non-visual subset robustness, but site×mode GLM is T1. Remediation: run mode×site interaction test; cost 0.5 day. | "Cross-site asymmetry is treated as a finding and a threat to generalization, not averaged away." |
docs/checkpoints/pre_run/topvenue_constraints.md:112:| F1 | Internal validity: isolate representation effects from scaffold/evaluator bugs | NEEDS_BIB_ENTRY: Cook & Campbell 1979; NEEDS_BIB_ENTRY: Sculley et al. 2014 | ⚠️ | Phase A fixes and bug catalog are strong, but `pre_rerun_audit.md §4.8.3` outlier task spot-check is TBD. Remediation: top/bottom task manual review; cost 2-3h. | "The rerun uses post-fix code; archived pre-fix data is appendix-only robustness." |
docs/checkpoints/pre_run/topvenue_constraints.md:113:| F2 | Construct validity: ensure metrics measure task success and not evaluator artifacts | `zhou2024webarena`; `koh2024visualwebarena`; NEEDS_BIB_ENTRY: Lipton & Steinhardt 2018 | ⚠️ | Adjusted SR and FP filters address evaluator artifacts; `pre_rerun_audit.md §4.5.4` LLM-judge reproducibility is TBD. Remediation: judge spot-check and per-evaluator-class sensitivity; cost 0.5 day. | "We report raw and adjusted success and isolate evaluator-class threats in limitations." |
docs/checkpoints/pre_run/topvenue_constraints.md:115:| F4 | Statistical conclusion validity: report uncertainty and sensitivity to thresholds | NeurIPS checklist Q7; NEEDS_BIB_ENTRY: Cook & Campbell 1979 | ⚠️ | CIs/meta-analysis exist; `pre_rerun_audit.md §4.8.1` K±1 and `§4.8.2` leave-one-cell-out are TBD. Remediation: add threshold-gradient and cell-removal table; cost 0.5 day. | "The decision rule is locked; sensitivity tables show how close the conclusion is to the threshold." |
docs/checkpoints/pre_run/topvenue_constraints.md:116:| F5 | Selection bias: curated mechanistic tasks must not be mistaken for population-wide behavior | NEEDS_BIB_ENTRY: Zhang et al. 2024 activation patching; NEEDS_BIB_ENTRY: Wang et al. 2023 | ⚠️ | `实验笔记.md §117.5` explicitly warns strong/reverse tier curation may correlate with effect size and launches 2x2 controls. Remediation: complete cells C/D and scope claims to curated tasks if controls fail; cost 0.5-1 day. | "Mechanistic claims are about curated mirage-disagreement tasks unless the 2x2 control supports broader bidirectionality." |
docs/checkpoints/pre_run/topvenue_constraints.md:117:| F6 | Instrumentation validity: verify cross-machine numerical consistency for mechanistic activations | NEEDS_BIB_ENTRY: Pineau et al. 2021; NEEDS_BIB_ENTRY: Zhang et al. 2024 activation patching | ⚠️ | `scripts/provenance/numerical_determinism_check.py` exists; `pre_rerun_audit.md §4.9.1` says comparison needs A100/Myriad. Remediation: run compare and report max hidden-state diff; cost 1-2h compute. | "Mechanistic layers are only interpreted after same-input hidden-state stability is verified within tolerance." |
docs/checkpoints/pre_run/topvenue_constraints.md:119:| F8 | Temporal validity: account for benchmark/software drift over time | `zhou2024webarena`; `drouin2024workarena`; NeurIPS checklist Q4 | ✓ | `pre_run/locked_versions.md` (created 2026-05-09) pins Playwright **1.58.0** + Chromium **revision 1208** (transitively pinned by Playwright). Plus VWA `832f037e` + HF model SHA `ebb281ec70b0...` + transformers 4.57.6 + torch (2.11.0+cu128 DGX / 2.1.0 Myriad with B-81 shims). T0/T1/T2/T3 evaluator-change protocol governs upgrades; same paper requires dual-report under post-lock T0 fix. | "Playwright 1.58.0 + Chromium 1208 + VWA 832f037e + HF Qwen3-VL-4B revision ebb281ec — all pinned in `pre_run/locked_versions.md` and auto-verified pre-launch." |
docs/checkpoints/pre_run/topvenue_constraints.md:131:| G7 | Control for task-selection bias with crossed task-tier × direction design | NEEDS_BIB_ENTRY: Zhang et al. 2024; NEEDS_BIB_ENTRY: Conmy et al. 2023 | ✓ | 4-cell 2x2 design completed 2026-05-09: A (fwd×strong N=24, Holm L17 p=0.011 \*\*) + B (rev×reverse N=15, Holm L11+L17) + C (fwd×reverse N=15, Δ=-0.10 weaker) + D (rev×strong N=24, Holm L11+L17 p=0.006/0.008 \*\*). Cross-cell Welch p=0.509-1.000 (NS) — magnitudes statistically indistinguishable across direction × tier; reverse direction Δ=-0.193 identical on both tiers (p=1.000), forward Δ=-0.143/-0.103 not Welch-distinguishable across tiers (p=0.577). Selection bias rejected; mechanism claim universal pending cross-site (cells F/G). | "The crossed 2x2 design rules out selection-bias artifact: reverse direction produces identical magnitude on both task tiers (Welch p=1.000), forward direction is Welch-indistinguishable across tiers (p=0.577); only Holm-significance differs due to N power asymmetry." |
docs/checkpoints/pre_run/topvenue_constraints.md:134:| G10 | Use mechanistic metrics tied to behavior/action tokens | NEEDS_BIB_ENTRY: Wang et al. 2023; NEEDS_BIB_ENTRY: Geva et al. 2021 | ⚠️ | Current Stage 2 metrics are continuation overlap and LD to target; direct action-token probability/action-switch tests are not clearly present. Remediation: add action-token/logit-diff metric where outputs contain actions; cost 0.5-1 day. | "Continuation metrics capture the mirage reasoning state; action-token metrics are added where the trace format makes them well-defined." |
docs/checkpoints/pre_run/topvenue_constraints.md:141:| H1 | Maintain a dedicated limitations section with strong assumptions and robustness failures | NeurIPS checklist Q2; NEEDS_BIB_ENTRY: Lipton & Steinhardt 2018 | ⚠️ | `section4_limitations_disclosure.md` exists and `pre_rerun_audit.md §4.4` enumerates limitations; final Section 8 prose is not complete. Remediation: consolidate into final limitations section; cost 0.5 day. | "Limitations are not hidden in appendices; they are surfaced alongside the main empirical claims." |
docs/checkpoints/pre_run/topvenue_constraints.md:143:| H3 | Scope deployment and safety claims for autonomous web agents | NeurIPS checklist Q10; `wei2026clawsafety`; `drouin2024workarena` | ✓ | `pre_run/ethics_license_coi_statements.md` (created 2026-05-09) covers 4-point misuse/safety scope: (a) lower automation cost broadens deployment surface (legitimate + misuse), (b) visual hijack patterns + adversarial-prompt risk, (c) benchmark contamination risk, (d) cross-bench generalization not established → CLAW-Safety review required for deployment. | "The paper is a scientific characterization, not deployment-ready. Misuse/safety scope (autonomous web agents lower automation cost, surface visual-hijack failure modes, accelerate benchmark contamination) is explicit in §8; safety-critical deployment requires domain-specific evaluation." |
docs/checkpoints/_status/section7_generalization.md:7:owner: 我 + DGX
docs/analysis/phantom_paper/B0_dom_shopping_diagnostic.md:96:| 121 | exact item cost | "SanDisk Extreme PRO 512GB CFast 2.0" | Exact product name from image yields price answer; 4 steps. |
docs/analysis/cross_sites/pilot_t0_decision_final.md:45:- **B1 (qwen3vl_agent) torch CUDA non-determinism**: even with `torch.manual_seed`, some CUDA ops (e.g., `cudnn.benchmark=True` defaults) are non-deterministic. We do not force `cudnn.deterministic=True` because of perf cost.
docs/analysis/cross_sites/pilot_t0_decision_final.md:64:Pilot wave-1 (launched 11:19 BST) was destroyed at 12:01-12:03 BST by another Claude session (Myriad HPC config work) that misinterpreted "busy:1 free wait" log entries as "stuck" and ran `clear_tasks.py --force`. Wave-2 (launched 12:13 BST) was a clean re-launch after sites recovered from a `wsl --shutdown` event. The infrastructure was robust enough that cleanup + re-launch took ~10 min and no Phase A schedule slip.
docs/analysis/vwa_classifieds/B0_Vision_digest.md:122:| Vision vs SoM total_cost | **2.9e-9** | Vision 更便宜 ★★★ |
docs/analysis/vwa_classifieds/B0_Vision_digest.md:123:| Vision vs DOM total_cost | **4.5e-12** | Vision 更便宜 ★★★ |
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:24:| Efficiency 3a Token cost | ✅ condition_summary_v2.json | ✅ fig3a_token_cost_intra_baseline.png (bundled with 3c) |
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:26:| Efficiency 3d Cost-SR Pareto | ✅ paper_cost_usd | ✅ fig3d_cost_sr_frontier.png |
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:40:- Fallback: live compute from `condition_summary_v2.json` per-cell if pre-aggregate missing
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:101:**Sub-code**: Efficiency 3c — Per-step latency separated from cost (paper §1 hook 4-fold drop-in property (b))
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:107:**This is paper §1 hook visualization** — currently latency only shown bundled with cost in fig3a. Dedicated figure single-message-clear for advisor sync + paper writing.
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:115:In `Makefile`, the `_figures` internal target (Phase 2 added) needs 6 new figure script invocations appended. Insert after existing fig3d_cost_sr_frontier line:
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:119:	# ... existing fig0c/0d/0e/0f/0g/1ab/1c/2/3a/3d/3_regional_carbon/_capability_b0_b1 ...
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:134:Each figure script must follow these conventions (see `fig0c_drop_one_oracle.py` / `fig3a_token_cost_intra_baseline.py` as templates):
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:197:- `scripts/analysis/figures/fig3a_token_cost_intra_baseline.py` (canonical Efficiency dimension figure template)
docs/checkpoints/codex_prompts/fix_figures_and_layer0_aggregator.md:8:2. ⚠️ **fig3d_cost_sr_frontier**: 漏 Phantom-SoM —— Layer 3d cost-SR Pareto 上 paper headline mode 缺席
docs/checkpoints/codex_prompts/fix_figures_and_layer0_aggregator.md:36:### 1. 修 fig3d_cost_sr_frontier.py（Layer 3d）
docs/checkpoints/codex_prompts/fix_figures_and_layer0_aggregator.md:38:**当前问题**：`scripts/analysis/figures/fig3d_cost_sr_frontier.py` 的 `ConditionSpec` 列表只有 4 mode（DOM / SoM / Vision / Phantom-DOM），完全没 Phantom-SoM。
docs/checkpoints/codex_prompts/fix_figures_and_layer0_aggregator.md:47:**验证**：跑完 fig7 应该有 5 个 mode marker（颜色匹配 fig3 一致性: DOM #4c78a8, SoM #f58518, Vision #54a24b, Phantom-SoM #b279a2, Phantom-DOM #e45756）。Layer 3d cost-SR Pareto 应该显示 P-SoM 落在"DOM-cost / DOM-SoM 中间 SR"位置（4-fold drop-in property (a) cost ≈ DOM 的核心 visual evidence）。
docs/checkpoints/codex_prompts/fix_figures_and_layer0_aggregator.md:158:- `results/phantom_paper/figures/fig3d_cost_sr_frontier.png` 重新生成，含 5 mode marker
docs/analysis/vwa_classifieds/B0_DOM_digest.md:153:| cost/ep | $0.0427 | $0.0415 | **$0.0248** |
docs/checkpoints/codex_prompts/refactor_phase1_run_registry.md:202:- `scripts/analysis/aggregate_cost_electricity.py` — same pattern
docs/checkpoints/codex_prompts/refactor_phase1_run_registry.md:213:- `scripts/analysis/figures/fig3a_token_cost_intra_baseline.py`
docs/checkpoints/codex_prompts/refactor_phase1_run_registry.md:214:- `scripts/analysis/figures/fig3d_cost_sr_frontier.py`
docs/checkpoints/paper_drafts/section4_empirical_findings.md:19:The first three arms are the original Phase 1 representation baselines. Phantom-SoM is the new ablation arm. P-text is a prompt-family control: it receives the same marks-text-only observation as Phantom-SoM but uses the DOM prompt. We report all five modes for descriptive SR, cost, and latency. For the main routing-value claim, we keep the primary drop-one oracle on the four-arm comparison used throughout the paper: DOM, SoM, Vision, and Phantom-SoM.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:25:The single-mode success rates show a site-modulated effect. On classifieds, full SoM remains the strongest individual representation. On reddit, Phantom-SoM is at least competitive with the strongest baselines, while using no image input. The table reports adjusted SR, because Figures 1, 2, 7, and 8 use episode-level `adjusted_success` for the paper comparisons. The latency column is p95 step latency from `condition_summary_v2.json`; cost is average total cost per task.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:27:| Site | Arm | Adjusted SR | Avg cost | p95 step latency | Metric |
docs/checkpoints/paper_drafts/section4_empirical_findings.md:46:The cost and latency columns make the routing tradeoff concrete. On classifieds, Phantom-SoM's average cost is effectively in the same band as DOM and SoM (**$0.044 vs $0.043 vs $0.041**), but its p95 step latency is much lower than full SoM (**18.2s vs 74.0s**, roughly 4x faster). On reddit, Phantom-SoM is the cheapest of the main text/SoM-style arms (**$0.038 vs SoM $0.041 vs DOM $0.052**) and remains faster at p95 step latency than full SoM (**51.4s vs 58.9s**). These numbers support the cost-aware routing interpretation in Figures 7 and 9 without requiring Phantom-SoM to win every site.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:65:The main empirical claim is therefore not that Phantom-SoM dominates the other modes. It does not. The claim is that it is an **independent routing arm**: it opens a distinct task pool at text-only cost, with the strongest relative benefit on the text-dominated reddit site and a clear visual-grounding disadvantage on classifieds.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:69:The five-mode result raises a confound: is Phantom-SoM useful because of the `[SOM_MARKS]` text representation, or because the SoM prompt changes the agent's confidence and behavior even without an image? P-text separates these factors. The full clean P-text runs are reported above for SR, cost, and latency; for behavioral mechanism, we use the verified same-task reddit subset of **N=48**, where all four cells of the prompt-by-representation ablation were manually checked.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:88:These findings explain why Phantom-SoM can be valuable despite not winning every single-mode comparison. Routing benefits depend on complementarity, not only average SR. A flat marks list can be worse for tasks that need hierarchy or visual layout, yet better for tasks where the same hierarchy induces over-searching. The practical implication is a cost-aware cascade: try cheap text representations first, use behavioral signals to detect when their exploration is unproductive, and escalate to full SoM when visual grounding is likely to matter.
docs/analysis/vwa_reddit/B0_B1_findings.md:12:> - **B1 数据非最终**：DGX 共享 GPU 同时跑多实例时存在 VRAM/算力争抢，B1 latency 数字受污染；最终 latency 待 Myriad HPC 上线后用独占 GPU 重跑。SR/cost/oracle 数字不受影响（推理逻辑 deterministic）。
docs/analysis/vwa_reddit/B0_B1_findings.md:45:> 注：B0 成本为 API 实际调用费用；B1 成本为本地 GPU 推理的 API 等价估算。两者成本体系不同，直接比较需谨慎。
docs/analysis/vwa_reddit/B0_B1_findings.md:266:8. **设计不对称需注意**：B0/B1 存在温度/max_tokens/scroll 等不对称（同 Classifieds），SR 差异无法完全归因于模型规模。**B1 latency 数字未最终化**（DGX 共享 GPU 争抢），Myriad HPC 上线后用独占 GPU 重跑。
docs/analysis/vwa_reddit/B0_B1_findings.md:275:- **成本比较**：B0（API 费用）与 B1（本地推理 GPU 等价成本）成本体系不同
docs/analysis/vwa_reddit/B0_B1_findings.md:383:- **B1 reddit Phantom-SoM (无图) 预测 ≥ Full SoM**（去掉 hijack +2pp + 省 50% token cost）
docs/analysis/vwa_reddit/B0_B1_findings.md:384:- **B0 reddit Phantom-SoM ≈ Full SoM**（gap +0.5pp 可忽略）→ **cost-saving win**（保 SR + 省 50% cost）
docs/analysis/vwa_reddit/B0_B1_findings.md:484:| **Phantom-SoM cost saving motivation** | 独立 |
docs/analysis/vwa_reddit/B0_B1_findings.md:494:- SoM 含图片 token (~50% input cost)，DOM/Phantom 无
docs/analysis/vwa_reddit/B0_B1_findings.md:495:- 同等 SR 下 → cost-saving win（universal）
docs/analysis/vwa_reddit/B0_B1_findings.md:497:- → **Phantom-SoM 是 universal cost-aware tool，不是反转 magnitude 的 hostage**
docs/checkpoints/pre_run/negative_results_registry.md:21:| **1** | 2026-04-28 | **Phantom-DOM 18 modes scope** (full factorial expansion across image / SoM / prompt axes) | 5-mode scope (DOM / SoM / Vision / phantom_som / phantom_dom) | 18 modes exceeded paper-grade focus; 13 ablation modes added marginal information at high compute cost | §3 paper-grade scope explicit; pre-rerun audit §1.1 enforces |
docs/checkpoints/pre_run/negative_results_registry.md:41:| C1 | 2026-04-26 | Phantom-SoM 4-fold drop-in property (cost ≈ DOM / latency ~50% / signal AUROC ≥ baseline / drop-one 1.7-3.8pp) | B0 reddit Phase A archived data + drop-one sig | Paper hook §1 (provisional pending 16-cell rerun) |
docs/checkpoints/codex_prompts/refactor_phase2_make_analysis_consolidation.md:22:aggregate-cross-site / aggregate-cost-electricity / aggregate-sr-fp / phantom-lift / routing-auroc
docs/checkpoints/codex_prompts/refactor_phase2_make_analysis_consolidation.md:84:	$(MAKE) aggregate-cost-electricity
docs/checkpoints/codex_prompts/refactor_phase2_make_analysis_consolidation.md:101:	$(PYTHON) scripts/analysis/figures/fig3a_token_cost_intra_baseline.py
docs/checkpoints/codex_prompts/refactor_phase2_make_analysis_consolidation.md:102:	$(PYTHON) scripts/analysis/figures/fig3d_cost_sr_frontier.py
docs/checkpoints/codex_prompts/refactor_phase2_make_analysis_consolidation.md:103:	$(PYTHON) scripts/analysis/figures/fig3_regional_carbon.py
docs/checkpoints/codex_prompts/refactor_phase2_make_analysis_consolidation.md:148:	@echo "  Per-run / per-cell:          make analyze RUN=<dir>"
docs/checkpoints/codex_prompts/refactor_phase2_make_analysis_consolidation.md:155:	@echo "    summary-collect / aggregate-cost-electricity / analyze-mechanism"
docs/checkpoints/codex_prompts/site_mechanism_dictionary.md:12:- `cost_per_mode.{json,md}` (Layer 3a/d cost classes)
docs/checkpoints/codex_prompts/site_mechanism_dictionary.md:119:   - `docs/analysis/cross_sites/cost_per_mode.json` (Layer 3a/d)
docs/checkpoints/pre_run/model_card.md:73:| Decoding mode | **Greedy** (`do_sample=False`), bitwise deterministic per (HF SHA + seed + GPU class) |
docs/checkpoints/pre_run/model_card.md:77:| Numerical determinism | bitwise across **A100 80GB / V100 / GB10 within tolerance** (≤1e-6 hidden-state delta per `numerical_determinism_check.py` audit F6). Math SDPA backend forced for cross-architecture portability per B-81h fix. |
docs/checkpoints/pre_run/model_card.md:90:- **GPU dependency**: bf16 inference requires sm_70+ (V100 OK, T4 OK with reduced batch); CPU inference too slow for paper-scale runs
docs/checkpoints/pre_run/model_card.md:91:- **Cross-machine numerical determinism caveat**: minor (<1e-6) hidden-state drift across V100 vs A100 due to cuDNN kernel selection — bounded under tolerance but worth noting (audit F6)
docs/checkpoints/pre_run/model_card.md:104:| Latency | Network-bound (~5-20s) | GPU-bound (~2-10s on A100) |
docs/analysis/vwa_reddit/B1_findings.md:14:> - **B1 数据非最终**：DGX 共享 GPU 同时跑多实例时存在 VRAM/算力争抢，B1 latency 数字受污染；最终 latency 待 Myriad HPC 上线后用独占 GPU 重跑。SR/cost/oracle 数字不受影响。
docs/analysis/vwa_reddit/B1_findings.md:85:| Vision vs SoM | total_cost | **1.7e-24** | Vision 更便宜 ★★★ |
docs/analysis/vwa_reddit/B1_findings.md:87:| Vision vs DOM | total_cost | **8.6e-29** | Vision 更便宜 ★★★ |
docs/analysis/vwa_reddit/B1_findings.md:89:| SoM vs DOM | total_cost | **1.2e-3** | SoM 更便宜 ★★★ |
docs/checkpoints/codex_prompts/swatch_form_value_changed_audit.md:19:URL: `http://100.95.81.103:7770/anime-throw-blanket-flannel-fleece-blanket-super-soft-cozy-warm-for-bedding-couch-sofa-plush-blanket-for-kids-adults-gift-50-x40.html` (DGX-reachable shopping site).
docs/checkpoints/codex_prompts/swatch_form_value_changed_audit.md:77:- The shopping site is shared (DGX accesses via Tailscale `100.95.81.103:7770`). Don't run destructive operations (add to cart, place order). Read-only HTML/JS probes only.
docs/checkpoints/codex_prompts/swatch_form_value_changed_audit.md:78:- Run on DGX Spark (`spark-9ea3`). Use `.venv/bin/python3` and ensure `PYTORCH_NVML_BASED_CUDA_CHECK=1` (though Playwright probe shouldn't need GPU).
docs/checkpoints/codex_prompts/refactor_layered_analysis.md:28:Layer 3  Efficiency        cost / latency / carbon
docs/checkpoints/codex_prompts/refactor_layered_analysis.md:29:  3a Token cost           condition_summary_v2.json
docs/checkpoints/codex_prompts/refactor_layered_analysis.md:32:  3d B0 vs B1 cost gap    fig3d_cost_sr_frontier
docs/checkpoints/codex_prompts/refactor_layered_analysis.md:71:# Layer 3 — Efficiency (cost / latency)
docs/checkpoints/codex_prompts/refactor_layered_analysis.md:74:    $(PYTHON) scripts/analysis/figures/fig3d_cost_sr_frontier.py        # 3d
docs/checkpoints/codex_prompts/refactor_layered_analysis.md:155:- `figures/fig3d_cost_sr_frontier.py` → Layer 3d
docs/analysis/vwa_classifieds/B0_B1_findings.md:12:> **B1 数据非最终**：DGX 共享 GPU 争抢污染 latency，待 Myriad HPC 独占 GPU 重跑。SR/cost/oracle 数字不受影响。
docs/analysis/vwa_classifieds/B0_B1_findings.md:47:> 注：B0 成本为 API 实际调用费用；B1 成本为本地 GPU 推理的 API 等价估算（基于 token 量 x API 定价）。两者成本体系不同，直接比较需谨慎。
docs/analysis/vwa_classifieds/B0_B1_findings.md:276:- **成本比较**：B0（API 费用）与 B1（本地推理 GPU 等价成本）成本体系不同，直接数值比较需注意
docs/analysis/vwa_classifieds/B0_B1_findings.md:365:| V1 省 token cost | ✅ | ✅ |
docs/analysis/vwa_classifieds/B0_B1_findings.md:371:- B0 classifieds: Phantom < Full SoM **(-7pp)** → 截图有真实视觉收益，省 cost 损失大
docs/analysis/vwa_classifieds/B0_B1_findings.md:377:- B/D subset: trade-off based on cost sensitivity
docs/analysis/vwa_classifieds/B0_B1_findings.md:465:| **Phantom-SoM cost saving motivation** | 独立 |
docs/analysis/vwa_classifieds/B0_B1_findings.md:475:- SoM 含图片 token (~50% input cost)，DOM/Phantom 无
docs/analysis/vwa_classifieds/B0_B1_findings.md:476:- 同等 SR 下 → cost-saving win（universal）
docs/analysis/vwa_classifieds/B0_B1_findings.md:478:- → **Phantom-SoM 是 universal cost-aware tool，不是反转 magnitude 的 hostage**
docs/checkpoints/paper_drafts/section2_background.md:5:Modern web agents differ less in the browser actions they expose than in the observation representation they give to the language model. Text-only agents typically serialize the Document Object Model or Accessibility Tree (AXTree) into a hierarchical text observation. WebArena uses this style of realistic browser environment to evaluate language-guided agents on shopping, forum, map, and software-development tasks \citep{zhou2024webarena}. Mind2Web similarly frames web interaction as selecting actions from structured page elements collected across real websites \citep{deng2023mind2web}. This line of work makes DOM-derived text the default low-cost representation: it is cheap, symbolic, and compatible with language-only models, but it can be verbose and blind to visual appearance.
docs/checkpoints/paper_drafts/section2_background.md:13:Routing has become a standard response to heterogeneous cost and capability. FrugalGPT frames inference as a cascade over multiple LLM APIs, learning when cheaper models can answer and when to escalate to stronger models [Chen et al. 2023]. RouteLLM similarly learns routers from preference data to choose between weaker and stronger LLMs under cost-quality tradeoffs [Ong et al. 2025]. These systems are important precedents for cost-aware inference, but their arms are models. The input representation is usually fixed while the backend model changes.
docs/checkpoints/paper_drafts/section2_background.md:33:Multimodal inference adds a second cost source: visual encoding. Image inputs increase prompt-processing time, memory pressure, and time-to-first-token. ModServe characterizes large multimodal model serving and shows that multimodal workloads have heterogeneous stages and resource requirements, motivating modality- and stage-aware resource disaggregation [Qiu et al. 2025]. In web agents, full SoM therefore has two costs: it prepares a marked screenshot and it sends image tokens to the model.
docs/checkpoints/paper_drafts/section2_background.md:35:Phantom-SoM explores a different kind of efficiency. It is not text pruning and it is not image scheduling. It is text reformatting. The `[SOM_MARKS]` list can be generated from the same browser/AXTree metadata already available to the agent, then sent without the marked screenshot. This removes image-token cost while preserving a discrete element index. In our runs the text observation is comparable in token length to the corresponding AXTree (within ±7% on reddit and classifieds, holding the system prompt fixed); the difference is in structure — flat indexed list versus nested hierarchy with url/tab metadata — rather than in length. The open question is whether such a representation is only a structural rewrite of DOM, or whether its format creates a distinct success pool. Our empirical sections answer the latter.
docs/checkpoints/paper_drafts/section2_background.md:37:This matters because many cost reductions trade away information: smaller models, shorter context windows, lower image resolution, or fewer retrieved lines. Phantom-SoM instead tests whether a cheap re-arrangement of already available text can expose a different reasoning path. If it succeeds on tasks missed by DOM, the gain is not merely compression; it is complementarity. That is why Section 4 reports both single-mode success and drop-one oracle value rather than treating token savings alone as the contribution.
docs/checkpoints/codex_prompts/section8_limitations_consolidation.md:48:- Power analysis (B9) shows minimum-detectable effect for cells with N=15 ~0.65 Cohen's d at α=0.05 — not powered for small mid-layer effects.
docs/checkpoints/codex_prompts/section8_limitations_consolidation.md:54:### 8.7 Compute, cost, and sustainability bounds (~80 words)
docs/checkpoints/codex_prompts/section8_limitations_consolidation.md:55:- Per-cell GPU-hours / USD / kg-CO₂ table per A10 reviewer-rebuttal. Cross-machine power profile (DGX Spark vs Myriad V100) is one limitation source for absolute energy numbers; relative comparisons within the same cell are unaffected.
docs/analysis/vwa_classifieds/B1_findings.md:13:> **B1 数据非最终**：DGX 共享 GPU 同时跑多实例时存在 VRAM/算力争抢，B1 latency 数字受污染；最终 latency 待 Myriad HPC 上线后用独占 GPU 重跑。SR/cost/oracle 数字不受影响。
docs/analysis/vwa_classifieds/B1_findings.md:71:| 对比 | cost p | latency p |
docs/checkpoints/paper_drafts/section3_definition.md:33:Phantom-SoM is a cost intervention, and the structure of the saving is best stated relative to two different baselines.
docs/checkpoints/paper_drafts/section3_definition.md:35:**Relative to DOM**, Phantom-SoM is essentially free. The `[SOM_MARKS]` block is produced by a regex filter over the VisualWebArena accessibility-tree text that the DOM baseline already consumes. VWA serializes interactive elements with bracketed numeric IDs of the form `[N] role 'label'`; in our implementation `_extract_text_marks` (see `p79/experiment/som.py`) walks `obs_text` line by line, keeps the lines that match `\[\d+\]`, and returns `(id, label)` pairs that are wrapped in a `[SOM_MARKS] ... [/SOM_MARKS]` block. There is no bounding-box lookup and no image work in this path; bounding boxes are only used by full SoM when drawing numeric labels onto the screenshot. Empirically this leaves text length roughly unchanged: holding the system prompt fixed at the DOM family, median total input is 3437 tokens for DOM versus 3661 for P-text on reddit, and 3008 versus 2948 on classifieds — within ±7% on both sites. The two formats see the same accessibility content; what differs is the surface form (flat indexed list versus nested hierarchy with url/tab metadata). We treat this as a representation property and study its behavioral effect mechanistically in Section 5; for cost accounting the implication is that switching DOM → Phantom-SoM at deployment time costs at most a regex pass over the same observation.
docs/checkpoints/paper_drafts/section3_definition.md:37:**Relative to full SoM**, Phantom-SoM saves two real layers of cost. (i) The on-server annotation step that draws numeric labels onto the page screenshot is unique to full SoM and is omitted in a Phantom-SoM deployment; in our research code we retain the marked image on disk for debugging, which is why both modes report ~30 ms median obs-prepare latency, but a production variant skips the draw entirely and recovers roughly 30 ms and on the order of $2e-5 per step. (ii) The marked screenshot is no longer encoded as image tokens at inference, removing the visual-encoding stage. Comparing step-level `tokens.input` medians between full SoM and P-text gives a same-prompt image-channel estimate of 733 input tokens per step on reddit (SoM 4275 versus P-text 3542; P-text partial live run, 145 episodes) and 1064 on classifieds (4034.5 versus 2970.5; 234 episodes). We attribute this median gap to the marked screenshot under our backend tokenization. These are the tokens that drive prompt-processing time, memory pressure, and time-to-first-token in multimodal serving (see Section 2.4); skipping them is the dominant component of the cost difference between full SoM and Phantom-SoM.
docs/checkpoints/paper_drafts/section3_definition.md:39:The combined picture is that Phantom-SoM sits at roughly DOM cost (its observation is a text filter of the same AXTree) while replacing the visual-evidence half of SoM with nothing at all. This is also a deployment-level claim, not only an analytical one: an existing full-SoM agent can be converted into a Phantom-SoM agent by changing only what the server forwards to the model — keep the `[SOM_MARKS]` text that is already being produced from the accessibility tree, stop drawing labels onto the screenshot, and stop attaching the marked image to the inference request. The model interface, the prompt, the action schema, and the evaluator are unchanged. There is no retraining, no new data path, and no marks-side prompt edit; the only mutation is on the backend annotation pipeline, after the AXTree filter and before the model call. We use this property in Section 4 to interpret cost-versus-success comparisons as deployment-time tradeoffs rather than research-only configurations, and in Section 5 to argue that Phantom-SoM's behavior is a property of the format the model already saw inside SoM, not an emergent capability that requires new infrastructure.
docs/checkpoints/paper_drafts/section3_definition.md:67:The fourth cell — AXTree observation paired with the SoM prompt — is intentionally excluded from Paper 1 because it is not a self-consistent design point. The SoM system prompt instructs the agent to interact via `[SOM_MARKS]` IDs (e.g. `click [42]` referring to the SoM-marked element 42), but AXTree text uses an independent accessibility-tree ID space; an action like `click [42]` becomes parsing-ambiguous when the two ID systems do not match. This hybrid mode (i) has no clean LLM mechanism, (ii) confounds the prompt-effect ablation with mismatched-ID parsing failure, and (iii) does not reduce token cost relative to P-text. We treat the 5-mode set (DOM, P-text, Phantom-SoM, full SoM, plus Vision as a separate screenshot-only arm) as the diagonal axis-by-axis path through the 2×2×2 (text-payload-structure × prompt × image) design cube; the four mismatched-prompt-representation hybrids are excluded for the same reason.
docs/checkpoints/paper_drafts/section3_definition.md:76:The 2x2 is not a routing policy by itself. It is a causal scaffold for Section 5: text-payload structure shapes exploration, while prompt wording tunes commitment confidence. Section 6 promotes this scaffold to a token-monotonic cascade — DOM → P-text (axis 1, text-payload swap, no token increase) → Phantom-SoM (axis 2, system-prompt swap, no data-token increase) → full SoM (axis 3, image embedding cost) — so a routing trigger never has to "add then remove" tokens.
docs/analysis/phantom_paper/som_vs_phantom_som_diagnostic.md:38:On classifieds, image-on SoM is both more successful and more efficient in steps. It needs fewer average steps (8.60 versus 12.05), fewer search-loop episodes (64.5% versus 76.9%), and fewer unique URLs, consistent with the screenshot resolving listing structure or visual object identity before the model over-searches. The latency tradeoff is large: SoM's p95 step latency is 74.0s versus 18.2s for Phantom-SoM, despite similar per-episode dollar cost ($0.0415 versus $0.0441).
docs/analysis/phantom_paper/som_vs_phantom_som_diagnostic.md:197:5. Cost and latency are part of the deployment mechanism. SoM and Phantom-SoM have similar average dollar cost in classifieds, but Phantom-SoM has much lower p95 step latency (18.2s versus 74.0s). On reddit, Phantom-SoM is both slightly cheaper ($0.0381 versus $0.0409) and higher SR, although the observed 3.33 pp gap should still be interpreted conservatively given run-to-run variance.
docs/analysis/phantom_paper/som_vs_phantom_som_diagnostic.md:201:> Marked screenshots are not a monotone visual upgrade. They provide spatial grounding, visual context, state recognition, and element disambiguation, which dominate on visually rich classifieds tasks. The same channel can also create false visual confidence, wrong-target attention, and direct-image loops, which become more costly on text-heavy reddit tasks. Phantom-SoM isolates this image channel: holding the SoM prompt fixed, removing the image lowers classifieds adjusted SR from 21.37% to 14.53% but raises reddit adjusted SR from 10.48% to 13.81%. The image effect is therefore site- and task-modulated, not a single visual-hijack axis.
docs/checkpoints/paper_drafts/section1_intro.md:5:This paper questions that bundling assumption as an experimental object. We characterize the **phantom routing space**: configurations on the "skip annotated image" boundary that retain some SoM-derived textual or prompt structure while removing the image. Its deployment-relevant representative is **Phantom-SoM**: the agent receives the SoM prompt and the `[SOM_MARKS]` textual element list, but no image. The structural controls are **P-text** (the `[SOM_MARKS]` text under the DOM prompt) and **P-prompt** (the SoM prompt over AXTree text). At the start of this project, Phantom-SoM looked like a broken ablation. The natural expectation was that removing the marked screenshot would collapse SoM into either a weak DOM surrogate or a nonsensical configuration: the prompt still suggests visual marks, but the visual substrate is absent. The data reject that expectation. Phantom-SoM solves tasks that DOM, full SoM, and Vision all miss, and on B0 reddit it matches or modestly exceeds full SoM by adjusted SR (**13.81% vs 10.48%, N=210**; the gap is within 2σ under the run-to-run variability we observe in same-condition repeats), while avoiding image-token cost. On classifieds, full SoM remains clearly stronger (**21.37% vs Phantom-SoM 14.53%, N=234**), the expected sanity check when marked screenshots carry real visual information.
docs/checkpoints/paper_drafts/section1_intro.md:7:Our first contribution is a controlled scientific evaluation of this phantom boundary. Across completed B0 VisualWebArena classifieds and reddit runs, we compare DOM, full SoM, Vision, and Phantom-SoM on the same task sets (**N=234 classifieds; N=210 reddit; same-task adjusted SR**) and use the P-text/P-prompt controls to test whether the effect collapses to one prompt trick or one text-format swap. Phantom-SoM is not the best single arm on every site, and we do not claim that it replaces full SoM. Its value is complementarity. Its task-success pool has low overlap with the established modes, with Jaccard similarity in the roughly **0.29-0.49** range against other arms, and its removal reduces the oracle. In drop-one oracle analysis, Phantom-SoM contributes **3.33 percentage points** of incremental oracle value on reddit (comparable to full SoM at **1.90 pp**) and **2.56 percentage points** on classifieds, placing Phantom-SoM consistently within the top routing-value arms despite using no image. Crucially, the cost of obtaining this configuration is essentially the cost of the DOM baseline: the `[SOM_MARKS]` block is produced by a regex pass over the same accessibility-tree text the DOM agent already consumes (interactive elements come pre-numbered as `[N] role 'label'`), so a deployment that can run DOM can run Phantom-SoM by changing what it forwards to the model: no bounding-box pipeline, no marked image, no extra inference modality. We therefore preserve the empirical **4-fold drop-in property** as the paper's practical finding: cost approximately DOM, lower image-stage latency, usable routing signal, and positive drop-one oracle value.
docs/checkpoints/pre_run/locked_versions.md:89:| Python | 3.12 (DGX) / 3.9.6 (Myriad) / 3.11 (Condense, future) | Multi-environment, behavior parity verified per audit F6 |
docs/checkpoints/pre_run/locked_versions.md:90:| torch | 2.11.0+cu128 (DGX) / 2.1.0 (Myriad module) | `sitecustomize.py` shim for Myriad's torch 2.1 missing `register_pytree_node` + `torch.compiler.is_compiling` (B-81b, B-81f) |
docs/checkpoints/pre_run/locked_versions.md:92:| numpy | <2 | torch 2.1 binary compiled against NumPy 1.x (Myriad B-81 stack) |
docs/checkpoints/pre_run/locked_versions.md:93:| urllib3 | <2 | RHEL 7 OpenSSL 1.0.2k incompatible with v2 (Myriad B-81c) |
docs/checkpoints/pre_run/locked_versions.md:94:| scipy | 1.17.1 (DGX) | Stat tests (Holm, Welch, Wilcoxon) |
docs/checkpoints/pre_run/locked_versions.md:145:  HF Qwen3-VL-4B `ebb281ec70b0...`, transformers 4.57.6 + Myriad shims (B-81 umbrella).
docs/analysis/cross_sites/mechanism_per_task_report.md:257:## Appendix E: E4 ranked per-cell actions
docs/checkpoints/pre_run/topvenue_constraints.md:33:| A1 | Preregister primary hypotheses, decision rules, and analysis families before post-rerun data are inspected | NEEDS_BIB_ENTRY: Pineau et al. 2021; NeurIPS checklist Q4/Q6 | ⚠️ | `docs/checkpoints/pre_run/preregistration.md` has H1-H8, Holm families, R1-R5, but frontmatter is `status: draft`, `registered_at`, `registered_git_sha`, `witnessed_by`, and OSF DOI are pending. Remediation: lock after advisor email, tag git, deposit OSF; cost 2-4h. | "The hypotheses and decision rules were written before the 16-cell rerun; the camera-ready will cite the lock SHA/OSF DOI once advisor witness is received." |
docs/checkpoints/pre_run/topvenue_constraints.md:34:| A2 | Release code, data, and exact reproduction instructions for main experimental claims | NeurIPS checklist Q4-Q5; NEEDS_BIB_ENTRY: Pineau et al. 2021 | ⚠️ | Code/scripts exist and `osf_lock_manifest.md` lists release artifacts, but `pre_rerun_audit.md §5.1.14` marks replication README as TBD and `§5.3.4` data-sharing policy as TBD. Remediation: write `README_REPLICATION.md` with clone→run→aggregate→figure commands and redaction policy; cost 1 day. | "We release the run manifest, environment snapshots, analysis scripts, and locked code; raw credentials/auth files are excluded and documented." |
docs/checkpoints/pre_run/topvenue_constraints.md:35:| A3 | Pin random seeds and disclose stochasticity limits | NeurIPS checklist Q6-Q7; NEEDS_BIB_ENTRY: Pineau et al. 2021 | ⚠️ | `configs/exp_v2_base.yaml` sets `seed: 42` and `runner/main.py::_seed_global_rng()` seeds Python/NumPy/Torch; `master_bug_catalog.md B-37` notes B0 proxy seeding remains best-effort and Anthropic-native seed support may be absent. Remediation: disclose B0 stochasticity and run B0 T=0 pilot sensitivity; cost 0.5-1 day. | "B1 is locally seeded/greedy; B0 uses best-effort API seed forwarding, so aggregate claims are reported as one controlled stochastic sample with bootstrap task uncertainty." |
docs/checkpoints/pre_run/topvenue_constraints.md:36:| A4 | Record software, hardware, model revisions, and evaluator code hashes per run | NeurIPS checklist Q8; NEEDS_BIB_ENTRY: Pineau et al. 2021 | ✓ | `scripts/provenance/snapshot_env.py` records Python, Torch, libraries, HF revision SHA, git commit/dirty state, GPU info, and `evaluator_code.combined_sha256`; `pre_rerun_audit.md §2.2` and `§3.3.6` require snapshots and rederive audit trails. | "Each paper-grade run carries machine/model/evaluator fingerprints sufficient to reconstruct scoring provenance." |
docs/checkpoints/pre_run/topvenue_constraints.md:38:| A6 | Provide complete hyperparameter/configuration disclosure | NeurIPS checklist Q6; NEEDS_BIB_ENTRY: Pineau et al. 2021 | ✓ | `configs/exp_v2_base.yaml` and per-site configs expose max steps, observation modes, backend temperature, cost rates, energy tracking, viewport, and seeds; `preregistration.md §4` locks mode definitions and inclusion rules. | "All experimental settings are in versioned YAML configs and summarized in the methods appendix." |
docs/checkpoints/pre_run/topvenue_constraints.md:42:| A10 | Include compute resources, total compute, and cost/carbon accounting | NeurIPS checklist Q8; `qiu2025modserve` | ⚠️ | `pre_rerun_audit.md §3.4` has per-cell GPU-hours, USD, and carbon tracking; cross-platform GPU power profile and Section 8 prose are TBD. Remediation: aggregate after rerun and add total compute table; cost 0.5 day. | "We report per-cell runtime, GPU type, API cost, and carbon estimates; cross-machine power variation is a limitation." |
docs/checkpoints/pre_run/topvenue_constraints.md:54:| B3 | Report effect sizes, not just p-values | NEEDS_BIB_ENTRY: Cohen 1988; NeurIPS checklist Q7 | ⚠️ | `preregistration.md §4` locks Cohen's h/d, but `pre_rerun_audit.md §4.1.4` says Cohen's h table is partial. Remediation: add h/d/CI columns to paper tables; cost 2-3h. | "We report pp lift plus standardized effect sizes with uncertainty for binary and continuous outcomes." |
docs/checkpoints/pre_run/topvenue_constraints.md:56:| B5 | Quantify annotation/label reliability for human-coded FP and failure modes | NEEDS_BIB_ENTRY: Cohen 1960 kappa; CONSORT-AI/Liu et al. 2020 | ❌ | `pre_rerun_audit.md §1.5` and `§4.3` mark FP labeling and failure-mode κ pilots as TBD. Remediation: two-rater 30-task pilot, κ≥0.7 target, adjudication log; cost 1 day. | "We will report κ for manually audited FP/failure-mode labels; until then qualitative mechanism claims remain caveated." |
docs/checkpoints/pre_run/topvenue_constraints.md:57:| B6 | Predefine missing-data/crashed-episode handling | CONSORT-AI/Liu et al. 2020; NeurIPS checklist Q7 | ✓ | Locked 2026-05-09 in `preregistration.md §4`: crashed episodes (uncaught exception / OOM / >30min timeout / browser crash) excluded from paired-N denominators, NOT imputed; not_logged_in / auth_drift after 3 retries excluded; missing artifacts excluded from per-step (not full-task); per-cell exclusion count + reason histogram in Appendix C. | "Listwise deletion only — crashes excluded from paired N, no success/failure imputation, exclusion histograms reported per cell." |
docs/checkpoints/pre_run/topvenue_constraints.md:59:| B8 | Analyze heterogeneity and avoid overclaiming pooled averages | NEEDS_BIB_ENTRY: DerSimonian & Laird 1986; NeurIPS checklist Q7 | ✓ | Locked 2026-05-09 in `preregistration.md §4`: random-effects DerSimonian-Laird primary, Cochran Q + I² + τ² reported, Higgins & Thompson 2002 I² thresholds (< 25% pooled / 25-50% both / 50-75% per-cell primary / > 75% no pool + heterogeneity-source decomposition by site/model/task-pool). Per-cell estimates always shown alongside pooled, so heterogeneity never averaged away. | "Random-effects with pre-specified I² thresholds — per-cell + heterogeneity-source decomposition reported when between-cell variance is high." |
docs/checkpoints/pre_run/topvenue_constraints.md:60:| B9 | Run power/MDE analysis for effect detectability | NeurIPS checklist Q7; CONSORT-AI/Liu et al. 2020 | ✓ | `docs/analysis/cross_sites/power_analysis.md` (consolidated 2026-05-09) reports observed-SR-conditioned MDE 4-7pp at 80% per-cell power across 3 sites + family-wise K-of-N power table at observed effect sizes (1-5pp) **<10%** → triggered methodological reframing: K-of-N rule retained as secondary transparency check, primary detection via B8 random-effects meta + TOST on N=910 pooled tasks. Bug fix: `power_analysis.py` stale interpretation block (claimed >0.95 K=12/16 family power at p_per=0.65; actual 0.289) corrected in same commit. Preregistration §4 updated with B9 row locking the corrected framing. | "Per-cell MDE 4-7pp at observed SR (8-15%); K-of-N rule documented as secondary transparency, primary detection via random-effects meta + pooled TOST." |
docs/checkpoints/pre_run/topvenue_constraints.md:67:| C2 | Reset environment/site state before each condition comparison | `zhou2024webarena`; `koh2024visualwebarena` | ⚠️ | `paper_drafts/section3_definition.md §3.5` states fresh reset; `pre_rerun_audit.md §2.1.1` enforces `RESET_BEFORE=1`, but `§3.2.3` state contamination snapshot is TBD. Remediation: implement pre/post `site_state_snapshot.sh`; cost 1 day. | "Each paper-grade condition starts from reset state; final artifacts will include state snapshots to verify no cross-condition contamination." |
docs/checkpoints/pre_run/topvenue_constraints.md:68:| C3 | Verify stateful website contamination, auth drift, and N/A tasks automatically | `zhou2024webarena`; NEEDS_BIB_ENTRY: Breck et al. 2017 | ⚠️ | Watchdog auth refresh and N/A filter exist; `pre_rerun_audit.md §3.2.9` has temporal auth/reset checks, but site-state snapshot remains TBD. Remediation: add strict site snapshot gate; cost 0.5-1 day. | "Temporal/auth drift checks are automated; state snapshotting is the remaining gate before declaring cells paper-grade." |
docs/checkpoints/pre_run/topvenue_constraints.md:76:| C11 | Provide task-level trace evidence for failure modes, not just aggregate SR | `koh2024visualwebarena`; `drouin2024workarena`; `furuta2024exposing` | ⚠️ | `pre_rerun_audit.md §4.4.b` lists `diag_pattern_match.py`, galleries, and pattern aggregators; per-cell failure-mode breakdown and κ are still TBD. Remediation: run post-rerun and add appendix table; cost 0.5-1 day. | "Trace-level diagnostics support the mechanism claims; automated pattern labels will be spot-checked against human labels." |
docs/checkpoints/pre_run/topvenue_constraints.md:77:| C12 | Report benchmark scope and avoid single-benchmark generalization claims | `deng2023mind2web`; `drouin2024workarena`; NEEDS_BIB_ENTRY: OSWorld/Xie et al. 2024 | ⚠️ | The intro and preregistration scope to VWA with planned WA expansion; `pre_rerun_audit.md §4.2.1` includes non-visual VWA+WA subset, but WA full expansion is untested. Remediation: scope claims to VWA or finish WA 480; cost 2-3 days. | "The main empirical claim is VWA-scoped; WebArena is reported only as planned/robustness unless completed." |
docs/checkpoints/pre_run/topvenue_constraints.md:78:| C13 | Disclose evaluator model drift when using LLM-as-judge (`ua_match`) | `zhou2024webarena`; NEEDS_BIB_ENTRY: Lipton & Steinhardt 2018 | ⚠️ | `master_bug_catalog.md B-20` documents GPT judge drift; `pre_rerun_audit.md §4.5.4` eval reproducibility spot-check is TBD. Remediation: rerun N=20 judge spot-check and pin prompt/model; cost 2-4h. | "LLM-judge tasks are isolated, prompt/model are pinned, and sensitivity to judge drift is disclosed." |
docs/checkpoints/pre_run/topvenue_constraints.md:85:| D1 | Claims must match evidence and scope; aspirational routing must not be stated as achieved | NeurIPS checklist Q1-Q2; NEEDS_BIB_ENTRY: Lipton & Steinhardt 2018 | ⚠️ | `paper_planning.md §1` labels the 4-fold drop-in hook provisional pending data; `preregistration.md R1-R5` maps framing to outcomes. Some older draft prose still says "hidden fourth routing arm" before 16-cell confirmation. Remediation: update intro after rerun based on R-rule; cost 2h. | "The final framing is data-conditional and tied to R1-R5; router deployment claims are deferred unless H7/H8 are locked and pass." |
docs/checkpoints/pre_run/topvenue_constraints.md:89:| D5 | Avoid cherry-picking task subsets or cells | NeurIPS checklist Q1/Q7; NEEDS_BIB_ENTRY: Lipton & Steinhardt 2018 | ⚠️ | `preregistration.md §4` defines cell inclusion and N floor; `pre_rerun_audit.md §4.8.2` counterfactual cell-removal stability is TBD. Remediation: run leave-one-cell-out decision test; cost 2h. | "All cells meeting locked criteria are included; leave-one-cell-out stability is reported as a falsification check." |
docs/checkpoints/pre_run/topvenue_constraints.md:90:| D6 | State failed assumptions and limitations in a dedicated section | NeurIPS checklist Q2; NEEDS_BIB_ENTRY: Lipton & Steinhardt 2018 | ⚠️ | `section4_limitations_disclosure.md` exists and `pre_rerun_audit.md §4.4` lists coverage, but final Section 8 prose is still TBD. Remediation: consolidate limitations in final paper; cost 0.5 day. | "The limitations appendix already enumerates evaluator, scaffold, numerical, and pre/post-fix threats." |
docs/checkpoints/pre_run/topvenue_constraints.md:94:## Category E — Web-Agent Specific (cross-bench / cost / latency)
docs/checkpoints/pre_run/topvenue_constraints.md:98:| E1 | Evaluate cost-quality tradeoffs, not success alone | `chen2023frugalgpt`; `ong2025routellm`; `qiu2025modserve` | ✓ | `preregistration.md H2` locks cost/latency/AUROC/drop-one drop-in claims; `paper_drafts/section3_definition.md §3.2` quantifies DOM vs P-text token and image-channel cost. | "The contribution is explicitly evaluated on success, drop-one complementarity, cost, latency, and routing signal quality." |
docs/checkpoints/pre_run/topvenue_constraints.md:99:| E2 | Include latency and serving-stage interpretation for multimodal modes | `qiu2025modserve`; `koh2024visualwebarena` | ⚠️ | Planning docs cite latency advantage; `pre_rerun_audit.md §3.4` tracks cost but cross-platform power/latency profiles are incomplete. Remediation: aggregate p50/p95 latency by mode and disclose hardware; cost 2-4h. | "Skipping image encoding removes the dominant multimodal serving stage; p50/p95 latency will be reported per cell." |
docs/checkpoints/pre_run/topvenue_constraints.md:100:| E3 | Use routing baselines that avoid train/test leakage | `chen2023frugalgpt`; `ong2025routellm`; `li2026avenirweb` | ⚠️ | `preregistration.md H7-H8` specifies 5-fold site-stratified CV and first-step leak-free router, but status is pending advisor lock and may be paper-2 deferred. Remediation: decide paper-1 vs paper-2 and implement split script; cost 1-2 days if included. | "Paper 1 does not require learned-router claims unless H7/H8 are locked; otherwise routing is oracle/characterization only." |
docs/checkpoints/pre_run/topvenue_constraints.md:102:| E5 | Test or clearly scope cross-benchmark generalization | `deng2023mind2web`; `drouin2024workarena`; NEEDS_BIB_ENTRY: GAIA/Mialon et al. 2023; NEEDS_BIB_ENTRY: OSWorld/Xie et al. 2024 | ❌ | Domain context says WA 480 is planned; current completed prose is VWA-centered. Remediation: either complete WA 480 non-visual subset or explicitly scope to VWA and move cross-bench to future work; cost 2-3 days or 1h to scope. | "The main paper is a VWA characterization; WA/Mind2Web/OSWorld generalization is future work unless reported in an appendix." |
docs/checkpoints/pre_run/topvenue_constraints.md:103:| E6 | Include model-scale contrast for agent behavior | `koh2024visualwebarena`; `drouin2024workarena`; `li2024effects` | ⚠️ | B0/B1 are included and `section1_intro.md` reports capability interaction, but B1 reddit phantom and B1 shop are still part of the 16-cell rerun plan. Remediation: finish 16-cell scope or weaken cross-capability claim; cost rerun-dependent. | "Capability contrast is limited to B0/B1 and interpreted as a scale probe, not a universal model-family law." |
docs/checkpoints/pre_run/topvenue_constraints.md:105:| E8 | Measure routing signal calibration/discriminability before claiming deployability | `chen2023frugalgpt`; `ong2025routellm`; NEEDS_BIB_ENTRY: Guo et al. 2017 calibration | ⚠️ | `preregistration.md H2(c)` locks AUROC threshold and signal universe; `pre_rerun_audit.md §4.2.12` has calibration diagnostics, but C4 DeLong/AUROC delta tests remain T1. Remediation: run AUROC delta CI/DeLong or bootstrap; cost 0.5 day. | "Routing-signal results are characterized with AUROC and not used as deployment proof without held-out router validation." |
docs/checkpoints/pre_run/topvenue_constraints.md:106:| E9 | Account for task visuality/site substrate when interpreting mode performance | `koh2024visualwebarena`; `guan2024hallusionbench`; `fu2024blink` | ⚠️ | `section1_intro.md` discusses classifieds visual-rich vs reddit text-dominated; `pre_rerun_audit.md §4.2.1` has non-visual subset robustness, but site×mode GLM is T1. Remediation: run mode×site interaction test; cost 0.5 day. | "Cross-site asymmetry is treated as a finding and a threat to generalization, not averaged away." |
docs/checkpoints/pre_run/topvenue_constraints.md:112:| F1 | Internal validity: isolate representation effects from scaffold/evaluator bugs | NEEDS_BIB_ENTRY: Cook & Campbell 1979; NEEDS_BIB_ENTRY: Sculley et al. 2014 | ⚠️ | Phase A fixes and bug catalog are strong, but `pre_rerun_audit.md §4.8.3` outlier task spot-check is TBD. Remediation: top/bottom task manual review; cost 2-3h. | "The rerun uses post-fix code; archived pre-fix data is appendix-only robustness." |
docs/checkpoints/pre_run/topvenue_constraints.md:113:| F2 | Construct validity: ensure metrics measure task success and not evaluator artifacts | `zhou2024webarena`; `koh2024visualwebarena`; NEEDS_BIB_ENTRY: Lipton & Steinhardt 2018 | ⚠️ | Adjusted SR and FP filters address evaluator artifacts; `pre_rerun_audit.md §4.5.4` LLM-judge reproducibility is TBD. Remediation: judge spot-check and per-evaluator-class sensitivity; cost 0.5 day. | "We report raw and adjusted success and isolate evaluator-class threats in limitations." |
docs/checkpoints/pre_run/topvenue_constraints.md:115:| F4 | Statistical conclusion validity: report uncertainty and sensitivity to thresholds | NeurIPS checklist Q7; NEEDS_BIB_ENTRY: Cook & Campbell 1979 | ⚠️ | CIs/meta-analysis exist; `pre_rerun_audit.md §4.8.1` K±1 and `§4.8.2` leave-one-cell-out are TBD. Remediation: add threshold-gradient and cell-removal table; cost 0.5 day. | "The decision rule is locked; sensitivity tables show how close the conclusion is to the threshold." |
docs/checkpoints/pre_run/topvenue_constraints.md:116:| F5 | Selection bias: curated mechanistic tasks must not be mistaken for population-wide behavior | NEEDS_BIB_ENTRY: Zhang et al. 2024 activation patching; NEEDS_BIB_ENTRY: Wang et al. 2023 | ⚠️ | `实验笔记.md §117.5` explicitly warns strong/reverse tier curation may correlate with effect size and launches 2x2 controls. Remediation: complete cells C/D and scope claims to curated tasks if controls fail; cost 0.5-1 day. | "Mechanistic claims are about curated mirage-disagreement tasks unless the 2x2 control supports broader bidirectionality." |
docs/checkpoints/pre_run/topvenue_constraints.md:117:| F6 | Instrumentation validity: verify cross-machine numerical consistency for mechanistic activations | NEEDS_BIB_ENTRY: Pineau et al. 2021; NEEDS_BIB_ENTRY: Zhang et al. 2024 activation patching | ⚠️ | `scripts/provenance/numerical_determinism_check.py` exists; `pre_rerun_audit.md §4.9.1` says comparison needs A100/Myriad. Remediation: run compare and report max hidden-state diff; cost 1-2h compute. | "Mechanistic layers are only interpreted after same-input hidden-state stability is verified within tolerance." |
docs/checkpoints/pre_run/topvenue_constraints.md:119:| F8 | Temporal validity: account for benchmark/software drift over time | `zhou2024webarena`; `drouin2024workarena`; NeurIPS checklist Q4 | ✓ | `pre_run/locked_versions.md` (created 2026-05-09) pins Playwright **1.58.0** + Chromium **revision 1208** (transitively pinned by Playwright). Plus VWA `832f037e` + HF model SHA `ebb281ec70b0...` + transformers 4.57.6 + torch (2.11.0+cu128 DGX / 2.1.0 Myriad with B-81 shims). T0/T1/T2/T3 evaluator-change protocol governs upgrades; same paper requires dual-report under post-lock T0 fix. | "Playwright 1.58.0 + Chromium 1208 + VWA 832f037e + HF Qwen3-VL-4B revision ebb281ec — all pinned in `pre_run/locked_versions.md` and auto-verified pre-launch." |
docs/checkpoints/pre_run/topvenue_constraints.md:131:| G7 | Control for task-selection bias with crossed task-tier × direction design | NEEDS_BIB_ENTRY: Zhang et al. 2024; NEEDS_BIB_ENTRY: Conmy et al. 2023 | ✓ | 4-cell 2x2 design completed 2026-05-09: A (fwd×strong N=24, Holm L17 p=0.011 \*\*) + B (rev×reverse N=15, Holm L11+L17) + C (fwd×reverse N=15, Δ=-0.10 weaker) + D (rev×strong N=24, Holm L11+L17 p=0.006/0.008 \*\*). Cross-cell Welch p=0.509-1.000 (NS) — magnitudes statistically indistinguishable across direction × tier; reverse direction Δ=-0.193 identical on both tiers (p=1.000), forward Δ=-0.143/-0.103 not Welch-distinguishable across tiers (p=0.577). Selection bias rejected; mechanism claim universal pending cross-site (cells F/G). | "The crossed 2x2 design rules out selection-bias artifact: reverse direction produces identical magnitude on both task tiers (Welch p=1.000), forward direction is Welch-indistinguishable across tiers (p=0.577); only Holm-significance differs due to N power asymmetry." |
docs/checkpoints/pre_run/topvenue_constraints.md:134:| G10 | Use mechanistic metrics tied to behavior/action tokens | NEEDS_BIB_ENTRY: Wang et al. 2023; NEEDS_BIB_ENTRY: Geva et al. 2021 | ⚠️ | Current Stage 2 metrics are continuation overlap and LD to target; direct action-token probability/action-switch tests are not clearly present. Remediation: add action-token/logit-diff metric where outputs contain actions; cost 0.5-1 day. | "Continuation metrics capture the mirage reasoning state; action-token metrics are added where the trace format makes them well-defined." |
docs/checkpoints/pre_run/topvenue_constraints.md:141:| H1 | Maintain a dedicated limitations section with strong assumptions and robustness failures | NeurIPS checklist Q2; NEEDS_BIB_ENTRY: Lipton & Steinhardt 2018 | ⚠️ | `section4_limitations_disclosure.md` exists and `pre_rerun_audit.md §4.4` enumerates limitations; final Section 8 prose is not complete. Remediation: consolidate into final limitations section; cost 0.5 day. | "Limitations are not hidden in appendices; they are surfaced alongside the main empirical claims." |
docs/checkpoints/pre_run/topvenue_constraints.md:143:| H3 | Scope deployment and safety claims for autonomous web agents | NeurIPS checklist Q10; `wei2026clawsafety`; `drouin2024workarena` | ✓ | `pre_run/ethics_license_coi_statements.md` (created 2026-05-09) covers 4-point misuse/safety scope: (a) lower automation cost broadens deployment surface (legitimate + misuse), (b) visual hijack patterns + adversarial-prompt risk, (c) benchmark contamination risk, (d) cross-bench generalization not established → CLAW-Safety review required for deployment. | "The paper is a scientific characterization, not deployment-ready. Misuse/safety scope (autonomous web agents lower automation cost, surface visual-hijack failure modes, accelerate benchmark contamination) is explicit in §8; safety-critical deployment requires domain-specific evaluation." |
docs/checkpoints/pre_run/release_redaction_checklist.md:27:  - Verify: `grep -rn "/home/jiaming" --include="*.py" --include="*.sh" --include="*.yaml" --include="*.md"` shows only DGX-specific docs (CLAUDE.md gitignored, runbooks are fine)
docs/checkpoints/pre_run/release_redaction_checklist.md:28:- [ ] No hardcoded `/home/ucab352/` paths outside `scripts/queues/qsub_*.sh` (Myriad-specific qsubs are fine)
docs/analysis/cross_sites/power_analysis_observed_sr20.md:3:**Configuration**: paired-design binary SR comparison; α=0.05 two-sided, β=0.2 (power=80%); baseline SR=0.20
docs/analysis/cross_sites/power_analysis_observed_sr20.md:13:## Per-cell power at assumed effect sizes
docs/analysis/cross_sites/power_analysis_observed_sr20.md:23:## Family-wise power (K-of-N rule)
docs/analysis/cross_sites/power_analysis_observed_sr20.md:26:Family-wise power assumes per-cell power is uniform (averaged across sites).
docs/analysis/cross_sites/power_analysis_observed_sr20.md:28:| K threshold | Per-cell power 0.50 | 0.65 | 0.80 | 0.90 |
docs/analysis/cross_sites/power_analysis_observed_sr20.md:37:- At baseline SR=0.20, smallest site (reddit N=210) detects effects ≥ 7.7pp at 80% power per cell.
docs/analysis/cross_sites/power_analysis_observed_sr20.md:38:- Largest site (shopping N=466) detects effects ≥ 5.2pp at 80% power per cell.
docs/analysis/cross_sites/power_analysis_observed_sr20.md:39:- At 3pp true effect, **per-cell power ≈ 0.19** (smallest site) → K_h1=12/16 family power = 0.000, K_h3=11/16 family power = 0.000.
docs/analysis/cross_sites/power_analysis_observed_sr20.md:40:- At 5pp true effect, **per-cell power ≈ 0.44** → K_h1=12/16 family power = 0.012, K_h3=11/16 = 0.042.
docs/analysis/cross_sites/power_analysis_observed_sr20.md:41:- **K_h1=12/16 is calibrated for ≥5pp effects.** For 2-3pp mechanism effects, K_h3=11/16 is the operative threshold; below ~3pp, neither K-of-N rule has paper-grade power and the paper relies on **TOST equivalence on pooled data** (N=234+210+466).
docs/analysis/cross_sites/power_analysis_observed_sr20.md:47:per-cell MDE = [7.3, 7.7, 5.2]pp 
docs/analysis/cross_sites/power_analysis_observed_sr20.md:48:for cls/red/shop respectively. The K_h1=12/16 family-wise rule has 1% power for 5pp effects 
docs/analysis/vwa_reddit/B1_DOM_digest.md:227:**注**：上表来自 A4b_fail_reason_cost_stats（跨 DOM+Vision 全条件统计），DOM 条件内的 search_repeat / click_back_loop 成本可能略高。
docs/analysis/cross_sites/swatch_form_change_audit.md:148:- Affects accuracy of `action_success`, `page_changed`, `page_change_reasons`, and downstream `wasted_cost`, `no_op_rate`, router `trigger_distribution`. These signals appear in shopping baseline tables.
docs/analysis/cross_sites/cost_per_mode.md:3:B0 reports avg_total_cost_usd from per-token API rates (Qwen3-VL-235B-A22B $0.001/1k input, $0.005/1k output). B1 reports avg_total_energy_kwh × $0.12/kWh as electricity-equivalent cost — local inference pays no API dollars; the per-token cost field in B1 condition_summary_v2.json is artifact (uses B0 rates) and is NOT comparable. B0 vs B1 dollar costs belong to different classes (API call cost vs electricity), so the paper presents both side-by-side, not a single ratio.
docs/analysis/cross_sites/cost_per_mode.md:7:| site | mode | avg_steps | avg_total_cost_usd ($/ep) |
docs/analysis/cross_sites/cost_per_mode.md:23:Computed as `avg_total_energy_kwh × $0.12/kWh` (DGX Spark, UK industrial rate per `metrics.energy.region: uk` in B1 yaml).
docs/analysis/cross_sites/cost_per_mode.md:43:The qualitative cost gap between API and local inference is large (2–3 orders of magnitude per these data) but is fundamentally a deployment-mode comparison, not a model-size ratio. Reporting a single multiplier (e.g. '30x') without specifying the cost class is misleading.
docs/checkpoints/codex_prompts/section5_reddit_prose.md:67:    * Efficiency 3b: red token cost +733 image tokens, no SR benefit
docs/analysis/cross_sites/mechanism_case_studies.md:9:Intent: same snare-drum target. DOM failed as above by stopping on page 2 after repeated no-progress scrolls. Phantom-SoM also began with `snare drum black red`, but it continued the pagination branch past DOM's boundary: step 10 clicked page 3, step 14 clicked page 4, and step 16 clicked `http://100.95.81.103:9980/index.php?page=item&id=50230`. The decisive step is not image use; Phantom-SoM has no screenshot. It is a prompt and indexed-mark persistence effect: the SoM-style action policy tolerated a longer search frontier and kept using marked pagination/listing controls after scroll failures. B1 makes the limit visible: on the same task, B1 DOM and B1 Phantom-SoM both failed. B1 Phantom-SoM did click `id=50230` at step 4, but then second-guessed the match, searched again, and ended in repeated no-progress type actions. This is useful negative evidence for treating P-SoM as DOM-cost, not as an image substitute.
docs/analysis/cross_sites/mechanism_case_studies.md:30:Across the eight cases, the shared mechanism is not "better perception" but boundary control after an initially plausible action stops producing evidence. Text-payload and prompt changes mostly alter the next branch after a stale viewport: go to a category (`classifieds #201`), go to the next page (`classifieds #103`), broaden a query (`reddit #7`), advance from one upvote control to the next (`reddit #81`), or keep searching instead of finishing (`reddit #167`). Axis 3 is different because it changes the evidence available for visually grounded predicates. `classifieds #17` shows the positive form: both B0 and B1 text-only traces hesitate because "red handlebars" is not a reliable text field, while SoM finishes from the image. `reddit #0` shows the negative form: the screenshot can over-weight the image anchor and suppress the post/comment URL branch. The B1 observations are consistent with the new aggregate result that B1 P-SoM classifieds (7.69%) is statistically indistinguishable from B1 DOM (8.55%, p=0.73): B1 P-SoM follows DOM-like text loops on `#201` and fails with DOM on `#17`, while B1 SoM succeeds on `#17`. That supports, but does not prove, the hypothesis that Axes 1 and 2 are mostly capability-tier invariant and Axis 3 is the real tier multiplier. Strongest evidence by drop-in property: (a) cost/no-image parity is `#201` B1; (b) trajectory complementarity is `#103`; (c) action-vocabulary/control-state sensitivity is `#81`; (d) image-only lift is `#17`.
docs/analysis/cross_sites/power_analysis_observed_sr15.md:3:**Configuration**: paired-design binary SR comparison; α=0.05 two-sided, β=0.2 (power=80%); baseline SR=0.15
docs/analysis/cross_sites/power_analysis_observed_sr15.md:13:## Per-cell power at assumed effect sizes
docs/analysis/cross_sites/power_analysis_observed_sr15.md:23:## Family-wise power (K-of-N rule)
docs/analysis/cross_sites/power_analysis_observed_sr15.md:26:Family-wise power assumes per-cell power is uniform (averaged across sites).
docs/analysis/cross_sites/power_analysis_observed_sr15.md:28:| K threshold | Per-cell power 0.50 | 0.65 | 0.80 | 0.90 |
docs/analysis/cross_sites/power_analysis_observed_sr15.md:37:- At baseline SR=0.15, smallest site (reddit N=210) detects effects ≥ 6.9pp at 80% power per cell.
docs/analysis/cross_sites/power_analysis_observed_sr15.md:38:- Largest site (shopping N=466) detects effects ≥ 4.6pp at 80% power per cell.
docs/analysis/cross_sites/power_analysis_observed_sr15.md:39:- At 3pp true effect, **per-cell power ≈ 0.23** (smallest site) → K_h1=12/16 family power = 0.000, K_h3=11/16 family power = 0.000.
docs/analysis/cross_sites/power_analysis_observed_sr15.md:40:- At 5pp true effect, **per-cell power ≈ 0.53** → K_h1=12/16 family power = 0.061, K_h3=11/16 = 0.151.
docs/analysis/cross_sites/power_analysis_observed_sr15.md:41:- **K_h1=12/16 is calibrated for ≥5pp effects.** For 2-3pp mechanism effects, K_h3=11/16 is the operative threshold; below ~3pp, neither K-of-N rule has paper-grade power and the paper relies on **TOST equivalence on pooled data** (N=234+210+466).
docs/analysis/cross_sites/power_analysis_observed_sr15.md:47:per-cell MDE = [6.5, 6.9, 4.6]pp 
docs/analysis/cross_sites/power_analysis_observed_sr15.md:48:for cls/red/shop respectively. The K_h1=12/16 family-wise rule has 6% power for 5pp effects 
docs/analysis/vwa_reddit/B0_findings.md:79:| Vision vs SoM | total_cost | **1.1e-11** | Vision 更便宜 ★★★ |
docs/analysis/vwa_reddit/B0_findings.md:81:| Vision vs DOM | total_cost | **9.9e-21** | Vision 更便宜 ★★★ |
docs/analysis/vwa_reddit/B0_findings.md:83:| SoM vs DOM | total_cost | **3.9e-4** | SoM 更便宜 ★★★ |
docs/checkpoints/codex_prompts/topvenue_constraints_audit.md:107:## Category E — Web-Agent Specific (cross-bench / cost / latency)
docs/checkpoints/codex_prompts/topvenue_constraints_audit.md:135:4. **Gaps marked ❌ MUST have remediation note** (specific action + cost estimate).
docs/checkpoints/codex_prompts/topvenue_constraints_audit.md:145:- **Hook**: Phantom routing space (3 arms: P-text / P-prompt / P-SoM) with 4-fold drop-in property (cost ≈ DOM / latency ~50% / signal AUROC ≥ baseline / drop-one 1.7-3.8pp). PROVISIONAL pending data confirmation.
docs/checkpoints/codex_prompts/topvenue_constraints_audit.md:151:- **Critical concerns we already know about**: pattern-based mirage curation brittleness, N=15-24 mechanistic limited power, single-model B1, single-site partially mitigated, post-hoc layer selection.
docs/checkpoints/pre_run/ethics_license_coi_statements.md:18:  `aggregate_phantom_lift.py` (paired McNemar), `power_analysis.py` (MDE)
docs/checkpoints/pre_run/ethics_license_coi_statements.md:34:- UCL Myriad HPC cluster (mixed V100/A100 nodes) — institutional research allocation
docs/checkpoints/pre_run/ethics_license_coi_statements.md:35:- UCL Condense A100 (40GB dedicated allocation) — institutional research allocation
docs/checkpoints/pre_run/ethics_license_coi_statements.md:36:- DGX Spark workstation (NVIDIA GB10) — shared research workstation, HolisticAI lab
docs/checkpoints/pre_run/ethics_license_coi_statements.md:39:Model API costs (B0 proxy access to Qwen3-Omni-235B) were covered by lab
docs/checkpoints/pre_run/ethics_license_coi_statements.md:66:1. **Lower automation cost broadens deployment surface**: The 4-fold drop-in
docs/checkpoints/pre_run/ethics_license_coi_statements.md:67:   property we characterize (cost ≈ DOM, latency ~50% lower) reduces the
docs/checkpoints/pre_run/ethics_license_coi_statements.md:101:sites with material consequences. Our cost / latency drop-in property
docs/checkpoints/codex_prompts/B0_dom_shopping_diagnostic.md:47:**H1. Aggregation/排序任务 cost**: A 类 intent 多含 "least/most expensive" / "highest rated" / "show me X with Y filter" — 需要遍历 list + 比较，DOM-only AXTree 在 Magento list page (12 items × ~10 fields each) 接近 token cap (12K obs cap)，agent 可能丢失 critical row。codex 应抽 ~10 个 A 类失败 task 的 episode JSONL，看 final thought / final action 是否表现"列表过长 → 选错"或"截断 → 看不到目标"。
docs/analysis/phantom_paper/disagreement_clusters.md:161:Why Phantom-SoM succeeded: direct step traces are unavailable, but §103 classifies these as mixed and text-compact quick-decision cases. The counterpart failures are dominated by visual-missing, click-loop, and search-loop patterns, consistent with Phantom opening a distinct low-cost solution basin.
docs/checkpoints/pre_run/dataset_card.md:24:| **classifieds** | 234 | `d36a20c1eaa1f5da...` | Visual-rich product listings (Magento-based) | DGX Tailscale → quark Docker container `:9980` (cf. `memory/MEMORY.md` "Docker 容器端口") |
docs/checkpoints/pre_run/dataset_card.md:25:| **reddit** | 210 | `ecd4ed4370740fd6...` | Text-dominated forum threads (Postmill) | DGX Tailscale → quark `:9999` |
docs/checkpoints/pre_run/dataset_card.md:26:| **shopping** | 466 | `07889e3646ee10e3...` | Mixed text + image product pages (Magento) | DGX Tailscale → quark `:7770` |
docs/checkpoints/pre_run/dataset_card.md:112:- **Run artifacts** (per-cell episode JSONLs, summaries, env_snapshots): paper-grade
docs/checkpoints/codex_prompts/mechanism_per_task_explanation.md:127:Plus click subtypes (heuristic via element_id text in obs around the click — may be costly; OPTIONAL):
docs/analysis/vwa_classifieds/B0_findings.md:83:| Vision vs SoM | total_cost | **2.9e-9** | Vision 更便宜 ★★★ |
docs/analysis/vwa_classifieds/B0_findings.md:85:| Vision vs DOM | total_cost | **4.5e-12** | Vision 更便宜 ★★★ |
docs/analysis/vwa_classifieds/B0_findings.md:87:| SoM vs DOM | total_cost | 0.496 | **无显著差异** |
docs/analysis/vwa_classifieds/B0_findings.md:220:高成本失败模式（avg cost per episode）：
docs/analysis/cross_sites/site_mechanism_dictionary.md:74:- DOM costs $0.0516/ep.
docs/analysis/cross_sites/site_mechanism_dictionary.md:75:- P-text costs $0.0459/ep.
docs/analysis/cross_sites/site_mechanism_dictionary.md:131:- P-text costs $0.0459/ep.
docs/analysis/cross_sites/site_mechanism_dictionary.md:132:- P-SoM costs $0.0381/ep.
docs/analysis/cross_sites/site_mechanism_dictionary.md:133:- The cost difference is driven by fewer steps, not image tokens.
docs/analysis/cross_sites/site_mechanism_dictionary.md:186:- P-SoM costs $0.0381/ep.
docs/analysis/cross_sites/site_mechanism_dictionary.md:187:- SoM costs $0.0409/ep.
docs/analysis/cross_sites/site_mechanism_dictionary.md:188:- SoM has image-token cost, but also fewer steps than DOM.
docs/analysis/cross_sites/site_mechanism_dictionary.md:237:Reddit's forum hierarchy makes representation and prompt the leading mechanisms. AXTree depth plus image-referential tasks push DOM into repeated keyword search; flat marks and SoM-style prompt priors open a different, low-cost trajectory basin. The image channel is useful for some content tasks but is not the site navigation substrate, so full SoM is not a monotone improvement over P-SoM.
docs/analysis/cross_sites/site_mechanism_dictionary.md:298:- DOM costs $0.0427/ep.
docs/analysis/cross_sites/site_mechanism_dictionary.md:299:- P-text costs $0.0397/ep.
docs/analysis/cross_sites/site_mechanism_dictionary.md:351:- P-text costs $0.0397/ep.
docs/analysis/cross_sites/site_mechanism_dictionary.md:352:- P-SoM costs $0.0441/ep.
docs/analysis/cross_sites/site_mechanism_dictionary.md:353:- P-SoM takes more steps in this site, so prompt does not automatically reduce cost.
docs/analysis/cross_sites/site_mechanism_dictionary.md:408:- P-SoM costs $0.0441/ep.
docs/analysis/cross_sites/site_mechanism_dictionary.md:409:- SoM costs $0.0415/ep.
docs/analysis/cross_sites/site_mechanism_dictionary.md:410:- SoM has image-token cost, but fewer steps: 8.60 vs 12.05.
docs/analysis/cross_sites/site_mechanism_dictionary.md:479:- No shopping cells exist in `axis_effect_size.json`, `axis1_microbehavior.json`, `mechanism_per_task.json`, `sr_fp_per_mode.json`, or `cost_per_mode.json`.
docs/analysis/cross_sites/site_mechanism_dictionary.md:513:- B0 DOM costs $0.0424/ep.
docs/analysis/cross_sites/site_mechanism_dictionary.md:515:- Wasted cost is $0.0374/ep, 88.2% of total.
docs/analysis/cross_sites/site_mechanism_dictionary.md:516:- No shopping cost cells exist in `cost_per_mode.json`.
docs/analysis/cross_sites/site_mechanism_dictionary.md:564:- No prompt-axis cost cell exists.
docs/analysis/cross_sites/site_mechanism_dictionary.md:609:- No image-axis cost cell exists.
docs/analysis/cross_sites/site_mechanism_dictionary.md:695:| shopping | missing in cost_per_mode | missing | B0 DOM digest reports $0.0424/ep |
docs/analysis/cross_sites/site_mechanism_dictionary.md:722:- Shopping cost/frontier cells should be added to `cost_per_mode.json` after the rerun.
docs/analysis/cross_sites/power_analysis_observed_sr10.md:3:**Configuration**: paired-design binary SR comparison; α=0.05 two-sided, β=0.2 (power=80%); baseline SR=0.10
docs/analysis/cross_sites/power_analysis_observed_sr10.md:13:## Per-cell power at assumed effect sizes
docs/analysis/cross_sites/power_analysis_observed_sr10.md:23:## Family-wise power (K-of-N rule)
docs/analysis/cross_sites/power_analysis_observed_sr10.md:26:Family-wise power assumes per-cell power is uniform (averaged across sites).
docs/analysis/cross_sites/power_analysis_observed_sr10.md:28:| K threshold | Per-cell power 0.50 | 0.65 | 0.80 | 0.90 |
docs/analysis/cross_sites/power_analysis_observed_sr10.md:37:- At baseline SR=0.10, smallest site (reddit N=210) detects effects ≥ 5.8pp at 80% power per cell.
docs/analysis/cross_sites/power_analysis_observed_sr10.md:38:- Largest site (shopping N=466) detects effects ≥ 3.9pp at 80% power per cell.
docs/analysis/cross_sites/power_analysis_observed_sr10.md:39:- At 3pp true effect, **per-cell power ≈ 0.30** (smallest site) → K_h1=12/16 family power = 0.000, K_h3=11/16 family power = 0.002.
docs/analysis/cross_sites/power_analysis_observed_sr10.md:40:- At 5pp true effect, **per-cell power ≈ 0.68** → K_h1=12/16 family power = 0.367, K_h3=11/16 = 0.577.
docs/analysis/cross_sites/power_analysis_observed_sr10.md:41:- **K_h1=12/16 is calibrated for ≥5pp effects.** For 2-3pp mechanism effects, K_h3=11/16 is the operative threshold; below ~3pp, neither K-of-N rule has paper-grade power and the paper relies on **TOST equivalence on pooled data** (N=234+210+466).
docs/analysis/cross_sites/power_analysis_observed_sr10.md:47:per-cell MDE = [5.5, 5.8, 3.9]pp 
docs/analysis/cross_sites/power_analysis_observed_sr10.md:48:for cls/red/shop respectively. The K_h1=12/16 family-wise rule has 37% power for 5pp effects 
docs/checkpoints/pre_run/preregistration.md:30:1. **Phantom-SoM is the deployment hero**: 4-fold drop-in property (cost ≈ DOM, latency ~50% lower, signal AUROC ≥ baseline, drop-one positive) is the headline practical contribution. This is pre-registered strict.
docs/checkpoints/pre_run/preregistration.md:53:- **H1(ii)** ≥ K_h1 of N_cells individually Holm-significant at α=0.05 within the per-cell P-SoM sub-family (m = N_cells), where **K_h1 = 0.75** (commit-locked, see §4).
docs/checkpoints/pre_run/preregistration.md:60:- **(a) Cost** — median cost(P-SoM) within ±10% of median cost(DOM); reflects the by-construction property that `[SOM_MARKS]` is an AXTree regex filter (no image embedding tokens). Tested empirically per cell.
docs/checkpoints/pre_run/preregistration.md:109:- **H7(ii)** ≥ K_h1 of N_cells individually Holm-significant on per-cell lift, bootstrap 95% CI lower-bound > 0.
docs/checkpoints/pre_run/preregistration.md:135:| Rule | Conditions | Paper hook framing | Hook power |
docs/checkpoints/pre_run/preregistration.md:140:| **R4** | H1 holds AND H2 partially fails (e.g., (a) cost or (b) latency fails on some site) | "Phantom-SoM partial drop-in" + §4 disclosure of failed sub-claim. | WEAK; substantial revision |
docs/checkpoints/pre_run/preregistration.md:149:- H1(ii) per-cell P-SoM Holm: m = N_cells.
docs/checkpoints/pre_run/preregistration.md:150:- H2 sub-claims (a)(b)(c)(d): m = 4 × N_cells (each per-cell sub-claim test).
docs/checkpoints/pre_run/preregistration.md:154:- H3(i) axis 1 per-cell: m = N_cells (bootstrap CI lower-bound > 0 test).
docs/checkpoints/pre_run/preregistration.md:155:- H3(ii) axis 2 per-cell: m = N_cells.
docs/checkpoints/pre_run/preregistration.md:161:- H7(ii) per-cell Tier 1 lift Holm: m = N_cells.
docs/checkpoints/pre_run/preregistration.md:191:| **Effect size (continuous)** | Cohen's d with bootstrap CI | For cost/latency H2(a)(b) |
docs/checkpoints/pre_run/preregistration.md:192:| **TOST equivalence margin δ** | **1.0pp** | ≈ 2 tasks in N=234, matches per-cell bootstrap SE; smaller is within sampling noise floor |
docs/checkpoints/pre_run/preregistration.md:193:| **H1 K_h1 cell-pass threshold** | **0.75** | Allows ~25% capability-outlier cells (e.g., B1 shopping power-limited); not so strict that single-cell noise breaks claim |
docs/checkpoints/pre_run/preregistration.md:198:| **N inclusion floor** | ≥ 100 ep per (cell × mode) | Statistical power baseline |
docs/checkpoints/pre_run/preregistration.md:209:| **Stopping rules / contamination halt criteria** (audit B7) | (a) **Pre-launch**: `make pre-launch-check` validates seed configured + HF SHA pinned + git working tree clean + GPU available + disk free > 20GB; failure halts launch (per audit C10). (b) **Smoke-test gate**: first 10 episodes per cell must show ≥ 1 success (or ≥ 1 N/A by ua_match) AND auth-state `logged_in=True`; otherwise halt + watchdog auth_refresh + restart, log incident in `master_bug_catalog.md`. (c) **Auth/site contamination halt**: ≥ 5 consecutive episodes with `not_logged_in` ⇒ stop cell, refresh auth, archive partial run as `_dirty_partial`, restart fresh. (d) **Eval drift halt**: if rerun on identical archived episode produces SR delta > 5pp via `validate_run.py --strict`, freeze cell + investigate evaluator code. (e) **OOM / hardware halt**: 3 consecutive job failures ⇒ stop cell, document hardware in incident log, manually re-queue with diagnostic output. | Halt rules protect data purity; halted cells restarted only after root-cause documented in `master_bug_catalog.md` + bug fix committed. Lock 2026-05-09. |
docs/checkpoints/pre_run/preregistration.md:210:| **Heterogeneity (random-effects, Q, I², τ²) pre-spec** (audit B8) | (a) **Primary estimator**: random-effects DerSimonian-Laird via `aggregate_phantom_meta.py` (already implemented). (b) **Heterogeneity reporting**: report Cochran Q (chi² test of homogeneity), I² (% of total variance attributable to between-cell heterogeneity), τ² (between-cell variance). (c) **Interpretation thresholds (pre-specified)**: I² < 25% = "low heterogeneity, pooled mean is primary"; 25%-50% = "moderate, report both pooled + per-cell"; 50%-75% = "high, per-cell estimates are primary, pooled is summary"; > 75% = "very high, do not pool — report only per-cell + heterogeneity-source analysis (site / model / task-pool)". (d) **Heterogeneity-source decomposition**: when I² > 50%, report meta-regression by site (cls / red / shop) and by model (B0 / B1) to identify dominant variance source. | Higgins & Thompson 2002 (I² thresholds). Per-cell estimates always shown alongside pooled, so heterogeneity is never averaged away. Lock 2026-05-09. |
docs/checkpoints/pre_run/preregistration.md:211:| **K-of-N rule scope** (audit B9 power-corrected) | The **K_h1=12/16 / K_h3=11/16** thresholds are retained as **secondary transparency checks** (count of cells *individually* clearing α=0.05 Holm), **not as gates on H1/H3 paper claims**. **Primary detection** = (a) DerSimonian-Laird random-effects meta-analysis on cells N≥10 (B8 lock above) + (b) TOST equivalence on N=910 pooled tasks at δ=1.0pp. Per `docs/analysis/cross_sites/power_analysis.md` §3-§5, K-of-N family power at observed effect sizes (1-5pp) is < 10%; the rule is calibrated for ≥7pp effects. This recharacterization is consistent with the original §4 "Primary metric" + B8 random-effects lock — K-of-N was always a transparency aggregator, not the primary test, and the corrected power analysis makes that explicit. | `power_analysis.py` bug (stale interpretation block) discovered 2026-05-09; fixed in same commit. K-of-N values themselves unchanged; only the framing as "secondary transparency vs primary gate" is added. Lock 2026-05-09. |
docs/checkpoints/pre_run/preregistration.md:262:one-out layer-selection** robustness — re-run patching on per-cell holdout
docs/checkpoints/pre_run/preregistration.md:330:| 2026-05-03 | Disconfirmation rule changed from "any cell fail" to data-conditional R1-R5 framing rule | "Any cell fail" too strict given single-cell power limits; framing rule maps data outcomes to paper hook revisions transparently |

exec
/bin/bash -lc "rg -n \"@\\w+\\{(lipton|pineau|neurips|zhou2024webarena|koh2024visualwebarena|deng2023mind2web|drouin2024workarena|yang2023som|zheng2024seeact|yang2025magma|wang2023interpretability|zhang2024patching|holm|derSimonian|der|cohen|cook|nousek|nosek|higgins|qiu2025modserve)\" docs/checkpoints/paper_drafts/paper.bib" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
4:@misc{yang2023som,
15:@inproceedings{koh2024visualwebarena,
27:@inproceedings{zhou2024webarena,
39:@inproceedings{deng2023mind2web,
51:@inproceedings{zheng2024seeact,
67:@inproceedings{yang2025magma,
159:@misc{qiu2025modserve,
316:@article{drouin2024workarena,

exec
/bin/bash -lc "rg -n \"@\\w+\\{\" docs/checkpoints/paper_drafts/paper.bib | sed -n '1,220p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
4:@misc{yang2023som,
15:@inproceedings{koh2024visualwebarena,
27:@inproceedings{zhou2024webarena,
39:@inproceedings{deng2023mind2web,
51:@inproceedings{zheng2024seeact,
67:@inproceedings{yang2025magma,
79:@inproceedings{li2025ferretui2,
91:@inproceedings{sclar2024promptformat,
103:@inproceedings{mishra2022reframing,
113:@inproceedings{ong2025routellm,
125:@article{chen2023frugalgpt,
137:@misc{kerboua2025focusagent,
148:@misc{li2026avenirweb,
159:@misc{qiu2025modserve,
170:@misc{yang2023dawnoflmms,
181:@misc{chen2026graph2eval,
197:@article{asadi2026mirageillusionvisualunderstanding,
205:@inproceedings{kaduri2024whatsintheimage,
214:@article{liu2024paying,
221:@inproceedings{wang2025xlrsbench,
229:@article{lu2025agentrewardbench,
236:@inproceedings{liu2025planandact,
243:@inproceedings{anonymous2025vilp,
250:@inproceedings{zheng2024uground,
257:@inproceedings{prabhu2026walt,
266:@article{furuta2024exposing,
275:@article{kate2025howgood,
283:@inproceedings{agentoccam2025,
292:@article{wang2024webquest,
300:@article{cai2024personalwab,
308:@article{su2025limits,
316:@article{drouin2024workarena,
324:@article{dhondt2024agente,
331:@article{qiao2025visescape,
338:@article{pan2024webcanvas,
346:@inproceedings{li2024effects,
353:@article{tan2026hmt,
361:@article{vu2026scaffold,
368:@inproceedings{tong2024eyes,
375:@article{jia2026decoding,
382:@article{alnazi2026omni,
389:@article{zhang2026factcheck,
396:@article{shu2025vqa,
403:@article{wei2026clawsafety,
411:@article{shukor2025scaling,
418:@inproceedings{luo2025probing,
425:@article{feng2025visually,
432:@article{chen2026analyzing,
439:@article{liu2026adaptive,
446:@article{li2025seetrek,
453:@inproceedings{anon2026inference,
463:@inproceedings{guan2024hallusionbench,
475:@inproceedings{fu2024blink,
488:@inproceedings{li2023pope,
502:@inproceedings{bittonguetta2023whoops,
514:@misc{liu2025devils,
525:@inproceedings{anon2026toolcalling,

exec
/bin/bash -lc 'test -e docs/checkpoints/paper_drafts/section8_limitations.md && wc -w docs/checkpoints/paper_drafts/section8_limitations.md || true' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
codex
The source files line up with the requested structure: external validity is Qwen-family/VWA-scoped, evaluator threats are handled through raw-plus-adjusted reporting, and the mechanism layer story must be explicitly post-hoc with L11/L17 treated as convergent rather than preregistered. I’m now drafting the consolidated body section and keeping it under the 1100-word target.
exec
/bin/bash -lc "rg -n \"G8|sparse|Levenshtein|IQR|median.*0|25%|strong-tier|4 of 5|4/5|task subsets\" docs/checkpoints/pre_run/topvenue_constraints.md docs/checkpoints/实验笔记.md docs/checkpoints/paper_drafts/section5_mechanism_reddit.md docs/analysis -g '*.md'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:7:Each layer/task pair is scored against the two unpatched continuations. `token_overlap_to_source` is the fraction of generated positions where the patched output matches the source token sequence, and `token_overlap_to_target` is the same fraction against the target sequence. `ld_to_source` and `ld_to_target` are token-level Levenshtein distances from patched output to source or target, bounded from 0 to 50 under the fixed 50-token continuation. Higher `token_overlap_to_target` and lower `ld_to_target` mean the patch had no effect; the disruption signal is a drop in target overlap or a rise in target Levenshtein distance.
docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:9:The layer-wise statistics follow the preregistration §4 bootstrap policy lock. For each direction and metric, tested layer L is compared with L35 using paired per-task differences L - L35. The canonical grid is L0, L5, L11, L17, L23, and L29, with Holm-Bonferroni correction across these six tests. For overlap metrics the paired t-test uses `alternative='less'`; for Levenshtein metrics it uses `alternative='greater'`. We report 1000-sample task-paired percentile bootstrap 95% confidence intervals on the mean paired difference, and Wilcoxon signed-rank is the non-parametric backup. The resampling unit is the task, not the token or episode.
docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:43:The macro and trajectory evidence explain why. In the cascade, the prompt axis dominates the reddit search-loop and typing shifts: search-loop falls by 13.81pp from P-text to P-SoM, type fraction falls by 6.58pp, and scroll fraction falls by 3.79pp (Macro 1b, prompt-axis dominant on search/type/scroll). The action-vocabulary decomposition shows the same policy change in local form: P-SoM reduces type by 0.0658 and increases tab focus by 0.0828 relative to P-text (Macro 1d, action-vocabulary shift). The per-task boundary analysis shows that this is often an early decision-prior change, not a late recovery: in the P-text versus P-SoM symmetric-difference set, the median first divergent step is 0 and all divergent cases are early (Micro 2f, N=15, median first divergent step 0, early divergence 100%).
docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:59:The positive side should also be kept in scope. The macro image axis reduces search and typing and changes first actions substantially, and the symmetric-difference set between P-SoM and SoM contains successful SoM-only tasks (Micro 2f, P-SoM versus SoM N=23, median first divergent step 0). The point is not that screenshots are useless on reddit; it is that the site does not organize navigation through visual product cards or spatial filters. The image channel can disambiguate content, but it is not the dominant route planner.
docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:67:The compound macro signature is exactly the one expected from this interaction. Relative to DOM, Phantom-SoM types less and uses more tab focus while leaving some coarse frequencies close to the endpoints (Macro 1d, compound type -0.089 and tab_focus +0.071). A router can exploit this complementarity because the behavioral difference appears before final success labels: the symmetric-difference set diverges early between DOM and P-SoM (Micro 2f, DOM<->P-SoM N=23, median first divergent step 0).
docs/checkpoints/pre_run/topvenue_constraints.md:9:- ✓ Done: 50 constraints (post-2026-05-09: G6 + G7 cell E + 2x2; B2 bootstrap policy lock; A14 + F3 reproducibility / external-validity scope; H2 negative-results registry; F7 + H3 ethics/license/COI/misuse-safety; G5 post-hoc layer disclosure; D8 + H5 section1 intro rewrite; B6 + B7 + B8 pre-spec policy lock; A5 + F8 version pinning; A11 + A12 + A13 release cards; G9 token alignment; G8 sparse-mechanism heterogeneity; C10 pre-launch + validate-strict Make targets; G1 §5.1 method box codex-delegated)
docs/checkpoints/pre_run/topvenue_constraints.md:55:| B4 | Use paired tests for same-task mode comparisons | `koh2024visualwebarena`; `zhou2024webarena`; NEEDS_BIB_ENTRY: McNemar 1947 | ✓ | `paper_drafts/section3_definition.md §3.5` and `preregistration.md §4` require same-task subsets; `axis1_microbehavior.py` and `aggregate_phantom_lift.py` include paired/McNemar/bootstrap infrastructure. | "All mode comparisons are task-paired; missing cells are excluded from the common universe rather than imputed." |
docs/checkpoints/pre_run/topvenue_constraints.md:59:| B8 | Analyze heterogeneity and avoid overclaiming pooled averages | NEEDS_BIB_ENTRY: DerSimonian & Laird 1986; NeurIPS checklist Q7 | ✓ | Locked 2026-05-09 in `preregistration.md §4`: random-effects DerSimonian-Laird primary, Cochran Q + I² + τ² reported, Higgins & Thompson 2002 I² thresholds (< 25% pooled / 25-50% both / 50-75% per-cell primary / > 75% no pool + heterogeneity-source decomposition by site/model/task-pool). Per-cell estimates always shown alongside pooled, so heterogeneity never averaged away. | "Random-effects with pre-specified I² thresholds — per-cell + heterogeneity-source decomposition reported when between-cell variance is high." |
docs/checkpoints/pre_run/topvenue_constraints.md:89:| D5 | Avoid cherry-picking task subsets or cells | NeurIPS checklist Q1/Q7; NEEDS_BIB_ENTRY: Lipton & Steinhardt 2018 | ⚠️ | `preregistration.md §4` defines cell inclusion and N floor; `pre_rerun_audit.md §4.8.2` counterfactual cell-removal stability is TBD. Remediation: run leave-one-cell-out decision test; cost 2h. | "All cells meeting locked criteria are included; leave-one-cell-out stability is reported as a falsification check." |
docs/checkpoints/pre_run/topvenue_constraints.md:125:| G1 | Define clean/corrupt/source/target prompts and behavioral metric for activation patching | NEEDS_BIB_ENTRY: Wang et al. 2023; NEEDS_BIB_ENTRY: Zhang et al. 2024 | ✓ | `paper_drafts/section5_mechanism_reddit.md §5.1 Method` (added 2026-05-09 via codex `section5_method_box.md` prompt): 7-paragraph method box covering source (`som`)/target (`phantom_som`) prompt definitions / 36-layer L0-L35 hidden state cache / forward+reverse direction / 4 metrics (token_overlap_to_source/target + LD_to_source/target) / Holm-Bonferroni paired t / 1000-sample task-paired percentile bootstrap / Wilcoxon backup / composite score curation / 5-cell A/B/C/D/E design / random-injection specifics (`randn_like(h) * h.std() + h.mean()`) / pre-vs-post-hoc layer disclosure cite §5.X. NEEDS_BIB_ENTRY flagged for `wang2023interpretability` + `zhang2024patching`. | "Section 5.1 defines the patching protocol explicitly: source = som with image, target = phantom_som without; per-layer hidden state cache + greedy continuation 50 tokens + token overlap / Levenshtein metrics; Holm-corrected paired test + bootstrap CI; cells A-E control selection-bias and content-specificity." |
docs/checkpoints/pre_run/topvenue_constraints.md:130:| G6 | Include null/random-injection controls | NEEDS_BIB_ENTRY: Heimersheim & Janiak 2024; NEEDS_BIB_ENTRY: Zhang et al. 2024 | ✓ | Cell E (job 335404, completed 2026-05-09 11:41) ran forward × strong-tier 24 task with Gaussian-matched random source hidden injection; results in `results/mechanistic/stage2b_celle_random_cls_strong_myriad/`. Random injection at any layer collapses target output (overlap→tgt 1.00→0.03-0.08, LD→tgt 0→17-36) vs real source's mild perturbation (L17 overlap 0.86, LD 6.0). Specificity ratio random-LD/real-LD = 5-19× across layers, demonstrating mid-layer L11/L17 disruption is content-specific, not generic injection effect. | "Cell E random-injection control collapses target output universally (overlap→tgt < 0.1) while real source patching at mid-layers preserves 86%+ overlap, ruling out 'any patching disrupts' interpretation." |
docs/checkpoints/pre_run/topvenue_constraints.md:132:| G8 | Report task heterogeneity, not only mean layer effects | NEEDS_BIB_ENTRY: Zhang et al. 2024; NEEDS_BIB_ENTRY: Wang et al. 2023 | ✓ | `scripts/analysis/stage2_heterogeneity_figure.py` (created 2026-05-09) generates 4-panel scatter+violin+median/IQR figures alongside existing mean+std band. Applied to all 5 done cells. **Critical finding**: cells A/B/C/D have median LD→tgt = 0 at L17 (mean carried by outlier tasks) — **mechanism is sparse task-conditional, not uniform across tasks**. Cell E random median = 38/50 (uniform destruction). Refines paper §5 framing: real source patching engages mid-layer mechanism only for high-salience-image tasks, sparse activation on most. | "Per-task scatter + violin overlay shows real-source mid-layer disruption is sparse (few tasks fully affected, median ≈ 0, mean carried by outliers); random-injection destroys all tasks (median = 38/50). Sparse content-specificity ≠ uniform circuit." |
docs/checkpoints/实验笔记.md:1718:- A 类（数据生成端 RU-1/4/5、VW-1，已污染历史 episode_summary，需修代码 + 写 `rederive_episode_summary.py` 从 step JSONL 重派生 page_unchanged_rate / energy_partial；RU-4 latency 缺 busy-wait 只能补标志位 + 论文披露）
docs/checkpoints/实验笔记.md:1727:- **代码改动**：`runner.py` (RU-1/2/4/5/6/7) / `vwa_wrapper.py` (VW-1/3) / `analysis.py` (A-1/2/3/4/5) / `analyze_reason_diagnostics.py` (RD-1~6/12/13) / `types.py` (EpisodeSummaryV2 加 `busy_wait_total_ms`/`energy_partial`/`energy_step_complete_count`) / 新建 `scripts/maintenance/rederive_episode_summary.py`
docs/checkpoints/实验笔记.md:3792:- VWA empirical: text-only 7.25% / multimodal 16.37% = **44% relative retention** (more conservative than 70-80% on real web)
docs/checkpoints/实验笔记.md:3929:**Codex queue task #10 状态**: 整合 done. 之前 estimated ~50-100K tokens (post Q1-4+6 reduction), Q5 add ~10K tokens. 实际 manual integration done by me (~30 min, no codex tokens spent on integration). Codex token 可用于 Section 4/5 prose 写作 (#11 Section 4 + #13 Section 5 mechanism).
docs/checkpoints/实验笔记.md:4470:| arXiv IDs (8 抽样) | **0/8 真** ❌ | 每个 V2 给的 ID fetch 后都指向无关论文 (plasma 物理 / sparse autoencoder 病理 / 天体物理 VLBI / clonal interference 数学 / Adaptive-k QA / Test-Time Compute / agent watermarking / graph certification) |
docs/checkpoints/实验笔记.md:4743:- 跟 source token-by-token 93% overlap, Levenshtein distance 1/15 token
docs/checkpoints/实验笔记.md:4786:> "Activation patching of source's (SoM with image) hidden state into target's (P-SoM no image) run at layer 11 (Qwen3-VL-4B, 31% depth) recovers source's full output continuation token-by-token in clean mirage case studies (cls task 0: 93% token overlap, Levenshtein 1/15). Patching at any other layer (L0-L10, L12-L35) has null effect on patched output, identifying layer 11 as the causal mirage layer where image-grounded perception emerges before downstream abstraction. Layer 11 placement is consistent with Stage 2A first-token logit shift L17 peak (5-task aggregate, +1.08±0.21) and Tool Calling Linear Steerable Circuit (Anonymous 2026 ICLR, Qwen3-4B L23 family) mid-layer 'computed feature emergence' zone."
docs/checkpoints/实验笔记.md:5608:DGX→quark Tailscale→Myriad SSH chain newly opened (笔记 §117.6 infra). Stage 2B forward (24 strong-tier tasks) + Stage 2C reverse (15 reverse-tier tasks) run on Myriad A100 80GB compute node `node-v00a-001` / `node-u00a-001`. ~30min wallclock each (V100 estimate had been 24h — A100 80GB ~48× speedup). Output dirs:
docs/checkpoints/实验笔记.md:5658:Composite is **task-text-only** (no patching leak), but plausibly correlates with patching effect size — strong-tier may be high-effect-size subset. Bidirectional finding could reflect mechanism universality OR task curation correlation.
docs/checkpoints/实验笔记.md:5664:- Cell D (new): reverse × strong-tier (24) — qsub 335340, qw
docs/checkpoints/实验笔记.md:5677:2. **Task heterogeneity**: std band ±0.22 ≈ 1.5× effect size at L17. Some tasks no disrupt, some near-flip. Need median+IQR alternative reporting OR per-task scatter alongside aggregate.
docs/analysis/experiment_matrix.md:21:| | som | **234/234** | 17.52% (41) | **13.25%** (31) | ✅ 完成 |
docs/analysis/cross_sites/tier5_evaluator_audit.md:10:`string_match` is not Levenshtein/token-Jaccard fuzzy matching. It is a mixture of exact lowercase comparison, token/substr containment, AND-over-list / OR-within-entry logic, and an LLM fuzzy judge. There is no normalized edit-distance threshold. In runner semantics, success requires total evaluator score `>= 1.0`; for the LLM fuzzy path the effective threshold is binary `1.0` if the completion contains `correct` and not `incorrect` / `partially correct`.
docs/analysis/cross_sites/tier5_evaluator_audit.md:98:- Effective fuzzy threshold: `1.0` binary label. There is no Levenshtein, edit distance, token Jaccard, embedding similarity, or numeric threshold.
docs/analysis/cross_sites/site_mechanism_dictionary.md:11:- Shopping is sparse: only `vwa_shopping/B0_findings.md` and `B0_DOM_digest.md` exist.
docs/analysis/cross_sites/site_mechanism_dictionary.md:587:Shopping product choice often depends on color, shape, pattern, packaging, and grid position. DOM can use product names and sparse alt text only; many task-critical attributes are absent from the AXTree. Therefore the image axis is expected to be decisive once SoM/Vision data exist.
docs/analysis/cross_sites/site_mechanism_dictionary.md:669:| Axis 1 text | PRIMARY: URL Jaccard 0.573, click-target Jaccard 0.463, +P-text +3.81pp oracle | Secondary: URL Jaccard 0.904, click-target Jaccard 0.561, +P-text +3.42pp oracle | Mixed/form-action; sparse evidence: click failure 24.7%, swatch bug 11 episodes |
docs/analysis/cross_sites/site_mechanism_dictionary.md:709:Shopping is sparse, but the failure audit makes its substrate visible. Magento combines visual product/variant identification with brittle form and cart interactions. DOM fails because many product attributes are absent from text, while the swatch audit shows a separate scaffold problem in form-state tracking. Treat shopping as forward-looking until B0/B1 5-mode cells exist.
docs/analysis/cross_sites/site_mechanism_dictionary.md:717:- Shopping needs B0/B1 5-mode data. Current mechanism statements are sparse and mostly forward-looking.
docs/analysis/cross_sites/site_mechanism_dictionary.md:733:| Shopping sparse caveat | explicit |
docs/analysis/cross_sites/probe_audit_verification.md:5:This replay pass treats Tier 1/2/4/5 outputs as candidate signatures, not fix scope. It used Playwright with the recorded `.auth` storage state, task `start_url`, logged prior actions where feasible, and live DOM/scroll/form snapshots at the target step. No framework code was changed.
docs/analysis/phantom_paper/B0_dom_shopping_diagnostic.md:71:- The A color-flag subset is 32 tasks with 6.25% adjusted SR; A no-color is only slightly better at 10.0%, so visual attributes hurt but do not fully explain the 8.54% bucket-level SR.
docs/analysis/phantom_paper/cross_site_pattern_consolidation.md:59:| search-loop | sparse but cross-site | classifieds: B0-61(B), B0-64(B), B0-167(B), B0-217(A), B1-40(D), B1-111(C); reddit: B0-14(B), B0-124(C), B0-162(A) | Search-loop is not purely a Postmill artifact. Reddit's slower Postmill navigation can amplify repeated search, but OSClass also induces repeated search/result scanning when item listings are visually dense or the target is described by reference-image attributes. |
docs/analysis/phantom_paper/cross_site_pattern_consolidation.md:97:4. **Non-universal: DOM search-loop in the exclusive slice.** DOM search-loop is sparse and changes direction by site (+6.3 pp on classifieds, -13.3 pp on reddit). Section 4 should avoid claiming that DOM search-loop dominates disagreement failures universally; it is better used as a whole-run trajectory-gradient finding, with disagreement failures skewing toward visual-missing.
docs/analysis/phantom_paper/cross_site_pattern_consolidation.md:99:## Implications for Section 4/5 Claims
docs/analysis/vwa_reddit/B1_DOM_digest.md:79:| page_reading | 0%（0/5） | 0% | 6.25%（1/16） | 6.25% |
docs/analysis/layered_evidence_status.md:155:- reddit: SoM median tokens/step 4301 vs P-SoM 3522; observed gap **778 tokens/step**
docs/analysis/layered_evidence_status.md:156:- classifieds: SoM median tokens/step 3975 vs P-SoM 3032; observed gap **943 tokens/step**
docs/analysis/vwa_classifieds/B1_Vision_digest.md:154:| Best single (SoM adjusted) | 13.25% |
docs/analysis/vwa_classifieds/B0_B1_findings.md:10:> **数据更新 (2026-04-26)**：04-26 全 condition rederive。当前 adjusted SR：B0 DOM 14.10% / SoM **21.37%** / Vision 13.68% · B1 DOM 8.55% / SoM **13.25%** / Vision 7.26%（漂移 <1.7pp，文字结论不变）。Mirage Gap：B0 +7.27pp / B1 +4.70pp（两个模型都强 SoM 优势）。
docs/analysis/vwa_classifieds/B1_findings.md:11:> **数据更新 (2026-04-26)**：04-26 全 condition rederive。当前 adjusted SR：DOM **8.55%** / SoM **13.25%** / Vision **7.26%**（v1: 7.59% / 13.84% / 7.14%，漂移 <1.0pp，三模式排序与结论不变，SoM 仍领先）。
docs/analysis/vwa_classifieds/B1_findings.md:132:| Best single (SoM) | 17.52% | 13.25% |

exec
/bin/bash -lc 'rg -n "Cell E|random|specificity|Gaussian|L17|median|IQR" docs/checkpoints/实验笔记.md docs/checkpoints/pre_run/topvenue_constraints.md docs/checkpoints/paper_drafts/section5_mechanism_reddit.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:9:The layer-wise statistics follow the preregistration §4 bootstrap policy lock. For each direction and metric, tested layer L is compared with L35 using paired per-task differences L - L35. The canonical grid is L0, L5, L11, L17, L23, and L29, with Holm-Bonferroni correction across these six tests. For overlap metrics the paired t-test uses `alternative='less'`; for Levenshtein metrics it uses `alternative='greater'`. We report 1000-sample task-paired percentile bootstrap 95% confidence intervals on the mean paired difference, and Wilcoxon signed-rank is the non-parametric backup. The resampling unit is the task, not the token or episode.
docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:11:Task curation uses `scripts/mechanistic/curate_mirage_tasks.py`. The composite score is `((src_neg - tgt_neg) + (tgt_aff - src_aff)) * (1 + divergence)`, where divergence is one minus source/target token overlap. Strong-tier tasks satisfy `composite >= 1.0` and overlap `< 0.5`; reverse-tier tasks satisfy `composite <= -1.5`. Per `dataset_card.md`, the released subsets contain classifieds 24 strong plus 15 reverse tasks, and reddit 47 strong plus 48 reverse tasks. The cell design is a 2-by-2 direction-by-tier control plus a random control: A = forward x classifieds-strong (N=24); B = reverse x classifieds-reverse (N=15); C = forward x classifieds-reverse (N=15), selection-bias control; D = reverse x classifieds-strong (N=24), selection-bias control; E = forward x classifieds-strong with random Gaussian source hidden states (N=24, seed=42), content-specificity control. Cells F/G are cross-site reddit and remain pending.
docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:13:For Cell E, each cached source tensor h is replaced by `randn_like(h) * h.std() + h.mean()` at the corresponding layer. This preserves activation magnitude while destroying task-specific structure. The specificity ratio is random-injection `ld_to_target` divided by real-source `ld_to_target`; it measures whether disruption comes from content-specific source information rather than generic non-zero injection.
docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:15:Finally, preregistration §5.X discloses the layer selection as post-hoc. L11/L17 were not pre-registered: L17 emerged from the Stage 2A logit-shift pilot, and L11 from an early single-task continuation case. The confirmatory evidence is Stage 2B forward overlap, Stage 2C reverse overlap, and Cell D cross-tier replication. Together with the independent Stage 2A logit-shift path, this gives four convergent paths (logit shift, forward overlap, reverse overlap, cross-tier) under Holm-corrected grid testing, but not a pre-registered exact-layer prediction.
docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:43:The macro and trajectory evidence explain why. In the cascade, the prompt axis dominates the reddit search-loop and typing shifts: search-loop falls by 13.81pp from P-text to P-SoM, type fraction falls by 6.58pp, and scroll fraction falls by 3.79pp (Macro 1b, prompt-axis dominant on search/type/scroll). The action-vocabulary decomposition shows the same policy change in local form: P-SoM reduces type by 0.0658 and increases tab focus by 0.0828 relative to P-text (Macro 1d, action-vocabulary shift). The per-task boundary analysis shows that this is often an early decision-prior change, not a late recovery: in the P-text versus P-SoM symmetric-difference set, the median first divergent step is 0 and all divergent cases are early (Micro 2f, N=15, median first divergent step 0, early divergence 100%).
docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:53:The outcome result rejects a naive monotone-vision story. Full SoM is below Phantom-SoM on adjusted reddit success (Outcome 0a, SoM 10.48% versus P-SoM 13.81%, a -3.33pp regression after adding the image). The image still changes behavior: the cascade shows an increase in finish rate and fewer steps when the image is added (Macro 1b, image-axis finish rate +10.95pp; step count -1.85), but on reddit this can mean earlier commitment from the wrong state rather than more correct routing. The click fraction also falls under the image axis (Macro 1b, click fraction -3.08pp), which is consistent with visual confidence substituting for link-following in some traces. The cost side makes the tradeoff concrete: adding the screenshot increases median token load without a reddit SR benefit (Efficiency 3b, SoM versus P-SoM observed gap 778 tokens/step).
docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:59:The positive side should also be kept in scope. The macro image axis reduces search and typing and changes first actions substantially, and the symmetric-difference set between P-SoM and SoM contains successful SoM-only tasks (Micro 2f, P-SoM versus SoM N=23, median first divergent step 0). The point is not that screenshots are useless on reddit; it is that the site does not organize navigation through visual product cards or spatial filters. The image channel can disambiguate content, but it is not the dominant route planner.
docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:67:The compound macro signature is exactly the one expected from this interaction. Relative to DOM, Phantom-SoM types less and uses more tab focus while leaving some coarse frequencies close to the endpoints (Macro 1d, compound type -0.089 and tab_focus +0.071). A router can exploit this complementarity because the behavioral difference appears before final success labels: the symmetric-difference set diverges early between DOM and P-SoM (Micro 2f, DOM<->P-SoM N=23, median first divergent step 0).
docs/checkpoints/pre_run/topvenue_constraints.md:22:| ~~5~~ | ~~G6 mechanistic random-injection/null controls~~ | ✓ done 2026-05-09 (cell E) | — |
docs/checkpoints/pre_run/topvenue_constraints.md:35:| A3 | Pin random seeds and disclose stochasticity limits | NeurIPS checklist Q6-Q7; NEEDS_BIB_ENTRY: Pineau et al. 2021 | ⚠️ | `configs/exp_v2_base.yaml` sets `seed: 42` and `runner/main.py::_seed_global_rng()` seeds Python/NumPy/Torch; `master_bug_catalog.md B-37` notes B0 proxy seeding remains best-effort and Anthropic-native seed support may be absent. Remediation: disclose B0 stochasticity and run B0 T=0 pilot sensitivity; cost 0.5-1 day. | "B1 is locally seeded/greedy; B0 uses best-effort API seed forwarding, so aggregate claims are reported as one controlled stochastic sample with bootstrap task uncertainty." |
docs/checkpoints/pre_run/topvenue_constraints.md:53:| B2 | Define confidence interval method, resampling unit, and assumptions | NeurIPS checklist Q7; NEEDS_BIB_ENTRY: Efron/Tibshirani bootstrap | ✓ | Locked 2026-05-09 in `preregistration.md §4`: 1000-resample percentile bootstrap (BCa as sensitivity check), task-level paired pairing across modes, single-level (task_id) clustering — no nested (cell × site) cluster bootstrap because meta-analysis at cell level uses random-effects + I²/τ² separately. | "Percentile + task-paired bootstrap is reported primary; BCa shown as appendix sensitivity. Cell-level uncertainty enters via random-effects meta-analysis, not nested bootstrap." |
docs/checkpoints/pre_run/topvenue_constraints.md:59:| B8 | Analyze heterogeneity and avoid overclaiming pooled averages | NEEDS_BIB_ENTRY: DerSimonian & Laird 1986; NeurIPS checklist Q7 | ✓ | Locked 2026-05-09 in `preregistration.md §4`: random-effects DerSimonian-Laird primary, Cochran Q + I² + τ² reported, Higgins & Thompson 2002 I² thresholds (< 25% pooled / 25-50% both / 50-75% per-cell primary / > 75% no pool + heterogeneity-source decomposition by site/model/task-pool). Per-cell estimates always shown alongside pooled, so heterogeneity never averaged away. | "Random-effects with pre-specified I² thresholds — per-cell + heterogeneity-source decomposition reported when between-cell variance is high." |
docs/checkpoints/pre_run/topvenue_constraints.md:60:| B9 | Run power/MDE analysis for effect detectability | NeurIPS checklist Q7; CONSORT-AI/Liu et al. 2020 | ✓ | `docs/analysis/cross_sites/power_analysis.md` (consolidated 2026-05-09) reports observed-SR-conditioned MDE 4-7pp at 80% per-cell power across 3 sites + family-wise K-of-N power table at observed effect sizes (1-5pp) **<10%** → triggered methodological reframing: K-of-N rule retained as secondary transparency check, primary detection via B8 random-effects meta + TOST on N=910 pooled tasks. Bug fix: `power_analysis.py` stale interpretation block (claimed >0.95 K=12/16 family power at p_per=0.65; actual 0.289) corrected in same commit. Preregistration §4 updated with B9 row locking the corrected framing. | "Per-cell MDE 4-7pp at observed SR (8-15%); K-of-N rule documented as secondary transparency, primary detection via random-effects meta + pooled TOST." |
docs/checkpoints/pre_run/topvenue_constraints.md:125:| G1 | Define clean/corrupt/source/target prompts and behavioral metric for activation patching | NEEDS_BIB_ENTRY: Wang et al. 2023; NEEDS_BIB_ENTRY: Zhang et al. 2024 | ✓ | `paper_drafts/section5_mechanism_reddit.md §5.1 Method` (added 2026-05-09 via codex `section5_method_box.md` prompt): 7-paragraph method box covering source (`som`)/target (`phantom_som`) prompt definitions / 36-layer L0-L35 hidden state cache / forward+reverse direction / 4 metrics (token_overlap_to_source/target + LD_to_source/target) / Holm-Bonferroni paired t / 1000-sample task-paired percentile bootstrap / Wilcoxon backup / composite score curation / 5-cell A/B/C/D/E design / random-injection specifics (`randn_like(h) * h.std() + h.mean()`) / pre-vs-post-hoc layer disclosure cite §5.X. NEEDS_BIB_ENTRY flagged for `wang2023interpretability` + `zhang2024patching`. | "Section 5.1 defines the patching protocol explicitly: source = som with image, target = phantom_som without; per-layer hidden state cache + greedy continuation 50 tokens + token overlap / Levenshtein metrics; Holm-corrected paired test + bootstrap CI; cells A-E control selection-bias and content-specificity." |
docs/checkpoints/pre_run/topvenue_constraints.md:129:| G5 | Avoid post-hoc layer selection by using independent confirmation or disclosure | NEEDS_BIB_ENTRY: Zhang et al. 2024; NEEDS_BIB_ENTRY: Wang et al. 2023 | ✓ | `preregistration.md §5.X` (added 2026-05-09) explicitly maps L11/L17 selection chain: Stage 2A logit_shift (hypothesis-generating) → §111 task-0 (single-task outlier, acknowledged) → Stage 2B 24-task / 2C 15-task / Cell D 24-task (3 confirmatory cells, all Holm-significant L17 + cell D L11). Holm-Bonferroni across 6 canonical layers handles multiple-comparison; convergence of 4 independent analysis paths (logit_shift, forward overlap, reverse overlap, cross-tier) constitutes confirmation. Cell E random-injection rules out generic-injection alternative. | "Layers L11/L17 are not pre-registered but emerged from Stage 2A pilot and are confirmed by 3 independent Stage 2B/2C/Cell-D tests with Holm correction — see preregistration §5.X." |
docs/checkpoints/pre_run/topvenue_constraints.md:130:| G6 | Include null/random-injection controls | NEEDS_BIB_ENTRY: Heimersheim & Janiak 2024; NEEDS_BIB_ENTRY: Zhang et al. 2024 | ✓ | Cell E (job 335404, completed 2026-05-09 11:41) ran forward × strong-tier 24 task with Gaussian-matched random source hidden injection; results in `results/mechanistic/stage2b_celle_random_cls_strong_myriad/`. Random injection at any layer collapses target output (overlap→tgt 1.00→0.03-0.08, LD→tgt 0→17-36) vs real source's mild perturbation (L17 overlap 0.86, LD 6.0). Specificity ratio random-LD/real-LD = 5-19× across layers, demonstrating mid-layer L11/L17 disruption is content-specific, not generic injection effect. | "Cell E random-injection control collapses target output universally (overlap→tgt < 0.1) while real source patching at mid-layers preserves 86%+ overlap, ruling out 'any patching disrupts' interpretation." |
docs/checkpoints/pre_run/topvenue_constraints.md:131:| G7 | Control for task-selection bias with crossed task-tier × direction design | NEEDS_BIB_ENTRY: Zhang et al. 2024; NEEDS_BIB_ENTRY: Conmy et al. 2023 | ✓ | 4-cell 2x2 design completed 2026-05-09: A (fwd×strong N=24, Holm L17 p=0.011 \*\*) + B (rev×reverse N=15, Holm L11+L17) + C (fwd×reverse N=15, Δ=-0.10 weaker) + D (rev×strong N=24, Holm L11+L17 p=0.006/0.008 \*\*). Cross-cell Welch p=0.509-1.000 (NS) — magnitudes statistically indistinguishable across direction × tier; reverse direction Δ=-0.193 identical on both tiers (p=1.000), forward Δ=-0.143/-0.103 not Welch-distinguishable across tiers (p=0.577). Selection bias rejected; mechanism claim universal pending cross-site (cells F/G). | "The crossed 2x2 design rules out selection-bias artifact: reverse direction produces identical magnitude on both task tiers (Welch p=1.000), forward direction is Welch-indistinguishable across tiers (p=0.577); only Holm-significance differs due to N power asymmetry." |
docs/checkpoints/pre_run/topvenue_constraints.md:132:| G8 | Report task heterogeneity, not only mean layer effects | NEEDS_BIB_ENTRY: Zhang et al. 2024; NEEDS_BIB_ENTRY: Wang et al. 2023 | ✓ | `scripts/analysis/stage2_heterogeneity_figure.py` (created 2026-05-09) generates 4-panel scatter+violin+median/IQR figures alongside existing mean+std band. Applied to all 5 done cells. **Critical finding**: cells A/B/C/D have median LD→tgt = 0 at L17 (mean carried by outlier tasks) — **mechanism is sparse task-conditional, not uniform across tasks**. Cell E random median = 38/50 (uniform destruction). Refines paper §5 framing: real source patching engages mid-layer mechanism only for high-salience-image tasks, sparse activation on most. | "Per-task scatter + violin overlay shows real-source mid-layer disruption is sparse (few tasks fully affected, median ≈ 0, mean carried by outliers); random-injection destroys all tasks (median = 38/50). Sparse content-specificity ≠ uniform circuit." |
docs/checkpoints/pre_run/topvenue_constraints.md:153:5. ~~❌ G6 — Run random-injection/null control for Stage 2 patching~~ ✓ DONE 2026-05-09 cell E (specificity ratio 5-19×).
docs/checkpoints/实验笔记.md:1740:- **`analyze_confidence_calibration.py`**：CC-1 `_rank_biserial` sign 反转修正（公式 `2U/(n1n2)-1`，验证 B0 cls `ep_mean_verbalized` rb=+0.49 正向）/ CC-4 benchmark 推断从 run_dir.parts（不再硬编码 vwa）/ CC-5 多 site per-site adjustment（不再硬编码 fallback `classifieds`）/ CC-6 `overall_usable` 加入 `mode_invariant!=False` 守卫 / CC-7 `_load_episode_summaries` task_id 防御（缺失/无效 skip 不 collide 到 -1）/ CC-2 `_optimal_threshold` 加 LOO-CV + bootstrap CI（threshold_loo_mean/std、sensitivity_loo、specificity_loo、threshold_ci_lower/upper、validation 字段）/ CC-3 `c10_composite_signals` 加 validation="in_sample" + rank + n_combinations_searched 列 + stdout disclaimer / CC-12 `unlink()` 防 IsADirectoryError
docs/checkpoints/实验笔记.md:2827:- Section 3.2 image-token estimate 从 total cost 反推 → step-level `tokens.input` median (`4d63c9f`)
docs/checkpoints/实验笔记.md:3385:- `runner/main.py`: `_seed_global_rng(seed)` per (condition, seed) — Python random + numpy + torch.manual_seed + cuda.manual_seed_all
docs/checkpoints/实验笔记.md:3389:- B-37 verification: grep `random.seed/np.random.seed/torch.manual_seed/set_seed` 在 p79/ 全 zero matches; 18 B0 configs 全用 T=0.1; payload 不传 seed; Anthropic API 协议根本没 seed 参数 → paper "seed=42 reproducibility" claim 之前是 metadata-only
docs/checkpoints/实验笔记.md:3869:   - Text-over-vision (M2 language prior override): Tong 2024 / Fu 2024 BLINK 24-30% near random / Bitton-Guetta 2023 WHOOPS! / MM-Vet OCR fallback
docs/checkpoints/实验笔记.md:4721:### 111.3 Stage 2A First-Token Logit-Shift Patching — L17 peak weak signal
docs/checkpoints/实验笔记.md:4729:- L11: +0.640 / **L17: +1.080 (peak)** / L23: +0.753 / L29: +0.617 / L35: +1.000
docs/checkpoints/实验笔记.md:4731:L17 peak std=0.214 (5/5 task ≥ 0.9), robust 跨 task. **Mid-layer "computed feature emergence" pattern** 跟 mechanistic 文献一致 (Tool Calling Linear Steerable Circuit Anonymous 2026 在 Qwen3-4B 定位 L23 是同类 finding). 但 absolute scale 弱 因为 source/target distribution 太接近.
docs/checkpoints/实验笔记.md:4744:- L0 / L5 / L17+: patched 完全 = target output (60% baseline overlap, no patch effect)
docs/checkpoints/实验笔记.md:4756:L17 (Stage 2A logit_shift peak, 5 task aggregate) vs L11 (Stage 2B continuation, 1 task case study) — 距离 6 layer ~17% depth gap. 不是同一精确层但**同一 mid-early region** (L11-L25), 跟 Tool Calling Linear Circuit (Qwen3-4B L23) 都属于 transformer 中段 "computed feature emergence" zone. 文献 cross-paper anchor.
docs/checkpoints/实验笔记.md:4786:> "Activation patching of source's (SoM with image) hidden state into target's (P-SoM no image) run at layer 11 (Qwen3-VL-4B, 31% depth) recovers source's full output continuation token-by-token in clean mirage case studies (cls task 0: 93% token overlap, Levenshtein 1/15). Patching at any other layer (L0-L10, L12-L35) has null effect on patched output, identifying layer 11 as the causal mirage layer where image-grounded perception emerges before downstream abstraction. Layer 11 placement is consistent with Stage 2A first-token logit shift L17 peak (5-task aggregate, +1.08±0.21) and Tool Calling Linear Steerable Circuit (Anonymous 2026 ICLR, Qwen3-4B L23 family) mid-layer 'computed feature emergence' zone."
docs/checkpoints/实验笔记.md:4986:- Capture: stage / direction (forward/reverse) / patch_config (source_mode / target_mode / layer indices / max_new_tokens / hook_position) / model.path + n_layers / input_dataset.task_ids / **outcomes_per_task** (per-task best_layer + L11/L17 overlap) / aggregate (best_layer mean / L11 mean / L17 mean) / env_snapshot_ref
docs/checkpoints/实验笔记.md:5300:- ✅ Cluster 4 (RNG seeding): `runner/main.py:81-94` random+numpy+torch seed
docs/checkpoints/实验笔记.md:5625:| **L17** | **0.86** | **0.81** | **6.0** | **6.9** |
docs/checkpoints/实验笔记.md:5630:Reverse direction shows **mid-layer L11-L17 disruption magnitude similar to forward** (Δoverlap→tgt: forward -0.143 ± 0.217 / reverse -0.193 ± 0.256 at L17). Welch's t @ L17 forward vs reverse p=0.535 (not distinguishable — consistent with bidirectional mechanism, but failure to reject ≠ proof of equality).
docs/checkpoints/实验笔记.md:5638:| Forward × overlap→tgt | **L17** (p_Holm=0.011 \*\*) | L11 marginal (p_Holm=0.092) |
docs/checkpoints/实验笔记.md:5639:| Forward × LD→tgt | **L17** (p_Holm=0.024 \*) | L11 marginal (p_Holm=0.080) |
docs/checkpoints/实验笔记.md:5640:| Reverse × overlap→tgt | **L11 + L17** (p_Holm=0.044 \* / 0.033 \*) | strongest signal |
docs/checkpoints/实验笔记.md:5641:| Reverse × LD→tgt | None Holm-significant | L17 raw p=0.014 marginal Holm 0.084 (N=15 power-limited) |
docs/checkpoints/实验笔记.md:5643:**L17 robust mid-layer mechanism layer** confirmed across 2 directions × 2 metrics (3 of 4 Holm-significant). L11 emerges as secondary in reverse but marginal in forward.
docs/checkpoints/实验笔记.md:5647:§111 reported "L11 flips 93% match in forward task 0" — that's **task-specific outlier**, not population pattern. 24-task forward average shows L11 overlap→tgt=0.908 (Δ=-0.093 from baseline), much weaker than task-0's near-flip. Distribution has long tail (some tasks fully flip, some don't disrupt). §111.5 "Stage 2A logit_shift L17 peak" is **consistent with N=24 finding** — both Stage 2A (5-task aggregate) and Stage 2B (24-task aggregate) point to L17 as primary.
docs/checkpoints/实验笔记.md:5649:Marked: §111 task-0 single-case study should NOT be cited as paper §5 representative finding. Use 24-task L17 aggregate + ±1σ band + Holm p-value instead.
docs/checkpoints/实验笔记.md:5667:- If C+D show similar L17 disruption → mechanism universal (paper §5 confident bidirectional claim)
docs/checkpoints/实验笔记.md:5676:1. **Effect size**: L17 forward overlap drop -0.143 ± 0.217 (~14% match degradation). Reviewer: "is 14% drop strong evidence?" Need to contrast with random-injection control (zero effect baseline).
docs/checkpoints/实验笔记.md:5677:2. **Task heterogeneity**: std band ±0.22 ≈ 1.5× effect size at L17. Some tasks no disrupt, some near-flip. Need median+IQR alternative reporting OR per-task scatter alongside aggregate.
docs/checkpoints/实验笔记.md:5679:4. **L11 vs L17 hierarchy**: Stage 2A peaked L17, Stage 2B forward L17 strongest, Stage 2B reverse both L11+L17. Possible "L11 = mirage emergence onset, L17 = mirage commitment / decision lock-in" two-stage interpretation. Speculative — paper section needs careful wording.

codex
I’m writing the new file now. The draft will cite existing VWA/WebArena-style references where possible and explicitly mark missing bibliography entries for checklist/statistics/patching citations instead of editing `paper.bib`.
apply patch
patch: completed
/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/checkpoints/paper_drafts/section8_limitations.md
diff --git a/docs/checkpoints/paper_drafts/section8_limitations.md b/docs/checkpoints/paper_drafts/section8_limitations.md
new file mode 100644
index 0000000000000000000000000000000000000000..b5f1526b2ec5bede526606fc5e12351774ee2057
--- /dev/null
+++ b/docs/checkpoints/paper_drafts/section8_limitations.md
@@ -0,0 +1,29 @@
+\subsection*{8.1 Scope \& external-validity bounds}
+
+This study is deliberately narrow: three sites (classifieds, reddit, shopping), one benchmark family (VisualWebArena plus WA-mini-style task structure), and two Qwen-family model classes: Qwen3-VL-4B locally and Qwen3-Omni-235B-Thinking through a proxy API. The blast radius is external validity, not the internal comparison: our claim is that a phantom routing space exists for Qwen-family agents on VWA-style tasks, not that every VLM, browser benchmark, or production site will expose the same arm ordering \citep{zhou2024webarena,koh2024visualwebarena,deng2023mind2web,drouin2024workarena}. The mechanism evidence in Section 5 is narrower still: it is B1-only, because the open-weight 4B model exposes activations and the B0 proxy model does not. This affects Sections 1, 5, and 6 by bounding language about universality and reproducibility: B1 behavior and mechanistic patching are byte-reproducible from released artifacts, while B0 is verifiable from traces and replayable subject to API access; cross-architecture claims, including GPT-4o-family claims, remain outside the paper.
+
+\subsection*{8.2 Construct validity \& evaluator threats}
+
+VWA success labels are imperfect measurements of task completion, especially around `ua_match` GPT-judge drift, the `string_match` `fuzzy_threshold` misnomer, brittle `program_html` selectors, and `finish_wrong_state` episodes. The blast radius is measurement-side: these four evaluator-class threats can flip individual labels or inflate raw success, but they do not redefine the task universe or the mode definitions; full appendix prose is kept in Section 4.X.1--4.X.4 rather than duplicated here. We therefore report both raw and adjusted success and isolate `na_fp`, `eval_fp`, and `visual_fp` filters as sensitivity layers, following the constraint-table rule that evaluator artifacts should be disclosed rather than hidden \citep{lipton2018troubling,neurips2024checklist}. This affects Sections 3 and 4: adjusted SR is the headline metric, raw SR remains visible, and claims that survive the filter ladder are defensible as representation effects rather than judge artifacts.
+
+\subsection*{8.3 Internal-validity threats: known scaffold bugs}
+
+Several scaffold bugs were real: the `in_viewport_ratio` operator-precedence bug exposed clipped DOM text, early scroll actions suffered direction-convention confusion, and Stage 2B/2C mechanism inputs came from pre-Phase-A archived browser states. The blast radius is bounded because these failures are mode-uniform within the relevant comparisons: the viewport bug affects DOM-derived text and Phantom-SoM's `[SOM_MARKS]` source together; scroll-direction confusion is a trajectory-execution threat rather than an evaluator rule; and Stage 2 uses frozen prompt/screenshot inputs, so Phase-A dispatch bugs affect which step an agent reached, not the model's forward pass on that saved step. This affects cross-mode interpretation in Sections 3--5: we treat environment failures as measurement threats and disclose their blast radius rather than folding them into cognitive claims.
+
+\subsection*{8.4 Pre-vs-post-hoc analyses \& retracted framings}
+
+The exact L11/L17 mechanism-layer story was not preregistered, and several earlier framings were retracted. The blast radius is epistemic status: preregistered H1--H3 gate the deployment and structural claims, H4 is exploratory, and H5--H6 are post-hoc explanatory layers; Section 5.X states that L17 emerged from the Stage 2A pilot and L11 from an early single-task continuation, then converged across logit shift, forward overlap, reverse overlap, and cross-tier tests under Holm correction. The negative-results registry records 12 retracted framings, including task-0 over-interpretation, the reverse-null claim overturned by the N=15 reverse sample, and the rejected selection-bias explanation (cross-tier Welch tests non-significant, with reverse magnitude identical across tiers). This affects Sections 1 and 5: the paper can claim a confirmed four-fold drop-in property and a sparse mid-layer mechanism, but not a preregistered exact-layer prediction or a universal single-task circuit.
+
+\subsection*{8.5 Statistical \& methodological limits}
+
+The statistical design is adequate for medium effects but underpowered for small per-cell effects and exact-layer micro-effects. The blast radius is precision, not directionality: Holm-Bonferroni is applied across the six canonical tested layers (L0/5/11/17/23/29), not all 36 cached layers; this matches the disclosed post-hoc grid but should not be read as a full-layer search correction \citep{holm1979sequentially,wang2023interpretability,zhang2024patching}. Bootstrap intervals use task-paired resampling, random-effects meta-analysis is limited to cells with N>=10 to avoid unstable tau-squared estimates, and complete-case deletion handles crashes or missing artifacts without imputation. Power analysis shows N=15 mechanism cells are not powered for small mid-layer effects (roughly Cohen's d below 0.65 at alpha=0.05), while site-level SR cells mainly detect 4--7pp effects. This affects Sections 4--5: null or marginal cells are interpreted as low-power evidence, and pooled estimates are paired with per-cell uncertainty.
+
+\subsection*{8.6 Sparse-mechanism caveat}
+
+The activation-patching effect is sparse: at L17, four of five completed cells have median Levenshtein-distance shift equal to zero, with IQRs including zero. The blast radius is the mechanism claim: mean disruption and Holm significance are carried by task subsets, approximately the high-salience-image quarter of the strong-tier cases, while many tasks show no visible continuation change; Cell E random injection, by contrast, destroys outputs broadly. This affects Section 5 by reframing the finding as a task-conditional sparse mechanism rather than a universal mid-layer circuit. The claim remains defensible because Section 5.1 reports per-task scatter/violin views alongside mean bands and because Cell E separates content-specific sparse patching from generic injection damage.
+
+\subsection*{8.7 Compute, cost, and sustainability bounds}
+
+Compute and carbon accounting are approximate because runs span DGX Spark, UCL Condense A100, and UCL Myriad V100/A100 profiles. The blast radius is absolute energy reporting: per-cell GPU-hours, proxy-API USD cost, and kg-CO2 estimates will be reported with hardware provenance, but cross-machine power variation limits precision \citep{qiu2025modserve}. Relative comparisons inside a cell remain valid because modes share the same hardware and evaluator path. This affects the cost part of the four-fold drop-in claim: token/API cost and latency comparisons are primary, while carbon totals are disclosed as bounded estimates.
+
+NEEDS_BIB_ENTRY: `lipton2018troubling`, `neurips2024checklist`, `holm1979sequentially`, `wang2023interpretability`, `zhang2024patching`.

diff --git a/docs/checkpoints/paper_drafts/section8_limitations.md b/docs/checkpoints/paper_drafts/section8_limitations.md
new file mode 100644
index 0000000000000000000000000000000000000000..b5f1526b2ec5bede526606fc5e12351774ee2057
--- /dev/null
+++ b/docs/checkpoints/paper_drafts/section8_limitations.md
@@ -0,0 +1,29 @@
+\subsection*{8.1 Scope \& external-validity bounds}
+
+This study is deliberately narrow: three sites (classifieds, reddit, shopping), one benchmark family (VisualWebArena plus WA-mini-style task structure), and two Qwen-family model classes: Qwen3-VL-4B locally and Qwen3-Omni-235B-Thinking through a proxy API. The blast radius is external validity, not the internal comparison: our claim is that a phantom routing space exists for Qwen-family agents on VWA-style tasks, not that every VLM, browser benchmark, or production site will expose the same arm ordering \citep{zhou2024webarena,koh2024visualwebarena,deng2023mind2web,drouin2024workarena}. The mechanism evidence in Section 5 is narrower still: it is B1-only, because the open-weight 4B model exposes activations and the B0 proxy model does not. This affects Sections 1, 5, and 6 by bounding language about universality and reproducibility: B1 behavior and mechanistic patching are byte-reproducible from released artifacts, while B0 is verifiable from traces and replayable subject to API access; cross-architecture claims, including GPT-4o-family claims, remain outside the paper.
+
+\subsection*{8.2 Construct validity \& evaluator threats}
+
+VWA success labels are imperfect measurements of task completion, especially around `ua_match` GPT-judge drift, the `string_match` `fuzzy_threshold` misnomer, brittle `program_html` selectors, and `finish_wrong_state` episodes. The blast radius is measurement-side: these four evaluator-class threats can flip individual labels or inflate raw success, but they do not redefine the task universe or the mode definitions; full appendix prose is kept in Section 4.X.1--4.X.4 rather than duplicated here. We therefore report both raw and adjusted success and isolate `na_fp`, `eval_fp`, and `visual_fp` filters as sensitivity layers, following the constraint-table rule that evaluator artifacts should be disclosed rather than hidden \citep{lipton2018troubling,neurips2024checklist}. This affects Sections 3 and 4: adjusted SR is the headline metric, raw SR remains visible, and claims that survive the filter ladder are defensible as representation effects rather than judge artifacts.
+
+\subsection*{8.3 Internal-validity threats: known scaffold bugs}
+
+Several scaffold bugs were real: the `in_viewport_ratio` operator-precedence bug exposed clipped DOM text, early scroll actions suffered direction-convention confusion, and Stage 2B/2C mechanism inputs came from pre-Phase-A archived browser states. The blast radius is bounded because these failures are mode-uniform within the relevant comparisons: the viewport bug affects DOM-derived text and Phantom-SoM's `[SOM_MARKS]` source together; scroll-direction confusion is a trajectory-execution threat rather than an evaluator rule; and Stage 2 uses frozen prompt/screenshot inputs, so Phase-A dispatch bugs affect which step an agent reached, not the model's forward pass on that saved step. This affects cross-mode interpretation in Sections 3--5: we treat environment failures as measurement threats and disclose their blast radius rather than folding them into cognitive claims.
+
+\subsection*{8.4 Pre-vs-post-hoc analyses \& retracted framings}
+
+The exact L11/L17 mechanism-layer story was not preregistered, and several earlier framings were retracted. The blast radius is epistemic status: preregistered H1--H3 gate the deployment and structural claims, H4 is exploratory, and H5--H6 are post-hoc explanatory layers; Section 5.X states that L17 emerged from the Stage 2A pilot and L11 from an early single-task continuation, then converged across logit shift, forward overlap, reverse overlap, and cross-tier tests under Holm correction. The negative-results registry records 12 retracted framings, including task-0 over-interpretation, the reverse-null claim overturned by the N=15 reverse sample, and the rejected selection-bias explanation (cross-tier Welch tests non-significant, with reverse magnitude identical across tiers). This affects Sections 1 and 5: the paper can claim a confirmed four-fold drop-in property and a sparse mid-layer mechanism, but not a preregistered exact-layer prediction or a universal single-task circuit.
+
+\subsection*{8.5 Statistical \& methodological limits}
+
+The statistical design is adequate for medium effects but underpowered for small per-cell effects and exact-layer micro-effects. The blast radius is precision, not directionality: Holm-Bonferroni is applied across the six canonical tested layers (L0/5/11/17/23/29), not all 36 cached layers; this matches the disclosed post-hoc grid but should not be read as a full-layer search correction \citep{holm1979sequentially,wang2023interpretability,zhang2024patching}. Bootstrap intervals use task-paired resampling, random-effects meta-analysis is limited to cells with N>=10 to avoid unstable tau-squared estimates, and complete-case deletion handles crashes or missing artifacts without imputation. Power analysis shows N=15 mechanism cells are not powered for small mid-layer effects (roughly Cohen's d below 0.65 at alpha=0.05), while site-level SR cells mainly detect 4--7pp effects. This affects Sections 4--5: null or marginal cells are interpreted as low-power evidence, and pooled estimates are paired with per-cell uncertainty.
+
+\subsection*{8.6 Sparse-mechanism caveat}
+
+The activation-patching effect is sparse: at L17, four of five completed cells have median Levenshtein-distance shift equal to zero, with IQRs including zero. The blast radius is the mechanism claim: mean disruption and Holm significance are carried by task subsets, approximately the high-salience-image quarter of the strong-tier cases, while many tasks show no visible continuation change; Cell E random injection, by contrast, destroys outputs broadly. This affects Section 5 by reframing the finding as a task-conditional sparse mechanism rather than a universal mid-layer circuit. The claim remains defensible because Section 5.1 reports per-task scatter/violin views alongside mean bands and because Cell E separates content-specific sparse patching from generic injection damage.
+
+\subsection*{8.7 Compute, cost, and sustainability bounds}
+
+Compute and carbon accounting are approximate because runs span DGX Spark, UCL Condense A100, and UCL Myriad V100/A100 profiles. The blast radius is absolute energy reporting: per-cell GPU-hours, proxy-API USD cost, and kg-CO2 estimates will be reported with hardware provenance, but cross-machine power variation limits precision \citep{qiu2025modserve}. Relative comparisons inside a cell remain valid because modes share the same hardware and evaluator path. This affects the cost part of the four-fold drop-in claim: token/API cost and latency comparisons are primary, while carbon totals are disclosed as bounded estimates.
+
+NEEDS_BIB_ENTRY: `lipton2018troubling`, `neurips2024checklist`, `holm1979sequentially`, `wang2023interpretability`, `zhang2024patching`.

exec
/bin/bash -lc 'wc -w docs/checkpoints/paper_drafts/section8_limitations.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
899 docs/checkpoints/paper_drafts/section8_limitations.md

exec
/bin/bash -lc 'rg -n "''^'"\\\\subsection\\*|NEEDS_BIB_ENTRY\" docs/checkpoints/paper_drafts/section8_limitations.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
29:NEEDS_BIB_ENTRY: `lipton2018troubling`, `neurips2024checklist`, `holm1979sequentially`, `wang2023interpretability`, `zhang2024patching`.

exec
/bin/bash -lc "python3 - <<'PY'
from pathlib import Path
p=Path('docs/checkpoints/paper_drafts/section8_limitations.md')
text=p.read_text()
parts=[]
cur=None
buf=[]
for line in text.splitlines():
    if line.startswith('\\\\subsection*{'):
        if cur is not None:
            parts.append((cur,'\\n'.join(buf)))
        cur=line.strip()
        buf=[]
    elif line.startswith('NEEDS_BIB_ENTRY'):
        if cur is not None:
            parts.append((cur,'\\n'.join(buf)))
            cur=None; buf=[]
    else:
        if cur is not None:
            buf.append(line)
if cur is not None:
    parts.append((cur,'\\n'.join(buf)))
for h,b in parts:
    words=b.split()
    print(f'{h}: {len(words)}')
print('subsection_total:', sum(len(b.split()) for _,b in parts))
PY" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
\subsection*{8.1 Scope \& external-validity bounds}: 143
\subsection*{8.2 Construct validity \& evaluator threats}: 129
\subsection*{8.3 Internal-validity threats: known scaffold bugs}: 118
\subsection*{8.4 Pre-vs-post-hoc analyses \& retracted framings}: 135
\subsection*{8.5 Statistical \& methodological limits}: 133
\subsection*{8.6 Sparse-mechanism caveat}: 110
\subsection*{8.7 Compute, cost, and sustainability bounds}: 88
subsection_total: 856

diff --git a/docs/checkpoints/paper_drafts/section8_limitations.md b/docs/checkpoints/paper_drafts/section8_limitations.md
new file mode 100644
index 0000000000000000000000000000000000000000..b5f1526b2ec5bede526606fc5e12351774ee2057
--- /dev/null
+++ b/docs/checkpoints/paper_drafts/section8_limitations.md
@@ -0,0 +1,29 @@
+\subsection*{8.1 Scope \& external-validity bounds}
+
+This study is deliberately narrow: three sites (classifieds, reddit, shopping), one benchmark family (VisualWebArena plus WA-mini-style task structure), and two Qwen-family model classes: Qwen3-VL-4B locally and Qwen3-Omni-235B-Thinking through a proxy API. The blast radius is external validity, not the internal comparison: our claim is that a phantom routing space exists for Qwen-family agents on VWA-style tasks, not that every VLM, browser benchmark, or production site will expose the same arm ordering \citep{zhou2024webarena,koh2024visualwebarena,deng2023mind2web,drouin2024workarena}. The mechanism evidence in Section 5 is narrower still: it is B1-only, because the open-weight 4B model exposes activations and the B0 proxy model does not. This affects Sections 1, 5, and 6 by bounding language about universality and reproducibility: B1 behavior and mechanistic patching are byte-reproducible from released artifacts, while B0 is verifiable from traces and replayable subject to API access; cross-architecture claims, including GPT-4o-family claims, remain outside the paper.
+
+\subsection*{8.2 Construct validity \& evaluator threats}
+
+VWA success labels are imperfect measurements of task completion, especially around `ua_match` GPT-judge drift, the `string_match` `fuzzy_threshold` misnomer, brittle `program_html` selectors, and `finish_wrong_state` episodes. The blast radius is measurement-side: these four evaluator-class threats can flip individual labels or inflate raw success, but they do not redefine the task universe or the mode definitions; full appendix prose is kept in Section 4.X.1--4.X.4 rather than duplicated here. We therefore report both raw and adjusted success and isolate `na_fp`, `eval_fp`, and `visual_fp` filters as sensitivity layers, following the constraint-table rule that evaluator artifacts should be disclosed rather than hidden \citep{lipton2018troubling,neurips2024checklist}. This affects Sections 3 and 4: adjusted SR is the headline metric, raw SR remains visible, and claims that survive the filter ladder are defensible as representation effects rather than judge artifacts.
+
+\subsection*{8.3 Internal-validity threats: known scaffold bugs}
+
+Several scaffold bugs were real: the `in_viewport_ratio` operator-precedence bug exposed clipped DOM text, early scroll actions suffered direction-convention confusion, and Stage 2B/2C mechanism inputs came from pre-Phase-A archived browser states. The blast radius is bounded because these failures are mode-uniform within the relevant comparisons: the viewport bug affects DOM-derived text and Phantom-SoM's `[SOM_MARKS]` source together; scroll-direction confusion is a trajectory-execution threat rather than an evaluator rule; and Stage 2 uses frozen prompt/screenshot inputs, so Phase-A dispatch bugs affect which step an agent reached, not the model's forward pass on that saved step. This affects cross-mode interpretation in Sections 3--5: we treat environment failures as measurement threats and disclose their blast radius rather than folding them into cognitive claims.
+
+\subsection*{8.4 Pre-vs-post-hoc analyses \& retracted framings}
+
+The exact L11/L17 mechanism-layer story was not preregistered, and several earlier framings were retracted. The blast radius is epistemic status: preregistered H1--H3 gate the deployment and structural claims, H4 is exploratory, and H5--H6 are post-hoc explanatory layers; Section 5.X states that L17 emerged from the Stage 2A pilot and L11 from an early single-task continuation, then converged across logit shift, forward overlap, reverse overlap, and cross-tier tests under Holm correction. The negative-results registry records 12 retracted framings, including task-0 over-interpretation, the reverse-null claim overturned by the N=15 reverse sample, and the rejected selection-bias explanation (cross-tier Welch tests non-significant, with reverse magnitude identical across tiers). This affects Sections 1 and 5: the paper can claim a confirmed four-fold drop-in property and a sparse mid-layer mechanism, but not a preregistered exact-layer prediction or a universal single-task circuit.
+
+\subsection*{8.5 Statistical \& methodological limits}
+
+The statistical design is adequate for medium effects but underpowered for small per-cell effects and exact-layer micro-effects. The blast radius is precision, not directionality: Holm-Bonferroni is applied across the six canonical tested layers (L0/5/11/17/23/29), not all 36 cached layers; this matches the disclosed post-hoc grid but should not be read as a full-layer search correction \citep{holm1979sequentially,wang2023interpretability,zhang2024patching}. Bootstrap intervals use task-paired resampling, random-effects meta-analysis is limited to cells with N>=10 to avoid unstable tau-squared estimates, and complete-case deletion handles crashes or missing artifacts without imputation. Power analysis shows N=15 mechanism cells are not powered for small mid-layer effects (roughly Cohen's d below 0.65 at alpha=0.05), while site-level SR cells mainly detect 4--7pp effects. This affects Sections 4--5: null or marginal cells are interpreted as low-power evidence, and pooled estimates are paired with per-cell uncertainty.
+
+\subsection*{8.6 Sparse-mechanism caveat}
+
+The activation-patching effect is sparse: at L17, four of five completed cells have median Levenshtein-distance shift equal to zero, with IQRs including zero. The blast radius is the mechanism claim: mean disruption and Holm significance are carried by task subsets, approximately the high-salience-image quarter of the strong-tier cases, while many tasks show no visible continuation change; Cell E random injection, by contrast, destroys outputs broadly. This affects Section 5 by reframing the finding as a task-conditional sparse mechanism rather than a universal mid-layer circuit. The claim remains defensible because Section 5.1 reports per-task scatter/violin views alongside mean bands and because Cell E separates content-specific sparse patching from generic injection damage.
+
+\subsection*{8.7 Compute, cost, and sustainability bounds}
+
+Compute and carbon accounting are approximate because runs span DGX Spark, UCL Condense A100, and UCL Myriad V100/A100 profiles. The blast radius is absolute energy reporting: per-cell GPU-hours, proxy-API USD cost, and kg-CO2 estimates will be reported with hardware provenance, but cross-machine power variation limits precision \citep{qiu2025modserve}. Relative comparisons inside a cell remain valid because modes share the same hardware and evaluator path. This affects the cost part of the four-fold drop-in claim: token/API cost and latency comparisons are primary, while carbon totals are disclosed as bounded estimates.
+
+NEEDS_BIB_ENTRY: `lipton2018troubling`, `neurips2024checklist`, `holm1979sequentially`, `wang2023interpretability`, `zhang2024patching`.

codex
The first pass came in under the requested subsection word range, so I’m tightening the compliance pass now: adding the missing missing-data and registry-bounding details, then recounting by subsection.
apply patch
patch: completed
/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/checkpoints/paper_drafts/section8_limitations.md
diff --git a/docs/checkpoints/paper_drafts/section8_limitations.md b/docs/checkpoints/paper_drafts/section8_limitations.md
new file mode 100644
index 0000000000000000000000000000000000000000..c4c440894aef0698543754949e5e7c9b285fb3b5
--- /dev/null
+++ b/docs/checkpoints/paper_drafts/section8_limitations.md
@@ -0,0 +1,29 @@
+\subsection*{8.1 Scope \& external-validity bounds}
+
+This study is deliberately narrow: three sites (classifieds, reddit, shopping), one benchmark family (VisualWebArena plus WA-mini-style task structure), and two Qwen-family model classes: Qwen3-VL-4B locally and Qwen3-Omni-235B-Thinking through a proxy API. The blast radius is external validity, not the internal comparison: our claim is that a phantom routing space exists for Qwen-family agents on VWA-style tasks, not that every VLM, browser benchmark, or production site will expose the same arm ordering \citep{zhou2024webarena,koh2024visualwebarena,deng2023mind2web,drouin2024workarena}. The mechanism evidence in Section 5 is narrower still: it is B1-only, because the open-weight 4B model exposes activations and the B0 proxy model does not. Shopping also has weaker intermediate mechanism coverage than classifieds/reddit, so site-level generalization is reported cell-by-cell rather than averaged into a universal web-agent claim. This affects Sections 1, 5, and 6 by bounding language about universality and reproducibility: B1 behavior and mechanistic patching are byte-reproducible from released artifacts, while B0 is verifiable from traces and replayable subject to API access; cross-architecture claims, including GPT-4o-family claims, remain outside the paper.
+
+\subsection*{8.2 Construct validity \& evaluator threats}
+
+VWA success labels are imperfect measurements of task completion, especially around `ua_match` GPT-judge drift, the `string_match` `fuzzy_threshold` misnomer, brittle `program_html` selectors, and `finish_wrong_state` episodes. The blast radius is measurement-side: these four evaluator-class threats can flip individual labels or inflate raw success, but they do not redefine the task universe or the mode definitions; full appendix prose is kept in Section 4.X.1--4.X.4 rather than duplicated here. We therefore report both raw and adjusted success and isolate `na_fp`, `eval_fp`, and `visual_fp` filters as sensitivity layers, including the audit-derived false-positive filters from Sections 78a and 95, following the constraint-table rule that evaluator artifacts should be disclosed rather than hidden \citep{lipton2018troubling,neurips2024checklist}. This affects Sections 3 and 4: adjusted SR is the headline metric, raw SR remains visible, and claims that survive the filter ladder are defensible as representation effects rather than judge artifacts.
+
+\subsection*{8.3 Internal-validity threats: known scaffold bugs}
+
+Several scaffold bugs were real: the `in_viewport_ratio` operator-precedence bug exposed clipped DOM text, early scroll actions suffered direction-convention confusion, and Stage 2B/2C mechanism inputs came from pre-Phase-A archived browser states. The blast radius is bounded because these failures are mode-uniform within the relevant comparisons: the viewport bug affects DOM-derived text and Phantom-SoM's `[SOM_MARKS]` source together; scroll-direction confusion is a trajectory-execution threat rather than an evaluator rule; and Stage 2 uses frozen prompt/screenshot inputs, so Phase-A dispatch bugs affect which step an agent reached, not the model's forward pass on that saved step. This affects cross-mode interpretation in Sections 3--5: we treat environment failures as measurement threats and disclose their blast radius rather than folding them into cognitive claims.
+
+\subsection*{8.4 Pre-vs-post-hoc analyses \& retracted framings}
+
+The exact L11/L17 mechanism-layer story was not preregistered, and several earlier framings were retracted. The blast radius is epistemic status: preregistered H1--H3 gate the deployment and structural claims, H4 is exploratory, and H5--H6 are post-hoc explanatory layers; Section 5.X states that L17 emerged from the Stage 2A pilot and L11 from an early single-task continuation, then converged across logit shift, forward overlap, reverse overlap, and cross-tier tests under Holm correction. The negative-results registry records 12 retracted framings, including task-0 over-interpretation, the reverse-null claim overturned by the N=15 reverse sample, and the rejected selection-bias explanation (cross-tier Welch tests non-significant, with reverse magnitude identical across tiers). It also records two framings that survived audit: the four-fold drop-in property and the sparse L11/L17 mechanism. This affects Sections 1 and 5: the paper can claim confirmed, registry-backed evidence for those two framings, but not a preregistered exact-layer prediction or a universal single-task circuit.
+
+\subsection*{8.5 Statistical \& methodological limits}
+
+The statistical design is adequate for medium effects but underpowered for small per-cell effects and exact-layer micro-effects. The blast radius is precision, not directionality: Holm-Bonferroni is applied across the six canonical tested layers (L0/5/11/17/23/29), not all 36 cached layers; this matches the disclosed post-hoc grid but should not be read as a full-layer search correction \citep{holm1979sequentially,wang2023interpretability,zhang2024patching}. Bootstrap intervals use task-paired resampling, random-effects meta-analysis is limited to cells with N>=10 to avoid unstable tau-squared estimates, and complete-case deletion handles crashes or missing artifacts without imputation. Exclusions are listwise only, at <=5% per cell under the B6 lock, so multiple imputation would add modeling assumptions without materially changing paired denominators. Power analysis shows N=15 mechanism cells are not powered for small mid-layer effects (roughly Cohen's d below 0.65 at alpha=0.05), while site-level SR cells mainly detect 4--7pp effects. This affects Sections 4--5: null or marginal cells are interpreted as low-power evidence, and pooled estimates are paired with per-cell uncertainty.
+
+\subsection*{8.6 Sparse-mechanism caveat}
+
+The activation-patching effect is sparse: at L17, four of five completed cells have median Levenshtein-distance shift equal to zero, with IQRs including zero. The blast radius is the mechanism claim: mean disruption and Holm significance are carried by task subsets, approximately the high-salience-image quarter of the strong-tier cases, while many tasks show no visible continuation change; Cell E random injection, by contrast, destroys outputs broadly. This affects Section 5 by reframing the finding as a task-conditional sparse mechanism rather than a universal mid-layer circuit. The claim remains defensible because Section 5.1 reports per-task scatter/violin views alongside mean bands and because Cell E separates content-specific sparse patching from generic injection damage.
+
+\subsection*{8.7 Compute, cost, and sustainability bounds}
+
+Compute and carbon accounting are approximate because runs span DGX Spark, UCL Condense A100, and UCL Myriad V100/A100 profiles. The blast radius is absolute energy reporting: the per-cell table reports GPU-hours, proxy-API USD cost, and kg-CO2 estimates with hardware provenance, but cross-machine power variation limits precision \citep{qiu2025modserve}. Relative comparisons inside a cell remain valid because modes share the same hardware, benchmark site, and evaluator path. This affects the cost part of the four-fold drop-in claim: token/API cost and latency comparisons are primary, while carbon totals are disclosed as bounded estimates rather than a fine-grained lifecycle assessment.
+
+NEEDS_BIB_ENTRY: `lipton2018troubling`, `neurips2024checklist`, `holm1979sequentially`, `wang2023interpretability`, `zhang2024patching`.

diff --git a/docs/checkpoints/paper_drafts/section8_limitations.md b/docs/checkpoints/paper_drafts/section8_limitations.md
new file mode 100644
index 0000000000000000000000000000000000000000..c4c440894aef0698543754949e5e7c9b285fb3b5
--- /dev/null
+++ b/docs/checkpoints/paper_drafts/section8_limitations.md
@@ -0,0 +1,29 @@
+\subsection*{8.1 Scope \& external-validity bounds}
+
+This study is deliberately narrow: three sites (classifieds, reddit, shopping), one benchmark family (VisualWebArena plus WA-mini-style task structure), and two Qwen-family model classes: Qwen3-VL-4B locally and Qwen3-Omni-235B-Thinking through a proxy API. The blast radius is external validity, not the internal comparison: our claim is that a phantom routing space exists for Qwen-family agents on VWA-style tasks, not that every VLM, browser benchmark, or production site will expose the same arm ordering \citep{zhou2024webarena,koh2024visualwebarena,deng2023mind2web,drouin2024workarena}. The mechanism evidence in Section 5 is narrower still: it is B1-only, because the open-weight 4B model exposes activations and the B0 proxy model does not. Shopping also has weaker intermediate mechanism coverage than classifieds/reddit, so site-level generalization is reported cell-by-cell rather than averaged into a universal web-agent claim. This affects Sections 1, 5, and 6 by bounding language about universality and reproducibility: B1 behavior and mechanistic patching are byte-reproducible from released artifacts, while B0 is verifiable from traces and replayable subject to API access; cross-architecture claims, including GPT-4o-family claims, remain outside the paper.
+
+\subsection*{8.2 Construct validity \& evaluator threats}
+
+VWA success labels are imperfect measurements of task completion, especially around `ua_match` GPT-judge drift, the `string_match` `fuzzy_threshold` misnomer, brittle `program_html` selectors, and `finish_wrong_state` episodes. The blast radius is measurement-side: these four evaluator-class threats can flip individual labels or inflate raw success, but they do not redefine the task universe or the mode definitions; full appendix prose is kept in Section 4.X.1--4.X.4 rather than duplicated here. We therefore report both raw and adjusted success and isolate `na_fp`, `eval_fp`, and `visual_fp` filters as sensitivity layers, including the audit-derived false-positive filters from Sections 78a and 95, following the constraint-table rule that evaluator artifacts should be disclosed rather than hidden \citep{lipton2018troubling,neurips2024checklist}. This affects Sections 3 and 4: adjusted SR is the headline metric, raw SR remains visible, and claims that survive the filter ladder are defensible as representation effects rather than judge artifacts.
+
+\subsection*{8.3 Internal-validity threats: known scaffold bugs}
+
+Several scaffold bugs were real: the `in_viewport_ratio` operator-precedence bug exposed clipped DOM text, early scroll actions suffered direction-convention confusion, and Stage 2B/2C mechanism inputs came from pre-Phase-A archived browser states. The blast radius is bounded because these failures are mode-uniform within the relevant comparisons: the viewport bug affects DOM-derived text and Phantom-SoM's `[SOM_MARKS]` source together; scroll-direction confusion is a trajectory-execution threat rather than an evaluator rule; and Stage 2 uses frozen prompt/screenshot inputs, so Phase-A dispatch bugs affect which step an agent reached, not the model's forward pass on that saved step. This affects cross-mode interpretation in Sections 3--5: we treat environment failures as measurement threats and disclose their blast radius rather than folding them into cognitive claims.
+
+\subsection*{8.4 Pre-vs-post-hoc analyses \& retracted framings}
+
+The exact L11/L17 mechanism-layer story was not preregistered, and several earlier framings were retracted. The blast radius is epistemic status: preregistered H1--H3 gate the deployment and structural claims, H4 is exploratory, and H5--H6 are post-hoc explanatory layers; Section 5.X states that L17 emerged from the Stage 2A pilot and L11 from an early single-task continuation, then converged across logit shift, forward overlap, reverse overlap, and cross-tier tests under Holm correction. The negative-results registry records 12 retracted framings, including task-0 over-interpretation, the reverse-null claim overturned by the N=15 reverse sample, and the rejected selection-bias explanation (cross-tier Welch tests non-significant, with reverse magnitude identical across tiers). It also records two framings that survived audit: the four-fold drop-in property and the sparse L11/L17 mechanism. This affects Sections 1 and 5: the paper can claim confirmed, registry-backed evidence for those two framings, but not a preregistered exact-layer prediction or a universal single-task circuit.
+
+\subsection*{8.5 Statistical \& methodological limits}
+
+The statistical design is adequate for medium effects but underpowered for small per-cell effects and exact-layer micro-effects. The blast radius is precision, not directionality: Holm-Bonferroni is applied across the six canonical tested layers (L0/5/11/17/23/29), not all 36 cached layers; this matches the disclosed post-hoc grid but should not be read as a full-layer search correction \citep{holm1979sequentially,wang2023interpretability,zhang2024patching}. Bootstrap intervals use task-paired resampling, random-effects meta-analysis is limited to cells with N>=10 to avoid unstable tau-squared estimates, and complete-case deletion handles crashes or missing artifacts without imputation. Exclusions are listwise only, at <=5% per cell under the B6 lock, so multiple imputation would add modeling assumptions without materially changing paired denominators. Power analysis shows N=15 mechanism cells are not powered for small mid-layer effects (roughly Cohen's d below 0.65 at alpha=0.05), while site-level SR cells mainly detect 4--7pp effects. This affects Sections 4--5: null or marginal cells are interpreted as low-power evidence, and pooled estimates are paired with per-cell uncertainty.
+
+\subsection*{8.6 Sparse-mechanism caveat}
+
+The activation-patching effect is sparse: at L17, four of five completed cells have median Levenshtein-distance shift equal to zero, with IQRs including zero. The blast radius is the mechanism claim: mean disruption and Holm significance are carried by task subsets, approximately the high-salience-image quarter of the strong-tier cases, while many tasks show no visible continuation change; Cell E random injection, by contrast, destroys outputs broadly. This affects Section 5 by reframing the finding as a task-conditional sparse mechanism rather than a universal mid-layer circuit. The claim remains defensible because Section 5.1 reports per-task scatter/violin views alongside mean bands and because Cell E separates content-specific sparse patching from generic injection damage.
+
+\subsection*{8.7 Compute, cost, and sustainability bounds}
+
+Compute and carbon accounting are approximate because runs span DGX Spark, UCL Condense A100, and UCL Myriad V100/A100 profiles. The blast radius is absolute energy reporting: the per-cell table reports GPU-hours, proxy-API USD cost, and kg-CO2 estimates with hardware provenance, but cross-machine power variation limits precision \citep{qiu2025modserve}. Relative comparisons inside a cell remain valid because modes share the same hardware, benchmark site, and evaluator path. This affects the cost part of the four-fold drop-in claim: token/API cost and latency comparisons are primary, while carbon totals are disclosed as bounded estimates rather than a fine-grained lifecycle assessment.
+
+NEEDS_BIB_ENTRY: `lipton2018troubling`, `neurips2024checklist`, `holm1979sequentially`, `wang2023interpretability`, `zhang2024patching`.

exec
/bin/bash -lc "python3 - <<'PY'
from pathlib import Path
p=Path('docs/checkpoints/paper_drafts/section8_limitations.md')
text=p.read_text()
parts=[]; cur=None; buf=[]
for line in text.splitlines():
    if line.startswith('\\\\subsection*{'):
        if cur is not None: parts.append((cur,'\\n'.join(buf)))
        cur=line.strip(); buf=[]
    elif line.startswith('NEEDS_BIB_ENTRY'):
        if cur is not None: parts.append((cur,'\\n'.join(buf))); cur=None; buf=[]
    else:
        if cur is not None: buf.append(line)
if cur is not None: parts.append((cur,'\\n'.join(buf)))
for h,b in parts:
    print(f'{h}: {len(b.split())}')
print('subsection_total:', sum(len(b.split()) for _,b in parts))
print('file_total:', len(text.split()))
PY" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
\subsection*{8.1 Scope \& external-validity bounds}: 166
\subsection*{8.2 Construct validity \& evaluator threats}: 139
\subsection*{8.3 Internal-validity threats: known scaffold bugs}: 118
\subsection*{8.4 Pre-vs-post-hoc analyses \& retracted framings}: 149
\subsection*{8.5 Statistical \& methodological limits}: 157
\subsection*{8.6 Sparse-mechanism caveat}: 110
\subsection*{8.7 Compute, cost, and sustainability bounds}: 96
subsection_total: 935
file_total: 978

exec
/bin/bash -lc 'rg -n "''^'"\\\\\\\\subsection\\*|NEEDS_BIB_ENTRY|\\[["'^]]+'"\\]\" docs/checkpoints/paper_drafts/section8_limitations.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
1:\subsection*{8.1 Scope \& external-validity bounds}
5:\subsection*{8.2 Construct validity \& evaluator threats}
9:\subsection*{8.3 Internal-validity threats: known scaffold bugs}
11:Several scaffold bugs were real: the `in_viewport_ratio` operator-precedence bug exposed clipped DOM text, early scroll actions suffered direction-convention confusion, and Stage 2B/2C mechanism inputs came from pre-Phase-A archived browser states. The blast radius is bounded because these failures are mode-uniform within the relevant comparisons: the viewport bug affects DOM-derived text and Phantom-SoM's `[SOM_MARKS]` source together; scroll-direction confusion is a trajectory-execution threat rather than an evaluator rule; and Stage 2 uses frozen prompt/screenshot inputs, so Phase-A dispatch bugs affect which step an agent reached, not the model's forward pass on that saved step. This affects cross-mode interpretation in Sections 3--5: we treat environment failures as measurement threats and disclose their blast radius rather than folding them into cognitive claims.
13:\subsection*{8.4 Pre-vs-post-hoc analyses \& retracted framings}
17:\subsection*{8.5 Statistical \& methodological limits}
21:\subsection*{8.6 Sparse-mechanism caveat}
25:\subsection*{8.7 Compute, cost, and sustainability bounds}
29:NEEDS_BIB_ENTRY: `lipton2018troubling`, `neurips2024checklist`, `holm1979sequentially`, `wang2023interpretability`, `zhang2024patching`.

exec
/bin/bash -lc 'git status --short docs/checkpoints/paper_drafts/section8_limitations.md docs/checkpoints/paper_drafts/section4_limitations_disclosure.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
 M docs/checkpoints/paper_drafts/section8_limitations.md

diff --git a/docs/checkpoints/paper_drafts/section8_limitations.md b/docs/checkpoints/paper_drafts/section8_limitations.md
new file mode 100644
index 0000000000000000000000000000000000000000..c4c440894aef0698543754949e5e7c9b285fb3b5
--- /dev/null
+++ b/docs/checkpoints/paper_drafts/section8_limitations.md
@@ -0,0 +1,29 @@
+\subsection*{8.1 Scope \& external-validity bounds}
+
+This study is deliberately narrow: three sites (classifieds, reddit, shopping), one benchmark family (VisualWebArena plus WA-mini-style task structure), and two Qwen-family model classes: Qwen3-VL-4B locally and Qwen3-Omni-235B-Thinking through a proxy API. The blast radius is external validity, not the internal comparison: our claim is that a phantom routing space exists for Qwen-family agents on VWA-style tasks, not that every VLM, browser benchmark, or production site will expose the same arm ordering \citep{zhou2024webarena,koh2024visualwebarena,deng2023mind2web,drouin2024workarena}. The mechanism evidence in Section 5 is narrower still: it is B1-only, because the open-weight 4B model exposes activations and the B0 proxy model does not. Shopping also has weaker intermediate mechanism coverage than classifieds/reddit, so site-level generalization is reported cell-by-cell rather than averaged into a universal web-agent claim. This affects Sections 1, 5, and 6 by bounding language about universality and reproducibility: B1 behavior and mechanistic patching are byte-reproducible from released artifacts, while B0 is verifiable from traces and replayable subject to API access; cross-architecture claims, including GPT-4o-family claims, remain outside the paper.
+
+\subsection*{8.2 Construct validity \& evaluator threats}
+
+VWA success labels are imperfect measurements of task completion, especially around `ua_match` GPT-judge drift, the `string_match` `fuzzy_threshold` misnomer, brittle `program_html` selectors, and `finish_wrong_state` episodes. The blast radius is measurement-side: these four evaluator-class threats can flip individual labels or inflate raw success, but they do not redefine the task universe or the mode definitions; full appendix prose is kept in Section 4.X.1--4.X.4 rather than duplicated here. We therefore report both raw and adjusted success and isolate `na_fp`, `eval_fp`, and `visual_fp` filters as sensitivity layers, including the audit-derived false-positive filters from Sections 78a and 95, following the constraint-table rule that evaluator artifacts should be disclosed rather than hidden \citep{lipton2018troubling,neurips2024checklist}. This affects Sections 3 and 4: adjusted SR is the headline metric, raw SR remains visible, and claims that survive the filter ladder are defensible as representation effects rather than judge artifacts.
+
+\subsection*{8.3 Internal-validity threats: known scaffold bugs}
+
+Several scaffold bugs were real: the `in_viewport_ratio` operator-precedence bug exposed clipped DOM text, early scroll actions suffered direction-convention confusion, and Stage 2B/2C mechanism inputs came from pre-Phase-A archived browser states. The blast radius is bounded because these failures are mode-uniform within the relevant comparisons: the viewport bug affects DOM-derived text and Phantom-SoM's `[SOM_MARKS]` source together; scroll-direction confusion is a trajectory-execution threat rather than an evaluator rule; and Stage 2 uses frozen prompt/screenshot inputs, so Phase-A dispatch bugs affect which step an agent reached, not the model's forward pass on that saved step. This affects cross-mode interpretation in Sections 3--5: we treat environment failures as measurement threats and disclose their blast radius rather than folding them into cognitive claims.
+
+\subsection*{8.4 Pre-vs-post-hoc analyses \& retracted framings}
+
+The exact L11/L17 mechanism-layer story was not preregistered, and several earlier framings were retracted. The blast radius is epistemic status: preregistered H1--H3 gate the deployment and structural claims, H4 is exploratory, and H5--H6 are post-hoc explanatory layers; Section 5.X states that L17 emerged from the Stage 2A pilot and L11 from an early single-task continuation, then converged across logit shift, forward overlap, reverse overlap, and cross-tier tests under Holm correction. The negative-results registry records 12 retracted framings, including task-0 over-interpretation, the reverse-null claim overturned by the N=15 reverse sample, and the rejected selection-bias explanation (cross-tier Welch tests non-significant, with reverse magnitude identical across tiers). It also records two framings that survived audit: the four-fold drop-in property and the sparse L11/L17 mechanism. This affects Sections 1 and 5: the paper can claim confirmed, registry-backed evidence for those two framings, but not a preregistered exact-layer prediction or a universal single-task circuit.
+
+\subsection*{8.5 Statistical \& methodological limits}
+
+The statistical design is adequate for medium effects but underpowered for small per-cell effects and exact-layer micro-effects. The blast radius is precision, not directionality: Holm-Bonferroni is applied across the six canonical tested layers (L0/5/11/17/23/29), not all 36 cached layers; this matches the disclosed post-hoc grid but should not be read as a full-layer search correction \citep{holm1979sequentially,wang2023interpretability,zhang2024patching}. Bootstrap intervals use task-paired resampling, random-effects meta-analysis is limited to cells with N>=10 to avoid unstable tau-squared estimates, and complete-case deletion handles crashes or missing artifacts without imputation. Exclusions are listwise only, at <=5% per cell under the B6 lock, so multiple imputation would add modeling assumptions without materially changing paired denominators. Power analysis shows N=15 mechanism cells are not powered for small mid-layer effects (roughly Cohen's d below 0.65 at alpha=0.05), while site-level SR cells mainly detect 4--7pp effects. This affects Sections 4--5: null or marginal cells are interpreted as low-power evidence, and pooled estimates are paired with per-cell uncertainty.
+
+\subsection*{8.6 Sparse-mechanism caveat}
+
+The activation-patching effect is sparse: at L17, four of five completed cells have median Levenshtein-distance shift equal to zero, with IQRs including zero. The blast radius is the mechanism claim: mean disruption and Holm significance are carried by task subsets, approximately the high-salience-image quarter of the strong-tier cases, while many tasks show no visible continuation change; Cell E random injection, by contrast, destroys outputs broadly. This affects Section 5 by reframing the finding as a task-conditional sparse mechanism rather than a universal mid-layer circuit. The claim remains defensible because Section 5.1 reports per-task scatter/violin views alongside mean bands and because Cell E separates content-specific sparse patching from generic injection damage.
+
+\subsection*{8.7 Compute, cost, and sustainability bounds}
+
+Compute and carbon accounting are approximate because runs span DGX Spark, UCL Condense A100, and UCL Myriad V100/A100 profiles. The blast radius is absolute energy reporting: the per-cell table reports GPU-hours, proxy-API USD cost, and kg-CO2 estimates with hardware provenance, but cross-machine power variation limits precision \citep{qiu2025modserve}. Relative comparisons inside a cell remain valid because modes share the same hardware, benchmark site, and evaluator path. This affects the cost part of the four-fold drop-in claim: token/API cost and latency comparisons are primary, while carbon totals are disclosed as bounded estimates rather than a fine-grained lifecycle assessment.
+
+NEEDS_BIB_ENTRY: `lipton2018troubling`, `neurips2024checklist`, `holm1979sequentially`, `wang2023interpretability`, `zhang2024patching`.

exec
/bin/bash -lc 'git diff -- docs/checkpoints/paper_drafts/section8_limitations.md --stat && git diff -- docs/checkpoints/paper_drafts/section4_limitations_disclosure.md --stat' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
diff --git a/docs/checkpoints/paper_drafts/section8_limitations.md b/docs/checkpoints/paper_drafts/section8_limitations.md
index b5f1526..c4c4408 100644
--- a/docs/checkpoints/paper_drafts/section8_limitations.md
+++ b/docs/checkpoints/paper_drafts/section8_limitations.md
@@ -1,10 +1,10 @@
 \subsection*{8.1 Scope \& external-validity bounds}
 
-This study is deliberately narrow: three sites (classifieds, reddit, shopping), one benchmark family (VisualWebArena plus WA-mini-style task structure), and two Qwen-family model classes: Qwen3-VL-4B locally and Qwen3-Omni-235B-Thinking through a proxy API. The blast radius is external validity, not the internal comparison: our claim is that a phantom routing space exists for Qwen-family agents on VWA-style tasks, not that every VLM, browser benchmark, or production site will expose the same arm ordering \citep{zhou2024webarena,koh2024visualwebarena,deng2023mind2web,drouin2024workarena}. The mechanism evidence in Section 5 is narrower still: it is B1-only, because the open-weight 4B model exposes activations and the B0 proxy model does not. This affects Sections 1, 5, and 6 by bounding language about universality and reproducibility: B1 behavior and mechanistic patching are byte-reproducible from released artifacts, while B0 is verifiable from traces and replayable subject to API access; cross-architecture claims, including GPT-4o-family claims, remain outside the paper.
+This study is deliberately narrow: three sites (classifieds, reddit, shopping), one benchmark family (VisualWebArena plus WA-mini-style task structure), and two Qwen-family model classes: Qwen3-VL-4B locally and Qwen3-Omni-235B-Thinking through a proxy API. The blast radius is external validity, not the internal comparison: our claim is that a phantom routing space exists for Qwen-family agents on VWA-style tasks, not that every VLM, browser benchmark, or production site will expose the same arm ordering \citep{zhou2024webarena,koh2024visualwebarena,deng2023mind2web,drouin2024workarena}. The mechanism evidence in Section 5 is narrower still: it is B1-only, because the open-weight 4B model exposes activations and the B0 proxy model does not. Shopping also has weaker intermediate mechanism coverage than classifieds/reddit, so site-level generalization is reported cell-by-cell rather than averaged into a universal web-agent claim. This affects Sections 1, 5, and 6 by bounding language about universality and reproducibility: B1 behavior and mechanistic patching are byte-reproducible from released artifacts, while B0 is verifiable from traces and replayable subject to API access; cross-architecture claims, including GPT-4o-family claims, remain outside the paper.
 
 \subsection*{8.2 Construct validity \& evaluator threats}
 
-VWA success labels are imperfect measurements of task completion, especially around `ua_match` GPT-judge drift, the `string_match` `fuzzy_threshold` misnomer, brittle `program_html` selectors, and `finish_wrong_state` episodes. The blast radius is measurement-side: these four evaluator-class threats can flip individual labels or inflate raw success, but they do not redefine the task universe or the mode definitions; full appendix prose is kept in Section 4.X.1--4.X.4 rather than duplicated here. We therefore report both raw and adjusted success and isolate `na_fp`, `eval_fp`, and `visual_fp` filters as sensitivity layers, following the constraint-table rule that evaluator artifacts should be disclosed rather than hidden \citep{lipton2018troubling,neurips2024checklist}. This affects Sections 3 and 4: adjusted SR is the headline metric, raw SR remains visible, and claims that survive the filter ladder are defensible as representation effects rather than judge artifacts.
+VWA success labels are imperfect measurements of task completion, especially around `ua_match` GPT-judge drift, the `string_match` `fuzzy_threshold` misnomer, brittle `program_html` selectors, and `finish_wrong_state` episodes. The blast radius is measurement-side: these four evaluator-class threats can flip individual labels or inflate raw success, but they do not redefine the task universe or the mode definitions; full appendix prose is kept in Section 4.X.1--4.X.4 rather than duplicated here. We therefore report both raw and adjusted success and isolate `na_fp`, `eval_fp`, and `visual_fp` filters as sensitivity layers, including the audit-derived false-positive filters from Sections 78a and 95, following the constraint-table rule that evaluator artifacts should be disclosed rather than hidden \citep{lipton2018troubling,neurips2024checklist}. This affects Sections 3 and 4: adjusted SR is the headline metric, raw SR remains visible, and claims that survive the filter ladder are defensible as representation effects rather than judge artifacts.
 
 \subsection*{8.3 Internal-validity threats: known scaffold bugs}
 
@@ -12,11 +12,11 @@ Several scaffold bugs were real: the `in_viewport_ratio` operator-precedence bug
 
 \subsection*{8.4 Pre-vs-post-hoc analyses \& retracted framings}
 
-The exact L11/L17 mechanism-layer story was not preregistered, and several earlier framings were retracted. The blast radius is epistemic status: preregistered H1--H3 gate the deployment and structural claims, H4 is exploratory, and H5--H6 are post-hoc explanatory layers; Section 5.X states that L17 emerged from the Stage 2A pilot and L11 from an early single-task continuation, then converged across logit shift, forward overlap, reverse overlap, and cross-tier tests under Holm correction. The negative-results registry records 12 retracted framings, including task-0 over-interpretation, the reverse-null claim overturned by the N=15 reverse sample, and the rejected selection-bias explanation (cross-tier Welch tests non-significant, with reverse magnitude identical across tiers). This affects Sections 1 and 5: the paper can claim a confirmed four-fold drop-in property and a sparse mid-layer mechanism, but not a preregistered exact-layer prediction or a universal single-task circuit.
+The exact L11/L17 mechanism-layer story was not preregistered, and several earlier framings were retracted. The blast radius is epistemic status: preregistered H1--H3 gate the deployment and structural claims, H4 is exploratory, and H5--H6 are post-hoc explanatory layers; Section 5.X states that L17 emerged from the Stage 2A pilot and L11 from an early single-task continuation, then converged across logit shift, forward overlap, reverse overlap, and cross-tier tests under Holm correction. The negative-results registry records 12 retracted framings, including task-0 over-interpretation, the reverse-null claim overturned by the N=15 reverse sample, and the rejected selection-bias explanation (cross-tier Welch tests non-significant, with reverse magnitude identical across tiers). It also records two framings that survived audit: the four-fold drop-in property and the sparse L11/L17 mechanism. This affects Sections 1 and 5: the paper can claim confirmed, registry-backed evidence for those two framings, but not a preregistered exact-layer prediction or a universal single-task circuit.
 
 \subsection*{8.5 Statistical \& methodological limits}
 
-The statistical design is adequate for medium effects but underpowered for small per-cell effects and exact-layer micro-effects. The blast radius is precision, not directionality: Holm-Bonferroni is applied across the six canonical tested layers (L0/5/11/17/23/29), not all 36 cached layers; this matches the disclosed post-hoc grid but should not be read as a full-layer search correction \citep{holm1979sequentially,wang2023interpretability,zhang2024patching}. Bootstrap intervals use task-paired resampling, random-effects meta-analysis is limited to cells with N>=10 to avoid unstable tau-squared estimates, and complete-case deletion handles crashes or missing artifacts without imputation. Power analysis shows N=15 mechanism cells are not powered for small mid-layer effects (roughly Cohen's d below 0.65 at alpha=0.05), while site-level SR cells mainly detect 4--7pp effects. This affects Sections 4--5: null or marginal cells are interpreted as low-power evidence, and pooled estimates are paired with per-cell uncertainty.
+The statistical design is adequate for medium effects but underpowered for small per-cell effects and exact-layer micro-effects. The blast radius is precision, not directionality: Holm-Bonferroni is applied across the six canonical tested layers (L0/5/11/17/23/29), not all 36 cached layers; this matches the disclosed post-hoc grid but should not be read as a full-layer search correction \citep{holm1979sequentially,wang2023interpretability,zhang2024patching}. Bootstrap intervals use task-paired resampling, random-effects meta-analysis is limited to cells with N>=10 to avoid unstable tau-squared estimates, and complete-case deletion handles crashes or missing artifacts without imputation. Exclusions are listwise only, at <=5% per cell under the B6 lock, so multiple imputation would add modeling assumptions without materially changing paired denominators. Power analysis shows N=15 mechanism cells are not powered for small mid-layer effects (roughly Cohen's d below 0.65 at alpha=0.05), while site-level SR cells mainly detect 4--7pp effects. This affects Sections 4--5: null or marginal cells are interpreted as low-power evidence, and pooled estimates are paired with per-cell uncertainty.
 
 \subsection*{8.6 Sparse-mechanism caveat}
 
@@ -24,6 +24,6 @@ The activation-patching effect is sparse: at L17, four of five completed cells h
 
 \subsection*{8.7 Compute, cost, and sustainability bounds}
 
-Compute and carbon accounting are approximate because runs span DGX Spark, UCL Condense A100, and UCL Myriad V100/A100 profiles. The blast radius is absolute energy reporting: per-cell GPU-hours, proxy-API USD cost, and kg-CO2 estimates will be reported with hardware provenance, but cross-machine power variation limits precision \citep{qiu2025modserve}. Relative comparisons inside a cell remain valid because modes share the same hardware and evaluator path. This affects the cost part of the four-fold drop-in claim: token/API cost and latency comparisons are primary, while carbon totals are disclosed as bounded estimates.
+Compute and carbon accounting are approximate because runs span DGX Spark, UCL Condense A100, and UCL Myriad V100/A100 profiles. The blast radius is absolute energy reporting: the per-cell table reports GPU-hours, proxy-API USD cost, and kg-CO2 estimates with hardware provenance, but cross-machine power variation limits precision \citep{qiu2025modserve}. Relative comparisons inside a cell remain valid because modes share the same hardware, benchmark site, and evaluator path. This affects the cost part of the four-fold drop-in claim: token/API cost and latency comparisons are primary, while carbon totals are disclosed as bounded estimates rather than a fine-grained lifecycle assessment.
 
 NEEDS_BIB_ENTRY: `lipton2018troubling`, `neurips2024checklist`, `holm1979sequentially`, `wang2023interpretability`, `zhang2024patching`.

exec
/bin/bash -lc "for k in lipton2018troubling neurips2024checklist holm1979sequentially wang2023interpretability zhang2024patching; do if rg -q \"@\\w+\\{"'$k," docs/checkpoints/paper_drafts/paper.bib; then echo "$k present"; else echo "$k missing"; fi; done' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
lipton2018troubling missing
neurips2024checklist missing
holm1979sequentially missing
wang2023interpretability missing
zhang2024patching missing

diff --git a/docs/checkpoints/paper_drafts/section8_limitations.md b/docs/checkpoints/paper_drafts/section8_limitations.md
new file mode 100644
index 0000000000000000000000000000000000000000..c4c440894aef0698543754949e5e7c9b285fb3b5
--- /dev/null
+++ b/docs/checkpoints/paper_drafts/section8_limitations.md
@@ -0,0 +1,29 @@
+\subsection*{8.1 Scope \& external-validity bounds}
+
+This study is deliberately narrow: three sites (classifieds, reddit, shopping), one benchmark family (VisualWebArena plus WA-mini-style task structure), and two Qwen-family model classes: Qwen3-VL-4B locally and Qwen3-Omni-235B-Thinking through a proxy API. The blast radius is external validity, not the internal comparison: our claim is that a phantom routing space exists for Qwen-family agents on VWA-style tasks, not that every VLM, browser benchmark, or production site will expose the same arm ordering \citep{zhou2024webarena,koh2024visualwebarena,deng2023mind2web,drouin2024workarena}. The mechanism evidence in Section 5 is narrower still: it is B1-only, because the open-weight 4B model exposes activations and the B0 proxy model does not. Shopping also has weaker intermediate mechanism coverage than classifieds/reddit, so site-level generalization is reported cell-by-cell rather than averaged into a universal web-agent claim. This affects Sections 1, 5, and 6 by bounding language about universality and reproducibility: B1 behavior and mechanistic patching are byte-reproducible from released artifacts, while B0 is verifiable from traces and replayable subject to API access; cross-architecture claims, including GPT-4o-family claims, remain outside the paper.
+
+\subsection*{8.2 Construct validity \& evaluator threats}
+
+VWA success labels are imperfect measurements of task completion, especially around `ua_match` GPT-judge drift, the `string_match` `fuzzy_threshold` misnomer, brittle `program_html` selectors, and `finish_wrong_state` episodes. The blast radius is measurement-side: these four evaluator-class threats can flip individual labels or inflate raw success, but they do not redefine the task universe or the mode definitions; full appendix prose is kept in Section 4.X.1--4.X.4 rather than duplicated here. We therefore report both raw and adjusted success and isolate `na_fp`, `eval_fp`, and `visual_fp` filters as sensitivity layers, including the audit-derived false-positive filters from Sections 78a and 95, following the constraint-table rule that evaluator artifacts should be disclosed rather than hidden \citep{lipton2018troubling,neurips2024checklist}. This affects Sections 3 and 4: adjusted SR is the headline metric, raw SR remains visible, and claims that survive the filter ladder are defensible as representation effects rather than judge artifacts.
+
+\subsection*{8.3 Internal-validity threats: known scaffold bugs}
+
+Several scaffold bugs were real: the `in_viewport_ratio` operator-precedence bug exposed clipped DOM text, early scroll actions suffered direction-convention confusion, and Stage 2B/2C mechanism inputs came from pre-Phase-A archived browser states. The blast radius is bounded because these failures are mode-uniform within the relevant comparisons: the viewport bug affects DOM-derived text and Phantom-SoM's `[SOM_MARKS]` source together; scroll-direction confusion is a trajectory-execution threat rather than an evaluator rule; and Stage 2 uses frozen prompt/screenshot inputs, so Phase-A dispatch bugs affect which step an agent reached, not the model's forward pass on that saved step. This affects cross-mode interpretation in Sections 3--5: we treat environment failures as measurement threats and disclose their blast radius rather than folding them into cognitive claims.
+
+\subsection*{8.4 Pre-vs-post-hoc analyses \& retracted framings}
+
+The exact L11/L17 mechanism-layer story was not preregistered, and several earlier framings were retracted. The blast radius is epistemic status: preregistered H1--H3 gate the deployment and structural claims, H4 is exploratory, and H5--H6 are post-hoc explanatory layers; Section 5.X states that L17 emerged from the Stage 2A pilot and L11 from an early single-task continuation, then converged across logit shift, forward overlap, reverse overlap, and cross-tier tests under Holm correction. The negative-results registry records 12 retracted framings, including task-0 over-interpretation, the reverse-null claim overturned by the N=15 reverse sample, and the rejected selection-bias explanation (cross-tier Welch tests non-significant, with reverse magnitude identical across tiers). It also records two framings that survived audit: the four-fold drop-in property and the sparse L11/L17 mechanism. This affects Sections 1 and 5: the paper can claim confirmed, registry-backed evidence for those two framings, but not a preregistered exact-layer prediction or a universal single-task circuit.
+
+\subsection*{8.5 Statistical \& methodological limits}
+
+The statistical design is adequate for medium effects but underpowered for small per-cell effects and exact-layer micro-effects. The blast radius is precision, not directionality: Holm-Bonferroni is applied across the six canonical tested layers (L0/5/11/17/23/29), not all 36 cached layers; this matches the disclosed post-hoc grid but should not be read as a full-layer search correction \citep{holm1979sequentially,wang2023interpretability,zhang2024patching}. Bootstrap intervals use task-paired resampling, random-effects meta-analysis is limited to cells with N>=10 to avoid unstable tau-squared estimates, and complete-case deletion handles crashes or missing artifacts without imputation. Exclusions are listwise only, at <=5% per cell under the B6 lock, so multiple imputation would add modeling assumptions without materially changing paired denominators. Power analysis shows N=15 mechanism cells are not powered for small mid-layer effects (roughly Cohen's d below 0.65 at alpha=0.05), while site-level SR cells mainly detect 4--7pp effects. This affects Sections 4--5: null or marginal cells are interpreted as low-power evidence, and pooled estimates are paired with per-cell uncertainty.
+
+\subsection*{8.6 Sparse-mechanism caveat}
+
+The activation-patching effect is sparse: at L17, four of five completed cells have median Levenshtein-distance shift equal to zero, with IQRs including zero. The blast radius is the mechanism claim: mean disruption and Holm significance are carried by task subsets, approximately the high-salience-image quarter of the strong-tier cases, while many tasks show no visible continuation change; Cell E random injection, by contrast, destroys outputs broadly. This affects Section 5 by reframing the finding as a task-conditional sparse mechanism rather than a universal mid-layer circuit. The claim remains defensible because Section 5.1 reports per-task scatter/violin views alongside mean bands and because Cell E separates content-specific sparse patching from generic injection damage.
+
+\subsection*{8.7 Compute, cost, and sustainability bounds}
+
+Compute and carbon accounting are approximate because runs span DGX Spark, UCL Condense A100, and UCL Myriad V100/A100 profiles. The blast radius is absolute energy reporting: the per-cell table reports GPU-hours, proxy-API USD cost, and kg-CO2 estimates with hardware provenance, but cross-machine power variation limits precision \citep{qiu2025modserve}. Relative comparisons inside a cell remain valid because modes share the same hardware, benchmark site, and evaluator path. This affects the cost part of the four-fold drop-in claim: token/API cost and latency comparisons are primary, while carbon totals are disclosed as bounded estimates rather than a fine-grained lifecycle assessment.
+
+NEEDS_BIB_ENTRY: `lipton2018troubling`, `neurips2024checklist`, `holm1979sequentially`, `wang2023interpretability`, `zhang2024patching`.

codex
DONE: wrote section8_limitations.md (935 words across 7 subsections)

- `8.1 Scope & external-validity bounds` — 166 words; `8.2 Construct validity & evaluator threats` — 139 words.
- `8.3 Internal-validity threats: known scaffold bugs` — 118 words; `8.4 Pre-vs-post-hoc analyses & retracted framings` — 149 words.
- `8.5 Statistical & methodological limits` — 157 words.
- `8.6 Sparse-mechanism caveat` — 110 words; `8.7 Compute, cost, and sustainability bounds` — 96 words.
- NEEDS_BIB_ENTRY: `lipton2018troubling`, `neurips2024checklist`, `holm1979sequentially`, `wang2023interpretability`, `zhang2024patching`.
diff --git a/docs/checkpoints/paper_drafts/section8_limitations.md b/docs/checkpoints/paper_drafts/section8_limitations.md
new file mode 100644
index 0000000000000000000000000000000000000000..c4c440894aef0698543754949e5e7c9b285fb3b5
--- /dev/null
+++ b/docs/checkpoints/paper_drafts/section8_limitations.md
@@ -0,0 +1,29 @@
+\subsection*{8.1 Scope \& external-validity bounds}
+
+This study is deliberately narrow: three sites (classifieds, reddit, shopping), one benchmark family (VisualWebArena plus WA-mini-style task structure), and two Qwen-family model classes: Qwen3-VL-4B locally and Qwen3-Omni-235B-Thinking through a proxy API. The blast radius is external validity, not the internal comparison: our claim is that a phantom routing space exists for Qwen-family agents on VWA-style tasks, not that every VLM, browser benchmark, or production site will expose the same arm ordering \citep{zhou2024webarena,koh2024visualwebarena,deng2023mind2web,drouin2024workarena}. The mechanism evidence in Section 5 is narrower still: it is B1-only, because the open-weight 4B model exposes activations and the B0 proxy model does not. Shopping also has weaker intermediate mechanism coverage than classifieds/reddit, so site-level generalization is reported cell-by-cell rather than averaged into a universal web-agent claim. This affects Sections 1, 5, and 6 by bounding language about universality and reproducibility: B1 behavior and mechanistic patching are byte-reproducible from released artifacts, while B0 is verifiable from traces and replayable subject to API access; cross-architecture claims, including GPT-4o-family claims, remain outside the paper.
+
+\subsection*{8.2 Construct validity \& evaluator threats}
+
+VWA success labels are imperfect measurements of task completion, especially around `ua_match` GPT-judge drift, the `string_match` `fuzzy_threshold` misnomer, brittle `program_html` selectors, and `finish_wrong_state` episodes. The blast radius is measurement-side: these four evaluator-class threats can flip individual labels or inflate raw success, but they do not redefine the task universe or the mode definitions; full appendix prose is kept in Section 4.X.1--4.X.4 rather than duplicated here. We therefore report both raw and adjusted success and isolate `na_fp`, `eval_fp`, and `visual_fp` filters as sensitivity layers, including the audit-derived false-positive filters from Sections 78a and 95, following the constraint-table rule that evaluator artifacts should be disclosed rather than hidden \citep{lipton2018troubling,neurips2024checklist}. This affects Sections 3 and 4: adjusted SR is the headline metric, raw SR remains visible, and claims that survive the filter ladder are defensible as representation effects rather than judge artifacts.
+
+\subsection*{8.3 Internal-validity threats: known scaffold bugs}
+
+Several scaffold bugs were real: the `in_viewport_ratio` operator-precedence bug exposed clipped DOM text, early scroll actions suffered direction-convention confusion, and Stage 2B/2C mechanism inputs came from pre-Phase-A archived browser states. The blast radius is bounded because these failures are mode-uniform within the relevant comparisons: the viewport bug affects DOM-derived text and Phantom-SoM's `[SOM_MARKS]` source together; scroll-direction confusion is a trajectory-execution threat rather than an evaluator rule; and Stage 2 uses frozen prompt/screenshot inputs, so Phase-A dispatch bugs affect which step an agent reached, not the model's forward pass on that saved step. This affects cross-mode interpretation in Sections 3--5: we treat environment failures as measurement threats and disclose their blast radius rather than folding them into cognitive claims.
+
+\subsection*{8.4 Pre-vs-post-hoc analyses \& retracted framings}
+
+The exact L11/L17 mechanism-layer story was not preregistered, and several earlier framings were retracted. The blast radius is epistemic status: preregistered H1--H3 gate the deployment and structural claims, H4 is exploratory, and H5--H6 are post-hoc explanatory layers; Section 5.X states that L17 emerged from the Stage 2A pilot and L11 from an early single-task continuation, then converged across logit shift, forward overlap, reverse overlap, and cross-tier tests under Holm correction. The negative-results registry records 12 retracted framings, including task-0 over-interpretation, the reverse-null claim overturned by the N=15 reverse sample, and the rejected selection-bias explanation (cross-tier Welch tests non-significant, with reverse magnitude identical across tiers). It also records two framings that survived audit: the four-fold drop-in property and the sparse L11/L17 mechanism. This affects Sections 1 and 5: the paper can claim confirmed, registry-backed evidence for those two framings, but not a preregistered exact-layer prediction or a universal single-task circuit.
+
+\subsection*{8.5 Statistical \& methodological limits}
+
+The statistical design is adequate for medium effects but underpowered for small per-cell effects and exact-layer micro-effects. The blast radius is precision, not directionality: Holm-Bonferroni is applied across the six canonical tested layers (L0/5/11/17/23/29), not all 36 cached layers; this matches the disclosed post-hoc grid but should not be read as a full-layer search correction \citep{holm1979sequentially,wang2023interpretability,zhang2024patching}. Bootstrap intervals use task-paired resampling, random-effects meta-analysis is limited to cells with N>=10 to avoid unstable tau-squared estimates, and complete-case deletion handles crashes or missing artifacts without imputation. Exclusions are listwise only, at <=5% per cell under the B6 lock, so multiple imputation would add modeling assumptions without materially changing paired denominators. Power analysis shows N=15 mechanism cells are not powered for small mid-layer effects (roughly Cohen's d below 0.65 at alpha=0.05), while site-level SR cells mainly detect 4--7pp effects. This affects Sections 4--5: null or marginal cells are interpreted as low-power evidence, and pooled estimates are paired with per-cell uncertainty.
+
+\subsection*{8.6 Sparse-mechanism caveat}
+
+The activation-patching effect is sparse: at L17, four of five completed cells have median Levenshtein-distance shift equal to zero, with IQRs including zero. The blast radius is the mechanism claim: mean disruption and Holm significance are carried by task subsets, approximately the high-salience-image quarter of the strong-tier cases, while many tasks show no visible continuation change; Cell E random injection, by contrast, destroys outputs broadly. This affects Section 5 by reframing the finding as a task-conditional sparse mechanism rather than a universal mid-layer circuit. The claim remains defensible because Section 5.1 reports per-task scatter/violin views alongside mean bands and because Cell E separates content-specific sparse patching from generic injection damage.
+
+\subsection*{8.7 Compute, cost, and sustainability bounds}
+
+Compute and carbon accounting are approximate because runs span DGX Spark, UCL Condense A100, and UCL Myriad V100/A100 profiles. The blast radius is absolute energy reporting: the per-cell table reports GPU-hours, proxy-API USD cost, and kg-CO2 estimates with hardware provenance, but cross-machine power variation limits precision \citep{qiu2025modserve}. Relative comparisons inside a cell remain valid because modes share the same hardware, benchmark site, and evaluator path. This affects the cost part of the four-fold drop-in claim: token/API cost and latency comparisons are primary, while carbon totals are disclosed as bounded estimates rather than a fine-grained lifecycle assessment.
+
+NEEDS_BIB_ENTRY: `lipton2018troubling`, `neurips2024checklist`, `holm1979sequentially`, `wang2023interpretability`, `zhang2024patching`.

tokens used
94,277
DONE: wrote section8_limitations.md (935 words across 7 subsections)

- `8.1 Scope & external-validity bounds` — 166 words; `8.2 Construct validity & evaluator threats` — 139 words.
- `8.3 Internal-validity threats: known scaffold bugs` — 118 words; `8.4 Pre-vs-post-hoc analyses & retracted framings` — 149 words.
- `8.5 Statistical & methodological limits` — 157 words.
- `8.6 Sparse-mechanism caveat` — 110 words; `8.7 Compute, cost, and sustainability bounds` — 96 words.
- NEEDS_BIB_ENTRY: `lipton2018troubling`, `neurips2024checklist`, `holm1979sequentially`, `wang2023interpretability`, `zhang2024patching`.
