---
type: pre-audit-ledger
audit_date: 2026-05-15
audited_by: claude-sonnet (ground-truth pre-audit before /stress A2)
gate: /stress A2.1-A2.8 (claim-audit mode)
---

# Ground-truth pre-audit (2026-05-15)

## Summary
- Files audited: 16
  - `docs/checkpoints/paper_planning.md` (2521 lines, multi-pass)
  - `docs/checkpoints/phase1_plan.md`
  - `docs/checkpoints/pre_run/preregistration.md`
  - `docs/checkpoints/pre_run/{locked_versions,model_card,dataset_card,osf_lock_manifest,pre_rerun_audit,reeval_audit_protocol,topvenue_constraints,evaluator_change_protocol,ethics_license_coi_statements,negative_results_registry,release_redaction_checklist}.md`
  - `docs/checkpoints/paper_drafts/section{1,2,3,4_empirical_findings,4_limitations_disclosure,5,8}_*.md`
- ❌ NO PROVENANCE: 3
- 🔁 DRIFT: 8
- 🧊 FOSSILIZED: 3
- ⚠️ STALE: 9
- 🗺️ MAP-ERROR (phase1_plan §A2): 1 (suspected) / 7 OK
- ✅ Files clean (no S1/S2 issues): `pre_run/ethics_license_coi_statements.md`, `pre_run/negative_results_registry.md`, `pre_run/release_redaction_checklist.md`, `pre_run/evaluator_change_protocol.md`, `pre_run/reeval_audit_protocol.md` (minor stale only; no claim-audit blocker)

---

## S1 — would mislead /stress A2.x audit (top issues, fix before /stress)

### S1.1 [🔁 DRIFT] B2 / Gemma3-VL scope not propagated to preregistration or paper drafts §1/§4/§5/§8
- **Files**: `pre_run/preregistration.md` frontmatter L12 + §2 H1/H3 (L48–166) + §4 row "N_cells statistical" L340 + §6 commit (4) L417; `paper_drafts/section1_intro.md` L13; `paper_drafts/section8_limitations.md` L3; `pre_run/model_card.md` (whole file, B0/B1 only); `pre_run/dataset_card.md` §63; `pre_run/locked_versions.md` §"Model substrate" (B0 + B1 only).
- **Conflict**: `phase1_plan.md` L29-36 (2026-05-15) locks scope at **3 baselines × 6 modes = 36 conditions / 6 cells** (B2 = `google/gemma-3-4b-it`); `paper_planning §19` 2026-05-14 entry (L1733) confirms "Gemma3-VL 正式入 baseline"; `paper_drafts/section3_definition.md` §3.5.1 already references "B0 / B1 / B2" code paths (L117–121). But every other gating doc still asserts **2 baselines × 6 modes = 24 conditions / 4 cells**.
- **Latest canonical**: `phase1_plan.md` (2026-05-15) + memory `feedback_chronicle_on_milestone` 实验笔记 §142+.
- **Why this matters for /stress**: A2.2 (Comparison rigor / B0/B1/B2 axis) + A2.3 (statistical design N=6 cells at k=6, not k=4) + A2.4 (evidence-claim coupling for 3 baselines) + A2.8 (preregistration completeness) all anchor on the 4-cell scope. Without fix, /stress will critique under the wrong N and miss the real B2 matched-capability question.
- **Fix**: propagate "3 baselines / 36 conditions / 6 cells" to preregistration frontmatter + §2 estimand language (k=6 not k=4) + §4 N_cells row + §6 commit (4); rewrite `model_card.md` to add B2 card; add B2 to `dataset_card.md` + `locked_versions.md`; update section1 L13 + section8 L3 from "two Qwen-family model classes" to "three baselines (B0 Qwen3-VL-235B-A22B, B1 Qwen3-VL-4B, B2 Gemma3-VL 4B)".

### S1.2 [⚠️ STALE] FP architecture (post-hoc `compute_adjusted_success`) retired 2026-05-14 but still cited everywhere
- **Files**: `pre_run/locked_versions.md` L104–105 (`compute_adjusted_success` pinned as the FP filter); `pre_run/dataset_card.md` L47–63 (still describes 3-variant ladder `raw / +na_fp / +na_fp+eval_fp` + `visual_fp DEPRECATED`); `paper_drafts/section4_limitations_disclosure.md` L24, L65, L82 (cites `na_fp` / `eval_fp` filter classes); `paper_drafts/section8_limitations.md` L7 (claims `na_fp / eval_fp / visual_fp` are "sensitivity layers"); `osf_lock_manifest.md` §2.2 (still references the K_h1=0.75 + K_h3=0.67 + raw/adjusted ladder rationale).
- **Conflict**: `preregistration.md §4` rows "FP filter architecture" / "N/A task exclusion" / "FP filter sensitivity" (L331–333, 2026-05-14) explicitly RETIRE the post-hoc layer: source-level B-91 fix + N/A excluded at task-load + `adjusted_success ≡ success`. Memory `reference_fp_architecture_2026-05-14.md` is canonical.
- **Why this matters for /stress**: A2.3 + A2.7 + A2.8 will accept the stale "3-variant FP ladder" as primary spec and miss that the current canonical metric is `success` (not `adjusted_success`). Section4_limitations_disclosure §4.X.1, §4.X.3, §4.X.4 prose rationale collapses under the new architecture.
- **Fix**: rewrite locked_versions.md L99–108 + dataset_card.md L47–73 + section4_limitations_disclosure.md §4.X (whole subsections rationale), removing the post-hoc filter language; cite memory `reference_fp_architecture_2026-05-14.md` + master_bug_catalog B-91 + preregistration §4 2026-05-14 row.

### S1.3 [⚠️ STALE] K-of-N gate (K_h1=0.75, K_h3=0.67) retired 2026-05-14 but `osf_lock_manifest.md` + `paper_planning §5` still cite as gating
- **Files**: `osf_lock_manifest.md §1` L15 + §2.2 L54–55 ("K_h1 transparency ratio (NOT a gate)" — language hedged but the ratios still appear as `0.75 → 3 of 4 cells`); `paper_planning.md §5` L1031 R1 cell ("Hero + Structural 全过 (K_h1≥0.75, K_h3≥0.67)"); §19 decision log L1722 + L1725 still presents K_h1/K_h3 as locked gate values.
- **Conflict**: `preregistration.md §2.5` (L249-269) + §4 row "K-of-N per-cell consistency" (L325) + Appendix A 2026-05-14 entry (L476) retire the ratios entirely ("at N=4, ⌈0.75×4⌉=⌈0.67×4⌉=3 — fake precision"). The single primary gate is now H1 one-sided FE superiority at δ=1.0pp.
- **Why this matters for /stress**: A2.3 (statistical design) + A2.8 (pre-registration completeness) will be confused by `paper_planning §5` probability tree which keys on K_h1/K_h3 thresholds — when the preregistration says these are transparency-only counts.
- **Fix**: rewrite `paper_planning §5` R1 cell ("K_h1≥0.75, K_h3≥0.67" → "FE θ > +1.0pp AND both H3 axes FE CI > 0"); `paper_planning §19` decision log: add a 2026-05-14 row that supersedes the 2026-05-03 K_h1/K_h3 lock; `osf_lock_manifest §2.2` rows: rename ratios → "transparency-only counts".

### S1.4 [🗺️ MAP-ERROR / ⚠️ STALE] `phase1_plan §A2.6` artifact `paper_planning §6` is now ambiguous — §6 is "Critical Risks", but the R3 framing language §A2.6 demands lives in §5
- **File**: `phase1_plan.md` L112 §A2.6: "Scope / external validity ... (artifacts: [[paper_planning]] §6 R1-R5 + `pre_run/preregistration.md` §7 reproducibility scope)".
- **Conflict**: `paper_planning.md §6` title "Critical Risks + Mitigation (4 risks)" (L1066); R1-R5 framing rule lives in `paper_planning §5` ("Final Scope + 顶刊概率", L1015–1047) + `preregistration §2` ("FRAMING DECISION RULE", L226–239). §A2.6 reviewer will land on §6 (risks) and miss the framing tree.
- **Why this matters for /stress**: A2.6 audit ("Phase 1a only cls+red → R3 framing risk") will not find the R1-R5 mapping in the cited §6. Either /stress concludes "no framing rule preregistered" (false negative) or asks for re-direction.
- **Fix**: `phase1_plan §A2.6` change pointer `[[paper_planning]] §6` → `[[paper_planning]] §5 + preregistration §2 framing rule R1-R5`.

### S1.5 [⚠️ STALE] `ADVISOR_SYNC.md` retired (file does not exist) but referenced from 4 active gating docs
- **Files**: `osf_lock_manifest.md` L6 (cites "笔记 §110 + ADVISOR_SYNC §F"), L126 ("`docs/checkpoints/ADVISOR_SYNC.md` §F (OSF DOI workflow detail)"); `preregistration.md` L26 ("`docs/checkpoints/ADVISOR_SYNC.md` §1 — advisor sync prep, lock decision questions"); `phase1_plan.md` L120 ("详细 framing decision log → [[paper_planning]] §19 + [[ADVISOR_SYNC]]"); `paper_planning.md` L1809, L1857, L1867, L2454 (4 hits) reference ADVISOR_SYNC.md status fields and workflow.
- **Conflict**: File does not exist on disk (`ls docs/checkpoints/ADVISOR_SYNC.md` → not found). Per CLAUDE.md memory log + 实验笔记 §142+, ADVISOR_SYNC.md was retired 2026-05-15; replaced by `_status/issues/issue_advisor_sync_2026-05-14.md` frontmatter pattern.
- **Why this matters for /stress**: A2.8 (pre-registration completeness, witness mechanism) checks that the 9 commit decisions trace to a witness register. If the audit follows the link, it dead-ends; /stress can wrongly conclude witness chain is broken.
- **Fix**: replace `ADVISOR_SYNC.md` pointers with either `_status/issues/issue_advisor_sync_2026-05-14.md` or `paper_planning §19 + 实验笔记 §137-§142`. Specifically: preregistration.md L26 + osf_lock_manifest.md L6/L126 + phase1_plan.md L120 + paper_planning.md L1809/L1857/L1867/L2454.

### S1.6 [🧊 FOSSILIZED] Router scope flipped 2026-05-14 ("Phase 1 并行核心") but preregistration H7/H8 still bannered "DEFERRED, NOT PART OF THIS DOI CLAIM"
- **Files**: `pre_run/preregistration.md §2` L189–224 (H7/H8 "ROUTER family — ⚠️ DEFERRED, NOT PART OF THIS DOI CLAIM"); `pre_run/preregistration.md §6` commit (8) L421 ("Router H7/H8 = DEFERRED (paper-2)"); contradicts `paper_planning §19` 2026-05-14 entry L1735 ("Router = Phase 1 并行核心线 — 双路线: (a) rule-based ... (b) learned classifier") + CLAUDE.md contribution scope §2 + `phase1_plan §C` (router as core line, gated on §B done).
- **Conflict**: Router moved from paper-2 defer → paper-1 core 2026-05-14 advisor sync. Preregistration H7/H8 "DEFERRED" banner is now obsolete; phase1_plan.md treats router as core (§A2.5 explicitly audits router operationalization).
- **Why this matters for /stress**: A2.5 (router operationalization rigor) will read preregistration § H7/H8 banner saying "deferred, no test" and conclude there is no preregistered router commitment — when the operational scope is now paper-1. A2.4 (evidence-claim coupling) similarly under-counts contributions if router H7/H8 is treated as deferred.
- **Fix**: rewrite `preregistration §2 ROUTER family` banner: remove "DEFERRED" status; mark H7/H8 as PRIMARY (or at minimum SECONDARY) gating for §6 routing contribution. Update §6 commit (8) accordingly. Add Appendix A 2026-05-14 entry on the router-flip decision.

### S1.7 [🔁 DRIFT] Reddit SoM raw SR triple drift: section4 13.81%, section1 implied raw≈13.81%, sr_fp_per_mode 11.90%, B0_cross_site_findings 13.33%
- **Files**: `paper_drafts/section4_empirical_findings.md` L35 (SoM reddit adjusted SR 10.48 — matches sr_fp_per_mode); `paper_drafts/section1_intro.md` L5 ("Phantom-SoM 13.81% vs full SoM 10.48%, N=210"); `docs/analysis/cross_sites/sr_fp_per_mode.md` L10 ("B0 | reddit | SoM | 210 | 11.90% | 10.48%"); `docs/analysis/cross_sites/B0_cross_site_findings.md` L20 ("SoM 13.33% (28/210) raw / 10.48% adjusted").
- **Conflict**: SoM reddit RAW SR appears in three different files as 11.90% / 13.33% / (implied via section1 hero claim CI). Adjusted SR is consistent at 10.48% across all sources. Section1's "head-to-head reddit single-mode comparison" hero claim and section5 §5.1 line 5 "+3.33pp gap" depend on raw vs adjusted being the same direction.
- **Why this matters for /stress**: A2.4 (evidence-claim coupling) will scrutinize whether the 13.81 vs 10.48 head-to-head is robust. If raw SR is 11.90% (sr_fp_per_mode = canonical post-§139.8) the head-to-head changes from 13.81/11.90 = +1.91pp raw to 13.81/10.48 = +3.33pp adjusted — a 2× swing depending on metric choice. Section1 frames "competitive parity" via the adjusted number; a /stress reviewer may detect the inconsistency across legacy docs and discount the claim.
- **Fix**: identify canonical post-§139.8 source (sr_fp_per_mode.md is current), retract B0_cross_site_findings.md L20 numbers, and add a footnote in section1 + section4 clarifying which SR variant is reported.

### S1.8 [🧊 FOSSILIZED] §5 mechanism = "暂搁" (advisor 2026-05-14) but section5_mechanism.md (42KB, 201 lines) still the paper's headline mechanism section
- **Files**: `paper_drafts/section5_mechanism.md` full file; `paper_planning §19` 2026-05-14 entry L1734 ("Mechanism 暂搁 — §5 (patching / layer probe / logit lens / SAE) 整个先不管"); CLAUDE.md "Paper-1 contribution scope" §3 ("**~~Mechanism~~ 暂搁** — advisor discussion 2026-05-14 'mechanism 部分先不要管了'").
- **Conflict**: Paper-1 contribution scope per CLAUDE.md = (1) Phenomenon + (2) Router; mechanism deferred. But section5_mechanism.md is 42KB of detailed L11-L17 patching / cosine geometry / lm_head amplification claims that section1 L11 + section8 L15 both endorse as Contribution 2 ("mechanism account for why the ablation works").
- **Why this matters for /stress**: A2.1 (research-question framing) will see section1 promising "mechanism contribution" but the project plan says mechanism is shelved. A2.4 will count §5 claims toward the "evidence-claim coupling" check that doesn't apply if §5 is out-of-scope. A2.8 will flag the preregistration §5.X "Stage 2 mechanism layer-selection disclosure DEFERRED" banner against the paper draft still claiming it.
- **Fix**: decide whether mechanism stays IN scope for the workshop submission (then update CLAUDE.md + §19 entry), or OUT (then rewrite section1 L11–13 to drop the mechanism contribution claim, mark section5_mechanism.md as `paper-2 draft`, and update section8 L15 accordingly). This is decision-level, not a copy-edit.

### S1.9 [🔁 DRIFT] Phase 1a condition count: preregistration says 24, phase1_plan says 36
- **Files**: `pre_run/preregistration.md` frontmatter L12 ("cls+red × B0+B1 × 6 modes = 24 operational conditions across 4 statistical cells"); `pre_run/preregistration.md §4` row "N_conditions Phase 1a (operational)" L339 ("24 conditions"); `pre_run/preregistration.md §4` row "N_conditions Phase 1b (main paper, deferred)" L341 ("+12 conditions = shop × 2 models × 6 modes"). VS `phase1_plan.md` §0 L36 ("Phase 1a = 36 conditions / 6 cells"); CLAUDE.md "Phase 1a scope split" line 28 ("⚠️ 2026-05-14 update: Gemma3-VL 正式入 baseline → Phase 1a 现含 3 模型 ... cls + red × {B0, B1, Gemma3-VL} × 6 modes = 36 conditions / 6 cells, 待 planning confirm").
- **Conflict**: Same numbers, different documents, different totals: 24 (preregistration) vs 36 (phase1_plan). This is the same drift as S1.1 but specifically affects the condition counter that A2.3 power analysis depends on.
- **Why this matters for /stress**: A2.3 (statistical design) — N=4 vs N=6 cells changes per-cell power, FE inverse-variance pooling effective sample size, and the "fake precision" K-of-N rationale (3/4 ≈ 3/6 ≠ same threshold logic). The preregistration §2.4 power statement (L243–248) and §2.5 decision flow (L249-269) are written for k=4 and may need rewrite at k=6.
- **Fix**: choose canonical scope (likely phase1_plan = current); update preregistration frontmatter L12 + §4 rows L339-341 + §2 estimand language to k=6 with 3 baselines. Re-evaluate power calculations (`docs/analysis/cross_sites/power_analysis.md`) at k=6.

### S1.10 [❌ NO PROVENANCE] "≈1.49% rescue rate" GLM fallback claim in section3.5.1 is "current B0 archive" but no provenance file cited
- **Files**: `paper_drafts/section3_definition.md` L121 (`§3.5.1 Parse-error recovery scaffold (provisional)` — "the rescue rate measured on the current B0 archive is ≈ 1.49 % of steps").
- **Provenance**: searched `docs/analysis/` + `docs/reference/` for "1.49" or "glm_fallback" → no aggregator output found. The figure is presented as `provisional` with "specific paper text pending advisor decision" caveat, but the audit count is not sourced.
- **Why this matters for /stress**: A2.7 (confound register / known asymmetries) will demand: which runs were aggregated, what step denominator, paper-grade or archive vintage, did the 1.49% include both pre-§139.8 and post-§139.8 runs? Without provenance file, the rate becomes hand-waving.
- **Fix**: add provenance pointer (e.g., `docs/analysis/.../glm_fallback_rate.md` aggregator output) or retract the specific number until aggregator lands. This is also S1.2-adjacent (paper_grade clean re-run hasn't happened post-§139.8).

### S1.11 [⚠️ STALE] section4_limitations_disclosure.md cites retracted `visual_fp` filter + `compute_adjusted_success` framework as live policy
- **Files**: `paper_drafts/section4_limitations_disclosure.md` L11–195 (multiple subsections cite "Mitigation: ... `na_fp` exclusion class" / "FP filter sensitivity ladder" / "Appendix D"); section8_limitations §8.2 L7 explicitly cites `visual_fp` as a sensitivity layer; `dataset_card.md` L54 ("visual_fp DEPRECATED").
- **Conflict**: per memory `reference_fp_architecture_2026-05-14.md` + preregistration §4 2026-05-14 row, all three FP classes (`na_fp`, `eval_fp`, `visual_fp`) are retired in favor of source-level fixes. `visual_fp` was already deprecated; now `na_fp` and `eval_fp` are also retired. section4_limitations_disclosure was written 2026-05-09 (per filemtime); it predates the retire.
- **Why this matters for /stress**: A2.7 — section4_limitations_disclosure §4.X.1 (ua_match GPT-judge drift mitigation = "na_fp exclusion class") and §4.X.3 (program_html selector brittleness = "eval_fp filter") are the disclosure paragraphs reviewers will read; they describe a defense that no longer exists.
- **Fix**: rewrite section4_limitations_disclosure §4.X.1, §4.X.3, §4.X.4 mitigation paragraphs against the new architecture: na_fp → fixed at evaluator level via B-91; eval_fp → branch dropped (RESET_BEFORE upstream); N/A tasks → excluded at task-load. Same fix as S1.2.

### S1.12 [❌ NO PROVENANCE] +50.0pp / +33.3pp B0→B1 SoM visual-hijack claim in section4 has only a paper_planning §2 anchor, no analysis-file source
- **Files**: `paper_drafts/section4_empirical_findings.md` L42 ("B0-to-B1 SoM visual-hijack/click-loop increasing by **+50.0 pp** on classifieds and **+33.3 pp** on reddit"). Section1 also implicitly trades on this via "cross-capability robustness check".
- **Provenance**: `paper_planning.md` L564 records the numbers without aggregator pointer ("SoM hijack flip cross-site: cls +50.0pp, red +33.3pp (vs aggregate +43.7pp)"). Searched `docs/analysis/cross_sites/failure_modes_per_cell.md` + `docs/analysis/vwa_*` — no aggregator file outputs these specific deltas. The "+43.7pp" aggregate was retired 2026-05-09 (CLAUDE.md "third contribution dropped 2026-05-09; B1 = cross-capability robustness check").
- **Why this matters for /stress**: A2.4 (evidence-claim coupling) will demand the source. Section4 makes this the section's mechanism-evidence pivot ("We treat that asymmetry as mechanism evidence rather than a setup bug"). Without backing aggregator, /stress will conclude the claim is unsupported.
- **Fix**: locate the failure-mode aggregator that produced +50.0pp/+33.3pp (probably an old `disagreement_clusters.md` variant), pin the pointer, OR retract the specific numbers and substitute a sourced cross-capability statement.

---

## S2 — internal inconsistency (fix opportunistically)

### S2.1 [🔁 DRIFT] section4 P-text reddit adjusted SR appears as 12.38 (table L37) but section5 hero_claim caveat says "11.90 / 12.38 / 12.38 — canonicalize"
- **Files**: `paper_drafts/section4_empirical_findings.md` L37 (P-text reddit adjusted 12.38); `section5_mechanism.md` L194 NOTE FOR HUMAN: "§4 table line 37 says 11.90, prose line 106 says 12.38, hero_claim_bootstrap_ci.md says 12.38". The NOTE FOR HUMAN identifies the inconsistency as cross-doc follow-up.
- **Conflict**: per section5_mechanism.md L194, the §4 table number drift is a known TODO. Current table reads 12.38 (matches hero_claim + sr_fp_per_mode), but the NOTE FOR HUMAN warning implies an earlier 11.90 version was committed elsewhere.
- **Why this matters for /stress**: A2.4 will spot the same inconsistency.
- **Fix**: as recorded in section5_mechanism.md L194, canonicalize to 12.38 (matches sr_fp_per_mode.md) and remove the NOTE FOR HUMAN.

### S2.2 [🧊 FOSSILIZED] "Phase 1 final scope: 6 sites × 3 models × 5 modes" in paper_planning §5 L1010 conflicts with current §4 scope of `paper §4` (2 sites × 2 models × 6 modes) + Phase 1b deferred (+ shop)
- **Files**: `paper_planning.md §5` L1004-1014 ("Benchmark: VWA 3 站 + WA 3 站 = 6 sites, ~1390 task per condition; Models: B0 + B1 + Claude Opus 4.7 = 3 model families; Modes: DOM/SoM/Vision/Phantom-SoM/P-text = 5 modes; Cells: 6 × 3 × 5 = ~90 cells").
- **Conflict**: Current preregistration Phase 1a = 4 cells × 6 modes (or 6 × 6 if B2). WA out-of-scope per §F3 (preregistration §7). "Claude Opus 4.7" as a baseline was dropped (paper_planning §19 2026-05-09 "third contribution capability×representation cut"). "5 modes" obsolete (now 6 with P-prompt re-included).
- **Why this matters for /stress**: A2.6 (scope / external validity) — /stress reading §5 "final scope" will model the comparison universe wrong. The §5 probability tree assumes WA + Claude + ~90 cells.
- **Fix**: rewrite §5 "Final Scope" block at top: Phase 1a (current paper-1 workshop) + Phase 1b (deferred main-paper) + paper-2 mechanism + paper-2/3 routing. Mark "6 sites × 3 models × 5 modes / 90 cells" as RETRACTED 2026-05-13 codex stress audit scope reframe.

### S2.3 [🔁 DRIFT] section1 explicitly says "Routing exploitation is deferred to a follow-up paper" (L13) — conflicts with router-now-in-scope decision 2026-05-14
- **Files**: `paper_drafts/section1_intro.md` L13. Conflicts with `paper_planning §19` 2026-05-14 router-un-defer entry L1735 + `phase1_plan §C` "router 双路线 checklist (gated on §B done)" L172.
- **Conflict**: Section1 currently routes router to paper-2; project decision routes router to paper-1.
- **Why this matters for /stress**: A2.4 evidence-claim coupling — /stress will check whether section1's contribution count (1 = phantom, 2 = mechanism, [implicit 3 = future router]) matches the project's planned contributions. Mismatch surfaces as "what is this paper actually claiming?" attack.
- **Fix**: depending on S1.8 resolution: if router is paper-1 core, rewrite section1 L13 from "deferred to a follow-up paper" → "Section 6 reports rule-based and learned-classifier router results"; add a third contribution paragraph.

### S2.4 [🔁 DRIFT] section1 L13 promises "Section 6 discusses generalization, Section 7 summarizes limitations" but draft files are `section{1,2,3,4,5,8}` — there is no §6 or §7 draft
- **Files**: `paper_drafts/section1_intro.md` L13 ("Section 6 discusses generalization, and Section 7 summarizes limitations and implications"); `ls paper_drafts/section*.md` → no section6 or section7. `section5_mechanism.md` L196 NOTE FOR HUMAN already flagged this: "§6 + §7 drafts missing".
- **Conflict**: Already a known TODO in section5's NOTE FOR HUMAN, but section1 promises an outline that the paper structure doesn't yet support.
- **Why this matters for /stress**: A2.6 (scope) — if /stress is told "Section 6 covers generalization" but no draft exists, it cannot verify the scope claims; will conclude "outline is fictional".
- **Fix**: either draft section6 + section7 minimal stubs, OR rewrite section1 L13 final paragraph to match the actual section layout (sections 1-5 + 8 limitations).

### S2.5 [🔁 DRIFT] paper_planning §1 hook L40 says "Latency ~50% lower (cls SoM 74s vs Phantom-SoM 18.2s = 4× faster)" — "~50% lower" vs "4× faster" are inconsistent magnitudes
- **Files**: `paper_planning.md §1` L40, L45, L102, L976, L984; `paper_drafts/section1_intro.md` L7 carefully says only "lower image-stage latency" (no specific multiplier); `section4_empirical_findings.md` L46 ("roughly 4x faster").
- **Conflict**: 18.2s / 74.0s ≈ 4× faster, i.e. latency ≈ 25% of SoM, i.e. ≈ -75% reduction. "~50% lower" implies latency ≈ 50% of SoM, which would be 37s not 18.2s.
- **Why this matters for /stress**: A2.4 — the 4-fold drop-in property "(b) Latency ~50% lower" is one of the 4 sub-claims; if the magnitude is loose, /stress will spot it and ask which is the canonical claim.
- **Fix**: standardize paper_planning to "~75% lower (4× faster)" on cls; on reddit it's 51.4s vs 58.9s = only ~12% lower; consider whether "(b) latency ~50% lower" is overstated on reddit and needs site-conditional language.

### S2.6 [⚠️ STALE] preregistration §6(a) commit-list still says "9 commit decisions" L413 but §19 + Appendix A continue to accrete (the 2026-05-14 "3A" entry added several knock-on cleanups not in the 9)
- **Files**: `preregistration.md §6(a)` L413-423.
- **Conflict**: Originally 5 commits (5/3), expanded to 8 (5/4), to 9 (5/14 "3A"). Appendix A 2026-05-14 entry (L476) adds: H1 simplification fold-in, K-of-N retire, heterogeneity-conditional reframe, H7/H8 banner, §2.4/§2.5 power+decision flow, §6(b) OSF workflow → osf_lock_manifest reference. None propagated into the witness commit list.
- **Why this matters for /stress**: A2.8 — pre-registration completeness audit will ask whether the witness mechanism covers ALL locked choices. Currently the witness chain commits 1-9 but Appendix A has 5+ more locked decisions not enumerated.
- **Fix**: re-enumerate witness commits in §6(a); either expand to 12-14 commits or add a clear "(plus all Appendix A 2026-05-14 entries)" inclusion clause.

### S2.7 [🔁 DRIFT] osf_lock_manifest §2.2 row "H1 PRIMARY gate" still cites "Pooled DerSimonian-Laird meta + magnitude θ_RE ≥ 1.0pp + one-sided superiority" — preregistration retired DL 2026-05-14
- **Files**: `osf_lock_manifest.md §2.2` L51 (H1 row); preregistration §2 H1 estimand L54–98 (FE, not DL); §4 row "Pooling estimator + heterogeneity pre-spec" L345.
- **Conflict**: osf_lock_manifest predates the 2026-05-14 "3A" decision that retired DL in favor of FE inverse-variance pooling over the 4 planned cells.
- **Why this matters for /stress**: A2.3 — /stress reading osf_lock_manifest as the OSF DOI artifact freeze will model the wrong estimator. If the OSF DOI is ever minted with this stale row, audit trail diverges from preregistration.
- **Fix**: rewrite osf_lock_manifest §2.2 H1 row to match preregistration "FE inverse-variance pooled θ_FE > 1.0pp one-sided" (single test, m=1).

### S2.8 [🔁 DRIFT] preregistration §2 H1 + §4 say k=4 cells but if Gemma3-VL (B2) is in scope, k=6 and the FE estimand changes
- **Files**: `preregistration.md §2` L58-64 estimand "4 *planned* (site, model) cells"; §4 row N_cells L340 "4 cells".
- **Conflict**: same as S1.9 but specifically about the FE estimand. At k=6 the "no τ² needed" rationale (Veroniki k<10) still holds, but the per-cell weights and SE_FE = sqrt(1/Σw_i) numerics shift.
- **Why this matters for /stress**: A2.3 statistical design — moving from k=4 to k=6 doubles the cell count for B2 inclusion. The decision flow §2.5 L249-269 (steps 1-7) hard-codes k=4.
- **Fix**: depends on S1.1 / S1.9 resolution; if 3 baselines locked, rewrite §2 estimand + §2.5 decision flow at k=6.

---

## S3 — minor / cosmetic (defer)

### S3.1 [⚠️ STALE] `paper_planning §1` retro-update note L13 (2026-05-04 deepest-evening late) cites "2026-05-04" framing — last updated date is 11 days stale
- **File**: `paper_planning.md` L13 ("**Last updated**: 2026-05-04 deepest-evening late ...").
- **Conflict**: paper_planning has had 2026-05-14 entries added to §19 (L1729-1736); the top-of-file last-updated is no longer accurate.
- **Fix**: bump top-of-file timestamp; cosmetic only.

### S3.2 [🧊 FOSSILIZED] negative_results_registry.md (May 9 vintage, 7.6KB) likely missing some 2026-05-12+ retractions (e.g. axis-2 logit lens of-means → per-task)
- **File**: `pre_run/negative_results_registry.md`.
- **Conflict**: paper §5 + §8 reference "12 retracted framings"; the registry would benefit from a check that the 12 include the 2026-05-13 "of-means lens understated reddit axis-1" reframe.
- **Fix**: verify the registry covers all retractions cited in section8 L15; not blocking /stress A2.

### S3.3 [⚠️ STALE] topvenue_constraints.md (54KB, dated 2026-05-14) row B2 ("bootstrap details") name-collision with B2 baseline (Gemma3-VL)
- **File**: `pre_run/topvenue_constraints.md` (L24, L53, L156, L176 all use "B2" as the audit-constraint ID).
- **Conflict**: post-B2-baseline-naming (2026-05-14), the audit-constraint "B2" letter-prefix is now ambiguous with the model "B2".
- **Fix**: rename audit-constraint IDs to `AUDIT-B2` or similar; cosmetic / future-clarity.

### S3.4 [🔁 DRIFT] section3_definition.md §3.5.1 disclosure L117-121 lists "B0 / B1 / B2" — but section2/section4/section8 still only mention B0/B1
- **File**: `section3_definition.md` L117–121.
- **Conflict**: section3 is the only paper draft that consistently uses B0/B1/B2 terminology. Section1, section2, section4, section5, section8 all 2-baseline.
- **Why this matters**: not critical for /stress (section3 disclosure is self-contained), but reflects incomplete propagation.
- **Fix**: covered by S1.1.

### S3.5 [⚠️ STALE] paper_planning.md §3 L805 "LLM mechanism = Explanation layer (paper Section 5)" — Section 5 暂搁 per S1.8 makes this section header obsolete
- **File**: `paper_planning.md` L805-826 (mechanism subsection).
- **Conflict**: covered by S1.8.

---

## phase1_plan §A2 map-check

- **§A2.1**: `paper_drafts/section1_intro.md` ✅ exists | `paper_planning §1` ✅ exists ⇒ **OK**
- **§A2.2**: `paper_planning §15 prior-work table` ✅ (L1481-1502) | `实验笔记 §138` — chronicle referenced not checked here ⇒ **OK**
- **§A2.3**: `pre_run/preregistration.md §2.4 / §3 / §4` ✅ exists | `power_analysis.py` referenced in §A2.3 as comments — `docs/analysis/cross_sites/power_analysis.md` exists; .py script not searched here ⇒ **OK** (verify power_analysis.py vs .md when audit runs)
- **§A2.4**: `paper_planning §3` ✅ exists (L603) | `paper_planning §21` ✅ exists (per L23 paper_planning header reference; not directly verified) ⇒ **OK**
- **§A2.5**: `paper_planning §8` ✅ exists (L1166 "Router Design (Tier 1+2)") | `p79/experiment/router.py` — code file, not audited here ⇒ **OK**
- **§A2.6**: `paper_planning §6` ⚠️ **MAP-ERROR** — §6 is "Critical Risks", but R1-R5 framing rule lives in §5 + preregistration §2. ⇒ See **S1.4** above.
- **§A2.7**: `CLAUDE.md Guard Rails` ✅ exists | `实验笔记 §139 B-86` + `memory/project_paper_hook.md` — chronicle/memory references not directly verified here ⇒ **OK**
- **§A2.8**: `pre_run/preregistration.md` ✅ exists | `pre_run/osf_lock_manifest.md` ✅ exists ⇒ **OK** (but see S1.5 ADVISOR_SYNC dead link + S2.6 commit count + S2.7 H1 DL→FE drift)

---

## Recommended fix order (top 5, what unblocks /stress A2.x fastest)

1. **S1.1 + S1.9 + S2.8** — Propagate B2 / Gemma3-VL / 3-baseline / 36-condition / 6-cell scope through preregistration frontmatter + §2 + §4 + paper_drafts/section1 + section8 + model_card + dataset_card + locked_versions. Single coordinated edit. **Unblocks A2.2 / A2.3 / A2.4 / A2.8.**
2. **S1.2 + S1.11** — Rewrite section4_limitations_disclosure §4.X.1/§4.X.3/§4.X.4 + locked_versions §"Evaluator code" + dataset_card §"Exclusion (post-FP-filter)" against retired post-hoc FP architecture (use memory `reference_fp_architecture_2026-05-14.md` as canonical). **Unblocks A2.3 / A2.7 / A2.8.**
3. **S1.6 + S2.3** — Decide router scope (paper-1 core vs paper-2 defer) and propagate: remove H7/H8 "DEFERRED" banner in preregistration §2 if paper-1; rewrite section1 L13 router-deferral sentence. **Unblocks A2.5 / A2.4.**
4. **S1.5 + S1.4 + S2.6** — Replace dead `ADVISOR_SYNC.md` pointers (4 docs) with `_status/issues/issue_advisor_sync_2026-05-14.md`; fix phase1_plan §A2.6 pointer §6 → §5 + preregistration §2; expand preregistration §6(a) witness commit list to cover Appendix A 2026-05-14 entries. **Unblocks A2.6 / A2.8.**
5. **S1.8** — Make the §5 mechanism in/out decision explicit. Either edit CLAUDE.md / §19 to put mechanism back in scope and validate the section5_mechanism draft; or mark section5_mechanism.md as `paper-2 draft` and rewrite section1 L11–13 to drop the mechanism contribution. **Unblocks A2.1 / A2.4 + clarifies which sections /stress should audit.**

---

## Files clean (no S1/S2 issues surfaced)

- `pre_run/ethics_license_coi_statements.md` (only includes ethics statements, no drift-prone framing claims)
- `pre_run/negative_results_registry.md` (registry-style, content not audited beyond S3.2 cosmetic check)
- `pre_run/release_redaction_checklist.md` (release-hygiene checklist, no claim-anchoring)
- `pre_run/evaluator_change_protocol.md` (Protocol A tier classification, T0-T3 framework still current)
- `pre_run/reeval_audit_protocol.md` (Protocol B audit-trail mechanics, still current)
- `paper_drafts/section2_background.md` (citations + 4-fold-drop-in framing; only minor 2-baseline language which is covered by S1.1)
- `paper_drafts/section3_definition.md` (the only paper draft already using B0/B1/B2 terminology; cost/measurement protocol §3.5.1 disclosures up-to-date except S1.10 GLM rate)

## Audit scope NOT covered (out of bounds)

- Post-data interpretation (deferred to post-clean-run /stress; current Phase 1a not yet rerun under §139.8 FP architecture, so adjusted_success numbers in section1/4 are pre-fix vintage and may shift)
- Reviewer rehearsal (deferred to M5 submit per phase1_plan §A boundary §1)
- Code↔prose mismatch (that's /stress A1.x territory — implementation-layer audit, code-audit mode)
- Chronicle-side fact-checks (实验笔记 §95 / §128 / §138 / §139 / §142 cross-referenced but not audited as primary source)
- Figure / `docs/analysis/figures/` provenance (covered by `EVIDENCE_LAYER_AUDIT.md` not re-audited here)
- paper.bib citation correctness (cited only as cross-reference; bibkey existence is `section5_mechanism.md` line 175 "Bibkeys audit 2026-05-12 21:18" job, not this audit)
