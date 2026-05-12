# Codex Prompt — Paper §5 Mechanism Prose Round v1

## Goal

Generate **paper §5 mechanism prose v1** from `docs/checkpoints/mechanism/plan.md` §5 evidence layer + today's (2026-05-12) cross-site replication results. The current paper draft `section5_mechanism_reddit.md` contains **behavioral routing evidence** (Outcome / Macro / Micro axis-1/2/3 analysis) that mis-fits §5 — it should be relocated to §4 empirical findings or a new behavioral section. Section 5 should become a **mechanism interpretability** chapter built around PCA cosine gap, mean-diff steering, activation patching, and H1 format variation.

This is the deferred prose round explicitly tracked in `plan.md` §7.4 "Decisions pending" → "Paper §5 prose round — Codex + me — After v2 full + Zekun decision". Method 4.4 v2 full sweep finalized 2026-05-11 22:00. Zekun decision still pending but does not block this v1 prose round — Method 4.5 (LA-HDMI vs SAE) belongs to paper §8 future work, not §5.

## Why

Plan.md §5 dashboard contains 5 complete sub-findings backed by extensive evidence:
- **§5.1** Method 4.2 PCA cosine gap, Qwen3-VL-4B B1 cls 288 examples × 37 layers, AUROC 1.0, 5/5 robustness tests passed
- **§5.2** Stage 2/3 patching disruption, 16 cells (10 Stage 2 + 6 Stage 3) covering A/B/C/D/E/F/G/Cr/Dr/Er/H-d/H-t/H-p × cls + reddit
- **§5.3** Method 4.4 v2 full 45/48-cell sweep, L33 α=10 H-mean 0.33 sweet spot (L17 α=5 smoke 0.44 refuted as variance artifact, 笔记 §126/§127)
- **§5.4** Image-axis peak-layer dichotomy, 8 mode-pair table, AXTree-side L04 (4/4), marks-side L17-L36 (4/4), zero overlap
- **§5.5** H1 format variation, 8 indexed-list variants + 2 controls, AXTree-DOM is unique format preserving L04 image-axis peak

Plus **§7.3.1 reddit cross-site results landed 2026-05-12** that are NOT YET in any paper draft:
- P5b reddit Mirage signature: P-SoM↔DOM L17 = 0.0098 (text-axis sibling), P-SoM↔SoM L17 = 0.0423 (image-axis split), DOM↔Vision peak L04 AUROC 1.0
- P5a reddit format H1: marks-like 4/6 peak L17 (true mid-layer fusion, cleaner than cls L36 monotonic-boundary artifact)
- P4 cls reverse-tier H1: same pattern as strong-tier, selection-bias defended
- Stage 3 6/6 cells closed today: cls H-d -0.33 @L17, reddit H-d -0.33 @L11, all in mid-layer L11-L17 disruption -0.19 to -0.33

The current paper §5 draft is from 2026-05-09 and is **out of date by ~3 days**: it says "Cells F/G are cross-site reddit and remain pending" (false), "B1 reddit phantom runs are still pending" (false), and "Section 6 routing implementation will leverage these mechanism insights" (contradicts the paper-1 → paper-2 routing deferral framing in `paper_planning.md` §1).

The audit constraint here is **paper-grade evidence-to-prose translation**, NOT new findings. Every claim in the prose must trace to a plan.md §5 row or a `docs/checkpoints/mechanism/results/*.md` file or a `results/mechanistic/*/pilot_summary.md`. No speculation, no over-claim, no new mechanism hypothesis.

## Repository context to read FIRST

1. `docs/checkpoints/mechanism/plan.md` — **PRIMARY source of truth**, especially:
   - §1.1-§1.4 Theory framework + Zoom 1-4 hierarchy
   - §5.1 Method 4.2 (cls baseline numbers)
   - §5.2 Stage 2/3 cells dashboard (10 + 6 cells)
   - §5.3 Method 4.4 v2 full sweep
   - §5.4 Image-axis peak-layer dichotomy
   - §5.5 H1 test (cls baseline)
   - §7.3.1 Reddit cross-site results (P5a + P5b, NEW 2026-05-12)
   - §5.2 Stage 3 cross-site table (NEW 2026-05-12)
2. `docs/checkpoints/paper_drafts/section5_mechanism_reddit.md` — CURRENT prose (77 lines, behavioral content to relocate; only §5.1 Method block on activation patching protocol can be retained and merged into new prose)
3. `docs/checkpoints/paper_drafts/section4_empirical_findings.md` — receiving location for relocated behavioral axis-1/2/3 content
4. `docs/checkpoints/paper_drafts/section2_background.md` — for Q5 / Wu et al. Tool Calling Linear Circuit lit anchor consistency (§2 should already cite, §5 should reference)
5. `docs/checkpoints/mechanism/results/`:
   - `format_variation_h1_test.md` (cls baseline H1)
   - `format_variation_h1_test_reddit.md` (NEW, cross-site H1)
   - `format_variation_h1_test_cls_reverse.md` (NEW, selection-bias defense)
   - `layer_axis_emergence.md` (image-axis peak-layer dichotomy)
   - `h1_per_task_fragility.md` (per-task variability)
6. `docs/checkpoints/stage4_method42_results.md` (cls Method 4.2) + `_reddit.md` (NEW cross-site Mirage)
7. `results/mechanistic/stage3_*/pilot_summary.md` (6 Stage 3 cells, Δoverlap numbers)
8. `docs/checkpoints/paper_drafts/paper.bib` — bibkeys: `wu2026tool` or equivalent (Tool Calling Linear Circuit Wu, Wang, Cho et al.), `khorasani2026hdmi` (HDMI completeness × selectivity), `kaduri2024crossmodal`, `fayyaz2026steermoe`, `sclar2024prompt` (5 lit anchors per plan §2)

## Output structure

Write **one new file**: `docs/checkpoints/paper_drafts/section5_mechanism.md` (replacing the misnamed `section5_mechanism_reddit.md` — delete the latter after relocating its useful content).

Target length: 80-130 lines of paragraphs (NOT bullet lists). Plain markdown, paper.bib citation style `\citep{key}`. Math allowed in LaTeX inline `$...$`.

Section structure:

### 5.1 Overview and theoretical framing (~15 lines)
- Lead with the question: why does P-SoM achieve DOM-cost + SoM-signal? Phantom routing space mid-layer mechanism is the explanation
- Position within Zoom 1-4 hierarchy (plan §1.1) and Q5 bidirectional mid-layer fusion lit anchor (plan §1.2, Wu et al. Tool Calling Linear Circuit + Kaduri cross-modal flow)
- Pre-commit the four mechanism claims that §5.2-§5.6 will defend:
  1. Mode means are geometrically separable in the residual stream (PCA cosine gap, AUROC 1.0)
  2. P-SoM is mid-layer text-axis sibling of DOM, image-axis sibling of SoM
  3. SoM→{no-image-arm} activation patching displaces target prediction at mid-layer L11-L17 with consistent magnitude across cls + reddit
  4. The trigger for shortcut activation is flat-list format, not specific token pattern — AXTree hierarchy is the unique defeating format
- Cross-site evidence stack: four vertical defenses (per-task fragility, selection-bias, cross-site H1, cross-site Mirage signature)

### 5.2 Method 4.2 — PCA cosine gap (~15 lines)
- Method (cosine gap, AUROC via mean-diff projection, PCA top-10 variance)
- 5/5 robustness tests (label perm, per-task, per-step, silhouette, bootstrap CI), each one sentence
- Cls baseline table (plan §5.1, 4 pairs @ L17): P-SoM↔P-text 0.0028, DOM↔P-prompt 0.0013, P-SoM↔SoM 0.0413, DOM↔Vision 0.0547
- Reddit replication table (plan §7.3.1): P-SoM↔DOM 0.0098 (text-axis sibling), P-SoM↔SoM 0.0423 (image-axis split), DOM↔Vision peak L04 AUROC 1.0
- Geometric interpretation: P-SoM is text-axis sibling at L17, not image-axis sibling — Mirage signature

### 5.3 Method 4.4 — mean-diff activation steering (~15 lines)
- Method (mean-diff direction, per-input addition with α scaling, HDMI completeness × selectivity → harmonic mean reliability)
- Full 45-cell sweep (layer × α grid: layers [11,17,23,29,33,34], α [1,2,5,10,20])
- L17 α=5 smoke 0.44 → full 0.16 (smoke variance artifact, 笔记 §126/§127)
- L33 α=10 H-mean 0.33 sweet spot (completeness 38%, selectivity 29%, late-layer over-steers JSON)
- Mid-layer vs late-layer dissociation: completeness vs selectivity trade-off (plan §5.3)
- **Acknowledge** as evidence ceiling, motivates §8 future LA-HDMI / SAE without claiming superiority

### 5.4 Stage 2/3 — Activation patching for causal mid-layer mechanism (~20 lines)
- Retain `section5_mechanism_reddit.md` §5.1 Method paragraph (clean/corrupt/source/target patching protocol, layer hooks, continuation scoring, Holm-Bonferroni grid)
- Update outdated claims: cells F/G NOT pending (done), B1 reddit phantom NOT pending (done)
- **Stage 2** P-SoM↔SoM 10-cell table (plan §5.2 Stage 2 sub-table)
- **Stage 3** 6-cell DOM-axis additivity table (plan §5.2 Stage 3 sub-table, cross-site cls + reddit):
  ```
                  SoM→DOM      SoM→P-text    SoM→P-prompt    range
  cls site:       -0.33 @L17   -0.25 @L17    -0.22 @L17     [-0.33, -0.22]
  reddit site:    -0.33 @L11   -0.24 @L11-17 -0.19 @L11     [-0.33, -0.19]
  ```
- Interpretation: image-feature axis is shared substrate across DOM / P-text / P-prompt arms; cross-site magnitude essentially identical; reddit fusion locus slightly earlier (L11 vs L17) but mechanism universal
- Negative controls: Cell E -0.03 (cls random injection), Cell Er ~0 (reddit random) — mechanism is content-specific, not noise

### 5.5 Image-axis peak-layer dichotomy and H1 format variation (~20 lines)
- §5.4 layer_axis_emergence: 8 mode-pair peak-layer table. AXTree-no-image side → L04 (4/4); marks-no-image side → L17-L36 (4/4); zero overlap. Cleanest single-pair signature of Mirage Effect mechanism
- H1 refined hypothesis: pretraining co-occurrence shortcut — "input contains mark-like indexed region list → activates visual-grounding pathway"
- 10-mode format variation grid: 6 marks-like + 2 controls (hash_id, plain_sentence) + 2 baselines (dom, som)
- Cls strong-tier result (plan §5.5): marks-like all L36 (monotonic boundary artifact), hash_id_control L36 (failed control), plain_sentence L17, dom L04 ✓
- Cls reverse-tier result (NEW P4, `format_variation_h1_test_cls_reverse.md`): identical pattern to strong-tier — H1 NOT a tier selection artifact, universal Qwen3-VL-4B training prior
- Reddit result (NEW P5a, `format_variation_h1_test_reddit.md`): marks-like 4/6 peak true L17 mid-layer (cleaner than cls boundary artifact), hash_id_control L04 (proper control), dom L04 ✓
- Cross-site reframing: cls L36 is monotonic-boundary artifact; reddit data reveals the **true** L11-L17 mid-layer fusion locus. This supports Q5 bidirectional mid-layer fusion hypothesis (Wu et al. tool-calling: "mid and late-layer attention heads")

### 5.6 Convergent four-vertical-defense evidence stack (~10 lines)
- Per-task fragility (P1 prior DGX work): single-task mean ≠ aggregate, mechanism is not population-level artifact
- Selection-bias defended (P4 NEW): cls reverse-tier replicates strong-tier H1 pattern
- Cross-site H1 (P5a NEW): reddit replicates indexed-list shortcut trigger directionally
- Cross-site Mirage (P5b NEW): reddit replicates P-SoM ≈ DOM at mid-layer and P-SoM ≠ SoM image-axis split
- Optional 5th defense (deferred): P2 cross-family Phi-3.5-Vision + P3 capacity Qwen2-VL-7B — defers to single-thread HF download recovery
- Position the four defenses as defending cross-site generalization, not B0/B1 capability scaling (defer §7)

### 5.7 Discussion and limits (~10 lines)
- Method 4.4 H-mean 0.33 ceiling: cosine-gap mid-layer (L17) vs steering ceiling late-layer (L33) — selectivity vs completeness trade-off; ceiling motivates §8 future work (LA-HDMI gradient steering, SAE feature steering)
- Cls boundary artifact vs reddit clean peak: Mirage Effect mechanism is universal but per-site curve shape varies; report effect-direction claim, not strict layer claim
- Connect to Q5 Zoom 4 lit anchor (Wu et al. Tool Calling, Kaduri cross-modal flow, Fayyaz SteerMoE): paper §5 is the multimodal web-agent application of the linear-readable + steerable + mid-late-layer-circuit framework, not a replication
- AXTree hierarchy as unique defeating format (paper §6 supplement open question): why hierarchy beats flat — likely cross-modal attention to indentation tokens, attribution pending

## Constraints

1. **NO new findings, NO speculation** beyond what plan.md §5 + today's results files claim
2. **All numerical claims trace to a source file** — paranoid about citing where each number comes from
3. **Citation style** `\citep{key}` for bibkeys; if a bibkey is needed but not in paper.bib, prefix with `NEEDS_BIB: ` so manual review catches it
4. **Layer indexing**: paper draft says "L0-L35 = 36 layers"; plan says "37 layers". Use the **plan** convention (37 layers including embedding output L0) and note the convention explicitly: "We index layers L0-L36, where L0 is the embedding-block output and L1-L36 are the 36 transformer decoder block outputs"
5. **Avoid "first to X" novelty claims** — see `intro_controlled_characterization_rewrite.md` precedent; phantom routing space is **controlled scientific characterization**, mechanism work is **multimodal application of existing linear-readable + steerable framework**
6. **Routing implementation deferral**: explicit one-sentence note that paper-1 §5 is mechanism-only; routing implementation moved to paper-2 (per `paper_planning.md` §1)
7. **Behavioral content relocation**: at the end of the prose, append a brief NOTE FOR HUMAN section listing which subsections of current `section5_mechanism_reddit.md` should be moved to `section4_empirical_findings.md` and which can be deleted (it's not your job to do the move, just flag it)

## Output

1. New file: `docs/checkpoints/paper_drafts/section5_mechanism.md` (80-130 lines, paragraphs, plain markdown)
2. End-of-file `## NOTE FOR HUMAN` block listing:
   - Bibkeys needed with `NEEDS_BIB:` prefix
   - Behavioral content subsections of current `section5_mechanism_reddit.md` to relocate to §4 (with line ranges)
   - Plan.md §5 numbers that should be double-checked before submission (e.g., "Stage 3 H-t-red Δ@L17 reported as -0.24 in plan §5.2 Stage 3 table but cellht_red pilot_summary.md only gives Best-L overlap→src=L01 (0.194); the Δoverlap→tgt number needs results.json verification")

Hand back when ready.
