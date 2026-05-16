---
type: issue
category: paper-grade-artifact-build
status: open
priority: high
action: Build 3 missing paper-grade canonical artifacts surfaced by /stress A1.4b-i. These are NEW producer scripts, not code bug fixes — total scope estimate 1.5-3 days. Defer to A1.4b-i follow-up scope.
created: 2026-05-16
updated: 2026-05-16
paper_section: "§1 hero + §3.6 evidence layer + §appendix reproducibility"
audit_source: /stress A1.4b-i (codex B2 + B8 + gemini C3)
file_paths:
  - results/phantom_paper/phase1_prereg_gate.csv  # MISSING — to build
  - results/phantom_paper/claim_manifest.json     # MISSING — to build
  - results/phantom_paper/hero_metrics.json       # MISSING — to build
  - scripts/analysis/aggregate_phantom_lift.py    # legacy 3→5 estimand (B2)
  - scripts/analysis/aggregate_phantom_meta.py    # appendix RE meta (B-182 done)
b_number: B-184 / B-185 / B-186 (reserved)
---

# Phase 1 canonical paper-grade artifacts — 3 missing

`/stress A1.4b-i` cross-AI audit (commits 60e6ce5 / 824e55a / de85d5a / 3f83a52)
identified that the **paper §1 hero claim source-of-truth** is currently
fragmented across legacy artifacts that compute the wrong estimand or lack
provenance. Three canonical files are missing from `make analysis` output;
this issue tracks the build scope.

## B-184 — `results/phantom_paper/phase1_prereg_gate.{csv,json}` (codex B2, P0 OOB)

**Problem**: `scripts/analysis/aggregate_phantom_lift.py:811-816` headline
declares PRIMARY as "3→5-mode lift over 5 arms (DOM/SoM/Vision + P-text/P-SoM)",
but `preregistration.md:68-83` locks PRIMARY as "FE inverse-variance pooled
P-SoM drop-one over six planned (site, model) cells, one-sided superiority
H0: θ_FE ≤ +1.0pp". These are DIFFERENT estimands — they can diverge when
P-text covers the same tasks as P-SoM or when P-prompt is present.

**Build**: per-cell compute `oracle(all_6) - oracle(all_6_without_PSoM)`,
bootstrap SE, FE pool over the six planned cells (seed=42 disclosed via B-176),
one-sided superiority test `z = (θ_FE - 1.0) / SE_FE`. Emit:
- `phase1_prereg_gate.csv` — per-cell drop-one + pooled FE row
- `phase1_prereg_gate.json` — pooled metadata + gate decision
- `phase1_prereg_gate.md` — markdown table for paper §1 prose to cite

**Wire into Makefile**: new `_aggregate` step before `phantom-lift`. Demote
`phantom_lift.csv` 3→5 + `meta_phantom_lift.csv` DerSimonian-Laird to
exploratory/appendix labels (already partial — B-182 added family_scope
columns to phantom_meta; phantom_lift still needs the demote prose).

**Effort estimate** (codex): 0.5-1 day.

## B-185 — `results/phantom_paper/claim_manifest.json` (codex B8, P1)

**Problem**: `analyze_run` writes per-run `statistical_tests.json` (raw
McNemar/Wilcoxon + Holm post B-178); `phantom-lift` writes
`phantom_lift.csv` (legacy 3→5 estimand); `phantom-meta` writes
`meta_phantom_lift.csv` (RE meta). These are 3 parallel branches producing
different statistical artifacts; paper-claim → producer mapping is
ambiguous. Reviewer pasting a number into the paper can't tell which file
backs it.

**Build**: `claim_manifest.json` maps each paper table/figure/prose claim to:
- `producer` — exact script + commit SHA
- `input_file_sha256` — input artifact hashes
- `family` — H1/H3-axis1/H3-axis2/H4/appendix
- `estimand` — drop-one over 6 cells / 3→5 lift / etc.
- `gating_status` — gate / sensitivity / exploratory

**Effort estimate** (codex): 0.5-1 day. Forces cleanup of stale figure/prose
dependencies.

## B-186 — `results/phantom_paper/hero_metrics.json` (gemini C3, P1)

**Problem**: Paper §1 hook claims "P-SoM 4-fold drop-in property": (a) cost,
(b) latency, (c) signal AUROC, (d) drop-one oracle. All 4 metrics ARE
computed in current pipeline:
- (a) `fig3d_cost_sr_frontier.py` ✓
- (b) `fig3c_latency_per_step.py` ✓
- (c) `aggregate_routing_auroc.py` → `auroc_cross_condition.csv` ✓
- (d) `aggregate_phantom_lift.py` → `phantom_lift.csv` ✓

But they are scattered across 4 different output CSVs. No single artifact
collects all 4 P-SoM numbers for paper table/prose to cite. Reviewer
verification requires reading 4 separate files.

**Build**: `hero_metrics.json` — single JSON with P-SoM (a)(b)(c)(d) numbers
per (site, model) cell + pooled. Producer reads `phantom_lift.csv` +
`auroc_cross_condition.csv` + the runner per-condition summaries.

**Effort estimate** (gemini): 0.5 day (mostly aggregation glue).

## Status

- [x] Issue file created (this file)
- [x] **B-184: build phase1_prereg_gate.{csv,json,md} producer** — landed commit (see chronicle §150.8). `scripts/analysis/aggregate_phase1_prereg_gate.py` + 17 unit tests. Live run currently emits `gate_status=INSUFFICIENT_DATA` (no cell has all 6 modes yet; Phase 1a rerun in flight). Producer gracefully degrades.
- [x] **B-184: wire into Makefile `_aggregate` chain** — runs BEFORE `phantom-lift` so canonical gate lands first; `phantom-lift` retained as appendix-exploratory.
- [ ] B-184: paper §1 hero prose update to cite new gate artifact (defer until Phase 1a data lands and gate produces non-INSUFFICIENT_DATA status)
- [ ] B-185: build claim_manifest.json producer
- [ ] B-185: emit input_file_sha256 for all aggregator inputs
- [ ] B-186: build hero_metrics.json producer

## Dependencies

- B-185 depends on B-184 (manifest needs canonical gate to reference)
- B-186 independent of B-184/B-185 (4-metric collector, no gate)
- All 3 deferred until after A1.4b-ii data plane audit (logger_v2 / io_utils /
  analysis / types / metrics / schema_migrations) — A1.4b-ii may surface
  additional schema fields that affect B-184/B-185 design.

## Cross-link

- Audit chronicle: 实验笔记 §150 (A1.4b-i full)
- Master bug catalog: §150 audit table (B-184/B-185/B-186 deferred entries)
- Cross-AI prompts: `docs/checkpoints/codex_prompts/{codex,gemini}_stress_a1_4b_i_083831.md`
- Cross-AI outputs: `docs/checkpoints/codex_outputs/{codex,gemini}_stress_a1_4b_i_083831.md`
