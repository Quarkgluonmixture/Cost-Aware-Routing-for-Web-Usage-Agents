# Pre-run hostile stress: 16-cell paper-grade rerun design (Phase A)

## Context

User is about to launch the **16-cell paper-grade rerun** that produces the supporting
data for paper-1 §1 hero claim + §4 hero tables + drop-one CI + per-mode forest plot.
This is the **pre-launch audit window** — no cells started yet. Once launched, the
data lock means no further preregistration changes are permitted.

User has already received independent audits on:
- Stage 4 mechanism pipeline (4 paper-grade bugs found and fixed last week)
- Phase 1 baseline code/data pipeline (`codex_full_prerun_audit_2026-05-13.md`)
- viz pipeline (5 viz pipeline bugs, fixed in 78d1efe)
- v2 retraction audit (3 codex paper-grade fixes)

This audit is **complementary**: it targets the **design / methodology / preregistration**
layer specifically, not the code/data pipeline. Code-side audits cover whether the
pipeline computes things correctly; this audit asks whether what's **being computed**
and **how it gates paper claims** is sound.

You are a hostile reviewer who has NOT seen Claude's analysis or the user's reasoning.
Read the design cold and attack.

## Your job

Find anything in the 16-cell rerun **design** that, if launched as-is, would:
- (a) Change a paper-grade number by ≥ 0.5pp, OR
- (b) Invalidate a paper-grade qualitative claim, OR
- (c) Give a top-tier reviewer (NeurIPS / ICML / ACL main / ICLR) ammunition strong
  enough to reject the paper hook framing, OR
- (d) Trigger a post-hoc framing drift accusation (garden-of-forking-paths / HARKing
  / undisclosed multiple testing / preregistration-vs-actual mismatch).

You are looking for things the user has **not yet seen** in prior audits. Existing
audits already covered the code/data layer; do not redo that. Focus on design.

## Input files (read cold, in order)

### Layer 1 — Preregistration (what's claimed)

- `docs/checkpoints/pre_run/preregistration.md` — **primary target**. H1-H8 hypotheses,
  family declaration, locked analysis choices, framing rule R1-R5, witness mechanism,
  decision log. ~330 lines. Read fully.
- `docs/checkpoints/pre_run/` — other pre-run docs (topvenue_constraints, pre_rerun_audit,
  dataset_card, model_card, locked_versions, evaluator_change_protocol, reeval_audit_protocol,
  negative_results_registry, osf_lock_manifest, ethics_license_coi_statements,
  release_redaction_checklist). Skim. Flag inconsistencies with preregistration.md.

### Layer 2 — Paper hook + theory framing (what the data is supposed to support)

- `docs/checkpoints/paper_planning.md` §1 (paper hook), §2 (theory framework, Evidence
  ⫨ Explanation, Zoom 1-4), §3 (cross-X patterns), §20 (doc workflow). The strategy notebook.
- `docs/checkpoints/paper_drafts/section1_intro.md` — final §1 prose. Does the prose
  claim match what H1-H8 actually gates?
- `docs/checkpoints/paper_drafts/section3_definition.md` — experimental design prose.
  Mode operational definitions vs preregistration §4 lock — agree?
- `docs/checkpoints/paper_drafts/section4_empirical_findings.md` — what numbers paper §4
  reports. Are these covered by the H1-H8 gating, or are some §4 claims exploratory?
- `docs/checkpoints/paper_drafts/section4_limitations_disclosure.md` — disclosure prose.

### Layer 3 — Advisor decisions / framing register

- `docs/checkpoints/ADVISOR_SYNC.md` — 5 framing decisions, lock state, what's
  decided vs pending. Cross-check vs preregistration frontmatter `status: draft` and
  the 8 commit decisions in §6.
- `docs/checkpoints/advisor_sync_5_5_followup.md` — Q1-Q11 followup, K_h1=12 /
  K_h3=11 / TOST δ=1.0pp / paper split 3v4 lock asks. Email pending.

### Layer 4 — Launch orchestrator + protocol

- `scripts/queues/queue_16cell_paper_grade.sh` — the only authorized launch path.
  3-chain orchestrator (cls / red / shop). Pre-launch gates. Reset protocol. ETA model.
- `docs/reference/launch_checklist.md` — 16-cell paper-grade rerun protocol checklist.
- `.claude/CLAUDE.md` — "实验启动 hard rules" + "三阶段实验设计" + "关键变量".
- `scripts/preflight_v2.sh` — preflight check what.

### Layer 5 — Cell scope ground truth

- `docs/checkpoints/_status/cells/cell_*.md` — 26 cells' frontmatter. Which match
  the 16 in preregistration §4 N_cells row? Which are archived pre-Phase-A vs
  Phase A post-fix? Any inclusion ambiguity?
- `docs/checkpoints/next_steps.md` §1 + §1a — 16-cell rerun launch sequence, pre-launch
  gates 1-6, ETA chain.

### Layer 6 — Related historical audit findings (so you don't redo)

- `docs/checkpoints/codex_outputs/codex_full_prerun_audit_2026-05-13.md` — already-completed
  Phase 1 code/data audit. **Read summary verdict + confirmed bugs only**. Use this
  to avoid duplication; flag design issues complementary to those.
- `docs/checkpoints/codex_outputs/v2_retraction_audit_2026-05-13.md` — v2 retraction.
- `docs/reference/master_bug_catalog.md` — known bug list. Has any "known design quirk"
  not yet escalated to preregistration?
- `docs/reference/EVIDENCE_LAYER_AUDIT.md` — figure registry. Each paper figure should
  trace to a gated H1-H8 sub-claim OR be flagged exploratory.

## Output format

### One-sentence pre-launch verdict

Pick one:
- "Design is paper-grade — safe to launch as-is"
- "Design has N paper-grade flaw(s) — must fix before launch"
- "Design has methodological concerns but no proven flaw — disclosure-only fix"
- "Insufficient time to verify — partial audit only"

### Confirmed design flaws

For each:
- **Layer**: preregistration / paper-prose / orchestrator / cell-scope / framing-rule / witness
- **Evidence**: quote file:line + the contradicting / drifting / underspecified text
- **Impact**: which paper claim or number is at risk
- **Severity**: HIGH (kills hook framing or invalidates ≥1 H-gate) / MED (forces
  disclosure language or appendix sensitivity check) / LOW (reviewer would ask question
  but answerable in rebuttal)
- **Defuse effort**: minutes / hours / requires advisor sync

### Probable concerns (suspicious, couldn't fully verify cold)

Same format, marked "needs further check by Claude or user".

### Methodology drift (preregistration ↔ actual)

If preregistration says X gates paper claim Y, but actual queue / code / paper prose
implements Z: report all three + which is right + what defuse looks like.

Special attention: any **operational definition gap** (e.g., "mode" defined two
different ways across docs, "N_cells" defined two different ways, "K-of-N" defined
two different ways, FP filter scope shifted).

### Reviewer ammunition (not bugs, but top-tier reviewer would ask)

What questions would a hostile NeurIPS / ICML / ACL reviewer ask after reading paper §1 + §3 + §4
+ the supplement preregistration appendix? List 5-10 specific reviewer questions
with **the answer the user currently has** (or "no answer prepared" if gap).

Calibrate to reviewer-3-skeptical-3/10 (peer-lab top-tier, not friendly).

### Cross-doc inconsistency (highest priority class)

If preregistration §4 says X, paper §3 prose says X', advisor_sync_5_5_followup says X'',
and queue_16cell says X''' — report all 4 + which is canonical + what action defuses.

### What you read and what you didn't

Brief enumeration with time-per-section. If a section was time-constrained out, say so.
If a file path didn't exist, note it (do not fabricate).

### Verdict on next steps

If design holds: tell user they can launch 16-cell rerun as soon as advisor email lands.
If design has flaws: prioritized list. Especially: **the one thing to fix tonight**
if user has 1-3h energy before advisor email arrives.

Specifically: should the user **delay launch** for any of these design issues, or
can they all be defused in post-data disclosure language?

## Calibration

- This is paper-grade design audit, not code review or prose polish. Don't flag style.
- Don't propose code fixes. Identify the suspect, the impact, the defuse cost.
- Calibrate severity by paper-grade impact, not aesthetic ugliness.
- Negative result is valid: if you find nothing ≥0.5pp suspect after 60 min reading,
  write the trust verdict and stop.
- Don't fabricate. If a file path doesn't exist, note it. If you can't trace a
  chain, say so.
- Be **specific**. "K-of-N rule needs more thought" is useless. "preregistration.md
  line 211 says K_h1=12 is 'secondary transparency check' but paper_drafts/section1_intro.md
  line 47 still uses K_h1 phrasing as if it gates H1 — this is a confirmed drift,
  severity MED, defuse = paper prose edit ~10 min" is useful.

## What this is NOT

- Not Claude's /stress (orthogonal context — you have NOT seen Claude's review of this design)
- Not a methodology blessing — your role is suspicion-mode, not approval-mode
- Not bound by typical Web agent / interpretability subfield pitfalls — set your own
  attack vectors based on what the docs show
- Not a 5-minute scan. Take the time you need within 60 min budget.

## Time budget

Up to 60 min. Codex foreground PID-based monitor will fire when done.
