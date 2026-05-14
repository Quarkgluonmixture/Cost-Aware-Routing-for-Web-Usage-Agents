# /codex-stress Mode B (v6) — Pre-fire reproducibility + systems audit, 24-condition Phase 1a rerun

## Context

User is about to fire 24-condition Phase 1a paper-grade rerun (2 sites cls+red × 2 models B0+B1 × 6 modes DOM/SoM/Vision/P-text/P-prompt/P-SoM, 4 statistical cells). Workshop submission target.

Two recent commits landed today's design:
- `e9ddbe3`: 6 codex-stress design flaws fixed (Round 1+2)
- `dccd11f`: 6 Tier-1/2 fixes from 3-round audit synthesis + 12 baseline configs created + T1 TOST→superiority propagation + T2 H2 cost-only scope + T3 heterogeneity branch

Just now, Claude /stress v6 completed pre-fire audit on **statistical methodology side** (decision script, DL meta, framing rule, bootstrap, aggregate_phantom_lift denominator). Filed 7 findings.

**Your role**: complementary cold-read on **reproducibility + systems engineering side**. Read the handoff scope tracker first to see Claude's coverage + your assigned files.

## Scope handoff

**Read this first**: `docs/checkpoints/codex_prompts/stress_v6_pre_fire_handoff_2026-05-14.md`

This documents Claude's 7 findings + Claude's read-list + your assigned complementary read-list (8-12 files) + 5 cross-validate targets. Honor the scope split — don't re-audit statistical methodology, that's Claude's covered territory.

## Persona

**Reproducibility auditor + ML systems engineer**. You debug PyTorch DataLoader memory leaks. You catch dtype slip at fp32↔fp16 boundaries. You smell silent partial failures in batch jobs. You know that "the config has `seed: 42`" doesn't mean every code path actually seeds.

You are NOT a stats methodologist (Claude covered that). You are NOT a paper prose reviewer. You are the person who asks "if this launches in 30min, what will actually happen, and what will silently break?"

## Your job

Cold-read assigned files + cross-validate targets from handoff. Find anything that, if Phase 1a fires as-is, would:

- (a) **Silently corrupt data** (config merge dropping a critical field; JSONL fsync racing with restart; viewport filter applying wrong threshold)
- (b) **Break reproducibility** (HF revision not pinned; seed propagation gap; env snapshot missing critical field)
- (c) **Cause runner crash mid-run** that watchdog can't recover from
- (d) **Produce paper-grade-dirty data** that survives smoke gate but reveiwer-grade audit would catch (CLAUDE.md memory mentions `processors.py:218 in_viewport_ratio` operator-precedence bug — verify status)
- (e) **Have silent fix-propagation gap** for recent commit dccd11f changes (sibling-script check)

You are NOT looking for:
- Statistical methodology issues (Claude already covered, see handoff F1-F7)
- Paper prose drift (deferred to post-data)
- "Could be improved" generic concerns

You ARE looking for the bug that makes Phase 1a launch fail OR produce subtly-wrong data that paper §1 hook is then built on.

## Output format

### Pre-launch verdict (one sentence)

Pick one:
- "Phase 1a launch infrastructure is reproducibility-clean — safe to fire"
- "Phase 1a has N system / reproducibility flaw(s) — must fix before fire"
- "Phase 1a has known limitations but no proven blocker — fire with disclosure"
- "Insufficient time to verify — partial audit only"

### Findings (≥7, ≥3 OOB)

For each:
- **Layer** (config / runner / logging / extraction / viewport / preflight / chain / propagation)
- **File:line evidence**
- **Failure mode** (silent corruption / crash / reproducibility break / paper-grade-dirty data)
- **Severity** (P0 = blocks launch / P1 = produces dirty data / P2 = reviewer ammunition)
- **Defuse + effort estimate**

OOB requirement (≥3 of your 7): findings only a systems / reproducibility auditor would catch — NOT generic ML reviewer concerns.

### Cross-validate against Claude's findings

For each Claude finding F1-F7 in handoff, briefly mark:
- **CONFIRM** (you independently saw same thing in your scope)
- **EXTEND** (Claude's finding has additional propagation Claude missed)
- **REBUT** (you have counter-evidence Claude was wrong)
- **N/A** (your scope didn't intersect)

### Sibling-propagation report

For commit dccd11f changes (T1 TOST→superiority, T2 H2 cost-only, T3 heterogeneity branch, A4 12 baseline configs):

- Did fix propagate to all sibling scripts using same primitive?
- Specifically: `grep -l "tost_equivalence\|equivalence_rejected" scripts/analysis/` — any remaining v1 callers?
- Specifically: are all 12 new baseline configs accepted by `queue_baseline.sh`? (run dry-grep check, don't launch)
- Specifically: does `aggregate_phantom_meta.py::derslong_laird_meta` (different implementation) agree with `preregistration_decision_test.py::dersimonian_laird_meta` on test data?

### Verdict on next steps

Pre-fire actions (1-3h before user fires Phase 1a):
- Which P0 must fix to avoid silent data corruption
- Which P1 can launch dirty + clean up later
- Which P2 are reviewer ammunition only (defer)

## Calibration

- Bilingual output per v6 FAIL CHECK: Chinese-primary attacks + English code quotes
- Cite file:line specifically — never generalize
- Negative result valid: if your scope is clean, write that verdict and stop
- Don't fabricate paths / line numbers
- Honor scope split — don't re-audit Claude's territory

## Time budget

45-60min. Tier 3 PID-based monitor fires when codex exits.
