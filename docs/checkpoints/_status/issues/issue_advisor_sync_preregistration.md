---
type: issue
category: blocker
status: active
priority: high
action: schedule advisor sync (30-45 min) + lock 5 commits in preregistration.md
created: 2026-05-03
---

# Advisor sync + pre-registration lock

Pre-registration framework reframed 2026-05-03 (Hero + Structural + Framing-rule R1-R5) — `docs/checkpoints/preregistration.md` is `status: draft` pending advisor sync to lock.

## Sync ask (5 commit decisions)

| # | Decision | 我倾向 |
|---|---|---|
| 1 | H1 K_h1 cell-pass threshold (P-SoM Holm-sig) | **0.75** |
| 2 | H3 K_h3 cell-pass threshold (axis non-overlap CI > 0) | **0.67** |
| 3 | TOST equivalence margin δ | **1.0pp** |
| 4 | Cell inclusion: Phase A only main + archived Appendix D | confirm |
| 5 | Witness: git SHA + advisor email + OSF DOI | confirm |

Plus framework buy-in question: 5-rule data-conditional framing rule (R1-R5) 是否被视为 disguised garden-of-forking-paths?

Plus旧 ask: VWA bug 单独成文? RunPod $200? Early-stop A/B/C? SteerMoE scope?

## Blocks

- T0e: `preregistration.md` flip status:draft → status:locked
- T0f: `fig_hypothesis_matrix.py` scaffold (post-rerun fill)
- T1 stats infra (F1 / F2 interaction tests etc. — useful sync prep but blocked by framework buy-in)
- 14-cell phantom rerun launch decision (cell list + Phase A only inclusion final)

## Unblocked by

Advisor sync session (30-45 min) using `docs/checkpoints/ADVISOR_SYNC.md` §1-§6 as prep notes.

## Refs

- `docs/checkpoints/preregistration.md` (draft)
- `docs/reference/EVIDENCE_LAYER_AUDIT.md` §2 (template + meta-rationale)
- `docs/checkpoints/ADVISOR_SYNC.md` §1.4 (5 lock decisions table)
- `docs/checkpoints/paper_planning.md` §1 (data-conditional hook reframe note) + §19 (decision log entries 2026-05-03)
