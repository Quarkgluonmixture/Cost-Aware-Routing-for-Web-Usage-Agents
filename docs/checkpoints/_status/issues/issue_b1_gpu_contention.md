---
type: issue
category: blocker
status: mitigated
priority: low
action: A100 dedicated allocated 5/6 (Steve approved), pending SSH verify; B1 GPU contention deprecated for paper-grade rerun
updated: 2026-05-06
resolved_pending: A100 SSH verify completes
---

# B1 phantom runner GPU contention (DGX shared) — MITIGATED by A100

## 5/6 update: A100 dedicated unblocks

UCL Condense A100 dedicated allocated 2026-05-06 evening (Steve approved). Once SSH verifies, paper-grade 16-cell rerun + mechanistic Stage 2B scale-up + Llama-4 cross-arch all migrate to A100. DGX shared seonglae sweep contention no longer paper-timeline blocker.

笔记 §112 details A100 unblock + compute path priority shift (A100 = Tier 0).

## Historical context (pre-5/6, kept for audit trail)

Original blocker: Current target was **B1 phantom_prompt classifieds** (PID 3826576, 110/234, ~5 ep/h, ETA ~25h). seonglae 并行任务持续抢占 GPU.

History of B1 cls phantom chain (DGX shared, despite contention):
- B1 phantom_som cls: ✅ done 2026-05-02 16:41
- B1 phantom_text cls: ✅ done 2026-05-02 10:46
- B1 phantom_prompt cls: ✅ done 2026-05-04 05:50 (sr_raw=10.68)

All three are **pre-Phase-A** data → 废 by 16-cell rerun (`issue_14cell_phantom_rerun.md`) on A100.

## Long-term resolution

✅ A100 dedicated (笔记 §112) eliminates GPU contention for paper-grade rerun. DGX shared remains as fallback for Phase A bug fix smoke test only.

## Refs

- `docs/checkpoints/_status/issues/issue_14cell_phantom_rerun.md` (16-cell rerun on A100)
- `docs/checkpoints/advisor_sync_5_5_outcomes.md §A.8` (compute path Tier 0 A100)
- `docs/checkpoints/实验笔记.md §112` (A100 allocation chronicle)
