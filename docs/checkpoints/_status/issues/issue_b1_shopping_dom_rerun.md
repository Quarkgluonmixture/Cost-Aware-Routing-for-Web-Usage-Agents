---
type: issue
category: blocker
status: active
priority: medium
action: queue_baseline.sh B1 dom shopping (post-A100 SSH verify)
updated: 2026-05-06
---

# B1 shopping DOM 466 ep — 待 A100 clean re-run

archived `_archive/B1_3mode_shopping_20260413_pre_magento_bug` (含 dom 465/466). Need clean rerun on dedicated GPU.

## Compute path (5/6 update)

⭐ **UCL Condense A100 dedicated** (allocated 5/6, pending SSH verify) — primary path. ~24h on A100 dedicated, no contention.

Previously planned RunPod 4090 ($0.6/h × 24h = ~$15) — deprecated by A100 self-allocation.

## Refs

- `docs/checkpoints/advisor_sync_5_5_outcomes.md §A.8` (compute path Tier 0 A100)
- `docs/checkpoints/实验笔记.md §112` (A100 allocation chronicle)
