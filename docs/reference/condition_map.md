# Condition 映射表

diag / write-analysis / report 三个 SKILL 共用的 condition → 路径映射。新 run 出现时只改此文件。

## 路径模式

```
results/{benchmark}/phase1/{run_id}/{condition_id}/episodes/{site}_task_{task_id}_steps_v2.jsonl
```

- `condition_id` 当前 paper-1 universe = 7 modes:
    - 3 baseline: `phase1_dom_router_0` / `phase1_som_router_0` / `phase1_vision_router_0`
    - 4 phantom (paper §3 hero): `phase1_phantom_som_router_0` / `phase1_phantom_text_router_0` / `phase1_phantom_dom_router_0` (legacy alias of `phantom_text`, pre-A1.7 archive dirs only) / `phase1_phantom_prompt_router_0`
- `site` 前缀：classifieds → `classifieds_task_`，reddit → `reddit_task_`，shopping → `shopping_task_`

> B-458 (/stress A1.4 P1-6-C gemini OOB, 2026-05-17): 4 phantom condition_ids 加入. Pre-fix this file 只列 3 baseline conditions, 所有 phantom-mode runs (paper §3 hero P-SoM/P-text/P-prompt) 都被 `diag` / `write-analysis` / `report` SKILL 当作 unknown condition 静默漏掉 — automated pipeline status aggregation 看不到 paper §3 数据.

## VWA (VisualWebArena)

benchmark = `visualwebarena`

| baseline | site | run_id | tasks |
|----------|------|--------|-------|
| B0 | classifieds | `B0_3mode_classifieds_20260413` | 234 |
| B0 | reddit | `B0_3mode_reddit_20260422` | 210 |
| B0 | shopping | `B0_3mode_shopping_20260421` | 466 |
| B1 | classifieds | `B1_3mode_classifieds_20260413` | 234 |
| B1 | reddit | `B1_3mode_reddit_20260413` | 210 |
| B1 | shopping | `B1_3mode_shopping_20260413` | 466 |

## WA (WebArena)

benchmark = `webarena`

| baseline | site | run_id | tasks |
|----------|------|--------|-------|
| B0 | shopping | `B0_wa_3mode_shopping_20260417` | 192 |
| B0 | shopping_admin | `B0_wa_3mode_shopping_admin_*` | 182 |
| B0 | reddit | `B0_wa_3mode_reddit_*` | 106 |

## 快速推断规则

run_id 格式固定，可从 baseline + benchmark + site 推导：
- VWA: `{B0|B1}_3mode_{site}_{YYYYMMDD}`
- WA: `{B0|B1}_wa_3mode_{site}_{YYYYMMDD}`

日期后缀不确定时，用 `ls results/{benchmark}/phase1/{B*}_*_{site}_*` 取最新。
