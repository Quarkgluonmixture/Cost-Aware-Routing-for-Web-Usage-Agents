---
type: task
status: planned
priority: P2
horizon: backlog
order: 5
blocker: "改 ALL_RULES 判据 ⇒ discover-then-freeze 纪律: bump RULESET_VERSION v12 + diag_rescan_all 全量重扫 + 53 份 digest 补 v12 数字块 + 重跑依赖 diag_scans 的两个下游; 不影响 SR / 失败桶"
eta: "user 2026-09-11 定: 放后续 todo, 不急"
detail: docs/reference/master_bug_catalog.md#B-1999
created: 2026-09-11
updated: 2026-09-11
---

# diag v12 规则批 (B-1999 + B-2000 + P33 收窄) + task 41 布局核实

**来源**: B5 补登记时的 Tier-2 与 0-token 复核 (笔记 §509.4–509.5, 汇总 `docs/analysis/vwa_classifieds/B5_classifieds_cross_mode_diag_summary.md` §3.3)。
三条都在 analysis 层, 不碰 fire 路径, 不需要 witness。

## 要做的

1. **B-1999 P31**: 「终态已到参考页」豁免从比 URL path 改为比完整 URL (与 url_match EXACT 同口径)。
   现状 cls 漏标 B0 203 · B1 436 · B2 453 · B5 147 个跑满预算的失败; reddit 0。
2. **B-2000 P10**: `_extract_numbers` 先吃千分位 (`\d{1,3}(?:,\d{3})+(?:\.\d+)?`) 再去逗号。
   只解释一小部分误报 (B5 失败侧 61→50, 成功侧 31→30); 若要让 P10 在强模型上可用, 另需把「价格 vs 日期 / 型号数字」分开比。
3. **P33 按 mode 收窄**: som / vision 上降为中性事件 (审计 som 4/4 非死因); 无图 mode 保留。
4. bump `RULESET_VERSION` → `12-*`, `diag_rescan_all.py --baseline-dir results/diag_scans/v11_vwa` 全量重扫 (B5 已在 extension 里被固定 run),
   53 份 digest 补 v12 数字块, 同步 `.claude/skills/diag/SKILL.md` 规则列表 (该文件仍写 v10, 已漂一版), 重跑
   `aggregate_conditional_failure_attribution` / `page_change_corrected_metrics`。
5. **task 41 布局核实** (独立小项): 从 A100 取一张 task 41 起始页截图, 数 gallery 每行几个。
   若确认 P79 视口下「第二行」≠ 参考答案假设的每行 3 个 ⇒ 登记为 benchmark-FP, 由 user 定是否走 AMENDMENT 剔除 (SR 绝对值影响上限 0.45pp, 不影响 mode 间差分)。

## 顺带可以做的 (未复核, 不要直接落码)
汇总 §5 的 #7–#9 (同一搜索词重复 ≥3 次 / 说了要核实下一步就 finish / 物品名词错配) —— 先做 0-token 全量复核看成功侧误伤, 再议。
