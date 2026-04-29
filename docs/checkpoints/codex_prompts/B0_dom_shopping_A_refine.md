# Codex prompt: refine A_NON_VISUAL_TEXT_ONLY 分类 (shopping)

## 用途

`docs/analysis/cross_sites/codex_audit_shopping.json` 04-26 把 466 shopping tasks 分 4 类。后续 diag (`B0_dom_shopping_diagnostic.md`) 发现 **A_NON_VISUAL_TEXT_ONLY 82 tasks 中 60% 含 latent visual attribute**（color/theme/design 但无 ref image），boundary 不准。

任务：**只 re-classify A 类 82 tasks**，保留 B/C/D 不动（它们已 verified clean 见 diag §3.H3）。

输出 → `docs/analysis/cross_sites/codex_audit_shopping_A_refined.json`

## 新 4-class A taxonomy

| Sub-class | 定义 | 期望 SR (B0 dom paper-grade clean) |
|---|---|---|
| **A1_PURE_TEXT** | 纯文本检索/查询/简单 form 操作；intent 只含 exact name / category / category + 数值条件；DOM AXTree 完全足够 (e.g. "What is the price of X" / "Add Y to wishlist") | 较高 (>15%) |
| **A2_LATENT_VISUAL** | intent 含 color / theme / design / pattern / shape / style 等 visual attribute 但 **无 ref image**, agent 必须从 DOM 文本推断 visual property (e.g. "red blanket" / "Rick and Morty themed" / "round watch") | 必败 (~5%) |
| **A3_AGGREGATION** | intent 含 比较 / 排序 / aggregation keyword: least/most expensive, cheapest, highest rated, lowest, most reviews, average, sum, count, top-N 等。需要遍历 list (Magento 12 items × ~10 fields) | 必败 (<5%) |
| **A4_FORM_ACTION** | intent 是 modify cart / change settings / submit review / update profile / cancel order 等 deterministic form 操作；agent 只需 navigate 到 form, 填字段, submit。无 retrieval 不 ambiguous | 较高 (>20%) |

(注: A2 + A3 可能重叠 e.g. "least expensive red blanket" — 选**主导 dimension** 标记: 是 visual attribute 主 (A2) 还是 aggregation 主 (A3)? 优先 A3 因为 aggregation 更基本)

## 输入

```
docs/analysis/cross_sites/codex_audit_shopping.json     # 04-26 audit, 取 category=A 的 82 tasks
external/visualwebarena/config_files/vwa/test_shopping.json  # 原 task config (intent / eval / image)
```

## 输出 schema

```json
[
  {
    "task_id": 0,
    "intent": "Buy the least expensive red blanket from \"Blankets & Throws\" category.",
    "original_category": "A_NON_VISUAL_TEXT_ONLY",
    "refined_category": "A3_AGGREGATION",
    "primary_signals": ["least expensive (aggregation keyword)", "red (visual attribute, secondary)"],
    "rationale": "Primary axis is price aggregation (least expensive), color is secondary visual filter. Agent must sort/compare across Blankets & Throws list; visual color verification is a downstream check after sort but failure mode is aggregation cost."
  },
  ...
]
```

## 验证 step

跑完后 codex 自验证: 82 tasks 全标，refined_category 分布合理 (~A1: 20-30, A2: 25-35, A3: 25-35, A4: 5-10)。每条 rationale 1-2 句解释 primary signal。

## token 预算

~50K (read 82 task intents + write 82 entries with rationale)

## 触发命令

```bash
codex run --prompt docs/checkpoints/codex_prompts/B0_dom_shopping_A_refine.md
# 或 paste prompt 整个内容到 codex CLI
```

## paper integration plan

跑完后:
1. 用新 taxonomy 复算 SR breakdown — 验证 A2/A3 SR ~0%, A1/A4 SR > 15%
2. paper Section 7 generalization prose 用 4-class A breakdown 替代单一 A 类
3. fig5 category × mode heatmap 加 A1/A2/A3/A4 column (paper Section 4)
4. 后续 codex disagreement clustering (#14 待 5-mode shopping 数据齐) 用 refined taxonomy 做 per-task analysis
