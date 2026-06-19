# B2 som classifieds — /diag failure attribution digest

**Run**: `B2_som_classifieds_20260611_210828_923656661_1218867_R3380` (manifest-bound authoritative)
**Condition**: phase1_som_router_0 · **Site**: classifieds · **Model**: B2 = Gemma3-4B · **Mode**: som (Set-of-Marks 标注图 + [SOM_MARKS] 文本)
**N**: 224 ep · **SR**: 5/224 = **2.2%** · **ruleset_version**: `5-domsomvispsom-b1860coord`
**Diag date**: 2026-06-19 (首次 B2 cls diag, Tier-1 全扫 + Tier-2 sonnet ×2 深挖 22 ep)

> ⚠️ 单 condition digest，不下 cross-mode 结论。cross-mode 定量待 B1+B2 cls freeze。

## 1. 三分类统计

| 类别 | 占比 | 说明 |
|---|---|---|
| **agent-limit** | ~100% (219/219 failed) | Gemma3-4B cls ~2% 地板。22 ep Tier-2 (19 no-hit + 3 success-audit) **全 agent-limit** |
| scaffold-bug | 0 | Tier-2 主动找 SoM 标注图未传 / mark-id 错位 等框架 bug，未发现 |
| benchmark-FP | 0 | no-hit finish answer 语义也错 |

## 2. Tier-1 规则分布 (failed per-rule, hit 总数)

`P5`(感知缺失循环)=244 · `P31`(budget耗尽未完成)=187 · `P14`(URL自环)=127 · `P4`(根节点误操作)=116 · `P12`(从不翻页)=79 · `P19`(url_match过早finish)=43 · `P18`(漏价格排序)=28 · `P33`(img-href幻觉)=27

→ P5+P31 主导 (Gemma 地板)。`P4`=116 显著 (som 也高) = element_id 误操作类，与 phantom_som P4=278 同源 (§322 element_id 幻觉, B2 4B 比 B1 更差)。

## 3. Tier-2 深挖

**no-hit failed (19, 全 agent-limit)** — 子类分布:
- **url_match 导航到错误 item** (最大子类, task 18/108/146/185/201/215...): B2 到错误 item 页就 finish，agent_url ≠ reference_url
- **视觉计数 / 类别误判** (192 partial-answer 只答 red 漏 white · 215 把 VR headset 当相机)
- **搜索循环卡死** (task 182: 重复 type 'Playstation' 16+ 次无结果 → 页面漂移到 contact → 提交管理员邮件)
- **错误页面循环陷阱** (task 5 user-items 打转 / 80 contact form 循环)
- **多约束聚合失败** (25 颜色+品类+日期 / 41 gallery 行内价格区间)

**success 审计 (3, P-rule fire)**:
- **task 87 / 124 = presence-only 伪成功 (`hit_causal=false`)**: B2 在极早 step 偶然到达正确 URL (url_match PASS)，但**完全不感知任务完成**，后续 25-29 步全是无效 click (no_op_rate 0.83-0.97 / page_unchanged_streak ≥15)，靠 runner 在 budget 耗尽时截最终 URL 救活 = SoM 下 Gemma 无法感知「已达目标」(§335 finish-less 极端版)。
- task 233 = **真实成功** (`hit_causal=true`): 从封面图正确识别 "The Lion King"。

> ⚠️ **测量隐患**: B2 的部分 url_match success (87/124) 是「runner 最终 URL 快照救活」而非「agent 主动完成」→ **B2 名义 SR 含运气/救活成分，真实能力 < 名义 SR**。paper 报 B2 SR 时需注 (与 B-1869 walk_fail-fallback-报 success 同类测量隐患, post-fire candidate)。

## 4. 🔁 Self-evolving — 提议 P-rule (post-fire candidates, 本轮不落码)

1. **P-wrong-url-navigation** (最高优先): `url_match eval + eval_source_agent_url != reference_url` → 高 prevalence + 零 FP，覆盖 no-hit 最大子类。
2. **P-presence-only-success**: `success=true + no_op_rate>0.8 + page_unchanged_streak≥15` → 剥离「运气成功 vs 有效成功」，对 Pareto / drop-one 剥离 B2 真实贡献有直接价值。
3. **P-search-loop-stuck**: `≥5 连续相同 type action + url 不变` (task 182)。

→ ruleset 冻结待 B1+B2 cls freeze 一起评估 (§0 diag_freeze_v6_plan)。

## 5. Actionable

- 无 scaffold-bug B-number · 无 benchmark-FP task 排除。
- **B2 som cls = agent-limit 地板**；⚠️ presence-only 伪成功 (87/124) = B2 SR 含 runner-救活成分，paper SR 报告需注「B2 真实能力 < 名义 SR」(post-fire, B-1869 sibling)。
