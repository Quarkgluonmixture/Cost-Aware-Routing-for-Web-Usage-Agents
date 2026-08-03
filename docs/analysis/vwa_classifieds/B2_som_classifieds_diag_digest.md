# B2 som classifieds — /diag failure attribution digest

**Run**: `B2_som_classifieds_20260611_210828_923656661_1218867_R3380` (manifest-bound authoritative)
**Condition**: phase1_som_router_0 · **Site**: classifieds · **Model**: B2 = Gemma3-4B · **Mode**: som (Set-of-Marks 标注图 + [SOM_MARKS] 文本)
**N**: 224 ep · **SR**: 5/224 = **2.2%** · **ruleset_version**: `5-domsomvispsom-b1860coord`
**Diag date**: 2026-06-19 (首次 B2 cls diag, Tier-1 全扫 + Tier-2 sonnet ×2 深挖 22 ep)

> ⚠️ 单 condition digest，不下 cross-mode 结论。cross-mode 定量待 B1+B2 cls freeze。


## 0v8. v8 freeze 补记（2026-07-27）— cls 行为**不是**字节不变

`RULESET_VERSION` 升至 **`8-reddit-p41p46-b1890fix`**。该批规则源自 **reddit** discover，但有两处**确实改变了 cls 行为**，
均已逐条定性核实（不是回归）：

1. **B-1890 修复**：`P35`/`P39` 原先 guard 在 `effective_mutating_action_count`，而该字段从未被 runner
   填充、恒为 0 → guard 是 **no-op**，规则比其 docstring 声称的更宽松。v8 改为从 step record 派生突变计数。
   抽查确认被移除的旧命中确实有 6–8 个突变步（即**旧命中是错的**）。
2. **P33 正则扩展**：加入 reddit 的 `/submission_images/` 路径。cls 侧因此 **+1 例**（cls task 233 —— 它的
   `sites` 只写 classifieds，但 intent 实际要求"the characters in the image **on Reddit**"，
   该 episode 真的访问了 `localhost:9999`，旧正则漏检）。

本 condition 的 v8 数字 —— **跨 condition / 跨站聚合请用这一组**：

| 指标 | v8 |
|---|---|
| SR | **2.23%** (5/224) |
| failed + hit | 216 |
| **failed NO-hit** | **3** |
| success + hit | 1 |

v8 新规则 failed 侧: {'P45': 214, 'P44': 102}；success 侧: 无。
（`P43` 在 cls 上大量命中属预期 —— 它标记"intent 需要视觉 + 该 mode 无页面截图"这一**中性组合**，
并非预测失败；§387.10 实测补上截图的增益 ≈0。）

全部 36 个 canonical condition 现处同一版本 → **cross-mode / cross-site 聚合解锁**。

---
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

---

### v11 数字块（`11-intent-text-fallback`，2026-08-03 补）

> 本 digest 正文成稿于更早的 ruleset。v10 落了 **+P49 / P36 carve-out / P14 carve-out**，
> v11 给 **P34/P48 换用 `_finish_intent_text()`**（answer 为空时 fallback 读 `thought`——
> B0 惯于把结论写进 `answer`，B1 留在 `thought`，旧口径因此变成了模型行为检测器）。
> 全部 48 个 canonical condition 已在 v11 下重扫，**cross-mode / cross-model 聚合以本块为准**。

| 字段 | 值 |
|---|---|
| Run | `B2_som_classifieds_20260611_210828_923656661_1218867_R3380` |
| Episodes | 224（success 5 · SR 2.23%） |
| 三子集 | failed+hit 215 · failed-NO-hit 4 · success+hit 1 |
| config_missing | 0 |

| 规则 | 含义 | step 级 | episode 级 |
|---|---|---:|---:|
| `P36` | WALK_FAIL_DEGENERATE | 1657 | 167 |
| `P5` | 感知缺失循环 | 244 | 153 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 214 | 132 |
| `P31` | budget耗尽未完成 | 115 | 115 |
| `P14` | URL 自环 | 121 | 98 |
| `P12` | 从不翻页 | 79 | 79 |
| `P44` | HALLUCINATED_ELEMENT_REF | 102 | 39 |
| `P4` | 根节点误操作 | 116 | 34 |
| `P18` | cheapest漏价格排序 | 28 | 28 |
| `P33` | 导航至裸图片URL幻觉 | 27 | 27 |
| `P20` | 评测目标页从未访问 | 23 | 23 |
| `P25` | 跨站任务跳过其中一站 | 12 | 12 |
| `P2` | 容器节点误点 | 12 | 6 |
| `P32` | 文本误入价格filter | 3 | 3 |
| `P13` | 搜索代替浏览 | 2 | 2 |
| `P22` | 图上数字dom不可读 | 2 | 2 |
| `P10` | 跨步数值记忆失败 | 2 | 2 |
| `P28` | benchmark-FP货币tokenize | 1 | 1 |
| `P19` | url_match过早搜索页finish | 1 | 1 |
| `P17` | click-back振荡 | 1 | 1 |
| `P37` | URL_HALLUCINATION | 1 | 1 |

> ⚠️ **解读约束**（`docs/analysis/_data_quality_audit.md`）：
> ① 本表是**症状分布，不是死因分布** —— P36/P31 经 10 例跨 benchmark 因果验证均判为 risk-marker；
> ② `P2`/`P4` 依赖 `element_bbox`，在 **vision 上结构性为 0（假 0）**；
> ③ `P36` 在 vision 上只覆盖 `type` 步（click 无 `locator_route_meta`）→ **分母与 dom/som 不同**。
