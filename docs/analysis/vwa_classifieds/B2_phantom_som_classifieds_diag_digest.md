# B2 phantom_som classifieds — /diag failure attribution digest

**Run**: `B2_phantom_som_classifieds_20260615_044451_093238285_1626673_R22577` (manifest-bound authoritative)
**Condition**: phase1_phantom_som_router_0 · **Site**: classifieds · **Model**: B2 = Gemma3-4B · **Mode**: phantom_som (SoM prompt + [SOM_MARKS] 文本 + 无标注图)
**N**: 224 ep · **SR**: 2/224 = **0.9%** · **ruleset_version**: `5-domsomvispsom-b1860coord`
**Diag date**: 2026-06-19 (Tier-1 全扫 + Tier-2 sonnet 深挖 8 ep)

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
| SR | **0.89%** (2/224) |
| failed + hit | 217 |
| **failed NO-hit** | **5** |
| success + hit | 0 |

v8 新规则 failed 侧: {'P45': 188, 'P44': 104, 'P43': 71}；success 侧: 无。
（`P43` 在 cls 上大量命中属预期 —— 它标记"intent 需要视觉 + 该 mode 无页面截图"这一**中性组合**，
并非预测失败；§387.10 实测补上截图的增益 ≈0。）

全部 36 个 canonical condition 现处同一版本 → **cross-mode / cross-site 聚合解锁**。

---
## 1. 三分类统计

| 类别 | 占比 | 说明 |
|---|---|---|
| **agent-limit** | ~99% (7/8 Tier-2) | Gemma3-4B phantom_som 地板。P4 element_id=1 幻觉 = 纯 agent-limit |
| **benchmark-FP** | 1 (task 142) | eval 仅 string_match('Pennsylvania') 不验 URL → 到错误 item 但页含 Pennsylvania → success (轻度 FP) |
| scaffold-bug | 0 | P4 element_id=1 确认在 [SOM_MARKS] 真实存在 = 非构造错 |

## 2. Tier-1 规则分布 (failed per-rule, hit 总数)

`P4`(根节点误操作)=**278** (全 6-mode 最高) · `P5`(感知缺失)=215 · `P31`(budget耗尽)=210 · `P14`(URL自环)=116 · `P19`=56 · `P12`=46 · `P33`(img-href幻觉)=46 · `P18`=32

→ **P4=278 全场最高** = element_id=1 幻觉最严重 (SoM-prompt click-priming，§322: psom>ptext)。

## 3. Tier-2 深挖

**no-hit failed (6, 全 agent-limit)**:
- **task 5**: cycling deadlock (element_id=4 循环点击 28/30 步)
- **task 83 / 85 / 174**: 图像识别能力缺失 (无图 → 无法区分含/不含特定视觉元素)
- **task 173 / 199**: ⭐ **P33 img-href 幻觉** (把 [SOM_MARKS] img src PNG 路径 / id=1 href 当作 website 答出，intent 问"图像内 website" ref=kaiyo.com)

**success 审计 (2)**:
- **task 201 = presence-only 伪成功** (`agent_finished=false` + `trajectory_incomplete=true`)。P4 (eid=1 click) hit 但 `hit_causal=false`。
- **task 142 = benchmark-FP** (genuine finish 但 eval string_match 宽松：访问非目标 item/65955 而非 ref item/22310，但两页都含 "Pennsylvania" → success)。

## 4. ⭐ 关键: P4 element_id=1 = 纯 agent-limit (§322 假说 B2 cross-model 确认)

task 199/201 实证: element_id=1 在 [SOM_MARKS] 中**真实存在** (root/body 标签, bbox=[0,0,10,10], action_success=True) = **非 scaffold 构造错**。B2 4B 在无法从当前 obs 获取目标信息时，把 id=1 的 href/URL 当作 **low-information fallback 信息源** (task 199 thought 明文记录推理链)。→ **§322 "element_id=1=幻觉 low-default 非 renumber-root" 假说从 B1(4B) 扩到 B2(Gemma 4B) = cross-model 确认**，且 B2 比 B1 更严重 (P4=278)。

## 5. 🔁 Self-evolving — 提议 P-rule (post-fire candidates)

1. **P33-img-url-combo**: finish answer 匹配 `localhost/oc-content/uploads/.+\.png` OR `localhost` domain 单体 OR element_id=1 的 href，且 intent 含 website/address/image → `image_content_invisible_hallucination` (覆盖 173/199，零 FP)
2. **P-presence-only** (task 201, 同 vision/som/ptext)

→ ruleset 冻结待 B1+B2 cls freeze (§0 diag_freeze_v6_plan)。

## 6. Actionable

- **benchmark-FP task 142**: eval string_match 不验 URL 宽松 FP (轻度，post-fire 评估是否排除/上游修)。
- **B2 psom cls = agent-limit 地板** + P4 element_id=1 幻觉 6-mode 最严重 (§322 B2 确认)。
- presence-only 伪成功跨 som+vision+ptext+psom **4 mode** 系统性。
