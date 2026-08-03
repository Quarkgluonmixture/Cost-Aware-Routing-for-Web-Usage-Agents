# B2 phantom_text classifieds — /diag failure attribution digest

**Run**: `B2_phantom_text_classifieds_20260614_020803_377049301_1495224_R14219` (manifest-bound authoritative)
**Condition**: phase1_phantom_text_router_0 · **Site**: classifieds · **Model**: B2 = Gemma3-4B · **Mode**: phantom_text (DOM prompt + [SOM_MARKS] 文本 + 无标注图)
**N**: 224 ep · **SR**: 1/224 = **0.4%** (6-mode 最低) · **ruleset_version**: `5-domsomvispsom-b1860coord`
**Diag date**: 2026-06-19 (Tier-1 全扫 + Tier-2 sonnet 深挖 5 ep)

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
| SR | **0.45%** (1/224) |
| failed + hit | 218 |
| **failed NO-hit** | **5** |
| success + hit | 0 |

v8 新规则 failed 侧: {'P44': 334, 'P45': 209, 'P43': 71}；success 侧: 无。
（`P43` 在 cls 上大量命中属预期 —— 它标记"intent 需要视觉 + 该 mode 无页面截图"这一**中性组合**，
并非预测失败；§387.10 实测补上截图的增益 ≈0。）

全部 36 个 canonical condition 现处同一版本 → **cross-mode / cross-site 聚合解锁**。

---
## 1. 三分类统计

| 类别 | 占比 | 说明 |
|---|---|---|
| **agent-limit** | ~100% (223/223 failed) | Gemma3-4B phantom_text 地板, 6-mode 最低 SR。5 ep Tier-2 全 agent-limit |
| scaffold-bug | 0 (1 telemetry gap) | 无 fatal bug；但 task 5 暴露 `effective_mutating_action_count=0` 漏计 GET-based 删除 (telemetry gap, 非 fatal) |
| benchmark-FP | 0 | — |

## 2. Tier-1 规则分布 (failed per-rule, hit 总数)

`P5`(感知缺失循环)=237 · `P31`(budget耗尽未完成)=209 · `P4`(根节点误操作)=**153** · `P14`(URL自环)=99 · `P33`(img-href幻觉)=66 · `P12`=52 · `P19`=45 · `P18`=26

→ P5+P31 主导。`P4`=153 显著 = element_id=1 根节点幻觉 (§322 low-default; phantom_text 裸 element_id + 无图最易幻觉，与 phantom_som P4=278 同源)。

## 3. Tier-2 深挖

**no-hit failed (4, 全 agent-limit)**:
- **task 12**: 类目推断缺失 (未搜 motorcycle，按 URL item-id 顺序猜"最新")
- **task 16**: 视觉识别图片定位 (无图 → 文字搜错 item，读错 email)
- **task 41**: ⭐ **gallery 行结构盲** ([SOM_MARKS] 文本不保留 2D grid 布局 → "second row of this page" 无法定位 = phantom_text 结构性盲区)
- **task 119**: 图片内容读取 (钞票面额读不到 → 用 listing price "9999" 代替 ref "50")
- → 16/41/119 = **phantom_text 结构性盲区** (no-image + no-grid-layout)

**success 确认 (task 5 = 唯一 success, presence-only 伪成功)**:
- agent step 6 **确实成功删除** item 84144 (reward=1.0, DOM 中消失)，但 `agent_finished=false` + `trajectory_incomplete=true` → agent 随后 24 步继续随机删除其他 listing 直到 budget 耗尽，runner 救活。
- ⚠️ **telemetry gap**: `effective_mutating_action_count=0` 即使删除成功 (GET-redirect 删除未被突变追踪器捕获) — 非 fatal，post-fire 记。

## 4. 🔁 Self-evolving — 提议 P-rule (post-fire candidates)

1. **P-vision-required**: intent 含 'in the picture' / 'shown in image' / 'denomination' + mode∈{phantom_text, dom} → agent-limit/vision-required (覆盖 16/119，扩展现有 vision-required 规则族)
2. **P-gallery-row**: intent 含 'second/nth row of this page' + start_url 含 `sShowAs=gallery` + mode∈{phantom_text, dom} → layout-blind (task 41)
3. **P-presence-only** (task 5, 同 vision/som — agent_finished=false + trajectory_incomplete=true + success=true)

→ ruleset 冻结待 B1+B2 cls freeze (§0 diag_freeze_v6_plan)。

## 5. Actionable

- 无 fatal scaffold-bug；**telemetry gap**: `effective_mutating_action_count` 漏计 GET-based 删除 (post-fire 候选，cross-ref B-1869 测量隐患族)。
- **B2 phantom_text cls = 0.4% 最低地板**；⚠️ **唯一 success 是 presence-only** → 名义 SR 虚高，真实 SR ≈ 0。
- **presence-only 伪成功跨 som+vision+phantom_text 三 mode 系统性** = B2 url_match/mutation success 普遍含「runner 救活」。

---

### v11 数字块（`11-intent-text-fallback`，2026-08-03 补）

> 本 digest 正文成稿于更早的 ruleset。v10 落了 **+P49 / P36 carve-out / P14 carve-out**，
> v11 给 **P34/P48 换用 `_finish_intent_text()`**（answer 为空时 fallback 读 `thought`——
> B0 惯于把结论写进 `answer`，B1 留在 `thought`，旧口径因此变成了模型行为检测器）。
> 全部 48 个 canonical condition 已在 v11 下重扫，**cross-mode / cross-model 聚合以本块为准**。

| 字段 | 值 |
|---|---|
| Run | `B2_phantom_text_classifieds_20260614_020803_377049301_1495224_R14219` |
| Episodes | 224（success 1 · SR 0.45%） |
| 三子集 | failed+hit 218 · failed-NO-hit 5 · success+hit 0 |
| config_missing | 0 |

| 规则 | 含义 | step 级 | episode 级 |
|---|---|---:|---:|
| `P36` | WALK_FAIL_DEGENERATE | 1521 | 178 |
| `P31` | budget耗尽未完成 | 143 | 143 |
| `P5` | 感知缺失循环 | 237 | 123 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 209 | 109 |
| `P14` | URL 自环 | 96 | 80 |
| `P44` | HALLUCINATED_ELEMENT_REF | 334 | 75 |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT | 71 | 71 |
| `P33` | 导航至裸图片URL幻觉 | 66 | 66 |
| `P4` | 根节点误操作 | 153 | 55 |
| `P12` | 从不翻页 | 52 | 52 |
| `P18` | cheapest漏价格排序 | 26 | 26 |
| `P20` | 评测目标页从未访问 | 25 | 25 |
| `P25` | 跨站任务跳过其中一站 | 13 | 13 |
| `P2` | 容器节点误点 | 6 | 5 |
| `P30` | 到达正确item后离开 | 5 | 5 |
| `P17` | click-back振荡 | 5 | 5 |
| `P10` | 跨步数值记忆失败 | 11 | 3 |
| `P22` | 图上数字dom不可读 | 2 | 2 |
| `P11` | 最新+地点组合 | 1 | 1 |
| `P38` | DOM_URL_AS_IMAGE | 1 | 1 |
| `P19` | url_match过早搜索页finish | 1 | 1 |
| `P13` | 搜索代替浏览 | 1 | 1 |

> ⚠️ **解读约束**（`docs/analysis/_data_quality_audit.md`）：
> ① 本表是**症状分布，不是死因分布** —— P36/P31 经 10 例跨 benchmark 因果验证均判为 risk-marker；
> ② `P2`/`P4` 依赖 `element_bbox`，在 **vision 上结构性为 0（假 0）**；
> ③ `P36` 在 vision 上只覆盖 `type` 步（click 无 `locator_route_meta`）→ **分母与 dom/som 不同**。
