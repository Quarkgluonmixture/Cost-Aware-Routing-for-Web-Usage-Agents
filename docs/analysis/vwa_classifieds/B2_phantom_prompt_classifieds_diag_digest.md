# B2 phantom_prompt classifieds — /diag failure attribution digest

**Run**: `B2_phantom_prompt_classifieds_20260616_142027_795794905_1801050_R10175` (manifest-bound authoritative)
**Condition**: phase1_phantom_prompt_router_0 · **Site**: classifieds · **Model**: B2 = Gemma3-4B · **Mode**: phantom_prompt (SoM prompt + AXTree 文本 + 无图; axis-2 control)
**N**: 224 ep · **SR**: 4/224 = **1.8%** · **ruleset_version**: `5-domsomvispsom-b1860coord`
**Diag date**: 2026-06-19 (Tier-1 全扫 + Tier-2 sonnet 深挖 9 ep)

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
| SR | **1.79%** (4/224) |
| failed + hit | 217 |
| **failed NO-hit** | **3** |
| success + hit | 0 |

v8 新规则 failed 侧: {'P44': 577, 'P45': 249, 'P43': 69}；success 侧: 无。
（`P43` 在 cls 上大量命中属预期 —— 它标记"intent 需要视觉 + 该 mode 无页面截图"这一**中性组合**，
并非预测失败；§387.10 实测补上截图的增益 ≈0。）

全部 36 个 canonical condition 现处同一版本 → **cross-mode / cross-site 聚合解锁**。

---
## 1. 三分类统计

| 类别 | 占比 | 说明 |
|---|---|---|
| **agent-limit** | 6/9 Tier-2 | Gemma3-4B phantom_prompt 地板 |
| **benchmark-FP** | **3/9** (task 5/110/142) | ⭐ pprompt = B2 cls 最多 FP 的 mode |
| scaffold-bug | 0 | — |

## 2. Tier-1 规则分布 (failed per-rule, hit 总数)

`P5`(感知缺失)=288 · `P31`(budget耗尽)=207 · `P14`(URL自环)=153 · `P2`(容器节点误点)=65 · `P19`=55 · `P12`=55 · `P33`=52 · `P18`=34

→ P5+P31 主导 (Gemma 地板)。

## 3. Tier-2 深挖

**no-hit failed (5, 全 agent-limit)**:
- task 40 (搜索识别错 Whirlpool→LG) · 61 (误读任务 video game→bowling) · 108 (幻觉直跳 item 10542) · 111 (无图判球衣队名失败) · 174 (Black Friday logo 图像识别, 无图)

**success 审计 (4) — ⭐ B2 success 大量非真能力**:
- **task 5**: presence-only + FP (item 84144 run 前已 404，agent 从未删除，success 来自 eval program_html 对 404 返回 PASS) — 注意 ≠ ptext task 5 (那个 agent 真删了)
- **task 106**: 真成功 (constraint-skipping 侥幸：正确导航到 Photo 类最贵 Canon item，email 答对，但跳过 animal-image 约束 — phantom 无图无法验证，碰巧最贵 item 即正解)
- **task 110**: benchmark-FP (lucky numeric guess "0"，未访问正确 item 34406，ref "0|OR|zero" string_match 宽松通过)
- **task 142**: benchmark-FP (访问错误 item 65955 而非 ref 22310，但两者都在 Pennsylvania，string_match("Pennsylvania") 单点巧合命中，eval 不验 URL)

## 4. ⭐ 关键: B2 cls 名义 SR 严重虚高 (真实有效 ≈ 0)

pprompt SR 4/224 拆: **3 benchmark-FP + 1 constraint-skipping 侥幸** → **真实有效 SR ≈ 0**。
- FP 三源: presence-only (runner 救活) + lucky-guess (string_match 宽松短答案) + string-coincidence (不验 URL)
- **task 142 跨 psom+pprompt 都 FP** (string_match 不验 URL = 系统性 eval 宽松)
- → 印证 §335-338 B2 cls 真地板，且**更强**: 连那 1-4 个 success 都非可靠能力。

## 5. 🔁 Self-evolving — 提议 P-rule (post-fire candidates)

1. **P-vision-required** (phantom mode image-identification: intent 含 image/logo/jersey + obs_mode∈phantom*) 覆盖 61/111/174
2. **P-lucky-numeric-FP**: `eval_type=string_match + reference ∈ 短数字/否定词 + agent_url ≠ correct item URL` → benchmark-FP candidate (task 110)
3. **P-presence-only-delete**: `agent_finished=false + trajectory_incomplete=true + delete_remove_count=0 + success=true` (task 5)

→ ruleset 冻结待 B1+B2 cls freeze (§0 diag_freeze_v6_plan)。

## 6. Actionable

- **benchmark-FP task 5/110/142**: post-fire 评估是否排除 (string_match 宽松 + presence-only；142 同 psom = 系统性 eval-不验-URL)。
- **B2 pprompt cls = agent-limit 地板**；名义 SR 1.8% 虚高 (真实有效 ≈ 0)。

---

### v11 数字块（`11-intent-text-fallback`，2026-08-03 补）

> 本 digest 正文成稿于更早的 ruleset。v10 落了 **+P49 / P36 carve-out / P14 carve-out**，
> v11 给 **P34/P48 换用 `_finish_intent_text()`**（answer 为空时 fallback 读 `thought`——
> B0 惯于把结论写进 `answer`，B1 留在 `thought`，旧口径因此变成了模型行为检测器）。
> 全部 48 个 canonical condition 已在 v11 下重扫，**cross-mode / cross-model 聚合以本块为准**。

| 字段 | 值 |
|---|---|
| Run | `B2_phantom_prompt_classifieds_20260616_142027_795794905_1801050_R10175` |
| Episodes | 224（success 4 · SR 1.79%） |
| 三子集 | failed+hit 217 · failed-NO-hit 3 · success+hit 0 |
| config_missing | 0 |

| 规则 | 含义 | step 级 | episode 级 |
|---|---|---:|---:|
| `P36` | WALK_FAIL_DEGENERATE | 1458 | 169 |
| `P5` | 感知缺失循环 | 288 | 151 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 249 | 139 |
| `P31` | budget耗尽未完成 | 134 | 134 |
| `P44` | HALLUCINATED_ELEMENT_REF | 577 | 114 |
| `P14` | URL 自环 | 151 | 113 |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT | 69 | 69 |
| `P12` | 从不翻页 | 55 | 55 |
| `P33` | 导航至裸图片URL幻觉 | 53 | 53 |
| `P18` | cheapest漏价格排序 | 34 | 34 |
| `P20` | 评测目标页从未访问 | 25 | 25 |
| `P2` | 容器节点误点 | 65 | 15 |
| `P25` | 跨站任务跳过其中一站 | 13 | 13 |
| `P30` | 到达正确item后离开 | 5 | 5 |
| `P10` | 跨步数值记忆失败 | 6 | 3 |
| `P22` | 图上数字dom不可读 | 2 | 2 |
| `P17` | click-back振荡 | 1 | 1 |
| `P13` | 搜索代替浏览 | 1 | 1 |
| `P4` | 根节点误操作 | 3 | 1 |

> ⚠️ **解读约束**（`docs/analysis/_data_quality_audit.md`）：
> ① 本表是**症状分布，不是死因分布** —— P36/P31 经 10 例跨 benchmark 因果验证均判为 risk-marker；
> ② `P2`/`P4` 依赖 `element_bbox`，在 **vision 上结构性为 0（假 0）**；
> ③ `P36` 在 vision 上只覆盖 `type` 步（click 无 `locator_route_meta`）→ **分母与 dom/som 不同**。
