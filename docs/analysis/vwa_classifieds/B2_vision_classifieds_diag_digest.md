# B2 vision classifieds — /diag failure attribution digest

**Run**: `B2_vision_classifieds_20260612_221910_098760264_1351451_R9288` (manifest-bound authoritative)
**Condition**: phase1_vision_router_0 · **Site**: classifieds · **Model**: B2 = Gemma3-4B · **Mode**: vision (raw screenshot only, 无 AXTree 文本)
**N**: 224 ep · **SR**: 5/224 = **2.2%** · **ruleset_version**: `5-domsomvispsom-b1860coord`
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
| SR | **2.23%** (5/224) |
| failed + hit | 214 |
| **failed NO-hit** | **5** |
| success + hit | 0 |

v8 新规则 failed 侧: 无；success 侧: 无。
（`P43` 在 cls 上大量命中属预期 —— 它标记"intent 需要视觉 + 该 mode 无页面截图"这一**中性组合**，
并非预测失败；§387.10 实测补上截图的增益 ≈0。）

全部 36 个 canonical condition 现处同一版本 → **cross-mode / cross-site 聚合解锁**。

---
## 1. 三分类统计

| 类别 | 占比 | 说明 |
|---|---|---|
| **agent-limit** | ~100% (219/219 failed) | Gemma3-4B vision grounding 地板。8 ep Tier-2 全 agent-limit |
| scaffold-bug | 0 | ⚠️ Tier-2 确认: **截图传递正常 + 坐标转换无误** = 无 vision scaffold bug |
| benchmark-FP | 0 | — |

## 2. Tier-1 规则分布 (failed per-rule, hit 总数)

`P5`(感知缺失循环)=**329** · `P31`(budget耗尽未完成)=197 · `P14`(URL自环)=119 · `P19`(url_match过早finish)=45 · `P18`(漏价格排序)=32 · `P20`(评测目标页从未访问)=25 · `P12`=13 · `P25`=13

→ **P5=329 是 6 mode 最高** = 纯像素 grounding 下感知缺失最严重 (无 AXTree 文本兜底 → Gemma 看不懂 screenshot 反复试)。

## 3. Tier-2 深挖

**no-hit failed (5, 全 agent-limit)**:
- **task 10**: early-abandon (到错误 item Canon 镜头 → 判 "task impossible, no furniture visible" → 2 步 finish)
- **task 50**: wrong-item navigation (读错 listing 的 email)
- **task 119**: thumbnail 视觉读取失败 (钞票面额读成 "$1" vs ref "$50")
- **task 215**: 无搜索策略 (随机点 Xbox 2 步 finish)
- **task 221**: 到达正确页但计数失败 (数碗答 "0" vs ref "6")

**success 审计 (3, 全 presence-only 伪成功 `hit_causal=false`)**:
- **task 121 / 123 / 193**: `agent_finished=false` + `trajectory_incomplete=true` + runner step-limit 截断时 url_match 通过，no_op_rate 64-83% (反复无效 click 偶然停在目标 URL) = **与 som 87/124 同模式跨 mode 复现**。

## 4. 🔁 Self-evolving — 提议 P-rule (post-fire candidates)

⭐ **P-presence-only (最高优先, vision 给出最精确信号)**:
```
agent_finished=false AND trajectory_incomplete=true AND success=true
AND eval_context_mode='agent_page'  →  presence_only_rescued
```
命中 vision 3/3 + som 87/124 = **跨 mode 系统性, 零 FP**。从 SR 剔除或单独报告 (剥离「runner 救活」vs「agent 完成」)。

→ ruleset 冻结待 B1+B2 cls freeze (§0 diag_freeze_v6_plan)。

## 5. Actionable

- 无 scaffold-bug · 无 benchmark-FP。
- **B2 vision cls = agent-limit 地板** (P5 感知缺失 6-mode 最严重 = 纯像素无文本兜底)。
- ⚠️ **presence-only 伪成功跨 som+vision 复现** = B2 url_match SR 系统性含 runner-救活成分 (paper 需注 + post-fire P-rule, B-1869 sibling)。

---

### v11 数字块（`11-intent-text-fallback`，2026-08-03 补）

> 本 digest 正文成稿于更早的 ruleset。v10 落了 **+P49 / P36 carve-out / P14 carve-out**，
> v11 给 **P34/P48 换用 `_finish_intent_text()`**（answer 为空时 fallback 读 `thought`——
> B0 惯于把结论写进 `answer`，B1 留在 `thought`，旧口径因此变成了模型行为检测器）。
> 全部 48 个 canonical condition 已在 v11 下重扫，**cross-mode / cross-model 聚合以本块为准**。

| 字段 | 值 |
|---|---|
| Run | `B2_vision_classifieds_20260612_221910_098760264_1351451_R9288` |
| Episodes | 224（success 5 · SR 2.23%） |
| 三子集 | failed+hit 214 · failed-NO-hit 5 · success+hit 0 |
| config_missing | 0 |

| 规则 | 含义 | step 级 | episode 级 |
|---|---|---:|---:|
| `P5` | 感知缺失循环 | 329 | 170 |
| `P36` | WALK_FAIL_DEGENERATE | 824 | 144 |
| `P31` | budget耗尽未完成 | 104 | 104 |
| `P14` | URL 自环 | 118 | 97 |
| `P18` | cheapest漏价格排序 | 32 | 32 |
| `P20` | 评测目标页从未访问 | 25 | 25 |
| `P12` | 从不翻页 | 13 | 13 |
| `P25` | 跨站任务跳过其中一站 | 13 | 13 |
| `P17` | click-back振荡 | 7 | 7 |
| `P19` | url_match过早搜索页finish | 4 | 4 |
| `P10` | 跨步数值记忆失败 | 2 | 2 |
| `P13` | 搜索代替浏览 | 2 | 2 |
| `P22` | 图上数字dom不可读 | 2 | 2 |
| `P33` | 导航至裸图片URL幻觉 | 2 | 2 |
| `P27` | 找不到即放弃 | 1 | 1 |
| `P24` | 不确定仍finish | 1 | 1 |
| `P30` | 到达正确item后离开 | 1 | 1 |
| `P1` | 元素中心越界 | 4 | 1 |

> ⚠️ **解读约束**（`docs/analysis/_data_quality_audit.md`）：
> ① 本表是**症状分布，不是死因分布** —— P36/P31 经 10 例跨 benchmark 因果验证均判为 risk-marker；
> ② `P2`/`P4` 依赖 `element_bbox`，在 **vision 上结构性为 0（假 0）**；
> ③ `P36` 在 vision 上只覆盖 `type` 步（click 无 `locator_route_meta`）→ **分母与 dom/som 不同**。
