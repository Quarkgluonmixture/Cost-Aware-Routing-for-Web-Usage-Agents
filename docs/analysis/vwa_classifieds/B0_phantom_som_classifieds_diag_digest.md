# B0 phantom_som classifieds — /diag digest

> **per-condition** (site × model × mode). Run-id 在 header `Run:` 行记 manifest-bound authoritative；re-run 覆盖同名、更新 run_id。


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
| SR | **15.62%** (35/224) |
| failed + hit | 159 |
| **failed NO-hit** | **30** |
| success + hit | 3 |

v8 新规则 failed 侧: {'P43': 68, 'P45': 21, 'P44': 2, 'P46': 1}；success 侧: 无。
（`P43` 在 cls 上大量命中属预期 —— 它标记"intent 需要视觉 + 该 mode 无页面截图"这一**中性组合**，
并非预测失败；§387.10 实测补上截图的增益 ≈0。）

全部 36 个 canonical condition 现处同一版本 → **cross-mode / cross-site 聚合解锁**。

---
## 1. Header

| 字段 | 值 |
|---|---|
| Run | `B0_phantom_som_classifieds_20260527_191300_844420226_914570_R32031` (manifest-bound authoritative) |
| Condition | `phase1_phantom_som_router_0` |
| Site / Model / Mode | classifieds / **B0** (Qwen3-VL-235B-A22B via AWS proxy) / **phantom_som** |
| N episodes | 224 (scored) |
| SR | **15.6%** (35 success / 189 failed) — A100 condition_summary `success_rate=0.15625` 交叉验证一致 |
| **ruleset_version** | **`5-domsomvispsom-b1860coord`** ⚠️ cross-mode 聚合前 verify 全 digest 同版本 |
| Tier-2 深挖 | 59 ep (54 failed-no-hit 全覆盖 + 5 success-hit FP 审计), 8 sonnet sub-agent 并行 |

> **定位声明 (discover-then-freeze 硬纪律 #1)**: 这是 phantom_som 的**单 condition discover digest**, 无对照。per-rule 分布只描述 phantom_som 自己。**禁止 cross-mode 定量比较** (当前 4/6 mode discovered: dom+som+vision+phantom_som; phantom_text/phantom_prompt/reddit/B1/B2 未跑)。6-mode freeze + 全量重扫后才能聚合。

## 2. 三分类统计 (Tier-1 + Tier-2 合并)

| 类别 | 占比 | 说明 |
|---|---|---|
| **agent-limit** | ~100% of diagnosed | phantom_som 无图 → 视觉任务结构性必败 + cross-mode 通用行为缺陷 |
| **scaffold-bug** | **0** | Tier-2 全审 54 no-hit + 5 success-hit, **0 框架 bug**。phantom_som runner/obs 构造干净 |
| **benchmark-FP** | **0** | **0 评测误判**。无 string_match 过严 / N/A task 漏排除 |
| **unclear** | 0 | — |

> **强信号**: deterministic 盲区 (54 no-hit) **100% agent-limit**, 没藏任何 scaffold-bug 或 benchmark-FP。对比 dom 时代 Tier-2 常挖出框架问题, phantom_som 失败极其"干净" —— 失败纯粹是"模型在无图模式下能力局限", 这正是 paper §4 phantom routing space **cost** 叙事需要的证据 (P-SoM 省 image token 的代价 = 视觉任务能力损失, 不是工程噪声)。

## 3. Tier-1 规则分布 (failed-only per-rule)

```
P31  69  budget耗尽未完成 (trajectory_incomplete)      ← 主导
P5   39  感知缺失循环 (无视觉反馈 → 重复无效动作)
P17  37  click-back 振荡
P19  37  url_match 过早搜索页 finish
P14  24  URL 自环
P7   19  sCity=州名
P33  16  导航至裸图片URL幻觉 (NEW, phantom_som 特有)    ← 本轮 self-evolve 落码
P20  14  评测目标页从未访问
P18  14  cheapest 漏价格排序
P10  12  跨步数值记忆失败
P25  11  跨站任务跳过其中一站
P30   9  到达正确item后离开
P23   9  oldest 误用价格排序
P4    5 / P27 5 / P13 3 / P12 3 / P24 2 / P22 2 / P2 1 / P28 1
```

读法: P-SoM 失败表层主因 = **P31 budget 耗尽 (30.8%)** + **P5 感知缺失循环 (18.8%)** —— 跟 dom 的 P14/P6 主导**形态不同** (但 ⚠️ 禁 cross-mode 定量, 仅 per-condition 描述)。P5 高位反映 phantom_som 无视觉反馈时 agent 反复点同一无效元素。视觉规则 P6/P15/P16/P21 **0 命中** (它们 `mode != "dom"` gate, phantom_som 不触发) —— 即 phantom_som 的视觉盲失败目前主要落在 Tier-2 no-hit 区 (见 §4)。

## 4. Tier-2 新发现 (54 no-hit + 5 success-hit 深挖)

### 4.1 no-hit 子集 (54 ep) — 全 agent-limit, 核心 = phantom_som 无图视觉盲

8 sub-agent 一致归因, 子模式 (按规模):

1. **listing 视觉属性盲** (最大): 颜色 (red handlebars / blue iPhone / white Xbox / not red / frame color) + listing 图片内容 (puppies / CD / tennis ball / animal shape / book cover pattern) + 图中 OCR (网址 / 球衣号 / 美钞面值 / 碗数量)。agent 无 listing 缩略图, 靠标题文本猜测 → 选错 item 或幻觉作答。
2. **ref-image 匹配盲**: task 带 ref 图 (传给 multimodal model, **可 OCR**) 但需 match 到无图的页面 listing → 选错 item ("I recall seeing this exact item" / "similar to the image" / "references the person in the image")。
3. **gallery 空间布局盲**: gallery row/grid 位置 ("second row") —— 纯文本列表无法确定网格行列。
4. **phantom-img-nav 幻觉** → **已落 P33** (见 §6)。
5. **导航/排序行为缺陷** (cross-mode 通用, 非视觉): 多候选视觉甄别 (植物 lamb vs 真羊 / 挂墙架 vs 落地架) / 翻页不足 / start_url 页约束误读。

### 4.2 success-hit FP 审计 (6 ep)

| task | 触发规则 | causal? | 结论 |
|---|---|---|---|
| 2 | P14 | **纯 FP** | 多字段表单填写 (连续同 URL 但每步 page_changed=True)。⚠️ 当前 P14 已有 `any type → skip` + `≥50% page_changed → skip` guard, **理论应已 catch → 需实证为何仍 fire** (follow-up) |
| 4 | P10 | causal | 跨步遗忘已写入的描述 (真记忆失败, 侥幸 success: 服务器保留了 step3 写入) |
| 5 | P5+P14 | causal | 删对了目标 item 但误以为删的是另一辆, 后续 17 步无效 scroll (真感知盲区, eval 凭 DB 404 side-effect pass) |
| 15 | P5+P14 | causal | 到达正确 item 页但连续 3 次点同一邮件链接 action_success=False (无视觉反馈) |
| 94 | P5+P14 | causal | 到达答案页但连续 9 次点图片元素 action_success=False (纯感知盲区循环) |
| 222 | **P33** | edge | string_match yes/no 凑巧答对, 但 agent 确实迷路到图片页 → P33 presence 真实, outcome 凑对。5.9% edge 可接受 |

> **关键**: P5 在 3 个 success 上 fire **全部 causal** (phantom_som 无视觉反馈 → genuine 重复无效动作), **不是 presence-only 噪声**。这与 dom 时代 P6/P14 大量 presence-only over-fire **不同** → 现有 ruleset 在 phantom_som 上 causal 纯度高, P5 **不需收窄**。

## 5. 代表 episode

**agent-limit / 颜色视觉盲**
- task 17 "cheapest bike with red handlebars": step6 thought "current view doesn't specify handlebar color, I should click the first" → 明知无颜色信息仍猜, 选 id=10865 vs ref 79747
- task 56 "cheapest snowblower not red": step3 thought "**cannot determine its color from this view**", step4 幻觉出多型号价格列表

**agent-limit / ref-image 匹配盲**
- task 96 "item similar to image" (ref=黄色吸尘器): agent OCR 出 ref 图但无 listing 缩略图比对, 选 id=26022 vs ref 5939
- task 136 "exact item I recall" (ref=蓝色童车): agent 读出 ref 图颜色细节但选错 listing id=260 vs ref 37999

**agent-limit / 图中 OCR**
- task 199 "website mentioned in the image" (ref=kaiyo.com): agent 答 "http://localhost:9980" (页面 URL, 非图中内容)
- task 221 "how many bowls in image" (ref=6): agent 答 5

**agent-limit / phantom-img-nav (P33)**
- task 128 "jersey numbers in image": [SOM_MARKS] 暴露图片 href → agent 点 element_id=18 → obs_url 落 `/oc-content/uploads/74603/74603.png` (mark_count=2 近空 DOM) → 幻觉 "10 and 7" vs ref 99,13,80,92
- task 187 "Lightning McQueen gallery": agent 离开 start_url gallery 改搜索 → finish 停在 `.../76299/76299.png` 裸图片 URL

**benchmark-FP / scaffold-bug**: **无** (Tier-2 全审 0)

## 6. 🔁 Self-evolving — 本轮 ruleset 演进 `4-domsomvis-b1860coord → 5-domsomvispsom-b1860coord`

### ✅ P33 落码 — 导航至裸图片URL幻觉 (phantom_som 特有)
- **signal**: 任一 step 的 `obs_url` 匹配 `RAW_IMAGE_URL_RE = /oc-content/uploads/.*\.(png|jpe?g|gif|webp)($|?)` (0-token URL 正则)
- **诱因**: phantom_som 的 [SOM_MARKS] 把每个 listing 图片的 href 暴露为带 ID 可点击元素 → agent "点进图片" → 裸 PNG 页无可读内容 → 幻觉。**phantom_som 结构性独有** (vision=截图非链接 / dom=无 SOM ID → 其他 mode 不可能触发); 但 signal 本身 mode-agnostic
- **验证**: failed-fire 16 / success-fire **1/17 = 5.9%** (task 222 string_match 凑对 edge, 可接受) → 干净, 落码
- **两 sub-agent 独立发现** (batch D task 128 + batch F task 187) → 高置信

### ❌ P34 提议但**回退** — phantom_som 视觉盲 (P6 color/ref-image 孪生)
- **试落**: `obs_mode==phantom_som` + (具体颜色词 OR ref-image-visual-match), 复用 P6 已 narrowed 正则
- **回退原因**: 重扫 success-fire **21/106 = 20% = presence-only** —— 复现 P6 历史上 88%-on-success dom over-fire 的同一陷阱。两个误报根因:
  1. **"navigate to my listing of the white car"** (task 4/5/75/76): 颜色是**自己 listing 的标识符**, 文本可匹配, 非视觉判断
  2. **"I recall seeing this exact item"** (task 44-48/140): ref 图 **可被 multimodal model OCR**, 只需 OCR→搜索, 不需 match 页面 listing 图
- **决策**: 颜色/ref-image intent 本质是 **presence detector** (颜色 intent 的 success vs fail 无法用 0-token 区分), narrow 到干净需大量工作且残留 FP → 按 discover 纪律**不落 presence-only 规则**。等 phantom-family (phantom_text/prompt) 全数据再设计 success-safe 版本 (如排除 "my listing" self-identifier + 从 image 分支移除 "exact item" OCR-able 子模式)

### 📋 Follow-up 候选 (暂不落码)
1. **P15/P16/P22 扩 phantom_som**: gallery-row (P15) / image-content (P16) / img-number (P22) 的正则**已存在**但 `mode != "dom"` gate。phantom_som 这些失败明确存在 (task 14/41/187 gallery, 110/119/199/221 OCR)。但扩 gate 涉及 "dom-specific vs 无图通用" 判断, 需 phantom_text/prompt 数据一次性设计 → 等 phantom-family 齐
2. **P14 task 2 FP 实证**: 当前 guard 理论应 catch 表单填写, 需读 task 2 实际 streak 确认为何仍 fire (可能 edge case 或 guard 漏 form_value_changed 语义)
3. **start_url 页约束离开** (task 78/187): `start_url 含 iPage=/gallery` 但 finish url 不含 → 候选规则, 但需 confirm 非 over-fire
4. **count-from-image** (task 110/221): "how many ... in image" + phantom_* → 但与 P22 重叠, 并入 P22 扩 gate 讨论

## 7. Actionable

- **scaffold-bug → B-number**: **无** (Tier-2 0 scaffold, phantom_som 框架干净)
- **benchmark-FP → task 排除**: **无** (Tier-2 0 FP)
- **AMENDMENT_08 exclude 候选**: 本 condition 未发现新 exclude task (T180/cross-site/B-21 sibling 已在 §299 R21557+R5313 sourced; phantom_som 无新增)
- **cross-mode**: ⚠️ 禁定量比较直至 6-mode freeze。本 digest = per-condition discover 产物

---

### v11 数字块（`11-intent-text-fallback`，2026-08-03 补）

> 本 digest 正文成稿于更早的 ruleset。v10 落了 **+P49 / P36 carve-out / P14 carve-out**，
> v11 给 **P34/P48 换用 `_finish_intent_text()`**（answer 为空时 fallback 读 `thought`——
> B0 惯于把结论写进 `answer`，B1 留在 `thought`，旧口径因此变成了模型行为检测器）。
> 全部 48 个 canonical condition 已在 v11 下重扫，**cross-mode / cross-model 聚合以本块为准**。

| 字段 | 值 |
|---|---|
| Run | `B0_phantom_som_classifieds_20260527_191300_844420226_914570_R32031` |
| Episodes | 224（success 35 · SR 15.62%） |
| 三子集 | failed+hit 156 · failed-NO-hit 33 · success+hit 3 |
| config_missing | 0 |

| 规则 | 含义 | step 级 | episode 级 |
|---|---|---:|---:|
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT | 68 | 68 |
| `P36` | WALK_FAIL_DEGENERATE | 140 | 52 |
| `P5` | 感知缺失循环 | 49 | 39 |
| `P17` | click-back振荡 | 37 | 37 |
| `P31` | budget耗尽未完成 | 31 | 31 |
| `P14` | URL 自环 | 27 | 24 |
| `P7` | sCity=州名 | 21 | 19 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 21 | 17 |
| `P33` | 导航至裸图片URL幻觉 | 17 | 17 |
| `P20` | 评测目标页从未访问 | 14 | 14 |
| `P18` | cheapest漏价格排序 | 14 | 14 |
| `P10` | 跨步数值记忆失败 | 11 | 11 |
| `P25` | 跨站任务跳过其中一站 | 11 | 11 |
| `P30` | 到达正确item后离开 | 9 | 9 |
| `P23` | oldest误用价格排序 | 9 | 9 |
| `P19` | url_match过早搜索页finish | 6 | 6 |
| `P4` | 根节点误操作 | 11 | 5 |
| `P27` | 找不到即放弃 | 5 | 5 |
| `P37` | URL_HALLUCINATION | 4 | 4 |
| `P13` | 搜索代替浏览 | 3 | 3 |
| `P12` | 从不翻页 | 3 | 3 |
| `P24` | 不确定仍finish | 2 | 2 |
| `P22` | 图上数字dom不可读 | 2 | 2 |
| `P2` | 容器节点误点 | 1 | 1 |
| `P44` | HALLUCINATED_ELEMENT_REF | 2 | 1 |
| `P28` | benchmark-FP货币tokenize | 1 | 1 |
| `P38` | DOM_URL_AS_IMAGE | 1 | 1 |
| `P46` | COMMENT_INTENT_NO_TYPE | 1 | 1 |

> ⚠️ **解读约束**（`docs/analysis/_data_quality_audit.md`）：
> ① 本表是**症状分布，不是死因分布** —— P36/P31 经 10 例跨 benchmark 因果验证均判为 risk-marker；
> ② `P2`/`P4` 依赖 `element_bbox`，在 **vision 上结构性为 0（假 0）**；
> ③ `P36` 在 vision 上只覆盖 `type` 步（click 无 `locator_route_meta`）→ **分母与 dom/som 不同**。
