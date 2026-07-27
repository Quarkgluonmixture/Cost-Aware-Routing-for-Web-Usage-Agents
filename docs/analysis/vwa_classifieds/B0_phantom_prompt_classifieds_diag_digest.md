# B0 phantom_prompt classifieds — /diag digest

> **per-condition** (site × model × mode). Run-id 在 header `Run:` 行记 manifest-bound authoritative。


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
| SR | **19.64%** (44/224) |
| failed + hit | 161 |
| **failed NO-hit** | **19** |
| success + hit | 9 |

v8 新规则 failed 侧: {'P44': 41, 'P43': 66, 'P45': 34}；success 侧: 无。
（`P43` 在 cls 上大量命中属预期 —— 它标记"intent 需要视觉 + 该 mode 无页面截图"这一**中性组合**，
并非预测失败；§387.10 实测补上截图的增益 ≈0。）

全部 36 个 canonical condition 现处同一版本 → **cross-mode / cross-site 聚合解锁**。

---
## 1. Header

| 字段 | 值 |
|---|---|
| Run | `B0_phantom_prompt_classifieds_20260528_040546_107246795_987141_R14655` |
| Condition | `phase1_phantom_prompt_router_0` |
| Site / Model / Mode | classifieds / **B0** (Qwen3-VL-235B-A22B via AWS proxy) / **phantom_prompt** (SoM-style prompt + AXTree text, **无标注图**) |
| N episodes | 224 |
| SR | **19.6%** (44 success / 180 failed) — A100 condition_summary `success_rate=0.196` 交叉验证 |
| **ruleset_version** | **`5-domsomvispsom-b1860coord`** (P33 含, P34 已回退) |
| Tier-2 深挖 | focused 42 ep = 20 unique no-hit + 6 shared verify + 16 success-hit FP; 6 sonnet 并行 |

> **定位声明 (discover-then-freeze 硬纪律 #1)**: 单 condition discover digest, 无对照, **禁 cross-mode 定量比较**。本 digest 与 psom 的对照仅做**定性** mechanism 描述 (视觉盲共享性), 不下定量结论。当前 **5/6 mode discovered** (dom+som+vision+phantom_som+phantom_prompt; phantom_text/reddit/B1/B2 未跑)。

> **方法论 — focused Tier-2**: phantom_prompt no-hit (53) 与 phantom_som no-hit (51) 高度重叠 (shared 33 / pprompt-unique 20)。shared 33 个 psom 已深挖确认视觉盲 → 不重复全挖, 只 (a) unique 20 全挖 + (b) success-hit 16 FP 审计 + (c) shared 抽 6 verify。省 quota + 聚焦新信息 (skill「跨 condition 选择性」)。

## 2. 三分类统计

| 类别 | 占比 | 说明 |
|---|---|---|
| **agent-limit** | ~100% of diagnosed | 无图视觉盲 + cross-mode 通用行为缺陷 + budget 耗尽 |
| **scaffold-bug** | **0** | unique 20 + shared 6 + success 16 全审, 0 框架 bug |
| **benchmark-FP** | **0** | 0 评测误判 |
| **unclear** | 0 | — |

## 3. Tier-1 规则分布 (failed-only per-rule, v5)

```
P31  62  budget耗尽未完成        ← 主导
P17  37  click-back 振荡
P5   36  感知缺失循环
P19  30  url_match 过早 finish
P14  18  URL 自环
P7   18  sCity=州名
P2   17  容器节点误点
P18  17  cheapest 漏价格排序
P20  15  评测目标页从未访问
P23   9 / P10 8 / P25 7 / P33 5 (phantom-img-nav) / P24 4 / P30 4 / 余 ≤3
```

## 4. Tier-2 新发现

### 4.1 unique no-hit (20 ep) — 全 agent-limit
phantom_prompt-unique (psom 上非 no-hit) 失败:
- **多数仍是无图视觉盲** (颜色 0/12/82 · ref-image 60/79/101 · listing 图片内容 83/131/150/151/188/192 · 图中 OCR 128/173/218 · cross-site image 230) — 与 psom 同根, 只是 psom 上走了别的规则或偶中
- **phantom_prompt 特有亚型 (task 173)**: AXTree 暴露图片 `src` (`.../oc-content/uploads/14834/14834.png`) → agent 把**图片文件路径当成"图中显示的网址"**返回 (ref=kaiyo.com)。**psom 不会** ([SOM_MARKS] 只给 numbered marker ID, 不暴露 src href)。样本仅 1, 提议不落 (见 §6)
- **off-page nav (task 79)**: start_url 已加载目标页 (iPage=4), 但 agent search 离开原页。batch A 推测与 pprompt prompt 风格有关, 但属推测 (psom 对照未控)

### 4.2 shared no-hit verify (6 ep) — **6/6 确认与 psom 同视觉盲**
batch F 明确: task 14/41 (gallery row) · 81 (cover) · 136 (ref-image) · 162 (gallery thumbnail) · 210 (listing image) **全部 `same_as_psom_visual_blind=true`**。phantom_prompt **无独立视觉盲死因类别**; 两处 nuance (162 长文字匹配轨迹 / 210 "Lambs Ear" 词干陷阱) 是程度/路径差异非新类别。
→ **P34 phantom-family-wide 强证据**: 视觉盲是 phantom_som + phantom_prompt 共享的结构性 agent-limit, 与 prompt 格式无关, 是 "no annotated image delivered to model" 的直接后果。

### 4.3 success-hit FP 审计 (16 ep) — **16/16 全 FP, 0 causal**
⚠️ **关键 finding**: phantom_prompt 的 success-hit **全部 presence-only** (psom 仅 6 success-hit 且多 causal — 本 digest 不做定量对照, 仅描述 phantom_prompt 自身):

| 规则 | success-FP | 根因 (representation-dependent) |
|---|---|---|
| P5 | 4 | success 后继续探索 / 翻页前页底 scroll / 单次 scroll 误入窗口 |
| P2 | 3 | 同目标下一 click 换 elem 成功 (正常重试) / homepage UI 延迟态 / **item 页内点图片链接失败 (无图)** |
| P14 | 3-4 | 编辑/管理页 type/scroll URL 稳定 (非 stuck) |
| P10 | 3 | 价格数值 vs 列表计数误配 / 年份数字 (语义类型混淆) |
| P17 | 3 | **无图导致 item 页 click-back 验证式往返, finish 最终收敛正确** |
| P18 | 0 | 未 fire (pprompt 倾向 price-filter 非 sort) |

**根因**: phantom_prompt 用完整 AXTree (暴露图片链接 + verbose) → agent 更多 (a) 点图片元素失败 (P2), (b) 进 item 页因 AXTree 无颜色触发验证 click-back (P17)。这是 **representation-dependent behavior 不是 agent failure** —— 现有 cross-mode 规则在无图+verbose-AXTree mode 的 presence-only 暴露。

## 5. 代表 episode

**视觉盲 (shared, 与 psom 同因)**
- task 81 "cheapest book hurricane on cover": 凭书名 "Hurricane Katrina" 文字推断封面有 hurricane, 选错 (id=21162 vs ref 4727)
- task 210 "cheapest lamb": 被 "Lambs Ear plants" 词干欺骗, 选植物非动物 (id=32759 vs ref 81060)
- task 136 ref-image bicycle: ref 图未传, agent 编造 "blue Frozen-theme bicycle", 选错 item

**phantom_prompt 特有 (task 173)**: 把 AXTree 图片 src `localhost:9980/oc-content/uploads/14834/14834.png` 当"图中网址"返回 (ref=kaiyo.com)

**success-fire FP (representation-dependent)**
- task 17 (P17): 无图无法确认 red handlebars, click-back 验证往返, 但最终 finish 收敛到正确 item 79747 = success
- task 48 (P2): homepage category dropdown 二次 click UI 延迟失败, 随即换 elem 成功

## 6. 🔁 Self-evolving — 本轮**纯 discover, ruleset 不动** (仍 `5-domsomvispsom-b1860coord`)

为什么不落码 (discover-then-freeze + skill 陷阱 #2):
- **P34/P35 视觉盲 phantom-family-wide**: batch F 6/6 + batch B 提议 (扩 P16 image-content 到 `{phantom_som, phantom_prompt, phantom_text}`) **定性证据强**, 但落码仍面临 psom 时的 presence-only 风险 (颜色/ref-image/image-content intent 本质 presence detector — agent 文本侥幸答对则 success-fire)。**等 phantom_text 数据齐 + 设计 success-safe 版本一次性落** (no-image-family gate + finish≠ref 类 success-safe 条件)
- **现有规则 (P2/P5/P10/P14/P17) success-fire presence-only**: batch D/E 收窄建议多为"看 success label"而非 0-token signal —— 规则是 presence detector, success 切分在 downstream, 不能内置 success label 检查。真正的 success-safe 收窄 (如 P17 "finish_url ≠ 振荡 item url 才 causal" / P10 "价格/年份数字类型隔离" / P2 "下一 click 换 elem 成功则 FP") 需 0-token 实现 + 全 phantom-family 验证, **避免 psom 时 P14 task 2 那样盲改 cross-mode 规则**
- **task 173 AXTree img-src OCR 混淆**: phantom_prompt 独有, 但样本仅 1 → 提议不落 (signal: finish_answer 含 `localhost.*oc-content/uploads` + intent 要图中文字; 待更多样本)

唯一既有收益: **P33 在 phantom_prompt 验证跨 mode 稳健** (success-fire 0/5, 比 psom 1/17 更干净) → 印证 P33 mode-agnostic 设计正确, AXTree 也暴露 img-href 但 agent 点的少 (5 vs psom 17)。

### Follow-up (freeze 前一次性处理)
1. **P34/P35 no-image-family-wide success-safe 版**: 等 phantom_text 数据 → 设计 gate `mode in {phantom_som,phantom_prompt,phantom_text}` + success-safe (区分"必败视觉任务"vs"文本侥幸成功")
2. **现有规则 success-safe 收窄** (P17/P10/P2/P5/P14): 用 0-token signal (非 success label), 全 phantom-family 验证后落
3. **task 173 img-src 亚型**: 攒样本

## 7. Actionable
- **scaffold-bug → B-number**: 无
- **benchmark-FP → task 排除**: 无
- **cross-mode**: ⚠️ 禁定量直至 6-mode freeze。phantom_prompt = 5th mode discover, findings 已记, ruleset 不动
