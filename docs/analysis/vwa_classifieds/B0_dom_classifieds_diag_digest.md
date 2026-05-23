# B0 dom classifieds — 失败错因 digest（diag skill）

> **生成方式**: `/diag` skill 3-tier pipeline (2026-05-23 run on R31194)。Tier-1 deterministic 全扫 (`diag_pattern_match.py`, 0 token, ruleset `1-dom`) → Tier-2 Claude sub-agent 深挖 (23 no-hit 全覆盖 + 12 failed-hit causal verify + 7 success-hit FP 审计 = 42 ep / 7 agents) → Tier-3 整合 (本文件)。
> **Run**: `B0_dom_classifieds_20260523_080127_508387150_37076_R31194` (Gate-3 fresh substrate, per-condition docker restart, manifest-bound authoritative)
> **Condition**: `phase1_dom_router_0` | site classifieds | mode **dom** | model **B0 = Qwen3-VL-235B (proxy)**
> **ruleset_version**: `2-dom` (discover 在 `1-dom`；本 session user fast-track 落码 P19-P23 + 收窄 P6/P14 → bump `1-dom`→`2-dom`，见下「Self-evolving changelog」。cross-mode 聚合前须 verify 全 digest 同版本 — 当前仍**禁止 cross-mode 定量比较**)
> **Supersedes**: R9755 digest (2026-05-21, pre-Gate3 first-completed try-run)。本 run 是**同一 condition 的 fresh Gate-3 重跑** → 构成 R9755 self-evolve 出的 P15-P18 的一次 **fresh-substrate out-of-sample 检验**（见下「跨 run 一致性」）。

> ⚠️ **定位声明（沿用 R9755 3-AI 审计共识，仍适用）**：本 digest 是 **internal 诊断记录，NOT paper-grade 结论**。
> - **单 condition + dom-only + 无对照**：单 model×mode×site（B0 dom cls）。"DOM 表征天花板 / routing 论点 / 换表征能救"需 **som/vision/phantom 对照**才成立（当前 ruleset `1-dom`，6-mode 数据未齐，**禁止任何 cross-mode 定量比较**）。
> - **presence ≠ causation**：190 failed 中仅 **23 no-hit + 12 failed-hit = 35 经 sub-agent 逐个证因**；其余 155 是规则命中未逐个证因。"agent-limit 主导"应读作"35 深挖 100% agent-limit + 155 命中 agent-limit 类规则"。本 run **causal verify 实测 12 个规则命中里 5 个规则名标错**（presence≠causation 在 fresh data 再次确认，见「Tier-2 causal verify」）。
> - **P-rule 仍部分 in-sample**：P15-P18 在 R9755 拟合。本 run 是它们的首次 fresh-substrate 检验，coverage 掉 5pp（93%→88%）= in-sample 虚高的实证，**非真泛化结论**（真泛化需 held-out 不同 condition）。
> - **per-rule 非互斥**：分布是 per-episode-per-rule 命中，P6∩P14 大量重叠，勿各行相加。
>
> **paper failure-analysis 待 6-mode + 多 condition 数据齐后重做，不复用本 digest 数字。**

---

## Verdict

| 维度 | 结论 |
|---|---|
| Episodes | 224 (34 success / **190 failed**) — SR **15.18%** |
| 三分类 | **agent-limit**（35 深挖证因 100% + 155 规则命中 agent-limit 类）· **scaffold-bug 0**（35 子集证因 + P8 全 run 0 命中）· **benchmark-FP 0**（35 子集 finish-vs-reference 全真错）|
| Deterministic coverage (failed) | **87.9%** (167/190 failed 命中) — fresh-substrate, vs R9755 in-sample 93% |
| no-hit failed | **23**（全部 agent-limit；含 13 个 R9755 未暴露的新盲区）|
| success 全部命中规则 | 34/34（**P6 在 30/34 success 上 fire** → 最强误报源）|

**R31194 (fresh Gate-3) 的失败 ~100% 指向真实模型能力局限，pipeline 干净**：35 深挖子集零框架 bug、零评测误判（parse/tool_call 全 valid、finish-vs-reference 全真错）+ scaffold 规则 P8 全 run 0 命中。低 SR (15.18%) 是 dom 表征对 cls 视觉任务的结构性天花板，**不是 bug**——这正是 paper-grade clean run 应有的样子。

---

## 跨 run 一致性（R9755 in-sample → R31194 fresh-substrate）

R31194 是 R9755 P15-P18 规则的首次 fresh-substrate 检验（同 condition，docker 重启全新 site state）：

| 指标 | R9755 (in-sample fit) | R31194 (fresh substrate) | 读法 |
|---|---|---|---|
| SR | 14.7% (33/224) | 15.18% (34/224) | 稳定，fresh substrate 未改变能力天花板 |
| failed-coverage | 93% (P15-P18 拟合后) | **87.9%** | **掉 5pp = in-sample 虚高实证**，规则有一定泛化但非 93% |
| no-hit failed | 13 | 23 | fresh run 暴露更多盲区（10/13 R9755 no-hit 仍 no-hit = 强一致；+13 新盲区）|
| scaffold / FP | 0 / 0 | **0 / 0** | 两 run 一致 → 结论 robust |

**no-hit 重叠**：R9755 的 13 no-hit `[84,97,106,119,124,129,131,162,207,208,210,221,230]` 中 **10 个在 R31194 仍 no-hit**（84/97/106/124/129/131/207/208/210/221）= 规则盲区高度可复现。R31194 新增 13 no-hit `[40,91,100,111,130,156,158,209,211,215,216,219,223]` 暴露 R9755 session 未surface 的模式（见下「Tier-2 新发现」）。

---

## Tier-1 规则分布 (failed-only, episode-level, ruleset 1-dom)

```
P14 URL自环              105  ████████████████████  55.3%  (与 P6 大量重叠)
P6  视觉任务DOM必败       94  ██████████████████    49.5%  ★最强误报源, success 30/34 也 fire
P16 图像内容             47  █████████             24.7%
P17 颜色/属性文本推断     38  ███████               20.0%
P5  感知缺失循环          37  ███████               19.5%
P10 跨步数值记忆失败      20  ████                  10.5%
P7  sCity=州名           19  ████                  10.0%
P18 cheapest漏排序        17  ███                    8.9%
P2  容器节点误点          16  ███                    8.4%
P15 gallery行位置          6  █                      3.2%
P13 搜索代浏览 / P12 不翻页 / P4 根节点  少量
```
**is_scaffold 命中: 0**（唯一 scaffold 规则 P8 全 run 零命中 → 但不等于无 scaffold bug，靠 Tier-2 主动找；本 run 35 深挖子集确认 0 scaffold）。

---

## Tier-2 新发现 (no-hit 盲区 23 + FP 审计)

### 23 no-hit 全部 agent-limit (0 scaffold / 0 FP)，主因子类型：

1. **纯视觉 URL 导航**（task 124/130/131/158 etc）— intent "navigate to item whose image <草地/日落/篮子/项链>" + url_match，DOM 无任何像素 → 结构性不可解。
2. **图像唯一信息**（task 100 图上车号 / 209 批量定价仅在图 / 221 数量"6"仅在图）— 答案 fact 只存在于图像，DOM 文本不含。
3. **跨站视觉过滤**（task 207 Nintendo Switch 配色匹配）— agent 正确执行价格排序但**完全忽略需视觉判断的过滤条件** → 直接支撑 P-SoM signal-AUROC 优势假设。
4. **视觉内容写评论**（task 208 识别昆虫"moth/butterfly"猜成"beetle"）。

### 新 deterministic 候选（Tier-1 当前未覆盖，self-evolving 提议见下）：
- **task 210 premature-finish-on-search-page**：url_match EXACT 但 finish 时 `eval_source_agent_url` 含 `page=search`（停在列表页未进 item 详情页）。注：R9755 把 210 判为"lamb→plant 关键词歧义"，fresh run sub-agent 找到更干净的状态信号 = agent 根本没进详情页。
- **task 223 wrong-target-page**：program_html 的 `eval_target_url` (id=12085) ≠ finish obs_url (id=13215)，且 target URL 从未在任何 step 出现 → agent 在错误 listing 上操作。
- **task 156 oldest-sort-mismatch**：intent "oldest" + step URL 出现 `sOrder=i_price`（用价格排序代替日期排序，因 UI 无 date-sort 选项）。
- **task 100/209/221 image-content/quantity-confusion**：intent 要图像内数字/数量 + finish answer 是价格 OR 含 "does not specify"/"cannot determine" + reference 为纯数字。
- **task 215 WANTED-vs-sale**：finish 前进入的 item title 含 `\bWANTED\b`（求购广告误认为 for-sale）。
- **DOM-mode 视觉幻觉**（task 1/3/91/219 etc）：obs_mode==dom 但 finish thought/answer 含 "as per the image"/"appears to be"/"image shows" → agent **声称看到了它在 dom 模式看不到的图**。强 paper finding（hallucinated visual grounding）。

### success-hit FP 审计 (7 个全成功但 fire 规则)：5 误报 / 2 真风险

- **P6 严重 over-fire**：5 次 fire 中 4 次纯误报（task 15/52/103/137）。根因：P6 触发条件 = `image != null`，但 **B0 多模态会把 task reference image 当"可读文字来源"OCR 出来走文本搜索**（task 137 从封面 OCR "Bastien Piano Basics" 直接搜到），绕过视觉需求 → "task 带图" ≠ "需视觉比对页面截图"。
- **P14 误报**（task 25）：navigate-then-finish（同 URL 停留 2 步）被当自环；真自环需 ≥3 重复 + ≥1 显式 back。
- 真风险 2 个：task 75 (P2+P5 容器点击失败+parse 错误 3+ 步停滞)、task 181 (P6+P14，pharaoh 主题混淆致 item ×4 访问 + ×3 back 真震荡)。

### Tier-2 causal verify (12 failed-hit)：7 真死因 / **5 规则名标错**（presence≠causation）

| task | Tier-1 标 | 真死因 | 类别 |
|---|---|---|---|
| 0/1 | P6 | ✓ 真死因（颜色/视觉属性 DOM 不可判）| agent-limit |
| 60/61 | P16 | ✓ 真死因（参考图理解对，但商品类目区分需缩略图：RV 误判为驾驶 game）| agent-limit |
| 7 | P17 | ✓ 真死因（颜色无 filter widget，从机型名推断颜色出错）| agent-limit |
| 5 | P14 | ✓ 真死因（30 步卡同一 URL，0 mutating action）| agent-limit |
| 11 | P5 | ✓ 真死因（蓝色 bike，item ×4 振荡）| agent-limit |
| **4** | P14 | ✗ 表层；真因 = **选错 listing**（Toyota 86 而非 white car）后卡 edit 页 | agent-limit |
| **8** | P5 | ✗ 表层；真因 = **item_add 表单交互失败**（11× no_progress 找不到 price 字段），感知其实成功 | agent-limit |
| **2** | P10 | ✗ 表层；真因 = **搜索词漂移**（red gem→red gemstone 改变结果集）| agent-limit |
| **3** | P10 | ✗ 表层；真因 = **颜色幻觉**（"not black as per the image" 在 dom 模式幻觉）| agent-limit |
| **9** | P17 | ✗ 表层；真因 = **数值漂移**（算出 $785 但表单填 $790）= P10 模式 | agent-limit |

→ **P16/P17 语义确认**：P16 = 参考图场景理解对但需商品缩略图区分类目；P17 = 颜色/规格属性无 DOM filter → 文本推断不可靠。

---

## Self-evolving changelog（已实现 `1-dom` → `2-dom`，2026-05-23 user fast-track）

> user 批准在 6-mode freeze 前 fast-track 落码（偏离默认 discover-then-freeze 时序）。已 bump `RULESET_VERSION="2-dom"`；R31194 重扫验证全部通过（每条新规则 **0 success-FP**、命中预期 Tier-2 task）。⚠️ 仍 dom-only → **cross-mode 定量比较仍禁止**；6-mode 数据齐时这些规则连同 som/vision/phantom discover 一并进 freeze 全量重扫。

| 新规则 | 0-token signal | 命中(全 failed) | success-FP | 验证 |
|---|---|---|---|---|
| **P19 url_match过早搜索页finish** | eval_types⊇url_match + finish obs_url 含 `page=search` + ref_url 非 search | **26** | **0** | task 210 ✓ |
| **P20 评测目标页从未访问** | eval_types⊇program_html + 任一 http target item id 全程未在 obs_url 出现 | **19** | **0** | task 223 ✓ |
| **P21 dom模式视觉幻觉** | mode=dom + **has_image=False** + finish 声称看到 listing/photographic 图像内容（排除 ref-image 措辞）| **9** | **0** | task 91/121/146/160... ✓ |
| **P22 图上数字/数量dom不可得** | string_match + ref 纯数字 + (intent 读图上数字 OR how-many+agent放弃) + answer 缺 ref 数 | **3** | **0** | task 100/221 ✓ |
| **P23 oldest误用价格排序** | intent 含 oldest/earliest + step URL 含 `sOrder=i_price` | **8** | **0** | task 156 ✓ |

**P-rule 收窄（降 FP，已实现）**：
- **P6**：has_image branch 从 blanket `image!=null`（success 30/34 fire = 88% 误报）收窄为 `has_image AND intent 含视觉匹配语 (selfie/this image/exact item/taken on...)`。**success-fire 30→22**（去掉纯 OCR-able ref-image 任务；救回 task 47/101 真视觉 TP）。残留 22 = has_image-visual-match 13（risk-marker 本质，agent 偶尔仍成功）+ **has_color 9（既有问题，本次未碰，留后续）**。
- **P14**：连续同 URL 阈值 **3→4**（outcome-independent；3 步太短无法判"卡死"，navigate-then-finish 长得一样）。**success-fire 13→9，failed-fire 105→71**；真卡死 loop（task 5 = 30 步）不受影响。

**关键洞察 — P21 的 has_image gate**：无 reference image 时 agent 说"listing image shows X"必指它在 dom 看不到的页面内容 = 真幻觉；有 ref image 时"image"有歧义（可能合法指 ref image，= P6 旧错）。gate 同时消 2 FP（task 62/63 echo intent）+ 保 9 TP。**P21 是本轮最强 paper finding**：dom-mode agent 不只是 fail 视觉任务，而是 **confabulate 视觉 grounding**（"image taken inside a garage" 等）。

**P-rule → router feature 连接**：P19/P21/P22 这类 0-token signal 本身就是 **learned router 的候选特征**（"此 task 需视觉 → route 到 som/vision" vs "此 task 行为缺陷 → 留 dom + retry"）。详见下「后续行动」。

---

## 后续行动

- **无 scaffold-bug** → 无需出 B-number（R31194 fresh Gate-3 框架层干净，与 R9755 一致）。
- **无 benchmark-FP** → 无需 task 排除（35 深挖子集 finish-vs-reference 全真错；N/A task 已 task-load 排除）。
- **paper router 设计输入**：P14 真自环 (task 5)、P5/P2 振荡 (task 11/75)、P10 数值漂移 (task 2/9)、搜索词漂移 (task 2) 是 **cross-mode agent 行为缺陷，换表征救不了 → 需 retry/memory 模块**；纯视觉盲区 (task 124/130/131/207...) 是 **表征失败 → 可 route 到 som/vision**。⚠️ 此分层是 dom-only provisional，须 6-mode 对照确证。
- **discover-then-freeze 下一步**：som/vision/phantom 4 mode 跑完 → 各自 discover → 合并 P19-P23 候选 + 收窄 P6/P14 → bump RULESET_VERSION → 全量重扫所有 condition → 才开 cross-mode 定量比较。
