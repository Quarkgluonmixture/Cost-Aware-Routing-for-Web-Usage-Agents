# B1 dom classifieds — 失败错因 digest（diag skill）

> **生成方式**: `/diag B1 dom` skill 3-tier pipeline (2026-06-05 run on R17188)。Tier-1 deterministic 全扫 (`diag_pattern_match.py`, 0 token, ruleset `5-domsomvispsom-b1860coord`) → Tier-2 Claude sub-agent 深挖 (**7 no-hit 全覆盖 + 12 success-hit FP 审计 + 10 failed-hit causal verify = 29 ep / 5 agents**, sonnet) → Tier-3 整合 (本文件)。
> **Run**: `B1_dom_classifieds_20260603_103630_477435114_112846_R17188` (manifest-bound authoritative)
> **Condition**: `phase1_dom_router_0` | site classifieds | mode **dom** | model **B1 = Qwen3-VL-4B (local)**
> **ruleset_version**: `5-domsomvispsom-b1860coord`（不变；B1 是**新增 model 维度**, 在现有 v5 ruleset 下扫描。本轮 discover findings → 提议 P34-P37 + 多条 FP-narrowing, 但**因并行 session 也在 diag dom 改 `diag_pattern_match.py`, 本轮一律不落码、不 bump version**, 提议留「Self-evolving」section 待协调。**禁止 cross-mode AND cross-model 定量比较** 直至 freeze + 全量重扫）

> ⚠️ **定位声明（沿用 R9755/R21557 3-AI 审计共识）**：本 digest 是 **internal 诊断记录，NOT paper-grade 结论**。
> - **单 condition + dom-only + 无对照**：单 model×mode×site（B1 dom cls）。"DOM 表征天花板 / routing 论点 / 换表征能救"需 **som/vision/phantom 对照**才成立。**禁止任何 cross-mode/cross-model 定量比较**（B1 vs B0 的对比在本文件仅作**定性 capability-gradient 观察**, 非统计 claim）。
> - **presence ≠ causation**：210 failed 中仅 **7 no-hit + 10 failed-hit causal verify + 12 success-hit FP 审计 = 29 经 sub-agent 逐个证因**；其余 193 failed-hit 是规则命中**未逐个 causal verify**。
> - **per-rule 非互斥**：分布是 per-episode-per-rule，P5∩P31∩P6 大量重叠，勿各行相加。

---


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
| SR | **6.25%** (14/224) |
| failed + hit | 207 |
| **failed NO-hit** | **3** |
| success + hit | 10 |

v8 新规则 failed 侧: {'P45': 96, 'P43': 70, 'P44': 68, 'P46': 2}；success 侧: 无。
（`P43` 在 cls 上大量命中属预期 —— 它标记"intent 需要视觉 + 该 mode 无页面截图"这一**中性组合**，
并非预测失败；§387.10 实测补上截图的增益 ≈0。）

全部 36 个 canonical condition 现处同一版本 → **cross-mode / cross-site 聚合解锁**。

---
## Verdict

| 维度 | 结论 |
|---|---|
| Episodes | 224 (**14 success / 210 failed**) — SR **6.25%** |
| 三分类 (29 深挖) | **agent-limit 100%**（17 failed 深挖 = 17 agent-limit + **0 scaffold-bug + 0 benchmark-FP**）· 12 success-hit = **16/16 rule-fire 全 FP** |
| Deterministic coverage (failed) | **96.7%** (203/210 failed 命中) — 远高于 B0 dom (88%)，因 B1 退化循环 (P5/P31) 极易被正则抓 |
| no-hit failed | **7** (16/106/119/129/208/213/221) — 全 agent-limit, 含 4 个**新模式**(P34-P37 候选) |
| success 命中规则 | 12/14 success fire 规则 (**P6 在 8/14 success 上 fire = 57% FP**; 16 hits **全部 hit_causal=false**) |
| pipeline 健康度 | **parse_valid / tool_call_valid 全 valid** — B1 action 解析层完全正常, 失败 100% 语义/能力/表征层, 非 scaffold |

**B1 dom cls SR = 6.25% (14/224)**，是 B0 dom cls (R21557 17.4% / R9755 14.7%) 的 **~1/3**。这不是 paper-grade cross-model claim（需统计检验），但定性上印证：4B 模型在 dom 盲表征下的失败**远早、远彻底**于 235B。

---

## 🔑 头条发现 — 失败画像相对 B0 发生「翻转」(定性 capability gradient)

| 主导失败信号 | B1 dom (R17188, 4B) | B0 dom (R9755, 235B) | 读法 |
|---|---|---|---|
| **P5 感知缺失循环** | **173 (头号)** | 36 | B1 反复 scroll/hover/type, page 不变 → 死循环 |
| **P31 budget 耗尽** | **131** | 未进 top | B1 走不完 trajectory 就 step 用尽 |
| P6 视觉任务 DOM 必败 | 98 | 96 | 两者相近（dom 视觉盲是共性）|
| P14 URL 自环 | 35 | **109 (B0 头号)** | B0 陷在导航循环；B1 连"陷循环"都做不到 |

**机制差异**：B0 (235B) 的头号失败是 **P14 URL 自环**（有能力反复导航但绕不出来）；B1 (4B) 的头号失败是 **P5 退化循环 + P31 budget 耗尽**（连有效导航都难维持，原地空转到 step 上限）。弱模型不是"换了种方式失败"，而是"更早卡死在更原始的循环里"。

---

## Tier-1 规则分布（failed-only per-rule, 210 failed）

```
P5  感知缺失循环          173  ████████████████████  ← 头号
P31 budget耗尽未完成      131  ███████████████
P6  视觉任务DOM必败        98  ███████████
P19 url_match过早finish    71  ████████
P2  容器节点误点           56  ██████
P16 视觉图像内容DOM必败    51  ██████
P17 click-back振荡         43  █████
P18 cheapest漏价格排序     35  ████
P14 URL自环                35  ████
P13 搜索代替浏览           22  ███
P20 目标页从未访问         21  ██
P10 跨步数值记忆失败       15  ██
P12 从不翻页               12  █
P25 跨站跳过一站            9  █
P30 到达正确item后离开      7  █
P15/P7/P33/P27/P22/P4/P21/P23  ≤6 各
```
**scaffold(is_scaffold) hits: 0** — 规则库仍 agent-limit 偏置（唯一 scaffold 规则 P8 零命中）。

success-fire (FP 源): P6=8 · P17=3 · P16=2 · P33=1 · P10=1 · P18=1（合计 16 hits / 12 success-ep, **全部经审定为 FP**）。

---

## Tier-2 新发现

### A. no-hit 深挖 (7 ep) — 全 agent-limit, 4 个新模式

deterministic 完全盲区的 7 个失败全是 agent-limit（0 scaffold / 0 FP）, parse_valid 全 True。挖出 4 个**现有规则未覆盖**的新模式：

| task | 死因 | 新模式 (候选规则) |
|---|---|---|
| 208 | 应提交 comment 表单, 但 agent 直接 `finish(answer=非空)` 从未 type 进 form | **P35 finish 代替 form 提交**（B1 特有「幻觉完成」: program_html + mutating_action_count=0 + finish）|
| 213 | if/else 条件 task（图含观众→留言；否则→返邮箱）, dom 看不到图 → 走错 branch | **P36 条件分支盲判**（if/else intent + dom + 早 finish）|
| 221 | "how many bowls" — dom 文本不含明确数字, B1 **从文本编造** "10"(ref=6) | **P37 数量幻觉**（how many + dom + 答案为另一数字; 区别于 P22 读图数字）|
| 16/106/129 | dom 盲 → 用关键词搜索**近似替代**看图 → 选错 item（coffee grinder≠mug 等）| **P34 图像内容→关键词替代**（P6 家族变体, 但走完整 trajectory 非早弃）|
| 119 | P6 类视觉弃答, 但 answer "No dollar bill image found" 措辞未被 P6 正则覆盖 | **P6 正则扩展**（`no.*image.*found\|image.*not.*found` 变体）|

> 为什么现有规则漏了这些: P6 抓的是「早弃 + N/A 措辞」, 但 B1 常常**走完整路径**(关键词搜索 / 选错 item / 编造数字), action 序列合法, 正则抓不到。这些是 **dom 盲 + 4B 弱推理的复合产物**, B0 较少出现（B0 更多直接 P14 循环）。

### B. success-hit FP 审计 (12 ep, 16 hits) — 100% FP + 关键发现「B1 dom 并非全盲」

**16 个 success-fire hit 全部 `hit_causal=false`**。最重要的机制发现：

> **🔬 B1 dom 模式下 model 仍接收 reference image 作为多模态输入, 会 OCR 出可搜索实体再走文本路径** —— 不是"看不到图"。

- **P6 image-match 系统性 FP**: task 44/45/48/87/103/153 的 agent 从 reference image OCR 出**书名/型号/人名/类别**（"Hidden Figures" / "HYDE NO. 10000" / "Lionel 1120" / "golf cart" / "Abraham Lincoln" / "matador"）→ 文本搜索 → AXTree title 匹配成功。P6 假设"dom 必然视觉盲"在 reference image 可 OCR 时**不成立**。
- **P6 color 分支 FP**: task 15/50 的颜色词("red")只是 seller-email 任务的**搜索 token**, 答案是 AXTree 结构化字段, 无需像素感知。
- **P16 FP**: task 94 `image=null` 却 fire（"in the image" 短语误触发）。
- **P17 success-振荡 FP**: task 52/183/219 振荡但**终点停在正确 item**（high-cost-but-success, 非 failure-risk）。
- **P10 结构性 FP**: task 219 `_extract_numbers()` 从 finish-action 的 URL answer 抽出 **port 9980** 与价格 20.0 比对 → 误 fire。这是 **diag 规则自身的 bug**。
- **P33 docstring 错**: task 153 在 dom (非 phantom_som) success 上 fire — agent 点 image href → 裸 PNG → `back` 自救。P33 "天然 success-safe" 声明对 dom 不成立。
- **P18 FP**: task 219 用 price-range filter（min=15）替代 i_price 排序也找到合法答案。

### C. failed-hit causal verify (10 ep) — P5 真死因/可路由, P31 复合, P17 仅症状

| 规则 | causal? | 根源 | router 含义 |
|---|---|---|---|
| **P5** (verify 3/3) | ✅ **真死因** | **representation-blind**: agent 反复 scroll/hover/type 因 **dom 不反馈 action 结果** (scroll 不改 AXTree / hover 无 alt-text / 无法确认属性) → 死循环 | **可路由** — SoM/Vision 给反馈后循环大概率消失 |
| **P31** (verify 2/2 独立) | ✅ 真死因但**复合** | task 40 = P5 循环耗尽（可路由）; task 86/207 = **capability ceiling**（hallucinated 登录凭据 user@example.com/password123 + 跨站 working memory 丢失）| **混合** — 不可单独作路由信号 |
| **P19** (verify 1/2) | ⚠️ 半 causal | task 82 真 finish-on-search (causal); **task 55 FP**: agent 从未 finish, budget 耗尽后 runner 在搜索页 post-hoc 评估被误判 | 需加 `agent_finished=true` guard |
| **P18** (verify 1/1) | ✅ 真死因 | task 210 无 sOrder 排序 → 取列表第一条 → 选错 cheapest | 偏 capability (规则跟随), 与表征无关 |
| **P17** (verify 0/3) | ❌ **仅症状** | task 55/117/146 振荡都不是死因: 或锁错 item 耗 budget (P31), 或 dom 视觉盲下的不确定性外化 | 不可作 failure 归因, 应降级 |
| **P6** (verify in B) | 见 B 节 | dom 盲是真天花板**但** reference image 可 OCR 时高 FP | mode-specific (可路由) 但需 OCR-bypass 收窄 |

> **paper 关键证据**: P5/P31 **不是同一死因的两个测量面**。**P5 = dom-representation-blind 强信号 (routable)**; **P31 = 复合信号** (P5 循环 + capability ceiling 如 auth/跨站记忆)。**P31 单独不可作路由依据** —— 这直接关系 router 论点的信号选择。

### 🔗 P31 跨模式收口（vs §317 B1 som — dom 与 som 含义相反）

并行 session 的 B1 **som** digest (§317) 判 **P31 = finish-less artifact** (0/4 causal; success-fire 10/10 presence-only = B1 到达 reference 页但不发 finish, url_match/program_html eval 不需 finish → 仍 pass)。**dom 实测相反**:

| 检查 | B1 dom (R17188) | B1 som (R31705, §317) | 读法 |
|---|---|---|---|
| **success finish-rate** | **14/14 = 100%** | success 含 finish-less arrival | dom 赢都是**真 finish**, 无 arrival artifact |
| overall finish-rate | 41.1% | 50.9% | dom 更 finish-less, 但... |
| finish-less 集中在 | **失败集** (failed finish 37.1%) = 真卡死 | success/near-success = 到达-沉默 | 同指标相反语义 |
| P31 causal verify | 86/207 真死因 (capability ceiling), 40 = P5 下游 | 0/4 (终点表象) | dom P31 是真死, som 是 artifact |

> **机制根因 = 表征**: som 能看标注图 → 常到达正确页 (只是不说停) → finish-less artifact; dom 表征盲 → **真到不了** → 空转死。∴ `trajectory_incomplete` / `trigger_distribution` **同时被 model (§317 教训1) 和 mode (本节) confound** —— 不可裸比, 不可单独作路由信号, cross-mode/cross-model 聚合前必须先剥离 finish-less 行为差异。

---

## 代表 episode（每类 2-3 个 + 证据 step）

**agent-limit — representation-blind 退化循环 (P5, 可路由)**
- **task 5**: steps 19-30 连续 scroll, page_changed=False 全程。thought 反复 "search bar not visible in tree, I will scroll" → dom 下 scroll 不改 AXTree, 永远感知不到搜索框 → 12 步空转耗尽 budget。
- **task 12**: steps 3-10 连续 8 步 hover 同一 element。thought 反复 "image might contain color, hover to reveal alt text" → dom 无图像内容, hover 永远无效 → 最终猜 "Black"(ref="red")。

**agent-limit — capability ceiling (P31, 不可路由)**
- **task 86**: step 2 已读出正确信息 (cheapest Toyota $400), 却离开去登录, steps 5-30 用 **hallucinated 凭据** (user@example.com/password123) 反复登录失败耗尽 budget。
- **task 207**: 跨站 task, step 1 正确读出 "Neon Red & Neon Blue", 20+ 步后 **working memory 丢失** (step 22 "I don't have access to OneStopMarket's color scheme") + 登录循环。

**agent-limit — dom 盲 + 弱推理新模式 (no-hit)**
- **task 208**: 应在 item 16826 提交 comment, 但 agent 离开去搜同类 item, 最终 `finish(answer="Ground Beetle")` **从未提交表单** → 评测查 .comments_list 为空。
- **task 221**: "how many bowls" — dom 描述模糊, B1 **编造** "10"(ref=6), 非读图失败而是文本数字幻觉。
- **task 16**: coffee mug 任务, dom 看不到图, agent 把 "coffee grinder" 当最接近项点进去, 返回错误卖家邮箱。

---

## 🔁 Self-evolving — 提议规则（⚠️ 本轮**不落码**, 待与并行 session 协调后统一 bump version + 全量重扫）

> **为什么 defer**: 并行 session 同时在 diag B1 **som** (§317) 并可能改 `diag_pattern_match.py`。若两 session 各自加规则 + bump `RULESET_VERSION` 会撞版本 + 破坏 discover-then-freeze 的"任一时刻所有 condition 同版本"纪律。本轮产出**规则草案**, 落码留到 **B1 dom+som discover 合并 freeze step** 一次性做。
>
> **⚠️ 与 §317 (som) 规则编号撞车 — 合并计划**: 两 session 各自占用 P34-P36 但含义不同。冻结时按下表 reconcile（§317 先到先得保留 P34-P36, 本 digest 项顺延）:
> | 本 digest 草案 | §317 (som) 草案 | 处置 |
> |---|---|---|
> | **P35 finish 代替 form 提交** (comment task208) | **P35 edit-未提交** (item_edit 5/5) | **合并** → 泛化为一条 "finish-without-mutation (program_html side-effect)" |
> | P34 图像→关键词替代 | P34 早放弃-否定 finish | 撞号, 本项**顺延 P37** |
> | P36 条件分支盲判 | P36 gallery 单步读错 | 撞号, 本项**顺延 P38** |
> | P37 数量幻觉 | — | **顺延 P39** |

### 新规则候选 (4 条, B1 discover 产出)
| ID | 模式 | 0-token signal | success-safe 条件 |
|---|---|---|---|
| **P35** | finish 代替 form 提交 | eval_type=program_html + require_reset + `effective_mutating_action_count=0` + `submit_create_count=0` + 有 finish(非空) | success=true 不触发 |
| **P36** | 条件分支盲判 | intent 含 `if.*depicts\|if.*shows\|unless.*visible` + eval=program_html + dom + steps≤2 + finish@start_url | success=true 不触发 |
| **P37** | 数量幻觉 | `intent.startswith("How many")` + answer.isdigit() + answer≠ref + dom + steps≤3 | answer==ref 不触发 |
| **P34** | 图像内容→关键词替代 | intent 含视觉内容词(animal/bird/...) + dom + 走完整 trajectory + answer≠空 + success=false | success-label gate |

### 现有规则 FP-narrowing (多条, 跨 B0/B1 均受益)
| 规则 | 问题 | 收窄建议 |
|---|---|---|
| **P10** | finish-action URL answer 抽出 port 9980 误 fire | finish + answer 匹配 URL 格式时, output_nums 剔除 localhost port (9980/7770/9999) + query-param id |
| **P19** | budget 耗尽 post-hoc 评估在搜索页被误判 finish-on-search | 加 `agent_finished=true` guard |
| **P17** | success/正确 item 上振荡误报 | finish url item == 最高频访问 item → 降级 `routing_detour` 非 failure-risk |
| **P6** | reference image 可 OCR 时高 FP (image-match) + 颜色词作搜索 token 时 FP (color) | image-match: agent step0-1 thought 提取出可搜索实体且 AXTree 文本匹配 → OCR-bypass 不计 failure; color: eval=string_match 且 ref 不含颜色词 → 跳过 color 分支 |
| **P16** | `image=null` 时误触发 | image=null 时不 fire |
| **P33** | docstring "天然 success-safe" 对 dom 错 | 改 docstring + 加 recovery check (下一步 back 且 finish url 正确 → detour 非 trap) |

---

## Actionable

1. **scaffold-bug → B-number 候选: 0**（episode 层无 scaffold; pipeline 干净, parse/tool_call 全 valid）。
   - 但 **diag 规则层 bug ≥3**: P10 port-FP / P19 post-hoc-eval-FP / P33 docstring — 这些是 `diag_pattern_match.py` 的修复项, 非 runner B-number。defer 落码（并行 session）。
2. **benchmark-FP → task 排除: 0**（深挖 29 ep 未发现语义对判错; B1 多为真失败, 不像 B0 偶有 string_match 过严）。
3. **router 论点证据 (强)**: P5 representation-blind (可路由) vs P31 capability-ceiling (不可路由) 的分层, 是 "换表征能救一部分、不能救全部" 的直接 per-episode 证据。**待 B1 som/vision 对照齐后**, 验证 P5 循环是否在 SoM/Vision 消失 = 干净的 routing 因果链。
4. **P6 解读修正**: B1 dom 的 98 个 P6-failed 不能直接读作 "98 个视觉盲失败" —— B1 会 OCR reference image 走文本路径, 失败更多是 "OCR/关键词搜索后选错 item"(degraded grounding) 而非纯盲。cross-mode 聚合时须标此 caveat。

---

## 与 B0 dom 的定性对照（⚠️ 非统计 claim, 待 freeze 后定量）

| 维度 | B1 dom (4B, R17188) | B0 dom (235B, R9755/R21557) |
|---|---|---|
| SR | 6.25% | 14.7% / 17.4% |
| 头号失败 | P5 退化循环 (173) + P31 budget (131) | P14 URL 自环 (109) |
| deterministic coverage | 96.7% | 88% |
| no-hit failed | 7 (3.3%) | 35 / 22 |
| 失败本质 | 原地空转 / 弱推理编造 / 表征盲循环 | 有能力导航但绕不出循环 |
| scaffold / benchmark-FP | 0 / 0 | 0 / ≥1 (R21557) |

**capability gradient 定性结论**: 弱模型 (B1) 的失败 deterministic 更易抓 (96.7% vs 88%), 因其失败模式更"原始"(退化循环 vs 复杂导航). no-hit 盲区更小 = Tier-2 quota 更省。这本身是 paper "失败画像随 capability 变化" 的一个可量化侧面 (待 cross-model 统一版本后定量)。

---

*Discover-phase 记录 (B1 dom = 新增 model 维度). 下一步: B1 som/vision 跑完后, 验证 P5 representation-blind 循环是否被换表征消解 (router 因果链); 协调并行 session 后统一 land P34-P37 + FP-narrowing → bump RULESET_VERSION → 全量重扫拉齐所有 condition 到同版本, 方可 cross-mode/cross-model 定量比较。*

---

### v11 数字块（`11-intent-text-fallback`，2026-08-03 补）

> 本 digest 正文成稿于更早的 ruleset。v10 落了 **+P49 / P36 carve-out / P14 carve-out**，
> v11 给 **P34/P48 换用 `_finish_intent_text()`**（answer 为空时 fallback 读 `thought`——
> B0 惯于把结论写进 `answer`，B1 留在 `thought`，旧口径因此变成了模型行为检测器）。
> 全部 48 个 canonical condition 已在 v11 下重扫，**cross-mode / cross-model 聚合以本块为准**。

| 字段 | 值 |
|---|---|
| Run | `B1_dom_classifieds_20260603_103630_477435114_112846_R17188` |
| Episodes | 224（success 14 · SR 6.25%） |
| 三子集 | failed+hit 207 · failed-NO-hit 3 · success+hit 10 |
| config_missing | 0 |

| 规则 | 含义 | step 级 | episode 级 |
|---|---|---:|---:|
| `P36` | WALK_FAIL_DEGENERATE | 1002 | 128 |
| `P5` | 感知缺失循环 | 173 | 98 |
| `P6` | 视觉任务 DOM 必然失败 | 98 | 98 |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT | 70 | 70 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 96 | 67 |
| `P16` | 视觉图像内容DOM必败 | 51 | 51 |
| `P31` | budget耗尽未完成 | 50 | 50 |
| `P17` | click-back振荡 | 43 | 43 |
| `P18` | cheapest漏价格排序 | 35 | 35 |
| `P14` | URL 自环 | 35 | 32 |
| `P13` | 搜索代替浏览 | 22 | 22 |
| `P20` | 评测目标页从未访问 | 21 | 21 |
| `P44` | HALLUCINATED_ELEMENT_REF | 68 | 20 |
| `P12` | 从不翻页 | 12 | 12 |
| `P19` | url_match过早搜索页finish | 12 | 12 |
| `P10` | 跨步数值记忆失败 | 12 | 10 |
| `P25` | 跨站任务跳过其中一站 | 9 | 9 |
| `P2` | 容器节点误点 | 56 | 7 |
| `P30` | 到达正确item后离开 | 7 | 7 |
| `P15` | gallery行位置DOM不可定位 | 6 | 6 |
| `P7` | sCity=州名 | 6 | 5 |
| `P33` | 导航至裸图片URL幻觉 | 5 | 5 |
| `P27` | 找不到即放弃 | 4 | 4 |
| `P37` | URL_HALLUCINATION | 3 | 3 |
| `P22` | 图上数字dom不可读 | 2 | 2 |
| `P38` | DOM_URL_AS_IMAGE | 2 | 2 |
| `P35` | MUTATION_MISSING | 2 | 2 |
| `P46` | COMMENT_INTENT_NO_TYPE | 2 | 2 |
| `P21` | dom模式视觉幻觉 | 1 | 1 |
| `P23` | oldest误用价格排序 | 1 | 1 |
| `P4` | 根节点误操作 | 2 | 1 |

> ⚠️ **解读约束**（`docs/analysis/_data_quality_audit.md`）：
> ① 本表是**症状分布，不是死因分布** —— P36/P31 经 10 例跨 benchmark 因果验证均判为 risk-marker；
> ② `P2`/`P4` 依赖 `element_bbox`，在 **vision 上结构性为 0（假 0）**；
> ③ `P36` 在 vision 上只覆盖 `type` 步（click 无 `locator_route_meta`）→ **分母与 dom/som 不同**。
