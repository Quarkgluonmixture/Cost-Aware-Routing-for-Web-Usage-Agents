# B0 som classifieds — 失败错因 digest（diag skill）

> **生成方式**: `/diag` skill 3-tier pipeline (2026-05-26 run on R5313)。Tier-1 deterministic 全扫 (`diag_pattern_match.py`, 0 token, ruleset `4-domsomvis-b1860coord`) → Tier-2 Claude sub-agent 深挖 (**24 no-hit 抽样 (cap 25/59) + 6 success-hit FP 审计 + 5 failed-hit causal verify = 35 ep / 6 agents**, sonnet) → Tier-3 整合 (本文件)。
> **Run**: `B0_som_classifieds_20260526_041601_863239369_602235_R5313` (AMENDMENT_07 sequential-id SoM fresh re-launch, per-condition docker restart, manifest-bound authoritative; condition_summary finalize 2026-05-26 13:17)
> **Condition**: `phase1_som_router_0` | site classifieds | mode **som** | model **B0 = Qwen3-VL-235B (proxy)**
> **ruleset_version**: `4-domsomvis-b1860coord`（不变；本 run 是 SoM-family **post-AMENDMENT_07 sequential-id** 第一个 fresh run — SoM nodeId churn 消除验证点。**禁止 cross-mode 定量比较** 直至 6-mode 数据齐 freeze + 全量重扫）
> **Supersedes**: R9725 som digest (2026-05-24, archived per AMENDMENT_07 在 `_archive_amend07_seqid_R9725_som`)。R9725 用 native nodeId churning, R5313 用 deterministic sequential-id — 二者是 SoM-fix 前后的 **同 condition fresh-vs-fresh** 对比 (见下「跨 run 一致性 + AMENDMENT_07 验证」)。

> ⚠️ **定位声明（沿用 R9755/R31194/R9725 3-AI 审计共识，仍适用）**：本 digest 是 **internal 诊断记录，NOT paper-grade 结论**。
> - **单 condition + som-only + 无对照**：单 model×mode×site（B0 som cls）。"som 救视觉 / routing 论点 / 换表征能救"需 dom/vision/phantom **同版本 ruleset** 对照才成立（6-mode 数据未齐：dom R21557 + som R5313 已落地，vision/P-text/P-prompt/P-SoM 待跑）。**禁止任何 cross-mode 定量比较**。
> - **presence ≠ causation**：163 failed 中仅 **24 no-hit 抽样 + 5 failed-hit causal verify + 6 success-hit FP 审计 = 35 经 sub-agent 逐个证因**；其余 128 failed-hit 是规则命中**未逐个 causal verify**。
> - **failed_NO_HIT = 59 (cap 抽 25 深挖)**: deterministic 盲区是 R21557 dom (22) 的 **2.7×**，主要因 P6/P15/P16 mode-gate 不 fire som，som-specific 失败模式现有 ruleset 未覆盖。Tier-2 揭示了 4 大类 SoM-specific 机制（见关键发现 1-4）— 这是本 digest 最 paper-relevant 部分。
> - **P10 在 6/6 success FP audit 全 non-causal**: P10 在 som 上 7 success-fire 全 false-positive (URL 端口 / element_id / DATE 残留)。
> - **per-rule 非互斥**：分布是 per-episode-per-rule。
>
> **paper failure-analysis 待 6-mode + 多 condition + 同版本 ruleset 数据齐后 cross-mode aggregator 重做，不复用本 digest 数字。**

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
| SR | **27.23%** (61/224) |
| failed + hit | 114 |
| **failed NO-hit** | **49** |
| success + hit | 10 |

v8 新规则 failed 侧: {'P45': 19, 'P44': 1}；success 侧: 无。
（`P43` 在 cls 上大量命中属预期 —— 它标记"intent 需要视觉 + 该 mode 无页面截图"这一**中性组合**，
并非预测失败；§387.10 实测补上截图的增益 ≈0。）

全部 36 个 canonical condition 现处同一版本 → **cross-mode / cross-site 聚合解锁**。

---
## Verdict

| 维度 | 结论 |
|---|---|
| Episodes | 224 (**61 success / 163 failed**) — SR **27.2%** |
| 三分类 | **agent-limit 主导**（35 深挖 = 33 agent-limit + 1 scaffold-bug + 1 边缘）· **scaffold-bug ≥1**（T36 cross-site cls+shopping，shopping 站 7770 不可达 30 步 about:blank — 见关键发现 6）· **benchmark-FP 0** (本批 sub-agent 深挖未发现新 FP，但 T216 dom 端的 cross-run FP 在 R5313 也复现 finish item=82390) |
| Deterministic coverage (failed) | **63.8%** (104/163 failed 命中) — 显著低于 R21557 dom 88.1%，差距源于规则库 dom-biased (P6/P15/P16 mode-gate) |
| no-hit failed | **59** (R21557 dom 的 2.7×) — 抽 25 深挖 (P-rule discover 优先) |
| success 命中规则 | 12/61（**P10 在 7/61 success 上 fire = 11.5%** → 全 non-causal, P10 在 som 上是高 FP source）|

**R5313 som SR 27.2% (vs R9725 30.4%, R21557 dom 17.4%)** — sequential-id 重跑后 som 上小幅下降 3.2pp (R9725 → R5313)，仍 dom < som < (typical vision/phantom)。Sequential-id 消除了 element-ID churn (R9725 → R5313 字节一致输入 + 同 task) → 残留方差 = B0 MoE 与 fuzzy judge (符合 §295 元素-ID 已 byte-identical 实证)。pipeline 干净：35 深挖子集 parse/tool_call 全 valid。

---

## 跨 run 一致性 + AMENDMENT_07 验证（R9725 native nodeId → R5313 sequential nodeId）

R5313 是 SoM-family AMENDMENT_07 修后第一个 fresh run，验证 element-ID churn 消除 + run-to-run 稳定性：

| 指标 | R9725 (pre-amend07, native nodeId) | R5313 (post-amend07, sequential) | 读法 |
|---|---|---|---|
| SR | 30.4% (68/224) | 27.2% (61/224) | -3.2pp; 仍在 som-cls 典型 floor 带 |
| failed-coverage | 74.4% (v3-domsom) → 重扫 81.0% (v4) | 63.8% | **下降 17pp** = R5313 暴露更多 R9725 未现的 som-specific 模式 (cap 25 抽 vs R9725 全 44) |
| no-hit failed | 44 (cap 35 深挖) → 59 (v4 重扫降) | **59** | 数字稳定; sequential 不增不减 no-hit count |
| element-ID churn | byte-divergent 同 task 同输入 (§282) | byte-identical (§295) | ✅ AMENDMENT_07 修复确认 |

**AMENDMENT_07 验证结论**: SoM nodeId 改 deterministic sequential 后, 同 task 同 obs 字节一致 — R5313 与 R9725 SR 差 3.2pp 完全来自 **B0 MoE / fuzzy judge** 残留方差 (符合 §295 + AMENDMENT_06 non-gating sensitivity 预期, 不是 ELement-ID 漏修 artifact)。

---

## Tier-1 规则分布 (failed-only & success-fire, ruleset 4-domsomvis-b1860coord)

```
rule  failed  success-fire  notes
P2      1     0            (P2 在 som 上几乎不 fire, 容器节点 dom-specific)
P4      5     3            
P5     26     1            perception-loop (som 上仍主导)
P7     21     0            sCity=州名 (mode-independent)
P10     9     7            ★ 7 success-fire 全 non-causal (URL 端口 / element_id / DATE 残留)
P12     4     0
P14    19     1            v3 修后 success-fire 仅 1 ✅
P17    26     1            click-back oscillation
P18    11     0
P19    28     0            url_match 过早 finish (CLEAN signal)
P20    13     0
P23     8     1            
P24     5     0
P25    11     1            
P27     9     0            abandonment
P28     3     0            bench-FP 货币 tokenize
P29     1     0            bench-FP 语义 yes/no
P30     1     0
P31    58     0            ★ budget-incomplete (high vol, CLEAN)
```

主导规则: **P31=58** (budget 耗尽 — 与 dom 同水平) · **P19=28** (url_match 过早 finish) · **P5=26** = **P17=26** (感知 vs click-back) · **P14=19** · **P20=13** · **P18=11** = **P25=11**。

**vs R21557 dom 对比 (NOT cross-mode 比较, 仅描述性差异)**:
- P6 / P15 / P16 mode-gate 不 fire som → som 上规则覆盖少 17pp 是结构性 (规则库 dom-biased)
- P2=68 dom → 1 som = 容器节点误点是 dom 上 mode-specific 失败
- P31 / P19 / P5 / P17 / P14 / P10 在两 mode 均高 = cross-mode 通用失败 ([[diag SKILL "通用 vs mode-specific 分层"]] 实证, 但**禁止据此 cross-mode 定量推论** 直至 6-mode 数据齐)

---

## Tier-2 新发现 (sub-agent 深挖 24 no-hit + 6 success FP audit + 5 failed verify)

### 关键发现 1 — SoM-specific (A): Viewport-bound SoM marks 导致 form submit button 看不见

**核心机制**: SoM 每步 [SOM_MARKS] 与标注图**仅含当前 viewport 内可见元素 (17-21 marks)**, item_edit / publish form 的 Save/Update/Publish 按钮位于视口下方, **从未出现在任何 mark 中**。Agent 填完 description + price 后直接 finish, 误以为提交成功。

**实证**:
- **T4** (update price): SoM step_8 DOM 24 行截止于 Description, 无 Update 按钮; **DOM mode 同 task success** (step_8 click `elem=7703` Update). `effective_mutating_action_count=0`, eval_source_url 停在 `item_edit&id=84144`.
- **T75 / T76** (update price/desc, 同模式): 同样 mark_count 17-21, Save 按钮永不可见; DOM mode 同 task success.
- **T8** (publish new listing): 同模式 + new_tab 触发 about_blank_recovery 丢草稿; 二次重填后点 SoM mark 17 (实为 Video gaming 类目导航), submit_create_count=0.

**含义**: SoM 在 long form (item_edit / publish) 任务上有结构性劣势, DOM 即使 `current_viewport_only=True` 也因 in_viewport_ratio bug (`processors.py:218`) 保留全 form 文本 + 较高 element_id (~7703) → submit button 仍可达。**这是 paper §3-§4 routing 论点的反直觉证据点**: "SoM 救视觉" 不是单方向, long form 任务 SoM 输 DOM。

**P-rule 草案 P-SOM-EDIT-NO-SAVE** (高价值):
> mode=som + action_type=finish + url 含 `action=item_edit` + trajectory 无任何成功的 click 让 URL 跳出 `item_edit` 到 `page=item` → flag som-form-submit-miss. 0-token, 天然 success-safe.

### 关键发现 2 — SoM-specific (B): Scroll-after-rerender 丢失 price/title context

**核心机制**: 当 agent scroll 到 item 详情页底部的 comment textarea 时, SoM 重渲染**物理丢弃顶部价格 span** ([SOM_MARKS] viewport-bound), agent 失去 price anchor → 计算错误的 "offer $N less" 数字。DOM 即使 `current_viewport_only=True`, 价格 span 因 `in_viewport_ratio` bug 仍保留 → 不丢 context。

**实证**:
- **T31** (offer $10 less than asking): step_6 thought "asking price is $150" (实际 $260); step_7 改为泛型 "no specific number"; 提交 comment 不含 `$\d+`. eval `must_include=['$250']` fail.
- **T32** (offer $10 less, GT Avalanche): step_9 thought "$395 → offer $385"; **step_16 thought "asking is $300, offer $290"** (context flip!); 提交 $290 ≠ $385. 推测受搜索 filter `sPriceMin=300` 锚点偏差 + scroll 丢 price.

**含义**: SoM 在 read-then-write-across-scroll 类任务 (commenting / offering) 上有特异性弱点。

**P-rule 草案 P-SOM-PRICE-LOST**:
> mode=som + intent 含 `offer.*less|offer.*lower` + eval program_html must_include 含 `\$\d+` + typed comment text 不含 `\$\d+` 模式 → flag som-price-context-lost-on-scroll. 0-token summary-level.

### 关键发现 3 — SoM-specific (C): 标注框 + JPEG 压缩颜色感知偏差

**核心机制**: SoM 标注框 (黄色 outline + 数字标签) 覆盖在原 thumbnail 上 + JPEG 编码压缩 → 红色 thumbnail 在标注图中可能偏橙色。Agent 视觉判颜色 vs DOM 文本明确写的颜色矛盾时, 倾向相信视觉判断。

**实证 T56** (cheapest snowblower **not** red):
- DOM 文本明确: `[2904] StaticText: 'Snow blowers...Toro Snow Shovel/Thrower/Blower(Red/Black)'`
- Agent step_4 thought: **"The image shows it is orange, not red, so it meets the criteria"** → 通过过滤
- 实际 item id=17599 是 Red/Black, 不符合 "not red" 约束 → fail

**含义**: 这是 **SoM 盲区 2.7×** 的一个清晰 mechanism path — DOM mode 不看图所以不会出错, SoM 看图但颜色判断不稳定 + 不 cross-reference DOM 文本。**最强 SoM-specific deterministic signal**。

**P-rule 草案 P-SOM-COLOR-MISMATCH**:
> mode=som + intent 含颜色排除约束 (`not\s+(?:red|blue|...)`) + agent finish 时 thought 声称颜色 X + DOM 文本含与 X 矛盾的颜色描述 → flag som-color-misjudge-vs-dom. 需要 step DOM artifact 解析, 中等成本.

### 关键发现 4 — SoM-specific (D): Gallery row 二维位置歧义

**实证 T14 + T41** ("second row painting" gallery 任务): SoM 标注图每个缩略图都有编号框但**无 row/col 语义**; [SOM_MARKS] 文本是线性序列, 无二维位置信息。Agent 把 row 1 col 2 误读为 second row, 或读取 scroll 后的新行 prices 误认为 row 2 paintings。

**P-rule 草案 P-SOM-GALLERY-ROW** (中等):
> mode=som + intent 含 `(?:second|third|N-th)\s+row` + eval=string_match + success=False → flag som-gallery-row-disambiguation. 

### 关键发现 5 — 6/6 success-hit P10 全 non-causal (P10 在 som 上是高 FP source)

T44/T46/T48/T87/T126/T152 经 sub-agent 审计: P10 命中**全部 false-positive**, 三类 FP source:
- **URL 端口数字渗漏** (T44/T46/T87/T126): finish answer 含 `localhost:9980/...item_id=XXXXX` → 端口 9980 / item_id 数字被 P10 当作 "跨步数值需记忆"
- **SoM element_id 数字渗漏** (T152): thought 中 `element_id=12` 是 UI 标签, output 中 "24.2MP" 是产品名规格 → P10 误配
- **DATE_CONTEXT_RE 月/日残留** (T44/T46/T48/T87/T126 通用): `2023/11/16` 中 `2023` 正确剥除但 `[11, 16]` 残留 + output 任何 >10 数字触发

**Carveout 提议** (P10 in `ALL_RULES`, 待 freeze 后落码):
- url_match task + output 含 URL → 过滤 URL port + item_id 数字
- som mode + thought 含 `element_id=N` 模式 → N 不计入 output_nums
- DATE_CONTEXT_RE 阈值 `>10` 提到 `>30` 或限定 3 位数+

### 关键发现 6 — 1 scaffold-bug: cross-site task scaffold issue

**T36** (cross-site cls+shopping): agent step_5 在 OSClass cls 成功找到目标 (Luigi's Mansion 3 id=35037, $50), 但 step_6-29 (24 步) 全部 `goto/new_tab` shopping site `localhost:7770` 失败, **全程 about_blank**。`page_changed=False`, P19 命中 (url_match 在搜索页 finish), 但 P19 是结果非原因。

**真因**: cross-site 任务在 paper-grade fire 期间, **cls 与 shopping 同 A100 host docker bridge 共享**, 同时只能跑一条 site chain (CLAUDE.md hard rule #3 "paper-grade fire 同一物理 host 同时只能跑一条 site chain", §0 ④ 提及 B-1581 cross-site contention)。R5313 fire 期间另一 cls chain 在跑 → shopping container 服务可能未启或被压。

**含义**: paper-grade fire **不应包含 cross-site cls+shopping 任务** (与 cls-only chain 冲突), 或必须 disclose 这类任务在 single-site fire 下必 fail。**B-number candidate** (类似 B-1581 cross-site contention).

### 关键发现 7 — failed-hit P-rule causal verify: 1/5 causal

- **T9 (P31 budget)** = real cause ✅ (form 填到一半 30 步用完, trajectory_incomplete=True)
- **T162** (P17 + P19 命中) → not causal; 真因 = agent 反复进同一 item 但无法判断 "tennis ball" 视觉特征 → P17 振荡是症状非起因, P19 是结果
- **T36** (P19 命中) → not causal; 真因 = **scaffold-bug** cross-site shopping 不可达 (见关键发现 6)
- **T0** (P5 命中) → P5 命中存疑 (所有 scroll page_changed=True 不符 P5 定义); 真因 = agent 在搜索结果列表上 finish 没进 item 详情
- **T50** (P14 命中) → P14 命中边缘 (4 步 stuck-loop vs 阈值); 真因 = agent 读取了错误 seller email

**结论**: P-rule hit 在 R5313 som 上 4/5 not causal, 与 R21557 dom 3/3 not causal 一致 = **presence-vs-causation gap 是 ruleset 通病**, 不限 mode。

---

## Self-evolving — 提议新规则 (待 6-mode freeze 后批量落码)

> ⚠️ 同 R21557 dom digest: discover-then-freeze 协议, 现 2/6 mode 完成 (dom + som), 4 mode 未跑。新规则草案先记录不落码。

som-specific 草案 (mode == "som" gate):

- **P-SOM-EDIT-NO-SAVE** (关键发现 1): mode=som + finish.url 含 `action=item_edit` + trajectory 无 click 让 URL 跳出 item_edit → som-form-submit-miss. 覆盖 T4/T75/T76. **0-token**, 天然 success-safe.
- **P-SOM-PUBLISH-NO-SUBMIT** (关键发现 1 sibling): mode=som + intent 含 `make\s+a\s+post|create\s+a\s+listing` + submit_create_count=0 + about_blank_recovery_count ≥1 → som-publish-form-disrupted. 覆盖 T8.
- **P-SOM-PRICE-LOST** (关键发现 2): mode=som + intent 含 `offer\s+\$?\d*\s*(?:less|lower)` + must_include `\$\d+` + typed comment 不含 `\$\d+` → som-price-context-scrolled-out. 覆盖 T31/T32.
- **P-SOM-COLOR-MISMATCH** (关键发现 3, 最强 SoM-specific): mode=som + intent 含 `not\s+(?:red|blue|...)` + DOM 文本含 X + agent thought 答非 X → som-color-misjudge. 覆盖 T56.
- **P-SOM-GALLERY-ROW** (关键发现 4): mode=som + intent 含 `(?:second|third|N-th)\s+row` + eval=string_match + success=False → som-gallery-row-disambiguation. 覆盖 T14/T41.
- **P-SOM-IMGCONSTRAINT** (T60 image-constraint-self-cancel): mode=som + reference image + agent thought 含 `unrelated\s+to|irrelevant` + 后 ≤2 步 finish → som-image-constraint-skipped.
- **P-SOM-ANIMAL** (T59): mode=som + has_reference_image + animal_in_intent + url_match + reference_id NOT in trajectory + small search results ≤5 → som-animal-misidentified.

cross-mode 草案 (mode-independent):

- **P-CROSS-SITE-UNREACHABLE** (关键发现 6, scaffold-bug): cross-site intent (cls + shopping/reddit) + ≥10 步 `about_blank` + goto/new_tab success=False → scaffold-bug cross-site-unreachable. 覆盖 T36. **B-number candidate**.

P10 carveout 提议 (modify existing rule, bump ruleset):
- url_match task + output 含 URL → 过滤 URL port (9980) + item_id 数字
- som mode + thought 含 `element_id=N` 模式 → N 不计入 output_nums
- DATE_CONTEXT_RE 阈值 `>10` 提到 `>30` (R21557 dom + R5313 som 共证)

**预估覆盖**: 上面 7 条 som-specific 规则可覆盖 24 抽样 no-hit 中 ~14 个 (T2/T3/T12 keyword-search 找不到目标无 deterministic signal; T20/T21/T34/T40 视觉识别错无 stable signal; T66/T67/T78 同; T47/T51 image-similarity 无 stable signal)。failed-coverage 63.8% → ~75-80%。**注意**: 这只是抽样 25/59 的延伸, 全 59 no-hit 还有 ~34 未深挖 → 实际 coverage 提升估算偏乐观。

---

## 代表 episodes

**SoM-specific 失败 4 大类代表**:
- **T75** (类 A viewport-bound form submit miss): 填 price 后 finish 但 URL 停 item_edit; DOM mode 同 task success.
- **T31** (类 B scroll-after-rerender price lost): "offer $10 less" 但 comment 不含 `$\d+`.
- **T56** (类 C color JPEG 标注偏差): DOM 文本 "Red/Black" agent 看图判 "orange" 通过 not-red 过滤.
- **T14** (类 D gallery row 二维歧义): "second row painting" 选了 row 1 col 2.

**scaffold-bug 1 个**:
- **T36** — cross-site cls+shopping, shopping 7770 不可达 30 步 about_blank. **B-number candidate**.

**budget-incomplete CLEAN signal**:
- **T9** (causal P31 confirmed): form 填到 category+title 时 30 步用完, trajectory_incomplete=True. **唯一 5/5 causal**, P31 是 budget signal CLEAN.

---

## paper-grade 含义 (待 6-mode 完成后再 finalize)

1. **SoM 盲区 2.7× 是真实结构性, 不是数据 artifact**: failed_NO_HIT 59 (som) vs 22 (dom) 来自 (a) P6/P15/P16 mode-gate dom-biased = 规则库覆盖原因 (~50% of gap), (b) som-specific 失败模式 (A-D 4 类) 现有 ruleset 未覆盖 = paper-findable 新机制 (~50% of gap)。**paper §3-§4 routing 论点反直觉证据**: "SoM 救视觉" 不是单方向, long form / commenting / gallery row 任务 SoM 输 DOM.
2. **AMENDMENT_07 SoM sequential-id 修复 validated**: R9725 (native nodeId churning) → R5313 (sequential byte-identical) SR 30.4% → 27.2% ±3.2pp 来自 B0 MoE / fuzzy judge 残留方差 (§295 一致), 不是 element-ID artifact. **AMENDMENT_06 non-gating sensitivity 现覆盖 MoE 残留**.
3. **scaffold-bug T36 cross-site fire 限制**: paper-grade fire 同 host 跑 cls-only chain 时 cross-site task 必 fail. paper §8 disclosure list + B-number candidate.
4. **P10 在 som 上高 FP-rate 7/12 success-fire = 58% non-causal**: P10 needs URL/element_id/date-residual 三类 carveout, R5313 + R21557 共证.

---

## Cross-link

- 实验笔记 §294-§296 (AMENDMENT_07 + sequential-id fix + run-to-run sensitivity)
- master_bug_catalog (T180 dom scaffold-bug + T36 som cross-site scaffold-bug 待加 B-number)
- `next_steps.md` §0 ④ — Phase 1a fire 跑中, 等 6 mode 完成 → discover-then-freeze 全量重扫
- R21557 dom digest (B0 dom cls, **同步刷新**)
- R9725 archived som digest (旧版本, supersedes by 本文件)
- diag SKILL.md "跨 condition / cross-mode 工作协议" (discover-then-freeze 协议 + 通用 vs mode-specific 分层)

---

### v11 数字块（`11-intent-text-fallback`，2026-08-03 补）

> 本 digest 正文成稿于更早的 ruleset。v10 落了 **+P49 / P36 carve-out / P14 carve-out**，
> v11 给 **P34/P48 换用 `_finish_intent_text()`**（answer 为空时 fallback 读 `thought`——
> B0 惯于把结论写进 `answer`，B1 留在 `thought`，旧口径因此变成了模型行为检测器）。
> 全部 48 个 canonical condition 已在 v11 下重扫，**cross-mode / cross-model 聚合以本块为准**。

| 字段 | 值 |
|---|---|
| Run | `B0_som_classifieds_20260526_041601_863239369_602235_R5313` |
| Episodes | 224（success 61 · SR 27.23%） |
| 三子集 | failed+hit 106 · failed-NO-hit 57 · success+hit 10 |
| config_missing | 0 |

| 规则 | 含义 | step 级 | episode 级 |
|---|---|---:|---:|
| `P31` | budget耗尽未完成 | 29 | 29 |
| `P17` | click-back振荡 | 26 | 26 |
| `P36` | WALK_FAIL_DEGENERATE | 99 | 25 |
| `P5` | 感知缺失循环 | 26 | 19 |
| `P7` | sCity=州名 | 21 | 19 |
| `P14` | URL 自环 | 17 | 15 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 19 | 15 |
| `P20` | 评测目标页从未访问 | 13 | 13 |
| `P18` | cheapest漏价格排序 | 11 | 11 |
| `P25` | 跨站任务跳过其中一站 | 11 | 11 |
| `P27` | 找不到即放弃 | 9 | 9 |
| `P10` | 跨步数值记忆失败 | 9 | 9 |
| `P23` | oldest误用价格排序 | 8 | 8 |
| `P19` | url_match过早搜索页finish | 5 | 5 |
| `P33` | 导航至裸图片URL幻觉 | 5 | 5 |
| `P24` | 不确定仍finish | 5 | 5 |
| `P12` | 从不翻页 | 4 | 4 |
| `P28` | benchmark-FP货币tokenize | 3 | 3 |
| `P37` | URL_HALLUCINATION | 3 | 3 |
| `P4` | 根节点误操作 | 5 | 2 |
| `P30` | 到达正确item后离开 | 1 | 1 |
| `P2` | 容器节点误点 | 1 | 1 |
| `P44` | HALLUCINATED_ELEMENT_REF | 1 | 1 |
| `P29` | benchmark-FP语义yes/no | 1 | 1 |

> ⚠️ **解读约束**（`docs/analysis/_data_quality_audit.md`）：
> ① 本表是**症状分布，不是死因分布** —— P36/P31 经 10 例跨 benchmark 因果验证均判为 risk-marker；
> ② `P2`/`P4` 依赖 `element_bbox`，在 **vision 上结构性为 0（假 0）**；
> ③ `P36` 在 vision 上只覆盖 `type` 步（click 无 `locator_route_meta`）→ **分母与 dom/som 不同**。
