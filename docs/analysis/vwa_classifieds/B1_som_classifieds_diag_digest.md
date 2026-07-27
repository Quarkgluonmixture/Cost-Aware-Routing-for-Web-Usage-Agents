# B1 som classifieds — 失败错因 digest（diag skill）

> **生成方式**: `/diag` skill 3-tier pipeline (2026-06-05)。Tier-1 deterministic 全扫 (`diag_pattern_match.py`, 0 token, ruleset `5-domsomvispsom-b1860coord`) → Tier-2 Claude sub-agent 深挖 (**36 no-hit 全覆盖 + 7 success-hit FP 审计 + 5 failed-hit causal verify = 48 ep / 8 agents**, sonnet) → Tier-3 整合 (本文件)。
> **Run**: `B1_som_classifieds_20260604_072456_562166453_226675_R31705` (Phase-A clean, manifest-bound authoritative; condition finalize 2026-06-05)
> **Condition**: `phase1_som_router_0` | site classifieds | mode **som** | model **B1 = Qwen3-VL-4B (local)**
> **ruleset_version**: `5-domsomvispsom-b1860coord`（**不变** — 本轮是 **B1 首次 diag = discover**，B1-specific 新规则只**提议不落码**，理由见下方并行 session 约束）

> ⚠️ **定位声明（B1 首次 diag，internal 诊断记录，NOT paper-grade 结论）**：
> - **单 condition + 首个 B1 + 无对照**：单 model×mode×site（B1 som cls）。现有 P-rule (P1–P33) **全部在 B0 (235B) 上 discover**，B1 (4B) 失败行为分布不同 → 套用 B0 ruleset 打分**只描述命中，不构成跨模型/跨模式结论**。**禁止任何 cross-mode / cross-model 定量比较**直至 freeze + 全量重扫。
> - **presence ≠ causation**：192 failed 中仅 **36 no-hit 全覆盖 + 5 failed-hit causal verify + 7 success-hit FP 审计 = 48 ep 经 sub-agent 逐个证因**；其余 ~144 failed-hit 是规则命中**未逐个 causal verify**。
> - **⛔ 并行 session 约束（本 session 关键纪律）**：另有并行 session 同时 diag 另一 som condition。`diag_pattern_match.py` / `RULESET_VERSION` / `diag_autorun.sh` 全量重扫是**共享文件/动作**，两个 discover session 并改 `ALL_RULES` 会 race。按 cross-mode 协议 discover/freeze 两阶段拆开：**本 digest 只 discover + 提议规则草案，绝不编辑 diag 脚本 / 不 bump version / 不全量重扫**。落码留到协调后的单独 freeze step。
> - **per-rule 非互斥**：分布是 per-episode-per-rule。

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
| SR | **14.29%** (32/224) |
| failed + hit | 162 |
| **failed NO-hit** | **30** |
| success + hit | 5 |

v8 新规则 failed 侧: {'P45': 71, 'P46': 2}；success 侧: 无。
（`P43` 在 cls 上大量命中属预期 —— 它标记"intent 需要视觉 + 该 mode 无页面截图"这一**中性组合**，
并非预测失败；§387.10 实测补上截图的增益 ≈0。）

全部 36 个 canonical condition 现处同一版本 → **cross-mode / cross-site 聚合解锁**。

---
## Verdict

| 维度 | 结论 |
|---|---|
| Episodes | 224 (**32 success / 192 failed**) — SR **14.29%** |
| 三分类（48 ep 深挖 + 全 run 规则推断） | **agent-limit 压倒性主导**（48 深挖 = 46 agent-limit + 2 边缘-FP；deterministic FP 规则 P28/P29 各 1）· **scaffold-bug = 0**（P8 0 命中；task 129/38 有 locator-walk-fail / parse-loop 的 scaffold-adjacent 信号但 framework 已兜底，根因仍 model）· **benchmark-FP ≈ 1.5%**（P28 货币 tokenize 1 + P29 语义 yes/no 1 + task 192 multi-value 边缘） |
| Deterministic coverage (failed) | **81.3%** (156/192 failed 命中) |
| no-hit failed | **36** — **全覆盖深挖**（无抽样，cap 50 内） |
| success 命中规则 (FP source) | **20/32 success 触发 hit** — P5 (14) / P14 (13) / P31 (10) / P12 (7) 为 presence-only 主源（FP 审计 18/20 hit non-causal） |

**核心结论（三路证据收口）**：B1 (4B) 失败本质 = **stop-condition 识别缺陷**，在频谱**两端同时失败**。dominant 信号 **P31（trajectory_incomplete，占失败 51.6%）不是死因，是 B1 finish-less 行为的结构性 artifact** —— 见下「关键发现 1」。这是本 digest 最 paper-relevant 部分，也是任何 B0↔B1 比较的硬约束。

---

## B1 行为画像（全 224 ep 聚合 — 双峰失败的统计骨架）

| 指标 | all | success | failed | 读法 |
|---|---|---|---|---|
| **agent_finished**（主动发 finish） | 50.9% | 65.6% | 48.4% | **只有半数 episode B1 主动 finish** — finish-less 是 B1 结构特征 |
| **trajectory_incomplete**（跑满 budget 未 finish） | 49.1% | **34.4%** | 51.6% | success 上也 34.4% → 评测器(url_match/program_html)不需 agent finish 即判成功 |
| **steps** | mean 18.0 / **median 26.5** / max 31 | — | — | **median > mean = 双峰指纹**（低步早放弃簇 + 31=budget 簇） |

**FAILED (n=192) 双峰拆分**：

| 失败极 | 信号 | 计数 | 占失败 |
|---|---|---|---|
| **极 A — 早放弃 (overconfident premature stop)** | `agent_finished & steps≤3` | 43 | **22.4%** |
| **极 B — budget 耗尽 (can't-stop oscillation)** | `trajectory_incomplete` | 99 | **51.6%** |
| 中间 | `finished & steps>3` | 50 | 26.0% |

→ B1 在 stop-condition 频谱**两端**都坏：要么 1–3 步草率断定"找不到"就 finish（极 A），要么无法识别已到达答案、click-back 循环到 budget 耗尽（极 B）。同一底层缺陷的两种表现。

---

## 关键发现

### 1. ⭐ P31 (trajectory_incomplete, 51.6%) 不是死因 — 是 finish-less 行为的 artifact（三路证据）

P31 表层是 B1 头号失败规则，但**三条独立证据线一致判定它非 causal**：

- **聚合层**：B1 全程只有 50.9% episode 主动 finish。半数 episode 直接跑到 budget = trajectory_incomplete，与成败无关（success 上也 34.4% 触发）。
- **FP 审计（success-fire 10/10 presence-only）**：success 的 trajectory_incomplete 全是 **"finish-less arrival"** — B1 导航到 reference item（url_match）或完成 comment side-effect（program_html）后**不会发 finish**，但 url_match 直接读 agent 当前 URL、program_html 用 isolated context 独立查 DOM，**两类评测器都不需要 agent finish** → eval pass。P31 只看 trajectory_incomplete、不看 success/eval_type → 必在这些 success 上误 fire。
- **Causal 验证（failed 0/4 causal）**：失败集里触发 P31 的 episode（task 1/8/45/56），模型早在 step 2–10 就陷入不可逃逸的振荡(P17)/感知循环(P5)/搜索页早 finish(P19)，**跑满 budget 只是振荡的必然终点**。用 P31 计数描述 B1 失败会**高估"步数不足"、遮蔽真实 agent-limit 机制**。

> **Paper 硬约束**：`trajectory_incomplete` / `trigger_distribution` 是 **cross-model 行为混淆变量** — B0 (235B) 会主动 finish，B1 (4B) 半数不发 finish。任何用这两个字段做 B0↔B1 对比都必须先剥离 finish-less 行为差异，否则把"B1 不会说停"误读成"B1 探索更久/失败更多"。

### 2. ⭐ B1 视觉精细 grounding 系统性弱 + confidence-accuracy 解耦（核心 agent-limit paper finding）

som mode B1 **能看图**（SoM 标注图 + [SOM_MARKS] + AXTree），但 4B 在缩略图/详情图上的**精细读取**系统性出错，且**高 verbalized confidence (0.95) 犯错**：
- 颜色识别：task 12 motorcycle 答 black（实 red）、task 25 red boats 数成 0、task 50 把 SoM 红框误读为"red palette"商品、task 192 双色只报一色。
- 数字/文本读取：task 118 手机屏幕时间 12:42（实 3:03）、task 119 钞票 $100（实 $50）、task 128 球衣号只读 2/4 个、task 199 把页面平台名 OsClass 当作图中 URL（实 kaiyo.com）。
- 图像-语义匹配：task 16 coffee mug ↔ coffee maker 混淆、task 60 车内场景 ↔ RTX4090、task 181 把参考图幻觉成"图坦卡蒙金面具"。

这是 B0 (235B) 罕见、B1 (4B) 结构性的能力天花板 → **router 论点的潜在证据**（换更强表征/模型能救），但需 cross-mode 同版本对照才能下定量结论。

### 3. ⭐ 早放弃极（极 A）— B1 特有的 premature-negative-finish

task 84(1步首页 finish)/107/110/150/186/25/26：B1 看第一屏/首页就断定"not found"/"0"/"cannot complete"，1–3 步 finish，**从不 scroll/翻页/换 filter**。与 B0 的"探索更久"相反。是 stop-condition 缺陷在"过早停"一端的体现，可 deterministic 化（见 Self-evolving P34 候选）。

### 4. ⭐ Edit-form-not-submitted — 最干净的新 scaffold-adjacent agent-limit 模式

task 4/75/76：B1 进入 `item_edit` 页、type 了含目标价格的新 description，**但从不点 Save/Submit 就 finish** → program_html 在 item 查看页检测到价格未变 → fail。`eval_source_agent_url 含 item_edit` 的 5 ep **全部失败**（5/5）。根因 = 4B 计划深度不足，丢失 "type→submit→verify" 三步链（B0 会继续点 Save）。signal 极干净、precision 高 → 最强 P-rule 候选（P35）。

### 5. FP 审计：现有规则的 success-safe 边界在 B1 上失效

20 hit 中 18 presence-only。**P14 的 13 success-fire 直接证伪 diag skill「P14 v3 修正 → success-fire 0」记录** —— v3 carve-out 用"有 type = productive"豁免，但 B1 的模式是"到达 reference item 页后**无 type 的反复无效 click**（trying to finish）"，v3 不覆盖这种 finish-less arrival。**这正是 cross-model discover 暴露的：B0 上验证过的 success-safe 边界不适配 B1**。修法见 Actionable。

---

## Tier-1 规则分布（failed episode-level，n=192）

| 规则 | failed-ep | % failed | success-fire | 性质 |
|---|---|---|---|---|
| **P31** budget耗尽未完成 | 99 | 51.6% | **10** | ⚠️ 非死因（见关键发现 1）；success-fire 全 presence-only |
| **P5** 感知缺失循环 | 65 | 33.9% | **14** | 部分 causal（前期卡壳）+ 大量 presence-only（到达页 stuck） |
| **P19** url_match过早搜索页finish | 62 | 32.3% | 0 | causal 度高（task 21/45 真死因）；success-fire 0 = 干净 |
| **P17** click-back振荡 | 42 | 21.9% | 2 | causal（task 1 教科书级振荡） |
| **P18** cheapest漏价格排序 | 37 | 19.3% | 3 | 部分 causal；success-fire(task17)=sPriceMin/Max 合法替代策略 FP |
| **P4** 根节点误操作 | 36 | 18.8% | 1 | success-fire(task217)=SoM label eid=1 ≠ DOM root 的 FP |
| **P14** URL自环 | 28 | 14.6% | **13** | ⚠️ success-fire 高，v3 carve-out 不覆盖 B1 finish-less arrival |
| P20 评测目标页从未访问 | 18 | 9.4% | 0 | — |
| P25 跨站跳过其中一站 | 12 | 6.2% | 1 | — |
| P12 从不翻页 | 12 | 6.2% | 7 | presence-only 主源之一 |
| P7 sCity=州名 | 11 | 5.7% | 0 | — |
| P10 跨步数值记忆失败 | 8 | 4.2% | 0 | — |
| P27 找不到即放弃 | 8 | 4.2% | 1 | success-fire(task217)=program_html side-effect 已成 FP |
| P30/P13/P24/P22/P23/P33/P2 | ≤7 each | — | 0 | 长尾 |
| **P28 货币tokenize / P29 语义yes/no** | 1 / 1 | — | 0 | **benchmark-FP 规则**（deterministic 已识别） |

---

## Tier-2 新发现

### no-hit 子集（36 ep 全覆盖）= 35 agent-limit + 1 边缘-FP

**0 scaffold-bug、0 纯 benchmark-FP**（task 192 multi-value must_include 部分答 = 边缘）。全部归 agent-limit，集中在：视觉 grounding 弱（发现 2）、早放弃（发现 3）、edit 未提交（发现 4）、gallery 单步读错（task 41/42/118）、跨步记忆崩溃 explore-loop（task 101 重访同一错误 item ≥3 次）。

### success-hit FP 审计（7 ep / 20 hit）= 18 presence-only + 2 causal

P31 10/10、P5 多数、P14 13、P12 7 全 presence-only（B1 finish-less arrival 在 reference 页反复无效 click）。仅 task 166 中段两条 hit（错误 item 页短暂真卡壳）causal，但 episode 最终成功 → 非死因。

### causal 验证（5 failed ep）

P31 **0/4 causal**（伴随现象）；P17 task 1 真死因；P5 task 8/21 前期真死因；P19 task 21/45 真死因；P18 task 56 真死因。**统一底层缺陷 = B1 缺乏"我已找到了"的 stop-condition 识别**：能感知局部状态（"Yaris 非红"/"ID 45196 visible"）但无法 commit+finish，只能退回"搜索"重试 → 振荡/早放弃/budget 耗尽都是同一缺陷的不同表现。

---

## 代表 episode

| 类别 | task | 死因 | 证据 |
|---|---|---|---|
| agent-limit · finish-less artifact | **1** | 感知 Yaris 非红但 28 步 click(element_65)→back 振荡至 budget；P31 是表象、P17 是机制 | step_02–29 重复同一 click+back 13 轮 |
| agent-limit · 早放弃 | **84** | 1 步在首页（未搜索）断定无 selfie 戒指 finish；reference item/43966 $6000 存在 | step_0 finish, obs_url='…/'(首页) |
| agent-limit · edit 未提交 | **75** | item_edit 页 type 含 $120 新 desc 但未点 Save 即 finish；price 未持久化 | step_6 finish while url=item&action=item_edit |
| agent-limit · 视觉精细读取 | **119** | 钞票面额答 $100（实 $50），step_0 已承认不确定却 step_1 高置信 finish | confidence 0.95, answer='100' |
| agent-limit · explore-loop | **101** | 11 步重访 item/12085(错误城堡画)≥3 次，每次重看同图做同样错判 | 跨步记忆崩溃 + 视觉辨别失败 |
| 边缘 benchmark-FP | **192** | reference must_include[red,white] 双值，B1 只答 red（感知到双色但只报最显著） | "evaluation asymmetry" 候选 |

---

## 🔁 Self-evolving — 提议 P34+ 规则（⛔ 本 session 不落码，留 freeze step）

> 并行 session 约束下**仅记录草案**。落码 = 单独协调后 `ALL_RULES` 加 `check_pN` + bump `RULESET_VERSION` + `diag_autorun.sh` 全量重扫。**每条都须内置 success-safe 条件**（避免再造 presence-only 规则）。

| 候选 | 模式 | 0-token signal | 覆盖 | success-safe | 优先级 |
|---|---|---|---|---|---|
| **P35 edit-form-not-submitted** | edit 页 type 后未 submit 即 finish | `eval_types 含 program_html & agent_finished & eval_source_agent_url 含 'item_edit'` | task 4/75/76（item_edit 5/5 全 fail） | program_html 已隐含（成功需 DOM 变更） | **最高**（signal 最干净） |
| **P34 premature-negative-finish** | 早放弃，否定句 finish | `steps≤3 & agent_finished & finish_answer 含否定('no'/'not found'/'cannot'/'none') & reference 非空` | task 84/107/110/150/186 等（极 A 簇 ~22%） | ⚠️ 须排除 P29 语义 yes/no 合法"no"；建议加 `success=False` gate | 高 |
| **P36 gallery-single-step-misread** | 落 gallery 页 1 步 finish 读错行/价 | `agent_action_step_count=1 & start_url 含 sShowAs=gallery & answer 含 price-range` | task 41/42/118 | success=False gate | 中 |
| P37 walk-fail-blind-finish | locator walk_fail 重复后不换策略 finish | `连续≥2 step locator_route_meta.error='walk_fail:no_actionable_within_walk' 同 eid & 后续 finish` | task 129 | success=False gate | 中（scaffold-adjacent） |
| P38 parse-loop-then-hallucinate | wait+parse_fail 连发后仓促 finish | `parse_error_injected_wait_count≥2` | task 38（全 run 仅 1 ep） | — | 低（样本太少，先观察） |

---

## Actionable — 现有规则 success-safe 收窄（⛔ 同样留 freeze step）

FP 审计给出的高优先收窄（消除 B1 上的 presence-only 误报）：

1. **P31 加 `success=False` gate** — 一行改动，消除全部 **10 次悖论性 success-fire**；零误伤（trajectory_incomplete 在 success 上恒为 finish-less artifact）。**最高优先**。
2. **P14：url_match 任务 `run_url==reference_url` 时 skip** — 精准消除 B1 finish-less arrival（比 success gate 在 failed ep 上保留更多检测力）；同时**修复 v3 carve-out 对 B1 的盲区**。
3. **P5 加 `success=False` gate** — 消除到达页 stuck-loop 的 presence-only success-fire。
4. **P4：SoM 模式 eid=1 触发时加 bbox size check**（仅 `bbox 宽>1200 & 高>680` 才算 root）— 消除 SoM label-id ↔ DOM root-id 混淆 FP（task 217）。
5. **P18：URL 含 `sPriceMin=`/`sPriceMax=` 时 skip** — 价格区间过滤是 cheapest 的合法替代策略（task 17）。
6. **P27 加 `success=False` gate 或 program_html carve-out** — side-effect 已被 isolated-context eval 独立验证时不算"放弃"（task 217）。

**无 B-number 候选**：scaffold-bug = 0；task 129/38 的 walk-fail/parse-loop 是 model 输出质量问题，framework 已兜底（注入 wait/retry），非框架 bug。pipeline 干净（48 深挖子集 parse/tool_call 全 valid）。

---

## 跨 condition 协议状态

- **仍禁止 cross-mode / cross-model 定量比较**：B1 首次 diag，ruleset 仍是 B0-discovered；B1-specific 规则未落码。
- **version 演进未推进**：`5-domsomvispsom-b1860coord` 不变（discover findings 入本 digest，ruleset 不动，与 phantom_prompt R14655 纯-discover 同处理）。
- **freeze 前置**：B1 的 P34/P35/P36 + 6 条 success-safe 收窄落码 → bump version → `diag_autorun.sh` 全量重扫所有已扫 condition → 才可做 cross-mode/cross-model 表。**须与并行 session 的 discover 产物合并后统一 freeze**（避免 ALL_RULES race）。
- **未跑**：B1 其余 mode (dom/vision/phantom*)、B1 reddit、B2 全系列。
