# B0 som classifieds — 失败错因 digest（diag skill）

> **生成方式**: `/diag` skill 3-tier pipeline (2026-05-24 run on R9725)。Tier-1 deterministic 全扫 (`diag_pattern_match.py`, 0 token, ruleset `2-dom`) → Tier-2 Claude sub-agent 深挖 (**44 no-hit 全覆盖 + 8 success-hit FP 审计 = 52 ep / 8 agents**, sonnet) → Tier-3 整合 (本文件)。
> **Run**: `B0_som_classifieds_20260523_234208_078993305_172000_R9725` (Gate-3 **Option-A 全新重跑**，per-condition docker restart fresh substrate, manifest-bound authoritative)
> **Condition**: `phase1_som_router_0` | site classifieds | mode **som** | model **B0 = Qwen3-VL-235B (proxy)**
> **ruleset_version**: `3-domsom`（discover 第 2 mode 落码：som Tier-2 发现的 P24-P30 + P14/P10/P20 FP 修正已 land，全量重扫 dom+som 落同版本——见「Self-evolving」。**本 digest 数字为 v3 重扫后**；v2(2-dom)→v3 变化记于 Self-evolving。仍 dom+som 2-mode discover，vision/phantom 未跑 → **禁止 cross-mode 定量比较**直至 6-mode freeze）
> **Supersedes**: R2815 som 试跑（2026-05-23，**B-1848 Playwright driver-wedge hang run，已 archived**）。R2815 **不是**干净对照（wedge 污染 → +7pp 非对称，见 B-1858/§282），故本 digest **无干净跨 run 一致性章节**（som 缺第二个健康 run = §0④ gate-blocking repro-replicate 的核心缺口）。

> ⚠️ **定位声明（沿用 R9755/R31194 3-AI 审计共识，仍适用）**：本 digest 是 **internal 诊断记录，NOT paper-grade 结论**。
> - **单 condition + 无干净对照**：单 model×mode×site（B0 som cls），且 som 仅此一个健康 run。"som 救视觉 / routing 论点 / 换表征能救"需 dom/vision/phantom **同版本 ruleset** 对照才成立（当前 `2-dom` dom-biased，6-mode 数据未齐，**禁止任何 cross-mode 定量比较**）。
> - **presence ≠ causation**：156 failed 中仅 **44 no-hit + 8 success-hit FP 审计 = 52 经 sub-agent 逐个证因**；其余 112 failed-hit 是规则命中**未逐个 causal verify**。"agent-limit 主导"应读作"52 深挖 = 49 agent-limit + 3 benchmark-FP + 0 scaffold"。
> - **P14 已实测为 presence-only**：8/8 success-hit 审计确认 P14 在成功 ep 的 FP rate ≈100%（表单填写/图像浏览/评论提交都需多步停留同 URL）→ **failed P14=65 需重度打折，不可读作"65 个因 URL 自环失败"**。
> - **per-rule 非互斥**：分布是 per-episode-per-rule，P14∩P17 等有重叠，勿各行相加。
>
> **paper failure-analysis 待 6-mode + 多 condition + 同版本 ruleset 数据齐后重做，不复用本 digest 数字。**

---

## Verdict

| 维度 | 结论 |
|---|---|
| Episodes | 224 (**68 success / 156 failed**) — SR **30.4%** |
| 三分类 | **agent-limit**（52 深挖 = 49 agent-limit + 112 failed-hit 命中 agent-limit 类规则）· **scaffold-bug 0**（52 子集主动找零框架 bug + P8 全 run 0 命中）· **benchmark-FP ≥3**（task 42/96/222，均在 no-hit 子集，finish-vs-reference 语义对却判 0）|
| Deterministic coverage (failed) | **74.4%** (116/156, v3 3-domsom) — was 71.8% (v2 dom-biased)；新规则救 ~12 ep，P14 修正诚实打回 ~8 假覆盖 |
| no-hit failed | v2 **44** → v3 **40**（全深挖基于 v2：41 agent-limit + 3 benchmark-FP + 0 scaffold；v3 新规则覆盖 ~12，P14 FP 去除打回 ~8 待深挖）|
| success 命中规则 | v2 21/68（**P14 在 15/68=22.1% success fire** = 最强误报源 presence-only）→ **v3 P14 修正后 success-fire 0/68** ✅ |

**B0 som cls 的失败 ~主要指向真实模型能力局限，pipeline 干净**：52 深挖子集零框架 bug（parse/tool_call 全 valid）+ scaffold 规则 P8 全 run 0 命中。但与 dom 不同，som 多出 **≥3 个 benchmark-FP**（均 string_match 过严 = **B-21** 货币 tokenize / semantic yes-no；**task96 复查确认 url_match 实际 pass，是 string_match 那半 fail**，非 scoring bug）→ 真实 SR 是 **30.4% 的下界**（去 3 FP 后 ≥31.7%）。

---

## som vs dom 定性对照（⚠️ NOT 定量 — ruleset 非同版本统一打分）

> 仅作 discover 阶段方向参考，**不进 paper**。定量 cross-mode 须等 freeze + 全量重扫同版本。

| 指标 | B0 dom (R31194) | B0 som (R9725) | 定性读法 |
|---|---|---|---|
| SR | 15.18% | **30.4%** | som 标注图救了"纯视觉 URL 导航 / 图像唯一信息"类任务 → SR 翻倍 |
| 最强失败规则 | P6 视觉DOM必败 (49.5%) | P14 URL自环 (41.7%) | **换表征救了感知，没救导航/finish 行为缺陷** → 支撑"通用规则需 module 不是 route" |
| 视觉天花板规则 P6/P15/P16 | 大量 fire | **0 fire**（mode gate 生效） | som 不再受 DOM 视觉盲区 → 表征确实补了像素信息 |
| failed coverage | 87.9%→85.8% | 71.8%→74.4% | v2→v3 (3-domsom)；P14 FP 去除致 dom 微降，som 升（新规则覆盖）|

---

## Tier-1 规则分布 (failed-only, episode-level — v2 `2-dom` discover 快照, 落码前)

> 下方是**落码前 (v2)** 分布，保留以解释为何修 P14/P10/P20。**v3 `3-domsom` 落码后分布 + 完整变化见下 Self-evolving section**。

```
P14 URL自环                65  ████████████████████  41.7%  ★最强误报源(success 22.1%也fire), 已确认presence-only
P17 click-back振荡         31  ██████████            19.9%  风险真实但可自愈, 需区分单item振荡vs多item探索
P19 url_match过早搜索页finish 27 █████████            17.3%
P5  感知缺失循环           26  ████████              16.7%  部分误报(提交重试/删除后搜索)
P7  sCity=州名             16  █████                 10.3%
P20 评测目标页从未访问     15  █████                  9.6%  delete-from-list场景误解(见P20修正)
P2  容器节点误点           12  ████                   7.7%
P18 cheapest漏价格排序     12  ████                   7.7%
P10 跨步数值记忆失败       10  ███                    6.4%  日期污染FP(年份数字误判)
P23 oldest误用价格排序      9  ███                    5.8%
P12 不翻页 / P13 搜索代浏览 / P22 图上数字  少量
```
**is_scaffold 命中: 0**（唯一 scaffold 规则 P8 全 run 零命中；52 深挖子集主动找亦 0 scaffold → som pipeline 干净）。

---

## Tier-2 新发现 (44 no-hit 盲区 + 8 success-hit FP 审计)

### A. 44 no-hit 分类：41 agent-limit + 3 benchmark-FP + 0 scaffold

**som-specific 新失败模式（dom 不会有，som discover 核心产出）**：
1. **correct-then-abandoned**（task 93）— agent step_0 点中正确 item(=reference)，看标注图后**自我否定**（"these are different items"）离开，最终 finish 错 item。**som 标注图让 agent 对"是不是目标"更易过度自我怀疑**。
2. **som_ui_chrome_content_confusion**（task 199）— intent "website mentioned in the image"，agent 把 [SOM_MARKS] 里站点 logo 文本 "OsClass" 当成 listing 图片内容，答 OsClass 而非图内水印 kaiyo.com。**SOM marks 把 UI chrome 与 image content 混在同一 observation**。

**跨 mode 通用失败模式（som 也有，行为缺陷类）**：
3. **premature_finish_with_uncertainty**（task 101/176/182/201/60/67）— finish thought/answer 带明确不确定限定语（"though it's actually stained glass"/"not explicitly mention"/"while the description does not"）却仍 finish 错 item。agent 知道不匹配但视觉搜索能力耗尽 → 投降式 finish。
4. **visual_attr_keyword_search**（task 21/123/169/172）— 面对颜色/材质视觉属性，agent 用文字关键词搜索（"dark color"/"yellow"/"black phone"）代替看图，关键词搜不到 reference item。
5. **task_abandonment**（task 118/163）— 到 detail page 找不到目标即放弃（"cannot be completed"/"does not display"），不返回上级重试。
6. **cross_site_skip**（task 227/232）— 多站任务（start_url 含 `|AND|`）agent 从未访问其中一站，跳过跨站视觉推理。task 232 另暴露 `%7CAND%7C` URL-encode 异常（agent 把多站 URL 当单 URL 打开 → 潜在 scaffold 信号，但根因仍是 agent 未用正确 tab 导航）。
7. **premature_multivalue_finish**（task 128）— 多值 VQA（must_include 4 个球衣号）agent 1 步直接 finish，只读到部分值。

### B. 8 success-hit FP 审计结论（presence ≠ causation，方法论修正）

| 规则 | causal 判定 | 修正建议 |
|---|---|---|
| **P14** URL自环 | **8/8 误报，FP≈100%** | 成功 ep 多步停留同 URL 是必然（表单/浏览/评论）。failed=65 **重度打折**；建议排除 edit/item-detail 页长停留，或降级为 risk-marker 不计 causal |
| **P5** 感知缺失循环 | 部分误报 | "提交重试"(task217 评论控件无响应) / "删除后继续搜索"(task5) 是 FP；真死循环(同动作+页面真不变)才 causal |
| **P10** 跨步数值记忆 | 日期污染 FP | task25：thought 含 "16th November 2023" 的 16/2023，answer="1"(计数)，规则误判数值不匹配。建议排除年份(>1900)/日期 pattern 数字 |
| **P17** click-back振荡 | 高风险真实但可自愈 | task100 单 item 访问 8 次=真风险但自纠正成功；task102 多 item 各 2-3 次=正常图像匹配探索。需区分两者 |
| **P20** 目标页从未访问 | delete 场景误解 | task5：delete 从 listings 页 AJAX 发起无需进 item 页，eval program_html 自行导航验证 404。建议排除 program_html+404(删除验证型)任务 |

---

## benchmark-FP 详查（3 个，影响 SR 真实值）

| task | eval_type | 现象 | 性质 |
|---|---|---|---|
| **96** | url_match EXACT + string | url_match **pass** (item/5939=ref) 但 string_match **fail**：must_include "14" vs answer "$14.00" — NLTK `word_tokenize` 切成 `['$','14.00']`，"14" 非独立 token → AND=0 | ✅ 复查完成 = **B-21** (string_match 货币 tokenize 假阴性, evaluators.py:173-177)，非 scoring bug；**P28 已捕获** |
| **42** | string_match | answer "$5.00 to $120.00" 语义对，但 NLTK `word_tokenize('$5.00')=['$','5.00']`，must_include "5" 不是独立 token → 0 分 | string 过严（货币格式 tokenize 假阴性） |
| **222** | string_match | answer "...is correct"，must_include "yes"，语义等价但字面不含 → 0 分 | string 过严（semantic yes/no mismatch） |

→ 真实 SR ≥ **31.7%**（71/224），30.4% 是下界。failed-hit 112 未逐个审，可能含更多 FP。

---

## 代表 episode

**agent-limit · 视觉细粒度**：
- task 15 — 忽略 "red case"（红琴盒）视觉属性，仅凭标题文本匹配错吉他（step_0 thought 无任何颜色确认）
- task 146 — 视觉计数错误，声称 item 图显示"恰好 4 本书"，实际选错（url EXACT 直接判负）

**agent-limit · 行为缺陷（通用，需 module）**：
- task 93 — correct-then-abandoned：step_0 到达正确 item 后自我否定离开
- task 201 — premature_finish_with_uncertainty：finish thought "while the description does not explicitly mention black and red" 仍 finish 错 item

**agent-limit · som-specific**：
- task 199 — som_ui_chrome_content_confusion：答 "OsClass"(站点 logo) 而非图内 "kaiyo.com"

**benchmark-FP**：
- task 96 — url_match pass 但 string_match fail（answer "$14.00" vs must_include "14"，NLTK 货币 tokenize = B-21；P28 捕获）
- task 42 — 货币格式 NLTK tokenize 假阴性

---

## 🔁 Self-evolving — P24-P30 已落码 + P14/P10/P20 修正（v2 `2-dom` → v3 `3-domsom`, 2026-05-24）

> 用户决策"规则库应更新" → discover 第 2 mode 落码（skill discover-then-freeze 的 **discover 阶段 = 逐 mode 累积字典**，非等 6-mode freeze）。`diag_pattern_match.py` bump `2-dom`→`3-domsom`，全量重扫 dom (R31194) + som (R9725) 落同版本。**P26 暂缓**（finish_at_search_page 难与合法 search-page 计数任务区分，留 Tier-2）。

**新规则 6 条** — failed ep-level / success-fire（success-safe 验证）:
| 规则 | 含义 | som failed | som succ-fire | dom failed | 覆盖 (som no-hit) |
|---|---|---|---|---|---|
| P24 | 不确定仍 finish (url_match wrong-item hedge) | 7 | 0 | 6 | 101/176/201 |
| P25 | 跨站任务跳过其中一站 | 12 | 1\* | 6 | 227/232 |
| P27 | 找不到即放弃 (abandonment) | 5 | 0 | 2(1\*dom) | 118/163 |
| P28 | benchmark-FP 货币 tokenize | 2 | 0 | 1 | 42 |
| P29 | benchmark-FP 语义 yes/no | 1 | 0 | 0 | 222 |
| P30 | 到达正确 item 后离开 (som self-doubt) | 1 | 0 | 4 | 93 |

\* 2 个边界 success-fire（**非逻辑 bug** = url_match 评测机制特性，记录不调优避免 over-fit）: **P25 task233**（跨站 url_match 单站即满足，未访 reddit 仍 success）/ **P27 task151**（url_match 读 live URL 不看 finish，agent 嘴上 "task cannot" 但 URL 碰巧匹配 = §282 t151）。FP rate ≤1/224=0.4% noise 级。

**修正 3 条** — success-fire 验证（核心成果）:
| 规则 | 修正 | som succ-fire | 效果 |
|---|---|---|---|
| **P14** | 排除 type 输入 / 多数 page_changed 的 "productive" 长停留（非卡死）| **15 → 0** ✅ | presence-only 根治；failed 65→19 |
| P10 | 双变量：matched 用全量 thought 数字（answer 可引用日期），"遗漏大数"只看去日期数字 | 4 → 3 | task25 日期污染修复，**无新增 FP**（先 strip-all 致 +4 FP，验证抓出 → 改双变量）|
| P20 | 排除 program_html required_contents 含 "404"（delete 验证型，AJAX 删除无需访问 item）| 1 → 1 | task5 场景 |

**coverage**: som failed 71.8% → **74.4%**（新规则救 ~12 ep；P14 修正诚实打回 ~8 假覆盖 ep 回 no-hit）· dom 87.9% → 85.8%（P14 FP 去除）。**新 no-hit 暴露 8 个 P14-假覆盖 ep**（26/35/43/75/112/136/137/142）= 下轮深挖目标。

---

## Actionable

| 项 | 处置 | 优先级 |
|---|---|---|
| **task 96/42** string_match 货币 tokenize FP | ✅ 复查完成 = **B-21 实例**（evaluators.py:173-177 word_tokenize 货币假阴性，2026-04-30 已 CONFIRMED）。P28 deterministic 检测；**不需新 B-number**。SR 影响：paper 报告时做 raw vs FP-adjusted 敏感性（fire 跑中不改 eval 保一致）| 🟡 paper SR 敏感性 |
| **task 42 / 222** string_match 过严 FP | 确认后 → 可能 task 排除 或 eval 上游修；与 dom/vision 是否同样 FP 待 cross-mode | 🟡 中 |
| **P14/P10/P20 误报修正** | diag_pattern_match.py self-evolve（bump version + 全量重扫 dom+som）| 🟡 中（待 user 决策落码时机）|
| **scaffold-bug 0** | som pipeline 干净 = paper-grade 好消息，无 action | ✅ |
| **som 缺第二健康 run** | = §0④ gate-blocking repro-replicate（B-1858）的核心缺口，post-fire/idle 跑第二个 B0 som 估 run-to-run std | 🔴 gate 前置（post-fire）|
