# B0 dom classifieds — 失败错因 digest（diag skill）

> **生成方式**: `/diag` skill 3-tier pipeline (2026-05-26 run on R21557)。Tier-1 deterministic 全扫 (`diag_pattern_match.py`, 0 token, ruleset `4-domsomvis-b1860coord`) → Tier-2 Claude sub-agent 深挖 (**22 no-hit 全覆盖 + 6 success-hit FP 审计 + 3 failed-hit causal verify = 31 ep / 6 agents**, sonnet) → Tier-3 整合 (本文件)。
> **Run**: `B0_dom_classifieds_20260525_194618_553890342_530647_R21557` (AMENDMENT_07 sequential-id post-fix fresh re-launch, per-condition docker restart, manifest-bound authoritative)
> **Condition**: `phase1_dom_router_0` | site classifieds | mode **dom** | model **B0 = Qwen3-VL-235B (proxy)**
> **ruleset_version**: `4-domsomvis-b1860coord`（不变；本 run 是 dom mode 的 sequential-id 重跑——dom 用 native nodeId 不受 AMENDMENT_07 影响，是 in-distribution clean re-run。**禁止 cross-mode 定量比较** 直至 6-mode 数据齐 freeze + 全量重扫）
> **Supersedes 修正 (2026-05-26)**: ~~R31194 digest archived per AMENDMENT_07 在 `_archive_amend07_seqid_R31194_dom`~~ — path 错误。R31194 raw data 实际位于 A100 `results/repro_replicates/B0_dom_classifieds_R31194_clean_replicate/` (§297 clean replicate path)。**R31194 digest 已从 git `de0ae65` 恢复** 为 [[B0_dom_classifieds_R31194_diag_digest]] (2026-05-26 user dual-digest 指令补救)。R31194 与 R21557 都用 native nodeId（AMENDMENT_07 只动 SoM-family），二者是同 condition 的 **fresh-vs-fresh out-of-sample 检验**（见下「跨 run 一致性」 + [[B0_dom_classifieds_diag_digest]] pointer 完整 run-to-run table）。

> ⚠️ **定位声明（沿用 R9755/R31194 3-AI 审计共识，仍适用）**：本 digest 是 **internal 诊断记录，NOT paper-grade 结论**。
> - **单 condition + dom-only + 无对照**：单 model×mode×site（B0 dom cls）。"DOM 表征天花板 / routing 论点 / 换表征能救"需 **som/vision/phantom 对照**才成立（当前 6-mode 数据未齐：dom R21557 + som R5313 已落地，vision/P-text/P-prompt/P-SoM 待跑）。**禁止任何 cross-mode 定量比较**。
> - **presence ≠ causation**：185 failed 中仅 **22 no-hit + 3 failed-hit causal verify + 6 success-hit FP 审计 = 31 经 sub-agent 逐个证因**；其余 160 failed-hit 是规则命中**未逐个 causal verify**。
> - **P6 在 30 success 上 fire 26 个 (87%)** = 最强 FP source，3 success-hit verify confirmed 5/6 non-causal → P6 dom mode 高 FP risk 沿用 R31194 结论。
> - **per-rule 非互斥**：分布是 per-episode-per-rule，P6∩P14∩P19 大量重叠，勿各行相加。
>
> **paper failure-analysis 待 6-mode + 多 condition 数据齐后 cross-mode aggregator 重做，不复用本 digest 数字。**

---

## Verdict

| 维度 | 结论 |
|---|---|
| Episodes | 224 (39 success / **185 failed**) — SR **17.4%** |
| 三分类 | **agent-limit 主导**（31 深挖 = 28 agent-limit + 1 scaffold-bug + 1 benchmark-FP + 1 unclear/edge）· **scaffold-bug ≥1**（T180 cls 评分 widget radio input AXTree 不暴露）· **benchmark-FP ≥1**（T216 cross-run cross-mode 全 4 run 落同一替代 item id=82390 而非 reference 66046）|
| Deterministic coverage (failed) | **88.1%** (163/185 failed 命中) — 与 R31194 fresh-substrate 87.9% 一致 |
| no-hit failed | **22**（21 agent-limit + 1 scaffold-bug + 1 benchmark-FP / 1 unclear; 13 个 R31194 已暴露的盲区在 R21557 仍 no-hit = 高一致）|
| success 命中规则 | 30/39（**P6 在 26/30 success 上 fire = 87% FP rate** 沿用 R31194 → P6 carveout 仍 open）|

**R21557 SR 17.4% (vs R31194 15.18%, R9755 14.7%)** — sequential-id 重跑后小幅上升 2.2pp，仍在 dom 表征对 cls 视觉任务结构性天花板的 floor 邻域 (§D4 dom partial @88 task 17%→25% Δ=+8pp McNemar n.s. = native nodeId churn + MoE 残留)。pipeline 干净：31 深挖子集 parse/tool_call 全 valid，agent-limit 主导 ~100%。

---

## 跨 run 一致性（R31194 → R21557, AMENDMENT_07 前后 dom）

R21557 是 R31194 经 SoM sequential-id fix 后 fresh 重跑（dom mode 与 AMENDMENT_07 无关，故 R21557 验证 ⚖ R31194 sequence robustness）：

| 指标 | R31194 (pre-amend07) | R21557 (post-amend07) | 读法 |
|---|---|---|---|
| SR | 15.18% (34/224) | 17.4% (39/224) | +2.2pp，仍 floor 邻域；MoE / token-trace 残留方差 (笔记 §242) |
| failed-coverage | 87.9% | 88.1% | 一致 = ruleset 规则集对 dom-cls 稳定 |
| no-hit failed | 23 | 22 | 一致；13 个 R31194 no-hit 在 R21557 仍 no-hit（84/97/106/124/129/131/207/208/216/217/218 等）= 规则盲区 highly reproducible |
| scaffold / FP | 0 / 0 | **1 / 1** | R21557 sub-agent 主动找到 T180 scaffold-bug + T216 benchmark-FP — 此前 R31194 sub-agent 未单独 spot 这两个（不同 batching 命中盲点不同；非新现象，是 sampling artifact）|

**no-hit 任务高度 cross-run 共享** = 验证 ruleset 没在 fresh data 上 collapse。

---

## Tier-1 规则分布 (failed-only & success-fire, ruleset 4-domsomvis-b1860coord)

```
rule  failed  success-fire  notes
P2     68      5            container-misclick (high vol)
P4      2      0
P5     48      3            perception-loop
P6     80     26            ★ dom 视觉天花板 (87% FP-rate on success)
P7     19      0            sCity=州名
P10    13      4            cross-step number recall (P10 FP 仅来 url_match 端口/element_id, 见 R5313)
P12     4      1
P13     3      0
P14    26      3            ★ v3 修后 success-fire 0→3 仍 OK (P14 v3 productive-stay carveout 起作用)
P15     5      1            gallery-row pos (dom-only)
P16    46      7            ★ visual-image-content (dom-only)
P17    32      5            click-back oscillation
P18    13      0
P19    35      0            url_match 过早 finish (CLEAN signal)
P20    15      0
P21     5      0
P22     4      0
P23     8      0
P24     5      0
P25     5      0
P27     7      0            abandonment (vision-extended + ref-carveout)
P28     2      0            bench-FP 货币 tokenize
P30     5      0
P31    56      2            ★ budget-incomplete (high vol; 2 success-fire 可能仍 productive)
P32     1      0            text-in-price-filter
```

主导规则: **P6=80** (视觉天花板) · **P2=68** (容器误点 — 多 P6 superseded) · **P31=56** (budget 耗尽) · **P5=48** (感知缺失循环) · **P16=46** (视觉图像内容) · **P19=35** (url_match 过早 finish) · **P17=32** (click-back 振荡) · **P14=26** (URL 自环, v3 修后 success-fire 仅 3)。

---

## Tier-2 新发现 (sub-agent 深挖 22 no-hit + 6 success FP audit + 3 failed verify)

### 关键发现 1 — 1 scaffold-bug (paper-relevant): cls 评分 widget radio AXTree 不可见

**T180** (任务: 给 listing 评 5-star + 评论)。Classifieds CSS 星级评分 widget 是 radio input，**完全不暴露在 AXTree 中** — DOM mode agent 看到的 group 节点内只有 `StaticText 'Rating'` + `textbox 'Rating Title'` + `textbox 'Comment'` + `button 'Send'`，**0 个 radio 节点**。Agent step_12 尝试 click 评分区域，locator 返回 `walk_fail:no_actionable_within_walk`。最终评论已提交但无星级 → eval `program_html` 检查 `.comments_list h3 must_include='5 of 5'` 必 fail。

**含义**: cls 任意涉及"give N stars"的 task 在 dom + som + vision 三 mode 都会 fail (radio 不在 AXTree → 视觉模式可能能点但 AXTree-grounded agent 看不到)。**可修**: VWA submodule 改写 `helper_functions.py` 把 radio 暴露 aria 或 interaction 层 JS 强制 click。**B-number candidate**: 待加入 master_bug_catalog (类似 B-21 货币 tokenize / B-1836 eval-timeout, 但更上游)。

### 关键发现 2 — 1 benchmark-FP cross-run cross-mode 一致替代 item

**T216** (任务: 找 $420-$430 区间最便宜的 oval table)。Agent 在 R21557 dom + R31194 dom + R24792 vision + R5313 som **4 个不同 run/mode** 全部一致落在 `item&id=82390` (Weiman Fruitwood Oval Two-Tier Cart, $420, 标题明确含 "Oval")。**Reference url=item&id=66046 在所有 4 个 run 的所有 DOM artifact 中均未出现** (`grep` 无命中) — 是从未在 search/browse 路径上的 item id。

**含义**: 跨 run/mode 一致的语义正确替代 → 强 benchmark-FP 信号 (类似 R31194 task 42/96/222 string-match FP)。建议加入 `vwa_manual_non_visual_task_ids.py` 旁的 FP-candidate 排除列表 (待 6-mode 完整数据后再 finalize)。

### 关键发现 3 — DOM 视觉天花板的 6 种新 phrasing 盲区 (P6/P16/P22 regex gap)

22 no-hit 中 **21 个全部** indicate "intent 含视觉属性 + dom 模式不可读"，但现有 P6/P16/P22 regex 没匹配。具体分布：

| Task | Intent 关键短语 | 现有规则 miss 原因 |
|---|---|---|
| 12, 13 | "color of the most recently listed..." | P6 VISUAL_COLOR_KEYWORDS 漏 `\bcolou?r\b` 名词本身 (只有具体色名) |
| 16 | "the item with the coffee mug **in the picture**" | P16/P22 漏 "in the picture/image/display" 通用 phrasing |
| 59 | task_config.image 非空 (外部参考图) | 无 task_config 字段直读规则 |
| 84 | "selfie image" (拍摄角度) | 纯视觉 attribute, 无文本 proxy |
| 97 | "shape of an animal" (物理外形) | P22 漏 "shape/picture of" |
| 106, 192 | "whose image has..." / "primary color of car..." | P16 phrasing variant |
| 107, 113 | "color of rims" / "jersey number" (raw PNG URL finish) | P16 漏 "color of N"; finish-on-raw-PNG 新模式 |
| 117, 123 | "similar color as the person" / "object color" + start_url 改写 | P6 漏 "similar color as"; start_url context 被改写未规则 |
| 129 | "listing image shows the price" | P22 漏 "image shows the price" |
| 131 | "puppies in the basket" (图像识别) | P16/P22 漏 "image with N" |
| 203 | "USB-C cable" 仅在图中 | 无 task_config.comments 字段直读规则 |
| 207, 208 | "color scheme matches" / "insect in picture" | P6 漏 "matches" 动词; "insect in picture" P16 漏 |
| 209, 218 | "batch 15 cheaper, only in image" / "USA coin in image" | 无 task_config.comments 直读规则 |
| 217 | comment form field 重复 type 覆盖标题 | 边缘 form-overwrite, deterministic 难 |

### 关键发现 4 — failed-hit P-rule causal verify: 3/3 not causal (presence-coincidence)

T3 (P2 容器误点) / T22 (P17 click-back) / T0 (P19 url_match 过早 finish) 经 sub-agent 逐 step verify：**全部命中 superseded by 更上游的 P6 (颜色不可见) + P15 (gallery 行位置)**。即 P2/P17/P19 hit 是 P6/P15 unrecoverable 死局下的下游症状。

**含义** (沿用 R31194 教训): failed-hit per-rule 计数读作"presence detector"非"causation"。R21557 dom 真实 agent-limit failure mass = **P6 + P15 + P16 + P31 (4 条规则) 联合主导**，其余规则计数虚高 by overlap。

### 关键发现 5 — success-hit FP audit: 5/6 P6 hit non-causal (P6 仍是高 FP-risk)

T4/T5/T15 (P6 "white car" / "red case" — DOM 文本本身含颜色词字面) + T44/T45 (B0 多模态 LLM 通过 reference image OCR 转 text search 绕过视觉匹配) 均 hit_causal=false。**仅 T11** ("first BLUE bike", DOM 无 color field, agent 走了 14 步含 6 次 back oscillation 才绕过感知障碍) hit_causal=true 但仍 success → near-causal grey-zone。

**建议 carveout** (待 6-mode discover 完成后落码):
- **P6 carveout-A**: DOM 文本本身含颜色词字面匹配 → 不 fire (类似 P14 v3)
- **P6 carveout-B**: B0 (多模态) + reference image OCR 路径可绕过 → 降级 presence-only

---

## Self-evolving — 提议新规则 (待 6-mode freeze 后批量落码)

> ⚠️ **按 discover-then-freeze 协议** (SKILL.md "跨 condition / cross-mode 工作协议"): 现 dom + som 两 mode discover 进行中，4 mode 未跑 (vision/P-text/P-prompt/P-SoM)。**新规则不立刻落码 ALL_RULES** — 等所有 6 mode 完成 + Tier-2 discover 完成 → 一次 freeze + 全量重扫 → 落同一 ruleset version。先记录草案。

dom-specific (mode != "dom" gate 默认) 草案:

- **P33** (color "not specified" + page=item): intent 含 `\bcolou?r\s+of\b` + obs_url 含 `page=item&id=` + finish.answer 含 `(?:not\s+(?:specified|mentioned|visible|available)|cannot\s+(?:be\s+)?determine)` → agent-limit visual-attr-not-in-dom. 覆盖 T12/T13 + 也覆盖 T192 self-report. **Success-safe**: finish 显式承认无法找到 → 成功 ep 不会写。
- **P34** (external reference image): `task_config["image"] != null` + eval_type 含 `url_match` + mode=dom → agent-limit external-ref-image-DOM-inaccessible. 覆盖 T59. **0-token**, 纯 config 读取，**最 deterministic 的新规则**。
- **P35** (in-the-picture phrasing): intent regex `r'(?:in|with|from)\s+the\s+(?:picture|image|display|photo)'` + mode=dom → agent-limit visual-image-content-DOM-blind. 覆盖 T16/T207/T208 + (P22 phrasing 扩展)。
- **P36** (raw PNG URL finish): mode=dom + finish step obs_url 匹配 `oc-content/uploads/.*\.png` + thought 含 "I (?:will\s+)?assume" → agent-limit visual-hallucination. 覆盖 T107/T113. **天然 success-safe**: url_match item-page reference 永不会是 raw PNG URL.
- **P37** (P22 phrasing 扩展): intent 含 `image\s+(?:shows?|has|with)\s+(?:the\s+)?(?:price|number|text)` + mode=dom → agent-limit image-embedded-text. 覆盖 T129. (P22 原 phrasing "number shown IN the image" 漏 T129 的 "listing image shows price")
- **P38** (cross-mode, scaffold-adjacent): scaffold-bug 提议 — intent 含 `\b\d+\s+star\b|\brating\b` + eval program_html `\d+\s+of\s+\d+` + click 评分区域 walk_fail → scaffold-bug star-widget-AXTree-blind. **B-number candidate**.
- **P39** (cross-mode, benchmark-FP detector): eval_type=url_match + reference_id NOT in any obs_url across all run trajectories + agent 跨 ≥2 run/mode 一致 finish 同 alternative_id → benchmark-FP candidate. 覆盖 T216. **可能需要 cross-run global view** (per-run Tier-1 看不到), Tier-3 整合时单 cross-run 脚本检测.
- **P40** (price boundary inclusive vs exclusive): obs_url 含 `sPriceMax=N` + finish step item 价格 == N + intent 含 `under|less\s+than|below` → agent-limit boundary-inclusive. 覆盖 T218. **0-token**.
- **P41** (P6/P16 regex 修正非新规则, 仍 bump ruleset version):
  - P6 VISUAL_COLOR_KEYWORDS 添加 `\bcolou?r\b` 名词本身
  - P6 IMAGE_VISUAL_MATCH_RE 添加 `similar\s+color\s+as` / `same\s+color\s+as` / `matches`
  - P22 phrasing 收宽（合并入 P37）

**预估覆盖**: 上面 8 条新规则可覆盖 22 no-hit 的 ~17 个 (T84 selfie 无 deterministic signal, T217 form overwrite 边缘, T203 USB-C cable 信号弱)。failed-coverage 88.1% → ~95%。

**FP-rate carveout 提议** (P6 / P10):
- P6 dom-mode + DOM 文本含目标颜色词字面 → 不 fire (R21557 success-FP 5/6 confirmed)
- P6 dom-mode + task_config.image != null + agent 用 reference-image OCR 后转 text search → 降级 presence-only
- P10 url_match task + output 含完整 URL → 过滤 URL 端口 (9980) 与 item_id 数字 (见 R5313 P10 FP audit, 6/6 non-causal)
- DATE_CONTEXT_RE 月/日残留阈值 `>10` 提高到 `>30` 或限定 3 位数+

---

## 代表 episodes

**scaffold-bug 1 个**:
- **T180** — rating widget radio AXTree 不可见 → 评论已提交但无星级 → eval `5 of 5` fail. **paper §8 / B-number candidate**.

**benchmark-FP 1 个**:
- **T216** — Weiman Fruitwood Oval Cart $420, 跨 4 run/mode 一致 finish item=82390 (语义正确), reference=66046 (从未出现). **FP-candidate list 加入**.

**agent-limit 代表**:
- **T59** — task_config.image 非空 (企鹅参考图), B0 dom 无法读图, 推断"penguin"但实际正确动物不同, finish item=15671 (penguin game) vs reference=6247. **P34 deterministic candidate** (单 config 读取).
- **T129** — "listing image shows the price" → DOM 文本看到的 $15 是图旁边 text, agent 选错 seller. **P37 candidate**.
- **T123** — start_url 已指定 cls clothes 搜索, agent step_0 type "yellow" 改写 sPattern, 后续操作脱离 task 指定上下文. **P38 candidate** (cross-mode applicable).

---

## paper-grade 含义 (待 6-mode 完成后再 finalize)

1. **DOM 视觉天花板**: 22 no-hit + 80 P6-hit + 46 P16-hit = **dom mode 视觉相关失败 mass ≥ 35%** of all failures. paper §3 hero `DOM 必败 / SoM 可救` 的 dom 端 evidence numerator. 但**禁止下 cross-mode 定量比较**直至 vision/SoM 同 ruleset 数据。
2. **scaffold-bug 暴露 1 个 (T180)**: cls 评分 widget AXTree 不暴露 — `master_bug_catalog` 待新 B-number; 暂归 P38 草案 + paper §8 discussion (类似 B-21 货币 tokenize 的 benchmark scaffold limitation).
3. **benchmark-FP 暴露 1 个 (T216)**: cross-run cross-mode 一致 + reference id 从未出现 = 强 FP 信号. paper §8 disclosure 列表加入.
4. **跨 run 高一致** (R31194 ↔ R21557): SR ±2.2pp, no-hit 13 重叠, failed-coverage 88.1% vs 87.9%. **AMENDMENT_07 SoM-fix 不影响 dom mode 失败模式**, 符合预期 (P79 dom 用 native nodeId).

---

## Cross-link

- 实验笔记 §294-§296 (AMENDMENT_07 + run-to-run sensitivity)
- master_bug_catalog (T180 scaffold-bug + T216 FP 待加 B-number)
- 上 game `next_steps.md` §0 ④ — Phase 1a fire 跑中, 等所有 6 mode 完成 → discover-then-freeze 全量重扫 → 才下 cross-mode 结论
- R31194 archived digest (旧版本, supersedes by 本文件)
- R5313 som digest (B0 som cls, 同步刷新)
