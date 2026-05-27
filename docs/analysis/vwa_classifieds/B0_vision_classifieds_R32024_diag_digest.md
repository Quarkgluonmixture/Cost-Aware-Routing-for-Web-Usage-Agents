# B0 vision classifieds — 失败错因 digest（diag skill, R32024）

> **生成方式**: `/diag` skill 3-tier pipeline (2026-05-26 run on R32024)。Tier-1 deterministic 全扫 (`diag_pattern_match.py`, ruleset `4-domsomvis-b1860coord`, 0 token, 222 ep scan / 224 raw — 差 2 推测 jsonl 跳过, 单一 note 不阻塞) → Tier-2 Claude sonnet sub-agent (40 no-hit / 6 agents + 11 success-hit FP audit / 1 agent + 5 failed-hit verify / 1 agent = 56 ep / 8 agents) → Tier-3 整合 (本文件).
> **Run**: `B0_vision_classifieds_20260526_141916_610351680_689390_R32024` (Phase 1a 第 3 次 re-launch 后首个 vision condition; AMENDMENT_07 sequential-SoM-id 落码后; B-1839 per-condition docker restart fresh substrate; manifest-bound authoritative)
> **Condition**: `phase1_vision_router_0` | site classifieds | mode **vision** (agent 只看裸截图, 无 AXTree 文本) | model **B0 = Qwen3-VL-235B (proxy)**
> **规模**: N=222 (diag scan) / 224 (raw episode summaries) | success=55 | failed=167 | **SR=24.8%** (55/222) | total_steps=3557 | avg_steps=15.9 | trajectory_incomplete=66/224 (29.5%) | **parse_error_rate=0.0%** (B-1860 修复 hold)
> **ruleset_version**: `4-domsomvis-b1860coord` (与 R24792 同版本; 3/6 mode discover, **禁 cross-mode 定量比较**)
>
> **🔁 Run-to-run sibling**: [[B0_vision_classifieds_R24792_diag_digest]] (archive AMENDMENT_07 前同 condition 同 ruleset, paper Risk 6 / H1 sensitivity 证据). Run-to-run 对照表 见 §0.

> ⚠️ **定位声明** (沿用 dom/som/R24792 vision 3-AI 审计共识): 本 digest 是 **internal 诊断记录, NOT paper-grade 结论**。
> - **单 condition + vision-only + 无 cross-mode 对照**: SR / 失败分布只描述 B0 vision cls 自己。"vision 表征局限 / routing 论点 / 换表征能救"需 6-mode 对照。
> - **presence ≠ causation**: 167 failed 中 40 no-hit 逐个证因 + 5 failed-hit causal verify (sample), 仍未 P-rule 全量 causal map。
> - **per-rule 非互斥**: 分布是 per-episode-per-rule 命中, P19∩P5 等重叠, 勿各行相加。
>
> **paper failure-analysis 待 6-mode + 多 condition 数据齐 + 全量重扫统一 ruleset 后重做, 不复用本 digest 数字。**

---

## §0 Run-to-run 对照 R24792 ↔ R32024 (⭐ 本轮 diag 增量价值)

> **paper Risk 6 / H1 sensitivity 实证**: 同 condition (B0 vision cls), 同 ruleset (`4-domsomvis-b1860coord`), 不同 fire date — 之间发生 AMENDMENT_07 SoM sequential-id 落码 + B-1862 manifest rebind + Phase 2 disk migration。**vision 模式不读 SoM 标注图** → AMENDMENT_07 对 vision **无直接影响**;两 run 差异源于 **(a) B0 MoE 非确定性** (§242, 字节相同输入 stochastic argmax) + **(b) per-condition docker fresh state** + **(c) manifest 池微差 (224→222, 2 ep 差异未定位)**。

### 头部指标对照

| 指标 | R24792 (archive) | R32024 (current) | Δ | 解读 |
|---|---|---|---|---|
| N (diag scan) | 224 | 222 | -2 | 2 ep 差异待源码 audit, manifest 池微差 |
| success | 54 | 55 | +1 | |
| **SR** | **24.1%** | **24.8%** | **+0.7pp** | non-gating sensitivity; **vision floor 比 dom 稳** (dom R31194→R21557 Δ=+2.2pp §298) |
| failed | 170 | 167 | -3 | |
| failed_NO_HIT | 50 (22%) | 52 (23%) | +2 | Tier-2 盲区比例稳 |
| success_hit | 9 | 11 | +2 | run-to-run 相位差 (成功 ep 在 P-rule 命中路径不同) |
| **success-hit causal rate** | **0% (9/9 全 presence-only)** | **47% (8/17 hits causal=true)** | **+47pp** | ⚠️ ✨ **run-to-run 大变** — 新发现 |
| benchmark-FP (净) | 0 (task 40/132 翻案) | 0 (task 192 sub-agent 误标修正) | 0 | "vision substrate 干净 → 失败=能力上限" 稳态持续 |
| scaffold-bug | 0 | 0 | 0 | substrate 干净持续 |
| parse_error_rate | 0.0273% (1/3662) | **0.0% (0/3557)** | 0.027 pp | B-1860 修复 hold + R32024 比 R24792 更干净 (单一 invalid_coord ep 消失) |

### Tier-1 P-rule 分布对照 (failed-only)

| Rule | R24792 | R32024 | Δ | 说明 |
|---|---|---|---|---|
| **P31** budget incomplete | **73** | **63** | -10 | 仍头号; vision 高 budget 不完成稳态 |
| **P19** url_match 过早 finish | 50 | **37** | -13 | run-to-run 显著降; 抽样 task 18 verify=true causal |
| **P14** URL 自环 | 35 | **38** | +3 | success-fire causal rate 从 0%→70%! (见下) |
| **P5** 感知缺失循环 | 41 | 32 | -9 | sample task 8 verify=presence-only (上游搜索参数错) |
| P7 sCity=州名 | 22 | 23 | +1 | 稳; task 69 verify=causal true |
| P17 click-back 振荡 | 20 | 15 | -5 | |
| P20 评测目标页从未访问 | 16 | 20 | +4 | |
| P18 cheapest 漏排序 | 13 | 11 | -2 | |
| P25 跨站任务跳过 | 11 | 11 | 0 | 稳 |
| P10 跨步数值记忆 | 5 | 9 | +4 | |
| P23 oldest 误用 price | 9 | 9 | 0 | 稳 |
| P27 找不到即放弃 | 8 | 4 | -4 | success-hit 比例下降反映 vision 措辞 run-to-run 漂 |
| P12 从不翻页 | 6 | 6 | 0 | 稳 |
| P24 不确定仍 finish | 4 | 1 | -3 | |
| P28 benchmark-FP 货币 | 2 | 3 | +1 | |
| P30/P32/P22/P29 | 各 ≤1 | 各 ≤1 | tail | |

### success-hit per-rule (P-rule FP source) 对照

| Rule | R24792 success-hit | R32024 success-hit | causal in R32024 |
|---|---|---|---|
| **P14** | (n=R24792 9 ep total presence-only) | **7** | **4/7 = 57% causal** (task 52/100/101/103 homepage dropdown 卡死真因) |
| P5 | (R24792 全 presence) | 4 | 2/4 = 50% causal (task 179/186 真 stuck) |
| P17 | (R24792 全 presence) | 2 | 1/2 = 50% (task 52 真 14761 振荡 / task 205 |AND| URL 误报) |
| P31 | (R24792 全 presence) | 2 | 0/2 = 0% (url_match 在 budget 耗时凭 final URL 通过) |
| P10/P18/P27 | (R24792 全 presence) | 各 1 | 各 0% 误报 |

### Run-to-run 拆解 + paper implication

**贡献来源**:
1. **B0 MoE 非确定性** (§242, §298): 字节相同输入 stochastic argmax — 跨步选择 random walk → vision 决策树某分支 stochastic
2. **per-condition docker fresh restart** (B-1839): 站点 state 互不串扰 → 每 run cart/comment/listing 空
3. **manifest 池微差** (224→222 vs 224): 2 ep 缺失待 audit (单一非阻塞)
4. **AMENDMENT_07 影响 = 0** (vision 不读 SoM 标注图; sequential-id 在 vision 通路不暴露)

**关键新发现 (R32024-only, R24792 阶段尚未 expose)**:
- ✅ **vision homepage dropdown 系统性视觉点击卡死** (task 52/100/101/103 各 9-21 步): P14 success-fire **真因果**, vision 无 AXTree 直点 category dropdown → 反复 click 失败靠 select_option / URL 改写突破。**这是 paper §3 vision efficiency 实证** (cost/latency 右尾, dom mode 不会因 AXTree 直点 dropdown)。
- ⚠️ **success-hit causal rate run-to-run 不稳态** (0%→47%): 同 condition 同 ruleset 下 P14 v3 "productive 长停留排除" 在 R32024 表现 closer to dom (~70% causal), 在 R24792 全 presence-only → **success 路径的 click-back/url-loop 模式 run-to-run 漂移大**。paper §3.5 vision-as-baseline 增 sensitivity disclosure: "vision success-path 不稳态, 跨 run 视觉点击失败模式相位差导致 P-rule causal rate 漂动"。

**paper Risk 6 (run-to-run sensitivity) 增量**:
- vision SR Δ=+0.7pp 是 **non-gating noise**, 远小于 dom +2.2pp drop-one (§298)
- vision 比 dom run-to-run 更稳的可能解释: vision 无元素 ID, 不受 SoM nodeId churn (AMENDMENT_07) 影响 → 减少了一类噪声来源
- **dom + vision 双 mode 已有 run-to-run replicate** (R31194+dom-archive R21557 / R24792-vision-archive+R32024 — vision floor demand by §0 ④ "vision floor 仍 gated" 现部分满足)

---

## §1 三分类统计 (本轮 47 ep 深挖)

| 类别 | 数量 | 占比 | 说明 |
|---|---|---|---|
| **agent-limit** | **46** (40 no-hit + 5 verify + minus 1 sub-agent 误标修正) | ~98% of audited | vision 视觉/推理上限, 不可修, paper finding |
| **benchmark-FP** | **0** (task 192 sub-agent 误标 → §4 修正为 agent-limit) | 0% | "vision 失败更纯" 论点持续 hold |
| **unclear** | 0 | 0% | |
| **scaffold-bug** | 0 (47 ep + 222 规则命中均未发现) | — | substrate 仍干净 (B-1860 + AMENDMENT_07 后) |
| **success-hit FP audit** | 8/17 hits causal=true (47% causal) | — | **R24792 0% → R32024 47%** (run-to-run 大变, §0) |
| **failed-hit causal verify (sample)** | 2/5 (40%) | — | P19/P7 真; P31/P14/P5 抽样里 presence-only |

**核心结论**: 与 R24792 一致 — **vision 失败几乎全 agent-limit, 零 scaffold-bug, 零 benchmark-FP**。Substrate 干净 (B-1860 修复持续 hold + parse_error 从 0.027% 进一步降到 0.0%)。**新发现 (§0)**: vision success 路径 P14 causal rate run-to-run 漂移大, 不稳态 → paper sensitivity disclosure 候选。

---

## §2 Tier-1 规则分布 (failed-only, 167 failed; 完整 18 条 hit)

```
P31  63 (budget 耗尽 incomplete)        ████████████████████████  ← vision 头号 (持续)
P14  38 (URL 自环)                       ███████████████
P19  37 (url_match 过早搜索页 finish)    ███████████████
P5   32 (感知缺失循环)                   █████████████
P7   23 (sCity=州名)                     █████████
P20  20 (评测目标页从未访问)             ████████
P17  15 (click-back 振荡)                ██████
P18  11 (cheapest 漏价格排序)            ████
P25  11 (跨站任务跳过其中一站)           ████
P10   9 (跨步数值记忆)                   ███
P23   9 (oldest 误用 price)              ███
P12   6 (从不翻页)                       ██
P27   4 (找不到即放弃, vision 扩展)      █
P28   3 · P22 1 · P24 1 · P30 1
```

**mode-gate 验证 ✅**: P6/P15/P16/P21 全 0 命中 (mode != "dom" gate 生效).

**P22 1 命中反常** (P22 应 dom-only "图上数字 dom 不可读"): 待源码审核 (可能漏 mode-gate, 或边界 carveout), 单一非阻塞, 标 `master_bug_catalog` 候选 follow-up (B-186x).

> ⚠️ per-rule 非 causal; 本轮 verify (5 samples) 显示 P31/P14/P5 在 vision 上 presence-only 倾向较高, P19/P7 真 causal — 与 dom R31194 §6 P14 70% presence-only 趋势可能相反 (vision 上 P14 真因果反多), 但样本 n=5 太小不下定论, 留待全 mode causal verify 后定。

---

## §3 Tier-2 新发现 (47 ep audit)

### 3.1 no-hit 40 (deterministic 盲区, 全 agent-limit)

vision-only 失败子类型分布 (本轮 audit):

| 子类型 | 代表 task (本轮) | 机制 |
|---|---|---|
| **图像语义误识别** | 49 (蓝色屏幕→蓝色 LED 物理灯混淆) · 50 (gallery 红色调色板誤识) · 56 (snowblower→snow ski 装备语义滑) · 63 (Iron Man 图→"Cast Iron Collection" 文字字面) · 65/67-类 roleplay 场景误识 · 208 (昆虫识 ladybug, ref=moth/butterfly) | 视觉→文字关键词转换失真; 无 AXTree alt-text 兜底 |
| **细节视觉误判** | 12 (摩托 black/red 误判) · 22 (gallery 红车定位错落 Ford 8400 vs 实际 103K miles) · 119 (钞票 $50→$1, Grant→Washington 误读) · 150 (price OCR $229→$249, "二九"/"四九"混淆) | 缩略图分辨率天花板 |
| **gallery 行列计数** | 14 (gallery 第二行错位) · 41 (gallery 1-step finish 价格盲判) · 146 (4 books 误数 6 books) | 缩略图密集时行列定位偏差 |
| **意图-坐标不一致** | 0 (plan-execute 解耦, thought 否定 click 但 action 仍 click) · 129 (soccer cleats coord click → smart watch item) | thought / coord 解耦; 像素坐标偏差 |
| **页面约束违反** | 78 (start_url iPage=4 被 sCity filter 离开) · 134 (Xbox games p5 被搜索 jersey 离开) | "on this page" / start-page-relative 约束被忽略 |
| **图像约束系统忽略** | 172 (蓝椅子 image 约束: agent 自承"无 visible 蓝椅子"仍 finish 最便宜) · 181 (golden mask image 忽略选最贵 pinball) · 184 (Charizard 找不到选金条) · 185 (Toronto 球队 image 未验证) | reference image 视觉匹配约束被 price/sort 优先策略覆盖 — vision 特有 high-frequency |
| **裸 PNG URL finish** | 132 (click 主图导航到 `/oc-content/uploads/*.png`, finish 在 PNG URL) | vision 无 AXTree 区分 link type → 点图链跳到 raw file (P-rule 候选 P-cand-A, §5) |
| **多步表单状态丢失** | 217 (评论 title 重填后状态丢失) · 208 (评论 body 视觉误判) | scroll 后表单字段状态无 AXTree 追踪 |
| **不排序/约束遵循** | 20 (intent 'most recent' 但无 sOrder, 未切排序直接 finish) · 35 (loveseat 任务无 sCategory filter) · 211 (排序第一不验证内容选了 shipping container) | 任务语义约束遵循率低 |
| **vision text-blind 放弃答** | 98 (滚动 3 次后 finish answer="engine hours not mentioned", ref="80"; 列表描述小字读不到) | vision 模式 small-text OCR 系统性盲区 (P-cand-C, §5) |

### 3.2 success-hit 11 FP audit (混合 causal — R24792 0% → R32024 47%)

**真因果 (8/17 hits)** — vision homepage dropdown 视觉点击系统性卡死:

| task | causal hits | 模式 |
|---|---|---|
| 52 | P14 + P17 | homepage 19 步 click 卡死 + item 14761 反复 4 次 |
| 100 | P14 | homepage 14 步 click 卡死 (vision 点不中 category dropdown) |
| 101 | P14 | homepage 9 步 click 卡死, 最终 select_option 突破 |
| 103 | P14 | homepage 11 步 click 卡死, 同上 |
| 179 | P5 + P14 | 搜索页 click 失败 8 步 靠 scroll 恢复 |
| 186 | P5 + P14 | back 后搜索页底 scroll 卡死, 翻页恢复 |

→ **vision 模式 homepage dropdown 视觉点击系统性卡死** = paper §3 efficiency 实证 (cost/latency 右尾分布, dom mode 因 AXTree 直点 dropdown 无此模式)。**这是 R24792 阶段 sub-agent 全标 "presence-only" 未识别的 vision-only stuck pattern** — R32024 sub-agent 在更细 task-by-task 审计下识别为真 causal。

**Presence-only (9/17 hits)**:
- task 5 P5+P27 (delete 副作用 success, list scroll 边界正常; "could not be found" 措辞误判)
- task 87 P31 (停在正确 item 页消耗 budget, eval url_match 凭 final URL 通过)
- task 151 P5+P14 (已到正确 item, 后续 click 为 post-nav confirm)
- task 205 P17 (|AND| cross-site URL 的 back 是 task setup 逃脱非 detail↔list 振荡)
- task 209 P18 (cheapest 在 quantity-discount 是量价折算非 cheapest-item)

### 3.3 failed-hit causal verify (5 samples, 40% true)

| task | tier1_rule | causal | real root cause |
|---|---|---|---|
| 9 | P31 | **false** | vision 价格 OCR $795→$799→$790 + 上传 dialog 9 次 click 失败 (上游 root, P31 末端症状) |
| 6 | P14 | **false** | 跨步记忆丢失 — 访问 3 个 URL 但 finish 只交 1 个 (上游 root, P14 头部症状) |
| 18 | P19 | **true** | vision 搜索结果缩略图识别 blue iPhone 失败 5 scroll 后 finish 在 search 页 (P19 描述吻合) |
| 8 | P5 | **false** | 搜索参数错 (用 `query=` 非 `sPattern=`) 上游致 13 次 scroll 在错结果页 (P5 末端) |
| 69 | P7 | **true** | step_3 type 'Delaware' 入 sCity 后 27 步 无救; sCity=州名 真根因 (P7 吻合) |

**含义**: P31/P14/P5 在 vision 上 presence-only 倾向较高 (sample 3/5 false); P19/P7 真 causal — n=5 太小不下定论, 留待全 causal verify。

---

## §4 forensic / sub-agent 误判修正

**task 192 修正: benchmark-FP → agent-limit**
- sub-agent (batch 6) 标为 benchmark-FP, 理由: "agent 答 'red' 但 must_include=['red','white'], partial match → eval 严"
- ❌ 错: multi-value `must_include` 评测语义 = **两词都要在 answer 里** (string_match 标准行为); agent 仅说 'red' 忽略 'white' = **agent 输出不全, 评测正确**
- ✅ 修正分类为 **agent-limit**; 体现 vision 模式 "multi-attribute finish 倾向只输出最显著一个属性" 的 systematic limitation (与 R24792 task 40 / 132 sub-agent log-only 误判模式同源 — log-only judgment 易过判 FP, **artifacts forensic / 评测语义复核才能纠正**)
- **净 benchmark-FP 0** (与 R24792 一致)

> **方法论教训 (R24792 §3 已注 + R32024 实证)**: log-only sub-agent FP 判定有系统性偏置, **多值 must_include 不知道 OR vs AND 语义** + **agent thought 自信声明易误导** → 用 sub-agent 判 FP 必须 cross-check (a) 评测语义 (eval_type + must_include shape) (b) artifacts forensic (截图 ground-truth). R32024 task 192 仅 cross-check (a) 即翻案, 不需 artifacts。

---

## §5 Self-evolving — P-rule 候选 (本轮 **不落码**, 累积 freeze 前一次性)

**决策 (2026-05-26)**: 本轮发现 6 个候选, 不立即 bump ruleset version — discover 阶段仅 vision 一 mode 新数据点, **统一等 phantom/reddit/B1/B2 跑完一次性 freeze 落码 + 全量重扫** (避免每 condition 一次 bump 造成版本碎片化)。当前 `4-domsomvis-b1860coord` 不动。

| 候选 | task | signal (0-token) | 类别 | 推荐 |
|---|---|---|---|---|
| **P-cand-A: vision PNG-finish** | 132 | `obs_url[-1] ~ r'/oc-content/uploads/.*\.png$' AND action_type[-1]=='finish'` | agent-limit | 🟡 hold — vision-only / single instance, 待累积 |
| **P-cand-B: 缺 iPage 离开 start_url** | 78, 134 | `start_url 含 iPage=N AND step_0 obs_url 无该 iPage AND obs_url 新增 sCity\|sPattern` | agent-limit | 🟡 hold — 待累积 cross-condition |
| **P-cand-C: vision text-blind 放弃答** ⭐ | 98 | `finish.answer ~ r"not (mentioned\|available\|provided\|visible\|specified)" AND eval_type=string_match` | agent-limit | ✅ **强推荐**: vision 高频, 0-token, success-safe (success ep 不会用 "not mentioned"), 横跨 vision-mode 通用 |
| **P-cand-D: gallery 1-step finish wrong** | 14, 41 | `start_url~'page=search' AND steps≤2 AND action_type[-1]=='finish' AND finish≠ref` | agent-limit | ❌ 不推荐 — `finish≠ref` 引入 eval-time 依赖 (现有 P-rule 不读 ref); 改 signal 用 `finish.answer` 短文本特征待研究 |
| **P-cand-E: intent='most recent' 无 sOrder 切换** ⭐ | 20 | `intent 含 'most recent\|recent'(re) AND ∀step: obs_url 无 sOrder=dt_pub_date AND no select_option action AND steps≤4` | agent-limit | ✅ **强推荐**: 跨 mode 候选 (dom/som 也会有), 待 cross-mode 数据齐验证 |
| **P-cand-F: image-constraint ignore** | 172, 181, 184, 185 | 难 0-token 化 (需任务 intent NLP) | agent-limit | ❌ 不落码 — agent-limit paper finding 类, 文字描述足够 |

**累积 freeze pre-flight checklist**:
- [ ] phantom_text + phantom_som + phantom_prompt B0 cls 跑完 → 候选 A/B/C 增累
- [ ] B1 cls + B2 cls 各 6 mode 跑完 → 候选 E 跨 model 验证
- [ ] reddit 各 cell 跑完 → 候选 B/C 跨 site 验证
- [ ] freeze: bump ruleset 到 `5-6mode-paper1a` 或类似 → 全量重扫所有 condition → 验证 success-fire 全 0 → SKILL.md 同步

---

## §6 Actionable

1. ✅ **Tier-1/2/3 digest 完成** (本文件; archive sibling [[B0_vision_classifieds_R24792_diag_digest]] 保留; dual-digest run-to-run preserved per user 2026-05-26)
2. ✅ **task 192 sub-agent 误标修正** 为 agent-limit (多值 must_include 评测语义)
3. ⏳ **P-cand-C/E 累积观察** → freeze 前一次性 bump ruleset
4. ⏳ **P22 1 命中反常调查** (P22 应 dom-only): 单一非阻塞, 标 master_bug_catalog 候选 B-186x — `mode != "dom"` gate 是否漏挂或边界 carveout
5. ⏳ **paper §3.5 vision-as-baseline sensitivity disclosure**: success-hit P14 causal rate run-to-run 漂移大 (R24792 0% → R32024 47% causal); vision SR Δ=+0.7pp non-gating 稳; benchmark-FP 持续 0 stable
6. ⏳ **diag scan 222 ep vs raw 224 ep 差 2 ep 源码 audit** (单一 note, manifest 池微差)
7. ⏳ **cross-mode 仍禁** (3/6 mode + 本 R32024 替换 R24792 作为 canonical-vision, R24792 archive 用于 run-to-run sensitivity 而非 cross-mode aggregation)

---

## 附录

- **B-1860 telemetry hold**: R32024 `parse_error_rate=0.0%` (vs R24792 0.027% / R3671 pre-fix 13.6%) — 坐标契约修复持续生效, vision substrate 进一步干净
- **AMENDMENT_07 影响 = 0** on vision (SoM nodeId 在 vision 通路不暴露)
- **B0 MoE 非确定性** (§242, §298) = vision run-to-run 差异主要来源
- **B-1839 per-condition docker restart** = run-to-run fresh substrate (cart/listing/comment 空) — 与 §0 ④ "每 condition 落地跑 diag" 协议吻合
- **Tier-2 token**: 8 sonnet sub-agents × ~7 ep avg ≈ 56 ep audit / ~720K total token (6 no-hit × 7 ep + 1 FP × 11 ep + 1 verify × 5 ep)
- **diag JSON**: `/tmp/B0_vision_classifieds_R32024_diag.json` (gitignored)
- **run-to-run sibling**: archive R24792 → [[B0_vision_classifieds_R24792_diag_digest]]; dom/som sibling → [[B0_dom_classifieds_diag_digest]] / [[B0_som_classifieds_diag_digest]] (不同 run id, 同 ruleset)
