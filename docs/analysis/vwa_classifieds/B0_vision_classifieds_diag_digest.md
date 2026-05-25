# B0 vision classifieds — 失败错因 digest（diag skill）

> **生成方式**: `/diag` skill 3-tier pipeline (2026-05-25 run on R24792)。Tier-1 deterministic 全扫 (`diag_pattern_match.py`, 0 token, ruleset `3-domsom-b1860coord`) → Tier-2 Claude sonnet sub-agent 深挖 (50 no-hit 全覆盖 / 7 agents + 9 success-hit FP 审计 / 1 agent = 59 ep / 8 agents) → Tier-3 整合 (本文件)。
> **Run**: `B0_vision_classifieds_20260525_011025_904578573_386602_R24792` (Gate-3 fresh substrate, per-condition docker restart, **B-1860 coord contract 修复后首个 vision run**, manifest-bound authoritative)
> **Condition**: `phase1_vision_router_0` | site classifieds | mode **vision** (agent 只看裸截图, 无 AXTree 文本) | model **B0 = Qwen3-VL-235B (proxy)**
> **规模**: N=224 ep | success=54 | failed=170 | **SR=24.1%** | total_steps=3662
> **ruleset_version**: `4-domsomvis-b1860coord` (本轮落码 P31/P32 + P27 扩展/carveout 后 bump; **dom R31194 / som R9725 已全量重扫到同版本** 2026-05-25, P27/P31/P32 success-fire 全 0 验证 → SKILL discover-then-freeze 硬纪律 2 满足, 但 phantom/reddit/B1/B2 未跑仍只 3/6 mode, **禁 cross-mode 定量比较**)。

> ⚠️ **定位声明 (沿用 dom/som 3-AI 审计共识)**: 本 digest 是 **internal 诊断记录, NOT paper-grade 结论**。
> - **单 condition + vision-only + 无对照**: SR / 失败分布只描述 B0 vision cls 自己。"vision 表征局限 / routing 论点 / 换表征能救"需 dom/som/phantom 对照 (当前 ruleset 含 vision 但仅 3/6 mode 已 diag, **禁止 cross-mode 定量比较**)。
> - **presence ≠ causation**: 170 failed 中 **50 no-hit + 0 failed-hit 逐个证因** (本轮 Tier-2 全力投 no-hit 盲区, failed-hit 未单独 causal verify — P19/P5 等主导规则的命中-死因映射待补; 见 §6)。
> - **per-rule 非互斥**: 分布是 per-episode-per-rule 命中, P19∩P5 等重叠, 勿各行相加。
>
> **paper failure-analysis 待 6-mode + 多 condition 数据齐 + 全量重扫统一 ruleset 后重做, 不复用本 digest 数字。**

---

## §0 B-1860 坐标契约修复验证 ⭐ 本轮 diag 核心目的

> next_steps §0 ④ 明确: vision diag 验证 "parse_error 13.6%→正常 + true_oob no-op telemetry"。R24792 = B-1860 contract APPLIED (merge `d977006` + amendment 05) 后首个 vision run。**结论: 修复完全生效, 铁证如下。**

| 指标 | pre-fix (R3671, archived) | post-fix (R24792) | 判定 |
|---|---|---|---|
| **parse_error_rate** | **13.6%** | **0.0273% (1/3662)** | ✅ 降 ~500× |
| **tool_call_invalid** | (高) | 0.0273% (1/3662, 同一 step) | ✅ |
| **b1860_coord_residual** (Tier-2 逐 ep) | — | **0 / 59 深挖 ep** | ✅ 7 agent 独立确认 |
| **glm_fallback_attempted** | — | **0** | ✅ B-991 retired 生效 |
| **coordinate telemetry** | 无 | regime/scale/recovered/oob 字段落盘 | ✅ 工作 |

**坐标 telemetry 结构** (B-1860 新增, 落 `action_executed.coordinate_normalization`):
```json
{"x_regime": "qwen_0_1000", "y_regime": "qwen_0_1000", "x_scale": 1000.0, "y_scale": 1000.0,
 "recovered": true, "true_oob": false, "dead_zone": false, "malformed": false}
```
→ Qwen 的 0-1000 坐标系被正确**识别** (regime) + **映射回 [0,1]** (scale=1000) + **恢复成功** (recovered=true)。这是 B-1860 修复的核心: pre-fix Qwen 0-1000 坐标被当 [0,1] 用 → 点击全部挤在左上角 ~0.1% 区域 → parse/grounding 崩。

**唯一残留 invalid_coord = fail-loud no-op 范例** (`task 9, step 29`):
- `parse_valid=False`, `tool_call_valid=False`, `parse_failure_reason=invalid_coord`
- **`action_executed=None`** ← 坐标无法恢复时 **no-op (不点击垃圾位置)** + 标记 invalid，而非 fail-silent garbage click。
- 这正是 "true_oob no-op telemetry" 的体现: 真无效坐标 → 拒绝执行 + 留痕, 非静默点错。

**error_category 全量分布** (3662 steps): `no_progress=764` (agent 行为层, 非 scaffold) / `policy_blocked_offsite=7` / `invalid_coord=1`。**无 parse/坐标类 scaffold flooding** = B-1860 后 vision substrate 干净。

---

## §1 三分类统计

| 类别 | 数量 (深挖 59 ep) | 占比 | 说明 |
|---|---|---|---|
| **agent-limit** | **49** (no-hit) | 98% of no-hit | vision-only 视觉/推理能力天花板, 不可修, **paper finding** |
| **benchmark-FP** | **0** (task 40 forensic 翻案) | 0% | 截图证据: ref LG 实际更 recent + agent 没用 vision 判材质 → agent-limit (§4) |
| **unclear** | **1** (task 129) | 2% | task 132 forensic 倾向 agent-limit (兔 brown/tan 非纯黑) |
| **scaffold-bug** | **0** | — | **59 深挖 + 129 规则命中均未发现可修代码 bug** |
| **success-hit FP** | 9/9 hit_causal=false | 100% 误报 | 全部成功 ep 的规则命中均为 presence-only |

**核心结论**: B-1860 修复后, vision 失败**几乎全是 agent-limit** (模型看图能力上限), **零 scaffold-bug**。这是 paper §3-§4 "vision baseline 局限 → phantom-SoM improvement potential" 的直接 evidence。

---

## §2 Tier-1 规则分布 (failed-only, 170 failed)

> **全量重扫 v3 (`4-domsomvis-b1860coord`, 2026-05-25)**: 新增 **P31=73** (budget 耗尽/incomplete — **超过 P19 成头号失败维度!** vision 失败大量是 30 步用尽未完成, 效率浪费维度) · P32=1 (text-in-price) · P27 2→8 (ABANDONMENT 扩展)。v3 完整: **P31 73** / P19 50 / P5 41 / P14 35 / P7 22 / P17 20 / P20 16 / P18 13 / P25 11 / P23 9 / P27 8 / P12 6 / P10 5 / P24 4 / P28 2 / P32 1 / P22 1 / P29 1。下方 ascii bar 是落码前 (`3-domsom-b1860coord`) 快照, 保留作对照。

```
P19  50 (url_match 过早在搜索页 finish)   ████████████████████  ← vision 头号
P5   41 (感知缺失循环)                     ████████████████
P14  35 (URL 自环)                         ██████████████
P7   22 (sCity=州名)                       █████████
P17  20 (click-back 振荡)                  ████████
P20  16 (评测目标页从未访问)               ██████
P18  13 (cheapest 漏价格排序)              █████
P25  11 (跨站任务跳过其中一站)             ████
P23   9 (oldest 误用价格排序)              ███
P12   6 · P10 5 · P24 4 · P28 2 · P27 2 · P22 1 · P29 1
```

**mode-gate 验证 ✅**: dom-only 视觉天花板规则 **P6/P15/P16/P21 全 0 命中** (它们有 `if mode != "dom": return []`)。vision 能看图 → 不触发 "DOM 看不到颜色/图像" 类规则。这印证规则库 mode-specific 分层 = router 论点证据基础。

> ⚠️ per-rule 非 causal: P19/P5/P14 是表层命中, 本轮**未对 failed-hit 做 causal verify** (Tier-2 全投 no-hit)。"P19 主导" 应读作 "P19 在 50 个 failed 命中", 非 "50 个因 P19 死"。

---

## §3 Tier-2 新发现

### no-hit 50 (deterministic 盲区, 全覆盖逐 ep 证因)

47/50 agent-limit, vision-only 失败子类型 (≠ dom 的 P14 自环 / P6 视觉天花板):

| 子类型 | 代表 task | 机制 |
|---|---|---|
| **图像语义误识别** | 65/67 (roleplay 场景误识别 soccer/basketball → 落同一 wrong item 15671) · 199 (URL OCR 读成 OsClass 平台水印, 漏 listing 内 kaiyo.com) · 208 (昆虫识成 ladybug, ref=moth/butterfly) | vision 无 AXTree 锚点, 纯截图推理被语义干扰 |
| **颜色/细节视觉误判** | 12 (摩托识成 black, ref=red) · 32 (价格 OCR $395 读成 $300, comment 报错值) · 117 (服装颜色误判 blue) | 截图细粒度 OCR/颜色精度不足 |
| **浅搜索即 finish** | 49 (列表页直接答 RAM=8, 未进详情页 ref=64) · 48/58/59 (只取列表首项/子集, 不比对) | 不进 item 页验证 / 不穷举 |
| **意图-坐标不一致** | 60 (thought 说点第 2 个 racing sim, 坐标命中第 1 个 pinball) · 22 (说 second-row 红车, 点到 Porsche) | vision list-item 位置估计误差 |
| **多步推理链断裂** | 185 (城市识成 Toronto → 队名 → collectible 三步全错) · 184 (Charizard "最贵" 误解为全局最贵) | multi-hop, 一步错步步错 |
| **不排序/任务理解偏差** | 20 (不按日期排序点首项) · 43 (page-4 已有 red 车却去搜索) · 78 ("on this page" 约束被忽略, 离开指定页) | 语义约束遵循失败 |

### success-hit 9 FP 审计 (9/9 纯误报, hit_causal=false)

全部成功 ep 的规则命中均 presence-only。两类机制:
- **Type A — productive 长停留误判** (task 87/118): agent 在**正确 item 页** scroll 确认 (obs_url 不变但 page scrolled), P14 仍 fire。
- **Type B — vision 操作受阻后策略切换** (task 5/101/103/166/170): vision 无法点中 dropdown/thumbnail → click fail streak (P5/P14/P10/P17 fire) → agent 改用 select_option / URL 构造 / 换关键词 **侥幸成功**。

> 🔑 **Type B latent-risk trace**: 这些 ep 虽 success, 但 "险中求胜" (如 task 101 homepage 21 步 click fail 后才突破), 是 vision 模式的 **efficiency waste 痕迹** (非 gate-fail)。值得 paper §3 标注: vision 即使成功, cost/latency 分布右尾更重 (待 cross-mode 量化)。

---

## §4 代表 episode

**agent-limit / 图像语义误识别**
- **task 65 + 67** (同源 pattern): reference image 是 roleplay 场景, agent 误识别为 soccer (65) / basketball (67), 两个都落到 `id=15671` ("Video Game collection") wrong item。→ vision 跨模态图像→搜索词映射失败的可复现实证。
- **task 199**: 1 步 finish, finish answer="OsClass" (站点平台水印), ref="kaiyo.com" (listing 图内文字)。vision OCR 混淆 chrome UI 与 listing 内容。

**agent-limit / 意图-坐标不一致**
- **task 60**: `step_5 thought='second listing is Racing Simulator $2450... I should click on it'` 但 `coordinate=[496,476]` 命中第 1 条 Pinball (id=7915); step_6 用 Pinball 作答。→ vision list 位置估计偏差 (非 B-1860 坐标 bug — telemetry recovered=true)。

**benchmark-FP 候选 → artifacts forensic 核查结论 (2026-05-25, 截图证据)**
- **task 40 → ❌ 翻案为 agent-limit** (非 FP)。排序列表截图 (step_007/008, `dt_pub_date desc`) 显示全部 listing 同为 2023/11/16, 但降序下 **ref LG(18607) 排在 GE(70627) 之前** = LG 更 recent, 且 LG 缩略图确为不锈钢色。agent **scroll 过 LG** (step_007 截图清晰可见) 却依赖标题文字 (LG 标题未写 "stainless steel", GE 写了) 跳过 LG 选更旧的 GE → **vision 未用视觉判断材质 = agent-limit**。⚠️ sub-agent (log-only) 误判 FP, **artifacts forensic 纠正** = presence≠causation 在 FP 判定层的范例 (agent thought 自信说 "GE is most recent" 但其自我认知错误, 只有截图能 ground-truth)。
- **task 132 → ⚠️ 倾向 agent-limit** (颜色误判)。id=21697 = "Netherland Dwarf Rabbits", step_002 截图 3 只兔为 brown/tan + 深色 **非纯黑**, agent 当 "3 black animals"。ref=69201 (未访问)。url_match EXACT 对模糊视觉 task 偏严的 note 保留, 但主因 agent 颜色判断偏差。
- **净结论: B0 vision benchmark-FP ≈ 0** (不像 dom 有 B-21 货币 tokenize 真 FP) → 不需 task 排除 / 上游 eval 修。失败更纯 agent-limit, 强化 "vision substrate 干净 → 失败=能力上限" 论点。

**success-hit FP**
- **task 5**: P14 误报。delete 任务, `step_02` click Delete + accept confirm dialog (page_changed=True) 删除已发生; step_03-29 是 vision 看不到刷新后 list 的多余 scroll (obs_url 全程 page=user&action=items); eval=program_html 查 id=84144 返回 404, **与最终 URL 无关** → success=true。P14 命中 30 步同 URL 但非死因。
- **task 101**: P5/P14 误报。homepage 21 步 click fail (vision 点不中 category dropdown) → `step_22` select_option 突破 → step_28 finish id=83533 success。no_op_rate=0.72 但最终成功。

---

## §5 Self-evolving — 落码 P31/P32 + 扩展 P27 (本轮已落, 全量重扫验证)

> **决策 (2026-05-25)**: 原"留 freeze 统一落"经复议改为**本轮落码** — discover-then-freeze 的 discover 阶段本就是每 mode 落码建字典 (dom 落 P15-P23, som 落 P24-P30), vision 落符合流程。**落码前 cross-check 现有规则避免重复** (发现 P19 已覆盖 url_match-finish, P27 已是 abandonment) → 实际落 **2 新规则 + 扩展 1 现有**, 质量优先非数量。

**已落码 (success-safe: dom/som/vision 三 condition success-fire 全 0 验证)**:

| 规则 | 类型 | signal (0-token) | success-safe 机制 | 命中 (failed) |
|---|---|---|---|---|
| **P31 budget耗尽未完成** | 新 | `summary.trajectory_incomplete` | program_html-404 carve-out (task 5 delete 副作用 success 排除, 同 P20) | vision 73 / dom 58 / som 49 |
| **P32 文本误入价格filter** | 新 | obs_url `sPrice(Min\|Max)=[^&]*[A-Za-z]` | 天然 (malformed URL 不入成功轨迹) | vision 1 (task 34) |
| **P27 找不到即放弃** | 扩展 | ABANDONMENT_RE +`no <noun>...is visible/was found`; + ref-carveout | finish obs_url==ref item 不 fire (task 151 url_match 主观放弃但停对页, 同 P24/P30) | vision 8 / dom 4 / som 9 |

**未落码 (cross-check 后剔除, 避免规则库膨胀/双计)**:
- ~~finished_on_list_page~~ (task 49/78) — **P19 已覆盖** url_match-on-search-page; string_match 版本 success 风险高 (答案可能 list 可见, e.g. task 233)。
- ~~visual_1step_finish~~ (task 199) — success 风险 (task 233 类 1 步成功合法); 需 finish≠ref 逻辑, 留待。
- ~~scroll_loop_no_finish~~ (task 136) — **P31 (incomplete) 已涵盖**; final_url=search 仅 detail 增益。

**RULESET_VERSION** `3-domsom-b1860coord` → **`4-domsomvis-b1860coord`**; SKILL.md「当前 P-rules」(30 条 P1-P32) +「当前相位」(3-mode discover) 已同步。**全量重扫教训**: 扩展 P27 正则在 dom task 151 引入回归 → 全量重扫 (非仅改动来源 mode) 才发现 → 改正则必全 condition 重扫。

---

## §6 Actionable

1. ✅ **benchmark-FP forensic 核查 (完成)**: task 40 翻案 agent-limit (截图: ref LG 实际更 recent + agent 没用 vision 判材质) + task 132 倾向 agent-limit (兔 brown/tan 非纯黑)。**净 FP≈0** → 不需 task 排除 / 上游修 (详 §4)。

2. ✅ **P-rule 落码 + 全量重扫 (完成)**: 落 P31 (budget) + P32 (text-in-price) + 扩展 P27 (vision 措辞 + ref-carveout); bump → `4-domsomvis-b1860coord`; dom/som/vision 全量重扫 success-fire 全 0; SKILL.md 同步 (详 §5)。

3. ⏳ **P14 v3 scroll_changed 豁免复核** (vision-specific refinement, 未做): success-fire Type A (task 87/118) 揭示 P14 v3 "productive 长停留排除" 当前可能只看 `url_changed`, 未 OR `scroll_changed` → vision 在正确 item 页 scroll 确认 (obs_url 不变) 仍被判自环。建议: v3 productive 判定 OR 包含 `scroll_changed=true`。改 → bump version → 全量重扫。

4. ⏳ **failed-hit causal verify 补做** (本轮缺口): P19 (50) / P5 (41) / P14 (35) 主导规则未做 presence→causation 验证。cross-mode 聚合前需补 (抽每主导规则 2-3 ep)。

5. ⏳ **cross-mode 解锁前置** (仍未满足): 三 condition 已落同版本 `4-domsomvis-b1860coord` (硬纪律 2 ✅), 但只 **3/6 mode** (phantom×4/reddit/B1/B2 未 diag) → **仍禁 cross-mode 定量比较**, 须等 6-mode discover 完 + freeze。

---

## 跨 run 一致性 / 附录

- **B-1860 telemetry 是 runner 层** (`coordinate_normalization`), 不是 diag rule 层 → `-b1860coord` 后缀主要标记 "扫的是 B-1860 后 substrate", ALL_RULES 规则集 (P1-P30, 无 P9/P26) 与 `3-domsom` 同。但版本号既已分叉, cross-mode 前仍须全量重扫对齐 (透明性优先)。
- **SR 24.1%** 仅记录, **不与 dom R9755 14.7% 比较** (禁 cross-mode)。
- Tier-2 token: 8 sonnet agents, ~59 ep, 合计 ~746K token (no-hit 7 agents + FP 1 agent)。B-1860 量化验证 (parse/oob/bbox 聚合) 由父 context 0-sub-agent-token 完成。
