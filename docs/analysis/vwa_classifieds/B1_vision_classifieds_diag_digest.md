# B1 vision classifieds — /diag failure attribution digest

| field | value |
|---|---|
| **Run** | `B1_vision_classifieds_20260605_012235_349047872_327631_R28622` (manifest-bound authoritative) |
| **Condition** | `phase1_vision_router_0` |
| **Site / Model / Mode** | classifieds / **B1 (Qwen3-VL-4B)** / **vision** (raw screenshot only, no AXTree) |
| **N episodes** | 224 |
| **SR** | **12.5%** (28/224 success) |
| **ruleset_version** | `5-domsomvispsom-b1860coord` ⚠️ **discover-only run — 未落码新规则, 未 bump version**; cross-mode 定量比较禁止直至 freeze |
| **Tier-2 深挖** | 48 episode / 7 sonnet sub-agent (25 no-hit 全覆盖 + 11 success-hit FP 审计 + 6 P5 机制因果 + 6 P19/P18 verify) + **4 截图 forensic** (task 0/1/40/20) |
| **姊妹 condition** | §317 (B1 som) · §318 (B1 dom) — B1 cls cross-mode discover 三联 |

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
| SR | **12.50%** (28/224) |
| failed + hit | 170 |
| **failed NO-hit** | **26** |
| success + hit | 1 |

v8 新规则 failed 侧: {'P46': 4}；success 侧: 无。
（`P43` 在 cls 上大量命中属预期 —— 它标记"intent 需要视觉 + 该 mode 无页面截图"这一**中性组合**，
并非预测失败；§387.10 实测补上截图的增益 ≈0。）

全部 36 个 canonical condition 现处同一版本 → **cross-mode / cross-site 聚合解锁**。

---
## 1. 三分类统计 (Tier-1 hit + Tier-2 深挖合并)

| 类别 | 占比 (深挖样本) | 说明 |
|---|---|---|
| **agent-limit** | 压倒 (~44/48 深挖) | 4B 视觉弱模型主导; 子型见 §4 |
| **scaffold-bug** | **0** | 48 深挖 0 命中; parse/tool_call 全 valid; P8 (唯一 is_scaffold) 0 fire |
| **benchmark-FP** | 1 confirmed (task 20) + 候选 | url_match EXACT wrong-item 歧义 (§7) |
| **rule-FP (presence-only)** | 大量 (见 §3/§5) | success-hit 11/11 全误报 + P19/P18 诊断 FP + finish-less artifact |

**一句话**: B1 vision 失败几乎全是 **4B 模型在纯截图下的能力极限** (视觉 grounding / 细粒度 OCR / 导航恢复 / 多跳规划), scaffold 干净, benchmark-FP 极少。失败画像比 dom (§318) 更"视觉化": detail-OCR 幻觉 (时钟/钞票/球衣号/数量) 成新主轴。

---

## 2. Tier-1 规则分布 (failed-only, episode-level)

```
P5=123  P31=128  P14=82  P19=71  P18=36  P20=22  P12=17  P7=15  P17=13
P25=12  P10=7  P27=6  P24=4  P28=3  P22=2  P13=1  P30=1  P29=1
```
no-hit failed = **25** (deterministic 盲区, coverage 87.2%); success-fire (FP源) = P5×7 P14×6 P31×5 P10×2 P12×2 P17×1 P25×1。

⚠️ **P5 + P31 表层主导 = 大量 presence-only 噪声** (见 §3 + §5)。裸读 per-rule 会严重误判 B1 vision 失败结构。

---

## 3. 🔑 ROUTER 关键 finding — dom→vision route 能否消解 P5? (本 diag 的科学目的)

**§318 假设**: B1 dom 死于 P5 感知盲循环 (AXTree 不反馈 action → 空转); 换 vision (给截图) 应消解 → 干净 routing 因果链。
**Tier-1 表层**: P5 在 vision **没消失, 仍 #1** (123/196 failed)。
**Tier-2 + 截图 forensic 机制判定 → 结论 = REFINE (非成立非全崩)**:

### (a) ✅ route-win 实证: "AXTree 丢失的已渲染文字" 型 (task 40, **截图确认**)
- **task 40 "stainless steel dishwasher 品牌?"**: **dom 30 步失败** (进 LG 详情页但 thought "does not specify the brand", 反复重搜 P5) → **vision 4 步成功** (搜索结果页截图清晰显示标题 `LG dishwasher`, agent 直接读出 "LG", no_op_rate=0.0 零循环)。**截图 ground-truth**: 列表标题 "LG dishwasher" 肉眼可读 ✓。
- = paper P-SoM/vision drop-in "vision 能读 AXTree 丢失的已渲染文字" 的**最干净正面证据**。这类 P5 **route 真有效**。

### (b) ❌ route 救不了: 表征无关的导航/搜索恢复失败 (task 0/1, **截图翻案**)
- Agent F log-only 判 task 0/1 = "perception-bound (4B 感知天花板)"。**截图 forensic 翻案** (§290 教训第三次实证):
  - **task 0 "blue kayak"**: 截图显示搜索结果是 fabric / Lionel 火车 / soup tureen —— 全是**描述含 "blue" 关键词的非-kayak**。agent 报"无 kayak"**字面正确**, 不是感知幻觉。失败 = 搜索词 ("blue") 召回错 + agent 不重构/不去 Boats 分类。
  - **task 1 "红色 Toyota $3-6k"**: 截图显示 2007 Toyota Yaris 是**银灰色**, agent 说"not red"**完全正确**。失败 = 价格区间内无红 Toyota + agent 不扩搜不翻页, 反复点同一辆。
- 两者 = agent **感知正确**, 循环是**导航/搜索恢复失败** → **表征无关** (dom 遇同样烂搜索结果也卡) → **需 module (search 重构 / pagination / category browse) 不是 route**。

### (c) ❌ route 救不了: 表征无关推理/输出错 (task 2, 6)
- task 2: scroll 阶段感知困难但最终找到 ruby bracelet, 输出选错 item (排序 reasoning 错)。task 6: P5 **误标**, 实为推理不一致 (明知 $5200 超范围仍写进 answer) + output format 错。

### (d) feedback-bound (task 3)
- agent 看得到价格 (知 $1499 超范围) 但不会用 price filter, 反复点同一 Nikon → navigation strategy failure, vision 作用微乎其微。

**精炼论点 (paper-usable)**:
> dom→vision route **实质救活** "页面已渲染、但 AXTree 未以 agent 可用方式暴露的文本信息 (品牌名/标签)" 型 P5 (task 40 实证, dom 30步→vision 4步)。route **救不了** 表征无关的导航/搜索恢复失败 (task 0/1)、推理/输出错 (task 2/6)、filter 策略缺失 (task 3) —— 这些需 **module**。**"4B 视觉感知天花板" 被 log-only 高估**: 截图证明 task 0/1 agent 感知正确, 真天花板在 detail-OCR (§4), 不在"找不到目标"的循环。

⚠️ **截图 forensic pending (router 强化)**: task 0 step_001 / task 1 早期搜索页已验; 若 paper 要 N≥3 route-win 样本, 需在 B1 som/full cross-mode 找更多 task-40 型 flip (dom-fail→vision-success 且根因=已渲染文字)。

---

## 4. Tier-2 新发现 — no-hit 子集 (25 task, deterministic 盲区)

**全 agent-limit (24/25) + 1 benchmark-FP (task 20)**, 0 scaffold。vision-specific 失败子型:

| 子型 | task | 死因 |
|---|---|---|
| **细粒度 OCR 幻觉** (correct page 上读错细节) | 118 (手机时钟 1:47 vs 3:03) · 119 ($50 钞票读成 $10) · 128 (球衣号只读 2/4) · 221 (碗数 8 vs 6) · 27 (RV 室内计数 2 vs 3) | 在**正确页面**上视觉细节读错 = vision 真天花板 (≠ "找不到目标") |
| **item-selection grounding 错** | 14 · 15 · 50 (red palette 误读成 Capsule 机) · 115 (踝 misread 膝) | gallery/list 视图点错语义相关但非目标 item |
| **multi-answer 漏报** | 192 (只报 red 漏 white) | string_match 多词 must_include 漏一 |
| **form-action 遗漏 (finish 代替 DOM 写)** | 75 (edit 未提交价格) · 208 · 213 (comment 未提交) | program_html eval, effective_mutating_action_count=0 ← **合并 §318 P35** |
| **URL 幻觉** | 35 (编 example.com 域名) | 从截图读不出真 localhost item URL → 捏造 |
| **放弃型** | 12 · 16 ("not listed" + confidence 0.0) | 视觉搜索失败即 finish 放弃 |
| **浅搜索即 finish (url_match 全库最值)** | 215 · 78 | 不穷举全库, 看到一个就 finish |
| **跨模态 grounding / UI 混淆** | 129 (UI 价格标签 vs 图内印刷价) · 199 (OsClass logo vs 图内 kaiyo.com) | 分不清 UI chrome 与 in-image text |

---

## 5. Tier-2 新发现 — success-hit FP 审计 (11/11 全 presence-only 误报)

**全部 11 个 success episode 的 hit 都是 hit_causal=false** (没一个 fired rule 是死因, 因为它们成功了)。两类:

### finish-less arrival artifact (5 个: 125/130/151/152/187, 全 eval=url_match)
- agent 第 1-2 步即到达正确 item URL → 此后 28-29 步在该页空转 (no_op_rate 0.93-0.97, page_unchanged_streak 26-27) → url_match **凭当前页 URL 自动通过, 不需 finish** → P5/P14/P31 全是到达后空转的 presence-only。
- **P31 在 vision = §317 som 同款 finish-less artifact, 且更极端** (5/11 vs som 较少): vision 无 AXTree element_id 锚点 → 到达正确页后更难自信 emit stop → 空转到 max_steps 更普遍。**= P31/trajectory_incomplete 跨模式 confound 第三次坐实** (som=finish-less §317 / dom=真卡死 §318 / **vision=finish-less 加剧**)。

### 其他 presence-only (6 个)
- **P10 在 url_match 任务系统性 FP** (task 45/87): P10 从 finish.answer 抽数字与 thought 比, 但 url_match 的 finish.answer = URL 字符串, 端口 9980 + item id 被当"应记忆数字" → 必误报。
- P17 (task 153): click-back 是有意反复核验策略, 成功了。P25 (task 233): start_url 含 `|AND|` 但 config.sites=['classifieds'] 单站 string_match。

---

## 6. Tier-2 — P19/P18 诊断 FP (Agent G, 方法论关键)

- **P19 (17/18/19/21): 0 真 benchmark-FP**。全 agent_finished=False → P19 走 **fallback 路径** (无 finish 时用 steps[-1].obs_url 代) 触发。真死因 = vision 找不到/识别不出正确 item + budget 耗尽 (P31)。**P19 在无-finish 轨迹上 = presence-only**。
- **P18 (36/37): 诊断 FP**。这俩是**跨站比价任务** (start_url `|AND|` 两站, eval=string_match), "cheapest" 是比较结果非单站排序需求。真死因 = **P25 跨站跳过** (agent 从未访问 shopping localhost:7770)。

---

## 7. 🔁 Self-evolving — 提议 P-rule (discover-only, 留合并 freeze step 落码)

> ⚠️ **本 run 不落码** (与 §317/§318 同步 discover-only)。下列与 B1 dom (§318) + som (§317) 的提议**合并去重后**在统一 freeze step 落 `diag_pattern_match.py` + bump RULESET_VERSION + `diag_autorun.sh` 全量重扫。

**新规则候选 (deterministic, 干净 signal)**:
1. **MUTATION_MISSING (P35 泛化, 合并 §318)**: `eval_type=program_html AND (eval_source_agent_url contains 'item_edit' OR locator contains '.comments_list') AND effective_mutating_action_count=0 AND agent_finished=true` — task 75/208/213。**= §318 预测的 "finish-without-mutation" 收敛**, 跨 mode 稳健。
2. **URL_HALLUCINATION**: `'example.com' in finish_answer AND 'localhost' in reference_answer` — task 35。无 FP 风险。
3. **URL_MATCH_WRONG_ITEM (benchmark-FP 候选)**: `eval_type=url_match AND agent landed page=item&id AND agent_url != reference_url AND agent_finished=true` — task 20。标 FP 候选供 exclude 审。
4. **VISUAL_DETAIL_MISREAD (correct-page)**: `correct_page_reached (agent_url≈reference) AND finish_answer NOT in must_include AND steps≤2` — task 118/119。vision detail-OCR 天花板专属。

**现有规则收窄 (success-safe, 跨 mode 都需)**:
- **P31** (最高优先): success=True 强制静默 **或** `url_match AND agent_url≈reference_url` 豁免 — finish-less artifact 不是死因。
- **P10**: finish.answer `startswith('http')` 时跳过数字比对 (URL 内嵌数字非记忆事实)。
- **P5/P14**: success-safe carve-out 或 `current_url≈reference_url` 豁免。
- **P19**: 区分 `has_finish=True` (causal) vs fallback `has_finish=False` (presence-only, 应归 P31)。
- **P18**: 入口 guard `if "|AND|" in start_url: return []` (跨站任务归 P25)。
- **P25**: 用 `config.sites` 字段判多站, 非 start_url 解析。
- **P17**: success=True 静默。

---

## 8. 代表 episode

| 类 | task | 一句话 |
|---|---|---|
| **route-win (router 核心证据)** | **40** | dom 30步失败 (AXTree 无品牌) → vision 4步成功 (截图读 "LG"); 截图确认 |
| **route 救不了 (导航失败, 截图翻案)** | **0 / 1** | agent 感知正确 (无 kayak / 车非红) 但不会恢复搜索 → 表征无关需 module |
| **finish-less artifact (FP)** | 125/151/152 | 第2步到达正确 URL → 28步空转 → url_match 自动 pass, P31 误报 |
| **vision detail-OCR 天花板** | 118 / 119 | 正确页上读错时钟 (1:47 vs 3:03) / 钞票 ($50→$10) |
| **MUTATION_MISSING** | 208 / 213 / 75 | 应写 comment/edit 却直接 finish, DOM 无 mutation |
| **benchmark-FP** | 20 | url_match EXACT 找到另一台白 Xbox, "most recently" 歧义 |

---

## 9. Actionable

- **benchmark-FP / exclude 候选** (并入 AMENDMENT_08 post-Phase-1a 审, 同 §299): task 20 (url_match EXACT 歧义)。
- **B-number 候选**: 无新 scaffold-bug (parse/tool_call 全 valid)。MUTATION_MISSING / finish-less artifact = agent-limit + rule-FP, 非代码 bug。
- **freeze step TODO**: 合并 B1 dom(§318)+som(§317)+vision 三 condition 的规则提议 → 去重 (P35/MUTATION_MISSING 收敛) → 落码 + bump version + 全量重扫 → 才解锁 cross-mode 定量。
- **cross-mode 定量仍禁** (discover-then-freeze, 6-mode 未齐)。

---

### v11 数字块（`11-intent-text-fallback`，2026-08-03 补）

> 本 digest 正文成稿于更早的 ruleset。v10 落了 **+P49 / P36 carve-out / P14 carve-out**，
> v11 给 **P34/P48 换用 `_finish_intent_text()`**（answer 为空时 fallback 读 `thought`——
> B0 惯于把结论写进 `answer`，B1 留在 `thought`，旧口径因此变成了模型行为检测器）。
> 全部 48 个 canonical condition 已在 v11 下重扫，**cross-mode / cross-model 聚合以本块为准**。

| 字段 | 值 |
|---|---|
| Run | `B1_vision_classifieds_20260605_012235_349047872_327631_R28622` |
| Episodes | 224（success 28 · SR 12.50%） |
| 三子集 | failed+hit 170 · failed-NO-hit 26 · success+hit 1 |
| config_missing | 0 |

| 规则 | 含义 | step 级 | episode 级 |
|---|---|---:|---:|
| `P5` | 感知缺失循环 | 213 | 123 |
| `P14` | URL 自环 | 104 | 82 |
| `P31` | budget耗尽未完成 | 58 | 58 |
| `P18` | cheapest漏价格排序 | 36 | 36 |
| `P20` | 评测目标页从未访问 | 22 | 22 |
| `P36` | WALK_FAIL_DEGENERATE | 74 | 18 |
| `P12` | 从不翻页 | 17 | 17 |
| `P17` | click-back振荡 | 13 | 13 |
| `P7` | sCity=州名 | 15 | 13 |
| `P25` | 跨站任务跳过其中一站 | 12 | 12 |
| `P19` | url_match过早搜索页finish | 10 | 10 |
| `P27` | 找不到即放弃 | 6 | 6 |
| `P24` | 不确定仍finish | 4 | 4 |
| `P46` | COMMENT_INTENT_NO_TYPE | 4 | 4 |
| `P35` | MUTATION_MISSING | 4 | 4 |
| `P10` | 跨步数值记忆失败 | 3 | 3 |
| `P28` | benchmark-FP货币tokenize | 3 | 3 |
| `P22` | 图上数字dom不可读 | 2 | 2 |
| `P37` | URL_HALLUCINATION | 1 | 1 |
| `P13` | 搜索代替浏览 | 1 | 1 |
| `P30` | 到达正确item后离开 | 1 | 1 |
| `P33` | 导航至裸图片URL幻觉 | 1 | 1 |
| `P29` | benchmark-FP语义yes/no | 1 | 1 |

> ⚠️ **解读约束**（`docs/analysis/_data_quality_audit.md`）：
> ① 本表是**症状分布，不是死因分布** —— P36/P31 经 10 例跨 benchmark 因果验证均判为 risk-marker；
> ② `P2`/`P4` 依赖 `element_bbox`，在 **vision 上结构性为 0（假 0）**；
> ③ `P36` 在 vision 上只覆盖 `type` 步（click 无 `locator_route_meta`）→ **分母与 dom/som 不同**。
