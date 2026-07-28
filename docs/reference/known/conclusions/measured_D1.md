# 测量结论 D1 (§1–§128.5)

> 来源: `scratchpad/batches/D1.jsonl` 219 条 MEASURED 记录, 覆盖项目最早期至 2026-05 中。
> 本文件只做**聚合 / 排序 / 去重**, 不做换算、不做合并、不做平均。
> 任何非台账原文的判断都显式标 `[聚合者推论]`。

---

## 1. [SOM_MARKS] 与 AXTree 的文本量 (format-not-information)

**当前值**: 同一 step 内两者字符量相当 —— AXTree 73 行 / 4188 chars / 70 elements with [N] tags vs [SOM_MARKS] 70 marks / 4169 chars, **比值 1.00×**; label / URL / role attr 全部保留, 唯一差异 = hierarchy indentation 被 strip。(B1 / classifieds / task 1 step 2 / 单 step 抽样, §124.4)
另有 §103 的另一口径数字: 3437 vs 3661 (reddit), 3008 vs 2948 (cls), 即 **±7%**。

**演变**: §103 报 ±7% 长度差 → §109.16 修正该 3008–3661 的 scope (是 full prompt context 不是 observation snapshot) → §124.4 逐字符核实得 1.00× 且内容逐项保留。

**已作废**: §103 的 "3008-3661 tokens" 作为 observation snapshot 量级的读法作废 —— §109.16 明确它是 full prompt context (system + task + observation + history)。

**caveats**: §124.4 是 **n=1 step 抽样**; 原文明写「与 §103 报的 ±7% 长度差是不同 step/口径, **不可合并**」。§103 的 image token 部分另被 §104 的 step-level median 取代 (见主题 3)。

**证据**: §103 §109.16 §124.4; `p79/experiment/som.py`; `memory/project_phantom_space_axes_format_not_information.md`

**原文片段**: 「AXTree 73 行 / 4188 chars / 70 elements with [N] tags; [SOM_MARKS] 70 marks / 4169 chars; 比值 1.00×; label / URL / role attr 全部保留, 唯一差异 = hierarchy indentation 被 strip」

---

## 2. Observation snapshot 的 token 规模 (P79 vs 工业 SDK)

**当前值**: P79 snapshot-alone ~1000–1500 tokens vs industry SDK ~200–400 tokens = **~3–5× format-axis gap** (B1 / classifieds / run `B1_phantom_prompt_classifieds_20260501` task_0 step_000 单 step 抽样, §109.16)。工业侧 200–400 tokens/snapshot 的两个引用点: agent-browser (Vercel Labs) 与 Playwright MCP (Microsoft) (§109.12)。

**演变**: §103 曾把 3008–3661 当 observation 量 → §109.16 code-verified 后拆出 snapshot-alone ~1000–1500。

**已作废**: 用 3008–3661 与工业 200–400 直接对比的读法 (scope 不同)。

**caveats**: §109.16 原文写「token estimate」= **估计值非 tokenizer 实测**; 单 step 抽样; industry 200–400 来自各 SDK README 自述, 未逐条 fetch 验证。

**证据**: §109.12 §109.16; `external/visualwebarena/browser_env/processors.py`

**原文片段**: 「P79 snapshot-alone ~1000-1500 tokens vs industry SDK ~200-400 tokens = ~3-5× format-axis gap; §1 hook table 的 3008-3661 tokens 是 full prompt context」

---

## 3. Image token 的 per-step 量与成本占比

**当前值**: step-level `tokens.input` median = **reddit 733 / classifieds 1064 tokens per step** (B0 / som / measured n=145 reddit, n=234 cls, §104)。图像绘制步骤本身 ~30ms + $2e-5/step (§103)。

**演变**: §103 从 total cost 反推估得 ~600 (reddit) / ~1100 (cls) per step → §104 改用 step-level tokens.input median 直测得 733 / 1064 (commit 4d63c9f)。

**已作废**: §103 的 ~600/1100 反推估计。

**caveats**: §104 明写这是 median 不是 mean; §103 版本的 caveats 里另有 token scope 修正 (见主题 1)。

**证据**: §103 §104

**原文片段**: 「从 step-level tokens.input median 直接测, 取代之前从 total cost 反推的估计 (commit 4d63c9f)」

---

## 4. classifieds 三模式 adjusted SR (visual_fp 时代定义 — 整体作废)

**当前值**: **无可用当前值。** 该时代所有 adjusted SR 依赖 `visual_fp` 层, 该层于 §95 被删除 (理由: 无文献先例 / 边界不可判 / 过滤范围过大 / 与 routing 冲突), §95 之后的 adjusted SR 与之不可比。

**演变 (按时间抄录, 不合并)**:
- §1: B1 cls DOM **0.85%** (2/234) / SoM **16.24%** (38/234) / Vision **8.12%** (19/234)
- §20: B1 cls DOM 21 个 raw 成功中 19 个判 visual FP → adjusted 0.85%
- §18: B1 DOM 成功中 visual lucky hit 占 **90.5%**
- §56: B0 cls DOM 35 个成功 = 7 na_fp + 10 visual_fp + 17 §56 盲区 + 1 真实 → adjusted **7.7%** (现行定义) 与 **~0.4%** (严格定义)
- §77: B0 cls SoM adjusted 从旧数据 12.05% 跃升至 **21.43%** (parse_error 修复后), McNemar p<0.001 vs DOM 与 vs Vision
- §89: B0 cls 全集 (scored 224) DOM **8.48%** / SoM **20.98%** / Vision **12.05%**; non-visual subset (n=72) DOM 1.39% / SoM 15.28% / Vision 缺
- §89: B1 cls 全集 DOM **4.91%** / SoM **13.84%** / Vision **7.14%**; non-visual subset DOM 0.42% / SoM 13.89% / Vision 6.94%
- §97 第四轮 (ACS-1/CMP-1 修复后): B0 cls dom raw 0.1496 → adj **0.141**; som raw 0.2308 → adj **0.2137**

**已作废**: 全部。`visual_fp` 层 §95 删除 → 上述 adjusted 数字全部与后续不可比。§56 的 "17 §56 盲区" 指当时 FP 规则未覆盖的类别。

**caveats**: §56 的 7.7% 与 ~0.4% 是**两种定义下的两个结果, 不是区间**。§89 的 non-visual subset n=72 是小样本; SoM non-visual 13.89% 略高于全集 13.84% 属噪声量级。§97 另发现: 修复前 `aggregate_cross_site.py` / `compare_b0_b1.py` 的 adjusted_sr 列对 non-stub condition **永远返回 None** (load_fp_stats 是 dead code path), 任何早于 2026-04-26 的 cross-site / cross-model 表若引用 adjusted_sr 列都是错的。⚠️ §77 与 §89 两组 B0 数字不一致, 见矛盾清单 C2。

**证据**: §1 §18 §20 §56 §77 §89 §95 §97; `docs/archive/analysis_pre_2026-05-15/vwa_classifieds/{B1_findings.md, B0_findings.md, B0_DOM_digest.md}`; `scripts/analysis/compare_b0_b1.py`

**原文片段**: 「adjusted 用的是当时含 visual_fp 层的三层 FP 体系 (§20/§27); DOM 0.85% 来自 21 成功中 19 判 visual FP; 该 FP 定义于 §95 被删除, 数字与 §95 之后的 adjusted SR 不可比」

---

## 5. 三模式 raw SR — §101 四 cell canonical 表

**当前值** (raw success, 单 seed):

| cell | DOM | SoM | Vision | SoM−DOM | DOM−Vision | SoM−Vision |
|---|---|---|---|---|---|---|
| B0 / cls (n=234) | 15.0% | 23.1% | 15.8% | +8.1pp | −0.9pp | +7.3pp |
| B1 / cls (n=234) | 11.1% | 17.5% | 11.1% | +6.4pp | 0.0pp | +6.4pp |
| B0 / reddit (n=210) | 11.4% | 11.9% | 8.6% | +0.5pp | +2.9pp | +3.3pp |
| B1 / reddit (n=210) | 10.0% | 8.1% | 4.8% | **−1.9pp (反转)** | +5.2pp | +3.3pp |

**演变**: §70 曾报 B0 cls DOM raw 14.96% / Vision 14.10% (2026-04-16 时点, SoM 当时 resume 17/234) → §90 后又清除补跑 10 个 keyword_finish 受影响的 cls episode → §101 给出上表。

**已作废**: §70 的 14.96 / 14.10 作为最终值 (原文自标"此后 §90 又清除并补跑, 数字可能再变")。

**caveats**: raw success **非 adjusted**; 单 seed; 原文明写「SoM−Vision gap 不是纯文本贡献 (含标签视觉差异)」, B1 cls 那行另标「不可解读为纯文本贡献」; B1 reddit 的反转量级是 **P79 setting-specific** (依赖 P79 SoM 设计), 「VWA 原版下可能 −0.5pp 或消失」。

**证据**: §70 §90 §101

**原文片段**: 「DOM 11.1% / SoM 17.5% / Vision 11.1% ; SoM-DOM +6.4pp / DOM-Vision 0.0pp / SoM-Vision +6.4pp ... SoM-Vision +6.4pp = 文本贡献 + 标签视觉影响 (净小), 不可解读为纯文本贡献」

---

## 6. 三模式 oracle ceiling / routing headroom / 独占集 (Venn)

**当前值** (§101, raw success 派生, 单 seed):

| cell | SoM only | DOM only | Vision only | D∩S | D∩V | S∩V | all 3 | oracle | headroom |
|---|---|---|---|---|---|---|---|---|---|
| B0 / cls | 21 | 7 | 11 | 10 | 3 | 8 | 15 | 32.1% | +9.0pp |
| B1 / cls | 14 | 9 | 5 | 6 | **0** | 10 | 11 | 23.5% | +6.0pp |
| B0 / reddit | 7 | 5 | 5 | 9 | 4 | 3 | 6 | 18.6% | +6.7pp |
| B1 / reddit | 3 | **7 (最大独占)** | 2 | 6 | **0** | 0 | 8 | 12.4% | +2.4pp |

**演变**: §77 曾报 B0 cls oracle ceiling **24.79% (adjusted)** / routing headroom 4.27pp / Vision oracle 占比 36.2% (21/58) → §97 rederive 后 B0 oracle (raw/adj) cls 0.32 / 0.31, reddit 0.19 / 0.17 (原文称 rederive 前后"稳定") → §101 给出 raw 口径的上表。

**已作废**: 无显式作废; 但 §77 的 24.79% 是 adjusted (visual_fp 时代定义), 与 §101 的 raw 32.1% **口径不同不可换算**。

**caveats**: exclusive set counts 基于单 seed raw success; B0 reddit 那行原文标"小样本 exclusive counts"; B1 reddit headroom 最小 (+2.4pp, all_fail 87.6%) 「使路由价值有限」; DOM∩Vision=0 在 B1 两站都成立。§77 另有「SoM+Vision 共享 13 tasks 可降级节省 32% 成本」—— 原文标为估算。

**证据**: §77 §97 §101

**原文片段**: 「DOM only 7 (最大独占) / SoM only 3 / Vision only 2 / DOM∩SoM 6 / DOM∩Vision 0 / SoM∩Vision 0 / all 3 = 8; oracle 12.4%, headroom +2.4pp (all_fail 87.6%)」

---

## 7. B1 reddit 的 DOM > SoM 反转

**当前值**: 反转存在但**统计上不显著**。§94 McNemar: B1 reddit DOM > SoM p=**0.344**, 95% CI DOM−SoM **[−1.0%, +4.8%]** (含 0); 只有 10 个不一致 pair (DOM-only 7, SoM-only 3), n=210。对照 B1 classifieds SoM > DOM: p=**0.014**, 95% CI [+1.7%, +11.1%], 显著。

**演变**: §94 首报 (adjusted 口径 DOM 6.83% > SoM 5.85%) → §100 引用另一组数字 (DOM 10.0% > SoM 8.1% > Vision 4.8%, 反转 −1.9pp; 同 § 内另写 "SoM 4.76%") → §101 给 raw 口径 DOM 10.0 / SoM 8.1 / Vision 4.8 → §101 Venn 显示这是唯一 DOM only (7) > SoM only (3) 的 cell (原文称"反转 fundamental 证据") → §101 per-category 显示 B1 reddit category B (n=84) SoM−DOM **−3.6pp** 是反转最严重 subset。

**已作废**: 无 (口径并存, 见矛盾清单 C4)。

**caveats**: 原文硬要求 ——「任何引用 'B1 reddit DOM>SoM 反转' 的地方**必须带这个不显著限定**」。对照 B0 reddit **未反转** (SoM adj 11.71% > DOM 8.78%), 尽管 B0 SoM 截断率更高 (33.9%); §94 对此的解释 (235B 有能力用截图弥补文本截断, 4B 不行) 原文自标「这是解读不是实验」。

**证据**: §94 §100 §101

**原文片段**: 「B1 Reddit DOM > SoM: p=0.344, 95% CI DOM−SoM [-1.0%, +4.8%], 不显著 (含 0)」/「任何引用'B1 reddit DOM>SoM 反转'的地方必须带这个不显著限定」

---

## 8. SoM max_marks=80 截断

**当前值**: 步骤级截断率 (mark_count=80): **Reddit B0 33.9% / B1 28.7%; Classifieds B0 0.2% / B1 0.5%**。任何步骤被截断的 task 比例: B1 reddit **74.3% (156/210)**, classifieds ≈0%。DOM-only 成功的 7 个 task 中 **5/7 被截断**。(§94)

**演变**: 单次测量, 无后续修订记录。

**已作废**: 无。

**caveats**: 根因是 `som.py:24` 硬编码 max_marks=80 后 break; Reddit 页面元素密集故触发率高。

**证据**: §94; `p79/experiment/som.py`

**原文片段**: 「步骤级截断率 (mark_count=80): Reddit B0 33.9% / B1 28.7%; Classifieds B0 0.2% / B1 0.5%。任何步骤被截断的 task 比例: B1 reddit 74.3% (156/210)」

---

## 9. SoM 标签 occlusion 对 OCR / visual recall 的损失

**当前值**: 最大效应 —— reddit_task_6 (111 marks) 上 mode-SoM vs mode-NoMarks 的 F1: **B0 18% vs 78% = −60pp; B1 15% vs 75% = 同样 −60pp**。实心填充覆盖元素文字开头 3–6 字符。(§100, §101 引用)

**完整 probe 矩阵** (§100, 5 图 × B0/B1 × SoM/NoMarks/WithText, visual recall %):

| 图 (marks) | B0 SoM/NoMarks/WithText | B1 SoM/NoMarks/WithText |
|---|---|---|
| classifieds_14 (33) | 41 / 36 / 46 | 27 / 36 / 32 |
| classifieds_15 (41) | 11 / 21 / 26 | 32 / 21 / 58 |
| reddit_164 step14 (54) | 46 / 46 / 50 | 46 / 36 / 96 |
| reddit_task_6 (111) | 18 / 78 / 80 | 15 / 75 / 81 |
| reddit_164 step0 (128) | 40 / 55 / 43 | 28 / 42 / 81 |

**已作废**: 无。

**caveats**: **n=5 图的 probe-level 测量, 不是 task SR**; ground truth 为 axtree link/button/heading label; 原文明写 claims「仅适用于 P79 SoM 设计 (标全部元素 + 固定青色 + simple placement), 不能 generalize 到 VWA 原版 SoM」; **−60pp 来自单张高密度图 (111 marks), 其余 4 图效应小得多甚至反向**; B0/B1 受影响幅度近似 → 判定 occlusion 是设计 bug 而非 capability 问题; §101 引用时标「量级依赖 P79 placement 设计 (VWA 8-候选 placement 可减少)」。

**证据**: §100 §101; `scripts/maintenance/probe_som_occlusion.py`

**原文片段**: 「最大效应来自单张高密度图 (111 marks), 其余 4 图效应小得多甚至反向 (见完整矩阵); B0/B1 受影响幅度近似 → 判定 occlusion 是设计 bug 而非 capability 问题」

---

## 10. SoM 数字标签的 attention hijack (num_ids)

**当前值** (B1, §100): classifieds_14 (33 marks) SoM/NoMarks/WithText = **0 / 0 / 0**; classifieds_15 (41) = **12 / 0 / 0**; reddit_164 step14 (54) = **1 / 0 / 0**; reddit_task_6 (111) = **88 / 0 / 0**; reddit_164 step0 (128) = **446 / 0 / 7**。§101 据此给出 hijack 临界阈值 **~111–128 marks 之间**。

**演变**: §100 测得 → §101 引用并插值出阈值。

**已作废**: 无。

**caveats**: **n=5 图**; 原文明写 num_ids 是「probe 输出里的数字计数, 是 **attention 倾向的代理指标不是 attention 直接测量**」; §101 明写阈值「是**插值推断非实测边界**」, 且该 § 的数字「来自 §100 SoM probe (本 § 引用), 非 §101 自测」; 阈值限定为 P79 设计。

**证据**: §100 §101; `scripts/maintenance/probe_som_occlusion.py`

**原文片段**: 「num_ids 是 probe 输出里的数字计数, 是 attention 倾向的代理指标不是 attention 直接测量」

---

## 11. B1 vs B0 无标签截图下的 visual recall + text-over-vision bias

**当前值** (§100): B1 mode-NoMarks 36 / 21 / 36 / 75 / 42 vs B0 mode-NoMarks 36 / 21 / 46 / 78 / 55 (按 33/41/54/111/128 marks 顺序), **差距 0–13pp**。text-over-vision (F4): B0 mode-WithText ≈ B0 mode-NoMarks (**80% vs 78%** on reddit_task_6) → 文本 fallback 完全 recover OCR loss; B1 mode-WithText 在 reddit_164 step14 上 **96% 反超 B0 WithText 50%**。

**已作废**: 无。

**caveats**: n=5 图; 结论「B1 视觉在无标签下接近 B0, 反驳 4B 视觉本质弱」**仅在 probe recall 这一指标上成立**; F4 是 n=1–2 图的对比, 且「解读 '4B 给文本时完全忽略截图' 是**推论**」。

**证据**: §100

**原文片段**: 「结论'B1 视觉在无标签下接近 B0, 反驳 4B 视觉本质弱'仅在 probe recall 这一指标上成立」

---

## 12. SoM 下非交互元素误点率 (M2) 与早停率

**当前值** (§101, per-cell 实测, click 事件级):

| cell | total clicks | non-interactive % | top non-int role |
|---|---|---|---|
| B0 / cls | 524 | 11.3% | image (46) |
| B1 / cls | 957 | **30.0%** ⚠️ | image (184) |
| B0 / reddit | 662 | 10.4% | heading (60) |
| B1 / reddit | 1488 | 9.5% | heading (49) |

**演变**: §96 先测 B0 vs B1 classifieds 的早停 × 误点关系 —— 早停率 (cycle/stuck) B0 **24/234 (10.3%)** vs B1 **109/234 (46.6%)**; 非交互 click 占总 click B0 **66/524 (12.6%)** vs B1 **307/957 (32.1%)**; 早停且含非交互 click B0 6/24 (25.0%) vs B1 54/109 (49.5%); 早停且末 3 步全为非交互 click B0 5/24 (20.8%) vs B1 37/109 (33.9%)。误点 role 分布: image 62% / StaticText 18% / heading 11% → §101 给出四 cell 的 per-cell 表。

**已作废**: 无 (§96 的 12.6%/32.1% 与 §101 的 11.3%/30.0% 是**不同 § 的两次统计**, 台账未 reconcile —— 见矛盾清单 C8)。

**caveats**: per-cell 实测**原文未给 CI**; 是 click 事件级比例不是 episode 级; §96 的误点 role 分布「未标是 B0 还是 B1 还是合计」; B1 reddit 9.5% ≈ B0 10.4% 是**反直觉结果** —— 即 M2 (误点) 在 reddit 上对 B1 非显著。

**证据**: §96 §101; `scripts/analysis/analyze_noninteractive_click_earlystop.py`

**原文片段**: 「反直觉: 9.5% ≈ B0 10.4%, 即 M2 (误点) 在 reddit 上对 B1 非显著」

---

## 13. Phantom modes 的单模式 SR

**当前值** (FRESH clean cell, §104, raw / adjusted):
- B0 / classifieds / P-SoM: **raw 15.81% / adj 14.53%** (n=234, FRESH 04-28)
- B0 / reddit / P-text: **raw 13.81% / adj 11.90%** (n=210)
- B0 / reddit / P-SoM: **raw 14.29% / adj 13.81%** (n=210)

**演变**:
- §103 (same-task subset, adjusted): B0 cls DOM 14.10 / SoM 21.37 / Vision 13.68 / Phantom **11.97**; B0 reddit DOM 9.52 / SoM 10.48 / Vision 6.67 / Phantom **10.95**
- §103 (N=48 same-task subset, raw→adj): DOM 18.75→12.50 / P-text 18.75→12.50 / P-SoM 18.75→16.67 / SoM 22.92→16.67
- §104 FRESH n=210/234 全量 (上表)
- §106 全量对比: cls P-SoM **15.81% vs DOM 14.96%** (表面≈); reddit P-SoM **14.29% > DOM 11.43% (+2.86pp)**
- §108.16 4-cell 交叉验证 P4 (single 别扭 ≥ DOM): B0 reddit ✅ (P-text 13.81 / P-SoM 14.29 > DOM 11.43); B0 cls ✅ (P-text 16.67 / P-SoM 15.81 > DOM 14.96); **B1 cls ❌ REVERSED** (P-text 10.26 < DOM 11.11, P-SoM 10.26 <)
- §127.9 bootstrap: reddit P-SoM vs SoM SR 差 point **+3.33pp, CI [−0.95, +7.62] 跨 0**, P(>0)=0.914; classifieds P-SoM vs SoM **−6.84pp, CI [−12.39, −1.28] strict-negative** (SoM dominates cls)

**已作废**: §103 的 N=48 narrative 被 §106 (FRESH N=210/234 全数据) supersede; §103 的「5/5 macro metrics P-text = P-SoM」结论被 §106 判为 small-sample artifact 并 RETRACTED。

**caveats**: §103/§104/§106/§108.16 全部是 **pre-Phase-A** 数据 (bug fix 3c15cd7 之前, early-stop 仍在); §103 原文明写 adj_SR aggregate 数字 collision 「不是 task-level identity (6/48 vs 6/48 stochastic collision, task-level Jaccard 仍 ~0.5)」; §127.9 的 CI 跨 0 → 「不能 claim exceeds, 只能说 competitive parity within 2σ」; §127.9 数据是 pre-fix archive; cls / reddit 方向相反 = cross-site asymmetry。

**证据**: §103 §104 §106 §108.16 §127.9; `docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md`

**原文片段**: 「CI 跨 0 → 不能 claim exceeds, 只能说 competitive parity within 2σ; 数据是 pre-fix archive (§132.3 明确 all current data is pre-fix)」

---

## 14. Drop-one oracle / routing value (HERO 指标)

**当前值**: 跨 cell 区间 **1.7–3.3pp** (B0+B1 / cls+reddit, §124.5)。最完整单 cell = B0 reddit 6-mode: **P-text +3.81pp / P-SoM +3.33pp / P-prompt +2.86pp** (§108.1)。bootstrap: reddit drop-one P-SoM **+3.33pp, CI [+0.95, +6.19] strict-positive**, P(Δ>0)=0.998; reddit drop-one SoM **+1.90pp, CI [+0.48, +3.81]**, P=0.980 (§127.9)。

**演变**:
- §103 (4-mode pool): B0 cls SoM −7.69pp > Vision −3.85 > DOM −2.14 > **Phantom −1.71**; B0 reddit SoM −2.86pp > **Phantom −2.38** > Vision −1.90 > DOM −1.43 (Phantom 第二高, 用于反驳 "Phantom = noise" 假说)
- §108.1 (6-mode, 唯一完整 cell): 上述 +3.81 / +3.33 / +2.86, 4-mode drop-one 全 sig
- §108.16 drop-one ranking 跨 capability **方向反转**: B0 reddit P-text 3.81 > P-SoM 3.33 > P-prompt 2.86 (text-axis > image-axis); B0 cls P-text 3.42 > P-SoM 2.56; **B1 cls P-text 0.85 < P-SoM 1.71 (image-axis > text-axis, REVERSED)**
- §127.9 加 bootstrap CI

**已作废**: 无数值被作废; 但 §103 的无 CI 版本被 §104 起的 bootstrap CI 版本补足。

**caveats**: §103 数据 **pre-Phase-A 且无 CI** (bootstrap CI 在 §104 才加); §108.1 是 **N=1 cell (B0 reddit only)**, 原文标 provisional 待 cls 6-mode + B1 phantom 数据 confirm, advisor sync Q3 标 provisional pending 14-cell rerun; §108.16 明写 **B1 magnitude 比 B0 弱 4×** (0.85–1.71 vs 2.56–3.81), provisional pending 14-cell rerun + cross-VLM-family validation; §127.9 是 pre-fix archive, 且原文明写「hero status **只在 oracle level 成立, 不在 single-mode head-to-head**」。

**证据**: §103 §108.1 §108.16 §124.5 §127.9; `docs/analysis/cross_sites/phantom_lift.md`; `scripts/analysis/figures/fig0c_drop_one_oracle.py`

**原文片段**: 「hero status 只在 oracle level 成立, 不在 single-mode head-to-head」/「B1 magnitude 比 B0 弱 4× (0.85-1.71 vs 2.56-3.81); provisional pending 14-cell rerun + cross-VLM-family validation」

---

## 15. 多 mode oracle ceiling 的边际增益

**当前值**: 5-mode vs 3-mode oracle lift: **B0 cls +4.70pp [2.14, 7.69]; B0 reddit +5.24pp [2.38, 8.11]** —— CI 排除 0 (bootstrap 1000 resample, §104)。6-mode 完整 cell (B0 reddit, §108.1): 6-mode vs 3-mode **+7.14pp [3.81, 10.48]** sig; 6-mode vs 5-mode marginal **+1.90pp [0.48, 3.81]** sig。

**已作废**: 无。

**caveats**: 全部 **pre-Phase-A 数据**; §108.1 是唯一 6-mode 完整 cell (N=1 cell), 原文标 provisional。

**证据**: §104 §108.1; `scripts/analysis/figures/fig0c_drop_one_oracle.py`

**原文片段**: 「6-mode oracle vs 3-mode +7.14pp [3.81, 10.48] sig; 6-mode vs 5-mode marginal +1.90pp [0.48, 3.81] sig」

---

## 16. 跨 cell meta-pooled phantom lift 与异质性

**当前值** (§124.5, DerSimonian-Laird 随机效应): **P-SoM +2.34pp [1.30, 3.37], I²=0% (零异质)**; **P-text +2.44pp [0.32, 4.56], I²=71%**。

**已作废**: 无。

**caveats**: 原文明写「P-SoM HERO 地位是 **statistically most across-cell-consistent 不是 mechanism-privileged**」; 具体 k (cell 数) **未在该 § 给出**; 底层 cell 数据为 **pre-§116** (early-stop 未关)。

**证据**: §124.5; `docs/analysis/cross_sites/meta_phantom_lift.md`

**原文片段**: 「P-SoM HERO 地位是 statistically most across-cell-consistent 不是 mechanism-privileged; 底层 cell 数据为 pre-§116」

---

## 17. Task-pool 互补性 (Jaccard / unique tasks / URL divergence)

**当前值**: phantom arms 的 task-pool Jaccard 跨 cell 区间 **0.29–0.49** (§124.5)。单 cell point estimates:
- §103: B0 reddit + cls DOM↔Phantom task pool Jaccard **0.40–0.48** (两站同水平)
- §104 Scenario C sentinel (P-SoM ↔ P-text): B0 cls **0.447** / B0 reddit **0.571** —— 均 safe (< 0.7 阈值)
- §106: B0 cls P-SoM vs DOM Jaccard **0.53**; P-SoM 解决 **12 个** DOM 解不了的 unique task, DOM 解决 **10 个** P-SoM 解不了的
- §106 compound URL signature divergence (DOM↔P-SoM): reddit **0.481 (52% URL diff)**; cls **0.66 path+query Jaccard** (path-only artifact 0.885) → 34% pages 不重合

**已作废**: 无。

**caveats**: §124.5 的 0.29–0.49 是**区间不是单 cell point estimate**; §104 的 sentinel 阈值 0.7 是**自定**; §106 的 path-only 0.885 被原文标为 **artifact**; 全部 pre-Phase-A。§106 原文定位: 「这是 routing-arm 价值在 task-pool complementarity 不在 SR delta 的核心证据」。

**证据**: §103 §104 §106 §124.5; `scripts/analysis/aggregate_phantom_lift.py`

**原文片段**: 「P-SoM 15.81% vs DOM 14.96% (表面 ≈); task-pool Jaccard = 0.53; P-SoM 解决 12 个 DOM 解不了的 unique task, DOM 解决 10 个 P-SoM 解不了的」

---

## 18. Routing signal 的 AUROC (confidence calibration)

**当前值**: **口径分裂, 三组数字并存**:
- **token-level logprob 信号: AUROC 0.497 (接近随机)** —— B1 白盒 logprob, 结论 = 对 4B 模型无效 → Phase 2 改用行为信号 (§7)
- **verbalized confidence: B0 reddit P-text 0.793 = 5-mode 最高, 超 baseline 0.766**, 5/5 mode overall_usable=True (§104)
- **fig0b_extra 口径: B0 cls best 0.846 (P-text), B0 reddit 0.817 (P-prompt)** (§124.7)

**演变**: §7 token-level 失败 → §104 verbalized 可用 → §124.7 另一 figure 口径给更高值。§97 第二轮另做 sign 修正验证: rank_biserial (CC-1 修公式 sign `2U/(n1n2)−1`) 后 B0 cls ep_mean_verbalized rb = **+0.49** (与 AUROC 方向一致)。

**已作废**: 无数值作废; 但 §7 的 0.497 在 §15/§24/§26 被复引 —— 原文明写「**均复引此同一数字 (不是独立测量)**」。

**caveats**: §104 与 §124.7 原文明写「**不同 metric/口径, 不可混**」(见矛盾清单 C5); 全部 pre-Phase-A / pre-§116 数据, 「16-cell rerun 后需刷新」; §97 的 rank_biserial 是「修复后的一致性验证, 不是新发现」。

**证据**: §7 §97 §104 §124.7; `scripts/analysis/analyze_confidence_calibration.py`; `docs/analysis/cross_sites/auroc_cross_condition_summary.md`; `scripts/analysis/figures/fig0b_extra_confidence_calibration.py`

**原文片段**: 「§124.7 后另有 fig0b_extra 报 B0 cls 0.846 P-text / B0 reddit 0.817 P-prompt — 不同 metric/口径, 不可混」

---

## 19. 「别扭 (mismatch) framework」4 predictions 的 cross-cell 验证

**当前值** (§108.16, 4 cells):
- **P1** (P-prompt drop-one 最低): B0 reddit ✅ 2.86 < 3.81/3.33
- **P2** (P-prompt raw SR < DOM): B0 reddit ✅ 10.48 < 11.43
- **P3** (Image-轴别扭 → low FP): B0 reddit ✅ P-SoM 0.48 lowest; B0 cls 🟡 P-SoM 1.28 < P-text 2.14 但 DOM 0.85 lowest; B1 cls 🟡 P-SoM 2.56 = DOM 2.56 tied
- **P4** (Single 别扭 ≥ DOM): B0 reddit ✅ / B0 cls ✅ / **B1 cls ❌ REVERSED** (P-text 10.26 < DOM 11.11, P-SoM 10.26 <)

**已作废**: 无。

**caveats**: 原文标 provisional —— 「现有数据**全部是 Phase A bug fix 之前 (pre-3c15cd7) 收集**; **N=4 cells 不足 statistical commit**; 14-cell rerun 后才是 commit time; B1 reddit phantom 数据缺」。

**证据**: §108.16

**原文片段**: 「原文标 provisional: 现有数据全部是 Phase A bug fix 之前 (pre-3c15cd7) 收集; N=4 cells 不足 statistical commit; 14-cell rerun 后才是 commit time」

---

## 20. Trajectory / strategy metrics (search-loop / steps / self-correct)

**当前值** (全数据口径, §106): B0 reddit **P-text search-loop 49.5% vs P-SoM 35.7% (差 13.8pp)**。

**演变**:
- §103 (N=210 gradient): B0 reddit DOM 27% search-loop / 38% type / 12.7 steps → Phantom_som 20% / 32% / 9.9 → SoM 12% / 23% / 8.1
- §103 (cls 弱 gradient): DOM vs Phantom_som — type 29% vs 32%, steps 11.6 vs 11.9, search-loop 19% vs 18%
- §103 (same-task subset n=26, 完全 fair): search-loop DOM 22.7 / P_som 9.1 / P_dom 9.1 | click-loop 9.1 / 18.2 / 18.2 | scroll 15.2 / 26.2 / 27.2 | finish 1.75 / 1.74 / 5.24 | self-correct/ep 0.31 / 0.35 / 0.08 | type 40.2 / 20.4 / 29.3 | avg steps 13.2 / 13.2 / 7.4
- §106 全数据 49.5% vs 35.7%
- §124.5 early-finish 比例: cls P-text **51.5%** vs P-SoM **44.5%**

**已作废**: §103 的 N=26/N=48 subset 结论被 §106 (N=210 全数据) supersede —— 原文: 「N=210 全数据显示 P-text ≠ P-SoM 显著, N=48 的 5/5 相等是 small-sample artifact」。

**caveats**: §103 subset 原文自标「N 小 noisy (avg steps 反低), 早期 directional signal only」; **所有 trajectory-derived 指标 pre-§116 contaminated** (§124.4 critique 3: early-stop 仍在, trajectory 被 truncate); §124.5 的 early-finish 原文标 partially confirms, 需 16-cell rerun 验证。§103 另判「representation 效应通过 micro-level element selection ordering manifest 而非 macro search-loop」。

**证据**: §103 §106 §124.4 §124.5; `docs/analysis/cross_sites/axis_effect_size_report.md`

**原文片段**: 「pre-Phase-A 数据 (early-stop 仍在, 见 §124.4 critique 3: 该类 trajectory-derived 指标 pre-§116 contaminated)」

---

## 21. Antagonistic mechanism pairs (两 axis 反向相消)

**当前值** (§106, cascade ablation 3 axis × 8 metric × 2 site = 48 cells, 6 个 antagonistic pair): reddit scroll text vs prompt **+0.15 vs −0.15**; reddit scroll text vs image **+0.15 vs −0.14**; cls selfcorr text vs prompt **−0.14★ vs +0.17★**; cls finish_rate prompt vs image **−0.13 vs +0.57★**; cls n_steps prompt vs image **+0.11 vs −0.33★**; cls action_repeat prompt vs image **+0.13 vs −0.42★**。

**关联测量** (§106): axis1_microbehavior ratio = mean(|axis-1 effect on decision-quality metrics|) / mean(|axis-1 effect on macro-action-freq metrics|) —— **reddit 2.28 ✓ / cls 1.02 (边界, technically ✓)**; verdict = generalizes。

**已作废**: 无。

**caveats**: pre-Phase-A trajectory-derived 指标, §124.4 critique 3 标 **contaminated**; ★ 是原文自带显著标记; **cls ratio 1.02 是边界值**; 该组 pair 的意义是「endpoint DOM↔SoM 比较看不到」。

**证据**: §106; `docs/analysis/cross_sites/axis_effect_size_report.md`, `axis1_microbehavior_report.md`

**原文片段**: 「6 个 antagonistic mechanism pair (两 axis 反向相消, endpoint DOM↔SoM 比较看不到)」/「cls ratio 1.02 是边界值; path-only 0.885 被标为 artifact」

---

## 22. B0 API 成本 vs B1 电费成本 (deployment-class gap)

**当前值** (§106, per episode): reddit **B0 $0.0399/ep vs B1 $0.000407/ep = 98×**; classifieds **B0 $0.0386/ep vs B1 $0.000366/ep = 105×**。

**关联**: §124.7 fig3b image token gap —— B0 reddit P-SoM **0.93× DOM cost**, classifieds P-SoM **0.93× DOM**。

**已作废**: 无。

**caveats**: 原文明写「两者是**不同 cost class 不是 capability ratio**」; B1 用 `avg_total_energy_kwh × $0.12/kWh` **UK industrial 电价假设**; 「**仅 deployment-class gap 解读有效**」; §124.7 的 0.93× 是 pre-§116 数据, 16-cell rerun 后需刷新。

**证据**: §106 §124.7; `docs/analysis/cross_sites/cost_per_mode.json`

**原文片段**: 「两者是不同 cost class 不是 capability ratio; B1 电价用 UK industrial $0.12/kWh 假设; 仅 deployment-class gap 解读有效」

---

## 23. obs_prepare CPU 成本相对 token 成本的量级

**当前值** (§69): obs_prepare 每步 **~$0.000005–0.000022** vs token cost 差值 **~$0.005/步** → **低两个数量级**。三模式 obs_prepare_ms: SoM ~50–200ms/步 vs Vision/DOM ~几 ms。

**已作废**: 无。

**caveats**: 费率 `overhead_cost_per_ms=0.00000011` (RTX 4090 摊算); 「对 router 决策几乎无影响, 只为方法论完整」; 原文明写「**换更贵 GPU 或更便宜模型时比例可能反转**」; 不需重跑 B1 (§35 已记 latency_ms.obs_prepare 可事后追算)。

**证据**: §69

**原文片段**: 「换更贵 GPU 或更便宜模型时比例可能反转」

---

## 24. VWA / WA 的 visual task 占比 (两套定义, 数字不可互换)

**当前值**: **两套定义并存, 台账明确禁止混用**。
- **自动列表定义** (`_load_visual_task_ids`, §89): VWA Classifieds **162/234 = 69.2%**; VWA Reddit **177/210 = 84.3%**; VWA Shopping **269/466 = 57.7%**; WA Shopping 0/192, WA Shopping Admin 0/182, WA Reddit 0/106 = 0%
- **Codex 手动审计定义** (§95): VWA 整体 **95.3%**; classifieds **96.2%** / reddit **99.5%** / shopping **92.9%**; 剔除后仅剩 **43 个 non-visual VWA task**
- **Codex 重新审计 4-category (§101, category A = NON_VISUAL_TEXT_ONLY)**: classifieds 旧 list 9 (3.8%) → 新 A **27 (11.5%)**, 3× ; reddit 旧 1 (0.5%) → 新 A **11 (5.2%)**, 11× ; shopping 旧 33 (7.1%) → 新 A **83 (17.8%)**, 2.5×

**演变**: §89 自动列表 → §95 codex 手动审计 (数字暴涨) → §101 codex 4-category 重审 (给出第三套 A 类计数)。

**已作废**: §101 之前的 `NON_VISUAL_TASK_IDS` 旧 list (9 / 1 / 33)。

**caveats**: §89 与 §95 原文明写「两种不同定义, **数字不可比、不可互换引用**」(见矛盾清单 C3); §95 的审计文件 docstring 写 'manual' 但**实际是 codex 判定**, §100 next step #5 要求重新独立审计; §101 的 4-category 同样是「**codex 判断非人工**」; reddit 首次有 11 个 non-visual subset 可统计但「仍是小样本」。

**证据**: §89 §95 §101; `docs/archive/analysis_pre_2026-05-15/cross_sites/vwa_manual_non_visual_task_ids.py`; `docs/analysis/cross_sites/codex_audit_{classifieds,reddit,shopping}.json`

**原文片段**: 「⚠️ 与 §89 的自动列表占比 (cls 69.2% / reddit 84.3% / shopping 57.7%) 是两种不同定义, 数字不可比、不可互换引用」

---

## 25. Benchmark task 计数与参考图任务数

**当前值**:
- 含参考图任务 (§46): classifieds **68/234**, reddit **84/210**, shopping **169/466** (约 29–40%)
- WA 三站拆分 (§71): 812 tasks 原始 → 拆为 **480** (shopping 192 + shopping_admin 182 + reddit 106); N/A 正确检测 **27** 个; WA visual 返回空集
- classifieds N/A task (§10/§27): **10 个**, 全部被当时的 visual FP 过滤覆盖, 且 §27 判定 **10 个全部 ua_match 误判 (双重根因)**

**演变**: 无 (各为一次性计数)。

**已作废**: §10/§27 的「N/A 与 visual FP 重叠关系」随 §95 删除 visual_fp 层而失效。

**caveats**: §46 的参考图任务在 B0 链路曾**全部静默丢失** (`api_proxy.py` 未把 `context.reference_images` 传给 `agent.step()`), B1 正常使用 → 「**修复前的 B0/B1 对比在这些任务上不公平**」; §71 有 5 个跨站任务归入 shopping primary site。⚠️ VWA/WA 总 task 数与 visual 占比在 §21 与 §89 冲突, 见矛盾清单 C1。

**证据**: §10 §21 §27 §46 §71 §89; `scripts/maintenance/split_wa_tasks.py`

**原文片段**: 「B0 链路 (api_proxy.py 未把 context.reference_images 传给 agent.step()) 全部静默忽略, B1 正常使用 → 修复前的 B0/B1 对比在这些任务上不公平」

---

## 26. Per-category subset SR (codex 4-category × 三模式)

**当前值** (§101, raw success, 单位 %, 列序 DOM / SoM / Vision / SoM−DOM):

| cell | A (non-visual) | B (ref-image) | C (page-screenshot) | D (uncertain) |
|---|---|---|---|---|
| B0 cls | n=27: 7.4 / 11.1 / 11.1 / +3.7 | n=68: 29.4 / 35.3 / 23.5 / +5.9 | n=96: 8.3 / 21.9 / 13.5 / **+13.5** | n=43: 11.6 / 14.0 / 11.6 / +2.3 |
| B1 cls | n=27: 11.1 / 11.1 / **0.0** / 0.0 | n=68: 16.2 / 19.1 / 11.8 / +2.9 | n=96: 8.3 / 18.8 / 14.6 / +10.4 | n=43: 9.3 / 16.3 / 9.3 / +7.0 |
| B0 reddit | n=11: 0/0/0/0 (全 fail) | n=84: 20.2 / 21.4 / 15.5 / +1.2 | n=113: 6.2 / 6.2 / 4.4 / 0 | n=2: 0/0/0/0 |
| B1 reddit | n=11: 0/0/0/0 | n=84: 16.7 / 13.1 / 7.1 / **−3.6 (反转最严重)** | n=113: 6.2 / 5.3 / 3.5 / −0.9 | n=2: 0/0/0/0 |

**关联** (§103, per-mode only-set 的 category 占比): B0 cls —— SoM only-set C 类 61% / Vision only-set C 类 56% / **Phantom only-set B 类 ref-img 75%** / DOM only-set mixed。B0 reddit —— SoM only-set B 类 83% / Vision only-set C 类 75% / Phantom only-set mixed (含唯一 A 类) / DOM only-set C 类 100% (n=3)。

**已作废**: 无。

**caveats**: category **由 codex 判断非人工**; subset n 小 (A=27, D=43; reddit A=11, **D=2 无统计意义**); raw success; B1 cls A subset Vision = 0.0% 即 n=27 全 fail; §103 的 only-set 「规模小, 百分比基于个位数-十位数 task」, DOM only-set n=3 时百分比无意义。

**证据**: §101 §103

**原文片段**: 「A subset 全 fail (n=11); D n=2 无统计意义; raw success」/「only-set 规模小, 百分比基于个位数-十位数 task; category 由 codex 判断」

---

## 27. 假阳性 (FP) 体系的构成与 §95 改革

**当前值**: **visual_fp 层已删除 (§95)**, 理由 = 无文献先例 / 边界不可判 / 过滤范围过大 / 与 routing 冲突。§95 之后 adjusted SR 定义变更, 与之前不可比。

**演变 / 被 §95 带走的测量**:
- §18 B1 DOM 成功中 visual lucky hit **90.5%**
- §20 B1 cls DOM 21 成功中 19 FP
- §56 B0 cls DOM 35 成功 = 7 na_fp + 10 visual_fp + 17 盲区 + 1 真实
- §83 18 个 E-FP 实例的 PUR **双峰 (低 <0.3 与高 >0.7)**, 阈值 PUR=0.5 由此设定 —— §95 判定该阈值**缺系统性验证并删除**
- §88 全量 6 个 candidate case 穷举验证 (reddit 69/72 正确标 E-FP; cls 5 / shopping 37 / reddit 160/188/189 均不受影响; B1 reddit cross_rep 确认 dom:2 + vision:2 eval_fp) —— §95 后 PUR / url_unique 阈值被删, 该验证**只对 §88 版本规则有效**
- §103 (N=48 same-task subset) FP gap / na_fp: DOM raw 18.75→adj 12.50 (gap 6.25, na_fp 3) / P-text 同 / P-SoM raw 18.75→16.67 (gap 2.08, na_fp 1) / SoM raw 22.92→16.67 (gap 6.25, na_fp 3)
- §108.16 P3 (image-轴别扭 → low FP): B0 reddit P-SoM FP 0.48 lowest; B0 cls P-SoM 1.28 < P-text 2.14 但 DOM 0.85 lowest; B1 cls P-SoM 2.56 = DOM 2.56 tied

**已作废**: 全部 visual_fp 相关比率与阈值。

**caveats**: §83 的双峰判断基于 **n=18**; §88 是 n=6 的穷举验证; §97 rederive 修正了 PUR 本身 (见主题 28)。

**证据**: §18 §20 §56 §83 §88 §95 §103 §108.16

**原文片段**: 「基于 visual_fp 判定, §95 已删除 visual_fp 层 (理由: 无文献先例 / 边界不可判 / 过滤范围过大 / 与 routing 冲突) → 该比例不再是可用指标」

---

## 28. page_unchanged_rate (PUR) 的 rederive 修正

**当前值** (§97, RU-1 修复后): B0 classifieds avg PUR **0.289 → 0.168 (−12pp)**; B0 reddit **0.286 → 0.192 (−9pp)**。

**已作废**: rederive 之前的 PUR 值 (含 finish step)。

**caveats**: 从 step JSONL 重派生 (**PUR = 排除 finish step 后的正确值**); 旧 summary 备份在 `episodes/.bak_pre_rederive/`; 覆盖 cls 702 ep + reddit 630 ep 三模式合计。

**证据**: §97; `scripts/maintenance/rederive_episode_summary.py`

**原文片段**: 「从 step JSONL 重派生 (PUR 排除 finish step 后的正确值); 旧 summary 备份在 episodes/.bak_pre_rederive/」

---

## 29. Evaluator 侧的静态缺陷与 eval_fp

**当前值** (§107 Tier 5 静态审计): **562/1598 program_html 条目使用 brittle selector**; ua_match 有 **GPT 4 种 drift modes**; string_match fuzzy=1.0 binary。

**关联**:
- §59 `page_image_query` evaluator crash: B1 reddit **28/210 (13.3%)** 全部 `evaluator_error: 'NoneType' object is not callable` (根因 `evaluator_router(config_file)` 未传 `captioning_fn`)
- §78 eval_fp 典型案例 3 个: WA shopping task 47/48 DOM (agent 被 scroll cycle 早停, 空答案匹配 "0 order/$0 total spend"); VWA reddit task 69 VISION (program_html 旧评论匹配, score=1.0)
- §87 evaluator 脏 page: B1 reddit task 138/143/150 (138/150 已由 watchdog 清除待重跑; 143 DOM 已标 benchmark_noise=True)

**已作废**: 无。

**caveats**: §107 Tier 5 是**静态 read 判定非运行时验证**; 「blast radius unknown (B-20/21/22 标 unknown blast, 走 paper §3 disclosure 而非 code fix)」; §78 是 **case 级例证不是比率**; §87 是 n=3 case。

**证据**: §59 §78 §87 §107; `docs/reference/master_bug_catalog.md`

**原文片段**: 「静态 read 判定非运行时验证; blast radius unknown (B-20/21/22 标 unknown blast, 走 paper §3 disclosure 而非 code fix)」

---

## 30. 早期 scaffold bug 对 SR / 步数的影响

**当前值 (各为独立一次性测量, 不可相加)**:
- **busy:1 中间态** (§1): B1 cls DOM raw SR **7.3% → 8.97%**; 平均步数 **15.3 → 14.9**。(§5) 步数侵蚀率: 旧 run DOM **11.8%** / SoM **24.5%**; SoM 受影响 task 高达 **80.1%**; 修后两模式均降至 **0%**
- **np.float32 bug** (§30): B1 cls vision SR **3.16%**, 全部来自不依赖 click/type 的评测路径 (url_match / program_html) —— 该期间所有 coordinate click 实际未执行
- **`<think>` 标签 parse_error** (§63): B0 cls DOM **18 steps / 16 tasks**; Vision **38 steps / 33 tasks**; 受影响 step 的 output tokens 300–600 (正常 100–150)
- **B0 SoM parse_error 率** (§67): **~20.1%** (修复前); (§90 引 §67) GLM fallback 后降至 **2–4%**
- **B0 SOM prompt 缺陷期** (§40): 33 episodes 中 SR **3% (1/33)**, avg **9.1 步**, **88% 被 action cycle 早停**
- **Magento swatch 漏检** (§105): B0 shopping DOM **11/465 ep (2.4%)** 命中 swatch-loop signature, 全部 success=False, 9/11 被 cycle 早停未 finish; `action_failed` false-positive **~30 步/run**
- **viewport ratio bug** (§80): WA Shopping DOM **14 tasks (7.5pp)**; Classifieds DOM **3 tasks (1.3pp)**
- **B0 WA shopping 首次 run** (§75): **0% SR (192/192 全失败)**; **51/192 (26.6%)** step_0 被 302 到 login 页; 其余 141 task 起始页正常但 reward=None

**已作废**: 上列各"修复前"数字均不作为 baseline。§30 的 3.16% 原文明写「此 SR 不可用作 baseline」; §40 的 33 episodes 已全部清除重跑; §105 影响的 run 数字需校 (「router signal AUROC / wasted_cost / no_op_rate 都受污染, paper §5/§6 数字需校」)。

**caveats**: §1 是**跨 run 对比非同 run**, 且是 raw success 非 adjusted; §5 的 SoM 受影响比例高「因处理链更长」; §80 的第二层影响"语义误导"**未量化** (原文标"待 Vision 对照量化"); §105 中 DOM/SoM 共享 snapshot 层 (Vision 不受影响)。

**证据**: §1 §5 §30 §40 §63 §67 §75 §80 §90 §105; `docs/analysis/cross_sites/swatch_form_change_audit.md`

**原文片段**: 「旧 run DOM 11.8% / SoM 24.5% 步数被侵蚀; SoM 受影响 task 高达 80.1%; 修后两模式均降至 0%」

---

## 31. 数据污染与清除规模 (episode / task 级)

**当前值 (逐条抄录, 不求和)**:
- §14 session cookie 过期: task 85–131 共 **47 个 require_login task** 全部丢失登录态 (根因 PHP `session.gc_maxlifetime`=1440s)
- §29 CDP 焦点丢失: B1 cls Vision **191 个 episode** 的 type 全部受影响 → 全量清除重跑 (另有修复前 3 个 vision episode 因 §28 被清除)
- §34 参考图 processor bug: B1 reddit **84/210 tasks** 受影响 (DOM 模式 obs.image=None → 不传 images 参数)
- §34 误用 `--clean`: B1 cls **702 episodes 全部丢失** (三模式合计)
- §39 B0 cls 首次启动三类 bug: 共清除 **250+ 污染 episodes**; Bug A (MySQL 初始化顺序) 使 **219 个 DOM episodes** 全部无效; Bug B (session GC 24 分钟) 两波污染 tasks 32–39 与 49–62, **19 个 SOM episodes** 无效
- §81 Wikipedia ZIM 版本 bug: reddit **33 tasks** + shopping **2 tasks**; reddit vision 33 task 已清理 (210→177 episodes)
- §82/§84 NOT-LOGGED-IN 误判: B1 cls DOM/SoM **各 6 个 task** (200/207/225/230/231/232, 均跨站)
- §86 auto-retry 漏洞: B0 cls DOM condition **9 个污染 task** (154–162)
- §90 keyword_finish scaffold confound: **23 个 episode** (22 B0 + 1 B1); B0 cls 1/7/2=10, B0 reddit 2/8/1=11, B0 shopping 1, B1 shopping 1; **SoM 占 68% (15/22)**
- §12 cycle detection 误杀: cls **3 个 task** (6/16/18)
- §57 tab_focus cycle 误判: B0 DOM task 229 只跑 3 步即被 cycle-1×3reps 终止
- §73 benchmark 推断失败: WA shopping **85 个任务**被 VWA visual 列表误标 (P0)

**已作废**: 不适用 (这些是污染规模计数)。

**caveats**: §90 的 SoM 占比高「因 SoM 降级时 prompt 变长, 235B API 更易输出冗长自然语言」—— 原文明写「**这是 scaffold confound — B0 命中概率远高于 B1 (4B local 输出结构化 JSON), 使 B0 vs B1 对比不公平**」; §34 的 702 episodes 是**事故性丢失**, 「旧数据本身不含 §33-§34 修复, 重跑后数据更一致」; §81 根因是 Kiwix ZIM 实为 2025-08 而 task config 硬编码 2022-05, 且「Playwright 成功加载 404 页不触发异常, runner 误分类为 agent fail」; §82/§84 真正根因是 watchdog 跨站检测 bug 而非 auth。

**证据**: §12 §14 §29 §34 §39 §57 §73 §81 §82 §84 §86 §90

**原文片段**: 「这是 scaffold confound — B0 命中概率远高于 B1 (4B local 输出结构化 JSON), 使 B0 vs B1 对比不公平; 23 个 episode 已清除补跑」

---

## 32. Phase A 系统性 bug mining 的 blast radius

**当前值** (§107, 全 archived runs B0+B1 × cls+red+shop × 全 mode):
- **Tier 2 silent failure scan**: 扫 **4493 ep / 46844 step JSONL signature**, 5 categories → **3052 ep / 76.5% failed traces**
- **Tier 4 invariant violations** (10 invariants × 4501 ep): I3 repeat click **481** / I7 finish-but-eval-reject **1552** / I9 element_id role drift **1127** / I10 page_changed false **288**
- **Tier 10 dispatch-effective-target probe** (Playwright 实测): **94.4% off-target on failed clicks; 100% off-target on type/select**
- **per-bug blast radius**: B-33 family (AXTree-DOM mapping) **3.0% all ep, 55.9% of click-loop signature** (远超 §106 报的 1.6%) | B-01 TYPE silent failure **12.22% / 549 ep** | B-02 union_bound center mismatch **1.6% / 27 ep** | B-09 page_changed false trigger **5.7% / 288 ep** | B-11/17/18 广义早停+重复 click **~15–20% trace cost inflate**

**演变**: §106 曾报 B-33 为 1.6% → §107 全量扫描后改判 3.0% all ep / 55.9% of click-loop signature。

**已作废**: §106 的 B-33 1.6% 读数。

**caveats**: Tier 2 是 **signature-based static scan, 非因果验证**, 「76.5% 是 failed traces 占比不是 bug 率」; Tier 4 「invariant violation ≠ 确认 bug, 是 **candidate signal**」; Tier 10 **n=18 cases 小样本, 且只测 failed clicks 子集 (selection on failures)**; per-bug 原文明写「B-11/17/18 的 15–20% 是 **cost inflate 估计非精确计数**; 各 bug **口径不同 (ep% vs signature%) 不可相加**」。

**证据**: §106 §107; `docs/reference/master_bug_catalog.md`; `scripts/maintenance/probe_tier10_dispatch_target.py`

**原文片段**: 「B-11/17/18 的 15-20% 是 cost inflate 估计非精确计数; 各 bug 口径不同 (ep% vs signature%) 不可相加」

---

## 33. Phase A fix 的 pilot validation Δ SR

**当前值** (§107, matched subset):
- **wave-2** (仅 Cluster 4 T=0 sanity, 60 ep = reddit 30 + shopping 30): **Δ = 0pp** matched-subset SR, 90% unique first action
- **wave-3** (Cluster 1+2+3+4 full bundle, 60 ep): reddit **20.0% vs paper-grade matched 16.67% → Δ = +3.33pp**; shopping **10.0% vs 13.33% → Δ = −3.33pp**; **combined N=60 net Δ = 0pp**; 0 Cluster 1 dispatch errors; 0 Cluster 3 fuzzy cycle activations

**已作废**: 无。

**caveats**: **N=60 小样本, ±5pp PASS 阈值是自定**; 「shopping −3.33pp **归因 sampling noise 无独立验证**」; Cluster 3 零触发 by design (min_reps=5 + short tasks)。

**证据**: §107; `docs/analysis/cross_sites/pilot_t0_wave3_final.md`

**原文片段**: 「N=60 小样本, ±5pp PASS 阈值是自定; shopping -3.33pp 归因 sampling noise 无独立验证」

---

## 34. B0 proxy 的 determinism 与定价一致性

**当前值** (§107.1, T=0 + top_p=1.0 + seed=42 forwarded, 同 prompt, 5 calls): **5/5 calls → 5 distinct byte-level digests** (token output 38/45/46/49/49 wide variance); 但 **5/5 calls → same action click [element_id=5]**。即 **token-level non-deterministic, decision-level convergent**。成本 ~$0.005。

**关联** (§17): B0 proxy 冒烟 1154 input + 72 output → **$0.001514** (与定价完全吻合)。

**已作废**: 无。

**caveats**: **n=5 单 prompt**; 「provider-side 是否 honor T=0 从 client 不可 verify」; 「结论**仅支持 SR-level robust**, token-level metrics (string_match exact / thought similarity) 仍需 disclose residual variance」; §17 是「仅 1 次调用的定价一致性验证, **不是吞吐/成本估计**」。

**证据**: §17 §107.1; `scripts/maintenance/probe_b37_api_determinism.py`

**原文片段**: 「5/5 calls → 5 distinct byte-level digests ... 但 5/5 calls → same action click [element_id=5]; 即 token-level non-deterministic, decision-level convergent」

---

## 35. GLM fallback 对 parse_error 的救回效果

**当前值** (§90 引 §67): parse_error 率从 **~20% 降至 2–4%**。§70 验证: 冒烟 **4/4 场景正确** (think_block / markdown / scroll / keyword_trap); 在线 **2/2 parse_error 救回** —— task_13 keyword_scroll→finish (32.5s), task_16 keyword_scroll→click eid=854 (49.5s, reward=1.0)。

**已作废**: 无 (注: 项目后期 GLM rescue 整体 retire, 但那不在 D1 范围内)。

**caveats**: **n=2 在线样本**; **2–4% 是区间未给单点**; 残留路径 keyword_finish 另由 §90 处理; GLM 需 3 项 bug 修复后才通 (api_proxy 未传 use_glm_fallback/glm_config; max_tokens 512→2048; timeout 30s→60s, DGX 到 api.z.ai 推理延迟 30–50s)。

**证据**: §67 §70 §90

**原文片段**: 「冒烟 4/4 场景正确 (think_block / markdown / scroll / keyword_trap); 在线 2/2 parse_error 救回」

---

## 36. select_option 与 dispatch 落点验证

**当前值** (§51, Playwright 端到端): category filter 页 (task_2) —— click→'' 无效, `select_option label='Electronics'`→'15' ✓, `label='Antiques'`→'2' ✓; publish 页 (item_add) element_id=1388 —— 'Electronics'→'15' ✓, 'Jewelry'→'19' ✓; AXTree OPTIONS 注入: combobox 行下注入 **23 个选项**, 两页均正常。

**关联**: §54 已选反馈缺失 —— B0 task=2 连续三次重复 `select_option label=Jewelry eid=147` → cycle detection 截断, step 1 thought 与 step 0 几乎一致。§107 Tier 10 —— **100% off-target on type/select**。

**已作废**: 无。

**caveats**: §51 是 **n=2 页面的功能性验证, 不是 SR 证据**; §54 是 **n=1 case**, fix 后 kill 所有进程并 clear task 0–10 (已跑数据作废) 重启。

**证据**: §51 §54 §107

**原文片段**: 「n=2 页面的功能性验证, 不是 SR 证据」

---

## 37. Validation / diagnosis tooling 的检出量

**当前值**:
- §91 `validate_run.py` 首轮: B0 classifieds 全 PASS; B0 reddit 检出 **vision task 75 缺失 (WARN)**; B1 shopping 检出 **44.2% 覆盖率 + 缺失 required files (FAIL)**
- §93 27-check validate_run 在 6 个 VWA run 上: B1 reddit digest 发现 **dom 55 条 / som 172 条重复**; B0 classifieds dom **时序 SR 退化 19.2%→11.5%**; C26/C27 全部 PASS
- §92 `diag_pattern_match` 在 B1 cls DOM failed-only (182 episodes): **全部有命中**; **P14 77.5% / P6 61.0% / P5 40.1%** 为主要模式
- §92 规则审计修复前后: P6 (只查 image 字段漏颜色词) **57→111 命中**; P14 (start_url 应来自 config 而非 steps[0].obs_url) **109→141**

**已作废**: §92 修复前的 P6=57 / P14=109 命中数。

**caveats**: §93 的时序退化「**是 run 内前后段对比, 不是两次实验对比**」; §92 「仅 failed episode; 规则本身有已知 FP」; §91 的 B1 shopping 44.2% 对应「B1 VWA shopping 已暂停, DOM 466 完整 SoM 4/466 中断」。

**证据**: §91 §92 §93; `scripts/analysis/validate_run.py`, `scripts/analysis/diag_pattern_match.py`

**原文片段**: 「B0 classifieds dom 时序 SR 退化 19.2%→11.5%」/「时序退化 19.2%→11.5% 是 run 内前后段对比, 不是两次实验对比」

---

## 38. Mechanistic Stage 1 — linear probe 全 trivial

**当前值** (§111.2, B1 Qwen3-VL-4B / classifieds): **三个 contrastive setup 全部 L0=0.5, L1+=1.0 全 trivial** —— Stage 1A (DOM vs P-SoM, empty obs, N=10 items) / Stage 1B (P-prompt vs P-SoM, archived obs, 96 items) / Stage 1C (SoM vs P-SoM, image-axis, 96 items)。

**已作废**: 无 (结论本身是 negative result)。

**caveats**: PCA=50 + C=0.01 强 regularize 仍 trivial; 原文归因 = 「input token 序列 fundamentally 不同 (text 内容/长度/image tokens), probe 在 **last input token position 永远 trivially 编码 input 差异**」; 结论 = 「**linear probe 对此 contrastive setup 是 wrong tool, mirage signature 必须用 patching (causal) 测**」; archived data 是 pre-Phase-A。

**证据**: §111.2; `p79/mechanistic/linear_probe.py`

**原文片段**: 「结论 = linear probe 对此 contrastive setup 是 wrong tool, mirage signature 必须用 patching (causal) 测」

---

## 39. Stage 2 patching — 早期 pilot (N=1..5) 与其被推翻

**当前值**: **早期 pilot 结论全部被 scale-up 推翻, 不应引用。**

**演变**:
- §111.3 Stage 2A first-token logit-shift (som→phantom_som, 5 task × 36 layer): logit_shift_to_source L0–L5 ~0 / L11 +0.640 / **L17 +1.080 (peak)** / L23 +0.753 / L29 +0.617 / L35 +1.000; L17 peak std=0.214 (5/5 task ≥ 0.9)
- §111.4 Stage 2B task-0 case study: **L11 patched 输出与 source 93% overlap, Levenshtein 1/15 token**; L0/L5/L17+ 完全 = target (60% baseline); task 1/2 null effect
- §111.5b Stage 2C reverse (N=1): 全 36 layer overlap→source = 0.51 (baseline), overlap→target = 1.0 → **任何层 null effect**

**已作废**:
- §111.4 的 task-0 93% flip 被 **§117.4 作废**: 「task-0 是 **task-specific outlier 不是 population pattern**, 24-task forward L11 overlap→tgt = 0.908 Δ=−0.093 **远弱**; 不应作 paper §5 representative finding」
- §111.5b 的 "reverse null" 被 **§117.2 推翻**: scale-up 到 N=15 reverse tasks 后 reverse 显示 mid-layer L11–L17 disruption **幅度与 forward 相当**

**caveats**: §111.3 的 argmax_match 全 1.0 与 KL 全 ~0 都是 **trivial** (source/target argmax 都是 token ID 515 = JSON `{` opener, chat template 强制), absolute scale 弱因 source/target distribution 太接近, N=5; §111.4 「N=3 太小, 仅 1/3 task 出 clean signal」, task 1/2 的 divergence 是 **non-mirage reason (thought framing 不同)** 故 patching 不能修, 且 dropdown OPTIONS 注入被跳过 (production prompt drift 1 处); 全部 archived pre-Phase-A。

**证据**: §111.3 §111.4 §111.5b §117.2 §117.4; `scripts/mechanistic/run_stage2{,b}_*_pilot.py`

**原文片段**: 「L11 patched 输出与 source token-by-token 93% overlap, Levenshtein distance 1/15 token」+ 作废理由「task-0 是 task-specific outlier 不是 population pattern, 24-task forward L11 overlap→tgt=0.908 Δ=-0.093 远弱」

---

## 40. Stage 2 patching — scale-up 后的 per-layer profile

**当前值** (§117.2, B1 / classifieds / som↔phantom_som, forward N=24 strong-tier + reverse N=15 reverse-tier, Myriad A100):

| layer | fwd overlap→tgt | rev overlap→tgt | fwd LD→tgt | rev LD→tgt |
|---|---|---|---|---|
| L0 | 0.98 | 0.88 | 0.9 | 2.4 |
| L5 | 0.97 | 0.89 | 1.3 | 2.6 |
| L11 | 0.91 | 0.82 | 3.9 | 6.7 |
| L17 | 0.86 | 0.81 | 6.0 | 6.9 |
| L23 | 0.93 | 0.87 | 2.2 | 2.8 |
| L29 | 0.95 | 0.93 | 2.5 | 1.1 |
| L35 | 1.00 | 1.00 | 0.0 | 0.0 |

L17 Δoverlap→target (vs L35 baseline): **forward −0.143 ± 0.217 (N=24) / reverse −0.193 ± 0.256 (N=15)**, Welch t @ L17 p=**0.535**。

**已作废**: §111.5b 的 reverse null。

**caveats**: 原文明写「**failure to reject ≠ proof of equality**」; 「std band ±0.22 ≈ **1.5× effect size** (强 task heterogeneity, 有 task 完全 flip 有 task 不 disrupt)」; **forward 24 tasks ≠ reverse 15 tasks** (direction-specific composite curation, **selection-bias 未控** → §117.5 2x2 control launched); archived pre-Phase-A obs。§128.4 另实证 L35 patched 的 overlap→target ≈ **1.00** (final-block patching 经验上等价 unpatched), 用于论证 Bug 6 (用 L35 patched 作 baseline) 的数值影响 ≈ 0, 故只文档化不改数。

**证据**: §117.2 §128.4; `results/mechanistic/stage2b_curated_b1_cls_myriad`; `scripts/analysis/stage2_layer_significance.py`

**原文片段**: 「原文明写 failure to reject ≠ proof of equality; std band ±0.22 ≈ 1.5× effect size (强 task heterogeneity, 有 task 完全 flip 有 task 不 disrupt)」

---

## 41. Stage 2 层显著性 — 8-cell L17 汇总 (mid-layer mechanism)

**当前值** (§123, 统一表, paired t-test vs L35 baseline, one-sided, Holm 跨 6 层):

| cell | 配置 | N | L17 p_Holm | Δ overlap→tgt |
|---|---|---|---|---|
| A | cls fwd strong | 24 | 0.011 ✓ | −0.143 |
| B | cls rev reverse | 15 | 0.033 ✓ | −0.193 |
| C | cls fwd reverse | 15 | **0.257 ✗ NULL** | −0.103 |
| D | cls rev strong | 24 | 0.010 ✓ | −0.193 |
| F | reddit fwd strong | 24 | 0.004 ✓✓ | −0.228 |
| G | reddit rev reverse | 15 | 0.036 ✓ | −0.271 |
| C-r | reddit fwd reverse | 15 | 0.012 ✓ | −0.320 |
| D-r | reddit rev strong | 24 | 0.041 ✓ | −0.155 |

**→ 7/8 cells L17 Holm-sig (88%), magnitudes −0.10 到 −0.32。**

L11 对照 (§120 4-cell + §123): Cell A 0.092 ✗ / B 0.044 ✓ / F 0.002 ✓✓ / G 0.049 ✓ / C-r 0.012 ✓ / D-r 0.116 ✗ / (C 0.257 ✗ / D 0.008 ✓)。
§118 Cell F 完整层剖面 (mean overlap→tgt): L0 0.885 Δ−0.115 p_Holm 0.031 ✓ | L5 0.937 Δ−0.063 0.063 ✗ | **L11 0.747 Δ−0.253 0.002 ✓✓** | **L17 0.772 Δ−0.228 0.004 ✓✓** | L23 0.847 Δ−0.153 0.023 ✓ | L29 0.937 Δ−0.063 0.069 ✗。
§120 Cell G 完整层剖面 (reddit reverse × reverse-tier, N=15, Myriad V100 job 336424): L0 0.817 Δ−0.183 p_Holm 0.086 ✗ | L5 0.799 Δ−0.201 0.054 ✗ | **L11 0.757 Δ−0.243 0.049 ✓** | **L17 0.729 Δ−0.271 0.036 ✓** | L23 0.873 Δ−0.127 0.124 ✗ | L29 0.896 Δ−0.104 0.124 ✗。
§117.3 metric 交叉: Forward × overlap→tgt L17 p_Holm 0.011 ✓ (L11 marginal 0.092); Forward × LD→tgt L17 0.024 ✓ (L11 0.080); Reverse × overlap→tgt L11 0.044 ✓ + L17 0.033 ✓; Reverse × LD→tgt 无 Holm-sig (L17 raw p=0.014 → Holm 0.084, N=15 power-limited)。

**演变**: §117.3 (cls 2 cells) → §118 (+reddit Cell F) → §120 (4 cells) → §121 (cls 2x2 补 C/D) → §123 (reddit 2x2 补 C-r/D-r, 成 8-cell 表)。

**已作废**: §118 曾把 cls Cell A L17=0.011 与 reddit Cell F L17=0.004 「放在同一 overlap→tgt 列标题下」的表述被 **§120 RETRACTED**。§123 的 site-asymmetric mechanism framing 被 **§124.4 RETRACTED**。

**caveats**: 唯一 NULL 是 cls Cell C —— 原文明写「**p_Holm=0.257 是 not Holm-sig 不是 proven null**」(95% CI L11 Δ=[−0.235,−0.012] 技术上排除 0, raw p=0.043, 但 Holm 跨 6 层 + N=15 惩罚重); N=15 cells power-limited; 全部基于 **archived pre-Phase-A observation artifacts** (原文注: mechanistic 为单步 inference, 不受 early-stop trajectory bug 影响); §118 另标 L11/L23 在 reddit 显著但 cls 不显著「可能 site-specific layer recruitment 或 cls 稀疏信号未达 Holm, 原文标 exploratory」。

**证据**: §117.3 §118 §120 §121 §123; `results/mechanistic/layer_significance_*.md`; `results/phantom_paper/figures/fig_mech_8cell_l17_forest.png`

**原文片段**: 「7/8 cells L17 Holm-sig (88%), magnitudes -0.10 到 -0.32; 唯一 NULL 是 cls Cell C (N=15 power-limited)」

---

## 42. Stage 2 的 sparse-mechanism caveat (mean vs median)

**当前值** (§118): L17 LD→target —— reddit Cell F **mean 8.000, median 0**; classifieds Cell A **L17 LD 6–8 (varies), median 0**。

**已作废**: 无。

**caveats**: 原文硬要求 ——「**median=0 表示多数 task 无 disruption; mean 由 ~25% high-salience-image task subset 撑起 — sparse-mechanism caveat 必须保留, 不能只报 mean**」。N=24 each。

**证据**: §118; `results/mechanistic/layer_significance_cellfg_reddit_20260509.md`

**原文片段**: 「median=0 表示多数 task 无 disruption; mean 由 ~25% high-salience-image task subset 撑起 — sparse-mechanism caveat 必须保留, 不能只报 mean」

---

## 43. Bidirectional symmetry (forward vs reverse 幅度)

**当前值** (§120, Welch t @ L17 overlap_to_target): **cls** forward Δ −0.143 ± 0.217 (N=24) vs reverse −0.193 ± 0.256 (N=15), **p=0.535** → bidirectional yes; **reddit** forward −0.228 ± 0.308 (N=24) vs reverse −0.271 ± 0.363 (N=15), **p=0.705** → bidirectional yes。§121 另有 Cell C vs Cell D cross-direction Welch @L17 **p=0.282**。

**已作废**: 无。

**caveats**: 原文明写「两 site forward/reverse magnitude statistically indistinguishable, 但 **failure to reject ≠ equality**; task subsets 不同 (direction-specific curation)」; §121 的 C vs D「原文明写 **power-limited, 不可解读为等价**」(C 是 null)。

**证据**: §120 §121

**原文片段**: 「两 site forward/reverse magnitude statistically indistinguishable, 但 failure to reject ≠ equality; task subsets 不同 (direction-specific curation)」

---

## 44. Selection-bias 2x2 control (direction × tier)

**当前值** (§121, cls, overlap_to_target 的 L11 / L17 p_Holm): Cell A fwd×strong×24 **0.092 / 0.011** (L17 ✓) | Cell B rev×reverse×15 **0.044 / 0.033** (L11+L17 ✓) | Cell C fwd×reverse-tier×15 **0.257 / 0.257 (✗ NULL)** | Cell D rev×strong-tier×24 **0.008 / 0.010** (L11+L17 ✓)。LD_to_target 同结论 (C 0.201/0.201 ✗, D 0.037/0.037 ✓)。
reddit 对应 (§123): **C-r** reddit×fwd×reverse-tier×15 L11 **0.012 ✓** / L17 **0.012 ✓**; **D-r** reddit×rev×strong-tier×24 L11 **0.116 ✗** / L17 **0.041 ✓**。

**演变**: §117.2 指出 selection-bias 未控 → §121 cls 2x2 land → §123 reddit 2x2 补齐 (§120 时点原文还标「reddit 2x2 control 未跑」)。

**已作废**: 无。

**caveats**: Cell C 的 NULL 是 **power-limited 不是 proven null**; 「composite score 是 **correlation-based heuristic**, strong/reverse 划分是**经验非理论**」。

**证据**: §120 §121 §123; `results/mechanistic/layer_significance_cell{cd_cls2x2,crdr_reddit2x2}_*.md`

**原文片段**: 「Cell C N=15 power-limited — p_Holm=0.257 是 not Holm-sig 不是 proven null (95% CI L11 Δ=[-0.235,-0.012] 技术上排除 0, raw p=0.043, 但 Holm 跨 6 层 + N=15 惩罚重)」

---

## 45. Random-injection negative control (Cell E / E-r)

**当前值** (§122, cls Cell E, forward × strong-tier × random-injection, N=24 同 Cell A 的 task):

| layer | overlap→tgt | LD→tgt | overlap→src |
|---|---|---|---|
| L0 | 0.028 | 17.21 | 0.032 |
| L5 | 0.084 | 9.71 | 0.027 |
| L11 | 0.029 | 32.79 | 0.017 |
| L17 | 0.040 | 29.62 | 0.015 |
| L23 | 0.026 | 36.25 | 0.010 |
| L29 | 0.044 | 22.75 | 0.024 |
| L35 | 0.034 | 28.88 | 0.019 |

real vs random 同 24 task 对比: L35 baseline **real 1.000 vs random 0.034 (diff −0.966)**; L17 **0.857 vs 0.040 (−0.817)**; L11 **0.908 vs 0.029 (−0.879)**。
reddit E-r (§123): baseline broken, overlap→tgt all-layer ~0.01–0.09 (vs L35=0.021), **无 L17 选择性** —— 与 cls Cell E 定性一致。
§128.5 另有 Gaussian random-injection 控制 (cls 359719 / red 359720): **定性 destroys output**, 两 cell 都 confirm codex 预测。

**已作废**: 无。

**caveats**: **Cell E 的 L35 baseline 本身 broken (0.034 不是 1.000) → Holm-paired-vs-L35 框架对 Cell E 根本不适用, 故不做 Holm**; N=24 **单一 random seed**; 三个 signature 差异原文明列: (1) L35 baseline broken (2) 跨层 flat 0.03–0.08 无 mid-layer peak 无 L23+ recovery (3) 是 **random-bigram 级残留不是 content-preserving**; §128.5 的 Gaussian 控制「**原文只给定性结论无数值**」。

**证据**: §122 §123 §128.5; `results/mechanistic/stage2b_celle_random_cls_strong_myriad`

**原文片段**: 「即 random injection 是 layer-agnostic 灾难性破坏, 与 Cell A 的 L17 选择性 -14% 结构不同」

---

## 46. Stage 3 attribution — 分 target mode 的 disruption 与可加性

**当前值**:
- **H-cells 层显著性** (§124.1, source=som forward × strong-tier, N=24 each): Ht_cls (target=P-text) L11 0.002 ✓✓ / L17 0.002 ✓✓, mean overlap 0.743 / 0.745 | Hp_cls (target=P-prompt) 0.009 ✓✓ / 0.011 ✓, 0.756 / 0.777 | Ht_red 0.005 ✓✓ / 0.006 ✓✓, 0.756 / 0.764 | Hp_red 0.005 ✓✓ / 0.011 ✓, 0.767 / 0.809
- **target-mode disruption 梯度** (§124.10, L17 overlap, cls, N=24 per cell): **DOM target −0.31 (最大) > P-text −0.26 > P-prompt −0.22 > P-SoM −0.14** —— 即 target 与 source 共享 axes 越多 disruption 越小
- **2x2 additivity 检验** (§124.10, Cell H-d-cls, predicted = Ht_cls + Hp_cls − Cell A vs observed Hd_cls): overlap L11 predicted −0.4092 / observed −0.3317 / diff **+0.0775 (sub-additive)**; overlap L17 −0.3342 / −0.3092 / **+0.0250 (✅ ADDITIVE, 交互项约主效应和的 5%)**; LD L11 +10.75 / +11.92 / +1.17 (additive); LD L17 +8.83 / +11.21 / +2.37 (super-additive, small)
- **cross-site 6-cell best-layer Δ** (§127.1): cls SoM→DOM **−0.352 (L18)** / SoM→P-text **−0.270 (L12)** / SoM→P-prompt **−0.273 (L13)**; reddit SoM→DOM **−0.338 (L14)** / SoM→P-text **−0.330 (L15)** / SoM→P-prompt **−0.322 (L14)**; **6 个 best layer 全落 L12–L18; Δ range [−0.27, −0.35]**
- **reddit H-d cell** (§127.1): L11 Δ=−0.335 / L17 Δ=−0.255 / best L14 −0.338

**已作废**: 无数值作废, 但**口径被改判** —— §136.2 F2 指出 H-d cells (som→dom) **同时翻 image+text+prompt 三轴**, §5 prose 把它当 axis-1 patching 是**错口径**; §136.6 把 axis-1 改判为分解轴。

**caveats**: §124.10 预注册预测为 Δ_to_target @ L11 ≈ +10.74, falsify 条件是观测落在 ±2pp 之外; 「其它 layer/metric 组合交互项为主效应的 **5–15%**, 方向依组合而异 (sub- 或 super-additive)」; §124.10 是**单 site (cls) 单 direction (forward)**; §127.1 的 **best-layer 是 post-hoc 选的** (原文记 W3 攻击: 需 Holm m=36 redo, 该 chunk 内未 close); §124.1 的 Hp 两 cell 是 bug 修复后重跑版 (jobs 342330/342331)。

**证据**: §124.1 §124.10 §127.1; `results/mechanistic/`

**原文片段**: 「DOM target -0.31 (最大) > P-text -0.26 > P-prompt -0.22 > P-SoM -0.14; 即 target 与 source 共享 axes 越多 disruption 越小」/「best-layer 是 post-hoc 选的 (W3 攻击: 需 Holm m=36 redo, 本 chunk 内未 close)」

---

## 47. Exp 5 — axis-2 prompt-only patching

**当前值** (§128.1, source=P-SoM → target=P-text, image + text-format held constant, @L17 overlap→tgt / LD→tgt): cls H-text (image+prompt) **0.75 / 9.2**; cls cellhprompt (prompt only) **0.79 / 8.5**; red H-text **0.76 / 8.6**; red cellhprompt (prompt only) **0.70 / 8.8**。

**关联** (§128.1, cosine peak vs patching causal peak 对照): Image —— cosine 0.041@L17 vs patching ~0.04–0.05@L11–L17 (**aligned**); Text-format —— cosine 0.029@L23 vs patching 待 reanalyze; Prompt-family —— cosine 0.011@L23 vs patching ~0.20–0.30@L11–L17 (**misaligned**); 在 L23 patching overlap→tgt 仅 0.96 / 0.89 (几乎无 displacement)。

**已作废**: 对照表中的 cosine 侧数字来自 **v1 buggy NPZ**, §128.5 后 invalidated (见主题 48)。

**caveats**: §129.4 CL2 指出「§5.4 引用的 axis-2 patching **0.20–0.30 与 cellhprm overlap=0.188 chance baseline 未区分**, prose 需 verify」; 该 cell 数据「一度被 watcher prefix bug 本地覆盖后从 Myriad 重拉」。

**证据**: §128.1 §129.4; `docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md`

**原文片段**: 「§129.4 CL2 指出 §5.4 引用的 axis-2 patching 0.20-0.30 与 cellhprm overlap=0.188 chance baseline 未区分, prose 需 verify」

---

## 48. Stage 4 Method 4.2 — mode-pair cosine gap 的 v1 → v2 崩塌

**当前值** (§128.5, v2 NPZ):

| mode pair | v1 | v2 | 变化 |
|---|---|---|---|
| DOM↔P-text (axis-1 anchor) | L23 0.0254 | **L36 0.0047** | −81%, peak 移到 boundary |
| P-prompt↔P-SoM (axis-1) | L23 0.0292 | **L36 0.0048** | −84% |
| P-text↔P-prompt | L23 0.0288 | **L36 0.0081** | −72% |
| P-text↔P-SoM (axis-2 anchor) | L23 0.0114 | **L36 0.0088** | −23% (本来就小) |
| P-SoM↔SoM (image anchor) | L17 0.0412 | **L36 0.0416** | magnitude 稳, peak 移位 |
| DOM↔Vision (image) | L04 0.0653 | **L04 0.0670** | unchanged |
| DOM↔P-prompt | L36 0.0067 | **L36 0.0068** | unchanged |

v2 cross-site (§128.5): image axis cos **0.045 (cls) / 0.042 (reddit)**; axis-1 **0.005 / 0.003**; axis-2 **0.009 / 0.009**。

**演变**: §124.6 首测 (Stage 1 缓存复用) → §125.2 六模式全层 profile → §127.1 reddit P5b (P-SoM↔DOM L17=0.0098; P-SoM↔SoM L17=0.0423) → §128.4 发现 Bug 2 → §128.5 v2 重跑。

**已作废**:
- **v1 NPZ 全部数字** (Bug 2: `build_som_marks` lossy regex, SOM_MARKS 只剩 3 行, **71/72 marks 被 drop**) —— §128.5 判整体 invalidated
- 「image axis localized at L17」的说法被 **retract** (v2 peak 移到 L36)
- §127.1 的 P5b 数字 (superseded by §128.5)

**caveats**: 原文硬要求 ——「**v1→v2 之间同时改了 3 处 (SOM_MARKS regex + tier filter + model revision pin) 且 N 从 48 降到 24 → §130.5 判为 4 个并发 confounder, −81% 不可归因单一 cause**」; §136 后 axis-1 pair 又因 **B-82 ([OPTIONS] 缺失)** 被判 confounded 需再次重抽; axis-2 判定不受 B-82 影响; DOM↔Vision 的 L04 peak 被认定为真 Mirage signature (v2 后仍成立) 但 **§136.2 F3 指出 vision mode 是三轴 different, 干净 image pair 只有 som↔phantom_som**; v2 两 site pattern 一致「是预期 (同一份 code fix), **不构成独立 cross-site 验证**」; §124.6 版本是 96 examples 复用既有 Stage 1 缓存 (archived pre-Phase-A obs)。

**证据**: §124.6 §125.2 §127.1 §128.4 §128.5 §130.5 §136; `docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md`; `scripts/analysis/stage4_pca_cosine_gap.py`

**原文片段**: 「v1→v2 之间同时改了 3 处 (SOM_MARKS regex + tier filter + model revision pin) 且 N 从 48 降到 24 → §130.5 判为 4 个并发 confounder, -81% 不可归因单一 cause」

---

## 49. 三轴 hierarchy 的幅度比 (已 retract)

**当前值**: **无。整个 hierarchy 已被 §128.5 retract。**

**演变 / 曾报的值**:
- §124.6 (Stage 1 缓存, 96 examples, L11/L17/L23 余弦): P-prompt↔P-SoM (text-axis) 0.9987 / 0.9984 / 0.9980; P-prompt↔SoM (image-axis) 0.9667 / 0.9583 / 0.9708; P-SoM↔SoM (image-axis) 0.9693 / 0.9615 / 0.9733 → **image-axis gap 3.0–3.3% vs text-axis 0.13% = 25× 不对称**
- §125.2 (288 examples): image-axis peak **0.06 at L4–L17** | text-axis peak **0.025 at L23** | prompt-axis peak **0.007 at L36**; magnitude ratio **image:text:prompt ≈ 10:4:1**
- §127.2 (v1 NPZ, cls+reddit): Image (P-SoM↔SoM) **L17 0.041 (4x)**; Text-format **L23 0.029 (3x)**; Prompt-family **L23 0.011 (1x ref)** → **4:3:1, cross-site stable**

**已作废**: §127.2 的 **4:3:1 hierarchy 整体 RETRACTED** (§128.5: v1 NPZ 由 buggy SOM_MARKS regex 产, 3 行 vs production 72 行)。§125.2 的 10:4:1 基于同一 v1-era pipeline。

**caveats**: cosine gap = 1 − cos(mean_A, mean_B), 是 **mean-vector 几何量非行为量**; §124.6 是 96 examples 且 archived pre-Phase-A obs。

**证据**: §124.6 §125.2 §127.2 §128.5; `docs/checkpoints/mechanism/results/axis2_layer_profile.md`

**原文片段**: 「v1 NPZ 由 buggy SOM_MARKS regex 产 (3 行 vs production 72 行) → §128.5 整个 4:3:1 hierarchy retracted」

---

## 50. Method 4.2 的 AUROC = 1.000 及其解释性攻击

**当前值** (§128.5, v2 NPZ, leave-one-task-out held-out): **1.000** (6 modes 仍完美线性可分, 尽管 non-image 轴的 mean-diff 是 sub-permille)。

**演变**: §125.2 (v1) 全部 **540 组 (15 pairs × 36 non-trivial layers) AUROC = 1.000** (唯一例外 DOM↔P-prompt L36 = 0.998) → §128.4 在 legacy buggy NPZ 上跑 lototask CV 得 **lototask = in_sample = 1.000** → §128.5 v2 NPZ 仍 1.000。

**已作废**: 无 (值不变), 但 **v1 的 in-sample 标注 (Bug 3) 已被 held-out 版本取代**。

**caveats**: §130.5 攻击 ——「AUROC 的 direction = (c1−c2)/||c1−c2|| **与 cosine 测同一方向** → AUROC 1.0 + cos 0.005 **不代表 modes well-separated**, 而是 within-mode variance ⊥ inter-mode direction」; §125.2 原文已注「这是 **input-encoding 层面的可分性** (§124.5 已指出 trivially separable), **不等于下游 policy 差异**」; §128.4 作者自评「separability 由 prompt + image 主导而非 text payload」。

**证据**: §125.2 §128.4 §128.5 §130.5; `scripts/analysis/stage4_pca_cosine_gap.py`

**原文片段**: 「AUROC 的 direction = (c1-c2)/||c1-c2|| 与 cosine 测同一方向 → AUROC 1.0 + cos 0.005 不代表 modes well-separated, 而是 within-mode variance ⊥ inter-mode direction」

---

## 51. Method 4.2 robustness 5 项 (A–E)

**当前值** (§125.3, B1 / cls / 6 modes / N=24 tasks = 288 examples):
- **A label permutation**: real AUROC 1.000 vs perm mean **0.629 ± 0.038**, p=0.005, **9.8σ** (原文注: perm baseline 在高维是 0.63 不是 0.5)
- **B per-task consistency**: 24 task **100%** 在 L17 6 个 pair 上 cosine gap 为正, std/mean ≤ 10%
- **C per-step invariance**: P-SoM↔SoM L17 step2 **0.0414** vs step5 **0.0411**
- **D silhouette**: L17 **0.471**, L23–L36 **0.51–0.52** (≥0.5 = strong clustering)
- **E bootstrap 1000 task resample**: P-SoM↔SoM L17 **0.0413 [0.0403, 0.0422]**, 宽度为均值的 4.6%

**已作废**: 无显式作废; 但底层 288-example 数据集是 v1-era pipeline [聚合者推论: 台账未在 §125.3 记录 v2 重跑版本, 引用前需确认]。

**caveats**: 原文明写「**全部在同一 288-example 数据集内, 非独立复现**」; 「L17 silhouette 0.471 **略低于 0.5 阈值**」。

**证据**: §125.3; `scripts/analysis/stage4_robustness.py`

**原文片段**: 「全部在同一 288-example 数据集内, 非独立复现; L17 silhouette 0.471 略低于 0.5 阈值」

---

## 52. P-SoM 的身份归属 (最近邻 mode)

**当前值** (§125.2, B1 / cls / 288 examples): **P-SoM 在 layer 4–36 每一层的最近 mode 都是 P-text**; L17 cosine gap P-SoM↔P-text **0.0028** vs P-SoM↔SoM **0.0412** = **14.7× 更远**; P-SoM↔SoM peak L17 = 0.0412 与 Stage 2 patching disruption locus 完全吻合。

**已作废**: 「P-SoM↔SoM peak 在 L17」的部分被 §128.5 v2 推翻 (peak 移到 L36, magnitude 稳)。最近邻结论本身台账未标作废。

**caveats**: 原文定位为「驳斥 'P-SoM = SoM minus image' 的直觉, 支持 'P-SoM = P-text + SoM-prompt tweak'; 两独立方法 (interventional patching + observational PCA) 收敛于 L17」—— 但该收敛论据的 PCA 侧数字来自 v1-era NPZ。

**证据**: §125.2 §128.5; `scripts/analysis/stage4_pca_cosine_gap.py`

**原文片段**: 「P-SoM 在 layer 4–36 每一层的最近 mode 都是 P-text; L17 cosine gap P-SoM↔P-text 0.0028 vs P-SoM↔SoM 0.0412 = 14.7× 更远」

---

## 53. Exp 3 logit lens output amplification (已 retract)

**当前值**: **无。§128.5 判 v1 NPZ invalidated。**

**曾报的值** (§127.3): axis-2 cos 0.011 @L23 → KL **0.16 (14x amp)**; axis-1 cos 0.029 → KL **0.69 (24x)**; KL@L36→0 (final layer mean collapse to common JSON header); mode-distinct signal 集中 **L23–L25**。

**已作废**: 全部 (v1 NPZ Bug 2)。

**caveats**: 另有两条独立于 Bug 2 的方法学缺陷 ——「KL 是 **lm_head(mean_h) 的 decoded-means KL 不是 per-task KL 平均** (§130.5 OOB1 Jensen artifact)」; 「bf16 强制 (~3 位小数) 使**第 4 位小数为噪声** (§130.5 OOB2)」。

**证据**: §127.3 §128.5 §130.5; `docs/checkpoints/mechanism/results/axis2_logit_lens.md`

**原文片段**: 「KL 是 lm_head(mean_h) 的 decoded-means KL 不是 per-task KL 平均 (§130.5 OOB1 Jensen artifact); bf16 强制 (~3 位小数) 使第 4 位小数为噪声」

---

## 54. axis-2 per-task fragility 分布

**当前值**: **v2 重跑后量级完全不同, v1 数字作废。**

**演变**:
- §127.9 (v1 NPZ, 24 task): cls @L23 mean=median=**0.0131**, IQR **[0.012, 0.014]**, 100% tasks > 0.010, zero outliers, no right-skew; reddit mean=median=**0.0120**, IQR **[0.011, 0.013]**, 100% tasks > 0.010
- **§129.2 v2 重跑给出完全不同量级: IQR [0.0024, 0.0040]**

**已作废**: §127.9 的 v1 数字 (mean 0.0131 / 0.0120 及其 IQR)。

**caveats**: 原文注 v1 结果「与 H1 dichotomy 11% strict per-task fragile 形成对比 (H1 是 **binary layer-comparison**, axis-2 是 **continuous mode-pair distance**)」—— 两者口径不同。

**证据**: §127.9 §129.2; `docs/checkpoints/mechanism/results/axis2_per_task_fragility.md`

**原文片段**: 「v1 NPZ; §129.2 v2 重跑给出完全不同量级 (IQR [0.0024, 0.0040])」

---

## 55. Method 4.4 steering — layer × α dose-response

**当前值**: **整条线在 §136 被判 triple-contaminated, 无可用值。**

**曾报的值**:
- §125.5 (smoke, 2 tasks × 2 steps × 6 layers × 5 α = 128 generations): direction norm **L11=4.48 → L34=68.24 (15× 增长)**; L33 α=10 = **50% shift rate + 25% JSON valid** (最高格); mid-layer L11/L17 max shift **25% at α=2–5**, JSON 100% valid; late-layer L33/L34 shift 高但破坏 JSON 外壳; **L23 = 0% shift across all α (dead zone)**
- §126.3 (HDMI H-mean): **L17 α=5 H-mean 0.44** (c=29%, s=100%) 为 sweet spot; 对照 **L33 α=10 H-mean 0.23** (c=57%, s=25%, over-steer 破 JSON)

**已作废**: §126.3 两条被 §132b supersede (v2 split sweep held-out 数字取代)。§136 判 Method 4.4 整条线 **triple-contaminated** (buggy P-SoM baseline builder + 缺 [OPTIONS] + 未归一化 α)。

**caveats**: **smoke 规模 (2 task)**, 24-task 全量版当时在 DGX 跑 ~2h 未回; 原文对照文献: 「50% << Wu et al. 93% on tool calling, 但与 Ma & Rui 的 Qwen3-family pattern (probe works + causal patching uneven) 一致」; §126.3 site/n 未写明 (48 cells sweep)。

**证据**: §125.5 §126.3 §132b §136; `scripts/mechanistic/run_stage4_method44_v2_sweep.py`

**原文片段**: 「smoke 规模 (2 task); 24-task 全量版当时在 DGX 跑 ~2h 未回; 50% << Wu et al. 93% on tool calling」

---

## 56. Mirage task curation 与 tier 污染

**当前值** (§113.2, B1 local / cls / som vs phantom_som 各 15 token greedy continuation / 209 task scored): Tier A (top 7) composite **+4.00 to +4.20** | Tier B 8 tasks **+3.00 to +3.99** | Tier C 9 tasks **+2.50 to +2.99** | **strong total (composite ≥ 1.0 ∧ token_overlap < 0.5) = 24 tasks** | **reverse (composite ≤ −1.5) = 11 tasks**; 209/234 scored (25 skipped 因 step_002 artifact 缺); ~30 min compute。
Top-7 cluster 共享 signature (§113.2): task 0 blue kayak / 81 hurricane book / 112 basketball / 113 football / 127 MCAT prep book / 201 snare drum / 224 wall rack $30-40 —— source (有图) 全部 "do not show any X", target (无图) 全部 "show items/listings related to X"。

**演变**: §113.2 报 reverse=11 → **后续 Stage 2C 实际用 15 reverse tasks**。§128.4 发现 **Bug 1 (manifest tier 未过滤)**: cls 24 strong + 15 reverse; reddit 47 strong + 48 reverse (mixed); lexicographic first-24 必然混入 reverse (cls_task_4 / _10 排序靠前) → 「**paper 全篇写的 24 strong-tier 实际是 mixed tier**」。§127.1 另跑 P4 cls reverse-tier format variation (15 task) 作 selection-bias defense, pattern 同 strong-tier (marks-like L36 + dom L04) → **不是 tier artifact**。

**已作废**: 「24 strong-tier」这一标签在 Stage 4 pipeline 中不成立 (Bug 1)。

**caveats**: composite = mirage_score × (1 + divergence), mirage_score = (src_neg − tgt_neg) + (tgt_aff − src_aff) 是 **regex heuristic** (task-text-only, 无 patching leak 但 **plausibly 与 patching effect size 相关** → 这是 §117.5 selection-bias 2x2 control 的动机); Top-7「全部是 cls search-result page 上 image-grounded ground-truth absence 场景; **heuristic regex 命中非人工标注**」; §127.1 P4 是 **n=15 小样本, 结论是 qualitative pattern 一致, 未给统计检验**; archived pre-Phase-A obs。

**证据**: §113.2 §127.1 §128.4; `results/mechanistic/curate_mirage_b1_classifieds/candidates.md`; `scripts/mechanistic/run_stage4_multimode_extract.py`

**原文片段**: 「composite = mirage_score × (1 + divergence) ... 是 regex heuristic (task-text-only, 无 patching leak 但 plausibly 与 patching effect size 相关 → §117.5 selection-bias 2x2 control 的动机)」

---

## 57. Stage 4 NPZ pipeline 的三个 bug 与数据规格

**当前值**:
- **Bug 2 (SOM_MARKS lossy regex)** spot-verify (§128.4): cls_task_1 step_002 —— obs 4188 chars; **OLD `build_som_marks` 输出 38 chars / 3 行 vs NEW production `_extract_text_marks` 4169 chars / 72 行 → 71/72 marks dropped**
- **Bug 1 (tier 未过滤)**: 见主题 56
- **Bug 3 (AUROC in-sample)**: 见主题 50
- **Stage 4 multimode hidden state 规格** (§125.1): **(288, 37, 2560)** = 24 cls strong-tier task × 2 steps × 6 modes × 37 layers × 2560 hidden dim; 每 mode 48 examples, balanced (Myriad job 351370, 前 3 次 344630/344647/348257 失败)
- **Myriad P5a reddit format-variation** (§127.1): shape **(430, 37, 2560)**; 4/6 marks-like peak **L17**; hash_id_control **L04** (proper control); dom **L04** (24 task × 2 step × 10 mode, job 354382)

**已作废**: 所有 v1 NPZ 派生结论 (主题 48/49/53/54)。

**caveats**: Bug 2 是**单 step spot-verify 非全量统计**; 影响范围**仅 Stage 4 NPZ pipeline** (Method 4.2 / Exp 1 / Exp 3 / layer_axis_emergence), **Stage 2/3 patching 直读 archive 不受影响**; §125.1 「4 波修复才 land (manifest.json fallback / Phase 0 sentinel 接受 hidden_states.npz / FORCE_MATH_SDP=1 因 V100 sm_70 缺 bf16 cutlass kernel)」; ⚠️ §127.1 说 format-variation extract 的 regex **保留 label** (故 8-variant 内部对比 valid) 但 **§133b 表把该脚本 Bug 2 状态标为 ✗** —— 见矛盾清单 C7。

**证据**: §125.1 §127.1 §128.4; `scripts/mechanistic/run_stage4_multimode_extract.py`

**原文片段**: 「OLD build_som_marks 输出 38 chars / 3 行 vs NEW production _extract_text_marks 4169 chars / 72 行 → 71/72 marks dropped」

---

## 58. Mechanistic compute wallclock 与算力

**当前值** (§117.1/§118): Myriad **A100 80GB 上 Stage 2B/2C 单 cell ~30 min each** (V100 estimate 曾为 24h → **~48× speedup**); Cell F **42 min wall** (10× faster than V100 estimate)。§123: reddit 2x2 + random 三 cell 每个 30–50 min, 但 **queue wait 8h50min**。

**已作废**: V100 的 24h estimate。

**caveats**: 「**queue wait 另计**」(§123 记录一次 8h50min 排队)。

**证据**: §117.1 §118 §123

**原文片段**: 「~30 min each (V100 estimate 曾为 24h, ~48× speedup); Cell F 42 min wall (10× faster than V100 estimate)」+「queue wait 另计」

---

## 59. Provenance / reproducibility lock 值

**当前值**:
- **B1 模型 revision SHA** (§114.3): `Qwen/Qwen3-VL-4B-Instruct` = **`ebb281ec70b05090aa6165b016eac8ec08e71b17`** (DGX smoke 2026-05-07, 0 errors)
- **Evaluator code SHA** (§115.3, post-§95 reform): combined SHA256 = **`ba7a9276d59269be30bc8eb2...`**; per-file analysis.py `9d6559495b61977d...` / environment.py `e9a12798677fd233...` / metrics.py `b7361fe724ee4d70...`
- **VWA docker fingerprint** (§114.3): **10 containers captured** (postmill / Magento / classifieds-db / sentiment_postgres / hai-admin-postgres 等); reddit HTML hash captured; classifieds/shopping hash **偶发 timeout** (Tailscale latency, errors recorded fail-soft)

**已作废**: 无。

**caveats**: 「`transformers>=4.37.0` 是 **floor 不是 lock** (仍是 open drift 面)」; evaluator SHA **原文只给前缀 (截断)**, combined = 3 文件 concatenated bytes 的 SHA256; docker fingerprint 中 **cls/shop hash 缺失**, 「未来需 A100 self-host 部署后再跑一次做 diff」。

**证据**: §114.3 §115.3; `scripts/provenance/snapshot_env.py`; `results/provenance/vwa_dgx_via_quark.json`

**原文片段**: 「需写入 OSF preregistration; transformers>=4.37.0 是 floor 不是 lock (仍是 open drift 面)」

---

## 60. 跨模型对比 (Claude vs Qwen, scroll-heavy 子集)

**当前值** (§72):
- **10 个共享 task 三条线**: SR —— DashScope 官方 **50% (5/10)** / Qwen proxy DOM **30% (3/10)** / Claude SoM **30% (3/10)**; Scroll up 比例 **3.2% / 6.9% / 36.2%**; Stuck scroll **19.4% / 26.4% / 0%**; Parse error **0% / 0% / 0%**
- **20 个 scroll-heavy task 全集**: Claude Sonnet 4.6 SoM vs Qwen3-VL-235B proxy DOM —— SR **40% (8/20) vs 20% (4/20)**; 平均 **12.4 步 vs 24.9 步**; 成本/episode **$0.070 vs $0.102**

**已作废**: 无。

**caveats**: **n=10 / n=20 且是 scroll-heavy 子集非随机抽样**; 「三条线的 **observation mode 不完全一致** (Claude 是 SoM, Qwen proxy 是 DOM)」; 原文明写「**该数据仅作 discussion 素材不进主实验**」。

**证据**: §72

**原文片段**: 「n=20 且是 scroll-heavy 子集; 模式不同 (SoM vs DOM); 该数据仅作 discussion 素材不进主实验」

---

## 61. 文献 fact-check 与工业 landscape

**当前值**:
- **国产 DR V2 报告 fact-check** (§109.18, 1188 行, 抽样 8 arXiv + 4 GitHub): arXiv IDs **0/8 真** (每个 fetch 后都指向无关论文); GitHub repos **2/4 存在** (alibaba/page-agent 真 17.5k stars v1.8.1 2026-04-27 vs V2 claim 12.4k v2.1.0 2026-03-22; bytedance/ui-tars 真 10.2k vs claim 15.7k; aliyun/alibabacloud-webagent 404; THUDM/Open-AutoGLM 404); **stars/版本号/release 日期全部 fabricated**; 量化指标表 (token/latency/SR/DAU) **全表不可用**; Layer 1 中国部署具体数 (1200 站点 / 30万小程序 / 98% 覆盖) fabricated 信号
- **(ii)×L2 工业 instance sweep** (§109.12): 从 1 instance (OmniParser-v2) 扩到 **11+** —— agent-browser (Vercel Labs, 81+ releases, v0.26.0 Apr 2026, CDP-direct Rust, dual-mode @eN / SoM [N], 200–400 tokens/snapshot) / Playwright MCP (Microsoft, a11y tree + [ref=e5] + incremental snapshots, 200–400 tokens) / Tarsier (Reworkd v0.6.0 Jun 2024, typed brackets, 内部 benchmark unimodal text beats GPT-4V + Tarsier-Screenshot by 10–20%) / Stagehand v3 / Browser Use SDK (31%→26% drop when search disabled) / Skyvern / Anchor Browser ($6M seed) / AgentQL / MultiOn / OpenClaw (361K GitHub stars by early 2026)
- **Hermes 文献副驾 first run** (§126.1, 5 天窗口 2026-05-06 至 05-11): **200 arxiv submissions + 29 GitHub repos → 11 hits (筛除率 95%)**: 1 篇 score 3 + 5 篇 score 2 + 5 个 score 1; **0 个 false positive**

**已作废**: 国产 DR V2 的**引用层与量化层**整体不可用。

**caveats**: §109.18 是**抽样验证非全量**; 原文保留正面判定 ——「9-cell mapping framework 本身 **structurally correct ✅**, 行业共识定性方向 **direction correct ✅** —— 只有引用层与量化层不可用」; §109.12 「Tarsier 的 10–20% 是**厂商内部 benchmark 非同行评审**; star 数 / 版本号是**时点快照**; 部分数字来自 README/发布页**未逐条 fetch 验证**」; §126.1 「**0 FP 是 agent 自己 self-audit 声明 (本期没明显 FP), 非独立核验**」。

**证据**: §109.12 §109.18 §126.1; `docs/literature/2026 年国产 Web Agents 技术全景与深度分析报告2.md`

**原文片段**: 「arXiv IDs 抽 8 个 → 0/8 真 (每个 fetch 后都指向无关论文) ... stars/版本号/release 日期全部 fabricated」

---

## 62. Cross-AI audit (codex) 的 meta 测量

**当前值**:
- **§128.3 codex /codex-stress Mode A 首跑**: 抓到 **5/6 weak claims** 是 Claude /stress 完全没 surface 的; output **2911 行**
- **§128.4 codex prompt 设计 meta-experiment**: v1 directive (persona + 8 bug categories + ~30 bullet questions) → **8141 行**, 多为 raw code exploration, **没生成 final structured verdict**; v2 lean (persona + scope + output format, STOP) → **6484 行**完整 6-section 报告, **6 个 paper-grade methodology bug (4 HIGH)**

**已作废**: 无。

**caveats**: §128.3 的 catch rate 是**自评统计**, 第 6 项与 Claude 有 overlap (W6 tokenization small-n caveat); §128.4 是 **n=1 对照, 两次 prompt 不只 lean/directive 一个变量不同**, 但「结论 (lean 更好) 被作为 memory rule 固化」。

**证据**: §128.3 §128.4; `docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md`

**原文片段**: 「n=1 对照, 两次 prompt 不只 lean/directive 一个变量不同; 结论 (lean 更好) 被作为 memory rule 固化」

---

## 63. 代码库 / 文档 bookkeeping 计数

**当前值 (全部为时点快照, 会继续增长)**:
- **§97 audit 规模**: ⚠️ 同一 § 内三处互不一致 —— 总结小节 **26 文件 / ~17500 行 + 13 YAML, 9 轮 audit, ~90 处真 bug + ~50 处 medium**; 完整审计统计小节 **13 个文件 / ~12000 行, ~70 处真 bug + ~50 medium**; 同表格合计行 **~16k 行 / ~80 修复** (见矛盾清单 C6)
- **§97 单测基线**: **81 测试全过 (74 原有 + 7 smoke)**; 演进 65 (§85) → 70 (§86/§87) → 74 (§88 报 76 passed / §97 前几轮报 74) → 81
- **§108.18 paper.bib**: 16 entries (Day 1 §104) → 51 → 56 (Q5 加 5) → **57** (加 anon2026toolcalling); §127.6 进一步 **61 → 67 entries / 578 → 638 行**, zero NEEDS_BIB remaining, 2 NEEDS_VERIFY 留在 bib note
- **§126.2 paper §5 mechanism lit anchor**: 4 篇 → **7 篇** (加 Peale 2605.07805 / Khorasani HDMI 2605.07631 / Lin&Liu 2605.08012); wu2026toolcalling eprint placeholder 2605.XXXXX 更正为 2605.07990
- **§108.20 实验笔记 tag 分布** (§1–§107, 2026-05-02): **107 entries 打标: 19 #finding / 5 #literature / 31 #bug / 44 #infra / 8 #design**
- **§116 pre_rerun_audit.md gate item 增长**: §116.3 12 sections → §116.11 13 sections ~150 items ~350 行 → §116.12 4 phases × 18 sub-sections ~150 items ~455 行 → §116.13 5 phases 22 sub-sections ~195 items ~610 行 → §116.14 **~210 items ~640 行** (2026-05-08 单日 8 轮 user audit prompt)
- **§116.8/§116.9 master_bug_catalog status**: 🛠️ FIXED 1 → ~45 → **~50** | ✅ CONFIRMED 1 → ~28 → **~10 (deferred with rationale)** | ❌ NOT_A_BUG 3 → **4** | 🔄 UNVERIFIED **0** (all triaged via static read)
- **§2 GLM-5.1 sidecar 归因**: **211 条自动归因 + 2 条人工补充** (task_184 / task_187); 置信度 **96.7% high / 3.3% medium**
- **§104 数据体积**: 单 condition run **1.7GB** (artifacts ~100MB / JSONL 仅 15MB); 全 results/ **21GB** (2026-04-28 时点)

**已作废**: 各为快照, 无正式作废。

**caveats**: §97 三处统计原文明写「**未 reconcile — 引用时必须说明取哪一处, 不要相加或平均**」; 单测「各 § 报的数字不同, **不要当同一时点**」; §116 item 计数是「**原文自述近似值**」, 状态分布 ~10 已在位 / ~15 partial / ~25 TBD (§116.10 时点); §116.8 的「~」是原文自述近似值; §2 的置信度是「**GLM 自报, 非外部校验**」; §127.6 的 NEEDS_VERIFY (zhang2024patching 等) 待 pre-submission 核验。

**证据**: §2 §97 §104 §108.18 §108.20 §116 §116.8 §116.9 §126.2 §127.6; `docs/reference/master_bug_catalog.md`; `docs/checkpoints/paper_drafts/paper.bib`

**原文片段**: 「⚠️ 同一 § 内三处统计互不一致 (26 文件/17500 行/90 bug vs 13 文件/12000 行/70 bug vs 表格 ~16k/~80), 未 reconcile — 引用时必须说明取哪一处, 不要相加或平均」

---

## 64. 运行吞吐与 GPU 争抢

**当前值** (§104): B1 classifieds phantom_dom runner rate 从早期 **~7.6 ep/h 降到 4 ep/h**; 实测 **1.36 min/task → ETA ~9h** (原估 ~16h)。

**已作废**: 原估 ~16h ETA。

**caveats**: 共享 GPU 争抢导致 (seonglae 5 train_intervention 进程 + StreamWriter 8.5GB 占 95% util); stall 4h 后 SIGTERM + relaunch。

**证据**: §104

**原文片段**: 「rate 从早期 ~7.6 ep/h 降到 4 ep/h; 实测 1.36 min/task → ETA ~9h (原估 ~16h)」

---

## ⚠️ 矛盾清单

> 同一个量在不同 § 有不同值、且原文没说清哪个对的。**两侧并列, 不选边。**

### C1. VWA+WA 的 task 总数与 visual 占比
- **§21**: VWA only **910 tasks, 67% visual / 33% non-visual**; VWA + WA 免费三站 **1415 tasks, 52% visual / 48% non-visual**
- **§89**: VWA only **910 tasks (~608 visual / ~302 non-visual) = 67% visual**; VWA + WA **~1390 tasks (~608 / ~782) = 44% visual**
- VWA-only 侧一致 (910 / 67%); **VWA+WA 侧 1415 vs ~1390 且 52% vs 44% 不一致**。原文两处均标「未 reconcile, 不要合并或平均」。

### C2. B0 classifieds 三模式 adjusted SR (visual_fp 时代)
- **§77 分类索引表行**: SoM **21.43%** / Vision **11.61%** / DOM **8.04%**
- **§89 表**: SoM **20.98%** / Vision **12.05%** / DOM **8.48%**
- 原文两处均标「不一致未 reconcile, 不要合并」。(注: 两组均属已作废的 visual_fp 时代定义。)

### C3. VWA visual task 占比的两套定义
- **§89 自动列表** (`_load_visual_task_ids`): cls **69.2%** / reddit **84.3%** / shopping **57.7%**
- **§95 Codex 手动审计**: VWA 整体 **95.3%**; cls **96.2%** / reddit **99.5%** / shopping **92.9%**
- 原文: 「两种不同定义, 数字**不可比、不可互换引用**」。另: §95 审计文件 docstring 写 'manual' 实际是 codex 判定, §100 要求重新独立审计。

### C4. B1 reddit 的 SoM SR (三处口径)
- **§94**: DOM adj **6.83%** > SoM adj **5.85%**
- **§100 (a)**: DOM **10.0%** > SoM **8.1%** > Vision **4.8%**
- **§100 (b), 同一 § 内另一处**: "B1 reddit task **SoM 4.76%** 反转 vs B0 reddit task SoM +0.5pp 微正"
- 原文: 「三处口径 (raw vs adjusted vs 其他) **未标明, 严禁互换或合并引用**」。

### C5. Routing signal AUROC 的口径
- **§104 (verbalized confidence)**: B0 reddit P-text **0.793** = 5-mode 最高, baseline **0.766**
- **§124.7 (fig0b_extra)**: B0 cls best **0.846** (P-text), B0 reddit **0.817** (P-prompt)
- 原文: 「不同 metric/口径, **不可混**」。
- (另有 §7 的 token-level **0.497**, 那是不同信号源, 不构成同量冲突。)

### C6. §97 audit 规模的三处自述
- 总结小节: **26 文件 / ~17500 行 + 13 YAML / 9 轮 / ~90 真 bug + ~50 medium**
- 完整审计统计小节: **13 文件 / ~12000 行 / ~70 真 bug + ~50 medium**
- 同表格合计行: **~16k 行 / ~80 修复**
- 原文: 「未 reconcile — 引用时必须说明取哪一处, **不要相加或平均**」。

### C7. Stage 4 format-variation 脚本的 Bug 2 状态
- **§127.1**: format-variation extract 的 **regex 保留 label** (与 multimode Bug 2 不同) 故 8-variant 内部对比 **valid**
- **§133b 表**: 把该脚本的 Bug 2 状态标为 **✗ (codex C1 P0 documented)**
- 原文: 「两处判断不一致, **未在本 chunk 内收口**」。

### C8. SoM 非交互 click 占比 (§96 vs §101)
- **§96 (classifieds)**: B0 非交互 click 占总 click **66/524 = 12.6%**; B1 **307/957 = 32.1%**
- **§101 (classifieds)**: B0 total clicks 524, non-interactive **11.3%**; B1 total clicks 957, non-interactive **30.0%**
- total clicks 分母相同 (524 / 957) 但比例不同。台账未 reconcile。
- [聚合者推论] 可能是"非交互"判定口径微调, 但**原文没说**, 不选边。

### C9. B0 VWA shopping DOM 的 SR
- **§81 follow-up**: **16.52% (77/466, 含 task 345 as fail)**
- **§103**: run `B0_3mode_shopping_20260421` DOM n=466 —— **raw 11.8% / adjusted 0%** (Magento auth bug 期间, 数据已 cleared)
- §81 原文自标: 「行动规划另记 `B0_3mode_shopping_20260421` (DOM 466 ep) 已因 broken auth 被 cleared — **需确认 16.52% 对应哪个 run 才能用**」。

### C10. master_bug_catalog 的 NOT_A_BUG 列表
- **§116.8**: B-12 / B-13 / B-14 / B-27 (4 个)
- **§107 表格**: B-12 / B-14 / B-27 (3 个)
- 原文: 「两处不完全一致, 按各自原文抄」。

### C11. Early-stop spec-vs-code drift 的起算日期
- **§116.5 原文**: 4/30 advisor 同意 → 5/8 用户问「关了吗」= **14 天**
- **§110 advisor sync 记录**: 该 sync 在 **5/5**
- 原文: 「日期在笔记内**不自洽, 按原文抄录**」。

### C12. paper §4 reddit P-text 的 SR (已由作者裁定, 但两值都曾出现)
- 两处不一致: **11.90** vs **12.38** → **§128.4 canonicalize 为 12.38** (commit 00076b1)
- 注: §104 FRESH cell 报的是 raw 13.81 / **adj 11.90**。裁定为 12.38 后, 11.90 的来源口径台账未再说明。

### C13. B-33 family blast radius
- **§106**: **1.6%**
- **§107**: **3.0% all ep, 55.9% of click-loop signature** (原文明标"远超 §106 报的 1.6%")
- 这一条原文给了方向性裁定 (§107 是全量扫描), 但两值口径不同 (ep% vs signature%), 并列备查。

---

## 未归主题的孤条

- **§104 数据体积快照**: 单 condition run 1.7GB (artifacts ~100MB / JSONL 15MB), 全 results/ 21GB (2026-04-28)。—— 已并入主题 63, 此处备注其为纯时点快照。
- **§116.5 early-stop drift 持续 14 天** (`p79/experiment/runner/main.py` vs `preregistration.md`): 原文注「若无这次 audit, 16-cell rerun 数据会被 contaminated」。属流程时长测量, 无同类可聚。
- **§128.4 Stage 3 Holm baseline 的数值影响**: L35 patched 的 overlap→target **≈ 1.00**, 用于论证 Bug 6 (用 L35 patched 作 baseline 而非 unpatched) 数值影响 ≈ 0 → **只文档化不改数, 属 defer 项**。(已在主题 40 caveats 引用。)
- **§68 AXTree 文本相似度的假阴性例证**: task 4 type 小范围编辑 `text_similarity=0.995` (阈值 0.95) 被判 `action_success=False`。n=1 step 例证, 原文未标 baseline (连 baseline 是 B0 还是 B1 都没写)。
- **§65 Vision 裸截图下 GLM digest 的坐标偏移检出率**: **23.6% (实际 >40%)**。原文明注「'实际 >40%' 是**估计不是精确测量**」; 修法 = 优先加载 `screenshot_annotated.png`。
- **§103 B0 shopping DOM (Magento auth bug 期间)**: raw 11.8% / adjusted 0% —— 「adj=0% 即所有 raw success 都被 FP 过滤; 数据已 cleared」。已进矛盾清单 C9。
