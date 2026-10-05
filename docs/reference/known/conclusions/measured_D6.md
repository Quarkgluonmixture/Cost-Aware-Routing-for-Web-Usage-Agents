---
type: conclusions
batch: D6
status: done
created: 2026-10-06
source: scratchpad/batches/D6.jsonl (170 条 MEASURED，§475.2–§527.2，2026-08-21→09-15)
---

# 测量结论 D6 (§475.2–§527.2)

> **读法**：每个主题的「当前值」是现在算数的数字；「已作废」的数字**禁引**。
> 本批时期：§475 匿名化与 ARR 投稿准备 → 毕设收尾（09-08 提交 119 页）→ reframe chain 跑 B5（GPT-5.6 terra）与 B0·reddit replicate，
> 噪声地板从 cls·B0 扩到 reddit / B5 / B1·red，**C1（serving-path 分组）提出后被 §497 连续攻击、gap 从 4.26pp 缩到 1.48pp**；
> §505（09-09）一天内把 router 全链重想（lookahead / bandit / 规则 / cascade / 方差分解），全部收敛到「对 random +1–2pp、对固定策略无 Pareto 胜」，
> 唯一过重跑存活的是**预算路由（step cap）**，并在 shop_B1 两个 held-out 臂 prospective 2/2 PASS（方向判据）；REALM 接收；showcase / 演讲准备。
> ⚠️ **本批之后**：§529.4（2026-10-06）登记 WA·B1·reddit 六对复测，**C1 已按预注册撤回**，总噪声带上沿 3.45pp → 5.77pp —— 凡涉及处在主题末尾标注。
> **noise 类数字一律各自带 scope 并列，禁止做加减法**（沿用 §397.10）。同一主题 §474 以前的演变见 `measured_D5.md`（并行批）。

---

# A. 噪声地板 / replicate / C1

## A1. B0·reddit 三个 phantom 臂的同 condition 地板

> §474 以前见 measured_D5.md

**当前值（B0 / VWA reddit / canonical scored universe n=203, AMENDMENT_08）**:
B0.red.ptext **7.39%**（self_drop 5.42/1.97pp）· B0.red.pprompt **11.33%**（6.90/4.43）· B0.red.psom **10.34%**（3.45/6.90）；
red_band: band **1.97-6.90pp** / observed **2.46-3.45pp** / one_sided_95 **3.14-3.89pp**。
§500.2 drift 检验（canonical 2026-06 vs replicate 2026-09，n=205 common）：flip 方向不对称比 som **11/6 = 1.83:1** · dom **13/7 = 1.86:1** · vision **6/6 = 1.00:1**，**三条全部 < 2:1**；start_url_mismatch 均 0，全部 flip 归 model_nondeterm ⇒ **7.39% 站得住，§478.4 的 CLAIM_UNVERIFIED 可关**。

**演变**:
- §478.4（按 205 手算，P-SoM 当时未完）：P-text 16/205 = 7.80%（SR 13.66%→9.76%）· P-prompt 25/205 = 12.20%（12.68%→10.24%）· P-SoM 19/165 = 11.52%（11.52%→13.33%）；P-text A-only 12 / B-only 4 强不对称 ⇒ 「登记前须先排除站点状态漂移」。
- §479.1 canonical 登记值（n=203）取代 §478.4。
- §496.3 再提：7.39% 端「仍挂着 §478.4 的 CLAIM_UNVERIFIED …至今未验」。
- §500.2 用三条新臂（som/dom/vision）检验：事前判据「≥2:1 且同号 ⇒ drift」未触发 ⇒ 3:1 是 ptext 那条臂自己的性质。

**已作废**: §478.4 的 **7.80 / 12.20 / 11.52%**（205 口径、P-SoM 基于 165/205）—— 台账 superseded_by 明写「仅为中间量」。⚠️ superseded_by 文字把 P-SoM 写成「11.2」，value 字段是「11.52%」，两处不一致（见矛盾清单 #2）。

**caveats**:
- 「red_band **只对文本侧成立** — reddit 没有 replicated 的 dom/som/vision 臂, 不可当作该站点图像侧的地板」（§479.1 时点；§500 起 reddit 已六臂全 replicate，见 A9）。
- 「阈值须按对口臂读 (§477.2), 不可与 cls_band 混取 min/max」。
- §500.2：「三条**方向同号**(archive 侧独有 PASS 更多), 只是幅度未达阈值 —— 弱同向趋势仍在, 不是零」。

**证据**: §478.4 / §479.1 / §496.3 / §500.2；`docs/analysis/cross_sites/noise_floor_inventory.json`、`results/visualwebarena/phase1/B0_phantom_{text,prompt,som}_reddit_2026*`、`aggregate_noise_floor_inventory.py` CLEAN_PAIRS 注释

**原文片段**: 「这是 canonical 值, **取代**同日按 205 手算的 7.80/12.20/11.2 (§478.4)」(§479.1)；「三条全部 < 2:1 … 7.39% 站得住」(§500.2)

⚠️ 2026-10-06 §529.4 已撤回 / 已变化，见实验笔记（C1 撤回；总噪声带上沿 3.45pp → 5.77pp）。

---

## A2. 噪声带字段（cls_band / floor_band / observed）与「floor 是否 mode-dependent」

> §474 以前见 measured_D5.md

**当前值**:
- §479.1 登记 reddit 三对后回归：**cls_band 逐字段不变**（band **4.46-7.59pp** / som **5.36-7.59** / som_absdiff **2.23**）；floor_band 判据字段 one_sided_95 **0.0/4.15 不变**（reddit 三臂 1s95 3.14-3.89 均低于 4.15）；仅叙述性字段变：n_draws **10→13**，observed_max **2.68→3.45pp**；fire manifest 11 个 replicate 全识别为已注册。
- 毕设 ch4:464 明写引用 'the six replicated cls·B0 arms'，范围 absdiff **0.89-2.68** / disc **10.27-14.29%**，未变。
- §504.4（审稿人 4s7L 问题，「现在可测」）：`B0·cls` 六臂 discordance **10.27–14.29%**；`B0·red` 六臂 **4.93–11.33%**（`noise_floor_inventory.md` 第 1 节）。
- §477.2 毕设 exact 阈值口径：「95% (exact)」= **零分布的 95 分位数**（X~Bin(d,1/2), D=(2X-d)/n, n=224）；六臂 exact: ptext **3.12** · pprompt **3.57** · dom/som/psom **4.02** · vision **4.46**。

**已作废**: §477.2 自记的一次误算 —— 用「最小拒绝阈」口径算 dom(d=27) 得 **4.91pp**（论文表 4.02pp），曾据此误判「论文算错了」。

**caveats**:
- 「observed_* 只出现在 fusion_premium 的 scope_warning 叙述文字里, 不参与任何判断 (判据是 one_sided_95)」。
- §504.4：「这是**读数不是结论** —— 尚未做『floor 随 mode 变』的统计检验 … **不进 camera-ready** … 记为 ARR 高优先项」；B0·red 下界 4.93% 来自 vision 臂（不携带 element id），「方向与 4s7L 的机制假设一致, 但 n=1 不构成证据」。
- §477.2：「**两种口径能差一个格点**(d=27 时 4.02 vs 4.91), 复算前必须先对齐口径」。
- ⚠️ 本批另有两处把「重跑噪声带」写成 **0.89-2.23pp**（§490.1）/ **2.23pp**（§491.6），与上面 0.89-2.68 / som_absdiff 2.23 的 scope 关系台账未说明（矛盾清单 #1）。

**证据**: §477.2 / §479.1 / §504.4；`docs/analysis/cross_sites/noise_floor_inventory.json`、`noise_floor_inventory.md`

**原文片段**: 「cls_band 逐字段**不变** … 仅叙述性字段变: n_draws 10→13, observed_max 2.68→3.45pp」(§479.1)

⚠️ 2026-10-06 §529.4 已撤回 / 已变化，见实验笔记（总噪声带上沿 3.45pp → 5.77pp）。

---

## A3. B5（GPT-5.6 terra）的同 condition 地板 —— 第二个 API 模型

> §474 以前见 measured_D5.md

**当前值（B5 / classifieds / dom / n=224 / A1 R29736 ↔ A2 R15476, pre-declared pair）**:
**29/224 = 12.95%**（A-only 13 / B-only 16）；net ΔSR **+1.34pp**。
§505.10 对照：dom 重跑 discordance **12.9% vs 12.1%**（B5 vs B0）；稳定 som→dom route-away **7/13（3.1pp, band 5.8–7.1）**。

**演变**: §478.3 首报时「尚未登记进 CLEAN_PAIRS」→ §505.2 起 18 对 CLEAN_PAIRS 已含 `cls_B5 dom`（见 C1）。

**caveats**:
- 「落在 B0 的 10.3-14.3% 带内 ⇒ … 地板是 **API serving 的属性**, 不是 B0/Qwen-MoE 特有。⚠️ 这**不**证明机制相同, 只证明两个独立厂商的 API 都有同量级地板」。
- B5 只有 dom 一条 replicate，六臂方差分解做不了（§505.10）。

**证据**: §478.3 / §505.10；`scripts/analysis/aggregate_noise_floor_inventory.py`（§478.3 时「待登记」，artifact_exists=false）

**原文片段**: 「29/224 = 12.95% (A-only 13 / B-only 16); net ΔSR +1.34pp」(§478.3)

⚠️ 2026-10-06 §529.4 已撤回 / 已变化，见实验笔记（C1 撤回；本条是 C1 API 组的一臂）。

---

## A4. 本地 backbone（B1 / B2）replicate 与功效

> §474 以前见 measured_D5.md

**当前值**:
- **B1 × VWA-reddit**（§496.1，本地 bf16，原 run 07-03·07-06 ↔ 重跑 08-30·09-01）：som **4/205 = 1.95%**（SR 8.3%→7.3%, Δ−1.0pp）· dom **7/205 = 3.41%**（SR 6.8%→6.3%, Δ−0.5pp）；7 个 flip 全部 model_nondeterm；reset-goto scan 205/205 一致；kappa **0.864 / 0.722**。203 口径：som 4/203=**1.97%** / dom 7/203=**3.45%**（「不改判读格」）。实测功效 d≈**8.8 (som) / 7.6 (dom)**，两格均 < 门槛 10 ⇒ **inventory only, 不得报 CI**。
- **cls_B1**（§505.24）：som/vision 的 replicate 与 canonical **逐 task 全同（0 翻转）**。
- **B2 replicate 的结构上限**（§515.4，= 两次 SR 之和，假设 replicate SR ≈ 原 SR）：B2·cls·som / B2·cls·vision（SR 2.23%）≤ ≈**4.46%** < API floor 下界 4.93% ⇒ **结构上不能证伪 C1**；B2·red·dom（SR 3.90%）≤ ≈**7.8%**，是三格中唯一可能越过 4.93% 的。

**演变**:
- §480.5 事前预测（d ~ n × SR × 0.59, n=203）：som SR 8.29% d=9.9 · dom 6.83% d=8.2 · phantom_som 6.83% d=8.2 · phantom_text 6.83% d=8.2 · phantom_prompt 6.34% d=7.6 · vision 2.93% d=3.5 —— **全部低于 d>=10**；吞吐实测 5.58-6.49 ep/h ⇒ 约 35h/mode。
- §496.1 实测落地，d 与预测同量级。

**已作废**: 同日早些时候「**B1×reddit 是 19 天窗口最划算一笔**」的表述 —— 被 §480.5 推翻（买不到区间）；价值改为「inventory 级 local-reddit 点仍能**证伪** C1」。

**caveats**:
- §496.1：「两对**尚未登记 CLEAN_PAIRS** —— 登记会动噪声地板 canonical, 属 estimand-adjacent, 留给 user」（§500.3 时 local 组已为 5 臂，见 A5）。
- §505.24：cls_B1 0 翻转「部分反映的是 replicate 缺乏独立性」。
- §515.4：「0.00% 在『本地确定性』与『几乎不成功』两种解释下都成立, 不区分二者; 上限是必要条件不是期望值」；replicate SR 若升到 ≈2.7% 以上，cls 两格上限随之超过 4.93%。
- 0.59 启发式本身的不确定见 A6。

**证据**: §480.5 / §496.1 / §505.24 / §515.4；`results/visualwebarena/phase1/B1_*reddit*`、`docs/analysis/cross_sites/serving_mode_floor.md`

**原文片段**: 「实测功效 d≈8.8 (som) / 7.6 (dom), **两格均 < 门槛 10** ⇒ inventory only, 不得报 CI」(§496.1)

⚠️ 2026-10-06 §529.4 已撤回 / 已变化，见实验笔记（C1 撤回）。

---

## A5. C1 —— 可复现性地板按 serving path 分组（serving_mode_floor）

> §474 以前见 measured_D5.md

**当前值（截至 §515.4，即撤回前最后一版 = §500.3，18 对 clean replicate）**:
API **13 臂** floor **4.93-14.29%**；local **5 臂** floor **0.00-3.45%**，且首次跨两站；gap **1.48pp**，仍不重叠；精确单边秩检验 **p=0.0001**。
限定到 d>=10 的 powered 子集：local 只剩 classifieds 两条（**0.00/0.00%**），gap **7.39pp**，**p=0.0110**。

**演变**:
- §480.1（13 对）：API 10 臂（Qwen+OpenAI, 2 站点）**7.39-14.29%** vs local 3 臂（Qwen, 1 站点）**0.00-3.12%**；gap **4.26pp**，p=**0.0035 (1/286)**；只看 d>=10：API 10/10、local 2/3，gap **7.39pp**，p=**0.0152**。
- §496.3（B1×reddit 落地后）：local 上界 **3.12% → 3.41%**（B1·red·dom，205 口径）；API 下界仍 **7.39%**；gap **4.26 → 3.98pp**；「≥7.39% ⇒ C1 撤回」未触发。
- §500.3（五格 replicate 注册后）：见当前值；新下界 4.93% 由 B0.red.vision 持有。
- §515.4：B2 replicate 的结构上限论证（见 A4）。

**已作废**: gap **4.26pp**（§480.1）与 **3.98pp**（§496.3）被 §500.3 的 **1.48pp** 取代；§496.3 时 `serving_mode_floor.md` 表内「local 0.00–3.12% / gap 4.26pp」是 stale（原文明记未重算）。

**caveats**（尽量一字不改）:
- 「**未拆 scale** — API 组 235B/未公开, local 组 4B; 第二个 API 家族拆掉的是『家族』读法不是『规模』读法」。
- 「**不做机制归因** (§302.5 裁定: 停在 observable provider-dependent floor)」。
- 「臂间不独立 … 该 p 是描述性分离统计量, **不可作 gate**」。
- 「local 侧只有一个家族且**买不到第二个** — B2 的 d≈1.8 是功效限制不是排期问题」。
- §500.3：「**『local 组跨两站』只在 inventory 级成立** … **不能只看站点数下跨站结论**」；「新下界 4.93% 由 B0.red.vision 持有, 而它是事前声明的 inventory-only 臂」。
- §496.3：「**gap 的两端并非同等可信**」（7.39% 端当时未验；§500.2 已关，见 A1）。
- 统计与 provenance 层面的攻击见 A6。

**证据**: §480.1 / §496.3 / §500.3 / §515.4；`docs/analysis/cross_sites/serving_mode_floor.{md,json}`、`naacl_evidence_delta.md` C1

**原文片段**: 「API 13 臂 (原 10) floor **4.93-14.29%** … gap 4.26 → **1.48pp**, 仍不重叠, 精确单边秩检验 p=0.0001」(§500.3)

⚠️ 2026-10-06 §529.4 已撤回 / 已变化，见实验笔记（WA·B1·reddit 六对复测登记后，C1 按预注册撤回；总噪声带上沿 3.45pp → 5.77pp）。

---

## A6. C1 的统计 / provenance 攻击（§497 codex Mode B + §497.7 Claude 复算）与单位陷阱

**当前值**:
- **伪重复单位**（§497.2）：13 arm 按 arm exchangeability **1/286=0.0034965**；按 4 cell mean 折叠 **1/4=0.25**；按 3 backbone mean **1/3**；仅复制一个已有 API arm 即变为 **1/364**。§497.7 独立复算：C(13,3)=286 ⇒ p=0.0035，复制一臂 ⇒ C(14,3)=364、p 降到 **0.0027**；cell 折叠 0.25，backbone 折叠 0.33。
- **powered 假标**（§497.2 / §497.7）：B1·cls·vision/som 实际 discordant_count **均为 0**，producer 用 n×mean_SR×0.59 算得 d=**16.52/18.88** 并标 powered=true；`serving_mode_floor.md:44` 的 B1.cls.som 标 floor=0.00% / d=18.9 / interval=yes。
- **SHA 覆盖边界**（§497.3）：SHA 只对 sorted task-ID list 做；reddit 203-ID SHA=`1ce29c8b...15c92`，删 task 0 后=`6155abc7...67d8c`；**13/13 clean_pairs 均无 SHA field**，`serving_mode_floor.json` 无 SHA/universe path。
- **单位陷阱**（§480.4）：clean_pairs 同一条记录里 `sr_a`/`sr_b` 是小数（0.25），`discordance_pct` 是百分数（14.29）；多除一次 100 会使 d 缩小 100 倍，**十三条臂全部误判为欠功效**；已在 `serving_mode_floor.py` 加 0<=sr<=1 断言。

**caveats**:
- 「cell/backbone 折叠 p 也不是推荐的替代 gate; serving 没有随机化且与 scale 共变, 此 probe 只证明 1/286 的单位伪重复」。
- 「0.59 是事前采购启发式, 三锚点 0.46/0.58/0.74 跨 1.61×; 本条攻击的是它被事后当作 exact interval-carrying gate」。
- 「inventory.margins 存在 site universe_sha, 但 C1 producer 只读 clean_pairs; ID-membership hash 本身也不覆盖 task/evaluator 内容或 exclusion-policy 版本」。
- §497.7：§497.1 与 §497.4 依赖 codex 自述 probe，「Claude **未独立复跑**」⇒ 置信度低于另三条；「本条只核算术与字段, 不代表认可 §497 的处置建议」。
- §480.4：「该错误的后果**不是数字略偏**, 而是一张会宣称『本项目没有任何可带区间的地板』的表 — 方向性错误」。

**证据**: §480.4 / §497.2 / §497.3 / §497.7；`scripts/analysis/serving_mode_floor.py`、`scripts/analysis/lib/canonical_task_universe.py`

**原文片段**: 「B1·cls·vision/som 实际 discordant_count 均为 0, producer 用 n×mean_SR×0.59 算得 d=16.52/18.88 并标 powered=true」(§497.2)

⚠️ 2026-10-06 §529.4 已撤回 / 已变化，见实验笔记（C1 撤回）。

---

## A7. C2 —— 「加一条新表征 ≈ 加一次重跑」在第二个站点

**当前值**: red·B0：**+4.93pp**（加 DOM）vs 实测重跑 band **1.97-6.90pp**（文本侧）⇒ **inside the rerun band**；该 cell 此前为 'no floor on this cell'。现共 **3/8 cell 有 band**（cls·B0 / red·B0 / WA·B1）。

**caveats**: 「reddit band **只覆盖文本侧三臂**; 被加的臂恰为 DOM (文本侧) 故对口成立 … 该 side gating 已程序化 (red_band.replicated_side + side_of_arm)」；「band 仍是若干 draw 的范围, 不是 bound」。

**证据**: §480.2；`docs/checkpoints/paper_drafts/ablation_tables.md` Table 18、noise_floor_inventory head_to_head

**原文片段**: 「red·B0: +4.93pp (加 DOM) vs 实测重跑 band 1.97-6.90pp (文本侧) ⇒ **inside the rerun band**」(§480.2)

⚠️ 2026-10-06 §529.4 已撤回 / 已变化，见实验笔记（总噪声带上沿已变）。

---

## A8. label instability（cls_B0 flip 富集）

> §474 以前见 measured_D5.md

**当前值（cell `cls_B0`, n=224, 六臂各重跑一次）**: 整格 flip **86/224 (38.4%)**；contested flip rate **81.8%**；补集 **10.3%**；enrichment **7.9×**（§504.3）/ **7.95x**（§479.3）；leave-replicated-out 控制 **不可用 (None)**。

**演变**（§504.3 汇总）: 两臂 → 三臂 → 六臂：flip **49/224 → 67/224 (29.9%) → 86/224 (38.4%)**；contested **48–52% → 67.0% → 81.8%**；补集 **2.9% → 5.9% → 10.3%**；enrichment **17.4× → 11.4× → 7.9×**。§479.3：replicated_arms 3 → 6，n_flipped 67 → 86，enrichment 11.4x → 7.95x，leave-replicated-out 2.92x → None。

**已作废**: 无（各版均为当时真实状态）。REALM #192 Table 29 引的是 08-06 快照（49 flips / two arms / 17.4x），「属提交时真实状态, 不改」。

**caveats**:
- 「7.95x **不可单独引用** — 它已失去反循环控制 (B-1995)」。
- 「**定性增强但 enrichment 单调下降** … 引用时必须同时给两个数(contested 率 + 补集率)」；产物自述每个数都是 flip 率的 **LOWER bound**。
- 毕设不引用这些数（已 grep 确认）；`red_B0` 虽六臂全 replicate 但未跑这套分层。

**证据**: §479.3 / §504.3；`docs/analysis/cross_sites/label_instability.json`

**原文片段**: 「整格 flip 49/224 → 67/224 (29.9%) → 86/224 (38.4%); contested flip rate 48–52% → 67.0% → 81.8%」(§504.3)

---

## A9. unique-solve 包络（2^6 assignment）与「SoM 不包含 DOM∪Vision」

**当前值**:
- §500.1 / §510.2，各臂 unique-solve **下界**：cls（n=224）SoM **6** · Vision **6** · P-text 0 · P-SoM 0 · P-prompt **1** ⇒ 视觉侧最低 6 vs 文本侧最高 1，**分离 +5**；red SoM **4** · Vision **2** · P-text 0 · P-SoM 0 · P-prompt **2** ⇒ **接触 +0**。
- §510.2 分母 205 → 203 后：`Vision` 上界 **6 → 5**，`P-prompt` 上界 **5 → 4**；六臂下界一个没动 ⇒ red_b0 仍 **+0**。task 160 在全-B assignment 下**只有 P-prompt 解出** = 被协议排除的任务在充当 unique solve。
- §503 信息单调性：SoM 解集**不包含** D∪V。单次读数 6 cell 合计 **81** 个任务被 DOM 或 Vision 解出而 SoM 没有；2³ 包络下界 cls_B0 **18**（max 28）· red_B0 **10**（max 20）；SoM 独有 12-16 / 5-10；cls_B2：SoM 解 5 / D∪V 解 8 / 交集 **0**。

**caveats**:
- §500.1：「**只覆盖 B0** … 本表不授权任何关于它们 [B1/B2] 的陈述」；「**reddit 这格没有 drift-free 子集** … 该 envelope 同时含 run-to-run 噪声与代码漂移, **不可当纯噪声带引用**」；「DOM 不属于任何一侧」。
- §510.2：「只动上界不动下界 ⇒ **§500 的『reddit 上两 side 接触』结论不受影响**」；cls_b0 完全不受影响。
- §503：「这是**第一性原理预测失败**的直接证据 … **用于反驳 sVJH『结论可从第一性原理预料』**」；尚未算 WA 两格。

**证据**: §500.1 / §503 / §510.2；`docs/analysis/cross_sites/unique_solve_envelope_cross_cell.md`、`results/phantom_paper/per_task_sr.csv`

**原文片段**: 「包络下界 >0 ⇒ 重跑解释不掉」(§503)

---

# B. Oracle / 两臂上界（回溯性，不可部署）

## B1. full cost-aware oracle（oracle_sr_cost）与「the oracle」措辞

**当前值**:
- §484（8 cell，对 always-cheapest）：**7/8 格 Pareto 胜**；唯一失败格 red·B2：SR **+5.42pp** 但成本 **+1.8%**（0.06958 vs 0.06833）；ΔSR **+4.91 到 +25.00pp**，Δcost **−17.7% 到 +1.8%**。
- §517.2（对**单一最佳看法**，talk 用）：B0 classifieds / reddit / WA reddit：成功 **+16.07 / +11.33 / +16.35pp**；成本 **−20.2 / −13.7 / −26.9%**；用时 **+0.5 / −2.0 / −33.9%**；CO₂e 估算 **−14.4 / −7.3 / −28.6%**；全部 7 格用时 −33.9% 到 +9.2%（cls·B2 +9.2%，red·B1 −22.4%，red·B2 −15.8%，cls·B1 −2.9%）。WA·B1 未算。
- §486.2 稿件：全稿把 triage_only 结果说成「the oracle」**9 处**（ch6 ×6, Abstract ×2, ch7 ×1）+ ch5 图注 1 处，**全部已限定**。

**caveats**:
- 「oracle_sr_cost 是**回溯性**的, 不可部署 … 毕设 Chapter 6 报的『oracle 7/8 失败』说的是 **triage_only** oracle —— 两个数字都是 7/8 但含义相反」；GPT 2026-08-27 review 称 full oracle 8/8，「系从两个 B0 格外推, red·B2 判错」。
- §517.2：「完美事后选择的上界 … 用时不是一致收益 … CO₂e 是 token 估算, B0 无能耗实测, 只可报比例」。
- ⚠️ §484 与 §517.2 的对照基线不同（always-cheapest vs 单一最佳看法），**不可互比**。
- §517.3：`router_objective_ordering._wa_matrix` 在 WA·B1 上 DOM 匹配到 2 个目录 ⇒ 整个设置被**静默跳过**；现有 md 是 09-15 前生成仍含 WA·B1，重跑会少一格（见 G3）。
- §486.2：「改一处断言时必须扫它的复述」。

**证据**: §484 / §486.2 / §517.2 / §517.3；`docs/analysis/cross_sites/router_objective_ordering.md`、`deliverables/showcase/talk/hindsight_efficiency.{py,json}`

**原文片段**: 「7/8 格 Pareto 胜 (SR 更高且成本更低)。唯一失败格是 red·B2」(§484)

---

## B2. two-arm action oracle 与 learned two-arm 策略

**当前值（§491.6/§491.7，8 cell = 6 VWA + 2 WA，18-特征匹配集，B=10000）**:
- 天花板：ΔSR **+0.99pp (red·B1) ~ +6.70pp (cls·B0)**，**8/8 Pareto 压制 oracle_triage**。
- learned nested：Pareto-beat **cross-fitted** always-cheapest = **1/8**（whole-cell 基线下 **3/8**）。唯一那个 win 是 cls·B2：**SR 完全相同 + 成本低 0.12%（$0.000087/episode）**，base SR 2.23%（5/224）。
- 「6/8 超出 2.23pp 重跑噪声带」里 cls·B2 只超出 **0.0021pp** ⇒ 可辩护版本 **5-6/8**。
- §491.3 frontier 检查（30 个 ordered pair）：**2/8 cell** 存在另一对两轴严格支配选中对 —— cls·B0 (SoM,Vision) **33.93%/0.06257** vs (P-prompt,Vision) **36.16%/0.06221**；red·B0 (SoM,Vision) **18.23%/0.09719** vs (DOM,Vision) **18.72%/0.09650**；其余 6 cell 0/30 ⇒ oracle_two_arm **不能叫 ceiling**，只能叫 *conditional oracle over the triage-selected pair*。

**演变**:
- §490.1：two_arm_action_oracle 8/8 严格 Pareto 压制 oracle_triage（ΔSR +0.99 到 +6.70pp, Δcost 全部 ≤0）；对 always-cheapest 的 Pareto-win 计数 **1/8 (oracle_triage) → 8/8 (two_arm_action_oracle)**；3/8 cell（red·B1 +0.99 / cls·B2 +2.23 / red·B2 +1.97pp）落在「已知 rerun noise band 0.89-2.23pp」内或边缘，5/8 清楚超出。
- §490.4：AUROC(z) < AUROC(y) 在 6/8 cell，其中 3 个 <0.5（cls·B0 **0.405**, cls·B1 **0.454**, wa_red·B0 **0.406**）；learned two-arm 对 always-cheapest **0/8**，与原稿 y-target 'learned (nested)' 的 0/8 打平；成本溢价更低（cls·B1 z: **+0.00pp/+3.5%** vs y: **+1.79pp/+35.9%**）。
- §491.6/7 修复后：见当前值。

**已作废**:
- §490.4 内 20-特征冒烟测试阶段「cls·B0 学出来的策略 Pareto-beat always-cheapest」—— 全量匹配特征集下消失，**已在 §490.4 一并收回，不能引用**。
- §490.4 的 learned **0/8** 被 §491.6/7 的 **1/8（cross-fitted）/ 3/8（whole-cell）** 取代（基线口径不同，见矛盾清单 #4）。
- 「cost 上的 8/8」是实测非定理（**§491.2 RETRACTED**，原文指向）。

**caveats**:
- 两个 oracle 都是 perfect-knowledge，不是可部署策略；「SR 方向有构造保证」。
- 「**两处刀刃, 脚本自动标注勿静默计入分子**」（cls·B2 的 0.0021pp 与唯一 Pareto win）。
- red·B2 是两个 target 下都出现的例外，「疑似 … 结构性原因, 未深挖」。

**证据**: §490.1 / §490.4 / §491.3 / §491.6 / §491.7；`scripts/analysis/two_arm_action_learnability.py`、`docs/analysis/cross_sites/two_arm_action_learnability.md`

**原文片段**: 「唯一那个 Pareto win 也是 cls·B2, 内容是 **SR 完全相同 + 成本低 0.12%** … 满足 Pareto 定义但不是任何人会据此部署的结果」(§491.6/7)

⚠️ 2026-10-06 §529.4 已撤回 / 已变化，见实验笔记（本主题引用的重跑噪声带数值已变）。

---

## B3. 三臂收敛（READ=dom / LOOK=vision / SoM）相对六臂

> §474 以前见 measured_D5.md（§455.3 的 3×2 vs 6×1）

**当前值（§505.5，11 cell）**: orc3−best vs orc6−best：cls_B0 **+11.2/+16.1pp** · cls_B1 +7.1/+10.3 · cls_B5 +9.4/+14.3 · red_B0 +8.8/+12.2 · **red_B0_WA +2.9/+16.3** · red_B1_WA +5.8/+14.4 · shop_B0 +8.0/+8.0。contested 3 臂 vs 6 臂 cls_B0 **31.2% vs 39.3%**。3-class 最小类 ≥10 行只 **2/11**（cls_B0 13, shop_B0 13；其余 1–8）。3 臂 LinUCB **11/11 低于 fixed-best**。3 路 lookahead 每格在 random-to-best 对照 ±1pp 内，dom 已是 best 的 5 格每个 f 都掉 SR。

**caveats**: 「WA 两格天花板损失 60–80% 是因为赢家是文本侧 phantom 臂 (ptext/pprompt), 三臂里没有它们」；「orc3 未对着『best 臂跑 3 次』的重跑对照读 (只有 2 个 som 重跑, §455.3 已示 3×2 与 6×1 不可区分)」；3-class 供给数字与 pilot c3 日志的 READ/LOOK/BOTH 不是同一定义。

**证据**: §505.5；笔记 §505.5 + digest §6（`docs/analysis/cross_sites/one_step_lookahead_2026-09-09.md`）

**原文片段**: 「3-class 最小类 ≥10 行只 **2/11**」(§505.5)

---

# C. §505 router 全链重想（2026-09-09，一天内）

> §474 以前见 measured_D5.md（§457 abstention / §459 在线 abort / §409.3 sequential 未观测）。
> 本组数据源全部来自 `results/router_llm_pilot_20260909/`，其协议排除任务污染见 G1。

## C1. realization 级配对检验与 step-0 单信号

**当前值**:
- **配对 realization 检验**（§505.2，18 对 CLEAN_PAIRS，326 个翻转 task，只用 step_idx=0）：P(succ>fail) = mean_logprob **0.476** / min_logprob 0.476 / mean_margin 0.510 / min_margin 0.472 / verbalized 0.523（107 ties）/ thought 长度 0.507 / 输出 token 0.511 / 推理延迟 0.521 / mean_entropy(B1 only) **0.333**；sign-test 双侧 **p 全部 > 0.30**。同 mode 重跑 step-0 动作相同 **24–75%**（B0），url_after 相同 **50–90%**。
- **step-0 单信号对自身 episode 成功的 AUROC**（§505.3，11 格 × mode 中位，max(a,1−a)）：verbalized **0.562** / mean_logprob 0.547 / min_logprob 0.546 / mean_margin 0.574 / min_margin 0.552 / thought 长度 0.543 / 输出 token 0.548（E3 整段聚合同名信号 **0.53–0.85**）。
- **TF-IDF**（1–2gram, 5 折）：intent-only ≥ thought-only 在 **29/33** 行（cls_B0·dom·self **0.789 vs 0.577**；red_B0·vision·upg:dom **0.785 vs 0.752**）。

**caveats**: 检验的是 **realization 级**信号（task 难度配对抵消），「task 级 (§457 abstention) 仍有 0.6–0.86 AUROC」；只看 step 0；B5 无 logprob；TF-IDF 是 steady-state 口径（template sibling 跨 fold，user 2026-09-09 裁定非 leakage），GroupKFold 下同降；语义 embedding 未测。

**证据**: §505.2 / §505.3；`docs/analysis/cross_sites/one_step_lookahead_2026-09-09.md` §1–§2

**原文片段**: 「sign-test 双侧 p 全部 > 0.30」(§505.2)

---

## C2. one-step lookahead（cheap step-0 → 是否升级）

**当前值**:
- **增量 AUROC**（§505.3，中位 Δ(MODEL0−OBS0)）：5 折 stratified self **−0.007** / any **−0.021** / rich **+0.005** / upg **+0.021**；GroupKFold(template) −0.010 / −0.031 / −0.011 / −0.004。绝对值（strat 中位）：OBS0 self 0.720 · any 0.723 · rich 0.685 · upg 0.604；+MODEL0 0.723 · 0.693 · 0.672 · 0.617；+POST0 0.721 · 0.692 · 0.674 · 0.639。upg 在 **16/26** (cell,pair) 过 shuffle-null p95。
- **离线策略**（§505.4，27 对）：Pareto 打赢 always-rich **11/27**（MODEL0），其中 9 对是 rich≤cheap 退化 / SR 1–4% / n=104 无 band；**唯一非退化且有 band 的：cls_B0 vision→som f=0.15：27.7%/$0.0659 vs always-som 27.2%/$0.0724 = +0.5pp, band 4.5–7.6**。对同规模随机升级均值增益 f=0.1/0.2/0.3 = **+0.74/+0.90/+1.04pp**，63–74% 对为正。cheap-restart 每行 ≤ always-cheap；upgrade 比 restart 高约 2–3pp（band 内）。step-0 peek 成本 = 整段 episode 的 **3.0–12.0%**（中位）。
- **四轴前沿**（§505.7）：11 格里 **0 个** mode 同时赢 SR / cost / latency / token 四轴；lookahead 在 SR≥always-rich 点上四轴同时不劣的非退化对 **3/21**（red_B0_WA dom→ptext n=104 / red_B2 vision→dom SR 4% / shop_B1 vision→ptext n=171），有 band 的格 **0 个**；cls_B0 vision→som f=0.15：cost **−7.8%**，token **−10.0%**，**latency +17.6%**。

**caveats**: 「四个标签在 full-factorial 设计下全部稠密, 所以这里失败的是 signal 不是 supply」；POST0 对升级不 splice-legal，只作信息上界；「操作点是扫出来的, 未 out-of-fold 选阈值 ⇒ 每个数是可部署性能的上界」；「与 paper cascade (+1.1~+2.1pp over random) 同量级、同结局」；跨 mode step-0 url_after 不同 27–70% ⇒ step≥1 splice 不合法；latency 方向取决于 rich 臂快慢；token ≈ cost 的镜像。

**证据**: §505.3 / §505.4 / §505.7；digest §2 / §3 / §8

**原文片段**: 「唯一非退化且有 band 的: **cls_B0 vision→som f=0.15: 27.7%/$0.0659 vs always-som 27.2%/$0.0724 = +0.5pp, band 4.5–7.6**」(§505.4)

---

## C3. contextual bandit（full-information 日志精确回放，sVJH 点名项）

**当前值（§505.5，11 cell，200 个随机 task 顺序）**:
- 6 臂 LinUCB/Thompson（λ=0 与 0.5）：**11/11 格低于 fixed-best 且 11/11 低于 fixed-cheapest**（cls_B0 22.4% vs som 27.2% / vision 25.0%；cls_B5 28.1% vs 37.1%），末段 1/4 各臂占比 10–38% 近均匀。
- 2 臂（cheapest/best）：11/11 低于 fixed-best（cls_B0 26.2 vs 27.2），best 臂末段占比 47–69%。
- 3 臂（dom/vision/som）：11/11 低于 fixed-best（cls_B0 24.1；cls_B5 31.1 vs 37.1），末段 37–56%。

**caveats**: 回放合法性依赖 full-information 日志，成立；单次 draw 的 reward 含 10–14% 重跑翻转；「未调 α / 先验; n=104–435 下任何合理超参都在探索期, 结论对超参不敏感 (2 臂已是最有利情形)」。context = OBS0 pre-flight 15 维，reward = success − λ·cost/median_cost。

**证据**: §505.5；digest §4

**原文片段**: 「**11/11 格低于 fixed-best 且 11/11 低于 fixed-cheapest**」(§505.5)

---

## C4. corr(best−2nd gap, Δ_router) —— 明确「不是 finding」

**当前值（§505.5，n=11）**: v3 all+costaware **−0.708**，bootstrap 95% **[−0.94, +0.63]**，LOO [−0.79, −0.62]；v1 −0.556 [−0.98, +0.16]；3-class −0.038 [−0.56, +0.81]。对照：corr(gap, oracle 加一臂增益) **+0.371 [−0.10, +0.69]**；corr(labelled%, oracle 加一臂增益) **+0.844 [+0.71, +0.96]**。

**已作废**: README 缺陷 #5 的 **−0.696**（user 原记 n=10）—— 与 −0.708「是同一现象不同 cell 子集」，两者都**不可引**。

**caveats**（一字不改）: 「**此条不是 finding**: Δ_router = SR_router − SR_best ≈ −(离开 best 比例)×gap + 有信息增益, router 一偏离就结构性负相关, 说的是 router 无知不是可路由性; 支持『两个相近臂 ⇒ 有活路』的量是 oracle 增益 vs gap, 而它 CI 过零」。

**证据**: §505.5；digest §5

**原文片段**: 「n=11 的相关系数本身不可引」(§505.5)

---

## C5. routing 目标的可学上界：方差分解 / 分级标签 / 跨 backbone 共享

**当前值**:
- **方差分解**（§505.7，仅两个六臂全 replicate 的格）：cls_B0（n=224）task 主效应 0.0773 / mode 主效应 0.0026 / task×mode 交互（扣噪声后）**0.0209** / replicate 噪声 0.0618 ⇒ **交互/噪声 = 0.34**；red_B0（n=203）0.0557 / 0.0004 / **0.0056** / 0.0459 ⇒ **0.12**。单次 run「离开 best 去 mode X」标签在第二次 run 复现：cls_B0 **20/66**，red_B0 **9/43**；稳定严格偏好每 mode 1–5 个 task。
- **分级 progress 标签**（§505.8，URL 覆盖率 leave-self-out）：18 对 replicate 分级 r(A,B) 0.50–0.78（pooled **0.74**）vs 二元 kappa 0.22–0.58；交互/噪声 cls_B0 二元 0.32 → 分级 **0.49**；red_B0 0.12 → **0.12**。标签只在 **9,841/21,283** episode 上有定义。（cls_B0 135 task / red_B0 62 task 子集）
- **near-miss enrich**（§505.12，red 为五臂口径）：交互/噪声 cls **0.34→0.36**，red **0.16→0.20**；route-away 复现 cls 20/66→26/82，red 7/36→9/41。
- **跨 backbone 共享**（§505.19）：r(难度) cls B0–B1 0.62 / B0–B5 0.68 / B1–B5 0.46，red B0–B1 0.66，WA 0.72，shop 0.54；×B2 0.13–0.46。r(交互) cls **0.18–0.22**，red B0–B1 0.13，WA 0.08，shop 0.24，×B2 −0.00–0.09。同 backbone 两次 run：难度 **0.88/0.93**，**交互 0.28 (cls_B0) / 0.09 (red_B0)**。best-arm 一致 44–59%（同族）。

**caveats**:
- §505.7：去噪是 noise/2 线性扣除的近似，「但 0.34/0.12 的量级不依赖该修正 (未扣时 0.75/0.54 仍 <1)」；两次 replicate 相隔 2–69 天含漂移；只两格且都是 B0；「这是『能不能学』的上界性诊断, 不是某个 router 的评估」。
- §505.8：是 LLM grader 的**上界代理**；对 universal-fail task 无定义；`reward` 字段不是逐步评分，勿当 graded signal。
- §505.12：「red 的 5 臂分解与 §505.7 的 6 臂 (0.12) 不同臂集, 二元值 0.16 是本口径的对照, **不可与 0.12 混引**」。
- §505.19：单 draw 跨 backbone 相关是共享结构的下界；B2 近地板，其相关不可读为「跨族不共享」的强证据。

**证据**: §505.7 / §505.8 / §505.12 / §505.19；digest §8 / §9 / §13 / §19

**原文片段**: 「交互/噪声 = 0.34 … ⇒ 0.12」(§505.7)

---

## C6. 失败原因桶与 near-miss —— 稳定性高、预测力无

**当前值**:
- **同 condition 重跑一致性**（§505.9，cls_B0 两次都失败的 task）：细桶一致 dom **74.1%**（n=174, kappa 0.66）/ vision **71.9%**（153, 0.64）/ som **76.0%**（146, 0.70）；粗桶 78.8–81.6%（kappa 0.59–0.62）；final_error_category 88.9–91.1%；loop_pattern 76.4–85.0%。§505.12 扩到六臂：cls_B0 细桶 kappa **0.55–0.70**（dom 0.66 / vision 0.64 / som 0.70 / ptext 0.55 / pprompt 0.55 / psom 0.62），粗桶 73–82%；red_B0 五臂 kappa 0.41–0.51。
- **预测力**（§505.9，cls_B0 cheap run A 失败的 task，oof-AUROC）：对 rich 成功 dom→som **0.567** / vision→som **0.341** / dom→vision **0.501** / som→dom **0.459**；对 cheap 自身重跑 0.511–0.627；对 switch-only 0.345–0.542（n_switch 10–31）。`fail_max_steps` 桶在 dom 下 rich 成功 **0/14**。
- **near-miss 比例**（§505.12）：cls_B0 六臂页面侧 near-miss 2–11 条/臂 = **1–6% 的失败**（url_match 失败里 2–9%）；答案侧给出答案 7–10 条，接近度 ≥0.8 仅 4–5 条，全 token 命中 1–4 条；red_B0 五臂页面侧 1–3%。`target_item_ever_visible` 全为 False（字段可能未实装）。

**caveats**: 桶由同一 ruleset 生成，kappa「含规则稳定性成分, 对 LLM 归因是上界」；「这是事后 (整段 cheap episode) 信号, 对应 cascade 形态不是 lookahead; 即便如此也无信号」；near-miss 预测 rich 成功与预测自身重跑同向 ⇒ task 属性；所有 n 都在个位到十位数；只 cls_B0（red 部分五臂，psom canonical 因 strict-identity 拒读缺）。

**证据**: §505.9 / §505.12；digest §10 / §13；各 run `analysis/` 下补跑的 reason_diagnostics

**原文片段**: 「`fail_max_steps` 桶在 dom 下 rich 成功 **0/14**」(§505.9)

---

## C7. cls_B5（最强格）相对 cls_B0 的成功矩阵结构

**当前值（§505.10）**: best som **37.1 vs 27.2%**；union **51.3 vs 43.3%**；**route-away task 32 (14.3pp) vs 36 (16.1pp)**，占 union 28% vs 37%；非 best 臂嵌套在 som 内 0.73 vs 0.68；mode 主效应方差 **0.0053 vs 0.0020**（11 格最高）；每臂对 som 独有解 11–13 vs 10–16；dom 重跑 discordance **12.9% vs 12.1%**；稳定 som→dom route-away 7/13（3.1pp, band 5.8–7.1）。11 格 route-away pp 随 best SR 涨到 14–16 即平（cls_B0 16.1 / red_B0_WA 16.3 / cls_B5 14.3 / red_B1_WA 14.4）。

**caveats**: B5 只有 dom 一条 replicate，7/13 对 som 侧翻转是盲的、是上界；「B5 无 logprob, 但 B0 有 logprob 也无信号, 不是主因」；「饱和曲线」只 11 点且 cell 间不独立，是描述不是拟合；「§6 的 ρ=0.952 用的 contested 定义含『仅 best 解出』行, 与此处 route-away 不同口径, 两者都对, 但只有 route-away 是路由价值」。

**证据**: §505.10；digest §11

**原文片段**: 「11 格 route-away pp 随 best SR 涨到 14–16 即平」(§505.10)

⚠️ 2026-10-06 §529.4 已撤回 / 已变化，见实验笔记（本条引用的 band 5.8–7.1 属噪声带族）。

---

## C8. 「便宜够不够」条件标签、两段式省钱策略、跨 backbone 迁移

**当前值**:
- **条件标签可学性**（§505.11，只在 best 臂解出的行上定义）：cls_B5 dom→som n=83（dom✓ 40）OBS0 AUROC **0.671** / +MODEL0 0.664 / shuffle-null p95 0.559 / GroupKFold 0.608，标签在 dom 双 replicate 间 agree 81% kappa **0.61**；shop_B0 vision→som n=66：0.648 / 0.674 / null p95 0.622 / Group 0.619；cls_B0 vision→som n=61：**0.421 / 0.328**（低于随机），kappa 0.45；red_B0 vision→dom n=30（6 正例）0.632/0.847，null p95 0.716。
- **两段式 pre-flight 省钱**（§505.11）：cls_B5 dom→som 送 dom 份额 0.3/0.6/0.8 ⇒ ΔSR vs always-som **−2.7/−5.8/−8.5pp**，省 4%/11%/17%，learned−random **+1.3/+2.2/+2.2pp**；oracle（181 题）+5.8pp 且省 **22%**；learned 收回 oracle-over-random 余量约 **15%**；省到 10% 需付约 5.5pp SR。shop_B0 vision→som：份额 0.6–0.8 时 SR +0.9~+1.6pp 且省 15–20%，但 always-vision 已省 23% 只丢 0.5pp。cls_B0 vision→som：0.2–0.5 份额输 random 0.9–1.6pp。
- **leave-one-backbone-out 迁移**（§505.19，特征 = OBS0）：any_solves（弃权）迁移 AUROC **0.691–0.868**，**11/11 格 ≥ 本格 within-cell CV**（0.645–0.828）；pooled 0.694–0.865。cheap_suffices|best：within 0.576–0.668（4 格）→ 迁移 **0.404–0.466**（cls_B5 0.428 / shop_B0 0.423 / cls_B1 0.404），例外 red_B0_WA 0.655（n=37）；pooled 0.408–0.548。

**caveats**: 「部署时不可观测哪些行属于它 ⇒ AUROC 是诊断不是策略」；「四个能算的格里两格有信号、一格低于随机、一格 n=30; 不是跨格一致的结果」；「red_B0 的 0.847 是 6 个正例上的数, **不可引**」；份额 in-sample 扫 ⇒ learned−random 是上界；「与 §387.16.4 同形: 有标签有 AUROC, 对 random 有 1–2pp, 对固定策略无 Pareto 胜出」；弃权迁移 ≥ within 部分因训练行数变多；「mode 契合度是 backbone 属性 (B5 vision 12.1% vs B0 25.0%)」。

**证据**: §505.11 / §505.19；digest §12 / §19

**原文片段**: 「learned 收回 oracle-over-random 余量约 15%」(§505.11)

---

## C9. 规则路由：reactive 触发器 / 手写工业规则 / 反向挖掘规则

**当前值**:
- **reactive rule router 离线投影**（§505.13，cls_B0 dom→som）：grounding/stuck 信号对 som 成功 AUROC **0.41–0.56**（ax_click_fail_rate 0.451 / page_unchanged_rate 0.529 / max_repeat_streak 0.538 / url_revisit 0.413）；`page_unchanged_streak≥2 ∨ repeat_streak≥2` 触发于 **57% 失败 + 33% 成功** episode，触发点中位第 7/19 步（45% 成本已花）；拼接投影 variant D ≈ **26.3% / $0.0869** vs always-dom 17.4%/$0.0696 vs always-som 27.2%/$0.0724；救回 24 / 白切 81 / 打断成功 13（som 解其中 9）。
- **手写工业规则**（§505.18，11 格，默认臂 = cell best）：均值 R1 视觉谓词→vision **−2.5pp**（cls_B5 −17.4）· R1'→som 0.0 · R2 −0.2 · R3 −0.3 · R4 AX大→vision −1.4（red_B0_WA −7.7）· R5 −2.1 · R5' −0.2 · R6 −1.1；正的格数 0–3/11；有 band 的格**无一超过 band**，最大正值 +1.8（cls_B0 R4）。
- **反向挖掘规则的重跑存活**（§505.18，1,235–1,265 候选中挑 top-20）：cls_B0 A **+2.4pp** → B **−0.6pp**，B>0 5/20，B>band **0/20**；red_B0 A +1.5 → B **+0.1**，8/20，0/20；反向（B 挖 A 验）+1.0 → +0.5（cls）/ +0.9（red），A>0 10/20 · 14/20。

**caveats**: 「**投影不是测量**: 切换后 som 起于 dom 已动过的站点, 该 sequential outcome 本项目从未观测 (§409.3)」；variant D 从未点火；手写规则默认臂 in-sample 选（对规则偏有利）；「规则的收益取决于 backbone 的臂形状, 不可跨 backbone 搬」；top-20 是 winner's curse 的极端形式，「但这正是『手写/挖规则然后上线』会遇到的情形」。

**证据**: §505.13 / §505.18；digest §14 / §18

**原文片段**: 「B>band 0/20」(§505.18)

---

## C10. cascade：事后重跑 cascade 与模型档位级联

**当前值**:
- **事后重跑 cascade**（§505.15，15 对可算）：learned − random f=25% **+0.79pp**（中位 +0.67, 87% 对为正），f=50% **+0.99pp**（中位 +0.89, 93% 为正）；全部重跑天花板 A∪B−A = **+1.8~+7.6pp @ 2× 成本**；最佳信号（6 选 1, in-sample）对「A 失败」AUROC 0.39–0.74，对「重跑能修」0.46–0.88。
- **模型档位级联**（§505.20，cls, mode=som）：B0→B5：both 41 / B0-only 20 / B5-only 42 / neither 121，union **46.0%** vs B5 37.1%；pre-flight「B0 成功」AUROC **0.710**；分流 learned−random −0.4/+2.0pp（share 0.5/0.7）；事后级联 f=0.5/0.7/0.9 SR 37.5/41.5/44.2% vs random 36.6/40.4/44.1%，成本 −3%/+22%/+37% vs always-B5。B1→B5：AUROC 0.587，分流 −1.7~−0.2pp，级联 −0.6~−1.0pp。成本 B0 $0.0724 / B1 $0.0603（电费折算）/ B5 $0.1671。

**caveats**: 信号逐对 in-sample 选 ⇒ 上界；「『重跑能修』的后半 (失败题重跑成功率 4–10%) 是硬币, 检测器只能找前半, 所以增益量级被结构性封在 ~1pp」；cls_B1 som/vision 与 red_B1 som 零/一个翻转不可算；「union 46% 的收益来自并行两家不是检测器; 检测器 (episode mean logprob) 对 B0 失败 AUROC 0.554」；B1 成本是电费折算，只作方向；B5 记账价偏低 25%（§478.3）。

**证据**: §505.15 / §505.20；digest §16 / §20

**原文片段**: 「增益量级被结构性封在 ~1pp」(§505.15)

---

## C11. 跨臂最终答案一致性作 verifier（非路由）

**当前值（§505.25，8 格 79 个臂对，非 url_match 任务）**: pooled P(正确|一致) **294/834 = 35%** vs P(正确|不一致) **434/2333 = 19%**；格均值（一致 − base）cls_B0 **+25pp** · shop_B0 +15 · cls_B5 +13 · red_B0 +10 · red_B1 +9 · red_B0_WA +6 · cls_B1 +5 · shop_B1 +4（**8/8 为正**）；逐对反向：red_B0_WA 4 对 <1×，red_B0 som+vision 0.5×，cls_B1 三对一致时 0%。

**演变**: §505.20 pilot（cls_B0 单格）：dom+ptext 一致 n=13 → dom 正确 **31%**，不一致 n=50 → 10%（base 14%）；dom+pprompt 12 → 25% vs 13%；**dom+som 7 → 43% vs 11%**；ptext+pprompt 12 → 17% vs 17% → §505.25 扩到 8 格。

**caveats**: pooled 计数不独立（同 task 进多对），只作方向；一致 n 多为个位到十几，单对不可引；base 高的格（WA）无增益；「是 verifier 不是 router: 成本 2× 换『答案交不交』的精度, 覆盖率 10–70%」；3 条 canonical 臂未能算（strict-identity 拒读 / 部分副本）。

**证据**: §505.20 / §505.25；digest §20 / §25

**原文片段**: 「pooled P(正确|一致) 294/834 = 35% vs P(正确|不一致) 434/2333 = 19%」(§505.25)

---

## C12. 选臂 pilot 的样本量

**当前值**:
- §505.23 全集裁判 n=20/50/100：cls_B0（差 2.2pp）**49/63/80%**；cls_B5（12.9pp）82/96/100%；red_B0_WA（8.7pp）66/96/100%；臂差 ≤0.5pp 的格 79–85 / 88–95 / 95–100%。replicate 裁判：cls_B0 n=100 → **79%**，n=224 → 100%；**red_B0 n=205 全集 → 0%**（A 的 best dom 14.6% 在 B 掉到 11.7%，B 的 best 是 psom；六臂 A 里 7.8–14.6）。
- §505.24 扩到 B1：cls_B1 三臂 n=10/20/30/50 → 88/91/96/98%，n≥100 → 100%；red_B1 两臂 n=10 起 100%。

**caveats**: 全集裁判 in-sample 偏乐观，replicate 裁判才是部署语义；「『2pp 内』是任意阈值, 对着 band 应更宽 (4–7pp) … 结论方向不变: 臂差在 band 内时选臂无意义」；cls_B1 replicate 0 翻转、red_B1 两臂差 1.0–1.5pp。成本估计 $45 按 B0 cls ≈ $0.07 × 600。

**证据**: §505.23 / §505.24；digest §23

**原文片段**: 「**red_B0 n=205 全集 → 0%**」(§505.23)

---

## C13. 延迟 / token / 成本结构（harness 层）

**当前值**:
- **推理占单步 wall-clock**（§505.13，step-0 中位）：API 臂 cls_B0 **22–24%**（推理 1.8–2.1 s / 单步 8.2–8.6 s）· cls_B5 28–44% · red_B0 16–27% · red_B0_WA 15–21% · shop_B0 15–16%；本地 4B 臂 cls_B1 55–59% · cls_B2 62–65% · red_B1 36–47% · red_B2 42–64%。cls_B0 六 mode 每步 7.0–7.4 s，som 最快（62 s）因步数最少（8 vs 10–14）；dom 每步 token 3857 vs vision 3504。
- **每步 token**（§505.14，7,230 episode 全步）：cls_B0 dom/vision/som input 3741/3402/4646，output **103/102/99**，thinking 0，output 占成本 9.7–13.1%；cls_B5 3016/2922/4206，163/151/147，thinking **0**，17.3–24.7%；red_B0 4438/3457/4970，100/98/92，0，8.5–12.4%；B1/B2 output 97–107。成功 vs 失败每步 output：B0 相同（103 vs 102），**B5 成功更少**（124–137 vs 154–170）。
- **input 随步数增长**（§505.20，cls_B0）：dom 3370→3857（×1.14），som 4367→4745（×1.09），vision 3139→3504（×1.12）；≥20 步 episode 每步均值 3979/4813/3564。
- **harness 历史窗口**（§508.5，代码事实）：`cfg.agent.history_window` 默认 **8 步**，每条 = 动作类型 + 目标 + 结果 + URL，**不含 thought**。

**caveats**: 综述引的 75–94% 来自带 reasoning 的 frontier 模型，「两者不矛盾只是 regime 不同」；「**本项目没有 thinking 区间的 run**: B5 经 proxy tokens.thinking 全 0 (§471.5 reasoning_effort 不透传)」；「harness 不累积完整历史 (短窗口) … 结论限于本 harness」；是否保留 thought 会改结果未测（候选消融）。

**证据**: §505.13 / §505.14 / §505.20 / §508.5；digest §14 / §15 / §20；`p79/experiment/runner/main.py:3052-3073`

**原文片段**: 「本项目没有 thinking 区间的 run」(§505.14)

---

# D. 预算路由（step cap）—— 本批唯一过重跑存活的路由形态

## D1. 失败 episode 的花费份额与固定 cap 前沿

**当前值（§505.21，11 cell × 6 mode 中 60 condition 中位）**: 失败 episode 占花费 **92%**（范围 76–99%）；成功 p50 步数 4–10（p90 14–30），失败中位 30（30–91% 撞满 cap 30）。固定 cap 前沿：cap 25 **−1.79pp/省 14%** · 20 −2.44/28% · **15 −3.12/43%** · 12 −4.39/53% · 10 −4.91/60% · 8 −5.85/67% · 5 −8.48/78%；步数节省 = 成本节省。

**caveats**: SR 损失是 pp 绝对值，占 SR 比例 cap 15 中位 27%；「对『成功慢』的臂 (B1 som p90 30 步) 固定 cap 伤害远大于中位 (cls_B1 som cap 12/15 −6.7pp)」；成本 ∝ 步是近似；latency 节省按每步约 7 s 推。

**证据**: §505.21；digest §21

**原文片段**: 「失败 episode 占花费 **92%**」(§505.21)

---

## D2. learned 预算路由（OBS0 → P(本 mode 成功)）

**当前值**:
- **过重跑**（§505.21，18 CLEAN_PAIRS，A 上训 B 上评）：learned 损失 **+2.15pp** vs random +4.06 vs 同成本 fixed +3.58，省 **41%**；learned < fixed 在 **14/18 对 (78%)**，均值优势 **+1.43pp**。
- **in-sample**（§505.21，f=0.5, cap=5, strat）：dom learned **+2.52pp/省 41%** vs random +4.54 vs fixed +3.65；vision +1.60/42% vs +3.11 vs +2.39；som +3.03/41% vs +4.71 vs +4.47；ptext +3.40 vs +5.41 vs +4.25；pprompt +3.28 vs +5.71 vs +4.49；psom +2.28 vs +4.63 vs +3.75 —— **6/6 mode learned < random 且 ≤ fixed**。GroupKFold：learned ≈ fixed（dom +3.59 vs +3.65；som +4.12 vs +4.19；vision +2.03 vs +2.31）。
- **五族前沿**（§505.22，55 condition 中位，cost 轴）：20% → fixed +2.23 / random2 +1.84 / **learned2 +0.89** / tier3 +1.34 / planner +2.68；40% → +3.12 / +3.30 / **+1.95** / +1.95 / +3.69；60% → +4.91 / +5.48 / +3.90 / **+3.12** / +4.02；70% → +6.73 / +7.68 / **+4.61** / +5.36 / +5.07。latency 轴同表 ±0.1。per mode 40% 处 learned2 < fixed **6/6**。
- **逐 cell**（§505.24，40% 处 fixed/learned/tier3）：learned < fixed **11/11 格 × 3 水平**；cls_B0 4.2/1.8/1.8 · cls_B1 2.2/0.4/0.7 · cls_B5 4.7/3.6/4.2 · red_B0 5.1/3.7/2.9 · red_B0_WA 6.2/4.3/5.8 · red_B1 3.7/1.5/2.0 · red_B1_WA 3.8/1.0/1.0 · shop_B0 3.0/2.1/2.3 · shop_B1 1.6/0.7/0.9 · red_B2 2.4/0.5/0.5 · cls_B2 1.1/0.0/0.0。固定 cap 最伤的臂：cls_B1 som **+6.7**（band 0–1.8）→ learned +2.7；red_B1 som +5.9 → +2.4；red_B0_WA ptext **+11.5** → +8.7。
- demo 三题示例（§506.1，scratchpad 重算，非冻结版）：130 满预算；76 two-tier cap 5 / three-tier cap 8；17 cap 5 / 弃权 ⇒ LOOK-76 $0.103→$0.019，但 4 个成功里截 2 个（READ-76 / BOTH-17）。

**caveats**:
- 「learned 对 fixed 的优势 (~1pp) 是 steady-state (template memory) 效应, 冷启动退化为固定 cap」。
- 「与 §505.18 mode 路由规则 0/20 过重跑对照: 预算路由读 task 难度 (稳定) 且动作施加于同一臂, 不需交互项」；「14/18 sign test p≈0.03, pooled 而非 per-cell 结论; per-cell 差在 band 内」。
- 前沿点位 in-sample 选（下包络），上界；「planner 族 … 在所有节省水平 ≥ fixed 的损失 ⇒ 『要多久』事前不可预测」；「部署时 cap 必须按臂定」。
- §506.1 原文写「§505.21–24 的整体前沿 (省 41%, 损失 1.2–1.8pp 优于 fixed cap)」—— 「1.2–1.8pp」在本批台账其他条目中**无对应出处**（矛盾清单 #6）。

**证据**: §505.21 / §505.22 / §505.24 / §506.1；digest §21 / §22 / §24

**原文片段**: 「learned < fixed 在 **14/18 对 (78%)**, 均值优势 **+1.43pp**」(§505.21)

---

## D3. prospective 检验（2026-09-09 冻结，shop_B1 两个 held-out 臂）

**当前值**: 预注册 PRIMARY policy（two_tier）方向判据 **2/2 PASS**。
- P-text（§510.3 / §515.3 重跑一致）：PRIMARY（prospective, n=216）full SR 4.2%：two_tier learned SR loss **+1.39pp** vs 同成本固定 cap 13 **+1.85pp** vs 随机 +2.11pp ⇒ **PASS**；three_tier +1.85pp 与固定 cap 12 打平 ⇒ **FAIL**；non-prospective 两个 policy 都 PASS。
- P-prompt（§515.3）：PRIMARY n=432 full SR 5.3%：two_tier **+0.69pp**（cost −44%）vs cap 15 +2.55pp vs 随机 +2.06pp ⇒ **PASS**；three_tier +0.69pp（cost −45%）vs cap 15 +2.55pp vs 随机 +1.99pp ⇒ **PASS**。

**演变**: §510.3 时 P-prompt 臂仍在跑（405/437）；PRIMARY set 因 universe 修正从 218 降到 216 → §515.3 两臂评完。

**caveats**: 「预注册的 PRIMARY policy 是 **two_tier**, three_tier 的 FAIL 不是事后挑的。判据是**方向**而非幅度 (shop_B1 没有 rerun band)」；「单 run、只有 B1、SR ≈5% ⇒ learned 与 fixed 的差距是 1-2 个任务量级 … 这是方向判据不是显著性」。

**证据**: §510.3 / §515.3；`results/visualwebarena/phase1/B1_phantom_text_shopping_20260908`、`pre_run/budget_router_prospective_shop_B1_20260909.json`

**原文片段**: 「两个 held-out 臂都评完 ⇒ 预注册 PRIMARY policy (two_tier) 方向判据 2/2 通过, 但这是方向判据不是显著性」(§515.3)

---

# E. B5 = GPT-5.6 terra（首个非-B0 API backbone）

> §474 以前见 measured_D5.md（registry 08-19 / §469.5 / §471.5）

## E1. B5 的 SR 与相对 B0

**当前值**:
- §478.3 classifieds dom n=224 ×2：A1 **23.66% (53/224)** / A2 **25.00% (56/224)**；对照 B0 17.41% · B1 6.25% · B2 1.34%。
- §492.2 逐 mode SR 差（B5 − B0，cls n=224，B0 侧取 appendix 表 A.1 canonical）：SoM **+9.82** / P-text **+8.49** / DOM **+7.59** / P-SoM **+7.15** / P-prompt **+2.24** / Vision **−12.95**（pp）。
- §508.2 cls 文本侧五臂 SR 范围：B5 **21.9-37.1%**（B0 14.3-29.5，B1 6.2-14.3，B2 0.4-2.2）。

**caveats**: 「P-prompt 的 +2.24pp 仅略高于 B5 自己那对 replicate 的 1.34pp, **不可当作可分辨效应**」；「B5 的 $/ep 是 B0 的 2-3 倍 ($0.137-0.227 vs $0.058-0.072)」；每格只一次运行（DOM 除外）；B5 无 logprobs（confidence 只有 verbalized）；单站点。Vision −12.95 受坐标制错配污染（见 E3）。⚠️ §492.2 的 DOM 差用 B5 哪个 run，台账未写。

**证据**: §478.3 / §492.2 / §508.2；`results/visualwebarena/phase1/B5_dom_classifieds_20260820_…_R29736` + `…_R15476`

**原文片段**: 「SoM +9.82 / P-text +8.49 / DOM +7.59 / P-SoM +7.15 / P-prompt +2.24 / Vision −12.95 (pp)」(§492.2)

---

## E2. 价格、成本与支出（registry 漂移 / 每格成本 / chain 支出 / 吞吐）

**当前值**:
- **proxy 价格快照**（§478.5, registry `proxy_model_registry_20260826_003006.json`, 59 models）：sol 0.00625/0.0375 · terra 0.0025/0.015 · luna 0.00025/0.0015（input/output per 1k）；terra 08-16 0.001/0.005 → 08-19 0.002/0.012 → 08-26 0.0025/0.015 = 十天 2.5x/3x。
- **config 记账价偏离**（§478.5）：60 个 config 全部偏离且方向不一：B0 **+2.5%**；B4 **−8.3%**；B5 **+25%**（⇒ B5 成本列低估 25%，§478.3）。
- **每 condition 实测成本**（§496.5，avg_total_billed_cost_usd × episode）：B0 cls $0.06–0.08/ep ⇒ $14–18/格 · **reddit $0.10–0.11/ep ⇒ $21–23/格** · shop $0.07–0.12/ep ⇒ $30–52/格；B5 cls $0.14–0.22/ep ⇒ $31–49/格，reddit 外推 ~$47（chain 报价 $52）⇒ **B0 比 B5 便宜一半以上**。
- **站点倍率**（§489.5，B0 同模型）：墙钟 red·dom 10.10 vs cls·dom 2.26 min/ep（**4.48x**）；red·ptext 10.94 vs 2.49（4.40x）；成本 red·dom $0.1013 vs cls·dom $0.0696/ep（**1.45x**）；red·ptext $0.0991 vs $0.0710（1.40x）。外推 B5 red ≈ $0.318/ep、19.5 min/ep（B5 cls 实测 $0.224/ep，4.89 min/ep vision）。
- **reframe chain 真实支出**（§492.5，08-20→08-30）：实花 **$342.50 = $400 天花板的 86%**；chain 自报 **$0.00**。
- **吞吐时间线**（§480.5）：Phase C 实测 **15.9 ep/h** ⇒ 5 cells ≈ 3.0 天；B5×reddit 按 5.7 ep/h ⇒ 3 cells ≈ 4.5 天 ⇒ GPU 09-02 晚空闲。

**caveats**:
- §478.5：「⚠️ **这条也会过期** — 它是快照不是常量, 引用前必须重跑 probe_proxy_model_registry.py。台账里该存的是『去哪里读』不是『读到了什么』」；不能用单一倍数校正；跨 mode 成本**比较**不受影响。
- §496.5：`cost_total_mixed_unit_warn_rate=1.00` 已知且已系统处理（B-565 / §181 / §260E），不影响 B0 自身绝对 API USD；是 avg×n 重建值；交叉验证 $342.50 ≈ B5·cls 6 格 + B0·red 3 格数量级自洽。
- §492.5：「缺陷本身**不是新的** —— §478.5 已裁定 _chain_cost 找的四个 key … 根本不存在 … 新的是**暴露量**」。
- §489.5 外推是跨模型缩放，「第一格落地后必须用实测重算」；§480.5 B5 reddit 吞吐未实测。

**证据**: §478.5 / §480.5 / §489.5 / §492.5 / §496.5；`docs/checkpoints/probes/proxy_model_registry_20260826_003006.json`、`scripts/queues/_reframe_chain.sh`、`logs/.reframe_chain_*.state`

**原文片段**: 「实花 $342.50 = 天花板的 86% … chain 自报 $0.00」(§492.5)

---

## E3. B5 vision 的坐标制错配（B-1997）与 agent 自登出

**当前值**:
- **click 后页面真的变了的比例**（§508.1，cls vision 全 click）：B2 **15.2%**（n=2123）· B1 24.7%（1686）· B0 **71.6%**（1256）· B5 **13.2%**（1380, 0826 run）/ **15.2%**（2077, 0827 run）。
- **坐标制式**（§508.1，±4 命中分页按钮）：B0 像素位 1 · 千分位 88；B5 0827 像素位 **174** · 千分位 30；B5 0826 像素位 70 · 千分位 16。中位 x B5 720/674 vs B0 500；B5 x>1000 占 1.5-1.7%，B0 0%。
- **agent-induced logout**（§488.3，watchdog 'PRESERVED 3 NOT-LOGGED-IN' 计数）：vision R24364 2 波/33 ep + R16160 1 波/136 ep = **3 波/169 ep**；som R31483 0/224；dom R15476 0/224 ⇒ dom+som **0 波/448 ep**；每波固定损失 3 个 episode。
- §488.2 R24364 第一波 perpetrator：task 4 step 27/28 连点 [895,44] / [892,44]（右上用户菜单区），step 29 后 URL 变为 `page=register` ⇒ 自点登出；task 5/6/7 为受害者。

**caveats**: 「B5 的数字是坐标制错配 (B-1997) 的产物, 不代表 GPT-5.6 的 grounding」；「模型步间换制 … 不是稳定的单一制」；「**不能读成『只有 vision 会 logout』** —— §329 实测 B2·dom 就有 (约 1 波/100 task)」，vision 侧只 3 个事件，「**该格跑满 224 后必须回看波数**」；「**vision 下 §329 的 forensic 方法失效** … 坐标通道让事后归因近乎失效」。

**证据**: §488.2 / §488.3 / §508.1；`master_bug_catalog` B-1997

**原文片段**: 「B5 的数字是坐标制错配 (B-1997) 的产物, 不代表 GPT-5.6 的 grounding」(§508.1)

---

## E4. B5 cls 的 diag 归因与失败构成

**当前值**:
- **diag 三分类**（§509.2，Tier-1 v11 + Tier-2 no-hit 全覆盖）：no-hit dom 17 · som 54 · P-text 34 · P-prompt 37 · P-SoM 37 = **179**；agent-limit **174** · scaffold **1**（t210 → B-1998）· benchmark-FP 嫌疑 **4**（全是 task 41）；子类 字面关键词代理 55 · 视觉选错 33 · 循环 26 · 丢约束 13 · 说了要核实却直接交 9 · 无图 mode 做看图题 7。
- **multiple_actions 判无效却已执行**（§509.3，cls 23 canonical condition）：B0 0 · B1 0 · B2 2–24 · B5 **49–89**（页面变了 80–95%）；B5 dom task 210 连续 3 次 abort（B-1998）。
- **字面关键词代理**（§509.5）：参考 item 标题不含 agent 搜过的任何词：失败 **66% (233/353)** vs 成功 **32% (53/164)**。
- **「这一页」类 intent 首动作即搜索**（§509.5）：B0 14% · B1 32% · B2 10% · B5 **47%**；SR 先搜索 vs 留在页上：B0 12.1 vs 23.4 · B1 17.8 vs 16.6 · B5 18.5 vs 27.9。
- **task 41**（§509.5）：5 月 25 日以来 32 个 run 全部失败；B5 八个 run 答案完全相同 $1,900–$27,995；参考 {1200,23750} = 每行 3 个时第 4–6 位，agent 读到第 7–9 位。
- **失败构成**（§509.6，占全部 episode）：答错 B0 46.9–57.6 → B5 **33.0–46.9**；跑不完 B0 25.0–30.8 → B5 **28.6–35.3**。

**caveats**: Tier-2 子类是 sonnet sub-agent 标签归一后的计数；sub-agent 读不到页面文本和截图；B-1998「这些步不耗 30 步预算 ⇒ B5 / B2 的步数与空转类指标 (§508.2) 带偏; B0/B1 为 0 ⇒ 预注册 cell 数字不受影响」；关键词代理与先搜索「是风险因素, 不能写成规则」；task 41「sub-agent 的『站点在 08-20→08-26 漂移』解释被此推翻; 对 SR 绝对值影响上限 0.45pp」，需 A100 截图确认；§509.6 只有 cls、B5 无 vision。

**证据**: §509.2 / §509.3 / §509.5 / §509.6；`docs/analysis/vwa_classifieds/B5_classifieds_cross_mode_diag_summary.md`、`master_bug_catalog` B-1998

**原文片段**: 「agent-limit 174 · scaffold 1 (t210 → B-1998) · benchmark-FP 嫌疑 4 (全是 task 41)」(§509.2)

---

# F. 动作通道与 backbone 行为剖面

## F1. 动作定位通道与 identifier contract —— grounding 是混淆项，但不沿表征轴

**当前值**:
- §484 B0×classifieds 前 60 episode：dom **511/511 element_id**；som 436/438 element_id（2 coordinate）；vision **491/491 coordinate，零 element_id**。
- §503 per-mode 整体动作成功率（6 cell, 17,089 次动作）：P-text **75.0%** · P-SoM 66.5% · SoM 61.8% · DOM 59.9% · P-prompt 57.7% · Vision **40.8%**；native nodeId 组 fallback 率（DOM 40.1% / P-prompt 35.2%）明显高于 compact 1..K 组（22.5-27.8%）；文本臂内部差距 **17.3pp** ≈ Vision↔DOM 的 **19.1pp**；fallback 率随 backbone 变弱上升（B0 12.2% → B1 34.7% → B2 37.0%）。
- §512.2 动作占比（8 格中位）：翻页 Vision **29.7%**（24.1–36.0），DOM 6.9 · P-text 5.6 · P-prompt 6.0 · P-SoM 5.6 · SoM 6.9 → Vision 是各自的 **4.3–5.3×**；打字 Vision 8.9% vs 17.3–22.0%（0.41–0.51×）；逐设置 Vision 翻页 ÷ 同格最多的另一种 = **1.25–7.05×**。

**已作废**: 毕设 `ch3_system.tex:43`「All modes ground actions through element identifiers」—— 被 §484 推翻。

**caveats**: 「论文自己点名的致命 confound, 在 Vision 上真实存在。涉及 Vision 的对比 (含 oracle/ceiling/drop-one) 混入了动作空间差异; 五个 element_id mode 之间的对比不受影响」(§484)；「分层不沿『文本 vs 像素』而沿 **identifier contract**」(§503)；海报「~4×」是中位数之比，「稳的说法是『每个设置 Vision 都翻页最多』(它最低 24.1% > 其他最高 20.8%)」(§512.2)。

**证据**: §484 / §503 / §512.2；`dispatch_path_audit.json`、`docs/analysis/cross_sites/per_mode_four_dimension_profile_with_wa.json`

**原文片段**: 「grounding 确实是混淆项但**不沿表征轴分布**」(§503)

---

## F2. cls 四个 backbone 的行为剖面（空转 / 步数 / 每成功价格 / 失败桶 / thought）

**当前值（classifieds canonical，vision 臂除外）**:
- **空转率 / 步数 / SR / $/ep**（§508.2，文本侧五臂范围）：空转 B2 42-56% · B1 25-32% · B0 7-15% · B5 4-6%；步数 B2 24-28 · B1 18-22 · B0 13.6-16.6 · B5 15.3-17.7；SR B2 0.4-2.2 · B1 6.2-14.3 · B0 14.3-29.5 · B5 21.9-37.1%；$/ep B2 0.07-0.09 · B1 0.06 · B0 0.07 · B5 0.14-0.17。
- **每次成功的价格**（§508.2）：som B0 **$0.24** · B1 $0.42 · B5 $0.45 · B2 $4.14；dom B0 $0.40 · B1 $0.90 · B5 $0.56 · B2 $5.9。
- **失败桶迁移**（§508.3，占 failed %，六 mode 范围）：cls 答错 B2 5-14 · B1 34-48 · B0 62-70%；cls 跑不完 B2 69-87 · B1 52-65 · B0 30-38%；red 答错 B2 4-16 · B1 19-33 · B0 29-50%；red 跑不完 B2 76-89 · B1 61-80 · B0 45-70%。（B5 补充见 E4 §509.6）
- **thought 行为**（§508.4，cls dom/som）：与上一步逐字相同 B2 11.2/14.3 · B1 11.5/11.8 · B0 0/0 · B5 0/0.1%；提到没成 B2 14.0/8.2 · B1 20.7/13.8 · B0 8.9/8.5 · B5 6.5/6.1%；中位词数 B2 42 · B1 46/52 · B0 41 · B5 23；空 thought B0 13.8/21.2%，其余 ≤1.5%。

**caveats**: 「是『按 SR 排序的四个 backbone』不是能力阶梯: family / API-vs-本地 / tool_call·response_format·文本JSON 三种输出路径 / B5 无 logprob 四重混淆」；replicate 两次空转率差 0.1-1.5pp 但未登记成 band；B1/B2 本地成本是能耗折算（且来源为 psutil 估算，见 G8）；「没算 band」；失败桶是确定性规则映射非人工标注；B0 空 thought 是 tool_call 缺字段的 fallback；thought 指标「是机制描述不是在线信号」。

**证据**: §508.2 / §508.3 / §508.4；`docs/analysis/cross_sites/failure_modes_per_cell.md`

**原文片段**: 「是『按 SR 排序的四个 backbone』不是能力阶梯」(§508.2)

---

# G. 管线 / 基础设施 / 卫生

> §474 以前见 measured_D5.md（§469.5 等）

## G1. reddit 双分母（205 / 203）与 universe 漏网

**当前值**:
- **B-1992**（§478.6）：`_reframe_chain.sh` 的 RED_N=203（scoring）被用作 collection 检查 ⇒ Phase B 三格各跑满 205 后报 `episodes=205 != expected=203` → HALT；主机空转 5.5h；数据完好；同期 `_b5_reddit_chain.sh` 初稿照抄 203。全仓复查其余 203 均为 analysis 层用途。
- **shopping 主 universe 分叉**（§497.3）：classifieds 224=224，reddit 203=203，**shopping expected_scored_ids=433 而 paper_scored_task_count=432**（tiers=(A,B) vs (A,B,E)）。
- **unique-solve 包络**（§510.2）：协议排除 task 58/160 混进分母 → 上界抬高（见 A9）。
- **persistent_state_leakage_audit**（§510.4）：加 canonical universe 限制后**零变化**（md 逐字节相同，WA leaked 0→0，VWA leaked 22→22）；`scan_b1969_contamination` 与 `early_abort_B0_classifieds` 重跑逐字节相同。
- **§505 数据源污染**（§510.5）：`step0.jsonl` 里协议排除任务 **82 行 / 21,291 = 0.39%**，其中 **46 个 success**；red_B0 36 行(14 succ) · red_B1 20 行(**19 succ**) · red_B2 12 行(6 succ) · shop_B0 6 行(6 succ) · shop_B1 8 行(1 succ)；cls 与两个 WA 格均为 0。**未重跑**。

**已作废**: §510.5 第一版用排除集并集 {58,160,463,465} 匹配得 **186 行** —— 错（把 cls 同号合法任务算进来）。

**caveats**: 「AMENDMENT_08 原文已把两个分母分开写明 … 是落地只取一半, 不是预案没写」；「『reddit=203』这个记忆本身有缺陷」；「校验在 phase 末尾 ⇒ 潜伏到三格 (~$66/4.5 天) 跑完才触发」；shopping 分叉「反证 canonical SHA 可以精确地 hash 一个错的 primary universe」；「『修了但数字不变』要如实说 —— 这是**确认原数字正确**, 不是『修复生效』」；「红线是 `sr_excluded` 过滤 ≠ universe 处理」；0.39% 对 §505 pp 级结论方向不构成威胁，但那批数字在 collected 口径上（digest `:286`「full 205 tasks」即此）。

**证据**: §478.6 / §497.3 / §510.2 / §510.4 / §510.5；`master_bug_catalog` B-1992、`scripts/analysis/lib/canonical_task_universe.py`、`results/router_llm_pilot_20260909/lookahead/step0.jsonl.gz`

**原文片段**: 「**必须按 cell 的 site 分别取排除集**」(§510.5)

---

## G2. fire gate / validator 的覆盖缺口

**当前值**:
- **watchdog 截断**（§492.3）：`experiment_watchdog.py:1185` 的 `[-400:]`；validate_fire_manifest stdout = **1948 字节**，stderr 在拼接串位置 0 ⇒ 截断 **100%** 落在 benign 行上，真诊断被完整挤出。
- **BASELINES 覆盖**（§492.4）：`BASELINES = ('B0','B1','B2')`（validate_fire_manifest.py:40）⇒ **B5 的 8 个 paper-grade run 一个都不进 ghost/binding 循环**。
- **paper_grade_check 可见率**（§492.6）：`for i in issues[:10]`，VERDICT 打 len(issues) ⇒ 长期显示 'ISSUES=24' 后跟 10 行，**14 条从未被看见**。
- **validator 对空 schema**（§497.1）：空 conditions+scored_task_count manifest 返回 **rc=0/[OK]**；绑一个无 condition summary 的在跑 run 为 authoritative 返回 rc=0/bound-clean=1。
- **replicate provenance**（§497.5）：FORCE_NEW 定向测试 2 passed；两臂 env git commit 不同且均 dirty=true，resolved-config leaf 差异 **12**；新 artifact 无 replicate_of/intent_sha/parent_run_id。
- **A100 vs DGX fire 路径一致性**（§515.4）：269 个文件逐个 sha256，**1 个不同**（`_lib_paper_grade_gates.sh`，只差注释/编号与换行，行为相同）。

**已作废**: §515.4 首次汇总用 `diff | uniq -c -f1` 误报 **40 个不同** —— 用 join 按文件名配对才对。

**caveats**: 「stdout 随注册的 replicate 数增多而继续变长 ⇒ 截断只会越来越严 … §469.5 (08-17) 已诊断出同一病因但**代码未改**」；「**B5 数字进 NAACL 稿前必须先补这道覆盖**」；paper_grade_check「『不是新问题』这个判断**只对可见的 10 条成立**」；§497.5 的 12 个差异都是 shopping_cart_reset 字段，「结论是缺 pair-compatibility/intent-join gate」；§497.1 Claude 未独立复跑（见 A6）；A100 与 DGX commit hash 不共享。

**证据**: §492.3 / §492.4 / §492.6 / §497.1 / §497.5 / §515.4；`scripts/analysis/validate_fire_manifest.py`、`scripts/maintenance/{experiment_watchdog,paper_grade_check}.py`、`scripts/queues/{queue_baseline.sh,_lib_paper_grade_gates.sh}`

**原文片段**: 「一个活着的 backbone 的全部数据坐在链外」(§492.4)

---

## G3. 手写注册表的「第二个 CLEAN_PAIRS」—— 新臂不登记就不进计算

**当前值**:
- §476.3 毕设表 4.6：cls·B0 **六条臂全部有 replicate**，论文表只有 3 条；`rerun_union_extrapolation.py` 的 `MODE_OF` 只映射 dom/som/vision，遇 ptext 直接 `SystemExit`（fail-closed）。补全后 3 行 → 6 行；新增三臂 SR P-prompt 19.64 / P-text 15.62 / P-SoM 15.62；headline 不变（SoM 27.23%，3.53pp 与 16.07pp headroom 不变）。
- §477.3 CLEAN_PAIRS 六臂化后 consumer 回归（无人跑过）：**5 个脚本**受影响 —— `retry_vs_switch_label_supply.py` exit=1 · `replicate_metric_noise.py` **静默**少算 · `fig_f10_rerun_discordance.py` 仍标 three · `fig_f9`+`fig_f10b` regex 未限定 cell 拉进 B1 零 discordance 行（band 0.00--4.15 / 0.00--7.59）· `export_ablation_tables.py` 数了 10 对却说 on one cell。
- §479.2 **B-1994**：`retry_vs_switch_label_supply.py` 自 08-17 起 dom/som/vision 臂被 B1/B5 数据覆盖（arity 检查只比 key 集合故通过）；`aggregate_label_instability.py` 静默串站；两产物 mtime 早于首次碰撞 ⇒ 已发布数字未受影响；修后 dom SR a=17.41%。
- §500.5 `aggregate_confidence_cascade.py` glob 双匹配：自 2026-08-18 起每次 MissingInput fail-loud，`confidence_cascade.md` 停在 08-03 达 **5 周**；重跑后内容**零变化**。
- §517.3 `router_objective_ordering._wa_matrix`：WA·B1 DOM 匹配 2 个目录 ⇒ 返回 None，**整格静默跳过**。

**caveats**: 「**`MODE_OF` / `REPLICATES` / `ARMS` 是第二个 CLEAN_PAIRS** … 五个里只有一个 fail-closed, 其余**静默**」；「**形状检查检不出内容被换掉**」；「fail-closed 把静默 bug 变成响亮 bug」；「**fail-loud 起了作用但没人看**」；§517.3「重跑前要先让 glob 只选 canonical 运行（或按 run_manifest 固定）」。

**证据**: §476.3 / §477.3 / §479.2 / §500.5 / §517.3；`scripts/analysis/{rerun_union_extrapolation,aggregate_confidence_cascade,router_objective_ordering}.py`、`master_bug_catalog` B-1994

**原文片段**: 「五个里只有一个 fail-closed, 其余**静默**」(§477.3)

---

## G4. runner / 环境层缺陷

**当前值**:
- **B-1991**（§478.6）：cls docker-restart reset 冷态实测 **262s** vs 阈值 240s(+10s) ⇒ SIGKILL；reset 实际成功、判据失败；热态同格仅 **78s**；一格 abort ⇒ 后 4 格（~$154）未跑；修复 480s。
- **VWA `_key2id` 表**（§487.1）：表大小 **129983** = SPECIAL_KEYS(15) + chr(32..127) + chr(129..129999) + "\n"；`_key2id["\n"]=129982`，与 fire log 逐位吻合；表外 = 控制字符（除 \n）+ chr(128) + chr(>=130000)。
- **cls reset 残留**（§487.2）：oc_t_alerts=**1**、oc_t_latest_searches=0；上游 reset.php 不碰这两张表，named volume 不重播种 ⇒ 管线无任何环节清理。
- **steps JSONL 与 summary 不一致第 3 例**（§505.26）：shop_B0 dom task 346 JSONL 30 步 vs summary steps=28；§400.1 两例之外，全库 62 条 canonical 臂中 **3 个 task**。
- **B-1996**（§506.9）：CLI `--max_steps` 不限制步数 —— 实际上限仍 30（live LOOK 跑了 24 步，run_meta 却记 12）。
- **B-1998**（§509.3）：runner 在 env.step() 之后才算 parse_valid（数值见 E4）。

**caveats**: 「240s headroom (B-1839) 是按**热主机**量的」；「代价被 fail-closed 放大」；`_key2id` 判定须复刻双重查表；§487.2「oc_t_user 非 admin 活跃数=1 … **不是**脏数据 … 删它会让整个 cls 无法登录」；§505.26 机制是按 §400.1 同形推断，三例都是失败→失败，「步级分析须排除并披露」；B-1996「正式实验不受影响」；B-1998 未修（fire 路径）。

**证据**: §478.6 / §487.1 / §487.2 / §505.26 / §506.9 / §509.3；`master_bug_catalog` B-1991 / B-1996 / B-1998、`external/visualwebarena/browser_env/{actions.py,constants.py}`、`reset_vwa_sites.sh:215-218`

**原文片段**: 「reset 成功, 判据失败」(§478.6)

---

## G5. diag ruleset 缺陷与 Table 41 按站点拆分

**当前值**:
- **B-1999 P31 豁免漏标**（§509.4）：cls B0 **203** · B1 436 · B2 453 · B5 147（每 condition 26–94，约占 incomplete 的 40–60%）；reddit 18 个 condition 全 0。
- **B-2000 P10 千分位**（§509.4）：失败侧 B0 56→50 · B1 39→29 · B2 16→14 · B5 61→50；成功侧 B0 20→20 · B1 1→1 · B5 31→30。
- **P33**（§509.4，B5 9 个命中）：som 4/4 非死因；无图 mode 5 个中 3 个是真编造。
- **Table 41 按站点拆**（§512.3，ruleset v11）：合并 8 格逐数一致（A 块 101 题：P27 2.31×(13) · P17 2.25×(61) · P16 2.24×(25) · P43 1.65×(196)；B 块 109 题：P49 3.61×(8) · P17 1.17× · P12 0.93× · P31 0.91×）；VWA 六格 A 块 P27 2.98×(11) …；WA 两格 B 块 P49 **1.53×**(8)。

**caveats**: Osclass 每页 path 都是 /index.php ⇒ cls url_match 任务一律豁免，「不影响 SR 与失败桶」；「千分位只解释一小部分」；P33 样本小，建议 som/vision 降为中性事件，未改规则；「P49 合并后的 3.61× 在 WA 内部只有 1.53×, 是合并时基线被 VWA 稀释造成的放大。富集是比值不是检验」；文本侧 4 臂、图像侧 2 臂两块不能互比。三条均未修（需 v12）。

**证据**: §509.4 / §512.3；`master_bug_catalog` B-1999 / B-2000、`docs/analysis/cross_sites/conditional_failure_attribution.json`

**原文片段**: 「P49 合并后的 3.61× 在 WA 内部只有 1.53×」(§512.3)

---

## G6. 测试套件与分析工具参数

**当前值**:
- §510.7 `make test` 预存失败：起点 24 条，本 session 清 3 条，**21 条剩余** = **16 条 paper prose 路径 stale**（test_stress_a2_1_phantom_framing 占 12 条）· **3 条 diag 规则 ratchet**（RULESET_VERSION 已到 `11-intent-text-fallback`，测试要求 `9-`）· **2 条其余**。
- §510.7.1 B-1959 修复的测试覆盖**改前为零**：现有测试 harness（tests/test_b1957_shop_tail_follow.py:107）**自己重新定义了**不含 `_consumed` 的 `_resume_filter_done`；新增测试反向验证：no-op 变红报 `got []`，还原 33 passed。
- §497.4 `power_analysis.py`：(alpha,beta)=(0.05,0.20)/(0.01,0.10)/(0.20,0.40) 三组全返回 MDE=**0.078062555438**、power(3pp)=**0.245339880464**；`--beta 0.10` 仍渲染 power=80%。

**caveats**: 「**16 条 prose 类不是改路径就能修** … 属 user 裁定」；「**一个测试如果在 harness 里重新定义了被测函数, 它测的就不是那个函数**」，未普查全库；power_analysis「默认 alpha=.05/beta=.20 的现有表格数值不因此变错」，Claude 未独立复跑（§497.7）。

**证据**: §497.4 / §510.7 / §510.7.1；`scripts/analysis/power_analysis.py`、`scripts/queues/queue_phase1_paper_grade.sh:645`

**原文片段**: 「一个测试如果在 harness 里重新定义了被测函数, 它测的就不是那个函数」(§510.7.1)

---

## G7. AWS proxy 共享预算

**当前值**: proxy 共享池实时余额 **$30.18**（`proxy_budget_watch.py --once`，A100，2026-09-15 08:59 UTC，§515.1；§516.3 同值）；一次三栏 live 约 **$0.15**（README 估算）。reframe chain 历史支出见 E2（$342.50 / 86%）。

**caveats**: 「单点、共享池 (他人也在花); 恰在 09-02 意图书的 $30 停机线上 ⇒ B0/B5 任何付费 replicate 都不应发车」；DGX 上那份停在 08-26（$383.67）是 stale，**勿引**；「floor $30」无程序强制；live server 额度用完时三栏以 error 提前结束、页面不说明原因（`why='budget'` 只表示跑满 12 步）。

**证据**: §515.1 / §516.3；`scripts/maintenance/proxy_budget_watch.py`、`deliverables/showcase/demo/live/{server.py,run_lane.sh}`

**原文片段**: 「恰在 09-02 意图书的 $30 停机线上」(§515.1)

---

## G8. 机器（DGX / A100 / quark）

**当前值**:
- **DGX 磁盘第一次告警**（§487.6）：repo 仅 12G，/home/jiaming 238G，其余 ~3.2T 属其他用户；清 FinQA 纯文本 Qwen3 55.9G + 缓存 5G ⇒ free **49G → 109G**。
- **DGX 第二次告警**（§514.2/§514.3）：free 50G → 13.5G（约 0.4GB/分钟）；同期他人 `ollama pull`（单个 96.5GB GGUF + 约 20GB）；删自己的 gemma-4-abliterated:31B 腾出 **19.87GB**，free 13.5G → 32.5G。
- **A100 nvidia 用户态/内核态裂开第二次**（§515.2）：unattended-upgrades 09-12 把 580.173.02 → 580.178.04，内核模块仍旧版；nvidia-smi 报 mismatch，但 torch CUDA 正常（只坏 NVML）。
- **A100 本地 run 能耗来源**（§515.2）：step `source` = **psutil_profile**（抽 2 run 首个 task 30/30 步）；A100 venv `import pynvml` ModuleNotFoundError。
- **quark ↔ DGX 连通**（§506.3）：DGX→quark Tailscale 通；quark→DGX Tailscale :8799 不通；quark→`ssh spark`（cloudflared）通。

**caveats**: 第一次告警文案「Prune logs/ artifacts/」归因错误；P79 三个 local baseline 权重未动；「fire 在 A100 而非 DGX」；第二次归因只到「时间+体量+速度吻合」，前 40 分钟写入者未确定（「时间共现不构成证据」）；「不能用 `ollama rm` 删自己的库」；「『CUDA available』不能当驱动健康的证据」（与 §387.1 同形第二次）；**A100 本地 run 的能耗/电费列是『CPU 利用率 × a100_pcie_40gb TDP 档』估算, 从未用 GPU 实测功耗**，写 cost/能耗句子须带此限定（只抽样 2 run）；连通性只测了当天当前网络。

**证据**: §487.6 / §506.3 / §514.2 / §514.3 / §515.2

**原文片段**: 「A100 本地 run 的能耗/电费列是『CPU 利用率 × a100_pcie_40gb TDP 档』估算, 从未用 GPU 实测功耗」(§515.2)

---

# H. 毕设 / 论文

> §474 以前见 measured_D5.md（§452 单向同步等）

## H1. 毕设提交规格与评审层落差

**当前值**: 最终提交（§502，2026-09-08 16:00）：**119 页** / 0 error / 0 undefined reference / overfull 9（= 加表单前基线），含 declaration form 2 页；**页数无上限**（user 2026-09-08 确认）。

**演变**: §475.8-adjacent（08-22）：user 报 **≤100 页**，全稿 **89 页** ⇒ ~11 页余量（user 原话「大概是100以内」是回忆不是 handbook）→ §502 确认无上限。

**已作废**: 「≤100 页」约束与「89 页」「114 页」页数 —— 「勿再引『89 页』『114 页』或任何页数预算约束」(§502)。

**caveats**: Overleaf 落差（§475.8-adjacent，08-22 时点）：Overleaf clone 停在 `cb02b22` / 2026-08-11 12:53，tex 最后大改 2026-08-21 22:24（`7229ed7`，de-jargon ~200 处，84→89 页）⇒ **落后 10 天**，「落差正好覆盖那轮为导师做的改写」；08-11 后台账/笔记无 advisor 交互记录（最近 07-24）。

**证据**: §475.8-adjacent（会话直接问答，未开新 §）/ §502；`scripts/maintenance/overleaf_thesis_sync.sh`、`final_dissertation/tex/main.pdf`

**原文片段**: 「勿再引『89 页』『114 页』或任何页数预算约束」(§502)

---

## H2. 毕设内容级修正

**当前值**:
- 表 4.6 replicate 臂 3 → 6（§476.3，见 G3）。
- exact 阈值口径 = 零分布 95 分位数（§477.2，见 A2）。
- F6 被 argmax 静默吃掉的并列（§481.2）：**2 处** —— cls·B2 SoM 与 Vision 同为 2.23%；wa_red·B1 DOM / P-prompt / P-text 三路同为 16.35%（底层计数确为 5/224 vs 5/224、17/104 ×3）。
- ch3:43「All modes ground actions through element identifiers」被 §484 推翻（见 F1）。
- 全稿「the oracle」→ triage_only 限定 9+1 处（§486.2，见 B1）。

**caveats**: 并列判定 abs(v-top)<1e-9；源表只有 2 位小数，底层计数不同时会误判（本例已核为同一计数）。

**证据**: §476.3 / §477.2 / §481.2 / §484 / §486.2；`router_objective_ordering.md`（经 _ordering_parse 对账）

**原文片段**: 「cls·B2 上 SoM 与 Vision 同为 2.23%」(§481.2)

---

## H3. 毕设排版与审读的测量

**当前值**:
- 图内散文串（§481.1）：**105 条 → 0 条**（check_no_prose.py，本仓自产 11 张）。
- em dash（§481.5）：**88 处 → 0 处**（en dash 数值区间保留）。
- 最小实词字高（§482.1）：**6.1pt**（排除 AXTree 代码纹理 2.0pt 与 Table 4.3 \tiny 5.3pt；115 页全书）。
- 未被 \ref 的图（§483.3）：改前 **6 张** → 0 张。
- float 跨页断句（§494.1 实测取代 §493.2 首行）：[t] 原始 **116 页 / 10 处**（p39/42/50/55/58/60/61/64/68/73）· [tb] 115/8 · [htb] 115/8 · [b] **120 页**/1 · **[!b] 116 页/1 处** ⇒ 真实修复 **10→1**。
- `--allow-partial` 字节（§493.3）：修前 `e2 80 93`（U+2013）→ 修后 `2d 2d`。
- Fig 4.2 面板（§493.5）：180.4 x 123.9 pt；FS_VALUE=7.5 下六个 mode 名并排需 **183pt** ⇒ 不存在水平排布能放下；改「只标非支配点」后正文点名却无标签的 mode **3 个**（§494.4，已加 PROSE_NAMED 白名单）。
- GPT 第二轮成品审读（§493.6）：**11 条真 / 3 条假**；「四处标题断词」是第二轮重复误报（两轮均 0 命中）。

**已作废**: §493.2 首行「[t]=8 处」——「那是**推的不是量的**」，§494.1 还原重编译实测 **10**；「真实修复是 10→1 而非 8→1」。

**caveats**: 散文判定是启发式；「量字号必须先滤标点 … ink bbox 高度 != 字号点数, 只可用于相对比较」；\ref 只验证存在不验证位置；`!` 解除 \bottomfraction 限制是关键；lmodern T1 打字机字体带 `--` 连字，与 08-29 路径断词是「两个不同失效」；Fig 4.2 是「明知偏离建议而偏离」+ 放置器调用顺序 bug；GPT 审读口径 117 页 vs 本地 116 页 ⇒「所有按页码定位的说法都必须按内容重核」，「既不能整体采信也不能整体驳回」。

**证据**: §481.1 / §481.5 / §482.1 / §483.3 / §493.2 / §493.3 / §493.5 / §493.6 / §494.1 / §494.4；`scripts/analysis/figures/thesis/fig_f7_cost_sr_frontier.py`

**原文片段**: 「真实修复是 10→1 而非 8→1」(§493.2)

---

## H4. REALM @ EMNLP 2026

**当前值**:
- §502.2：**接受**（2026-09-08），forum `EAplLx6gCD`，camera-ready **2026-09-14 AoE**；OpenReview #192，提交 2026-08-06。
- §504.3 审稿人 4s7L 要求 reconcile：**36 = VWA 六格 × 6 mode**（不含 WA）；**48 = 含 WA 两格的 6×8**；`2_setup.tex:19` 的 **7,686 = 224×18 + 203×18**，纯 VWA；WA 全量若需另算（104×6×2=1,248）。

**caveats**: 「**camera-ready 未做, 且含不可逆决策** … 选归档 ⇒ 正式发表 ⇒ **占掉 NAACL ARR 投稿权**; 保持非归档才留住主会资格。09-14 前必须裁定」（09-08 时点）；「**两个数各自都对**, 缺的只是 scope 声明 —— 修法是标注 scope, **不是**改数字」。4s7L 的 noise-floor-mode 问题见 A2。

**证据**: §502.2 / §504.3；`sections/2_setup.tex:5,19`、`sections/3_complementarity.tex:3`

**原文片段**: 「7,686 = 224×18 + 203×18」(§504.3)

---

# I. 投稿合规 / 匿名化（ARR / NAACL 2027）

## I1. public repo 暴露面与匿名 export

**当前值**:
- §475.2（`git ls-files` + `git log --all`，2026-08-21 HEAD）：identity：**1705/1705 commits 单一 author**；`portfolio/Jiaming-Wei-*.{pdf,pptx}`；`.gitmodules` 指向个人 fork。credential：AWS proxy endpoint 明文在 **97 个文件**（key 未泄）；`.auth/{reddit,shopping}_state.json` 曾于 `ef2c283` commit、`85da448` 删除。
- §475.4 匿名 export 脚本首跑（1231 文件）：**5 个真 bug**，全属「白名单/边界」类（扩展名白名单 / `\b` 在文件名里失效 / `HolisticAI` 无分隔 ×7 文件 / 省略号 host / shell 默认值里的 ntfy topic）。

**caveats**: 「**后两条都不构成行动理由** … `.gitmodules` 那条**是**真泄漏, 匿名 export 必须改写为上游 URL + patch bundle」；统一修法「**按内容嗅探而非按扩展名**(首 8KiB 有 NUL 即二进制)」；主机名嵌在文件名里时 `\b` 系统性失效，需 `(?<![A-Za-z0-9])`。

**证据**: §475.2 / §475.4；`scripts/maintenance/export_anonymous_repo.py`

**原文片段**: 「`.gitmodules` 那条**是**真泄漏」(§475.2)

---

## I2. ARR 政策与 NAACL 2027 deadline

**当前值**:
- ARR submission form 含 Existing Preprints 字段；preprint 定义是「功能性而非平台性」（§475.8）。
- 毕设：「University dissertations and theses do not count as prior publications」⇒ 不触发 dual-submission；但稿件「must not contain explicit references to the authors' prior work」，similar prior work 须匿名方式 disclose/cite（§475.8）。
- NAACL 2027 commitment deadline 两处冲突：ARR dates 表 **December 20, 2026** vs NAACL CFP（2026-08-18 发布）**December 23, 2026** ⇒ **取 12-23**；ARR submission deadline **2026-10-12** 两处一致。

**caveats**: 「**『含论文源的 GitHub repo 是否算 preprint』政策未点名, 属 inference 而非明文** … 提交时按保守处理(申报), 但不要把它当作已确证的规则引用」；「非归档 workshop 稿的具体引用格式政策未规定」，两条约束「作用于**稿件写法**, 与 repo 可见性无关」；deadline 冲突「临近前须再核一次」。

**证据**: §475.8；aclrollingreview.org/submissionform、aclrollingreview.org/authors、2027.naacl.org/calls/main_conference_papers/

**原文片段**: 「**取 12-23**(venue-specific CFP 比 ARR 汇总表权威, 且发布更晚)」(§475.8)

---

# J. Showcase / 海报 / demo / 演讲

## J1. 海报排版度量（build_poster.py）

**当前值**:
- 字体行高因子（§495，ImageFont.getmetrics）：Arimo(=Arial) **1.118** · NotoSerif **1.362** · DejaVuSansMono **1.165** × 字号。
- LibreOffice 行推进因子（§498.2，Liberation Sans）：**1.20**（17.65pt × 1.25 行距实测 9.3mm/行；字体文件 1.118 → 8.7mm/行，低估 7%；图注同比 1.07）。
- WIDTH_CAL（§499.8）：**0.97**（另需在 – — / 之后允许断行）。

**caveats**: 漏乘因子系统性少算约 12%；「该因子随字体族变, 换字体必须重测」；「§495 的『漏乘 ascent+descent 少 12%』与本条是同一个洞的两层, 修一层不等于修完」；PowerPoint 原生渲染未测；✓ ✗ 在 Arimo 无字形、PIL 量为零宽，须在渲染图里核；ROW_BOTTOM 与 footer 之间 15mm「是人为留白, 不是可用余量」。

**证据**: §495 / §498.2 / §499.8；`deliverables/showcase/build_poster.py`

**原文片段**: 「修一层不等于修完」(§498.2)

---

## J2. demo 选题与三栏结果（cls·B0）

**当前值**:
- kayak（cls task 0）三表征**全部失败**：canonical dom 30 步 / som 6 / vision 4；replicate 30 / 12 / 5；$0.144 / 0.032 / 0.015（§499.2）。
- 三表征结果不同且 canonical↔replicate 每模式 ✓✗ 一致的任务 **24/224**（✓✗✓ 13 · ✗✗✓ 4 · ✓✗✗ 3 · ✗✓✓ 2 · ✗✓✗ 2）；三种表征结果不同的题共 70，其中 46 在某模式上翻转（§499.2）。
- demo 三题（canonical）：130 Look ✓ 2 步 $0.007 · Read ✗ 21 步 $0.096 · Both ✓ 3 步 $0.014 | 76 Look ✗ 30 步 $0.116 · Read ✓ 9 步 $0.079 · Both ✗ 6 步 $0.029 | 17 Look ✗ 19 步 $0.075 · Read ✗ 6 步 $0.025 · Both ✓ 7 步 $0.037（§499.2）。
- task 76 两个 replicate 的不同页面数（§499.9）：READ 12 步 → **8** 个；LOOK 26 步 → **9** 个，最大重复组 7 帧像素级相同。
- learned router（L1 LR, oof）对三题（§506.1）：130 → dom/READ（✗）· 76 → dom/READ（✓, 两次都 ✓）· 17 → phantom_prompt（canonical ✓ / replicate ✗）；router 海报成绩 **0 of 8**（vs always-cheapest）。预算路由对三题见 D2。

**caveats**: kayak「只能演『三种表征长什么样』, 演不出结果差异」，「『蓝色只能靠看』未验证」；「单次运行选 demo 题会有 2/3 概率选到不稳的」；「不得说某题『天生是视觉题』, 只能说『这次运行』」；§499.9 阈值 2.0 目视标定，「是一个任务的轶事级证据」，screenshot 是动作**后**状态而 obs_url 是动作**前**；三题按 view 差异挑，不是按 router 表现；SoM 逐步 artifacts 已清。

**证据**: §499.2 / §499.9 / §506.1；`deliverables/showcase/figures/demo_strip.json`、`deliverables/showcase/poster_figures.py`、`results/phantom_paper/l1_router_offline_20260715/router_offline_replay.json`

**原文片段**: 「单次运行选 demo 题会有 2/3 概率选到不稳的」(§499.2)

---

## J3. demo 前端与 deck 版式

**当前值**:
- 步进动画（§501.3）：指针 252ms 走过半程 / 585ms 到九成；涟漪 t=1001ms，画面 t=1350ms 才换；无闪：亮度基线 211.8，最低 **165.7**（重建会掉到 ~13）。
- 便携单文件（§501.5）：82 帧 lossless WebP **15.0 MB → 8.7 MB (58.0%)**，逐帧字节一致；内联后 **11.7 MB**；冷开 ~3s。
- demo iframe 截图高度（§511.3，task 130 第 1 步）：套模板前 1920×1080 328px · 1536×864 149 · 1280×720 **0** · 1440×900 149 · 1280×800 56；套模板后 248 / 78 / 0 / 78 / 0；固定 1880×960 整体缩放后 299 / 240 / 200 / 242 / 215。
- 更宽字体下 deck 余量（§511.6）：第 3、6 张标题两行；内容最低 86.8 / 84.4 / 89.6vh；最右 93.5vw；无裁切。

**caveats**: 「顺序由数据决定不是美术选择: 每步存的 screenshot 是动作**之前**的页面」；「『无损』是**可失败的构建断言**而非声称」，像素 diff「**自己先出过两次假阳性**」；iframe 只量 task 130 第 1 步、DGX headless Chromium、未在真屏上测；DejaVu 只是替身，「Phase 5 要在 MacBook 上实际打开一遍」。

**证据**: §501.3 / §501.5 / §511.3 / §511.6；commit 4fc1453 + 15c0cf8、`deliverables/showcase/demo/build_portable.py`、`deliverables/showcase/talk/index.html`

**原文片段**: 「先换图再点等于把结果放在原因前」(§501.3)

---

## J4. live 页（DGX 站点，B0 via proxy）

**当前值**:
- DGX bgrins arm64 classifieds 与录像站点一致性抽查（§506.6）：3 个 item 页与录像一致（item 19604 / 79747 / 11376）；可见差别：页头文字而非 OsClass logo。
- 首次真站点端到端（§506.6，step cap 12，单次）：**72 s** 全部结束；LOOK 6 步 $0.023 · READ 7 步 $0.029 · BOTH 5 步 $0.025；合计 **$0.077**；console 无错误。
- 按下 Run 到第一步（§506.9）：改前 **31–36 s** → 改后（撤起跑线 + 后台预登录）**19–21 s**。
- 由 live 页发现 B-1996（见 G4）；额度表现与余额见 G7。

**caveats**: 「只抽查了 3 条, 不等于全库一致 … 只能用于不评分的 demo, 不能用于任何测量」；live run「不评分, 不可与录像或论文的任何成功率比较; 三个答案的对错是人工目测判断」；剩余 ~20 s 是冷启动，常驻 runner 未做。

**证据**: §506.6 / §506.9 / §516.3；session scratchpad 截图、`day-of.html`

**原文片段**: 「只能用于不评分的 demo, 不能用于任何测量」(§506.6)

---

## J5. 演讲台本与片子素材数字

**当前值**:
- 台本 v0（§507.5）：**735 词**；÷140 = 5.2 分，÷120 = 6.1 分；加停顿 ≈ 7.0–8.2 分；片子每张词数 13/21/41/32/45/48/31。
- 台本 D24 重写后（§527.2）：**961 词**（改前 932）。
- Playwright MCP browser_snapshot 一次抓取（§524.2，Wikipedia 注册页）：**161 行 YAML / 7,691 字节 / 640 词**；片子节选 32 行。
- 片子用的「~4×」翻页（见 F1 §512.2）与 hindsight 效率表（见 B1 §517.2）。

**caveats**: 「字数只是事前估, 真判据是 Phase 4 掐表 ≤ 9:30」；「只是提醒，不是时长；仍未出声掐表」；snapshot「行数与字节不是 token 数，不能据此说文本比截图贵或便宜；不代表一般网页」。

**证据**: §507.5 / §524.2 / §527.2；`deliverables/showcase/talk/rehearsal-script.md`、`check_talk.py`、`SHOWCASE_PREP §5`

**原文片段**: 「行数与字节不是 token 数」(§524.2)

---

# ⚠️ 矛盾清单

> 一律并列，不选边。

1. **「重跑噪声带」上沿：0.89-2.23pp / 2.23pp vs 0.89-2.68**
   - §490.1 写「已知 rerun noise band 0.89-2.23pp」，§491.6 写「2.23pp 重跑噪声带」
   - §479.1 记毕设引用 cls·B0 六臂 absdiff **0.89-2.68**；同节 som_absdiff **2.23**
   - 台账未说明 §490/§491 用的是哪个 scope（som 单臂？三臂旧值？）。→ A2 / B2。
2. **§478.4 P-SoM 中间值：value 字段 11.52% vs superseded_by 文字 11.2**。两者都已作废（canonical 10.34%），但若有人从 superseded_by 抄会抄到另一个数。→ A1。
3. **C1 API 组下界：7.39%（§480.1/§496.3，B0.red.ptext）vs 4.93%（§500.3/§504.4，B0.red.vision）**
   - 不是矛盾而是锚点换人：§500.2 刚证 7.39% 站得住，§500.3 下界就被事前声明为 inventory-only 的 vision 臂拉到 4.93%。§515.4 的「B2 不能证伪 C1」以 4.93% 为准。→ A5。
4. **learned 策略对 always-cheapest 的 Pareto 胜数**
   - §490.4 two-arm learned **0/8** → §491.6/7 **1/8（cross-fitted）/ 3/8（whole-cell）**；唯一 win 是 cls·B2 的 SR 相同 + 成本低 0.12%
   - 旧结论层 D4 B11：§392.2 learned triage **0/6**、§399 **0/26**
   - 不同 policy（triage vs two-arm）、不同基线口径，**不可合并成一个计数**。→ B2。
5. **「7/8」两义**：oracle_sr_cost **7/8 Pareto 胜** always-cheapest（§484）vs 毕设 Ch6 triage_only「oracle **7/8 失败**」—— 同数相反含义（§484 自注）。→ B1。
6. **预算路由「损失 1.2–1.8pp 优于 fixed cap」（§506.1 转述 §505.21–24）** vs §505.21 实测 learned−fixed 均值优势 **+1.43pp**、§505.24 逐格差 —— 「1.2–1.8」在本批台账中无出处。→ D2。
7. **§492.2 B5−B0 DOM 差 +7.59pp 用的 B5 run**：§478.3 B5 dom 有 A1 23.66% / A2 25.00% 两个，台账未写取哪个。→ E1。
8. **float 断句起点：8 vs 10** —— §493.2 首版 8（推断）→ §494.1 实测 10，已在同批更正。→ H3。

---

## 对旧结论层的 supersede

> 本批条目**没有**明文 RETRACT 任何 §397 以前的数字；以下是状态更新 / 新增限定 / 方法适用范围收窄，逐条注明性质。

- **§398.2（measured_D4 Z1）「B1/B2 本地格 replicate 一个都没有」 → §480.1 / §496.1 / §500.3** —— 状态更新：local 臂已有 3 → 5 条（B1 cls + B1 red）；B2 仍无（§515.4 只给结构上限）。条目未明文引 §398.2。
- **§397.10(3)（measured_D4 A3 caveat）「self-oracle noise 是 B0-MoE 上界、不可外推到本地」 → §478.3 / §480.1** —— 补充：第二个 API 家族（B5）落在 B0 同带内 ⇒「API serving 属性」；随后 C1 分组提出，又于 **§529.4（本批之后）撤回**。
- **§329 B2·dom 自登出的 forensic 方法 → §488.2** —— 适用范围收窄：「**vision 下 §329 的 forensic 方法失效**」。
- **§387.1 A100 驱动裂开 → §515.2** —— 同形第二次；新增「『CUDA available』不能当驱动健康的证据」。
- **全部 B1/B2 本地成本（B-565 / §181 / §260E 的电费折算口径） → §515.2** —— 新增限定：A100 本地 run 能耗是 psutil × TDP 档估算，从未 GPU 实测。
- **§387.15.1 AMENDMENT_08（collection 205 / scoring 203，measured_D4 A6） → §478.6 B-1992** —— 非推翻：预案正确，落地只取一半；§497.3 另发现 shopping 433 vs 432 分叉。
- **§392.2 / §399（measured_D4 B11）learned 0/6 · 0/26 → §491.6/7** —— 非推翻、并列：不同 policy（two-arm），1/8 的唯一 win 不具部署意义（矛盾清单 #4）。
- **§387.16.4 triage learnability → §505.11** —— 非推翻：「与 §387.16.4 同形」再次复现。

---

# 覆盖性闭合

- 本批条目数 **170** = 实际用到的条目数 **170**（每条至少归入一个主题；多主题支撑的条目如 §476.3 / §484 / §486.2 / §509.3 / §510.2 / §510.5 / §517.3 在多处出现）。
- **未归入任何主题的条目：0 条。**
- 主题数 **53**（A 9 · B 3 · C 13 · D 3 · E 4 · F 2 · G 8 · H 4 · I 2 · J 5），矛盾清单 8 条，对旧结论层 supersede / 限定 8 条。
- 唯一带 `superseded_by` 字段的条目：§478.4（→ §479.1），已在 A1 处理。
- §529.4 的撤回提示已加在 A1 / A2 / A3 / A4 / A5 / A6 / A7 / B2 / C7 末尾。

*本文件由 D6 批 170 条 MEASURED 记录聚合而成。数字一律原样抄写，未做任何算术。*
