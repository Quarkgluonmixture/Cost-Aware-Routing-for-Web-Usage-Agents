# CLAIM_EVIDENCE_MATRIX

> 起手文件 2/5（Guide §21 + §26）。每个 headline claim → 主证据 → 三角验证 → 局限 → 落在哪一章。
>
> **本项目加了两列，指南里没有**：`台账溯源` 和 `⚠️ 已作废的说法`。
> 原因很实在——这个项目最大的写作风险不是"claim 没证据"，而是**引用了自己已经推翻的那一版**。
> §394（AUROC 叙述）、§396.2（天花板全在 cost）、§442.7（污染 p 值）都被自己 RETRACT 过，
> 而作废的数字仍散落在旧稿、旧笔记、旧图注里。写每一句前用
> `known.py <关键词>` 查一次，比事后被 examiner 或 reviewer 抓到便宜得多。

**建立日期**：2026-08-09 · **对齐**：[THESIS_ONE_SENTENCE.md](THESIS_ONE_SENTENCE.md)

---

## 主论证链（四步，缺一步则论证不闭合）

| ID | Research question | Claim | 主证据 | 三角验证 | Limitation | 章节 | Appendix |
|---|---|---|---|---|---|---|---|
| **C1** | 表征之间真的有值得路由的差异吗？ | **上限真实存在**：oracle 相对最强单模 SR **+3.45~16.35pp**，成本低 **13.7–35.3%** | `router_objective_ordering.md`（`oracle_sr_cost` 行）· `phase1_full_prereg_decision.json` · `sr_per_mode.json` | **8 cells 一致方向，跨两个 benchmark**（6 VWA + 2 WA）；FE inverse-variance pooled | oracle 是**事后**上界，非可达策略；FE 估计量只说这些格 | Ch4 | A |
| **C1b** | 上限里有没有**不受标签供给约束**的那一半？ | `triage_only`（该放弃的任务用最便宜的模式）在 **8/8 cell** 零 SR 损失、省 **9.5–30.6%** 成本。它的标签是二元 solvable，**训练侧供给不受限**（与 C5 的 which-mode 半形成对照）—— 但**它同样不可达**，见 C4 | `router_objective_ordering.md` §Across cells | 8/8 一致，含 WA 两格 | ⚠️ **纯 oracle**：零 SR 损失是 by construction（报告原文 "zero SR change by construction"），不是发现。⚠️ 且该标签**本身带噪**——C3 测得同模式重跑 discordance 14.3%，所以"零损失"部分是事后运气 | Ch4 → 立即交棒 Ch6 | A |
| **C2** | 这个上限有结构基础，还是只是随机涨落？ | 三个 phantom arm **各自独解**一批任务（drop-one oracle **1.7–3.3pp**），且 format / prompt-style 双轴独立 | `meta_phantom_lift.csv` · drop-one oracle 表 | 22/24 arm 观测 drop-one 为正 | ⚠️ **H3 过门是弱证据**（§397.8）：门测 ≠0，但同策略跑两次本来就 ≠0 | Ch4 | A |
| **C3** | 这个结构比"什么都不做重跑一次"更大吗？ | **不**——结构量级落在同模式重跑的噪声地板内 | `compare_cross_run_same_condition.py`：discordance **14.3%**（32/224），Cohen κ **0.614** | self-oracle drop-one A→B 6.7pp / B→A 7.6pp | ⚠️ self-oracle drop 只能当 **instability diagnostic**，**不是 bias estimate**（§293）；净差 +0.9pp 不是噪声地板本身 | Ch4 末 / Ch6 | B |
| **C4** | 那能不能学一个便宜的预测器？ | **不能**：真嵌套 CV 下 **0/6 cell** 的 learned triage 能 Pareto 胜过平凡的 always-cheapest | `router_triage_learnability.md`（§392.2 真嵌套版） | label-shuffle 置换零分布（B=200）+ always-cheapest 固定策略，**两道控制缺一不可** | 权衡点而非压制点：cls·B0 SR +1.79pp 但 cost +11%；red·B2 +1.97pp 但 cost +2.4%。⚠️ **仅 VWA 6 格**——见下方缺口表 | Ch6 | C |
| **C5** | 为什么学不到——假设类、标签定义、还是别的？ | 瓶颈是标签的**产生率**：标签只在任务被解开时诞生，base SR 2–27% 时无法靠重新切分制造事件 | `router_label_supply_diagnosis.md`：**4/6 cell 无可训练分类器**（可训练标签 15–97 个） | 三条独立路径全堵：连续标签（VWA score 纯二值）／池化（矛盾率 cls 57.4% / red 56.0%）／重标注（cost-tier 是唯一有收益的，但不制造新 solve 事件） | 结论限定在观察到的 SR 体制内，不外推到强模型 | Ch6 / Ch7 | C |

**链条完整性检查**：C1 不成立 ⇒ 全文无题；C2 不成立 ⇒ C1 可能是噪声；
C3 是**焊接枢纽**（同时关掉 C2 的正面读法、并证明 C4 不是估计器的锅）；
C4 无 C5 ⇒ 只是"我们没做出来"；C5 无 C4 ⇒ 只是理论担忧。

---

## ⚠️ 已作废的说法 — 写作时**不得**引用

| 作废内容 | 出处 | 现在该怎么说 |
|---|---|---|
| "AUROC 0.65–0.72 in 5/6 cells"（暗示判别力够） | §394 **RETRACTED** | 第 6 格 red·B2 是 **0.483（低于随机）**，而它偏偏是**唯一显著的那格** ⇒ 全局判别与尾部可用性在本数据上**解耦** |
| "天花板全在 cost 维度 / oracle SR 逐格等于最强单模" | §396.2 **RETRACTED** | 稿件当时错引了 `triage_only` 列；真 oracle（`oracle_sr_cost`）SR 增益 **+3.45~16.07pp** |
| "~1/4 标签由 MODES 硬编码顺序 tie-break 决定" | §387.16 E 条 **推翻** | `true_tie` 在 6 个 cell **全为 0**；真实缺陷更严重——**12.5–54.6%** 的标签上 MODES 顺序返回了一个**严格更贵**的成功 mode |
| "确有污染 p=0.024 / drop-one ≤0.45pp"（B-1969） | §442.7 → **§442.8 作废** | O=4 vs E=6.31，plus-one **p=0.24 ⇒ 未识别出因果效应**；1.91% 是**探测下界非发生率** |
| "linear probe 显示 AUROC=1.0 的可分性" | §111.2 判 **trivial** | probe 在 last input token position 永远 trivially 编码 input 差异 ⇒ **linear probe 对该 contrastive setup 是 wrong tool** |
| 机制层 patching 的全部现存结论（含"图像轴位移 0.475/0.390"） | B-1966，正负面**全部不可用** | 24 cell 已按修复重跑；**未重新聚合前不得引用任何机制层数字** |
| "5.13% vs 12.65% / 156 次超时" | §442.8 | 作废数字，见到即替换 |

> 用法：`.venv/bin/python3 scripts/maintenance/known.py <关键词>` — 台账 2186 条，
> `[RETRACTED]` / `SUPERSEDED BY` 标记会直接显示。**动笔前查，不要靠记忆。**

---

## 证据缺口（写作前必须处理）

| 缺口 | 影响的 claim | 现状 | 处理方式 |
|---|---|---|---|
| **同模式重跑的噪声地板只有 vision 一格** | C3 | §302 只做了 B0·cls·vision | 要么把 C3 措辞限定到"在已测的那一格"，要么补跑（A100 排队 + 预算见底 ⇒ 大概率**限定措辞**） |
| **机制层未重新聚合** | 不影响 C1–C5 | 24 cell 已重跑完，聚合脚本已扩展 | 机制层**不进毕设主线**（advisor 2026-05-14 搁置）；若进 appendix 必须先重聚合 |
| ⭐ **WA 只进了 oracle 层，没进 learnability 层** | C4 / C5 的外部效度 | `router_objective_ordering.md` 已含 `wa_reddit·B0/B1`（**8 cells 跨两个 benchmark**）；但 `router_triage_learnability` **stays VWA-only**——它依赖 `extract_50_features.py`，其中 `PHASE1_ROOT`(:55) / `VWA_CONFIG`(:56) / `CELLS`(:65-67) 全是 VWA 常量 | 不花钱不占 GPU，但**不是「改两行路径」**——见下方专条。若打通，C4 从「6 格 VWA」变「8 格跨 benchmark」，直接回答 Guide §14.2 Ch6 第 5 问 + NAACL 攻击面 #2 |
| **B4 / shop / 其余 WA 站未落地** | 会进一步加强 C1/C4 外部效度 | proxy 预算见底，等续额度 | 毕设不依赖它们；若 09-01 前落地则进 Ch6 external validation |
| **rubric #2 系统结构图未画** | 全篇可读性 | 未开工 | Guide §14.3 点名要这张图，且"应该比任何 architecture 细节更早出现" |

---

### ⚔️ 已知攻击（3-AI /stress 2026-08-09）— 写作时必须正面处理，不能绕

这三条来自 Gemini（Mode C）冷读，**当前没有反驳它们的数据**。写进来是因为
指南 §1.6 说得对：不利证据写全是加分项，被 reviewer 发现才是扣分项。

| # | 攻击 | 为什么它有杀伤力 | Defuse 需要什么 | 现状 |
|---|---|---|---|---|
| **A1** | **C4 的"结构性"可能只是欠采样** —— C5 自己说 4/6 cell 只有 15–97 个可训练标签。用 15 个正例在嵌套 CV 里学不出 Pareto 最优，是欠采样的平凡后果，不是"体制的结构性质" | **直击 headline**。它把一个负结果降级成"你数据太少" —— 而这恰好是最便宜的驳回理由 | 标签最多那格的**学习曲线**；或证明**无正则化模型过拟合训练集仍分不开**（训练误差也差 ⇒ 特征无信号；只有 CV 差 ⇒ 只证明数据少）| 未做，09-01 后 |
| **A2** | **drop-one 1.7–3.3pp 可能只是 pass@K** —— C3 的噪声地板 discordance **14.3%** 比它大一个量级。跑同一个最强 arm 两次也会翻掉一些失败 | 若成立，C2 的"双轴独立"直接崩。台账查证：**`pass@` 0 match，项目从没做过这个对照** | 与 `pass@2` 基线对比：若 `Best_Mode × 2 > Best_Mode + Phantom_Arm`，双轴 claim 不成立。**可用现有 archive 的同 condition 重跑数据算，不需新 fire** | 未做，2–3h |
| **A3** | **OOD 论证混淆 visual matching 与 visual grounding** —— 没有 reference image 不等于不需要视觉 grounding；SoM 服务的是当前屏幕的元素定位 | 我原来的措辞（"驱动 SoM 的需求根本不存在"）站不住 | 实证 DOM-only 在 WA 上追平 SoM 而在 VWA 上追不平 | ✅ **措辞已改**（`corpus_eda.md` §2 + 本表 C1b），claim 降级为"任务规格的差异，不必然是模态需求的差异" |

### 专条：把 WA 接进 learnability，代价比它看起来大

初读 `router_objective_ordering.md` 的 caveat 会以为这是路径参数化的活。实际读
`extract_50_features.py`（879 行）后，是三层，只有第一层是机械的：

1. **路径**（机械）— `PHASE1_ROOT` / `VWA_CONFIG` / `CELLS` 三个常量；`--cells` 参数已存在
2. **scored universe**（中等）— 该文件专门处理过「两个 reddit cell 都是 205 且无 scored-universe SHA」
   的历史缺陷（见 :182 / :399 注释）。WA 的 104 要正确接进同一套记账，不能沿用 VWA 常量
3. ⚠️ **特征集会退化**（方法学问题，不是工程问题）—
   20 个特征里，`difficulty` 在 **WA 语料根本不存在**、`has_ref_image` 在 WA **恒为 0**
   （两者都是今晚 benchmark EDA 的实测结论）。

第 3 点的后果要写清楚而不是绕开：**跨 benchmark 的 C4 比较不是同一个分类器**。
两个诚实的处理，二选一，都必须在正文说明：

- **(a) 交集特征集** — 只用两个 benchmark 都有的特征，VWA 侧也重跑一遍 ⇒ 可比，但 VWA 结果变弱
- **(b) 各自最优特征集** — 各自用满，报告为「两次独立的可学性检验」而**非**一次跨 benchmark 比较

**倾向 (a)**：C4 是负结果，用**更弱**的特征集重跑只会让「学不到」更稳；
而若 (a) 下 VWA 反而变好，那本身就是必须报告的发现。

## 与 Guide 硬规则的对照

- **T6 证据可溯源** ✅ 每行都有产物文件名
- **T9 三角验证** ✅ 每个 claim 至少一列独立证据；C4 的两道控制是本文最强的一处
- **T10 claim 校准** ✅ C3/C4/C5 全部是负向或限定表述
- **T11 负结果是证据** ✅ 主论证链的后两步都是负结果
- **T15 viva test** — 逐行过 Guide §22.5 六问，**尚未做**，定稿前必须补

---

→ 下三个文件：`TERMS.md` · `FIGURE_PLAN.md` · `CHAPTER_CHAIN.md`
