# FIGURE_PLAN

> 起手文件 4/5（Guide §26 + §7）。每张图 = **question + takeaway + main/appendix**。
>
> **本项目加了两列，指南里没有**：`数据源 + mtime` 和 `图注雷区`。
> 理由和 CLAIM_EVIDENCE_MATRIX 加两列的理由是同一个——这个项目的写作风险不是"图没数据"，
> 而是**图的数据是那个已经被推翻的版本**。`docs/analysis/` 有 200+ 个产物、`_with_wa` 后缀无索引，
> 光看图名分不出新旧。所以每张图必须钉住产物文件名，且**用 mtime 判新鲜度**。

**建立日期**：2026-08-10 · **状态**：Stage A（图与论证的绑定）已定；Stage B（页数预算）待 handbook
**对齐**：[THESIS_ONE_SENTENCE.md](THESIS_ONE_SENTENCE.md) · [CLAIM_EVIDENCE_MATRIX.md](CLAIM_EVIDENCE_MATRIX.md) · [CHAPTER_CHAIN.md](CHAPTER_CHAIN.md) · [TERMS.md](TERMS.md)

---

## 0. 三条硬规则

**R1 — 每张图过 Guide §7.2 四检查**，第 8 节有登记表：
① Question：它回答哪个问题？ ② Caption：只看图注能否知道图是什么、数据是什么、设置是什么？
③ Text：正文有没有告诉读者看哪一部分？ ④ Claim：删掉它，哪个 claim 会失去证据？
**①③④ 都答不上 = decorative，删。**

**R2 — Figure-first drafting**（Guide §7.3）。先做核心图再写 Results。
> 如果没法把 evidence 画清楚，通常也还没把 claim 想清楚。

**R3 — 图注不得引用作废数字，也不得省略 scope。**
动笔写任何图注前跑一次 `.venv/bin/python3 scripts/maintenance/known.py <关键词>`。
**"没有 scope 的数字"和"作废的数字"一样危险**——见第 7 节，本文件建立当天就抓到一例。

---

## 1. Blocker 与两层写法

**Stage A（本文件现在的内容）**：图 ↔ 章问题 ↔ claim ↔ 数据源的绑定。不依赖 handbook。
**Stage B（待填）**：图数上限、主文/appendix 切分线、是否计入页数、跨栏尺寸与最小字号。

⛔ **依赖 2025/26 COMP0191 handbook**（`CHAPTER_CHAIN.md:239` 明写 FIGURE_PLAN deferred until
handbook limits are known）。**user 2026-08-10 已走 AskUCL 官方渠道询问**。

### ⭐ P1 搜索已回：blocker 的性质变了 —— **公开资料不可能有，别再搜**

`search_results/UCL_AISD_COMP0191_Dissertation_Rules_2025-26.md`（2026-08-10）9 项里 **8 项
"公开资料未见"**，证据等级全 C（官方但间接）。两个有价值的确定结论：

1. ✅ **模块代码**：`COMP0191` = *MSc Artificial Intelligence for Sustainable Development
   Project*（60 credit，100% dissertation）；`COMP0190` 只是 **Project Preparation**。
   ⚠️ 该结论标 **D**——因为公开 catalogue 已切到 2026/27，2025/26 版需从 Portico 调 MID。
2. ✅ **元规则**（UCL Academic Manual 2025/26）：**若** assessment 设 word count，
   **则必须**在 module/assessment instructions 里说明 figures / tables / appendices /
   references 是否计入。⇒ **这些数字只可能在 Moodle 或系里**，搜公开网页永远搜不到。

**因此**：Stage B 不再"等搜索"，只等 AskUCL / Moodle。**保守默认继续生效**，且
FIGURE_PLAN 不因此停工——Stage A 与全部图的实现都不依赖它，只有最后的割刀依赖。

**AskUCL 追问清单**（P1 §14 已备好 9 条英文问句，可直接贴）：正文上限是字数还是页数 ·
图表/caption/公式/脚注是否计入 · appendix 是否排除+有无上限 · reference 是否排除 ·
有无强制模板 · PDF 是否须印 word count · GenAI declaration 的措辞与位置 · 引用格式 ·
确认 COMP0191 就是 2025/26 cohort 的 final project 模块。

**在此期间使用的保守默认（拿到 handbook 后必须回校）**：

| 参数 | 保守默认 | 若 handbook 更宽松则 |
|---|---|---|
| 主文图数 | ≤ 16 | 把第 5 节条件性图与部分 appendix 图上提 |
| appendix 图数 | 不限，但每张必须有正文指针（rubric #6 / T12） | 不变 |
| 图是否计入页数 | 假设**计入** | 放宽尺寸，F1/F4 从 1 栏改整页 |
| 单图占版 | ≤ 半页，F0/F1/F3 可整页 | 不变 |

⚠️ 切分线（哪张进主文）目前**按论证必要性排**，不按页数排。Guide §8 的规则可直接套用：
> 如果把某项从主文移走会让 headline claim 难以验证，它就属于正文。

---

## 2. 章号绑定（7 章版）

**2026-08-10 user 决策：disc+concl 合并为 Ch7**（学长 rubric #5，结构级 ⭐）。
⚠️ `CHAPTER_CHAIN.md` 仍是 8 章（Ch7 Discussion + Ch8 Conclusion 分开），**待同步**。
合并不影响本文件图号——原 Ch8 本来不配图（Guide：Conclusion 不引入新证据）。

| 章 | 章的那一个问题（CHAPTER_CHAIN 原文） | 图 |
|---|---|---|
| Ch1 Introduction | Why is selective page representation a research problem worth studying in web agents? | F0 F1 |
| Ch2 Background | What is already known …, and what remains unresolved enough to justify this study? | F2 |
| Ch3 System | How can representation choice and routing be compared **fairly enough that later differences are interpretable**? | F3 F4 F5 |
| Ch4 Representation | Is there **enough heterogeneous and complementary value** across observation modes to make routing scientifically worthwhile? | F6 F7 F8 F9 F10 F10b |
| Ch5 Predictability | Can a router infer when another representation is worth using **from information available at serving time**? | F11 F12 |
| Ch6 End-to-end | Does routing actually beat defensible fixed alternatives **after all relevant overhead is counted**? | F13 (+cond) |
| Ch7 Disc & Concl | What has actually been learned …, and how far can those conclusions travel? | F14 F15 |

---

## 3. 主文图清单

状态图例：🆕 新画 · ♻️ 已有图改注/重跑 · ⛔ 数据待产 · 🟡 条件性

### Ch1 — 让读者在看任何技术细节前先看见问题

#### F0 — One-figure thesis overview 【✅ **已完成 2026-08-10**】

> **产物**：`final_dissertation/figures/fig_f0_thesis_overview.{png,pdf}`
> **脚本**：`scripts/analysis/figures/thesis/fig_f0_thesis_overview.py`（无数据依赖）
> **底部结论带 = 论证链本身**，所以图和 chapter chain 不会漂移。
> 已按 C2 降级同步：第 ① 条自带"一次重跑就买到 2.0–7.6pp"的限定。
> ⚠️ scope 措辞已两次校正：WA **只有 reddit 且只有两个 backbone**，不能写成"两个 benchmark × 三 backbone"。

- **Question**：这篇论文在做什么？（读者只看这一张图就要能答）
- **Takeaway**：同一个任务可以用便宜或昂贵的方式喂给同一个 agent；本文测量这个选择的**上限**、**可预测性**、**可实现性**，答案依次是 **有 / 没有 / 不适用**。
- **必须让读者看出四件事**（Guide §7.1 F0）：输入是什么 · 流程有哪些阶段 · 新东西在哪 · 评估从哪出来。
- **实现形式**（已落地）：左=网页任务；中=6 mode 表征空间；右=agent loop → measured；
  下方绿框=router 决策点（*the decision this thesis is about*）；底部结论带=三步论证链。
- **四检查**：① 全文 ② 需写明"这不是系统性能图，是论证地图" ③ Ch1 结尾整段导览 ④ 删掉不丢 claim，但**丢掉 second marker**（T13）

#### F1 — Motivating example 【✅ **已完成 2026-08-10**】

> **产物**：`final_dissertation/figures/fig_f1_motivating_example.{png,pdf}`
> **脚本**：`scripts/analysis/figures/thesis/fig_f1_motivating_example.py`
> **三列 = 三个 mode 真正送进模型的东西**：DOM 的 AXTree 文本（+"no image is sent"）·
> SoM 的**真实带编号标注截图** + `[SOM_MARKS]` 文本 · Vision 的原始截图。
>
> ⭐ **量化点比图像本身更重要**：`SoM 文本 = DOM 文本 × 1.008`（**只差 0.8%**）
> ⇒ **多花的钱几乎全在那张图上**。这正是 phantom 系列的构造缝，所以 F1 同时铺垫了 Ch3。
> ⚠️ **素材来源三处，图注全标**：标注/原始截图对来自 `周报/weekly-dashboard` assets（1280×660）·
> 文本来自 `mechanistic/_obs_mirror` · bytes+tokens 来自 phase1 step record。
> ⚠️ **同页面是验证过的**：step-000 是 task start URL，dom-run 与 vision-run 的该步截图
> **md5 逐字节相同**（脚本内置校验，不同就拒绝出图）。

- **Question**：昂贵的上下文具体贵在哪、又具体多给了什么？
- **Takeaway**：同一个页面，DOM / SoM / Vision 三路输入的**内容与单价并排**——差价是真实的，而额外信息**不是每步都用得上**。
- **形式**：一行三列。真 screenshot（含 SoM 标注框）+ AXTree 文本截段 + 每列底部标 token 数/单价。
- **数据源**：`results/visualwebarena/phase1/*/artifacts/`（真 artifact，不许示意图）；单价用 §449 实测（shop：dom \$0.1198 / som \$0.0979 / vision \$0.0722 每 episode）
- ⚠️ **Guide Red flag 6 直接点名这张图**：*"Too much jargon before the reader sees a web page → Fix: early worked example."*
- **四检查**：④ 删掉不丢 claim，丢掉**问题的可感知性**——Guide §7.1 F1 的全部作用

### Ch2 — 文献图谱

#### F2 — Literature map 【✅ **已完成 2026-08-10**】

> **产物**：`final_dissertation/figures/fig_f2_literature_map.{png,pdf}`
> **脚本**：`scripts/analysis/figures/thesis/fig_f2_literature_map.py`（从 P2 搜索结果解析簇与标题）
> **按决策变量分簇，不按年份**（Guide §8.1 明确反对时间线）：
> 每簇标"它选的是什么"，中心标"没有一个选这个"——**缺口是推导出来的，不是断言的**。
> ① 观测表征 (8)：选**哪些元素**进 prompt，表征是固定配置 · ② 模型/模态 routing (6)：选**哪个模型**，输入表征不动
> · ③ confidence deferral (6)：选**答不答**，信号通常要先付全额推理 · ④ cost-aware inference (7)：选**花多少算力**，是深度/宽度不是看什么
> ⚠️ 簇下那句概括是**作者综述，不是引文**，脚注已声明。

> **P2 搜索已回并核验完**：`search_results/web_agent_representation_routing_related_work.md`
> 四簇共 **27 个 arXiv ID，逐个过 arXiv API，0 problem**（标题全部相符）。
> 每篇自带"它优化的是什么 / 因此为什么不回答本文问题"两列——正是画出缺口所需的那一维。
> 该文件还附了一节 **"与本文最接近、需要重点防撞的两篇"**（*Read More, Think More:
> Revisiting Observation Reduction for Web Agents* · *Learning-to-Defer with
> Expert-Conditioned Advice*），⚠️ **这两篇必须在 Ch2 正面处理**，不能只列在图上。

- **Question**：已有工作在优化什么，因此**为什么它们不回答本文的问题**？
- **Takeaway**：四簇（web-agent 观测表征 / 模型-模态 routing 与 cascade / confidence-based deferral / cost-aware inference）各自优化的目标都不是"**给定同一个 agent，这一步该喂哪种表征**"，缺口在四簇交叉处。
- **形式**：二维簇图，不是时间线（Guide §8.1 明确反对按年份排）。
- ⚠️ 每篇必须带**一句"它优化的是什么"**——图谱能不能显出缺口，取决于素材本来是否带这一维。
- **四检查**：④ 删掉 ⇒ gap 只剩文字断言（rubric #8 点名要这张）

### Ch3 — 实验合同

#### F3 — Causal comparison boundary 【✅ **已完成 2026-08-10** · rubric #2】

> **产物**：`final_dissertation/figures/fig_f3_comparison_boundary.{png,pdf}`
> **脚本**：`scripts/analysis/figures/thesis/fig_f3_comparison_boundary.py`（无数据依赖）
> 五段 pipeline 全部灰色标 "same …"，**只有 observation construction 一段是橙色 + ★ the only
> varied stage**；虚线边界框标 "everything inside is held identical"。
> ⚠️ 输出箭头从**边界底边**出发而非从 mode 框——mode 框是 varied stage 的取值注解，
> 不是流程节点；从它出发会读成"mode 产生了 outcome"。

- **Question**：凭什么后面测到的差异可以归因到表征，而不是别的东西？
- **Takeaway**：**同一任务 + 同一 agent + 同一动作循环**，只有观测构造这一段被替换；成本与结果在同一边界内计量。
- **实现形式**（已落地，照 `CHAPTER_CHAIN.md:237` 的链条）：
  `same task + same agent` → mode-specific observation construction → same action loop → benchmark outcome + measured cost
- ⚠️ **不是装饰性架构图**。CHAPTER_CHAIN 原文：*"Not a decorative architecture diagram."*
  也**不要**画成 `p79/` 的模块依赖图——那回答的是"代码怎么组织"，不是"为什么可比"。
- **Guide §14.3 说它"应该比任何 architecture 细节更早出现"**。
- **四检查**：④ 删掉 ⇒ **C1–C5 全部**失去内部效度的可视论证

#### F4 — Corpus EDA 【✅ **已完成 2026-08-10** · rubric #10 收尾】

> **产物**：`final_dissertation/figures/fig_f4_corpus_eda.{png,pdf}`
> **脚本**：`scripts/analysis/figures/thesis/fig_f4_corpus_eda.py`（从 `corpus_eda.json` 解析）
> 四 panel = 四个读者本来得盲信的问题：**A** run set vs scored set 双口径（解释 205↔203 / 435↔432 并存）·
> **B** 语料自带哪些标注 · **C** 什么算成功 · **D** 指令多长。
> ⭐ **panel B 是下游最重要的一块**：WA 三站 **ref-image 全 0 且无 difficulty 标注**，
> 而 F11 显示 difficulty 标注恰恰承担了 VWA 上大部分判别力 ⇒ 两图互相印证。
> ⚠️ **措辞已按攻击 A3 降级并写进 panel 标题**：ref image 缺失 = **任务规格的差异**，
> 不自动等于"这些任务不需要视觉 grounding"。
> ⚠️ 两个 corpus 错别字（`mediun` / `hrad`）在脚注点名，**保持不修**并排除出难度统计。

- **Question**：这些结论建立在什么样的语料上，它们之间可比吗？
- **Takeaway**：六语料在难度分布、评测类型、指令长度、**是否自带 reference image** 上系统性不同——WA 零 ref image 是 OOD 的**结构性理由**，不是偶然。
- **数据源**：`docs/analysis/benchmark_eda/corpus_eda.json`（08-09）· 脚本 `scripts/analysis/benchmark_eda.py`
- **必须画进去的三个副产品**（笔记 §445）：run set vs scored set 双口径（解释 205↔203 / 435↔432 并存）· benchmark 自带 2 个错别字 · WA 零 ref image
- ⚠️ **措辞降级已生效**（攻击 A3）：ref image 缺失 = **任务规格的差异，不必然是模态需求的差异**。图注不得写"驱动 SoM 的需求不存在"。
- **风格**：dashboard style（少黑话少字多图）
- **四检查**：④ 删掉 ⇒ 外部效度与 A3 的应答都只剩文字（rubric #10 的尾巴）

#### F5 — Experimental design matrix 【🆕】

- **Question**：一共跑了什么，哪些格是主线、哪些是外部验证？
- **Takeaway**：site × model × 6 mode 的 42 conditions / 6 cells 主线，加 WA 2 cell 外部验证；**空格与条件格一目了然**。
- ⚠️ **术语**：condition = (site, model, mode) 启动单位；cell = (site, model) 统计分层单位。**图注里不得混用**。
- **四检查**：④ 删掉 ⇒ 读者无法判断覆盖度（Guide §7.1 F4：实验多时尤其适合）

### Ch4 — 上限真实存在（+ 它有多脆）

#### F6 — SR per mode 【♻️】

- **Question**：各 mode 的成功率差多少？
- **Takeaway**：没有一个 mode 在所有 (site, model) 上占优。
- **数据源**：`docs/analysis/cross_sites/sr_per_mode.json`（08-10 ✅ 新鲜）· 脚本 `fig0a_sr_per_mode_heatmap.py`

#### F7 — Cost–SR frontier 【♻️】

- **Question**：贵的 mode 是否买到了成比例的成功率？
- **Takeaway**：成本与成功率**不共线**，前沿上有多个非支配点——这是路由有意义的前提。
- **数据源**：`cost_per_mode.json` · 脚本 `fig3d_cost_sr_frontier.py`
- ⚠️ 成本口径锁 `total_billed` primary（memory `project_cost_latency_canonical_estimand`），图注写明**计量边界**。

#### F8 — Oracle ceiling + rerun band 【✅ **已完成 2026-08-10**】

> **产物**：`final_dissertation/figures/fig_f8_oracle_ceiling.{png,pdf}`
> **脚本**：`scripts/analysis/figures/thesis/fig_f8_oracle_ceiling.py`（从 md 解析）
> 独立解析出的 rerun band = **2.00–7.59pp**，与 `noise_floor_inventory.md` §3 的
> "2.0–7.6pp" 吻合 → 交叉验证通过。
> ⚠️ **band 没有从柱上减掉**，两者都标了 arm count——inventory 明确写
> *"the 6-mode ceiling gain (five arms added) is NOT comparable to a one-rerun floor"*，
> 做那个减法正是它禁止的。like-for-like 的一臂对照在 F10b。
> 视觉结论：三个 cell 的 **5-arm** 增益就落在 **1-arm** 重跑带内外缘（4.91 / 4.43 / 3.45pp）。

- **Question**：如果**事后**知道该用哪个 mode，能拿到多少？
- **Takeaway**：oracle 相对最强单模 SR **+3.45~16.35pp**、成本低 **13.7–35.3%**（8 格跨两个 benchmark）。
- **数据源**：`results/phantom_paper/phase1_full_prereg_decision.json`（08-10 ✅）· `docs/analysis/cross_sites/router_objective_ordering.md`（08-03）· 脚本 `fig_ceilings.py`（**png 当前不存在，需重跑**）
- ⚠️ **scope 必须进图注**：+16.35pp 来自 `wa_reddit·B0`，VWA-only 口径的上界是 **+16.07pp**（`cls·B0`）。见第 7 节雷区 #1。
- ⚠️ **oracle 是事后上界，不是可达策略**——图注必须自带这句，否则读者会当成结果。
- 🔴 **必须把重跑基线画进这张图**（`noise_floor_inventory.md` §3 ①，2026-08-10）：
  原文要求 *"the headline needs the rerun baseline printed next to it"* —— **一次重跑就买到 2.0–7.6pp**。
  也就是 16.07pp 的天花板里，有 2–7.6pp 是"把同一个 mode 再跑一遍"也能拿到的。
  **形式**：在 oracle 柱旁加一条 rerun-band 参考带（阴影区间）。
  **这是全文唯一还站着的正面主张，这条限定省掉就是过度主张。**
- **四检查**：④ 删掉 ⇒ **C1 塌，全文无题**

#### F9 — Complementarity（Venn + drop-one forest）【♻️】

- **Question**：这个上限有结构，还是随机涨落？
- 🔴 **Takeaway 已降级（2026-08-10，`noise_floor_inventory.md` §3 ②）**：可以说
  "三个 phantom arm 各自独解一批任务，drop-one oracle **1.7–3.3pp**，22/24 arm 为正"，
  但**不能把它当作正面的结构性证据**——pooled 双轴效应 **1.35 / 2.09pp** 低于
  **最宽松的已测重跑地板 2.0pp**。原文判词：***"does not survive as a positive claim"***。
  正确措辞：**"观察到方向一致的互补结构，但其幅度未超过重复采样地板"**。
- ⚠️ 因此本图的角色从"证明结构存在"改为**"展示结构的方向性 + 明确它未过噪声门槛"**，
  必须与 F10 同页或紧邻，否则单看这张图会读成正面结论。
- **数据源**：`results/phantom_paper/meta_phantom_lift.csv`（**mtime 05-17，比其余产物老近三个月 → 定稿前必须核它是否还对应当前 42-condition 口径**）· 脚本 `fig_phantom_structure_venn.py` + `fig_forest_drop_one.py`
- ⚠️ **H3 过门是弱证据**（§397.8）：门测 ≠0，但同策略跑两次本来就 ≠0。图注**不得**把过门当作结构存在的证明——它的对照在 F10。

#### F10 — 重跑 discordance 【✅ **已完成 2026-08-10** · **半栏小图**（user 决策）】

> **产物**：`final_dissertation/figures/fig_f10_rerun_discordance.{png,pdf}`（5.6×2.9 in）
> **脚本**：`scripts/analysis/figures/thesis/fig_f10_rerun_discordance.py`
> （从 `noise_floor_inventory.md` §1 + `phase0b_noise_floor.md` §2 双源解析）
>
> **定位已改**：F10b 承担论证，F10 只回答读者紧接着会问的一句"**这个地板是怎么测的**"
> ⇒ 缩成半栏注脚，不与 F10b 并列。
> **内容**：同一 cell（B0×VWA-cls, n=224）三个独立复现 arm 的 task 级翻面率
> —— Vision **14.29%**（κ 0.614）· SoM **12.95%** · DOM **12.05%**（κ 0.559）。
> ⚠️ κ **只有两对算过**，缺的那对就空着，**绝不跨 arm 借用**。
> ⚠️ WA 那个复现 pooled 5 modes × 10 tasks，**不是自身 baseline arm 的地板** ⇒ 只在脚注提名，不并排画。
> ⚠️ 术语锁：mode 名走 `MODE_LABEL` 映射，`capitalize()` 会得到错的 "Som"。

- **Question**：这个结构比"什么都不做、重跑一次"更大吗？
- **Takeaway**：**不**——结构量级落在同模式重跑的噪声地板内：discordance **14.3%**（32/224）、Cohen κ **0.614**。
- **数据源**：`docs/analysis/cross_sites/label_instability.md`（08-03）· `compare_cross_run_same_condition.py`
- ⚠️ **只有 B0·cls·vision 一格**（§302）。图注与正文都必须限定到"在已测的那一格"，不得推广到 corpus。补跑已判为不划算（A100 排队 + 预算见底）。
- ⚠️ self-oracle drop（A→B 6.7pp / B→A 7.6pp）只能当 **instability diagnostic，不是 bias estimate**（§293）。
- **四检查**：④ 删掉 ⇒ **C3 焊接枢纽消失**：C2 的正面读法无人关掉，C4 的失败也就说不清是不是估计器的锅

#### F10b — 表征 arm vs 重跑 arm（one-arm margin）【✅ **已完成 2026-08-10**】

> **产物**：`final_dissertation/figures/fig_f10b_one_arm_margin.{png,pdf}`
> **脚本**：`scripts/analysis/figures/thesis/fig_f10b_one_arm_margin.py`（从 md 解析）
> 独立算出 **2/2 有地板的 cell 落在带内** + 门槛 **3.82–4.15pp**，与 inventory 吻合。
> ⚠️ 六个无地板的 cell 在**轴标签**里标 "(no rerun floor measured)"，**绝不借用他格的带**。
> ⚠️ 门槛线标注了 "(one cell)"——它只从 `B0·VWA-cls` 的三个复现 arm 导出。

> **2026-08-10 更正**：本图原标"⛔ 数据待产 · 2–3h"。实际上
> `docs/analysis/cross_sites/noise_floor_inventory.md`（08-04）**已经做完了这个对照**，
> 只是没用 "pass@k" 这个词（所以台账 `pass@` 0 match 是**假阴性**）。
> **不需要新分析，只需要把它画出来。**

- **Question**：drop-one 的增益会不会只是"多跑一次"？
- 🔴 **Takeaway（实测，结论对我们不利）**：在 `B0·VWA-cls` 上，
  **+1 个最佳异表征 arm = 7.14pp**，**+1 次同 arm 重跑 = 4.91–7.59pp**
  ⇒ ***"indistinguishable — inside the rerun band"***。
  两个有地板的 cell 都是这个结论：*"Neither cell shows a representation arm worth
  appreciably more than a rerun arm; one shows it worth no more at all."*
  ⚠️ 而 7.14pp 已是 **best** 异表征，phantom arm 的贡献只会更小。
- ⭐ **更硬的一条（§1b）**：measured floor 的上边缘 2.23pp 与其自身标准差 2.32–2.53pp 同量级
  ⇒ ***"clears the band is not clears the noise"***：一个效应要达到 **3.82–4.15pp**
  才不太可能由单次重跑产生。**drop-one 的 1.7–3.3pp 落在这条线以下。**
- **形式**：one-arm-margin 对照条形/区间图——每个有地板的 cell 一组，
  异表征 arm 增益 vs 重跑 arm 增益带，叠上 3.82–4.15pp 的"超噪声门槛"线。
- **数据源**：`docs/analysis/cross_sites/noise_floor_inventory.{md,json}`（08-04）·
  脚本 `scripts/analysis/aggregate_noise_floor_inventory.py`
- ⚠️ **口径纪律**：这是 **1-arm baseline +1 arm**。C2 的 drop-one 是 **6-arm oracle −1 arm**。
  inventory 明确拒绝跨口径比较（*"the 6-mode ceiling gain is NOT comparable to a
  one-rerun floor"*）。6-arm 口径的对照**做不了**——只有 1 个重跑 arm，凑不出六臂。
  图注必须写明这一条，否则等于替读者做了它自己拒绝的那个比较。
- ⚠️ **WA 行不能照读**（inventory 明确警告）：其 rerun 数字把 5 个 mode 的
  pilot-vs-full 池化到 **10 个 registered pilot task** 上（pooled_n=50 但只有 10 个独立 task），
  而旁边的异表征列是 **104** 个 task 上从 `dom` 起算的；且 `dom` 自己的 pilot-vs-full 是
  **0 flips**。⇒ 只能读作"该 cell 的*某些* arm 在重复下移动 2–4pp"，**不是 `dom` 的地板**。
- ⚠️ 台账查证：`pass@` **0 match**，项目从没做过这个对照（攻击 A2）。C3 的地板 14.3% 比 drop-one 大一个量级，**examiner 自己会问**。
- ✅ **方法学锚已到位（P3 搜索 2026-08-10，已核）**：pass@k 的**原始来源是 Kulal et al.
  *SPoC: Search-based Pseudocode to Code*（`arXiv:1906.04908`）**，Chen et al. Codex
  （`arXiv:2107.03374`）沿用并标准化了无偏估计式 —— 引用时**两篇都给**，别只给 Codex。
  异质 vs 同质的 matched-budget 对照锚：Bian & Wang（`DOI:10.3233/HIS-2007-4204`）+
  `arXiv:2502.11027`（⚠️ 真实标题是 **"On the Effect of Sampling Diversity in Scaling LLM
  Inference"**，P3 文件里写的 "Diversified Sampling Improves…" 是错的，见 §10）。
- **四检查**：④ 删掉 ⇒ C2 留着一个最便宜的驳回口

### Ch5 — 学不到（先证明"不是估计器的锅"）

#### F11 — 标签可预测性：**两个特征集并排** 【♻️→🆕 设计已升级 2026-08-10】

- **Question**：serving-time 特征能不能预测该不该升级表征？
- **Takeaway（升级后，比原设计强得多）**：能预测的那部分**主要来自一列部署拿不到的特征**。
  20 特征版 VWA 六格 AUROC **0.615–0.864**，其中 **5/6 格的最强单特征是 `reasoning_difficulty`**
  ——VisualWebArena **task config 自带的人工难度标注**。去掉它（+ `has_ref_image`）的 18 特征匹配集上，
  **6 格里 5 格 AUROC 下降**（red·B1 **0.864 → 0.723**），只剩 0.526–0.723。
- **形式**：左右并排两块 heatmap（20 特征 VWA-only / 18 特征匹配集含 WA），中间画每格 Δ；
  在 20 特征那块上**标出最强单特征是哪一列**——`reasoning_difficulty` 那 5 格要视觉突出。
- **数据源**：`router_triage_learnability.md`（08-03, 20 特征 6 格）+ `router_triage_learnability_with_wa.md`（08-03, 18 特征 8 格）· 现有脚本 `fig0g_routing_auroc_heatmap.py` 只画单块，**需扩成双块**
- ⭐ **这张图同时反击攻击 A1**（"C4 的结构性只是欠采样"）：不是样本少学不到，而是
  **唯一稳定携带信号的那一列，部署侧不存在**。产物原文：*"it is reading the benchmark's own
  statement of how hard the task is, which no deployment has."*
- 🚨 **图注雷区**：不得沿用 "AUROC 0.65–0.72 in 5/6 cells"（§394 RETRACTED）；
  也不得沿用 with_wa 散文里的 "0.651–0.717"——**那个数两个口径都不是**（见雷区 #6）。
- ⚠️ 两块**不可比但必须并排**：18 特征集不是 20 特征表的子集，每格都重拟合过。图注必须写明，否则等于把不可比的列并排放而不声明。
- **四检查**：④ 删掉 ⇒ C5 少掉最便宜的那条机制证据，A1 也没了正面应答

#### F12 — 双控制（permutation null + always-cheapest）【🆕】

- **Question**：把"学到了"这件事按最严格的方式检验，还剩什么？
- **Takeaway**：两道控制**缺一不可**——bundle-permutation 零分布（**B=10000**）说明信号不是巧合；而 **always-cheapest 固定策略**说明即便有信号也不值钱。
- **数据源**：`router_triage_learnability_with_wa.json`（08-03）
- **形式**：每格一行：置换零分布直方图 + 实测点 + always-cheapest 参考线 + Holm 阈值线。
- ✅ **m 的口径已解决（2026-08-10）**：Holm 阈值一直是动态的，8 格族阈值 `0.05/8 = 6.25e-3`
  正确写进了产物；只有散文标签错，已修。本图正常报 **m=8，1/8 reject**（reddit·B2 p=0.0004 vs 0.0063）。详见雷区 #6。
- ⚠️ 置换单位是**整个 task bundle**（y, succ, cost）对 X，不是只 permute `y`——只 permute y
  会让标签与定义它的 outcome 脱钩，且误差不是单向的。图注要写这句，它是本文最强的一处方法学控制。
- ✅ **plus-one 估计量有规范出处（P3，已核）**：Phipson & Smyth, *Permutation P-values Should
  Never Be Zero*（`DOI:10.2202/1544-6115.1585`，SAGMB 2010；`arXiv:1603.05766`）——
  有限次置换下不能用会产生 p=0 的朴素 `k/B`。本项目用的 `(k+1)/(B+1)` 正是它。
  Holm 校正引 Holm 1979（⚠️ 只有 **JSTOR stable 4615733**，crossref 查不到，引用用期刊信息）。
- ✅ **小样本 nested CV 的偏差锚**：*Machine learning algorithm validation with a limited
  sample size*（`DOI:10.1371/journal.pone.0224365`）—— 正确的 nested CV 大体保持近似无偏，
  小 n 主要损害的是**精度**而非无偏性。这条直接支撑对攻击 A1 的回应措辞。
- **四检查**：④ 删掉 ⇒ C4 变成"我们没调好"（Guide §7.1 F8：让 triangulation 变成视觉证据）

### Ch6 — 端到端：主结果

#### F13 — Dominance plane，0/8 【✅ **已完成 2026-08-10**】

> **产物**：`final_dissertation/figures/fig_f13_dominance_plane.{png,pdf}`
> **脚本**：`scripts/analysis/figures/thesis/fig_f13_dominance_plane.py`
> （**数字从 `router_triage_learnability_with_wa.json` 读取，不硬编码**）
> 脚本独立算出的 win-region 计数 = **0/8**，与产物文件 `:124` 的 "0 of 8" 吻合 → 交叉验证通过。
>
> 🔴 **画图时发现的自我限定，已写进图内副标题**：把每格归一到 always-cheapest 后，
> **连事后 `oracle_triage` 也只有 1/8 落进胜区**。因为 always-cheapest 是**成本下界**，
> 任何保住 SR 的策略必然更贵。⇒ C4 的正确表述是"**这个权衡不划算**"，
> 而**不是**"学习器不行"。0/8 仍成立，但性质从"learner 失败"校准为
> "**在该成本口径下不存在划算的可部署点**"。不主动说，examiner 必问。

- **Question**：学到的 router 打得过"永远用最便宜那个"吗？
- **Takeaway**：**0/8 格**的 learned triage 能 Pareto 胜过平凡的 always-cheapest 固定策略。唯一显著的那格（red·B2）多保住 **+2.46pp** SR 但多付 **+1.7%** 成本——**是权衡点，不是压制点**。更极端的权衡在别的格：`wa_reddit·B1` **+6.73pp 换 +41.7% cost**、`cls·B1` **+1.79pp 换 +36.1%**（8 格口径，2026-08-10 逐格重算；旧稿的 "+1.97pp / ~2.4%" 是 6 格值，"cls·B0 +1.79pp/+11%" 是串行错配）。
- **数据源**：`router_triage_learnability_with_wa.md:124`（"0 of 8"）· `router_objective_ordering.md`
- ⚠️ **现有 `fig_router_pareto_plane.png` 是 07-15 的 6 格旧口径，不能直接用**。
- ⚠️ 同时要画 `triage_only` 的**双面性**：8/8 格零 SR 损失、省 9.5–30.6%，但零损失是 **by construction**（报告原文 "zero SR change by construction"），**且它同样不可达**。图注不得让它看起来像个可部署结果。
- 🆕 **形式已定（P4 搜索 2026-08-10，两个 panel）** —— 原设想"每格一条 ladder"被否掉：
  P4 §3.1 明确**不建议 8 个独立 Pareto 小图**，因为读者要看 8 次才能得出"没有一个赢"。

  **Panel A — Baseline-normalized Dominance Plane**（8 格叠在**一张**图上）：
  对每格把 always-cheapest 固定为原点，变换坐标
  `x = log₂(C / C_baseline)` · `y = S − S_baseline`（**用百分点 pp，不用相对百分比**——
  base SR 只有 2% 时 2%→3% 显示成 "+50%" 会把 +1pp 夸大成大效应）。
  于是**所有格的 baseline 都落在 (0,0)**，左上象限 `x ≤ 0, y ≥ 0` 淡底色标
  **"Pareto improvement region / Dominates fixed policy"**。
  ⇒ "0/8" 变成**视觉事实**：没有一个点落进那个区域。
  log-ratio 而非差值，是因为各格 baseline cost 量级不同。

  **Panel B — 8-row Constrained-Gain Certificate Plot**：每格一行的标量证书，
  避免读者自己数点。
- ⭐ **P4 §11 提了一个我没考虑过的统计风险**：**Pareto ordering 本身可能受估计噪声影响**
  —— "0/8" 是**点估计**判定。建议加 **bootstrap dominance probability**（每格 bootstrap
  重采样后落进 win region 的概率）。这同时是对攻击 A1/A2 的正面加固，且与 F10 的噪声地板呼应。
  **未做，列为 F13 的可选加固项**（估 1–2h，纯 archive 计算）。
- ⚠️ **被支配的 routed 配置不许隐藏**（CHAPTER_CHAIN §6.2）。
- **四检查**：④ 删掉 ⇒ **C4 headline 没有图**

### Ch7 — 为什么学不到（这才是贡献）

#### F14 — 标签供给衰减 + 可训练门槛 【✅ **已完成 2026-08-10**】

> **产物**：`final_dissertation/figures/fig_f14_label_supply_attrition.{png,pdf}`
> **脚本**：`scripts/analysis/figures/thesis/fig_f14_label_supply_attrition.py`
> （**从 `router_label_supply_diagnosis.md` 解析，不硬编码；解析不全直接报错拒绝出图**）
>
> **对 P4 模板做了一处诚实偏离**：P4 建议四级（All → Solved → Labels → Trainable），
> 但本项目 **第 2、3 级是同一个集合**——which-mode 标签恰好在任务被解开时存在。
> 这个恒等式**就是机制本身**，所以画成一级并标注，而不是硬凑成两级。
> **门槛也不在标签总数上**：判据是 `N_MIN_CLASS_TRAIN=10` 且需 **≥2 个类别各自够 10 个**，
> 所以 panel B 把阈值放在真正咬住的地方——**可用类别数**。结论：**4/6 格 0 个或 1 个类别过线**。

- **Question**：为什么学不到——假设类不对、标签定义不对，还是别的？
- **Takeaway**：瓶颈是标签的**产生率**。which-mode 标签**只在任务被解开时诞生**，而 base SR 只有 2–27% ⇒ **4/6 格没有可训练的分类器**（可训练标签 15–97 个）。这不是切分方式的问题，重新切分**制造不出事件**。
- **数据源**：`docs/analysis/cross_sites/router_label_supply_diagnosis.md`（07-28）
- 🆕 **形式已定（P4 搜索 2026-08-10）** —— 原设想的漏斗/瀑布**被否掉**：P4 §13 指出
  Sankey/funnel 擅长表达"逐级流失"，但本图的 takeaway 不是"流失很多"，而是
  **"流失之后剩余监督量跌到了可训练阈值以下"** ⇒ 这是 **threshold-crossing problem**，
  不是 flow-composition problem。漏斗会把注意力引到流宽和百分比上，恰好不强调"最终的 n 够不够训"。

  **改用 Thresholded Attrition Connected-Dot Plot**：每格一行，四个 connected dots
  `All tasks → Solved → Labels available → Trainable`，加一条**竖直的 `n_min`
  trainability threshold**，左侧淡灰底标 "insufficient supervision"、右侧 "trainable regime"。
  ⇒ 结论从"漏斗底部很窄"升级为 **"4/6 final-stage points remain left of the trainability
  threshold"**，强得多。**主轴用绝对计数**（P4 §17），不只用百分比。
- ⭐ **P4 §16：把结构性关系直接标在图上** —— `Label Exists ⇒ Task Solved`（没解开的任务
  **定义上**无法产生监督标签）。这个蕴含关系标在 `All tasks → Solved` 那一段，就是 C5 的机制本身，
  而不是一句需要读者自己推的注解。
- **实证锚**：`route_only` 的 which-mode 标签在 cls·B0 只存在于 **97/224** 已解任务上，
  red·B0 只有 **15/203**。⚠️ `n_min` 的取值要自己定并在正文说明理由（P4 未给经验值）。
- ⚠️ 与 `triage_only` 的对照必须画出来：**二元 solvable 标签在训练侧供给不受限**（C1b），受限的是 which-mode 那一半。两个半边的**标签供给不对称**才是机制。
- ⚠️ 结论限定在**观察到的 SR 体制内**，不外推到强模型。
- **四检查**：④ 删掉 ⇒ C5 塌 ⇒ C4 从"体制的边界"退回"我们没做出来"

#### F15 — 三条替代路径全堵 【🆕】

- **Question**：绕过标签稀缺的常规办法为什么都不行？
- **Takeaway**：三条独立路径全堵——连续标签（VWA score 纯二值）· 池化（矛盾率 cls **57.4%** / red **56.0%**）· 重标注（cost-tier 是唯一有收益的，但**不制造新的 solve 事件**）。
- **数据源**：同 F14 文件
- **四检查**：④ 删掉 ⇒ C5 只剩一条路径的证据，"结构性"降级为"没试够"

---

## 4. Appendix 图（每张必须有正文指针 — rubric #6 / T12）

| ID | 内容 | 支撑 | 现状 | 正文指针写在哪 |
|---|---|---|---|---|
| A1 | 逐格失败模式分布 | C1/C5 | ♻️ `fig_failure_modes_per_cell.py` | Ch4 SR 讨论处 |
| A2 | 延迟分解（含 router overhead） | C4 | ♻️ `fig3c_latency_per_step.py` + `latency_decomposition.json` | Ch6 §overhead |
| A3 | token/image 成本构成 | C1 | ♻️ `fig3a` + `fig3b` | Ch3 §成本口径 |
| A4 | energy / CO₂e | — | ♻️ `fig3_regional_carbon.py` + `energy_carbon_audit.json` | Ch3 §可持续性口径 |
| A5 | 逐格 forest（meta pooled） | C1/C2 | ♻️ `fig_meta_forest.py` | Ch4 §pooling |
| A6 | 完整 feature dictionary | C4 | 🆕 表非图 | Ch5 §5.2 |
| A7 | 提示模板 / action schema | 复现 | 🆕 表非图 | Ch3 §3.9 |
| A8 | 任务池 Jaccard / 类别热图 | C2 | ♻️ `fig0d` + `fig0e` | Ch4 §互补性 |
| A9 | confidence calibration | C4 | ♻️ `fig0b` | Ch5 §5.5 |

🚨 **A4 的 T14 红线 —— P5 搜索已回，术语按下表锁死**（`search_results/sustainability_measurement_methodology.md`）：

| 我测到的量 | **只能**叫它 | 不能叫它 |
|---|---|---|
| token 数 | computational demand / computational efficiency | 碳排、能耗 |
| 延迟 | runtime efficiency | — |
| 美元 | economic efficiency | — |
| GPU 遥测能耗 | **GPU-device** operational energy | 系统能耗 |
| GPU 能耗 × 显式电网碳强度 | **GPU-device operational emissions estimate** | total carbon footprint / SCI score / 环境影响评估 |

**正式锚 = ISO/IEC 21031:2024（Software Carbon Intensity, SCI）**：`O = E × I`
（operational emissions = 能耗 × 电网碳强度），完整 SCI = `(O + M) / R`，其中 **M = 嵌入排放**、
R = functional unit。⇒ **token 本身不是碳排项**，它最多是 workload / functional-unit 描述符；
CO₂e 必须同时有能量项和碳强度项。缺 M（以及 PUE、网络、制造阶段）就**不得**自称 SCI 或 total footprint。

**决定**：A4 **留在 appendix**，主文只用 "computational efficiency" 措辞，正文一句指针指过来。
`energy_carbon_audit.json` 里的 CO₂e 数字全部改标 "GPU-device operational emissions estimate"
并显式列出缺失的边界组件。可引用的方法学出处（均已核）：Strubell et al. ACL 2019
（`10.18653/v1/P19-1355`）· Patterson et al.（`arXiv:2104.10350`）· Green Algorithms
（`10.1002/advs.202100707`）· GREENER principles（`10.1038/s43588-023-00461-y`）·
Chasing Carbon（`10.1109/MM.2022.3163226`）· BLOOM footprint（`arXiv:2211.02001`）·
Fernandez et al. ACL 2025（`10.18653/v1/2025.acl-long.1563`）。
GB 电网碳强度用 **NESO Carbon Intensity API**；⚠️ 需声明用的是 average(location-based) 还是 marginal。

🚫 **机制层图一律不进**（`fig_mech_*` / `fig_axis2_*` / `fig_stage4_*`）：
advisor 2026-05-14 搁置 §5，且 **B-1966 使正负面结论全部不可用，24 cell 已重跑但未重新聚合**。
**未重新聚合前引用任何机制层数字都是违规**（CLAIM_EVIDENCE_MATRIX 作废表）。

---

## 5. 条件性图（落地才进 — user 2026-08-10 决策）

主论证链 C1–C5 **不依赖**下面任何一项。落地则加图，不落地则一句话交代，全文不动。

| ID | 内容 | 依赖 | ETA | 不落地时怎么写 |
|---|---|---|---|---|
| C-1 | shop 三站外部验证（B0 vision 补尾 + B1 dom/som/P-SoM） | A100 在跑 | ~08-15 | "shop 未纳入主线（R3 framing）" |
| C-2 | WA shopping / shopping_admin × B0 | 未起，等 B1 跑完 + \$180 额度 | 09-01 前大概率不全 | 只报已落地的 wa_reddit 两格 |

⚠️ **09-01 硬截止 − 17 天**。条件性图的**图位不预留在主文版式里**——落地了再插，避免开天窗。

---

## 6. 实现分派

| 类别 | 图 | 依赖 | 估时 |
|---|---|---|---|
| **手绘（无数据依赖，今天可做）** | F0 F3 F5 | 无 | 各 1–2h |
| **有数据、需新脚本** | F4 F12 F13 F14 F15 | 产物已在 | 各 1–2h |
| **已有图，改注/重跑** | F6 F7 F8 F9 F10 F11 + A1–A5 A8 A9 | 产物已在 | 各 15–40min |
| **需外部素材** | F1（真 artifact + screenshot）· F2（GPT P2 文献） | artifact 已有 / P2 待搜 | F1 2h · F2 2h |
| ~~需先算数据~~ **数据已存在，只需画图** | F10b | `noise_floor_inventory.{md,json}`（08-04） | ~~2–3h~~ **~40min** |

**优先级**：F0 → F3 → F14 → F13 → F10b/F8 → F1 → 其余。
理由：F0/F3 决定读者能不能进门（且零数据依赖，不会被实验进度卡）；F14/F13 是两个 headline
的唯一载体。~~F10b 是唯一有 deadline 风险的分析活~~ —— **已解除**（2026-08-10 发现分析
08-04 就做完了），但它反过来把 **F8 变成了必改项**（oracle 柱必须叠 rerun band）。

---

## 7. 🚨 图注雷区（写图注前必读）

### #1 — oracle 上界的两个口径，都对，但**指代不同的 cell 集**

本文件建立当天（2026-08-10）核出的不一致：

| 文件 | 写法 | 实际指代 |
|---|---|---|
| `CLAIM_EVIDENCE_MATRIX.md` C1 | +3.45~**16.35**pp | **8 格含 WA**（16.35 来自 `wa_reddit·B0`）— 与该行"8 cells 跨两个 benchmark"自洽 ✅ |
| `THESIS_ONE_SENTENCE.md` | +3.45~**16.07**pp | **VWA-only 6 格**（16.07 来自 `cls·B0`）— 但**没标 scope** ⚠️ |

两个数都能在 `router_objective_ordering.md` 里查到（16.07 在 :21-24，16.35 在 :146-154）。
**它们不是打字错误，是口径差**。规则：**任何 oracle 数字出现时必须紧跟 cell 集**。
`THESIS_ONE_SENTENCE.md` 已补 scope 标注（2026-08-10）。

### #2 — AUROC 的叙述已 RETRACTED（§394）

- ❌ 不得写："AUROC 0.65–0.72 in 5/6 cells"（暗示判别力够）
- ✅ 应写：第 6 格 red·B2 是 **0.483（低于随机）**，而它偏偏是**唯一显著的那格** ⇒
  全局判别与尾部可用性在本数据上**解耦**。AUROC 打的是全局排序，省下的钱来自尾部。

### #3 — 其余作废数字（见即替换）

| 作废 | 现在该怎么说 |
|---|---|
| "天花板全在 cost 维度"（§396.2） | 当时错引了 `triage_only` 列；真 oracle 有 SR 增益 |
| "~1/4 标签由 MODES 硬编码 tie-break 决定"（§387.16 E） | `true_tie` 6 格全为 0；真实缺陷是 **12.5–54.6%** 的标签上 MODES 顺序返回了**严格更贵**的成功 mode |
| "确有污染 p=0.024 / drop-one ≤0.45pp"（B-1969） | plus-one **p=0.24 ⇒ 未识别出因果效应**；1.91% 是**探测下界非发生率** |
| "linear probe AUROC=1.0 可分性"（§111.2） | probe 在 last input token position **trivially** 编码 input 差异 ⇒ wrong tool |
| 机制层 patching 全部现存结论（含"0.475/0.390"）（B-1966） | 未重新聚合前**不得引用** |
| "5.13% vs 12.65% / 156 次超时"（§442.8） | 作废 |

### #4 — 主证据在 gitignored 目录

`results/phantom_paper/*` 整个被 `.gitignore:35` 排除，而 **C1 的 `phase1_full_prereg_decision.json`
和 C2 的 `meta_phantom_lift.csv` 都在里面**。后果两条：
① 复现包/OSF 必须显式打包这两个文件，否则"证据可溯源"（T6）在外部只是空指针；
② 它们的新鲜度只能靠 mtime 判——`meta_phantom_lift.csv` **05-17**，比其余产物老近三个月，**定稿前必核**。

### #5 — 术语锁（TERMS.md）在图注里同样生效

- `dom` 是**基于 AXTree** 的条件标签，不是 raw HTML DOM。图注不得写 "the model reads the raw DOM"。
- condition ≠ cell。
- oracle 一律带"retrospective / not deployable"。

### #6 — ✅ 8 格产物的散文硬编码 **已修（2026-08-10 当日）**，但它连带修掉的 5 个数字要用新值

**先说更正**：初判"m 未随 8 格更新 ⇒ 显著性结论没重算过"是**错的**。
Holm 阈值由 `m = len(ps)` **动态**计算（`router_triage_learnability.py:733/:736`），
产物里 `p=0.0004 vs 0.0063` 的 `0.0063` 就是 `0.05/8`。**算术一直是对的，错的只有标签。**
所以不需要重算统计，也不需要在"8 格报 Pareto / 6 格报显著性"之间二选一。

**已做**：散文全部参数化（cell 数、AUROC 区间、存活 cell 的逐项数字、base SR 区间）
+ 新增 `--from-json` 入口（免 40min 置换重跑即可重渲染散文——**这正是该缺陷长期不修的根因**）。
重渲染后**所有表格行逐字一致**，两个口径各自自洽（6 格版写 m=6，8 格版写 m=8）。

**⚠️ 连带修掉的 5 个数字——图注必须用右列**：

| 旧值（作废） | 新值 |
|---|---|
| 存活 cell "AUROC is **0.483**，below chance，best single **0.711**" | **6 格：0.615 / 0.800** · **8 格：0.526 / 0.790**。⚠️ 两个口径都**高于随机**，"below chance" 是错的 |
| "**AUROC 0.651-0.717** in five of six cells" | **6 格：0.615–0.864** · **8 格：0.526–0.758** |
| "**Two** cells yield no SR-lossless saving at all" | **6 格：0** · **8 格：1**。旧值与它自己的 §3 表格矛盾（六格 saving 全非零） |
| "at **2-27%** base SR" | ✅ 6 格确是 2–27%，**8 格是 2–36%**。且 `base SR` = `baseline_policy.sr_pct`，**不是** `solvable_rate_pct`（后者 7–43% / 7–52%）——两个量别混 |
| red·B2 "keeps **1.97pp** SR, pays **~2.4%** cost" | **6 格：+1.97pp / +2.2%** · **8 格：+2.46pp / +1.7%** |

**对图的约束**：
- **F13 用 8 格 `0/8`**，权衡点举例用 8 格值：最贵 `wa_reddit·B1` **+6.73pp / +41.7%**、`cls·B1` **+1.79pp / +36.1%**；最便宜 `red·B2` **+2.46pp / +1.7%**
- **F12 可以正常报 m=8 的 Holm**（1/8 reject，reddit·B2 p=0.0004 vs 0.0063）
- 图注引用 with_wa 散文**现在是安全的**——但引用**旧稿或旧图注**里的上表左列仍然不安全

---

## 8. 四检查登记表（定稿前必须全绿）

| 图 | ①Question | ②Caption | ③Text pointer | ④Claim test | 主/附 |
|---|:-:|:-:|:-:|:-:|:-:|
| F0 | ✅ | ✅ | ☐ 待 Ch1 prose | ✅ | 主 |
| F1 | ✅ | ✅ | ☐ 待 Ch1 prose | ✅ | 主 |
| F2 | ✅ | ✅ | ☐ 待 Ch2 prose | ✅ | 主 |
| F3 | ✅ | ✅ | ☐ 待 Ch3 prose | ✅ | 主 |
| F4 | ✅ | ✅ | ☐ 待 Ch3 prose | ✅ | 主 |
| F5 | ☐ | ☐ | ☐ | ☐ | 主 |
| F6 | ☐ | ☐ | ☐ | ☐ | 主 |
| F7 | ☐ | ☐ | ☐ | ☐ | 主 |
| F8 | ✅ | ✅ | ☐ 待 Ch4 prose | ✅ | 主 |
| F9 | ☐ | ☐ | ☐ | ☐ | 主 |
| F10 | ✅ | ✅ | ☐ 待 Ch4 prose | ✅ | 主(半栏) |
| F10b | ✅ | ✅ | ☐ 待 Ch4 prose | ✅ | 主 |
| F11 | ☐ | ☐ | ☐ | ☐ | 主 |
| F12 | ☐ | ☐ | ☐ | ☐ | 主 |
| F13 | ✅ | ✅ | ☐ 待 Ch6 prose | ✅ | 主 |
| F14 | ✅ | ✅ | ☐ 待 Ch7 prose | ✅ | 主 |
| F15 | ☐ | ☐ | ☐ | ☐ | 主 |
| A1–A9 | ☐ | ☐ | ☐ | ☐ | 附 |

**主文 17 张**（含 F10b），保守默认上限 16 ⇒ **Stage B 拿到 handbook 后必然要动一次刀**。
预先定好割序（从后往前割）：**A8/A9 已在附 → 先割 F5（设计矩阵可退成表）→ 再割 F15（并入 F14 的第四级）→ F7（并入 F6 双轴）**。
**F0 / F3 / F8 / F10 / F13 / F14 不可割**——每一张删掉都会让某个 headline claim 失去唯一载体。

---

## 9. GPT 搜索端结果的落地状态（2026-08-10 全部回收）

产物在 `search_results/`，**核验层单列在 `search_results/VERIFICATION.md`**（原始文件不改）。

| Prompt | 落到哪 | 状态 |
|---|---|---|
| **P1** handbook | §1 Stage B | 🟡 **blocker 未解但性质变了**：公开资料不可能有，只能 Moodle/AskUCL。确认 `COMP0191` 才是 final project 模块。保守默认继续 |
| **P2** 文献图谱 | F2 | ✅ **可开工**：27 个 arXiv ID **0 problem**；附"需重点防撞的两篇" |
| **P3** 方法学锚 | F10b · F12 | ✅ pass@k=**Kulal SPoC + Chen Codex 两篇** · plus-one=**Phipson & Smyth** · 小样本 nested CV=**PLOS ONE 0224365** |
| **P4** 负结果图范式 | F13 · F14 | ✅ **两张主图形式全换**：8 小图→**Dominance Plane**；漏斗→**Thresholded Attrition Dot Plot**。另得一条加固项（bootstrap dominance probability） |
| **P5** T14 措辞 | A4 | ✅ 术语表锁死 + **ISO/IEC 21031:2024 SCI** 作正式锚；A4 **留 appendix** |

**引文核验总账**：33 arXiv + 27 DOI → **实质错误仅 1 处**（`arXiv:2502.11027` 标题，
真实为 *On the Effect of Sampling Diversity in Scaling LLM Inference*）+ 1 处需换引用方式
（Holm 1979 只有 JSTOR ID）。⚠️ 12 处初判 MISMATCH 里 11 处是**核验脚本自己的误报**——
教训见 VERIFICATION.md §4。

---

## 10. Guide 硬规则对照

- **T1 每张图服务一个问题** ✅ 第 3 节每张图第一行就是 question
- **T6 证据可溯源** ✅ 每张图带产物文件名 + mtime；⚠️ 雷区 #4 是唯一缺口
- **T8 Figures are arguments** ✅ 第 8 节 ④ claim test 是硬门；F5/F7/F15 已预判为可割
- **T10 claim 校准** ✅ F8/F10/F13 的图注自带"事后上界 / 只测了一格 / 权衡非压制"
- **T11 负结果是证据** ✅ F13/F14 是主文最重的两张，都是负结果
- **T12 appendix 有指针** ✅ 第 4 节最后一列就是指针位置
- **T14 sustainability 不偷换** ✅ **已定（P5）**：术语表锁死，A4 留 appendix，正式锚 ISO/IEC 21031:2024 SCI（`O=E×I`，缺 M 就不得自称 SCI/total footprint）
- **rubric #2 系统结构图** → F3 · **#8 文献图谱** → F2 · **#9 visualization** → 全表 dashboard style · **#10 EDA 图** → F4

---

→ 五个起手文件到此齐备：[THESIS_ONE_SENTENCE](THESIS_ONE_SENTENCE.md) · [CLAIM_EVIDENCE_MATRIX](CLAIM_EVIDENCE_MATRIX.md) · [TERMS](TERMS.md) · **FIGURE_PLAN**（本文件） · [CHAPTER_CHAIN](CHAPTER_CHAIN.md)
