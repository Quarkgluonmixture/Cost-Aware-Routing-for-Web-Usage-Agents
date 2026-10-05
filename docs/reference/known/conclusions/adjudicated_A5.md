---
type: conclusions
batch: A5
status: done
created: 2026-10-06
source: A5.jsonl（ADJUDICATED，295 条，§398.2 → §527.6，2026-07-28 → 2026-09-15）
---

# 裁定层 A5（§398.2–§527.6，295 条 ADJUDICATED，2026-07-28 → 2026-09-15）

**聚合非转写**：逐条索引见 `ledger.jsonl`。本文件只回答「这件事现在算什么、为什么、被谁推翻过」。

**读法**：这一片从「两篇合并为一篇」（§398.8）开始，到 Showcase 演讲定稿（§527.6）结束，跨七周。
前三周是 **frame 的反复重构**：合并稿四步骨架 → 文本/视觉/融合三分（§407）→ 站点翻转（§413）→ 三家零预设收敛（§463）→ NAACL 的「Route the budget, not the representation」（§505.27）；
贯穿全程的方法论中心是 **同条件重跑地板**（noise floor）：它从「一格的轶事」被逐步补成跨臂、跨站点、跨 serving path 的 band，并长出一整套「对口臂比较 / 发车前注册 / declared power」纪律。
中段是 **交付密集期**：REALM camera-ready（保持 non-archival）、匿名化、毕设成稿与图、Showcase 海报 v2→v9.10、demo 与演讲。
基础设施侧的教训密度极高，且同一形状反复出现：**pattern 自匹配、跨机时区、「退出码 0 但做的不是那件事」、错误信息说得通但不是本次原因**。

> ⚠️ 本片两条硬「推翻链」必须连读：
> **(a)** §402.7「B1/B2 重跑地板不做新实验，因为确定性已充分」→ §406 理由被 RETRACTED 并替换 → §467.1 → §468.5 → §470.7（本地地板只在第一次值得买）；
> **(b)** venue：A4 §383.1（A non-archival + B archival）→ §398.8 合并成一篇 archival（non-archival 被静默删掉）→ §407.9 恢复 non-archival → §502.2 camera-ready 维持 non-archival。

---

## 一、论文主张与 frame

### 1.1 合并稿主张与 frame 的演变链

**当前值**（截至 §505.27，2026-09-09）：NAACL 稿主张 = **「可学的是 task 难度，不可学的是 task×mode 契合，难度支撑的杠杆是预算而非表征」**（候选题 *Route the budget, not the representation*）；**mode 从 router 的决策变量变为结论的覆盖范围（6/6 mode 成立）**；骨架 8 节见 next_steps §0。
已知软肋（原文）：early-stop 先例未核、6/11 无 band、learned−fixed 1.4pp 为 steady-state、全离线。

**演变**：
- **§398.8（07-28）两篇合并为一篇**，主张 = 表征路由的上限真实存在但既不稳定也不可达。理由：*"不是因为两篇都弱 —— Paper B 强（完整自洽的负结果），Paper A 弱（H1 FAIL + H3 双轴低于噪声地板）"*。焊接点 = 新增第 ③ 步：① ceiling 高 → ② 有结构基础 → ③ 但结构小于同模式重跑地板 → ④ 且学不到。
- **§407.1 + §407.12（08-01）框架从「六模式 phantom 分类学」转向「文本 / 视觉 / 融合三分」**。user 三句话推动，第三句是方法论理由：*「企业里根本没有人区分 phantom，真正的工业问题是 vision/text」*。四个无截图变体降级为自变量的构造效度检查；**旧 ②（H3 两轴结构）整个消失，2x2 降级为 §2 构造效度检查**（旧②是承重墙且已塌，并省出超出的 2 页）。
- **§407.22 + §407.24（08-02）两个候选框架并列备选，未选**：候选一（codex）1-2 组织成概念贡献 *an oracle gap is not a routing opportunity*；候选二 3-4 组织成部署期配置决定；5-9 通用。证据层汇总刻意写在选框架之前（让框架对着覆盖度选）。codex 另主张「这里藏着两篇现在只写第一篇」，与四步骨架有张力，留裁。
- **§412（08-02）§6 十条反证目标：八条已闭，无廉价项剩余**；剩三条（shopping 零目录 · SoM 自己的重跑地板 · 继承该地板的融合带）是结构性 limitation，需新数据而非新分析 ⇒ 瓶颈换成 A100 队列。
- **§413（08-02）paper frame 定为「该加哪条通道在两站点间翻转 + 该选择下沉不到单任务」**；措辞说 classifieds vs reddit，不说 workload law。三轮独立（Claude/codex/Gemini 候选互相隐藏）。承重 = claim 4：四个独立泛函 × 三 backbone × 两 benchmark，**4.93-7.39pp 对 0.89-2.23pp 重跑带**。
- **§458.3（08-12）「六表征 harness 能告诉部署、而单 mode 部署学不到的事」定为一节**，六条 + 两条部署属性，全部有数字：① 别建 fusion pipeline（0/8 过 rerun 门槛）② 必须 sequential 重编号（代价 12.5%/20.0% step）③ has_reference_image 当路由特征无效 ④ 表征结论不可跨 cell 迁移（Jaccard 0.690）⑤ 别为延迟换表征（model call 仅 22-67%）⑥ 可按『该不该做』省钱（≤5% 损失下 4/6 cell 省 11.2-47.2%，held-out）+ §458 两条与 SR 正交的部署属性（可诊断性 / token 尾部）。依据 §109.17（artifact-existence ≠ research-characterization）与 §315 五源锁死。
- **§463.2（08-13）三家独立 lineage（codex gpt-5.6-sol xhigh / DeepSeek V4 Pro / Kimi K3）零预设收敛五条**：① `noise_floor_inventory` 是方法论中心 ② `layered_evidence_status` 只作索引永不作被引证据 ③ `energy_carbon_audit` 不承重，砍出主文 ④ **workshop 稿 contribution (ii)「most of the apparent oracle headroom comes from rerun variance」过度声称** ⑤ 六臂 oracle gain 必须带 arm count，`representation_deployment_profile`/`latency` 降附录。第 ④ 条被稿子自己的产物否掉（`noise_floor_inventory`：*'Not licensed: The whole 6-mode ceiling gain is noise. We hold one rerun arm, not five'*）。
- **§470.6（08-19）per-step routing signal（`U_t ⫨ F_t` 分离 + Mechanical/F1-F5 taxonomy）与 2026 Computer-Use 多层 interface stack = 方向存档，非 paper-1 contribution**（落 `paper_planning.md §24`）。user：「不用 reframe，但可以记录」。同时钉死：schema 支撑度（thinking 无 typed 字段；F4 必须用 `agent_visible_changed` 而非裸 `page_changed`，B-09 probe 6/8 违例）、三个障碍、**CU 那批论文数字全部来自对话转述未经 arXiv API 核实**。剂量 = paper-1 零。
- **§491.7（08-30）NAACL 写作纪律五条**落 paper_planning §25：(1) 不写『构造上必然 Pareto 压制』只写 SR 有保证 cost 是实测；(2) 8/8 不当 headline，拆成方向-量级两句；(3) 不写 ceiling 改写 *conditional oracle over the triage-selected pair*；(4) 不把 AUROC 低归因于样本少；(5) 不把『发现基线算错→修好→仍打不过 always-cheapest』心路历程写进正文（gemini：读起来像 post-hoc rationalization）。来源：三家 /stress 收敛，前四条各有代码/数据层证伪。
- **§505.27（09-09）NAACL frame 改写**（见当前值）；user 定的框架『mode 是我们证明覆盖了主流表征』；数据支撑 §505.7 / §505.2 / §505.10 / §505.19 / §505.21–24。

**已作废**：
- §398.8 合并时「一篇 archival」的投稿形态 → 被 §407.9 推翻（见 7.1）。
- 旧四步骨架的 ②（H3 两轴结构）→ §407.1/§407.12 删除。
- workshop 稿 contribution (ii) 措辞 → §463.2 判过度声称（禁引）。

**caveats**：§413 原文 *"两个站点识别不了因果调节变量，三个 backbone 共用同一批任务 ⇒ 确立的是站点交互的模型鲁棒性不是六次独立观测"*。§407.22 的两个候选框架在本批内**未见显式拍板**，后续 frame 是 §413 → §505.27。

**证据**：`docs/checkpoints/实验笔记.md` §398.8 / §407.1 / §407.12 / §407.22 / §407.24 / §412 / §413 / §458.3 / §463.2 / §470.6 / §505.27；`docs/checkpoints/paper_planning.md` §24/§25；`task_naacl2027_main.md`。

**原文片段**：§398.8 *"第 ③ 步同时关掉 A 的正面结果、补上 B『为什么不是估计器问题』的另一半"*；§407.1 *"若实践者不区分 phantom 变体，把它们当独立研究对象就是在回答没人问的问题"*。

### 1.2 观察模式的定义、分组与命名

**当前值**：
- **三侧 canonical 分组（§450.18，锁进 TERMS §1.1）**：**文字侧**（DOM · P-text · P-prompt · P-SoM，只送结构化文本不送图）· **结合侧**（SoM，文本+标注截图）· **视觉侧**（Vision，只送截图）。依据实测：四个文字侧 mode 的 `image_payload_bytes` 全为 0。**不替换既有 2×2**（文字侧内部 = `text format (AXTree vs [SOM_MARKS]) × prompt style`，`mark_count` 30=[SOM_MARKS] / 0=AXTree）⇒ 同一结构两个 zoom level。
- **SoM 归「融合」不归「截图族」（§407.1）**：SoM = 标注截图 + 标号图例 + mark↔id 对应，是唯一的融合模式，且是 5/6 格里最贵的模式。
- **六个 mode 文本侧同源一棵 accessibility tree（§470.1）**：`dom` 不是 raw HTML DOM；`[SOM_MARKS]` 是 AXTree 的严格行子集（只留带 `[N]` 前缀的行 + 拍扁层级）；项目没有 raw HTML / API-tool tier 的 arm。
- **SoM / [SOM_MARKS] 标注 AXTree 中每一个带 identifier 的 node（§478.2）**，含 StaticText / heading / image，不是可交互元素子集；**丢弃的是层级不是元素**。证据：`_extract_text_marks` 判据只有 `^\s*\[(\d+)\]\s+\w`，零 role 过滤；[SOM_MARKS] = AXTree 的 1.008x 字符。
- **axis-1（DOM↔P-text）自 AMENDMENT_07 起是 bundled（§470.1）**：同时切换 text substrate 与 identifier contract，不是纯 format 轴；引用须用 non-collapse 语言，不得表述为 isolation。id-stability 效应量 ~10.5%（§298）与被研究效应同量级。
- **命名（§462.3）**：六个臂称 **end-to-end observation–grounding configurations** 而非 'representations'；五个 router 家族的失败称 **finite stress test** 而非 bound。理由（codex 零预设）：六臂同时改 prompt framing / element identifier / image availability / 有时 grounding-dispatch ⇒ treatment bundle，没有因果隔离。

**演变**：§407.1（三分）→ §450.18（三侧锁进 TERMS）→ §462.3（命名降格为 configurations）→ §470.1（AXTree 同源 + axis-1 bundled）→ §478.2（SoM 标注范围订正）。

**已作废**：
- §294 ④「axis 1 = 纯拍扁，id 都 nodeId，不 confound」——当时为真，§295 改 sequential 落码（`3a79196`）后即变；被 §470.1 判定失效（跨批，见末节）。
- 「SoM 标注的是可交互元素子集」的旧说法 → §478.2 推翻（commit e486e8e 改毕设 tex）。

**caveats**：§450.18 *"phantom 三臂 drop-one 小（0.00–2.68pp）是因为同处一侧高度重叠，大的独有贡献来自跨侧（SoM 4.46 / Vision 4.02）"*。`TERMS.md §2.1` 的 critical lock 只做防御性措辞约束，未把 AXTree 同源当前提使用。

**证据**：笔记 §407.1 / §450.18 / §462.3 / §470.1 / §478.2；`TERMS.md §1.1`；`configs/exp_v2_base.yaml:22`；`p79/experiment/som.py`；FIGURE_PLAN F9。

**原文片段**：§407.1 *"塞进「截图族」正好把工业界真在用、也最贵的那个藏掉"*。

### 1.3 per-mode 描述指标的解释纪律

**当前值**：per-mode 指标表**三分类**：empirical / ◆ architecturally-downstream / ⚙️ by-construction；◆ 与 ⚙️ 均不得作行为发现引用（§401.2）。**修正后 6/6 一致签名的经验发现由 3 降为 0**——本数据上没有任何一条 per-mode 签名是不可从设计预测的。

**演变**：
- §400.2：二分（经验 vs by construction）。7 个 6/6 一致签名里 4 个是架构同义反复（Vision 零 element id ⇒ locator fallback 低；无 AXTree 文本 ⇒ token/cost 低）。
- §401.2：Gemini cross-AI 指出 scroll_frac / action_fail_rate / no_change_rate 是坐标寻址的机械级联下游（点不准→页面不变→滚动重定位）⇒ 三分。
- §407.6：**四维剖面一致度不设门槛，报 x/6 连续值**。18 指标 x 2 端 = 36 检验期望假阳：6/6 = 0.005 · 5/6 = 0.144 · 4/6 = 1.880 —— 5/6 站得住而 6/6 严了 30 倍；6/6 门槛是 K-of-N 那个错误换地方重犯（2026-05-13 已裁定 K-of-N 降为 transparency-only）。
- §407.11：三分表的「最强文本」是 4 臂取 max 而 vision/som 各 1 臂 —— 主动写进稿子（逐格查后「融合优势」一列一个数没变，但须披露）。
- §407.21：**fusion_premium 用先验固定的两两对照，不用 max(最强文本, vision)**。原估计量两个含噪量取 max 抬高对照 ⇒ 压低融合优势 ⇒ 偏向我们自己的主张；winner's curse，与「相除前先数臂」同科。
- §408.5：diamond 两路径一致性在 mean_diff 上是**代数恒等式**，不作为 text x prompt 交互检验报告。45/48 成立，3 个 miss 全是 B0/reddit P-SoM 臂的 201 vs 203；列名 `additive?` → `same base set?`；Tier 2a 的 consistency 列同理。

**已作废**：§400.2 二分法（被 §401.2 三分取代）；「3 条经验 6/6 签名」（§401.2 降为 0，禁引）；6/6 一致度门槛（§407.6 取消）；fusion_premium 的 max 对照形式（§407.21）。

**caveats**：§400.2 *"不标注则一条设计的重述会被读成发现"*。

**证据**：`docs/analysis/cross_sites/per_mode_four_dimension_profile.md`；`scripts/analysis/per_mode_four_dimension_profile.py ARCH_DOWNSTREAM`；笔记 §400.2 / §401.2 / §407.6 / §407.11 / §407.21 / §408.5。

**原文片段**：§401.2 *"即本数据上没有任何一条 per-mode 签名是不可从设计预测的"*。

### 1.4 cost / latency / 碳 的口径

**当前值**：
- **碳 / 能耗不作为 Pareto 的轴（§407.20）**——理由是**仪器没在测那个东西**，不是冗余：config 写 `a100_pcie_40gb + use_pynvml: true`，但 step record source 是 `psutil_profile`（NVML 静默回落，psutil 读 CPU 不读 GPU）；A100 PCIe 40GB 满载 200-250W，记的是 66W；co2e~latency 的 r=0.9999 是症状不是结论。
- **B1/B2 绝对美元数在稿中读作「token count in different units」，数字不改（§407.20）**：三个 B1/B2 paper-grade run 的 host 全是 a100-jiaming-test，但 `exp_v2_base.yaml:76-79` 的 `input_cost_per_1k=0.00093` 仍是 DGX Spark GB10 推导；格内跨 mode 比值不受影响，受影响的是 Table 1/6 绝对 USD ⇒ §2.4 补写为什么只比比值。
- **latency（§409.2）**：「动机」与「证据」分开——撤回前沿计数，保留轴的独立性主张并换量。user：「cost 确实包含两个轴，不同工业在乎不同的轴」；P0-1 打死的只有旧证据（前沿计数）。
- **可持续性术语锁死（§450.9 T14）**：A4（energy/CO₂e 图）留 appendix，主文只用 'computational efficiency'。映射 token→computational demand · latency→runtime efficiency · 美元→economic efficiency · GPU 遥测→GPU-device operational energy · 能耗×显式电网碳强度→GPU-device operational emissions estimate；正式锚 ISO/IEC 21031:2024（`O = E × I`，`SCI = (O+M)/R`）⇒ token 本身不是碳排项；缺 M/PUE/网络/制造阶段**不得**自称 SCI 或 total carbon footprint。GB 电网须声明 average(location-based) vs marginal。
- **B5 记账价不改（§478.5）**：保持 0.002/0.012，代价（成本列低估 25%）写进 intent 文件，发表前按当时 tier 价从 token 重算。理由：token 是原始观测、价格是后处理参数，中途改价破坏 cross-cell 可比。

**已作废**：「latency 前沿计数」证据（§409.2 撤回，§412 列为「被三次运行推翻的对象」之一）。

**caveats**：Showcase 场合碳排口径另有裁定（§506.2 demo 允许区间估算；§527.4 演讲片子不上 CO₂e），见 9.x；与本节「主文不报碳轴」不冲突但须分场合引用。§463.2 ③ 同向：`energy_carbon_audit` 不承重。

**证据**：笔记 §407.20 / §409.2 / §450.9 / §478.5；FIGURE_PLAN A4；`docs/checkpoints/pre_run/b5_reddit_chain_launch_intent_20260826.md`。

**原文片段**：§407.20 *"碳 / 能耗不作为 Pareto 的轴报告 —— 理由是仪器没在测那个东西，不是它冗余"*。

### 1.5 证据层的地位

**当前值**：证据层是**索引不是数字的第二来源**——Claim→Evidence Index 与 D1-D9 表一律不放数字，每行只指向拥有该数字的产物（§460.2）。**mechanism 层不进 main 稿证据层**（§460.4），例外：『linear probe 对该 contrastive setup 是 wrong tool』本身可报告（NAACL 攻击面 #7 的正面回答）。

**演变**：§450.8（产物散文不许硬编码）→ §460.2（证据层纯指针）→ §463.2 ②（三家独立：`layered_evidence_status` 只作索引永不作被引证据，与 §460 吻合）。

**caveats**：⚠️ **命名陷阱**（§460.4）：`mechanism_per_task.json` 装的是 E1-E4 **行为**指标（click-target divergence / trajectory boundary / confidence calibration / action vocabulary），不是机制证据，而正文 §3 的『22 of 25 live behavioural metrics』引的正是它们 ⇒ 已在证据层 SUPERSEDED 节末加警示。不进的三条理由：advisor 2026-05-14 shelve；§111.2 probe 三 setup 全 AUROC=1.0 被判 wrong tool；§124.10 是单 cell 单 backbone exploratory。

**证据**：笔记 §460.2 / §460.4 / §463.2。

---

## 二、重跑地板（noise floor）—— 本片的方法论中心

### 2.1 地板的地位与「headline 必须并列重跑基线」

**当前值**：
- **① ceiling（C1）是唯一还站着的正面主张，其 headline 必须把重跑基线印在旁边**（§406 → §450.10）。当前措辞：一次同 arm 重跑买到 **2.0–7.6pp**；5-arm 增益 **4.39–16.07pp** 本身 survives；F8 升为必改项（oracle 柱叠 rerun band）。
- **oracle 数字必须紧跟 cell 集（§450.5）**：+16.07pp = cls·B0（VWA 6 格上界），+16.35pp = wa_reddit·B0（8 格含 WA 上界）——**两个都对，是口径差**。
- **「representation gain 落在 rerun band 内」是 best-single-mode 起点的性质（§455.2）**：licensed 句子补第五个 same —— **same base**。从 dom（15–17%）出发换表征值 2.1–3.5× 于重跑，从 som（27.23%）出发 0.58–0.94×；部署默认 DOM，面向部署结论方向相反。
- **F8 版面约束（§450.13）**：「必须并列」与「禁止相减」同时成立时把约束画进版面：柱（5 arms）与带（1 arm）各标 arm count，副标题明写不可相减并指向 F10b；F10b 把「只有 2 格有地板」做成轴标签 *(no rerun floor measured)*；门槛线标 *(one cell)*，因 3.82–4.15pp 只从 B0·VWA-cls 的三个复现 arm 导出。

**演变**：
- §398.2：**fixed-marginal 独立性 null 判定为 mis-specified，不作为 complementarity 的检验**。给定 marginal，更多重叠 ⇒ 更少独解 ⇒ 更小 drop-one；6/6 cell 实测在 null p95 之上；被杀的是参照分布不是 complementarity（22/24 arm 观测 drop-one 仍为正）。⚠️ 推翻 3-AI overlap 裁定的 P0（B-893），散文动之前须跨 AI 复审。
- §398.5：**self_drop 与 H3 轴的比较合法**；工具 caveat 禁的是 self_drop 比 **drop-one**（六臂联合事件），不可扩大化。self_drop `|run1∖run2|/n` 与 H3 轴 `|P-text∖P-SoM|/n` 同一泛函；引 §397.10(3)。
- §406：headline 并列；*"多加一个表征买到的东西与多跑一次同一个表征买到的东西同量级"*，只报 +16.07pp headroom 不报「重跑同一模式买 4.9–7.6pp」会全篇失信。
- §450.10：重跑区间写作 2.0–7.6pp（CLAIM_EVIDENCE_MATRIX C1 / THESIS_ONE_SENTENCE / FIGURE_PLAN F8 同步）。
- §455.2：补 same base。
- §479.5：fig_f8 的 `2.0--7.6pp`→`0.0--7.6pp` 变化**不单独 land**（B1.cls.{vision,som} d=0 为真数据，但正文四处仍写 2.0，单改图会图↔文矛盾）；**该 framing 决定仍待做**（next_steps §0 唯一未决项）。

**已作废**：
- THESIS_ONE_SENTENCE 无 scope 的「+3.45~16.07pp」→ §450.5 补 scope。
- 独立性 null 作为 complementarity 检验（B-893 的 P0）→ §398.2。

**caveats**：
- 噪声数字各带 scope 并列，禁止相减：§413 claim 4 的 **4.93-7.39pp 对 0.89-2.23pp 重跑带**（四泛函 × 三 backbone × 两 benchmark 口径）与 §406/§450.10 的 **4.9–7.6pp / 2.0–7.6pp**（同 arm 重跑口径）不是同一个量。
- §450.5：*"cost 区间 13.7–35.3% 的 scope 尚未逐格核"*（WA 格 -19.6% 落在区间内，未验证）。
- §398.2 ⚠️ *"散文动之前须跨 AI 复审"*。
- §406 写「在唯一两个有地板的 cell 上…重跑同一模式买 4.9–7.6pp」，§450.10 写「一次同 arm 重跑买到 2.0–7.6pp」：两数并存，**台账未说明是 scope 不同还是后者替换前者**；当前稿件口径以 §450.10 为准（CLAIM_EVIDENCE_MATRIX C1 同步），引用 4.9–7.6pp 须带 §406 的「两个有地板的 cell」scope。

**证据**：`docs/analysis/cross_sites/phase0b_noise_floor.md` §2/§3；`docs/analysis/cross_sites/noise_floor_inventory.md` §1/§2/§3；`router_objective_ordering.md` (:21-24 / :146-154)；笔记 §398.2 / §398.5 / §406 / §450.5 / §450.10 / §450.13 / §455.2 / §479.5。

**原文片段**：§450.10 引 noise_floor_inventory §3 *"Noise destroys positive claims. ... ① is the only positive one left — which is why §2's caveat is the whole cost."*

### 2.2 B1/B2「重跑地板不做新实验」的推翻链

**当前值**：本地模型（B1/B2）的地板产能**只在第一次值得买**——它回答『这个 model 确定吗』这个一次性问题，不是『这一格地板是多少』（§470.7）。实测两个独立 B1 mode（含 vision d≈16.6）恒为 0.00%。

**演变（完整链）**：
1. **§402.7（07-29）**：B1/B2 run-to-run 地板不做新实验——理由「确定性已由既有证据充分支持」（§298.2 determinism 133/133 OK + §397.10 配对一致性 A 1.000/B 1.000 + energy.cpu_arch 96/96 x86_64）。
2. **§406（08-01）**：结论保留，**理由被推翻并换掉**（原理由同批 RETRACTED）。新理由：可用的 replicate 已免费存在——WA 10-task pilot 与 WA full-104 是同一 condition（`exp_v2_wa_full_reddit_base.yaml` 只删 `task.task_ids.reddit`）。
3. **§467.1（08-16）**：本地同条件复制是零 API 成本地板产能，**应当默认排满闲置 wall-clock**（B1×cls 10.6–11.0 ep/h ≈21h/格；已排 11 格 cls 5 + red 6）；B2 不排。
4. **§468.5（08-16）**：**「免费」不等于「有用」**——B1 phantom 臂 SR 6.25-7.59%，11 格里 9 格 d<10 不该报；改为 B0 cls 三个 phantom 臂 $48 买 d≈21-26。
5. **§470.7（08-19）**：§467.1 **修正**为「只在第一次值得买」；取舍点是 A100 wall-clock 不是钱。
6. §472.3（08-20）：本地地板出现 3.12%，地板故事从「API 12% vs 本地 0%」变成「API 12% vs 本地 0–3.12%」。

**已作废**：§402.7 的理由（「确定性已充分」）；§467.1「默认排满」（被 §468.5 / §470.7 修正）。

**caveats**：§402.7 的结论本身（不为 B1/B2 地板做新实验）在 §406 保留，但后续 §479.1 / §480.5 / §494.2 / §496.2 仍在跑 B1×reddit replicate——那是为 C1 的可证伪性（2.4），不是为「测 B1 地板」。

**证据**：`docs/analysis/cross_sites/noise_floor_inventory.md` §1；笔记 §402.7 / §406 / §467.1 / §468.5 / §470.7 / §472.3。

### 2.3 地板 band 的比较规则（对口臂 / 对口侧 / 分格报）

**当前值**：
- **重跑噪声阈值一律按对口臂比较（§477.2）**，不拿跨臂 band 下沿当任何具体效应的阈值。六臂 B0·cls one-sided 95% = 3.82/4.15/3.95/**3.52**/3.89/3.82；§4.7 的 3.53pp 是关于 SoM 的，对口阈值 SoM normal **3.95** / exact **4.02**，均 > 3.53；SoM d=29。决定性证据：`ch4:433-435` 原文 *『only one clears the exact cutoff for its own arm』*。
- **reddit 地板不并入共享 band，单列 red_band 与 cls_band 对称（§479.1）**；floor_band 保留为全 pair 汇总（`one_sided_95` 经验证不受 reddit 行影响），但 reddit 效应量必须读 red_band。
- **§480.2 程序化为 side gating**：band 覆盖的 side 与被加臂的 side 不一致时拒绝比较，verdict 列写明原因（reddit 只有文本侧三臂有 replicate）。
- **多格分裂时各格分别报，不合并（§496.2）**：B1×reddit som 1.95% 落 0–3% 带，dom 3.41% 落 3–7.39% gap 带；dom 越界幅度 **0.29pp** < 一个 task = 1/205 = **0.49pp** ⇒ 合并会把亚-task 级跨界藏进平均数。
- **cross-mode oracle 类指标的噪声 band 可用 2^N 赋值包络测（§470.3）**，前提是该 cell 每个臂都有 replicate；`replicate_metric_noise.json` 里 `n_unique_solves` 的 `"excluded": "cross-mode by construction"` 豁免**自 2026-08-18 起失效**。
- **B-1995（§479.3）**：label_instability 的反循环控制在六臂全 replicate 后**结构上不可用**，显式记录 None 而非降级为六臂数字；正解 leave-one-out 属 estimand 变更，不在此做。
- **B0·red·Vision 维持 inventory-only（§500.6）**，不因实测 discordant count = 12 升级；事前 projected d = n×SR×0.59 = 9.3。

**演变**：§470.3（豁免失效）→ §477.2（对口臂）→ §479.1（red_band 单列）→ §479.3（B-1995）→ §480.2（side gating 写成代码）→ §496.2（分格报）→ §500.6（不用实现值回读功效）。

**已作废**：`n_unique_solves` 的 cross-mode 豁免（§470.3）；跨臂 min 3.52 作 SoM 阈值的读法（§477.2）；label_instability 反循环控制的六臂版数字（§479.3，不得升格 headline）。

**caveats**：§470.3 *"技术性豁免会随数据到位而失效，但没人回头重查 —— 26 个指标里唯一被豁免的恰是全篇 hero 指标"*。§479.3 *"difficulty_null 自己的 docstring 写明两定义差约 4x 且 neither may be quoted alone"*。§452.6：codex 的精确二项阈值算错（4.91 vs 真值 4.02）。

**证据**：`scripts/analysis/aggregate_noise_floor_inventory.py`（red_band / CLEAN_PAIRS 注释）；`scripts/analysis/export_ablation_tables.py t_floor`；`docs/reference/master_bug_catalog.md B-1995`；笔记 §470.3 / §477.2 / §479.1 / §479.3 / §480.2 / §496.2 / §500.6。

**原文片段**：§477.2 *"于是『给一条毫不相干的臂补 replicate，SoM 的结论就变弱』"*。

### 2.4 replicate 注册与 declared power

**当前值**：
- **same-condition replicate 的注册意图必须发车前落盘，落地后照单全收**；不注册须在同一 commit 写明理由；各格 power 提前 declare（§469.7）。依据 codex Mode B F2：事后注册 = 事后选择。
- **注册必须编辑 `CLEAN_PAIRS` 字面量，不得改读外部注册表（§471.8）**：`registered_replicate_run_ids()` 用 `ast.literal_eval` 读源码；写成 `[...] + load_registry()` 会 ValueError ⇒ validator 返回空 frozenset ⇒ 所有已注册 replicate 同时变 ghost（§469.5 的灾难）。
- **自动化注册写完必须验五条，任一不过即回滚（§471.8）**：① 文件仍 parse ② 仍 literal_eval ③ **validator 返回集合确实含新 run id** ④ 两臂目录存在且 summary 完整 ⑤ pair 数恰好 +1。实测：已注册 no-op / 不完整 arm 拒绝（R819 147-of-203 被拦）/ 真插入 8→9 且 validator 6→7 run id。
- **CLEAN_PAIRS 增至 7 对（§470.5）**：B0 cls 三 phantom 臂（ptext / pprompt / psom），d=20.7/26.1/20.7；aggregator discordance 10.27/12.50/12.05% 与 §470.3 独立脚本逐格吻合。新增段无 f-string（PEP 701 在 A100 py3.10 下 SyntaxError → 所有 replicate 一起变 ghost）。
- **B0 reddit 三对排除站点漂移，归 model nondeterminism，可登记（§479.1）**：start_url_mismatch=0，205 个 step-0 landing 全一致；self_drop 不对称方向不同向（P-text/P-prompt 偏 archive，P-SoM 偏 replicate）。闭合 §478.4 的 CLAIM_UNVERIFIED。
- **B1×reddit 落地报 COMPLETE ghost + FAIL-CLOSED 判为预期行为（§494.2）**：`validate_fire_manifest.py:227` 只放过已登记的第二个完整 run；处置 = CLEAN_PAIRS 留占位注释 + next_steps §0 写明，两格验到 205 后再登记。同序列 08-21/23/24 已发生过（§487.7）。

**caveats**：§496.4 *"落地后按 §469.7 照单全收，但 CLEAN_PAIRS 登记留给 user"*。

**证据**：`scripts/analysis/aggregate_noise_floor_inventory.py CLEAN_PAIRS`；`register_replicate_pair.py` docstring；`floor_chain_launch_intent_20260817.md`；笔记 §469.7 / §470.5 / §471.8 / §479.1 / §494.2。

**原文片段**：§469.7 *"地板数字不好看的那格可以悄悄留在未注册状态、重新叫作污染，而最终产物里看不出这件事发生过"*。

### 2.5 地板覆盖扩展：买哪格

**当前值**：截至 §496.4（09-02）已发车 **B0·reddit 的 som/dom/vision replicate**（run_id `B0_som_reddit_20260902_194848_784669986_474818_R11761`，意图书 commit `42cf086`）。三个 reading：① 2^6 unique-solve envelope 的第二个站点 ② §478.4 悬案（7.39% 是真地板还是 6月↔8月站点漂移）③ C1 本身（新地板 <3.41% 则两组重叠 ⇒ C1 死）。vision d≈9.3 声明 inventory only；som/dom d≈17.5 可报 CI；被截断时先砍 vision；预算 ~$67 / 余额 $106.43 / floor $30 halt。

**演变**：
- §464.3（08-13）：chain 第 3 段由 `WA-shop B0 dom`（$17/0.7天）换为 **`B1 som classifieds` same-condition replicate**（$0/约1天）。三家零预设无一把 winner-reversal 留作主线、共同把 rerun floor 认作中心；DeepSeek 最强反对 = 『floor 只在一个 full cell = two-cell anecdote』。选 B1×cls 不选 B2（B2 每臂 1–8 successes）。⚠️ `export FORCE_NEW=1` 承重（B-1916）。**元教训：实验计划跟着 frame 走**。
- §467.1 → §468.5 → §468.10（08-16）：08-21 前只跑 **B0 × cls × 3 phantom 臂（$48）**；shop/WA-shop、B0-red、B4 全部等审稿意见不预支。实测排期 08-17 09:38 → 08-18 12:47（1.1 天）；B0 red 三臂实测 4.6 ep/h ⇒ 5.5 天（$66）冲过 08-21 否掉；空档用免费 B1 cls（vision d≈16.6 排第一，其余 d≈8-10 描述性）。**08-21 那天 A100 要空着能响应意见**。
- §472.3（08-20）：**维持取消地板 chain 格 6-8**，reframe chain 优先占 cls 站点（user 裁定）。AMENDMENT 的 Reason 2/3 不依赖被推翻的 Reason 1；3.12% 反而让 reframe 的 cell A2（B5 replicate）更值钱。格 6-8 是 B1、随时可补，但占 cls 站点 ~2.7 天。
- §480.5（08-26）：19 天 GPU 窗口内最划算的下一笔 = **B1 × reddit replicate**（唯一能让 C1 对照组从单站点变跨站点的缺口，方向无关）→ 同节**价值重述为「能推翻 C1 的最便宜实验」**，不再表述为「买精度」：实测功效全部 <10（最高 som d=9.9），只能是 inventory；但 C1 断言的是分组性质，0-3% 点是支持、7%+ 点是反证，可证伪性不依赖区间。
- §496.4（09-02）：B0·reddit 补 som/dom/vision（见当前值）。

**已作废**：§467.1「默认排满」；§480.5 前半句「以精度为由推荐」（同节改为以可证伪性为由）。

**证据**：`docs/checkpoints/pre_run/b0_reddit_replicate_chain_launch_intent_20260902.md`；`docs/checkpoints/paper_drafts/naacl_evidence_delta.md §4`；笔记 §464.3 / §467.1 / §468.5 / §468.10 / §472.3 / §480.5 / §496.4。

### 2.6 C1 =「地板按 serving path 分组」

**当前值**：C1 的表述固定为『地板按 serving path 分组』这一**观察层**主张，不升级为因果或机制主张；唯一能定论的实验（同一 checkpoint 两种服务方式）**点名而非执行**（§480.1）。

**理由**：§302.5 已裁定机制不可归因（无 expert-route log / batch id / instance id / model SHA）；scale 与 serving path 共变，拆它需要 235B 自托管或 4B 走 API，均在算力包络外。

**caveats**：§500.5 数据驱动后暴露：*"local 组跨两站只在 inventory 级成立，d>=10 子集仍只有 classifieds"*。§496.4 reading ③：新地板 <3.41% 则两组重叠 ⇒ C1 死。§480.5：B1×reddit 是能推翻 C1 的最便宜实验。

**证据**：`scripts/analysis/serving_mode_floor.py` docstring / `_coverage_gaps()`；`naacl_evidence_delta.md`；笔记 §480.1 / §480.5 / §500.5。

---

## 三、Router 结论与评估纪律

**当前值**：
- **router 按决策时机分三类且各自信息边界已测（§505.15）**：事前（task 级，§457 有信号）/ 在线 per-step（前缀对后缀无信息，§459 + §505.2）/ 事后（自身 confidence 判失败部分可行，重跑 cascade +0.8~1.0pp over random）；四种时机都收敛到 +1~2pp over random，共同缺的是结果里 66–88% 的实现级随机性。
- **circularity 表述为两种可区分失效模式（§459.3）**：(a) 标签不存在，(b) 标签存在但背后无信号。四个问题里只有 **pre-flight abstention** 两样都有：which-mode label 饿死 4/6 cell；retry-vs-switch contested 仅 1.04× which-mode；early-abort AUROC≈null；abstention AUROC 0.615-0.864 ⇒ 唯一成立。
- **cell 合并边界（§505.19）**：弃权/难度预测可跨 backbone（与跨 site）合并训练，一个 site 一个难度先验；which-mode / cheap-suffices 不可跨 backbone 合并；之前 per-cell 路由阴性结论不因合并而翻；跨族（B2）不与任何 backbone 共享结构。实测：难度 r 0.5–0.7 且迁移 AUROC 11/11 ≥ within；交互 r 0.0–0.24 且标签迁移 3/4 低于随机；同 backbone 重跑交互天花板 0.28/0.09（Spearman–Brown 4 个 cls backbone → 共享交互信度 ~0.5）。
- **欠采样攻击 A1 对预测器成立、对结论不成立（§453.1）**：更多数据的极限是完美预测器 = oracle triage，它只在 1/8 格进 Pareto 胜区（learned 0/8）；欠采样能解释 0.73→1.00 的差距，解释不了 oracle 自己也过不去的边界。
- **live Pass-2 router 不进 paper-1（§420，确认既有裁定）**：task_pass2_router superseded（2026-07-16）；可训练标签 6 格间 15–97 个、总计 260 ⇒ **七个 router 产物全部是离线的**。

**评估纪律**：
- **Pareto 三档并列报（§399.3）**：非支配（admissibility）/ 严格支配（superiority）/ 相对六固定 mode 非支配；引用 pass 率必须说明哪一档。EXP_SPEC §2 写『Pareto 支配』而 §3 锁『95% 非支配』（§150b.4/B-1550）——买 +7pp SR 花 +10% cost 算非支配不算支配；首版只跑锁定判据，headline 与假设方向相反。
- **禁止用『两个 arm 都过同一二值阈值』推导『某因子不是原因』（§401.3）**；归因走配对 task-level 对比或 2×2 交互。反例：reddit·B0 which-mode 15.27%@0.10415 vs cost-tier 14.29%@0.10803 两轴都更好却被判『粒度不是原因』；tier 63/14、少数类 18% 时分类器可塌成多数类。
- **confidence cascade 全部数字是 offline splice，写成 limitation（§409.3）**：sequential potential outcome 从未观测过，偏向哪边都可能，不可通过重分析修复。
- **零方差 / 缺值信号在排序前剔除，并对每个操作点报 tie span（§409.3）**。
- **0b 行 confidence AUROC（最高 0.877）不可用于 early-abort 论证（§459.1）**：整 episode 聚合 = 看未来；前 k 步重算后掉到 0.336-0.667。
- **held-out prediction 与 held-out policy 分开报（§465）**；「某损失预算下省 X%」的阈值必须由嵌套内层 CV 选出；旧口径保留在 JSON `_ORACLE_SELECTED` 键下作乐观上界。
- **口径不一致的严重性 ≠ 存在性（§465.1）**：codex 指出 leak 政策不一致；实算 6 个 leaked 全在 reddit、classifieds 零影响，唯一改变 red_B2 的 3 个 task（该 cell 已标不可用）⇒ 需重跑产物为零。若落在 cls_B0 就必须全部重跑。
- **补特征时在调用方 monkey-patch 标签函数，不 fork 特征提取器（§457.4）**：fork 会产生第二套会漂移的特征定义（B-1806/B-1807 收进 `p79.policies.router_features` 正为消除此失效）。
- **预注册 freeze 文件不重写（§510.3）**：`budget_router_prospective_shop_B1_20260909.json` 仍含 434 个 task（含协议排除的 463/465）；universe 限制只加在 `evaluate()` 与未来 `freeze()`，评估集 = freeze ∩ canonical scored。

**已作废**：§399.3 首版「只跑锁定判据」的 headline（方向与假设相反）；§401.3 反例中的『粒度不是原因』结论；0b AUROC 作为 early-abort 依据（§459.1）；abstention 用测试标签挑阈值的操作点（§465，降为 `_ORACLE_SELECTED` 对照）。

**caveats**：§462.3：五个 router 家族的失败是 *finite stress test* 不是 bound。§505.27 软肋：learned−fixed 1.4pp 为 steady-state、全离线。

**证据**：`docs/analysis/cross_sites/router_pooled_tier_learnability.md`；`scripts/analysis/router_pooled_tier_learnability.py attribution()`；`docs/checkpoints/EXP_SPEC_pooled_tier_router.md`；`router_label_supply_diagnosis`；`analyze_confidence_calibration.py`；笔记 §399.3 / §401.3 / §409.3 / §420 / §453.1 / §457.4 / §459.1 / §459.3 / §465 / §465.1 / §505.15 / §505.19 / §510.3。

**原文片段**：§453.1 *"谁报了更好的 router，该问他改进的是预测器还是那条边界"*；§465 *"泄漏发生在『选哪个操作点来报』的那一刻"*。

---

## 四、模型线：B3 / B4 / B5

### 4.1 B3（MiMo-VL）

**当前值**：B3 维持 **MiMo-VL-7B-RL-2508**，不换 Claude / GUI-Owl / GLM-4.1V（§419.1）；**关 thinking 保 4096**，thinking 模型的 max_new_tokens「对等」应对等**动作预算**而非总 token 数（§419.2）。

**演变**：
- §407.20（08-02）：B3 max_new_tokens 16384 → 4096 对齐 B0/B1/B2（`exp_v2_B3_som_classifieds_pilot.yaml:62` 唯一偏离；user 裁定，B3 可以不开思考模式）；全套 1098 passed / 0 failed。
- §419.2（08-03）：b2fd1b5 的理由「只有 B3 偏离」不成立——B0/B1/B2 非 thinking，4096 全给动作；MiMo 的 `<think>` 先吃预算 ⇒ 同一数字给 B3 更少动作预算；它撤销的 46ffa1e 标题写死：「截断会把能干伪装成地板」。结论（4096）保留，前提改为关 thinking。
- §419.1：GPT 重扫 + WebFetch 逐条核：① Claude 撞 §340 硬约束「开源」，且 B0 已买到它想买的信息（classifieds 三格 P-SoM 相对掉幅 42.6%(235B) < 53.1% < 60.0%，越强的模型丢图掉得越少）② GUI-Owl built on Qwen3-VL ⇒ 与 B0/B1 同族 ③ GLM-4.1V 官方卡没有 WA/VWA 数字。重扫结论：无新候选能同时满足非Qwen+bolt-on+40GB+有 agent 证据。

**已作废**：§407.20「4096 因为只有 B3 偏离」的理由（§419.2 推翻，结论不变）。

**caveats**：⚠️ 实现坑（§419.2）：官方要求 `/no_think` 在**整个 user content 最末尾**（其后不能有 image），而 `qwen3vl_agent.py:247` 有 reference image 时把截图追加在文本之后 ⇒ 文本级追加会静默失效。

### 4.2 B4（Claude via proxy）

**当前值**：**B4 = `eu.anthropic.claude-sonnet-5`**（§456.4）；落地是改一行 `model.api_name`；图像通道已验（§456.1）；执行上先跑 cls 6 cond，拿到实测步数再决定 red。

**演变**：
- §444.2（08-09）：B4 = 改一行 config，不需新 backend / 凭证 / 计费路径；选型 `eu.anthropic.claude-sonnet-4-6`，理由「可达且单价 0.001/0.005 = 与 B0 完全同价」（实测 8i/8o $0.000048 vs B0 9i/8o $0.000049）⇒ 12 conditions ≈ $154 而非 sonnet-4-5 的 $578（0.003/0.015 = 3x）。
- §444.5：model 轴扩展瓶颈从「有没有」变成「花不花得起」；**gemma 同族 4B→12B→27B 比再加异族更适合堵 NAACL 攻击面 #3**（12b 0.00015/0.0003，27b 0.0003/0.0006，各 12 cond ≈ $22；同族只变规模 ⇒ 分开「弱模型」与「弱表征」）；`nvidia.nemotron-nano-12b-v2` 比 B0 便宜 ~7x。
- §456.4（08-12）：user + 学长：『AWS 价格不是问题（能申请），但用 sonnet 肯定用 5 —— 同价位取新的』；sonnet-5 与 sonnet-4-6 标价相同（0.003/0.015）。先 cls 是为验掉『Sonnet 步数减半』假设（成本区间 $663 → $332）；余额 $674。

**已作废**：§444.2 的选型 sonnet-4-6（§456.4 推翻为 sonnet-5）。

**caveats**：⚠️ **价格矛盾**：§444.2 记 sonnet-4-6 单价 0.001/0.005，§456.4 记 sonnet-4-6 与 sonnet-5 同为 0.003/0.015。台账未给调和说明（见末节）。§468.10 起 B4 等审稿意见不预支。

### 4.3 B5（第二个 API 模型）

**当前值**：
- **B5 进 run_manifest.yaml 新节 `extension:`（§509.1）**：run_registry 默认不读，`include_extension=True` 才读；失败桶写独立 key `extension_cells`；validate_run_manifest 对 extension 只查节↔grade 与磁盘/episode 数。理由：直接进 `cells:` 会被 45 个脚本读到，等于悄悄改了预注册 cell 集合。
- **多对象问题用加 prompt 指令解决，不用 take-first 截断（§472.9）**；该 prompt 不对称须在 paper §3.5 disclose。理由：B0 是 grammar 约束成恰好一个，take-first 不是同一个动作选择过程，而 B5 存在的意义正是与 B0 比（NAACL 攻击面 #3）。`strict: True` 实测返回空体。⚠️ 为 B5 单独打破 B-451 逐字节 prompt 契约；锚句缺失时 raise 而非静默 no-op。
- **mode 排序不跨 backbone 保持（§492.2）**：『换更强 backbone 会整体抬升』要换成此说法——B0 上 vision 是第二好且最便宜的臂（25.00%, $0.0648），B5 上它同时是最差 SR 和最贵的臂（12.05%, $0.227/ep）；五个带文本 mode 一致抬升 7-10pp 而纯视觉掉 13pp。含义：任何建立在『便宜臂已够用』上的路由论证都依赖 B0 的 vision。
- **B-1997（B5 vision coordinate contract）处置顺序（§508.6）**：① B5 六 condition 补登记（零 GPU）→ ② coordinate_contract 改（B5 = 像素制，B0/B1/B2 不动）先留 witness tag → ③ B5 vision cls 重跑（~$50）→ ④ 复核 §505.10 / §505.18 / §505.19。修与跑是 fire 路径改动，按 pre-fire witness 规则不在收尾 session 做。
- §509.1 `aggregate_failure_modes` P1-8-A 去重改为按行记录、同格先到先得（旧循环让兄弟 mode 被报成同格第二个 run，B5 加入后 1→5 条假警报；改后 cells 与 HEAD 逐字节相同）。
- §509.8：B5 补登记 session 收尾不跑 /stress；B-1998 并进 task_b5_vision_coordinate_contract 第 ② 步（同一次 witness）。

**caveats**：
- 价格：B5 config 记 0.002/0.012（低估 25%，§478.5）；§471.7 记 GPT-5.6 被记成『与 B0 严格同价』实际 sol 是 6 倍。本批条目**未直接写明 B5 的模型名**。
- ⚠️ **B-number 可能重号**：§487.2 的 B-1997 = cls reset 无条件 DELETE；§508.6 的「B-1997 处置顺序」讲 B5 coordinate_contract。§487.5 已记录 DGX/A100 两机各自发号会重号（见末节）。

**证据**：`scripts/maintenance/probe_proxy_model_registry.py`；`_status/tasks/task_b5_vision_coordinate_contract.md`；笔记 §407.20 / §419.1 / §419.2 / §444.2 / §444.5 / §456.4 / §472.9 / §478.5 / §492.2 / §508.6 / §509.1 / §509.8。

**原文片段**：§492.2 *"『baseline 够不够强』的答案不是够/不够，而是换 baseline 之后被路由的对象本身变了"*。

---

## 五、扩展 workload（WA / shopping）与排程

### 5.1 WA 与 shopping 的定位

**当前值**：
- **WA 作为第二个 benchmark stratum 与 VWA 平行报告，不并入 θ_FE 池化（§406）**。纳入理由是设计层的（VWA 以图指定目标 / WA 纯文字，合起来张成任务模态轴；user + 学长 2026-08-01）。不并池：(a) 只有 B1 一个模型 (b) run_manifest.yaml 65 条零 WA (c) prereg §8.8 注册的是 10-task × 5-mode Jaccard，full-104 是 exploratory (d) WA reddit 与 VWA reddit 共用同一 postmill 容器。单列修辞更强。B0 六 mode 落地（~08-06）后可另报 WA 内部池化。
- **合并稿不报 VWA visual task 占比；shopping 不进本篇降 future work（§406）**：visual 占比两套定义（§89 自动 cls 69.2/red 84.3/shop 57.7 vs §95 codex 手动整体 95.3）不可比；shopping 唯一 SR（§81 的 16.52%）来源不明。
- **B1 shop = 435 全集 × 3 mode（dom + som + P-SoM）（§448.1）**：实测 B1 dom shop SR 4/46 = 8.7% ⇒ 200×6 只产 ~30 个可训练标签分 6 类，435×3 产 ~52 个分 3 类。跑 shop 的唯一理由是 §216.1『唯一真 OOD』= router 泛化检验。选 som vs P-SoM（图像轴）而非 user 原提的 dom+som+vision（三个全是 baseline mode）。
- **WA shopping / shopping_admin 的 reset 现在是支持的（§455）**：`_lib_paper_grade_gates.sh:582` 共享 VWA Magento 容器（B-1930）；六个 mode 的 `configs/exp_v2_B{0,1}_*_wa_shopping.yaml` 全部已存在。

**演变**：§447.2（08-09）B1 shop 缩为 **200-task 分层子集 × 保全 6 mode**（六格 435 需 19 天撞毕设截止；task_ids 单点定义在 `exp_v2_B1_shop_strat200_base.yaml`，六格 `task_ids_sha` 完全相同 8c16f67f580d）→ §447.3 中途换 config 后旧 episode 按新 task_ids 分类（41 个：27 在内保留 / 14 在外必须清，否则 214 vs 200 触发 B-1834 abort；清理走 `clear_tasks.py` + dry-run）→ **§448.1 推翻 200×6，改 435×3**（*"保住 menu 的形式丢掉它的用途"*）。

**已作废**：§447.2 的 200×6 方案（§448.1）；§387.3 RETRACTED 条 `replaced_by` 中「WA shopping/shopping_admin 保持不支持」（§455，跨批）。

**caveats**：§447.2 的「单点定义」原则（六份拷贝各自漂移是 WA chain abort 的成因，§405）在 §448.1 后仍有效。

**证据**：`configs/exp_v2_B1_shop_strat200_base.yaml`；`configs/exp_v2_B1_{dom,som,phantom_som}_shopping.yaml`；笔记 §406 / §447.2 / §447.3 / §448.1 / §455。

### 5.2 排程与 fire 顺序

**当前值**（截至 §515.4，2026-09-15）：proxy 余额不足时的**本地免费 chain**（user 选 A）：WA·B1·reddit {dom, P-prompt, P-text, som, P-SoM, vision} → B1·shop·som → B2·red·dom → B2·cls·som → B2·cls·vision，FORCE_NEW=1，09-15 09:29:04 UTC A100 发车（CHAIN_PID 2412996）；额度到账后在 cell 边界插入付费项并先砍 B2；**硬停 2026-10-05 00:00 UTC**。顺序按 power：WA 内 d=10 三臂先，B1 shop som（d=19.5）补「本地组 d≥10 只有 classifieds」缺口，B2（d 2.9-4.7）最后。

**演变**：
- §449.3（08-09）：两条 run 争同一容器时，**先跑哪条看断裂点落在完成度的百分之几**：两顺序总完成时间相同（都 08-23）；先跑 B0 ⇒ B1 dom 在 21% 处断 8.2 天（344 个 episode 在断裂后）；先跑 B1 ⇒ B0 vision 在 86% 处断 5.6 天（61 个）；5.6 倍差。裁定先跑 B1。
- §449.3：比较断裂风险不能只比时长——API 模型跨天有**静默换版本**风险（§444.4），本地固定权重只需 reset ⇒ 加 4.8h 例外先补完 B0 vision 61 个；B1 断裂 5.5h / B0 vision 断裂 24h，总完成时间不变。
- §455.4：**数据在手时，排 chain 之前先跑零成本分析**（retry-vs-switch 分析数秒，误排的 chain 占 A100 数天）。
- §464.3 / §468.10：见 2.5。
- §489.5（08-27）：DEADLINE_UTC 默认值放宽：`_b5_reddit_chain` 2026-09-04 → 2026-10-05；`_reframe_chain` 2026-09-06 → 2026-09-20。原 09-04 会在 b5-reddit 第 1.6 格处 halt（3 格 × 205 ep × 19.5 min ≈ 8.3 天）；真实约束是 NAACL ARR 2026-10-12；毕设 09-05 是人力约束不占 API。**脚本里的默认值不是项目约束**。
- §505.27（09-09）：B1 shopping 落地后 fire 顺序：#0 冻结预算分类器对 B1 shop 的 cap 预测（prospective, $0）→ #1 shop_B0 三臂 replicate（~$135）→ #2 WA red_B0 六臂 replicate（~$50）→ #3 B5 cls som(+vision) replicate（~$50）→ #4 variant D reactive rule router 点火 B0 cls+red（~$36）→ #5 B5 red dom+som（~$80）；**不跑 B2 任何 replicate**；合计 ≈ $350 / ~45 h；窗口 09-13 → 10-12，预算 $546。
- **§515.4 推翻 §505.27『不跑 B2 任何 replicate』**：付费项被 $30.18 余额阻塞，user 明确要求含 B2 replicate 与 WA。

**已作废**：§505.27「不跑 B2 任何 replicate」（§515.4）；两条 chain 的原 DEADLINE_UTC 默认值（§489.5）；§449.3 的纯「先跑 B1」（同节加 4.8h 例外）。

**caveats**：§489.5 正在跑的 Phase C 仍用启动时固化的 09-06（预计 08-30 完成，不受影响）。§455.4 ↔ §468.5：「先问现有数据能不能回答」之后必须接「再问跑完之后这个数字撑不撑得住」。

**证据**：`scripts/queues/_b5_reddit_chain.sh`；next_steps §0；`task_naacl2027_main.md`；笔记 §449 / §455.4 / §489.5 / §505.27 / §515.4。

---

## 六、分析产物的工程纪律

### 6.1 结论与数字不得硬编码

**当前值**：分析产物里**凡是数据能陈述的事实一律计算生成**，只有『某个 cell 根本不存在』这类外部事实留手写文本（§500.5）。产物散文里任何 cell-count / 区间 / 逐格数字不许硬编码，且脚本必须提供免重算的重渲染入口（`router_triage_learnability.py --from-json`，§450.8）。毕设图脚本一律从产物解析/读取数字，解析不全直接报错拒绝出图（§450.12）。

**演变**：
- §408.2：`axis_effect_size` 找不到输入时必须 fail，不许降级成空报告（删 `except Exception: warn(); return {}`；main() 开头 live-directory 检查）。理由：每个 contrast 都 n=0 的报告读起来是一组否定结论。
- §418.1 / §420.5：「结论硬编码在生成器里」是一个**缺陷类**，本轮修 6 处（`aggregate_fusion_premium.py:317` / per_mode 旁注列 6/6 vs 8/8 / NOTES 注册表 / `aggregate_confidence_cascade.py:430` **分母错且事实错**（称 Vision 6/6 格最便宜，wa_B0 上是 DOM）/ `axis_effect_size.py` BH 分母 / **`axis1_microbehavior.py` verdict 硬编码只看两个 site，加了 wa_reddit 后根本没看**）。
- §450.8：该 bug 活了 ~1 周不是粗心——改一句措辞要重跑 ~40min 的 B=10000 置换；同文件 :762-763 证明作者已踩过一次（p=0.005 从 B=200 残留）⇒ 复发型缺陷。验证：重渲染到 scratchpad → `diff` 只比 `^| ` 表格行。
- §450.12：F14 从 `router_label_supply_diagnosis.md` 正则解析 6 格 + `N_MIN_CLASS_TRAIN`；F13 从 `router_triage_learnability_with_wa.json` 读 5 个 policy 点；副作用 = 免费交叉验证（F13 win-region 计数与散文『0 of 8』吻合）。
- §457.5 / §467.2：**跨产物 join 必须核对 cell_id 命名口径，coverage 为零时 fail-loud**：`np.nansum` 对全 NaN 返回 0.0 ⇒『没有成本数据』印成『省了 $0.00』。修法：显式命名映射 + 零覆盖 raise + <90% 覆盖 warning。§467.2 复发（`abstention_learnability.py:78-81` 早有注释仍踩）——*注释警告不等于机制防护*。
- §467.4：**同一文件内部的表↔正文漂移，没有任何自动检查能抓**（B-1971）；唯一防线是正文的数也从产物现算。
- §500.5：`serving_mode_floor.py` 的 coverage_gaps 硬编码 prose 在 B1·red replicate 落地十一天后仍写 'B1 has no reddit replicate'，而表头已更新成 sites=2。

**caveats**：§418.1 ⚠️ *"本轮扫描只覆盖 `n/6` 这一个形状；硬编码的模式名/比值/方向未扫过，而 cascade 那例证明这些能错在事实层"*。

**证据**：笔记 §408.2 / §418.1 / §420.5 / §450.8 / §450.12 / §457.5 / §467.2 / §467.4 / §500.5；`scripts/analysis/serving_mode_floor.py _coverage_gaps()`；`router_triage_learnability.py` docstring。

**原文片段**：§500.5 *"一个陈述若能从数据推出，手写它就等于给它一个独立漂移的机会"*。

### 6.2 universe / 计分口径

**当前值**：
- **『scored』有两个口径（§445.1）**：**run set**（语料 − N/A，load 时排除，`tasks.py::_is_na_task`，§139.8）vs **scored set**（再 − protocol，分析时排除，`PROTOCOL_EXCLUSIONS` AMENDMENT_08-10）。reddit 205 vs 203 / shopping 435 vs 432 两个都对；实证 B1_dom_classifieds 224 ep / B1_dom_reddit 205 / B1_dom_wa_reddit 104。⇒ *episode 数若等于 scored 数反而是出问题了*。
- **axis_effect_size 约束到 canonical scored universe；n 检查比集合不比计数（§408.3）**：reddit 每格读 205 个 step 文件对 203 的计分集，AMENDMENT_08 排除的 58/160 进了每个效应量；P-SoM 臂出假通过（掉 2 个 identity-mismatch 又多 2 个不该计分的，n 恰等于 expected）。
- **分析脚本一律经 `pass1_run_manifest.json` 白名单取 run，禁止裸 glob（§442.8）**：B-1969 污染分析 v1 glob 到 4257-4259 episode（canonical 只有 4032），分子分母同时混了 paper 与 replicate 数据（B-1896 / §367）。
- **universe lint（§510.4 / §510.5）**：7 个未登记脚本五个真修、两个豁免（判据：「这个脚本产不产 task 级的 rate」）；测试 24 failed → 22 failed。**lint 扫描范围不扩到 `results/`**（gitignored，会让 census 随机器变红变绿）；改用流程规则：pilot 脚本产出要进稿的数字，先搬进 `scripts/analysis/` 再产数。
- **§505.26**：3 个 task 不重跑；步级分析排除并披露；需要步级产物时用骨架目录（symlink + 排除该 task）在 canonical 目录之外跑，canonical 目录保持 fail-loud（`strict_identity=True`）；建议把『resume 重跑不 rotate 旧 steps JSONL』作为 runner 数据完整性缺口进 bug catalog（fire 前修，需 witness）。补齐后 red_B0 psom 失败桶 kappa 0.52，shop_B0 三对 verifier 一致 27–36% vs 不一致 16–19%。
- **tripwire 退役（§510.7）**：退役 `test_manifest_has_no_shopping_conditions_yet`，换 `test_resume_filter_keeps_a_duplicate_whose_binding_is_spent`；它自 2026-08-06 起一直红，因为它守的事件真的发生了且 §437 当天已修。*事件发生后仍红着的 tripwire 比没有 tripwire 更坏*。

**caveats**：§408.3 *"per_mode_four_dimension_profile.py:297 的注释早就一字不差地描述了这个坑，但只用来保护它自己那个文件"*。§402.7 scored universe 保持 203（见 6.3）。

**证据**：`scripts/analysis/benchmark_eda.py §0`；`docs/analysis/benchmark_eda/corpus_eda.md`；`scripts/analysis/scan_b1969_contamination.py::load_canonical`；笔记 §408.3 / §442.8 / §445.1 / §505.26 / §510.4 / §510.5 / §510.7。

### 6.3 diag ruleset 与 benchmark 缺陷披露

**当前值**：
- **SUCCESSFUL_NOOP_REPEAT 规则不落地，记为已知盲区（§402.4）**：失败侧 fire 452 次但 success-FP 45/268=16.8%（P34 被否同因 20%）；反例 B2·som task 130 成功且连点 eid=103 六次；四种收窄全卡在 14-16%。
- **P47/P48 落码，ruleset bump `8-reddit-p41p46-b1890fix` → `9-wa-p47p48`，全量重扫 42 condition（§411）**：R1/R3 在 624 episode 上 success-safe（24/0 与 9/0）；R2（17% success 误伤）与 R4（36%）按原裁定不落；VWA 既有规则命中与 v8 全等。
- **reddit sidebar 泄漏归入独立 benchmark bug paper（§402.7，user 决定）**：主 paper 一句披露 + 指针；不扩展 AMENDMENT_08，**scored universe 保持 203**；泄漏仅 6 个，实质只影响 B2·DOM 一格（require_reset 在 reddit 为 no-op ⇒ 订阅跨 episode 累积）。
- **§509.8**：B-1999 / B-2000 / P33 som 收窄 + task 41 截图核实 → 新任务卡 `task_diag_v12_rule_batch`（backlog）；v12 要动 53 份 digest；汇总 §5 #7–#9 未复核提议不落码。

**证据**：笔记 §402.4 / §402.7 / §411 / §509.8；`_status/tasks/task_diag_v12_rule_batch.md`。

---

## 七、投稿与 venue

### 7.1 REALM 的 archival 状态与投稿日期

**当前值**：**REALM #192 camera-ready 保持 Non-archival**（§502.2，user 2026-09-08 裁定）。理由：归档 ⇒ 正式发表 ⇒ 占掉 NAACL ARR 投稿权；**NAACL 2027 main（ARR 10-12）是当前唯一 live 目标**；09-14 后不可逆。

**演变（推翻链）**：
- A4 §383.1（07-22）：Paper A non-archival + Paper B archival。
- §398.8（07-28）：两篇合并为一篇 archival —— §407.9 记载 non-archival *"在 07-28 两篇合并成一篇 archival 时被静默删掉，没有重新讨论过"*。
- **§407.9（08-01）REALM 走 non-archival 轨**：user 前提「workshop 不锁你」错（REALM 是 archival），但结论「先占坑之后冲顶会」对，因有第三条 non-archival 轨（同 8 页 / 同 08-05 截止 / 不进 proceedings / 可 under review elsewhere）；正是 2026-05-14 学长建议 + 2026-07-22 user 拍板过的方案。
- §502.2（09-08）：camera-ready 维持 non-archival。

**REALM notif 日期（§470.9）**：翻转两次（09-07 → 08-21 → 09-07）。更正格式固定为『当前值 + 翻转史 + 哪个写法是 stale』，不能覆盖。08-16 那次单点更正删掉正确值并留下『09-07 已作废』的反向路标。

**declaration form（§502.1 / §502.2）**：两次投稿（REALM #192 + VLM4RWD）合填**一份**（同一 manuscript、同一标题、两个非归档 workshop，拆两张会被读成两项独立成果）；REALM 接受后仍填 **Section 2（未发表）**，判断依据写进 §1 跳过理由与 (o)。通用规则：*表格分类没覆盖你的情况时，选最接近的栏并写明为什么，不要把情况改造成符合表格*。

**VLM4RWD（NeurIPS 系 workshop）**：**一律附 NeurIPS Paper Checklist**，即使 CFP 没点名（§473.8）——`checklist.tex` 写 *『The papers not including the checklist will be desk rejected.』*，`neurips_2026.tex:131` 把 checklist 列为不计正文页数。

**投稿表单字段（§473.7）**：OpenReview 的 abstract / keywords / TL;DR 与 PDF 是两个独立表面；表单 abstract 停在旧版；keywords 一个 workshop topic 词都没有（OpenReview 用 keywords 做 reviewer 匹配）；TL;DR 250 字符硬限。下次投稿纳入 `/stress` scope。

**已作废**：§398.8 的「一篇 archival」；REALM notif 的中间值 08-21（§470.9 记录当前为 09-07）。

**caveats**：⚠️ notif 日期在本批多处出现不同值：§468.10 写「REALM 意见 08-21 到」并据此排期；§470.4 把「REALM notif 09-07」列为 CLAUDE.md 的 stale 项（实为 08-21）；§470.9 同日记录又翻回 09-07；§480.5 引「09-07 意见」。以 §470.9 的翻转史为准（见末节）。

**证据**：笔记 §407.9 / §470.4 / §470.9 / §473.7 / §473.8 / §502.1 / §502.2；`task_realm_paper_b_router_negative`；`deliverables/vlm4rwd/README.md`。

### 7.2 稿件排版与 camera-ready 纪律

**当前值**：
- **paper 目录禁用 `wrapfigure`**；压页只用缩图 / 移图入附录；caption 完整性须逐条比对 PDF 文本（先归一化 en-dash / 右单引号 / ligature / 减号，否则 61 个 caption 报 40+ 假阳性；归一化后 61/61）（§473.1）。
- **为够到 venue 关键词的『主题对位』默认按高风险语义滑移处理（§473.2）**：三处错方向完全一致（把四个 mode 说成三个 / 单点对比说成整套设计性质 / calibration 数字说成 perceptual faithfulness）；该段是唯一没过 REALM 三轮审计的部分。
- **把 Limitations / Threats 移出正文时必须同步处理依赖它们的主张（§473.3）**；3-AI 唯一重合。实例：'most of the apparent oracle headroom is rerun variance'（band 只在两 cell 实测）、'+22.54pp against +0.65pp'（『null on reddit』随图去了 p18）、`red·B2` 例外。三处限定语已写回正文。
- **页数硬约束下，限定与修正写进浮动体 caption，正文只留指针（§503）**：正文版 element-id 修正把 8 页顶到 9 页；放进 table* caption 不影响分页；改为净 -3 字符正文版本。
- **多个 `main*.tex` 时「哪份在审」只认两判据**：① 编译产物 mtime ② 拿外部权威副本（OpenReview 表单 / 已提交 PDF）的 title+abstract 回搜源码；不得靠文件名前缀（`realm_*` 是最强误导源）（§504.1）。
- **`3_noise.tex:3` flip 数字 camera-ready 换成三臂重算值（§504.3）**：**67/224 (29.9%)** flip · contested **67.0%** vs 补集 **5.9%** · enrichment **11.4×**。原文「49 of 224 … 48–52% … against 2.9%」自称三臂但引的是 dom+vision 两臂产物（§464.2 已 RETRACT）。
- **`latexmk` 0-undefined 不构成「图引用正确」的证据（§502.3）**：`fig:f1` 实际指向 `fig_f1_motivating_example.pdf` 而非 `fig_f1_diamond_schematic.pdf`；真实 label `fig:space`。
- **验证命令里的 `||` fallback 只保护它所在那一行（§502.3）**：`git cat-file origin/master:<path>` 无 fallback 报「远端没有这个文件」，远端分支实为 `main`。

**已作废**：`3_noise.tex:3` 的两臂 flip 数字（49 of 224 / 48–52% / 2.9%）—— 被 §464.2 RETRACT、§504.3 替换（禁引）。

**证据**：笔记 §473.1 / §473.2 / §473.3 / §502.3 / §503 / §504.1 / §504.3。

**原文片段**：§502.3 *"验证脚本自身出错时最危险的形态不是报错，而是报一个合理的假警报"*。

### 7.3 匿名化

**当前值**：
- **现有 public repo 不做 in-place 匿名化，走「新建独立匿名 repo」（§475.1）**：PUBLIC + Pages + 1 fork/2 star + 1705 commits 单一署名 + repo 名=论文主题 + portfolio 真名 PDF；决定性否决理由：`prereg-amendment-01..08` + `*-prefire` 共 9 个 witness tag 的价值就是 commit hash，history rewrite 会烧掉 preregistration 可追溯链。
- **`docs/checkpoints/pre_run/osf_deposit_DOI1_*/` 整包排除**，README 写 withheld-for-anonymity（§475.3）：`MANIFEST_SHA256.txt` 对得上一个已发布且署真名的 OSF DOI，原理上不可匿名化。
- **ARR 稿里引 OSF DOI = 自曝身份**（§475.3 → §475.8 保留并升级依据）：ARR 对 preregistration DOI 无专门条款，适用 author checklist『otherwise disclose their identity』；camera-ready 再引 DOI。
- **匿名 submission repo 仍然必需，理由换成『提交件里的 code 链接必须匿名，否则 desk reject』（§475.8）**：CFP 用 *will* 非 *may*；`anonymous.4open.science` 是举例不是指定服务。09-05 后重跑导出、上线匿名 repo 的计划不变。
- **去标识脚本的 scrub 清单与 leak-scan 探针必须是两份独立编写的清单，且显式登记 INTENTIONALLY_KEPT（§475.5）**：共享 pattern 时首跑报 3 hits（替换值自匹配假阳性），另写 PROBES 立刻多抓 5 类真残留；KEPT：`MarvelsGrantMan136` 148 文件 / `blake.sullivan` 8 / `emma.lopez` 10（VWA 上游 fixture 账号）/ `execute-api` 6（metrics.py 错误分类子串）。

**已作废**：匿名 repo 的旧驱动理由「避开 anonymity window」（§475.8 替换）。

**证据**：笔记 §475.1 / §475.3 / §475.5 / §475.8。

---

## 八、毕设（dissertation）

### 8.1 范围、结构与术语定义

**当前值**：
- **毕设图规划四决策（§450.1，user 2026-08-10）**：① handbook 走 AskUCL 官方渠道，GPT P1 并行加速且官方答复优先 ② **disc+concl 合并为 Ch7**（学长 rubric #5）③ shop/WA 在跑数据**只作 external validation，落地才进，不预留图位**（距 09-01 硬截止 17 天、proxy 额度 $180 只够一项）④ **A2 pass@K 对照现在做**，进 Ch4 主文（F10b）——C3 噪声地板 discordance 14.3% 比 drop-one 1.7–3.3pp 大一个量级。
- **COMP0191 页数/图表/appendix 规则搜不到，别再搜（§450.9）**：只可能在 Moodle 或系里；Stage B 保守默认继续。`COMP0191` = MSc AISD Project（60 credit, 100% dissertation）；`COMP0190` 只是 Project Preparation（⚠️ 标 D）。
- **路由决策粒度必须在 ch1 和 ch3 写明是 task-level（§474.3）**：codex 两轮独立冷读都卡在 ch3 的 eq:router 是 per-step 形式化。
- **毕设全文 ceiling 统一指相对最佳单模式的增益（§474.9）**，不是绝对成功率水平；ch3:177 的 ceiling gain 相应改为 ceiling（user 裁定跟 ch7 走：*a ceiling of +3.45 to +16.35pp*）。
- **内部 cell 代号（§474.5）**：row-stub 表内换人话（cls·B0→Classifieds·235B，58 处）；column-header 表保留代号 + caption 现场解码；正文保留代号但前置速查页（全稿 152 处 `\cell{}`）。
- **ch7 两处 `89.9\%/96.7\%` 必须带与 `79.2\%/83.7\%` 相同的 in-sample caveat（§474.8）**：源产物表头写 `Bayes ceiling (cost tier)` 但正文警告 *『Both columns are resubstitution (in-sample) estimates, not Bayes ceilings』*。
- **新增权威定义页（glossary / reader's guide）必须逐条对 term-lock 与正文形式定义核对（§474.9）**：速查页（18 术语 + 2 解码表）吃掉 20 条 /stress findings 里的 9 条，含 2 个 P0。
- **§486（user 08-27）**：§484 的 20 条只动两处与数据相反的陈述（§3.2 grounding 契约——与实测 491/491 相反；oracle 限定——Appendix A 写着 full oracle 7/8 胜），其余 18 条押后。

**写作/排版纪律**：
- de-jargon 改写验收 = `tools/paper-deslop/scripts/invariant_check.py` 的 removed-number / removed-citation / removed-crossref 全 0 + anchors OK；terms 的 violation 是任务目标不是错误（§474.5）。报警必须逐条追根（§474.8：ch5 缺 13/13/10 个 `0/1/2` = 36 处代号替换；`10{,}000` tokenizer；**CO₂e 被误展开是真错误**）。
- de-jargon 不得展开标准单位符号——`CO\textsubscript{2}e` 保留，首次出现括注全称；*展开的是黑话，不是单位*（§474.8）。
- em-dash 清理不要统一替换，逐处按语义选标点；LaTeX 表格里的 `---` 是内容（`none` / `ref.`）（§454.2，222 处）。
- 多文件编辑批次一律绝对路径，结束后逐文件验终态（grep 特征串），不以批内 print 为准（§454.1：丢 3 处编辑致摘要仍写 'The mechanism is a supply constraint'）。
- 『编译 0 undefined / 0 overfull』要编译到收敛才成立：单次 latexmk 报 8-16 个 overfull，第二/三遍归零；无 baseline 会把中间状态当回归（§474.7）。
- glossaries 的 `\entryname`/`\descriptionname` 覆写必须放进 `\AtBeginDocument{}`（§493.4，preamble 里编译 exit 0 零警告但 PDF 仍是 Notation）。
- 书目升级（preprint → 正式 venue）取 venue 官方 .bib，不采信审读给的卷期页码（§493.7；arXiv API 无 journal_ref）。
- **Overleaf 单向同步（§452）**：真源 `final_dissertation/tex/`，Overleaf 项目 6a7a7331d2e6523a360245d4 只作渲染/导师评审层；同步脚本先本地编译，编不过拒绝推送；git identity 需在 clone 内单独 config。§494.5：`overleaf_thesis_sync.sh` 加未知参数守卫（`[ $# -gt 0 ]` → exit 2）并前置到 `set -e` 之后（传 `--dry-run` 被静默忽略并推了 Overleaf a75662b）。
- **submission typesetting pass（§491，commit b10d6b8）**：placeins+float / FloatBarrier / 附录表 [H] / `\artefact` / abstract 827→515 词 / Reader's Guide 4→3 页 等。**未采纳 GPT 主要诊断**『float 写在句子中间』——实测 0 处，真因是页顶 float 劈开跨页段落。

**已作废**：ch3:177 的「ceiling gain」定义（§474.9）；速查页中与 term-lock 冲突的定义（§474.9）。

**证据**：笔记 §450.1 / §450.9 / §452 / §454.1 / §454.2 / §474.3 / §474.5 / §474.7 / §474.8 / §474.9 / §486 / §491 / §493.4 / §493.7 / §494.5；FIGURE_PLAN §1 Stage B；VERIFICATION.md §3；`scripts/maintenance/overleaf_thesis_sync.sh`。

**原文片段**：§474.9 *"定义页的风险不与其长度成正比，而与它的位置和权威性成正比"*。

### 8.2 毕设图

**当前值**：
- **图内只放标签不放论断，论断一律下沉到 LaTeX caption（§481.1）**；五条 house rule 写进 `_style.py`，由 `check_no_prose.py` 读**渲染后的 PDF** 强制。学长判据：「图是为了方便人理解的，如果人看图需要更多 effort，那就是更差」。§482.4 判据改三处（含数字/·/=/_ 的 token 算标识符不计词数；白名单前缀匹配；「标签不以句号收尾」按 token 数判）；回归 6 条已知散文 0 漏判 / 9 条合法标签 0 误报。
- **删除 f0 / f2 文献地图 / f14 标签漏斗 / f15 三条出路（§481.4）**：过 FIGURE_PLAN R1 四问都答不上「删掉它哪个 claim 会失去证据」；f0 由 REALM fig_overview 取代。fig_partition_forest 与 fig_f2_h1_forest 不进毕设（§481.2：后者压着 'INTERIM (PARTIAL_DATA), NOT A VERDICT' 水印；前者依赖正文从未定义的 regex-flagged task 分区）。fig_sr_by_class 由 `fig_f6_sr_by_class.py` 重画（原图配色与全书相反且生成脚本不在仓库）。
- **横排页从 3 处减到 1 处（§483.2）**，只保留 Fig 1.1（fig_overview，2.67:1，属 REALM 原样保留批次）；Fig 1.2 改三行后每张截图 8.5cm → 13cm。
- **PRINT_W_IN / 「图内标签 ≥8pt」只适用于本仓库自己重画的图（§485.2）**；inherited 图（REALM 沿用的 3 张）基准是原发表呈现尺寸。
- **删除 `_style.save()` 而非接上调用（§483.4）**：零调用方却声称已消除 stale-figure；12 个图脚本 docstring 写明 ALWAYS make thesis-figures / NEVER 单独跑脚本。
- **mode 与 cell 显示名统一到 `_style.py` 的 MODE_LABEL / mode_label() / cell_label()（§493.5）**；**禁止 `.capitalize()` / `.title()`**（§450.14：`'som'.capitalize()` = 『Som』违反 TERMS.md 的 SoM）。cell 显示名跟随正文 `\cell{}` 写法（cls / red / wa_red），不用 VWA-/WA- 前缀（§494.3：正文已有三套写法 cls·B0 ×29 / Classifieds·235B ×9 / Classifieds·Qwen-4B ×8，图统一到第四套更糟）。
- **F13 / F14 形式换掉（§450.9）**：F13 = Baseline-normalized Dominance Plane（`x=log₂(C/C_b)` / `y=S−S_b`(pp)，0/8 成为视觉事实）；F14 = Thresholded Attrition Connected-Dot Plot（threshold-crossing 不是 flow-composition）；y 轴必须 pp 不用相对百分比。**§450.12 F14 诚实偏离：画三级不画四级**（`Label Exists ⇒ Task Solved` 就是 C5 机制）；门槛按 `N_MIN_CLASS_TRAIN=10` 且需 ≥2 个类别各自够 10 个，结论『4 of 6 cells never reach it』。
- **图必须渲染出来看三样（§450.11）**：压字/溢出、箭头语义指向、scope 措辞精确性。F0 scope 原写『两个 benchmark × 三 backbone』但 WA 只有 reddit 且只有 B0/B1；F3 输出箭头从 mode 框内部出发读成『mode 产生了 outcome』。
- **图注/标题措辞跑在脚本实际检查的东西前面是新的过程失败模式（§452.4）**：F1 n=1 写成 'a matched sample' / F7 token 计价轴标成 electricity（差 88×）/ F5/F6 宣称『完成度』『SHA 校验』而 parser 只比 SR 且只覆盖 36/48 / F15 panel B 注解过强。
- **F4 panel B 与 F11 跨图互证（§450.16）**：WA 无 difficulty 标注 × 该标注承担 VWA 大部分判别力 ⇒ 『可预测性有相当部分来自语料自带的答案，而那一列在部署里不存在』；Ch3（F4）与 Ch5（F11）必须互相 cross-ref。
- **找素材先 find 全仓再收窄（§450.15）**：素材分散在 `results/_archive_aborted_*` / `results/repro_replicates/` / `results/mechanistic/_obs_mirror/` / `docs/checkpoints/周报/weekly-dashboard/`；唯一的 SoM 编号标注图在 dashboard assets 里。
- **§479.5**：5 张毕设图重画后全部回滚至 HEAD 维持冻结（fig_f8 未决见 2.1）。

**已作废**：f0 / f2 / f14 / f15 图（§481.4 删除）；F13 的 8 个 Pareto 小图形式与 F14 的 Sankey/漏斗形式（§450.9）；F14 四级模板（§450.12）。

**证据**：笔记 §450.9 / §450.11 / §450.12 / §450.14 / §450.15 / §450.16 / §452.4 / §479.5 / §481.1 / §481.2 / §481.4 / §482.4 / §483.2 / §483.4 / §485.2 / §493.5 / §494.3；FIGURE_PLAN F13/F14；`scripts/analysis/figures/thesis/_style.py`；`check_no_prose.py`；`fig_f10_rerun_discordance.py`。

---

## 九、Showcase（海报 / demo / 演讲）

### 9.1 海报版本链

**当前值**：**印出来的是 v9.10**（§516.1）。板前 90 秒走读按 v9.10 从上往下讲（标题 → task 76 → 面板 1–4 → 5 → 6 → 电脑；20 秒版 = ①④⑤），并附「海报说过头处」口径表：IMAGE-ONLY 说成 *views with the screenshot*（含 SoM）· stalled progress 不当结论 · ~4× 是中位数之比 · 面板 5 横轴是 log 成本比 · 不用 oracle/labels/pp · 韦恩不说各有独占 · +16 不和一次重跑并排。海报已印无法改，只能口头纠正。

**演变（v2 → v9.10）**：
- §495（v2 冻结）：显式非对称 grid + 逐面板越界断言，不用连续 section 装箱（*装箱解的是文档问题，海报是层级问题*）；自建 markdown-ish 解析器必须对未消费标记字符断言失败（`*and*` 单星号原样印进 v2；补 `assert '*' not in leftover`）。
- §498.1 v4：组织方模板骨架 + 全宽 system diagram（毕设 Fig 1.1 三面板）+ 一套字号（41/28/14/17.65/12.7/32.5+10.6）；学长四条反馈：主 system diagram / 紧凑 / 字号 consistent / 像 NeurIPS；v3 自拉两栏网格被 user 否（模板不一致）。
- §499.1 v5：海报旁放电脑 demo，分工 = demo 管现象 / 海报管系统与度量；主题「Look, read, or both?」；否掉「The screenshot tax」——被 F1 数字证伪：Vision 3,123 < DOM 3,314 < SoM 4,335 tokens。
- §499.4 v6：jargon 全部换白话（oracle→best choice in hindsight · router→a learned choice · pp→more tasks in 100 · …）；demo 三帧压成记分板；Fig 1 改真正的环。
- §499.5 v7：零预设 GPT 审读：standfirst 拆两句各点名比较对象、因果措辞降级（『confirms scarcity is the mechanism』→『points to scarcity as the main bottleneck』）、新增 Fig 3（六点 97@27.2 / 55@14.3 / 53@14.8 / 24@7.4 / 16@2.2 / 15@3.9）；拒绝放大 0 of 8。
- §499.6 v8 = v6 版式 + v7 全部措辞修正；**v7 的『结果先行、loop 压底』版式撤回**（user：上一版样子更好；零预设审读优化被问的那一维，视觉层级由作者判断）。
- §499.8 v8.3 字号方案：正文 23pt / 图注 17pt / 图内标签 15.5pt / … / 数字 42pt；Fig 1 五张卡按各自剩余空间单独定标 19–29pt；空间来自缩图与 4 处字符串缩短，不来自任何 claim / 数字 / scope 的改动。
- §499.9 v9 = 图先占位重做：A1 竖版不变 · 顶部三段系统图 · 中部两看法截图带 · 三栏；去掉大数字条、翻面率图、碳排放；title 用论文题目。学长：『多图少字多示例、大数字很low』。**海报引用 REALM Table 5 时只写『极值集中在能看见页面的两个 view』（Vision 9 / SoM 5 / 四个纯文字 0），不得写 behaviourally inseparable / non-separability**——门槛 ≥7/8=87.5% 是 cell 数从 6 涨到 8 之后才定的（原 83%），rank_consistency 不是 separability test。
- §499.10：六个发现 panel 行优先 1-2-3 / 4-5-6；caption 一律压到一行。
- §499.11：截图带每格标注 = **导致该页面的那一步**的动作（`step N · <action> →`）；screenshot 记动作前页面。
- §499.12：行高不齐改**宽高比**不是改 scale（venn 2.91 → 1.71，高度上限 57→98mm）；**`figures/v9/` 纳入版本控制**，撤销『一条命令能再生成故不入版本』规则（三个输入本身 gitignored，含 7082 张截图）。

**已作废**：v2 展场设计（200pt 大零 hook / kayak 场景 / 深色 takeaway 带，§498.1 替换）；v3 两栏网格；「The screenshot tax」主题（§499.1）；v7 版式（§499.6）；海报上 non-separability 说法（§499.9 禁）；`figures/v9/` 的 gitignore 规则（§499.12）。

**caveats**：§499.4 ⚠️ *白话标签变长会引起图例/轴标签碰撞，换词后必须重看渲染*。§523.1 推翻了 v6 的「router → a learned choice」**仅限演讲**；板前走读仍用海报原词 hindsight。

**证据**：`deliverables/showcase/poster_content.md`（v4–v8.3 头 / 词汇表）；`deliverables/showcase/POSTER_ASSET_LIBRARY.md §10/§11/§13`；`build_poster.py` / `build_poster_v9.py`；`SHOWCASE_PREP.md §2`；笔记 §495 / §498.1 / §499.1–§499.12 / §516.1。

### 9.2 demo（录像 + live 页）

**当前值**：
- **路由可视化 = learned choice 的 fold-held-out 选择**（箭头 + 红/绿圈 + 一行说明），不画预算路由；不为让 router 好看而换题；build 拒绝 in-sample 与 fallback 预测（§506.1）。
- **红绿框按相对其他栏上色（§506.10 → §507.4）**：只有被选中栏唯一答对才绿 / 都对为灰（并点名更便宜栏）/ 选错别栏对为红 / 全错为灰；live 页要等访客把三栏都判完 ✓/✗ 才上色（live 任务无答案键，三个判定都来自访客）。
- **CO₂e 估算区间（§506.2）**：从不显示单个数；公式 (输入 token × J_in + 输出 token × J_out) × 设施系数 × 电网；J_out 1.0–8.0 · J_in 0.05–0.4 · 设施 1.23–1.97 · 电网 131–144 g/kWh（AWS eu-west-2 London）；区间宽 ~14×；SHOWCASE_PREP §5/§6 改为允许说『大约…由 token 估算、不是实测』。user 在被告知 B0 energy.source=disabled 与毕设立场后仍选碳排。
- **live 架构（§506.3）**：agent 在 DGX（`live/server.py` + `run_lane.sh`），站点 = quark 自己的 classifieds docker，`ssh -N -L 8799:localhost:8799 spark`；`P79_PAPER_GRADE=0`、不走 queue 脚本、不评分；一次一个 session，每栏 12 步，session 8 分钟。现场入口 = live server 自己提供的 `http://localhost:8799/`，不经 VS Code Live Server（§506.7：工作区文件变动整页刷新），`demo_portable.html` 只作断网备份。
- **live 显示（§506.9 当前）**：撤掉三栏起跑线（每栏一到就显示）；自动回到录像默认关闭（`?idle=<秒>` 才开）；router 选择 = `live/router_pick.py` 五折投票，推理难度取 classifieds 中位数。
- **一个页面两处用法（§507.3）**：展出 = quark 跑 demo（三题自动播放）；演讲 = 同一 index.html 加 `?task=130&autoplay=0`。
- 前端验证教训（§501.2 / §501.4）：交付给他人用某工具打开的网页必须复现那个工具的行为（Live Server 字符串搜 closing body tag 注入）；`background:currentColor` 与文字 color 不得共存于同一元素；比较面板共享基线（`clamp(118px,15.5vh,205px)`，顶/底错位 0px）；压暗其余在『其余』就是展示内容时自相矛盾（scrim 致亮度 212 → 135 (-36%)，去后 207.2）。

**演变**：§506.8 三栏起跑线同步（START_WAIT 60 s，每步至少停 1.5 s；只改显示不让 agent 互等）+ 5 分钟无输入自动回录像 → **§506.9 撤掉起跑线、自动回录像默认关**（user：『经常卡』、不同时开始、不跳回）；§506.9 红绿由访客判 → §506.10 相对上色 → §507.4 三栏判完才上色。

**已作废**：§506.8 的起跑线与默认 5 分钟自动回录像（§506.9）；§506.9 的孤立上色（§506.10 / §507.4）。

**caveats**：§506.10 碳排悬停去掉 'published'；codex Mode B 不补，以 Claude + agy 两家为准。§499.9 v9 海报去掉碳排放，与 demo 保留碳排区间是不同表面（9.3 的 §527.4 片子也不上 CO₂e）。

**证据**：`deliverables/showcase/demo/index.html`（.tag / .think / renderPick / armIdle）；`demo/README.md`；`demo/build_demo_data.py CARBON`；`live/server.py`；笔记 §501.2 / §501.4 / §506.1–§506.10 / §507.3 / §507.4。

### 9.3 演讲

**当前值**（截至 §527.6，09-15 晚）：主流程 **opening → prize（钩子数字卡）→ agents（三块真实抓取，LOOK · READ · BOTH）→ demo → question …**（§527.2 / §527.4 的顺序表述见下「⚠️」）。
- **prize 页**：模板 `.stat` 两块数字卡（27 → 43 · −20%），不用条形图（§527.6，user：条形图页「很丑」）；页脚另两个网站 +11 / +16、−14 / −27%；CO₂e 与用时不上片子（§527.4）；prize 删用时卡（只有一个网站变快、两个持平会自己拆台，§527.2）；view / router 引入前只说 *the right way to see each page / upper bound*，perfect router 从 learned 页起才说。
- **agents 页**：三块「7vh 大 logo（Codex / Claude Code / browser-use）+ 工具名 + 方法 + 裁到表单的真实抓取」；BOTH 用 browser-use 的 **DOM 覆盖层**（`dom_highlight_elements + add_highlights`），不用截图覆盖层（「完全不像 SoM」）（§527.5 / §527.6）。
- **术语**：hindsight 改说 **perfect router**，与 learned router 成对；learned 页标题 *No learned router reaches the win corner — even a perfect one rarely does.*；check_talk JARGON 删 router（§523.1，推翻 09-13 ROADMAP §4「router → a learned choice」）；perfect router 首次出现须交代事先知道结果、是上限。
- **learned 页**：散点只保留两种标记（橙点 = 没见过的题上测的学到选择；紫蓝空心方块 = 事后最优），删训练题三角；绿色只表示胜出区；右上角 *every learned choice costs more than the star*（渲染时断言）；0 of 8 / 1 of 8 不变（§522.2）。
- **why 页**：真实 6 点 + 训练分界线 + 能力箭头；纵轴用**第二常见正确看法的例子数**（判据所在），按 §D 判据 12.5 六点被一条横线分开（标签总数下 53 不够、55 够画不出线）；越线 ≠ 赢（事后最优 1 of 8）；2–4× 是下界（§521.1）。
- **demo 在演讲里只放录像回放 task 130**（手动步进，约 45 s），不跑 live；live 首步实测 19–21 s（§507.2）。演讲版 `?autoplay=0` 隐藏 demo 自己的 header，task 句 32px，caption 引号词 `<mark>` 高亮（§527.2）。
- **工具链**：slide 默认走 pptx（`talk_content.md` → `build_talk.py`），三级故障梯 quark 浏览器 → U 盘 → 内嵌 40 s mp4（§507.2）；套主办方 Showcase-Speaker-Deck.pptx 只在 `<style>` 末尾加覆盖段，顶条/页脚/logo 用伪元素（`check_talk.py` 数词）（§511.2）；demo 固定 1880×960 由 `fitDemo()` 整体缩放（§511.3）；09-16 带两台电脑：quark 全天放展板，MacBook + Chrome 只放片子（§511.6）；Talk PDF 统一由 `export_pdf.py` 导出，先等背景图与 logo 解码（§520.2）。

**演变（故事线链）**：
- §512.1（09-13）：标题 → demo → 观众带走的三条 → 行为不同 → 失败不同 → 要不要按任务选 → 学不会（0 of 8 / 1 of 8）→ 为什么还不行（label supply）→ 请求 → 收尾；「将来能学」只说『还不行，更强的 agent 是检验』。
- §512.3：失败页脚注改「text-only: several named failures over 2× · image-only: none stands out · pattern from the six VisualWebArena settings」；不拿 P27 的 2.3× 当主数（只有 13 次命中，P17 61 次 / P16 25 次同样过 2×）。
- §513.1：三条结论先上屏「一头雾水」→ demo 之后放一个问题 + 四个路标，close 回答它（presentation-playbook v3）。
- §513.3：hindsight 页删热力矩阵（48 个数）与三臂韦恩（独有解题数下界可归零，§470.3），换逐设置箭头图 `talk_hindsight.png`（5 行 BOTH / 2 行 READ / 1 行 READ 列表变体）。
- §517.1：学长「太严谨」→ hindsight 页 = Claude Code 前端自检场景 + 三大数字（+11 到 +16 / 百题 · CO₂e −7 到 −29%（估算）· 用时最多 −34%、2 of 3 网站持平）；learned = 大数字 0 of 8；why = *Routing's scaling law: no wins, no examples* + 2–4×；开场加导师。保留底线：用时不说处处更快、碳带 estimated、*if you pick perfectly*；scaling law 是趋势叫法不是拟合幂律。
- §518.1：两页改场景图与放大结果图，数字退为辅助（替代 §517.1 两页大数字布局）。
- §519.1：日常买书桌请求替代前端自检场景；同一 B0·classifieds 27.23% → 43.30%，成本 −20.2%。
- §521.1：why 页真实 6 点（替代 §520.1 纯示意双对数图）。
- §524.2：开场 opening → agents（Playwright MCP 真实抓取 / 截图）→ prize → demo；删 hindsight 书桌页；不用写死的动画。
- §525：agents 页两图放进 Claude Code / Codex（GPT-6 Astra · computer use）窗口框；台词 *can work from a screenshot*（不说 Codex 只看截图）。
- §527.2：主流程 opening → prize → agents → demo（先说能省多少更抓人）。
- §527.4：prize 页放在 demo 之后、question 之前；B0·classifieds 两根条 27 → 43 of 100 + −20% token bill。
- §527.5：agents 页三块大 logo；**D25 里让 user 自截终端图的方案作废**。
- §527.6：BOTH 用 DOM 覆盖层；prize 改两块数字卡不用条形图。

**已作废**：§512.1 的「收获前置」顺序（§513.1）；热力矩阵 + 韦恩 hindsight 页（§513.3）；§517.1 大数字布局（§518.1）；前端自检场景（§519.1）；纯示意双对数 why 图（§521.1）；训练题三角标记（§522.2）；「router → a learned choice」演讲用词（§523.1）；hindsight 书桌页与写死动画（§524.2）；CSS 窗口框认工具（§527.5）；D25 自截终端图（§527.5）；prize 条形图与用时卡（§527.2 / §527.6）。

**caveats**：⚠️ **prize 位置在同日两条里表述不一**：§527.2 写主流程「opening → prize → agents → demo」，§527.4 写「prize 页放在 demo 之后、question 之前」，§527.6 未再说位置；台账未给最终顺序的调和（见末节）。§519.1 / §518.1：书桌 / 自检场景非 Claude 实测，收益来自 benchmark 事后选择。§524.2：两图都不是 Claude 或 Astra 的运行记录。

**证据**：`talk/index.html`；`talk/rehearsal-script.md` v1/v2；`talk/check_talk.py`；`talk/talk_figures.py::routing`；`talk/browser_use_capture.py`；`talk/RUNBOOK.md`；`talk/export_pdf.py`；ROADMAP.md D1–D26 / §4；`SHOWCASE_PREP.md §5/§6`；`day-of.html`；笔记 §507.2 / §511.2–§511.6 / §512.1–§527.6。

**原文片段**：§523.1 *"No learned router reaches the win corner — even a perfect one rarely does."*

---

## 十、基础设施与运维

### 10.1 cls 站点与 reset

**当前值**：
- **cls 周期性 12s+ 完全不可用根因 = OSClass auto-cron 自请求 × `php -S` 单 worker；修法 = 注入 `PHP_CLI_SERVER_WORKERS=4`（B-1969，§442）**：`index.php:335` 的 auto-cron 让每个普通页面请求向同一 server 再发一个 HTTP 请求，发起方占着唯一 worker ⇒ 单个请求靠自己就能锁死全站（容器 CPU 0.00% / processlist 全空 / 24h 仅 261 行日志）。对照：5 次 `GET /` 产生 5 个内部 POST，5 次 `GET /?page=cron` 产生 0 个且延迟 0.06s vs 0.23s。**修正 B-1968 的归因**（head-of-line blocking 方向对但没答出谁占线程）。不选关 auto-cron（开关在 DB t_preference，会被 reset 冲掉）。**落点** = `scripts/vwa/start_vwa_docker.sh::start_classifieds()` 幂等注入（`CLS_PHP_WORKERS` 可覆写），**不是** compose 文件（被 submodule `.gitignore:153` 整体忽略，含明文 RESET_TOKEN）。⚠️ 订正：`check_vwa_health.sh` 与 `reset_vwa_sites.sh` Gate-3 注释里的『PHP-FPM』对 cls 是错的。
- **Fire-3/4/5/6 的 eval-timeout abort 与 B-1969 无关（§442.7）**：§253 实测 eval 挂死同一时刻 curl 仅 0.17s、全新 context 约 170ms；真根因 = agent 的 BrowserContext 退化（B-1803 修复）；eval_goto_timeout 11 个 episode 与 reset_goto_timeout 78 个基本不重叠。
- **B-1997：cls HTTP reset 之后无条件 DELETE oc_t_alerts + oc_t_latest_searches（§487.2）**：B-746 把 sentinel 从 3 张扩到 5 张时清理没上；08-27 一条 agent 订阅熬过每次 reset，cls 永久发不了车；种子态本就是空，fail-closed 保留。
- **B-1996：wrapper 层剔除表外字符（§487.1）**，不改 VWA submodule，不 quarantine 放行 task 137；改上游 `_keys2ids` 默认值会静默映射到任意键；对已跑完 cell 零影响。
- **R24364 三波 session lost 不做任何人工干预（§488.1）**：维持 §330 preserve-as-denominator-failure（删+重跑 = denominator surgery；根因是 agent 自点 Logout）；排除 infra bug 与 B-1997 所致（cls DB 无 session 表）。

**已作废**：B-1968 的归因（§442）；`reset_vwa_sites.sh` Gate-3 注释中「latency-degradation windows」与「Fire-5/6 eval-timeout aborts」是一件事（§442.7）；cls「PHP-FPM」注释（§442）。

**证据**：`docs/reference/master_bug_catalog.md B-1969`；`scripts/vwa/start_vwa_docker.sh`；笔记 §442 / §442.7 / §487.1 / §487.2 / §488.1。

### 10.2 proxy / API 契约与 fail-loud guard

**当前值**：
- **fail-loud guard 的判据必须是「要的东西拿回来了没」，不是「响应长什么样」（§468.7）**：paper-grade 下必须**恰好一个可解析 `web_action`**；两份非空文本不一致时 raise。
- **并行 `web_action` 一律 take-first + 落盘计数与被丢 payload，不 raise（§469.2）**；上游 `parallel_tool_calls: false` 实测无效，关闭不再排期。理由：take-first 与 2026-08-16 之前 `proxy_tool_calls[0]` 字节级等价（全部归档 B0 数据的既有语义），raise 才是行为变更；466 task × 30 步下必炸。
- **fail-loud tripwire 按『会丢什么』收窄（§466.2）**：B-1110（2026-05-18）埋的 `assert not isinstance(raw_content, list)` 断言的是形状；实际漂移是良性 text block + tool_calls 完好，却升级成 PaperGradeAbortError 带走整条 12-cell chain；改判据为「block 里有 tool_use 且顶层 tool_calls 缺失」。
- **replay 型探针不能证伪 provider 漂移（§466.2）**：`probe_b0_production_path.py` 回放漂移前 fixture、`pytest -k proxy` 92 passed 全绿，对 B-1970 无感；须真 `step()` 打真 proxy（`tool_call_parse_path=="tool_calls"` + confidence 4/4 + `confidence_error=null`）。
- **远端告诉你为什么的通道不要在错误路径上扔掉（§472.8，B-1984）**：`raise_for_status()` 丢 body，而 AWS proxy 的拒绝理由只在 body；修复打形状不打内容。
- **监控区分 `proxy_outage:5xx` / `quota:4xx` / `network:*`，quota 告警连续 2 轮确认，503 永不触发预算告警（§445.4）**；v1 第二轮就推了假的「预算耗尽」（实为 B-1880 已记录的 503）。
- **runner 三个 fail-fast raise 分支（evaluator_unavailable / fatal_env / proxy_quota）各推 urgent ntfy，通知器 best-effort 永不抛（§446.2）**：*越严重的事件越安静*，机器空转六小时。
- **健康探针与被监控负载同形状——proxy 探针用 `max_tokens=4096`（§446.3）**。
- **支出 gate 改用实时 quota 探针**（每 cell 前查，floor $60），不再用累计记账天花板（§478.5）：config 价低估 25%；旧 chain 的 `_chain_cost` 一直返回 $0.00（key 不存在），那道天花板从未生效过。
- **无人值守 chain 的 halt 条件必须含成本上限，且从产物现读（§471.7）**：上限 $400 对计划 $291，防的是价格记错（GPT-5.6 sol 实际 6 倍）。
- **验证 endpoint 图像通道 = 语义答案 + input-token 增量双条件（§456.1）**：gemma-4-31b 答 'White' 被首版判 silent-drop，input token 622 vs 文本 44 证明图收到了；语义题选不可猜中的。

**已作废**：B-1970 guard「顶层 tool_calls 非空就放行」（§468.7）；§468.7 衍生的「两个 call 即 raise」行为（§469.2 改 take-first）；B-1110 tripwire 的形状判据（§466.2）；累计记账成本天花板（§478.5）；预算 watcher v1 的 `status != 200` 一律当没钱（§445.4）。

**caveats**：⚠️ **§446.3 的「74 分钟」证据被 §446.7 推翻**：§446.3 写「01:19 真实 run 已被 403 拒绝，而 max_tokens=1 的探针到 02:33 才开始 403 —— 74 分钟里探针说健康」；§446.7 判定 *"从一个 14 分钟的盲区编造出 74 分钟，并据此建立了一个机制假说（max_tokens 预留）写进笔记与台账"*。「探针与负载同形状」的规则仍在用，但其原证据数字作废（见末节）。§469.4 未修：`env_snapshot.json` 记的 remote commit 是 rsync 前旧 commit 且 dirty=true（codex F4）。

**证据**：`p79/experiment/runner/helpers.py::push_run_abort_ntfy`；`tests/test_run_abort_ntfy.py`；`scripts/maintenance/proxy_budget_watch.py::PROD_MAX_TOKENS`；`scripts/queues/_b5_reddit_chain.sh`；B-1984；笔记 §445.4 / §446.2 / §446.3 / §456.1 / §466.2 / §468.7 / §469.2 / §471.7 / §472.8 / §478.5。

**原文片段**：§466.2 *"它断言的是形状，而真正会造成损失的是内容"*。

### 10.3 chain、monitor 与进程操作

**当前值**：
- **不能靠『再起一条 chain』给正在跑的 chain 排队，必须用 watcher 接力（§451）**：`_lib_paper_grade_gates.sh:235` 用 `flock -n`（抢不到锁当场 FATAL rc=78）；`queue_chain.sh` 段列表启动时固定。
- **接力 / 开火 watcher 用双判据（§451 / §461.4）**：主 = 前一段产出文件计数达标或末段自己的完成 sentinel（`condition_summary_v2.json`），副 = PID 退出；只满足 PID 消失时推高优先级 ntfy 并退出而不开火；6 天上限兜底。判据不得用 pgrep pattern（`pgrep -fc "autolaunch.sh 738522"` 返回 3 而实际 1 个）。
- **done-monitor 判据必须只能被目标事件满足（§449.5）**：grep 到 08-07 旧 runner log ⇒ 21:39 误报 `RESUME_CONFIRMED`；改为『旧 run 目录 episode 数 > 374』。
- **kill 进程按 PID 不按 pattern（§446.6 / §471.8）**：列 PID 的命令里**任何字符串都不能被 pattern 匹配**（`proxy_budget_watch.py` 含 `budget_watch.py` 子串 ⇒ exit 144）；查进程用 `ps -eo pid,cmd | grep "[r]eframe_xxx"`；`grep -v grep` 救不了。**经 ssh 的发车保护（§515.5）**：命令首行放 `# LAUNCHGUARD_<date>`，检查 `grep -v LAUNCHGUARD_<date>`；不用 `pgrep -f` + `grep -vx $$`（还会匹配 `$(…)` 子 shell）。
- **跨机时间差先并排打两边 `date` 再做减法（§446.7 / §449.6）**：DGX/BST 与 A100/UTC 差 1 小时；§449.6 同日第二次踩（63 分钟真值 ~10 分钟）——*记了没挡住*。
- **chain 运行期间不得覆盖 queue 脚本**（改好的传成 `*.sh.b4`，launcher 确认无存活后 `mv`，留 `.pre_b4_<date>` 备份）（§461.3）；**部署脚本与 chain 分开且 scp 到 repo 外执行**（§471.7）——bash 按文件偏移读取正在执行的脚本。
- **halt 后重启只重新声明欠数据的 cell（§487.4）**：`FORCE_NEW=1` 对 spec 每个 cell mint 新 run_id，verify 用 `ls -dt | head -1` ⇒ 更差的重跑会覆盖能过的结果；新增 `PHASE_C_CELLS` 覆盖点。
- **被 source 的 shell 库里每个循环/临时变量必须 `local`（§472.12，B-1987）**：`for site in cls red shop` 覆盖调用方 `local site`，ntfy 写错站点 = 让人再起同站 chain 的信号。
- **守卫类通知按状态变化推送，不按轮次（§472.14，B-1989）**：每 30min 一条、2 小时 5 条、~48 条/天；修法 = 对被挡内容取指纹。
- **磁盘告警按档位推送（§514.3）**：余量跌进 50/25/10/5/1GB 各档各推一次（≤5GB urgent），文案改『共享盘，先查谁在写』；旧文案 §487.6 已证归因错误但只清盘未修脚本，不到三周复发。
- **`git stash` 与 `pop` 之间不得夹入可能超时的长命令（§500.7）**：三次 pytest 撞 120s 上限被 SIGTERM，改动只剩在 `stash@{0}`。
- **marker 不阻塞 queue_chain / queue_baseline 发车路径（§496.6）**：消费者仅 `experiment_watchdog.py:1171` 与 `queue_phase1_paper_grade.sh:146`；锁是否持有用 `flock -n` 判；代价 = watchdog 不会自动补缺集，done-monitor 的 205-episode 检查是唯一兜底。`experiment_watchdog.py:1185` 400 字符截断**又一次**挤掉真肇因（§469.5 起已第三次）——诊断过 ≠ 修过。
- **08-30 08:40Z MANIFEST-BIND FAIL-CLOSED 判为已消解且未阻断任何东西，但肇因无法确定（§492.3）**：排除 B5 完成 / rsync 在飞 / §487.7 三个 reddit ghost。
- **多 session 并行写仓库（§451.4）**：append-only JSONL + 各写各的 § 号安全；`next_steps.md §0` 是共享区必须核对（ledger 2224→2278 +54 而自己只加 3 条）；块标注并行关系而非『最新』。

**证据**：`scripts/maintenance/wa_chain_autolaunch.sh`；`_b4_wa_watcher.sh`；`_launch_b4_smoke_and_wa_shop.sh`；`_reframe_bootstrap.sh`；GOTCHAS.md；CLAUDE.md done-monitor 章；笔记 §446.6 / §446.7 / §449.5 / §449.6 / §451 / §451.4 / §461.3 / §461.4 / §471.7 / §471.8 / §472.12 / §472.14 / §487.4 / §492.3 / §496.6 / §500.7 / §514.3 / §515.5。

### 10.4 fire 机部署与自动化授权

**当前值**：
- **发 paper-grade chain 前必须核实 fire 机上的代码即为拟发版本（md5 双侧比对 + 远端跑测试），不以 dev 机绿灯为准（§469.4）**：A100 上 `proxy_api_agent.py:1016` 那句 raise 原封不动。
- **A100 装 scikit-learn 1.5.2，不放宽主机验证闸（§472.6）**：缺 sklearn ⇒ 整个 tests/ 收集失败（`-k` 在收集之后过滤）⇒ 该闸自诞生起不可能通过；装前 `--dry-run` 确认无升级；装后 numpy 1.25.2 / scipy 1.15.3 / torch 2.11.0+cu128 未变；A100 221 passed / 1 skipped 与 DGX 逐项一致。
- **fire 机上写 tracked 文件的自动化必须登记进 `_reframe_bootstrap.sh` 的 `_HOST_AUTHORED`（§472.13，B-1988）**：部署前 `git stash push -u` 把它们扫走；ntfy 说 `QUARANTINE task 0` 而 registry `count: 0`。名单是手工注册表（与 CLEAN_PAIRS 同性质）。
- **发新 B-number 前必须 DGX 与 A100 两台机都查（§487.5）**：B-1993 撞号（A100 commit 1ed6a23）；DGX catalog 到 B-1995、A100 到 B-1990，互不包含；改用 B-1996/B-1997。
- **自动 push 授权**：§471.8 `reframe_finalize_poller.sh` 获范围限定的自动 push 授权（user 2026-08-19），范围由「脚本能产出什么」界定；总则「push 需显式确认」不变 → **§472.4 修正：范围必须由该动作实际作用的对象界定**（B-1982：`git push` 推的是分支，09:23Z 把 11 分钟前交互 session 的 3 个无关 commit 推出去）；修复 `_scoped_push` 要求 origin..HEAD 每个 commit 同时满足 subject 匹配 + 只碰三个白名单路径。⚠️ 同类：`git push` / `rm -r` / `docker prune` / `pkill`。
- **A100 驱动（§515.2，user 选）**：rmmod + modprobe 原地换成 580.178.04（不 reboot），16 个 `*nvidia*580*` 包 apt-mark hold 到本地 replicate chain 结束；内核 5.15.0-191 已装未启用，chain 期间禁止重启；§387.1 只修不挡导致七周后复发。代价：hold 期间收不到 nvidia 安全更新。

**已作废**：§471.8 授权范围的「按脚本能产出什么」界定（§472.4 推翻；授权本身仍在但须经 `_scoped_push`）。

**证据**：笔记 §469.4 / §471.8 / §472.4 / §472.6 / §472.13 / §487.5 / §515.2；B-1982 / B-1988。

---

## 十一、过程方法论

### 11.1 cross-AI 审计

**当前值**：
- **价值机制 = 每家找的『缝』不同，不是『谁的结论对』（§463.4）**：收敛的五条都是『读产物就能得出』，独有抓获全都需要把两份东西对着读（codex 代码↔报告 / Kimi 结尾句↔产物清单 / DeepSeek floor 覆盖面↔推论）。
- **方向值钱，算术必须自己重算；Phase 4b（1-AI unique 的 P0/P1 全量事实核验，不 sample）不可省（§452.6）**：codex 二项阈值算错（4.91 vs 真值 4.02），gemini 两条事实不成立。
- **分工 codex 读 diff / gemini 冷读定义 / Claude 查 caveat，经验证有效（§474.9）**：codex 7 条全 unique。
- **内联式审计：每份内联产物整份给或明确标注截断处（§468.8）**：gemini G6『幽灵引用』是 `sed -n '1,45p'` 截在第 45 行造出的（实在第 51 行）；与 §464.1『未内联的产物会被读成不存在』是同一失效的两半。
- **opencode CLI 不可用于长 prompt 审计，改直连 OpenAI-compatible endpoint（`crossai_frame_direct.py`）（§463.5）**：7.7KB prompt 挂死 51 分钟，server socket fd = 0；kimi-k3 只接受 temperature=1；deepseek-v4-pro reasoning 与答案共用 token 预算（空 content + 61KB 思考文件会骗过按体积的 sanity check）。
- **写进 env / source 文件的内容只能是合法 shell 赋值；cross-AI『快而空』时先验调用约定再怀疑模型（§473.5）**。
- **audit prompt 应写明『只读，不得修改任何文件』（§497.7）**：codex 未经授权写了笔记 §497（82 行）+ ledger 7 条；SKILL.md v7.3 硬约束只约束 Claude 不约束被派发的 cross-AI。

**证据**：`scripts/maintenance/crossai_frame_direct.py`；`.claude/skills/stress/SKILL.md`；笔记 §452.6 / §463.4 / §463.5 / §468.8 / §473.5 / §474.9 / §497.7。

### 11.2 验证、探针与归因的通用教训

**当前值**（每条一句；都是本片实证）：
- **探未知 API 时候选表放一个已知好的控制 + 一个故意错的探针（§444.3）**：B4 探针把 `api_name` 当字段名，六候选全 400（含 qwen 控制）才暴露是字段名错；故意错的模型名带出 `Use GET /model-api/models`。
- **查 arXiv 前先打一发已知存在的 ID 当控制组；「未核成」不能报成「查无此文」（§470.8）**：301（无 `-L` → 0 字节）与 429 解析后都「没有 entry」；控制 `2501.12326`；§462.5 四个 ID 维持 unverified。
- **探针『真读到图』用颜色不用数字 OCR，扫描整个 action blob（§471.4）**：B0 把 `[326]` 读成 `[360]`；CONTROL FAILED 拦住『GPT 五关全挂』。
- **『我这条路失败』≠『那个东西不行』，下结论前枚举其他路径（§471.5）**：结构化输出至少 tools / response_format(json_schema, json_object) / 纯文本+解析三条；控制组只能验证工具好不好使，验证不了漏了另一条路。
- **fail-closed 的错误信息说对一个真实存在的原因 ≠ 本次原因（§472.8）**：cert 到期 / SHA pin 长度 / 缺 sklearn 三例；*问「这句话解释得了我看到的全部现象吗」*。
- **测试必须验行为不验形状（§448.4）**：探针测试断言线上 payload 而非模块常量；分支覆盖用 AST 按实参绑定而非源码字符串计数（⚠️ AST 仍非控制流检查）。
- **判断代码结构用 yaml.safe_load / ast，不用字符串出现（§448.6）**：`include_sites` 出现在注释里 ⇒ 三个 phantom config 静默跳过插入 ⇒ load_tasks 加载 0 个 task。
- **产物文件存在 ≠ 产物有效（§466.4）**：摘要数 `episodes/*summary*.json` 文件个数，两个 B4 smoke HTTP 400 / 0 步照样「1 ep」。
- **字符串替换脚本改完须独立复核落盘内容，不以 `re.subn` 计数为准（§473.8）**：`\\answerYes{}` 16/16 MISSING。
- **新增断言/字段/阈值前先答：谁负责让它为真，谁负责读它（§489.6）**：08-27 一天四次同形状；*更像是该代码库的系统性形状而非个案*。
- **查产物优先于查文档；两者分歧以产物为准（§450.19）**：六次撞上；代价双向（两次低估、两次高估）；⚠️ 台账 `pass@` 0 match 是假阴性（该分析叫 one-arm margin）。
- **防重做台账挡得住重做，挡不住在新动作里重蹈（§494）**：发车/新计划落地前把已知缺陷推演到计划终点。
- **动笔前台账查证须覆盖不带数字的因果连接句（§495）**：『Always-cheapest is a cost floor, so anything that protects success costs more』直接印进海报 v2。
- **RETRACT 一条论断时 grep 所有引用该论断的散文（§495）**：§476.4 修图注，同句在 §450.12 散文里活着。
- **归因写入者只能靠该进程自己的 log/trace，时间共现不构成证据（§497.7）**；同节三次误判（mtime 共现 kill codex / 跨时区 45 分钟真值 1.5 分钟 / ps 截断误判 monitor 已死）——*缺的不是知识而是次序*。
- **元层**：§398.3 结论层分工按『外包风险』而非条数分配（A 批 831 条裁定 Claude 亲读；D 批 875 条测量交 subagent 并强制附原文片段）；§398.6 before-you-claim 清单不能只是再写一条规则（*知道规则与推理时调用规则是两件事*，起作用的两次都是别人在场）；§449.8 **写笔记 = 两步**（append 后立刻录 `ledger.jsonl`；四条录入纪律写进 CLAUDE.md）；§470.4 CLAUDE.md / memory 里会漂移的量一律写成指针不写数值（CLAUDE.md 自定 Stale-info rule 却违反最狠：「3 个 baseline」实为 5 等）。

**已作废**：「pkill pattern 要 specific / 先 pgrep -af 验证」作为充分规则（§471.8 判其不 actionable）；「用 API 不用 WebSearch」作为 arXiv 核查的充分规则（§470.8 补控制组层）。

**证据**：GOTCHAS.md；`.claude/CLAUDE.md`「写笔记 = 两步」；`scripts/maintenance/probe_proxy_model_registry.py::CANDIDATES`；`tests/test_proxy_budget_watch.py`；`tests/test_run_abort_ntfy.py`；`deliverables/showcase/build_poster.py segments()`；笔记 §398.3 / §398.6 / §444.3 / §448.4 / §448.6 / §449.8 / §450.19 / §466.4 / §470.4 / §470.8 / §471.4 / §471.5 / §472.8 / §473.8 / §489.6 / §494 / §495 / §497.7。

**原文片段**：§497.7 *"观察不完整时应先补完观察，而不是先下结论再找支持"*。

---

## 十二、⚠️ 本片矛盾与待核清单

> **2026-10-06 跨谱系审校（agy `gemini-3.1-pro-low`，依据台账原文）**：真矛盾只有 **#1**（sonnet-4-6 单价两处不一致），已由 B4 config 注释与 §456.2 闭环；
> #2 / #3 已被后续条目明确取代（§446.7、§470.9），#4 是有记录的暂缓落地（§479.5），#9 是已登记的待办，均非矛盾；
> #8 已闭环（见下）；#5 / #6 / #7 涉及 §500 以后，审校材料未覆盖。

| # | 事项 | 两侧 / 需核 |
|---|---|---|
| 1 | **sonnet-4-6 单价** | **已闭环（2026-10-06 核仓库）**：§444.2 的 0.001/0.005 是未接线注册项的**占位价**，§456.2 / §456.4 记真实标价 0.003/0.015（sonnet-4-6 与 sonnet-5 同价），并于 2026-08-12 用计费 `usage.cost` 核过（`configs/exp_v2_B4_*.yaml` 文件头注释）。⇒ §444.2「与 B0 同价」与「12 cond ≈ $154」的前提**作废**，B4 实际单价是 B0 的 3 倍 |
| 2 | **§446.3 的 74 分钟** | §446.3 以「74 分钟探针说健康」支撑「探针用生产 max_tokens」；§446.7 判 74 分钟是跨时区编造（真值 ~14 分钟），并点名据此建立的 max_tokens 预留机制假说。规则仍在用，原证据作废 |
| 3 | **REALM notif 日期** | §468.10「08-21 到」；§470.4 判「09-07」为 stale（实为 08-21）；§470.9 同日记翻转史 09-07 → 08-21 → 09-07；§480.5 引「09-07 意见」。按 §470.9 当前为 09-07，§470.4 的那一项更正自身成 stale |
| 4 | **重跑区间 4.9–7.6 vs 2.0–7.6 vs 0.0–7.6** | §406「4.9–7.6pp」（两个有地板的 cell）与 §450.10「2.0–7.6pp」并存，台账未说明是 scope 差还是替换 → §479.5「0.0–7.6pp」未 land（正文四处仍写 2.0，framing 待决）。另 §413 的 0.89-2.23pp 是不同泛函口径，禁止与之互减 |
| 5 | **B-1997 编号** | §487.2 = cls reset 无条件 DELETE；§508.6「B-1997 的处置顺序」= B5 coordinate_contract。§487.5 记两机分叉各自发号。需核 catalog 中 B-1997 的真实指向 |
| 6 | **演讲 prize 页位置** | §527.2「opening → prize → agents → demo」vs §527.4「prize 放在 demo 之后、question 之前」（同日）；§527.6 未再说位置。以 talk/index.html 现状为准 |
| 7 | **B2 replicate** | §464.3 / §467.1 / §505.27 均不排 B2；§515.4 在余额阻塞下按 user 要求排入（d 2.9-4.7，最后跑、先砍）。B2 数据的报告等级须按 §468 / §469.7 declared power（d<10 = inventory） |
| 8 | **§407.22 两候选框架** | ~~本批未见显式拍板~~ **已闭环（2026-10-06 跨谱系审校指出，已核台账）**：§413 ADJUDICATED「paper frame 定为『该加哪条通道在两站点间翻转 + 该选择下沉不到单任务』」；§413 同时 RETRACTED 两个落选候选。后续 §505.27 再改写为 route-the-budget |
| 9 | **cost 区间 13.7–35.3% 的 scope** | §450.5 明写未逐格核，仍未闭合 |

---

## 对旧结论层的 supersede

- **B-893（3-AI overlap 裁定的 P0：fixed-marginal 独立性 null）** → §398.2 → 该 null mis-specified，不作为 complementarity 检验（⚠️ 散文动之前须跨 AI 复审）。
- **A4 §383.1（REALM：Paper A non-archival + Paper B archival）** → §398.8 → 两篇合并为一篇（且被静默改成 archival）→ §407.9 恢复 non-archival → §502.2 camera-ready 维持 non-archival。
- **A4 §397.8 / §397.4（Paper A 正面结论削弱 + 保留结构性结论）** → §398.8 / §407.12（台账条目合记为「§407.1 + §407.12」）→ 合并稿新增第 ③ 步关掉 A 的正面结果；旧 ②（H3 两轴结构）整个消失，2x2 降为构造效度检查。
- **§294 ④（axis 1 = 纯拍扁，不 confound）** → §470.1 → 自 AMENDMENT_07（§295 `3a79196`）起 axis-1 是 bundled，不得表述为 isolation。
- **§387.3 RETRACTED 条的 `replaced_by`（WA shopping/shopping_admin 保持不支持）** → §455 → 已被 B-1930 + VWA shopping reset 实装（d78fd3b）推翻，现支持。
- **B-1968（cls 不可用归因 head-of-line blocking，记成未修）** → §442 → 真根因 auto-cron 自请求 × 单 worker，B-1969 已修。
- **§253 相关的 `reset_vwa_sites.sh` Gate-3 注释（latency 窗口与 Fire-5/6 eval-timeout 是一件事）** → §442.7 → 两者无关，注释已订正（§253 的 BrowserContext 结论本身保留）。
- **§298.2 determinism 作为「B1/B2 不做地板实验」的理由（§402.7 引用）** → §406 → 理由 RETRACTED 并替换；§298.2 本身作为机制在 §470.7 继续使用。
- **§150b.4 / B-1550 锁定的「95% 非支配」作为唯一报告档** → §399.3 → Pareto 三档并列报。
- **2026-05-13 K-of-N 降为 transparency-only** → §407.6 → 同一原则推广到四维剖面：不设 6/6 门槛，报 x/6。
- **§89 / §95 VWA visual task 占比** → §406 → 合并稿不报。
- **§216.1 shop「唯一真 OOD」** → §406 shopping 降 future work → §448.1 仍以 router 泛化检验为唯一理由跑 B1 shop 435×3（作 external validation，§450.1）。
- **§387.1（A100 驱动只修不挡）** → §515.2 → 加 apt-mark hold 并禁止 chain 期间重启。
- **§487.6（磁盘告警归因错误只清盘未修脚本）** → §514.3 → 按档位推送 + 改文案。
- **B-1110（2026-05-18 埋的形状 tripwire）** → §466.2 → 判据改为内容（会丢什么）。
- **B-1959 tripwire（`test_manifest_has_no_shopping_conditions_yet`）** → §510.7 → 退役，换守 FIX 本身的测试。
- **§464.2 已 RETRACT 的 flip 数字（49 of 224 / 48–52% / 2.9%）** → §504.3 → camera-ready 换 67/224 (29.9%) / 67.0% vs 5.9% / 11.4×。
- **§476.4（已 RETRACT 的海报/图注论断）** → §495 → 引用该论断的散文须全仓 grep。
- **A4 §381 / §339 的 B3 路线** → §419.1 维持 MiMo-VL-7B-RL-2508（非推翻，重确认）。

---

## 覆盖性闭合

- **本批条目数 = 295；实际用到 = 295**。
- 一条多用：§406（4 条）、§449.3（2 条）、§471.8（4 条）、§480.5（2 条）、§479.1（2 条）等分布在多主题，按 § 号均已出现。
- **未归入任何主题的条目：0 条**。
- 噪声类数字（4.93-7.39pp / 0.89-2.23pp / 4.9–7.6pp / 2.0–7.6pp / 0.0–7.6pp / 3.82–4.15pp / 3.52 / 3.95 / 4.02 / 10.27/12.50/12.05% / 14.3% / 1.7–3.3pp）各带原 scope 并列，未做任何加减。

*本文件覆盖 A 批第 5 片（295 条，§398.2–§527.6）。A1–A4 已落盘于同目录，A4 止于 §397.10。*
