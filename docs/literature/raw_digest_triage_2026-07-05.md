# Lit Triage — docs/literature/raw 11 个 digest 全量核验 (45 papers, 2026-07-05)

**Source:** `docs/literature/raw/` 全部 11 个 digest (2026-05-29 → 2026-07-05)，共约 50+ 条目；扣除已在 06-15 triage (`cron_digest_2026-06-12_triage.md`) 处理过的与已在 bib 的 10 篇（MIRAGE / Scaffold / AVR / WebChallenger / CSCR / SteerMoE / Seeing-but-not-Believing / Logit Sharpness / early-exit / Same-Task-Different-Circuits），本轮核验 **45 篇**。
**Verification:** 4 个并行 sub-agent，全部经 arXiv API `id_list` 核验 + 原文实读（LIVE 候选读 HTML 全文重点章节，park/drop 级读摘要+结论）。**45/45 ID 真实存在，零幻觉引用**；1 篇 OpenReview 条目找到 arXiv 真身（YtWZdwEG5K = 2602.21704）。
**Read depth:** router 簇 13 篇 / web-agent 观测簇 10 篇 / confidence 信号簇 8 篇（前三簇 ★ 篇全文实读）；mechanism 簇 14 篇轻量核验（§5 已 shelved，默认 park）。
**Bottom line:** 45 篇中 **零篇做 observation/representation 轴路由、零篇在 WebArena/VWA 上做 routing** — P79 的 phantom routing space 轴差异 novelty 经全量文献扫描后完整存活。真正进 paper-1 LIVE 的新引用候选 **12 篇**（多为 related-work/motivation 权重，无一承重）；最有行动价值的是 **2606.22864 触发的 router-AUROC 加固**（见 §2.1）。digest 系统性问题延续 06-15 诊断：§5-anchor 漂移未修 + 无 bib-dedup + **类比通胀**（新发现，见 §4）。

---

## 0. Verdict 总表 (45 篇)

### LIVE-cite 候选 (12)

| arXiv | 题 (真题) | 落点 | 权重/注意 |
|---|---|---|---|
| 2603.04445 | Dynamic Model Routing & Cascading Survey (Moslem & Kelleher) | **§1 gap 声明 + §6 总起** | 最硬 gap 引文：六 paradigm 全在 model 轴，representation 轴零覆盖（grep 全文无 observation/WebArena）。勿写"首个 survey"（无据） |
| 2605.14290 | Web Agents Should Adopt the Plan-Then-Execute Paradigm (Piet et al., Berkeley) | §2 contrast pole + §7 一句 defuse | digest 的 THREAT 定性**高估**：安全动机 position paper（cs.CR, Popa/Wagner 组），零实证 agent；PTE 内部 18.72% 任务的 LLM 子例程仍要消费页面观测 → 表示选择问题不消失 |
| 2605.29397 | Revisiting Observation Reduction for Web Agents (Enomoto et al.) | §2 | 最近的 observation-efficiency 邻居，但只做**单表示内** HTML extractive 压缩、无 per-task 自适应 → 衬托 P79 表示间路由。数字比 digest 更强（评测加速实测 290×，非 100×） |
| 2606.16158 | Focus When Necessary (LazyMCoT, TencentBAC) | §6 adaptive-visual-input 邻居 | 13 篇 router 簇中唯一非 model 轴：难度触发的视觉输入 escalation（单 mode 内），对位 P79 mode-menu + complementarity。⚠️ digest "GUI grounding" 错——实为高分辨率自然图像 VQA |
| 2606.25249 | Adaptive Re-Ranking (Genc et al., UMass) | §6 + router 结果讨论 | oracle headroom vs learned-router gap（router acc 仅 ~0.65）的最强跨域类比。⚠️ 其 oracle = full per-query argmax，**≠** P79 drop-one 构造，引用必须区分 |
| 2605.07180 | Learning Agent Routing From Early Experience (BoundaryRouter) | §6 | training-free cold-start 路由 + "routing is surprisingly hard" 互证；87 题小 benchmark，不作强证据 |
| 2606.06708 | Signal-Driven Observation (SDO, position paper) | §2 一句 | 正交轴（WHEN to observe vs P79 WHICH representation）；顺带 cross-capability 动机呼应 |
| 2606.00435 | Detect Before You Leap: Mirage Detection | §2 一句（MIRAGE cite 旁） | 12-backbone 确证 mirage regime 系统性；定位话术 "detect-and-abstain vs P79 route-and-exploit"。TC-LIA 对无图 phantom 臂不适用 |
| 2604.20940 | Sema: Semantic Transport | substrate 段低权重 | ablation（纯视觉 token 75.5% → +结构文本 93.3%）佐证"结构化文本承重"。⚠️ 引用措辞禁用"丢图不掉点"（Sema 保留视觉 token，与零图 P-SoM 不同构；禁忌 frame 红线） |
| 2511.19477 | Building Browser Agents (Vardanyan, 生产报告) | §1/§6 rule-router 动机低权重 | selective-vision escalation = 产业界手工表示路由实例；标注 non-peer-reviewed technical report |
| 2602.15580 | How Vision Becomes Language (PID 逐层) | §1/§2 motivation 1-2 句 | mechanism 簇唯一破例：晚层 82% language-unique + 信息指纹 task-dependent = phantom 合理性 + per-task 路由双动机锚。引作 motivation 非 evidence（LLaVA-only、非行为层）；兼 paper-2 must-cite |
| 2606.27023 | Medical VQA Verbalized Calibration (2×2 factorial) | §3 设计先例一句（可选） | black-image factorial 与 phantom "去图保结构"同源；注意其 2×2 是训练信号构造非评估设计 |

### Reviewer-defense（预案弹药，不进正文或至多脚注）(6)

| arXiv | 题 | 防的什么 |
|---|---|---|
| **2606.22864** | When AUC 0.998 Is Not Enough | **唯一带 pre-deadline 行动项**（见 §2.1）：同域（CUA+Mind2Web+Qwen-VL）probe construct-validity 警钟；对 LIVE routing-AUROC 的可迁移攻击面 = trivial 协变量驱动 |
| 2605.02241 | Zero-Shot Confidence Estimation | 监督 router OOD 崩塌（AUROC 0.51-0.56）→ LR 跨站泛化质疑的承认+design-motivation 引文 |
| 2605.18796 | UCCI | "为何不用 confidence cascade" 之问：cascade 需先跑完小模型 episode，web agent 成本模型完全不同 |
| 2604.09839 | Steered Activations Non-Surjective | ⚠️ digest THREAT 定性**反了**：phantom = prompt-level on-manifold 干预，恰在其划的安全侧 → **辩护资产**（"为何路由表征而非 steer 激活"现成答案）；paper-2 caveat must-cite |
| 2606.19868 | Black-Box UE 横评 | "为何不用花哨 UE 做路由信号"：24 方法无赢家 + 强方法需多次推理与 cost-aware 冲突 |
| 2606.17234 | Self-Assessing Tongues (MT) | verbal 与 internal 零相关**且互不占优、都弱**（digest 漏后半句）→ 不构成对 logprob 记录的威胁 |

### paper-2-park (16)

mechanism 簇 13 篇 + 3 篇跨簇：

- **2602.15580**（兼上表 light-cite）/ 2602.20330 Circuit Tracing VLM（"Findings of CVPR 2026" 属实——CVPR 2026 新设 Findings Track，但 = 主会 reject 后 opt-in、无额外评审，引用不可写 "CVPR 2026"）/ 2606.15733 Vernier / 2604.09839（兼 defense）/ 2605.13156 Dual-Pathway / 2606.06333 SASA / **2606.16193 CSAE（park 优先级高：实验模型族 = Qwen3-VL + Gemma-3，与 P79 baseline 重合）** / 2606.00726 LRS（低优先）/ 2604.08524 OV-not-QK / 2606.09131 DPVR（不 light-cite：2602.15580 已覆盖 + 在投未审 + "vision token routing" 撞词危险）/ 2602.22787 Knowledge Attribution（与 2606.02907 绑定为 probe 可信性辩证 pair-cite）/ 2606.19946 GEMS / 2606.22686 Geometry of Refusal（最低优先，可出簇）/ 2602.21704 (=YtWZdwEG5K) Dynamic Multimodal Activation Steering（ICLR 2026 作者自报，OpenReview 侧未能独立确认）
- 2603.17839 How do LLMs Compute Verbal Confidence?（DeepMind 6 作者）— mechanism 主体 park；§6 信号选择可一句带过（它证 verbal confidence 超出 logprob → 预案：P79 用 proxy top-2 logprob 零成本，verbal 需额外轮次且 tool_choice=required 下无自然出口）
- 2606.00251 Capability Self-Assessment — RL self-routing（Self-SOLVE/DELEGATE），learned-router 扩展路线锚；Qwen3 族对齐
- 2606.00376 Deterministic Horizon 的实证部分（horizon-feature 灵感）— paper-1 **drop**（见下）

### background-only (5)

2606.27457 Cluster-Route-Escalate（可被 survey 吸收后 drop）/ 2604.00136 ParetoBandit（"$ 硬预算"实为 per-request 率上限）/ 2505.17616 Runaway-early-exit（**已在 bib** `lu2025earlyexit`，且 Limitations 自证排除 web navigation——§6.8 表引用无需动）/ 2606.17645 Beyond Domains (SkillMigrator)（仅 "policy-facing LLM completions dominate latency and cost" 一句可作 cost-estimand 佐证脚注）/ 2606.17389 VRP（digest "phantom sibling" 过度类比——不操纵格式不做路由，仅共享"grounding≠表现"主题；mentorship 型论文权重低）

### drop (8)

| arXiv | 题 | 为什么 |
|---|---|---|
| 2606.00232 | TIGER | **digest 严重错配**：multimodal generation 幻觉修复，"observation graph" 是图像事实三元组非 UI 观测；其 "AXTree 改 graph 表示" 建议 = 冲刺期推翻 42-condition 预注册 + 6-cell 统计设计，明确否决 |
| 2606.00376 | Deterministic Horizon | digest "作 §6 router 理论依据"否决：ICML 2026 仅 metadata 自称不可核实；理论有参数循环红旗（L_eff 反推再自证）；域 = deterministic state-tracking，对表示选择无蕴含，引之引火 |
| 2606.04627 | MIRAGE mobile agents | 推理 token 压缩轴 + 移动域纯截图，与观测表示路由零交集（与 P79 的 mirage effect 同名不同物） |
| 2606.18671 | HANSEL | HCI 人类验证界面，全轴零交集 |
| 2606.18668 | EARS | 多 agent abstention 协议；P79 无 N/A 出口（已知局限），引之反而招问 |
| 2604.03527 | Topaz | XAI 模型路由 system demo，无定量任务效果 |
| 2606.20544 | Calibrated MoE (ICML 2026 ✓ journal-ref) | MoE 内部路由 + 小分类器尺度，双重 scope 外（survey 同样把 MoE 划界外） |
| 2606.24420 | Beyond Logprobs (ExtractConf) | digest "与三臂同构"牵强：dual-call ensemble+deferral ≠ pre-execution 路由；双臂全跑 = 成本翻倍 + stateful web env 需 reset 隔离。Hunter/Mapper disagreement 思路留 paper-2 idea pool |

（Score-1 awareness 级条目未列入本轮 45 篇：digest 已自标 OUT-OF-SCOPE，维持原判。）

---

## 1. LIVE adds 要点（引用纪律）

1. **Survey gap 引文是本轮最大收获**（2603.04445）：§1 "现有 efficiency routing 全在 model/compute 轴" 的 claim 从此有权威 survey 背书——其六 paradigm（difficulty/preference/clustering/RL/UQ/cascading）无一涉 observation representation。§6.8 总起句可直接挂。
2. **§6.8 表新增候选行**：LazyMCoT（difficulty-triggered visual-input escalation | compute(input detail) 轴）、BoundaryRouter（LLM-vs-agent 执行策略轴）、Adaptive Re-Ranking（IR 管线档位轴 + oracle-gap 叙事）。三行都是"填新格子"，不与现有 FrugalGPT/RouteLLM/CSCR/PANDO/DMR/ReVision/AVR/WebChallenger 重复。
3. **标题纪律**：BoundaryRouter / SkillMigrator / LazyMCoT 均为系统名 ≠ 论文标题，bib 条目必须用真题（"Learning Agent Routing From Early Experience" / "Beyond Domains: ..." / "Focus When Necessary: ..."）。
4. **措辞红线**：Sema 引用句锚 "structured text is load-bearing / transport-layer efficiency"，禁 "no-image no-drop"（拖回 cheaper-SoM 禁忌 frame）。2602.15580 引作 motivation 非 evidence。

## 2. Reviewer-defense 要点

### 2.1 ⚡ 唯一 pre-deadline 行动项：routing-signal AUROC 加固（源 2606.22864）

该文与 P79 同域（multimodal CUA + Mind2Web + Qwen2.5-VL + hidden-state 信号），审稿撞见概率不低。其 C1 诊断可迁移成对 LIVE AUROC claim 的攻击："AUROC 是否主要由 site/task-template/长度这类 trivial 协变量驱动？VWA task 模板在 train/test 重复 → 'learnable' 可能部分是 'memorizable'"。**廉价防线（现有数据可算）**：
- (a) §6 并排报告 **scalar-covariate 基线**（仅 site + length + 模板特征的 logistic）vs learned router AUROC；
- (b) **template-disjoint split** 敏感性一行。
同时 defuse 泄漏质疑与 2606.22864 型攻击。paper-2（probe 复活时）该文必引。

### 2.2 THREAT 定性双向修正

- Plan-Then-Execute (2605.14290)：THREAT ↓（安全 position paper 零实证；PTE executor 仍消费观测）→ 转化为 §2 架构频谱 contrast pole。
- Non-Surjective (2604.09839)：THREAT → **asset**（phantom 是 prompt-level on-manifold 干预）。
- 2606.22864：THREAT 维持但精确化——打不到预测效用型 AUROC 本身，打的是协变量混淆，可低成本 defuse（§2.1）。

## 3. Digest 质量反馈（cron 侧改进项，延续 06-15 §4）

1. **零幻觉**：45/45 ID 真实、数字类声称基本准确（2605.29397 甚至保守：100× 实为 290×）。web-search 重建模式下这个可靠度超预期，但**核验仍必须**——正是核验抓出下面 2-4。
2. **§5-anchor 漂移未修**（06-15 已报，仍在犯）：05-29 → 07-05 每期都有 Score-3 锚 "§5 mechanism"，而 §5 自 2026-05-14 已 shelved。对 paper-1 价值持续系统性高估。
3. **新发现——类比通胀**：digest 倾向把主题呼应写成结构同构。实锤 4 例：VRP "几乎是 phantom space 实证 sibling"（仅主题呼应）、ExtractConf "与三臂 disagreement 同构"（ensemble-deferral ≠ 路由）、Deterministic Horizon "作 §6 理论依据"（域错配+理论红旗）、Adaptive Re-Ranking "与 drop-one oracle 干净同构"（full oracle ≠ drop-one）。
4. **域读错 1 例**：TIGER（multimodal generation 幻觉修复被读成 web 观测路由）。
5. **仍无 bib-dedup**：AVR/MIRAGE/Scaffold/SteerMoE/early-exit/CSCR 等已 bib 论文反复被 surface。
6. venue 核验结果：ICML 2026 ✓ (2606.20544 journal-ref) / EMNLP 2025 Findings ✓ (2505.17616 comment) / CHI 2026 HCXAI Workshop ✓ (2604.03527 comment) / **CVPR 2026 Findings Track 真实存在**（2602.20330；但 = opt-in 无额外评审 track，引用需注明）/ ICLR 2026 (2602.21704) 作者自报未独立确认 / ICML 2026 (2606.00376) **不可核实**。

## 4. Disposition 行动表

**执行状态 (2026-07-05, user 拍板"全做")**: ✅ 13 条入 `paper.bib` (99→112: 12 LIVE + `li2026aucnotenough`); ✅ 合稿 `aaai27_main.md` §2 两段 + §6 covariate-baseline 句 (`<TBD>` 槽); ✅ 长版 section2 (§2.2 survey/三邻居/PTE + §2.4 enomoto/SDO/sema/wu/chowdhury) + section3 (§3.4 factorial 先例) + section6 (§6.8 survey opener + 表 3 行 + 差异化段 + vardanyan); ✅ paper_planning §14.1 +4 defense 行; ✅ NUMBERS_TODO §0 +1 槽位; ✅ digest skill patch → `raw/_digest_skill_patch_2026-07-05.md` (prompt 本体在 Hermes/quark 侧, 需 user 微信端粘入); ⏳ AUROC 加固数字 = `router_covariate_baseline.py` 分析 agent 在跑。**draft prose (aaai27_main + section2/3/6) 未 commit, 待 /stress** (CLAUDE.md auto-trigger; 06-15 先例同)。

| 动作 | 对象 | 优先级 | 落点 |
|---|---|:--:|---|
| **AUROC 加固**：scalar-covariate 基线 + template-disjoint split | （数据侧，非引用） | **高 ⚡** | §6 router 结果表 + 预案 |
| 加 bib + §1 gap 句 + §6.8 总起 | `moslem2026routingsurvey` | **高** | §1 / §6.8 |
| 加 bib + §6.8 表 3 新行 | `wang2026lazymcot` / `wang2026boundaryrouter` / `genc2026adaptivereranking` | 中 | §6.8 |
| 加 bib + §2 邻居句 ×4 | `piet2026plan` / `enomoto2026observation` / `gaur2026sdo` / `chowdhury2026miragedetect` | 中 | §2 |
| 加 bib + 低权重 light-cite ×4 | `meng2026sema`(措辞红线) / `vardanyan2025browser` / `wu2026visionbecomeslanguage`(2602.15580) / `senoglu2026medvqacalib`(可选) | 低 | substrate 段 / §1 / §3 |
| 预案条目 ×6 | 2606.22864 / 2605.02241 / 2605.18796 / 2604.09839 / 2606.19868 / 2606.17234 | 中 | reviewer 预案 (paper_planning) |
| paper-2 park 登记 | §0 park 表 16 篇 | paper-2 | 复活时取用 |
| digest skill 改进 3 条 | §5-anchor 修正 + bib-dedup gate + 类比降温 prompt | 低 | cron sidecar prompt |

**本轮全部未动 `paper.bib` / draft prose** — 12 篇 LIVE 候选中除 survey 外均为 related-work 权重，建议 user 决定采纳范围后一次性入 bib + 过 /stress。
