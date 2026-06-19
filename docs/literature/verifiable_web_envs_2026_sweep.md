# Lit Sweep — 2026 可验证 Web/GUI 环境合成线 (4 papers + MobileGym)

**Found via:** MobileGym (arXiv 2605.26114, 见 [[mobilegym_paper_note]]) bibliography 挖掘, 2026-06-10。4 篇 arXiv ID 经 API 核验, 各由独立 sub-agent 精读全文后汇总。
**Use case for our project:** **bug paper related work 的骨架素材 (primary)** + paper-1 两个锚点 (secondary)。这条线 = "社区已承认 verifier bottleneck, 走合成新环境路线"; 我们 bug paper 走互补路线 = 审计存量 benchmark 的 evaluator FP。

---

## 0. 集群定位轴: realism × judge determinism

| Paper | 机构 | 环境 | Judge | 对 P79 最有用的一个数字 |
|---|---|---|---|---|
| **WebGym** (2601.02439v6) | Microsoft AI Frontiers + UIUC + CMU | 127,645 **真实 live 站**, 292k tasks | GPT-4o rubric (非确定) | judge 人工验证仅 **n=80**; rubric 提 precision ~73→93% 伤 recall ~96→85%, 自承不可根除 |
| **InfiniteWeb** (2601.04126v2) | PKU + 南大 + MSRA | 合成 localStorage 静态站, $1.93/站 | 确定性 JS evaluator + instrumentation 防篡改, dense 0-1 | purpose-built evaluator 仍 **~5% 错误率** (人工核 95/100, n=100, 95% CI≈[88.7, 98.4]) |
| **AutoWebWorld** (2602.14296) | HKUST-GZ + DeepWisdom + PKU + Mila | 29 个 FSM→Vue 合成站 | FSM goal predicate, BFS 枚举, "无 LLM judge" | 11,663 条 verified traj **dedup 后仅 1,215 distinct**; 自己评 WebVoyager 仍用 Gemini-3-Flash LLM judge |
| **OpenApps** (2511.20766) | Meta FAIR + NYU + Brown | 6 个玩具 FastHTML app + YAML 参数化变体 | 全 state deterministic indicator | fixed-app 测的 SR std 低估部署方差 **>2×** (Qwen2.5-VL: within 16.8 vs across-variation 32.0) |
| MobileGym (2605.26114) | 中科院自动化所 + PKU + CUHK | 浏览器内 React 类 Android, 28 apps | state-diff 确定性 judge | VLM judge 10.2% 误判 (12/118, GPT-5.4 重判同 12/118) |

Realism 递减 / judge 确定性递增: WebGym → InfiniteWeb → AutoWebWorld。OpenApps 正交 (测环境扰动下 SR 方差), MobileGym 是 mobile 域同款思路。

## 1. Bug paper 映射 (primary — related work 骨架现成)

叙事线: **"verifier bottleneck 已被社区承认 → 一条路线 = 合成可验证新环境 (AutoWebWorld/InfiniteWeb/MobileGym), 另一条 = 真实站 + LLM judge 并承受测得的 precision/recall 损失 (WebGym) → 我们走互补第三路线: 审计存量高引用 benchmark 的 deterministic evaluator 实现 FP"**。

支撑弹药 (每条都有精确出处):
- **即使 purpose-built 也躲不掉**: InfiniteWeb 自动 evaluator ~5% 错误 (n=100); WebGym judge 验证仅 n=80 且 recall regression 自承 "almost impossible to eliminate"; MobileGym VLM judge 10.2% (n=118) → "eval validation 普遍 underpowered" 论据。
- **OpenApps 原句**: "when reporting agent failure, authors typically rely on anecdotal evidence alone" → 直接支撑我们 systematic FP audit 的 motivation。
- **OpenApps reward 设计讨论**: human-trajectory rewards 过度限制 ("many roads lead to Rome") vs change-based checks 可被 reward hacking (引 Zhu et al. 2025) — VWA evaluator FP 审计的天然锚点。
- **AutoWebWorld Table 1**: external verifier 成本 $0.15-1.00/条 — verifier 经济性论据。
- **自我矛盾例证**: AutoWebWorld 批判 LLM judge 却用 Gemini-3-Flash 评 WebVoyager — 问题普遍性的活例。
- 共同上游引文: **Xue et al. "An Illusion of Progress?" (2504.01382)** — WebGym/InfiniteWeb/AutoWebWorld 三篇都引, eval-reliability 同脉络, 我们 bib 应有 (待核)。

## 2. Paper-1 (routing) 映射 (secondary — 两个锚点)

1. **OpenApps GPT-4o vision-only SR 0/0/0 vs +AXTree 82-90** (15 tasks×3 seeds×8 variations): 极端 observation-mode 效应, 文本表征对视觉变体 robust — 与 phantom space "skip annotated image" 论点同向, related work 级佐证。
2. **WebGym 作 gap 锚**: 最大规模训练环境也把 obs 表征当 model 能力的固定函数 (trained agent 纯 screenshot+坐标; SoM 仅给 GPT-4o 当能力补偿), 不研究 mode 间成本-成功率权衡 → 反衬 paper-1 把 obs mode 当一级变量的定位。Thinking-vs-Instruct 2× 输出长度的 performance-efficiency 取舍可呼应 cost-aware 动机。
3. **Limitations caveat (来自 OpenApps)**: 我们 mode 间 Δ 在 fixed VWA clone 上测得; OpenApps 显示 fixed-clone 方差低估部署方差 >2×, 且变体敏感度跨模型异质 (German 只打 Kimi-VL) — (a) FE pooled 结论应措辞为 "within this environment instantiation"; (b) 异质性反而支持我们 per-(site,model) cell 分层设计。

## 3. Hostile-reader caveats (引用前自检)

- **WebGym**: GPT-4o 三重闭环 (rubric 生成者=judge=被比 baseline), 26.2→42.9 由同一 judge 度量, reward-hacking 不可排除; live 站不可复现。
- **InfiniteWeb**: 正文 vs 图表数字打架 (去 TCTDD 80.6 vs 82.6); baseline compute 不对齐; localStorage 静态站 ≤12 页无 auth, 与 VWA 真 Magento/PostMill 栈差距大。
- **AutoWebWorld**: WebVoyager 数字不可比文献 (9/15 站 + 15 步 + 自选 judge, GPT-5.1 仅 18.96 被系统性压低); "no LLM judge" 只覆盖状态转移层, query↔轨迹语义对齐未验证; "Infinite" vs 1,215 distinct。
- **OpenApps**: 15 个 toy task, min/max 是小 n 极值; arXiv v1 缺附录 (正文引 Tables 4-13 不在 PDF 内); **其 "reliability" = SR 对环境扰动的方差, ≠ 我们 bug paper 的 evaluator correctness — 不同 estimand, 引用必须区分**。
- 全部为 preprint (OpenApps 2025-11, 其余 2026-01/02), 引用标注 preprint 状态。

## 4. Disposition

- **Bug paper**: §1 列出的叙事线 + 弹药直接进 related work 草稿; AutoWebWorld/InfiniteWeb/WebGym/OpenApps/MobileGym 五连引成 cluster。
- **Paper-1**: OpenApps vision-only 锚点 + WebGym gap 锚 → related work 候选; OpenApps 方差 caveat → limitations 候选。均待 prose round 时定。
- **Xue 2504.01382 已核 (2026-06-10): paper.bib 无此条** — bug paper / paper-1 eval-FP 段落动笔时补入。
- 均未入 `paper.bib`。Sub-agent 精读 digest 原文较本 note 更细 (含训练超参/全部数字), 如需可重新派 agent 深挖单篇。
