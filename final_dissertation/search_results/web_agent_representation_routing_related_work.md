# 文献综述结构化梳理：Web Agent 表征路由、模型路由、选择性预测与自适应推理

## 研究问题

> Web agent（能看网页、选动作、在浏览器里执行、循环直到完成任务的 AI 系统）每一步都要把当前页面“喂”给模型。  
> 页面表征可以是便宜的结构化文本，例如 accessibility tree，也可以是昂贵的多模态输入，例如带标注框的截图 + 完整结构文本。
>
> 本文关心两个问题：
>
> 1. 昂贵表征在什么时候才真正必要？
> 2. 这种“何时必要”能否仅用**决策前就能获得的廉价信号**预测？

本文献表不以“论文做了什么”为主要组织方式，而是严格区分：

- 论文真正优化的**决策变量**是什么；
- gating / routing 信号是在昂贵路径执行**之前**还是**之后**才能得到；
- 该工作为何仍未回答：  
  **“给定同一个 web agent，在当前 step 到底应该喂 cheap representation 还是 rich representation？”**

---

## 簇 1：Web / GUI Agent 的页面表征

| 簇 | 标题 | 一作 | 年 | arXiv ID 或 DOI | 是否 peer-reviewed（会议/期刊名） | 它优化的是什么 | 因此它为什么不回答我的问题 |
|---|---|---|---:|---|---|---|---|
| 1 | Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V | Jianwei Yang | 2023 | `arXiv:2310.11441` | 未核实正式发表；arXiv preprint | **决策变量是视觉 prompt 的编码方式**：是否给图像区域叠加可引用的 marks / boxes，以提高视觉 grounding。 | 它证明一种**固定的更丰富视觉表征**可能更好，但不判断 web agent 的**某一步**是否值得购买该表征，更没有 cheap-before-decision gate。 |
| 1 | Mind2Web: Towards a Generalist Agent for the Web | Xiang Deng | 2023 | `arXiv:2306.06070` | NeurIPS 2023 Datasets & Benchmarks | **决策变量是保留哪些 HTML 元素**：先做 element filtering，再让 agent 预测动作，以降低 HTML context 成本。 | 它做的是**同一文本表征内部的相关元素裁剪**，不是在 accessibility tree、HTML、screenshot+structure 等不同 observation modalities 之间预测逐步边际价值。 |
| 1 | WebArena: A Realistic Web Environment for Building Autonomous Agents | Shuyan Zhou | 2024 | `arXiv:2307.13854` | ICLR 2024 | **优化/评估对象是 agent policy 的端到端任务成功率**；页面 observation interface 在 baseline 中是配置，而不是在线学习的决策变量。 | 它提供了适合做这类实验的环境，但**没有把“本 step 选哪种页面表征”定义成 action**，因此不存在 representation-value router。 |
| 1 | VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks | Jing Yu Koh | 2024 | `arXiv:2401.13649`; DOI `10.18653/v1/2024.acl-long.50` | ACL 2024 Long | **评估维度是 multimodal web agent 在 visually grounded tasks 上的任务成功与动作能力**，而不是动态 observation selection。 | 它能回答“**哪些任务总体包含视觉需求**”，却不回答固定 agent 在**当前 step** 是否需要视觉，以及能否仅靠当前 cheap observation 在付视觉成本前预测。 |
| 1 | GPT-4V(ision) is a Generalist Web Agent, if Grounded | Boyuan Zheng | 2024 | `arXiv:2401.01614` | ICML 2024 | **决策变量主要是 grounding strategy**：如何把 LMM 的意图映射到网页元素，并比较视觉、HTML 以及组合 grounding。 | 它比较/设计的是**固定 grounding 配置**，没有学习一个仅基于 cheap page state、逐 step 决定是否获取 rich multimodal observation 的 policy。 |
| 1 | WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models | Hongliang He | 2024 | `arXiv:2401.13919`; DOI `10.18653/v1/2024.acl-long.371` | ACL 2024 Long | **决策变量是 multimodal agent 的下一浏览器动作**；视觉+文本 observation 基本作为固定 agent 输入方案，并与 text-only setup 比较。 | 它说明固定使用 multimodal input 可以带来收益，但没有把**“这一 step 是否使用 multimodal input”**本身变成成本敏感决策。 |
| 1 | WebLINX: Real-World Website Navigation with Multi-Turn Dialogue | Xing Han Lu | 2024 | `arXiv:2402.05930` | ICML 2024 | **决策变量之一是保留哪些 HTML 元素**：用 retrieval-style ranker 压缩页面，再基于选中元素、截图和 history 做动作预测。 | 它学习的是**HTML relevance / pruning**；截图等额外模态仍属于既定 downstream observation，不预测“截图相对于 cheap tree 的这一步增量价值”。 |
| 1 | Read More, Think More: Revisiting Observation Reduction for Web Agents | Masafumi Enomoto | 2026 | `arXiv:2604.01535` | 未核实正式 archival venue；arXiv preprint | **实验决策轴直接是 observation representation 与 thinking budget**：比较 compact AxTree、详细 HTML、history / diff 等在不同模型能力与推理预算下的效用。 | **这是本簇最接近本文的问题之一**，但它得到的是“按模型能力/全局 thinking budget 选 observation”的配置规律，而不是对固定 agent 的**每个页面状态**用 cheap pre-decision features 预测 rich observation 的边际收益。 |

### 可核验入口

- `2310.11441` — https://arxiv.org/abs/2310.11441
- `2306.06070` — https://arxiv.org/abs/2306.06070
- `2307.13854` — https://arxiv.org/abs/2307.13854
- `2401.13649` — https://arxiv.org/abs/2401.13649
- `2401.01614` — https://arxiv.org/abs/2401.01614
- `2401.13919` — https://arxiv.org/abs/2401.13919
- `2402.05930` — https://arxiv.org/abs/2402.05930
- `2604.01535` — https://arxiv.org/abs/2604.01535

---

## 簇 2：模型或模态层面的 Routing 与 Cascade

| 簇 | 标题 | 一作 | 年 | arXiv ID 或 DOI | 是否 peer-reviewed（会议/期刊名） | 它优化的是什么 | 因此它为什么不回答我的问题 |
|---|---|---|---:|---|---|---|---|
| 2 | FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance | Lingjiao Chen | 2024 | `arXiv:2305.05176` | TMLR 2024 | **决策变量是调用哪些 LLM、以什么 cascade / 组合调用它们**，目标是在回答质量与 API 成本之间优化。 | 它 route 的是**模型/API**而不是同一个 agent 的输入表征；其 cascade 还可以根据前级模型输出继续决策，而本文的目标 gate 必须在 rich representation 推理前成立。 |
| 2 | AutoMix: Automatically Mixing Language Models | Pranjal Aggarwal | 2024 | `arXiv:2310.12963`; DOI `10.52202/079017-4164` | NeurIPS 2024 | **决策变量是是否从小模型升级至更大的模型**，依据小模型答案的 self-verification / confidence，并优化成本–性能。 | 它不但切换的是**模型而非 observation**，而且 gate 信号需要**先生成小模型答案**才能获得；这不等于“昂贵页面/主 agent 推理之前就能获得的 cheap page signal”。 |
| 2 | Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing | Dujian Ding | 2024 | `arXiv:2404.14618` | ICLR 2024 | **决策变量是 query 发给 small 还是 large LLM**，router 根据预测的 query difficulty 和目标质量控制 cost-quality trade-off。 | 它的 gate 在结构上很接近**事前预测**，但 action 是**换模型**；没有固定模型后预测“cheap observation → rich observation”的 counterfactual improvement。 |
| 2 | RouteLLM: Learning to Route LLMs with Preference Data | Isaac Ong | 2025 | `arXiv:2406.18665` | ICLR 2025 | **决策变量是 strong / weak LLM 的选择**，训练目标利用模型间 preference data 学习质量–成本 routing boundary。 | 它需要的是**不同模型的相对表现标签**，并改变 model identity；本文的 counterfactual 是同一模型/agent 在两种页面表征下的结果差异。 |
| 2 | RouterDC: Query-Based Router by Dual Contrastive Learning for Assembling Large Language Models | Shuhao Chen | 2024 | `arXiv:2409.19886`; DOI `10.52202/079017-2120` | NeurIPS 2024 | **决策变量是从多个 LLM experts 中选哪个模型**，用 query–LLM contrastive representation 学习适配关系。 | query embedding 可以是事前 cheap signal，但它预测的是**哪个模型擅长这个 query**，不是哪个网页 state 会从额外视觉/结构信息中受益。 |
| 2 | TensorOpera Router: A Multi-Model Router for Efficient LLM Inference | Dimitris Stripelis | 2024 | `arXiv:2408.12320`; DOI `10.18653/v1/2024.emnlp-industry.34` | EMNLP 2024 Industry | **决策变量是把 query 路由给哪个 expert LLM**，目标联合考虑性能、成本和推理效率。 | 它仍然把异质性放在**模型集合**上，而本文的异质性来自固定 agent 面对同一 state 时的**observation fidelity / cost**。 |

### 可核验入口

- `2305.05176` — https://arxiv.org/abs/2305.05176
- `2310.12963` — https://arxiv.org/abs/2310.12963
- `2404.14618` — https://arxiv.org/abs/2404.14618
- `2406.18665` — https://arxiv.org/abs/2406.18665
- `2409.19886` — https://arxiv.org/abs/2409.19886
- `2408.12320` — https://arxiv.org/abs/2408.12320

---

## 簇 3：基于置信度的弃权、延迟决策与选择性预测

| 簇 | 标题 | 一作 | 年 | arXiv ID 或 DOI | 是否 peer-reviewed（会议/期刊名） | 它优化的是什么 | 因此它为什么不回答我的问题 |
|---|---|---|---:|---|---|---|---|
| 3 | Selective Classification for Deep Neural Networks | Yonatan Geifman | 2017 | `arXiv:1705.08500` | NIPS 2017 | **决策变量是 predict 还是 reject**；通过 confidence threshold 在给定风险水平下最大化 coverage。 | 它决定的是**是否接受已经算出的预测**；confidence 通常来自模型输出，因此已经支付 predictor inference，而不是事前决定是否购买 richer input。 |
| 3 | SelectiveNet: A Deep Neural Network with an Integrated Reject Option | Yonatan Geifman | 2019 | `arXiv:1901.09192` | ICML 2019 | **决策变量是 classify/regress 还是 reject**，网络联合优化 prediction loss 与 selection / risk–coverage trade-off。 | 它把“不确定样本”**丢弃/弃权**，没有定义一个付费 acquisition action 来取得 richer representation 后继续由同一 predictor 决策。 |
| 3 | Deep Gamblers: Learning to Abstain with Portfolio Theory | Ziyin Liu | 2019 | `arXiv:1907.00208` | NeurIPS 2019 | **决策变量是预测某类还是下注于 abstention class**，优化给定 coverage 下的 selective accuracy。 | 它的 uncertainty 是预测网络内部产生的，且替代动作是**不预测**而不是“购买更丰富 observation 后再预测”。 |
| 3 | Consistent Estimators for Learning to Defer to an Expert | Hussein Mozannar | 2020 | `arXiv:2006.01862` | ICML 2020 | **决策变量是本模型预测还是 defer 给外部 expert**，以 cost-sensitive surrogate 学习 classifier + rejector。 | 它的 escalation endpoint 是**另一个决策者/expert**，而不是同一 web agent 获取更多页面信息；训练也显式利用 expert decisions。 |
| 3 | How to Fix a Broken Confidence Estimator: Evaluating Post-hoc Methods for Selective Classification with Deep Neural Networks | Luís Felipe Cattelan | 2024 | `arXiv:2305.15508` | UAI 2024 | **决策变量仍是 accept/reject，但优化的是用于排序样本的 post-hoc confidence estimator**，直接从 logits 计算 selection score。 | 它尤其说明本文的差别：gate 输入是**完整 classifier forward 已产生的 logits**，不是付 rich-view 成本之前可取得的页面结构信号。 |
| 3 | Learning-to-Defer with Expert-Conditioned Advice | Yannis Montreuil | 2026 | `arXiv:2603.14324` | ICML 2026 AgenticUQ Workshop poster；非 ICML main conference | **决策变量已经扩展为联合选择 expert 和要提供给该 expert 的额外 advice / information**，并把 acquisition cost 纳入 composite action。 | **这是理论上最接近本文的近邻之一**，但它是一般的一步式 supervised learning-to-defer formulation；没有刻画 sequential browser state、AxTree→screenshot+structure 的固定-agent acquisition，也没有专门要求 router 只依赖当前 cheap web observation 来预测 task-level marginal gain。 |

### 可核验入口

- `1705.08500` — https://arxiv.org/abs/1705.08500
- `1901.09192` — https://arxiv.org/abs/1901.09192
- `1907.00208` — https://arxiv.org/abs/1907.00208
- `2006.01862` — https://arxiv.org/abs/2006.01862
- `2305.15508` — https://arxiv.org/abs/2305.15508
- `2603.14324` — https://arxiv.org/abs/2603.14324

---

## 簇 4：Cost-Aware / Adaptive Inference

| 簇 | 标题 | 一作 | 年 | arXiv ID 或 DOI | 是否 peer-reviewed（会议/期刊名） | 它优化的是什么 | 因此它为什么不回答我的问题 |
|---|---|---|---:|---|---|---|---|
| 4 | Adaptive Computation Time for Recurrent Neural Networks | Alex Graves | 2016 | `arXiv:1603.08983` | 未核实正式会议/期刊版本；arXiv preprint | **决策变量是每个输入在输出之前执行多少个 recurrent computation steps**。 | 它自适应的是**模型内部计算深度**；输入信息已经给定，不涉及是否额外获取更昂贵的外部 observation。 |
| 4 | Adaptive Neural Networks for Efficient Inference | Tolga Bolukbasi | 2017 | `arXiv:1702.07811` | ICML 2017 | **决策变量是计算哪些网络组件、是否 early exit，或选择哪一个成本不同的网络**，全局优化准确率–计算量。 | 它节省的是**network evaluation cost**；其 expensive branch 增加的是模型计算而非页面信息，所以没有估计 rich observation 对固定 agent 的增量信息价值。 |
| 4 | DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference | Ji Xin | 2020 | `arXiv:2004.12993`; DOI `10.18653/v1/2020.acl-main.204` | ACL 2020 | **决策变量是在哪一 Transformer layer 提前退出**，以较小质量损失减少 inference time。 | gate 依据**已经计算出的中间层状态/预测**选择更多内部 layers；它没有在模型推理前选择不同输入 representation。 |
| 4 | DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification | Yongming Rao | 2021 | `arXiv:2106.02034` | NeurIPS 2021 | **决策变量是保留哪些视觉 tokens**，根据中间 feature 的 token-importance 动态 pruning，以优化 FLOPs–accuracy。 | 虽然表面上也是“少看一些视觉信息”，但 gate 必须先**编码图像并取得中间 visual features**；不是从免费/便宜网页结构预测是否值得获取截图。 |
| 4 | Not All Images are Worth 16x16 Words: Dynamic Transformers for Efficient Image Recognition | Yulin Wang | 2021 | `arXiv:2105.15075` | NeurIPS 2021 | **决策变量直接是每张图需要多少视觉 tokens / 多细的表示**：从粗粒度 representation 级联到更细粒度，足够 confident 就停止。 | **这是 cost-aware representation 最直接的视觉类比之一**，但 escalation signal 是运行粗粒度视觉 Transformer 后的分类 confidence；任务也是独立图像分类，而非 sequential web agent 用非视觉 cheap state 在截图 acquisition 前预测收益。 |
| 4 | Confident Adaptive Language Modeling | Tal Schuster | 2022 | `arXiv:2207.07061` | NeurIPS 2022 | **决策变量是每个输入/生成 timestep 应执行到哪一层**，confidence-controlled early exit 分配不同计算量。 | 它动态购买的是**更多模型层计算**，且 confidence 来自已有中间 computation；没有外部 observation acquisition 或 cheap-page-only gate。 |
| 4 | Token-Budget-Aware LLM Reasoning | Tingxu Han | 2025 | `arXiv:2412.18547`; DOI `10.18653/v1/2025.findings-acl.1274` | Findings of ACL 2025 | **决策变量是给每个问题分配多少 reasoning tokens**，依据预测的问题复杂度平衡 token cost 与答案质量。 | 它说明可以用**事前低成本特征预测所需资源量**，但资源是 output-side reasoning budget；没有预测额外 webpage modality / representation 对固定 agent 的 marginal utility。 |

### 可核验入口

- `1603.08983` — https://arxiv.org/abs/1603.08983
- `1702.07811` — https://arxiv.org/abs/1702.07811
- `2004.12993` — https://arxiv.org/abs/2004.12993
- `2106.02034` — https://arxiv.org/abs/2106.02034
- `2105.15075` — https://arxiv.org/abs/2105.15075
- `2207.07061` — https://arxiv.org/abs/2207.07061
- `2412.18547` — https://arxiv.org/abs/2412.18547

---

# GAP

现有 web-agent 工作已经表明，**observation representation 会显著影响 grounding、上下文成本与任务表现**，而且最新工作进一步显示最优表示会随模型能力、页面表征和推理预算变化；与此同时，LLM routing、selective prediction 与 adaptive-inference 文献已经证明，可以按样本动态分配模型、专家或内部计算资源。

**然而，这些方向尚未共同回答一个更细粒度的问题：在固定 web agent 的每个交互 step，能否仅利用获取昂贵 observation 之前已经存在的低成本页面信号，预测 richer multimodal representation 相对于 cheap structured representation 的边际任务价值。** 因此，本文将 **observation fidelity 本身作为逐步 routing action**，研究何时视觉/高保真页面信息确有必要，以及这种 necessity 是否可以由 **pre-decision cheap signals** 可靠预测。

---

# 与本文最接近、需要重点防撞的两篇工作

## 1. Read More, Think More: Revisiting Observation Reduction for Web Agents

**ID：** `arXiv:2604.01535`

这篇工作意味着本文不宜把 novelty 写成：

> “Previous web-agent work has not studied adaptive observations.”

这个说法过宽。

更稳妥的区别是：

> Prior work studies how observation representations interact with model capability and reasoning budget at the configuration level, whereas we ask whether observation fidelity can be routed **online at each browser step**, for a **fixed agent**, using only **cheap signals available before acquiring the richer observation**.

也就是说，novelty 应钉在：

- step-level；
- fixed agent；
- pre-decision；
- cheap observable features；
- prediction of **marginal value of richer representation**。

---

## 2. Learning-to-Defer with Expert-Conditioned Advice

**ID：** `arXiv:2603.14324`

这篇工作意味着本文也不宜声称：

> “No prior work studies costly information acquisition or adaptive provision of additional information.”

这个范围同样过宽。

更精确的区别是：

> General learning-to-defer formulations can jointly choose an expert and additional information supplied to that expert, but they do not study sequential web interaction in which a fixed browser agent must decide, at each state, whether a richer page representation is worth acquiring based only on signals already present in the cheap representation.

---

# 概念定位：四个文献簇与本文之间的差别

可以把四个方向压缩成下面四种 routing 变量：

| 文献方向 | 在线决策变量 | expensive branch 增加了什么 | gate 常见来源 | 与本文的核心差别 |
|---|---|---|---|---|
| Web / GUI representation | 页面如何编码、哪些元素保留 | 更多/不同网页信息 | 通常固定配置，或 representation 内部 relevance | 通常没有逐 step 学习“是否购买 richer representation” |
| LLM routing / cascade | 调哪个模型 | 更强模型能力 | query features、前级模型输出、confidence | 换的是**模型**，不是同一 agent 的输入表征 |
| Selective prediction / defer | 接受、拒绝或交给专家 | 不预测 / 换决策者 | logits、confidence、classifier output | gate 往往在一次主要 inference **之后**，或 endpoint 是 expert |
| Adaptive inference | 算多少层、多少 token、多少视觉 token | 更多内部计算 | 中间表示、confidence、difficulty | 买的是**compute**，不是新的外部 webpage information |
| **本文** | **当前 step 用 cheap 还是 rich page representation** | **额外视觉/结构页面信息** | **只允许使用 rich acquisition 之前的廉价网页信号** | **固定 agent、逐步 observation routing、预测 richer view 的 marginal utility** |

---

# 可以直接用于 Related Work 的收束段落

Existing work has explored web-page representations, model routing, selective prediction, and adaptive computation largely as separate problems. Web-agent studies show that accessibility trees, HTML, screenshots, and multimodal grounding can induce substantially different cost–performance profiles, while routing and adaptive-inference methods dynamically allocate models or computation according to predicted difficulty or confidence. However, these lines of work do not directly address **step-level observation routing for a fixed web agent**: whether the marginal value of acquiring a richer page representation can be predicted using only low-cost signals available **before** that acquisition. We therefore treat observation fidelity itself as the routing decision and study when richer multimodal context is genuinely necessary, and whether that necessity is predictable in advance.

---

# UNCERTAIN

以下条目没有从主表删除，是因为其 **arXiv ID / 标题本身有把握**，但正式发表状态或版本对应关系在本轮检索中没有达到同等确定性，因此不应擅自写成正式 archival publication。

### Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V

- arXiv ID：`2310.11441`
- 标题与 ID 已核实。
- 本轮未确认可信的正式 peer-reviewed proceedings 版本。
- 因此应按 **arXiv preprint** 引用，除非后续另行核实正式版本。

### Read More, Think More: Revisiting Observation Reduction for Web Agents

- arXiv ID：`2604.01535`
- 标题与 ID 已核实。
- 截至 2026-08-10，本轮未确认正式 archival venue。
- 因为它与本文问题高度接近，**即使只是 preprint 也应该在 related work 中认真处理**。

### Adaptive Computation Time for Recurrent Neural Networks

- arXiv ID：`1603.08983`
- 标题与 ID 已核实。
- 本轮未确认独立的正式 conference / journal proceedings 版本。
- 主文中如引用，可直接引用 arXiv 版本，不应自行填写会议名。

### Learning-to-Defer with Expert-Conditioned Advice

- arXiv ID：`2603.14324`
- arXiv 主版本标题为 *Learning-to-Defer with Expert-Conditioned Advice*。
- 可核到 ICML 2026 AgenticUQ Workshop 版本，但这**不是 ICML main conference paper**。
- 因此 related work 中应避免写成 “ICML 2026 paper”；更安全的表述是 **ICML 2026 AgenticUQ Workshop / arXiv preprint**。

---

# 最终定位

这批文献共同说明，本文真正需要守住的 novelty 不是笼统的：

> “动态选择更便宜的输入。”

而是一个更窄、也更可辩护的研究问题：

> **Given a fixed web agent and a cheap observation already available at the current browser step, can we predict—before acquiring or processing the expensive representation—whether richer page information will improve the agent enough to justify its additional cost?**

对应到决策变量：

\[
a_t \in \{\text{cheap representation},\ \text{rich representation}\}
\]

router 只能看到：

\[
x_t^{\text{cheap}}
\]

而真正需要预测的是类似：

\[
\Delta U_t
=
U(a_t=\text{rich}\mid s_t)
-
U(a_t=\text{cheap}\mid s_t),
\]

其中 \(U\) 可以进一步展开为成功概率、动作正确率或长期任务价值与推理成本之间的权衡。

因此，与传统 model routing 最本质的区别不是“也用了一个 classifier”，而是：

**router 所预测的 counterfactual 是同一个 agent 在同一个 browser state 下，由不同 observation fidelity 导致的增量价值，而不是不同模型之间的能力差。**
