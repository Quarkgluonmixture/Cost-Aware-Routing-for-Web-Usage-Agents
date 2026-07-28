---
title: "Web Agent 研究笔记：MAG、表征路由、Grounding 与可复现性"
type: research-agent-brief
status: working-note
updated: 2026-07-28
primary_topic: web-agent
subtopics:
  - observation representation routing
  - action grounding routing
  - Set-of-Marks
  - sequential element identifiers
  - web-agent reinforcement learning
  - run-to-run reproducibility
  - oracle headroom
  - evaluator uncertainty
primary_paper:
  title: "MAG: A Web-Agent Benchmark and Harness for Multimodal Action and Guide Generation"
  arxiv: "2607.10079"
  preferred_version: "v3"
  versions: {v1: 2026-07-11, v3: 2026-07-16}
  authors: "Gan, Wei, Liang, Cai, Zhang, Ni"
verification:
  date: 2026-07-28
  method: "arXiv API (id_list) + v3 full-text HTML fetch, 逐数字比对"
  status: "所有引用数字已对 v3 正文/附录核实通过; 引用标记已从 v2 升级为 v3"
project_context: "P79 — cost-aware representation routing for web agents"
intended_reader: "Research Agent"
---

# Web Agent 研究笔记：MAG、表征路由、Grounding 与可复现性

## 0. 给 Research Agent 的使用说明

这份笔记不是普通论文摘要，而是一个面向持续研究的知识节点。阅读时请严格区分三类内容：

1. **论文直接支持的事实**：MAG 的任务定义、实验设置、数值结果和作者明示的限制。
2. **基于论文和项目经验得出的分析**：例如它对 representation routing、sequential ID、run-to-run noise 的启示。
3. **尚待验证的研究问题**：例如 AXTree 与 DOM 标识的稳定性、不同 grounding 的重复运行方差、WebGym 和 WebArena-Verified 的最新变化。

后续扩展时不要把第 2、3 类内容改写成“论文已经证明”。任何新增结论都应绑定来源、版本、日期与适用范围。

> **2026-07-28 核实轮次说明**：本笔记原版所有数字引 v2 且未经核对。已完成一轮 v3 全文比对（方法见 §4.1），结果：**所有数字通过**，但补入了 4 项原版缺失的重要事实——作者的 CI 与显著性判断（§6.1）、三-judge evaluator 交叉验证（§9.2a）、长程能力断崖（§10.7）、usefulness check 的单人评分性质（§10.5）。§11 因与项目台账冲突已整节重写。**凡标 ✅ 的条目可直接引用；未标的仍属第 2/3 类。**

---

# 1. 当前最重要的结论

## 1.1 Web Agent 的表征不应被视为固定输入

“信息更多”不等于表现更好。更多模态可能同时带来：

- 更高 token、图像推理和服务成本；
- 更长的端到端延迟；
- 文本与视觉信息竞争；
- text-over-vision 或 visual distraction；
- 不同 observation 中元素标识不一致；
- 更复杂的 action contract；
- 更高的 grounding 与执行错误面；
- 更难控制的运行时随机性。

因此，更合适的研究抽象是：

> **网页观测表征是一种可以按任务、模型与运行状态动态路由的资源，而不是所有任务固定使用的输入模板。**

## 1.2 Routing 至少应拆成两个维度

目前应明确区分：

1. **Observation representation routing**  
   选择 Agent 如何观察网页，例如 DOM、AXTree、截图、SoM、P-text、P-prompt、P-SoM。

2. **Action grounding routing**  
   在 observation 基本不变时，选择 Agent 如何指向动作目标，例如 sequential mark index、browser node ID、像素坐标或结构化 API/tool call。

MAG 主要研究第二类，不应被误写成 screenshot-vs-DOM 或 text-vs-vision 的 observation routing 论文。

## 1.3 MAG 对 P79 的最大价值

MAG 提供了三条与 P79 高度相关的外部证据：

- **模型对 grounding 的偏好可以很大，但可证的只有一例**：Gemini 3.5 Flash 偏好 SoM +13.8pp（CI [+6.9, +21.3]，显著），而 GPT-5.5 与 Claude 的 CI 都跨零。支持 model-specific routing 的**方向**，但不要写成"三个模型各有偏好"（详见 §6.1）；
- SoM 使用 representation-local sequential index，而非 Chromium 内部 AX node ID —— 这不是给 P79 的建议，而是**对 P79 已锁决策（AMENDMENT_07，2026-05-25）的事后外部佐证**（详见 §11.1）；
- MAG 用单次 solved-set union 推导 routing headroom，却没有校准 execution-level run-to-run variance —— 这既是 P79 reproducibility audit 的最新案例，**也是一把指向 P79 自身 H3 hero 的刀**（详见 §11.5）。

## 1.4 最重要的方法学警告

单次运行中两个 mode 的 solved-set overlap 小，并不自动说明两种 mode 存在稳定、可学习的互补能力。它也可能来自：

- trajectory stochasticity；
- provider nondeterminism；
- page/environment variation；
- evaluator variation；
- grounding interface 的噪声差异；
- browser element identifier churn。

因此：

> **Single-run oracle union 是 routing opportunity 的候选上界，不是 stability-calibrated routing estimand。**

---

# 2. Web Agent 方向的待整理知识节点

以下是当前笔记中尚未展开、但应由 Research Agent 继续补全的主题。

## 2.1 WebGym

待研究：

- WebGym 的任务目标、环境规模、训练数据生成方式和 RL 接口；
- 它解决的是 environment scarcity、trajectory collection，还是 evaluator/benchmark coverage；
- 与 WebArena、BrowserGym、Online-Mind2Web、WebGames 的关系；
- 是否支持真实网站、镜像站点、可重置状态或大规模并行 rollout；
- 最新版本、代码仓库、训练复现和失败案例。

## 2.2 WebArena-Verified

待研究：

- 它修订了 WebArena 的哪些任务、evaluator 和成功条件；
- 是否减少 LLM judge、substring matching 和 brittle programmatic evaluators；
- 对旧 WebArena leaderboard 的可比性影响；
- task-level 修订、人工复核和 evaluator determinism 的具体实现；
- 最新 release、任务数、覆盖网站与公开代码状态。

## 2.3 Agent state tracking

一个更成熟的 Web Agent 应显式记录：

- 当前与历史 URL；
- tabs 与 active tab；
- 表单状态；
- 页面变化；
- 已完成与未完成子目标；
- 页面是否 unchanged；
- navigation history；
- write action 与 irreversible action；
- 当前 action grounding contract。

需要研究：这些状态应该由 LLM 自己总结、由 harness 显式维护，还是由两者混合维护。

## 2.4 AXTree、DOM 与元素标识稳定性

已知风险：AXTree 中暴露的 browser-internal node ID 可能受浏览器无障碍对象构建时序影响。

待研究：

- DOM node 标识是否也存在类似的生命周期和重建问题；
- BrowserGym、Playwright、CDP 与 WebArena 分别如何生成 element identifiers；
- 哪些 ID 是文档内稳定、跨 observation 稳定、跨 reload 稳定或仅单 snapshot 有效；
- sequential ID、DOM path、backendNodeId、AXID、CSS/XPath、bbox fingerprint 的稳定性比较；
- element-ID churn 对不同模型、不同 observation mode 和不同 action contract 的影响。

## 2.5 图像推理的延迟与隐性代价

图像 observation 的成本不止是图像 token：

- screenshot 获取与编码；
- 图像上传；
- 多模态 prefill；
- 更长服务队列；
- 视觉 grounding；
- 图像与文本对齐；
- marked screenshot 的渲染；
- context competition；
- 重复截图和历史截图带来的上下文膨胀。

需要区分：model latency、environment latency、observation preparation latency 和 end-to-end task latency。

---

# 3. 可用于 Representation Router 的信号

## 3.1 任务前信号

可在 episode 开始前获得：

- intent 文本；
- 网站或页面类型；
- 是否包含 reference image；
- 历史任务难度；
- 页面复杂度；
- 是否涉及写操作或不可逆操作；
- 是否明确需要多模态信息；
- 是否需要比较图片、颜色、布局或图标；
- 任务模板或 task family；
- 模型身份与能力档位；
- **预计任务长度 / gold trajectory 步数**（MAG §10.7 证据：>8 gold step 时小模型全线崩到 <3%，而 API 模型维持 20-38% —— 这是一个与 grounding **正交**的失败轴，representation routing 触及不到，可能需要路由到"换模型"而非"换表征"）；
- 预计成本与 latency budget。

## 3.2 运行中信号

可在 trajectory 中动态更新：

- page unchanged；
- repeated action；
- URL revisit；
- action diversity；
- no-progress streak；
- grounding confidence；
- verbalized confidence；
- context usage；
- observation length；
- loop / cycle indicators；
- tab churn；
- action parse failures；
- page load / busy state；
- visual-content requirement 是否突然出现；
- 当前表征是否无法提供目标信息。

## 3.3 白盒信号

仅在可访问模型内部状态时使用：

- next-token entropy；
- action margin；
- mode-selection margin；
- hidden-state probes；
- tool-selection representations；
- layer-wise confidence；
- attention allocation between text and image；
- representation-specific activation signatures。

## 3.4 更完整的路由形式

长期目标不应只选择 observation mode，而应联合选择：

\[
r(q,s,m,b) \rightarrow
(\text{observation},\text{grounding},\text{model},\text{tool},\text{fallback})
\]

其中：

- \(q\)：任务 intent；
- \(s\)：当前网页与 trajectory state；
- \(m\)：可用模型能力；
- \(b\)：成本、延迟与安全预算。

可能输出包括：

- P-SoM + sequential mark ID；
- screenshot + coordinate；
- AXTree + native element ID；
- DOM + structured browser action；
- API/tool call；
- human confirmation；
- abstain 或 fallback。

---

# 4. MAG 论文定位

## 4.1 基本信息

- 标题：*MAG: A Web-Agent Benchmark and Harness for Multimodal Action and Guide Generation*
- 作者：Chengguang Gan, Hanjun Wei, Yunhao Liang, Zhixi Cai, Qinghao Zhang, Shiwen Ni
- arXiv：2607.10079 [cs.AI]
- **v1：2026-07-11**（arXiv API `published`）
- **v3：2026-07-16**（arXiv API `updated`，当前最新）
- 链接：<https://arxiv.org/html/2607.10079v3>

**版本核实状态（2026-07-28）**：本笔记原先所有数字均引 v2。已抓取 v3 全文逐一比对，**§6.1 / §7.1 / §7.2 / §9.1 / §9.3 / §10.3 / §10.5 的全部数字在 v3 中均原样存活**，引用标记已统一升级为 v3。

> ⚠️ 检索陷阱备忘：v3 主表用**小数记法**（`.345` / `.207` / `.132`）而非百分数。直接 grep `34.5` 会 0 命中并造成"数字在新版被删"的假象。核版本时必须两种记法都试。

## 4.2 任务定义

MAG 将两个通常分开的目标合并：

1. Agent 在 WebArena 环境中执行多步任务；
2. Agent 每一步同时生成未来普通用户可以照着操作的 guide。

作者将其定位为 Digital Adoption Platform 自动化：Agent 不只替用户操作企业软件，还自动生成可复用的新手引导。

数据构建概况：

- 来源于 WebArena 六个网站；
- 以 OpAgent 成功演示过的 581 个任务为起点；
- 最终形成 563 个带逐步 screenshot、action 和人工校正 guide 的任务；
- 作者称其为首个联合评估多步网页执行与逐步 guide generation 的 benchmark。

来源：<https://arxiv.org/html/2607.10079v3>

---

# 5. MAG 实际比较的是什么

## 5.1 两种 action grounding

MAG 比较：

- **SoM grounding**：输出当前候选元素的 sequential index，\(\rho_t\in\{1,\ldots,n_t\}\)（记号与 §6.2 统一：\(t\) = step，\(n_t\) = 该 step 的候选元素数）；
- **Coordinate grounding**：输出页面像素坐标。

## 5.2 两个 arm 的 observation 基本相同

两种 grounding 都能看到：

- 带编号框的 screenshot；
- 候选元素文字菜单；
- visible text；
- task intent；
- guide history。

区别主要是动作输出接口：选择 mark index，或生成 coordinate。

因此，MAG 研究的是：

> **在 SoM-enriched observation 固定时，index selection 与 coordinate emission 之间的 action-grounding preference。**

它不能单独证明：

- screenshot 比 DOM 更好；
- SoM image 比 raw screenshot 更好；
- visual-only 比 structured text 更好；
- marker overlay 本身有效；
- observation modality routing 的收益。

论文在综述中应归入 Grounding、Action Interface、Model-specific Routing，而不能简单归为 observation-modality comparison。

---

# 6. MAG 的主要实验发现

## 6.1 Grounding preference 明显依赖模型

论文 Table 1（SR 列，n=174 test tasks）+ Appendix F 配对 bootstrap CI：

| 模型 | SoM | coord | Δ | 95% CI | 判读 |
|---|---:|---:|---:|---|---|
| Gemini 3.5 Flash | 34.5% (60) | 20.7% (36) | **+13.8** | **[+6.9, +21.3]** | 显著偏好 SoM |
| GPT-5.5 | 35.6% (62) | 37.4% (65) | +1.7 (偏 coord) | [−4.0, +7.5] | 跨零，无偏好 |
| Claude Sonnet 4.6 | 27.6% (48) | 27.0% (47) | +0.6 | [−5.2, +6.3] | 跨零，无偏好 |
| Qwen3.5-9B GRPO r10 | 13.2% (23) | 8.0% (14) | — | 见下 | 仅 SoM 获益 |

补充事实（原笔记缺）：

- **只有 Gemini 一个模型的 grounding 偏好统计显著**，另两个 API 模型的 CI 都跨零。"model-specific routing" 的证据强度实际是 **1/3 模型**，不是"三个模型各有偏好"。
- 9B 的 coordinate 路线不只是"没有稳定获益"，而是**先升后降**：SFT 7.5% → GRPO r5 9.2% → **r10 8.0%**（回落到 base SoM 的水平）。论文 intro 引用的 "13.2% versus 9.2%" 是拿 r10 SoM 对 **r5** coord（各自最佳轮次），不是同轮对比。
- 论文自己给的 grounding-only unique-solve 计数：Gemini 34 vs 10；GPT-5.5 14 vs 11；Claude 13 vs 12。

来源：<https://arxiv.org/html/2607.10079v3>（Table 1 + Appendix F，2026-07-28 核实）

最稳妥的解释：

> Grounding preference 不是全局常量，而是 model × task × environment × training regime 的交互。

这为动态 grounding router 提供了动机，但还不等于证明某个 router 可以可靠预测每个 task 的最佳 grounding。

## 6.2 MAG 支持 SoM 使用 sequential ID

MAG 将 SoM grounding 定义为当前候选元素集合上的局部 sequential selection index：

\[
\rho_t \in \{1,\ldots,n_t\}
\]

这类 ID 是 representation-local interface，不是 Chromium 生命周期相关的 AXID/nodeId。

对 P79 的意义：

- SoM-family 使用 sequential ID 有外部实现依据；
- sequential ID 更符合 SoM 的“选择菜单”语义；
- 浏览器内部 node ID 不应被当成 SoM 语义的一部分；
- P79 的 axis-1 应描述为 **AXTree-native representation → SoM-indexed flat representation**，而不是“纯 flattening”。

---

# 7. 训练层启示

## 7.1 SFT 学会 contract，不等于学会 competence

MAG 对 Qwen3.5-9B 的结果显示：

- **OFCR（output format correctness rate）**：base coord 13.7% / base SoM 18.6% → SFT 97.4% / 97.6%（base 模型 81.4% 的 step 不可解析）；
- 任务成功率没有同步提高；
- **SoM SFT 从 base 8.0% 反降至 6.9%**（论文原文：*below the 8.0% of the untuned base*）；
- 作者归因：*"the tuned model learns to declare completion too early"*。

来源：<https://arxiv.org/html/2607.10079v3>（Table 1 OFCR 列 + §1 + Appendix F，2026-07-28 核实）

训练能力应至少分成五层：

1. **Contract learning**：输出合法 JSON、动作格式和 guide 格式；
2. **Local grounding/action imitation**：在单步或局部状态选对元素和动作；
3. **Long-horizon competence**：保持多步任务目标、处理状态变化；
4. **Recovery and exploration**：从错误、循环和未知状态中恢复；
5. **Outcome optimization**：真正提高最终功能成功率。

许多 Web Agent 工作只证明了前两层，却将其包装成完整 Agent 能力。MAG 在这一点上提供了较诚实的反例。

## 7.2 Plain GRPO 在低成功率 Web Agent 上容易失去梯度

当一个 task 的一组 rollout 全部失败，binary reward 全为 0，group advantage 也为 0，无法产生有效学习信号。

在个位数成功率的长 horizon Web Agent 上，all-fail group 很常见。因此问题往往不是单纯选择哪种 advantage estimator，而是：

> **当前策略能否探索到至少一条正轨迹。**

MAG 的 expert-augmented 做法：

- 每个 task 生成 6 条 on-policy trajectory；
- 最多加入 2 条缓存的 GPT-5.5 成功 expert trajectory；
- 让原本全失败的 group 获得正 reward support。

结果：SoM 路线从 SFT 的 6.9% 提升到 13.2%，task-bootstrap 95% CI 约为 [+1.1,+11.5]pp。

**但作者自己对"SoM 才获益"这个对比留了后手**（原笔记缺，且这条直接支持 §10.4）：

> SoM 与 coordinate 的**增益之差 +5.7pp 仍然跨零**（CI [−0.6, +12.1]），所以作者只说 "SoM is the mode where training makes progress"，明确**拒绝**声称这是一个 proven contrast。

也就是说 "GRPO 只帮 SoM 不帮 coordinate" 这句话，在论文内部只有 **directional** 支持，没有统计支持。引用时不要写成"论文证明 GRPO 对 SoM 有选择性增益"。

来源：<https://arxiv.org/html/2607.10079v3>（§5 + Appendix F，2026-07-28 核实）

更稳妥的领域结论：

> Expert injection 是该论文实验中最受支持的有效机制；它缓解了低成功率 binary-reward RL 的探索稀疏问题，但现有证据不足以证明它是唯一必要条件。

---

# 8. Guide history 作为外显 memory

MAG 每一步输入：

- 当前 screenshot；
- candidate menu；
- task intent；
- 之前生成的 guide history。

Agent 面向用户生成的文本，同时成为下一步的状态摘要。这可以称为：

> **Externalized explanation as agent memory**

## 8.1 潜在优势

- guide 可以直接复用；
- memory 对用户可见、可审计；
- 强迫 Agent 维持操作语义连贯；
- 降低对隐藏 chain-of-thought 的依赖；
- 可用于人类接管、回放或教学。

## 8.2 主要风险

- 错误 guide 会变成持续污染后续决策的错误记忆；
- 面向用户的表达未必是最优内部状态压缩；
- guide quality 与 task success 可能仅为共同能力导致的相关性；
- explanation、memory 和 action policy 被耦合，难以分离因果贡献；
- 过度简化 guide 可能丢失执行所需的隐含状态。

因此，成功 episode 的 guide 更好，不等于 guide generation 因果上提高了成功率。

> 补充（v3 核实）：论文设有 output contract 约束——**任何被评估的 step 都不得把 mark id 或 coordinate 泄漏进 guide 文本**，作者报告全部 run 零违例，并评论 *"the constraint is easy to satisfy; being right is not"*。这说明 guide 与 action 的耦合在**格式层**是被隔离的，但在**语义层**（guide history 作为下一步输入）仍然耦合，上述风险不受该约束影响。

来源：<https://arxiv.org/html/2607.10079v3>（§4.2 + §5，2026-07-28 核实）

---

# 9. MAG 的 routing headroom 与 run-to-run 问题

## 9.1 论文的 headroom 观察

论文 Appendix F "Solved set overlap"：

> Its SoM and coordinate successes overlap on only 8 tasks; the union covers 29 of 174 (16.7%), **an immediate headroom for modality routing**.

**完整数字（原笔记只给了 overlap 和 union，导致下面的公式无法求值）**：

| 量 | 值 | 来源 |
|---|---:|---|
| checkpoint | **GRPO round 10**（非 base / 非 SFT） | Appendix F |
| \(\lvert S_{\text{SoM}}\rvert\) | **23** (13.2%) | Table 1 `.132` |
| \(\lvert S_{\text{coord}}\rvert\) | **14** (8.0%) | Table 1 `.080` |
| overlap | 8 | Appendix F |
| union | 29 (16.7%) | 23 + 14 − 8 = 29 ✓ 内部自洽 |

对应 estimator：

\[
H_{\text{route}}
=
|S_{\text{SoM}}\cup S_{\text{coord}}|
-
\max(|S_{\text{SoM}}|,|S_{\text{coord}}|)
= 29 - 23 = \mathbf{6\ tasks} = \mathbf{3.4pp}
\]

它属于 single-run solved-set union/extrema functional。**论文正文从未把 headroom 化成这个 3.4pp 的数**——只给了 union 的 16.7%，读起来比实际可路由增量大得多（16.7% vs 3.4pp）。引用时务必用 \(H_{\text{route}}\)，不要用 union 比例。

来源：<https://arxiv.org/html/2607.10079v3>（Table 1 + Appendix F，2026-07-28 核实）

## 9.2 为什么它可能高估稳定 routing opportunity

主表具有以下特征：

- API 模型每个 condition 只有一次 greedy sweep；
- 9B 主结果每个 grounding 也只有一个 seed；
- bootstrap 只重采样 task；
- 没有重复执行同一 task-condition 来估 execution-level variance。

因此，observed complementarity 可能同时包含：

- 稳定的 grounding capability difference；
- trajectory stochasticity；
- provider nondeterminism；
- environment variation；
- ~~evaluator variation~~ → **见下方 §9.2a，这一条已被作者部分 defuse，不要再当攻击点用**；
- SoM 与 coordinate 的执行噪声差异。

**两条量化强化（用作者自己的数字，比上面的定性列表有力得多）**：

1. **作者自己在 Limitations 给出了噪声尺度**：
   > The test set holds 174 tasks, so **single digit differences carry roughly ±2 point uncertainty**, and the GRPO result rests on **one seed per grounding scheme** and one teacher model.

   而 \(H_{\text{route}} = 3.4\text{pp}\)。**routing headroom 只有作者自承不确定度的 1.7 倍**，且建立在单 seed 上。这是用论文自身材料构成的最干净反驳，不需要任何外部假设。

2. **提供 unique-solve 的那条 arm 本身没有可证的训练效应**：coordinate r10 = 8.0%，对 SFT 的增量是 **+0.6pp，CI [−3.4, +4.6]**（跨零）；r5 的 +1.7pp CI [−2.9, +6.3] 也跨零。也就是说 union 里 coord 独家贡献的那 6 个 task，来自一条统计上与 SFT 不可区分的 arm。**"稳定互补能力"的解释需要 coord arm 先有可证能力，而这个前提在论文内部不成立。**

更准确的表述应为：

> The observed union is an upper-bound-style single-sweep routing opportunity, not a stability-calibrated routing estimand.

## 9.2a 必须承认：evaluator variation 已被作者部分排除

原笔记把 "evaluator variation" 列为 union 噪声来源之一。**v3 Appendix F "Judge agreement" 显示作者已经做了三-judge 交叉验证**，继续用这一条攻击会被一查即破：

- 29 个 semantic comparison task 原由 GPT-5.5 judge 评分（该 judge 自家系统也在被评，存在自评偏袒风险）；
- 作者用**两个独立 judge 对全部 15 runs、435 verdicts 重评**；
- Gemini 3.5 Flash 与原 judge 一致率 **94.9%**，且**没有任何 run 移动超过 2 个 task（1.1pp）**；
- Claude Sonnet 4.6 系统性更宽松（一致率 79.1% / 82.8%），抬高绝对值但**不改变论文解读的任何 ordering**；
- GPT-5.5 judge 在自家行上无 self-preference（与独立 Gemini judge 相差 ≤1 task）。

**结论**：evaluator variation 在这篇论文里被压到 ≤1.1pp 的 run-level 影响。剩余噪声来源应集中在 **trajectory stochasticity / provider nondeterminism / environment variation**——这三条作者确实没有校准。攻击面收窄了，但也因此更精准。

## 9.3 论文内部对 trajectory variation 的旁证

论文 Appendix G 对 SFT 和 GRPO 分别进行 temperature=1.0 的六次独立采样（1,044 episodes/model，同 harness 同 judge）：

| | greedy | pass@1 | pass@3 | pass@6 | never solved |
|---|---:|---:|---:|---:|---:|
| SFT SoM | 6.9% | **4.9%** | 10.1% | 14.4% | 149/174 |
| GRPO r10 | 13.2% | 12.1% | 17.6% | 21.3% | 137/174 |

这表明同一模型、同一任务的 solved set 会随 trajectory 明显变化：**SFT 的 greedy 6.9% 在采样下掉到 pass@1 4.9%**，而 pass@6 又升到 14.4%——同一 checkpoint 的"解出/未解出"标签在 ±2× 范围内浮动。

> ⚠️ **引用纪律：这是再解读（reinterpretation），不是作者结论。** 作者用同一组 pass@k 数据论证的是相反方向的命题——"GRPO adds capability, **not just concentration**"（pass@6 从 14.4% 升到 21.3%，+6.9pp，CI [+1.2, +13.2]，21 tasks only-GRPO vs 9 only-SFT），即 RL 学到了**新能力**而非只是把分布变尖。
>
> 两种读法不冲突（同一数据的不同投影），但笔记引用时必须标明"作者未作此推论"，否则会被指为误引。

来源：<https://arxiv.org/html/2607.10079v3>（Appendix G，2026-07-28 核实）

## 9.4 对 P79 的方法学价值

> ### 🚫 禁止条款：不要拿噪声地板去减 hero 数字
>
> 读到这里最自然的下一步动作是做减法——"P79 self-oracle drop-one 实测 6.7/7.6pp（§302），hero drop-one 只有 1.7-3.3pp，所以 hero 死了"。**这个动作在 P79 内部已被明令禁止两次，且做过它的那条结论已被作废**：
>
> - **§293 ADJUDICATED (2026-05-25)**：self-oracle 数字必须**对称报告**（双向 self_drop + discordance + κ），且**只作 instability diagnostic，NOT bias estimate**。
> - **§397.10 已 RETRACTED**，作废的正是"§302 的线性减法 12.1% ≈ 10.5% + 1-2pp"这一类操作。
> - **§293 记录的 GPT 纠正**：drop-one 的**正偏方向不是数学必然**。它取决于 task pool 结构——真-unique task 上的 noise 会让 drop-one **下降**，jointly-solvable / all-fail task 上的 noise 会让它**上升**。VWA 近边界任务多，所以担忧合理，但**必须限定为 conditional**，不能当成已证明的方向性偏倚。
>
> 正确用法：把 self-oracle 当**不稳定性诊断**（"这个量测环境有多吵"），不当**偏倚估计**（"hero 里有多少是假的"）。
>
> ### ⚠️ 但别把禁令扩大化 —— 有一个比较是合法的（2026-07-28 主 session 补）
>
> 上面禁的是 **self_drop vs drop-one hero**。理由是 drop-one 是**六臂联合事件**
> （P-SoM 成 ∧ 其余五臂全败），与单臂两跑的翻转率不同构；工具自带的 caveat
> `same-mode discordance != P-SoM-vs-5-competitors false-unique` 说的正是这一对。
>
> **但 self_drop vs H3 轴是同一泛函**：
>
> | 量 | 形式 |
> |---|---|
> | `self_drop(run1, run2)` | \|{run1 解出} ∖ {run2 解出}\| / n |
> | H3 axis-1 | \|{P-text 解出} ∖ {P-SoM 解出}\| / n |
> |
>
> 在"两臂可互换"的零假设下，轴应当测出两次同模式重跑测出的量 —— 那正是地板。
> §397.10(3) 原话即"**正是 H3 轴的估计量形式**"，且 `paperA/limitations.md`
> 整节 "The structural test is against the wrong null" 已经做过这个比较。
>
> **实证教训**：主 session 2026-07-28 曾用 drop-one 的禁令拒答 H3 的比较，
> 而依据就写在它自己引用过的三份文档里。**禁令扩大化与禁令缺失一样有害。**

P79 应明确区分：

- **single-run oracle headroom**；
- **stable cross-representation complementarity**；
- **run-to-run pseudo-complementarity**。

Router training label 不应长期停留在：

> 某 mode 在某一次运行成功，所以该 mode 是 task label。

更成熟的目标应是：

- repeated success probability；
- expected utility；
- cost-adjusted success；
- stability-calibrated labels；
- uncertainty-aware routing；
- abstain/fallback；
- distributional rather than hard best-mode targets。

---

# 10. MAG 的主要局限

## 10.1 “Raw coordinate grounding” 容易造成误读

Coordinate arm 不是 raw screenshot-only agent。两个 arm 都接收 marked screenshot、candidate menu 和 visible text，只是动作输出不同。

因此，论文只能支持 action grounding interface comparison，不能支持 observation modality comparison。

## 10.2 可解任务筛选偏差

MAG 继承 OpAgent 已经成功过的 581/812 个 WebArena task，因此数据偏向已有强 agent 至少能完成一次的任务。

这意味着它更适合研究：

> 在存在已知成功演示的 WebArena 子集上，如何学习执行与 guide generation。

不能将结果直接当成完整 WebArena 难度分布，也不应与一般 WebArena leaderboard 直接比较。

来源：<https://arxiv.org/html/2607.10079v3>

## 10.3 Train/test template overlap 很高

174 个 test task 中：

- 159 个与 training task 共享 intent template；
- 只有 15 个 template unseen。

GRPO 的提升主要集中在 seen-template tasks：**seen 上 round-10 SoM 达 13.8%，unseen 上仅 1/15，与 SFT anchor 完全相同**。对照组：API 模型（未训练）在 unseen 上保持水平（GPT-5.5 coord 解出 7/15，对 seen 的 36.5%）——**说明 unseen slice 本身不是"更难"，而是训练收益不迁移**。

作者自己的措辞：训练线应被读作 "largely **within template** generalization"，并承认 unseen slice 太小不足以下定论。

最稳妥的结论：

> Expert-augmented GRPO improves within-template task competence.

不能据此声称学到了跨 task family 的通用 Web Agent 能力。

来源：<https://arxiv.org/html/2607.10079v3>（Appendix F "Intent template overlap"，2026-07-28 核实）

## 10.4 Expert-GRPO 因果证据有限

限制包括：

- expert-augmented 主结果每个 grounding 只有一个 seed；
- plain GRPO 的多次尝试配置不完全一致；
- 部分比较使用 training-batch statistics，而非固定 test set；
- 一些旧 run 发生在 evaluator/judge pipeline 尚未完全稳定时；
- 作者将其描述为 documentary evidence，而非严格 controlled ablation。

因此，只能说 expert injection 是当前实验中最受支持的机制，不能说它被证明为唯一必要条件。

## 10.5 Guide metric 偏表面相似度

Guide quality 由 BLEU-1、BLEU-2、ROUGE-1、ROUGE-L 的平均构成，再通过 success gate 形成 GGS。

问题包括：

- 单一 reference；
- 多种合法表达可能被低估；
- BLEU/ROUGE 不擅长判断操作充分性和错误细节；
- \(\gamma=0.4\) 为人为设定；
- GGS 与 SR 机械耦合，不能独立证明 guide 提升。

论文另有 50 个任务的人类 usefulness check，其中 82% guide 被认为可帮助首次用户完成任务。该证据更接近真实用途，但**有两个原笔记未记录的限制**：

- **评分者是"one author"**（论文原文：*one author rated the step by step gold guide of 50 randomly sampled test tasks*）。这是**单人自评**，不是独立标注，没有 inter-rater agreement；
- 完整分布是 41 useful (82%) / 2 ambiguous (4%) / **7 useless (14%)**，Wilson 95% CI = **[69.2%, 90.2%]**——下界接近 69%，比 "82%" 单点读起来弱不少；
- 站点间高度不均：shopping admin 与 map 100%，shopping 90%，GitLab 83%，**Reddit 明显更低**。
- 作者归因：不达标案例几乎全是 **Set-of-Mark step mapping effect**（终端控件的标注映射问题），而非措辞错误。

引用时不要写成"人类评估确认 82% guide 有用"，应写成"一位作者自评 50 个样本，82%（Wilson CI [69.2, 90.2]）"。

来源：<https://arxiv.org/html/2607.10079v3>（Appendix B "Usefulness check"，2026-07-28 核实）

## 10.6 只有 task bootstrap，没有 run bootstrap

Task-level paired bootstrap 回答的是：

> 给定这一次运行产生的 labels，如果重新抽取 task，结果会如何波动？

它没有回答：

> 同一个 task 在同一 condition 下重跑，success label 是否变化？

总不确定性至少可以概念性拆成：

\[
\operatorname{Var}_{\text{total}}
=
\operatorname{Var}_{\text{task}}
+
\operatorname{Var}_{\text{execution}}
+
\operatorname{Var}_{\text{evaluation}}
+
\cdots
\]

MAG 主要估计第一项。第三项已由三-judge 交叉验证部分约束（§9.2a，≤1.1pp）；**第二项完全未估**——作者自承 "the GRPO result rests on **one seed per grounding scheme**"。

## 10.7 长程能力断崖（v3 主要 finding，原笔记完全缺失）

Table A6 按 gold demonstration 长度分解 SR（167 个有标注轨迹的 test task）：

- GRPO 增益集中在**短/中**任务（+9.8pp / +9.1pp）；
- **超过 8 个 gold step，所有 9B 变体都掉到 3% 以下**（多数为 0.0%）；
- 同区间 API 模型仍维持 **20–38%**。

作者结论：*"The remaining teacher gap is a **long horizon gap**, not a per step accuracy gap."*

对 P79 的意义：这是一个**与 grounding 正交的失败轴**。如果 P79 的 router 只在 observation/grounding 维度上选择，它在长程任务上可能面对的是一个 representation routing **无法触及**的瓶颈。§3 的路由信号清单里 `no-progress streak` / `context usage` 部分覆盖了这一点，但笔记此前没有把"任务长度"本身作为**任务前信号**列入 §3.1——建议补入（VWA 任务可从 reference trajectory 长度估计）。

---

# 11. 对 P79 的直接映射

> **⚠️ 本节 2026-07-28 重写。** 原版把下列内容写成"建议 P79 采纳"，但对照防重做台账（`scripts/maintenance/known.py`）后发现**大部分已经落地两个月**。把已完成的事写成建议，会让笔记读者重做已有工作，也会低估 MAG 的真实价值——它不是"告诉 P79 该怎么做"，而是**对已锁决策的事后外部佐证**，这在 amendment 需要辩护时更值钱。

## 11.1 Sequential ID 决策 —— ✅ 已落地（2026-05-25），MAG 是事后佐证

MAG 为以下表述提供外部支持：

> SoM-style interfaces use representation-local sequential selection identifiers rather than browser-internal AX node IDs.

**P79 现状（不是待办）**：

| 项 | 状态 | 出处 |
|---|---|---|
| SoM-family 改 deterministic sequential (1..K) | ✅ 已实施 | **AMENDMENT_07_SOM_IDENTIFIER_CONTRACT_20260525.md** + `git_witness_SOM_IDENTIFIER_CONTRACT_20260525.txt` + B-1862 |
| "element_id 不 patch、churn 是真实部署噪声" | ❌ **已 RETRACTED** | §293 → replaced_by §295 |
| 判定"是 design-change 而非 bug-fix" | ✅ ADJUDICATED 2026-05-25 | §295（git blame 证 upstream-VWA 原生，P79 从无 sequential→nodeId 改动） |
| §2 framing 改"结构-格式"表述 | ✅ 已改 | §207.3 / B-900 |
| DOM arm | 保留 native nodeId（AMENDMENT_07 未动） | §297.1 caveat |

**关键**：P79 当前的 axis-1 表述**已经**是 "preserves hierarchical nesting **vs flattened into sequential indexed list**"（§207.3），更精确的机制描述见 §303.7 replaced_by：

> P-text 真机制 = AXTree → [SOM_MARKS] structure flatten **+ AMENDMENT_07 id namespace re-key (CDP nodeId → sequential 1..K)**

所以原笔记要攻击的 "pure flattening of identical identifiers" 是 **P79 改版前的旧说法**，现在打的是稻草人。

**MAG 在这里的真实用途**：AMENDMENT_07 是 fire 中途改 substrate 的 amendment，最怕审稿人问"你凭什么中途改"。§295 当时的辩护理由是"production/标准 SoM 全是 sequential 重编号（WebVoyager / SeeAct-Choice / AndroidControl / browser-use，连 VWA 自己的 `image_som` 也是）"。**MAG（2026-07，独立第三方，同期）把这份证据从"我们查到的惯例"升级为"同期同行评议工作的独立实现"**。这是 §11 里对论文最有用的一条。

⚠️ 但引用强度要诚实：MAG 只证明**它自己**用 representation-local sequential index，不证明"所有 SoM 必须如此"。不要把它写成规范性论据。

## 11.2 Routing 研究范围扩展

P79 当前核心是 observation representation routing。MAG 表明未来应进一步研究 action grounding routing。

可形成二维或联合 router：

| Observation | Grounding | 示例 |
|---|---|---|
| AXTree | native ID | DOM search / structured action |
| SoM text | sequential mark ID | P-text / P-SoM |
| Marked screenshot | sequential mark ID | full SoM |
| Screenshot | coordinate | direct visual grounding |
| Structured API | schema/tool call | agentic web / MCP |

## 11.3 Router labels

当前 single-run oracle labels 可作为第一阶段工程近似，但论文必须承认：

- 它们可能包含 trajectory luck；
- set-union headroom 可能包含 pseudo-complementarity；
- cost-aware routing 最终应优化 expected utility，而非 hard best-mode label。

## 11.4 评测与复现 —— 多数已落地，逐项对照

原版列了 6 条"建议 P79 报告"。实际状态：

| 建议项 | P79 状态 | 出处 |
|---|---|---|
| task-level uncertainty | ✅ 已有（task-level paired bootstrap 是主 gate 的一部分） | preregistration §2.5 |
| run-to-run discordance | ✅ **已实测**：14.3% (32/224)，Cohen κ = 0.614 | §302.1/§302.4（B0/cls/vision, n=224） |
| same-mode self-oracle diagnostic | ✅ **已实测**：drop-one A→B 6.7pp / B→A 7.6pp（对称报告） | §302.4 |
| evaluator-only stability | ⚠️ 部分（VWA 是 programmatic evaluator，无 LLM judge 层，性质与 MAG 不同） | — |
| representation-specific instability | ⚠️ 部分：P-text ~9% per-task flip (§292/§294)、dom 12.1% discordance (§297.1) | 各 §|
| canonical gate 与 non-gating replicate sensitivity 的边界 | ✅ **已制度化** | **AMENDMENT_06_REPRODUCIBILITY_SENSITIVITY_20260525.md** |

§293 的三平行 mitigation（ADJUDICATED 2026-05-25）已明确边界：**主 gate 原样执行 / 另设 witnessed non-gating reproducibility sensitivity 层 / sensitivity 不稳则 prose 主动降级**。

## 11.5 仍然开放的那一条（本笔记对 P79 的最高价值处）

上表之外，有一条**至今未 defuse**，而且就在昨天被重新裁定成立：

> **§396.2 ADJUDICATED 2026-07-27**：self-oracle noise floor 攻击**确认为真且两个月未 defuse**——H3 没有同模式重跑的噪声地板（Gemini #4 = codex #8）。项目自己的 `scripts/analysis/extract_50_features.py:636` 早就警告过 N=1 oracle 需要噪声天花板；这是 2026-05-15 Mode C pilot 提过的同一条攻击。

> ### ✅ 状态更新 2026-07-28（本笔记写就当日晚，主 session Phase 0b）
>
> **地板已经测出来了**，这条攻击的"没有地板"部分不再成立 —— 但测出来的结果对我们不利：
>
> | | scope | 值 |
> |---|---|---|
> | H3 axis-1 / axis-2 pooled | 6 cells | **1.35 / 2.09 pp** |
> | self_drop，**vision** clean pair | B0·cls n=224 | **6.7 / 7.6 pp**（逐位复现 §302.1） |
> | self_drop，**dom** clean pair（首次跑到 canonical N） | B0·cls n=224 | **4.9 / 7.1 pp** |
>
> **两个轴都低于两个地板。** 且地板不是 vision-specific（两个不同 mode 同量级），
> 两对均 0 reset 污染、flip 全为 model-nondeterm。
> 附带纠正：`limitations.md` 现写"we hold **one** same-condition replicate pair,
> on the strongest backbone under the **screenshot-only** mode" —— 实为**两对**且
> 第二对是文本模式，该段论证因此更强。
> 详见 `docs/analysis/cross_sites/phase0b_noise_floor.md`。
>
> ⚠️ 仍**未**解决的：地板全部来自 B0（API-served MoE），本地确定性 backbone（B1/B2）
> 的 replicate **一个都没有**（manifest 里的候选目录多数不在磁盘，唯一存在的那个
> 三个 subdir 各只剩 1 个 episode）。地板是**上界**，不可直接外推。

相关未决条目：

- **§242 CLAIM_UNVERIFIED**：drop-one oracle 的 1.7-3.3pp 必须证明显著高于 stochastic noise floor（B0 cls 12% per-task flip）；建议同 condition 重跑 2× 量 SR 的 stochastic SD——**重跑尚未做**。
- **§293 CLAIM_UNVERIFIED**：H1 的 task-level paired bootstrap **物理上看不到 run-to-run 方差**；正偏方向经 GPT 纠正为**非数学必然**（conditional on task pool 结构），**尚无 replicate-calibrated 定量**。

**MAG 与这条的关系**：MAG 用 single-sweep union 推 routing headroom、单 seed、不校准 execution variance——**与 P79 H3 是同一类估计量的同一个缺陷**。这意味着这份笔记同时是弹药和风险：

- 作为弹药：它证明这是**领域现状**，而非 P79 独有的疏漏；
- 作为风险：任何读过 MAG 批评的审稿人，会立刻把同一把刀转向 P79 的 drop-one hero。

**framing 决策（主 session 2026-07-28，已定）**：取**反转**，但表述收窄。

不写"领域惯例如此"（一篇论文证明不了惯例），写：**同期一篇 harness 工程与统计严谨性
高于平均的工作（MAG, 2026-07），仍以 single-sweep union 推 routing headroom 且单 seed
——说明 noise calibration 的缺口不是疏忽，是方法惯例里本就没有这一步。**

之所以敢反转，是因为 P79 这边**已经把地板测出来了**（见上方状态更新），
所以立场不是"我们也一样"而是"我们做了这一步，并如实报告它推翻了我们自己的正面结果"。
MAG 的 `H_route = 3.4pp` 对上作者自承 ±2 point uncertainty = **1.7 倍**，
与 P79 的 1.35/2.09pp vs 4.9–7.6pp **同构** —— 这是外部佐证，不是攻击对象。

---

# 12. Research Agent 后续任务清单

## 12.1 核实 MAG 最新版本 —— ✅ 已于 2026-07-28 完成

方法：arXiv API `id_list` 查询（**不用 WebSearch**，当月索引滞后）+ v3 全文 HTML 抓取逐数字比对。

- [x] 核对 arXiv 版本状态 —— v1 2026-07-11 / v3 2026-07-16（最新），**无 v4**；
- [x] 核实模型名称、成功率、训练设置和 sample 数 —— §6.1 表格全部通过；
- [x] 核实 guide metric、human usefulness study 细节 —— §10.5 已补单人评分 + Wilson CI；
- [x] 核实 evaluator 细节 —— §9.2a 已补三-judge 交叉验证（原笔记完全缺失）；
- [x] 确认 "raw coordinate" 的正式措辞 —— 摘要作 *"two grounding schemes **over screenshots**: Set-of-Mark element selection and raw pixel coordinates"*，**直接印证 §5.2 / §10.1 的分析**（两 arm 同为 screenshot 之上）；
- [ ] 查找代码和数据是否公开 —— 论文称 "we release everything" / checkpoints "released with the harness"，**但未核实仓库实际可达性**（唯一遗留项）。

## 12.1a 遗留核实项

- MAG 代码/数据/checkpoint 仓库是否真实公开可下载（论文只给了 HF 的 Qwen3.5-9B base 链接，harness 仓库地址待查）；
- 是否有第三方复现或 issue（§12.3 反证查询尚未执行）。

## 12.2 论文引用滚雪球

向后追：

- OpAgent；
- WebArena；
- SoM grounding；
- expert-augmented GRPO；
- WebRL；
- WebAgent-R1；
- DAPO；
- LUFFY。

向前追：

- 是否已有复现；
- 是否有人质疑 template overlap；
- 是否有人讨论 guide-as-memory；
- 是否有人研究 SoM/coordinate router；
- 是否有后续 benchmark 使用 repeated-run evaluation。

## 12.3 反证查询

至少搜索：

- `MAG web agent replication`
- `2607.10079 limitation`
- `MAG benchmark template leakage`
- `MAG GRPO ablation`
- `SoM coordinate grounding reproducibility`
- `web agent solved set overlap stochasticity`
- `WebArena run-to-run reproducibility`
- `Set-of-Marks sequential identifier AXID`

## 12.4 与 Web Agent 全景文档的集成位置

MAG 应至少进入以下章节：

1. Benchmark：action + guide generation；
2. Grounding：index selection vs coordinate emission；
3. Training：contract vs competence、all-fail GRPO；
4. Memory：externalized guide history；
5. Routing：model-specific grounding preference；
6. Evaluation：single-run union 与 execution variance；
7. P79：sequential ID 与 representation-grounding joint routing。

---

# 13. Claim ledger

v3 核实状态列（2026-07-28）：✅ = 已对 v3 正文/附录逐字核实；➖ = 分析性推论，无需核实。

| Claim | 当前判断 | 置信度 | 核实 | 边界 |
|---|---|---:|:--:|---|
| MAG 联合网页执行与逐步 guide generation | 论文直接支持 | 高 | ✅ | v3 摘要 "first benchmark that unifies…" |
| MAG 主要比较 action grounding，而非 observation modality | 由输入控制设计直接推出 | 高 | ✅ | v3 摘要作 "two grounding schemes **over screenshots**"，作者措辞直接印证 |
| Grounding preference 具有明显模型依赖性 | 实验支持，**但仅 1/3 模型显著** | 中 ↓ | ✅ | Gemini +13.8 [+6.9,+21.3] 显著；GPT-5.5 与 Claude 的 CI 跨零 |
| 9B 的 GRPO 增益是 SoM 专属 | **作者自己拒绝此断言** | 低 ↓ | ✅ | 增益之差 +5.7pp CI [−0.6,+12.1] 跨零；原文只说 "the mode where training makes progress" |
| SoM sequential ID 比 Chromium AXID 更符合 selection interface | 论文实现 + P79 §295 共同支持 | 高 | ✅ | 仅证 MAG 自身实现，非规范性论据；P79 侧已由 AMENDMENT_07 落地 |
| SFT 更容易学 contract 而非长程 competence | 本文结果支持 | 中高 | ✅ | SoM SFT 6.9% < base 8.0%；单论文单设置 |
| Expert injection 缓解 all-fail GRPO 信号稀疏 | 本文最强机制证据 | 中 | ✅ | 作者自称 documentary 而非 controlled ablation；单 seed |
| Guide history 可以充当外显 memory | 架构事实 | 高 | ➖ | 是否提高成功率未被因果证明 |
| MAG 的 solved-set union 是稳定 routing headroom | 未被证明 | 低 | ✅ | 缺 execution-level repeated runs；作者自承 ±2pp 不确定度 + 单 seed |
| MAG 的 union 是有价值的 single-sweep routing opportunity | 支持 | 中高 | ✅ | 真实增量 \(H_{\text{route}}\)=**3.4pp**（非 union 的 16.7%）；应标为未校准上界 |
| evaluator variation 是 union 的主要噪声来源 | **已被作者部分排除** | 低 ↓ | ✅ | 三-judge 交叉 435 verdicts，run-level 影响 ≤1.1pp（§9.2a） |
| 长程 (>8 gold step) 是与 grounding 正交的失败轴 | 论文直接支持 | 高 | ✅ | 9B 全变体 <3%，API 20-38%（Table A6） |
| P79 应研究 stability-calibrated router labels | 方法学建议，**且是 P79 未 defuse 的活攻击** | 高 ↑ | ➖ | §396.2 ADJUDICATED 2026-07-27；见 §11.5 |

---

# 14. 总评价

| 维度 | 评价 |
|---|---|
| 新任务定义 | 强：action + guide 有明确产品场景 |
| Harness 工程 | 很强：反映了 live Web RL 的真实工程问题 |
| Grounding 对比 | 设计相对干净，但范围比“modality comparison”更窄；可证的模型依赖仅 1/3 |
| GRPO 方法 | 有价值，证据中等，需要多 seed 和受控消融；作者自己拒绝 SoM-vs-coord 的 proven contrast |
| Benchmark 泛化 | 有限：solvable-subset + 159/174 template overlap + 训练收益不迁移到 unseen |
| 统计严谨性 | **相对基准 = Web Agent 领域平均，非绝对标准**。加分：三-judge 交叉验证、pass@k 消融、CI 全报、Limitations 主动自曝单 seed 与 ±2pp。扣分：execution-level variance 完全未估、GRPO 单 seed、guide metric 单参考且由一位作者自评 |
| 学术诚实度 | **高**：多处主动限定自己的结论（"rather than claim a proven contrast"、"documentary rather than a controlled ablation"、"too small for firm conclusions"）——批评这篇论文时应当承认这一点，否则显得不公 |
| 对 P79 的相关性 | 非常高（且**双向**：既是可引证据，也是指向 P79 H3 的同类攻击） |

最终判断：

> MAG 是 2026 年 Web Agent 方向值得重点跟踪的论文。它为 model-specific action-grounding routing 和 SoM sequential selection ID 提供了重要证据，也展示了低成功率 Web RL 中 contract learning、探索稀疏和 expert trajectory injection 的关键问题。与此同时，它用单次 solved-set union 推导 routing headroom，却没有校准 run-to-run noise，因此也是研究 single-run oracle、pseudo-complementarity 与 reproducibility 风险的典型案例。
