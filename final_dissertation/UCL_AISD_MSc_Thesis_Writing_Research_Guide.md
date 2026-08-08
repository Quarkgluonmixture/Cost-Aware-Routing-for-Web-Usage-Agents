# UCL AISD MSc Dissertation 写作深度调研与执行指南

> **用途**：把 UCL 官方要求、4 篇优秀 MSc dissertation 的实际写法、以及 Zekun 学长给出的直接写作规则，压缩成一套可以真正用于写作、改稿和最终审查的 thesis playbook。
>
> **研究日期**：2026-08-08
>
> **最重要的一句话**：**优秀 MSc thesis 不是“把做过的东西全写进去”，而是把一条可追踪、可验证、可防守的研究论证链写清楚：为什么有这个问题 → 为什么现有方法不够 → 你具体做了什么 → 每一步为什么要做 → 看到了什么 → 为什么可信 → 到哪里为止 → 下一步因此是什么。**

---

## 0. 资料范围、证据等级与使用边界

### 0.1 本调研使用的 4 篇论文

1. **`UCL_MSc_Thesis.pdf`**  
   *From Text-based Stereotype Detection to Anti-Bias Benchmarking: An Ethical Auditing Methodology for Large Language Models*  
   MSc Artificial Intelligence for Sustainable Development, UCL, 2023。该文与 UCL 对 Zekun Wu 的公开毕业生介绍中的 dissertation 主题一致；UCL 也公开确认 Zekun 是该届 AISD 的最高表现学生之一。

2. **`COMP0091_SFDZ8.pdf`**  
   *Climate Change Policy Exploration using Reinforcement Learning: Learning and Interpreting Trajectories in a World-Earth System model*  
   MSc Machine Learning, UCL, 2022；导师包括 Maria Perez-Ortiz。

3. **`Project (79)-2.pdf`**  
   *AI-powered Risk Assessment for Healthy Ageing*  
   MSc Artificial Intelligence for Biomedicine and Healthcare, UCL, 2023；导师 Maria Perez-Ortiz。

4. **`Thesis_bias_llm_clinical_FINAL.pdf`**  
   *Bias in Large Language Models: Evaluation and Mitigation in Real-World Clinical Cases*  
   MSc Artificial Intelligence for Biomedicine and Healthcare, UCL, 2024；导师 Maria Perez-Ortiz、Jackie Kay。

这四篇不是“官方模板”。它们的价值在于：它们让我们看到 **UCL AI/ML 方向、尤其 Maria 相关项目里，真正成功的研究叙事长什么样**。

### 0.2 证据等级

本文按以下优先级使用信息：

- **A 级：你当前年度的项目 handbook / Moodle / supervisor 明确要求** —— 一旦与其他资料冲突，以它为准。
- **B 级：UCL 当前官方网页、Module Catalogue、Academic Manual** —— 用来确定项目定位、Level 7 学术要求、评价原则等。
- **C 级：Zekun 对你的直接指导** —— 不是法规，但因为来自同项目高分学长并针对实际 thesis 写作，具有非常高的实操价值。
- **D 级：4 篇优秀 dissertation 的共同模式** —— 用来归纳“什么通常有效”，不能反推成硬性规则。

### 0.3 当前公开资料里没有可靠确认的东西

截至本次检索，**UCL 公开网页没有暴露你这一届 AISD final project 的完整 dissertation marking rubric、精确 word/page limit、最终模板细节、appendix 是否计入限制等**。因此：

- 不要从 2022/2023/2024 的旧 thesis 页数反推出你今年的限制；
- 不要把某篇优秀论文的章节结构当成强制格式；
- 最终提交前必须以 **当前 Moodle / module handbook / project brief / supervisor 邮件** 做一次 hard-constraint audit。

这不是小细节。**旧优秀论文告诉你“怎样写得好”，当前 handbook 才告诉你“什么能交”。**

---

# 1. 先给结论：优秀 thesis 的共同 DNA

把四篇论文、Zekun 的建议和 UCL 官方 Level 7 要求放在一起后，最稳定的共同特征不是某一种 LaTeX 风格，而是以下八点。

## 1.1 研究问题是一条链，不是一堆工作包

最强的论文几乎都可以还原成：

> **Problem → Gap → Research Question → Design Choice → Experiment → Observation → Interpretation → Next Question → Final Claim**

论文不是实验日志，但也不能把研究过程完全抹平后只给一个“完成品”。真正好的写法会让 examiner 看懂：

- 你为什么先做 A；
- A 暴露了什么问题；
- 所以为什么有 B；
- B 的结果为什么支持/不支持原假设；
- 因而下一步做 C 是否合理。

这就是你说的“**把自己的研究过程写出来**”的准确版本：**不是流水账，而是把研究决策的因果链写出来。**

## 1.2 每章都有一个问题，并把下一个问题“交棒”出去

Zekun 的规则非常关键：

> 每一章回答一个问题，并以“这导致了下一章的问题”收尾。

这在优秀论文中可以直接看到。Zekun 的 thesis 在 Introduction 里明确说每个 objective 是下一个 objective 的基础；后文又使用类似“有了透明的 classifier 之后，我们转向……”的显式过渡。RL thesis 则把五个实验排成越来越具体的干预序列，而不是独立实验的拼盘。

**判断你的章节是否合理的一个狠招：**

如果删掉章节标题，只读每章最后一段和下一章第一段，研究逻辑仍应该连得起来。

## 1.3 图不是装饰，是论证的一部分

对四篇 PDF 做 caption-label 自动统计（包含 appendix，且不同排版可能造成少量误差）：

| Thesis | 独立 Figure 标签（约） | 独立 Table 标签（约） |
|---|---:|---:|
| Zekun / stereotype-bias | 23 | 11 |
| RL / climate | 24 | 6 |
| Healthy ageing | 22 | 10 |
| Clinical LLM bias | 29 | 84 |

这里真正值得学的不是“必须画 23 张图”，而是：**四篇不同方向的优秀 AI thesis 都高度视觉化。**

更重要的是图的角色：

- **开篇 overview / system figure**：让 examiner 在进入细节前拥有 mental model；
- **pipeline / preprocessing figure**：替代几段难读的流程描述；
- **experiment-result figure**：让证据先被看见，再由文字解释；
- **failure / trajectory / example figure**：把抽象指标变成可理解机制；
- **appendix figure**：承载完整结果而不淹没正文。

Zekun 在 Introduction 早期就放了完整 System Flowchart；Clinical thesis 在 Methodology 第一页放 Study Design；RL thesis 大量用 trajectory plot 来解释 agent 到底学了什么。

## 1.4 Motivation 在 technical detail 前面

优秀写法的顺序通常是：

> **为什么需要这个东西 → 它直觉上做什么 → 形式化定义 / 算法 / 参数**

不是：

> “We use X/Y/Z…” → 两页公式 → 最后才说为什么。

RL thesis 在加噪实验里先说“真实世界不是 deterministic，因此要测试 robustness”，然后才给 Gaussian-noise 公式。这与 Zekun 的 Rule 0 完全一致：**plain mechanism first, term name second。**

## 1.5 结果段落同时写 observation、interpretation 和 uncertainty

弱论文：

> “Figure 4 shows model A achieves 0.72 and model B 0.68.”

强论文：

> 1. **What happened?** 哪条曲线/哪组数变了；
> 2. **What is surprising?** 与预期哪里一致/不一致；
> 3. **Why might this happen?** 给出机制解释；
> 4. **How sure are we?** seed、variance、数据范围、替代解释；
> 5. **What does it imply for the next experiment/claim?**

RL thesis 是这一点最好的样本：它会直接写 “the answer … is on average: no”，也会写某个表现 “surprisingly well” 但把它归因于 randomness 且不继续扩张结论。

## 1.6 负结果、失败和局限不是扣分项；掩盖它们才是

四篇中最成熟的文本都有明显的 self-critique：

- Healthy Ageing 的 abstract 直接承认一个 Deep Cox Mixtures pipeline 很可能把 hyperparameters overfit 到 test set；
- RL thesis 专门有 **Challenges**，讨论 tuning backfire、interpretability 和环境模型过于简化；
- Clinical thesis 讨论 identity category 的简化、US-centric 数据、不同 debiasing 方法效果不稳定；
- Zekun 明确把未完成的 Bias Mitigation 留作 future work，并说明 time/compute constraints。

这正对应 Level 7 的“critical evaluation of methodologies”，也对应 Zekun 的 defence calibration：**你的目标不是让每一项都成功，而是让每一句话都站得住。**

## 1.7 Appendix 是正文论证的外存，不是垃圾桶

好 appendix 的模式：

- 正文给足以支持 claim 的核心证据；
- 完整 curves / prompts / hyperparameters / extra examples / detailed tables 放 appendix；
- **正文必须明确指向它。**

RL thesis 会为了主图可读性裁掉过长曲线，并明确告诉读者 full plots 在 Appendix A。Clinical thesis 用 appendix 保存 prompts、迭代信息、训练资源、扩展结果。Zekun 也将 counterfactual scenarios、attribute words、prompt、architecture、SHAP 等放进 appendix。

这与你得到的规则完全一致：**没有正文 pointer 的 appendix，要么不该存在，要么 pointer 丢了。**

## 1.8 写给“聪明但不是你子领域的人”

UCL COMP0190 的官方学习目标明确要求能向 **academic specialists and non-specialists** 清晰沟通。Level 7 当然要求 frontier-level technical understanding，但这并不等于论文必须充满缩写。

所以 Zekun 的 “jargon rule 0” 不是风格偏好，而是与课程目标高度一致：

> **先让 examiner 理解机制，再给术语；不要要求一个 neighbouring-field examiner 预装你的上下文。**

---

# 2. UCL / AISD 到底在考什么

## 2.1 AISD final project 的项目定位

UCL 当前项目介绍把 AISD 定位为：

- 强技术 AI 训练 + 环境/人道/可持续发展问题；
- final research project/dissertation 是核心 assessment；
- individual project 通常与面对真实 sustainable-development challenge 的 stakeholder 合作；
- 不只是“跑模型”，而是把 AI 用到 real-world problem 并考虑其影响。

因此，AISD thesis 的“好”至少有两层：

1. **作为 AI research：技术问题、方法、实验和证据成立；**
2. **作为 AI for Sustainable Development：你知道这个技术介入现实系统后意味着什么，哪里帮助、哪里可能产生代价或风险。**

不一定要把整章写成 SDG 宣传材料；相反，最好的做法往往是把 sustainability 变成**研究问题或评价维度的一部分**。

## 2.2 COMP0190 给出的最直接 thesis 能力要求

UCL Module Catalogue 对 COMP0190（AI for Domain-Specific Applications Project Preparation）的描述非常有用，因为它直接说了学生最终应该能做什么：

- 把 final project 放进已有研究语境；
- 理解文献中 experimental studies 的 rationale；
- 考虑 ML/AI 在 Sustainable Development / Healthcare 中的 ethical implications；
- **critique existing research**；
- **design own research**；
- **carry out own analysis**；
- 向 specialist 和 non-specialist **communicate clearly**；
- 掌握与项目相关的 state-of-the-art algorithms、datasets、research methods；
- scope project 并定位到 literature；
- 处理 dataset gathering / curation。

把这些翻译成 examiner 真正在看什么，大概是：

> **你是否像一个初级独立研究者，而不是一个完成了工程任务的学生。**

工程量可以很大，但如果 examiner 看不到“为什么这样设计、替代方案是什么、证据如何支持 claim、局限在哪里”，它仍然不像研究。

## 2.3 UCL Level 7 的底线不是“会用方法”，而是“能评价方法”

UCL Level 7 descriptor 强调：

- 对当前/前沿知识有 systematic understanding；
- 对适用于自己研究的 techniques 有 comprehensive understanding；
- 在知识应用上体现 originality；
- 能 critical evaluate current research；
- 能评价 methodologies、提出 critique，并在合适时提出新假设。

因此 thesis 中很值钱的句子往往不是：

> “We used method X because it is state of the art.”

而是：

> “X is appropriate here because assumptions A/B match our setting; however, it does not model C, so we use Y as a robustness check / interpret the result only under scope D.”

**方法选择本身就是论证。**

## 2.4 为什么你必须为 second marker 写

UCL Academic Manual 要求 dissertation / research project 进行 full, independent second-marking；assessment 是 criterion-referenced。

这带来一个非常实际的写作推论：

> **不要写给已经知道你半年研究过程的 supervisor；要写给第一次打开 PDF 的第二阅卷人。**

Supervisor 知道：

- 为什么这个 baseline 很难；
- 为什么你当时换了 metric；
- 为什么某个实验只跑了三次；
- 为什么某个 failure 很有意义。

第二阅卷人不知道。**论文必须把这些“隐形上下文”显式化。**

这也是为什么：overview figure、term definition、research questions、experiment rationale、chapter transitions 非常值钱。

## 2.5 AISD 独有的一层：AI 既可能帮助 sustainability，也可能伤害它

UCL 自己关于 AISD 的课程案例明确强调：学生要批判性分析 AI 对 SDGs 的应用，并考虑 **social, economic and environmental sustainability**；AI 可能 enable SDGs，也可能因为 compute、bias、inequality、governance 等反过来 inhibit progress。

因此，一个成熟 AISD discussion 不只写：

> “Our method saves cost, therefore it is sustainable.”

而应该问：

- 你优化的是 money、latency、energy、API calls、GPU compute 还是一个 proxy？
- proxy 与真实 environmental impact 的关系有多强？
- 更低成本是否可能以 accuracy / accessibility / fairness 为代价？
- 系统部署后谁受益、谁承担错误？
- 你的 benchmark 场景能否代表 deployment？

**Sustainability 最好被 operationalised，而不是被当成最后一页的价值宣言。**

---

# 3. Zekun 的“8-beat abstract”应该升级成整篇论文的骨架

学长称它为 “7 beats”，但实际列出了 8 个动作。这里按实际的 8 个来用。

## 3.1 Abstract 的 8 beats

### Beat 1 — Problem

**真正坏掉的是什么？**

不是领域背景，而是具体 failure mode。

- 弱：LLMs are increasingly important in many applications.
- 强：Using the most capable input representation for every web-agent step can waste inference cost when cheaper representations would have produced the same useful action.

### Beat 2 — Stakes

**为什么这个 failure 值得研究？**

要落到现实后果：cost、latency、energy、safety、fairness、deployment feasibility、human impact 等。

### Beat 3 — Pivot / Gap

**别人做到哪，你转了哪一步？**

Gap 必须可以被前人文献支持，不能是“很少有人研究”式空白句。

最理想形式：

> Existing work optimises X or evaluates Y, but does not decide Z at the point where Z matters. We therefore …

### Beat 4 — Method

**你到底做了什么？**

给足够具体的信息，但每个专有名词都要让邻域 examiner 能理解。

### Beat 5 — Finding

**最重要的结果是什么？**

最好至少有一个数字或明确相对关系；不要只写 “effective / promising”。

### Beat 6 — Triangulation

**为什么相信它不是偶然或指标游戏？**

可能来自：

- multiple seeds / confidence interval；
- multiple benchmarks / sites / tasks；
- ablation；
- alternative metrics；
- oracle / random / heuristic baseline；
- failure analysis；
- train–serve symmetry check；
- cross-domain validation。

### Beat 7 — Scope

**你的结论到哪里为止？**

Scope 不是自我否定，而是给 claim 画合法边界。

### Beat 8 — Implication

**别人因此能做什么？**

不是“future work is needed”，而是“你的结果使什么成为可能”。

---

## 3.2 为什么 Zekun 的 thesis 本身能验证这套结构

其 abstract 的实际顺序就是：

1. LLM stereotypes / bias 是问题；
2. 给出现实 harm 的 stakes；
3. 指出现有 stereotype detection 与 bias analysis 之间缺乏连接；
4. 建 MGS dataset + sentence/token classifiers + XAI；
5. classifiers 优于 baselines；
6. 再用 feature attribution、token classifier、多个 metrics 做交叉验证/测量；
7. 研究范围落在 text stereotypes、若干社会维度；
8. 最终产出 reusable dataset / classifiers / metrics / benchmarks / auditing framework。

关键是：**abstract 不是目录压缩，而是 argument 压缩。**

---

# 4. 四篇优秀论文逐篇拆解：什么值得借，什么别照抄

## 4.1 Zekun thesis：最值得借的是“系统化 + 贡献链”

### 它做得特别好的地方

#### A. Introduction 不是泛泛背景，而是五步推进

其目录直接把 Introduction 拆成：

1. Problem Context and Research Gap
2. System Structure
3. Objective and Contributions
4. Thesis Structure
5. Public Access to Resources

这个顺序非常强：**先让人知道为什么 → 再给全局 mental model → 再说你贡献了什么 → 再告诉读者后面怎么读。**

#### B. 很早给 System Flowchart

System Flowchart 在正文很早出现，四个 stage 用不同颜色，读者先知道“整台机器”长什么样，再进入 dataset、classifier、elicitation、metrics。

这类图对复杂 agent/router thesis 尤其有效：**复杂度不可怕，没有 overview 才可怕。**

#### C. Objectives 与 contributions 一一对应

论文明确写：每个 objective 是下一步的 foundation，并对应 research contribution。

可直接借鉴一种写法：

> **Objective O1 → Evidence E1 → Contribution C1 → motivates O2**

这样 Introduction、Results、Conclusion 就能使用同一套编号，形成 traceability。

#### D. Chapter transition 非常显式

它不怕写“now that X is established, we move to Y”。这在学术写作里并不幼稚，反而降低读者认知负担。

#### E. 多种证据组合

不是只给 F1：classification + XAI + elicitation + multiple bias metrics + category analysis。即使某些具体方法今天看未必最佳，这种 **triangulation mindset** 很值得借。

### 不应照抄的地方

- 个别句子语法和词语选择并不干净；高分论文不是语言无瑕疵的证据；
- 某些现实案例在 abstract/introduction 中铺得偏多，容易稀释自己的 contribution；
- 部分术语密度仍高，所以 Zekun 后来给你的 “Jargon Rule 0” 可以看成对自己早期写法的进一步提炼；
- 章节编号/图号存在一些不够统一的地方，不要模仿排版细节。

**结论：借它的 architecture，不借它的所有 surface form。**

---

## 4.2 RL climate thesis：最值得借的是“研究过程本身就是叙事”

这是四篇里最适合学习“怎么把 research process 写出来”的。

### A. Experiment ladder 极其清楚

Introduction 直接告诉读者实验序列：

1. 复现/对比 previous benchmark；
2. 给 policy action 加 cost；
3. 换成 sparse/simple reward；
4. 给 environment 加 noise 测 robustness；
5. 让 environment fully observable，测试 partial observability 的影响。

这不是“我们跑了五个实验”，而是一串 **research interventions**。

### B. 每个实验都有 Why → Implementation → Result → Interpretation

例如 noise experiment：

> 真实世界有噪声 → deterministic setting 太理想 → 加 parameter noise → 预期性能会下降 → 加 random agent 作比较 → 实际观察结果 → 解释 robustness。

读者永远知道为什么现在要看这个实验。

### C. 它敢让一个实验推翻原本预期

论文写了类似：

> Are agents trained in a noisy environment more adaptable? **On average: no.**

然后讨论 seed variance、complex agents 的不稳定性，并把一个异常表现谨慎归因于 randomness，而不是包装成新发现。

这是非常强的 defence calibration。

### D. 它会因为实验设计变化而改变 evaluation metric

Reward function 在不同实验中变了，作者意识到 raw reward 不能跨实验比较，于是定义独立于 reward 的 Success Rate，再用 trajectory qualitative analysis 补充。

这展示了一件 examiner 很看重的事：

> **不是机械地沿用指标，而是知道指标为什么能/不能回答研究问题。**

### E. Figure → observation → mechanism 的写法非常直白

Trajectory 图后不是重复 caption，而是直接描述 pattern：前几步采取什么 action、什么时候切换、为什么可能这样做。

### F. Discussion 有真正的 “What Have We Learnt?” 和 “Challenges”

它会总结：

- off-policy vs on-policy 的差异；
- reward choice 的巨大影响；
- noise 的意外效果；
- hyperparameter tuning 可能 backfire；
- model/environment 简化限制政策解释。

这比只写 “limitations and future work” 更像研究者复盘。

### 不应照抄的地方

- 部分机制解释使用 “we believe” 而缺少进一步验证；你的 thesis 若能用 ablation / diagnostic 直接检验，就不要停在推测；
- 某些图很复杂，视觉密度高；今天可以进一步改善 figure hierarchy；
- climate-specific background 较长，这是领域需要，不代表你的 thesis 也要写相同比例的 primer。

---

## 4.3 Healthy Ageing thesis：最值得借的是“research objectives 清晰 + 诚实报告”

### A. 问题到 objectives 很直接

Introduction 给出 central research problem，再列三个 objectives：传统 survival model、ML predictors、phenotype analysis。

对于工程内容很多的 thesis，这种 numbered objectives 能防止贡献散掉。

### B. Abstract 敢放不漂亮的发现

它没有把所有模型都描述成成功：Elastic Net 平均表现更好、XGBoost 某方面更 robust；并且明确指出 Deep Cox Mixtures 可能发生 test-set hyperparameter overfit。

这非常值得学：**abstract 不是宣传页。**

### C. Comparison 而非孤立 SOTA

它同时放传统 statistical model 与 ML model，使“为什么需要 ML”真正变成可检验的问题。

### 不应照抄的地方

- 其某些因果/泛化措辞可以更克制；
- background 很 domain-heavy，是 healthcare 项目的需求，不适合作为所有 CS thesis 的模板。

---

## 4.4 Clinical LLM bias thesis：最值得借的是“问题式 related work + 强 reproducibility appendix”

### A. Related Work 的标题直接是问题

例如：

- What are LLMs and how do they work?
- How do bias manifest in LLMs?
- How can we build clinically relevant bias evaluation in LLMs?

这种结构天然迫使 literature review 不变成论文列表，而是服务于 research question。

### B. Methodology 第一页就是 Study Design

一张 overview 图把：clinical cases → counterfactual variants → LLM tasks → bias dimensions → debiasing → metrics 放在一起。

复杂 pipeline 尤其适合这种做法。

### C. 实验编号形成完整 research program

- baseline；
- counterfactual variation；
- prompt mitigation；
- fine-tuning mitigation；
- ablation / no-option QA。

读者可以快速把 claim 映射回 experiment。

### D. Appendix 真正承担 reproducibility

它放入：

- prompt templates；
- model/training iterations；
- resource / hardware information；
- extended experiment results；
- 详细表格与统计结果。

这就是“appendix 作为外存”的典型。

### E. Discussion 会拆开 outcome 和 reasoning

即使 MCQ answer 正确，也可能 reasoning 中有 bias；这说明作者不是只看 headline metric，而是在问 metric 是否真正测到目标构念。

### 不应照抄的地方

- 134 页并不代表越长越好；大量 appendix/table 是任务性质造成的；
- related work 中的 LLM primer 对熟悉领域的 examiner 可能偏长；
- 缩写（CPV/MCQ/XPL 等）仍有较高认知负荷，你可以更激进地执行 term lock 和 jargon rule。

---

# 5. 可以通用照搬的规则 vs 只能按项目取舍的东西

## 5.1 高置信度通用规则

这些基本可以直接写进你的 thesis constitution：

1. **一个核心研究问题必须能用一句普通英语说出来。**
2. Introduction 必须清楚给出 Problem / Stakes / Gap / Approach / Contributions / Scope。
3. 每章只承担一个主要逻辑任务。
4. 每个 major experiment 都必须回答一个显式问题或检验一个假设。
5. Motivation 必须出现在公式/模型/参数细节之前。
6. 第一次出现术语时，先解释机制或直觉，再给名字/缩写。
7. One term = one definition；同一个词不要在不同章节漂移含义。
8. 每个核心 claim 都要能指回 figure/table/analysis。
9. 每个 figure/table 在正文中都要被引用，并明确说明 reader 应该看什么。
10. Results 不能只报数字，必须写 interpretation；Discussion 不能重复 Results，必须综合和限定。
11. 反例、失败、null result、variance、hyperparameter sensitivity 可以提高可信度。
12. Appendix 中所有实质内容必须从正文有入口。
13. Abstract 最后写，并用 8-beat worksheet 检查。
14. 每句话做 viva test：**如果 examiner 问 “how do you know?”, 你能立刻指出证据吗？**

## 5.2 不要通用化的东西

以下应由你的研究决定，不要因为某篇优秀论文这么做就跟：

- 一定要有独立 Background chapter；
- 一定要把 Results 和 Discussion 分开；
- 一定要有 20+ figures；
- 一定要 70/90/130 页；
- 一定要从基本 Transformer/RL 原理讲起；
- 一定要有 XAI；
- 一定要做 significance test；
- 一定要把所有实验都放主文；
- 一定要沿用前人 thesis 的 UCL title page / disclaimer / section numbering。

**形式服务于论证，不反过来。**

---

# 6. “把研究过程写出来”的标准模板

这是本调研最建议你直接采用的写作单元。

对每个 major experiment / analysis，用下面的五问结构。

## 6.1 Q-D-O-I-N 模板

### Q — Question

这一节到底在问什么？

> We next ask whether …

### D — Design / Why this test

为什么这个实验能回答问题？对照组是什么？改变了哪个变量？

> To isolate X from Y, we hold … fixed and vary …

### O — Observation

先只说发生了什么，不急着讲故事。

> Figure 4 shows …

### I — Interpretation

结果意味着什么？可能机制是什么？是否有 alternative explanation？

> This is consistent with …; however, … could also explain …

### N — Next

这个发现为什么会引出下一个实验/分析？

> This leaves open whether …; we test this in Section 4.3.

---

## 6.2 一个更完整的“实验段落合同”

每个重要实验至少能回答：

- **Question**：它在检验什么？
- **Prior expectation**：为什么此前会有这个预期？
- **Controlled change**：相对前一个实验只改变了什么？
- **Metric**：这个 metric 为什么对这个 question 合法？
- **Result**：主要 effect size / ranking / uncertainty 是什么？
- **Mechanism**：为什么可能出现这个结果？
- **Alternative explanation**：还有什么可能？
- **Scope**：这个实验不能说明什么？
- **Consequence**：下一步因此做什么？

如果一项实验填不出来，通常说明两种情况之一：

1. 它不值得占主文篇幅；或
2. 你还没有把它从“engineering activity”提炼成“research evidence”。

---

# 7. Figure strategy：不是“图多”，而是“每个认知瓶颈都用图拆掉”

## 7.1 推荐的 figure taxonomy

一个复杂 AI thesis 通常可以考虑下面几类图。不是全都必须有。

### F0 — One-figure thesis overview

读者只看这一张图，也能回答：

- 输入是什么；
- 系统/研究流程有哪些阶段；
- 你的新东西在哪里；
- 评估从哪里出来。

**优先级极高。** Zekun 和 Clinical thesis 都做了。

### F1 — Problem / motivating example

给一个具体任务/失败案例，让抽象问题变成可感知问题。

### F2 — Data / benchmark / representation schematic

解释 data unit、modalities、preprocessing、sampling、split。

### F3 — Method / router / architecture

重点不是把每个 tensor 都画进去，而是让读者理解 decision boundary / data flow。

### F4 — Experimental design matrix

模型 × benchmark × representation × baseline × intervention，尤其适合实验很多时。

### F5 — Main result figure

只承载 headline claim；不要把 15 个次要 ablation 全塞进来。

### F6 — Trade-off / frontier figure

当 thesis 核心是 cost–accuracy / performance–resource trade-off 时，通常比单一 accuracy table 更有解释力。

### F7 — Diagnostic / mechanism figure

failure mode、trajectory、feature importance、routing decision distribution、unchanged-page rate 等，回答“**why**”。

### F8 — Robustness / external validation

让 triangulation 变成视觉证据。

### F9 — Qualitative examples

用于给 aggregate metric 一个现实含义，但不要拿 anecdote 替代 quantitative evidence。

## 7.2 每张图必须通过 4 个检查

1. **Question test**：这张图在回答哪一个问题？
2. **Caption test**：只看 caption，读者是否知道图是什么、数据是什么、主要设置是什么？
3. **Text test**：正文有没有告诉读者应该看哪一部分？
4. **Claim test**：删掉这张图，会不会有某个重要 claim 失去证据？

如果 1/3/4 都答不上来，它可能只是 decorative figure。

## 7.3 Figure-first drafting

推荐先完成核心 figures，再写 Results。原因很简单：

> **如果你没法把 evidence 画清楚，通常也还没把 claim 想清楚。**

建议先做一张 `claim ↔ figure` 表：

| Claim ID | Claim | Primary evidence | Robustness evidence | Main / Appendix |
|---|---|---|---|---|
| C1 | … | Fig. 4.2 | App. Fig. A.3 | Main |
| C2 | … | Table 4.1 | Fig. B.2 | Main + App |

---

# 8. Literature Review：不要写成“论文摘要串烧”

## 8.1 Related Work 的任务不是证明你读得多

它必须完成三个动作：

1. **Define the problem space**：有哪些关键概念/设定；
2. **Organise prior approaches**：按照与你的问题有关的轴分类，而不是按作者年份排列；
3. **Derive the gap**：这些方法为什么还不足以回答你的 research question。

### 弱结构

> Paper A did X. Paper B did Y. Paper C improved B.

### 强结构

> Existing approaches make one of three choices about when rich information is used: always-on, manually fixed, or dynamically selected. The first maximises available context but ignores cost; the second reduces cost but cannot adapt to task state; the third is closest to our setting, but prior routing work optimises … rather than …

这里 literature review 已经在“推导你的方法”。

## 8.2 用问题式 subsection title

Clinical thesis 的做法很值得借：让 section title 本身就是问题或争议。

例如：

- When does visual information improve web-agent decisions?
- How is inference cost measured in multimodal agents?
- What has dynamic model/modal routing optimised before?
- Why are existing routing signals insufficient for interactive web tasks?

这会天然减少“百科全书式 background”。

## 8.3 Background 的 stop rule

只解释满足下面至少一个条件的知识：

- 后文方法需要它；
- 后文实验需要 examiner 用它理解结果；
- 它是 gap 的一部分；
- 它定义了你的 metric / assumption / scope。

否则可以引用，不必教学。

**邻域 examiner 需要被带进来，但不是从本科第一章重新教起。**

---

# 9. Methodology：写“设计理由”，不要只写 implementation manual

## 9.1 方法章节应该回答四层问题

### Layer 1 — What is the problem formulation?

- input / output；
- unit of decision；
- objective / estimand；
- constraints；
- definitions。

### Layer 2 — Why is this design appropriate?

- 为什么这个 representation / model / split / threshold / feature；
- 与替代方案相比有什么取舍；
- 哪些 assumption 被接受。

### Layer 3 — How exactly is it implemented?

- enough detail for reproducibility；
- hyperparameters、models、data pipeline、seeds；
- full boilerplate 可以去 appendix。

### Layer 4 — How could this design fail?

- leakage；
- confounding；
- train–serve mismatch；
- metric validity；
- contamination；
- unstable seeds；
- missing modality / unobserved variable。

Methodology 写到 Layer 4，才真正像 Level 7 research。

---

# 10. Results 与 Discussion：严格分离“看到了什么”和“能说明什么”

## 10.1 Results 的最小单元

推荐每个 subsection 使用：

> **Question → Figure/Table → direct observation → uncertainty → local interpretation**

其中 local interpretation 只解释与这一结果直接相关的机制。

## 10.2 Discussion 才做跨实验综合

Discussion 应回答：

- 所有实验合起来，最稳定的 pattern 是什么？
- 哪些 findings 相互加强？
- 哪些结果彼此冲突？
- 与 related work 的结论哪里一致/不一致？
- 你的 original hypothesis 哪部分被推翻？
- 什么是 mechanism，什么只是 correlation？
- 对真实 deployment / sustainability 的意义是什么？
- 哪些 claim 不能跨出 benchmark/task/model 范围？

### 一个很有用的区别

**Results：** “Hybrid achieved X while DOM achieved Y.”  
**Discussion：** “The gain is concentrated in states with …, suggesting that visual context is valuable selectively rather than uniformly; however, this interpretation depends on …”

---

# 11. Claim calibration：把整篇论文写成你能 live defend 的版本

Zekun 的 defence calibration 建议可以形式化为四级 claim ladder。

## 11.1 Claim ladder

### Level A — Direct observation

> Under our evaluation, X was higher than Y by …

通常最容易 defend。

### Level B — Empirical pattern

> Across the evaluated sites/seeds, X consistently …

需要跨条件证据。

### Level C — Mechanistic interpretation

> This suggests X helps because …

需要 diagnostic / ablation；否则用 suggests / is consistent with。

### Level D — General-world implication

> Dynamic routing reduces the environmental footprint of agentic AI.

这是最危险的层级，因为可能需要真实 energy/carbon measurement，而不仅是 token/API cost proxy。

**原则：证据在哪一层，claim 就写到哪一层。不要用 Level A 证据写 Level D 句子。**

## 11.2 常用 calibration 词

- **强证据**：demonstrates / shows（谨慎使用）
- **中等证据**：indicates / supports
- **机制尚未直接验证**：suggests / is consistent with
- **探索性**：may / could / points to

不需要把每句都 hedge 成雾，但需要让语言强度与证据强度一致。

---

# 12. Appendix 设计成“可复现层”

## 12.1 主文应该留下什么

主文保留：

- 理解研究问题所需定义；
- 方法的关键设计；
- headline results；
- 对核心 claim 必要的 robustness；
- 最重要的 failure analysis。

## 12.2 Appendix 适合放什么

- complete hyperparameters；
- prompts / templates；
- full per-site/per-seed/per-model tables；
- secondary plots；
- detailed algorithms / pseudo-code；
- dataset schema；
- implementation details；
- extended qualitative examples；
- additional ablations；
- resource usage；
- negative/diagnostic results that support completeness but would interrupt narrative。

## 12.3 Main-text pointer 不能敷衍

弱：

> More results are in the appendix.

强：

> Appendix C.2 reports the same comparison for all sites and random seeds; the ranking is unchanged except on Site X, where variance overlaps.

**Pointer 本身应该告诉 reader 为什么值得点过去。**

---

# 13. Term Lock：给整篇论文建一个“术语合同”

建议在写作目录旁维护一个 `TERMS.md` 或表格：

| Canonical term | Definition | First defined at | Forbidden aliases |
|---|---|---|---|
| representation | … | §2.1 | modality/input type（若含义不同） |
| routing decision | … | §3.1 | selection/prediction（除非明确） |
| success | … | §3.2 | accuracy（若不是同一指标） |

尤其 AI thesis 很容易发生：

- model / agent 混用；
- task success / action accuracy 混用；
- modality / representation / input 混用；
- cost / token count / monetary cost / compute 混用；
- bias / stereotype / fairness 混用。

Zekun 自己在 thesis 中专门先区分 Bias 与 Stereotype；这不是定义洁癖，而是**后续所有 claim 的合法性依赖定义稳定。**

---

# 14. 针对你的 cost–accuracy / web-agent router 方向：建议的 thesis 论证架构

> **说明**：这一节按你当前“sustainable agentic AI / cost–accuracy routing”研究方向做映射；具体章节名应随着最终实验和今年 handbook 调整，不把它当硬模板。

## 14.1 一句话 thesis question 的理想形态

不要以“我们提出一个 router”开头。更好的上位问题是：

> **When is expensive multimodal context actually necessary for a web agent, and can we predict that need cheaply enough to preserve task performance while reducing inference cost?**

这句话天然包含：

- 现象问题：rich context 是否总有必要；
- 研究问题：什么时候必要；
- 方法问题：能否预测；
- trade-off：performance vs cost。

## 14.2 建议的 argument chain

### Chapter 1 — Introduction: 为什么要 route？

回答：**What is broken in always-on rich inference?**

结束时交棒：如果昂贵表示并非总有价值，就必须先知道不同表示在什么状态下帮助。

### Chapter 2 — Background & Related Work

回答：**Prior work tells us what about web agents, multimodal representations, cost-aware inference and routing—and what remains unresolved?**

结束时交棒：缺口必须被转化为可测的 decision problem。

### Chapter 3 — Problem Formulation & Experimental Foundation

回答：**What exactly constitutes a routing decision, success, cost, and available evidence?**

这里锁定：

- decision unit；
- representation modes；
- task/action space；
- cost proxy；
- evaluation metrics；
- data split；
- no-leakage / train–serve assumptions。

### Chapter 4 — Empirical Motivation / Representation Study

回答：**Is there actually enough heterogeneity to make routing worthwhile?**

先证明：

- rich representation 并非 uniformly necessary；
- cheap representation 也并非 uniformly sufficient；
- failure modes 有可预测结构。

没有这一步，router 会像 solution looking for a problem。

### Chapter 5 — Router Method

回答：**Can that heterogeneity be predicted without spending the very cost we hope to save?**

重点讲：

- features why available at decision time；
- labels/target how constructed；
- train–serve symmetry；
- threshold selection；
- baselines；
- leakage controls。

### Chapter 6 — Evaluation

最好按“问题”而非“数据集”组织：

1. Does routing improve the cost–accuracy trade-off over static policies?
2. How close is it to an oracle / upper bound?
3. Which signals matter?（ablation）
4. Where does it fail?（failure analysis）
5. Does the pattern survive site/domain/benchmark shift?（external validation）
6. How sensitive is the result to threshold/cost definition/model choice?

### Chapter 7 — Discussion

回答：**What have we learned about selective multimodal reasoning—not just this router implementation?**

这才是 thesis 的 scientific contribution。

可以分：

- representation value is state-dependent；
- routing learnability；
- cost–accuracy frontier；
- implications for sustainable agentic AI；
- benchmark/deployment gap；
- limitations / threats to validity；
- future work。

## 14.3 你尤其应该做的一张图

一张 **end-to-end routing overview**，同时标出：

- agent 当前 state；
- cheap available signals；
- router decision；
- DOM / SoM / Vision（或你最终模式）；
- downstream action；
- performance measurement；
- cost accumulation。

这张图应该比任何 architecture 细节更早出现。

## 14.4 你尤其应该做的一类结果图

因为主题本身是 trade-off，强烈建议主结果不是只用 accuracy table，而是至少有一张：

> **x-axis = cost / resource proxy，y-axis = task performance，标出 static baselines、router、oracle/upper bound。**

读者一眼就能理解 thesis 的核心价值。

---

# 15. Abstract worksheet：直接可填版本

不要一上来写连贯 prose。先填下面 8 格，每格 1–2 句。

## 1. Problem

What is actually broken?

> [具体 failure，不讲大背景]

## 2. Stakes

Why does it matter?

> [cost / latency / compute / practical deployment / sustainability consequence]

## 3. Pivot / Gap

What does existing work miss?

> [前人做到 X/Y，但没有解决 Z]

## 4. Method

What did you build/do?

> [dataset/benchmark setup + router + representations + decision mechanism]

## 5. Finding

What is the headline empirical result?

> [最重要数字/相对变化，带 uncertainty]

## 6. Triangulation

Why should we believe it?

> [baselines + ablations + multiple sites/benchmarks + failure analysis]

## 7. Scope

What does this NOT establish?

> [models/tasks/benchmarks/cost proxy/generalisation limits]

## 8. Implication

What does this enable?

> [selective multimodal inference / more efficient agent design / research insight]

填完后连读。如果 8 格之间不自然，问题通常不在英语，而在 thesis argument 还没锁定。

---

# 16. Chapter chain 模板：可以机械使用

## 16.1 每章开头

> The previous chapter established **X**. The remaining question is **Y**. This chapter therefore **Z**.

不一定逐字这么写，但逻辑应存在。

## 16.2 每章结尾

> This chapter showed **finding A**, with **evidence B**, under **scope C**. However, it does not yet establish **open question D**. The next chapter tests **D** by **method E**.

这能强制你区分：

- 已经证明的；
- 还没有证明的；
- 为什么下一章存在。

---

# 17. Paragraph-level 写作规则

## 17.1 一个段落一个主动作

UCL Academic Writing guidance 也建议 one main point per paragraph。对技术 thesis，可以把段落写成：

> **Claim / topic sentence → evidence / mechanism → qualification → link forward**

## 17.2 第一遍 draft 禁止“空头形容词”

看到这些词，要求自己补 evidence：

- significant
- robust
- effective
- efficient
- scalable
- substantial
- novel
- comprehensive
- state-of-the-art

例如：

> “The router is efficient.”

必须改成可测句：

> “At matched task success within X pp, the router reduces [defined cost metric] by Y% relative to always-Vision.”

## 17.3 避免 AI-paper 腔的三个模式

### 模式 A：宏大开场

> In today’s rapidly evolving landscape of artificial intelligence…

删。

### 模式 B：没有信息量的 transition

> It is important to note that…  
> It is worth mentioning that…

直接说重要的东西。

### 模式 C：三连形容词包装

> robust, comprehensive, and effective framework

拆成可验证属性。

## 17.4 “写得像你会说出来”不等于口语化

Zekun 的 voice rule 更准确地理解为：

- 句子有明确主语；
- 动机先于细节；
- 用具体例子；
- 少用 nominalisation；
- 不为了“academic”而把简单句改成抽象名词堆。

---

# 18. Reproducibility：论文中要让人看到研究是真的做过

Clinical thesis 的 appendix 和 RL thesis 的 technical details 都体现了这一点。

至少记录并决定放主文/appendix：

- datasets / versions / dates；
- exact split logic；
- models / checkpoints / APIs；
- seeds；
- decoding；
- preprocessing；
- hyperparameters；
- threshold tuning procedure；
- compute / budget if relevant；
- missing / failed runs；
- exclusion criteria；
- code/resources availability；
- metric implementation；
- prompt / system instruction；
- environment versions。

对 agent benchmark，尤其要写：

- action space；
- maximum steps；
- termination condition；
- retry/error handling；
- browser/environment version；
- how stale / failed webpages were treated；
- task success evaluator；
- how screenshots/DOM/SoM were generated。

这些不是低级 implementation trivia；**它们决定实验是否可重现、是否公平。**

---

# 19. Sustainability discussion：怎样避免“硬贴 SDG”

推荐至少分开四层：

## 19.1 Direct technical effect

你实际测量了什么？

- calls；
- tokens；
- monetary API cost；
- latency；
- GPU time；
- FLOPs proxy；
- energy（若真的测）；
- carbon（若真的估算并有 region/model assumptions）。

## 19.2 System-level trade-off

节省资源是否牺牲：

- task success；
- reliability；
- safety；
- accessibility；
- fairness；
- robustness。

## 19.3 Deployment interpretation

你的 benchmark saving 在真实 workload 下是否可能保持？

## 19.4 Broader sustainability claim

只有在证据支持时才上升到 energy/carbon/environmental benefit。

**Cost reduction ≠ automatically carbon reduction。** 如果只测 monetary cost 或 token count，就明确叫它 cost/resource proxy，而不是直接宣称 environmental sustainability。

---

# 20. 写作顺序：不建议从 Introduction 第一行一路写到 Conclusion

更高效的顺序通常是：

## Phase 1 — Claim map

先列 3–5 个最终要 defend 的 claims。

## Phase 2 — Figures & tables

把每个 claim 的 evidence 做成图/表。

## Phase 3 — Results

先写你已经知道答案的部分。

## Phase 4 — Methods

根据 Results 需要的信息，倒推必须交代的设计与复现细节。

## Phase 5 — Discussion

把跨实验的 meaning、limitations、sustainability、prior work 写清楚。

## Phase 6 — Related Work

此时你最知道真正需要哪些文献来定义 gap。

## Phase 7 — Introduction

把整条链压缩成读者 roadmap。

## Phase 8 — Abstract

最后用 8 beats 写。

这种顺序能显著降低“Introduction 写得很漂亮，后面实验却不支持它”的风险。

---

# 21. 一个可执行的 Thesis Evidence Matrix

建议马上建立并持续维护：

| ID | Research question | Claim | Primary evidence | Triangulation | Limitation | Thesis section | Appendix |
|---|---|---|---|---|---|---|---|
| RQ1/C1 | … | … | Fig. X | Table Y / ablation | … | §4.2 | A.1 |
| RQ2/C2 | … | … | Fig. X | external benchmark | … | §5.1 | B.3 |

这个表有四个用途：

1. 防止 claim 没证据；
2. 防止有实验但不知道它支持什么；
3. 自动决定主文和 appendix 分工；
4. final audit 时可以逐行 viva-check。

---

# 22. 最终改稿时的“高分 thesis lint”

## 22.1 Structure lint

- [ ] 一句话能说清 thesis question。
- [ ] Introduction 明确 Problem / Stakes / Gap / Approach / Contributions / Scope。
- [ ] 每章只回答一个主要问题。
- [ ] 每章末尾自然导向下一章。
- [ ] Conclusion 回答的 research questions 与 Introduction 完全对应。

## 22.2 Jargon lint

- [ ] 每个缩写第一次出现时定义。
- [ ] 定义前有直觉/机制。
- [ ] 同一概念全篇同一个词。
- [ ] 不同概念没有共享一个模糊词。
- [ ] 邻域 AI examiner 不查外部资料也能读懂核心 contribution。

## 22.3 Evidence lint

- [ ] 每个 headline claim 能定位到 primary evidence。
- [ ] 关键结果至少有一种 triangulation。
- [ ] uncertainty / variance / seeds 在适用处可见。
- [ ] 没用 anecdote 替代 aggregate evidence。
- [ ] alternative explanation 被讨论。

## 22.4 Figure lint

- [ ] 第一眼有 overview figure。
- [ ] 每张图正文都有 pointer。
- [ ] 每张图正文都有一句 takeaway，而不只是 “Figure X shows…”。
- [ ] caption 自包含。
- [ ] 字号在最终 PDF 100% zoom 下可读。
- [ ] 图的视觉编码前后一致。
- [ ] 主结果图没有被次要实验挤爆。

## 22.5 Defence lint

对每个结论问：

- [ ] What exactly do you mean?
- [ ] How do you know?
- [ ] Compared with what?
- [ ] Could something else explain it?
- [ ] Where does this stop being true?
- [ ] Why did you choose this metric/design?

如果一句话经不住这六问，就降级 claim 或补证据。

## 22.6 Appendix lint

- [ ] 每个 appendix section 在主文至少被引用一次。
- [ ] 主文不依赖 appendix 才能理解基本方法或 headline result。
- [ ] appendix 不是未整理的 dump。
- [ ] prompts/configs/full tables 等能支持复现。

## 22.7 Sustainability lint

- [ ] 明确定义“cost / efficiency / sustainability”分别指什么。
- [ ] 没把 money/token proxy 偷换成 energy/carbon。
- [ ] 讨论了 performance–resource trade-off。
- [ ] 说明 real-world deployment 的外推边界。

---

# 23. 建议给 Zekun / supervisor 问的“高信息量问题”

既然你可以直接得到 Zekun 指导，不要问“你觉得我写得怎么样”这种宽问题。问以下这种能直接改变稿件的问题：

1. **If you were the second marker, what is the one-sentence thesis claim you would expect after reading my Introduction? Is that the claim my experiments actually support?**
2. **Which result deserves to be the first main figure, and which results are appendix-only?**
3. **Is my research narrative a sequence of questions, or does it still read like a list of things I implemented?**
4. **Where do I assume too much neighbouring-field knowledge?**
5. **Which three claims would you challenge first in a viva?**
6. **Which part sounds like I am over-claiming sustainability / generalisation?**
7. **Does every chapter hand a concrete unresolved question to the next?**
8. **If I had to remove 20% of the thesis, what should go first?**

这种反馈比逐句 proofreading 的价值高得多。

---

# 24. UCL GenAI 使用：写作流程里要提前留记录

UCL 当前通用政策不是“一刀切禁用 GenAI”，而是按 assessment category 决定。官方说明：

- Category 1 通常只用于准备/复习，submitted work 要自己完成；
- Category 2 可以用于 ideas、drafting、proofreading，但需要记录并 acknowledge；
- Category 3 还会要求 assessment 本身使用 GenAI；
- 不管哪类，都必须理解并能 defend 自己提交的内容；
- 具体 assessment instruction 优先。

因此对 thesis 最稳妥的做法是现在就维护一个轻量 `AI_USE_LOG.md`：

| Date | Tool/model | Purpose | Input type | What was accepted/rejected | Acknowledgement needed? |
|---|---|---|---|---|---|
| … | … | outline critique / grammar / code debugging | … | … | … |

**先查清你 final project 的 assessment category 和 declaration 规则。** 不要等提交前再反向重建使用记录。

---

# 25. 本调研最后形成的“Thesis Constitution v1”

如果只保留一页规则，就保留下面这些。

> ### Thesis Constitution
>
> **T1 — Research question first.** 每个章节、实验、图都必须服务一个研究问题或 claim。  
> **T2 — Motivation before mechanism.** 先说为什么，再说怎么做。  
> **T3 — Plain mechanism before jargon.** 第一次出现术语时，直觉/机制先行。  
> **T4 — Term lock.** 一个概念一个名字，一个名字一个定义。  
> **T5 — Chapter chain.** 每章回答一个问题，并明确留下下一问题。  
> **T6 — Evidence traceability.** 每个 headline claim 都能指到具体 figure/table/analysis。  
> **T7 — Research process, not diary.** 写出决策逻辑：观察 → 假设 → 测试 → 结果 → 下一步，而不是时间流水账。  
> **T8 — Figures are arguments.** 图必须降低认知负担或承载证据，不做装饰。  
> **T9 — Triangulate.** 关键结论至少用一种独立证据检查。  
> **T10 — Calibrate claims.** 语言强度不得超过证据强度。  
> **T11 — Negative results are evidence.** 诚实报告失败、variance、sensitivity 和不确定性。  
> **T12 — Appendix is referenced external memory.** 没正文 pointer 的 appendix 内容不应该存在。  
> **T13 — Write for the independent second marker.** 不假设对方知道你的项目历史。  
> **T14 — Sustainability must be operationalised.** 不把 cost proxy 偷换成 environmental impact。  
> **T15 — Viva test every sentence.** 每个 substantive claim 都准备好回答 “how do you know?”。

---

# 26. 立即执行的下一步

在真正开始大规模 prose drafting 前，建议先产生 5 个小文件：

1. `THESIS_ONE_SENTENCE.md`  
   只放：problem、RQ、headline answer，各一句。
2. `CLAIM_EVIDENCE_MATRIX.md`  
   使用第 21 节的表。
3. `TERMS.md`  
   锁所有核心术语和定义。
4. `FIGURE_PLAN.md`  
   每张图 = question + takeaway + main/appendix。
5. `CHAPTER_CHAIN.md`  
   每章只写：Question / Answer / Evidence / Handoff。

**这五个文件稳定后，再写长 prose，返工会少很多。**

---

# 27. 官方与外部资料来源

以下链接用于确认 UCL 当前公开要求与项目定位；具体年度 handbook/Moodle 要求仍优先。

1. UCL — Artificial Intelligence for Sustainable Development MSc  
   https://www.ucl.ac.uk/prospective-students/graduate/taught-degrees/artificial-intelligence-sustainable-development-msc

2. UCL Module Catalogue — Artificial Intelligence for Domain-Specific Applications Project Preparation (COMP0190)  
   https://www.ucl.ac.uk/module-catalogue/modules/artificial-intelligence-for-domain-specific-applications-project-preparation-COMP0190

3. UCL Academic Manual — Level Descriptors, Level 7  
   https://www.ucl.ac.uk/study/current-students/academic-manual/chapters/chapter-7-course-and-module-approval-and-amendment-framework/section-5-level-descriptors

4. UCL Academic Manual — Marking & Moderation  
   https://www.ucl.ac.uk/study/current-students/academic-manual/chapters/chapter-4-assessment-framework-taught-programmes/section-4-marking-moderation

5. UCL — Using artificial intelligence to support sustainable development  
   https://www.ucl.ac.uk/sustainable-development-goals/case-studies/2022/aug/using-artificial-intelligence-support-sustainable-development

6. UCL — GenAI and academic integrity in assessment  
   https://www.ucl.ac.uk/teaching-learning/generative-ai-hub/genai-and-academic-integrity-assessment

7. UCL Academic Writing Centre — Argument, voice, structure  
   https://www.ucl.ac.uk/ioe/departments-and-centres/academic-writing-centre/resources-academic-reading-and-writing/argument-voice-structure

8. UCL Library — Dissertation / research project support  
   https://www.ucl.ac.uk/library/news/2024/apr/discover-ucl-support-dissertations-and-research-projects

9. UCL — Zekun Wu student experience  
   https://www.ucl.ac.uk/engineering/computer-science/study/postgraduate-taught/postgraduate-student-experiences/zekun-wu

10. UCL — Top AISD students academic excellence award  
    https://www.ucl.ac.uk/engineering/news/2024/may/top-students-ai-sustainable-development-msc-win-award-academic-excellence

---

# 28. 对四篇 thesis 的快速索引（方便回看原文）

## Zekun thesis

优先回看：

- Abstract：PDF p.3
- Introduction / Problem Context & Gap：p.5 起
- System Structure + System Flowchart：约 p.6–7
- Objective and Contributions：约 p.7–10
- Thesis Structure：约 p.10
- Dataset preprocessing diagrams：约 p.28–30
- Classification / XAI：约 p.38–53
- Bias Examination：约 p.55–69
- Limitations / Future Work：约 p.71–75
- Appendices：约 p.76 起

## RL climate thesis

优先回看：

- Abstract：PDF p.3
- Introduction → Approach：约 p.6–10
- Experiment design / evaluation：约 p.32–34
- Benchmark trajectories：约 p.35–37
- Reward interventions：约 p.38–44
- Noisy environment：约 p.45–48
- Full Markov state：约 p.48–51
- Summary table / Further Analysis：约 p.51–63
- What Have We Learnt? / Challenges：约 p.64–66

## Healthy Ageing thesis

优先回看：

- Abstract：PDF p.2
- Introduction / Aims and objectives：约 p.6–8
- Methods pipeline：约 p.28–39
- Results：约 p.40–52
- Discussion / Limitations：约 p.53–57
- Conclusion / Future Work：约 p.58–60+

## Clinical LLM bias thesis

优先回看：

- Abstract：PDF p.2
- Introduction / Problem / Approach / Contributions：front matter 后
- Question-driven Related Work：Chapter 2
- Study Design：Methodology 第一页（PDF 约 p.32）
- Experiments 0–4：Methodology §3.3
- Results：Chapter 4
- Discussion / Limitations：后部 discussion chapter
- Prompts / iterations / extended results：Appendices

---

## Final takeaway

四篇优秀论文虽然题目、年份、programme 都不同，但它们共同指向同一个判断标准：

> **Examiner 最终不是在奖励“做了多少”，而是在判断你是否对一个问题形成了可理解、可检验、可批判、可复现、可限定的研究回答。**

Zekun 给你的几条规则——少 jargon、先 motivation、chapter chain、appendix referral、claim 可 viva defend、abstract beats——不是单独的文风技巧。它们实际上都在服务同一个目标：

> **降低 examiner 重建你研究逻辑的成本，同时提高每个 claim 的证据密度。**

这应该成为整篇 thesis 的总设计原则。
