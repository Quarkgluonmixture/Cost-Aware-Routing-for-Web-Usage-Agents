# 方法学检索笔记：如何防守两个负结果

> **主题 1：** 如何证明“多样性收益”不只是“多试了一次”？  
> **主题 2：** 如何区分“特征里没有可利用信号”与“样本量不够”？  
> **检索日期：** 2026-08-10  
> **用途：** 论文方法学设计、负结果解释、reviewer rebuttal、规范引文核验  
> **原则：** 只保留有可核验 arXiv ID 或 DOI 的条目；凡没有找到公认专名或强判据的地方明确标 `UNCERTAIN`。

---

## 0. 结论先行

### 问题 1：多样性收益 vs “只是多抽一次”

你真正需要的不是只报告：

\[
P(A\text{ succeeds}\lor B\text{ succeeds})
\]

比

\[
P(A\text{ succeeds})
\]

高多少，而是做一个**相同计算预算的同质重复采样对照**：

\[
A+B
\quad\text{vs}\quad
A+A
\]

其中：

- `A+B`：两个不同 mode，各跑一次；
- `A+A`：同一个 mode，在相同 decoding / 环境设定下独立跑两次；
- 两边都只允许两次调用，预算完全一致。

如果

\[
P(A\lor B) > P(A_1\lor A_2),
\]

才能把**超过重复采样 baseline 的那一部分**解释为 mode heterogeneity / complementarity，而不是普通 pass@2 效应。

LLM/code-generation 里，pass@k 的标准定义与后来广泛使用的无偏估计式来自 Chen et al. (2021), `arXiv:2107.03374`。  
传统 ensemble 文献把相近的对照框架称为 **homogeneous vs heterogeneous ensembles**；近期 LLM inference-scaling 文献直接比较 **diverse prompts vs stationary prompts**。

**重要限制：**你目前“每个 mode × 每个 task 只有一次 run”的数据，不能经验估计 same-mode `pass@2`。若要正面回答 reviewer，需要补 same-mode independent repeats；除非你的推理过程严格 deterministic，使同一 mode 重跑必然得到完全相同结果。

---

### 问题 2：无信号 vs 样本不足

你原先最想引用的判断：

> “训练误差也差 ⇒ 特征无信号；只有 CV 误差差 ⇒ 只证明数据少”

**不能按这个强度写。**

规范 learning-curve 文献支持的是：

\[
\text{训练误差高 + 验证误差高 + gap 小}
\Rightarrow
\text{high bias / underfitting}
\]

而：

\[
\text{训练误差低 + 验证误差高 + gap 大}
\Rightarrow
\text{high variance / data-limited}
\]

前者只能说明：

> 在**当前特征表示 + 当前模型族 + 当前学习 procedure**下，增加更多同分布训练数据本身通常不能解决问题。

它**不能逻辑推出**：

\[
X \perp Y
\]

或“features contain no signal”。

要判断“完整分类 pipeline 是否提取到了超出 no-association null 的 predictive structure”，规范做法是 **label-permutation test**。但即便 permutation test 不显著，也只能说“没有足够证据拒绝 no-association null”，不能证明真正的“零信号”。

因此最稳的论文结论应是：

> **No reliable out-of-sample predictive signal was detected under the evaluated feature representation, model family, and available sample sizes.**

而不是：

> **The features contain no signal.**

---

# 1. 总表：需求 → 标准做法 → 规范引文

| 我的需求 | 标准做法 / 概念叫什么 | 确立或规范它的文献 | arXiv ID / DOI | 一句话：具体怎么做 |
|---|---|---|---|---|
| **Q1(a) pass@k 定义** | **pass@k / functional correctness under k samples** | Kulal et al., *SPoC: Search-based Pseudocode to Code*；Chen et al. 明确沿用并标准化计算 | `arXiv:1906.04908`; `arXiv:2107.03374` | 每个任务生成 \(k\) 个候选，只要任意一个成功就计该任务 solved；最终对任务取 solved fraction。 |
| **Q1(a) pass@k 无偏估计** | **unbiased pass@k estimator** | Chen et al., *Evaluating Large Language Models Trained on Code* | `arXiv:2107.03374` | 每题生成 \(n\ge k\) 个样本，\(c\) 个成功，使用 \(1-\binom{n-c}{k}/\binom nk\)；不要直接把 empirical pass@1 代入 \(1-(1-\hat p)^k\)。 |
| **Q1(b) 真多样性 vs 重复采样** | **matched-budget homogeneous-vs-heterogeneous control**；LLM 中可表述为 **diverse-prompt vs stationary-prompt sampling** | Bian & Wang, *On diversity and accuracy of homogeneous and heterogeneous ensembles*；Wang et al., *Diversified Sampling Improves Scaling LLM inference* | `DOI:10.3233/HIS-2007-4204`; `arXiv:2502.11027` | 在相同调用次数、相同 decoder budget 下比较 `A+B` 与 independent `A+A`；异构组合只有超过同质重复采样才构成额外 diversity gain。 |
| **Q1(c) resampling 本身会产生 diversity** | **resampling-induced ensemble diversity** | Valdovinos, Sánchez & Gasca, *Influence of Resampling and Weighting on Diversity and Accuracy of Classifier Ensembles* | `DOI:10.1007/978-3-540-72849-8_32` | 显式研究 resampling 方法如何改变 ensemble diversity 与 accuracy，说明“多抽样”自身就是一个必须控制的 diversity 来源。 |
| **Q1(c) diversity 如何量化** | **classifier ensemble diversity measures** | Kuncheva & Whitaker, *Measures of diversity in classifier ensembles and their relationship with the ensemble accuracy* | `DOI:10.1023/A:1022859003006` | 可用 disagreement、double-fault、Q statistic 等量化错误互补性；但该文也强调 diversity 没有唯一公认定义，且 diversity 指标本身不等价于性能收益。 |
| **Q2(a) 数据量是否仍是瓶颈** | **sample-size learning curve** | Viering & Loog, *The Shape of Learning Curves: a Review* | `arXiv:2103.10948`; `DOI:10.1109/TPAMI.2022.3220744` | 在逐步增加的训练样本数上重跑完整学习 procedure，画 generalization performance vs training-set size；看末端是否仍持续改善或已经平台化。 |
| **Q2(b) train/CV 判据** | **learning-curve high-bias vs high-variance diagnosis** | Emmert-Streib & Dehmer, *Evaluation of Regression Models: Model Assessment, Model Selection and Generalization Error* | `DOI:10.3390/make1010032` | train 与 test 都差且 gap 小 → high bias；train 好而 test 差、gap 大 → high variance；前者不等价于“features 无信号”。 |
| **Q2(b) 检验是否超出 no-association null** | **label-permutation classifier significance test** | Ojala & Garriga, *Permutation Tests for Studying Classifier Performance* | **ICDM 版本：** `DOI:10.1109/ICDM.2009.108`；2010 JMLR 扩展版无单独 DOI | 打乱 labels，重新运行完整分类 pipeline，构造“类别结构不存在”时的 performance null distribution，再比较 observed score。 |
| **Q2(c) 调参与性能评估必须隔离** | **nested cross-validation** | Varma & Simon, *Bias in error estimation when using cross-validation for model selection* | `DOI:10.1186/1471-2105-7-91` | inner CV 做 tuning/selection，outer CV 只做 assessment；否则用同一 CV 同时选模型和报性能会产生 optimistic selection bias。 |
| **Q2(c) 小样本下 nested CV 的方向性 bias** | **proper nested CV can remain approximately unbiased; small-n mainly harms precision** | Vabalas et al., *Machine learning algorithm validation with a limited sample size* | `DOI:10.1371/journal.pone.0224365` | 其模拟中 fully nested CV 在不同样本规模下仍接近 unbiased；不能笼统写“小样本导致 nested CV 系统性偏高/偏低”。 |
| **Q2(c) 小样本 CV 的主要已知问题** | **large CV error bars / high estimator variance** | Varoquaux, *Cross-validation failure: Small sample sizes lead to large error bars* | `arXiv:1706.07581`; `DOI:10.1016/j.neuroimage.2017.06.061` | 小样本时 CV performance estimate 本身高度不稳定，fold 间标准误还可能严重低估真正的不确定性。 |
| **Q2(c) Monte-Carlo permutation p-value** | **permutation p-values should never be zero** | Phipson & Smyth, *Permutation P-values Should Never Be Zero* | `arXiv:1603.05766`; `DOI:10.2202/1544-6115.1585` | 有限随机 permutations 时不能用会产生 \(p=0\) 的朴素 \(b/B\)；常用实现是把 observed arrangement 一并计入，例如 \((b+1)/(B+1)\)。 |
| **Q2(d) Holm correction** | **Holm step-down FWER control** | Holm, *A Simple Sequentially Rejective Multiple Test Procedure* | `DOI:10.2307/4615733` | 对 family 中的 \(m\) 个 p-values 排序并顺序检验，控制 family-wise error rate。 |
| **Q2(d) family 怎么划** | **family defined by the joint inferential conclusion / selective emphasis** | Bender & Lange, *Adjusting for Multiple Testing—When and How?*；Hoffmann et al., *When to Adjust for Multiple Testing: A Unifying Guiding Principle* | `DOI:10.1016/S0895-4356(00)00314-0`; `DOI:10.1002/bimj.70148` | 若多个检验共同支撑一个最终结论，或小 p-value 会导致某个结果被选择性强调，应将它们视为同一 multiplicity family。 |

---

# 2. 问题 1：如何证明“多样性收益”不只是“多试了一次”

## 2.1 pass@k 的标准定义

在 code-generation 文献里，Kulal et al. 已使用 top-\(k\)/multiple-candidate functional correctness 思路；Chen et al. (2021) 将后来广泛使用的 pass@k 定义和估计方式写得最清楚。

对任务 \(i\)，若产生 \(k\) 个候选：

\[
Y_{i,k}
=
\mathbf 1
\left[
\text{至少一个候选成功}
\right].
\]

于是：

\[
\mathrm{pass@}k
=
\frac{1}{N}\sum_{i=1}^{N}Y_{i,k}.
\]

也就是说，pass@k 的科学问题是：

> 在每个问题允许 \(k\) 次候选生成时，至少一次成功的任务比例是多少？

### 核心文献

**Chen et al. (2021)**  
*Evaluating Large Language Models Trained on Code*  
arXiv: `2107.03374`  
https://arxiv.org/abs/2107.03374

**Kulal et al. (2019)**  
*SPoC: Search-based Pseudocode to Code*  
arXiv: `1906.04908`  
https://arxiv.org/abs/1906.04908

---

## 2.2 Chen et al. 的无偏 pass@k 估计式

如果每个任务不是只抽 \(k\) 个，而是先独立生成：

\[
n\ge k
\]

个候选，其中：

\[
c
\]

个成功，则该任务的无偏 pass@k estimator 为：

\[
\widehat{\mathrm{pass@}k}
=
1-
\frac{
\binom{n-c}{k}
}{
\binom{n}{k}
}.
\]

对所有任务取平均：

\[
\widehat{\mathrm{pass@}k}_{\mathrm{dataset}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left[
1-
\frac{
\binom{n_i-c_i}{k}
}{
\binom{n_i}{k}
}
\right].
\]

其直观意义是：

- \(\binom{n}{k}\)：从 \(n\) 个候选里选 \(k\) 个的全部组合；
- \(\binom{n-c}{k}\)：选出的 \(k\) 个全是失败样本的组合数；
- 两者之比是“\(k\) 个全失败”的概率；
- 用 1 减去它，就是“至少一个成功”。

### 不应使用的朴素估计

不要直接把 empirical pass@1：

\[
\hat p=\frac cn
\]

代入：

\[
1-(1-\hat p)^k.
\]

Chen et al. 明确讨论了它在有限 \(n\) 下的 bias，并给出组合式 estimator 作为无偏替代。

---

# 3. 你的 Q1 真正需要的对照：`A+B` vs `A+A`

假设六种 mode 中，单独表现最好的 mode 是 \(A\)，另一个候选 mode 是 \(B\)。

你目前的结果实际上是在比较：

\[
S_A=P(A\text{ succeeds})
\]

与：

\[
S_{AB}
=
P(A\text{ succeeds}\lor B\text{ succeeds}).
\]

如果：

\[
S_{AB}-S_A
=
1.7\%-3.3\%
\]

reviewer 完全可以说：

> 这只是第二次独立生成带来的普通 pass@2 收益。

因为即使完全不换 mode，只再跑一遍 \(A\)，也可能解决第一次失败的若干任务。

---

## 3.1 最直接的 matched-budget control

增加：

\[
S_{AA}
=
P(A_1\text{ succeeds}\lor A_2\text{ succeeds}),
\]

其中：

- \(A_1\) 和 \(A_2\) 使用完全相同 mode；
- decoding 配置相同；
- token/action budget 相同；
- 唯一差异是 independent stochastic draw / independent environment rollout；
- 总调用次数与 `A+B` 完全相同。

然后定义：

\[
\Delta_{\text{div}}
=
S_{AB}-S_{AA}.
\]

解释：

- 若 \(\Delta_{\text{div}}\approx0\)：你的 1.7–3.3pp 基本可由“第二次机会”解释；
- 若 \(\Delta_{\text{div}}>0\)：存在超过 same-mode repeated sampling 的 heterogeneity/complementarity gain；
- 若 \(\Delta_{\text{div}}<0\)：第二个异构 mode 还不如再抽一次最佳 mode。

这才是 reviewer 所问的真正 counterfactual。

---

## 3.2 更贴你论文问题的 rescue-rate decomposition

可以把“第二个 mode 救回了多少第一次失败任务”写成：

\[
P(B=1\mid A=0)
\]

并和 same-mode 第二次尝试比较：

\[
P(A_2=1\mid A_1=0).
\]

则异构 mode 的额外 rescue effect 是：

\[
\Delta_{\text{rescue}}
=
P(B=1\mid A=0)
-
P(A_2=1\mid A_1=0).
\]

整体组合收益可以写成：

\[
\Delta_{\text{div}}
=
P(A=0)\cdot
\Delta_{\text{rescue}}.
\]

这比只报 pair-union accuracy 更容易直接回答：

> “第二种 representation 到底是否专门救回了第一种 representation 的失败案例？”

---

# 4. 这个设计在文献里叫什么

## 4.1 传统 ensemble 文献：homogeneous vs heterogeneous ensembles

**Bian & Wang (2007)**  
*On diversity and accuracy of homogeneous and heterogeneous ensembles*  
DOI: `10.3233/HIS-2007-4204`  
https://doi.org/10.3233/HIS-2007-4204

该文明确区分：

- **homogeneous ensemble**：成员来自同类 learning algorithm；
- **heterogeneous ensemble**：成员来自不同 learning algorithms；

并研究二者 diversity 与 accuracy 的差异。

它不是专门为 LLM mode-routing 写的，但其方法学逻辑与你的：

\[
A+A \text{ vs } A+B
\]

高度一致。

---

## 4.2 LLM inference-scaling 中的直接近邻：diverse prompts vs stationary prompts

**Wang et al. (2025)**  
*Diversified Sampling Improves Scaling LLM inference*  
arXiv: `2502.11027`  
https://arxiv.org/abs/2502.11027

该文的核心动机就是：

> 如果始终从同一 stationary prompt distribution 重复采样，输出可能高度冗余；改变 prompt / perturbation 以增加 sampling diversity 可能比单纯继续采样更有效。

因此，你可以把自己的 control 在论文里描述为：

> **a matched-budget stationary-resampling control**

或：

> **a homogeneous repeated-sampling baseline**

并将跨 mode 的组合描述为：

> **heterogeneous / diversified sampling**。

---

# 5. Q1(c)：有没有专门讨论 ensemble diversity vs resampling 的方法学文献

有，但没有一个全领域统一的“diversity-vs-resampling test”专名。

## 5.1 Resampling 自身就是 diversity 来源

**Valdovinos, Sánchez & Gasca (2007)**  
*Influence of Resampling and Weighting on Diversity and Accuracy of Classifier Ensembles*  
DOI: `10.1007/978-3-540-72849-8_32`  
https://doi.org/10.1007/978-3-540-72849-8_32

该文直接研究：

- 不同 resampling 方法；
- ensemble diversity；
- ensemble accuracy；

之间的关系。

对你的意义：

> reviewer 说“只是又抽了一次”不是无意义的杠精点，而是一个真正的方法学 confound：resampling 本身就可以创造错误差异与 ensemble gain，因此必须有 matched resampling control。

---

## 5.2 Diversity 指标并不能替代性能对照

**Kuncheva & Whitaker (2003)**  
*Measures of diversity in classifier ensembles and their relationship with the ensemble accuracy*  
DOI: `10.1023/A:1022859003006`  
https://doi.org/10.1023/A:1022859003006

该文系统比较多种 diversity measures，例如：

- disagreement;
- double-fault;
- Q statistic;
- correlation;
- entropy-based measures.

但一个特别重要的结论是：

> **不存在一个被普遍接受的唯一 diversity 定义，而且 diversity measure 本身并不能稳定预测 ensemble accuracy。**

因此你的主证据最好不是：

> “A 和 B 的 disagreement 很高，所以有多样性收益。”

而应是：

> **在相同调用预算下，`A+B` 的任务成功率显著/稳定超过 `A+A`。**

diversity metric 只能作为机制性辅助证据。

---

# 6. 你当前数据是否足够做 Q1 的这个检验

## 6.1 如果每个 mode × task 只有一次 run：不够

你目前每个 task 对每个 mode 只运行一次，因此有：

\[
A_1
\]

但没有：

\[
A_2.
\]

因此你无法从现有数据直接估计：

\[
P(A_1\lor A_2).
\]

Chen et al. 的 pass@k estimator 同样要求：

\[
n\ge k.
\]

如果要估计 pass@2，至少需要每个 task 的同一 mode 有两个 independent samples；实际为了稳定估计，通常应有更多 repeats。

---

## 6.2 唯一例外：严格 deterministic inference

如果你的实验满足：

- greedy decoding；
- temperature = 0；
- deterministic model kernel；
- deterministic environment；
- 相同 initial state；
- 相同 observation；
- 相同 action execution；

并且重复执行同一个 mode \(A\) 必然产生完全相同 trajectory，那么：

\[
A_1=A_2
\]

从而：

\[
S_{AA}=S_A.
\]

这时 reviewer 所谓 stochastic `pass@2` 解释不成立。

但需要注意 web agent 往往还有环境 stochasticity：

- page timing；
- network state；
- dynamic page content；
- browser rendering；
- tool execution；
- external site state。

因此不能只因为 LLM 是 greedy 就自动声称整个 agent rollout deterministic。

---

# 7. Q1 还有一个容易被 reviewer 抓的第二层 confound：post-hoc pair selection

如果你是看完这批 evaluation tasks 后：

1. 找到了“best single mode”；
2. 又遍历所有额外 mode；
3. 选出 union gain 最大的 pair；

那么：

\[
A+B
\]

的 1.7–3.3pp 还包含一个 **selection-on-the-test-set** 问题。

更稳妥的设计是：

- 在 development / validation tasks 上选择 \(A\) 和 \(B\)；
- 在 untouched test tasks 上只评估预先固定的：
  - \(A\)
  - \(A+A\)
  - \(A+B\)

否则 reviewer 可以说你的 pair gain 是 six-mode search 后的 winner’s curse，而不只是 pass@2。

---

# 8. 问题 2(a)：用 learning curve 区分“样本不足”与“当前模型/表示受限”

## 8.1 标准概念

**Viering & Loog (2021/2022)**  
*The Shape of Learning Curves: a Review*  
arXiv: `2103.10948`  
DOI: `10.1109/TPAMI.2022.3220744`

稳定链接：

- https://arxiv.org/abs/2103.10948
- https://doi.org/10.1109/TPAMI.2022.3220744

其核心定义是：

> learning curve 描述 learner 的 generalization performance 如何随 training-set size 改变。

用途包括：

- 判断增加数据是否仍有收益；
- 数据需求预测；
- model diagnosis；
- model selection。

---

# 9. 你的 nested-CV setting 中 learning curve 应该怎么画

假设每个 data cell 有独立数据集 \(D\)。

对训练规模：

\[
n_1<n_2<\cdots<n_J
\]

重复以下 procedure。

## Step 1：固定 outer split

使用你原来的 outer CV split。

outer test fold：

- 始终完全不可用于 feature selection；
- 不可用于 normalization fit；
- 不可用于 hyperparameter tuning；
- 不可用于 training-size selection。

---

## Step 2：只在 outer-training fold 内 subsample

例如训练比例：

\[
20\%,40\%,60\%,80\%,100\%.
\]

但你有些格只有 15 个 positives，因此不要机械设置会导致：

- 某训练 subset 无正例；
- inner folds 某折无正例；

的比例。

更实际的是按**最小正例数约束**定义 training sizes。

---

## Step 3：每个 \(n_j\) 都重新执行完整 inner pipeline

必须重新做：

- preprocessing；
- feature selection；
- hyperparameter tuning；
- classifier fit；
- threshold selection（如果有）。

不能：

1. 在全数据上先挑好特征；
2. 再对不同 \(n\) 只重新 fit classifier。

否则 learning curve 会混入 information leakage。

---

## Step 4：同时记录 training 与 held-out performance

对每个 \(n_j\) 记录：

\[
M_{\mathrm{train}}(n_j)
\]

和：

\[
M_{\mathrm{outer}}(n_j).
\]

这里 \(M\) 最好与你论文实际主评价指标一致。

如果类别极不平衡，不应仅因为方便改画 raw accuracy；应使用与你主结果一致、且对任务有解释意义的 metric。

---

## Step 5：重复 subsampling

每个 training size 不应只有一次随机 subsample。

可在 outer-training fold 内做多次 stratified subsampling，并汇总：

- median / mean；
- percentile interval；
- bootstrap CI（如设计允许）；
- 或跨 outer repetitions 的 distribution。

重点是把小样本下的高不确定性画出来，而不是只画一条光滑均值线。

---

# 10. 问题 2(b)：你最需要的“判据”到底是什么

## 10.1 有规范出处的版本

**Emmert-Streib & Dehmer (2019)**  
*Evaluation of Regression Models: Model Assessment, Model Selection and Generalization Error*  
DOI: `10.3390/make1010032`  
https://doi.org/10.3390/make1010032

该文明确讨论 idealized learning curves。

### High bias / low variance pattern

若：

\[
E_{\mathrm{train}}(n)
\]

和：

\[
E_{\mathrm{test}}(n)
\]

随着 \(n\) 增大都收敛到较差水平，而且两者 gap 小，则是典型：

> **high bias / low variance**

diagnosis。

其含义是：

> 单纯增加同分布训练样本不会解决当前模型族的主要问题。

---

### Low bias / high variance pattern

若：

- training error 很低；
- test/CV error 明显更高；
- train-test gap 大；
- 随着 \(n\) 增大 gap 逐渐缩小；

则是：

> **high variance**

pattern。

其含义是：

> 增加训练样本可能显著改善 generalization。

---

# 11. 为什么“训练误差也差 ⇒ 特征无信号”不成立

即使：

\[
E_{\mathrm{train}}
\]

和：

\[
E_{\mathrm{CV}}
\]

都很差，也可能有多种原因：

1. 特征确实几乎没有 predictive information；
2. classifier family 太受限；
3. feature representation 把真实结构压坏了；
4. regularization 太强；
5. optimization 没有找到合适解；
6. decision boundary 与模型假设不匹配；
7. label noise / measurement error 限制了 attainable performance；
8. 训练 objective 与最终评价指标不匹配。

因此：

\[
\text{high training error}
\not\Rightarrow
X\perp Y.
\]

真正有文献根据的表述是：

> **The learning curves are consistent with a high-bias regime rather than a purely variance-limited regime.**

而不是：

> **The features contain no signal.**

---

# 12. Q2(b) 真正接近“有没有可利用信号”的规范方法：label permutation

## 12.1 核心文献

**Ojala & Garriga**

*Permutation Tests for Studying Classifier Performance*

- ICDM 2009 conference version:  
  `DOI:10.1109/ICDM.2009.108`
- JMLR 2010 extended article:  
  *Journal of Machine Learning Research*, 11:1833–1863  
  https://www.jmlr.org/papers/v11/ojala10a.html

注意：

> `10.1109/ICDM.2009.108` 是 **2009 ICDM proceedings version 的 DOI**；JMLR 2010 扩展版通常以 JMLR stable URL 引用，不应伪装成它自己的 DOI。

---

## 12.2 第一类 permutation test 的 null hypothesis

Ojala & Garriga 的第一类检验通过打乱 labels 构造 null distribution。

直观 null 是：

> observed features 与 class labels 之间不存在能让当前 classifier procedure 获得真实性能优势的类别结构。

算法：

### Observed statistic

用原始 labels 运行完整 pipeline：

\[
T_{\mathrm{obs}}.
\]

例如：

- outer-CV AUROC；
- balanced accuracy；
- MCC；
- 或你预先定义的 classifier utility。

### Permutations

对 \(b=1,\dots,B\)：

1. permutation labels；
2. 重新执行 feature selection；
3. 重新 inner-CV tuning；
4. 重新 fit；
5. 重新 outer-CV evaluation；
6. 得到：
   \[
   T_b.
   \]

最后比较：

\[
T_{\mathrm{obs}}
\]

与：

\[
\{T_1,\dots,T_B\}.
\]

如果你只 permute label 后拿已经训练好的 model 再测，不是同一个 hypothesis test。

---

# 13. permutation test 仍然不能“证明无信号”

如果：

\[
p>0.05,
\]

正确结论是：

> 未能拒绝 no-association null。

不是：

> 证明 features 与 labels 独立。

尤其你有的 cell 只有：

\[
15-97
\]

个 positive examples，小 \(n\) 会让 test power 很低。

因此应把：

- learning curve；
- permutation test；
- uncertainty interval；

放在一起解释。

最稳的层级是：

### 层级 1：观察结果

> The learned classifiers did not outperform the fixed baseline under nested cross-validation.

### 层级 2：variance diagnostic

> Learning curves did / did not show a substantial train–validation gap that closed with increasing sample size.

### 层级 3：null benchmark

> Label-permutation tests did / did not show performance distinguishable from the no-association null.

### 层级 4：谨慎结论

> We found no reliable evidence that the evaluated cheap features support useful out-of-sample routing under the available data and model family.

这比：

> “there is no signal”

强得多，也更难被 reviewer 用统计学反驳。

---

# 14. 问题 2(c)：小样本 nested CV 到底有哪些已知问题

这里必须把两件事分开：

1. **selection bias**
2. **estimation variance / uncertainty**

---

## 14.1 非 nested model selection 会产生 optimistic bias

**Varma & Simon (2006)**  
*Bias in error estimation when using cross-validation for model selection*  
DOI: `10.1186/1471-2105-7-91`  
https://doi.org/10.1186/1471-2105-7-91

核心问题：

如果你：

1. 用 CV 选择 hyperparameters / feature set；
2. 又把同一个 CV 的最好分数当最终 generalization estimate；

那这个 performance estimate 会因 model selection 而 optimistic。

规范解决：

- **inner CV**：model / feature / hyperparameter selection；
- **outer CV**：final performance assessment。

而且：

> 所有 data-adaptive steps 都必须在 inner loop 内重跑。

---

## 14.2 proper nested CV 并不因为 n 小就自动有固定方向 bias

**Vabalas et al. (2019)**  
*Machine learning algorithm validation with a limited sample size*  
DOI: `10.1371/journal.pone.0224365`  
https://doi.org/10.1371/journal.pone.0224365

该文系统改变 sample size，并比较：

- ordinary K-fold CV；
- nested CV；
- train/test split；
- partially nested procedures。

其模拟结果中：

> fully nested CV 的 performance estimates 在不同 sample sizes 下保持近似 unbiased，而非 nested / partially nested procedure 可明显 optimistic。

因此论文里不要写：

> “nested CV is biased because n is small.”

更准确的是：

> **Small n limits the precision and stability of nested-CV estimates, even when nesting protects against model-selection optimism.**

---

## 14.3 小样本 CV 最直接的规范警告：large error bars

**Varoquaux (2018)**  
*Cross-validation failure: Small sample sizes lead to large error bars*  

- arXiv: `1706.07581`
- DOI: `10.1016/j.neuroimage.2017.06.061`

链接：

- https://arxiv.org/abs/1706.07581
- https://doi.org/10.1016/j.neuroimage.2017.06.061

其核心论点是：

> 小样本下 cross-validation performance estimate 的误差条本身可以非常大，而“fold-to-fold standard error”还可能显著低估真正的不确定性。

这对你的 15–97 positive cases 非常关键。

因此 reviewer 说：

> “这些格样本太少，所以 negative CV result 不能说明没有 signal”

在统计逻辑上有一部分是对的：

- 小 \(n\) 不必导致 systematic pessimistic bias；
- 但它会导致：
  - low power；
  - high variance；
  - wide uncertainty；
  - unstable model selection。

所以你的目标不是证明“小样本绝对没影响”，而是区分：

> **data-limited pattern**

与：

> **high-bias / representation-limited pattern**。

---

# 15. permutation test 的 p-value 应怎么计算

**Phipson & Smyth (2010)**  
*Permutation P-values Should Never Be Zero: Calculating Exact P-values When Permutations Are Randomly Drawn*

- DOI: `10.2202/1544-6115.1585`
- arXiv: `1603.05766`

链接：

- https://doi.org/10.2202/1544-6115.1585
- https://arxiv.org/abs/1603.05766

如果做 \(B\) 次随机 permutations，记：

\[
b
=
\#\{T_{\mathrm{perm}}\ge T_{\mathrm{obs}}\},
\]

不要使用可能给出：

\[
p=0
\]

的朴素估计。

常见 Monte-Carlo permutation 实现为：

\[
p
=
\frac{b+1}{B+1}.
\]

因此如果你希望 permutation test 能分辨到 Holm 校正后的较小阈值，\(B\) 不能太小。

例如 \(B=999\) 时最小非零分辨率约：

\[
0.001.
\]

---

# 16. 问题 2(d)：8 个 data cells，Holm 的 \(m\) 应不应该是 8

## 16.1 如果 8 个检验共同回答一个论文层面的 claim：默认是 8

假设你对 8 个 data cells 分别检验：

\[
H_{0,1},\dots,H_{0,8},
\]

而论文最终会写：

> “cheap pre-decision features contain detectable predictive value in at least one evaluated setting”

或：

> “we found no predictive setting across the eight cells”

并且你会因为其中任意一个显著而特别强调该 cell，那么这 8 个 tests 共同进入一个上层推论。

在这种情况下：

\[
m=8
\]

是最自然、最可防守的 family definition。

---

# 17. family boundary 的规范依据

## 17.1 Bender & Lange (2001)

**Bender & Lange**  
*Adjusting for Multiple Testing—When and How?*  
DOI: `10.1016/S0895-4356(00)00314-0`  
https://doi.org/10.1016/S0895-4356(00)00314-0

其核心原则：

> confirmatory setting 中，如果多个 significance tests 的结果需要组合成一个最终 conclusion / decision，应考虑 multiplicity adjustment。

也就是说 family 的边界不是“同一张 table 有几行”，而是**inferential conclusion**。

---

## 17.2 Hoffmann et al. (2026)：更直接讨论 family 边界

**Hoffmann et al. (2026)**  
*When to Adjust for Multiple Testing: A Unifying Guiding Principle*  
DOI: `10.1002/bimj.70148`  
https://doi.org/10.1002/bimj.70148

该文直接处理：

> 什么时候应该 adjustment，以及 adjustment 应覆盖哪些 tests？

提出的统一指导原则是：

> 如果作者会因为一个或多个 test 的 p-value 小，而在 reporting / interpretation 中对这些结果给予更多强调，那么这些 tests 应进入共同的 multiplicity consideration。

这与你的 8-cell 场景高度相关。

---

# 18. Holm procedure

**Holm (1979)**  
*A Simple Sequentially Rejective Multiple Test Procedure*  
DOI: `10.2307/4615733`  
https://doi.org/10.2307/4615733

若：

\[
m=8,
\]

将原始 p-values 排序：

\[
p_{(1)}
\le
p_{(2)}
\le
\cdots
\le
p_{(8)}.
\]

然后依次比较：

\[
p_{(1)}
\le
\frac{\alpha}{8},
\]

\[
p_{(2)}
\le
\frac{\alpha}{7},
\]

直到第一次不能拒绝；之后剩余假设不拒绝。

相比 plain Bonferroni，Holm：

- 仍控制 FWER；
- 通常更有 power；
- 不要求 tests 独立。

---

# 19. 什么时候 family 可能不是 8

## 情况 A：8 个完全预先独立的问题

如果八格各自对应八个科学问题：

- 每个都有独立 hypothesis；
- 不做“至少一个格有效”的 aggregate claim；
- 不会因为某一格 p 小就把它挑出来作为核心发现；

那么将它们拆成独立 families 有方法学空间。

但必须**事先**有清晰的 scientific rationale，不能在看完 p-values 后再拆 family。

---

## 情况 B：每格还测试多个 classifier，并挑最好一个

如果每格：

1. 测 5 个 classifiers；
2. 看谁最好；
3. 报它的 permutation p-value；

而这个“挑最好 classifier”的过程没有被完整包进 inner/nested procedure 或 permutation procedure，那么 multiplicity 并不只是：

\[
m=8.
\]

你还有 algorithm-selection multiplicity。

正确做法之一是：

> 把“从 classifier set 中选择最佳模型”定义为 learning pipeline 的一部分，并在每次 outer split / permutation 中重新执行。

这样最终可以保持：

> 每 cell 一个 pipeline-level hypothesis test。

---

# 20. 推荐给你的完整实验设计

## Q1：多样性收益

对预先选定的最佳 mode \(A\) 和补充 mode \(B\)：

### Primary comparison

\[
A+B
\quad\text{vs}\quad
A+A.
\]

要求：

- same number of calls；
- same per-call budget；
- same decoding settings；
- same environment conditions；
- only mode identity differs.

报告：

\[
S_A,\quad
S_{AA},\quad
S_{AB},
\]

以及：

\[
\Delta_{\mathrm{sampling}}
=
S_{AA}-S_A,
\]

\[
\Delta_{\mathrm{div}}
=
S_{AB}-S_{AA}.
\]

这样总 union gain：

\[
S_{AB}-S_A
\]

被分解成：

\[
\underbrace{S_{AA}-S_A}_{\text{extra-draw gain}}
+
\underbrace{S_{AB}-S_{AA}}_{\text{heterogeneity gain}}.
\]

这正是 reviewer 问题的最直接回答。

---

## Q2：无 predictor Pareto-dominates fixed baseline

对每个 data cell：

### A. 原始 nested-CV performance

报告：

- learned classifier；
- fixed baseline；
- uncertainty interval；
- 不只报 point estimate。

### B. Learning curve

训练规模：

\[
n_1,\dots,n_J.
\]

每个 \(n_j\)：

- subsample outer-training data；
- 完整 inner tuning；
- 记录 training 与 outer-validation performance；
- 重复多次 subsampling。

### C. Label permutation

每 cell 做：

\[
B
\]

次 permutation。

每次 permutation 完整重跑：

- preprocessing；
- feature selection；
- inner tuning；
- model selection；
- thresholding；
- outer evaluation。

### D. Holm

如果八格共同支撑同一论文层面的 inferential claim：

\[
m=8.
\]

### E. 结论分级

不要直接写：

> features contain no signal.

而是按证据强度：

1. **没有 outperform baseline；**
2. **learning curve 显示 variance-limited 还是 high-bias-like；**
3. **permutation 是否拒绝 no-association null；**
4. **再决定能否说“no reliable predictive evidence”。**

---

# 21. 可直接写入论文的方法表述

## 21.1 Q1 Methods：控制“只是多试一次”

> To distinguish representation complementarity from the trivial benefit of an additional stochastic attempt, we compare heterogeneous two-run unions against a matched-budget homogeneous resampling control. For a reference mode \(A\) and an additional mode \(B\), the heterogeneous condition evaluates whether either \(A\) or \(B\) succeeds, whereas the homogeneous control evaluates whether either of two independent runs of \(A\) succeeds. Both conditions therefore use the same number of agent runs; any improvement of \(A+B\) over \(A+A\) reflects complementarity beyond repeated sampling.

可引用：

- Chen et al. 2021, `arXiv:2107.03374`
- Bian & Wang 2007, `DOI:10.3233/HIS-2007-4204`
- Wang et al. 2025, `arXiv:2502.11027`

---

## 21.2 Q1 Results：不要把全部 union gain 都叫 diversity gain

建议：

> The union of modes \(A\) and \(B\) improved success by \(X\) percentage points over \(A\) alone. However, \(Y\) points of this increase were also obtained by a second independent sample from \(A\). The residual matched-budget gain attributable to cross-mode complementarity was therefore \(X-Y\) points.

不要写：

> Adding mode \(B\) provides \(X\) points of diversity gain.

除非：

\[
X
\]

已经是相对 `A+A` 的差。

---

# 22. Q2(b) 可直接写入论文的推荐句子

## 如果 learning curve 是 high-bias-like

> Training and held-out learning curves converged at similarly poor performance, with only a small generalization gap. This pattern is consistent with a high-bias regime rather than a purely variance-limited one: increasing the number of samples from the same distribution would not, by itself, be expected to remove the observed limitation under the evaluated representation and model family.

引用：

- Emmert-Streib & Dehmer 2019, `DOI:10.3390/make1010032`
- Viering & Loog 2022, `DOI:10.1109/TPAMI.2022.3220744`

注意这句话没有说：

> no signal exists.

---

## 如果 train 好、CV 差

> The large train–validation gap is instead characteristic of a high-variance regime, so the negative held-out result cannot be separated from limited sample size with the present data.

这是面对 15-positive cell 时非常重要的诚实结论。

---

## 如果 permutation 也不显著

> Under label permutation, the observed classifier performance was not distinguishable from the no-association null after multiplicity correction. We therefore found no statistically reliable evidence that the evaluated feature representation supports out-of-sample discrimination in this cell.

不要改成：

> The features are uninformative.

---

# 23. 一个更严谨的 reviewer-response 逻辑

Reviewer:

> “Your classifiers fail only because some cells have very few positive examples.”

回答不应是：

> “No, because training performance is also poor.”

而应分三层：

### 1. 承认 power / precision limitation

> Small cells necessarily yield less precise performance estimates; we therefore do not interpret a non-significant result as proof of no signal.

引用 Varoquaux 2018。

### 2. 展示 learning-curve diagnosis

> In cells where training and validation curves remain separated and improve with \(n\), the data-limitation explanation remains plausible.

> In cells where both curves converge at poor performance and flatten, the observed failure is high-bias-like rather than purely variance-limited.

引用 Emmert-Streib & Dehmer；Viering & Loog。

### 3. 用 permutation test 定义 null benchmark

> We additionally evaluate whether the complete nested learning procedure extracts structure beyond a label-permutation null.

引用 Ojala & Garriga。

这样你没有过度声称“no signal”，但也没有让 reviewer 用一句“small sample”把所有八格的负结果全部抹掉。

---

# 24. 论文里应避免的几句话

## 不建议 1

> The features contain no signal because training accuracy is also low.

问题：

\[
\text{high training error}
\not\Rightarrow
\text{no feature-label dependence}.
\]

---

## 不建议 2

> Nested CV is biased in our small cells.

问题：

proper nested CV 的主要作用就是降低 model-selection optimism；Vabalas et al. 的模拟还显示 fully nested CV 在其设置下跨 sample sizes 保持近似 unbiased。

更稳：

> Nested CV limits model-selection bias, but the resulting performance estimates remain highly uncertain at small sample sizes.

---

## 不建议 3

> The 3pp union improvement proves representation diversity is useful.

问题：

没有 `A+A` control 就无法排除普通 repeated-sampling gain。

更稳：

> The 3pp union improvement shows complementarity at the observed-run level; attributing it specifically to representation diversity requires comparison with a matched same-mode repeated-sampling baseline.

---

# 25. 最小可行新增实验

如果论文时间紧，只补最必要的：

## Experiment 1：same-mode repeat baseline

对最佳 mode \(A\)：

- 每 task 再采若干 independent runs；
- 估：
  - pass@1；
  - pass@2；
- 对比：
  \[
  A+A
  \]
  和：
  \[
  A+B.
  \]

这是 Q1 最关键的补洞。

---

## Experiment 2：8-cell learning curves

每格画：

- x：training sample count；
- y：primary predictive metric；
- training curve；
- outer held-out curve；
- uncertainty ribbon。

不需要把每格塞大量模型，只需要围绕最终 classifier pipeline。

---

## Experiment 3：pipeline-level label permutation

对每格：

- 定义一个 pre-specified statistic；
- permutation labels；
- 完整重跑 nested procedure；
- 得一个 raw \(p_i\)。

然后：

\[
p_1,\dots,p_8
\]

做 Holm。

如果 computational budget 不够，宁愿降低模型搜索复杂度，也不要做“permute 后只重新 fit 最终模型但不重新 selection”的半套 test。

---

# 26. Q2(d) 针对你当前八格的结论

在你当前描述下：

> “在 8 个数据格上各跑一次同样的检验，想知道 cheap features 是否在这些设置里提供可靠 predictive value。”

默认最合理的是：

\[
\boxed{m=8}
\]

前提：

- 每格输出一个预先定义的 pipeline-level p-value；
- 八格共同服务一个跨格结论；
- 任意格显著都会被当作 positive evidence。

如果你把八格当成八个完全不同、预注册且独立解释的问题，family 才有可能另行划分。

---

# 27. `UNCERTAIN`

以下部分必须明确区分“已有规范术语”与“我对文献实践的描述”。

## UNCERTAIN 1：Q1(b) 没有找到一个全领域统一命名的 test

没有找到一篇经典论文宣布：

> “区分 diversity gain 与 resampling gain 的标准 test 就叫 X test。”

文献中真实存在的是几个相邻概念：

- homogeneous vs heterogeneous ensembles；
- resampling-induced diversity；
- stationary vs diversified sampling；
- matched-compute / matched-budget baseline。

因此：

> **matched-budget homogeneous-vs-heterogeneous sampling control**

是一个准确的方法学描述，但不要声称它是某篇论文正式命名的专有统计检验。

**确定有出处的部分：**

- homogeneous / heterogeneous ensemble：Bian & Wang 2007；
- resampling 对 diversity 的影响：Valdovinos et al. 2007；
- diverse vs stationary prompt sampling：Wang et al. 2025。

---

## UNCERTAIN 2：没有找到规范出处支持“训练误差差 ⇒ features 无信号”

这一点不是简单的“没搜到”。

现有规范 learning-curve 文献明确把这种 pattern 解释为：

> high bias / underfitting

而不是：

> no feature signal.

所以 Q2(b) 若想要一个“有出处的判据”，正确判据是：

\[
\boxed{
\text{train/test both poor + small gap}
\Rightarrow
\text{high-bias-like}
}
\]

而不是：

\[
\boxed{
\text{train/test both poor}
\Rightarrow
\text{no signal}
}
\]

后一个 implication 不应写进论文。

---

## UNCERTAIN 3：“permutation 不显著”也不是 no-signal proof

label permutation 是最接近你问题的 pipeline-level null test，但：

\[
p>0.05
\]

只能说明当前数据没有足够 evidence 拒绝 null。

尤其你的一些格只有 15 个 positives，这可能同时意味着：

- predictive signal 很弱；
- sample size 太小；
- 两者都有。

如果要真的建立“effect 小到可以视为 practically absent”，那会进入 equivalence / smallest-effect-of-interest 的问题，已经超出普通 classifier permutation test 能给出的结论。

---

## UNCERTAIN 4：“small-sample nested CV bias”不能笼统说成固定方向

我核到的直接规范文献更支持：

- non-nested selection → optimistic bias；
- proper nested CV → 大幅降低 selection bias；
- small \(n\) → high variance / wide uncertainty / low power。

所以不要把：

> small sample

自动翻译成：

> nested CV has pessimistic bias

或：

> nested CV has optimistic bias.

---

# 28. 推荐引用优先级

如果主文空间有限，只留最关键的 8 篇：

1. **Chen et al. 2021** — pass@k 无偏 estimator  
   `arXiv:2107.03374`

2. **Bian & Wang 2007** — homogeneous vs heterogeneous ensemble  
   `DOI:10.3233/HIS-2007-4204`

3. **Wang et al. 2025** — diverse vs stationary LLM sampling  
   `arXiv:2502.11027`

4. **Viering & Loog 2022** — learning curves 综述  
   `DOI:10.1109/TPAMI.2022.3220744` / `arXiv:2103.10948`

5. **Emmert-Streib & Dehmer 2019** — train/test learning-curve high-bias vs high-variance 判据  
   `DOI:10.3390/make1010032`

6. **Ojala & Garriga 2009/2010** — classifier label permutation  
   `DOI:10.1109/ICDM.2009.108`

7. **Varma & Simon 2006** — nested CV / model-selection bias  
   `DOI:10.1186/1471-2105-7-91`

8. **Varoquaux 2018** — small-sample CV 大误差条  
   `DOI:10.1016/j.neuroimage.2017.06.061` / `arXiv:1706.07581`

若论文里实际进行 permutation + multiple correction，再加：

9. **Phipson & Smyth 2010**  
   `DOI:10.2202/1544-6115.1585`

10. **Holm 1979**  
    `DOI:10.2307/4615733`

11. **Hoffmann et al. 2026**  
    `DOI:10.1002/bimj.70148`

---

# 29. 完整参考文献与稳定 ID

## [R1] Kulal et al. — pass@k 前序 functional-search 文献

Kulal, S., Pasupat, P., Chandra, K., Lee, M., Padon, O., Aiken, A., & Liang, P.  
**SPoC: Search-based Pseudocode to Code.** 2019.

- arXiv: `1906.04908`
- https://arxiv.org/abs/1906.04908

---

## [R2] Chen et al. — pass@k 无偏 estimator

Chen, M., Tworek, J., Jun, H., et al.  
**Evaluating Large Language Models Trained on Code.** 2021.

- arXiv: `2107.03374`
- https://arxiv.org/abs/2107.03374

关键用途：

- pass@k 定义；
- \(n\ge k\) 多样本估计；
- 无偏 estimator：
  \[
  1-\frac{\binom{n-c}{k}}{\binom nk}.
  \]

---

## [R3] Bian & Wang — homogeneous vs heterogeneous ensemble

Bian, S., & Wang, W.  
**On diversity and accuracy of homogeneous and heterogeneous ensembles.**  
*International Journal of Hybrid Intelligent Systems*, 4(2), 103–128, 2007.

- DOI: `10.3233/HIS-2007-4204`
- https://doi.org/10.3233/HIS-2007-4204

---

## [R4] Wang et al. — LLM diversified sampling

Wang, T., Liu, Z., Chen, Y., Light, J., Chen, H., Zhang, X., & Cheng, W.  
**Diversified Sampling Improves Scaling LLM inference.** 2025.

- arXiv: `2502.11027`
- https://arxiv.org/abs/2502.11027

---

## [R5] Valdovinos et al. — resampling 与 diversity

Valdovinos, R. M., Sánchez, J. S., & Gasca, E.  
**Influence of Resampling and Weighting on Diversity and Accuracy of Classifier Ensembles.**  
IbPRIA 2007.

- DOI: `10.1007/978-3-540-72849-8_32`
- https://doi.org/10.1007/978-3-540-72849-8_32

---

## [R6] Kuncheva & Whitaker — diversity measures

Kuncheva, L. I., & Whitaker, C. J.  
**Measures of diversity in classifier ensembles and their relationship with the ensemble accuracy.**  
*Machine Learning*, 51(2), 181–207, 2003.

- DOI: `10.1023/A:1022859003006`
- https://doi.org/10.1023/A:1022859003006

---

## [R7] Viering & Loog — learning curves review

Viering, T., & Loog, M.  
**The Shape of Learning Curves: a Review.**

- arXiv: `2103.10948`
- DOI: `10.1109/TPAMI.2022.3220744`
- https://arxiv.org/abs/2103.10948
- https://doi.org/10.1109/TPAMI.2022.3220744

---

## [R8] Emmert-Streib & Dehmer — bias/variance learning-curve diagnosis

Emmert-Streib, F., & Dehmer, M.  
**Evaluation of Regression Models: Model Assessment, Model Selection and Generalization Error.**  
*Machine Learning and Knowledge Extraction*, 1(1), 521–551, 2019.

- DOI: `10.3390/make1010032`
- https://doi.org/10.3390/make1010032

---

## [R9] Ojala & Garriga — classifier permutation test

Ojala, M., & Garriga, G. C.  
**Permutation Tests for Studying Classifier Performance.**

ICDM 2009 conference version:

- DOI: `10.1109/ICDM.2009.108`
- https://doi.org/10.1109/ICDM.2009.108

Extended JMLR version:

- *Journal of Machine Learning Research*, 11, 1833–1863, 2010
- https://www.jmlr.org/papers/v11/ojala10a.html

**注意：**不要把 ICDM DOI 错写成 JMLR 文章自己的 DOI。

---

## [R10] Varma & Simon — nested CV

Varma, S., & Simon, R.  
**Bias in error estimation when using cross-validation for model selection.**  
*BMC Bioinformatics*, 7, 91, 2006.

- DOI: `10.1186/1471-2105-7-91`
- https://doi.org/10.1186/1471-2105-7-91

---

## [R11] Vabalas et al. — limited-sample validation

Vabalas, A., Gowen, E., Poliakoff, E., & Casson, A. J.  
**Machine learning algorithm validation with a limited sample size.**  
*PLOS ONE*, 14(11), e0224365, 2019.

- DOI: `10.1371/journal.pone.0224365`
- https://doi.org/10.1371/journal.pone.0224365

---

## [R12] Varoquaux — small-sample CV uncertainty

Varoquaux, G.  
**Cross-validation failure: Small sample sizes lead to large error bars.**  
*NeuroImage*, 180, 68–77, 2018.

- arXiv: `1706.07581`
- DOI: `10.1016/j.neuroimage.2017.06.061`
- https://arxiv.org/abs/1706.07581
- https://doi.org/10.1016/j.neuroimage.2017.06.061

---

## [R13] Phipson & Smyth — permutation p-values

Phipson, B., & Smyth, G. K.  
**Permutation P-values Should Never Be Zero: Calculating Exact P-values When Permutations Are Randomly Drawn.**  
*Statistical Applications in Genetics and Molecular Biology*, 9, Article 39, 2010.

- DOI: `10.2202/1544-6115.1585`
- arXiv: `1603.05766`
- https://doi.org/10.2202/1544-6115.1585
- https://arxiv.org/abs/1603.05766

---

## [R14] Holm — multiple testing

Holm, S.  
**A Simple Sequentially Rejective Multiple Test Procedure.**  
*Scandinavian Journal of Statistics*, 6(2), 65–70, 1979.

- DOI / JSTOR stable identifier: `10.2307/4615733`
- https://doi.org/10.2307/4615733

---

## [R15] Bender & Lange — family / when to adjust

Bender, R., & Lange, S.  
**Adjusting for Multiple Testing—When and How?**  
*Journal of Clinical Epidemiology*, 54(4), 343–349, 2001.

- DOI: `10.1016/S0895-4356(00)00314-0`
- https://doi.org/10.1016/S0895-4356(00)00314-0

---

## [R16] Hoffmann et al. — family boundary / selective emphasis

Hoffmann, S., Lemster, S., Collins, G., Hapfelmeier, A., Heinze, G., Mayr, A., Schmid, M., Wilcke, J. C., & Boulesteix, A.-L.  
**When to Adjust for Multiple Testing: A Unifying Guiding Principle.**  
*Biometrical Journal*, 68(4), e70148, 2026.

- DOI: `10.1002/bimj.70148`
- https://doi.org/10.1002/bimj.70148

---

# 30. 最终推荐：论文中的科学主张边界

## 对 Q1

在没有 same-mode repeated runs 前，你最多可以说：

> Different modes show observed-run complementarity.

不要说：

> Diversity provides an improvement beyond resampling.

补上 `A+A` 后，如果：

\[
A+B>A+A,
\]

才可以写：

> Cross-mode complementarity exceeds the gain obtainable from an additional same-mode sample under a matched inference budget.

---

## 对 Q2

如果 classifier 在八格都失败，正确主张不是：

> The cheap features contain no signal.

而是：

> No evaluated classifier achieved a Pareto improvement over the fixed policy under nested cross-validation.

然后根据 learning curve 分格：

- **high variance / still rising**：
  > data insufficiency remains a plausible explanation;

- **high bias / plateaued**：
  > additional same-distribution data alone appears unlikely to resolve the limitation under the evaluated representation/model family;

再结合 permutation：

> no statistically reliable predictive structure was detected beyond the label-permutation null.

这是现有方法学文献能够支持、而且 reviewer 最难一句话打掉的版本。
