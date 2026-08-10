# 负结果学术图表设计范式：文献实例与论文图形方案

## 研究背景

本研究的核心结果是**负结果（negative results）**，并且需要通过图形而不是仅靠文字论证，让读者快速理解两个核心结论：

1. **学到的分类器没有 Pareto 胜过平凡固定策略。**  
   在 8 个数据格（cells）上，每格都有多个候选策略，每个策略对应二维坐标：
   - 成功率（success rate）
   - 成本（cost）

   结果是：**没有一个 learned classifier / learned strategy 同时做到成功率不低于固定策略、且成本不高于固定策略。**

2. **失败机制来自标签产生率（label yield）不足。**  
   标签只会在任务被成功解开后产生，而基础成功率只有约 **2–27%**。  
   因此监督信号经历如下逐级损耗：

   ```text
   总任务数
      ↓
   被解开的任务
      ↓
   真正产生标签
      ↓
   足以训练分类器的有效样本
   ```

   最终 6 个数据格中有 **4/6 根本没有足够标签进入可训练区间**。

这两个结果的共同问题是：

> 负结果如果只是“点不高”“柱子不大”“曲线没提升”，很容易显得空洞。  
> 最有说服力的设计通常不是强调“结果很差”，而是先定义**方法如果成功，图上应该出现在哪里**，再让读者一眼看到那个区域为空、或关键阈值没有被跨过。

---

# 1. 可直接借鉴的公开论文图形范式

以下论文都适合作为“如何把方法不 work / 信号不存在 / baseline 没被击败”画得有说服力的参考。

| 标题 | 年 | 可核验 ID / 链接 | 图号 | 具体图形做法 | 为什么有效 |
|---|---:|---|---|---|---|
| **LLMRouterBench: A Massive Benchmark and Unified Framework for LLM Routing** | 2026 | DOI: `10.18653/v1/2026.findings-acl.1881`；ACL Anthology: https://aclanthology.org/2026.findings-acl.1881/ | **Fig. 6** | **每个 router 一行的 baseline-relative 水平比较图**。用一条共同的 `0` 竖直参考线表示“相对于 Best Single / GPT-5 baseline 没有变化”。一侧画 performance gain，另一项表示在保持 baseline accuracy 时的 cost saving；无法达到 baseline accuracy 的方法直接标为 **N/A**。因此失败方法表现为“performance 落在 0 左侧”或“根本不存在 cost-saving operating point”。 | 把原本二维的 accuracy–cost trade-off 转成“是否跨过 baseline 可行性边界”的**证书式图形**；读者不用自己逐个检查 Pareto 平面。 |
| **Do ImageNet Classifiers Generalize to ImageNet?** | 2019 | arXiv: `1902.10811`；https://arxiv.org/abs/1902.10811 | **Fig. 1** | 两个 scatter panel。每个分类器一个点：`x = 原 ImageNet test accuracy`，`y = 新测试集 accuracy`。粗黑虚线画出 **`y = x` 的 ideal reproducibility line**，同时画实际拟合线和 bootstrap CI。几乎整团点系统性落在 identity line 下方。 | 把“应该复现”定义成一个明确的**几何边界**。负结论不是靠文字宣布，而是理想线以上/附近的位置系统性缺失。 |
| **On Calibration of Modern Neural Networks** | 2017 | arXiv: `1706.04599`；https://arxiv.org/abs/1706.04599 | **Fig. 1** | 下排使用 **reliability diagram**：每个 confidence bin 的实际 accuracy 与 perfect calibration 对角关系比较，并把两者之间的区域明确表现为 calibration **Gap**。 | “信号/关系失效”被转换成“距离理想边界有多远”。图不是简单显示柱高，而是直接编码“应该在哪里”和“实际偏离多少”。 |
| **Deep Reinforcement Learning That Matters** | 2018 | DOI: `10.1609/aaai.v32i1.11694`；arXiv: `1709.06560`；https://arxiv.org/abs/1709.06560 | **Fig. 5** | 同一个 TRPO、同一套 hyperparameters，把 10 个随机种子拆成两组各 5 个，分别画成**两条平均 learning curve + 不确定性带**。理论上同分布、同设置应该高度重合，但两组曲线持续分离。 | 把“不可复现”定义成“本来应该重合却没有重合”，负结果依赖一个非常清晰的**反事实参照**。 |
| **A Closer Look at Few-shot Classification** | 2019 | arXiv: `1904.04232`；https://arxiv.org/abs/1904.04232 | **Fig. 4** | 把不同实验场景按 **domain difference 从小到大**排序；每个场景并列多种复杂 few-shot 方法和简单 Baseline。随着 domain shift 增大，复杂方法优势逐渐消失，部分设置下 Baseline 反而更高。 | 不强调单个“失败点”，而是把**优势随条件变化系统性坍塌**画成趋势，因此负结论更像结构性结果而不是偶然波动。 |

---

# 2. 这些论文真正值得借鉴的共同原则

上述图虽然形式不同，但设计逻辑高度一致：

> **先定义“方法如果真的 work，图上应该出现在哪里”，再用数据证明该区域为空、该边界没有被跨过、或该理想关系没有出现。**

几个典型例子：

- Recht et al.：
  - 理想结果 = 点应该贴近 `y = x`
  - 实际结果 = 整体系统性落在理想线下

- Guo et al.：
  - 理想结果 = confidence 与 empirical accuracy 一致
  - 实际结果 = 对角线与真实柱高之间形成可见 gap

- LLMRouterBench：
  - 理想结果 = router 应该进入“性能不降且成本下降”的区域
  - 实际结果 = 很多方法连 baseline accuracy 都无法维持，cost saving 无法定义

因此，负结果图最重要的不是“画得满”，而是：

> **让“空白”本身具有语义。**

---

# 3. Q1：8 个数据格 × 多策略 × 二维成本–成功率  
## 如何让“没有一个赢”一眼看出来？

## 3.1 不建议把 8 个独立 Pareto 小图作为主图

最直接的做法是：

```text
Cell 1: cost × success Pareto plane
Cell 2: cost × success Pareto plane
...
Cell 8: cost × success Pareto plane
```

问题在于读者必须重复进行 8 次认知操作：

1. 找到固定 baseline；
2. 找到哪个方向是更好；
3. 检查 learned points 是否进入 dominance region；
4. 最后自己汇总：“哦，8 个格都没有赢。”

这在 appendix 里非常适合用于**审计和细节验证**，但不适合作为 headline figure。

---

# 4. 推荐主图 A：Baseline-normalized Dominance Plane

对每个 cell \(j\)，固定 baseline 为：

\[
(C_{b,j}, S_{b,j})
\]

对 cell \(j\) 中的每个 learned strategy \(i\)，变换坐标：

\[
x_{ij}
=
\log_2 \frac{C_{ij}}{C_{b,j}}
\]

\[
y_{ij}
=
S_{ij} - S_{b,j}
\]

于是，不同 cell 的固定 baseline 都被统一映射到：

\[
(0,0)
\]

## 4.1 坐标轴含义

### 横轴：相对成本

\[
x = \log_2(C / C_b)
\]

解释非常直观：

- \(x = 0\)：与 baseline 成本相同
- \(x = -1\)：成本减半
- \(x = +1\)：成本翻倍
- \(x < 0\)：比 baseline 更便宜
- \(x > 0\)：比 baseline 更贵

相对于直接画：

\[
C - C_b
\]

log-ratio 更适合不同数据格之间 baseline cost 量级不一致的情况。

---

### 纵轴：成功率增益

\[
y = S - S_b
\]

建议直接用**百分点（percentage points, pp）**，例如：

```text
+3.2 pp
-1.8 pp
```

而不要优先使用相对百分比：

\[
\frac{S-S_b}{S_b}
\]

因为你的基础成功率可能只有约 2%。  
此时从 2% 到 3% 看起来是“+50%”，视觉和叙述上很容易夸大一个只有 **+1 pp** 的绝对提升。

---

# 5. 关键视觉设计：把 Pareto-winning region 画成一个明确的区域

若目标是：

- 成功率不低于 baseline：
  \[
  y \ge 0
  \]

- 成本不高于 baseline：
  \[
  x \le 0
  \]

那么左上象限就是：

\[
\boxed{x \le 0,\quad y \ge 0}
\]

也就是：

> **Pareto-dominates fixed baseline**

建议把这个象限使用非常淡的底色，并直接在区域内标注：

```text
Pareto improvement region
```

或：

```text
Dominates fixed policy
```

概念示意：

```text
Δ success (pp)

           ↑
      +    │ █████████████████
           │ █               █
           │ █   WIN REGION  █
           │ █               █
           │ █████████████████
           │
───────────●────────────────────────→ relative cost
        baseline
           │
           │        •
      -    │    •        •
           │          •
           │                 •

       cheaper                 more expensive
```

如果实验结果确实是：

> 8 个 cell 中没有任何 learned strategy Pareto 胜过固定策略

那么整个左上 dominance quadrant 将保持空白。

这个“空白”不是图没有信息，而是：

> **理论上最重要的区域中一个点都没有。**

---

# 6. 图内应该直接写结果，而不是逼读者自己计数

建议在图中或 caption 中直接写：

> **0 / 8 cells contain a learned operating point that Pareto-dominates the fixed baseline.**

或者更短：

> **Pareto wins: 0 / 8 cells**

这样读者同时得到：

1. 几何证据：winning region 为空；
2. 汇总证据：0/8；
3. individual points：失败到底是因为贵、因为 success 低，还是两者都差。

---

# 7. 推荐主图 B：8-row Constrained-Gain Certificate Plot

如果版面允许，最强的组合不是单独一个二维图，而是：

- **Panel A**：baseline-normalized dominance plane
- **Panel B**：每格一行的 constrained-gain dot plot

对每个 cell \(j\)，定义：

\[
G_j
=
\max_{i:\;C_{ij}\le C_{b,j}}
\left(S_{ij}-S_{b,j}\right)
\]

解释：

> 在所有**成本不高于 baseline** 的 learned operating points 中，最好能获得多少成功率提升？

那么：

- \(G_j > 0\)：存在 Pareto improvement
- \(G_j = 0\)：最多只能打平 baseline
- \(G_j < 0\)：即使限制成本不超过 baseline，最好的 learned strategy 仍然更差
- 如果根本没有任何 learned point 满足 \(C_i \le C_b\)，则标：
  - `N/A`
  - 或 `no feasible point`

图形形式：

```text
Success gain under cost ≤ baseline

Cell 1            ●───────│ 0
Cell 2        ●───────────│ 0
Cell 3              ●─────│ 0
Cell 4      ●─────────────│ 0
Cell 5           ●────────│ 0
Cell 6        ●───────────│ 0
Cell 7             ●──────│ 0
Cell 8          ●─────────│ 0

                             ↑
                      Pareto threshold
```

如果八个 cell 的点全部落在 0 左侧：

> “没有一个赢”会比 8 个 small multiples 更直接。

---

# 8. 为什么这个 scalar 比“Pareto distance”更安全

一个诱人的替代方案是定义：

```text
distance to Pareto dominance
```

例如把 success deficit 与 cost excess 合成一个 scalar。

不建议把它作为主结果，因为你必须人为决定：

- cost 与 success 如何归一化；
- 两者如何加权；
- 是 L1、L2 还是其他距离；
- 成本增加多少等价于成功率下降多少。

这会人为引入一个没有实验依据的 utility function。

相比之下：

\[
G_j
=
\max_{i:\;C_i\le C_b}(S_i-S_b)
\]

没有跨维度加权。

它直接回答：

> “在不花更多钱的前提下，learned method 能不能成功率更高？”

因此更像一个**可证伪的 Pareto certificate**。

---

# 9. 最推荐的 Result 1 图形组合

## 主文

### Panel A — Normalized Pareto dominance plane

所有 8 个 cell 的 learned operating points 统一到一个坐标系。

显示：

- baseline = `(0,0)`
- 左上 dominance region
- learned points
- `Pareto wins = 0/8`

---

### Panel B — 8-row constrained-gain certificate

每格一行。

显示：

\[
G_j
\]

并以：

```text
0 pp
```

为共同竖直基准线。

---

## Appendix

放完整：

```text
8 × original cost–success Pareto planes
```

用于：

- 审稿人核验；
- 展示原始绝对成本；
- 展示每格不同 baseline；
- 避免别人怀疑 normalization 掩盖结构。

---

# 10. Result 1 的建议 caption

可以考虑这种结构：

> **No learned strategy Pareto-dominates the fixed policy.**  
> (a) Cost and success are normalized to the fixed policy within each cell, placing every baseline at the origin. The shaded upper-left quadrant contains operating points that achieve at least baseline success at no greater cost; no learned operating point enters this region.  
> (b) For each cell, we report the maximum success-rate gain achievable subject to cost not exceeding the fixed-policy cost. All feasible gains are non-positive, yielding 0/8 Pareto wins.

---

# 11. 统计上的一个重要风险：Pareto ordering 可能受估计噪声影响

如果 success rate 是有限测试任务上的比例估计，例如：

\[
\hat{S} = \frac{k}{n}
\]

那么一个：

```text
+0.3 pp
```

或：

```text
-0.4 pp
```

的差异可能只是 sampling noise。

审稿人可能问：

> “你说它没有 Pareto dominate，是基于 point estimate，还是这种 ordering 在 bootstrap 后仍稳定？”

建议至少考虑以下一种做法。

---

## 11.1 Bootstrap confidence interval

在二维图中对 success axis 加 CI。

或者对：

\[
G_j
\]

画 bootstrap CI。

---

## 11.2 Bootstrap dominance probability

直接估计：

\[
P(
S_i \ge S_b
\land
C_i \le C_b
)
\]

如果成本近似 deterministic，则主要 bootstrap success。

进一步可以报告：

```text
P(any learned strategy dominates baseline)
```

这能避免一个极小噪声导致硬性的：

```text
win / no win
```

判断。

---

# 12. Q2：漏斗式标签耗尽应该怎么画？

你的真实机制是：

```text
Total tasks
   ↓
Solved tasks
   ↓
Labels produced
   ↓
Actually trainable examples
```

而：

```text
solve rate ≈ 2–27%
```

因此最终：

```text
4 / 6 cells
```

没有足够监督信号训练 classifier。

---

# 13. 为什么 Sankey / funnel 不是最优

Sankey 和传统 funnel 擅长表达：

> 数据逐级流失。

但你的核心 takeaway 并不仅是：

> “流失很多。”

真正关键的是：

> **流失之后，剩余监督样本跌到了 classifier 可训练阈值以下。**

因此这是一个：

\[
\textbf{threshold-crossing problem}
\]

而不仅仅是：

\[
\textbf{flow-composition problem}
\]

Sankey 会把读者注意力放在：

- 流宽；
- 百分比；
- 阶段之间谁掉得更多；

但不容易强调：

> **最终剩下的 n 是否足够训练。**

---

# 14. 推荐图：Thresholded Attrition Connected-Dot Plot

最建议的图形：

> **每个 cell 一行，四个阶段作为四个 connected dots，并加一条 trainability threshold。**

示意：

```text
Number of usable examples

                 insufficient         sufficient
                       │
Cell A      ●────●────●│────────────●
Cell B         ●────●──│────────────●
Cell C    ●──●──●──────│──────────────●
Cell D       ●─●─●─────│───────────────●
Cell E             ●───│────●───────────●
Cell F          ●──────│──────●─────────●
                       │
                    n_min
```

四个点分别代表：

1. `All tasks`
2. `Solved`
3. `Labels available`
4. `Trainable / usable`

---

# 15. 关键不是 funnel shape，而是 trainability threshold

定义一个最低监督量：

\[
n_{\min}
\]

对应：

> 低于该数量，训练 classifier 不再具有可接受的统计/工程意义。

在图上画一条明确的竖直线：

```text
minimum supervision required
n_min
```

阈值左侧淡灰底：

```text
insufficient supervision
```

右侧：

```text
trainable regime
```

那么结果会非常直接：

> **4 / 6 final-stage points remain left of the trainability threshold.**

比：

```text
funnel 最下面很窄
```

强得多。

---

# 16. 建议把结构性瓶颈直接标出来

你的标签不是普通 missing data。

更准确的关系是：

\[
\text{Label Exists}
\Rightarrow
\text{Task Solved}
\]

也就是说：

> 没解开的任务，从定义上就无法产生监督标签。

所以在：

```text
All tasks → Solved
```

这一段上可以直接标：

```text
base solve rate: 2–27%
```

以及：

```text
labels only exist for solved tasks
```

这会让读者理解：

> 样本稀缺不是随机的数据清洗损失，而是任务结构决定的监督上限。

---

# 17. 为什么主轴应该用 absolute count，而不是只用百分比

假设：

```text
Cell A
10,000 tasks × 2% solved
= 200 labels
```

而：

```text
Cell B
200 tasks × 27% solved
= 54 labels
```

只看 retention rate：

```text
2% vs 27%
```

会觉得 Cell B 更健康。

但如果 classifier 训练需要：

```text
n_min = 100
```

那么：

- Cell A：可能够训练
- Cell B：依然不够

因此你真正关心的是：

\[
\boxed{\text{absolute usable supervision}}
\]

不是单纯：

\[
\boxed{\text{retention percentage}}
\]

---

# 18. 推荐同时标百分比，但不要让百分比成为主轴

例如每一段旁边小字：

```text
234
↓ 7.2%
17
↓ 88%
15
↓ 60%
9
```

或者：

```text
Solved: 17 / 234 (7.3%)
Labels: 15 / 17 (88%)
Trainable: 9 / 15 (60%)
```

但视觉位置由 absolute \(n\) 决定。

---

# 19. x 轴是否应该用 log scale？

取决于不同 cell 的 task count 是否跨数量级。

## 如果数量级差不大

优先：

```text
linear scale
```

因为“最后一级触底”在 linear scale 上最直观。

---

## 如果不同 cell 跨 10× 甚至 100×

可以考虑：

```text
log x-axis
```

但需要小心。

如果 final count 出现：

```text
0
1
2
```

log scale 会变得尴尬，也可能把真正的“触底”视觉上放大成一个看起来并不低的位置。

因此如果低值本身是结果，宁可：

- 保持 linear；
- 使用局部 inset；
- 或拆成两个 panel；

也不要为了压缩尺度牺牲 takeaway。

---

# 20. Result 2 的推荐主图布局

建议做成：

```text
6 rows × 4 stages
```

每个 cell 一行。

阶段：

```text
Total
Solved
Labelled
Trainable
```

视觉编码：

- `Total`：浅色点
- `Solved`：中等强调
- `Labelled`：进一步强调
- `Trainable`：最深 / 实心点
- 阶段之间用细线连接
- `n_min`：粗一点的竖直阈值线
- 阈值左侧浅灰底：
  `insufficient supervision`

caption 直接写：

> **Supervision collapses before the classifier-training stage. Four of six cells end below the minimum supervision required for training because labels can only be generated on successfully solved tasks, whose base success rates range from 2% to 27%.**

---

# 21. Result 1 与 Result 2 可以使用同一种视觉语言

这是整个设计里最值得利用的一点。

两个负结果其实可以被统一描述成：

> **没有跨过一个明确的 feasibility boundary。**

---

## Result 1

边界是：

\[
x \le 0,\quad y \ge 0
\]

即：

```text
Pareto feasibility boundary
```

结论：

> 一个 learned point 都没有进入 Pareto-winning region。

---

## Result 2

边界是：

\[
n \ge n_{\min}
\]

即：

```text
supervision / trainability feasibility boundary
```

结论：

> 4/6 cells 没有跨过 classifier trainability threshold。

---

# 22. 论文中可以形成一个非常统一的叙事

可以把两张图连续安排：

---

## Figure X — Learning fails to improve the cost–success frontier

核心视觉：

```text
expected winning region = empty
```

回答：

> **Does learning beat the trivial fixed policy?**

答案：

> **No.**

---

## Figure Y — Learning fails because supervision collapses upstream

核心视觉：

```text
required training threshold = not reached
```

回答：

> **Why does learning fail?**

答案：

> **Because labels can only be produced after successful task completion, and low base success rates leave most cells below the trainability regime.**

---

这样图与图之间形成真正的因果链：

\[
\text{No Pareto gain}
\]

↓

\[
\text{Insufficient learned signal}
\]

↓

\[
\text{Insufficient labels}
\]

↓

\[
\text{Low task success rate}
\]

---

# 23. 推荐最终主文版式

## Figure 1：Pareto failure

### Panel A
**Normalized cost–success dominance plane**

内容：

- 所有 8 个 cell 的 learned operating points
- 所有 fixed baselines 对齐到 `(0,0)`
- 左上角 shaded Pareto-winning region
- 图内：
  ```text
  Pareto wins: 0 / 8
  ```

### Panel B
**Cell-wise constrained success gain**

每个 cell 一行：

\[
G_j
=
\max_{i:C_i \le C_b}
(S_i-S_b)
\]

共同 baseline：

```text
0 pp
```

---

## Figure 2：Supervision attrition

**Thresholded connected-dot attrition plot**

每个 cell 一行：

```text
Total → Solved → Labelled → Trainable
```

加：

```text
n_min
```

阈值线。

图内：

```text
4 / 6 cells below trainability threshold
```

并在：

```text
Total → Solved
```

附近标：

```text
base success = 2–27%
```

---

# 24. 为什么这套设计比“8 个 Pareto 图 + 一个 funnel”更强

原方案：

```text
8 × Pareto small multiples
+
1 × funnel
```

主要问题：

### 问题 1：主结论需要读者自己汇总

读者必须看完八格才知道：

```text
0 / 8
```

---

### 问题 2：funnel 强调的是流失比例，而不是统计可训练性

它无法直接回答：

```text
为什么 classifier 没学出来？
```

---

### 问题 3：两张图缺少统一视觉语法

Pareto 图讲二维 trade-off。

Funnel 讲 flow。

两者视觉上没有共同的“失败机制”。

---

新的方案：

```text
Figure 1:
没有进入 winning region

Figure 2:
没有跨过 training threshold
```

统一成：

> **feasibility boundary not crossed**

这更像一篇 measurement / negative-result paper 的叙事，而不是两个独立 exploratory charts。

---

# 25. 推荐视觉细节

## 25.1 不要用红色把失败画得过于戏剧化

最好：

- baseline / threshold：黑或深灰
- learned methods：正常论文配色
- forbidden / insufficient 区域：非常浅的灰
- winning region：非常淡的强调色

重点应该是：

```text
geometry
```

而不是：

```text
warning color
```

---

## 25.2 空区域必须有明确 label

不要只靠读者自己推断左上角是什么。

直接标：

```text
Pareto-improving region
```

否则“一个点都没有”可能被理解成：

```text
数据没有覆盖这个区域
```

而不是：

```text
方法没有赢
```

---

## 25.3 阈值必须有语义名称

不要只写：

```text
n = 100
```

而写：

```text
minimum training support
n_min = 100
```

或者：

```text
trainability threshold
```

---

## 25.4 重要 summary number 直接进入图形

例如：

```text
0 / 8 Pareto wins
```

以及：

```text
4 / 6 cells below training threshold
```

这不是“把 caption 写进图”，而是让图完成 headline communication。

---

# 26. 论文正文可以怎么解释这两张图

可以形成非常简洁的正文逻辑：

### 第一段

> Across all eight evaluation cells, none of the learned strategies Pareto-dominates the corresponding fixed policy. After normalizing cost and success relative to the fixed strategy within each cell, no learned operating point enters the region corresponding to non-inferior success at non-greater cost.

### 第二段

> The failure is consistent with a supervision bottleneck rather than merely poor classifier choice. Labels are only generated for tasks that are successfully solved, while base success rates range from 2% to 27%. Consequently, supervision collapses before the classifier-training stage, leaving four of six training cells below the minimum support required for learning.

---

# 27. 可以借鉴但不建议直接照搬的图形

## Sankey

适合：

```text
不同来源流向不同终点
```

不适合本研究，因为真正 takeaway 是：

```text
是否达到训练阈值
```

---

## Funnel

适合：

```text
conversion / retention
```

但容易把：

```text
2%
27%
```

这种 percentage 变成视觉主体，而弱化 absolute sample count。

---

## 8 个原始 Pareto small multiples

适合 appendix。

主文中结论提取成本太高。

---

## 单一综合 Pareto distance

不推荐。

因为需要隐藏式 utility weighting。

---

# 28. 最终建议

如果主文空间允许两张核心图：

## Figure A — “No learned strategy wins”

使用：

> **baseline-normalized dominance plane + 8-row constrained-gain certificate**

核心 takeaway：

```text
0 / 8 Pareto wins
```

---

## Figure B — “Why learning fails”

使用：

> **thresholded attrition connected-dot plot**

核心 takeaway：

```text
4 / 6 cells fall below the trainability threshold
```

---

这两张图的统一设计原则可以概括为：

> **Negative results are most persuasive when the figure explicitly marks where success would have appeared, and the data visibly fail to cross that boundary.**

你的两个结果正好分别对应：

```text
empty success region
```

和：

```text
uncrossed feasibility threshold
```

这是比传统的：

```text
“柱子比较矮”
```

或者：

```text
“漏斗最后比较窄”
```

更适合论文 headline figure 的表达。

---

# 29. 文献清单

## LLMRouterBench

**LLMRouterBench: A Massive Benchmark and Unified Framework for LLM Routing**

- Year: 2026
- Venue: Findings of ACL 2026
- DOI: `10.18653/v1/2026.findings-acl.1881`
- ACL Anthology:
  https://aclanthology.org/2026.findings-acl.1881/
- PDF:
  https://aclanthology.org/2026.findings-acl.1881.pdf
- Relevant figure: **Fig. 6**

---

## Recht et al.

**Do ImageNet Classifiers Generalize to ImageNet?**

- Year: 2019
- arXiv: `1902.10811`
- URL:
  https://arxiv.org/abs/1902.10811
- Relevant figure: **Fig. 1**

---

## Guo et al.

**On Calibration of Modern Neural Networks**

- Year: 2017
- arXiv: `1706.04599`
- URL:
  https://arxiv.org/abs/1706.04599
- Relevant figure: **Fig. 1**

---

## Henderson et al.

**Deep Reinforcement Learning That Matters**

- Year: 2018
- DOI: `10.1609/aaai.v32i1.11694`
- arXiv: `1709.06560`
- URL:
  https://arxiv.org/abs/1709.06560
- Relevant figure: **Fig. 5**

---

## Chen et al.

**A Closer Look at Few-shot Classification**

- Year: 2019
- arXiv: `1904.04232`
- URL:
  https://arxiv.org/abs/1904.04232
- Relevant figure: **Fig. 4**

---

# 30. UNCERTAIN

**None in the literature table above.**

The listed figure numbers and high-level graphical descriptions were treated as verified in the preceding literature pass.

However, before the final manuscript is frozen, the following elements of the **proposed new figures** still depend on project-specific definitions and therefore should not be presented as externally established facts:

1. The exact definition and numerical value of:
   \[
   n_{\min}
   \]
   i.e. the trainability threshold.

2. Whether:
   \[
   C_i \le C_b
   \]
   should use point estimates, expected cost, median cost, or another project-specific cost definition.

3. Whether Pareto comparisons should be made using point estimates or uncertainty-aware intervals.

4. Whether all 8 cells can legitimately be normalized onto the same cost-ratio axis without hiding a qualitatively important difference in absolute cost.

These are **analysis/design choices**, not uncertainties about the cited figure numbers.

---

# One-sentence takeaway

> **For both negative results, the strongest design is not to visualize “poor performance” itself, but to visualize a clearly defined success boundary and show that the learned system never crosses it.**
