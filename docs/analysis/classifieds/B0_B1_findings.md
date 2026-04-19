# B0 vs B1 Classifieds 跨模型对比报告

> B0: Qwen3-VL-235B-A22B（proxy API，temperature=0.1，max_tokens=4096）
> B1: Qwen3-VL-4B bf16（本地推理，do_sample=False，max_new_tokens=384）
> 站点: Classifieds (OSClass), 234 tasks × 3 modes
> 本报告关注模型规模（4B vs 235B）对三种观测模式的差异化影响

---

## 1. 核心对比表

### 1.1 Adjusted SR 对比

| 模式 | B0 (235B) | B1 (4B) | 差值 | 方向 |
|------|-----------|---------|------|------|
| DOM | **8.07%** | 0.85% | **+7.22pp** | B0 >> B1 |
| SoM | 12.05% | **16.24%** | **-4.19pp** | B0 << B1 |
| Vision | **10.71%** | 8.12% | **+2.59pp** | B0 > B1 |

**SoM 模式出现反转**：更大的 235B 模型在 SoM 上不如 4B 模型，这是核心异常。DOM 和 Vision 方向符合模型规模预期（更大 = 更强）。

### 1.2 Raw SR 对比

| 模式 | B0 Raw | B1 Raw | 差值 |
|------|--------|--------|------|
| DOM | 15.02% | 8.97% | +6.05pp |
| SoM | 15.81% | 20.51% | -4.70pp |
| Vision | 14.10% | 12.39% | +1.71pp |

Raw SR 方向与 adjusted SR 一致，SoM 反转在去噪前后均成立。

### 1.3 效率对比

| 指标 | B0 DOM | B1 DOM | B0 SoM | B1 SoM | B0 Vision | B1 Vision |
|------|--------|--------|--------|--------|-----------|-----------|
| 平均步数 | **14.1** | 14.9 | **8.3** | 11.8 | **8.0** | 8.0 |
| 平均成本/ep | $0.046 | $0.074 | $0.041 | $0.077 | $0.026 | $0.029 |

> 注：B0 成本为 API 实际调用费用；B1 成本为本地 GPU 推理的 API 等价估算（基于 token 量×API 定价）。两者成本体系不同，直接比较需谨慎。

B0 步数在 DOM 和 SoM 上均低于 B1（DOM: 14.1 vs 14.9；SoM: 8.3 vs 11.8），说明 235B 模型决策更快，不需要像 4B 模型那样多探索步骤。

---

## 2. SoM 模式反转深度分析

### 2.1 现象

B0 (235B) SoM adjusted SR = 12.05%，比 B1 (4B) SoM 16.24% 低 4.19pp。这与模型规模假设（更大模型一般更强）相反。

### 2.2 假说 A：parse_error drag（最可能解释）

| 指标 | B0 SoM | B1 SoM |
|------|--------|--------|
| parse_error 率 | **20.1%** (45/224) | ~<5%（估算） |
| 解码策略 | temperature=0.1 | do_sample=False（贪婪） |
| max_tokens | 4096 | 384 |

**B0 SoM 的 20.1% parse_error 率拖累了成功率**。

假设 parse_error episode 若能正常执行，SR 提升估算：
- 最乐观（所有 parse_error 都能成功）：12.05% + 45/(224) × 调整 = ~32%（高估，不现实）
- 保守估算（10% parse_error episode 可成功）：12.05% + 4-5pp ≈ 16-17%

修复 parse_error 后 B0 SoM **可能接近或超过 B1 SoM（16.24%）**，SoM 反转可能消失。

parse_error 高发原因：
1. **temperature=0.1 + 多模态输入**：SoM 同时发送图文，推理更复杂，轻度随机采样偶尔生成格式错误的 JSON
2. **B1 贪婪解码（do_sample=False）**：输出极度确定性，JSON 格式极稳定
3. **可能的 proxy API 格式问题**（§46/§47 已修复部分）

### 2.3 假说 B：Text-over-Vision 随模型规模增强

SoM 模式同时提供图文，"文字推理盖过图像"（text-over-vision）的 Mirage Effect 可能随模型规模增大而增强：

| 模型 | 文字推理能力 | text-over-vision 程度 | SoM 的视觉利用率 |
|------|------------|---------------------|----------------|
| B1 (4B) | 弱（有时文字跟不上） | 中 | 较高（文字覆盖不完全，视觉补充） |
| B0 (235B) | 强（详尽文字推理） | **可能更强** | 可能更低（文字主导） |

证据：B1 Vision_digest task 14 显示 SoM 模式下 thought 与 DOM 完全一致（text-over-vision 典型案例）。235B 模型生成更长、更详细的 thought，可能进一步压制图像感知。

**这个假说无法仅从汇总数据验证**，需要 episode-level thought 内容分析。

### 2.4 假说 C：Capability-Environment Gap 在 SoM 下更严重

B0 DOM digest 记录了 `<select>` 下拉菜单问题：235B 模型更执着于"正确"路径，cycle detection 更快触发。这个模式可能在 SoM 模式下更普遍——235B 模型的自信反而减少了有益探索。

B1 4B 模型因推理不确定性偶尔偏离当前路径，反而找到有效替代方案（如发现侧边栏分类链接）。这是"exploration vs exploitation" tradeoff 的规模悖论。

### 2.5 假说权重评估

| 假说 | 证据强度 | 可验证性 | 预期贡献 |
|------|---------|---------|---------|
| A: parse_error drag | 强（20.1% 可直接观测） | 高（修复 parse_error 重跑） | **主要因素**（~3-4pp） |
| B: text-over-vision scale | 中（间接推断） | 中（需 thought 分析） | 次要因素（~1-2pp） |
| C: capability-environment gap | 弱（task 级案例） | 高（cycle 触发统计） | 少量因素（~0-1pp） |

---

## 3. DOM 模式：模型规模带来显著提升

### 3.1 SR 对比

| 指标 | B0 DOM | B1 DOM | 倍数 |
|------|--------|--------|------|
| Raw SR | 15.02% | 8.97% | 1.67× |
| Adjusted SR | 8.07% | 0.85% | **9.5×** |
| 真实成功数（估计） | 18 | 2 | 9× |

Adjusted 后的 9.5 倍差距几乎都来自非视觉任务的能力提升。

### 3.2 B0 DOM 特有行为（B1 不具备）

1. **翻页（33+ task）**：B0 主动点击分页控件导航到 iPage=2/3/4，B1 从不翻页
2. **价格区间筛选（21+ task）**：独立填写 sPriceMin/sPriceMax 字段，B1 很少使用
3. **多 Tab 切换（4 task）**：B0 prompt 包含 tab_focus，B1 无此能力
4. **更稳定的表单字段聚焦**：通过 element_id 精准定位每个字段

### 3.3 B0 DOM 能力上限估算

- DOM 模式的**视觉信息瓶颈**仍然存在：62/234 tasks（~26.5%）是纯视觉任务，DOM 结构性不可达
- 去除视觉不可达任务（26.5%）和 N/A 任务（4.3%）后，DOM 理论可达任务约 163/234（69.7%）
- B0 DOM 在这些可达任务上 SR ≈ 18/163 ≈ 11%，说明仍有大量可达任务失败——模型能力仍有提升空间
- B1 DOM 在可达任务上 SR ≈ 2/163 ≈ 1.2%，显示 4B 模型基本无能力在 DOM 模式完成非视觉 classifieds 任务

---

## 4. Vision 模式：适度提升

### 4.1 SR 对比

| 指标 | B0 Vision | B1 Vision | 差值 |
|------|-----------|-----------|------|
| Raw SR | 14.10% | 12.39% | +1.71pp |
| Adjusted SR | 10.71% | 8.12% | +2.59pp |

Vision 模式 2.59pp 的提升符合预期：更大模型在纯视觉理解（颜色识别、形状识别、OCR）上更准确。

### 4.2 失败模式差异

| 失败原因 | B0 Vision | B1 Vision | 分析 |
|---------|-----------|-----------|------|
| fail_no_progress | **32.1%** | —（未直接列） | B0 更高，可能因更执着尝试同一操作 |
| 坐标 misclick | 高 | 高 | 两模型均受 VWA viewport 限制 |
| 坐标自纠正 | **有限自纠正** | **零自纠正** | B0 会改变坐标重试；B1 重复完全相同坐标 |
| page_unchanged_rate | 41.8% | 30.1% (B1) | B0 更高，执着行为副作用 |

B0 Vision 的 page_unchanged_rate（41.8%）高于 B1（30.1%），但 SR 更高——说明即使有更多无效步骤，235B 的视觉推理准确性弥补了效率损失。

**坐标自纠正差异**：B1(4B) 在 misclick 后以完全相同的坐标和 confidence 重复点击（如连续 3-4 步 [0.43, 0.85]，见 B1_Vision_digest 3.1 节）。B0(235B) 则会尝试修改坐标（如 task 115 step 1→2 从 [0.43, 0.85] 改为 [428, 841]），但纠正方向不一定正确（该例 y=841 越出 viewport）。这反映了 235B 更强的状态感知能力，尽管坐标精度仍不足以可靠恢复。

### 4.3 Oracle 中 Vision 贡献（adjusted）

| 模型 | Vision oracle 选择 | Vision-only 成功（估）|
|------|--------------------|----------------------|
| B0 | 20 / 46 (43.5%) | ~10-12 |
| B1 | 18 / 46 (39.1%) | 7（精确）|

B0 Vision 在 oracle 中贡献略多，说明 235B 在纯视觉任务上的成功集合更大。

---

## 5. Eager Completion：235B 的过度自信问题

### 5.1 现象

B0 (235B) 存在显著的 **eager completion** 行为：模型在首屏即给出答案，不滚动查看完整页面。从 ≤2 步 genuine finish（排除 parse_error fallback）统计：

| 条件 | Eager finish (≤2步) | 其中成功 | 失败率 | 占总 episode |
|------|---------------------|---------|--------|-------------|
| **B0 SoM** | **71** | 2 | **97.2%** | **45.5%** |
| B0 Vision | 24 | 4 | 83.3% | 11.8% |
| B0 DOM | 23 | 4 | 82.6% | 9.9% |

B0 SoM 近一半 episode 是"看一眼就答"，且 97% 都错了。B1 (4B) 因不确定性更高，平均步数更多（SoM: 11.8 vs B0: 8.3），反而有更多机会通过探索找到答案。

### 5.2 SoM 模式尤其严重的原因

SoM 同时提供截图 + 文字标注，给了 235B 模型 **最大信息密度的首屏**，导致：

1. **过早满足**：模型在首屏看到部分匹配项（如 2 辆红车），判断信息已充足，直接 finish
2. **"on this page" = 当前视口**：模型将 "on this page" 理解为当前可见区域，而非整个网页
3. **不会预判折叠线以下有更多内容**：即使页面头部显示 "37-48 of 7606 listings"（暗示 12 个 item），模型也不据此推理需要 scroll

典型案例：task 43（"red vehicles on this page" 的价格范围），B0 三模式全部在首屏看到 2 辆红车就作答，漏掉 $9999 那辆。

### 5.3 与 capability-environment gap 的关系

Eager completion 是 capability-environment gap 的另一面：

| 维度 | 表现 | 根因 |
|------|------|------|
| Capability-environment gap（§2.5 假说 C） | 235B 更执着于"正确"路径（如反复 click `<select>`） | 能力越强 → 越自信路径正确 → 探索越少 |
| **Eager completion** | 235B 看首屏就 finish，不 scroll | 能力越强 → 越自信信息充足 → 验证越少 |

两者共享相同根因：**235B 的高置信度抑制了探索行为**。4B 模型因推理不确定性被迫多步探索，反而在需要全页扫描的聚合型任务上更有可能成功。

### 5.4 对 SoM 反转的额外解释

这为 §2（SoM B0 < B1 反转）提供了 **假说 D**：

| 假说 | 贡献估算 |
|------|---------|
| A: parse_error drag (20.1%) | ~3-4pp |
| B: text-over-vision scale | ~1-2pp |
| C: capability-environment gap | ~0-1pp |
| **D: eager completion (45.5% SoM)** | **~2-3pp**（69 个失败 eager episode 中估计 5-7 个可通过探索成功）|

假说 A（parse_error）和 D（eager completion）可能共同解释 B0 SoM 的全部反转幅度（4.19pp）。

---

## 6. 路由格局对比

### 6.1 Oracle Ceiling 对比

| 指标 | B0 | B1 |
|------|----|----|
| 最优单模式 SR | 12.05% (SoM) | 16.24% (SoM) |
| Oracle ceiling（adjusted） | **20.54%** | 19.66% |
| Routing headroom | **+8.49pp** | +3.42pp |
| Oracle DOM 贡献 | **12** | 2 |
| Oracle SoM 贡献 | 14 | **26** |
| Oracle Vision 贡献 | **20** | 18 |

**B0 routing headroom（8.49pp）是 B1（3.42pp）的 2.5 倍**——原因是 B0 三模式 SR 更均衡，每种模式都有独特擅长的任务集合，路由有更大空间。

### 6.2 路由策略对比

**B1 最优策略**：以 SoM 为默认（16.24%），视 action_diversity 信号路由到 Vision（7 个 Vision-only 成功），不引入 DOM（贡献极小）。路由提升上限约 +3.42pp。

**B0 最优策略**：三模式均有路由价值：
- **Vision ↔ SoM**：Vision 成本仅 62% of SoM，Vision 贡献 20 个 oracle 选择
- **DOM 重新进入路由**：B0 DOM 12 个 oracle 选择，特别在 single_navigation+program_html（DOM 唯一 7.1%）和 page_reading+url_match（DOM SR 21.4%）类型上有优势
- 三模式路由理论提升上限 +8.49pp，远超 B1

### 6.3 模型能力对路由的含义（§9 Capability-Aware Routing）

B0 vs B1 的路由格局变化验证了 B1_findings.md §7.3 中的预测：

> "最优表征不是绝对的，是模型能力的函数（Read More, Think More 文献）。B0 DOM SR 若显著高于 B1，则 DOM 在高能力模型下重新有路由价值。"

实验证实：B0 DOM SR（8.07%）显著高于 B1（0.85%），DOM **确实重新成为路由候选**。Capability-aware routing 的假设得到支持。

---

## 7. 设计不对称对结论的影响

B0 和 B1 存在以下已知设计不对称（见 MEMORY），对 SR 对比有系统性影响：

| 不对称项 | B0 | B1 | 影响方向 |
|---------|----|----|---------|
| 解码策略 | temperature=0.1 | do_sample=False | SoM parse_error：B0 损失 SR |
| max_new_tokens | 4096 | 384 | B1 JSON 截断：B1 损失 SR（SoM/Vision 尤甚）|
| Scroll dy 约定 | 不稳定（±0.5） | 始终 +500 | B0 部分 scroll 方向错误：轻微损失 SR |
| 模型规模 | 235B (22B active) | 4B | 主要能力变量 |

**净效应**：设计不对称对 B1 SoM/Vision 不利（max_new_tokens 截断）；对 B0 SoM 不利（parse_error）。B0 SoM 的劣势可能被这两个方向抵消，模型真实能力差异比观测数据更小。

---

## 8. 关键发现汇总

1. **DOM 能力呈强规模依赖性**：235B vs 4B 在 DOM adjusted SR 上 9.5×（8.07% vs 0.85%），原因是 235B 掌握翻页、价格筛选等复杂导航策略，而 4B 几乎不会。

2. **SoM 反转（B0 < B1）**：可能主要由 parse_error 基础设施问题（20.1%）解释，而非 235B 的真实 SoM 能力不如 4B。修复 parse_error 后预期反转消失。

3. **Vision 符合规模预期**：235B 在纯视觉模式下 +2.59pp，视觉理解准确性提升，但 no_progress 率更高（执着行为副作用）。

4. **B0 路由 headroom 是 B1 的 2.5 倍**（8.49pp vs 3.42pp），三模式均有路由价值，Capability-Aware Routing 假设得到支持。

5. **Eager completion 是 B0 的系统性缺陷**：B0 SoM 45.5% episode 在 ≤2 步 finish（97% 失败），235B 的高置信度抑制了全页探索，与 capability-environment gap 共享根因。这为 SoM 反转提供了假说 D（~2-3pp 贡献）。

6. **SoM 仍是 B0 最优单模式**（12.05%），但领先优势比 B1 小得多。B0 三模式 SR 更均衡（8-12% 区间），相比 B1 的极端分化（0.85% - 16.24%）。

7. **DOM 在高能力模型下重获路由价值**：B0 DOM 贡献 12 个 oracle 选择（vs B1 的 2 个），特别在 program_html 评测类型上（DOM 7.1% SR，SoM/Vision 0%）。

---

## 方法论说明

- **比较局限**：B0 和 B1 存在多项设计不对称（温度/max_tokens/scroll），SR 差异无法完全归因于模型规模
- **DOM adjusted SR 不确定性**：B0 DOM §56 盲区 17 tasks 未完全分类，true adjusted SR 在 [0.4%, 8.07%] 区间
- **McNemar 检验**：B0 vs B1 跨模型的配对检验需要 task-level 成功标签（相同 task 对比），此处仅汇总统计
- **成本比较**：B0（API 费用）与 B1（本地推理 GPU 等价成本）成本体系不同，直接数值比较需注意

---

*生成时间：2026-04-15*
*数据来源：B0_3mode_classifieds_20260413 + B1_3mode_classifieds_20260413（Stub）*
*B0 三模式详情：B0_findings.md；B1 三模式详情：B1_findings.md*
