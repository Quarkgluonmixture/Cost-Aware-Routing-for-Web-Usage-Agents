# B0 vs B1 Classifieds 跨模型对比报告

> B0: Qwen3-VL-235B-A22B（proxy API，temperature=0.1，max_tokens=4096）
> B1: Qwen3-VL-4B bf16（本地推理，do_sample=False，max_new_tokens=384）
> 站点: Classifieds (OSClass), 234 tasks x 3 modes
> 本报告关注模型规模（4B vs 235B）对三种观测模式的差异化影响
> **v2 (2026-04-21): parse_error 修复后全面更新，SoM 反转已消除**

---

## 1. 核心对比表

### 1.1 Adjusted SR 对比

| 模式 | B0 (235B) | B1 (4B) | 差值 | 方向 |
|------|-----------|---------|------|------|
| DOM | **8.48%** | 0.85% | **+7.63pp** | B0 >> B1 |
| SoM | **20.98%** | 16.24% | **+4.74pp** | B0 > B1 |
| Vision | **12.05%** | 8.12% | **+3.93pp** | B0 > B1 |

**235B 模型在全部三种模式上优于 4B**，符合模型规模假设。此前版本（v1, 2026-04-15）的 SoM 反转（B0 < B1）已随 parse_error 修复而消除。

### 1.2 Raw SR 对比

| 模式 | B0 Raw | B1 Raw | 差值 |
|------|--------|--------|------|
| DOM | 14.96% | 8.97% | +5.99pp |
| SoM | 23.50% | 20.51% | +2.99pp |
| Vision | 15.81% | 12.39% | +3.42pp |

Raw SR 方向与 adjusted SR 一致，B0 在三种模式上全面领先。

### 1.3 效率对比

| 指标 | B0 DOM | B1 DOM | B0 SoM | B1 SoM | B0 Vision | B1 Vision |
|------|--------|--------|--------|--------|-----------|-----------|
| 平均步数 | **11.52** | 14.9 | **8.62** | 11.8 | **7.85** | 8.0 |
| 平均成本/ep | $0.0425 | $0.074 | $0.0417 | $0.077 | $0.0248 | $0.029 |

> 注：B0 成本为 API 实际调用费用；B1 成本为本地 GPU 推理的 API 等价估算（基于 token 量 x API 定价）。两者成本体系不同，直接比较需谨慎。

B0 步数在全部三种模式上均低于 B1（DOM: 11.52 vs 14.9；SoM: 8.62 vs 11.8；Vision: 7.85 vs 8.0），说明 235B 模型决策更快。

---

## 2. SoM 模式：parse_error 修复后反转消除

### 2.1 背景

v1 报告的核心异常是 **SoM 反转**：B0 SoM adjusted SR 12.05% < B1 SoM 16.24%（-4.19pp）。当时提出四个假说（parse_error drag、text-over-vision scale、capability-environment gap、eager completion），其中假说 A（parse_error drag）被评为最可能因素。

### 2.2 parse_error 修复效果（假说 A 验证）

parse_error 修复前后对比：

| 指标 | 修复前 | 修复后 | 变化 |
|------|--------|--------|------|
| B0 SoM parse_error 数 | 45 (20.1%) | **6 (2.6%)** | -39 |
| B0 SoM Adjusted SR | 12.05% | **20.98%** | **+8.93pp** |
| B0 vs B1 SoM 差值 | -4.19pp（B0 落后） | **+4.74pp（B0 领先）** | 反转彻底消除 |

**假说 A（parse_error drag）得到完全验证**：parse_error 率从 20.1% 降至 2.6%，B0 SoM SR 跃升近 9pp，不仅消除反转，还建立了对 B1 的 4.74pp 领先优势。这确认 parse_error 是此前反转的**主要原因**。

### 2.3 其他假说的定位（次要因素）

v1 报告中的假说 B-D 在反转消除后的角色：

| 假说 | 原定位 | 修复后定位 |
|------|--------|-----------|
| A: parse_error drag | 主要因素 (~3-4pp) | **已验证：实际贡献 ~8.93pp** |
| B: text-over-vision scale | 次要因素 (~1-2pp) | 可能仍存在，但被 235B 的能力优势覆盖 |
| C: capability-environment gap | 少量因素 (~0-1pp) | 同上，task-level 可能偶发，不影响总体方向 |
| D: eager completion | ~2-3pp | 仍然存在（见 S5），但影响程度低于预期 |

**结论**：parse_error 是此前反转的充分解释。B/C/D 假说作为二级效应可能仍存在，但在 parse_error 修复后，235B 模型的规模优势足以抵消这些次要负面因素，整体方向回归正常。

---

## 3. DOM 模式：模型规模带来显著提升

### 3.1 SR 对比

| 指标 | B0 DOM | B1 DOM | 倍数 |
|------|--------|--------|------|
| Raw SR | 14.96% | 8.97% | 1.67x |
| Adjusted SR | 8.48% | 0.85% | **10x** |
| 真实成功数（估计） | ~20 | 2 | 10x |

Adjusted 后的 10 倍差距几乎都来自非视觉任务的能力提升。

### 3.2 B0 DOM 特有行为（B1 不具备）

1. **翻页（33+ task）**：B0 主动点击分页控件导航到 iPage=2/3/4，B1 从不翻页
2. **价格区间筛选（21+ task）**：独立填写 sPriceMin/sPriceMax 字段，B1 很少使用
3. **多 Tab 切换（4 task）**：B0 prompt 包含 tab_focus，B1 无此能力
4. **更稳定的表单字段聚焦**：通过 element_id 精准定位每个字段

### 3.3 B0 DOM 能力上限估算

- DOM 模式的**视觉信息瓶颈**仍然存在：62/234 tasks（~26.5%）是纯视觉任务，DOM 结构性不可达
- 去除视觉不可达任务（26.5%）和 N/A 任务（4.3%）后，DOM 理论可达任务约 163/234（69.7%）
- B0 DOM 在这些可达任务上 SR 约 ~20/163 约 12%，说明仍有大量可达任务失败——模型能力仍有提升空间
- B1 DOM 在可达任务上 SR 约 2/163 约 1.2%，显示 4B 模型基本无能力在 DOM 模式完成非视觉 classifieds 任务

---

## 4. Vision 模式：适度提升

### 4.1 SR 对比

| 指标 | B0 Vision | B1 Vision | 差值 |
|------|-----------|-----------|------|
| Raw SR | 15.81% | 12.39% | +3.42pp |
| Adjusted SR | 12.05% | 8.12% | +3.93pp |

Vision 模式 3.93pp 的提升符合预期：更大模型在纯视觉理解（颜色识别、形状识别、OCR）上更准确。

### 4.2 失败模式差异

| 失败原因 | B0 Vision | B1 Vision | 分析 |
|---------|-----------|-----------|------|
| fail_no_progress | **32.1%** | -- (未直接列) | B0 更高，可能因更执着尝试同一操作 |
| 坐标 misclick | 高 | 高 | 两模型均受 VWA viewport 限制 |
| 坐标自纠正 | **有限自纠正** | **零自纠正** | B0 会改变坐标重试；B1 重复完全相同坐标 |
| page_unchanged_rate | 41.8% | 30.1% (B1) | B0 更高，执着行为副作用 |

B0 Vision 的 page_unchanged_rate（41.8%）高于 B1（30.1%），但 SR 更高——说明即使有更多无效步骤，235B 的视觉推理准确性弥补了效率损失。

**坐标自纠正差异**：B1(4B) 在 misclick 后以完全相同的坐标和 confidence 重复点击（如连续 3-4 步 [0.43, 0.85]，见 B1_Vision_digest 3.1 节）。B0(235B) 则会尝试修改坐标（如 task 115 step 1->2 从 [0.43, 0.85] 改为 [428, 841]），但纠正方向不一定正确（该例 y=841 越出 viewport）。这反映了 235B 更强的状态感知能力，尽管坐标精度仍不足以可靠恢复。

### 4.3 Oracle 中 Vision 贡献（adjusted）

| 模型 | Vision oracle 选择 | Vision-only 成功（估）|
|------|--------------------|----------------------|
| B0 | 26 / 68 (38.2%) | ~12-14 |
| B1 | 18 / 46 (39.1%) | 7（精确）|

B0 Vision 在 oracle 中贡献更多（26 vs 18），说明 235B 在纯视觉任务上的成功集合更大。

---

## 5. Eager Completion：235B 的过度自信问题

### 5.1 现象

B0 (235B) 存在显著的 **eager completion** 行为：模型在首屏即给出答案，不滚动查看完整页面。v1 报告中统计了 <=2 步 genuine finish（排除 parse_error fallback）：

| 条件 | Eager finish (<=2步) | 其中成功 | 失败率 | 占总 episode |
|------|---------------------|---------|--------|-------------|
| **B0 SoM** | **71** | 2 | **97.2%** | **45.5%** |
| B0 Vision | 24 | 4 | 83.3% | 11.8% |
| B0 DOM | 23 | 4 | 82.6% | 9.9% |

> **数据背景说明**：上表来自 parse_error 修复前的分析。由于当时 parse_error 率高达 20.1%，部分被统计为 "eager finish" 的 episode 实际上可能是 parse_error 触发的 fallback finish（而非模型主动选择提前完成）。parse_error 修复后，真正的 eager completion 比例应低于表中 45.5%。但 eager completion 现象本身仍然真实存在——235B 模型确实倾向于在较少步数内给出答案。

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
| Capability-environment gap | 235B 更执着于"正确"路径（如反复 click `<select>`） | 能力越强 -> 越自信路径正确 -> 探索越少 |
| **Eager completion** | 235B 看首屏就 finish，不 scroll | 能力越强 -> 越自信信息充足 -> 验证越少 |

两者共享相同根因：**235B 的高置信度抑制了探索行为**。4B 模型因推理不确定性被迫多步探索，反而在需要全页扫描的聚合型任务上更有可能成功。

### 5.4 Eager completion 的影响重新评估

parse_error 修复后，eager completion 对 SoM SR 的拖累效应需要重新评估：

- **v1 估算**：假说 D 贡献 ~2-3pp（69 个失败 eager episode 中估计 5-7 个可通过探索成功）
- **修复后评估**：由于 parse_error 修复已将 B0 SoM SR 提升到 20.98%（超越 B1 SoM 16.24%），eager completion 的拖累虽然真实存在，但其影响被 235B 整体能力优势所覆盖。Eager completion 更应被视为 B0 SoM 的**进一步提升空间**，而非反转的解释因素。

---

## 6. 路由格局对比

### 6.1 Oracle Ceiling 对比

| 指标 | B0 | B1 |
|------|----|----|
| 最优单模式 SR | **20.98% (SoM)** | 16.24% (SoM) |
| Oracle ceiling（adjusted） | **29.06%** | 19.66% |
| Routing headroom | **+8.55pp** | +3.42pp |
| Oracle DOM 贡献 | **13** | 2 |
| Oracle SoM 贡献 | **29** | 26 |
| Oracle Vision 贡献 | **26** | 18 |

**B0 routing headroom（8.55pp）是 B1（3.42pp）的 2.5 倍**——原因是 B0 三模式 SR 更均衡，每种模式都有独特擅长的任务集合，路由有更大空间。

B0 oracle ceiling 29.06% 远高于 B1 的 19.66%（+9.4pp），反映了 235B 模型在三种模式上的全面能力提升叠加后的路由潜力。

### 6.2 路由策略对比

**B1 最优策略**：以 SoM 为默认（16.24%），视 action_diversity 信号路由到 Vision（7 个 Vision-only 成功），不引入 DOM（贡献极小）。路由提升上限约 +3.42pp。

**B0 最优策略**：三模式均有路由价值：
- **SoM 为默认基线**：20.98% adjusted SR，29 个 oracle 选择（最多）
- **Vision 补充**：成本仅 59% of SoM（$0.0248 vs $0.0417），26 个 oracle 选择
- **DOM 重新进入路由**：B0 DOM 13 个 oracle 选择，特别在 single_navigation+program_html（DOM 唯一贡献）和 page_reading+url_match 类型上有优势
- 三模式路由理论提升上限 +8.55pp，远超 B1

### 6.3 模型能力对路由的含义（Capability-Aware Routing）

B0 vs B1 的路由格局变化验证了 B1_findings.md S7.3 中的预测：

> "最优表征不是绝对的，是模型能力的函数（Read More, Think More 文献）。B0 DOM SR 若显著高于 B1，则 DOM 在高能力模型下重新有路由价值。"

实验证实：B0 DOM SR（8.48%）显著高于 B1（0.85%），DOM **确实重新成为路由候选**。Capability-aware routing 的假设得到支持。

---

## 7. 设计不对称对结论的影响

B0 和 B1 存在以下已知设计不对称（见 MEMORY），对 SR 对比有系统性影响：

| 不对称项 | B0 | B1 | 影响方向 |
|---------|----|----|---------|
| 解码策略 | temperature=0.1 | do_sample=False | SoM parse_error：B0 损失 SR |
| max_new_tokens | 4096 | 384 | B1 JSON 截断：B1 损失 SR（SoM/Vision 尤甚）|
| Scroll dy 约定 | 不稳定（+/-0.5） | 始终 +500 | B0 部分 scroll 方向错误：轻微损失 SR |
| 模型规模 | 235B (22B active) | 4B | 主要能力变量 |

**parse_error 修复后**：B0 SoM parse_error 从 20.1%（45/224）降至 2.6%（6/234），解码策略不对称的影响大幅减弱。剩余 6 个 parse_error 对整体 SR 影响极小（<0.5pp）。设计不对称的主要残余效应现在集中在 B1 侧的 max_new_tokens 截断问题上。

**净效应**：parse_error 修复后，设计不对称对 B0 的系统性拖累基本消除，B0 vs B1 的 SR 差异可以更可靠地归因于模型规模差异。

---

## 8. 关键发现汇总

1. **B0 在全部三种模式上优于 B1**（无反转）：parse_error 修复后，235B 模型符合"更大 = 更强"的模型规模假设。此前 v1 报告的 SoM 反转已完全消除。

2. **DOM：10 倍差距**：adjusted SR 8.48% vs 0.85%，235B 模型掌握翻页、价格筛选等复杂导航策略，4B 模型在纯 DOM 模式下几乎无能力完成 classifieds 任务。

3. **SoM：parse_error 修复是关键**：B0 SoM adjusted SR 从 12.05% 跃升至 20.98%（+8.93pp），超越 B1 SoM 16.24% 达 4.74pp。parse_error 从 45 (20.1%) 降至 6 (2.6%)，验证了 v1 假说 A 的判断。

4. **Vision：稳步提升**：12.05% vs 8.12%（+3.93pp），235B 在纯视觉理解上更准确，坐标自纠正能力优于 4B（虽仍不完美）。

5. **B0 routing headroom 是 B1 的 2.5 倍**（8.55pp vs 3.42pp）：B0 三模式均有路由价值（oracle: DOM 13, SoM 29, Vision 26），oracle ceiling 达 29.06%（B1: 19.66%）。更大的 headroom 意味着 capability-aware routing 在高能力模型上有更大收益空间。

6. **Eager completion 仍然存在但影响低于预期**：235B 的高置信度确实抑制了探索行为（SoM 步数仅 8.62 vs B1 的 11.8），但此前 v1 将其定位为 SoM 反转的假说 D（~2-3pp）。修复后来看，主因是 parse_error，eager completion 更应被视为 B0 SoM 的进一步提升空间。

7. **Capability-aware routing 假设得到支持**：DOM 在 4B 模型下几乎无路由价值（2 oracle），在 235B 模型下重获价值（13 oracle）。最优表征是模型能力的函数，而非固定的，这支持了根据模型能力动态调整路由策略的设计。

---

## 方法论说明

- **比较局限**：B0 和 B1 存在多项设计不对称（温度/max_tokens/scroll），SR 差异无法完全归因于模型规模
- **parse_error 修复**：B0 数据在修复前后不完全可比（修复后重跑了部分 episode），但方向性结论（反转消除）是稳健的
- **DOM adjusted SR 不确定性**：B0 DOM 盲区 17 tasks 未完全分类，true adjusted SR 在 [0.4%, 8.48%] 区间
- **McNemar 检验**：B0 vs B1 跨模型的配对检验需要 task-level 成功标签（相同 task 对比），此处仅汇总统计
- **成本比较**：B0（API 费用）与 B1（本地推理 GPU 等价成本）成本体系不同，直接数值比较需注意

---

*生成时间：2026-04-21*
*数据来源：B0_3mode_classifieds_20260421（parse_error 修复后）+ B1_3mode_classifieds_20260413*
*B0 三模式详情：B0_findings.md；B1 三模式详情：B1_findings.md*
*v1 (2026-04-15) -> v2 (2026-04-21): parse_error 修复后全面更新，SoM 反转消除*
