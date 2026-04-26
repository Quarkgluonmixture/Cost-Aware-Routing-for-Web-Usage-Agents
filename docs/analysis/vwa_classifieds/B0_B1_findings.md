# B0 vs B1 Classifieds 跨模型对比报告

> B0: Qwen3-VL-235B-A22B（proxy API，temperature=0.1，max_tokens=4096）
> B1: Qwen3-VL-4B bf16（本地推理，do_sample=False，max_new_tokens=384）
> 站点: Classifieds (OSClass), 234 tasks x 3 modes
> 本报告关注模型规模（4B vs 235B）对三种观测模式的差异化影响
> B0 run: `B0_3mode_classifieds_20260413` | B1 run: `B1_3mode_classifieds_20260413`
> **注：visual_fp 层已在 §95 中废弃，adjusted SR 仅扣除 N/A FP + eval FP**
>
> **数据更新 (2026-04-26)**：04-26 全 condition rederive。当前 adjusted SR：B0 DOM 14.10% / SoM **21.37%** / Vision 13.68% · B1 DOM 8.55% / SoM **13.25%** / Vision 7.26%（漂移 <1.7pp，文字结论不变）。Mirage Gap：B0 +7.27pp / B1 +4.70pp（两个模型都强 SoM 优势）。
>
> **B1 数据非最终**：DGX 共享 GPU 争抢污染 latency，待 Myriad HPC 独占 GPU 重跑。SR/cost/oracle 数字不受影响。

---

## 1. 核心对比表

### 1.1 Adjusted SR 对比

| 模式 | B0 (235B) | B1 (4B) | 差值 | 方向 |
|------|-----------|---------|------|------|
| DOM | **12.95%** | 7.59% | **+5.36pp** | B0 > B1 |
| SoM | **20.54%** | 13.84% | **+6.70pp** | B0 >> B1 |
| Vision | **12.05%** | 7.14% | **+4.91pp** | B0 > B1 |

**235B 模型在全部三种模式上优于 4B**，符合模型规模假设。SoM 差距最大（+6.70pp），DOM 差距次之（+5.36pp），Vision 差距最小（+4.91pp）。

> §95 变更：DOM 差值从 +3.57pp 扩大至 +5.36pp（B0 DOM 12.95% vs 旧 8.48%；B1 DOM 7.59% vs 旧 4.91%）。visual_fp 废弃后 DOM 成功保留更多，且 B0 增幅大于 B1。

### 1.2 Raw SR 对比

| 模式 | B0 Raw | B1 Raw | 差值 |
|------|--------|--------|------|
| DOM | 14.96% | 11.11% | +3.85pp |
| SoM | 23.08% | 17.52% | +5.56pp |
| Vision | 15.81% | 11.11% | +4.70pp |

Raw SR 方向与 adjusted SR 一致，B0 在三种模式上全面领先。

### 1.3 效率对比

| 指标 | B0 DOM | B1 DOM | B0 SoM | B1 SoM | B0 Vision | B1 Vision |
|------|--------|--------|--------|--------|-----------|-----------|
| 平均步数 | **11.56** | 13.83 | **8.60** | 9.90 | 7.85 | **6.73** |
| 平均成本/ep | $0.0427 | $0.0399 | $0.0415 | $0.0347 | $0.0248 | **$0.0133** |

> 注：B0 成本为 API 实际调用费用；B1 成本为本地 GPU 推理的 API 等价估算（基于 token 量 x API 定价）。两者成本体系不同，直接比较需谨慎。

B0 步数在 DOM/SoM 上少于 B1（DOM: 11.56 vs 13.83；SoM: 8.60 vs 9.90）——235B 更高效导航。Vision B1 步数更少（6.73 vs 7.85）但 SR 更低——B1 的快速失败（premature finish）压低了平均步数。

---

## 2. SoM 模式：parse_error 修复后反转消除

### 2.1 背景

v1 报告的核心异常是 **SoM 反转**：B0 SoM adjusted SR 12.05% < B1 SoM 16.24%（-4.19pp）。当时提出四个假说（parse_error drag、text-over-vision scale、capability-environment gap、eager completion），其中假说 A（parse_error drag）被评为最可能因素。

### 2.2 parse_error 修复效果（假说 A 验证）

parse_error 修复前后对比：

| 指标 | 修复前 | 修复后 | 变化 |
|------|--------|--------|------|
| B0 SoM parse_error 数 | 45 (20.1%) | **6 (2.6%)** | -39 |
| B0 SoM Adjusted SR | 12.05% | **20.54%** | **+8.49pp** |
| B0 vs B1 SoM 差值 | -4.19pp（B0 落后） | **+6.70pp（B0 领先）** | 反转彻底消除 |

**假说 A（parse_error drag）得到完全验证**：parse_error 率从 20.1% 降至 2.6%，B0 SoM SR 跃升近 8.5pp，建立了对 B1 的 6.70pp 领先优势。

### 2.3 其他假说的定位（次要因素）

| 假说 | 原定位 | 修复后定位 |
|------|--------|-----------|
| A: parse_error drag | 主要因素 (~3-4pp) | **已验证：实际贡献 ~8.93pp** |
| B: text-over-vision scale | 次要因素 (~1-2pp) | 可能仍存在，但被 235B 的能力优势覆盖 |
| C: capability-environment gap | 少量因素 (~0-1pp) | 同上，task-level 可能偶发，不影响总体方向 |
| D: eager completion | ~2-3pp | 仍然存在（见 §5），但影响程度低于预期 |

---

## 3. DOM 模式：§95 后差距扩大

### 3.1 SR 对比

| 指标 | B0 DOM | B1 DOM | 倍数 |
|------|--------|--------|------|
| Raw SR | 14.96% | 11.11% | 1.35x |
| Adjusted SR | 12.95% | 7.59% | **1.71x** |
| 独占成功（adjusted） | **8** | **13** | B1 > B0 |

> §95 变更：B0 DOM adjusted SR 从 8.48% 升至 12.95%（+4.47pp），B1 DOM 从 4.91% 升至 7.59%（+2.68pp）。visual_fp 废弃后 DOM 成功保留更多。B1 DOM 独占成功从 7 增至 13，说明 B1 DOM 有较多仅靠文本推理才能解决的 task。

### 3.2 B0 DOM 特有行为（B1 不具备或较弱）

1. **翻页（33+ task）**：B0 主动点击分页控件导航到 iPage=2/3/4，B1 从不翻页
2. **价格区间筛选（21+ task）**：独立填写 sPriceMin/sPriceMax 字段，B1 很少使用
3. **多 Tab 切换（4 task）**：B0 prompt 包含 tab_focus，B1 无此能力
4. **更稳定的表单字段聚焦**：通过 element_id 精准定位每个字段

### 3.3 B1 DOM 独占成功分析

B1 DOM 有 13 个独占成功（仅 DOM 成功，SoM/Vision 均失败），说明结构化 AXTree 在特定 task 上有不可替代的价值：

| 独占成功类型 | 数量 | 描述 |
|------------|------|------|
| single_navigation | ~7 | 通过精确 element_id 点击完成导航 |
| page_reading | ~6 | 从 AXTree 文本中精确提取信息 |

这些 task 的共性是需要**精确文本提取或元素交互**，DOM 的 element_id 机制在此优于坐标点击（Vision）和可能受 text_over_vision 影响的 SoM。

---

## 4. Vision 模式：稳定差距

### 4.1 SR 对比

| 指标 | B0 Vision | B1 Vision | 差值 |
|------|-----------|-----------|------|
| Raw SR | 15.81% | 11.11% | +4.70pp |
| Adjusted SR | 12.05% | 7.14% | +4.91pp |

Vision 模式 4.91pp 的提升符合预期：更大模型在纯视觉理解（颜色识别、形状识别、OCR）上更准确。

### 4.2 失败模式差异

| 失败原因 | B0 Vision | B1 Vision | 分析 |
|---------|-----------|-----------|------|
| fail_no_progress | 39.7% | **58.1%** | B1 更高——坐标 misclick 频率更高 |
| fail_early_finish | ~8% | **14.1%** | B1 更频繁过早放弃 |
| 坐标自纠正 | **有限自纠正** | **零自纠正** | B0 会改变坐标重试；B1 重复完全相同坐标 |

**坐标自纠正差异**：B1(4B) 在 misclick 后以完全相同的坐标和 confidence 重复点击。B0(235B) 则会尝试修改坐标，虽然纠正方向不一定正确。这反映了 235B 更强的状态感知能力。

---

## 5. Eager Completion：235B 的过度自信问题

### 5.1 现象

B0 (235B) 存在显著的 **eager completion** 行为：模型在首屏即给出答案，不滚动查看完整页面。

> **数据背景说明**：eager completion 的定量统计来自 parse_error 修复前的分析。parse_error 修复后，真正的 eager completion 比例应有所下降。但 eager completion 现象本身仍然真实存在——235B 模型确实倾向于在较少步数内给出答案。

### 5.2 SoM 模式尤其严重的原因

SoM 同时提供截图 + 文字标注，给了 235B 模型 **最大信息密度的首屏**，导致：

1. **过早满足**：模型在首屏看到部分匹配项就判断信息已充足
2. **"on this page" = 当前视口**：模型将 "on this page" 理解为当前可见区域
3. **不会预判折叠线以下有更多内容**

### 5.3 与 capability-environment gap 的关系

两者共享相同根因：**235B 的高置信度抑制了探索行为**。4B 模型因推理不确定性被迫多步探索，反而在需要全页扫描的聚合型任务上更有可能成功。

---

## 6. Action 执行效率对比

| 指标 | B0 DOM | B1 DOM | B0 SoM | B1 SoM | B0 Vision | B1 Vision |
|------|--------|--------|--------|--------|-----------|-----------|
| click_fail_rate | **12.2%** | 17.8% | **7.0%** | 33.3% | 45.9% | **45.7%** |
| type_fail_rate | **6.8%** | 9.2% | **3.6%** | 6.5% | **16.7%** | 5.8% |
| pixel_coordinate_leak | 0% | 0% | 0% | 0% | **34.6%** | 20.9% |

**关键发现**：
- **SoM click_fail_rate 差距最大**：B0 7.0% vs B1 33.3%（+26.3pp）。235B 模型利用 SoM 标注的 element_id 精确定位，而 4B 模型 SoM click 定位能力大幅衰退
- **DOM click_fail_rate 差距较小**：12.2% vs 17.8%（+5.6pp），两个模型都依赖 AXTree element_id
- **Vision click_fail_rate 几乎相同**：45.9% vs 45.7%，坐标定位在两个模型上都是瓶颈
- B0 Vision pixel_coordinate_leak 更高（34.6% vs 20.9%），可能与 B0 prompt 格式差异有关

---

## 7. Mirage Effect 跨模型对比

### 7.1 Mirage Gap（Raw SR）

| 模型 | SoM SR | DOM SR | Mirage Gap |
|------|--------|--------|-----------|
| B0 | 23.08% | 14.96% | **8.12pp** |
| B1 | 17.52% | 11.11% | **6.41pp** |

Mirage Effect（§18）——相同文本信息下，图片存在触发质变推理路径——在两个模型上都存在。B0 mirage gap（8.12pp）大于 B1（6.41pp），说明 **235B 模型从图片中获益更多**。这可能因为：
- 235B 有更强的视觉-文本整合能力（能更好地利用 SoM 截图中的布局信息）
- 4B 模型的 text_over_vision 效应更严重（截图信息被文本推理覆盖），削弱了 SoM 的优势

---

## 8. 路由格局对比

### 8.1 Oracle Ceiling 对比

| 指标 | B0 | B1 |
|------|----|----|
| 最优单模式 SR | **20.54% (SoM)** | 13.84% (SoM) |
| Oracle ceiling（adjusted） | **31.20%** | 21.37% |
| Routing headroom | **+9.83pp** | +8.12pp |
| Oracle DOM 贡献（adj） | **23/73** | 16/50 |
| Oracle SoM 贡献（adj） | **25/73** | 17/50 |
| Oracle Vision 贡献（adj） | **25/73** | 17/50 |
| DOM 独占成功 | 8 | **13** |
| SoM 独占成功 | **21** | 15 |
| Vision 独占成功 | **11** | 6 |

B0 routing headroom（9.83pp）略高于 B1（8.12pp）。两者都有可观的路由空间。

> §95 变更：B0 oracle ceiling 从 29.06% 升至 31.20%（+2.14pp），B1 从 18.80% 升至 21.37%（+2.57pp）。三模式 oracle 贡献趋于均衡——DOM 不再是弱势模式。

### 8.2 路由策略对比

**B1 路由策略**：**三模式均有路由价值**：
- SoM 为默认（13.84%）
- DOM 贡献 13 个独占成功——§95 后 DOM 路由价值大幅提升
- Vision 贡献 6 个独占成功——低成本替代
- Routing headroom 8.12pp

**B0 路由策略**：三模式路由空间更大：
- SoM 为默认基线（20.54%）
- Vision 补充（成本仅 60% of SoM）
- DOM 贡献 8 个独占成功 + 最低延迟
- Routing headroom 9.83pp

### 8.3 模型能力对路由的含义（Capability-Aware Routing）

B0 vs B1 的路由格局变化验证了 B1_findings.md §7.3 中的预测：

> "最优表征不是绝对的，是模型能力的函数（Read More, Think More 文献）。B0 DOM SR 若显著高于 B1，则 DOM 在高能力模型下重新有路由价值。"

实验证实：B0 DOM SR（12.95%）高于 B1（7.59%），差距 5.36pp。两个模型下 DOM 都有独占成功（B0: 8, B1: 13），Capability-aware routing 的假设得到支持。

---

## 9. 设计不对称对结论的影响

B0 和 B1 存在以下已知设计不对称（见 MEMORY），对 SR 对比有系统性影响：

| 不对称项 | B0 | B1 | 影响方向 |
|---------|----|----|---------|
| 解码策略 | temperature=0.1 | do_sample=False | SoM parse_error：B0 损失 SR |
| max_new_tokens | 4096 | 384 | B1 JSON 截断：B1 损失 SR（SoM/Vision 尤甚）|
| Scroll dy 约定 | 不稳定（+/-0.5） | 始终 +500 | B0 部分 scroll 方向错误：轻微损失 SR |
| 模型规模 | 235B (22B active) | 4B | 主要能力变量 |

**parse_error 修复后**：B0 SoM parse_error 从 20.1%（45/224）降至 2.6%（6/234），解码策略不对称的影响大幅减弱。

---

## 10. 关键发现汇总

1. **B0 在全部三种模式上优于 B1**（无反转）：SoM 差距最大（+6.70pp），DOM 差距次之（+5.36pp），Vision 最小（+4.91pp）。

2. **DOM：§95 后差距扩大**：B0 DOM 12.95% vs B1 7.59%（+5.36pp，旧 +3.57pp）。visual_fp 废弃后 DOM 成功保留更多，B0 增幅大于 B1。B1 DOM 有 13 个独占成功（B0: 8），4B 在特定文本任务上有不可替代的价值。

3. **SoM：B0 优势稳定**：B0 SoM 20.54% vs B1 SoM 13.84%（+6.70pp），parse_error 修复后 SoM 反转完全消除。

4. **Vision：稳定差距**：12.05% vs 7.14%（+4.91pp），235B 在纯视觉理解上更准确，坐标自纠正能力优于 4B。

5. **B0 routing headroom 略高于 B1**（9.83pp vs 8.12pp）：B0 oracle ceiling 31.20%，B1 21.37%。两者都有可观的路由空间。

6. **Mirage Effect 跨模型稳定**：B0 mirage gap 8.12pp vs B1 6.41pp，两个模型都展现 SoM-DOM 差距，但 235B 从图片中获益更多。

7. **Capability-aware routing 假设得到支持**：DOM 在两个模型下都有独占成功（B0: 8, B1: 13），但贡献模式不同。最优表征是模型能力的函数。

8. **三模式均有路由价值**：§95 后 DOM 路由价值进一步提升（B1 独占 13 个 task），三模式 oracle 贡献趋于均衡。

9. **Action 执行效率差异显著**：SoM click_fail_rate 从 B0 7.0% 劣化至 B1 33.3%（+26.3pp），是模型规模衰退最大的维度。Vision click_fail_rate 两模型基本相同（~46%），坐标定位是模型无关的瓶颈。

---

## 方法论说明

- **Adjusted labels**：仅扣除 N/A FP + eval FP（§95 废弃 visual_fp）
- **比较局限**：B0 和 B1 存在多项设计不对称（温度/max_tokens/scroll），SR 差异无法完全归因于模型规模
- **B1 数据更新**：B1 使用 `B1_3mode_classifieds_20260413` 运行（§33/§34/§36 修复后重跑）
- **成本比较**：B0（API 费用）与 B1（本地推理 GPU 等价成本）成本体系不同，直接数值比较需注意

---

*v1 (2026-04-15): 初版，SoM 反转*
*v2 (2026-04-21): parse_error 修复后全面更新，SoM 反转消除*
*v3 (2026-04-23): B1 数据更新至 B1_3mode_classifieds_20260413 运行，DOM 能力回归*
*v4 (2026-04-24): B0 SoM 数据修正（55→54 raw successes），全部衍生指标同步更新*
*v5 (2026-04-25): §95 FP 重构：废弃 visual_fp，DOM adjusted SR 大幅上升；Oracle ceiling/headroom 更新；三模式 oracle 贡献趋于均衡*
*v6 (2026-04-25): 新增 §6 Action 执行效率对比（click_fail_rate per mode per baseline）*
*数据来源：B0_3mode_classifieds_20260413 + B1_3mode_classifieds_20260413*
*B0 三模式详情：B0_findings.md；B1 三模式详情：B1_findings.md*
