# B0 Vision Baseline 分析报告（Classifieds）

> B0 = Qwen3-VL-235B-A22B（proxy API），Vision 模式，classifieds 站点
> 对应 B1 分析见 `B1_Vision_digest.md`；B0 vs B1 跨模型对比见 `B0_B1_findings.md`
> **注：visual_fp 层已在 §95 中废弃，adjusted SR 仅扣除 N/A FP + eval FP**
>
> 数据来源：`phase1_vision_router_0`，classifieds 全部 234 tasks
> 分析方法：自动化 post analysis（condition_summary_v2 + reason_diagnostics + cross_representation）
> GLM Digest 覆盖：126/197 failed episodes 有完整 GLM 根因分析（2026-04-23 补跑）
> **注：visual_fp 层已在 §95 中废弃，adjusted SR 仅扣除 N/A FP + eval FP**

---

## 一、总体概况

| 指标 | 数值 |
|------|------|
| Vision condition 总 episode 数 | 234 |
| 成功（raw） | 37（15.81%） |
| N/A FP | 10（10 个 N/A reference task 中 10 个误判） |
| Eval FP | 0 |
| **Adjusted SR** | **12.05%**（27/224） |
| 平均步数 | 7.85 步 |
| 平均成本 | $0.0248 / episode |
| P95 延迟 | 44,984ms |

### 与 DOM / SoM 对比（B0 三模式）

| 指标 | DOM | SoM | **Vision** |
|------|-----|-----|-----------|
| Raw SR | 14.96% | **23.08%** | 15.81% |
| Adjusted SR | 12.95% | **20.54%** | 12.05% |
| 平均步数 | 11.56 | 8.60 | **7.85** |
| 平均成本 | $0.0427 | $0.0415 | **$0.0248** |
| P95 延迟 | **37,513ms** | 74,004ms | 44,984ms |
| 主导失败原因 | no_progress(26.5%) | wrong_url(24.4%) | **no_progress(39.7%)** |

B0 Vision 成本最低（$0.0248/ep），adjusted SR 与 DOM 接近（12.05% vs 12.95%）。

### 统计显著性

| 对比 | p 值 | 显著性 |
|------|------|--------|
| Vision vs DOM | 0.627 | — (n.s.) |
| SoM vs Vision | 0.085 | — (n.s.) |

> §95 变更：Vision vs DOM 从此前的 p=0.016（显著）变为 p=0.627（不显著）——因为 DOM adjusted SR 上升（visual_fp 废弃），DOM 与 Vision 差距几乎消失。

### 与 B1 Vision 对比

| 模型 | Raw SR | Adjusted SR | avg Steps | avg Cost |
|------|--------|-------------|-----------|----------|
| **B0 (235B)** | 15.81% | **12.05%** | 7.85 | $0.0248 |
| B1 (4B) | 11.11% | 7.14% | 6.73 | $0.0133 |

**B0 Vision 优于 B1 Vision（+4.91pp adjusted）**。

---

## 二、失败原因分布

| 失败原因 | 数量 | 比例 | 备注 |
|---------|------|------|------|
| **fail_no_progress** | **93** | **39.7%** | 最大失败源，Vision 特有高发 |
| success | 37 | 15.8% | (raw) |
| fail_incomplete_or_stuck | 21 | 9.0% | 页面卡住/信息不完整 |
| fail_finish_wrong_url_not_found | 21 | 9.0% | 完成时 URL 不匹配 |
| fail_early_finish | 19 | 8.1% | 过早结束 |
| fail_finish_eval_mismatch | 18 | 7.7% | 评测不一致 |
| fail_finish_empty_answer | 9 | 3.8% | 空答案 |
| fail_finish_claim_missing | 7 | 3.0% | finish 时声明缺失 |
| fail_max_steps_target_unreachable | 5 | 2.1% | 目标不可达 |
| fail_max_steps | 3 | 1.3% | 达到最大步数 |
| fail_max_steps_search_repeat | 1 | 0.4% | 搜索循环 |

---

## 三、核心异常：fail_no_progress 率 39.7%

B0 Vision 的 fail_no_progress 率（39.7%，93/234）是三模式中最高的：DOM 26.5%、SoM 8.1%。这是 Vision 模式特有的高发失败。

### 3.1 机制分析

Vision 模式高发的多重原因：

- **坐标 misclick 积累**：纯靠归一化坐标 `[x,y]` 点击，连续 misclick 累积 no_progress 计数
- **action_failed 极高**：Vision 模式 action_failed 次数三模式最高
- **scroll 到底后持续 scroll**：没有 AXTree 文本指引
- **page_unchanged_rate 最高**：37.9%

### 3.2 GLM Digest 根因分类（N=126 full GLM）

| 排名 | GLM 根因类别 | 数量 | 占比 |
|------|------------|------|------|
| 1 | **目标不可达** | 48 | 38.1% |
| 2 | **执行停滞** | 32 | **25.4%** |
| 3 | 过早结束 | 18 | 14.3% |
| 4 | 搜索循环 | 11 | 8.7% |
| 5 | 答案对齐错误 | 10 | 7.9% |

**执行停滞（25.4%）是 Vision 的特征性高频类别**——SoM 仅 9.5%、DOM 更低。Vision 模式因坐标点击失败导致 agent 反复尝试无效操作。

**坐标偏移是 Vision 失败的核心技术瓶颈**（42.9%），与 no-op 率和 action_failed 互相印证。

---

## 四、成本效率分析

### 4.1 Vision 成本优势

| 指标 | B0 Vision | B0 SoM | B0 DOM |
|------|-----------|--------|--------|
| 平均成本/ep | **$0.0248** | $0.0415 | $0.0427 |
| 每成功 ep 成本（adjusted） | $0.0248/0.1205 = **$0.206** | $0.0415/0.2054 = $0.202 | $0.0427/0.1295 = $0.330 |

**Vision 每 episode 成本最低**（$0.0248）。SoM 每成功 episode 成本略优（$0.202 vs $0.206），两者差距极小。

### 4.2 Wilcoxon 成本检验

| 对比 | p 值 | 方向 |
|------|------|------|
| Vision vs SoM total_cost | **2.9e-9** | Vision 更便宜 ★★★ |
| Vision vs DOM total_cost | **4.5e-12** | Vision 更便宜 ★★★ |

---

## 五、TinyMCE iframe 交互限制（三模式共性）

Classifieds 编辑页面的"Description"字段使用 TinyMCE 富文本编辑器，渲染在 `<iframe>` 内。三种观测模式均无法正常编辑此字段。对 Vision SR 影响 < 1pp，不构成对 Vision 的不公平。

---

## 六、共性脚手架缺陷（与 B1 Vision 相同）

- **无结构化导航信息**：搜索框、分类链接、分页控件只能靠视觉识别
- **N/A 任务 False Positive（10/10）**：全部 10 个 N/A reference task 均误判为 success
- **极少翻页**：Vision 模式分页控件需视觉识别，即使 235B 也很少翻页
- **confirm 弹窗不可交互**：Delete 操作在 Vision 模式下同样受 VWA 框架限制

---

## 七、Vision 在路由中的角色

三模式 cross_representation 分析：

| 指标 | 数值 |
|------|------|
| Oracle 选择（raw） | **27/75**（36.0%） |
| Oracle 选择（adjusted） | **25/73**（34.2%） |
| Vision 独占成功（adjusted） | **11 tasks** |
| only_vision 占比 | 4.7% |

**Vision 路由价值**：
1. **独占贡献**：11 个 task 仅 Vision 成功，不可被其他模式替代——路由必须覆盖 Vision 通道
2. **Oracle 选择占比均衡**：adjusted 25/73，与 SoM（25）和 DOM（23）几乎持平
3. **低成本优先通道**：$0.0248/ep，是路由策略中自然的"低成本优先尝试"选项
4. **与 DOM 持平**：McNemar p=0.627（不显著），Vision 在 adjusted SR 上与 DOM 几乎相同（12.05% vs 12.95%）

---

*更新时间：2026-04-25（§95 FP 重构：废弃 visual_fp；Vision vs DOM 从显著变为不显著；更新全部定量数据）*
*数据来源：B0_3mode_classifieds_20260413 phase1_vision_router_0*
*B0 三模式定量对比见 `B0_findings.md`；B0 vs B1 跨模型对比见 `B0_B1_findings.md`*
