# B0 Classifieds — DOM 模式分析

> B0 = Qwen3-VL-235B-A22B（proxy API），DOM 模式，classifieds 站点
> 对应 B1 分析见 `B1_DOM_digest.md`；B0 的核心分析价值是 235B vs 4B 的能力差异
> **注：visual_fp 层已在 §95 中废弃，adjusted SR 仅扣除 N/A FP + eval FP**

---

## 一、总体概况

| 指标 | 数值 |
|------|------|
| DOM condition 总 episode 数 | 234 |
| 成功（raw） | 35（14.96%） |
| N/A FP | 6 |
| Eval FP | 0 |
| **Adjusted SR** | **12.95%**（29/224） |
| 平均步数 | 11.56 步 |
| 平均成本 | $0.0427 / episode |
| P95 延迟 | 37,513ms |

> §95 变更：visual_fp 层废弃后，DOM adjusted SR 从 8.48% 上升至 12.95%（+4.47pp）。此前被标为 visual FP 的 DOM 成功现在保留为有效成功——DOM 模式虽无截图，但部分视觉任务可通过文本推理间接完成。

### 与 SoM / Vision 对比（B0 三模式）

| 指标 | DOM | **SoM** | Vision |
|------|-----|---------|--------|
| Raw SR | 14.96% | **23.08%** | 15.81% |
| Adjusted SR | 12.95% | **20.54%** | 12.05% |
| 平均步数 | 11.56 | **8.60** | 7.85 |
| 平均成本 | $0.0427 | $0.0415 | **$0.0248** |
| P95 延迟 | **37,513ms** | 74,004ms | 44,984ms |

### 统计显著性（McNemar 精确检验）

| 对比 | p 值 | 显著性 |
|------|------|--------|
| SoM vs DOM | **0.0115** | ★ |
| Vision vs DOM | 0.627 | — (n.s.) |
| SoM vs Vision | 0.085 | — (n.s.) |

**SoM 显著优于 DOM**（p=0.012）。Vision 与 DOM 差异不显著。

> 与旧数据对比：SoM vs DOM p 值从 1.4e-5 上升至 0.012，仍显著但幅度缩小——因为 DOM adjusted SR 上升（visual_fp 移除），DOM 与 SoM 差距缩小。

---

## 二、失败原因分布

| 失败原因 | 数量 | 比例 | 备注 |
|---------|------|------|------|
| **fail_no_progress** | **62** | **26.5%** | 最大失败源 |
| fail_finish_wrong_url_not_found | 44 | 18.8% | URL 不匹配 |
| success | 35 | 15.0% | (raw) |
| fail_finish_eval_mismatch | 32 | 13.7% | 评测不一致 |
| fail_max_steps_target_unreachable | 14 | 6.0% | 目标不可达 |
| fail_max_steps_click_back_loop | 11 | 4.7% | click-back 循环 |
| fail_early_finish | 10 | 4.3% | 过早结束 |
| fail_incomplete_or_stuck | 7 | 3.0% | 页面卡住 |
| fail_finish_empty_answer | 7 | 3.0% | 空答案 |
| fail_finish_claim_missing | 6 | 2.6% | 声称缺失 |
| fail_max_steps_search_repeat | 3 | 1.3% | 搜索循环 |
| fail_max_steps | 2 | 0.9% | 达到最大步数 |
| fail_finish_wrong_url_left_target | 1 | 0.4% | 离开目标 URL |

DOM 模式的 `fail_no_progress`（26.5%）显著高于 SoM（8.1%），因为 DOM 纯文本无截图辅助，agent 在长 AXTree 中反复选择相同（错误的）元素但无法得到视觉反馈进行纠正。

---

## 三、B0 vs B1 能力差异：B0 特有正向行为

以下行为在 B1（4B）中从未或极少出现，B0（235B）稳定使用，是模型规模带来的实质性能力提升。

### 1. 翻页导航（paginate）

B0 DOM 中 **33+ 个 task** 出现了 `iPage=2/3/4` 的 URL 翻页，agent 主动点击"下一页"按钮导航到第 2、3、4 甚至更后面的页面。B1 DOM 在 classifieds 几乎从不翻页。**B0 的翻页能力使其能遍历更多候选结果，是 raw SR 高于 B1 的重要原因之一。**

### 2. 价格上下限筛选（sPriceMin / sPriceMax）

B0 DOM 中 **21+ 个 task** 正确使用了价格区间筛选，独立填写 sPriceMin 和 sPriceMax 两个字段并通过 Enter 提交。B1 DOM 极少使用价格筛选控件。

### 3. 多 Tab 切换（tab_focus）

B0 prompt 明确包含 `tab_focus` 指令，agent 能识别多 tab 任务并调用 `tab_focus {page_number}` 切换浏览器 tab。classifieds B0 DOM 中有 4 个 task 触发了 `tab_focus`（148/150/163/229）。

### 4. 表单字段精准聚焦

B0 通过 AXTree element ID 精准定位并逐一聚焦表单字段，每个字段独立 type + Enter，不混用字段。B1 有时在 type 时选错字段或丢失焦点。

---

## 四、B0 特有行为：scroll dy 约定随机混用

235B 模型对 scroll `delta=[dx, dy]` 的符号约定不稳定，在同一个 episode 内混用 dy>0（prompt 约定向下）和 dy<0（训练先验向下）两种约定。

**根因（§72）**：大模型训练数据中存在多种 scroll 约定，temperature=0.1 的轻度随机允许采样切换。跨 API 控制实验确认 scroll 约定不稳定是 Qwen3-VL-235B 的**模型固有行为**，非 proxy 或 prompt 导致。

**处置**：不修复。已实现的缓解方案（§67）：Tool schema 将 `delta: [dx, dy]` 替换为 `scroll_direction: enum("up","down")`。分析时将 scroll 方向错误视为模型随机噪声。

---

## 五、B0 特有行为：`<select>` 下拉菜单 capability-environment gap

B0（235B）识别出 classifieds 首页的 category `<select>` 元素是正确入口，反复 click 同一 eid，每个 task 只走 3 步即被 cycle detection 截断。B1（4B）因不确定性而随机游走，偶然找到侧边栏链接。**认知越准确，越执着于"正确"路径，cycle detection 触发越快，探索越少**。

---

## 六、FP 分类统计（B0 DOM，234 episodes）

**成功总数：35**（raw SR 14.96%，234 tasks）

| 类别 | 数量 | 机制 |
|------|------|------|
| na_fp | 6 | N/A reference task，agent 未真正完成 |
| eval_fp | 0 | — |
| **净 FP** | **6** | — |
| **真实成功** | **29** | 35 - 6 = 29 |

**Adjusted SR**：12.95%（29/224，扣除 10 个 N/A reference tasks 后分母 224）

> §95 变更：此前 DOM 有 14 个 visual_fp（其中 4 个与 na_fp 重叠），net unique FP=16，adjusted SR 仅 8.48%（19/224）。visual_fp 废弃后，这些 DOM 视觉任务成功保留为有效成功。

---

## 七、典型失败模式

### 7.1 跨步数值记忆失败

Task 31/32/33 等任务中，agent 在 thought 中正确读到价格但在 finish 时输入错误报价——模型在长轨迹中丢失关键数值。

### 7.2 元素中心越界导致点击/输入失败

当元素中心 y > 720px（viewport 高度）时，click 静默失败或 TYPE 失焦导致全选蓝。

### 7.3 容器节点误点导致导航失败

classifieds 搜索结果的文字区域对应大容器 div，AXTree 把整个容器暴露为单一节点。其中心落在 price/date 文字上（非 `<a>` 子节点），click 无导航效果。

### 7.4 视觉幻觉（DOM 模式下 235B 幻觉视觉内容）

Task 222 等案例中，agent 在 DOM 模式下幻觉了图像内容并据此作答。这是 B0 DOM 中最危险的失败模式之一。

---

## 八、定量结果

| 指标 | DOM | SoM | Vision |
|------|-----|-----|--------|
| Raw SR | 14.96% (35/234) | **23.08%** | 15.81% |
| Adjusted SR | 12.95% (29/224) | **20.54%** | 12.05% |
| FP 分解 | N/A FP: 6, eval FP: 0 | N/A FP: 8 | N/A FP: 10 |
| avg steps | 11.56 | 8.60 | **7.85** |
| cost/ep | $0.0427 | $0.0415 | **$0.0248** |
| P95 latency | **37,513ms** | 74,004ms | 44,984ms |

### McNemar 精确检验（adjusted labels）

| 对比 | 不一致对 (A-only / B-only) | p 值 | 显著性 |
|------|--------------------------|------|--------|
| SoM vs DOM | 29 / 12 | **0.0115** | ★ |
| Vision vs DOM | 17 / 21 | 0.627 | — (n.s.) |
| SoM vs Vision | 31 / 18 | 0.085 | — (n.s.) |

### Bootstrap 95% CI（cross_rep adjusted labels, /234 分母）

| 模式 | SR | CI 下界 | CI 上界 |
|------|-----|---------|---------|
| SoM | 21.37% | 16.24% | 26.92% |
| Vision | 15.81% | 11.11% | 20.51% |
| DOM | 14.10% | 9.83% | 18.80% |

> 与 B1 对比：DOM 12.95% vs B1 7.59%（+5.36pp）；SoM 20.54% vs B1 13.84%（+6.70pp）；Vision 12.05% vs B1 7.14%（+4.91pp）。B0 235B 在所有三种模式下均优于 B1 4B。

---

*最后更新：2026-04-25（§95 FP 重构：废弃 visual_fp 层，DOM adjusted SR 从 8.48% 上升至 12.95%；更新全部定量数据和统计检验）*
