# 相关文献对 P79 实验设计的启发

> 本文档综合分析 26 篇近期 web agent / GUI grounding / cost-aware routing 论文（2024-2026），
> 提取对 P79 (Cost-Aware Routing for Web Usage Agents) 实验设计直接相关的发现。
> 每条启发标注来源论文、具体证据和映射到 P79 的哪个实验阶段。

---

## 一、核心发现汇总

### A. 观测表征（Phase 1 直接相关）

#### 1. 小模型用紧凑表征（a11y/SoM）远优于 HTML

**来源**：Read More, Think More (Enomoto et al., 2026, arXiv:2604.01535)

**证据**：
- gpt-oss-20b + HTML: 27.6%，比 a11y **差 18.8pp**；gpt-oss-120b + HTML 也比 a11y 差 7.9pp
- 高能力模型（gpt-5.1）反而用 HTML 更好（+17.5pp），说明最优表征取决于模型能力
- Diff-based 历史可以在保留 full history 性能的同时将 token 缩减到约 **1/3**

**交叉验证**：
- WebWorld [9] 采用 A11y Tree 作为主表征，理由：universal applicability、high information density、LLM-friendly
- WEBSERV [15] 的 DOM parser 过滤不可见/无关节点后达到 46.7% (WebArena-Lite)

**对 P79 的映射**：
- **Phase 1** — 强预测：SoM 将显著优于 DOM（预计 10-20pp），这是 P79 最可靠的假设
- **M4 Memory** — diff-based 历史是小模型 token 节约的关键手段

---

#### 2. 视觉输入不可或缺，hybrid > text-only

**来源**：Chain-of-Ground (Li et al., 2025, arXiv:2512.01979) + Ego2Web (Yu et al., 2026, arXiv:2603.22529)

**证据**：
- CoG: Image-based feedback **65.8%** > Text-based 64.3% > No feedback 61.4%
- Ego2Web: No Visual **4.4%** → Caption Only 23.6% → Raw Video **48.2%**（消融差距巨大）
- DMAST [5]: 视觉模态比文本模态更脆弱（image attack 34.4% vs text attack 24.1%），说明视觉信号承载关键信息

**对 P79 的映射**：
- **Phase 1 (A2 Observation)** — 支持 hybrid（DOM+截图）优于 dom_only
- 截图不仅是"额外信息"，而是提供结构化空间推理的关键模态
- 注意：不要为节省 token 过度压缩截图质量

---

#### 3. 语义元素描述 >> 原始 ID

**来源**：HMT (Tan et al., 2026, arXiv:2603.07024)

**证据**：
- 使用 raw element identifiers 导致灾难性下降：39.7% → **12.4%** StepSR（Mind2Web Cross-Website）
- 有效的语义描述需包含：role、label、visible text、relative position、structural context
- 去掉 pre/post-conditions 仅降 2.5%，说明语义描述比状态跟踪更关键

**对 P79 的映射**：
- DOM/SoM 表征必须包含语义标签，不能只有 raw ID
- 这也解释了为什么 AXTree（包含 role/name）比纯 DOM 结构更适合小模型

---

### B. Cost-Aware 路由（Phase 2 直接相关）

#### 4. Cost-aware 路由可实现 70-88% 成本降低

**来源**：WebRouter (Li et al., 2025, arXiv:2510.11221) + AVR (Liu et al., 2026, arXiv:2603.12823)

**证据**：
- WebRouter: ca-VIB 目标，**87.8%** 成本降低（$0.98→$0.12），仅 3.8% 准确率下降
- AVR: difficulty + confidence + safety 三机制，**78%** 成本降低，准确率在 2pp 内
- 关键发现：prompt tokens 占总成本 **>70%**，减少 prompt 是最有效的成本优化

**对 P79 的映射**：
- **Phase 2 (B1 Router)** — 成本目标：40-70% token 降低，成功率在 5pp 内
- 路由规则应优先减少 prompt 长度（选择更紧凑的表征）
- 注意：WebRouter 是 query-level 路由（每任务一次决策），P79 需要 step-level 路由

---

#### 5. 三机制路由框架

**来源**：AVR (Liu et al., 2026) + CATTS (Lee et al., 2026, arXiv:2602.12276)

**证据**：
- AVR 的三机制：
  1. **Difficulty Classification**: 按表单密度/交互元素数/历史长度预分类 easy/medium/hard
  2. **Confidence-Based Routing**: logprob < 阈值 → 升级到更强策略
  3. **Safety-Integrated Routing**: 高风险操作（提交/删除）强制双阶段推理
- CATTS: margin-gated confidence-aware scaling 实现 **56%** token 降低（405K vs 920K）
- AVR memory injection 将 7B 模型 confidence 从 0.83 提升到 0.96，实现 86% 成本降低

**对 P79 的映射**：
- **Phase 2** — P79 路由器的直接设计模板
- Confidence 阈值建议：conf < 0.85 → 触发 M3；conf < 0.70 → 触发 M2；conf < 0.50 → 触发 M1+memory
- 注意：小模型 confidence 校准质量未知（文献空白 #6），阈值需在 VWA 上调优

---

#### 6. 异构模型/表征组合 > 单一最强配置

**来源**：Chain-of-Ground (Li et al., 2025, arXiv:2512.01979)

**证据**：
- Triple-step CoG 用三个不同模型（UI-TARS-1.5-7B → Qwen3-VL-235B → Qwen3-VL-32B）达到 68.4%，超过任何单一模型
- 原因：不同模型有不同的"视觉盲区"，异构组合实现互补
- 即使最弱的 UI-TARS-1.5-7B（单独 42.0%）作为 anchor，后续 refine 也能达到 SOTA

**对 P79 的映射**：
- **Phase 2** — 直接理论支撑：最优策略不是固定用最强配置，而是根据场景动态选择
- **Phase 3 (M4)** — 验证分层推理的价值

---

#### 7. MoGE ≈ 硬编码规则路由器

**来源**：Avenir-Web (Li et al., 2025, arXiv:2602.02468)

**证据**：
- MoGE 分层 fallback：标准交互→坐标点击；文本输入失败→虚拟键盘→结构定位→语义搜索；`<select>`→script 直接赋值；iframe→视觉坐标穿透
- 去掉 MoGE 成功率从 48% 降到 40%（-8%）

**对 P79 的映射**：
- **Phase 2 (B1 Router)** — MoGE 验证了规则路由器的可行性
- 区别：MoGE 路由 grounding 方式，P79 路由观测表征
- 可借鉴：按 UI 元素类型（按钮/输入框/下拉/iframe）触发不同处理策略

---

#### 8. Pareto 通用路由：两层决策

**来源**：MoMA (Guo et al., 2025, arXiv:2509.07571)

**证据**：
- 两层路由：Layer 1 选模型（按任务难度），Layer 2 选 agent/策略（按当前状态）
- Performance-priority 模式：+2.9% 成绩，-31.46% 成本
- Auto-routing 模式：-37.19% 成本，成绩仍超 deepseek-v3

**对 P79 的映射**：
- **Phase 2** — P79 可映射为：Layer 1 = 选表征（dom/som/hybrid），Layer 2 = 选模块（M1-M4）
- Pareto frontier 概念直接适用于 P79 的成本-成功率权衡分析

---

### C. 记忆管理（M4 直接相关）

#### 9. 全量上下文有害，需要记忆压缩

**来源**：Avenir-Web (Li et al., 2025) + M² (Yan et al., 2026, arXiv:2603.00503)

**证据**：
- Avenir-Web: W=∞（全量）成功率 36% vs W=5（压缩）48%，差 **12%**
- M²: 双层记忆（内部摘要 + 外部 insight bank）实现 **58.7%** token 降低 + **16.2%** 成功率提升（Qwen3-VL-32B on WebVoyager）
- M² Insight Retrieval 延迟仅 **6ms**，几乎无开销

**交叉验证**：
- AgentSwing [1]: 自适应 context 管理路由（Discard-All / Keep-Last-N / Summary 并行评估），token 减少时成功率反而提升
- WebCanvas [23]: 小模型（GPT-3.5）从 memory 获益（+5.6%），大模型（GPT-4）反而受损（-1.0%）——**记忆对小模型更有价值**

**对 P79 的映射**：
- **Phase 3 M4** — 双层记忆是 P79 的核心模块之一
- 短期（最近 4-5 步全量）+ 长期（每 5 步摘要 + skill library）
- 目标：40-70% token 降低，同时维持或提升成功率

---

#### 10. 三级层次记忆 >> 扁平记忆

**来源**：HMT (Tan et al., 2026, arXiv:2603.07024) + StructuredAgent (Lobo et al., 2025, arXiv:2603.05294)

**证据**：
- HMT 三级（Intent / Stage / Action）：71% 成本降低，recall 84.2% vs 扁平 65.8%
- HMT: flat memory 仅 +6.6%，hierarchical +更多
- StructuredAgent: AND/OR 树规划，hard tasks +5%（Structured Memory），WebArena 52.6%

**对 P79 的映射**：
- **M4 实现模板**：Intent = 任务目标；Stage = 功能子目标（导航/筛选/选择）；Action = 具体操作 + 语义描述
- 层次记忆 + skill library（预计算 50-100 条成功 trajectory）

---

#### 11. 并行 context 管理路由

**来源**：AgentSwing (Feng et al., 2026, arXiv:2603.27490)

**证据**：
- 三种 context 策略并行评估（Discard-All / Keep-Last-N / Summary），lookahead k 步选最优
- **k=3 最优**（60.0%），k=1 差（52.5%），k=5 无进一步提升（55.0%）
- WebArena: 56.7% 成功率 + 190.3K token vs Keep-Last-N 的 47.3% + 205.4K token

**对 P79 的映射**：
- **Phase 2** — 路由可扩展为 context 策略选择
- 对 Qwen3-VL-4B（context 有限），在 70-80% max context 时触发策略选择
- 注意：AgentSwing 用 ≥30B 模型，小模型效果需验证

---

### D. 失败恢复（M1/M2 直接相关）

#### 12. 确定性验证 + Fault Localization + Local Repair

**来源**：ContractSkill (2026) + OpAgent (2026, arXiv:2602.13559)

**证据**：
- ContractSkill: 确定性验证（URL/DOM/表单值变化）+ fault localization → VWA 上 +19.5pp（58.0%→77.5%）
- 去掉 fault localization: GLM 65.0% vs full 77.5%（差 12.5pp）
- ContractSkill cross-model transfer: +47.8pp（VWA 平均 32.6%→80.4%）
- OpAgent: Reflector 分析失败原因 + Summarizer 压缩历史，模块化框架

**对 P79 的映射**：
- **M1 (Retry)** — 不是简单重试，而是先验证→定位→对症修复
- **M2 (Fallback)** — 按故障类型分发：element not found→换表征；action no effect→等+重试；wrong element→加上下文

---

#### 13. 状态变化检测是可靠的失败信号

**来源**：Avenir-Web (Li et al., 2025)

**证据**：
- 4 层检测：执行错误 → 页面状态变化 → 动作特定验证（type 后读回比对）→ 战略模式分析（连续失败触发重评估）
- WEBSERV [15]: 拦截网络事件 + 等待页面静默（quiescence）后才返回观测

**对 P79 的映射**：
- 我们的 `state_change.py` 对应第 2 层，方向一致
- 第 4 层（连续失败模式检测）对应 `error_category: "no_progress"`

---

#### 14. 迭代精炼有效但收益递减

**来源**：Chain-of-Ground (Li et al., 2025)

**证据**：
- Single-step: 63.9% → Dual-step: 66.7%（+2.8%）→ Triple-step: 68.4%（仅 +1.7%）
- 成本线性增长（2x, 3x），收益递减

**交叉验证**：
- CATTS [8]: semantic deduplication 关键——不去重时 N=32 准确率反而下降（83.3%→80.1%），去重后 N=8 就提升到 84.5%

**对 P79 的映射**：
- **M3 retry** — 第一次 retry 性价比最高，之后迅速下降
- Router 应设 max_retry，根据 confidence 动态决定是否值得重试

---

### E. 两阶段推理（M3 直接相关）

#### 15. 知识驱动 CoT：程序知识提升最大

**来源**：Web-CogReasoner (Guo et al., 2026, arXiv:2508.01858, ICLR 2026)

**证据**：
- 三层知识累积增益：Factual +17.9%，Conceptual +11.3%，**Procedural +19.2%**（最大）
- 整体 84.4%（Web-CogBench），超过 Claude Sonnet 4（76.8%）和 Gemini 2.5 Pro（80.4%）
- 基础模型 Qwen2.5-VL-7B 在 WebVoyager 上 30.2%，平均步数 7.00（最低）

**交叉验证**：
- WebWorld [9]: 仅 1,000 条 CoT 样本即可激活小模型推理能力（0.561 score，超过 10x 数据的 direct reasoning 0.510）
- StructuredAgent [13]: AND/OR 树规划 WebArena 52.6%

**对 P79 的映射**：
- **M3 实现模板**：Stage 1 检索知识（Factual→Conceptual→Procedural），Stage 2 基于推理选动作
- 关键：Procedural 知识（"如何操作筛选器"/"如何提交表单"）收益最大——对应我们缺少的站点先验
- 不需要 fine-tuning，prompt-based CoT 即可（WebWorld 验证）

---

#### 16. 站点先验知识注入效果最大

**来源**：Avenir-Web (Li et al., 2025)

**证据**：
- EIP 在任务开始前搜索站点帮助文档，合成 2-4 条导航指令
- 去掉 EIP: 48% → **36%**（消融中降幅最大，-12%）
- 对比：去掉 MoGE -8%，去掉 Checklist -4%

**对 P79 的映射**：
- 解释了 classifieds 站点 agent 不知如何使用地点筛选的根因——缺乏站点先验
- **低成本替代**：为四个 VWA 站点各写一份静态操作指南注入 system prompt（成本为零）

---

### F. VWA 框架级限制

#### 17. `<select>` 下拉菜单是 VWA 框架级不可达缺陷

**来源**：Avenir-Web (Li et al., 2025) + P79 自有分析

**证据**：
- Avenir-Web 设计了专用 `select` action 绕过，system prompt 硬编码"绝对不要 click 下拉菜单"
- P79 分析确认三层不可达：bbox=0 过滤 → viewport 交集过滤 → scroll 只滚页面

**对 P79 的映射**：
- task_58/72/74 中 `<select>` 相关失败标注为**环境不可达**，不归因模型能力

---

#### 18. Task-Tracking Checklist 防止导航漂移

**来源**：Avenir-Web (Li et al., 2025)

**证据**：
- 轻量 Qwen-3-VL-8B 维护 2-6 个 atomic milestone checklist
- 去掉后成功率降 4%

**对 P79 的映射**：
- **M4 two-stage** — Checklist 可作为 two-stage 的轻量实现
- 我们的 state_change 检测看"页面是否变化"，checklist 看"任务是否推进"——粒度不同

---

## 二、对 P79 三阶段实验的综合建议

### Phase 1（表征筛选）

| 假设 | 文献支撑 | 强度 | 来源 |
|------|---------|------|------|
| SoM > DOM（小模型） | a11y 比 HTML 好 18.8pp（20B 模型）；WebWorld/WEBSERV 均选 a11y | **强** | [2][9][15] |
| hybrid > dom_only | 视觉反馈 > 文本反馈（+1.5%~+24%）；视觉输入不可替代 | **强** | CoG, [17] |
| SoM 辅助定位有效 | 视觉标记提升 grounding；语义描述 >> raw ID（差 27pp） | **强** | CoG, [6] |
| diff-based 历史高效 | 性能接近 full history，token 仅 1/3 | 中等 | [2] |
| SoM 降级时性能下降 | 标记大小影响有限（大 66.7% vs 小 65.5%） | 弱 | CoG |

### Phase 2（路由研究）

| 假设 | 文献支撑 | 强度 | 来源 |
|------|---------|------|------|
| 规则路由 > 固定策略 | MoGE 验证；AVR 三机制 78% 成本降低 | **强** | Avenir-Web, [7] |
| confidence-based 路由有效 | logprob 阈值路由广泛验证；margin-gated 56% token 降低 | **强** | [7][8] |
| 异构组合优于同构 | 不同模型互补（68.4% vs 63.9%）；MoMA Pareto 验证 | **强** | CoG, [24] |
| 路由应基于 UI 元素类型 | select/input/button/iframe 各有专用处理 | **强** | Avenir-Web |
| 难度自适应路由有效 | AVR difficulty classification + 选择性 compute allocation | 中等 | [7][20] |
| Lookahead k=3 最优 | AgentSwing 验证，k=1 差，k=5 无进一步提升 | 中等 | [1] |
| Prompt token 占成本 >70% | WebRouter 量化验证 | 中等 | [4] |

### Phase 3（模块消融）

| 模块 | 文献支撑 | 预期贡献 | 优先级 | 来源 |
|------|---------|---------|--------|------|
| M4 Memory（双层层次） | M² 58.7% token↓ + 16.2%↑；HMT 71% cost↓；小模型受益更大 | **10-20pp↑** | **最高** | [3][6][23] |
| M3 Two-stage | Web-CogReasoner Procedural +19.2%；WebWorld 1K CoT 激活推理 | **5-15pp↑** | **高** | [14][9] |
| M1 Retry（自适应） | ContractSkill fault localization +12.5pp；CoG 迭代精炼 +4.5% | **3-10pp↑** | 高 | [10], CoG |
| M2 Fallback | ContractSkill cross-model +47.8pp（极端值）；DMAST 自适应 > 固定 | **5-10pp↑** | 高 | [10][5] |
| M5 站点先验（候选） | Avenir-Web EIP -12%（消融中最大） | **5-12pp↑** | 中 | Avenir-Web |
| M6 Checklist（候选） | Avenir-Web -4% | **2-5pp↑** | 中 | Avenir-Web |

---

## 三、模型/系统在各 Benchmark 上的表现参考

### Web Agent 端到端成功率

| 配置 | 成功率 | Benchmark | 来源 |
|------|--------|-----------|------|
| OpAgent (SOTA, 72B+RL) | 71.6% | WebArena | [11] |
| AgentSwing + DeepSeek-v3.2 | 62.5% | WebArena | [1] |
| AgentSwing + GPT-OSS-120B | 60.0% | WebArena | [1] |
| StructuredAgent + Claude 3.7 | 52.6% | WebArena | [13] |
| CATTS margin-gated (GPT-OSS-120B) | 47.9% | WebArena-Lite | [8] |
| WEBSERV + Claude 4.5 | 46.7% (Shopping) | WebArena-Lite | [15] |
| HMT + GPT-4o | 38.7% | WebArena | [6] |
| ContractSkill + Qwen3.5-Plus | 81.0% | VWA (subset) | [10] |
| ContractSkill + GLM-4.6V | 77.5% | VWA (subset) | [10] |
| ContractSkill baseline (no skill) | 56.5-58.0% | VWA (subset) | [10] |
| DMAST + Gemma-3-12B-IT | 10.2% | VWA | [5] |
| DMAST baseline Gemma-3-12B-IT | 6.2% | VWA | [5] |
| Avenir-Web + Gemini 3 Pro | 53.7% | Online-Mind2Web | Avenir-Web |
| Avenir-Web + Qwen-3-VL-8B | 25.7% | Online-Mind2Web | Avenir-Web |
| M² + Qwen3-VL-32B | 74.0% (+16.2%) | WebVoyager | [3] |
| M² + Claude-3.7-Sonnet | 84.5% (+12.5%) | WebVoyager | [3] |
| Web-CogReasoner (Qwen2.5-VL-7B) | 30.2% | WebVoyager | [14] |
| WebCanvas best (GPT-4) | 23.1% | Mind2Web-Live | [23] |

### Grounding 准确率

| 配置 | 准确率 | Benchmark | 来源 |
|------|--------|-----------|------|
| CoG Triple-step (mixed Qwen) | 68.4% | ScreenSpot-Pro | CoG |
| Qwen3-VL-235B single-step | 63.9% | ScreenSpot-Pro | CoG |
| Qwen3-VL-32B single-step | 61.4% | ScreenSpot-Pro | CoG |
| Qwen2.5-VL-72B | 43.6% | ScreenSpot-Pro | [7] |
| Qwen2.5-VL-3B | 24.2% | ScreenSpot-Pro | [7] |
| OS-Atlas-7B | 18.9% | ScreenSpot-Pro | [7] |

### Cost-Aware 路由效果

| 方法 | 成本降低 | 准确率变化 | 来源 |
|------|---------|-----------|------|
| WebRouter (ca-VIB) | 87.8% | -3.8% | [4] |
| AVR (Warm + Difficulty) | 78% | -0.8pp | [7] |
| HMT (层次记忆) | 71.0% | +6.6% (WebArena) | [6] |
| M² (双层记忆) | 58.7% token↓ | +16.2% | [3] |
| CATTS (margin-gated) | 56% token↓ | +4.7% | [8] |
| MoMA (auto-routing) | 37.19% | +2.9% | [24] |

### P79 Qwen3-VL-4B 预期基线

基于文献中相近模型的表现：
- **VWA 基线成功率预期：10-30%**（参考 DMAST 12B=6.2%，Avenir-Web 8B=25.7%，Web-CogReasoner 7B=30.2%）
- Qwen2.5-VL 3B→72B 的 grounding 准确率只从 24.2%→43.6%（24x 参数仅 1.8x 提升），说明**小模型 grounding 天花板不高**
- 记忆模块对小模型受益更大（WebCanvas: GPT-3.5 +5.6% vs GPT-4 -1.0%）

---

## 四、文献空白（P79 可填补）

| 空白 | 描述 | P79 对应 |
|------|------|---------|
| **Gap 1** | Step-level adaptive routing for small VLM (≤10B) on complex benchmark — 无论文同时研究 | P79 核心定位 |
| **Gap 2** | 观测模态 × failure recovery 交互作用 — DOM retry 是否应换 SoM？ | Phase 1 × M1/M2 |
| **Gap 3** | 小 VLM 的视觉记忆管理 — 存全部/关键/不存截图？ | M4 设计 |
| **Gap 4** | 多模块联合消融 — M1-M4 交互效应（协同 vs 冗余？） | Phase 3 |
| **Gap 5** | Benchmark-specific 路由策略 — VWA 特有的任务特征对路由的影响 | Phase 2 |
| **Gap 6** | 小模型 confidence 校准 — Qwen3-VL-4B 的 logprob 是否可信？ | Phase 2 路由依据 |
| **Gap 7** | 真实部署（延迟/限速/动态内容）对路由效果的影响 | Future work |

---

## 五、参考文献

### 已精读论文

1. **Avenir-Web**: Li, A.Y. et al. "Avenir-Web: Human-Experience-Imitating Multimodal Web Agents with Mixture of Grounding Experts." arXiv:2602.02468, 2025.
2. **Chain-of-Ground**: Li, A.Y. et al. "Chain-of-Ground: Improving GUI Grounding via Iterative Reasoning and Reference Feedback." arXiv:2512.01979, 2025.

### Literature Review 覆盖论文（24 篇）

[1] Feng et al. "AGENTSWING: Adaptive Parallel Context Management Routing for Long-Horizon Web Agents." arXiv:2603.27490, 2026.
[2] Enomoto et al. "Read More, Think More: Revisiting Observation Reduction for Web Agents." arXiv:2604.01535, 2026.
[3] Yan et al. "M²: Dual-Memory Augmentation for Long-Horizon Web Agents via Trajectory Summarization and Insight Retrieval." arXiv:2603.00503, 2026.
[4] Li et al. "WEBROUTER: Query-Specific Router via Variational Information Bottleneck for Cost-Sensitive Web Agent." arXiv:2510.11221, 2025.
[5] Liu et al. "Dual-Modality Multi-Stage Adversarial Safety Training." arXiv:2603.04364, 2026.
[6] Tan et al. "Enhancing Web Agents with a Hierarchical Memory Tree." arXiv:2603.07024, 2026.
[7] Liu et al. "Adaptive Vision-Language Model Routing for Computer Use Agents." arXiv:2603.12823, 2026.
[8] Lee et al. "Agentic Test-Time Scaling for WebAgents." arXiv:2602.12276, 2026.
[9] Xiao et al. "WebWorld: A Large-Scale World Model for Web Agent Training." arXiv:2602.14721, 2026.
[10] "ContractSkill: Deterministic Verification and Repair of Multimodal Web Skills." 2026.
[11] "OpAgent (Operator Agent)." arXiv:2602.13559, 2026.
[12] Zheng et al. "SkillWeaver: Web Agents can Self-Improve by Discovering and Honing Skills." arXiv:2504.07079, 2025.
[13] Lobo et al. "STRUCTUREDAGENT: Planning with AND/OR Trees for Long-Horizon Web Tasks." arXiv:2603.05294, 2025.
[14] Guo et al. "WEB-COGREASONER: Towards Knowledge-Induced Cognitive Reasoning for Web Agents." arXiv:2508.01858, ICLR 2026.
[15] Lu et al. "WEBSERV: A Browser-Server Environment for Efficient Training of RL-based Web Agents at Scale." arXiv:2510.16252, 2025.
[16] Wu et al. "Mixture-of-Experts Meets In-Context Reinforcement Learning." arXiv:2506.05426, 2025.
[17] Yu et al. "Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos." arXiv:2603.22529, 2026.
[18] Zhang et al. "Optimizing Generative AI Networking: A Dual Perspective with MAS and MoE." arXiv:2405.12472, 2024.
[19] Yang et al. "Egocentric Co-Pilot: Web-Native Smart-Glasses Agents." WWW '26, 2026.
[20] Kumar et al. "Throttling Web Agents Using Reasoning Gates." arXiv:2509.01619, 2025.
[22] Qian et al. "WebGraphEval: Multi-Turn Trajectory Evaluation via Graph Representation." NeurIPS 2025 Workshop.
[23] Pan et al. "WebCanvas: Benchmarking Web Agents in Online Environments." arXiv:2406.12373, 2024.
[24] Guo et al. "Towards Generalized Routing: Model and Agent Orchestration for Adaptive and Efficient Inference." arXiv:2509.07571, 2025.

---

*初版：2026-04-07（2 篇精读）*
*更新：2026-04-07（合并 24 篇 Literature Review，共 26 篇）*
