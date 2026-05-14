# 学长好，想跟您确认几件事

---

## Part 1 — 我目前想到的，手里有的创新点，想和您确认下paper的分配

> **数据状态**：行为层（Phase 1 SR 数字）= pre-fix archived，24-condition 干净重跑后替换；机制层（§5）= B1 v2 post-fix（已修过 SOM regex bug + 重抽过 NPZ）。下面标 `(provisional)` 的是前者，结构性 claim 不依赖具体数字。

### 1. Phantom routing space (核心发现)

把 web agent 的观测拆成 **三个维度** (文本格式 / 系统 prompt / 图片存在) → 一共 8 个实验 cube. 现在 paper 测的 6 modes 里, **4 个 corner 是"不带标注图"** (我叫它 phantom space), 这块以前没人系统性地测过. 发现 phantom 内部不是塌成一个点 (不是只有 phantom som), 是 **二维的结构** (text-flattening 跟 SoM-prompt 各自能解 P-SoM 解不了的任务). 这个 space 的中心 P-SoM (`[SOM_MARKS]` text + SoM-prompt + 无图) 是 hero, 满足以下四个最主要的优势：

| | 数据 |
|---|---|
| (a) cost ≈ DOM | 没有 image token |
| (b) latency ~50% lower | 省 image inference 那步 (cls SoM 74s → P-SoM 18s, 4×) `(provisional)` |
| (c) routing signal AUROC ≥ baseline | 5-mode 信号强度全 > baseline `(provisional)` |
| (d) 每个 phantom 模式独立成功 1.7-3.8pp | 6-mode 路由上限比 som+dom+vision 三 mode 大 +7.14pp (B0 reddit) `(provisional)` |

> 表里数字是 pre-fix archived 数据，24-condition Phase 1a 干净重跑后会被 fresh meta 替换；**结构性 claim（二维结构 / P-SoM 是中心 hero / 四重 drop-in 性质）不依赖具体数字**。

并且我已经做好了对应的 4-dimension Evidence 和 4-zoom 解释层。

### 2. Routing (router design)

利用 phantom space 的 +7.1pp 路由上限 `(provisional)` 做 router, 目前信号基础设施已经完备 (deferred to paper-2).

### 3. VWA bug fix

VWA / WebArena benchmark 本身的 bug —— 我跑实验过程发现的 + 之前 paper 提过没修的 (Patched WebArena 那种). 包括 Magento FPC / Postmill PHP gc / Wikipedia ZIM 版本 / select dropdown 7 层 / Phase A 4-cluster (dispatch + page_changed + fuzzy cycle + RNG). 这个您建议做 ACL position paper / survey + community repo 持续更新.

### 4. Mechanistic interpretability —— 已有具体结果，不再是"到时候分析"

不看 model 输出, 看 **model 内部** (hidden states / activations) 在 phantom mode 下的 signature 差异. **B1 (Qwen3-VL-4B local) 上的分析已经跑完** (v2 post-fix NPZ), 三条工具线产出一个 **probe–causal–steering 三分结论**：

| 层 | 方法（实际做的）| 结果 | 含义 |
|---|---|---|---|
| **Probe** | Method 4.2 PCA cosine gap | mode AUROC **1.000** 全 layer | mirage 信息在 hidden state 里**完全可分** |
| **Causal** | §5.4 activation patching (per-task paired) | L11–L17 displacement **0.20–0.35**, held-out 仍成立 | mechanism 真实存在且**可被 causal 使用** |
| **Steering** | Method 4.4 mean-diff steering | held-out H-mean **0.12** vs in-sample 0.29 (gap +0.16) | 单一 population direction **不 transfer** |

→ 核心 finding: **mechanism 是真实的, 但是 per-task encode 的** —— 没有单一固定方向能 capture 它. 这反过来 motivate 了 **SAE feature steering / 自适应方向作为 future work**（不是"修一个 ceiling", 是一个方向性结论 —— 对齐 IOI 的 "feature *used* vs feature *encoded*"）.

> 注：之前这里写"SAE 已做"不准确 —— 实际做的是 Method 4.4 mean-diff steering，SAE 现在是 future work（paper-2 候选）。

cross-architecture (Phi-3.5-Vision / Qwen2-VL-7B) 的 extraction pipeline 也已 paper-grade 就绪 → 可直接跑 H1' capacity-limit 对照 (4B 的 shortcut 是容量限制, 还是训练分布先验). 目前 mechanistic 仅 B1 —— B0 走 proxy API, 无 model internals access.

### 5. 其他 supporting innovations (paper_planning §22 列的)

- **9-cell intervention taxonomy**: 3 spectrum (bug fix / synthesis / channel addition) × 3 layer (server-side / agent-pipeline / LLM-internal), 12+ verified industry instances 西方+中国
- **B0 5-call probe (reproducibility 诚实 disclosure)**: API token-non-deterministic 但 action-convergent
- **4-dimension Evidence framework**: Outcome / Macro / Micro / Efficiency 正交
- **Industry-vs-research epistemic distinction**: 工业 deploy at artifact level (单 mode 部署省钱), paper 在 research level 做 controlled cross-mode comparison — 不同 epistemic level



### 您觉得上面这些，应该分为几篇paper？以哪篇为主呢

---