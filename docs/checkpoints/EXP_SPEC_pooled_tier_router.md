---
type: spec
status: executed
created: 2026-07-28
executed: 2026-07-28
purpose: 同族 pooled × cost-tier router 的可学性实验 — 规格，供新 session 直接执行
---

> **已执行 2026-07-28 — H-pool 不支持。** 产物
> `docs/analysis/cross_sites/router_pooled_tier_learnability.{md,json}`，
> producer `scripts/analysis/router_pooled_tier_learnability.py`，chronicle 见 实验笔记 §399。
> 严格支配 always-cheapest 0/26 arm×cell（最有利角落 0/4）；相对六固定 mode 菜单非支配 0/26。
>
> **本规格的两处需要下次注意**：
> 1. **§2 与 §3 口径分叉** —— §2 把 H-pool 写成「Pareto **支配**」，§3 把判据锁成
>    「95% **非支配**」。非支配是 admissibility（+7pp SR 换 +10% cost 也算过），
>    支配是 superiority。只跑锁定判据会给出与假设相反方向的 headline。产物已改三档并列报。
> 2. **§4 第 5 步的对照臂不够** —— 只隔离了「同族」与「粒度」，没隔离**池化本身**。
>    执行时补了第 5 臂 `per_cell × cost_tier`，事后看它是信息量最大的一臂：
>    唯一通过锁定判据的 reddit·B0 在 per-cell 训练下同样通过，
>    所以那些 pass 不是池化的功劳。（同型风险即 §316 所指。）

# 实验规格：同族 pooled × cost-tier router

**一句话**：现有 router 负结果全部建立在「跨族池 × which-mode 标签」上。
2026-07-28 分层显示这恰好是最不利的组合。本实验测最有利的组合是否可学。

**执行者注意**：规格里每条「不可动」都对应一条已裁定，改它等于改 estimand。
动手前先 `known.py` 查那条裁定。

---

## 1. 为什么做

`router_pooling_by_family.py`（2026-07-28）把原本混算的 pooled conflict rate 分层：

| | which-mode 冲突 | **cost-tier 冲突** | **tier 天花板** |
|---|---|---|---|
| **同族 B0+B1** cls | **48.0%** | **24.0%** | 88.0% |
| **同族 B0+B1** red | **45.0%** | **5.0%** | **97.5%** |
| 跨族 B1+B2 cls | 81.8% | 45.5% | 77.3% |
| 混合（论文现报） | 57.4% / 56.0% | 31.5% / 12.0% | 85.6% / 94.8% |

同族 + 粗粒度是最有利角落，而**它从没被训过**。

另有一条独立理由：**cost-tier 标签天然免疫 §395.2 的缺陷**。
which-mode 标签靠 `MODES` 先验顺序做 tie-break，而 B-1806 实测该顺序与实际成本不符
且跨 cell 反转（B0 上实测最便宜的 vision 在列表里排最后）⇒ 12.5–54.64% 的标签返回了
严格更贵的成功 mode。tier 只分 image / text-only，**不需要 tier 内排序**。

## 2. 假设

**H-pool**：在同族池（B0+B1）上、以 cost-tier 为标签训练的 router，
其 (SR, cost) 操作点 Pareto 支配 always-cheapest 固定策略，
在 ≥1 个 (site) 上成立。

两个结果都有用：
- **支配** ⇒ Paper B 的结论需实质修改：不是「学不到」，是**原来找错了池和粒度**
- **不支配** ⇒ 结论加限定后**更强**：连最有利配置（同族一致、天花板 97.5%、供给充足）
  都打不过白送的

## 3. 不可动的（每条对应一条已裁定）

| 项 | 锁定值 | 出处 |
|---|---|---|
| **CV 协议** | **task-held-out 5-fold within fixed cells** | §216.1 —— LOCO 已被 supersede（与 per-cell LR-head 架构不兼容，且 cells 共享 task pool 不是真分布外），LOCO 只作 Appendix |
| **tie-break** | 保留 `MODES` 先验顺序，**不得**换成实测成本 | B-1806 —— 实测顺序跨 cell 反转 + episode cost 内生于行为会引入 cost←outcome 循环 + n_succ 14-61 太小。「Do NOT silently switch — that is an oracle-label estimand change」<br>（本实验用 tier 标签，**结构上不触发这条**，但若中途回退到 which-mode 必须遵守） |
| **对照** | `always-cheapest` 固定策略 | 六格全部 = **Vision**（丢整棵 AXTree 省的 token 比图像 token 贵得多；cls·B1 Vision 0.04316 vs DOM 0.05951，−27.5%） |
| **Pareto 判据** | per-cell **paired bootstrap 95% 非支配** | §150b.4 / B-1550 two-layer operational deployment criterion |
| **成本口径** | `total_billed_cost_usd`，**仅 cell 内可比** | B0 计 proxy API 账、B1/B2 是电费当量，跨 backbone 不可比 |
| **嵌套** | **真嵌套**（阈值与 best_mode 都必须在训练折内选） | §392.2 —— §388.4 的「嵌套」被 §388.7.3 (B-1903) 指出训练侧复用了全局 OOF 分数、best_mode/cheap_mode 用全部六格已实现结局选 |

## 4. 做什么

1. **构池**：`load_pool()`（`router_label_supply_diagnosis.py`）→ 只留 `backbone ∈ {B0, B1}`，
   按 site 分开（cls / red 各一个池）
2. **标签**：`MODE_COST_TIER`（`p79/policies/router_features.py`）→ 二值 `image` / `text_only`
3. **训练**：走现有 nested CV 管线（`router_offline_replay` 那套），task-held-out 5-fold，
   阈值在训练折内选
4. **评估**：每个 (site, backbone) cell 上算 router 的 (SR, cost) 与 always-cheapest 对比，
   paired bootstrap 95%
5. **对照臂**（必须一起报，否则无法归因）：
   - 同族池 × **which-mode** 标签 —— 隔离「粒度」的贡献
   - 跨族池 × **cost-tier** 标签 —— 隔离「同族」的贡献
   - 现状（跨族 × which-mode）—— 复现已知负结果作 sanity check

⚠️ 没有第 5 步，就算 H-pool 成立也说不清是「同族」还是「粗粒度」带来的。
这是 §316「单臂测量制造幽灵 confound」的同型风险。

## 5. 已知限制（写进产物，不可省）

- **n 小**：同族 shared task 只有 cls 50 / red 20
- **天花板 ≠ 可学性**：§394 —— red·B2 是六格唯一显著的，AUROC 却 0.483
  （全局判别与尾部可用性在本数据上解耦）
- **48% / 45% 的 which-mode 冲突**可能是真实模型差异，也可能是噪声，现有数据分不开
- **B0 vs B1 混池的成本不可比** —— 池化只用于**标签与特征**，(SR, cost) 评估仍须 per-cell
- 产物必须标 `post_hoc_exploratory=True` / `h10_eligible=False`
  —— 这不是预注册的 H10，是探索性分析（同 `router_objective_ordering.md` 的做法）

## 6. 参照数字（B0·cls，来自 `router_objective_ordering.md`）

| policy | SR % | mean cost | Δcost |
|---|---|---|---|
| `cheapest` (= `single:Vision`) | 25.00 | 0.06481 | −10.4% |
| `best_sr` (= `single:SoM`) | 27.23 | 0.07236 | +0.0% |
| `triage_only` | 27.23 | 0.06312 | −12.8% |
| `oracle_sr` | 43.30 | 0.06259 | −13.5% |
| `oracle_sr_cost` | 43.30 | 0.05777 | −20.2% |

router 要有意义，必须落在 `cheapest` 右上方（SR 更高或成本更低且另一维不劣）。

## 7. 产物

- `docs/analysis/cross_sites/router_pooled_tier_learnability.md`（+ `.json`）
- 结论无论正负都 append 实验笔记（下一个 § 号）
- 若 H-pool 成立 → **回 REBUILD_PLAN 改 Phase 3 的论文结构决策**（Paper B 的核心论断要改）
