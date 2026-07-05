# Learned Router — Scalar-Covariate 基线 + Template-Disjoint Split 敏感性 (2026-07-05)

> **动机 (lit 2606.22864 "When AUC 0.998 Is Not Enough" 可迁移攻击面)**: LR 的 "learnable routing signal" 是否主要由 trivial 协变量 (文本长度 / 模板) 驱动? VWA task 是模板实例化的 (`intent_template_id`: cls **75 模板 / 234 task**, red **87 / 210**), canonical Stage-2 fold 按 *task* 切 (B-1871 per-site KFold) → 同一模板的实例横跨 train/test。实测 canonical split 下 **72.2% (B0_cls) / 52.7% (B1_cls) 的 holdout task 其模板在 train fold 中出现过** → "learnable" 有可能部分是 "memorizable"。本报告 = 数据侧防线实测。

## 0. 数据世代 (重要 — 结论有效范围)

| 项 | 值 |
|---|---|
| 数据源 | `results/phantom_paper/l1_router_rehearsal_20260702/raw_features_phase1a.npz` (2026-07-02 Pass-2 管线预演, 建自已 landed Pass-1 paper-grade runs) |
| 覆盖 | 4/6 cell 有数据 (B0_cls 97 / B0_red 51 / B1_cls 55 / B2_cls 16 个有标签 task); B1_red / B2_red 无 Pass-1 run |
| 可训练 cell | **仅 2 个**: B0_classifieds (5/5 folds), B1_classifieds (4/5 folds)。B0_reddit / B2_cls 在 B-995 min-class filter (n≥10) 下全 fold 退化为单类 → untrainable (与 canonical `stage3_summary.json` 一致) |
| Oracle labels | N=1 draws (每 condition 单 run), tie-break = prior cost order (§6.6 已披露) |
| **有效范围** | **pre-Pass-2 rehearsal vintage**。Phase 1a 全部 landed 后应在 canonical Stage-1 NPZ 上重跑本脚本 (一条命令, 见 §5) |

## 1. 方法 (与 canonical 协议完全同轨)

脚本: `scripts/analysis/router_covariate_baseline.py` (analysis 层新增, import 复用 `train_l1_router_with_mi` / `train_l1_router`, 零逻辑复制)。

- **Eval 协议 = canonical Stage 2+3 逐件复现**: per-site shared KFold seed=42 (fold map 与 rehearsal `*_fold_assignment.json` **逐 task 验证一致, 4/4 cell**) + B-995 min-class filter + `StandardScaler + LogisticRegression(class_weight=None, C=1.0, max_iter=2000)` + pooled out-of-fold 评估。full_lr 的 Stage-2 (fold-local TF-IDF(30) + pooled-MI top-18) 按 fold 原样重放。
- **指标**: τ-free argmax mode-match accuracy + **macro one-vs-rest AUROC** (multiclass mode 预测; 与 `aggregate_routing_auroc.py` 的 per-mode confidence-signal AUROC 是**不同 estimand**, 见 §4 警示)。canonical Stage-3 的 `cv_mode_match_acc` 含 τ fallback: B0_cls 0.4415 / B1_cls 0.366 — 本脚本 τ-free 复现 0.443 / 0.381, 一致性 OK。
- **Feature sets**: ① `full_lr` = canonical 18-feat (TF-IDF+MI); ② `scalar_min` = {intent 字符长, intent 词数, has_reference_image} (模板级浅特征); ③ `scalar_plus` = +reasoning_difficulty; ④ `template_onehot_oracle` = template_id one-hot (显式记忆上界参照, 非可辩护特征集)。site one-hot 不适用 — canonical 架构 per-cell, site 为 cell 常量 (设计已排除)。
- **Split regimes**: `standard` (canonical B-1871) vs `template_disjoint` (GroupKFold by `intent_template_id`, 同 site universe, 模板零跨界)。

## 2. 关键数字

### 2.1 B0_classifieds (n_oof=97; majority-class acc = 0.402)

| Feature set | Split | Acc | macro-AUROC | wAUROC |
|---|---|---:|---:|---:|
| **full_lr (canonical 18-feat)** | standard | 0.443 | **0.522** [0.475, 0.575] | 0.567 |
| scalar_min (3 个 trivial 协变量) | standard | 0.423 | **0.535** [0.497, 0.577] | 0.596 |
| scalar_plus | standard | 0.433 | 0.534 | 0.596 |
| template_onehot_oracle | standard | **0.485** | 0.538 | 0.606 |
| full_lr | template-disjoint | 0.433 | 0.520 [0.468, 0.570] | 0.556 |
| scalar_min | template-disjoint | 0.443 | 0.523 | 0.573 |
| template_onehot_oracle | template-disjoint | 0.402 (=majority) | 0.427 | 0.387 |

配对 bootstrap (B=2000, 按 task 对齐):
- **full_lr − scalar_min**: Δacc **+0.021** [−0.072, +0.103]; ΔmacroAUROC **−0.013** [−0.048, +0.018] → **无可分辨优势, AUROC 点估计甚至反向**
- template_oracle − full_lr: Δacc +0.041 [−0.052, +0.144] → 存在 full_lr 未吃到的可记忆成分 (方向性)
- full_lr disjoint − standard: Δacc −0.010 [−0.093, +0.072]; ΔAUROC −0.001 [−0.058, +0.058] → **掉幅 ≈ 0**

### 2.2 B1_classifieds (n_oof=42 standard / 39 disjoint; majority acc = 0.357/0.385)

| Feature set | Split | Acc | macro-AUROC |
|---|---|---:|---:|
| **full_lr** | standard | 0.381 | **0.567** [0.534, 0.623] |
| scalar_min | standard | **0.452** | 0.560 |
| template_onehot_oracle | standard | 0.381 | 0.536 |
| full_lr | template-disjoint | 0.436 | 0.533 [0.469, 0.602] |
| scalar_min | template-disjoint | 0.538 | 0.564 |
| template_onehot_oracle | template-disjoint | 0.385 (=majority) | 0.448 |

配对 bootstrap:
- **full_lr − scalar_min**: Δacc **−0.071** [−0.167, +0.024] (scalar 基线 acc 反超); ΔmacroAUROC +0.007 [−0.054, +0.073]
- full_lr disjoint − standard (32 common tasks): ΔmacroAUROC **−0.048** [−0.149, +0.032]

### 2.3 Sanity checks

- template one-hot oracle 在 disjoint 下**按构造塌缩**到 majority (acc 恰=majority, AUROC<0.5) — split 实现正确的阴性对照。✓
- fold map 与 canonical artifact 逐 task 一致 (4/4 cell)。✓
- B0_red / B2_cls untrainable 与 canonical stage3 结论一致。✓

## 3. 解读 (诚实版 — 结果对 "LR 学到非平凡信号" 不利)

1. **Trivial 协变量差不多打平 full LR**。两个可训练 cell 上, 3 特征 scalar 基线 (长度/词数/是否带参考图) 的 AUROC 与 18 特征 TF-IDF+MI LR 统计不可分 (Δ CI 全部横跨 0), B1_cls 上 acc 还反超 7pp。按 2606.22864 的框架: **当前 LR 展示的 "learnability" 几乎全部可由 trivial 协变量达成**, 尚无证据表明 TF-IDF 文本语义贡献了增量信号。
2. **Template-disjoint 掉幅有限 (B0_cls ≈0 / B1_cls −0.05 AUROC), 但这不是加固**: 掉幅小不是因为 LR 泛化好, 而是因为 standard split 下它本来就只有 0.52-0.57 的 macro-AUROC — **没有多少 template 记忆可掉** (对照: 显式记忆的 template oracle 从 0.485 acc 塌到 majority)。同时 template oracle 在 standard 下 acc 比 full_lr 高 4pp, 说明数据里确有可记忆模板结构, 只是 30-word TF-IDF 词表没吃满它。
3. **对 §6 claim 的含义**: §6 目前是 v0 placeholder (H10 全 TBD), 没有已落地的 LR AUROC prose claim — 所以这不是 "推翻已发表数字", 而是**前瞻性红旗**: 若 Phase 1a 全量数据后 LR 相对 scalar 基线仍无可分辨优势, §6.6 "低成功 cell 上 router = majority-class baseline 而非 learned boundary" 的披露需要升级为**所有 cell** 的定性, 且 §6.5 智能基线梯队应把 **scalar-covariate 基线与 template-disjoint 敏感性作为标配行** (比现有 decision-stump 行更尖锐)。反过来若全量数据后 full_lr 显著超过 scalar 基线, 本脚本即为现成的防线证据。
4. **样本量警示**: B0_cls n=97 / B1_cls n=42, 6 类退化到 post-filter 2-3 类 (dom/som/pprompt 存活)。0.52 vs 0.535 级别的差异在这个 n 下没有功效可言 — 这既保护 claim (打不死) 也保护攻击 (立不住), 唯一出路是 Phase 1a 全量 + (Phase 1b) shop 扩样。

## 4. Estimand 警示 (不要误用本结果)

本报告攻击/检验的是 **LR 的 multiclass mode-prediction AUROC** (task 文本特征 → 预测 per-task 最优 mode)。它 **不是** §1/aaai27 引用的 "routing-signal AUROC ≥ baseline" (0.766 P-SoM vs 0.673 DOM cls·B0) — 那是 `aggregate_routing_auroc.py` 的 **per-mode confidence-signal AUROC** (模型置信信号预测 within-condition 成功), 特征根本不含 task 文本, 不受 template 记忆攻击面影响 (per-task 协变量攻击面是否适用需另行分析, 与本脚本无关)。两个 estimand 别混。

## 5. 复现 / Phase 1a 全量后重跑

```bash
# rehearsal vintage (本报告):
.venv/bin/python scripts/analysis/router_covariate_baseline.py

# Phase 1a 全量 landed 后 (canonical Stage-1 NPZ 重新生成后):
.venv/bin/python scripts/analysis/router_covariate_baseline.py \
    --raw-features results/phantom_paper/l1_router/raw_features_phase1a.npz \
    --out-json results/phantom_paper/l1_router/covariate_baseline.json
```

输出 JSON (含 per-row OOF dump 供配对 bootstrap): `results/phantom_paper/l1_router_rehearsal_20260702/covariate_baseline.json`。
Bootstrap CI 复算: 见 JSON `oof_rows` (task_id 对齐) — 本报告 CI 用 B=2000, seed=42。

---
*生成: 2026-07-05, 脚本 `scripts/analysis/router_covariate_baseline.py` (schema `2026-07-05-router-covariate-baseline-v1`), 数据 `l1_router_rehearsal_20260702` (pre-Pass-2 vintage)。未 commit。*
