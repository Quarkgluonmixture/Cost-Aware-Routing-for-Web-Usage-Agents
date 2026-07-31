---
type: task
status: active
priority: P0
horizon: now
order: 2
blocker: "B2_reddit 已齐 (36/36 conditions, 07-29 实证) — 原 blocker 解除。现串在 08-03 对账 → 骨架之后"
eta: "k=6 重灌 → slotsheet 重生成 → 两篇 REALM 稿 splice (deadline 08-05 周三)。⚠️ h10 artifacts 最新仍是 07-22 的 k=5 版 (h10_artifact_regen_provenance_2026-07-22.md, 自标非提交终态) — 二次 promote 未做"
detail: preregistration
created: 2026-05-22
updated: 2026-07-31
---

# Analysis + gating (H1/H2/H3 + H10)

H1/H2/H3 = FE pooled bootstrap percentile p (R1-R5 framing per §2.5). H10 = router
Pareto non-dominance + 5/6 grid (§6). 详 [[paper_planning]] §16 + [[preregistration]] §2.

## k=5 verdict 已落 (2026-07-16, PN06 Branch B)

## k=6 重灌 — 现为关键路径 (2026-07-22 起, 笔记 §383)

AAAI 撤出后 deadline 变 **08-05 (REALM)**, B2_reddit ~07-26/27 落地 → k=6 **够得着且是实质升级**:
"5 of 6" 免责段整段消失 + **B-1284 cross-family modifier 解除** (Gemma 两站齐 → 跨族复制主张可用)
+ Protocol Note 06 两轨制披露整块可删。

**重灌前置 (硬)**
- [x] B-1887 mode 齐全性守卫已修 (commit `554cc7c`) —— 否则不加 `--cells` 重跑会静默吃进
      残缺 B2_reddit 并产出污染 oracle 标签
- [ ] B2_reddit 收尾 → bind → promote 进 run_manifest (36 条齐)
- [ ] Stage 1/2/3 全 6 cell 重跑 → canonical entropy gate **二次 promote** (k=5 版见
      `pre_run/h10_artifact_regen_provenance_2026-07-22.md`, 明标非提交终态)
- [ ] `make analysis` → slotsheet 重生成 → 两篇 splice

## 稿件层已知需改 (k=5 重生成产物发现, 笔记 §383.2)

1. **"two cells lack trainable labels" → "three cells"** (B0_reddit / B1_reddit / B2_classifieds
   均 0/5 可训练折; 旧数字引自 07-15 的 4-cell 快照, 当时 B0_reddit 尚未入池)
2. **机制口径**: 熵闸门**是过的** (2.1-2.2 bits vs 1.0)，阻断是 `insufficient_train_data`
   (`N_MIN_CLASS_TRAIN=10`)，**不是**标签集中
3. **补披露 Pass-2 从未 fire** (每 cell `Pass-2 runs: 0` → `k_of_n="0/0"` 是独立成因)

## 要和学长对的 Phase 1 清单（2026-07-29 user 定）

不在本轮定去留，对账时一并过：

- **mechanistic canonical sweep**（24 cell，driver pid 38603，deadline 08-01）——
  mechanism 线 2026-05-14 已暂搁，跑完了用 / 不用 / 存档待定
- **WA（WebArena）** —— 现为 future work，是否进 Phase 1 待对
