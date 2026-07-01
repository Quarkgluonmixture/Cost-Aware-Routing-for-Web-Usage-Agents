# AAAI-27 draft — 数字填充清单 (companion to `aaai27_main.md`)

> 每个 `<TBD>` / `[P]` / `[V]` 槽位 → 数据来源 + producer 命令。落一批数据就跑一轮，
> 把 tag 逐步升级到 [A]。**全部 verdict slot 填完之前不得投稿**。
> Deadline 锚: abstract **2026-07-21** / full paper **2026-07-28** / supp **2026-07-31** (UTC-12)。

## 0. 聚合链已打通 ✅（2026-07-01）— sync 配方 + 剩余项

**✅ RESOLVED 2026-07-01**：聚合门控链 = A100 fire 侧 `validate_fire_manifest --apply` auto-bind →
`docs/checkpoints/pre_run/fire_manifest.json`（21 bound）→ **手动 promote** 进
`results/phantom_paper/run_manifest.yaml` `cells:`（grade: paper-grade）→ aggregators。
本轮已做：rsync fire_manifest A100→DGX + promote 15 cells（B1/B2 cls ×12 + B0 red dom/som/vision）
→ `make analysis FAST=1` 吃进 **21 cells**；H1 interim k=3 / H2(a) 3/3 / H3 interim 已进 draft slots。
**新 cell land 后的 sync 配方**：`rsync condense-a100:.../fire_manifest.json` → promote 新 bound 条目进
run_manifest.yaml → `make analysis FAST=1` → 更新 draft tags/slots。

| 剩余槽位 | 现状 | 动作 |
|---|---|---|
| red·B0 P-text `[P]→[A]` | run 已完成 (07-01) 但 fire 侧还没 auto-bind | 等 A100 validate_fire_manifest 下一轮 bind → 走 sync 配方 |
| §5.4 latency canonical (retry-adjusted) | 只有 archive p95 [V] | aggregator emit `total_minus_retry_ms` per-mode 表（cross_sites cost_per_mode 目前无 wall-clock 列，需补列） |
| §6 AUROC 全 cell | auroc_cross_condition 需确认是否已扩到 21 cells | 查 `auroc_cross_condition.md`，扩了就把 §6 那句的 per-cell 范围更新 |
| ⚠️ figures 阶段 2 个 pre-existing 崩点 | ① `fig0c` 系 `max() empty`（首轮）② `axis1_microbehavior.py` KeyError `'mean_jaccard'`（partial reddit 数据） | 修 figure 脚本对 partial-cell 的容错；不 block 聚合（aggregators 在 figures 之前已完成） |
| ⚠️ red·B0 drop-one 慎用 | `fig0c` 的 b0_red 行（DOM 4.88 / SoM 3.90 / Vision 3.41）是 **3-mode partial portfolio** 上的 drop-one，与 6-mode 定义不可比 | draft 不引用，等 6 modes 齐 |

## 1. 等 fire 落地（顺序 = chain 自动推进）

| 槽位 | 依赖 | ETA 估算 (B0 red ~2-2.5d/cond) |
|---|---|---|
| Table 2 red·B0 P-prompt / P-SoM | fire 进行中 (P-text 07-01 已 land → 下两条) | ~07-05 |
| Table 2 red·B1 6 modes | B0 red 完成后 chain 进 B1 | ~07-05 → +6 cond |
| Table 2 red·B2 6 modes | B1 red 之后 | 再 +6 cond ⚠️ 极可能压线/超线 |
| `<H1-VERDICT>` `<H3-VERDICT>` | ≥2 个 6-mode 齐 cell (cls·B0 + red·B0 即可先出首个 pooled 读数; 正式 gate 要 6 cell) | red·B0 齐 (~07-05) 出 k=2 预读; k=6 取决于 B1/B2 red |
| Table 4 + `<H10-VERDICT>` | Pass-2 router fire (`queue_phase1_router_paper_grade.sh`, 6 cond, ~3-5d) | 排在 Pass-1 后 ⚠️ 高风险 |

**⚠️ Pass-2 预演发现 (2026-07-02, rehearsal dir `results/phantom_paper/l1_router_rehearsal_20260702/`)**:
训练链 3 段全通 (extract_50_features → with_mi → train_l1_router), `h10_entropy_gate.json` 正常 emit
(h10_status=ok, global_min_bits=2.10 > 1.0 DEFER 门槛) — H10 fail-closed 缺件已在 verdict 周前排除。
**但 B2_classifieds router 不可训** (best-mode 标签仅 16 task, 5 fold 全 insufficient_train_data,
B-1640 runtime 会 loud-fail) — 若 B2_reddit 同地板, H10 ≥5/6 判据数学上最多 4/6 → §6 descriptive
分支概率高 (已进 advisor 四合一消息 3️⃣(b))。**verdict-day/Pass-2 前必做**: 全 Pass-1 land 后在
**canonical** `results/phantom_paper/l1_router/` 重跑同链 (rehearsal 目录只是预演, 勿混用), 再 rsync
lr_fold pkls → A100 (Pass-2 queue 的 artifact gate 检查 30 pkl)。

**产出命令**: `python scripts/analysis/aggregate_phase1_full_prereg_decision.py` (H1/H2a/H3/R-rule) ·
`python scripts/analysis/aggregate_h10_pareto.py` (H10) · `make analysis` (全管线)。

## 2. Deadline 风险账 (2026-07-01 起算)

- 剩余 Pass-1: red·B0 ×2 (~4-5d) + red·B1 ×6 + red·B2 ×6。若 B1/B2 red 与 cls 同速 (~1d/cond)，
  Pass-1 全齐 ≈ **07-17±3**；Pass-2 router 6 cond ≈ +3-5d → **07-20~07-24**，分析 +1d。
- ⇒ 全量 6-cell verdict 版本 vs 07-28 full deadline: **可行但零余量**；abstract (07-21) 时
  大概率只有 k=2~4 cell。
- **降级预案 (跟学长确认)**: (a) 主文以 landed cells + pooled-k<6 透明披露投稿（prereg 有
  fixed-cells 设计，k 不齐必须明写，不能静默降 k）; (b) H10 若 Pass-2 赶不上 → §6 走
  descriptive operating-points 分支（draft 已双分支预写）; (c) 弃 AAAI 回 D11 early-Sep 原计划。
- ⚠️ **决定 (a)/(b)/(c) = estimand-adjacent 决策，需 advisor + witness**，不是工程可拍。

## 3. 写作侧遗留 (不依赖数据)

- [ ] Figure F1: 2×2 diamond 示意图（新画; canvas `phantom_space` 有底稿）
- [ ] Figure F2: 6-cell drop-one forest — `scripts/analysis/figures/fig_forest_drop_one.py`
- [ ] Figure F3: per-cell Pareto scatter — `aggregate_h10_pareto.py` 附带
- [ ] paper.bib 编译核对（全部 [@key] 已按 §1/§2 摘要核对存在; `nikankin2025sametask` 留 rebuttal）
- [ ] AAAI reproducibility checklist 逐项填写
- [ ] 匿名化 pass（host 名 / 用户名 / OSF 匿名视图）
- [ ] LaTeX 转换 (pandoc → aaai27.sty)
- [ ] **commit 本 prose 前必须跑 /stress (+ codex + gemini chain)** — CLAUDE.md auto-trigger #2
