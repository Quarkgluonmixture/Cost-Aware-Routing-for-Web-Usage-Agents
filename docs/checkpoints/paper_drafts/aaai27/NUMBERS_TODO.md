# AAAI-27 draft — 数字填充清单 (companion to `aaai27_main.md`)

> 每个 `<TBD>` / `[P]` / `[V]` 槽位 → 数据来源 + producer 命令。落一批数据就跑一轮，
> 把 tag 逐步升级到 [A]。**全部 verdict slot 填完之前不得投稿**。
> Deadline 锚: abstract **2026-07-21** / full paper **2026-07-28** / supp **2026-07-31** (UTC-12)。

## 0. 聚合链已打通 ✅（2026-07-01）— sync 配方 + 剩余项

**✅ RESOLVED 2026-07-01**：聚合门控链 = A100 fire 侧 `validate_fire_manifest --apply` auto-bind →
`docs/checkpoints/pre_run/fire_manifest.json`（21 bound conditions）→ **手动 promote** 进
`results/phantom_paper/run_manifest.yaml` `cells:`（grade: paper-grade）→ aggregators。
本轮已做：rsync fire_manifest A100→DGX + promote 15 conditions（B1/B2 cls ×12 + B0 red dom/som/vision）
→ `make analysis FAST=1` 吃进 **21 conditions**；H1 interim k=3 / H2(a) 3/3 / H3 interim 已进 draft slots。
**新 condition land 后的 sync 配方**：`rsync condense-a100:.../fire_manifest.json` → promote 新 bound 条目进
run_manifest.yaml → `make analysis FAST=1` → 更新 draft tags/slots。

| 剩余槽位 | 现状 | 动作 |
|---|---|---|
| ~~red·B0 P-text `[P]→[A]`~~ | ✅ DONE 2026-07-02: R32139 promoted + 22-condition 聚合 + Table 2 升 [A] | (队列⑧ 补丁顺带完成; promotion-gap watch 已 cron 化) |
| §5.4 latency canonical (retry-adjusted) | 只有 archive p95 [V] | aggregator emit `total_minus_retry_ms` per-mode 表（cross_sites cost_per_mode 目前无 wall-clock 列，需补列） |
| §6 AUROC 全 statistical cells | auroc_cross_condition 需确认是否已纳入 21 conditions | 查 `auroc_cross_condition.md`，扩了就把 §6 那句的 per-cell 范围更新 |
| §6 covariate 基线 + template-disjoint split `<TBD>` | ⚠️ **rehearsal vintage 首轮已出且是红旗** (2026-07-05): B0_cls/B1_cls 上 18-feat LR vs 3-协变量 scalar 基线 ΔAUROC −0.013/+0.007 (CI 全含 0) — 当前可测 learnability 全部可由 trivial 协变量达成; disjoint 掉幅 0~−0.05。**不推翻 landed claim (§6 无 landed AUROC), 但若 canonical 全量后格局不变 → §6.6 披露扩全 cell + scalar 基线进 §6.5 梯队**。⚠️ 勿混淆 estimand: 攻击面 = LR mode-prediction AUROC, ≠ §1 per-mode confidence-signal AUROC (0.766 P-SoM, 不含 task 文本特征) | Pass-1 全 land 后 `python scripts/analysis/router_covariate_baseline.py` 在 canonical NPZ 重跑 → 填 §6 `<TBD>`; report `docs/analysis/cross_sites/router_covariate_baseline_2026-07-05.md` |
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

### 1.1 Verdict/branch 命名槽总账（唯一映射）

每个 slot 一行；branch 文件只消费这里登记的 ID。`partial-data` 只允许预览，不能 splice 为终局；`needs-producer-field` 表示 estimand 已锁但 producer 尚未发出该稳定字段。

| slot_id | target anchor | estimand | producer command | artifact field | status |
|---|---|---|---|---|---|
| THETA | `branch_prewrites_s1_abstract.md` / all `«THETA»` | H1 six-mode P-SoM drop-one FE point estimate (pp) over complete planned cells | `python scripts/analysis/aggregate_phase1_full_prereg_decision.py` | `results/phantom_paper/phase1_full_prereg_decision.json::pooled_h1_fe.theta_FE_pp` | partial-data (k=3/6) |
| CI_LO | branch / all `«CI_LO»` | H1 primary pooled task-paired bootstrap 95% CI lower bound (pp) | same H1 producer | `...json::pooled_h1_bootstrap.ci95_lo_pp_bootstrap` | partial-data (k=3/6) |
| CI_HI | branch / all `«CI_HI»` | H1 primary pooled task-paired bootstrap 95% CI upper bound (pp) | same H1 producer | `...json::pooled_h1_bootstrap.ci95_hi_pp_bootstrap` | partial-data (k=3/6) |
| P_BOOT | branch / all `«P_BOOT»` | H1 one-sided paired-bootstrap p-value for H0: θ_FE ≤ +1.0pp | same H1 producer | `...json::pooled_h1_bootstrap.p_one_sided_bootstrap` | partial-data (k=3/6) |
| K | branch / all `«K»` | number of complete six-mode (site, backbone) statistical cells in the reported FE pool | same H1 producer | `...json::k_actual` | partial-data (3; final design 6) |
| AX1 | branch / all `«AX1»` | H3 axis-1 FE point estimate: per-cell P-text \ P-SoM unique contribution (pp) | same H1 producer | `...json::h3_axis1_pooled_fe.theta_FE_pp` | partial-data |
| AX2 | branch / all `«AX2»` | H3 axis-2 FE point estimate: per-cell P-prompt \ P-SoM unique contribution (pp) | same H1 producer | `...json::h3_axis2_pooled_fe.theta_FE_pp` | partial-data |
| AX1_CI_LO | branch complete abstracts / `«AX1_CI_LO»` | H3 axis-1 primary bootstrap 95% CI lower bound (pp) | same H1 producer | `...json::h3_axis1_pooled_fe.ci95_lo_pp_bootstrap` | partial-data |
| AX1_CI_HI | branch complete abstracts / `«AX1_CI_HI»` | H3 axis-1 primary bootstrap 95% CI upper bound (pp) | same H1 producer | `...json::h3_axis1_pooled_fe.ci95_hi_pp_bootstrap` | partial-data |
| AX2_CI_LO | branch complete abstracts / `«AX2_CI_LO»` | H3 axis-2 primary bootstrap 95% CI lower bound (pp) | same H1 producer | `...json::h3_axis2_pooled_fe.ci95_lo_pp_bootstrap` | partial-data |
| AX2_CI_HI | branch complete abstracts / `«AX2_CI_HI»` | H3 axis-2 primary bootstrap 95% CI upper bound (pp) | same H1 producer | `...json::h3_axis2_pooled_fe.ci95_hi_pp_bootstrap` | partial-data |
| UNIQ_CLS | branch / all `«UNIQ_CLS»` | **distinct Classifieds task IDs** in the union across the three canonical backbones of `{t: P-SoM succeeds and every other menu arm fails within that cell}`; deduplicate task IDs across backbones | `python scripts/analysis/aggregate_phase1_full_prereg_decision.py` | required stable field `...json::site_unique_psom_union.classifieds.n_unique_task_ids` | needs-producer-field; do not splice |
| UNIQ_RED | branch / all `«UNIQ_RED»` | **distinct Reddit task IDs** under the same across-backbone union and within-cell P-SoM-only rule | same H1 producer | required stable field `...json::site_unique_psom_union.reddit.n_unique_task_ids` | needs-producer-field; do not splice |

**UNIQ 选择理由。** Main §7 currently exposes per-backbone counts (cls·B0/B1/B2 = 2/3/1). A single site slot cannot select one backbone, and summing would double-count the same benchmark task when it is unique in multiple backbone cells. The deduplicated union is therefore the only site-level scalar that still means “number of tasks”; until the producer emits the two registered fields, these slots are intentionally blocked.

### 1.2 §8 strict-gate power producer

| slot_id | target anchor | estimand | producer command | artifact field | status |
|---|---|---|---|---|---|
| STRICT_GATE_POWER | `aaai27_main.md` §8 Statistics / `strict-gate power is reported as <TBD>` | achieved/prospective power for the **strict six-mode H1 drop-one gate** using the realized canonical effect and paired-bootstrap SE, not the archive four-mode additive proxy | recompute by the method in `docs/analysis/cross_sites/power_analysis.md` after the six-mode verdict artifact lands | verdict-day power ledger field `strict_6mode_h1.power_at_realized_effect` | needs-verdict-data |

## 2. Deadline 风险账 (2026-07-01 起算)

- 剩余 Pass-1: red·B0 ×2 (~4-5d) + red·B1 ×6 + red·B2 ×6。若 B1/B2 red 与 cls 同速 (~1d/cond)，
  Pass-1 全齐 ≈ **07-17±3**；Pass-2 router 6 cond ≈ +3-5d → **07-20~07-24**，分析 +1d。
- ⇒ 全量 6-cell verdict 版本 vs 07-28 full deadline: **可行但零余量**；abstract (07-21) 时
  大概率只有 k=2~4 cell。
- **✅ 降级预案已定 (user 拍板 2026-07-02): a+b 组合** — (a) deadline 时以 landed cells + pooled-k<6
  透明披露投稿 (fixed-cells 设计, k 不齐必须明写); (b) H10 若 Pass-2 赶不上 → §6 走 descriptive
  operating-points 分支。(c) 弃 AAAI 已排除 — **venue 同日拍板 = 投 AAAI-27**。
- 附带同日决策: Amendment 03 (FWER) 不走 (论文保持现有 hedge 表述, prereg 不动); B-1885 两死任务
  **分母 205 不动**, §8 Scope 段披露句已落 (2026-07-02)。

## 3. 写作侧遗留 (不依赖数据)

- [ ] Figure F1: 2×2 diamond 示意图（新画; canvas `phantom_space` 有底稿）
- [ ] Figure F2: 6-cell drop-one forest — `scripts/analysis/figures/fig_forest_drop_one.py`
- [ ] Figure F3: per-cell Pareto scatter — `aggregate_h10_pareto.py` 附带
- [ ] paper.bib 编译核对（全部 [@key] 已按 §1/§2 摘要核对存在; `nikankin2025sametask` 留 rebuttal）
- [ ] AAAI reproducibility checklist 逐项填写
- [ ] 匿名化 pass（host 名 / 用户名 / OSF 匿名视图）
- [ ] LaTeX 转换 (pandoc → aaai27.sty)
- [ ] **commit 本 prose 前必须跑 /stress (+ codex + gemini chain)** — CLAUDE.md auto-trigger #2
