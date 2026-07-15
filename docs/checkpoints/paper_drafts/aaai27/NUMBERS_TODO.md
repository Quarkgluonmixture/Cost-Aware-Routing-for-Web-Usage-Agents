# AAAI-27 draft — 数字填充清单 (companion to `aaai27_main.md`)

> 每个 `<TBD>` / `[P]` / `[V]` 槽位 → 数据来源 + producer 命令。落一批数据就跑一轮，
> 把 tag 逐步升级到 [A]。**全部 verdict slot 填完之前不得投稿**。
> Deadline 锚 (OpenReview 实测 2026-07-14, 双标注 UTC-0 + AoE): abstract registration
> **Jul 22 11:59 AM UTC-0 = Jul 21 23:59 AoE** / full submission **Jul 29 11:59 AM UTC-0
> = Jul 28 23:59 AoE** / reviewer-nomination 冻结 **Jul 21 AoE** (资格作者未提名 = desk-reject
> 风险)。旧 "supp 07-31" 在 live 表单无对应独立 deadline (supplements 是同表单字段) — 按随
> full submission 交齐处理, 除非 CFP 另有说明。系统 = **OpenReview 非 CMT**。

## 0. 聚合链已打通 ✅（2026-07-01）— sync 配方 + 剩余项

**✅ RESOLVED 2026-07-01**：聚合门控链 = A100 fire 侧 `validate_fire_manifest --apply` auto-bind →
`docs/checkpoints/pre_run/fire_manifest.json`（28 bound conditions）→ **手动 promote** 进
`results/phantom_paper/run_manifest.yaml` `cells:`（grade: paper-grade）→ aggregators。
截至 2026-07-14：rsync fire_manifest A100→DGX + promote **28 conditions**；`sr_per_mode.json`
`summary_table` 的 28 行均为 `complete_exact=true`，H1 interim k=3 / H2(a) 3/3 / H3 interim 已进 draft slots。
**新 condition land 后的 sync 配方**：`rsync condense-a100:.../fire_manifest.json` → promote 新 bound 条目进
run_manifest.yaml → `make analysis FAST=1` → 更新 draft tags/slots。

| 剩余槽位 | 现状 | 动作 |
|---|---|---|
| Reddit 已落 SR 回填 | ✅ 2026-07-14: 28-condition 聚合；B0 red P-SoM + B1 red DOM/SoM/Vision/P-text/P-SoM 均为 exact [A] | Table 2 已从 `sr_per_mode.json` 按 ROUND_HALF_UP 一位小数回填；promotion-gap watch 已 cron 化 |
| §5.4 latency canonical (retry-adjusted) | 只有 archive p95 [V] | aggregator emit `total_minus_retry_ms` per-mode 表（cross_sites cost_per_mode 目前无 wall-clock 列，需补列） |
| §6 AUROC 全 statistical cells | auroc_cross_condition 需确认是否已纳入当前 28 conditions | 查 `auroc_cross_condition.md`，扩了就把 §6 那句的 per-cell 范围更新 |
| §6 covariate 基线 + template-disjoint split `<TBD>` | ⚠️ **rehearsal vintage 首轮已出且是红旗** (2026-07-05): B0_cls/B1_cls 上 18-feat LR vs 3-协变量 scalar 基线 ΔAUROC −0.013/+0.007 (CI 全含 0) — 当前可测 learnability 全部可由 trivial 协变量达成; disjoint 掉幅 0~−0.05。**不推翻 landed claim (§6 无 landed AUROC), 但若 canonical 全量后格局不变 → §6.6 披露扩全 cell + scalar 基线进 §6.5 梯队**。⚠️ 勿混淆 estimand: 攻击面 = LR mode-prediction AUROC, ≠ §1 per-mode confidence-signal AUROC (0.766 P-SoM, 不含 task 文本特征) | Pass-1 全 land 后 `.venv/bin/python scripts/analysis/router_covariate_baseline.py --raw-features results/phantom_paper/l1_router/raw_features_phase1a.npz --out-json results/phantom_paper/l1_router/covariate_baseline.json` → 填 §6 `<TBD>`; report `docs/analysis/cross_sites/router_covariate_baseline_2026-07-05.md` |
| ⚠️ red·B0 drop-one 慎用 | `fig0c` 的 b0_red 行（DOM 4.88 / SoM 3.90 / Vision 3.41）是 **3-mode partial portfolio** 上的 drop-one，与 6-mode 定义不可比 | draft 不引用，等 6 modes 齐 |

## 1. 等 fire 落地（顺序 = chain 自动推进）

| 槽位 | 依赖 | ETA 估算（截至 2026-07-14） |
|---|---|---|
| Table 2 red·B0 P-prompt | 尚未 land；其余 5 modes 已为 [A] | ETA 待确认；仍是 red·B0 成为完整六模式 cell 的唯一缺口 |
| Table 2 red·B1 P-prompt | 在跑；其余 5 modes 已为 [A] | ~07-15 |
| Table 2 red·B2 6 modes | 队列等待 B1 P-prompt 完成 | ~07-15 后启动，+6 cond ⚠️ 压线 |
| `<H1-VERDICT>` `<H3-VERDICT>` | 当前 k=3；正式 gate 要 6 cell | k=4 先取决于 B1 P-prompt；k=6 仍取决于 B0 P-prompt + B2 red ×6 |
| Table 4 + `<H10-VERDICT>` | Pass-2 router fire (`queue_phase1_router_paper_grade.sh`, 6 cond, ~3-5d) | 排在 Pass-1 后 ⚠️ 高风险 |

**⚠️ Pass-2 预演发现 (2026-07-02, rehearsal dir `results/phantom_paper/l1_router_rehearsal_20260702/`)**:
训练链 3 段全通 (extract_50_features → with_mi → train_l1_router), `h10_entropy_gate.json` 正常 emit
(h10_status=ok, global_min_bits=2.10 > 1.0 DEFER 门槛) — H10 fail-closed 缺件已在 verdict 周前排除。
**但 B2_classifieds router 不可训** (best-mode 标签仅 16 task, 5 fold 全 insufficient_train_data,
B-1640 runtime 会 loud-fail) — 若 B2_reddit 同地板, H10 ≥5/6 判据数学上最多 4/6 → §6 descriptive
分支概率高 (已进 advisor 四合一消息 3️⃣(b))。**verdict-day/Pass-2 前必做**: 全 Pass-1 land 后在
**canonical** `results/phantom_paper/l1_router/` 重跑同链 (rehearsal 目录只是预演, 勿混用), 再 rsync
lr_fold pkls → A100 (Pass-2 queue 的 artifact gate 检查 30 pkl)。

**⚠️ 2026-07-15 offline replay 更新 (隔离目录 `l1_router_offline_20260715/`, 笔记 §367)**:
① **H10 可训性恶化至 3/4 cell 无完整 policy**: B1_reddit 虽 6/6 condition 齐但 union-success
仅 26 labels → 0/5 fold 可训 (同 B2_cls 16-label 地板); B1_cls 4/5 — **H10 ≥5/6 已结构性不可达
(≤4/6 上限, 与调度无关)**。② 唯一完整 cell B0_cls 的 OOF offline routed SR = 25.45% <
best-single SoM 27.23% (−1.79pp, 成本 +4.42%) — 当前 learned router 无附加值证据 (§6 预锁降级
规则的直接佐证; oracle 43.30% ceiling 仍在 = complementarity 主张不受影响)。③ **extractor 坑**:
`extract_50_features.py` 无 `--manifest` CLI, canonical manifest JSON 缺失时静默 glob 会把
stale run (B1_3mode_20260413) 抓成第 7 条 — **canonical 重跑前必须先给脚本加 manifest 白名单
(或复用 offline 目录的 scratch 模式), 否则 canonical 训练数据被污染**。

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
| K | branch / all `«K»` | number of complete six-mode (site, backbone) statistical cells in the reported FE pool | same H1 producer | `...json::pooled_h1_bootstrap.k_cells` | partial-data (3; final design 6) |
| AX1 | branch / all `«AX1»` | H3 axis-1 FE point estimate: per-cell P-text \ P-SoM unique contribution (pp) | same H1 producer | `...json::h3_axis1_pooled_fe.theta_FE_pp` | partial-data |
| AX2 | branch / all `«AX2»` | H3 axis-2 FE point estimate: per-cell P-prompt \ P-SoM unique contribution (pp) | same H1 producer | `...json::h3_axis2_pooled_fe.theta_FE_pp` | partial-data |
| AX1_CI_LO | branch complete abstracts / `«AX1_CI_LO»` | H3 axis-1 primary bootstrap 95% CI lower bound (pp) | same H1 producer | `...json::h3_axis1_pooled_fe.ci95_lo_pp_bootstrap` | partial-data |
| AX1_CI_HI | branch complete abstracts / `«AX1_CI_HI»` | H3 axis-1 primary bootstrap 95% CI upper bound (pp) | same H1 producer | `...json::h3_axis1_pooled_fe.ci95_hi_pp_bootstrap` | partial-data |
| AX2_CI_LO | branch complete abstracts / `«AX2_CI_LO»` | H3 axis-2 primary bootstrap 95% CI lower bound (pp) | same H1 producer | `...json::h3_axis2_pooled_fe.ci95_lo_pp_bootstrap` | partial-data |
| AX2_CI_HI | branch complete abstracts / `«AX2_CI_HI»` | H3 axis-2 primary bootstrap 95% CI upper bound (pp) | same H1 producer | `...json::h3_axis2_pooled_fe.ci95_hi_pp_bootstrap` | partial-data |
| UNIQ_CLS | branch / all `«UNIQ_CLS»` | **distinct Classifieds task IDs** in the union across the three canonical backbones of `{t: P-SoM succeeds and every other menu arm fails within that cell}`; deduplicate task IDs across backbones | `python scripts/analysis/aggregate_phase1_full_prereg_decision.py` | required stable field `...json::site_unique_psom_union.classifieds.n_unique_task_ids` | needs-producer-field; do not splice |
| UNIQ_RED | branch / all `«UNIQ_RED»` | **distinct Reddit task IDs** under the same across-backbone union and within-cell P-SoM-only rule | same H1 producer | required stable field `...json::site_unique_psom_union.reddit.n_unique_task_ids` | needs-producer-field; do not splice |

**K 字段选择。** `pooled_h1_bootstrap.k_cells` 在 partial 与最终 COMPLETE artifact 中都稳定存在；top-level `k_actual` 只在 `k<6` 分支写出，因此不能作为终态槽源。

**UNIQ 选择理由。** Main §7 currently exposes per-backbone counts (cls·B0/B1/B2 = 2/3/1). A single site slot cannot select one backbone, and summing would double-count the same benchmark task when it is unique in multiple backbone cells. The deduplicated union is therefore the only site-level scalar that still means “number of tasks”; until the producer emits the two registered fields, these slots are intentionally blocked.

### 1.2 §8 strict-gate power producer

| slot_id | target anchor | estimand | producer command | artifact field | status |
|---|---|---|---|---|---|
| STRICT_GATE_POWER | `aaai27_main.md` §8 Statistics / `strict-gate power is reported as <TBD>` | achieved/prospective power for the **strict six-mode H1 drop-one gate** using the realized canonical effect and paired-bootstrap SE, not the archive four-mode additive proxy | recompute by the method in `docs/analysis/cross_sites/power_analysis.md` after the six-mode verdict artifact lands | verdict-day power ledger field `strict_6mode_h1.power_at_realized_effect` | needs-verdict-data |

## 2. Deadline 风险账 (2026-07-01 起算)

- 剩余 Pass-1: red·B0 P-prompt（未 land，ETA 待确认）+ red·B1 P-prompt（在跑，~07-15）+
  red·B2 ×6（队列等待 B1 完成）。若 B2 red 仍按 ~1d/cond 串行，最早约 **07-21** 齐 B2；
  Pass-1 全齐仍额外取决于 B0 P-prompt。Pass-2 router 6 cond 再需 +3-5d，分析 +1d。
- ⇒ 全量 6-cell verdict 版本 vs 07-28 full deadline: **可行但余量极小且受 B0 P-prompt 未定 ETA 约束**；
  abstract (07-21) 时大概率为 k=4（若 B1 P-prompt 如期 land）。
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

## 4. K5 激活后的字数账 (2026-07-15)

- **PROTOCOL_NOTE_06 激活变更集给 §4(+1 句) / §8(+1 段) 净增 ~210 词** (`wc -w` 6085→6295 raw)。
  7 页零余量 → 最终打包时用 cut_prewrites 抵消; 若 k=6 升级触发, 删全部 K5-CONDITIONAL 语言自动回吐。
- [ ] verdict-day 打包前重跑 convert.sh 页数账 (checklist item 6b) 确认正文仍 ≤ 第 7 页末
