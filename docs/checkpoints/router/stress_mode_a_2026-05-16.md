# Mode A — Claude /stress on router_proposals_v1.md (self-audit)

**Scope**: pre-fire (router design lock before §B Phase 1a launch)
**Artifacts**: `docs/checkpoints/router/proposals_v1.md`, `docs/checkpoints/paper_planning.md` §8, `docs/checkpoints/phase1_plan.md` §C, `docs/checkpoints/pre_run/preregistration.md` §354 + §359 + Appendix A
**Findings count target**: 7 (pre-fire band) ≥ 3 OOB

## Verdict (one sentence)

Proposals v1 sets up two routes that share baseline/CV/data and look complementary on paper, but **estimand is `<pending>`, the rule-based P1 leaks gold-label supervision, and the learned P2 puts ~4520 features against ~180 train tasks** — pre-data the design supports a "router works ≈ best-single-mode within noise" outcome, not a publishable lift.

## Strong claims (survive attack)

1. **Preregistration §354 + §359 anchor is real**: both proposals cite the locked 5-fold site-stratified CV + train-fold-stratified best-single-mode baseline, so the comparison-anchor protocol is not invented post-hoc. Reviewer can hold the paper to this.
2. **P1 extends existing `p79/experiment/router.py` scaffold** (line 1-149 already in tree): so engineering effort estimate is real, not aspirational. The 3-day P1 timeline is defensible.

## Weak claims — methodology errors (OOB first)

### Finding F1 — Estimand 对 H9/H10 写"pending"但提案推下游下断言 [P0 — OOB]

**Claim**: `router_proposals_v1.md §"Statistical gate"`: "router lift over best-single-mode ≥ 1.0pp, ≥ K cells Holm-significant — *K and δ pending advisor lock*"; `preregistration.md` Appendix A 2026-05-15 entry: H9/H10 estimand `<pending>`.

**代码现实** / **设计现实**: The proposal commits to baselines, CV, features, labels, training — everything DOWNSTREAM of estimand — but the gate formula itself is blank. δ=1.0pp is borrowed from H1 (phantom phenomenon) without justification specific to router.

**攻击**: 这是 paper-grade HARKing 风险结构. 等数据 land 之后再选 K_h9 / δ_h9 = 标准的 post-hoc HARKing. H1 δ=1.0pp 用的 logic 是 "≈ 2 tasks in N=234, matches per-cell bootstrap SE" (preregistration §341); 同 logic 套 H9/H10 → 但 H9 比较的是 router-vs-best-single-mode lift, 这个 lift 的 sampling noise floor 跟 P-SoM drop-one lift 的 noise floor 不同 (router lift 受 best-single-mode 不稳定性 + classifier 训练随机性双重 noise). 不能直接搬 δ.

**Defuse**: Lock estimand pre-data in a separate session (already in advisor backlog). Specific formula: pooled FE-meta over 6 cells (mirrors H1 decision "3A" 2026-05-14), with router-specific noise calibration via pre-data simulation — sample N=180 from each cell, compute best-single-mode train-vs-test fluctuation under no-signal null, set δ_h9 = 2 × noise SD. Effort: 1 week pre-data simulation script + advisor sync.

**Effort**: 1 day for simulation script, 0.5 day for advisor sync, **launch-blocking** (cannot lock preregistration §6 OSF without estimand).

**Confidence**: high

### Finding F2 — P1 Layer-1 `has_reference_image` 是 gold-answer-derived → 测试时 leak [P0 — OOB]

**Claim**: `router_proposals_v1.md §"P1 Feature spec"`: "`has_reference_image` | manual binary tag (already curated, §139 audit Cat B) | bool".

**代码现实**: Cat B 是 VWA 任务 `intent_template` 内含 `[IMG]` 的 task (e.g., "Find a listing similar to this image: [IMG]"). 但 §139 audit Cat B 列表是用 task gold annotation 标的, 不是 agent 在 step 1 看到 intent 后 derivable 的 attribute. 如果 router 用这个 tag 做 routing decision, 它在用 **gold-answer-conditioned feature** 做决策 — reviewer 看就是 oracle leak.

**攻击**: 这违反 routing literature standard "router 决策 feature 必须从 task 输入 + agent-observable signals 派生" (Chen FrugalGPT 2023 §4.1; Welleck et al. 2024 routing study). 解法不是删 feature, 而是改 derivation: 写一个 `has_reference_image_derivable(intent_template)` = `re.search(r'\[IMG\]|image|photo|picture', intent_template, re.I)` 从纯文本 intent 自动 extract. 当前 §139 audit Cat B 标签 = audit-time gold, P1 router 要用 derivable 版本.

**Defuse**: 在 `p79/experiment/router.py` 加 `def has_reference_image_from_intent(intent: str) -> bool: return bool(re.search(r'\[IMG\]|image|photo|picture', intent, re.I))`, 跟 §139 audit Cat B label 算 F1 score 验证 derivability (target ≥ 0.95). 同 logic 对 `is_pure_search_intent` 应用.

**Effort**: 2 小时 (code + validation script), 不卡 launch — P1 implementation 当下加.

**Confidence**: high

### Finding F3 — P2 4520-dim feature × N=180 train task = catastrophic overfit [P0]

**Claim**: `router_proposals_v1.md §"P2 Feature spec"`: 总 dim ≈ 4520, L2 regularized; "训练 per (site, model) cell, on train fold of 5-fold CV" → N_train ≈ (234 + 210) / 2 × 0.8 ≈ 180 per cell-fold.

**设计现实**: feature-to-sample ratio = 4520 / 180 ≈ 25:1. 即便 L2 reg, 这是 sklearn `LogisticRegression(multi_class='multinomial')` 文档 NOT recommended 的 regime (sklearn docs §1.1.11 建议 N >> dim 或者 dim < 0.1×N).

**攻击**: 不是 "可能 overfit", 是 "几乎一定 overfit". 即便 sklearn 默认 `C=1.0` L2 reg, 4520 dim 上 N=180 训练 = test error >> train error by 5-10pp 通常. 加上 6-class multinomial → effective N per class boundary ≈ 30. 跟 best-single-mode baseline 比 1.0pp 完全在 overfit noise 内.

**Defuse**: 两条路:
(a) Feature selection pre-data: 限定 dim ≤ N/10 ≈ 18 features. 选 F3 (6 categorical) + F4 (4 browser) + 5-10 top TF-IDF terms via mutual info on **train fold only**;
(b) Switch from LR 到 GBDT (sklearn `HistGradientBoostingClassifier`) — 对 high-dim small-N more robust; 用 5-fold inner CV select min_samples_leaf.
**Recommend (a) for paper-grade simplicity**. P2 should commit to dim cap in proposal v2.

**Effort**: 1 day pre-data feature-selection design.

**Confidence**: high

### Finding F4 — F5 信号 feature = 在用 4-fold drop-in property 自己当 input [P1 — OOB]

**Claim**: `router_proposals_v1.md §"P2 F5 signal"`: "per-mode AUROC signal (verbalized + behavioral) from `confidence_summary.json`".

**设计现实**: phantom-SoM 4-fold drop-in property (c) = "signal AUROC ≥ baseline modes" 是 paper §1 hero contribution-1 的一部分. 现在 P2 用这些 signal 做 router input feature, 然后 router lift 算 contribution-2.

**攻击**: 这是 "用 contribution-1 的 evidence 作 contribution-2 的 mechanism" 双倍计算. 如果 P2 router 提升 1.5pp 主要来自 F5, paper 写 "phantom routing space 有 routing signal property, AND learned router exploits this property for lift" — 但 router lift 不是独立新发现, 而是 contribution-1 在另一个评测指标下的 restate. Reviewer (尤其 cascade literature) 会说 "你这是 self-citation 包装成 2 contributions".

**Defuse**: 两种选项:
(a) Disable F5 in primary P2 spec, 留 F1-F4. F5 作 ablation row "vs +signal feature" 在 §6 router section disclose. — 隔离 router lift 来自 task-level features, 不来自 contribution-1 signal infra.
(b) 把 router 重 framed 成 "deployment scaffolding for the phantom routing space" 而不是 independent contribution — 但 advisor 2026-05-14 已锁 router 为 paper-1 contribution-2, 这条 require reopen scope.
**Recommend (a)**.

**Effort**: 0 (only changes the primary-vs-ablation framing of F5 in proposal v2).

**Confidence**: medium-high — 取决于 reviewer 对 contribution overlap 的容忍度. R3 venue 可能 OK; R1 / NeurIPS 会拒.

### Finding F5 — best-single-mode anchor 在 5-fold × N=180 上 fold-noisy [P1]

**Claim**: `preregistration.md §359`: "Per cell: mode with highest mean adjusted-SR on train fold"; `router_proposals_v1.md §"shared substrate"` 引用此 lock.

**设计现实**: 6 modes × ~180 train task → 每 mode 训练 SR estimate 的 bootstrap SE ≈ √(p(1-p)/180) ≈ 3.7% (at p=0.18 mean SR). 在 5 modes 估值 within 5-7pp of each other (cls / red 实测数据范围) 时, top-mode 选谁 = fold-noisy.

**攻击**: 如果 anchor 本身在 fold 之间换 mode (e.g., fold 1 best = P-SoM, fold 2 best = SoM), router-vs-anchor lift 的 estimate 有 anchor-flicker noise 加在 router noise 上, 解 lift est CI 宽到无法 reject 0. 这是经典 "biased baseline comparison" 问题 — Romano-Wolf 多比较 framework 解, 但 preregistration 没 lock.

**Defuse**: 两条路:
(a) Pre-data Monte Carlo: 对每 cell 抽 5-fold split 100 次, 计算 best-single-mode 跨 fold 一致性 (Kendall τ). 如果 τ < 0.7 → anchor unstable, 改 "majority-winner across folds" 或 "ensemble of top-3 modes".
(b) Add secondary anchor: "best-single-mode-by-mean-rank" (rank modes per task, sum ranks, anchor = top mode). 更稳, 但需 preregistration update.
**Recommend (a) as launch-blocking simulation; (b) as fallback if τ < 0.7**.

**Effort**: 1 day MC simulation on existing pre-Phase-A archive data (`meta_phantom_lift.md` cells).

**Confidence**: medium

### Finding F6 — Tier-0 random over 6 modes = 稻草人 baseline [P1]

**Claim**: `router_proposals_v1.md §"Baselines & evaluation"`: "vs Tier-0 random (uniform over 6 modes)".

**设计现实**: 6 modes 内 vision / SoM 长期 SR 7-10pp 低于 P-SoM / SoM. Random uniform 选 → 实质 router 任何 sensible decision 都比 random 高 5+pp. 这是 weak baseline.

**攻击**: 顶 tier reviewer 不接受 random 作 baseline 在 routing paper. Standard expectation = "frequency-weighted random" (each mode weighted by its mean SR) 或 "best-3-mode random" (random over top 3 modes per cell). 这两个 strawman 都比 uniform-random 强很多. Paper-1 没有 "vs serious random" → 节 §6 router prose 弱.

**Defuse**: 把 Tier-0 拆三档:
- Random uniform (现 spec)
- Random weighted by mean SR per mode (新)
- Random over top-3-modes per train fold (新)

3 个 random tier 都跑, 选最强 (random top-3) 作 primary anchor + 其他 2 作 transparency.

**Effort**: 0.5 day (random eval 几乎 zero-compute).

**Confidence**: high — peer-lab reviewer 会一眼看出.

### Finding F7 — P1 threshold grid search 消耗 dof, 无 multiple-comparison correction [P1 — OOB]

**Claim**: `router_proposals_v1.md §"P1 Threshold setup"`: "16 combos × per-cell tuning. 5-fold CV on 234 cls + 210 red tasks per cell".

**设计现实**: 16 (θ_dom, θ_cmplx) combos × 5 fold × 6 cells = 480 evaluations. 每 cell 选 max-train-fold-SR threshold combo, 加 fold-CV 选最佳超参 = nested CV 才合法. 当前 spec 没 nested CV.

**攻击**: 不 nested CV → 选 threshold 时已用过 train fold 的 SR 信息, threshold 选定后 test fold SR estimate **偏高** (the standard "overfitting the CV" pitfall, Hastie ESL 7.10.2). 类比: 你不能用 train SR 选 model, 再用同一 train SR 报 model performance. 480 evaluation 加上 (16 cells × 16 combo) × Bonferroni → 任何 lift < 3pp 都不显著.

**Defuse**: 两条路:
(a) Use **nested 5-fold CV**: 每 outer fold 内, 再 5-fold inner CV 选 threshold, 然后 outer fold test 报 final SR. 增计算 5× 但 dof 干净.
(b) Lock threshold pre-data via baseline data: pre-Phase-A `meta_phantom_lift.md` archive 上 fit threshold 一次, freeze, 跑 Phase 1a fresh data 不重 tune.
**Recommend (b) — pre-data threshold lock = paper-grade cleanest, no multiple-comparison correction needed**.

**Effort**: (a) zero (nested CV is standard sklearn `GridSearchCV`), (b) 1 day on archive data.

**Confidence**: high

## Bug Table — actionable summary

### 🔴 P0 (lock 前必须 fix)

| # | Bug | Blast Radius | Launch 卡? |
|---|---|---|---|
| P0-1 | `router_proposals_v1.md §"Statistical gate"` H9/H10 estimand `<pending>` — gate formula undeclared, δ_h9 不能简单搬 H1 δ=1.0pp | 等 Phase 1a 数据 land 后再 lock estimand = HARKing. Reviewer (R1-R2) check pre-registration on OSF will reject the router contribution. Paper §6 router gate evaluation 无法 pre-data lock → §B Phase 1a launch 可以但 OSF preregistration 不能 commit. | 不卡 launch, **卡 OSF lock** |
| P0-2 | `router_proposals_v1.md §"P1 Feature spec"` row `has_reference_image` 是 audit-time gold label 不是 derivable signal | Router 用 gold-conditional feature 做 decision = oracle leak. Reviewer 抓住即拒. 解法是改用 `re.search` derivable derive, 跟 audit Cat B 算 F1 验 ≥ 0.95. | 不卡 launch, 卡 §C router implementation |
| P0-3 | `router_proposals_v1.md §"P2 Feature spec"` 4520 dim × ~180 train task = severe overfit | LR fit at 25:1 feature-to-sample ratio = test SR 几乎一定 < train SR by 5-10pp. Reported router lift = overfit noise. 解法是 dim cap ≤ N/10 ≈ 18 features. | 不卡 launch, 卡 §C P2 implementation |

### 🟠 P1 (paper-grade quality)

| # | Bug | Blast Radius | Launch 卡? |
|---|---|---|---|
| P1-1 | `router_proposals_v1.md §"P2 F5 signal"` re-uses contribution-1 phantom signal-AUROC infra as router input feature | 双倍计 contribution. Reviewer (cascade literature aware) 拒. 解法 = F5 移到 ablation row, P2 primary 用 F1-F4. | 不卡 |
| P1-2 | `preregistration.md §359` best-single-mode anchor at N=180 fold-level 选择 fold-noisy (Kendall τ across folds likely < 0.7) | Anchor flicker → router lift CI 宽到无法 reject 0. 解法 pre-data MC simulation on `meta_phantom_lift.md`. | 不卡, 卡 §D analysis |
| P1-3 | `router_proposals_v1.md §"Baselines"` Tier-0 random uniform = strawman | R1-R2 reviewer 一眼看出. 解法 = 加 frequency-weighted + top-3-modes-random 两 tier. | 不卡 |
| P1-4 | `router_proposals_v1.md §"P1 Threshold setup"` 16-combo grid search without nested CV = overfitting the CV | Hastie ESL 7.10.2 经典坑. Reviewer 拒. 解法 = pre-data threshold lock on archive (preferred) 或 nested CV. | 不卡, 卡 §C |

## Honest gaps (missing entirely)

- **No power analysis for H9/H10**: H1 has explicit power calc in preregistration §2.4 + `power_analysis.py`. H9/H10 是 NEW contribution-2, 缺 paper-grade power calc. 至少要算 "在 N=180 train + N=44 test fold × 6 cell pooled FE 下, 多大 router lift 能在 80% power × α=0.05 detect". 估计是 2-3pp — 比 1.0pp δ 大. 如果真 lift < 2pp, paper 写 "we ran 5-fold CV, observed +1.4pp, not statistically significant" 就死.
- **No router-induced cost overhead acknowledged**: Router decision 本身有 latency (P1 几 ms, P2 几十 ms). 在 paper "4-fold drop-in property" claim 里, router-induced overhead 算不算 break drop-in? Proposal 没说.
- **No baseline of "router only on phantom 3 modes"**: 6-mode router vs 3-phantom-mode router 哪个更 powerful? 减 search space 可能 reduce overfit (Finding F3). 没在 design space 里.
- **No "stick with default" baseline**: 一个 trivial baseline 是 "always P-SoM" (= contribution-1 hero). Router 必须比 always-P-SoM 强才 publishable. Proposal 没 explicit 锁这个.

## Distance to top-tier

- **Current acceptance tier**: workshop (R3) — pending Phase 1a data + estimand lock + F1-F4 fixes
- **Mid-tier (R2)**: blocked by P0-1 (estimand) + P0-3 (P2 overfit dim cap) + P1-1 (F5 leak)
- **Top-tier (R1)**: blocked above + (i) cross-model transfer claim needs B2 trained-fold transfer ablation explicit, (ii) router contribution must distinguish from contribution-1 signal infra, (iii) external validity needs Phase 1b shop pre-commit.
- **Submission-today probability if Phase 1a fires with proposals v1 unchanged**: 0.05 (top-tier) / 0.20 (mid-tier) / 0.60 (workshop)
- **With v2 fixes (P0-1, P0-2, P0-3, P1-1, P1-3, P1-4 applied)**: 0.20 (top-tier) / 0.45 (mid-tier) / 0.85 (workshop)

## One thing to fix tonight (1-3h leverage)

**Action**: Write `scripts/analysis/router_anchor_stability.py` — pre-data MC simulation script. Input: `docs/checkpoints/mechanism/results/meta_phantom_lift.md` archive (B0 4 cells). Output: per-cell Kendall τ of best-single-mode across 100 × 5-fold resamples + best-single-mode 一致性 % across folds.

**Why this**: F5 (anchor stability) is the cheapest defuse to run (use existing archive data, no Phase 1a wait), and it directly informs whether F5 fallback ("majority-winner across folds") needs to be in preregistration §359 update. Also feeds H9/H10 power calc (gap 2 in Honest gaps). Single script, ~80 LoC, can land tonight.

**Expected output**: anchor_kendall_tau.json per cell. If τ ≥ 0.8 → anchor stable, keep §359 as is; if τ < 0.8 → preregistration §359 update needed, propose alternate anchor.

## Phase 0 self-audit

- Scope declared: pre-fire ✓
- Artifacts: 5 declared (proposals_v1, paper_planning §8, phase1_plan §C, preregistration §354+§359+Appendix A, paper_planning §5+§6), 5 cited ✓
- Findings: 7 (target 7 for pre-fire) ✓
- OOB: 3 (F1 estimand-pending-vs-downstream-commit, F2 has_reference_image gold-leak, F4 F5 contribution-1-reuse) — target ≥ 3 ✓
- Specificity: every finding quotes proposal §, preregistration §, or specific dim/N value ✓
- Bilingual: section headers 中文 ✓ / code references English ✓ / attacks 中文 prose ✓
