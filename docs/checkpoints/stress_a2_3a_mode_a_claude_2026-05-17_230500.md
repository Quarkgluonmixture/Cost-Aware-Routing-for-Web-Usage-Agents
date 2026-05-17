# /stress A2.3a — Mode A (Claude self-audit) — Statistical design (power + sample)

**Scope**: pre-fire (advisor-blocking design-layer gate per phase1_plan §A2 + L317)
**Artifacts**:
- `scripts/analysis/power_analysis.py` (180 LOC)
- `docs/checkpoints/pre_run/preregistration.md` §2.4 (L321-329 power acknowledgment), §2.5 (L330-350 H1 PASS/FAIL flow), §3 (L368-405 multiple-comparison family), §4 (L408-441 locked analysis rows incl. B8/B9), Appendix A (L578-602)
- Memory `feedback_kofn_transparency_only.md` (K-of-N reclassification reference)
- Empirical verification: 3-baseline-SR spot-check of `power_analysis.py` output + FE-pool gate power recompute

**Mode A finding count**: 10 (≥7 pre-fire target ✓), OOB count: 5 (≥3 target ✓)
**Persona**: stats methodologist with paired-binary / meta-analysis / TOST implementation experience

---

## Verdict

**Phase 1a fire-blocking risk: HIGH on statistical-design substrate**. The preregistration's PRIMARY gate (H1 = one-sided FE superiority H0: θ_FE ≤ 1.0pp at α=0.05) is **48.3% powered at the observed +2.34pp pooled drop-one** — well below conventional 80%. The supporting power_analysis.py script has 3 distinct mathematical errors (1-sample SE used as paired-McNemar SE; two-sided z_α for a one-sided gate; inconsistent SE convention between MDE + power functions) that, when corrected, REDUCE rather than increase the reported power. Phase 1a could fire $thousands of A100 compute + still fail to reject H0 ~52% of the time due to **power, not real effect being null**. Reviewer will catch this on first read.

---

## Strong claims (survive attack)

1. **K-of-N retirement as gate is principled** (prereg §4 L441 + memory `feedback_kofn_transparency_only.md`). Power analysis correctly shows K-of-N would be dysfunctional at observed effect sizes (per-cell power ≈ 0.30 at 1-3pp); retiring it from gate to transparency-only is sound stats. Cohen's h + paired bootstrap CI + percentile (vs BCa for bounded proportions at small N) are all defensible methodology choices.

2. **Estimand-first design discipline** (Appendix A 2026-05-14 decision "3A"). The shift from random-effects DL/REML → fixed-effects inverse-variance over 6 planned cells correctly **dissolves** the k<10 fragility (Veroniki et al. 2016 τ²-bias, IntHout et al. 2014 anti-conservative RE Wald CI). The 6 cells ARE the design, not a population sample → no τ² in estimand → no estimator fragility from estimating τ² at small k. This is methodologically clean.

3. **N/A task exclusion at task-load time** (§4 L427) is the correct post-§139.8 architecture. Excluding 73 N/A tasks (5.3% of 1390) at selection, not as post-hoc denominator drop, avoids the silent-bias-via-fp-filter trap. WONDERBREAD precedent cited correctly.

---

## Weak claims — principled methodology errors (out-of-box first)

### Finding 1 — `mde_paired_binary` uses 1-sample Wald SE for paired McNemar test [P0 — OOB]

**Claim** — `power_analysis.py:43-59` docstring claims "paired-design McNemar-equivalent normal approximation" and prereg §2.4 cites the resulting MDE numbers as authoritative.

**代码现实** — `scripts/analysis/power_analysis.py:54-58`:
```python
z_alpha = 1.96   # two-sided α=0.05
z_beta = 0.842
p = baseline_sr
se_paired = math.sqrt(2 * p * (1 - p) / n)
mde = (z_alpha + z_beta) * se_paired / math.sqrt(2)
```
Algebra: `sqrt(2p(1-p)/n) / sqrt(2)` = `sqrt(p(1-p)/n)` — this is the **1-sample Wald SE for a single proportion at sample N**.

**攻击** — McNemar paired-binary test的 SE 取决于 discordant-pair proportion πD = π10 + π01,**不是** marginal proportion p。πD 可以从 0 (perfect agreement) 到 2×min(p, 1-p) (perfect disagreement) 任意取值,具体取决于两 mode 之间 task-level outcome 的相关性 ρ。脚本隐含假设 πD ≈ p(1-p),对应 **ρ=0 (零相关)** — 这是 paired test 的最坏情况(paired test 的统计 power 收益来自 ρ>0)。论文 prose claim "paired-design McNemar-equivalent" 但实际公式既非 paired (没有 πD), 也非 two-sample (差 sqrt(2) 因子) — 是 1-sample Wald SE。**Reviewer 头一遍就会问: "你们的 πD 假设是什么? 凭什么用 marginal p(1-p)?"** — 答不上来。

**Defuse** — 改写为正确 McNemar SE:
```python
# Conservative πD assumption (worst case 2*min(p, 1-p))
pi_D = 2 * min(p, 1-p)
se_mcnemar = math.sqrt(pi_D / n)
# Or use observed-ρ-conditional: pi_D = p_marginal*(1-rho) + delta correction
```
然后在 prereg §2.4 加 sentence: "Assumed discordant-pair proportion πD = p(1-p), corresponding to zero between-mode correlation — a conservative bound that under-states paired-test power at realistic ρ > 0."

**Effort** — 30 min code fix + 30 min prose update in §2.4 + 1-2h advisor sync to confirm πD assumption framing

**Confidence** — high (math is unambiguous)

---

### Finding 2 — PRIMARY FE-pool gate has 48.3% power at observed effect [P0 — OOB]

**Claim** — Prereg §2.5 L336-337 locks H1 PASS as "one-sided FE superiority test z = (θ_FE − 1.0) / SE_FE, p = 1 − Φ(z); p < 0.05 → H1 PASSES". Prereg §6 L509 lists this as the PRIMARY paper-1 gate. Prereg §2.4 L327 claims "fixed-effects estimand is the mitigation" for per-cell power ≈ 0.30.

**代码现实** — `power_analysis.py` (full script) computes ONLY per-cell power. **No FE-pool power calculation anywhere**. Empirical recompute (this audit, 2026-05-17):
```python
# 6 cells: cls × B0/B1/B2 (n=224 post-exclusion), red × B0/B1/B2 (n=205)
p = 0.10  # prereg-cited realistic adjusted-SR
se_cls = sqrt(0.10*0.90/224) = 0.0200  # 2.00pp per cell
se_red = sqrt(0.10*0.90/205) = 0.0210  # 2.10pp per cell
SE_FE = sqrt(1 / (3/se_cls² + 3/se_red²)) = 0.0084 = 0.84pp
# 80% power one-sided gate (H0: θ ≤ 1.0pp, α=0.05):
MDE = 1.0 + (1.645 + 0.842) × 0.84pp = 3.09pp
# Power at observed +2.34pp pooled drop-one (from meta_phantom_lift.md):
z_obs = (2.34 - 1.0) / 0.84 = 1.60
power = 1 - Φ(1.645 - 1.60) = 1 - Φ(0.05) = 1 - 0.520 = 0.480
```

**攻击** — 整个 prereg 的 PRIMARY gate **observed effect 上 power 48% — 不到一半**。Prereg §2.4 "FE pooling = mitigation" 的论断 **没有任何 power 计算支撑**;实际算下来 FE pool 在 +2.34pp observed effect 上 **比抛硬币好一点点**。Phase 1a 烧 36 conditions × 235B model API + 4B local A100 18-day 跑下来,有 **52% 概率 fail to reject H0 仅因 power 不足**, 不是真效应不存在。**Reviewer 一句话击毙**: "你们的 hero gate 在 reported effect 上 power < 50%, 数据再 clean 也没意义 — 这是 design-stage failure, not data-stage failure". Workshop reviewer 可能放过; EMNLP main / NeurIPS reviewer-3 必杀。

**Defuse** — 选 (a) (b) (c) 之一:
- (a) **降 δ threshold**: δ=0.5pp → 80% power 在 observed +2.34pp 上达到(but 0.5pp ≈ 1 task in N=224 — sampling noise floor, advisor 应该反对)
- (b) **加 Phase 1b shop site提前**: 9 cells × n=435 → SE_FE → 0.55pp, 80% power MDE = 2.37pp ≈ observed effect (但 shop 需 +18 conditions extra wallclock)
- (c) **公开承认 design under-powered**, paper §3 prose disclose "power=48% at observed effect; gate-pass conclusion is fragile;独立 replication strongly recommended"。Reviewer-honest path,但 hero claim 自废武功

**Effort** — (a) 1h prose; (b) 5-7 day A100 wallclock + 1h prereg amendment; (c) 2h paper §3 prose + 1h §8 limitations expansion

**Confidence** — very high (empirical recompute matches paired-binary FE-pool standard formula)

---

### Finding 3 — One-sided vs two-sided α confusion [P0 — OOB]

**Claim** — Prereg §2.5 L336 specifies "**one-sided** superiority test: z = (θ_FE − 1.0) / SE_FE, p = 1 − Φ(z)" — one-tailed at α=0.05 ⇒ z_α = 1.645.

**代码现实** — `power_analysis.py:54`:
```python
z_alpha = 1.96   # two-sided α=0.05
```
Per-cell MDE 和 per_cell_power_at_effect (L132-141) 都用 1.96。

**攻击** — Script 算的是 **两侧** test 的 MDE,但 prereg lock 的 gate 是 **单侧** test。两侧 α=0.05 ↔ 单侧 α=0.025; 单侧 α=0.05 用 z=1.645,**MDE 应小 19%** (= 1.96/1.645 比例)。Script reported cls MDE @ baseline=0.10 = 5.50pp; 单侧 corrected = 4.61pp。**§2.4 acknowledgment 引用的 "5-7pp 在 8-15% adjusted-SR" 都是 inflated 19%**;实际数字 reviewer 重算时会发现"哎你们说 5-7pp 我算出来 4-6pp,你们的 script 跟你们的 prereg gate 不一致"。**Cross-doc lock 失败**(prereg lock 跟脚本 disagreement = OSF audit trail integrity 破裂)。

**Defuse** — `power_analysis.py:54`:
```python
# z_alpha = 1.96  # two-sided α=0.05  ← WRONG for prereg one-sided gate
z_alpha = 1.645  # one-sided α=0.05 (matches prereg §2.5 one-sided FE gate)
```
重跑 §2.4 数字 + 重新 advisor confirm (one-sided framing 是 advisor 上次 sign-off 的 implicit 假设,需 explicit re-confirm)。

**Effort** — 5 min code fix + 1h §2.4 prose re-derivation + 1h advisor 单侧/双侧 sign-off email

**Confidence** — very high (prereg §2.5 line 336 explicitly says "one-sided")

---

### Finding 4 — §2.4 k=6 power TODO open 3+ days, blocking lock [P0]

**Claim** — Prereg L194 milestone "`preregistration.md` status `draft → locked`" depends on §A2.3a/b/c 全勾 + advisor confirm. §2.4 L326 has open TODO: `⏳ TODO (advisor lock pending): re-derive per-cell power at k=6 with B2 cells included.`

**代码现实** — TODO logged 2026-05-14 (B2 addition date), today is 2026-05-17 — **3 days open**. The "k=4 vintage" numbers continue to be cited authoritatively in §2.4 + §4 K-of-N row L441 + §6 commit lock list L518-522. **`docs/analysis/cross_sites/power_analysis.md`** (referenced in §4 K-of-N row) does not exist in `find docs/analysis -name "power_analysis*"` (verified at audit time — only `scripts/analysis/power_analysis.py` exists; the prereg references an output `.md` file that is not in repo).

**攻击** — Prereg 状态 `draft → locked` 卡在 §2.4 的 ⏳ TODO 已 3 天。Prereg 自己引用了一个 **不存在的** `docs/analysis/cross_sites/power_analysis.md` 作为权威源 (§4 row B9 L441 + §2.4 多处 reference)。Reviewer 沿着 paper §3 → prereg §4 row B9 → `docs/analysis/cross_sites/power_analysis.md` 链找,**死链**。OSF audit trail 完全破裂。

**Defuse** — 立即跑:
```bash
.venv/bin/python3 scripts/analysis/power_analysis.py --baseline-sr 0.10 \
  --output docs/analysis/cross_sites/power_analysis.md
git add docs/analysis/cross_sites/power_analysis.md
```
然后 §2.4 TODO 转 `⏳ FILLED 2026-05-17: per-cell power at k=6 — [link to power_analysis.md]`。**Caveat**: 这只 fix 死链问题,不 fix Finding 1/2/3 的 underlying methodology issues。

**Effort** — 5 min script run + 30 min prereg §2.4 update + advisor sign-off pending

**Confidence** — high

---

### Finding 5 — N pre-exclusion hardcoded; operational N is post-exclusion [P1]

**Claim** — `power_analysis.py:97-101` comment: `"intentionally the pre-exclusion design N — this is a pre-registered design-time power computation and the preregistration power section is locked to these numbers"`. Sites hardcoded `cls 234 / red 210 / shop 466`.

**代码现实** — Post-§139.8 retirement (Appendix A 2026-05-14 entry), N/A tasks excluded at task-load time → `scored_task_count` from `p79/experiment/analysis.py` returns cls 224 / red 205 / shop 435. Operational paired bootstrap CI在 224/205/435 上跑,不是 234/210/466。

**攻击** — 脚本注释自我 justify "would desync the committed prereg" — 但这是 **循环论证**: locks 应该正确,而不是 preserved-because-locked。Reviewer 检查 OSF 审计 trail 时会问"你们 prereg 锁 N=234,但实际数据集 N=224 (因 N/A 排除),power numbers 应该用哪个?"答案应该是 **operational N = 224** (实际跑的样本)。**MDE shift 约 2%** (sqrt(234/224) ≈ 1.022) — numerically small,但 OSF 审计 reviewer 会扣分: "prereg ↔ operational misalignment without explicit reconciliation"。

**Defuse** — 两选一:
- (a) Update L97-101 to use post-§139.8 scored counts (224/205/435), update §2.4 numbers
- (b) Keep pre-exclusion N as prereg-locked but add explicit reconciliation row to §4 + §2.4: "Power computed on pre-exclusion N=234/210/466; operational denominators are 224/205/435 post-N/A-exclusion; MDE delta < 2.5% — see Appendix X reconciliation table."

**Effort** — (a) 30 min; (b) 15 min prose addition

**Confidence** — high

---

### Finding 6 — TOST equivalence δ=1.0pp power not computed [P1 — OOB]

**Claim** — Prereg §4 L419 calls TOST equivalence δ=1.0pp "the tightest test"; §2.4 L327 calls FE pooling "the mitigation" for sub-3pp effects.

**代码现实** — `power_analysis.py` 整个 script 没有 TOST equivalence 的 power 计算。TOST = Two One-Sided Tests, requires BOTH CI ends inside [-δ, +δ] interval (Schuirmann 1987)。Required sample size scales roughly as `n ≥ (z_α + z_β)² × σ² / δ²` per side。在 SR=0.30, paired-binary 上,δ=1.0pp 的 TOST 需要 **N ≈ 5000+ per cell** at 80% equivalence-power (远超 cls 234 / red 210)。

**攻击** — Prereg 把 TOST 当成 fallback / 兜底 ("tightest test, relies on cross-cell pooling for adequate CI width") — 但 **从来没人算过 TOST 在实际 N 上的 power**。即使 pooled (Σ N = 234+210 = 444 per row), 远低于 TOST 需要的 5000+ per cell。如果 Phase 1a FE-pool gate 没 pass (Finding 2 显示 52% 概率), paper fallback 到 TOST → reviewer "你们 TOST equivalence 算的 power 是多少?" → 没算 → "那这个 fallback 也是 fake fallback"。**Hero claim 的 escape route 不 work**。

**Defuse** — Compute TOST power explicitly in `power_analysis.py`:
```python
def tost_power(n, p, delta, alpha=0.05, beta=0.20):
    # TOST requires both CI ends inside [-delta, +delta]
    # n_required for 80% TOST power at delta:
    from scipy.stats import norm
    z_a = norm.ppf(1 - alpha)  # one-sided per TOST end
    z_b = norm.ppf(1 - beta)
    se = sqrt(2*p*(1-p)/n)  # assumes paired McNemar approximation
    # TOST power: P(|theta_hat| < delta - z_a*se) under H1: theta = 0
    # ≈ 2*Phi((delta - z_a*se)/se) - 1
    ...
```
然后 §4 row "FP filter sensitivity" → "TOST equivalence" 加 power column。如果 < 80% → §2.4 honest disclose "TOST equivalence is informational, not a power-validated fallback".

**Effort** — 1-2h coding + 30 min prereg prose update + 1h advisor δ discussion

**Confidence** — high (TOST sample-size formulas are textbook Schuirmann 1987 + Walker+Nowacki 2011)

---

### Finding 7 — Inconsistent SE convention between mde + per_cell_power functions [P1]

**Claim** — Script docstring promises consistent "paired-design McNemar-equivalent" framework.

**代码现实** — Two SE conventions internally:
- `mde_paired_binary` L57-58: `se_paired = sqrt(2p(1-p)/n); mde = (z_α+z_β) × se_paired / sqrt(2)` ⇒ effective SE = `sqrt(p(1-p)/n)` (1-sample)
- `per_cell_power_at_effect` L132-141: `se = sqrt(2p(1-p)/n); z_score_at_effect = (eff/100) × sqrt(2) / se` ⇒ effective SE = `sqrt(p(1-p)/n)` (also 1-sample, but via different algebra)

Two-step transformation **algebraically consistent** but obfuscates the test framing — a reader can't tell from the function bodies which test is being framed.

**攻击** — 两个 function 都隐含用 1-sample SE,但 **通过不同的 sqrt(2) 操作**到达同一结果。Reviewer 看 mde_paired_binary 以为是 two-sample test (因为有 `2p(1-p)`), 看 per_cell_power_at_effect 也以为是 two-sample test (同 `2p(1-p)`)。但**算出来的 power 跟 1-sample test 一致** (因 sqrt(2) 撤销)。**代码自己不知道在测什么**, reviewer 也看不出来。即使 Finding 1 的 root cause fix 了,这个 code clarity bug 单独存在 — code-review 头一眼就发现"这两 function 的 SE convention 不一样,但 magically 给出 consistent power numbers"。

**Defuse** — Refactor to share single `_se_function(n, p, test_type="paired_mcnemar")` helper:
```python
def paired_mcnemar_se(n, p, pi_D=None):
    """SE of paired McNemar difference. If pi_D None, use 2*min(p, 1-p) worst case."""
    if pi_D is None:
        pi_D = 2 * min(p, 1-p)
    return math.sqrt(pi_D / n)

def mde_paired_binary(n, p, alpha, beta, one_sided=True):
    z_a = 1.645 if one_sided else 1.96
    z_b = 0.842
    return (z_a + z_b) * paired_mcnemar_se(n, p)

def per_cell_power(n, p, effect_pp, alpha, one_sided=True):
    z_a = 1.645 if one_sided else 1.96
    se = paired_mcnemar_se(n, p)
    z_score = (effect_pp / 100) / se
    return 1 - phi(z_a - z_score)
```

**Effort** — 30 min refactor + 30 min test (output should remain numerically close after Finding 1+3 fixes)

**Confidence** — high

---

### Finding 8 — I² heterogeneity threshold rule treats I² as reliable at k=6 [P1 — OOB]

**Claim** — Prereg §4 row L440 ("Pooling estimator + heterogeneity pre-spec") locks I² thresholds: "I² < 25% = pooled FE meaningful; 25-75% = FE + per-cell forest; > 75% = §4 prose leads with per-cell forest, hook capped at R3". I² > 75% triggers a paper-hook downgrade (R1 → R3 cap per §2.5 L347).

**代码现实** — I² = (Q - df) / Q where Q is Cochran's heterogeneity statistic. Q is **known to be unreliable at small k** (Higgins & Thompson 2002; Veroniki et al. 2016 — same source prereg correctly cites for τ²). At k=6, sampling variance of Q is large → I² is itself a noisy point estimate with wide CI (often 0-99% in practice).

**攻击** — Prereg 自己 cite Veroniki et al. 2016 + IntHout et al. 2014 as rationale for retreating from random-effects to fixed-effects at k<10 (because τ² estimation is fragile)。**但同样的 fragility 适用于 I²** — I² 是 Q 的 deterministic function, Q 在 k=6 上 sampling noise 大 → I² 也是 noisy point estimate。**"I² > 75% caps at R3" rule 可能 spuriously fire** (sampling noise that pushed Q into "I²=80%" territory)。Prereg 把 τ² 当 fragile 但 I² 当 reliable — **methodologically inconsistent**。Reviewer-3 看到这个 inconsistency 一句话毙: "你们 retreat from random-effects 因为 τ² 在 k=4-6 fragile, 但 rely on I² threshold 来 cap hook claim — I² shares the same fragility, why is it OK here?"。

**Defuse** — 两选一:
- (a) Add CI on I² (Wald / Q-profile method) + only fire R3 cap if **lower bound** of I² CI > 75%。比较 conservative,不太可能 spuriously fire,但 cap rule 实际几乎从不触发(I² CI 在 k=6 太宽)
- (b) 撤掉 hard I² threshold rule,改用 narrative judgment: "I² > 50% triggers per-cell forest as primary display + FE pooled as secondary summary; no hard hook downgrade rule"。

**Effort** — (a) 2h code + 1h prose; (b) 30 min prose

**Confidence** — high (I² fragility at k<10 is well-documented in same literature prereg already cites)

---

### Finding 9 — `k_of_n_power` orphan code (K-of-N retired) [P1]

**Claim** — Prereg §4 L441 + script header L5-9 + script L147-151 all confirm K-of-N retired as gate.

**代码现实** — `power_analysis.py:62-74`:
```python
def k_of_n_power(n_cells: int, k_threshold: int, per_cell_power: float) -> float:
    """Family-wise power for K-of-N pass rule."""
    from math import comb
    p = per_cell_power
    return sum(comb(n_cells, k) * (p ** k) * ((1 - p) ** (n_cells - k))
               for k in range(k_threshold, n_cells + 1))
```
Function defined but never called in `main()`. Pure orphan code.

**攻击** — K-of-N retired as gate (decision "3A" 2026-05-14), 但 `k_of_n_power` function 仍在 script 里。**Future maintainer** 给 script 加 `--k-of-n-check` flag 时,会重新计算 retired K thresholds, ressurecting the retired decision rule by accident。**Paper-grade hygiene**: 死代码 + spec drift。

**Defuse** — Delete L62-74 + 加 ABOVE comment: `# K-of-N retired as gate per prereg §4 L441 (decision 3A 2026-05-14); K-of-N is now transparency-only count, no power computation needed.`

**Effort** — 5 min

**Confidence** — high (dead code is unambiguous)

---

### Finding 10 — Default baseline-SR=0.30 doesn't match observed prereg-cited 8-15% [P2]

**Claim** — `power_analysis.py:84` defaults to `--baseline-sr 0.30`.

**代码现实** — Prereg §2.4 L325 cites "observed adjusted-SR levels (8-15%)" as the realistic baseline. User running `.venv/bin/python3 scripts/analysis/power_analysis.py` (no args) gets MDE = 8.39pp on cls — overly pessimistic vs the prereg-cited realistic baseline 0.10 → 5.50pp.

**攻击** — Default 0.30 不匹配 prereg's own cited reality 0.10。Casual reader runs script no-args → 看到 inflated MDE → 误以为 design 更 underpowered than reality (虽然现在 reality 也很 underpowered per Finding 2)。**Reproducibility hygiene**: defaults should match the actual operational regime。

**Defuse** — `power_analysis.py:84`:
```python
p.add_argument("--baseline-sr", type=float, default=0.10,
               help="Baseline success rate (default 0.10 — matches prereg §2.4 observed adjusted-SR 8-15% lower bound)")
```

**Effort** — 5 min

**Confidence** — high

---

## Bug Table — A2.3a Mode A (Claude) findings

### 🔴 P0 (lock 前必须 fix)

| # | Bug | Blast Radius | Launch 卡? |
|---|---|---|---|
| P0-1-A* | `power_analysis.py:54-58` `mde_paired_binary` 用 1-sample Wald SE (`sqrt(p(1-p)/n)`),不是 paired McNemar SE (依赖 discordant πD) 也不是 two-sample SE。Docstring claims "paired McNemar-equivalent" 但实际既非 paired 也非 two-sample。 | Prereg §2.4 cited per-cell MDE numbers 都基于此公式 — 隐含 πD ≈ p(1-p) (ρ=0, paired test 最坏情况)。**Reviewer 一句话击毙: "你们的 πD 假设是什么? 凭什么用 marginal p(1-p)?"** OSF prereg lock 跟 actual computation framework drift,审计 trail 失效。 | 不卡 launch,卡 advisor sign-off (single-sided/two-sided + πD 假设需 explicit confirm) + 卡 paper §3 prose round 写法 |
| P0-2-A* | `power_analysis.py` 没有 FE-pool gate power 计算; **empirical recompute: PRIMARY H1 gate (FE 单侧 H0: θ≤1.0pp) 在 observed +2.34pp 上 power = 48.3%**, 远低于 conventional 80%。 | Phase 1a 烧 36 conditions × A100 18-day, **52% 概率 fail to reject H0 仅因 power 不足**,不是真效应不存在。Prereg §2.4 "FE pool = mitigation" rhetoric 无 power 计算支撑。Workshop reviewer 可能放过;EMNLP main / NeurIPS reviewer-3 必杀 (hero gate < 50% power = design-stage failure)。Mitigation: 降 δ (advisor 反对) / 加 Phase 1b (5-7 day extra wallclock) / honest disclose (hero 自废)。 | **YES — 启动前必须决定**: (a) 加 Phase 1b shop 提前到 Phase 1a 同跑 / (b) δ 改 0.5pp / (c) honest disclose under-powered。三选一都需 advisor sign-off + prereg amendment。 |
| P0-3-A* | `power_analysis.py:54` `z_alpha = 1.96` (两侧 α=0.05) 但 prereg §2.5 L336 specifies 单侧 FE 测试 (应 z_α = 1.645)。Per-cell MDE numbers in §2.4 acknowledgment **inflated by ~19%** (= 1.96/1.645)。 | Script ↔ prereg gate 不一致 = OSF audit trail integrity 破裂。Reviewer 重算时发现 script 跟 prereg lock disagreement → "你们 prereg 锁 5-7pp MDE 但 script 算出 4-6pp,哪个 authoritative?"。Cross-doc 一致性是 paper-grade lock 的底线。 | 不卡 launch,卡 OSF DOI lock (advisor 必须 explicit sign-off 单/双侧 framing) + 卡 §2.4 prose 重新 derive |
| P0-4-A | `preregistration.md:326` §2.4 ⏳ TODO "re-derive per-cell power at k=6" open 3+ days (since 2026-05-14)。Prereg references `docs/analysis/cross_sites/power_analysis.md` 但该文件 **不存在** (only `scripts/analysis/power_analysis.py` exists)。 | Prereg `draft → locked` 卡 §2.4 TODO + 死链 (`docs/analysis/cross_sites/power_analysis.md`)。Reviewer 沿 paper §3 → prereg §4 row B9 链找 → 404 file not found → OSF audit trail 完全破裂。 | 不卡 launch,卡 OSF DOI lock (advisor sign-off 等 §2.4 TODO 关闭)。死链 fix 5 min。 |

### 🟠 P1 (paper-grade quality)

| # | Bug | Blast Radius | Launch 卡? |
|---|---|---|---|
| P1-5-A | `power_analysis.py:97-101` 硬编码 pre-exclusion N (234/210/466),实际 scored N = 224/205/435 (post-§139.8 N/A 排除)。Script 注释 self-justify "would desync committed prereg" — 循环论证。 | MDE delta ~2.2% (numerically tiny),但 OSF reviewer 扣分: "prereg N ≠ operational N without explicit reconciliation"。Paper §3 prose 引用 prereg 数字 reviewer 重算时发现 1-2% 偏差。 | 不卡 launch,paper §3 prose 写时加 reconciliation row + footnote |
| P1-6-A* | `power_analysis.py` 没有 TOST equivalence δ=1.0pp 的 power 计算。Prereg §4 L419 + §2.4 L327 都把 TOST 当 "tightest test" / fallback,但 **从来没人算过 TOST 在实际 N 上的 power**。Schuirmann 公式估算需 N ≈ 5000+ per cell at 80% TOST power, 远超 cls 234 / red 210。 | Phase 1a FE-pool gate 没 pass (~52% 概率 per P0-2) → paper fallback 到 TOST → reviewer "你们 TOST equivalence 算的 power 是多少?" → 没算 → "这个 fallback 也是 fake fallback"。Hero claim escape route 不 work。 | 不卡 launch,卡 paper §3 写"fallback test" 部分时必须有 power number。Mitigation: 跑 TOST power 计算 + 如果 < 80% 在 §2.4 honest disclose "TOST is informational only, not a power-validated fallback"。 |
| P1-7-A | `mde_paired_binary` 和 `per_cell_power_at_effect` 用 不同的 SE convention algebra 但 magically 一致 1-sample SE。Code clarity bug: reader 不知道 script 在测 1-sample / 2-sample / paired。 | Code-review reviewer (paper-grade audit) 头一眼发现"两 function SE convention 不一样"。Refactor 到 shared `paired_mcnemar_se()` helper 让 framing explicit。 | 不卡 launch,paper-grade code release 时必须 refactor |
| P1-8-A* | Prereg §4 row L440 ("I² < 25% / 25-75% / > 75%" thresholds) 在 k=6 上 spuriously fire 风险高 — I² 跟 τ² 一样在 k<10 上 fragile (Veroniki/IntHout 已 cite for τ²,但 prereg 把 I² 当 reliable)。 | "I² > 75% → R3 cap" rule 可能 spuriously trigger 在 sampling-noise-driven I² estimate 上,paper hook 莫名其妙降级。Reviewer-3 一句话毙: "你们 retreat from RE 因为 τ² fragile, 但 rely on I² threshold cap, I² shares same fragility — methodologically inconsistent"。 | 不卡 launch,卡 advisor sign-off (I² CI lower-bound rule 或撤 hard threshold)。Phase 1a 数据 land 后才知道是否 spurious-fire。 |
| P1-9-A | `power_analysis.py:62-74` `k_of_n_power` function 仍存在,但 K-of-N retired as gate 2026-05-14。Pure orphan code, future maintainer 加 `--k-of-n-check` flag 时会 ressurect retired decision rule。 | Paper-grade code hygiene + spec drift。Phase 1a launch 不卡,但 paper release artifact (OSF code repo) 应 cleanup。 | 不卡 |

### 🟡 P2 (defer-able)

| # | Bug | Blast Radius | Launch 卡? |
|---|---|---|---|
| P2-10-A | `power_analysis.py:84` default `--baseline-sr 0.30` 不匹配 prereg §2.4 cited "observed adjusted-SR 8-15%"。User no-args 跑得 inflated MDE 8.39pp (cls)。 | Reproducibility hygiene。Casual reader 误以为 design 更 underpowered。Phase 1a launch 不卡。 | 不卡 |

---

## Honest gaps (missing not weak)

1. **No paired-McNemar SE assumption documentation** — prereg + script 都没说明 πD 假设。即使 fix Finding 1 + 7, 需 explicit "we assume πD = X based on pilot data Y / conservative bound Z"。
2. **No advisor witness on δ=1.0pp at observed effect feasibility** — prereg locks δ=1.0pp 时引用 archive +2.34pp drop-one (Decision "3A" 2026-05-14), 但没人算 "given +2.34pp observed, power to reject θ≤1.0pp = ?" 这次 audit Finding 2 第一次算 → 48.3%。Advisor 不知道 sign 的 δ=1.0pp 在 observed effect 上 power < 50%。
3. **No Phase 1a → Phase 1b power transition analysis** — Phase 1b 加 shop site (n=435 × 3 models = 3 cells) → 9 cells total。SE_FE → sqrt(1/(3/0.0200² + 3/0.0210² + 3/0.0144²)) ≈ 0.55pp → 80% power MDE = 2.37pp ≈ observed +2.34pp。**如果 Phase 1b 同跑, hero gate power 拉到 ~80%**, 解决 Finding 2。但 prereg 把 Phase 1b 当 deferred main-paper-only — should re-evaluate this scoping in light of Finding 2 numeric。
4. **No replication-only path** — paper §8 limitations 应该 disclose "design under-powered for hero gate at observed effect; independent replication strongly recommended" 但 §8 limitations 当前内容未 audit (Mode C scope)。
5. **No sensitivity analysis on δ choice** — δ=1.0pp 是 "Decision 3A" pick,但没 sensitivity: at δ=0.5pp / 1.5pp / 2.0pp 各自的 power numbers 是多少? Advisor 选 1.0pp 是基于什么 power tradeoff?

---

## Distance to top-tier

**Current tier**: **workshop-borderline / EMNLP-main-fail** on statistical-design substrate alone (before considering data).

**Specific blockers**:

1. **Finding 2 (FE-gate 48% power)** — single biggest blocker。If unfixed, reviewer-3 一句话 reject paper-grade venue。Unblock: (a) δ降到 0.5pp (advisor 反对) / (b) Phase 1b shop site 同跑 (5-7 day extra wallclock) / (c) honest disclose under-powered。
2. **Finding 1 + 3 (script ↔ prereg framework drift)** — OSF audit trail integrity 必修。Unblock: 1h code fix + 1h prereg prose update + 1h advisor sign-off。
3. **Finding 4 (k=6 TODO + 死链)** — prereg lock blocker。Unblock: 5 min script run + 30 min prose update。
4. **Finding 6 (TOST 没 power 算)** — paper §3 escape-route 不 work。Unblock: 1-2h coding + 1h advisor δ sign-off。

**Unblock plan stitched**: ~2 days focused work (Finding 1+3+4+9 ≈ 4h code+prose; Finding 6 ≈ 3h; Finding 2 needs strategic decision + advisor sync → 1-3 day depending on (a)/(b)/(c) choice)。

**Submission-today probability**: **0.05-0.10 NeurIPS / 0.15-0.20 EMNLP main / 0.40-0.50 workshop** (workshop-grade 仅因 reviewer 容忍度,统计 design 仍弱)。

---

## One thing to fix tonight (1-3h leverage)

**Finding 2 (FE-gate 48% power at observed effect)** 是 single highest-impact action,但需 advisor strategic decision 不可单方面 fix。

**Tonight feasible (1-3h Claude solo)**: 联合 fix **Finding 1 + 3 + 4 + 9** — 全是 code-level + 已 known issue:

```bash
# 1. Fix Finding 4 dead link (5 min)
.venv/bin/python3 scripts/analysis/power_analysis.py --baseline-sr 0.10 \
  --output docs/analysis/cross_sites/power_analysis.md
git add docs/analysis/cross_sites/power_analysis.md

# 2. Fix Finding 3 z_alpha (5 min)
# Edit power_analysis.py:54: z_alpha = 1.645 (one-sided per prereg §2.5)

# 3. Fix Finding 1 SE formula (30 min)
# Add paired_mcnemar_se(n, p, pi_D=None) helper, refactor mde + per_cell_power

# 4. Delete Finding 9 orphan code (5 min)
# Remove k_of_n_power L62-74

# 5. Re-run + update §2.4 numbers (30 min)
# Update prereg §2.4 with new MDE numbers at one-sided α + corrected SE

# 6. Empirical FE-pool power addition (1h)
# Add fe_pool_power() function showing 48% at observed effect → forces F2 surface in prereg
```

之后 commit + chronicle: "stress A2.3a Mode A: 4 code-layer fixes + FE-pool power surfacing — Finding 2 (强 hero gate underpowered) escalated to advisor sync"。

---

## Phase 0 self-audit

- **Scope declared**: pre-fire ✓
- **Artifacts**: 5 declared, 5 cited ✓ (power_analysis.py + preregistration.md §2.4/§3/§4/Appendix A + memory)
- **Findings**: 10 (target 7 pre-fire) ✓ Δ=+3
- **OOB**: 5 (F1/F2/F3/F6/F8 — target 3) ✓ Δ=+2
- **Specificity**: all findings quote file:line + specific numbers (e.g. F2 cites empirical SE_FE=0.84pp, power=48.3%) ✓
- **Length/byte/cost specificity**: ✓ — F2 quotes empirical recompute on disk; F4 quotes verified-non-existent file path; F5 quotes scored count numbers from §4 row source

**Self-flag**: 我没 audit `paper_drafts/section3_definition.md` 跟 paper prose 怎么 cite power claims — 这是 Mode C gemini 的 scope (paper drafts), 但如果 paper §3 prose has additional independent power claims 跟 prereg §2.4 drift, Mode A 会 miss。Trust Mode C 覆盖。

**Self-flag**: I did not audit `aggregate_phantom_meta.py` 实际是否 implement FE inverse-variance pooling (vs default to DL random-effects) — 这是 Mode B codex 的 scope。If aggregator code disagrees with prereg lock B8 row, Mode A misses。Trust Mode B 覆盖。
