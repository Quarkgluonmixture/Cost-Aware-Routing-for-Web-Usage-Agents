---
type: analysis
status: exploratory
created: 2026-09-09
purpose: can a router that runs the cheap mode's step-0 inference first (one-step lookahead), a contextual bandit, or a 3-arm collapse do what the paper's five constructions could not — evaluated entirely on the 18,294 logged episodes, no new fire
scope_warning: 11 cells, canonical run per (cell,mode) chosen by the pilot's rule (largest run, tie -> earliest; 13,523/13,523 agree with full_table2.json). Effect sizes are read against the rerun band where one exists (5 cells); the other 6 cells cannot resolve a pp-level difference. cls_B5 is self-routing (router and agent both GPT-5.6) in the LLM-router rows only; the analyses here use no LLM router. All policy numbers are offline splices at the ONE splice-legal point (before any action), so the §409.3 sequential-state caveat does NOT apply to the lookahead rows; it still applies to nothing here because no post-action escalation is evaluated.
producer: results/router_llm_pilot_20260909/scripts/{extract_step0.py,lookahead_eval.py,bandit_replay.py} (results/ is gitignored; raw outputs in results/router_llm_pilot_20260909/lookahead/)
---

# One-step lookahead, contextual bandit, 3-arm collapse — three more routing constructions, all on logged data

Regenerate:
```bash
P=results/router_llm_pilot_20260909; L=$P/lookahead
.venv/bin/python3 $P/scripts/extract_step0.py $L/step0.jsonl          # 21,291 episodes (canonical + 18 replicate arms), ~20 s on 16 cores
.venv/bin/python3 $P/scripts/lookahead_eval.py $L $L/lookahead_results.json   # ~25 min
.venv/bin/python3 $P/scripts/bandit_replay.py $L 0.0; ... $L 0.5
```

## 0. The question and why it is not a reskin of the paper's cascade

The paper's cascade (§5, policy 4) escalates on **episode-aggregated** confidence of a full cheap run, so (a) it pays the whole cheap episode before deciding and (b) its rich-arm outcome is an offline splice onto a site the cheap arm has already touched (§409.3). A one-step lookahead decides **after the cheap model's step-0 inference and before any action**: the page state is the reset start state for every mode, so splicing the rich arm's logged episode is exact, and the peek costs a single inference.

| cell | step-0 inference as share of a full episode's cost (median, by mode) |
|---|---|
| cls_B0 | dom 8.0% · vision 6.9% · som 12.0% · ptext 7.4% · pprompt 9.2% · psom 6.9% |
| cls_B1 / cls_B2 / red_B1 / red_B2 / shop_B1 | 3.0–4.5% (every mode) |
| cls_B5 | dom 7.1% · som 8.6% · vision 3.1% |
| red_B0 / red_B0_WA / red_B1_WA / shop_B0 | 3.2–8.6% |

So structurally it is a different construction with a strictly better cost profile and a clean estimand. The three tests below ask whether the *information* is there.

## 1. Paired realization test — is the rerun flip visible at step 0? **No.**

18 same-condition replicate pairs (`noise_floor_inventory.json` CLEAN_PAIRS), 326 tasks whose outcome flipped between the two runs. For each flip task, compare the step-0 signal of the run that succeeded with the run that failed (same task, same mode, same model — task difficulty cancels exactly).

| step-0 signal | succ > fail | fail > succ | ties | P(succ > fail) | sign-test p |
|---|---:|---:|---:|---:|---:|
| mean_logprob | 140 | 154 | 3 | 0.476 | 0.45 |
| min_logprob | 140 | 154 | 3 | 0.476 | 0.45 |
| mean_margin | 150 | 144 | 3 | 0.510 | 0.77 |
| min_margin | 134 | 150 | 13 | 0.472 | 0.37 |
| verbalized | 57 | 52 | 107 | 0.523 | 0.70 |
| thought length | 154 | 150 | 22 | 0.507 | 0.86 |
| output tokens | 156 | 149 | 21 | 0.511 | 0.73 |
| inference latency | 170 | 156 | 0 | 0.521 | 0.47 |
| mean_entropy (B1 pairs only) | 5 | 10 | 3 | 0.333 | 0.30 |

Nothing the cheap model emits at step 0 separates the realization that will succeed from the one that will fail. The nondeterminism *starts* at step 0 (same-mode replicates take the identical step-0 action in only 24–75% of tasks on B0, land on the same URL in 50–90%), but it does not *decide* there. This is the mechanism behind §6's "the rows a router must learn from are the rows that flip between reruns": the flip is decided after the router has to decide.

## 2. Incremental AUROC — what does the model's first output add over model-free pre-flight features? **≈ 0.**

Three nested feature tiers from the cheap mode's own step-0 record: **OBS0** = page/task stats with no model call (dom_complexity, text length, mark count, has reference image, intent length + 18 intent keywords — the §457 abstention feature family); **+MODEL0** = the cheap model's step-0 output (4–6 confidence fields, verbalized, thought length/6 regex flags, output tokens, latency, action type, element-id presence, parse validity); **+POST0** = result of executing the step-0 action (action_success, page_changed, url changed — not splice-legal for upgrade, reported as an information bound). Labels are all dense: **self** = cheap succeeds, **any** = some mode succeeds, **rich** = rich mode succeeds, **upg** = cheap fails ∧ rich succeeds. L2 logistic regression, 5-fold task-level CV (stratified) and GroupKFold by `intent_template_id`; shuffle-null = permute training labels, 30 reps.

Median out-of-fold AUROC over (cell, cheap-mode) rows; in brackets the median gap over the shuffle-null median:

| label | rows | OBS0 | +MODEL0 | +POST0 | Δ(MODEL0−OBS0) | share Δ>0 |
|---|---:|---|---|---|---:|---:|
| self (strat) | 55 | 0.720 (+0.215) | 0.723 (+0.208) | 0.721 (+0.208) | −0.007 | 45% |
| any (strat) | 62 | 0.723 (+0.217) | 0.693 (+0.189) | 0.692 (+0.183) | −0.021 | 29% |
| rich (strat) | 27 | 0.685 (+0.179) | 0.672 (+0.160) | 0.674 (+0.167) | +0.005 | 59% |
| **upg (strat)** | 26 | 0.604 (+0.091) | 0.617 (+0.132) | 0.639 (+0.134) | +0.021 | 58% |
| self (group) | 55 | 0.654 (+0.186) | 0.624 (+0.139) | 0.633 (+0.145) | −0.010 | 47% |
| any (group) | 62 | 0.657 (+0.167) | 0.626 (+0.136) | 0.623 (+0.139) | −0.031 | 31% |
| rich (group) | 27 | 0.594 (+0.121) | 0.587 (+0.119) | 0.590 (+0.101) | −0.011 | 41% |
| upg (group) | 26 | 0.541 (+0.107) | 0.578 (+0.114) | 0.578 (+0.123) | −0.004 | 46% |

The routing target `upg` has *some* signal (MODEL0 beats the shuffle-null 95th percentile in 16/26 pairs under stratified CV), but the model's own output contributes a median +0.021 to it, and nothing to the others; under GroupKFold every increment is ≤ 0. Raw single-signal step-0 AUROC for own success (no fitting, median over 11 cells × modes): verbalized 0.562, mean_logprob 0.547, mean_margin 0.574, thought length 0.543 — against the 0.53–0.85 the *episode-aggregated* versions of the same signals reach in E3.

**Why the thought carries nothing new:** TF-IDF (1–2-gram) on the full step-0 thought text vs TF-IDF on the task intent alone, same CV, 33 (cell, cheap, label) rows: intent-TF-IDF ≥ thought-TF-IDF in 29/33 (e.g. cls_B0·dom·self 0.789 vs 0.577; red_B0·vision·upg:dom 0.785 vs 0.752). The step-0 thought is a paraphrase of the intent, and the intent is free. (Both numbers are steady-state-regime, template siblings across folds; both fall under GroupKFold.)

## 3. Offline policy — escalate top-f by out-of-fold P(upg), five controls

27 (cheap → rich) pairs = {cheapest-cost mode, dom, vision} → {best-SR mode, som}. Cost of an escalated task = cheap step-0 inference + rich full episode. Controls: always-cheap, always-rich, random escalation of the same size (exact expectation), oracle (escalate exactly the `upg` tasks), and **cheap-restart** (same f, but the escalated task gets the cheap arm's *replicate* run instead of the rich arm — available where a replicate exists).

- **Pareto-beats always-rich (SR ≥ and cost ≤) at some f:** 11/27 pairs (MODEL0). Of the 11, 9 are pairs where the rich arm is no better than the cheap one (ties/reversals: red_B0 dom→som 14.6=14.6, red_B0_WA dom→som, red_B1_WA dom→som, shop_B1 dom→som, shop_B0 vision→som 15.2 vs 14.7 at +31% cost) or SR is 1–4% (cls_B2, red_B2), or n=104 without a band (red_B1_WA vision→dom: 16.3–17.3% at −17…−30% cost, +6.35pp over random). The one non-degenerate win on a banded cell: **cls_B0 vision→som at f=0.15: 27.7% / $0.0659 vs always-som 27.2% / $0.0724** — +0.5pp SR against a rerun band of 4.5–7.6pp.
- **Gain over random escalation of the same size:** mean +0.74 / +0.90 / +1.04pp at f = 0.1 / 0.2 / 0.3 (MODEL0), positive in 63–74% of pairs. Same order as the paper's cascade (+1.1 to +2.1pp).
- **Cheap-restart control** (cls_B0 dom/vision/som, red_B0 five arms, cls_B5 dom): restarting the cheap arm on the same escalated set never beats continuing it (restart SR ≤ always-cheap in every row); upgrading beats restarting by ~2–3pp in cls_B0 vision→som — inside the band.
- **Cross-mode divergence at step 0** (for anyone tempted by a later splice point): url_after differs between cheap and rich in 27–70% of tasks, action_type in 20–58%. README defect #10's "decision window ≥2 steps in 64%" is a URL-equality statement, not a state-equality one.

## 4. Contextual bandit replay (sVJH's ask) — does not reach a fixed arm

Full-information logs (every arm's outcome is logged per task) allow exact replay. Context = OBS0 pre-flight stats; reward = success − λ·cost/median_cost; LinUCB(α=1) and Thompson; 200–300 random task orders.

| arms | λ | cells where bandit < fixed-best | cells where bandit < fixed-cheapest | best-arm share in last quarter of the stream |
|---|---|---|---|---|
| 6 (all modes) | 0 | 11/11 (e.g. cls_B0 22.4% vs som 27.2%) | 11/11 (vs vision 25.0%) | 10–38% per arm, near-uniform |
| 6 | 0.5 | 11/11 | 11/11 | vision 24–48% |
| 2 (cheapest, best) | 0 | 11/11 (cls_B0 26.2% vs 27.2%; cls_B5 34.1% vs 37.1%) | — | 47–69% |
| 3 (dom, vision, som) | 0 | 11/11 (cls_B0 24.1% vs 27.2%; cls_B5 31.1% vs 37.1%) | — | 37–56% |

At 104–435 tasks per cell with Bernoulli rewards at p = 0.02–0.37 and a 10–14% rerun flip, exploration cost alone exceeds the oracle headroom. "Online" adds nothing that "learn from the same rows offline" did not already fail at.

## 5. The best−2nd gap correlation (README defect #5) — reproduces, and is not a finding

| router | corr(gap, Δ_router) | bootstrap 95% | leave-one-out range |
|---|---:|---|---|
| v3 all+costaware | −0.708 | [−0.94, +0.63] | [−0.79, −0.62] |
| v1 intent-only | −0.556 | [−0.98, +0.16] | [−0.90, −0.36] |
| 3-class R/L/B | −0.038 | [−0.56, +0.81] | [−0.25, +0.30] |
| *oracle* one-added-arm gain vs gap | +0.371 | [−0.10, +0.69] | — |
| oracle gain vs labelled % | +0.844 | [+0.71, +0.96] | — |

Δ_router = SR_router − SR_best ≈ −(share of off-best choices) × gap + informed gain. If the router's off-best picks are uninformed, corr(gap, Δ_router) is negative by construction whenever the router deviates at all; it is a statement about the router's ignorance, not about routability. The quantity that would support "two similar arms make routing viable" is the oracle gain vs gap, and it is +0.37 with a CI through zero. What headroom tracks is labelled % (+0.84) — the paper's ρ = 0.952 again.

## 6. Collapsing to three arms (READ = dom, LOOK = vision, SoM) — helps supply and convergence a little, changes no verdict

| cell | orc3 − best | orc6 − best | band | contested (3 / 6 arms) | 3-class rows (cheapest solver) | min class ≥ 10 |
|---|---:|---:|---|---|---|---|
| cls_B0 | +11.2pp | +16.1pp | 4.5–7.6 | 31.2% / 39.3% | dom 17 · vision 56 · som 13 | ✓ |
| cls_B1 | +7.1 | +10.3 | 0.0–1.8 | 19.2 / 23.2 | 8 · 28 · 12 | ✗ |
| cls_B5 | +9.4 | +14.3 | 5.8–7.1 | 44.6 / 50.4 | 53 · 8 · 43 | ✗ |
| red_B0 | +8.8 | +12.2 | 2.0–6.9 | 21.0 / 24.9 | 24 · 16 · 8 | ✗ |
| red_B0_WA | +2.9 | +16.3 | — | 27.9 / 43.3 | 28 · 6 · 6 | ✗ |
| red_B1_WA | +5.8 | +14.4 | — | 17.3 / 26.0 | 11 · 10 · 2 | ✗ |
| shop_B0 | +8.0 | +8.0 | — | 17.9 / 17.9 | 13 · 64 · 24 | ✓ |
| others (cls_B2, red_B1, red_B2, shop_B1) | +2.4…+5.8 | +4.4…+9.9 | — | | | ✗ |

Three arms keep ~2/3 of the six-arm ceiling on VWA and lose 60–80% of it on the two WA cells, whose winners are text-side phantom arms. The minimum which-mode class rises from single digits to 8–13 in the large cells but clears the ten-row filter in only 2/11 (the six-class filter cleared 2/6). The 3-arm bandit still trails fixed-best in 11/11; a 3-way lookahead (dom step-0 → {stay, vision, som}) tracks the random-to-best control within ±1pp in every cell and loses SR at every f in the five cells where dom is already best.

## 7. What this closes and what it does not

- Closes, with data, three of sVJH's four named gaps: LLM-based routing (pilot v1–v3, all inside the band), contextual bandit (§4), online routing after partial interaction (§1–3, at the only splice-legal point). The fourth, graded trajectory supervision, is unchanged: this analysis found no data-native step-level label that carries more than the task text.
- Sharpens §6: the contested rows flip between reruns **and the flip is not observable before the first action** — the two halves of supply–value coupling meet at step 0.
- Does not test: escalation at step ≥ 1 (not splice-legal; URL-equality ≠ state-equality), text-embedding routers beyond TF-IDF, or any router that changes the *grounding path* (identifier contract) rather than the observation — the one axis on which per-step labels are dense (`locator_route_meta.success`, ~13k actions) and which §503.1 shows moves SR as much as the representation does.

## 8. Is the two-stage recipe (learn success, then learn the cheapest solver) the wrong direction? — the target of stage 2 is smaller than the noise

Six-arm double-replicate cells allow a variance decomposition of the task × mode success matrix:

| component | cls_B0 | red_B0 | learned by |
|---|---:|---:|---|
| task main effect (difficulty) | 0.0773 | 0.0557 | abstention / triage (§457, works) |
| mode main effect (which fixed arm) | 0.0026 | 0.0004 | a fixed policy |
| **task × mode interaction (the routing target)**, after removing noise leaking into 2-draw cell means | **0.0209** | **0.0056** | which-mode / lookahead / bandit |
| replicate noise | 0.0618 | 0.0459 | — |
| interaction / noise | **0.34** | **0.12** | |

Single-run "route away from som to X" labels that reproduce in the second run: cls_B0 20/66 (dom 3/16, vision 3/15, ptext 5/11, pprompt 4/14, psom 5/10), red_B0 9/43. Stable strict preferences (2/2 vs 0/2) number 1–5 tasks per mode. The decomposition is not the mistake: stage 1 has SNR > 1 and is exactly abstention; stage 2's target is one-third to one-eighth of the rerun noise and 70–80% of its training labels are coin flips, so no estimator — two-stage, direct expected-utility (§2–3 above), or online (§4) — can recover it at one draw per (task, mode). Denoising labels to interaction > noise/k needs k ≥ 3 replicates per (task, mode) on cls_B0 and k ≥ 9 on red_B0.

Four-axis check (SR / cost / latency / tokens): no mode wins all four axes in any of 11 cells (cheapest is mostly vision, fastest mostly som or vision), so a fixed-arm frontier exists; the lookahead policy dominates always-rich on all four axes in 3/21 non-degenerate pairs, none on a banded cell; on cls_B0 vision→som at matched SR it saves 7.8% cost and 10% tokens but adds 17.6% latency.

## 9. Would richer labels (an LLM grader / planner / failure-signature synthesiser) change this? — an upper-bound proxy says: a little on cls, nothing on red

Cleanest testable form: is a *graded* label more reproducible than binary success? Proxy for an LLM grader, without grader noise or mode bias: progress = share of the URLs visited by the task's other successful episodes (any mode, any arm; leave-self-out) that this episode also visited. Defined on 9,841/21,283 episodes (tasks somebody solved).

| | binary success | graded progress |
|---|---|---|
| replicate agreement, 18 pairs | kappa 0.22–0.58 | r 0.50–0.78 (pooled 0.74) |
| on tasks whose outcome flipped | — | |Δprogress| 0.15–0.29 (vs 0.03–0.15 on non-flip) |
| interaction / noise, cls_B0 (same 135 tasks) | 0.32 | **0.49** |
| interaction / noise, red_B0 (same 62 tasks) | 0.12 | **0.12** |

Grading cuts the noise about 4× but cuts the interaction about 3×: how far a mode gets differs across modes as little as whether it finishes. On cls_B0 it buys roughly one replicate's worth of denoising (k ≥ 3 → k ≥ 2); on red_B0 nothing. It is undefined exactly where an LLM grader would add coverage (universal-fail tasks), and there is no ground truth there to validate it. The other two forms were tested above: a planner is a function of intent + page and is the step-0 thought (§2: no information beyond the intent); an LLM synthesising the 40 failure rules is the pilot's v2/v3 router input (all inside the band). The step-level `reward` field is not a per-step evaluator (95.7% of failed episodes have reward > 0 at step 0).

## 10. The failure *cause* is stable across reruns; it still does not say whether another mode would succeed

Proxy for an LLM that synthesises "why it failed, what would fix it": the deterministic `reason_bucket` from `analyze_reason_diagnostics.py` (10+ buckets), computed on both arms of the three cls_B0 replicate pairs that have reruns of dom, vision and som.

| | dom | vision | som |
|---|---|---|---|
| tasks failed in both runs | 174 | 153 | 146 |
| fine bucket agreement (kappa) | 74.1% (0.66) | 71.9% (0.64) | 76.0% (0.70) |
| coarse stuck / finish-wrong / early-finish (kappa) | 81.6% (0.61) | 80.4% (0.62) | 78.8% (0.59) |
| final_error_category / loop_pattern agreement | 90.2% / 76.4% | 88.9% / 85.0% | 91.1% / 81.5% |

The cause is far more reproducible than the outcome on contested rows (kappa 0.22–0.58). But out-of-fold AUROC of the cheap arm's bucket for predicting the *rich* arm's success is 0.567 (dom→som), 0.341 (vision→som), 0.501 (dom→vision), 0.459 (som→dom); for switch-only (rich succeeds ∧ cheap rerun fails) 0.35–0.54; for the cheap arm's own rerun success 0.51–0.63. The stable part of the cause is a task property (hard / hopeless: `fail_max_steps` → 0/14 rich successes), not a mode-fit property. Across §8–§10, every label-enrichment route buys stability (graded +50% SNR on cls; cause kappa 0.7) and none buys the interaction term.

## 11. Why the strongest cell (cls_B5, GPT-5.6) is no more learnable: capability raised the mode main effect, not the interaction, and the noise floor did not move

| | cls_B0 | cls_B5 |
|---|---:|---:|
| best (som) / union | 27.2% / 43.3% | 37.1% / 51.3% |
| route-away tasks (some arm solves, best does not) | 36 (16.1pp) | **32 (14.3pp)** |
| route-away as share of union | 37% | **28%** |
| nestedness of non-best arms inside som (mean |m ∩ som| / |m|) | 0.68 | **0.73** |
| mode main-effect variance (highest of 11 cells) | 0.0020 | **0.0053** |
| unique-vs-som successes per arm | 10–16 | 11–13 |
| dom rerun discordance | 12.1% | **12.9%** |
| stable som→dom route-away (dom A ✓, som ✗, dom B ✓) | — | 7/13 = 3.1pp vs band 5.8–7.1 |

Across the 11 cells the route-away share rises with best-arm SR up to 14–16pp and then flattens (cls_B0 16.1, red_B0_WA 16.3, cls_B5 14.3, red_B1_WA 14.4). §6's "contested set tracks SR at ρ = 0.952" is correct but counts rows where only the best arm solves, which need no router. The routing value is the route-away share, and B5 shows a stronger backbone can make one representation dominate more rather than make representations complement each other. The reversal §6 predicts requires "stronger and non-nested"; B5 is one data point where the two diverge. (One replicate arm only; 7/13 is an upper bound because som-side flips are unobserved.)

## 12. The nested half: can "the cheap arm also suffices" be learned pre-flight? — yes on cls_B5 (AUROC 0.67, label kappa 0.61), and it cashes out at +1–2pp over random

Conditional label = cheapest arm succeeds, restricted to tasks the best arm succeeds on. Pre-flight features only (OBS0) or plus the cheap arm's step-0 output (+MODEL0).

| cell | cheap→best | n (best ✓) | cheap ✓ | OBS0 | +MODEL0 | null p95 | GroupKFold | label stability across cheap reruns |
|---|---|---:|---:|---:|---:|---:|---:|---|
| cls_B5 | dom→som | 83 | 40 | **0.671** | 0.664 | 0.559 | 0.608 | 81% agree, kappa **0.61** |
| shop_B0 | vision→som | 66 | 42 | 0.648 | 0.674 | 0.622 | 0.619 | — |
| cls_B0 | vision→som | 61 | 41 | 0.421 | 0.328 | 0.619 | 0.345 | 75%, kappa 0.45 |
| red_B0 | vision→dom | 30 | 6 | 0.632 | 0.847 | 0.716 | 0.653 | kappa 0.56 |

Deployed as a whole-cell pre-flight policy (score for "send to cheap" = P(nobody) + P(somebody)·P(cheap suffices | somebody), all out-of-fold), cls_B5 dom→som: sending 30 / 60 / 80% of tasks to dom loses 2.7 / 5.8 / 8.5pp against always-som and saves 4 / 11 / 17% of cost; random assignment of the same share loses 4.0 / 8.0 / 10.7pp, so the learned policy is worth +1.3 / +2.2 / +2.2pp at matched cost. The cost-ceiling oracle (dom suffices or nobody can → dom, 181 tasks) gains +5.8pp and saves 22%; the learned policy recovers about 15% of the oracle-over-random headroom, and reaching a 10% saving costs ~5.5pp of SR, the lower edge of the cell's rerun band (5.8–7.1). shop_B0 lands near the line between always-vision and always-som; cls_B0's conditional signal is below chance. Same shape as §5 policy 2 and §387.16.4: labels exist, AUROC clears the null, the fixed policies are not beaten.

## 13. What the failed episodes still carry — three "one step away" readings, all task-level

| reading | quantity | cls_B0 (6 arms × 2) | red_B0 (5 arms × 2) |
|---|---|---|---|
| failure cause (`reason_bucket`) stability across reruns | fine-bucket kappa | 0.55–0.70 | 0.41–0.51 |
| page-side near-miss (visited reference URL / final URL matched, eval failed) | share of failed episodes | 1–6% (2–11 per arm) | 1–3% |
| answer-side near-miss (≥80% of reference tokens in the final answer) | per arm, among 39–55 non-URL failures | 4–5 | — |
| interaction / noise after relabelling success ∨ near-miss | six-arm decomposition | 0.34 → 0.36 | 0.16 → 0.20 |
| single-run route-away labels reproduced in run 2 | binary → enriched | 20/66 → 26/82 | 7/36 → 9/41 |

"One step from correct" barely exists for this agent: failures are lost (not found / stuck / finished with the wrong answer), not near the goal. What is stable on the failure side (cause kappa 0.7; answer closeness correlated across modes at r = 0.57; near-miss predicting the arm's own rerun at 50% vs 4%) is a property of the task; the cheap arm's near-miss predicts the rich arm's success in the same direction it predicts its own rerun, and cheap-arm answers close to the reference are fixed by no other arm (0% of 5–6). Failure data is not wasted: hopeless buckets are the abstention positives (§457), the 40 failure signatures of §3 are failure-side products, and the replicate flip pairs (326 tasks whose two runs diverge from an identical start state) are a preference-pair substrate for training the agent, which is a different paper from routing it. (red_B0 psom canonical is absent from this table: task 149's step JSONL fails the strict-identity check.)

## 14. Calibrating against the July–September 2026 computer-use landscape (semantic-first, vision on demand)

Where our runs' wall-clock goes (median step-0 backend inference / total step): API arms 15–28% (B0·cls: 1.8–2.1 s of 8.2–8.6 s), local 4B arms 36–65%. In cls_B0 every mode costs 7.0–7.4 s per step; som is the fastest arm (62 s) only because it takes the fewest steps (8 vs 10–14), and dom spends 3,857 tokens per step against vision's 3,504. So "turns, not tokens" holds here, while "planning is 75–94% of latency" (OSWorld-Human, frontier reasoning models) does not: on a self-hosted VWA stack the browser is the bottleneck.

The one arm of the landscape's canonical experiment this project has never run is the reactive heuristic router (AX first, escalate to vision on an observable failure): `p79/experiment/router.py` implements it (unchanged-page / no-progress / action-failure streak triggers), and no router-on condition exists in `results/`. It cannot be evaluated by splicing (the rich arm would start from a site the cheap arm has already acted on). Offline, the triggers' selectivity is poor: on cls_B0 dom failures, grounding/stuck signals reach AUROC 0.41–0.56 for som's success; the streak trigger fires in 57% of failed and 33% of successful dom episodes, at a median step 7 of 19 (45% of the episode's cost already spent). An indicative (not splice-legal) projection puts it at 26.3% / $0.0869 against always-som's 27.2% / $0.0724: 24 rescued, 81 switched in vain, 13 successes disrupted. That is a prediction for a live run, not a measurement.
