---
type: action-ledger
status: rolling
updated: 2026-05-18
---

# Next Steps — Forward Action Ledger

> **Future-only**. Live state 不在这里:
> - Today / 瓶颈 / cron health → [[PLAYBOOK#§1]] + [[PLAYBOOK#§2]] (🤖 GLM @daily)
> - Real-time active runs / GPU → `make active` CLI
> - Cell snapshot (active 跑中 / pending / done) → `cells.base`
> - Paper section progress → `status.base`
> - 过去 chronicle → [[实验笔记]] (latest §240, 2026-05-20)
> - Strategy / theory → [[paper_planning]]
> - **Phase 1 执行计划 + audit checklist** → [[phase1_plan]] ⭐ canonical
> - Advisor sync prep → [[issue_advisor_sync_2026-05-14]]
> - OSF DOI lock workflow → [[osf_lock_manifest]]
> - Compute infrastructure → [[COMPUTE_INFRASTRUCTURE]]
>
> 🔧 新数据 → `make analysis` (~5-10min). Cron 每 10min 自动 sync cell frontmatter.

---

## §0a CURRENT — Fire-6 Pass-1 LIVE (2026-05-21)

> [!success] **Fire-6 Pass-1 RE-FIRED with B-1803** (2026-05-21). First launch (~12:06) aborted at cls B0 dom **task 4** (id=84144) — the 4th fire (Fire-3/4/5/6) killed by the same `EvaluatorUnavailableError: Page.goto 30s×3` on a program_html eval. **Diagnosis refuted the prior "agent-modified substrate" RCA**: item 84144 is NOT deleted (DB `b_active=1`) and the URL is healthy (curl 0.17s) — the real cause is the **degraded long-lived BrowserContext** by task ~4, which C1's same-context `new_page` did not escape. **Fix B-1803** (commit `4baac19`, Fire-6 RCA C1b): eval isolation → FRESH browser context (clean Chromium profile + auth from config storage_state file). Re-fired ~12:51 (orchestrator 566503, HEAD `4baac19`, all gates green, cls→red sequential). Pre-fire /stress (B-1796..B-1802) + Addendum 01a witness + re-Smoke A PASS all landed earlier. **✅ B-1803 CONFIRMED (14:18)**: re-fire passed task 4 (maxtask=5, evalerr=0) — the 4-fire id=84144 eval blocker is broken; Fire-6 cls now flowing past it. Multi-day run on standing tooling (A100 shell healthcheck cron + daily Claude review + watchdog). **Nothing required while it runs.**
>
> **Monitoring** (all zero-touch): A100 cron `scripts/maintenance/fire6_monitor.sh` (anomaly-only ntfy every 30min + daily heartbeat → `p79-exp-dgx-spark`) · daily Claude intelligent review cron (subtle-anomaly digest) · per-runner `experiment_watchdog` (auth refresh / clean re-run) · `make active` / `cells.base` / PLAYBOOK §1+§2.

> [!todo] Forward plan (priority order, 2026-05-21)
> 1. **[~9d, passive] Pass-1 lands** → `queue_phase1_paper_grade.sh status` shows 36/36 → `make analysis` (cross-condition + figures) → sanity-check §1 hero numbers (phantom-SoM 4-fold drop-in; H1/H2/H3 per cell). Fire-6 = first B0 fair measurement under schema≡validator.
> 2. **Pass-2 learned router (6 cond)** ⭐⭐ — post-Pass-1: LR training pipeline (Pass-1 outcomes → oracle label matrix → entropy defer gate → per-cell LR heads → artifact smoke) → `queue_phase1_router_paper_grade.sh`. ~3-5d. (B-1671 launch-pass2 raises until LR pipeline lands.) 详 [[phase1_plan]] §B-router + §C.
> 3. **Analysis + gating** post Pass-1+2 — H1/H2/H3 (FE pooled bootstrap percentile p, R1-R5 framing per §2.5) + H10 (router Pareto non-dominance + 5/6 grid, §6). 详 [[paper_planning]] §16 + [[preregistration]] §2.
> 4. **DOI 2 reproducibility bundle** ⭐⭐ — trigger = Pass-1+2 done + analysis frozen + paper §1-8 finalized. 42 condition summaries + episodes + steps + CSV + figures + scripts + drafts; README `cited_by DOI 1`. 详 [[osf_lock_manifest]] §3b.
> 5. **Paper writing (codex prose round)** — §1 hero numbers from Fire-6 clean data; §3.5.1 already carries Protocol Reset + B-1794 + B-1796..B-1802 disclosure. **Fold in deferred P1-11** (gemini: "upstream-aligned/core" labeling → Abstract non-comparability statement; deferred this /stress round per Q4=A no-codex-round).
> 6. **R-tier decision** (R1/R3/R5) + workshop vs Phase 1b shop expansion timing — post-data. 详 [[paper_planning]] §16.
> 7. **(deferred) GRL-layer / bug workshop sub-paper** — B-91 LLM-judge polarity + B-1796..B-1802 reproducible-accounting cluster. Track A/B 2026-06, NOT critical-path. 详 [[workshop_subpaper_plan]] §0.1.
>
> **2026-05-21 lineage**: pre-Fire-6 /stress 3-AI (18 findings → 8 fix) + Addendum 01a witness + re-Smoke A PASS + **Fire-6 Pass-1 FIRED**. 详 [[实验笔记]] §252 + §252.1 + §252.2.

---

## §0b Router pipeline pre-fire /stress — deferred follow-ups (2026-05-21)

> Router-pipeline pre-fire /stress (3-AI A+B+C, 22 findings; [[实验笔记]] §255). **16 P0/P1 fixed this round** — 簇 α B-1805~1809 (`25d4608`) / 簇 β B-1810/1811/1818 (`d4bc0bd`) / 簇 γ B-1812/1813 (`9a3703a`) / 簇 δ F7 B-1815 + F3/C3 disclosure. Below = deferred items, all gated on **Pass-1 data landing** (router code runs on empty data now, `n_pooled_total:0`, so effects can't be validated yet).

> [!todo] Deferred — do after Pass-1 lands (real data to validate against)
> 1. **Write the Pass-1 run manifest** (C2/B-1810) — `results/phantom_paper/l1_router/pass1_run_manifest.json` with the exact 36 paper-grade Pass-1 run IDs → upgrades `discover_runs` from "glob + reject smoke" to strict whitelist. Then run `aggregate_h10_pareto.py --require-full-coverage` (C8/B-1811) for the paper-grade H10 verdict (fail-closed on incomplete coverage).
> 2. **F3/G2 (B-1814) τ objective — sensitivity, NOT a refactor** — after Pass-1 lands, run a **τ sensitivity comparison: (a) current accuracy-τ vs (b) fixed τ vs (c) outcome-matrix Pareto-τ** on the real data; adopt (c) only if it materially shifts the frontier. **Do NOT refactor Stage 1-3 now** (user directive 2026-05-21). (c) needs the per-task per-mode outcome matrix threaded Stage 1 → Stage 3 inner-CV. 2nd-order: τ-selection only, outer H10 eval already clean (Stage-2 selectors + Stage-3 LR heads never see fold-k holdout).
> 3. **C3 (B-1816) inner-CV MI leak** — inner-CV reuses the Stage-2 outer-pool MI selector (mild, 2nd-order, 3rd-order magnitude). Nested per-inner-fold MI would conflict with the user-confirmed E'' pooled-cross-cell selector (per-cell ~30 vs pooled ~1124 MI samples). Decide on real data whether the leak materially moves τ; if not, the disclosure (in `tau_tuning_disclosure` + docstring) suffices.
> 4. **§6 disclosure block** (write into paper §6 / §3.5 once data lands):
>    - self-oracle noise ceiling (G1/B-1809) — oracle labels are N=1; report the noise floor before claiming the router beats it (needs a 2nd independent Pass-1 set, or report the multi-success-task fraction as the noise-sensitive subset; both already emitted in `oracle_provenance`).
>    - deployment realism (C1/gemini#3) — no-success tasks are now routed (out-of-sample fold LR); report their cost in the cost-benefit.
>    - MODES cost order (F2/B-1806) — verify the prior ascending-cost order against measured per-mode mean cost (summary_v2 `total_model_cost_usd`); switch to a measured-cost tie-break if they disagree.
>    - router_strictly_better (F7/B-1815) — report `k_cells_router_strictly_better` (θ CI lower > 0) alongside non-dominance; do NOT claim a learned-routing benefit if the router collapses toward a single baseline.
>    - sklearn/numpy version metadata (C6/B-1813) — store at train time + validate at serve time (defence-in-depth beyond the loader hard-fail).
> 5. ✅ **F6-followup DONE** (B-1821, 2026-05-21, commit `b0c79f2`) — 3 diagnostic scripts (`p1_archive_simulation` / `l2_partial_trajectory_auroc` / `router_archive_diagnostic`) now `from p79.policies.router_features import MODES` (identity-shared, verified `is`). Fixed a latent **cost-bias**: the local copies sat in the buggy pre-F2 order, biasing the archive oracle / G-1 tie-break toward expensive som/vision. 详 [[master_bug_catalog]] B-1821.
> 6. **F9 (P2)** — delete/guard the deprecated 8-dim `predict_mode` path (`learned_router.py:425-494`) after confirming no caller (runner uses `predict_mode_fold_aware`).
>
> 详 [[实验笔记]] §255 (→ §256 fix wave) + [[master_bug_catalog]] B-1805~B-1818.

---

## §0 Direction

**Paper hook**: → [[paper_planning#§1]] (canonical, phantom routing space 3 arms / 4-fold drop-in)

> [!warning] ⤵️ SUPERSEDED by §0a CURRENT (2026-05-21). NOTE: the RCA below mis-attributed the pattern to "agent-modified substrate" — the 2026-05-21 Fire-6 diagnosis REFUTED that (item not deleted; URL curl 0.17s). The real cause is a **degraded long-lived BrowserContext**, and the C1 same-context isolation (which the earlier "resolved" framing relied on) was INSUFFICIENT — Fire-6 was the 4th occurrence. Actual fix = **B-1803** fresh-browser-context eval isolation (§0a). Kept below for historical RCA context only; the substrate hypothesis in it is wrong.

> [!todo] Top forward actions (priority order, **2026-05-20 00:35 BST — Fire-5 aborted at cls task 4 / Fire-3 task 75 pattern repeated / 3-fire stateful-modify-task signal**)
> 1. **🚨 Fire-5 aborted 00:27:46 BST — RCA + Fire-6 prep needed** ⭐⭐⭐ — Fire-5 launched 2026-05-20 00:00:18 BST with all 8 preflight gates PASS (incl new Gate 8). cls B0 dom condition crashed at **task 4 (delete white car listing, eval URL id=84144) ~15 min in** via `EvaluatorUnavailableError: Page.goto Timeout 30000ms × 3 retries`. Master P0-2-B sentinel-wait correctly halted (rc=1 → NOT launching red ✓ first successful production fire-day fail-closed defense). Run dir: `B0_dom_classifieds_20260520_000018_844563552_322810_R14647`. Tasks 0-3 done in ~10min (task 2 only success), task 4 = ~15min then EvaluatorUnavailableError. **M1 PaperGradeAbortError gate did NOT fire** — because EvaluatorUnavailableError raises BEFORE summary write at `runner/main.py:1505` (B-1662 path), so M1's after-summary gate by design doesn't apply. Quarantine event appended to registry (task 4: 1 unclassified). Gate 8 now correctly HALTs Fire-6 until task 4 classified.
>    **3-fire pattern (cumulative state hypothesis upgrade from M7 transient_drift → suspected substrate-systemic)**:
>    - Fire-3 cls task 75 (edit white vase $80→$120, eval id=84148) → `EvaluatorUnavailableError Page.goto timeout` 30s × 3
>    - Fire-4 cls task 75 (same task) → `Page.screenshot timeout` 30s at step 22 (agent step, not evaluator)
>    - Fire-5 cls task 4 (delete white car, eval id=84144) → `EvaluatorUnavailableError Page.goto timeout` 30s × 3
>    **Common factor**: all 3 are *stateful modify tasks (edit/delete listing)* where agent modifies DB substrate, then evaluator does `Page.goto(item_url)` to verify, and that Page.goto times out 30s. M7 isolated Playwright MCP reproduction of id=84148 (no agent prior state) loaded in **1040ms clean** — so the page itself is fine. Failure correlates with **agent-edited substrate + multi-process A100 system load + Chromium-process cumulative state**.
>    **Immediate next-step decision tree** (recommend execute b → a → classify → c-or-d):
>    - (a) **Reproduce task 4 via MCP** with login session + simulate the delete action sequence (more accurate reproduce than fresh load), determine if delete actually committed + classify the `Page.goto Timeout` root cause. 1-2h.
>    - (b) **Direct DB query** (`docker exec classifieds_db mysql -uroot osclass`) to check if `pk_i_id=84144` row state (active/deleted) at time of crash → proves agent action commit + invalidation latency. 5min. Lowest-effort highest-evidence step.
>    - (c) **Classify task 4** + re-fire Fire-6. But strategic Q: if 3-fire pattern means structurally-fragile evaluator-on-modified-substrate, re-fire alone likely Fire-6 dies same way.
>    - (d) **Strategic re-design**: if the 3-fire signal indicates evaluator Page.goto on agent-modified items is structurally fragile, consider whether `program_html` evaluator path needs M5-style timeout taxonomy + isolated browser context per evaluation (not the same long-lived runner browser) + 90s eval timeout (not 30s) to absorb invalidation latency.
>    详 (will append) [[实验笔记]] §239 Fire-5 RCA.
> 2. **DB MCP setup deferred to post-Fire-5/6** (reference memory `reference-playwright-mcp` set 2026-05-20):
>    - cls_db = MySQL 8.1, root/password, database=osclass, **no host port mapping** (internal docker net only)
>    - red_db = postmill image has MySQL 3306 + Postgres 5432 in container, **no host port mapping**
>    - shop = Magento MySQL, **no host port mapping**
>    - **Constraint**: adding `-p 13306:3306 -p 15432:5432` host port mappings requires `docker restart` which would kill any running Fire (can't do during Fire). Defer to between Fires. Until then use `docker exec` pattern via SSH+Bash (already in `reset_vwa_sites.sh`).
>    - **Post Fire-5 done (now)**: this is the window — Fire-5 already aborted, no active fire, OK to restart containers + install MySQL MCP. But also OK to defer until after Fire-6 if priority is fast Fire-6 turnaround.
> 2. ✅ **OSF DOI 1 — MINTED 2026-05-18T23:10:06Z UTC** at **`10.17605/OSF.IO/9QCWU`** (Registration GUID `9qcwu`, parent project `kv9sf`; URL https://osf.io/9qcwu). Archive auto-completed instantly (no 48h wait). License CC-By 4.0. Pre-launch canonical witness 2026-05-18T21:16:28Z tier 1 strict; full-file SHA `6056b905...` / content-only SHA `011fa4c0...` per `DOI_1_README §"Witness file hash convention"`. Post-mint promoted tag `preregistration-doi1-minted-osf-9QCWU` @ 5edac3b (same anchor as witness tag). Backfills landed: `05caf27` (DOI/GUID/UTC + receipt) + this commit (§2.1 + §2.5 lock SHAs + chronicle §235). Outstanding paper-cite-as task: §3 + §4 + Appendix D footnotes cite `10.17605/OSF.IO/9QCWU` (defer to next codex paper round per advisor 2026-05-14 "mechanism 暂搁"). Cite as: `OSF preregistration 10.17605/OSF.IO/9QCWU, submitted 2026-05-18T23:10:06Z UTC, pre-canonical-outcome-creation witness`. 详 [[实验笔记]] §235 + [[osf_lock_manifest]] §2.
> 3. **Pass-2 learned router (6 conditions) sequential post-Pass-1** ⭐⭐ — fire trigger: `queue_phase1_router_paper_grade.sh` on A100 after Pass-1 complete (`bash scripts/queues/queue_phase1_paper_grade.sh status` shows 36/36 cells). ~3-5 天 wallclock. 详 [[phase1_plan]] §B-router + §C router operationalization. **Substrate gap**: B-1671 launch-pass2 currently raises delegation message until LR training pipeline lands (depends on Pass-1 outcomes for LR feature extraction).
> 4. **DOI 2 reproducibility bundle ⭐⭐** — mint trigger = Pass-1 + Pass-2 complete + analysis scripts frozen + paper §1-§8 finalized (~2-3 周 post Fire-3 start). Bundle includes: 42 condition_summary_v2.json + episode summaries + steps JSONL + aggregate CSV + figures + analysis scripts + finalized paper drafts。DOI 2 README explicit `cited_by DOI 1`。详 [[osf_lock_manifest]] §3b 8-step workflow.
> 5. **R-tier 决策 (R1/R3/R5)** post-data — workshop submission (workshop_R1 = H1+H2(a) only per prereg §2.7 bifurcation) vs Phase 1b shop expansion timing. 详 [[paper_planning]] §16.0 multi-submission matrix.
> 6. **独立 bug 研究 paper** (Track B B-91 LLM judge polarity standalone workshop note, R5 fallback per prereg §2 R5 row) — optional, 详 §11.
>
> **重大变化 lineage (sticky chronology)**:
> - 2026-05-14 收口: mechanism (§5/§0a) 整个暂搁; Gemma3-VL 入 baseline; 学生 focus = experiment execution.
> - 2026-05-15 host migration: paper-grade canonical run on A100 self-host VWA Docker (NOT DGX→quark Tailscale).
> - 2026-05-16 v7 walk-back: Phase 1a expanded 36 → 42 conditions (Pass-1 baseline 36 + Pass-2 learned router 6).
> - **2026-05-18 §A2 14/14 closed**: A2.1+A2.2+A2.3a/b/c/d+A2.4a/b+A2.5+A2.6a/b/c+A2.7+A2.8+A2.9+A2.10 全 closed via cross-AI /stress 3-AI cycles. Phase 1a fire substrate-ready.
> - **2026-05-18 B-1570 doctrine shift**: advisor email = optional post-fire collateral, NOT fire/lock blocker. OSF DOI replaces advisor email as primary witness function.
> - **2026-05-18 evening Q3=A doctrine fix wave** (commits 7925f71→72b93c9→79daf91 + B-1750~B-1759): 8 doctrine drift bugs in OSF DOI 1 deposit + 2 topvenue residue + 7 staging cleanup issues all closed. Substance-lock at `preregistration-locked-q3a` @ 72b93c9.
> - **2026-05-18 evening Fire-3 LIVE 🔥** (attempt #6 @ 21:27:28Z, commit `5edac3b`, tag `preregistration-doi1-witnessed-20260518T211628Z`): tier 1 pre-launch witness strategy supersedes 5-min post-PID-alive window; cls→red sequential chain running on A100。详 [[实验笔记]] §233.
> - **2026-05-19 evening Fire-3 cls/red dead + Phase 0 /stress hardening wave** (commits `04e1d9b` Phase 1 + `8d2a327` Phase 2 + this commit Phase 3): F1 async_envs.py RCA retracted; 3-AI /stress unified bug list 18 findings; cold re-fire after /stress on diff + user push approval。详 [[实验笔记]] §237.

---

## §0a Mechanism (§5) — ⏸️ 暂搁 (advisor discussion 2026-05-14)

> ⏸️ **2026-05-14 收口**: 学长 "mechanism 部分先不要管了". 整个 §5 (activation patching / layer probe / logit lens / SAE) 暂搁; 下面 forward items 全部冻结, **不进当前 paper scope**. §133/§136 已 land 的 mechanism v2 工作存档保留 (见 [[实验笔记]] §138.3). 以下内容保留作未来 paper-2 / 解冻参考.

**DONE (2026-05-11 → 05-14, 见 [[实验笔记]] §125-§133)**: Stage 4 全部 4 方法 land —
Method 4.2 PCA cosine gap (AUROC 1.000) / activation patching Exp 5 cellhprompt (L11-L17 displacement 0.20-0.30) /
Exp 3 logit lens (per-task KL, axis-2 ratio 1.1-3.95×) / Method 4.4 mean-diff steering (v2 train/eval split,
held-out 0.12 vs in-sample 0.29 → A5 counter-claim succeeds). §5 prose v2 reframe = probe–causal–steering trichotomy.
v2 NPZ re-extraction done. Pipeline audit + 5 paper-grade fix commits.

**Forward**:

| Pri | Item | Effort | Gating |
|---|---|---|---|
| ⭐⭐ | **Cross-family P2/P3 fire** — Phi-3.5-Vision + Qwen2-VL-7B extraction (scripts 已修 Bug 2/5, paper-grade safe). H1' capacity-limit test: 4B shortcut 是容量限制还是训练分布先验 | ~1-2h GPU/model | advisor mechanistic scope 决策 (B1-only → 不跑; cross-arch → 跑) |
| ⭐ | **SAE feature steering** — 把 "steering 不 transfer" 翻成 positive intervention. 当前倾向: 不做, 留 paper-2 (三分结论已自洽, SAE 引入新举证负担) | weeks | advisor 决策 (SAE 进 paper-1?) |
| 🟡 | **format_variation `fmt_som_standard` v1-ish 修复** — codex C1 P0, data-altering, 需 re-extract. 当前 documented NOT patched | 30-60min + extraction | 决定是否影响 H1/format-variation baseline 可比性 |
| 🟢 | `run_stage1_pilot.py` NPZ schema gap (older pipeline) | low | 非阻塞 |

---

## §1 Phase 1a paper-grade rerun → [[phase1_plan]]

**Scope** (2026-05-16 v7 walk-back final, [[实验笔记]] §200-§224): **42 operational conditions / 6 statistical cells** = Pass-1 baseline 36 (cls + red × {B0, B1, B2 = Gemma3-VL} × 6 modes) + Pass-2 learned router 6 (cls + red × {B0,B1,B2} × 1 learned-router cond/cell, `obs_mode="learned"` sentinel)。旧 24/4 + 36-only 已废 (H10 router Pass-2 inclusion per B-264+B-267 /stress A1.7 2026-05-16)。
**Phase 1b** (post-workshop deferred) = + shop × 3 × (6 baseline + 1 learned router) = 21 cond, feeds R3 → R1 main paper expansion 决策。

**Terminology hard rule**: "condition" = 1 (site, model, mode-or-router) launch unit; "cell" = 1 (site, model) stratification unit. **不要混用**。

**Canonical 执行 checklist + critical path + milestones + pre-launch gates + post-completion**: → [[phase1_plan]] §0 + §A + §B + §E

**当前 launch 主 blockers (snapshot 2026-05-18 post §A2 14/14 + prep)**:
- ✅ **#11 A100 VM VWA docker bring-up** — DONE 2026-05-13 (cls 9980 / red 9999 / shop 7770 / wiki 8888 all UP)
- ✅ **A100 substrate sync** — DONE 2026-05-18 (HEAD `c86fc9e`, VWA submodule `2f9b0b4`, per-task configs 910, B0 probe 5-gate PASS, VWA SBOM 4-match, Gemma Tier 1+2 + 13 invariant tests)
- ✅ **§A2 design-layer audit cascade** — 14/14 CLOSED via cross-AI /stress (A2.1-A2.10)
- ✅ **3 launch smokes** — B-1430 paper_grade (44/44 pytest) + §C LR feature step-0 (StepRecordV2 schema) + B-1425 Gemma Tier 3 (13 invariant tests + 4 DRY=1 wire combos)
- ✅ **Prep artifacts** — advisor email draft (Email 1 FYI only, Email 2 retired per OSF doctrine) + 2 audit walkthroughs + OSF deposit manifest committed `fa1f824`
- ⏳ **`preregistration.md` status `draft → locked`** — flip at fire event (1-min Edit + commit + git tag `preregistration-locked` + push --tags); substance fully RESOLVED per §A2 14/14 cascade; advisor email = optional FYI not blocker per B-1570
- ⏳ **`paper_drafts_locked/` snapshot** — `cp -r paper_drafts paper_drafts_locked` at fire event (1 sec)
- ⏳ **A100 final provenance snapshot** — `snapshot_env.py results/provenance/env_a100_lock.json` + `snapshot_vwa.sh results/provenance/vwa_a100_lock.json` at fire event (30 sec SSH)
- ⏭ **`queue_phase1_paper_grade.sh launch` ⭐ THE FIRE** — etc user `fire` signal
- ⏳ **#10 analysis 层 3-model 改造** (gates §D 不 gate launch; post-fire OK)

**Fire event sequence (~6 min wallclock, my execute + 1 user push ack)**:
1. `cp -r docs/checkpoints/paper_drafts docs/checkpoints/paper_drafts_locked` (1s)
2. SSH A100 → `snapshot_env.py` + `snapshot_vwa.sh` (30s)
3. Pull artifacts back to DGX + commit (10s)
4. Flip prereg frontmatter `status: draft → locked` + fill `registered_at` + `registered_git_sha` (10s)
5. `git commit -am "lock(prereg): Phase 1a Pass-1 fire-event lock + provenance snapshot"` (5s)
6. `git tag -a preregistration-locked -m "Phase 1a Pass-1 launch — $(date) + Git SHA <SHA>"` (1s)
7. `git push origin master + git push --tags` — needs user `push ok` ack per `feedback_git_push_requires_confirm` (~5s wait + 5s push)
8. SSH A100 → `setsid nohup bash scripts/queues/queue_phase1_paper_grade.sh launch &` (30s setup)
9. Watchdog Tier-1 file marker monitor + ntfy push notification setup (~5 min)

**Post-data-lands doc updates** (post Pass-1 baseline 完成): §4 P-prompt column / §1 hero numbers (phantom 4-fold drop-in actual numbers) / §8 limitations final / `compute_cost_carbon_table.md` numerical fill via aggregator / per-cell `validate_run.py --strict` / `make analysis` full pipeline.

---

## §2 OSF DOI 8-step lock workflow (B-1570 doctrine 2026-05-18)

**Trigger** (B-1570 update 2026-05-18, [[实验笔记 §220]] + `osf_lock_manifest.md §3` updated header):
- **Old**: "post advisor email reply" (deprecated; advisor email batch sign-off was originally required gate)
- **NEW**: "post Phase 1a Pass-1 baseline data complete" — advisor email reply 现 optional collateral, NOT gating event per `preregistration.md §6 §(a)` updated doctrine.

**Why doctrine shifted**: §A2 audit cascade (14/14 closed) substantively locked the 14 commit decisions via Git SHA refs at master HEAD; advisor sync 2026-05-14 directive ("student focus = experiment execution") shifted advisor's witness role from formal pre-data lock to post-fact OSF DOI mint completeness collateral. OSF DOI (public + immutable + cryptographic) is strictly stronger witness than advisor email (private + human-attested + non-cryptographic) → OSF DOI replaces advisor email's witness function.

**8 steps** (详 [[osf_lock_manifest]] §3):
1. **(optional, post-fire collateral)** Save advisor batch sign-off email PDF if/when advisor signs — NOT blocking per B-1570 doctrine
2. **Update `preregistration.md`** with confirmed thresholds + decision log entry
3. **Run `python3 scripts/provenance/snapshot_env.py`** on **A100 (paper-1 canonical)** + DGX (archive) + Myriad (optional)
4. **Run `bash scripts/provenance/snapshot_vwa.sh`** on each VWA-bearing host
5. **Snapshot paper drafts** → `cp -r paper_drafts paper_drafts_locked` + commit
6. **Tag git** → `git tag -a preregistration-locked` + push --tags
7. **Mint OSF DOI** at https://osf.io/registries/ — link OSF page to GitHub tag URL
8. **Backfill `osf_lock_manifest.md`** with SHAs + timestamps + DOI

**Steps 1-6 happen at fire event** (per §1 fire event sequence). **Steps 7-8 happen post-Pass-1-data-complete** (~30 min user-side via OSF UI + ~1-24h DOI 分配).

**Artifacts ready** (2026-05-18 fully pre-staged per [[osf_deposit_package_manifest_2026-05-18]]):
✅ 15 `pre_run/*.md` docs (preregistration + osf_lock_manifest + locked_versions + model_card + dataset_card + evaluator_change_protocol + reeval_audit_protocol + pre_rerun_audit + negative_results_registry + ethics_license_coi_statements + **NEW neurips_checklist + compute_cost_carbon_table** + release_redaction_checklist + topvenue_constraints + 2 walkthrough artifacts)
✅ `paper_drafts/section1-8 + paper.bib` (snapshot to `paper_drafts_locked/` at fire event)
✅ A100 snapshot scripts ready + previous A100 SBOM probe 2026-05-18 PASS (head + base + chain + lock all True)
✅ Reusable patch artifact provenance (B-440/B-91/B-535 per `osf_lock_manifest.md §2.5`)

### Advisor email logic (2026-05-18 clarified) — **2 emails 简化为 1**

| Email | 目的 | 是否需要 |
|---|---|---|
| **Email 1** — informal FYI to Maria + Zekun ("P79 实验启动了") | Supervision loop courtesy + collegial information flow | ✅ YES (send at fire event with Git SHA filled in) — see [[advisor_email_draft_2026-05-18]] |
| **Email 2** — formal pre-registration witness reply ask (advisor "I witness ... 14 lock decisions as of Git SHA <SHA>" 1-line reply) | OSF mint 前的 interim witness (pre-OSF era convention) | ❌ NO — **retired per B-1570 doctrine 2026-05-18**: OSF DOI (cryptographic public witness) supersedes advisor email witness function. `witnessed_by:` field in prereg frontmatter populated as "Git tag `preregistration-locked` + OSF DOI <to-be-assigned>" at OSF mint, replacing original "advisor name" plan |

---

## §3 Statistical methodology — advisor sync questions (deferred)

From §133 codex Round C + §134 /stress v6:

| Item | Question | 倾向 |
|---|---|---|
| C-M1 / F1 | DL meta τ² biased at k<10 (Veroniki 2016) — preregistration `decision 3A` 2026-05-14 retired DL in favor of FE estimand (no τ² needed). k=4 → k=6 per B2 addition 2026-05-14 eases but does not eliminate k<10 fragility under any RE estimator | RESOLVED via FE estimand (no advisor decision needed for DL replacement); k=6 power numbers ⏳ pending advisor lock |
| C-M2 / F2 | Wald 1.96 CI anti-conservative at k<10 (IntHout 2014) — FE estimand side-steps (FE Wald is sound at any k under CLT on per-cell θ_i). No Hartung-Knapp needed since no RE estimator in primary gate | RESOLVED via FE estimand 2026-05-14 |
| C-M2 | `power_analysis.md` rewrite for 4-cell Phase 1a scope (现仍 12/16 + N≥10 mismatch) | post-sync |
| C-1 / F4 | `aggregate_phantom_lift.py` denominator inconsistency (sr_3 universe_5 vs u_psom) — archived-data analysis, 影响 Appendix D | post-Phase-1a |
| C-2 | `aggregate_phantom_lift.py` H3 axis-2 universe — 可能 flip H3(ii) false negative | post-Phase-1a |

---

## §4 Audit follow-ups (DGX-side, independent)

| Pri | Item | Effort | Status |
|---|---|---|---|
| 🟡 R1 | Preflight v2 extension (B0 XOR B1 conflict / archive_subset checks) | 45 min | partially done §134 |
| 🟡 R4 | Stage 2B `--resume` flag for reboot recovery | 10 min | independent |
| 🟡 R6 | `check_evaluator_consistency.py` (Gate 7 evaluator_code_sha == lock-time SHA) | 30 min | OSF lock prep |
| 🟡 **B-1760** | **DOM mode `screenshot.png` regression — `obs.image=None` for accessibility_tree across 91/91 step records on Fire-3 cls B0 DOM.** Archive 2026-05-15 had it; logic byte-identical archive↔HEAD; runtime instrument needed. Trigger: post cls B0 SoM cell land (~5 days, SoM has its own screenshot path so isolates accessibility_tree regression). Acceptance: re-fire smoke6 / 10-task pilot before paper-grade re-fire, verify `screenshot.png` per step + `annotate_screenshots.py` produces `screenshot_annotated.png`. Paper §3 evidence layer NOT blocked (DOM trajectory `observation_dom.txt` + schema-v2 canonical fields all present); screenshot is audit-layer only. See §236.3 chronicle for full diagnostic trail. | 2 h | deferred — post cls B0 SoM land |
| 🟢 **R2-P2-10-C** | **Appendix E.3 temporal language fix** — `preregistration.md` Appendix E.3 says Phase 0 audit artifacts "witnessed alongside DOI 1 anchor" but artifacts are timestamped 2026-05-19 and DOI 1 minted 2026-05-18T23:10:06Z. Strictly post-DOI-1. Fix: rephrase to "post-DOI-1 forward disclosures, appending to the DOI 1 anchor without modifying its locked estimands". Gemini Mode C F5 / R2 round 2. Honesty surface, NOT a re-witness trigger. | 5 min prose | deferred — next /stress round or paper finalize |
| 🟢 **R2-P2-11-B** | **Schema 4-place sync test enumeration coverage** — `tests/test_schema_4place_sync.py:test_phase2_intervention_fields_present` only explicitly enumerates 4 step + 2 episode intervention fields. Other 18 Phase 2 fields (10 attempt-lineage + 8 footprint) covered indirectly via dataclass-sync only. Future drop of one field from `_EPISODE_OPTIONAL_FIELD_TYPES` still passes test. Fix: add `test_phase2_attempt_lineage_fields_present` + `test_phase2_footprint_fields_present` with explicit field sets. codex Mode B F6 / R2 round 2. Test-suite enhancement, no production impact. | 15 min | deferred — next /stress round or schema v3 prep |
| 🟢 N1 | Bonferroni / Holm correction paper §3 paragraph | 10 min | paper write phase |
| 🟢 N3 | Phantom variant FP rules | 1 h | post Phase 1a rerun |

---

## §5 Router — ⭐ Phase 1 并行核心线 (advisor 2026-05-14 收口)

> ⭐ **2026-05-14 收口**: router 从 "Section 6 / paper-2 deferred" 升为 **Phase 1 并行核心 contribution**. 两条基础路线并行做, 与 cls+red baseline clean run 同步.
>
> **执行 checklist + blockers + 完成判定** → [[phase1_plan]] §C

**路线 (a) rule-based router** — 根据 task 属性 / 任务区分来 route。
**路线 (b) learned router** — 训练一个 classifier 做 routing。
**未来扩展** — 根据不同 mode 的行为模式 route。

Routing signal infra (✅ ready): `confidence_summary.json` per-condition。
Train/test split protocol → 倾向 5-fold site-stratified CV (vs LOSO)。设计细节 → [[paper_planning#§8]]。

---

## §6 Sustainability / Green AI (Section 8 end-stage)

| Item | Status |
|---|---|
| fig regional carbon sensitivity (B1, 45 region) | ✅ done |
| B1 measured energy (cls + red × modes) | ✅ ready |
| Multi-metric Pareto (cost + lat + carbon) | ⏳ Section 8 前置 (~2h) |
| B0 token-based carbon estimator | ❌ optional Tier 3 future |
| Section 8 prose | ❌ paper end-stage |

---

## §7 Codex task queue

![[codex.base#Ready to send (now)]]

![[codex.base#Running / In flight]]

![[codex.base#Blocked / Queued]]

**Pending Python scripts (非 codex)**:
- ⏳ Multi-metric Pareto (cost + lat + carbon) — Section 8 前置 (~2h)
- ⏳ TF-IDF + binary feature extraction — Section 6 Tier 1 router 前置 (~1h)

---

## §8 Open issues

![[issues.base#Active blockers]]

![[issues.base#Backlog]]

---

## §9 Advisor align

详 [[issue_advisor_sync_2026-05-14]] (2026-05-14 sync — Part 1 novelty + Part 2 决策点). Sync 后:
decision log 写 [[paper_planning]] + framing decisions register → [[issue_advisor_sync_2026-05-14]] status open → discussed (ADVISOR_SYNC.md retired 2026-05-15).

---

## §10 References + quick links

### Phase 1 canonical
- [[phase1_plan]] — 统领性 audit/execute checklist (§A1 实现层 stress + §A2 设计层 stress + §B clean run + §C router + §D evidence + §E milestones)

### Paper drafts (final prose)
```
docs/checkpoints/paper_drafts/
  section1_intro.md          ✅
  section2_background.md     ✅ + paper.bib
  section3_definition.md     ✅
  section4_findings.md       🟡 待 P-prompt column + Phase 1a hero numbers
  section5_mechanism.md      ✅ v2 (probe-causal-steering trichotomy)
  section6_routing.md        ❌ paper-2 / Tier 1+2 prototype
  section7_generalization.md ❌ 待 WA
  section8_discussion.md     ❌ paper end-stage
```

### Figures
`results/phantom_paper/figures/` — regenerate via `make analysis` / `make figures`:
- §1 hook: fig0a_sr_per_mode_heatmap / fig0b_fp_rate_per_mode / fig0c_drop_one_oracle / fig0g_routing_auroc_heatmap / fig3b_image_token_gap
- §5 mechanism: fig_stage4_method42_v2_{cls,reddit} / fig_axis2_logit_lens_v2 / fig_axis2_layer_profile_v2 / fig_mech_8cell_l17_forest / fig_mech_real_vs_random / fig_layer_axis_emergence_v2_{cls,reddit} / fig_phantom_structure_venn

### Key infra paths
```
configs/exp_v2_*.yaml                              per-site experiment configs (含 12 baseline configs §133 A4)
scripts/queues/queue_phase1_paper_grade.sh         Phase 1a 36-cond orchestrator (§134 harden; expanded 24 → 36 per B2 addition 2026-05-14)
scripts/queues/queue_chain.sh                      sequential chain (§134 C3 crash-detect)
scripts/queues/queue_{baseline,phantom_*}.sh       per-condition launch (§134 FORCE_NEW)
scripts/analysis/preregistration_decision_test.py  H1/H3/TOST canonical (§133 T3 heterogeneity branch)
scripts/mechanistic/run_stage4_*.py                Stage 4 mechanism pipeline (v2 post-fix)
scripts/mechanistic/run_stage4_h1_{phi35,qwen2vl}.py  cross-family extraction (Bug 2/5 fixed §133)
scripts/provenance/snapshot_{env,vwa}.*            provenance fingerprint
p79/mechanistic/activation_patching.py             patching infra (layer-index convention documented)
```

### Provenance artifacts (paper-cite-able)
```
results/provenance/env_dgx_baseline.json           DGX baseline (HF Qwen3-VL-4B SHA ebb281ec...)
results/provenance/vwa_dgx_via_quark.json          VWA stack fingerprint
docs/checkpoints/pre_run/osf_lock_manifest.md      8-step DOI workflow
docs/checkpoints/pre_run/preregistration.md        R1-R5 framing rule + K-of-N transparency-only
```

---

## §11 独立 bug 研究 paper (workshop-targeted)

> advisor 2026-05-14 收口: bug 部分可**单独再发一篇 paper** 投 workshop — 独立于主 paper, **不替换**主 paper 的 workshop 节点.

**方向**: cross-benchmark bug 聚合研究, 针对现有 web agent benchmark.
**参考**: agisdk (https://github.com/agi-inc/agisdk).
**素材基础**: 项目 dual-track environment / VWA bug fix 工作 ([[实验笔记]] §109 dual-track 9-cell taxonomy / B-82 / `master_bug_catalog.md` 37+ bugs).
**状态**: 方向已 locked; 具体 scope + benchmark 选型 + 时间线待 planning.

---

> 📖 **Doc update workflow** (when X happens, update which docs) → [[paper_planning#§20]]
