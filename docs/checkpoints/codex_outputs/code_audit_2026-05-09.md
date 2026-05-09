# P79 Code Audit (Paper-Grade Bug Scan) — 2026-05-09 codex

## Executive Summary

- Total findings: 41
- Severity: BLOCKER × 14 / HIGH × 21 / MEDIUM × 6 / LOW × 0
- Files audited: 31
- Top-3 most-urgent fixes (one-line each)
  - F01/F02: stop aggregators from silently running on archived/pre-bug/no-cell registry state and clobbering paper outputs.
  - F20/F21/F23/F26: make `adjusted_success` fail-closed and parse multi-eval types so §4 FP-filtered SR is canonical.
  - F12/F16: fix Stage 2 layer indexing before interpreting L0/L35 or Holm-tested layers in §5 mechanism claims.

## Findings Table

| ID | File:line | Severity | Category | Issue (1 sentence) | Why it affects paper-grade | Suggested fix (1-2 lines) |
|---|---|---|---|---|---|---|
| F01 | scripts/analysis/lib/run_registry.py:38 | BLOCKER | 5 schema/config | Default paper aggregation includes `paper-grade-pre-bug` cells. | §1/§4/§6 aggregators can mix known pre-bug data with current cells by default. | Change `DEFAULT_GRADE_FILTER` to `["paper-grade"]`; require explicit `--include-pre-bug` for sensitivity only. |
| F02 | scripts/analysis/aggregate_phantom_lift.py:626 | BLOCKER | 3 silent fail | Empty registry/no valid cells writes an effectively empty `phantom_lift.csv` instead of failing. | A rerun with all cells archived can erase the input to all phantom-lift/meta figures while exiting success. | If `rows` is empty, raise `SystemExit` before writing; include skipped reasons and manifest grade summary. |
| F03 | scripts/analysis/aggregate_phantom_lift.py:209 | BLOCKER | 2 stat | TOST default uses `delta_pp=0.5` while the report prose says commit-locked `δ=1.0pp`. | §4 equivalence/nonzero conclusions are computed under a different margin than the locked method. | Set default to `1.0`; put the margin in a constant used by computation and markdown. |
| F04 | scripts/analysis/aggregate_phantom_lift.py:231 | BLOCKER | 2 stat | Bootstrap TOST p-value is directionally inverted/mislabeled for positive effects. | Strong positive lift produces large `p_upper`, so the table can fail the very nonzero/equivalence claim it says it tests. | Reimplement TOST explicitly: define whether the claim is equivalence or nonzero, compute both one-sided p-values analytically or by bootstrap tails matching that claim. |
| F05 | scripts/analysis/aggregate_phantom_lift.py:94 | HIGH | 3 silent fail | Corrupt per-episode summaries are skipped with `except Exception: continue`. | Missing successful/failed tasks silently changes oracle unions and pp lift. | Count corrupt files and fail for paper-grade cells; only allow `--allow-corrupt` for exploratory runs. |
| F06 | scripts/analysis/aggregate_phantom_lift.py:295 | HIGH | 3 silent fail | Modes below `MIN_EP_FOR_CELL` are silently dropped before required-mode checks. | Partial runs can change whether optional arms enter the denominator without an explicit audit trail. | Record skipped modes with counts; fail if any required or optional paper arm is below expected N unless `--partial`. |
| F07 | scripts/analysis/aggregate_phantom_lift.py:308 | BLOCKER | 2 stat | The common task universe intersects across all present modes, including optional P-prompt. | Adding a partial sixth arm can change the denominator for the primary 3-vs-5 lift even though P-prompt is not part of that estimand. | Compute a per-comparison universe from only the arms in that comparison; keep P-prompt out of 3-vs-5 denominators. |
| F08 | scripts/analysis/aggregate_phantom_meta.py:81 | BLOCKER | 2 stat | Random-effects meta-analysis has no B8 N>=10 per-cell floor. | Tiny/partial cells can enter pooled §4 estimates with deceptively precise SEs derived from bootstrap CIs. | Carry `n_common` from `phantom_lift.csv`; filter `n_common >= 10` and report excluded cells. |
| F09 | scripts/analysis/aggregate_phantom_meta.py:122 | HIGH | 2 stat | DL pooled CI always uses normal `1.96` even for very small k. | With k<5 cells, §4 meta CI/p-values can be anti-conservative and disagree with standard small-k meta-analysis. | Use SciPy/statsmodels meta-analysis if available or apply t/Knapp-Hartung small-sample adjustment; print method. |
| F10 | scripts/analysis/sensitivity_loo_meta.py:150 | BLOCKER | 2 stat | LOO table labels raw per-arm p-values as `holm_pass`. | Downstream of F4, the appendix can claim Holm robustness without applying the secondary-family m=3 correction. | For each LOO scenario, recompute all arms together and apply Holm within family before verdicts. |
| F11 | scripts/analysis/sensitivity_loo_meta.py:104 | HIGH | 5 schema/config | `parse_forest_csv(path)` ignores its `path` argument and hardcodes upstream `phantom_lift.csv`. | Passing `--input` gives a false sense that the sensitivity report used the requested frozen data. | Use the `path` parameter directly; rename the CLI input to `--forest-csv` if that is the intended source. |
| F12 | scripts/analysis/stage2_layer_significance.py:31 | BLOCKER | 7 layer indexing | Stage 2 significance labels L0 as embedding output, but the patching result L0 is transformer block 0 output. | §5 layer claims and Holm-tested layers are off by one relative to the stated Qwen3-VL convention. | Align `per_layer` schema with `HiddenStateExtractor` (`n_layers+1`) or relabel patch layers as block indices B0-B35. |
| F13 | scripts/analysis/stage2_layer_significance.py:140 | HIGH | 2 stat | Holm correction is applied only to t-test p-values while Wilcoxon is printed but never used as backup. | Non-normal/constant task effects can make the displayed non-parametric backup irrelevant to the decision. | Define the primary test; if Wilcoxon is backup, Holm-correct and gate on it when t-test is invalid or pre-specified. |
| F14 | scripts/analysis/stage2_layer_significance.py:169 | HIGH | 2 stat | Cross-direction comparison uses unpaired Welch tests even when 2x2 controls can share task IDs. | §5 asymmetry conclusions can lose pairing power or compare selection effects instead of task-matched effects. | Join forward/reverse rows by `task_id` where possible and use paired tests; only use Welch for disjoint task sets with a warning. |
| F15 | scripts/analysis/stage2_layer_significance.py:305 | MEDIUM | 1 stale claim | Comment says LD is negated internally, but code only switches `alternative`. | The report invites reviewers to trust a transformation that is not implemented, raising stale-claim risk after the recent B9 precedent. | Delete the stale negation note or actually materialize a signed disruption metric used consistently in tables. |
| F16 | p79/mechanistic/activation_patching.py:42 | BLOCKER | 7 layer indexing | `ActivationPatcher.n_layers` is `len(transformer blocks)` and excludes embedding output. | All Stage 2B/2C results are indexed as 0..35 but downstream text treats L0 as embedding and L35 as final baseline. | Add an explicit embedding/residual hook if L0 is required, or rename patch layers to block indices and adjust significance layers. |
| F17 | p79/mechanistic/activation_patching.py:357 | HIGH | 4 nondeterminism | Random-injection control uses `torch.randn_like` without a seed parameter in the library API. | Calling `patching_grid_continuation(..., randomize_source_hidden=True)` outside the CLI is not reproducible. | Add `random_seed`/`generator` to the function and use `torch.Generator(device=h.device).manual_seed(seed)`. |
| F18 | scripts/mechanistic/run_stage2b_continuation_pilot.py:299 | HIGH | 5 schema/config | Primary incremental results JSON omits `reverse`, `tier`, `random_inject`, and `random_seed`. | Stage 2 significance consumes this JSON and cannot reconstruct which causal/control cell produced the numbers. | Include full CLI provenance in every incremental save, not only `env_snapshot`/manifest. |
| F19 | scripts/mechanistic/run_stage2b_continuation_pilot.py:390 | HIGH | 1 stale claim | Reverse runs still print source as clean/image and target as mirage/no-image. | §5 summaries can invert the causal direction in prose while the data were actually swapped. | Build display labels after the `if args.reverse` swap from actual `source_inputs`/`target_inputs` roles. |
| F20 | p79/experiment/analysis.py:80 | BLOCKER | 6 FP filter | Canonical FP filter compares `eval_type == "string_match"` / `"program_html"` exactly. | Runner and diagnostics encode multi-eval tasks as `string_match|program_html`, so eval FPs can survive into primary adjusted SR. | Parse eval types as a set: split on `|`, accept lists, and test membership. |
| F21 | p79/experiment/analysis.py:117 | HIGH | 5 schema/config | `bool(r["agent_finished"])` treats `NaN` as `True`. | Legacy rows with missing `agent_finished` can be considered actively finished, bypassing `na_fp`/`eval_fp`. | Use `pd.notna` before bool conversion; pass `None` for missing values. |
| F22 | p79/experiment/analysis.py:101 | BLOCKER | 6 FP filter | Batch fast-path trusts any non-null `adjusted_success` without validating `fp_reason` values. | Legacy or visual-FP-era summaries can bypass the locked `{'', 'na_fp', 'eval_fp'}` primary policy. | Validate `fp_reason` set and schema version; recompute or fail when invalid/deprecated reasons appear. |
| F23 | p79/experiment/runner/main.py:1593 | HIGH | 3 silent fail | Runner catches all adjusted-success errors and writes `adjusted_success=None`. | Downstream scripts commonly fall back to raw `success`, so FP filtering can silently disappear for an episode. | Treat adjusted-success failure as fatal for paper-grade runs; at minimum set `fp_reason="adjustment_error"` and fail validation. |
| F24 | p79/experiment/io_utils.py:43 | HIGH | 3 silent fail | Corrupt JSONL lines are dropped and analysis continues. | Step-count, final-action, and `agent_finished` reconstruction can be wrong while only logging a warning. | Return corrupt-line counts and make strict analysis/validation fail when any corrupt line exists. |
| F25 | p79/experiment/io_utils.py:24 | MEDIUM | 5 schema/config | Restart dedup only detects valid `step_idx == 0` resets. | If the first line of a restarted run is corrupt or resume starts at nonzero, old and new steps can be mixed. | Dedup by run/session identifier when present; otherwise fail strict mode if reset evidence is ambiguous. |
| F26 | scripts/analysis/aggregate_sr_fp_per_mode.py:79 | BLOCKER | 6 FP filter | SR+FP aggregation falls back to raw `success` when `adjusted_success` is absent. | Primary §4 adjusted SR can silently include unfiltered legacy runs. | Require `adjusted_success`/`fp_reason` for paper-grade cells; recompute with canonical function if missing. |
| F27 | scripts/analysis/analyze_confidence_calibration.py:2242 | HIGH | 6 FP filter | If site detection fails, calibration sets adjusted labels equal to raw labels. | §6 routing AUROC can train/evaluate on labels that skipped the locked FP filter. | Fail closed when `benchmark_site` is unavailable; require caller to supply site/benchmark. |
| F28 | scripts/analysis/analyze_cross_representation.py:891 | HIGH | 3 silent fail | Wilcoxon failures in cross-representation cost tests are swallowed with `pass`. | Missing paired-test output is indistinguishable from “no significant difference” in summaries. | Append a structured warning row with comparison, metric, and exception. |
| F29 | scripts/analysis/aggregate_cross_site.py:81 | MEDIUM | 5 schema/config | Site inference checks `"shopping"` before `"shopping_admin"`. | WA/VWA shopping-admin runs can be mislabeled as shopping in cross-site tables. | Check longest/specific site names first, or parse `benchmark_site` from summaries before run names. |
| F30 | scripts/analysis/aggregate_cross_site.py:137 | HIGH | 3 silent fail | FP stats loading catches exceptions and returns `{}`. | Cross-site adjusted/raw comparisons can omit FP counts without failing or warning. | Log/fail on unreadable `cross_representation_summary.json` in strict mode. |
| F31 | scripts/analysis/aggregate_routing_auroc.py:139 | HIGH | 1 stale claim | Summary says “AUROC >= baseline” but code only reports each mode's maximum AUROC. | §6 can claim phantom signals beat DOM/SoM/Vision without computing the baseline contrast or uncertainty. | Add explicit baseline-mode comparison rows and CIs/p-values for the stated claim; soften prose until then. |
| F32 | scripts/analysis/preregistration_decision_test.py:1 | BLOCKER | 1 stale claim | Script still declares itself the canonical preregistration decision test around K-of-N rules. | Downstream of B9, K-of-N was reframed as secondary transparency; this file can resurrect the stale primary decision rule. | Mark as deprecated or rewrite around random-effects meta + locked TOST; include B9 note in header and output. |
| F33 | scripts/analysis/preregistration_decision_test.py:135 | MEDIUM | 2 stat | TOST implementation tests relative cost equivalence, not phantom lift equivalence/nonzero. | The name and CLI `TOST_delta=1.0` can be confused with the §4 phantom-lift TOST lock. | Rename to `evaluate_cost_tost` and keep lift TOST in the phantom-lift aggregator with distinct constants. |
| F34 | scripts/maintenance/glm/glm_cell_autoupdate.py:212 | HIGH | 5 schema/config | `latest_match` always prefers finalized summaries over newer in-flight runs. | Cron status can miss a rerun until it finalizes, leaving cells marked done against stale archived data. | Sort primarily by outer run recency/start time, then prefer finalized only within the same run ID. |
| F35 | scripts/maintenance/glm/glm_cell_autoupdate.py:352 | HIGH | 3 silent fail | Cell status flips to done based on episode count even without `condition_summary_v2.json`. | Status notes can declare a paper-grade cell done before final summaries and SR fields exist. | Require a finalized summary for `status=done`; use `episodes>=expected` only for `active_complete_pending_summary`. |
| F36 | scripts/maintenance/glm/myriad_watcher.py:96 | HIGH | 3 silent fail | SSH/qstat failure exits 0 silently and does not notify. | A broken Myriad watcher can hide failed/running jobs, undermining rerun provenance. | Persist failure count and notify/fail after N consecutive SSH failures. |
| F37 | scripts/maintenance/glm/glm_pre_launch_check.py:83 | HIGH | 3 silent fail | GLM pre-launch check allows launch when the GLM call fails. | The gate advertised as contamination protection is bypassed by network/API errors. | Return BLOCK/WARN on review failure for paper-grade launches, or require `--allow-glm-fail`. |
| F38 | Makefile:134 | MEDIUM | 5 schema/config | `pre-launch-check` only checks tracked dirty files, not untracked files. | Untracked source/config changes can affect a run while the gate prints “clean.” | Add `test -z "$$(git status --porcelain --untracked-files=all)"` or equivalent. |
| F39 | Makefile:154 | MEDIUM | 4 nondeterminism | Seed gate greps only `configs/exp_v2_base.yaml`, not the actual launch config. | A paper run can override the seed while pre-launch still passes. | Accept `CONFIG=<path>` and validate the resolved config seed used by the queue. |
| F40 | scripts/analysis/figures/fig1ab_cascade_diamond.py:30 | HIGH | 5 schema/config | Figure reads hardcoded 202604 archived run directories instead of the registry. | Paper figures can be regenerated from archived/pre-bug runs even after registry changes. | Replace hardcoded `STEP_DIRS` with `get_cells(..., grade="paper-grade")`; fail if cells missing. |
| F41 | scripts/analysis/figures/fig_capability_b0_b1.py:101 | BLOCKER | 1 stale claim | Plot annotation hardcodes `+43.7 pp` instead of using the parsed `shift` value. | This exactly matches the stale-claim pattern: generated figure text can contradict the source table. | Use `f"{shift[highlight]:+.1f} pp"` and assert it matches the highlighted row. |

## Per-finding detail

### F01 — scripts/analysis/lib/run_registry.py:38 — BLOCKER
Excerpt:
```python
Grade = Literal["paper-grade", "paper-grade-pre-bug", "in-flight", "archived"]
GRADES = ("paper-grade", "paper-grade-pre-bug", "in-flight", "archived")
DEFAULT_GRADE_FILTER: list[Grade] = ["paper-grade", "paper-grade-pre-bug"]
```
The default registry filter includes cells explicitly named `paper-grade-pre-bug`. Because `aggregate_phantom_lift.py`, `aggregate_routing_auroc.py`, figures, and status scripts use `get_cells()`/`get_run_dirs_paper_vwa()` without an explicit stricter grade, pre-bug data can enter primary §1/§4/§6 outputs by default. This is a config drift blocker, especially downstream of C10 gates and the B9 stale-claim precedent. Fix by making `paper-grade` the only default and requiring an explicit sensitivity flag for pre-bug cells.

### F02 — scripts/analysis/aggregate_phantom_lift.py:626 — BLOCKER
Excerpt:
```python
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
    w.writeheader()
    w.writerows(rows)
```
If the registry yields no valid cells, the script still writes the primary `phantom_lift.csv`. Given the current manifest can intentionally archive cells, this can clobber the input to meta-analysis and figures while exiting normally. Paper-grade analysis should fail loudly when zero cells are eligible. Raise `SystemExit` before writing and print a manifest/skip diagnostic.

### F03 — scripts/analysis/aggregate_phantom_lift.py:209 — BLOCKER
Excerpt:
```python
def bootstrap_tost_p(in_a: np.ndarray, in_b: np.ndarray,
                     delta_pp: float = 0.5, B: int = 1000, seed: int = 42
                     ) -> Optional[float]:
```
The function computes with `0.5pp`, but the same file later prints “TOST p ... at δ=1.0pp (commit-locked)” at line 665. This is a direct computation/prose contradiction for §4 phantom lift. Use a single constant, set it to the locked `1.0pp`, and reference that constant in the markdown. Cross-reference: this is the same stale-claim class as B9 `power_analysis.py`.

### F04 — scripts/analysis/aggregate_phantom_lift.py:231 — BLOCKER
Excerpt:
```python
p_lower = float(np.mean(lifts <= -delta_pp))
p_upper = float(np.mean(lifts >= delta_pp))
return max(p_lower, p_upper)
```
The prose says `p < alpha` rejects equivalence/nonzero, but for a clearly positive lift above `delta_pp`, `p_upper` approaches 1. This makes the reported TOST significance move opposite the substantive phantom-lift claim. Decide whether the test is equivalence (`|lift| < delta`) or practical nonzero (`lift > delta`) and compute the corresponding one-sided tail(s). Until fixed, TOST columns should not be used in paper text.

### F05 — scripts/analysis/aggregate_phantom_lift.py:94 — HIGH
Excerpt:
```python
try:
    rec = json.loads(p.read_text())
except Exception:
    continue
if rec.get("adjusted_success", rec.get("success", False)):
```
Any unreadable episode summary silently disappears from both observed and success sets. A single corrupt summary can alter oracle unions and pp lift while leaving only no visible warning. This is dangerous because §4 lift is task-set based. Count corrupt summaries and fail for paper-grade cells unless an explicit exploratory flag is passed.

### F06 — scripts/analysis/aggregate_phantom_lift.py:295 — HIGH
Excerpt:
```python
if len(o) < MIN_EP_FOR_CELL:
    # Skip undersized modes silently; allow rest of cell to still build.
    continue
```
The code drops undersized modes and then proceeds with whatever remains. That can make a cell look valid for a subset of arms and can change optional-arm behavior without a durable warning. Paper-grade output should list every missing/partial mode and fail when expected-N policy is violated. This affects §1/§4 routing-lift tables.

### F07 — scripts/analysis/aggregate_phantom_lift.py:308 — BLOCKER
Excerpt:
```python
# Common observed universe (intersection across all present modes)
common = set.intersection(*obs.values())
n = len(common)
```
The denominator for the primary 3-mode vs 5-mode lift is computed across all present modes, including P-prompt when it happens to be present. That means the sixth arm can change the primary 3-vs-5 estimand even though it is not part of that comparison. Compute `common` separately for each comparison family. This can directly corrupt §4 pp lift and meta inputs.

### F08 — scripts/analysis/aggregate_phantom_meta.py:81 — BLOCKER
Excerpt:
```python
paired = [(t, s) for t, s in zip(thetas, ses) if t is not None and s is not None and s > 0]
if len(paired) == 0:
    return None
```
B8 locks random-effects meta-analysis to N>=10 per cell, but this function only filters by nonzero SE. Since `load_per_cell_data()` does not carry `n_common`, tiny or partial cells can be pooled. Add `n_common` to the per-cell record and enforce the floor before `derslong_laird_meta()`. This affects §4 pooled phantom lift and the forest plots.

### F09 — scripts/analysis/aggregate_phantom_meta.py:122 — HIGH
Excerpt:
```python
ci_lo = theta_re - 1.96 * se_re
ci_hi = theta_re + 1.96 * se_re
z = theta_re / se_re if se_re > 0 else None
```
The random-effects CI and p-value use a large-sample normal approximation even when the notes admit k<5 cells. With very few cells, DL normal intervals can be too narrow. Use a validated meta-analysis implementation or a small-sample adjustment and state the method. This is downstream of F4 LOO/meta robustness concerns.

### F10 — scripts/analysis/sensitivity_loo_meta.py:150 — BLOCKER
Excerpt:
```python
base["holm_pass"] = base.get("p_one_sided", 1.0) < holm_alpha
...
loo["holm_pass"] = loo.get("p_one_sided", 1.0) < holm_alpha
```
The LOO appendix labels raw one-sided p-values as Holm decisions. The markdown later says the primary aggregator applies the secondary-family m=3 Holm correction, but the verdict logic does not reproduce it. This can falsely mark an arm robust after single-cell removal. Recompute the whole family for each LOO scenario and use adjusted p-values in the verdict.

### F11 — scripts/analysis/sensitivity_loo_meta.py:104 — HIGH
Excerpt:
```python
def parse_forest_csv(path: Path) -> dict[str, list[dict]]:
    ...
    forest_csv = REPO / "results/phantom_paper/phantom_lift.csv"
    if not forest_csv.exists():
```
The function accepts `path` but ignores it. A user can pass `--input` and receive a report generated from the live default file instead. This breaks reproducibility for appendix sensitivity runs. Use the provided `path` and fail if it is not the expected wide forest CSV.

### F12 — scripts/analysis/stage2_layer_significance.py:31 — BLOCKER
Excerpt:
```python
# Tested layers (vs L35 baseline). L0 is embedding output (often near-target);
# L35 is final post-block (output should == target by construction).
TEST_LAYERS = [0, 5, 11, 17, 23, 29]
```
The significance script assumes L0 is embedding output. However Stage 2B `per_layer` comes from `ActivationPatcher`, which registers hooks only on transformer blocks. This makes every §5 layer label ambiguous or off by one. Fix jointly with F16 before making mechanism claims or Holm-corrected layer statements.

### F13 — scripts/analysis/stage2_layer_significance.py:140 — HIGH
Excerpt:
```python
raw_pvals.append(t_p_one)
...
holm = holm_correct(raw_pvals)
```
The script computes Wilcoxon p-values but never lets them affect the Holm decision. If the t-test is invalid, non-normal, or mismatched to the task distribution, the “backup” is only decorative. Pre-specify the primary test and implement the backup decision rule. This affects §5 mechanism significance.

### F14 — scripts/analysis/stage2_layer_significance.py:169 — HIGH
Excerpt:
```python
fwd_diff = fwd_grid[:, layer] - fwd_grid[:, BASELINE_LAYER]
rev_diff = rev_grid[:, layer] - rev_grid[:, BASELINE_LAYER]
t_stat, t_p_two = stats.ttest_ind(fwd_diff, rev_diff, equal_var=False)
```
Cross-direction comparison is always unpaired. For the 2x2 controls, same task IDs can exist across forward and reverse conditions; ignoring pairing wastes power and can confound task selection with direction. Join by `task_id` where possible and use paired tests. Use Welch only for explicitly disjoint sets and say so in output.

### F15 — scripts/analysis/stage2_layer_significance.py:305 — MEDIUM
Excerpt:
```python
# directionality of "less than baseline" inverts. We flip sign of diff
# internally via the metric name handling — but the test is paired so the
# interpretation needs care. To keep consistent with overlap interpretation,
```
The code does not negate the LD metric; it only switches `alternative` to `"greater"`. This is stale interpretation text in exactly the class requested for broad audit. The table values may be numerically fine, but the prose claims a transformation that is absent. Remove the note or compute an explicit signed disruption metric.

### F16 — p79/mechanistic/activation_patching.py:42 — BLOCKER
Excerpt:
```python
self.layers = get_transformer_layers(model)
self.n_layers = len(self.layers)
```
Activation patching exposes 36 transformer block hooks, not the 37 hidden-state sequence with embedding output used by `HiddenStateExtractor`. Downstream scripts and plots call layer 0 “embedding” and layer 35 “final baseline,” so the mechanism story can be mislabeled. Either include an embedding hook or relabel patch layers as block outputs. Cross-reference: this is the root cause for F12 and the G8 heterogeneity figure label drift.

### F17 — p79/mechanistic/activation_patching.py:357 — HIGH
Excerpt:
```python
std = h.std()
noise = _torch_for_random.randn_like(h) * std + mean
randomized.append(noise)
```
The library-level random injection has no seed argument. The CLI currently seeds before calling it, but direct use or future scripts will produce nondeterministic Cell E controls. Add a generator/seed parameter to the function and store it in returned metadata. This protects §5 random-injection reproducibility.

### F18 — scripts/mechanistic/run_stage2b_continuation_pilot.py:299 — HIGH
Excerpt:
```python
json.dump({
    "config": {
        "site": args.site, "n_tasks": args.n_tasks, "step": args.step,
        "max_new_tokens": args.max_new_tokens,
```
The main results file consumed by Stage 2 analysis omits `reverse`, `tier`, `random_inject`, and `random_seed`. Those fields exist in other sidecars but not in the canonical `patching_continuation_results.json`. If the JSON is copied alone, the analysis cannot tell whether it is a forward, reverse, or random-injection control. Put full provenance into the results JSON at every incremental save.

### F19 — scripts/mechanistic/run_stage2b_continuation_pilot.py:390 — HIGH
Excerpt:
```python
- Source: `{args.source_mode}` (with image — clean) / Target: `{args.target_mode}` (no image — mirage)
- Direction: {"reverse (target→source)" if args.reverse else "forward (source→target)"}
```
For `--reverse`, the code swaps `source_inputs` and `target_inputs`, but the summary still describes source as clean/image and target as mirage/no-image. This is a stale interpretation block in a paper-facing mechanism summary. Build source/target labels after the swap and include both original modes and actual patched direction.

### F20 — p79/experiment/analysis.py:80 — BLOCKER
Excerpt:
```python
if agent_finished is not None and not agent_finished:
    if eval_type == "string_match":
        return (False, "eval_fp")
    if eval_type == "program_html":
```
The canonical FP filter expects exact eval-type strings. Runner and diagnostics construct composite strings such as `string_match|program_html`, so those rows do not match either branch. This under-filters `eval_fp` and directly corrupts §4 primary adjusted SR. Parse eval types as a list/set and use membership tests.

### F21 — p79/experiment/analysis.py:117 — HIGH
Excerpt:
```python
agent_finished=bool(r["agent_finished"]) if "agent_finished" in r.index else None,
eval_type=str(r["eval_type"]) if "eval_type" in r.index else None,
has_effective_action=bool(r["has_effective_action"]) if "has_effective_action" in r.index else None,
```
In pandas, `bool(np.nan)` is `True`. If legacy rows contain the column but have missing values, the batch path can treat unknown `agent_finished` as actively finished and skip FP downgrades. Use `pd.notna()` guards and pass `None` for missing. This matters for post-rerun JSON with evolving schema fields.

### F22 — p79/experiment/analysis.py:101 — BLOCKER
Excerpt:
```python
if (
    not ep_df.empty
    and "adjusted_success" in ep_df.columns
    and "fp_reason" in ep_df.columns
    and ep_df["adjusted_success"].notna().all()
):
    return ep_df
```
The fast path accepts any existing adjusted labels and does not validate `fp_reason`. A legacy row with `visual_fp` or another deprecated reason can bypass the §95 lock. Validate `fp_reason` against `{"", "na_fp", "eval_fp"}` and recompute/fail otherwise. This is the main FP consistency blocker.

### F23 — p79/experiment/runner/main.py:1593 — HIGH
Excerpt:
```python
except Exception as _adj_exc:
    logger.warning(...)
    episode_summary.setdefault("adjusted_success", None)
    episode_summary.setdefault("fp_reason", "")
```
If adjusted-success computation fails, the runner writes a summary that downstream code often treats as raw success. This is a fail-open path for the primary FP filter. In paper-grade mode, this should be fatal. If kept nonfatal for exploratory runs, set a distinct error reason and make `validate-strict` reject it.

### F24 — p79/experiment/io_utils.py:43 — HIGH
Excerpt:
```python
except json.JSONDecodeError:
    logger.warning("Dropped corrupt JSONL line %d in %s: %.100s", line_num, path, line)
    continue
```
Corrupt step lines are discarded. That can change final-action reconstruction, restart dedup, and `agent_finished` evidence while only logging a warning. Return both records and corrupt-line count, then make strict analysis fail on any corrupt line. This is a silent-failure risk for §4 FP filtering and mechanism diagnostics.

### F25 — p79/experiment/io_utils.py:24 — MEDIUM
Excerpt:
```python
for i, rec in enumerate(file_lines):
    if i > 0 and rec.get("step_idx", -1) == 0:
        last_run_start = i
return file_lines[last_run_start:]
```
Restart dedup depends entirely on a valid reset line with `step_idx == 0`. If the reset line is corrupt or a resumed retry starts at a nonzero step, old and new run fragments remain mixed. Add a run/session ID to step records or fail strict mode when dedup cannot prove one run. This is medium because the current runner usually starts at step 0.

### F26 — scripts/analysis/aggregate_sr_fp_per_mode.py:79 — BLOCKER
Excerpt:
```python
raw = bool(row.get("success", False))
adjusted = bool(row.get("adjusted_success", row.get("success", False)))
```
The standalone SR/FP aggregator falls back to raw success when adjusted labels are missing. That is the opposite of fail-closed behavior for the locked §4 primary outcome. Require adjusted labels for paper-grade cells or recompute with the canonical function. Also validate `fp_reason` values before counting.

### F27 — scripts/analysis/analyze_confidence_calibration.py:2242 — HIGH
Excerpt:
```python
if not sites:
    print("  ⚠ Cannot detect benchmark_site from summaries; skipping adjustment.")
    ep_df["raw_success"] = ep_df["success"]
    ep_df["adjusted_success"] = ep_df["success"]
```
When site detection fails, routing calibration silently proceeds with raw labels. That can contaminate §6 AUROC inputs with false positives. Require site/benchmark as CLI arguments or fail when no site is inferable. Do not synthesize adjusted labels equal to raw labels in paper-grade mode.

### F28 — scripts/analysis/analyze_cross_representation.py:891 — HIGH
Excerpt:
```python
try:
    stat, p = scipy_stats.wilcoxon(ca[valid], cb[valid])
    stat_tests[f"{ma}_vs_{mb}"] = {...}
except Exception:
    pass
```
A failed paired test disappears from the summary. Readers cannot distinguish “test not run” from “no effect.” Add a warning/error table with the exception and comparison name. This affects cost-at-success and cross-mode statistical reporting.

### F29 — scripts/analysis/aggregate_cross_site.py:81 — MEDIUM
Excerpt:
```python
for site in ("classifieds", "reddit", "shopping"):
    if site in run_id.lower() or site in run_dir.name.lower():
        return site
```
`shopping_admin` contains `shopping`, so this inference can return the wrong site. That is a schema drift bug for WA/VWA cross-site outputs. Check `shopping_admin` before `shopping` or prefer explicit `benchmark_site` from summaries. This is medium because the priority paper cells are cls/red, but the scope includes broader automation.

### F30 — scripts/analysis/aggregate_cross_site.py:137 — HIGH
Excerpt:
```python
for p in run_dir.glob(pattern):
    try:
        return _read_json(p)
    except Exception:
        pass
return {}
```
Failure to read FP stats is silently converted to an empty dict. Cross-site tables can omit FP information while looking complete. For paper-grade aggregation, unreadable FP summaries should be a hard failure. At minimum, propagate a structured warning into the output JSON/markdown.

### F31 — scripts/analysis/aggregate_routing_auroc.py:139 — HIGH
Excerpt:
```python
summary_lines = [
    "# Routing signal AUROC summary — max per (baseline, site, mode)",
    "",
    "Section 6 claim: AUROC ≥ baseline (DOM/SoM/Vision) for Phantom modes.",
```
The script reports the maximum AUROC within each mode but never computes phantom-vs-baseline contrasts. The prose states a §6 comparative claim that the table does not test. Add explicit baseline comparison rows with uncertainty, or change the summary to “max signal per mode” only. This is a stale-claim risk for routing results.

### F32 — scripts/analysis/preregistration_decision_test.py:1 — BLOCKER
Excerpt:
```python
"""Preregistration decision test — H1 / H3 / TOST canonical implementation.

Single source of truth for the paper §5 / Table 5 decision rules.
```
This file still presents K-of-N as the canonical decision framework. The user noted B9 reframed K-of-N as secondary transparency after the stale `power_analysis.py` bug. Leaving this script active risks reintroducing the old primary rule in paper tables. Mark it deprecated or rewrite it around the current random-effects meta and locked TOST policy.

### F33 — scripts/analysis/preregistration_decision_test.py:135 — MEDIUM
Excerpt:
```python
def evaluate_tost(cells: list[dict], delta_pp: float) -> dict:
    """TOST (two one-sided tests) for cost equivalence within ±delta_pp.
```
This TOST is for relative cost equivalence, while the phantom-lift scripts discuss TOST for pp lift/equivalence/nonzero. The shared name and `TOST_delta=1.0` CLI wording can confuse prereg decisions. Rename this to `evaluate_cost_tost` and keep lift TOST separate. This is a medium stale-method risk.

### F34 — scripts/maintenance/glm/glm_cell_autoupdate.py:212 — HIGH
Excerpt:
```python
def sort_key(m: dict):
    if m["summary_path"]:
        return (1, m["summary_path"].stat().st_mtime)
    return (0, m["cond_dir"].stat().st_mtime)
return max(matches, key=sort_key)
```
An older finalized run always beats a newer in-flight rerun because finalized matches sort with leading `1`. That can keep cell notes pointing at stale data until the rerun writes a final summary. Sort by run recency first and only prefer summaries within the same run. This affects cron provenance and rerun tracking.

### F35 — scripts/maintenance/glm/glm_cell_autoupdate.py:352 — HIGH
Excerpt:
```python
if episodes >= expected_n and new_fm.get("status") != "done":
    new_fm["status"] = "done"
    new_fm["finalized_at"] = datetime.now(timezone.utc).date().isoformat()
```
For in-flight matches, `episodes` is just a count of episode summaries and `sr` can be `None`. The cell can become `done` before `condition_summary_v2.json` exists, leaving incomplete metrics. Require a finalized summary before status `done`; otherwise use a distinct pending-finalization state. This affects the C10 automation gate context.

### F36 — scripts/maintenance/glm/myriad_watcher.py:96 — HIGH
Excerpt:
```python
stdout = ssh_chain("qstat -u ucab352")
if stdout is None:
    return 0
```
The watcher intentionally exits success on SSH/qstat failure. Transient silence is acceptable once, but repeated failures hide job states and failed reruns. Persist a failure counter and notify after a threshold. For paper-grade cron, repeated watcher failure should be visible.

### F37 — scripts/maintenance/glm/glm_pre_launch_check.py:83 — HIGH
Excerpt:
```python
try:
    raw = _call_glm_chat(glm_cfg, messages, timeout_s=60)
except Exception as e:
    return True, f"(GLM call failed: {e}, allowing launch)"
```
The pre-launch GLM gate allows launch when the review call fails. Since the prompt calls same-site B0/B1 and reset violations “paper-grade contamination,” this is a fail-open control. Return WARN/BLOCK on GLM failure for paper-grade launches unless an explicit override is supplied.

### F38 — Makefile:134 — MEDIUM
Excerpt:
```make
@git diff-index --quiet HEAD -- 2>/dev/null || (echo "❌ git working tree has uncommitted changes"; exit 1)
```
`git diff-index` does not catch untracked files. A new untracked config or Python file can influence a run while pre-launch reports “clean.” Use `git status --porcelain --untracked-files=all` or also check `git ls-files --others --exclude-standard`. This is a C10 gate completeness issue.

### F39 — Makefile:154 — MEDIUM
Excerpt:
```make
@echo "6. Seed configured in base config..."
@grep -q "seed: 42" configs/exp_v2_base.yaml || (echo "❌ seed=42 not in configs/exp_v2_base.yaml"; exit 1)
```
The gate validates the base config, not the resolved launch config. A queue-specific config can override the seed and still pass. Accept `CONFIG=<path>` and parse the actual YAML used by the launch. This is medium because runner seeding exists, but the gate can certify the wrong file.

### F40 — scripts/analysis/figures/fig1ab_cascade_diamond.py:30 — HIGH
Excerpt:
```python
STEP_DIRS = {
    "reddit": {
        "DOM": RESULTS / "B0_3mode_reddit_20260422/phase1_dom_router_0/episodes",
        "P-text": RESULTS / "B0_phantom_text_reddit_20260427/phase1_phantom_dom_router_0/episodes",
```
The figure bypasses `run_registry.py` and hardcodes old run directories. Regenerating paper figures after a rerun can still draw from archived/pre-bug data. Replace hardcoded paths with registry lookups and fail if required paper-grade cells are unavailable. This affects §5 mechanism visuals.

### F41 — scripts/analysis/figures/fig_capability_b0_b1.py:101 — BLOCKER
Excerpt:
```python
ax.annotate(
    "+43.7 pp",
    xy=(highlight + width / 2, b1[highlight]),
```
The figure parses the shift values from `disagreement_clusters.md`, but the highlighted annotation is hardcoded. If the source table changes, the bar label can contradict the computed data, exactly matching the stale-claim bug pattern from B9. Replace with `f"{shift[highlight]:+.1f} pp"` and assert that the highlighted pattern exists. This affects §5 capability/representation claims.

DONE: code_audit_2026-05-09.md (41 findings, 14 BLOCKER, 21 HIGH)
