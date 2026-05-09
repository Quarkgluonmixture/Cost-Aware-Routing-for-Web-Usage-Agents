# Codex Prompt — Paper-Grade Code Audit (broad scan)

## Goal

Audit the P79 codebase for **paper-grade bugs that could corrupt published numbers**. Output a prioritized findings table that the human + Claude can triage and fix.

This is a *broad* audit (~20 analysis scripts + mechanistic pipeline + cron automation). Trade depth-per-file for breadth-of-coverage. The paper depends on these scripts producing reproducible, defensible numbers, so even subtle bugs (off-by-one, wrong stat direction, hardcoded interpretation) are blockers.

**Recent precedent**: in this same session (2026-05-09), `scripts/analysis/power_analysis.py` was found to have a **stale interpretation block** claiming ">0.95 family power at p_per=0.65" while the same file's K-of-N table showed actual=0.289. That bug was dormant for weeks until B9 audit caught it. There are likely 3-5 more such stale-claim / off-by-one / silent-fail bugs in the codebase. Your job: find them all.

## Scope (audit these directories, in priority order)

1. **`scripts/analysis/*.py`** (~20 files) — paper number generation. **Highest priority**.
   - `aggregate_phantom_lift.py` / `aggregate_phantom_meta.py` (phantom routing pp lift)
   - `power_analysis.py` (now fixed; verify no other instances of stale claim pattern)
   - `sensitivity_loo_meta.py` (just written, double-check DL implementation matches scipy meta-analysis if available)
   - `stage2_layer_significance.py` / `stage2_heterogeneity_figure.py`
   - `aggregate_routing_auroc.py` / `compute_adjusted_success.py` (FP filter logic)
   - All other `scripts/analysis/*.py` files
2. **`p79/experiment/analysis.py`** — `compute_adjusted_success()` canonical, must match preregistration §4 (na_fp + eval_fp combined PRIMARY, visual_fp deprecated)
3. **`p79/experiment/io_utils.py`** — `read_jsonl_dedup` (restart-dedup correctness, corrupt line handling)
4. **`scripts/mechanistic/*.py`** + **`p79/mechanistic/*.py`** — Stage 2 patching pipeline. Especially `run_stage2b_continuation_pilot.py` and `activation_patching.py`
5. **`scripts/maintenance/glm/*.py`** — cron automation; `myriad_watcher.py` SSH chain logic; `glm_cell_autoupdate.py` re-run detection logic
6. **`Makefile`** — `pre-launch-check` and `validate-strict` gates (just added 2026-05-09 in C10)
7. **`p79/experiment/runner/*.py`** — main orchestrator; FP filter integration
8. Skip: tests/, docs/, deprecated dirs, queue scripts (those are shell)

## Bug patterns to look for (with paper-grade priority)

### Category 1 — Stale interpretation / hardcoded claim contradicting computed value
**Paper-grade severity: BLOCKER**

Like the power_analysis bug: a Markdown / docstring / print statement claims X but a same-file computation produces Y. Look for:
- f-string / `.format()` / `print()` / appended `lines` lists that have *literal numbers* in interpretation prose
- "TODO" / "XXX" / "FIXME" markers in stat output
- Default values used as if they were computed (e.g. `baseline=0.30` hardcoded in interpretation)

Search heuristic: any file that emits Markdown tables. Compare numbers in tables vs numbers in surrounding prose.

### Category 2 — Statistical errors
**Paper-grade severity: BLOCKER**

- **Wrong test direction**: `alternative='greater'` on metric that's expected to *decrease* (e.g. Levenshtein distance for "patching disrupts target" — disruption = LD increases vs target, but "matches target" = LD decreases). Verify each test's directional alternative matches the substantive claim direction.
- **Paired vs unpaired**: bootstrap that should be task-paired but resamples episodes; t-test that should be paired but uses `ttest_ind`. Check all `ttest_*`, `wilcoxon`, `bootstrap` calls.
- **Multiple comparisons**: Holm/Bonferroni applied to wrong family size. Pre-reg locks Holm across **6 layers per direction** (per `preregistration.md §4`); verify scripts use 6, not 7 (incl L35) or 36.
- **Bootstrap clustering policy** (B2 lock 2026-05-09): single-level task_id only, no nested cluster. Search for any `groupby` or `cluster` in bootstrap code.
- **Random-effects meta**: B8 lock requires N≥10 per cell. Verify aggregators apply this floor.
- **TOST equivalence**: δ=1.0pp lock; verify usage in `aggregate_phantom_lift.py` / `aggregate_phantom_meta.py`.
- **One-sided p-values**: H1 directional (theta > 0). Verify aggregators use `1 - phi(z)`, not `2 * (1 - phi(|z|))`.

### Category 3 — Silent failures
**Paper-grade severity: HIGH**

- `except Exception: pass` / `except: pass` — swallows all errors, paper-grade death sentence
- `except (...) as e: continue` without log — same
- `try: ...; except: return None / return {}` — caller can't distinguish "no data" from "failure"
- File-not-found fallbacks that silently return empty instead of warning

### Category 4 — Non-determinism / missing seed
**Paper-grade severity: HIGH**

- `np.random.choice(...)` / `random.shuffle(...)` / `torch.randn(...)` without explicit seed at function or module level
- `np.random.default_rng()` without seed
- Bootstrap iterations missing seed parameter (each call should be deterministic given same input data)
- Cell E random injection in `run_stage2b_continuation_pilot.py` — verify seed=42 propagates to all `randn_like` calls
- `dict.items()` order dependency (post-Python 3.7 OK but worth flagging if combined with hash-based sorting)

### Category 5 — Schema / config drift
**Paper-grade severity: MEDIUM**

- Dataclass field defaults that fail on post-rerun JSON missing fields (`KeyError` in `compute_adjusted_success`)
- `str.format` references to dict keys that may not exist (`row.get` vs `row[]`)
- CSV column names hardcoded but renamed elsewhere (e.g., `lift_5_vs_3_pp` vs `lift_pp` confusion seen in this session)
- Mode label string-match: `mode == "som"` vs `mode == "SoM"` vs `mode == "SOM"` inconsistency
- Site label: `classifieds` vs `cls` vs `vwa_classifieds`
- Baseline label: `B0` vs `b0` vs `proxy_api`

### Category 6 — FP filter consistency (P79-specific)
**Paper-grade severity: BLOCKER**

Per `preregistration.md §4`:
- PRIMARY: `na_fp + eval_fp` combined (locked 2026-05-09)
- DEPRECATED: `visual_fp` (over-filters 95.3% VWA tasks per 实验笔记 §95)

Search for:
- Any code using `visual_fp` outside Appendix-D non-visual-subset robustness path
- `compute_adjusted_success()` returning unexpected `fp_reason` values
- Per-row `fp_reason` strings that don't match the `{'', 'na_fp', 'eval_fp'}` lock

### Category 7 — Layer indexing (mechanism-specific)
**Paper-grade severity: HIGH**

Per G9 audit (2026-05-09):
- Qwen3-VL-4B has 36 layers (L0=embedding output, L1-L35=transformer blocks)
- Patching iterates over `range(n_layers)` where `n_layers=36`
- Holm correction across **6 layers per direction** (L0/5/11/17/23/29 vs L35 baseline)

Verify:
- No off-by-one (e.g. iterating `range(35)` missing L35)
- L35 baseline is *not* included in the 6-layer Holm family (it's the comparison anchor, not a tested layer)
- Cross-cell comparisons use identical layer indexing

### Category 8 — Resource exhaustion / OOM smell
**Paper-grade severity: LOW (but operational)**

- Loading entire JSONL files into memory (large cells > 10MB)
- Storing all 36 hidden states × 50 tokens × N tasks in RAM (could be ~10GB)
- Missing `del`/`gc.collect()` in long loops

## Output

Write to `docs/checkpoints/codex_outputs/code_audit_2026-05-09.md` with this exact structure:

```markdown
# P79 Code Audit (Paper-Grade Bug Scan) — 2026-05-09 codex

## Executive Summary

- Total findings: N
- Severity: BLOCKER × N1 / HIGH × N2 / MEDIUM × N3 / LOW × N4
- Files audited: N5
- Top-3 most-urgent fixes (one-line each)

## Findings Table

| ID | File:line | Severity | Category | Issue (1 sentence) | Why it affects paper-grade | Suggested fix (1-2 lines) |
|---|---|---|---|---|---|---|
| F01 | scripts/analysis/foo.py:42 | BLOCKER | 2 stat | ... | ... | ... |
...

## Per-finding detail

### F01 — <file>:<line> — BLOCKER
[3-5 sentences with code excerpt + reasoning + concrete fix]

### F02 ...
```

Each finding must include:
- Exact file:line reference
- Severity tier (BLOCKER paper / HIGH / MEDIUM / LOW)
- Bug pattern category (1-8 from list above, or "OTHER")
- Code excerpt (3-5 lines max)
- Why this corrupts paper-grade number / breaks reproducibility / silently swallows errors
- Suggested fix (concrete code change, not "consider refactoring")

## Constraints

- **Read-only** — do not modify any source file. Only write `code_audit_2026-05-09.md`.
- Aim for **30-50 findings**. Quality > quantity, but be aggressive — false positives are easy to dismiss; missed bugs corrupt the paper.
- **Cross-reference recent fixes** — explicitly note when a finding is downstream of B9 (power_analysis), F4 (LOO meta), G8 (heterogeneity figure), G1 (§5.1 method), C10 (Make gates), so we don't double-count.
- **Categorize by paper section affected** when possible: §1/§4 SR claims, §5 mechanism, §6 routing, §8 limitations.
- **Skip benign style nits**: docstring formatting, unused imports, type annotation gaps, line length. Focus on correctness.
- **Don't audit tests/** — assume tests are correct.
- **Don't audit deprecated/archive code** unless flagged in latest preregistration as "appendix robustness".
- **Final line**: print exactly `DONE: code_audit_2026-05-09.md (N findings, N1 BLOCKER, N2 HIGH)`

## Time budget

You have unlimited reasoning. Use 30-90 min wall clock. Report findings as you go — incremental writes to the output file are fine. Final commit/merge of all findings into the markdown file at end.
