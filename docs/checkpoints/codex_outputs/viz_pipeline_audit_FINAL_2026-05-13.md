### Finding 1 — H1 format analyzer was of-means, not paired [P1]
**Claim** — same-observation format test, `stage4_format_variation_analyze.py:4-6`  
**Code reality** — pre-patch `main()` built mode means, then cosine of means, `stage4_format_variation_analyze.py:84-90`  
**Attack** — This is a different estimand. Paired perturbation pipelines must compute per task/step deltas before averaging; otherwise task composition leaks into the “format” axis.  
**Defuse** — Patched `main()` to group by `(task_id, step)` and average paired cosine gaps; added `H.shape[1] == 37`.  
**Effort** — minutes  
**Confidence** — high

### Finding 2 — Missing Method 4.4 cells rendered as 0.0 [P1]
**Claim** — v2 sweep re-aggregation should summarize raw layer×alpha cells, `reaggregate_method44_v2_hmean.py:4-10`  
**Code reality** — pre-patch MD tables used `.get(..., 0.0)` for missing aggregate cells, `reaggregate_method44_v2_hmean.py:80,92,104,116,128`  
**Attack** — This is silent zero imputation. Under HDMI-style completeness/selectivity reporting, absent extraction is missingness, not evidence of zero reliability.  
**Defuse** — Patched missing layer/alpha detection to raise before writing JSON/MD; table reads now fail closed.  
**Effort** — minutes  
**Confidence** — high

### Finding 3 — Zero-success P-prompt Venn arm was hidden as “pending” [P1]
**Claim** — 2-circle Venn means P-prompt pending, `fig_phantom_structure_venn.py:186-187`  
**Code reality** — pre-patch `draw_panel()` used `optional in sets_r and sets_r[optional]`; an observed-but-empty P-prompt set became a 2-circle panel.  
**Attack** — A measured zero-success arm is negative structural evidence, not missing data. This silently upgrades a failed axis into “data pending.”  
**Defuse** — Patched branch to render 3-circle whenever P-prompt is observed, including empty sets; smoke-tested with empty circle.  
**Effort** — minutes  
**Confidence** — high

### Finding 4 — H1 fragility counted a control as marks-like [P1]
**Claim** — controls are `hash_id_control` and `plain_sentence`, `stage4_format_variation_analyze.py:12`  
**Code reality** — pre-patch `stage4_h1_per_task_fragility.py:25-27` put `hash_id_control` inside `MARKS_LIKE`, making the verdict `≥4/7` instead of `≥4/6`.  
**Attack** — This contaminates the per-task dichotomy denominator with a negative control. Reviewers will correctly call that hypothesis leakage.  
**Defuse** — Patched `MARKS_LIKE` to six variants, moved both controls to `CONTROLS`, and updated verdict text.  
**Effort** — minutes  
**Confidence** — high

### Finding 5 — Axis-2 fragility MD hardcoded v1 sample shape [P2]
**Claim** — v2 is task×mode×one-step style; report should reflect observed NPZ shape.  
**Code reality** — pre-patch text said “2 steps” and “24 tasks” unconditionally, `stage4_axis2_per_task_fragility.py:120-125,179-181`  
**Attack** — This is a paper-table footgun: correct computation, stale denominator prose.  
**Defuse** — Patched output to derive task count and step-count distribution from NPZ; added L0-L36 assertion.  
**Effort** — minutes  
**Confidence** — high

Diff summary: 5 files changed, 82 insertions, 40 deletions.

Verification:
`python3 -m py_compile` passed for all edited scripts. `.venv/bin/python` compile passed. Venn empty-set smoke test passed under `.venv` with expected “Circle C has zero area” warning. `git diff --check` passed.