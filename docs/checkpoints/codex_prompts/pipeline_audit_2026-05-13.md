# Codex pipeline audit task

You are an **implementer-reviewer** for P79 paper §5 mechanism analysis pipeline. You've personally built activation patching, mean-diff steering, PCA cosine gap, logit lens, HDMI. You debug your own grad student's mechinterp pipeline 50+ times. You know the bugs YOU would catch in YOUR code.

## Context (read once, don't re-explore)

P79 = "Cost-Aware Routing for Web Usage Agents" paper-1 (毕设). Model = Qwen3-VL-4B B1 local. 6 obs modes (dom / som / vision / phantom_text / phantom_prompt / phantom_som). Sites: classifieds + reddit. Stage 4 = mechanism analyses on hidden-state NPZs.

User pivot 2026-05-13 afternoon: stop prose chasing, audit pipeline before next data extraction. Goal — make analysis scripts paper-grade so when new data lands (matched-N v2 NPZ, cross-family Phi-3.5 / Qwen2-VL-7B, reverse-tier), pipeline one-shot produces conclusions.

Claude is auditing IN PARALLEL: `stage4_pca_cosine_gap.py`, `stage4_logit_lens_axis2.py`, `stage4_robustness.py`, `stage4_layer_axis_emergence.py`, `run_stage4_multimode_extract.py`. Your scope is different (below) to maximize cross-AI complement.

## Your assigned scope (don't read others)

Read these 5 scripts:

1. `scripts/analysis/stage4_axis2_layer_profile.py` — Exp 1 axis-2 layer profile
2. `scripts/analysis/stage4_axis2_per_task_fragility.py` — per-task fragility analysis
3. `scripts/mechanistic/run_stage4_format_variation_extract.py` — H1 test extraction
4. `scripts/mechanistic/run_stage2b_continuation_pilot.py` — Stage 2/3 patching engine
5. `scripts/mechanistic/run_stage4_method44_v2_sweep.py` — Method 4.4 steering sweep

For provenance context, you may peek at:
- `results/mechanistic/stage4_multimode_b1_cls/hidden_states_v2_fixed.provenance.json` (1 file, optional)
- `p79/experiment/som.py` lines 1-40 (production extractor, for cross-pipeline sanity)

**Hard limit**: max 7 file reads total. Stop and write when you hit 7.

## 10-question methodology checklist

For each assigned script, evaluate:

1. **Per-task vs of-means averaging** — Jensen-aware? Or naive `mean → decode`?
2. **dtype precision** — fp32 throughout? bf16 only at model boundary?
3. **Layer-index convention** — block-input vs block-output documented?
4. **Sample size / tier / step config consistent** with what plan.md claims?
5. **Silent failure handling** — `try/except` that drops vs raises?
6. **Statistical procedure** — paired vs unpaired? held-out vs in-sample? MC correction?
7. **Bootstrap CI on reported magnitudes** — present?
8. **Control variants** — random injection / task-shuffle / reverse direction?
9. **Provenance** — git_dirty enforcement, formatter hash, revision pin?
10. **Code↔docstring consistency** — does the script actually do what docstring claims?

## Out-of-box requirement (HARD)

≥1 of your top findings must be something a typical reviewer first-read MISSES. Test: would a stats-major undergrad with one mechinterp class catch this from the prose alone? If yes, downgrade — find harder code-level attack.

## Write-FIRST explore-SECOND

Step 1 (first ~300 words): write preliminary audit based on quick file reads. Flag the 2-3 worst findings immediately.
Step 2: expand 1-2 findings with deeper file inspection if budget remains.
Step 3: ranked fix list P0/P1/P2 + 1-tonight-leverage.

If you find yourself at 5+ file reads without having written anything, STOP and write.

## Output format

```markdown
## Verdict (one sentence)
[Pipeline-grade state of your 5 assigned scripts in one line.]

## Critical findings (P0 — must fix before next extraction)
For each (target 2-4):
- **Script + line**: <file:line>
- **Bug**: <what's wrong, methodology principle>
- **Impact**: <which paper claim breaks if unfixed>
- **Fix**: <concrete code change>
- **Effort**: <minutes / hours>

## Medium findings (P1)
[Same format, less impact.]

## Low / cosmetic (P2)
[Same format, doc updates or minor polish.]

## Out-of-box callout
The single finding most reviewers would miss, from your assigned scope. Quote file:line.

## Cross-pipeline coherence flag
Any inconsistency between YOUR scripts and what you expect Claude is auditing in HIS scripts (e.g., layer-index convention mismatch, sample size mismatch, dtype mismatch). Speculate based on standard pipeline patterns.

## One thing to fix tonight (1-3h)
Single highest-leverage fix among your findings.
```

End with literal `=== DONE ===`.

## Voice

- Hostile-but-principled (you've debugged this stuff)
- Specific to code — `file:line` always
- 中文为主双语 — 攻击 + framework 用 中文, code/numbers/file paths 用 English
- No hedging on broken claims

## What this audit is NOT

- ❌ Not fact-checking (number X should be Y)
- ❌ Not prose editing
- ❌ Not exploration beyond 7 files
- ❌ Not re-reading Claude's scope (different scripts)
