# Evidence snapshot 2026-10-06 — the 11-cell router pilot (2026-09-09)

## Why this exists

笔记 §505 (router 重想: one-step-lookahead / bandit / 三臂收敛 / 预算路由) is the only analysis
that covers all **11 cells** (VWA cls × {B0,B1,B2,B5} · red × {B0,B1,B2} · shop × {B0,B1} ·
WA red × {B0,B1}). The 08-02 cross-site product suite under `docs/analysis/cross_sites/` covers
8 of them — shopping and cls_B5 are absent from almost all of it (实验笔记 §531).

Until today the scripts and raw outputs behind §505 lived only in
`results/router_llm_pilot_20260909/`, which is gitignored and (since 2026-10-06) a junction to
`E:\p79-runs\router_llm_pilot_20260909\`. The tracked digest
`docs/analysis/cross_sites/one_step_lookahead_2026-09-09.md` carries the tables, but not the code
that produced them. This snapshot puts the code and the small outputs under version control.

**Frozen copy, not a working directory.** The scripts still read and write
`results/router_llm_pilot_20260909/...`; run them from the repo root against that path, not
from here. Nothing in this folder should be edited — if a script is fixed, fix the copy at the
original path and take a new snapshot.

## What is here

| path | what |
|---|---|
| `ORIGINAL_README.md` | the pilot's own README, incl. the 10 known defects + the 09-09 evening re-analysis notes |
| `NEXT_SESSION_PROMPT.md` | the handoff the user wrote on 09-09 |
| `scripts/*.py` (23 files) | LLM routers v1/v2/v3/3-class (**call the paid proxy**), `build_full_table.py`, eval / pareto / lookahead / bandit / step-0 extraction |
| `choices/` | the router LLM's per-task choices (GPT-5.6 luna/terra). **Paid API output — the one part that cannot be regenerated for free** |
| `tables/full_table2.json` | per-task `[success, cost_usd, latency_ms, tokens, steps]` for every `<cell>|<mode>` |
| `lookahead/` | `budget_frontier.json` (§505.24 budget routing) · `lookahead_results.json` · `task_meta.json` · the four text summaries · `scan_cum.py` |
| `*.log` | the run logs of the day |

## What was left out, and how to get it back

| left out | size | regenerate |
|---|---|---|
| `lookahead/step0.jsonl.gz` | 15.5 MB | `scripts/extract_step0.py` over the episode dirs |
| `lookahead/cum_scan.json.gz` | 5.2 MB | `lookahead/scan_cum.py` |
| `lookahead/skeleton_reason_diag/` | 4.7 MB | per-run reason-row CSVs; `scripts/analysis/` reason diagnostics |

All three are derived from episode records that are in `E:\p79-runs` (and in the A100 /
DGX mirrors on E:). Copy taken from `E:\p79-runs\router_llm_pilot_20260909\` on 2026-10-06;
file mtimes preserved.
