# Prospective test — pre-flight budget router on B1 × VWA-shopping held-out arms (frozen 2026-09-09)

**Why this exists.** Every budget-router number in 笔记 §505.21–24 is an offline splice of runs
that had already landed (the A→B replicate test in §505.21 included). The B1 shopping chain
(`b1_shopping_chain_launch_intent_20260906.md`) is still landing `P-text` and has not started
`P-prompt`, so their outcomes can be predicted **before they are read**. That is the only
prospective test this project can run at zero cost, and it is fixed here before any number
can influence it.

**What is frozen.** `budget_router_prospective_shop_B1_20260909.json` (sha256 in the commit
that adds it): one difficulty score per shopping task (mean predicted P(success) over the four
landed shop_B1 arms, from a logistic model on pre-flight features — step-0 page stats + intent
regexes + condition one-hot — trained on shop_B1 {dom, som, vision, psom} and shop_B0 {dom,
som, vision}, 3,038 rows; `train_label_sha256` pins the labels used), and the tier assignment
for two pre-declared policies: **two_tier** (lowest 50% → cap 5, rest full) and **three_tier**
(lowest 20% → abstain, next 30% → cap 8, rest full). P-text / P-prompt outcomes were never read.

**Held-out sets.** PRIMARY = P-text tasks not in `seen_ptext_ids` (the 216 P-text episodes on
disk at freeze time) + all P-prompt tasks. The 216 seen P-text tasks are reported separately
and labelled non-prospective.

**Evaluation** (exact truncation on per-step cost and latency):
`python scripts/analysis/budget_router_prospective_eval.py eval --run <ptext_run_dir> --run <pprompt_run_dir>`
Comparators: fixed cap at matched cost; random tier assignment (expectation over 200 draws).

**Pre-declared criterion (direction only; shop_B1 has no rerun band):** two_tier on the PRIMARY
set loses less SR than the fixed cap at matched cost AND less than random tiers. Whatever the
verdict, it is reported; a FAIL is written as a FAIL. Reference: 18 replicate pairs gave
learned −2.15pp vs fixed −3.58pp vs random −4.06pp at −41% cost (§505.21, 14/18 pairs).
