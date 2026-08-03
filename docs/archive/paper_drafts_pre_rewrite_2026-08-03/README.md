# Paper drafts archived 2026-08-03 — superseded by a rewrite, not by a revision

Everything here was moved out of `docs/checkpoints/paper_drafts/` on 2026-08-03 by
`git mv`, so the history is intact and any file can be brought back with a single
`git mv` in the other direction. Nothing was deleted.

## Why

The paper is being **rewritten against a different frame**, not edited. Three frames were
proposed and judged too weak between 2026-08-01 and 2026-08-03 (笔记 §407, §413, §414),
and the evidence layer itself changed materially in that window:

- the **eighth cell** (`wa_B0`) landed and invalidated three wordings outright (笔记 §415)
- **six conclusions were found hardcoded in their producers**, one wrong on the fact and
  not merely on the denominator (笔记 §418.1, §420.5)
- a **selection bias** was found in the number a candidate frame rested on — `P43`'s hit
  set is outcome-dependent by construction (笔记 §419.3)
- the **three-class framing** (no-image / vision-only / hybrid) arrived on 2026-08-03 and
  reorganises what the phenomenon section is even about

Editing drafts written against the old frame would have carried those defects forward
sentence by sentence. Starting from the products is cheaper and safer.

## What is here

| path | what it was | last touched |
|---|---|---|
| `paperA/` | Paper A — the phenomenon paper (findings + behaviour) | 2026-07-28 |
| `paperB/` | Paper B — the routing-negative paper (ceiling / supply / triage / relabelling) | 2026-08-02 |
| `aaai27/` | AAAI-27 campaign drafts; that venue was withdrawn 2026-07-22 | 2026-07-16 |
| `section*.md`, `trackB_*.md` | the original single-paper section set, superseded when the work split into A/B | 2026-06-11 – 2026-07-27 |

## What stayed behind, and why

- `latex/` — the ACL kit and `convert.sh`. Infrastructure, not content. ⚠️ `convert.sh`
  still names `paperA` and `paperB` as its only build targets and **will fail until the
  new skeleton exists**; that is deliberate, a silent fallback would be worse.
- `paper.bib` — 1,243 lines of references, all still applicable.
- `ablation_tables.md` — generated from the product JSONs by
  `scripts/analysis/export_ablation_tables.py`, and the intended starting point for the
  rewrite.

## Reading these afterwards

Numbers in these files were correct against the six-cell evidence layer and the pre-v11
diag ruleset. **Do not copy figures out of them.** Several are now known-stale in specific
ways recorded in `EVIDENCE_LAYER_SUMMARY.md` — in particular the `/6` denominators, the
`P43` framing, and anything quoting the fusion premium's clustered interval.
