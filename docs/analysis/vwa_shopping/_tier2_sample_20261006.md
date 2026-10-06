---
type: analysis
status: tier-2 sample, one reader
purpose: read a sample of the shopping failures no v11 rule explains
---

# VWA shopping — Tier-2 sample of the no-rule failures (2026-10-06)

**Sample.** 5 failed episodes per condition × 9 conditions = 45, drawn with `random.Random(20261006)` from the
`no_hit_task_ids` of `_tier1_summary.json` (draw: `_tier2_sample_20261006_draw.json`). Read by one Claude sub-agent
from the step JSONL + task config, read-only. Per-episode classes, mechanisms and evidence: `_tier2_sample_20261006.json`.

**Result.** agent-limit **41** · benchmark-FP **2** · scaffold-bug **1** · unclear **1**. The no-rule failures are, in this
sample, overwhelmingly the model's own mistakes; the large no-rule share on shopping (`_tier1_summary.md` §1) is a gap in the
rule set's vocabulary, not hidden pipeline or evaluator failure at scale.

Most common agent mistakes: leaving the page the task points to and keyword-searching instead (13); breaking a constraint
stated in its own reasoning (6); wrong grid position (4); reading the sidebar price filter as a price range (4); ending
"buy" tasks without ordering (4); clicking the header "My Wish List" link as if it added the item (4).

## The four non-agent episodes, and what was checked independently

| episode | class | what the reader saw | independently checked |
|---|---|---|---|
| B1_som 160 | scaffold-bug | wishlist carries over between tasks; new item lands off page 1 | the four wishlist-VQA tasks (108/159/160/163) are 0/9 across all nine conditions, and no reset step clears the wishlist → **B-2003**. Page-1/ordering detail not verified (needs the live site) |
| B0_vision 77 | benchmark-FP | correct 3rd item of the first row added; evaluator expects an item not in that row as served | **not verified** |
| B1_vision 61 | benchmark-FP | expected $26.99 item not in the first row as served | **not verified** |
| B0_dom 56 | unclear | expected orange item not in the first row as served | — |

The last three share one claimed cause: the category grid order (sorted by "Position") differs between runs (B0 vision on
08-07 vs B1 vision on 09-06). Whether that drift comes from our environment or the benchmark is unknown. Until checked, "first
row / Nth product" tasks on shopping should be treated as possibly non-stationary across runs.

**Found in passing, verified, and larger than this sample suggests:** typing into a search box that already holds a query
appends instead of replacing (`Sony` + `Sony headphones` → `SonySony headphones`): 1,446 of 3,388 such type actions on
shopping, under 1% on classifieds / reddit → **B-2002**. It decided only one sampled episode's fate (B1_phantom_text 428, which
would have failed anyway), but it is uneven across modes, so it matters for shopping mode comparisons.
