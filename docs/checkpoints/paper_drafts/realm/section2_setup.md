## 2. Setup

We evaluate on VisualWebArena \citep{koh2024visualwebarena} and
WebArena \citep{zhou2024webarena}.

<!-- EMPTY, but this one does NOT depend on the frame and can be written any time.

Content it needs, all already established:
  - 8 cells = (site x backbone). VWA classifieds/reddit x {B0 Qwen3-VL-235B-A22B,
    B1 Qwen3-VL-4B, B2 Gemma-3-4B} + WebArena reddit x {B0, B1}. **WA x B2 does not
    exist** — B2 never ran WebArena.
  - 6 observation modes, and the 3 deployment classes they group into.
  - Scored universes: cls 224 / red 203 (AMENDMENT_08) / WA 104 (six-mode intersection,
    WA has no exclusion list).
  - Estimands that must be stated because products disagree if they are not: cost =
    `total_billed_cost_usd`; latency = canonical (retry / busy-wait / recovered removed);
    success = binary, the evaluator emits exactly two values over 7,686 episodes.
  - The rerun band 0.89-2.23pp and what it is a floor *of*.
-->
