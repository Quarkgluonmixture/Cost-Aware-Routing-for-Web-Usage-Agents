# /diag digest index — failure attribution coverage

- **41 digests**: 37 with a readable three-way split · 3 that declare their own attribution incomplete · 1 pointer files (numbers live in the run-specific digests they forward to) · 0 this script still cannot parse
- built by `scripts/analysis/index_diag_digests.py`
- **navigation layer only** — per-rule detail, Tier-2 deep dives and P-rule false-positive audits exist solely in the digests

> ⚠️ **`?` means not looked at, not zero.** An earlier revision of this index treated unparsed digests as all-zero and concluded the corpus was clean; two digests contradict that (`B1_som_classifieds` benchmark-FP ≈1.5%, `B2_vision_reddit` task 160 / B-1889), and one had already written the warning the script violated. Coverage is now reported before content.

Classes: **agent-limit** = model capability · **scaffold-bug** = our pipeline · **benchmark-FP** = evaluator misjudged.

## Coverage and attribution

| baseline | site | mode | coverage | failed | agent-limit | scaffold-bug | benchmark-FP | free-text flag |
|---|---|---|---|---|---|---|---|---|
| B0 | classifieds | dom | ✅ | — | 0 | 0 | 0 | — |
| B0 | classifieds | phantom_prompt | ✅ | — | 100 | 0 | 0 | FP |
| B0 | classifieds | phantom_som | ✅ | — | 100 | 0 | 0 | FP |
| B0 | classifieds | phantom_text | ✅ | — | 188 (100%) | 0 | 0 (0%) | — |
| B0 | classifieds | som | ⚠️ 自称不完整 | — | 33 | 1 | 0 | scaffold |
| B0 | classifieds | vision | ↪ 指针(数字在 run digest) | — | ↪ | ↪ | ↪ | — |
| B0 | reddit | dom | ✅ | 175 | 175 (100%) | 0 (0%) | 0 (0%) | FP |
| B0 | reddit | phantom_prompt | ✅ | — | 5 | 0 | 0 | FP |
| B0 | reddit | phantom_som | ✅ | — | 5 | 0 | 1 | FP |
| B0 | reddit | phantom_text | ✅ | — | 5 | 1 | 1 | scaffold, FP |
| B0 | reddit | som | ⚠️ 自称不完整 | — | 41 | 0 | 4 | FP |
| B0 | reddit | vision | ✅ | — | 40 | 0 | 2 | FP |
| B1 | classifieds | dom | ✅ | — | 17 | 0 | 0 | — |
| B1 | classifieds | phantom_prompt | ✅ | — | 22 | 0 | 1 | — |
| B1 | classifieds | phantom_som | ✅ | — | 45 | 0 | 0 | — |
| B1 | classifieds | phantom_text | ✅ | — | 40 | 0 | 0 | — |
| B1 | classifieds | som | ✅ | — | 46 | 0 | 0 | scaffold, FP |
| B1 | classifieds | vision | ✅ | — | 44 | 0 | 1 | — |
| B1 | reddit | dom | ✅ | — | 5 | 0 | 2 | FP |
| B1 | reddit | phantom_prompt | ✅ | — | 5 | 0 | 2 | FP |
| B1 | reddit | phantom_som | ✅ | — | 5 | 0 | 2 | FP |
| B1 | reddit | phantom_text | ✅ | — | 5 | 0 | 2 | FP |
| B1 | reddit | som | ✅ | — | 5 | 0 | 2 | FP |
| B1 | reddit | vision | ✅ | — | 5 | 0 | 1 | FP |
| B2 | classifieds | dom | ✅ | — | 100 | 0 | 0 | FP |
| B2 | classifieds | phantom_prompt | ✅ | — | 6 | 0 | 3 | — |
| B2 | classifieds | phantom_som | ✅ | — | 99 | 0 | 1 | — |
| B2 | classifieds | phantom_text | ✅ | — | 100 | 0 | 0 | — |
| B2 | classifieds | som | ✅ | — | 100 | 0 | 0 | — |
| B2 | classifieds | vision | ✅ | — | 100 | 0 | 0 | — |
| B2 | reddit | dom | ✅ | — | 0 | 0 | 0 | FP |
| B2 | reddit | phantom_prompt | ✅ | — | 0 | 0 | 0 | FP |
| B2 | reddit | phantom_som | ✅ | — | 0 | 0 | 0 | FP |
| B2 | reddit | phantom_text | ✅ | — | 0 | 0 | 0 | FP |
| B2 | reddit | som | ✅ | — | 0 | 0 | 0 | FP |
| B2 | reddit | vision | ✅ | — | 0 | 0 | 0 | FP |

## ⚠️ Digests carrying a non-agent-limit signal

Either a non-zero structured count, or free text naming a benchmark-FP / scaffold issue. **A failure-analysis section must read these directly.**

- **B0_dom_classifieds_R31194** — free-text benchmark-FP mention · `docs/analysis/vwa_classifieds/B0_dom_classifieds_R31194_diag_digest.md`
- **B0_dom_classifieds_R21557** — scaffold-bug 1; benchmark-FP 1; free-text scaffold mention · `docs/analysis/vwa_classifieds/B0_dom_classifieds_R21557_diag_digest.md`
- **B0_phantom_prompt_classifieds** — free-text benchmark-FP mention · `docs/analysis/vwa_classifieds/B0_phantom_prompt_classifieds_diag_digest.md`
- **B0_phantom_som_classifieds** — free-text benchmark-FP mention · `docs/analysis/vwa_classifieds/B0_phantom_som_classifieds_diag_digest.md`
- **B0_som_classifieds** — scaffold-bug 1; free-text scaffold mention · `docs/analysis/vwa_classifieds/B0_som_classifieds_diag_digest.md`
- **B0_dom_reddit** — free-text benchmark-FP mention · `docs/analysis/vwa_reddit/B0_dom_reddit_diag_digest.md`
- **B0_phantom_prompt_reddit** — free-text benchmark-FP mention · `docs/analysis/vwa_reddit/B0_phantom_prompt_reddit_diag_digest.md`
- **B0_phantom_som_reddit** — benchmark-FP 1; free-text benchmark-FP mention · `docs/analysis/vwa_reddit/B0_phantom_som_reddit_diag_digest.md`
- **B0_phantom_text_reddit** — scaffold-bug 1; benchmark-FP 1; free-text scaffold mention; free-text benchmark-FP mention · `docs/analysis/vwa_reddit/B0_phantom_text_reddit_diag_digest.md`
- **B0_som_reddit** — benchmark-FP 4; free-text benchmark-FP mention · `docs/analysis/vwa_reddit/B0_som_reddit_diag_digest.md`
- **B0_vision_reddit** — benchmark-FP 2; free-text benchmark-FP mention · `docs/analysis/vwa_reddit/B0_vision_reddit_diag_digest.md`
- **B1_phantom_prompt_classifieds** — benchmark-FP 1 · `docs/analysis/vwa_classifieds/B1_phantom_prompt_classifieds_diag_digest.md`
- **B1_som_classifieds** — free-text scaffold mention; free-text benchmark-FP mention · `docs/analysis/vwa_classifieds/B1_som_classifieds_diag_digest.md`
- **B1_vision_classifieds** — benchmark-FP 1 · `docs/analysis/vwa_classifieds/B1_vision_classifieds_diag_digest.md`
- **B1_dom_reddit** — benchmark-FP 2; free-text benchmark-FP mention · `docs/analysis/vwa_reddit/B1_dom_reddit_diag_digest.md`
- **B1_phantom_prompt_reddit** — benchmark-FP 2; free-text benchmark-FP mention · `docs/analysis/vwa_reddit/B1_phantom_prompt_reddit_diag_digest.md`
- **B1_phantom_som_reddit** — benchmark-FP 2; free-text benchmark-FP mention · `docs/analysis/vwa_reddit/B1_phantom_som_reddit_diag_digest.md`
- **B1_phantom_text_reddit** — benchmark-FP 2; free-text benchmark-FP mention · `docs/analysis/vwa_reddit/B1_phantom_text_reddit_diag_digest.md`
- **B1_som_reddit** — benchmark-FP 2; free-text benchmark-FP mention · `docs/analysis/vwa_reddit/B1_som_reddit_diag_digest.md`
- **B1_vision_reddit** — benchmark-FP 1; free-text benchmark-FP mention · `docs/analysis/vwa_reddit/B1_vision_reddit_diag_digest.md`
- **B2_dom_classifieds** — free-text benchmark-FP mention · `docs/analysis/vwa_classifieds/B2_dom_classifieds_diag_digest.md`
- **B2_dom_classifieds_R17895** — scaffold-bug 2 · `docs/analysis/vwa_classifieds/B2_dom_classifieds_R17895_diag_digest.md`
- **B2_phantom_prompt_classifieds** — benchmark-FP 3 · `docs/analysis/vwa_classifieds/B2_phantom_prompt_classifieds_diag_digest.md`
- **B2_phantom_som_classifieds** — benchmark-FP 1 · `docs/analysis/vwa_classifieds/B2_phantom_som_classifieds_diag_digest.md`
- **B2_dom_reddit** — free-text benchmark-FP mention · `docs/analysis/vwa_reddit/B2_dom_reddit_diag_digest.md`
- **B2_phantom_prompt_reddit** — free-text benchmark-FP mention · `docs/analysis/vwa_reddit/B2_phantom_prompt_reddit_diag_digest.md`
- **B2_phantom_som_reddit** — free-text benchmark-FP mention · `docs/analysis/vwa_reddit/B2_phantom_som_reddit_diag_digest.md`
- **B2_phantom_text_reddit** — free-text benchmark-FP mention · `docs/analysis/vwa_reddit/B2_phantom_text_reddit_diag_digest.md`
- **B2_som_reddit** — free-text benchmark-FP mention · `docs/analysis/vwa_reddit/B2_som_reddit_diag_digest.md`
- **B2_vision_reddit** — free-text benchmark-FP mention · `docs/analysis/vwa_reddit/B2_vision_reddit_diag_digest.md`

## ⚠️ Digests that declare their own attribution incomplete

For these, a blank scaffold-bug / benchmark-FP cell means **not investigated**. Citing them as evidence of a clean pipeline is exactly the inference they warn against.

- **B0_dom_classifieds_R31194** — matched `未逐个证因。"agent-limit 主导"应读作"35 深挖` · `docs/analysis/vwa_classifieds/B0_dom_classifieds_R31194_diag_digest.md`
- **B0_som_classifieds** — matched `未深挖` · `docs/analysis/vwa_classifieds/B0_som_classifieds_diag_digest.md`
- **B0_som_reddit** — matched `未补录前本 digest 三分类统计基于 34/40 深挖` · `docs/analysis/vwa_reddit/B0_som_reddit_diag_digest.md`

## Corpus-level verdict

**Not admissible.** Only 37/41 digests expose a machine-readable attribution table, so no statement of the form "the pipeline is clean across all conditions" can be made from this index. The per-condition rows above are the usable unit.

## Run-specific digests (replicates / ablation arms)

- `B0_dom_classifieds_R31194` (R31194) — ⚠️ 自称不完整 · `docs/analysis/vwa_classifieds/B0_dom_classifieds_R31194_diag_digest.md`
- `B0_dom_classifieds_R21557` (R21557) — ✅ · `docs/analysis/vwa_classifieds/B0_dom_classifieds_R21557_diag_digest.md`
- `B0_vision_classifieds_R24792` (R24792) — ✅ · `docs/analysis/vwa_classifieds/B0_vision_classifieds_R24792_diag_digest.md`
- `B0_vision_classifieds_R32024` (R32024) — ✅ · `docs/analysis/vwa_classifieds/B0_vision_classifieds_R32024_diag_digest.md`
- `B2_dom_classifieds_R17895` (R17895) — ✅ · `docs/analysis/vwa_classifieds/B2_dom_classifieds_R17895_diag_digest.md`

## Headlines (where the digest states one)

- **B0_dom_reddit** — B0 dom reddit 的 14.6% SR **纯能力地板** —— 无 pipeline bug、无评测 FP（两侧皆 0）。失败全部是 DOM 模式 + 4B/235B 能力在 reddit 任务族上的真实局限。这对 paper 是干净的 agent-limit 证据，不需要修代码。
