# Codex prompt: Structured site-mechanism dictionary (Tier 1.5)

## 用途

Section 5 prose 写作（codex queue #13）需要 lookup 各 site 的 mechanism 解释。当前散落多处：
- 9 个 site digest (`docs/analysis/vwa_<site>/B*_{DOM,SoM,Vision}_digest.md`) — qualitative + per-mode
- `paper_planning §2.x site mechanical substrate` — high-level characterization
- `mechanism_per_task.json` (E1-E4) — quantitative per-task signatures
- `axis_effect_size.json` (Layer 1 cascade) — quantitative per-axis effect
- `axis1_microbehavior.json` (Layer 2 micro) — URL/keyword signatures
- `phantom_lift.{md,csv}` (Layer 0c/d oracle/Jaccard)
- `cost_per_mode.{json,md}` (Layer 3a/d cost classes)
- `sr_fp_per_mode.{json,md}` (Layer 0a/b)
- `disagreement_clusters.md` (04-27 stale snapshot, 3 baseline modes only — note this caveat)
- `codex_audit_shopping_A_refined.json` (shopping A1/A2/A3/A4 sub-classification)
- `swatch_form_change_audit.md` (§105 shopping bug)

**目标**: 把散落 inputs 综合成单一结构化 `site_mechanism_dictionary.json`，让 Section 5 prose codex (#13) 一次 lookup 所有 site × axis × mechanism 信息，narrative consistency 高。

## 输出 schema

`docs/analysis/cross_sites/site_mechanism_dictionary.json`:

```json
{
  "method": "synthesis of 9 site digests + §2.x substrate + Layer 0-3 quantitative + audit + disagreement_clusters",
  "schema_version": "1.0",
  "generated_at": "2026-04-29T...",
  "sites": {
    "reddit": {
      "n_total_tasks": 210,
      "site_summary": {
        "information_structure": "forum hierarchy (forum → posts → comments)",
        "navigation_affordance": "sidebar f/<forum> links + search box",
        "image_role": "content (post attachments), NOT navigation affordance",
        "intrinsic_search": false,  // search-loop is failure, not normal
        "url_routing": "path-based (/f/<forum>/<post>/<comment>)"
      },
      "axes": {
        "axis_1_text": {
          "dominance": "PRIMARY",
          "mechanism": "AXTree hierarchical embeds sidebar in deep tree → search-loop pathology. [SOM_MARKS] flat list makes sidebar f/<forum> directly clickable.",
          "evidence_layer_0": {"P-text drop-one": "+3.81pp [Layer 0c]", ...},
          "evidence_layer_1": {"axis_1 macro effects": "..."},
          "evidence_layer_2": {"axis_1 URL Jaccard": 0.573, "click-target Jaccard": 0.463},
          "case_studies": ["reddit task 4 wheat", "task 23 pumpkin robot search-loop"],
          "digest_quote": "DOM 模式看不到图片内容...重复搜索同一关键词 5-15 次不变 (B0_DOM_digest.md §2.1)"
        },
        "axis_2_prompt": {... same schema},
        "axis_3_image": {... weak/balanced},
        "compound_DOM_to_PSoM": {
          "task_pool_jaccard": 0.46,
          "click_target_jaccard": 0.421,
          "url_signature_jaccard": 0.481,
          "macro_independence_cells": "4/8 fully independent (Layer 1a)"
        }
      },
      "failure_modes": {
        "search_repeat": {"count_DOM": 29, "frac": 0.138, "case_tasks": [23, 30, 4]},
        "no_progress": {...},
        "finish_eval_mismatch": {...},
        ...
      },
      "site_specific_quirks": [
        "Postmill cookies persistent (no NOT-LOGGED-IN events 04-29 audit)",
        "task 345 image upstream-removed (§81 noted)",
        ...
      ]
    },
    "classifieds": {... same schema, axis 3 PRIMARY ...},
    "shopping": {... same schema, sparse data, mostly forward-looking ...}
  },
  "cross_site_table": {
    "axis_dominance": {
      "axis_1": ["reddit (PRIMARY)", "classifieds (secondary)", "shopping (mixed)"],
      "axis_2": ["reddit (macro driver search/type)", "classifieds (type/selfcorr)", "shopping (prompt × text task split)"],
      "axis_3": ["reddit (weak, content)", "classifieds (PRIMARY, finish h=+0.57)", "shopping (PRIMARY, visual variant)"]
    },
    "site_mechanism_thesis": "Per paper_planning §2.x: each site's mechanical substrate (information structure / navigation affordance / image role) determines which axis dominates."
  },
  "section5_narrative_anchors": {
    "reddit_substrate_claim": "AXTree hierarchical depth makes sidebar invisible → search-loop. P-text flatness fixes this without any model change.",
    "classifieds_substrate_claim": "Visual product identity makes image axis decisive. Without image P-SoM cls collapses toward DOM behavior, but task-pool diverges (Jaccard 0.53) — aggregate macro misleads.",
    "shopping_substrate_claim": "Form-interaction complexity (Magento custom-options + cart) requires precise element selection (axis 1) AND visual variant ID (axis 3). §105 swatch bug exposes form-state-tracking as orthogonal failure mode.",
    "cross_site_invariant_claim": "P-SoM is always task-pool complementary (Jaccard ≤ 0.7) regardless of site, supporting routing arm thesis even when aggregate SR ≈ DOM."
  },
  "open_questions": [
    "shopping section needs B0/B1 5-mode data (currently sparse)",
    "WA cross-bench data missing (Section 7 generalization)",
    "disagreement_clusters.md 04-27 stale, refresh via codex queue #14c after phantom data complete"
  ]
}
```

## 输出 markdown

`docs/analysis/cross_sites/site_mechanism_dictionary.md` (paper-ready, ~600-800 lines):

- Per-site section (3 sites): substrate + 3 axes × dominance × mechanism × evidence × case_studies × quotes
- Cross-site table: axis dominance × site
- Section 5 narrative anchors (3-4 sentence per site, paper-quotable)
- Open questions / data gaps

## Inputs (read these in order)

1. **Site digests** (深 in mode, primary qualitative source):
   - `docs/analysis/vwa_classifieds/B0_findings.md`, `B0_DOM_digest.md`, `B0_SOM_digest.md`, `B0_Vision_digest.md`, `B1_findings.md`, `B1_DOM_digest.md`, `B1_SOM_digest.md`, `B1_Vision_digest.md`, `B0_B1_findings.md`
   - 同 path 下 `docs/analysis/vwa_reddit/...` (9 files)
   - `docs/analysis/vwa_shopping/` 如果有 (sparse; check `ls`)

2. **Theory framework**:
   - `docs/checkpoints/paper_planning.md` §2 (axes 1/2/3 + Site mechanical substrate + Capability layer + Cross-axis interaction)

3. **Quantitative aggregations**:
   - `docs/analysis/cross_sites/axis_effect_size.json` (Layer 1 cascade × 8 metric × 3 axis)
   - `docs/analysis/cross_sites/axis1_microbehavior.json` (Layer 2 URL Jaccard, target hit, keywords)
   - `docs/analysis/cross_sites/mechanism_per_task.json` (E1-E4 per-task mechanism)
   - `docs/analysis/cross_sites/sr_fp_per_mode.json` (Layer 0a/b SR + FP)
   - `docs/analysis/cross_sites/cost_per_mode.json` (Layer 3a/d)
   - `results/phantom_paper/phantom_lift.csv` (Layer 0c/d oracle + Jaccard)

4. **Audit + bugs**:
   - `docs/analysis/cross_sites/codex_audit_shopping_A_refined.json` (A1/A2/A3/A4 shopping sub-class)
   - `docs/analysis/cross_sites/swatch_form_change_audit.md` (§105 shopping bug)
   - `docs/analysis/phantom_paper/disagreement_clusters.md` (04-27 stale; **note this caveat**)

5. **Layered status**:
   - `docs/analysis/layered_evidence_status.md` (auto-generated, summarizes all)

## 任务

1. 读所有 inputs（不需要 grep / map / 主动查 step JSONL — 所有需要的数字已在 aggregated 文件里）
2. 综合成 `site_mechanism_dictionary.{json,md}` 按上面 schema
3. 每个 axis × site 必须 cite specific quote / number / case task — 不要泛泛
4. cross-site invariant claim 必须 explicit + verifiable
5. open questions / data gaps 必须明列

## 验证

跑完 self-check:
- 3 sites × 3 axes × 4-5 mechanism fields per cell, 全有 evidence cite
- 至少 6 个 case study task IDs cross-site
- 所有 number 跟 source files 一致 (spot-check 3-5 个)
- shopping section 标 "sparse / mostly forward-looking" caveat
- disagreement_clusters cite 处必须标 "04-27 snapshot, baseline-only" caveat
- 不 hallucinate Number / Quote — 缺数据就标 missing

## 不要做的事

- 不要重新分析 step JSONL — 用 aggregated 文件
- 不要 propose new mechanism (依据现有 input)
- 不要 commit
- 不要扩展到 WA / Claude (out-of-scope, 加进 open_questions)
- 不要重写 site digests — 综合不替代

## token 预算

~50K (read 9 digests ~30K + aggregate JSONs + write JSON + write markdown)

## 触发命令

```bash
~/.npm-global/bin/codex exec --skip-git-repo-check \
  -C /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents \
  -s danger-full-access \
  --output-last-message logs/codex_site_mechanism_dictionary.last.md \
  - < docs/checkpoints/codex_prompts/site_mechanism_dictionary.md \
  > logs/codex_site_mechanism_dictionary.run.log 2>&1 &
```
