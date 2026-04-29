# Codex prompt: B0 dom shopping single-mode failure mode deep dive

## 用途

分析 B0 dom shopping (paper-grade clean re-run, 466/466 done, SR adj 13.55%) 的 failure modes，重点解释 counter-intuitive 发现：**A_NON_VISUAL_TEXT_ONLY (8.54% adj) < B_VISUAL_REF_IMAGE (21.30% adj)** ——为什么 DOM-only 在 text-only 任务上反而更弱？

输出 → `docs/analysis/phantom_paper/B0_dom_shopping_diagnostic.md`

## 输入数据

| 文件 | 用途 |
|---|---|
| `results/visualwebarena/phase1/B0_dom_shopping_20260428/phase1_dom_router_0/condition_summary_v2.json` | overall metrics |
| `.../analysis/reason_diagnostics/condition_overview.csv` | per-condition stats |
| `.../analysis/reason_diagnostics/condition_reason_summary.csv` | failure modes |
| `.../analysis/reason_diagnostics/episode_reason_rows.csv` | per-episode attribution |
| `.../analysis/reason_diagnostics/state_change_by_outcome.csv` | page change rate |
| `.../analysis/reason_diagnostics/final_thought_signature_summary.csv` | final-step thought clusters |
| `.../analysis/reason_diagnostics/bucket_thought_similarity_summary.csv` | trajectory bucket clusters |
| `.../analysis/reason_diagnostics/bucket_high_similarity_templates.csv` | high-overlap thought templates |
| `.../analysis/results/_overview/tables/condition_metrics.csv` | analyze_experiment summary |
| `.../analysis/signals/combined/tables/auroc_all_metrics.csv` | per-signal AUROC |
| `docs/analysis/cross_sites/codex_audit_shopping.json` | 466 task pre-classified visual taxonomy |
| `external/visualwebarena/config_files/vwa/test_shopping.json` (via submodule) | original task configs (intent/eval) |
| Sample episode JSONLs: `phase1_dom_router_0/episodes/shopping_task_<id>_steps_v2.jsonl` (focus failure cases) | step trajectory for case studies |

## 关键数据 anchor

```
B0 dom shopping SR by visual taxonomy:
  A_NON_VISUAL_TEXT_ONLY         82  SR raw  8.54%  SR adj  8.54%   ← LOWEST (counter-intuitive)
  B_VISUAL_REQUIRED_REFERENCE_IMAGE  169  SR raw 24.26%  SR adj 21.30%   ← HIGHEST
  C_VISUAL_REQUIRED_PAGE_SCREENSHOT  205  SR raw 13.66%  SR adj  9.76%
  D_UNCERTAIN                     9  SR raw 11.11%  SR adj  0.00%
  TOTAL                         465  SR raw 16.56%  SR adj 13.55%
  ----
  Visual-required (B+C): 374 tasks, SR adj 14.97%
  Non-visual (A):         82 tasks, SR adj  8.54%
```

## 调查问题

### Q1: 为什么 A 类 (text-only intent) DOM-only SR 反而最低？

3 个 working hypotheses 给 codex 测：

**H1. Aggregation/排序任务 cost**: A 类 intent 多含 "least/most expensive" / "highest rated" / "show me X with Y filter" — 需要遍历 list + 比较，DOM-only AXTree 在 Magento list page (12 items × ~10 fields each) 接近 token cap (12K obs cap)，agent 可能丢失 critical row。codex 应抽 ~10 个 A 类失败 task 的 episode JSONL，看 final thought / final action 是否表现"列表过长 → 选错"或"截断 → 看不到目标"。

**H2. 复杂多步 navigation**: A 类多需要 navigate 多个 category page (e.g., Home > Electronics > VR Headsets > sort by price)。DOM mode 缺 image 视觉指引 → 容易 click wrong link。检查 task 平均 steps + page_change_rate.

**H3. Codex 04-26 audit 分类 boundary**: 9 个 D_UNCERTAIN 全 fail；A 类可能含一些被 codex 误分的伪 non-visual。复核 ~20 个 A 类失败 task — 是否实际 visual-cue dependent？(eg. "buy the blanket" 没说颜色但 paper image 有 visual cue)

### Q2: B_VISUAL_REF_IMAGE 21.3% 高的 mechanism？

- DOM mode receives reference image as part of task input (per section3_definition.md line 80)
- B 类 task 的 ref image 是否帮 DOM agent 把 visual goal cast 成 text query (e.g. via OCR / caption inferred)?
- 检查 ~5 个 B 类成功 task 的 system prompt + first-step thought — agent 如何 reference 引用图

### Q3: 14 FP (raw 16.56% → adj 13.55% drop 3pp) 来源

- 14 FP 中 visual_fp / na_fp / eval_fp 分布
- 有无 prompt 模板系统性触发 fallback finish
- cross-mode FP rate compare (vs cls/red B0 dom paper-grade clean)

## 输出格式

`docs/analysis/phantom_paper/B0_dom_shopping_diagnostic.md`，section 结构:

```
1. Headline finding (1 句话)
2. SR breakdown table (含 visual taxonomy + step counts + page_change_rate)
3. H1/H2/H3 evidence — 每个 hypothesis 给 quantitative + 2-3 case study (task_id + intent excerpt + final thought + verdict)
4. Section 5 mechanism implications (axis 1 representation effect on Magento complex pages)
5. Section 7 generalization implications (B0 dom shopping vs cls/red — 跨 site capability profile shift)
6. paper-ready 1-paragraph summary (~100 words) for Section 4 prose embed
```

## 大致 token 预算

- Reading inputs: ~200K (含 episode JSONL samples)
- Writing analysis: ~50K
- 总: ~250K

## 触发命令

```bash
codex run --prompt docs/checkpoints/codex_prompts/B0_dom_shopping_diagnostic.md
# 或 paste 整个 prompt 到 codex CLI
```
