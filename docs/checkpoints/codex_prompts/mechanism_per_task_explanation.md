# Codex prompt: Per-task mechanism evidence (E1-E4 Tier 1 quick wins)

## 用途

`paper_planning §3` 4-Layer Evidence Framework 主要测 "what shifts under mode swap"。Section 5 mechanism explanation 需要更精细的 per-task per-step decision evidence 来回答 **"why these shifts happen"**。这次 Tier 1 quick win 加 4 个新 per-task metrics：

- **E1 click-target divergence**：clicked page transitions Jaccard between modes (mode-invariant, step-invariant signature)
- **E2 trajectory boundary**：for symmetric-difference tasks (DOM succeeds but P-SoM fails or vice versa), find first divergent step
- **E3 confidence calibration cross-condition aggregator**：existing `analyze_confidence_calibration.py` is per-run; need cross-condition summary table for paper Section 5
- **E4 action vocabulary distribution**：full action_type × subtype (click→link/button/textbox; type+keyword) distribution per (mode, site)

## 重要方法学约束

### Element identifier MUST be mode-invariant + step-invariant

`element_id` 不能用 — 三个原因：
1. AXTree IDs reset on every page navigation (within mode, 跨 step 不可比)
2. AXTree int vs [SOM_MARKS] int = different ID systems (跨 mode 不可比)
3. Even within same page, ID assignment can shift if DOM tree changes

**Use instead**:
- **Click signature**: `(pre_click_url_path_query, post_click_url_path_query)` — server-determined URL transition is fully mode-invariant + step-invariant
- For type actions: `action.text` (lowercased, stripped) — already mode-invariant
- For scroll/wait/finish: action_type alone (no per-step variation needed)

### URL signature (mode-invariant)

Use `urlparse(url)` decompose; signature = `f"{path}?{relevant_query_keys}"`:
- reddit: just `path` is sufficient (path-based routing, e.g. `/f/memes/123`)
- classifieds: must include `page=X&id=Y&sCategory=Z&sPattern=...` because OSClass uses query routing on `/index.php`

Already implemented in `axis1_microbehavior.py:url_path_query()` — reuse that.

## Inputs

Step JSONL, same as `axis_effect_size.py` and `axis1_microbehavior.py`:
```python
STEP_DIRS = {
    "reddit": {
        "DOM": "results/visualwebarena/phase1/B0_3mode_reddit_20260422/phase1_dom_router_0/episodes",
        "Vision": "results/visualwebarena/phase1/B0_3mode_reddit_20260422/phase1_vision_router_0/episodes",
        "SoM": "results/visualwebarena/phase1/B0_3mode_reddit_20260422/phase1_som_router_0/episodes",
        "Phantom-SoM": "results/visualwebarena/phase1/B0_phantom_reddit_20260428/phase1_phantom_som_router_0/episodes",
        "P-text": "results/visualwebarena/phase1/B0_phantom_text_reddit_20260427/phase1_phantom_dom_router_0/episodes",
    },
    "classifieds": {... same shape ...},
}
```

Each step JSONL line has at least: `obs_url`, `action_type` (or `action.action_type`), `action.text`, `action.thought` (for self-correction tokens), `confidence` (if available), `step_idx`. Adjusted-success outcome from each task's `*_summary_v2.json`.

## Tasks

### E1: Click-target divergence

Per task per mode, build `click_target_set`:
```python
click_targets = set()
for i, step in enumerate(steps):
    at = step.get("action_type") or step.get("action", {}).get("action_type")
    if at != "click":
        continue
    pre = url_path_query(step.get("obs_url"))
    post = url_path_query(steps[i+1].get("obs_url")) if i+1 < len(steps) else ""
    if pre and post and pre != post:  # only track URL-changing clicks
        click_targets.add((pre, post))
```

Per (site, axis_contrast), compute:
- mean Jaccard of `click_target_set` across paired tasks (paired contrast: DOM↔P-text, P-text↔P-SoM, P-SoM↔SoM, compound DOM↔P-SoM)
- std + median + N
- distribution of |click_target_set| sizes

### E2: Trajectory boundary divergence

For each (site, mode_a, mode_b) cascade-adjacent contrast (DOM-P-text, P-text-P-SoM, P-SoM-SoM, DOM-P-SoM):
- Filter to **symmetric-difference success tasks**: tasks where `mode_a` adjusted-success ≠ `mode_b` adjusted-success
- For each such task, find **first_divergent_step**:
  ```python
  for step_idx, (a_url, b_url) in enumerate(zip(mode_a_urls, mode_b_urls)):
      if a_url != b_url:
          first_divergent_step = step_idx
          break
  ```
- Aggregate: distribution of first_divergent_step per axis (histogram), median, fraction of "early divergence" (step ≤ 3) vs "late divergence" (step ≥ 10)
- For each axis, list 3 **case study task IDs** with extreme behavior (earliest/latest/most divergent)

### E3: Confidence calibration cross-condition aggregator

`analyze_confidence_calibration.py` outputs per-run JSON files (one per run dir) with calibration metrics. Need cross-condition aggregator:

For each (model, site, mode) cell, locate output of `analyze_confidence_calibration.py`:
- look for `{run_dir}/analysis/confidence/per_mode_summary.csv` or similar (check actual output path; may need to invoke per-run if missing)
- Extract: ECE, MCE, Brier, AUROC (token-level + verbalized + behavioral signals)

Build cross-condition table:
```
| model | site | mode | ECE_token | ECE_verbal | AUROC_token | AUROC_verbal | AUROC_behavioral_max |
| B0    | red  | DOM  | 0.12      | 0.08       | 0.78        | 0.82         | 0.79                  |
| B0    | red  | P-SoM| 0.05      | 0.03       | 0.75        | 0.79         | 0.72                  |
...
```

Highlights:
- "honest commit" mode: lowest ECE per (model, site)
- "best signal AUROC" mode: highest AUROC per (model, site)
- Layer 0b FP rate (`sr_fp_per_mode.md` 已有) cross-reference: low ECE ↔ low FP?

### E4: Action vocabulary distribution

Per (model, site, mode), full per-step action distribution:
```python
action_dist = {
    "click": int,
    "type": int,
    "scroll": int,
    "select_option": int,
    "wait": int,
    "back": int,
    "forward": int,
    "finish": int,
    "tab_focus": int,
    "other": int,
}
```

Plus click subtypes (heuristic via element_id text in obs around the click — may be costly; OPTIONAL):
- click_link / click_button / click_textbox / click_other

For now, **just full action_type distribution** (without click subtype) is sufficient.

Per axis paired contrast:
- mean per-task action_type fraction shift (DOM→P-text, etc.)
- highlight modes that use uncommon actions: e.g., "P-text cls uses select_option 5× more than DOM" (per cls task 12 case study evidence)

## Output

- `scripts/analysis/mechanism_per_task.py` (NEW, ~400-500 lines, pure stdlib)
- `docs/analysis/cross_sites/mechanism_per_task.json` (machine-readable)
- `docs/analysis/cross_sites/mechanism_per_task_report.md` (paper-ready, ~300-400 lines)

JSON schema:
```json
{
  "method": "...",
  "E1_click_target_divergence": {
    "reddit": {
      "axis_1_text":   {"mean_jaccard": ..., "std": ..., "n": 210, ...},
      "axis_2_prompt": {...},
      "axis_3_image":  {...},
      "compound_DOM_to_PSoM": {...}
    },
    "classifieds": {... same shape ...}
  },
  "E2_trajectory_boundary": {
    "reddit": {
      "DOM_vs_P-SoM": {
        "n_symmetric_diff_tasks": 22,  // 12 P-SoM unique + 10 DOM unique
        "median_first_divergent_step": 4,
        "early_divergence_rate": 0.45,  // step ≤ 3
        "late_divergence_rate": 0.18,   // step ≥ 10
        "case_study_task_ids": [4, 23, 30]
      },
      ...
    }
  },
  "E3_confidence_calibration": {
    "B0/reddit/DOM": {"ECE_token": ..., "AUROC_verbal": ..., ...},
    ...
  },
  "E4_action_vocabulary": {
    "B0/reddit/DOM": {"click": 0.42, "type": 0.38, "scroll": 0.15, ...},
    ...
  },
  "paper_section5_implications": {
    "E1_headline": "...",
    "E2_headline": "...",
    "E3_headline": "...",
    "E4_headline": "..."
  }
}
```

Markdown report:
- 4 sections (E1, E2, E3, E4), each with summary table + per-axis breakdown + 1-2 case studies
- Closing "Mechanism evidence for paper Section 5" 摘要

## Layer mapping (for layered_status.py integration)

| Metric | Layer |
|---|---|
| E1 click-target divergence | **Layer 2 micro** (extends axis1_microbehavior URL Jaccard with click-event granularity) |
| E2 trajectory boundary | **Layer 2 micro** (per-step decision quality) |
| E3 confidence calibration | **Layer 0b** (FP-related) + **Layer 1 supporting** |
| E4 action vocabulary | **Layer 1 macro** (extends axis_effect_size's 4 action-type metrics to full distribution) |

加进 Makefile:
```makefile
analyze-mechanism:
	$(PYTHON) scripts/analysis/mechanism_per_task.py
.PHONY: analyze-mechanism
```

并 chain 进 `analyze-layered` 作 final step (after Layer 0/1/2/3, before layered_status.py)。

## 验证

跑完 self-check:
- N(reddit) per axis 1 = 210, N(cls) = 234 (跟 axis1_microbehavior 一致)
- E1 mean Jaccard 落 [0, 1]
- E2 first_divergent_step 落 [0, 30]（max_steps=30）
- E3 N(model×site×mode) cells == sum of paper-grade B0 + B1 cells
- E4 action_dist sums to 1.0 ± 0.001 per cell
- 各 metric 至少 1 个 cell 显示 |effect| > 0.1（否则 prompt 错了）

## 不要做的事

- 不要重写 `analyze_confidence_calibration.py`，只做 cross-condition aggregator 读它输出
- 不要尝试 element-text extraction（复杂 HTML 解析，留 future work）
- 不要 commit
- 不要碰 P-prompt 数据（reddit 跑中, partial），如果检测到 < 200 episodes 就 mark "partial / pending"

## token 预算

~80K (read 5 mode × 444 step JSONL summaries + write script + run + write JSON+MD)

## 触发命令

```bash
~/.npm-global/bin/codex exec --skip-git-repo-check \
  -C /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents \
  -s danger-full-access \
  --output-last-message logs/codex_mechanism_per_task.last.md \
  - < docs/checkpoints/codex_prompts/mechanism_per_task_explanation.md \
  > logs/codex_mechanism_per_task.run.log 2>&1 &
```
