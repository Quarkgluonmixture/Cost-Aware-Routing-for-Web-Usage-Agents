# Codex prompt: 3-axis effect size ablation (prompt vs image)

## 用途

`scripts/analysis/figures/fig1c_strategy_gradient.py` 用全数据 (reddit N=210, cls N=234) 显示 5 mode 的 4 个 macro behavior metrics。视觉看 reddit 行：P-DOM 靠 DOM、P-SoM 靠 SoM —— 暗示 **prompt axis 主导 strategy 方向**，**image axis 主导 refinement**。需要量化的 effect size 替代 visual inspection，写进 paper Section 5。

## Theory framing

3-axis hierarchical design cube (text-payload × prompt × image)。5 个跑过的 mode 在 cube 里坐标：

| Mode | text | prompt | image |
|---|---|---|---|
| DOM | AXTree | DOM | ❌ |
| Vision | empty | Vision | ✅ |
| SoM | [SOM_MARKS] | SoM | ✅ |
| Phantom-SoM | [SOM_MARKS] | SoM | ❌ |
| Phantom-DOM | [SOM_MARKS] | DOM | ❌ |

**两条干净的 axis 对比**（把另外两 axis 控住）：
- **Prompt axis**: `P-DOM vs P-SoM` — 控住 text=[SOM_MARKS] + image=no, 仅换 prompt
- **Image axis**: `SoM vs P-SoM` — 控住 text=[SOM_MARKS] + prompt=SoM, 仅去图

## 输入

每个 site × mode 的 step JSONL 在：
```
results/visualwebarena/phase1/B0_3mode_<site>_<DATE>/phase1_dom_router_0/episodes/<site>_task_<id>_steps_v2.jsonl
results/visualwebarena/phase1/B0_3mode_<site>_<DATE>/phase1_som_router_0/episodes/...
results/visualwebarena/phase1/B0_3mode_<site>_<DATE>/phase1_vision_router_0/episodes/...
results/visualwebarena/phase1/B0_phantom_<site>_<DATE>/phase1_phantom_som_router_0/episodes/...
results/visualwebarena/phase1/B0_phantom_dom_<site>_<DATE>/phase1_phantom_dom_router_0/episodes/...
```

具体路径（与 fig3 脚本一致）:
```python
STEP_DIRS = {
    "reddit": {
        "DOM": "B0_3mode_reddit_20260422/phase1_dom_router_0/episodes",
        "Vision": "B0_3mode_reddit_20260422/phase1_vision_router_0/episodes",
        "SoM": "B0_3mode_reddit_20260422/phase1_som_router_0/episodes",
        "Phantom-SoM": "B0_phantom_reddit_20260428/phase1_phantom_som_router_0/episodes",
        "Phantom-DOM": "B0_phantom_dom_reddit_20260427/phase1_phantom_dom_router_0/episodes",
    },
    "classifieds": {
        "DOM": "B0_3mode_classifieds_20260413/phase1_dom_router_0/episodes",
        "Vision": "B0_3mode_classifieds_20260413/phase1_vision_router_0/episodes",
        "SoM": "B0_3mode_classifieds_20260413/phase1_som_router_0/episodes",
        "Phantom-SoM": "B0_phantom_classifieds_20260426/phase1_phantom_som_router_0/episodes",
        "Phantom-DOM": "B0_phantom_dom_classifieds_20260427/phase1_phantom_dom_router_0/episodes",
    },
}
```

文件名 pattern：`<site>_task_<id>_steps_v2.jsonl`，task_id 从 0 开始（reddit 0..209, cls 0..233）。

## Per-task metric 计算

每个 task 的 step JSONL 读完后算 4 个 per-task metric:

```python
total_steps = len(steps)
typed_steps = sum(1 for s in steps if s.get("action_type") == "type" or s.get("action", {}).get("action_type") == "type")
scroll_steps = sum(1 for s in steps if s.get("action_type") == "scroll" or s.get("action", {}).get("action_type") == "scroll")

# search_steps: step's obs_url contains '/search' (reddit) or 'page=search'/'/search' (cls);
# OR is a "type" step whose NEXT step's obs_url is search (i.e., agent triggered search)
SEARCH_MARKERS = {"reddit": ("/search",), "classifieds": ("page=search", "/search")}
search_steps = 0
for i, s in enumerate(steps):
    url = s.get("obs_url", "")
    next_url = steps[i+1].get("obs_url", "") if i+1 < len(steps) else ""
    at = s.get("action_type") or s.get("action", {}).get("action_type")
    if any(m in url for m in SEARCH_MARKERS[site]):
        search_steps += 1
    elif at == "type" and any(m in next_url for m in SEARCH_MARKERS[site]):
        search_steps += 1

# self-correction: thought tokens
selfcorr_steps = 0
for s in steps:
    a = s.get("action") or {}
    thought = (a.get("thought", "") if isinstance(a, dict) else "").lower()
    if any(t in thought for t in ("mistake", "wrong", "try again", "go back")):
        selfcorr_steps += 1

# Per-task metrics:
per_task["search_loop_bin"] = 1 if search_steps >= 2 else 0     # binary
per_task["type_frac"]       = typed_steps / total_steps if total_steps else None
per_task["scroll_frac"]     = scroll_steps / total_steps if total_steps else None
per_task["selfcorr_count"]  = selfcorr_steps                    # count, not normalized to keep scale同 fig3
```

## Effect size 计算

对每个 site × metric 组合，做两个 paired contrast（同 task）：

**Prompt axis**: `pair_prompt = (P-DOM[task_id], P-SoM[task_id])` for each task with both modes' data。
**Image axis**: `pair_image = (SoM[task_id], P-SoM[task_id])` for each task with both modes' data。

每对计算：
1. **mean diff** (signed: prompt = P-DOM − P-SoM; image = SoM − P-SoM)
2. **Cohen's d_z** = mean(diff) / std(diff)，paired version。对 binary metric (search_loop) 用 **Cohen's h** 改写：h = 2·(arcsin(√p1) − arcsin(√p2))
3. **Bootstrap 95% CI** of mean diff (n_boot=2000, paired resampling at task level)
4. **Wilcoxon signed-rank p-value** (two-sided, exclude ties)

## 输出

写到 `docs/analysis/cross_sites/axis_effect_size.json`:

```json
{
  "method": "paired contrasts on per-task metrics; Cohen's d_z (continuous) / Cohen's h (binary search-loop); bootstrap 95% CI (n=2000); Wilcoxon signed-rank.",
  "axes": {
    "prompt": {"contrast": "P-DOM minus P-SoM", "controls": "text=[SOM_MARKS], image=no"},
    "image":  {"contrast": "SoM minus P-SoM",  "controls": "text=[SOM_MARKS], prompt=SoM"}
  },
  "results": {
    "reddit": {
      "search_loop": {
        "prompt": {"n": 210, "mean_diff_pct_pts": 13.81, "cohen_h": 0.29, "ci95": [...], "wilcoxon_p": 0.001},
        "image":  {"n": 210, "mean_diff_pct_pts": -4.29, "cohen_h": -0.10, "ci95": [...], "wilcoxon_p": 0.34}
      },
      "type_frac": { "prompt": {...}, "image": {...} },
      "scroll_frac": {...},
      "selfcorr_count": {...}
    },
    "classifieds": {... same structure ...}
  },
  "interpretation": {
    "headline": "...",
    "prompt_dominates": ["search_loop@reddit", "type_frac@reddit", ...],
    "image_dominates":  ["scroll_frac@reddit", "selfcorr@reddit", ...],
    "site_specific_notes": {
      "classifieds": "..."
    }
  }
}
```

并写一份 ~250-word markdown report 到 `docs/analysis/cross_sites/axis_effect_size_report.md`，结构：

```
## Headline finding
[一句话：是否支持 "prompt 主导 strategy 方向 / image 主导 refinement"]

## Per-metric breakdown table
| site | metric | prompt-axis effect | image-axis effect | which dominates |
|---|---|---|---|---|

## Reddit 行 (cleanest case)
[3-4 sentences explaining the pattern + which metrics confirm theory]

## Classifieds 行 (site-specific divergence)
[3-4 sentences flagging where cls deviates and probable mechanism]

## Paper Section 5 implication
[2-3 sentences: 是否 strengthen "axis 2 prompt has independent first-order macro effect" + how to write up]
```

## 验证 step

跑完后 codex 自验证:
- N(reddit) per pair contrast = 210 (full overlap, since all 5 modes covered all 210 tasks)
- N(cls) per pair contrast = 234
- 任何 |Cohen's d_z| > 0.5 或 |Cohen's h| > 0.3 都视作 "meaningful"
- 任何 metric 上 prompt 和 image effect 的 CI 完全重叠 → flag "indistinguishable"

## 触发命令

```bash
~/.npm-global/bin/codex exec --skip-git-repo-check \
  -C /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents \
  -s workspace-write \
  --output-last-message logs/codex_axis_effect.last.md \
  - < docs/checkpoints/codex_prompts/axis_effect_size_ablation.md
```

## token 预算

~30-50K (read 5 modes × 444 step JSONL files header + compute + write JSON + report)
