# Codex prompt: axis-1 micro-behavior decomposition (reddit + classifieds, cross-site validation)

## 用途

`scripts/analysis/axis_effect_size.py` 的 fig3 ablation 显示 **axis 1 (text payload structure: AXTree → [SOM_MARKS]) 在 4 个 macro-action-frequency 指标 (search/type/scroll/selfcorr) 上 0/8 dominant**，但 fig2/fig11/oracle drop-one 都显示 axis 1 在 **task SR 层 first-order**。这种 macro-vs-outcome 张力的 paper hypothesis：**axis 1 改的是 per-step decision quality 不是 action-type 频率**——agent 用同样比例打字 / 点击，但**点的元素和走的页面完全不同**。

直接 element-id Jaccard 不可行（AXTree id 和 [SOM_MARKS] mark 不同 ID 系统，不可比）。改用 **mode-invariant** 的 anchor：URL trajectory、target-page hit、search keyword、first-action transition。

## 关键要求：cross-site 验证

**不能只算 reddit，必须 reddit + classifieds 都做**——如果只 reddit 看到 "axis 1 strong on decision-quality" 而 cls 看不到，结论就是 **overfit 到 reddit search-loop 失败模式** 这一个特定 case，paper 主 claim 站不住。

cls 的不同之处需要 codex 处理：
- cls 的 search 是 OSClass 任务**正常路径**（参考 `scripts/analysis/figures/fig1c_strategy_gradient.py` footnote: "OSClass tasks intrinsically use search pages"），所以 search-keyword 重复率 absolute 值都高——要看 **axis 间 differential**（DOM vs P-DOM 的差），不看绝对 level
- cls 的 target page 是 listing/category，URL pattern 不同
- cls 视觉信息（产品图）至关重要，axis 3 (image) 在 cls 上预期更强（这与 fig3 ablation 一致）

## Inputs

### Step JSONL（5 mode × 2 site）

```python
STEP_DIRS = {
    "reddit": {
        "DOM": "results/visualwebarena/phase1/B0_3mode_reddit_20260422/phase1_dom_router_0/episodes",
        "Vision": "results/visualwebarena/phase1/B0_3mode_reddit_20260422/phase1_vision_router_0/episodes",
        "SoM": "results/visualwebarena/phase1/B0_3mode_reddit_20260422/phase1_som_router_0/episodes",
        "Phantom-SoM": "results/visualwebarena/phase1/B0_phantom_reddit_20260428/phase1_phantom_som_router_0/episodes",
        "Phantom-DOM": "results/visualwebarena/phase1/B0_phantom_dom_reddit_20260427/phase1_phantom_dom_router_0/episodes",
    },
    "classifieds": {
        "DOM": "results/visualwebarena/phase1/B0_3mode_classifieds_20260413/phase1_dom_router_0/episodes",
        "Vision": "results/visualwebarena/phase1/B0_3mode_classifieds_20260413/phase1_vision_router_0/episodes",
        "SoM": "results/visualwebarena/phase1/B0_3mode_classifieds_20260413/phase1_som_router_0/episodes",
        "Phantom-SoM": "results/visualwebarena/phase1/B0_phantom_classifieds_20260426/phase1_phantom_som_router_0/episodes",
        "Phantom-DOM": "results/visualwebarena/phase1/B0_phantom_dom_classifieds_20260427/phase1_phantom_dom_router_0/episodes",
    },
}
```

每 step JSONL line 是一个 dict，至少含 `obs_url`, `action_type` (or `action.action_type`), `action.text` (for type actions), `step_idx`。

### Task config（用于 target-page extraction）

- reddit: `external/visualwebarena/config_files/vwa/test_reddit.json` — 每个 task 有 `intent`, `eval` (含 `reference_url` 或 `program_html` 的 url pattern)
- classifieds: `external/visualwebarena/config_files/vwa/test_classifieds.json` — 同上

target URL 从 `eval.reference_url` 或 `eval.program_html[].url` 提取（取第一个非 empty 的 URL）；如果没有就跳过该 task 的 target-hit metric（其他 metric 仍可计算）。

## Mode-invariant metrics（per task per mode）

每个 task 在每个 mode 下提一组：

```python
per_task_mode_metrics = {
    "url_set": set(steps_obs_url),                          # 该 task 该 mode 访问过的 distinct URL 集合
    "url_path_set": set(URL.path for url in url_set),       # 仅 path 部分（忽略 query/fragment),  cross-mode 更稳
    "n_url_visits": len(url_set),                            # coverage 大小
    "target_url_visited": bool(target_url_prefix in any url),# 是否 reach target page
    "search_keywords": [action.text for type-actions],       # 所有打字动作 text（lowercased, stripped）
    "n_type_actions": len(search_keywords),
    "max_keyword_repeat": max(Counter(search_keywords).values() or [0]),  # 最重复的 keyword 出现次数（>5 = search loop)
    "distinct_keywords": len(set(search_keywords)),
    "first_action_type": steps[0].action_type,               # type / click / scroll / finish
    "first_action_target_url_path": steps[1].obs_url.path if len(steps) > 1 else None,  # step 0 click 后落在哪
    "n_steps": len(steps),
    "reached_finish": last action is finish,
}
```

## Cross-mode contrasts（per site）

对每对 cascade-adjacent mode（同 axis swap）算 paired contrast：

| Contrast | Axis | Same task → 比什么 |
|---|---|---|
| `DOM vs P-DOM` | axis 1 (text) | URL-set Jaccard / target-hit-rate diff / search-keyword diff |
| `P-DOM vs P-SoM` | axis 2 (prompt) | 同上 |
| `P-SoM vs SoM` | axis 3 (image) | 同上 |

每个 axis 上的 metric 算：
- **URL-path Jaccard**: per task, J = |A ∩ B| / |A ∪ B|，axis 上的 mean Jaccard 越低 = decision divergence 越大
- **Target-hit rate diff**: per axis, share of tasks reaching target URL — 比较两 mode 的 hit rate（McNemar test 或 paired binary）
- **Max keyword repeat diff**: per axis, paired diff in `max_keyword_repeat` (search-loop intensity)
- **Distinct keywords diff**: paired diff in `distinct_keywords` (是否多样 reformulate vs 单一关键词重复)
- **First-action divergence rate**: per axis, share of tasks where first action type differs

## 关键 paper claim 检验（cross-site)

**Claim**: axis 1 (text 结构) 的 decision-quality effect 大于 macro-action-frequency effect。

**Test**: 比较 axis 1 在两类 metric 上的 effect size:
- "decision-quality bucket": URL-Jaccard, target-hit, first-action divergence
- "macro-action-freq bucket": (从已存的 `axis_effect_size.json` 读 axis 1 的 search/type/scroll/selfcorr effect)

per site 计算 `mean(|effect|_decision) / mean(|effect|_macro)`，如果 ratio > 1 ✓ claim 成立。

**Cross-site validity**:
- 如果 reddit 和 cls 都 ratio > 1 → claim **generalizes**, paper 可写
- 如果只 reddit 满足 → 标 "site-specific", paper 限定到 reddit 论述
- 如果两 site 都不满足 → claim 不成立，paper 改写

JSON 里 `cross_site_validity` block 写明结论。

## 输出

- `scripts/analysis/axis1_microbehavior.py` (新文件，不动 axis_effect_size.py)
- `docs/analysis/cross_sites/axis1_microbehavior.json`
- `docs/analysis/cross_sites/axis1_microbehavior_report.md`

JSON schema:
```json
{
  "method": "...",
  "metrics_per_task_per_mode": {  /* 不要 dump 全 N=210/234 list, 只放 summary stats per mode */
    "reddit": { "DOM": {"mean_url_set_size": ..., ...}, ... },
    "classifieds": {...}
  },
  "axis_contrasts": {
    "reddit": {
      "axis_1_text": {
        "url_jaccard_mean": 0.42,
        "target_hit_rate_diff": -0.18,  /* P-DOM higher than DOM by 18pp */
        "max_keyword_repeat_diff": -3.5, /* P-DOM has 3.5 less repeat */
        "first_action_divergence_rate": 0.62,
        ...
      },
      "axis_2_prompt": {...},
      "axis_3_image": {...}
    },
    "classifieds": {... same shape ...}
  },
  "cross_site_validity": {
    "claim": "axis 1 decision-quality effect > axis 1 macro-action-freq effect",
    "reddit_ratio": 2.4,
    "classifieds_ratio": 1.1,
    "verdict": "generalizes / site-specific / not supported",
    "narrative": "..."
  }
}
```

Markdown report:
- Headline finding
- Per-site table: axis 1 effect on each metric
- Cross-site validity verdict
- 3-5 case studies (具体 reddit task_23 / 30 / 4 + 类似 cls task; 看 DOM trajectory vs P-DOM trajectory)
- Paper Section 5 implication

## 验证

- N(reddit) per axis 1 contrast = 210
- N(cls) per axis 1 contrast = 234
- target-hit rate per task 计算时 task_config 里能 extract URL 的 task 子集报清楚 N
- url-path Jaccard mean 落在 [0, 1]
- cross-site verdict 必须给一个明确字符串（不能 "inconclusive"）

## token 预算

~50K (read step JSONL summaries + task configs + write script + run + write JSON+md)

## 触发命令（**run in background**）

```bash
~/.npm-global/bin/codex exec --skip-git-repo-check \
  -C /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents \
  -s danger-full-access \
  --output-last-message logs/codex_axis1_micro.last.md \
  - < docs/checkpoints/codex_prompts/axis1_microbehavior.md \
  > logs/codex_axis1_micro.run.log 2>&1 &
```
