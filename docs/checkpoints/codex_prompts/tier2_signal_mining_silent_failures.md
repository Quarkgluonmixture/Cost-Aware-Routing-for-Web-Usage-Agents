# Codex Tier 2 — Signal mining for non-click silent failures (TYPE / SCROLL / SELECT / FINISH / cross-step)

## 任务背景

Probe agent 已经把 click silent failures 分完类（n=93 → §106 27ep + agent decision 33ep + popup 12ep + button/AJAX 7ep + other 14ep）。但 **non-click action types 还没扫**。可能藏 §107 / §108 类同质 bug。

我们 paper-grade dataset：

```
results/visualwebarena/phase1/
  B0_3mode_classifieds_20260413/   234 ep × 3 mode (DOM/SoM/Vision)
  B0_3mode_reddit_20260422/        210 ep × 3 mode
  B0_phantom_som_classifieds_20260426/    234 ep × 1 mode (P-SoM)
  B0_phantom_som_reddit_20260428/         210 ep × 1 mode
  B0_phantom_text_classifieds_20260427/   234 ep × 1 mode (P-text)
  B0_phantom_text_reddit_20260427/        210 ep × 1 mode
  B0_phantom_prompt_reddit_20260429/      ~155/210 ep (P-prompt, live partial)
  B0_dom_shopping_20260428/        465/466 ep (DOM only)
  B1_3mode_classifieds_20260413/   234 ep × 3 mode
  B1_3mode_reddit_20260413/        210 ep × 3 mode
  B1_phantom_som_classifieds_20260428/    234 ep × 1 mode
  B1_phantom_dom_classifieds_20260429/    ~67/234 ep (P-text, live partial)
```

总 ~3500 episode 跨 baseline × site × mode。每个 episode 有 `*_steps_v2.jsonl` (per-step record) 和 `*_summary_v2.json` (episode aggregate)。

## 任务

对每个 non-click ActionType，挖**silent failure signature**——agent 发出 action，runner 记录 action_success=True 或 False，但 page 实际状态与 agent 期望不一致，且 agent 自身没有 trajectory signal 知道这件事。

### 挖矿目标 ActionType（5 类）

#### 1. TYPE silent failure

Pattern：`action_type == "type"` 但下一帧 obs 里目标 input field 的值没变／为空／不是 agent type 的 text。

具体 signature：
- 同一 `element_id` 上连续 ≥ 2 次 type 但 next obs 该 field 仍是 placeholder
- type 后下一 step agent 仍在 same URL（no submit triggered）但 obs `[input_field_value]` 没出现
- env_step_ms 异常（≥ 5s 暗示 keyboard event 未触发）
- `action.text` 长度 > 0 但 next-step `obs_text` 没含该 substring

Hint：reddit 上 agent 经常 type 进 search box 但 search 没提交（probe agent 已发现 22 ep 这种 case 但归 click）。

#### 2. SCROLL silent failure

Pattern：scroll action 但 viewport 实际没动。

Signature：
- 连续 ≥ 2 scroll 但 obs `viewport_position_y` 不变（如果 obs 含此字段；否则用 obs_text hash）
- scroll 后 obs_text similarity > 0.95（页面没换内容）
- env_step_ms < 200ms (scroll 太快暗示 short-circuit)

#### 3. SELECT_OPTION silent failure

Pattern：agent select 一个 dropdown option 但 page state 没更新。

Signature：
- `action_type == "select_option"` 但下一帧 obs 该 dropdown 仍显示原 default
- form submit 没触发 (URL 没变)

#### 4. FINISH wrong-state failure

Pattern：agent 自报 finish 但 task 未完成。Adjusted_success=False but agent thought it done.

Signature：
- action_type == "finish" 且 `episode["raw_success"] == False` (agent 认为成功 → string_match 等评估被骗)
- 或 `adjusted_success == False but episode["finish_attempted"] == True`
- agent thought 含 "done" / "completed" / "found" 但 obs URL ≠ target URL

这条已有部分 evidence (我们的 FP rate 0.b)。Tier 2 任务是 cluster FP 的 root cause sub-categories（误读 obs / agent 假信心 / 评估 mismatch / hallucinated answer string）。

#### 5. Cross-step trajectory anomaly

Pattern：连续两 step AXTree 大幅变化但没 navigation action（`obs_url` 不变）→ 暗示 frame-side 异常注入（popup overlay / async content load / AJAX inject / stale cache）。

Signature：
- step N URL == step N+1 URL but axtree text similarity < 0.7
- step N action 不是 navigation type (click 触发了 popup/modal 才合理)
- env_step_ms (step N+1) 异常长

## 输入

```python
RUNS = [
    "results/visualwebarena/phase1/B0_3mode_classifieds_20260413",
    "results/visualwebarena/phase1/B0_3mode_reddit_20260422",
    "results/visualwebarena/phase1/B0_phantom_som_classifieds_20260426",
    "results/visualwebarena/phase1/B0_phantom_som_reddit_20260428",
    "results/visualwebarena/phase1/B0_phantom_text_classifieds_20260427",
    "results/visualwebarena/phase1/B0_phantom_text_reddit_20260427",
    "results/visualwebarena/phase1/B0_phantom_prompt_reddit_20260429",  # partial
    "results/visualwebarena/phase1/B0_dom_shopping_20260428",
    "results/visualwebarena/phase1/B1_3mode_classifieds_20260413",
    "results/visualwebarena/phase1/B1_3mode_reddit_20260413",
    "results/visualwebarena/phase1/B1_phantom_som_classifieds_20260428",
    "results/visualwebarena/phase1/B1_phantom_dom_classifieds_20260429",  # partial
]
```

每 run 下 condition_dir/episodes/*_steps_v2.jsonl 和 *_summary_v2.json。

读 step JSONL 用 `p79.experiment.io_utils.read_jsonl_dedup` (已在仓库)。

## 输出

```
docs/analysis/cross_sites/tier2_silent_failure_catalog.json    # machine-readable
docs/analysis/cross_sites/tier2_silent_failure_catalog.md      # paper-ready (~2000 words)
```

JSON schema:

```json
{
  "audit_date": "2026-04-30",
  "total_episodes_scanned": 3500,
  "total_steps_scanned": 70000,
  "categories": {
    "type_silent_failure": {
      "n_episodes": 47,
      "blast_radius_pct": 1.34,
      "mode_breakdown": {"DOM": 18, "SoM": 12, ...},
      "site_breakdown": {"classifieds": 10, "reddit": 30, "shopping": 7},
      "case_study_task_ids": [{"site": "reddit", "task": 81, "run": "B0_3mode_reddit", "mode": "DOM"}, ...],
      "candidate_root_cause": "..."
    },
    "scroll_silent_failure": {...},
    "select_option_silent_failure": {...},
    "finish_wrong_state": {
      "subcategories": {
        "agent_hallucinated_answer": 12,
        "agent_finished_on_search_results_page_not_target": 8,
        ...
      }
    },
    "cross_step_trajectory_anomaly": {...}
  },
  "cross_action_summary": {
    "total_silent_failures_estimated": 200,
    "fraction_of_all_failures": 0.X,
    "site_with_highest_concentration": "...",
    "mode_with_highest_concentration": "..."
  }
}
```

Markdown 同样结构 + per-category 1-2 case study task ID 解读 + paper Section 4 wiring 建议。

## 验证

跑完 self-check：
- 至少检测出 ≥ 5 个 episode per category（除非真的没有，那也明确说）
- mode_breakdown 加总 = n_episodes
- case_study_task_ids 至少 3 条 per category，给具体 (site, task_id, run, mode)
- cross_action_summary 给 total estimate

## 不要做的事

- 不要 commit
- 不要重做 click 分类（probe agent 已完成，引用即可）
- 不要扫 WA runs (我们没 WA 数据)
- 不要尝试 fix——这是 audit not fix
- 不要 mock 数据—— absent run path 跳过并记 "skipped (data not available)"
- 不要超过 3500 episode 总扫描——partial run 当 partial 处理

## 跟 Tier 1 + click probe 的协作

- Tier 1 (静态扫) 输出 candidate bug list, Tier 2 (信号挖矿) 输出 actual blast radius —— 两边对照能 confirm/refute candidate
- click probe (已完成, 93 ep 分类) + Tier 2 (5 类 non-click) = 完整 silent-failure 全景

汇总放 `docs/analysis/cross_sites/silent_failure_master_catalog.md`（不在此次任务，等 3 个 audit 都完成后另起）

## token 预算

~80K (read step JSONL × ~70K step + write catalog + json)

## 触发命令

```bash
~/.npm-global/bin/codex exec --skip-git-repo-check \
  -C /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents \
  -s danger-full-access \
  --output-last-message logs/codex_tier2_signal_mining.last.md \
  - < docs/checkpoints/codex_prompts/tier2_signal_mining_silent_failures.md \
  > logs/codex_tier2_signal_mining.run.log 2>&1
```
