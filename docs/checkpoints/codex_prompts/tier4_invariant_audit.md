# Codex Tier 4 — Invariant audit on paper-grade episode data

## 任务背景

Tier 1 (静态扫) + Tier 2 (信号挖矿) + Tier 3 (lit review) + Click probe 覆盖 source × approach × layer × scope 矩阵的 80%。**残缺 cell：adversarial / invariant-based audit**——我们至今没 systematically 找过"runner state 与 page state 矛盾的 episode"。

Tier 3 综述提到的 5-category taxonomy 里第 5 类（Actionability Check Masking and Timeout Swallowing）和第 4 类（Evaluator State Drift）正是这种"action_success 跟实际 page state 不一致"的家族。Tier 4 的目标是把这类 bug 从 1730 paper-grade ep 数据里**主动 invariant violation 触发出来**——不依赖人脑想到 signature，让逻辑矛盾自报。

## 仓库

`/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents`，必须用 `.venv/bin/python3`。**不要 commit**。

## 待审计的 invariants (10 条)

每条 invariant 形式 `(violation_signature) → suspect category`。Codex 要：
1. 实现每条 invariant 检测函数
2. 跑 over 3500+ paper-grade episodes (per-step 数据)
3. 按 violation count 排序输出
4. 每条 violation 给 ≥ 3 case study task IDs

```python
INVARIANTS = [
    # I1: action 报成功但页面没变（runner 误报或者 silent NoOp）
    ("inv_action_success_but_no_change",
     lambda step, next_step: (
       step["action_success"] is True
       and step.get("page_changed") is False
       and step.get("text_similarity", 0) > 0.99
     )),

    # I2: action 报失败但页面变了（runner 漏报 success）
    ("inv_action_fail_but_page_changed",
     lambda step, next_step: (
       step["action_success"] is False
       and step.get("page_changed") is True
     )),

    # I3: 同 element_id click ≥3 次连续（agent 死循环）—— 应该被 cycle-detect 提前停
    ("inv_repeat_click_no_cycle_break",
     lambda steps: ...),

    # I4: env_step_ms ≥ 30s 但 action_type ∉ {wait, type-长文}
    # → 暗示 hidden Playwright timeout 被吞
    ("inv_long_step_unexplained",
     lambda step: (
       step["env_step_ms"] > 30000
       and step["action_type"] not in {"wait", "type", "scroll"}
     )),

    # I5: obs_url 不在前一步可达 URL set
    # → 神秘跳转, 暗示 popup / redirect / frame switch 异常
    ("inv_unexplained_url_jump",
     lambda step, prev_step: (
       step["obs_url"] != prev_step["obs_url"]
       and prev_step["action_type"] not in {"click", "goto", "back", "forward"}
       and not is_redirect_chain(prev_step["obs_url"], step["obs_url"])
     )),

    # I6: 连续两 step AXTree text similarity < 0.7 但 obs_url 没变
    # → 暗示 AJAX content load / popup overlay / async render
    ("inv_axtree_drift_same_url",
     lambda step, next_step: (
       step["obs_url"] == next_step["obs_url"]
       and text_similarity(step["obs_text"], next_step["obs_text"]) < 0.7
       and step["action_type"] not in {"goto", "click"}  # click 可能 popup 合理
     )),

    # I7: agent 自报 finish 但 task evaluator 拒收 (FP)
    ("inv_finish_but_eval_reject",
     lambda episode: (
       episode["finish_attempted"] is True
       and episode["raw_success"] is False
     )),

    # I8: episode 在 max_step 截断 但 last action 是 click 类
    # → max_iter masking silent failure
    ("inv_max_step_truncate_at_click",
     lambda episode: (
       episode["truncated_at_max_step"] is True
       and episode["last_action_type"] == "click"
     )),

    # I9: 同 element_id 在两个不同 step 上分配给不同 type 元素
    # (e.g. step 3 element_id=42 是 link, step 5 element_id=42 是 button)
    # → AXTree mapping stale-cache risk
    ("inv_element_id_role_drift",
     lambda steps: ...),

    # I10: action_success=True 且 page_changed=True 但 next obs_text 跟 prev 完全相同
    # → step 记录 与 实际 obs 不同步 (logger bug?)
    ("inv_state_change_but_obs_same",
     lambda step, next_step: ...),
]
```

如果某条 invariant 实现需要 fields 我们 step JSONL 没有的（例如 `page_changed` 真不存在），请明示并 skip 该 invariant + 在 output 里 mark "skipped: field N/A"。

## Inputs

```python
RUNS = [
    "results/visualwebarena/phase1/B0_3mode_classifieds_20260413",
    "results/visualwebarena/phase1/B0_3mode_reddit_20260422",
    "results/visualwebarena/phase1/B0_phantom_som_classifieds_20260426",
    "results/visualwebarena/phase1/B0_phantom_som_reddit_20260428",
    "results/visualwebarena/phase1/B0_phantom_text_classifieds_20260427",
    "results/visualwebarena/phase1/B0_phantom_text_reddit_20260427",
    "results/visualwebarena/phase1/B0_phantom_prompt_reddit_20260429",
    "results/visualwebarena/phase1/B0_dom_shopping_20260428",
    "results/visualwebarena/phase1/B1_3mode_classifieds_20260413",
    "results/visualwebarena/phase1/B1_3mode_reddit_20260413",
    "results/visualwebarena/phase1/B1_phantom_som_classifieds_20260428",
    "results/visualwebarena/phase1/B1_phantom_dom_classifieds_20260429",
]
```

读 step JSONL 用 `from p79.experiment.io_utils import read_jsonl_dedup`.

## 输出

```
docs/analysis/cross_sites/tier4_invariant_audit.md     (paper-ready ~1500-2500 words)
docs/analysis/cross_sites/tier4_invariant_audit.json   (machine-readable)
```

JSON schema:

```json
{
  "audit_date": "2026-04-30",
  "total_episodes_scanned": 3500,
  "total_steps_scanned": 70000,
  "invariants": [
    {
      "id": "I1",
      "name": "inv_action_success_but_no_change",
      "description": "...",
      "implementable": true,
      "n_violations": 47,
      "violation_pct_of_steps": 0.067,
      "mode_breakdown": {"DOM": 18, "SoM": 12, ...},
      "site_breakdown": {"classifieds": 10, "reddit": 30, "shopping": 7},
      "case_study_examples": [
        {"site": "reddit", "task": 81, "step": 4, "run": "B0_3mode_reddit", "mode": "DOM",
         "snippet": "<key obs / action excerpt>"}
      ],
      "candidate_root_cause": "...",
      "tier3_taxonomy_match": "Type 1 / Type 4 / ...",
      "novelty_assessment": "matches §106 already-known | matches probe agent's already-known | NEW finding"
    },
    ...
  ],
  "summary": {
    "n_new_findings_not_covered_by_t1_t2_probe": 0,
    "highest_novelty_invariant_id": "I3",
    "recommended_followup": "..."
  }
}
```

Markdown 同结构 + per-invariant 1 case study + Section 4 wiring 建议。

## 验证

跑完 self-check：
- 至少 7/10 invariants implementable (能拿到 fields)
- 每条 implementable invariant 给 ≥ 3 case study examples (除非 0 violation)
- summary section 说"哪些 invariant 触发的 violation 是 Tier 1/2/probe 没 cover 的 NEW finding"

## 不要做的事

- 不要 commit
- 不要重新跑 click probe (已完成)
- 不要重新分类 §106 ep (probe 已完成)
- 不要写 fix patch
- 如果 step JSONL 没某个 field, 不要 mock; mark "field unavailable" 然后 skip
- 不要 hand-roll text_similarity 全 corpus (用 simple shingle / hash 即可)

## token 预算

~50K (read step JSONL × 70K step + 实现 10 invariants + write report)

## 触发命令（参考）

```bash
~/.npm-global/bin/codex exec --skip-git-repo-check \
  -C /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents \
  -s danger-full-access \
  --output-last-message logs/codex_tier4_invariant_audit.last.md \
  - < docs/checkpoints/codex_prompts/tier4_invariant_audit.md \
  > logs/codex_tier4_invariant_audit.run.log 2>&1
```
