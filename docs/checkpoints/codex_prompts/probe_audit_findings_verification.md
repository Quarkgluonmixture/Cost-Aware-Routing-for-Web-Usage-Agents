# Codex Probe — Verify Tier 1/2/4/5 audit findings via Playwright replay

## 任务背景

Tier 1-5 audit 给出的是 **signature-based candidate bug list**。上次 §106 经历教训：原 scan signature "5.4% click silent fail" 经 Playwright replay 后真实 §106 只有 **1.6%**——其余是 agent decision error / popup / AJAX。

**所以现在每个 audit category 都需要同样级别 replay 验证才能进入 fix scope**。

仓库：`/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents`，`.venv/bin/python3` + Playwright。**不要 commit, 不要 modify framework**。

## 验证目标 — 6 个候选，按 paper-impact 排序

### A. TYPE silent failure (Tier 2: 549 ep / 12.22%)

**Claim**：Tier 2 signature 报 549 ep 出现 type silent fail（runner success=True 但 next obs 没 echo input value）。Tier 1 静态 audit 进一步指出 TYPE 用 element_id center → page.keyboard.type，**外加 Meta+A + Backspace** 可能清错字段。

**Verify 步骤**：

1. 从 `tier2_silent_failure_catalog.json` 的 `type_silent_failure.case_study_examples` 抽 **15 个 case**（不少于 5 site × mode 多样性，含 cls/red/shop 各几个）
2. 每 case 用 `.auth/<site>_state.json` + step JSONL 重放到该 step 前一帧
3. 抓取 element_id 对应的 union_bound + AXTree node + 真实 DOM 元素信息
4. **三种 click + type 路径对照**：
   - 现 framework: `page.mouse.click(union_bound_center)` then `page.keyboard.press("Meta+A")` then `Backspace` then `page.keyboard.type(text)`
   - locator path: `page.locator(...).click()` then `locator.fill(text)` (BrowserGym style)
   - JS click + value set: `el.focus()` then `el.value = text; el.dispatchEvent('input')`
5. 比较三条路径的：
   - 是否触发了**正确**的 `<input>` 元素
   - 是否触发了 form/AJAX submit
   - text 是否真进 input.value
   - 是否 `Meta+A` 误清了**其他**字段（关键 — 如果 center 命中错元素，`Meta+A` 会全选页面文字，`Backspace` 删掉 selected text）
6. 分类每 case → {`SCAFFOLD_TYPE_BUG`, `AGENT_BLANK_TYPE`, `AGENT_NO_FOLLOWUP`, `FRAMEWORK_CONFIG`, `REPLAY_FAIL`, `OTHER`}

**期望产出**：confusion matrix + 每类 case study + **真实 §107 (TYPE bug) blast radius 估计**（从 549 audit ep 修正到 N ep）。

### B. SCROLL silent failure (Tier 2: 667 ep / 14.85%)

**Claim**：Tier 2 报 SCROLL 后 viewport 不变 / obs_text similarity > 0.95。但 SCROLL 在底部已是 last page 时 viewport 不变是**合理**的，不是 bug。

**Verify 步骤**：

1. 从 case_study_examples 抽 10 case
2. Replay 到 step 前一帧，记录 viewport_position_y
3. Run scroll action via Playwright `page.evaluate("window.scrollBy(0, window.innerHeight)")`
4. 比较 scroll 前后 viewport_position_y 和 obs_text
5. 区分：
   - `LEGIT_SCROLL_AT_BOTTOM`: 已经在底，scroll 不动**正常**
   - `LEGIT_SCROLL_NO_OVERFLOW`: 页面就这么短
   - `SCAFFOLD_SCROLL_BUG`: 应该能 scroll 但 framework 漏掉（找 scroll-to-element）
   - `MODAL_SCROLL_TRAP`: agent scroll 文档但 modal/dialog 阻止
   - `REPLAY_FAIL`

**期望产出**：667 ep 中真 scaffold bug 比例（很可能 << 100%，多数是 LEGIT）。

### C. SELECT_OPTION silent failure (Tier 2: 149 ep)

**Claim**：Tier 1 静态发现 SELECT_OPTION 用 `locator.select_option()` **不传参数**（`parsed_code[-1]` 没 forward）。如果属实，dropdown 选择会无值或 clear。

**Verify 步骤**：

1. 从 case_study 抽 8 case
2. Replay 到 step 前一帧
3. 比较 framework dispatch vs `locator.select_option("正确值")` vs raw JS `el.value = X; el.dispatchEvent('change')`
4. 验证：framework 是否真的丢失参数，dropdown state 之后是否正确

**期望产出**：149 ep 中真 SELECT_OPTION arg-drop bug 数。

### D. I9 element_id role drift (Tier 4: 1127 violations)

**Claim**：同 element_id 在 step N 是 link，step M 是 button —— AXTree 节点 reuse / CDP nodeId 漂移。

**Verify 步骤**：

1. 从 audit JSON 抽 10 case study (step pairs where element_id 同但 role 不同)
2. Replay 到该 step
3. 检查 AXTree 该 element_id 当前指向什么 backend DOM node
4. 跨 step 是否真 reuse 了同 nodeId 给不同元素
5. 区分：
   - `SAME_ELEMENT_AXTree_RESHAPE`: 同元素本身改了 role（rare 但合理 — e.g., disabled state）
   - `STALE_NODEID_REUSE`: 框架 reuse 了 nodeId 给完全不同元素（**真 bug**）
   - `LOGGING_ARTIFACT`: step record 写错

### E. I10 state_change but obs same (Tier 4: 288 violations)

**Claim**：runner 记 `page_changed=True` 但 obs_text 没变 → **logger consistency bug**。

**Verify 步骤**：

1. 抽 6 case
2. Replay 到 step
3. 实际执行 action，比较 page state 是否真变 + obs reload 是否真没显示变化
4. 区分：
   - `LOGGER_BUG`: page 真没变，runner 记错
   - `OBS_CACHE_BUG`: page 变了 obs 没刷新
   - `INVISIBLE_CHANGE`: page hash 变了但 visible content 没变（e.g. 隐藏元素更新）

### F. I2 action_fail but page_changed (Tier 4: 25 violations)

**Claim**：runner 记 action_success=False 但实际 page changed → runner false negative。

**Verify**：抽 5 case，replay，验证 page 是否真变 + 为何 runner 判 fail。

## Replay 通用流程（每 case）

```python
# 1. Setup
ctx = browser.new_context(storage_state=".auth/<site>_state.json")
page = await ctx.new_page()

# 2. Navigate to task start_url
await page.goto(start_url)

# 3. Replay prior steps from step JSONL
for step in steps[:target_step_idx]:
    if step["action_type"] == "click":
        # 用现 framework dispatch (page.mouse.click(center))
        ...

# 4. At target step, run BOTH framework dispatch AND alternative dispatch
result_framework = framework_dispatch(action)
result_alternative = alternative_dispatch(action)

# 5. Diff outcomes — verify claim
```

**重要**：每 case 给 < 60s 限制。如 replay 时 site state 不可达（e.g., login expired / item deleted），mark `REPLAY_FAIL` 跳过，不再尝试。

## 输出

```
docs/analysis/cross_sites/probe_audit_verification.md         (~2000 words)
docs/analysis/cross_sites/probe_audit_verification.json
```

JSON schema:

```json
{
  "audit_date": "2026-04-30",
  "categories": {
    "type_silent_failure": {
      "audit_claim_blast_radius_pct": 12.22,
      "audit_claim_n_episodes": 549,
      "verified_n_cases_probed": 15,
      "classification_breakdown": {
        "SCAFFOLD_TYPE_BUG": 5,
        "AGENT_BLANK_TYPE": 4,
        "AGENT_NO_FOLLOWUP": 3,
        "FRAMEWORK_CONFIG": 1,
        "REPLAY_FAIL": 2,
        "OTHER": 0
      },
      "true_scaffold_bug_fraction": 0.33,
      "extrapolated_blast_radius_pct": 4.07,
      "case_studies_per_class": [...]
    },
    "scroll_silent_failure": {...},
    ...
  },
  "summary": {
    "tier_audit_overestimate_factor": "TYPE 3×, SCROLL 5×, ..." ,
    "true_paper_relevant_bugs": [...],
    "fix_scope_recommendation": [...]
  }
}
```

## 不要做的事

- 不要 commit
- 不要 modify framework code (probe only, no fix)
- 不要超过 60s per case (timeout → REPLAY_FAIL)
- 不要 mock 数据
- 不要重复跑 click probe (已完成)
- 不要 sample > 50 case 总（time budget control）
- 不要尝试 fix patch（this is verification, fix is separate phase）

## 验证

跑完 self-check：
- 6 个 category 全部 sample 至少 5 case (除非 audit ep 数 < 5, 那也明确说)
- 每类给真 vs claimed 的 fraction
- summary 节有 fix_scope_recommendation 列出真值得 fix 的 N 类

## token 预算

~80K (Playwright replay 每 case ~30-60s, 50 case × ~45s = ~37min wall + JSON/MD)

## 触发命令

```bash
~/.npm-global/bin/codex exec --skip-git-repo-check \
  -C /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents \
  -s danger-full-access \
  --output-last-message logs/codex_probe_audit_verification.last.md \
  - < docs/checkpoints/codex_prompts/probe_audit_findings_verification.md \
  > logs/codex_probe_audit_verification.run.log 2>&1
```
