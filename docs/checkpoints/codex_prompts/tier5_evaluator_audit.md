# Codex Tier 5 — Evaluator-side static audit

## 任务背景

Tier 1-4 + click probe 已 cover dispatch / observation / trajectory / invariant 4 个层面。**剩下未 audit 的层 = evaluator logic 本身**——也就是 `string_match` / `ua_match` / `program_html` 这三个 evaluator 在判定 success 时的 systematic bias。

我们已经 build 了 FP filter 体系（na_fp / eval_fp / visual_fp，详见 `docs/checkpoints/实验笔记.md` §78a + §95），它做的是 **post-hoc detect + correct evaluator FP**——但只能修正 GPT-judge 二判能 catch 的 case。如果 evaluator code 本身有 logic bug（e.g. fuzzy threshold 漏报、selector brittle、prompt template drift），FP filter 是 mitigation 不是 root-cause fix。

Tier 5 目标：**read evaluator 源码，找 logic-level bug**——补 evaluator 这层的 audit coverage，paper Section 4 限制条款用。

## 仓库

`/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents`，必须用 `.venv/bin/python3`。**不要 commit**。

## 待审计文件

```
external/visualwebarena/evaluation_harness/
├── evaluators.py              ← 主 entry, 包含 string_match / ua_match / program_html 路由
├── helper_functions.py        ← url 解析, llm_fuzzy_match, eval helpers
├── webarena_utils.py          (如果存在)
└── (其他 *.py)

external/webarena/evaluation_harness/      (同款上游)
└── ...

# 我们 fork 的 FP 体系（对照参考，不审计）
docs/checkpoints/实验笔记.md  §78a (na_fp 定义) + §95 (eval_fp/visual_fp 最新方案)
p79/experiment/analysis.py  (adjusted_success canonical)
scripts/analysis/analyze_*.py    (FP filter 实现)
```

## 审计内容

### Audit A — `ua_match` GPT-judge prompt template

`ua_match` 用 GPT-4o-mini 二判 task answer 是否符合 intent。审计：

1. **Prompt template 全文** — 找 file:lineno + 全文 paste 进 audit doc
2. **Known drift modes**：
   - 是否对 N/A task 有 systematic bias?（agent 说"task is impossible"时 GPT 倾向 accept 为 success，这是 §78a na_fp 的根因）
   - 是否对 ambiguous task instruction 有 inconsistent 判定?（同 trajectory 跑 N 次，输出会不会变化？）
   - Temperature / model_name 是否 fixed?
3. **Existing FP filter 是否完整 cover**：交叉对照 `analyze_reason_diagnostics.py` / `analyze_search_over_browse.py` 等的 filter logic
4. **Recommended fixes**：
   - Prompt template wording 改进
   - 多次 sampling + majority vote
   - 替换 GPT 为 deterministic rule

### Audit B — `string_match` fuzzy threshold

`string_match` 直接对比 agent 输出 string 与 target answer。审计：

1. **Fuzzy logic file:lineno + 函数全文**
2. **Threshold 设定**：
   - 是否 case-sensitive?
   - 是否 strip whitespace / punctuation?
   - 是否 fuzzy match (Levenshtein / token Jaccard / substring)?
   - threshold 在哪里设, 是否合理?
3. **Edge case 风险**：
   - agent 输出 "$5.99" vs target "5.99" → match 还是 miss?
   - agent 输出 "five point nine nine dollars" vs target "5.99"?
   - Multi-answer task (target 是 list) 怎么 match?
4. **False positive / false negative 估计**：从我们 paper-grade 数据 sample 20 episode 验 (or 抽样 raw_success 与 GPT 二判结果差异最大的 case)

### Audit C — `program_html` DOM selector brittleness

`program_html` 评估通过 navigate 到 target URL + run JS 检查 DOM state 是否符合 target。审计：

1. **Selector pattern**：
   - 用 `document.querySelector` / `XPath` / `CSS selector` 哪种?
   - selector 是 `id` / `class` / `nth-of-type` / `[attr=value]`?
2. **Brittleness 风险**：
   - target site (Magento / Postmill / Reddit) 改版后 selector 是否 still valid?
   - DOM async render 时 selector 是否有 wait 逻辑?
3. **Examples from our task pool**：抽 10 个 program_html eval task, 列 selector + 评估 brittleness

### Audit D — Cross-reference 我们的 FP 体系

读 `docs/checkpoints/实验笔记.md §78a + §95` (我们的 na_fp / eval_fp / visual_fp 现状)。然后回答：

1. Audit A-C 找到的 evaluator bug 中, 哪些已经被我们 FP filter 盖住?
2. 哪些还没盖住 (gap)?
3. 推荐 paper Section 4 怎么 frame: 
   - "We extend prior FP-mitigation approaches with adjusted_success metric, identifying additional eval-side biases [from Audit A-C] not covered by post-hoc filtering" 
   或类似

## 输出

```
docs/analysis/cross_sites/tier5_evaluator_audit.md      (~2000-3000 words)
docs/analysis/cross_sites/tier5_evaluator_audit.json    
```

JSON schema:

```json
{
  "audit_date": "2026-04-30",
  "ua_match": {
    "prompt_template_file": "external/visualwebarena/evaluation_harness/helper_functions.py",
    "prompt_template_lineno": 234,
    "prompt_template_full_text": "...",
    "drift_modes_identified": [
      {"mode": "N/A task accept bias", "evidence": "...", "covered_by_existing_fp_filter": true},
      ...
    ],
    "recommended_fixes": [...]
  },
  "string_match": {
    "function_file": "...",
    "function_lineno": ...,
    "fuzzy_threshold": 0.X,
    "case_sensitivity": true,
    "edge_case_concerns": [...]
  },
  "program_html": {
    "selector_pattern": "...",
    "brittleness_concerns": [...],
    "task_pool_audit": [...]
  },
  "fp_filter_cross_reference": {
    "existing_coverage": "...",
    "gaps": [...],
    "section4_framing_recommendation": "..."
  }
}
```

## 验证

完成后 self-check：
- ua_match prompt template 全文进入 JSON (truncate 到 2000 char if 超长)
- string_match fuzzy threshold 数字明确给出
- program_html ≥ 5 selector example
- fp_filter_cross_reference 至少 1 个 gap (除非 evaluator 真的 bug-free, 那也明确说)

## 不要做的事

- 不要 commit
- 不要 modify evaluator 源码 (audit not fix)
- 不要重新 verify §105 swatch (我们 state_change.py 已修)
- 不要重新跑 episode (静态读 + sample 抽查即可)
- 不要无视 §78a / §95 chronicle (那是关键参考)

## token 预算

~30K (read evaluator 源码 + 我们 §78a/§95 + write audit doc + json)

## 触发命令

```bash
~/.npm-global/bin/codex exec --skip-git-repo-check \
  -C /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents \
  -s danger-full-access \
  --output-last-message logs/codex_tier5_evaluator_audit.last.md \
  - < docs/checkpoints/codex_prompts/tier5_evaluator_audit.md \
  > logs/codex_tier5_evaluator_audit.run.log 2>&1
```
