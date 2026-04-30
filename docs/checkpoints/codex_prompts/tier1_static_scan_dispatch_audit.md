# Codex Tier 1 — Static scan of WA/VWA upstream dispatch + AXTree extraction

## 任务背景

我们在 `web-arena-x/visualwebarena` 和 `web-arena-x/webarena` 上游代码里已确认两个 silent-failure bug：

- **§105 swatch radio dict-key collision** — 我们的 `state_change.py` 里，product variant radio 修改没被 `_key()` 收录，已修
- **§106 union_bound center mismatch** — `browser_env/processors.py:786-795 get_element_center()` 用 AXTree 节点 `union_bound` (父容器 bbox) 中心做 click target，再 `browser_env/actions.py:1305-1308` 调 `page.mouse.click(x, y)` 直接发坐标，导致 inline 多行 `<a>` 或 listing-card 容器场景下命中父元素 silent-fail。Probe 实证 27 ep / 1.6% blast radius，DOM:SoM = 1.7×。

Probe agent 同时发现：
- 12 ep (13%) `POPUP_OR_TARGET_BLANK` — `<a target="_blank">` 在 headless playwright 下 click 不跟新 tab
- 7 ep (8%) `BUTTON_OR_AJAX` — `<button>` form/CSRF 失败，与 click dispatch 无关

现在要 systematically scan 上游代码，**找出还没被 surface 的 §107/§108-class bugs**——尤其关注：
1. 其他 `ActionType` (TYPE / SCROLL / SELECT_OPTION / HOVER / KEY_PRESS / GOTO / NEW_TAB / TAB_FOCUS / FINISH) 是否有同样的 "low-level dispatch when locator should work" 模式
2. AXTree extraction (`processors.py`) 在哪里决定 `element_id → DOM element` 映射，是否有 stale-cache / role="link" 误纳入 / 父子节点 bbox 错位等结构性问题

## 仓库

`/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents`，必须用 `.venv/bin/python3`。**不要 commit**。

## Scope

### 重点文件（必读）

```
external/visualwebarena/browser_env/
├── processors.py        ← AXTree extraction, get_element_center, observation 构造
├── actions.py           ← ActionTypes 分支 + 每个 type 的 dispatch 实现
├── envs.py              ← StepInfo / observation 流程
├── auto_login.py        ← 登录状态管理
└── helper_functions.py  ← 杂项 (URL parsing, element resolution)

external/webarena/browser_env/    (同款 上游 fork)
├── processors.py
├── actions.py
└── ...
```

### 待审计内容

#### Audit A — Action dispatch path matrix

针对每个 `ActionType` 列举两件事：

```
ActionType   | dispatch 方式           | 期望 vs 实际可能问题
─────────────┼─────────────────────────┼────────────────────
CLICK        | page.mouse.click(x,y)   | §106 已知; bbox center 命中父元素
TYPE         | ?                       | 是 page.keyboard.type(text) 还是 locator.fill()?
SCROLL       | ?                       | scroll-into-view 还是 mouse.wheel?
SELECT_OPTION| ?                       | 哪个 API? value 验证?
HOVER        | ?                       | ?
KEY_PRESS    | ?                       | ?
GOTO         | page.goto(url)          | 应该 robust
NEW_TAB      | ?                       | 在 headless 下 popup 行为?
TAB_FOCUS    | ?                       | ?
FINISH       | (本地处理)              | 不发到 page
```

对每个 ActionType 给 **(file:lineno, dispatch API call, suspicion category)**。

Suspicion 类别：
- `LOW`: 用 locator-based API，应该 robust
- `MEDIUM`: 用 low-level API 但只在简单 case 触发
- `HIGH`: 用 low-level API 且涉及 element identity（element_id / bbox / coord），同 §106 模式
- `INFO`: 设计如此，没问题

#### Audit B — AXTree extraction & element_id mapping

读 `browser_env/processors.py` 的 AXTree extraction 流程，回答：

1. **哪些 AXTree node 被纳入 `obs_nodes_info`？**
   - 哪些 `role` 被 include?（link / button / heading / textbox / searchbox / ...）
   - 哪些 `role` 被 filter out?
   - role 决定的 include 规则是什么文件?
2. **`element_id` 如何分配？**
   - 是顺序自增？基于 backendNodeId？
   - 跨 step 是否稳定？
3. **`union_bound` 来自哪里？**
   - `getBoundingClientRect()` 直接返回？还是 wrap 了？
   - 是否处理 inline multi-line / scroll position?
4. **是否有 stale-cache risk？**
   - JS 后渲染的 element 在 step N 时纳入 AXTree，到 step N+1 还有效吗?
   - `page.click` 失败后下一帧 obs 重新拍，element_id 是否复用旧 mapping?
5. **role="link" 的非 `<a href>` 元素怎么处理？**
   - `<button role="link">` / `<span onclick>` / `<h2>` 是否进 obs?
   - 如果进，`get_element_center` 后 `page.mouse.click(center)` 命中的是 button 而非 anchor，行为如何?

输出 audit doc，每条问题给 **(file:lineno, code excerpt, finding, paper-relevance rating)**。

#### Audit C — Cross-fork comparison

BrowserGym fork 用 bid 注入 + `page.locator("[bid=N]").click()` 解决 §106。我们 fork 仍用 `page.mouse.click`。审计：

1. BrowserGym 在 `actions.py` 里如何处理同样的 ActionType？link 到对应代码（如果你能找到 BrowserGym 源）。
2. 我们 fork 对每个 ActionType 跟 BrowserGym 比，diff 是什么?
3. 哪些 BrowserGym fix 我们可以 cherry-pick (drop-in patch)?

如果 BrowserGym 源不在仓库，report "BrowserGym source not available locally; recommendation: clone for comparison"——不要 mock 数据。

## 输出

`docs/analysis/cross_sites/tier1_dispatch_audit.md` (~1500-3000 words)，结构：

```markdown
# Tier 1 — WA/VWA Dispatch + AXTree Extraction Audit

## Executive summary
- N candidate bugs identified (HIGH suspicion)
- M structural concerns in AXTree extraction
- Recommended fix priority list

## Section A — ActionType dispatch matrix
Table of all ActionType × dispatch API + suspicion + (file:lineno) + reasoning

## Section B — AXTree extraction findings
- Q1 role inclusion rules: (findings)
- Q2 element_id assignment: (findings)
- Q3 union_bound source: (findings)
- Q4 stale-cache risk assessment: (findings)
- Q5 role="link" non-<a> handling: (findings)

## Section C — Cross-fork comparison (best effort)
Items that BrowserGym fixes that we don't

## Section D — Recommended fix priority
Top 5 bugs (with §106 ranking) + each: (description, fix sketch ~5-10 line, blast radius估)

## Section E — Open questions for follow-up
Things that need empirical Playwright probe to confirm
```

同时输出 `docs/analysis/cross_sites/tier1_dispatch_audit.json`：

```json
{
  "audit_date": "2026-04-30",
  "candidate_bugs": [
    {
      "id": "candidate_1",
      "action_type": "TYPE",
      "file": "external/visualwebarena/browser_env/actions.py",
      "lineno": 1234,
      "suspicion": "HIGH",
      "description": "...",
      "fix_sketch": "...",
      "blast_radius_estimate": "needs probe"
    }
  ],
  "axtree_findings": [
    {
      "question": "Q5 role='link' non-<a>",
      "answer": "...",
      "code_ref": "processors.py:567",
      "concern_level": "MEDIUM"
    }
  ]
}
```

## 不要做的事

- 不要 commit
- 不要 modify 任何源码（这是 audit not fix）
- 不要重新 verify §106（已有 probe 数据，引用即可）
- 不要 mock 数据—— BrowserGym 源若不在 repo，明示 "source not available"
- 不要写 fix patch（Tier 1 只 audit，fix 是另一阶段）

## 验证

完成后跑：
```bash
ls -la docs/analysis/cross_sites/tier1_dispatch_audit.{md,json}
.venv/bin/python3 -c "import json; d = json.load(open('docs/analysis/cross_sites/tier1_dispatch_audit.json')); print(f'{len(d[\"candidate_bugs\"])} candidate bugs, {len(d[\"axtree_findings\"])} axtree findings')"
```

期望 candidate_bugs ≥ 3 (除非整个 dispatch 出奇地干净，那也是 finding——明确说 "no high-suspicion candidates beyond §106")。

## token 预算

~50K (read 5-8 source file × 几百-几千 line + write audit doc + json)
