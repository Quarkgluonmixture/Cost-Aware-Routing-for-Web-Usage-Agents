# Process rule: 阶段性成果 → auto-trigger 写入实验笔记

User set 2026-05-12 in death-march mode after 17-commit day.

## Why

Forward-progress mode without chronicle write-in:
- Context compress / session 切换后 timeline narrative 丢失
- 跨 session 工作 lacks 时间锚
- Reviewer rebuttal 阶段不能 trace 何时每个 evidence layer landed
- Memory persists facts; 笔记 persists timeline. 两者互补.

## The 8 triggers

`docs/checkpoints/实验笔记.md` (append-only chronicle) **必须 append** 当以下任一事件发生:

1. User signals compress: "compress 对话", "/compact", "session 切换", "context 太长"
2. Multi-commit burst (≥5 paper-related commits since last 笔记 append)
3. 新 finding land (mechanism / experiment result)
4. /stress invocation completes (attacks identified or defused)
5. Major experiment lands (Myriad cell finish, cross-site evidence closure)
6. New infra/skill 上线 (`/stress` was 2026-05-12 example)
7. Paper section v1 / 重大 prose update
8. Day-end signal: "今天 land", "今晚收尾", "tomorrow morning"

## Format

```markdown
## N. 标题 (YYYY-MM-DD) [tag1][tag2] #tag1 #tag2

[concise body per 笔记 写作规范]
```

Multi-faceted bursts → single § with sub-headers (§127.1, §127.2, ...).

Tag categorization with line-count caps:
- `[finding]` — 1 line + `→ 见 {digest path}` pointer
- `[infra]` — 5-10 lines
- `[design]` — 8-15 lines
- `[literature]` — 8-12 lines
- `[bug]` — 3-5 lines

Cross-link:
- git commits via short hash (`commit be07296`)
- result files via `→ 见 docs/checkpoints/mechanism/results/*.md`

## Bypass

Only when user explicitly says "skip 笔记" / "no chronicle". Default always append.

## Canonical 模板

`docs/checkpoints/实验笔记.md` §127 (2026-05-12 死磕日) is the canonical multi-faceted example:
- 17 commits / 4 Myriad cells / 2 DGX experiments / 1 new skill / 2 stress attacks resolved
- → 9 sub-sections under one § (§127.1 - §127.9)
- Total ~130 lines for a full death-march day

## Locations

- Project rule: `.claude/CLAUDE.md` "阶段性成果" section (gitignored)
- This file: `docs/checkpoints/process/chronicle_on_milestone_rule.md` (git-tracked replica)
- Memory feedback: `~/.claude/projects/.../memory/feedback_chronicle_on_milestone.md` (cross-session)

## Versioning

v1 (2026-05-12) — initial rule after 17-commit day surfaced timeline-loss risk during context compression discussion.
