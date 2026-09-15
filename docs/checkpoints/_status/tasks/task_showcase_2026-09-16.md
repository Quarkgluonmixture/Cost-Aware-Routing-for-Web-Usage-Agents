---
type: task
status: active
priority: P0
horizon: now
order: 0
blocker: ""
eta: "09-15 深夜 D25：opening → agents（三块真实抓取，待换 user 自截的 Claude Code / Codex CLI 终端图）→ demo（演讲版减负、sunset 高亮）→ prize（27 → 43 两根条 + −20%）→ question…，11 页 · 剩 收截图换 agents 页 / 掐表（约 988 词）/ DGX tmux / 发最新 PDF · event 09-16（user 09:50 到场帮忙布展 · 投票 10:00–14:35 · 下午学生演讲第二个，约 14:55）"
detail: deliverables/showcase
created: 2026-09-02
updated: 2026-09-15
---

# Holistic AI × UCL CDI showcase (16 Sep 2026) — poster + laptop demo

**Poster**: `deliverables/showcase/poster_jiaming_wei.pdf` = **v9.10** (2026-09-04), file `poster_v9_jiaming_wei.pdf`.
Template three-column skeleton · loop diagram on top with real screenshots ·
number strip + three comparison definitions · THE CATCH / why / Fig 3 / takeaway
(left) · Fig 2 / verdicts / laptop bridge (right). No jargon. Every number parsed
and scoped — `poster_content.md` is the source of truth (read its v8.3/v8.2/v8 headers).
Board-distance pass (v8.3): 23pt body, 17pt captions, 15.5pt diagram labels,
17.5pt section headers, 42pt headline numbers, 24/18pt byline and affiliation;
the five loop cards set their own 19–29pt against the air each one had. Room
came from Fig 2 (5.45in) and Fig 3 (3.85in), not from any claim. Verified slack:
1.7mm left / 0.1mm right — on a `line_count` whose width metric is now
calibrated to LibreOffice (`WIDTH_CAL`), so earlier slack figures read high.

**Send**: Slack DM Zekun by **09-04** with the PDF + `SHOWCASE_PREP.md §1`
(oral-slot pitch + table/power request for the demo). Footer email is
`jiaming.wei.25@ucl.ac.uk` (v2–v8 carried a wrong `.22`).

**Demo** (v2, 2026-09-10, 笔记 §506): three recorded tasks (130 look · 76 read · 17 both)
with cost / time / CO₂e≈ meters under each lane and the learned choice's fold-held-out
pick (arrow + ring); tab 4 "✎ try your own" runs a typed task live in all three views
(DGX site + server, unscored, visitor judges ✓/✗). **Two ways to open**: live =
`http://localhost:8799/` through `ssh -N -L 8799:localhost:8799 spark` on quark; offline =
double-click `demo_portable.html`. Never through VS Code Live Server. Runbook:
`demo/README.md` → *Live*.
**Talk**: 10-minute slot confirmed (user 09-10), slides on a template still to come,
demo woven in.

**Chronicle**: 笔记 §495 (v2) · §498 (v4) · §499.1–.12 (v5→v9.10). Ledger under those §.
**真实尺寸校验**: `print_test_tiles.py` 把任意区域按 1:1 切成 A4, 打印须选「实际大小」。

**Roadmap (2026-09-11)**: `deliverables/showcase/ROADMAP.md` — six phases with done-conditions,
decisions D1–D10 (talk demo = replay of task 130 only; pptx by default; live tab only at the
board), the 7-slide skeleton mapped to the six printed panels, and the day-of fallback ladder.
⚠️ `SHOWCASE_PREP.md §2/§3/§5` are v8-based and stale for the printed v9.10 — rewritten in Phase 2.

**Talk v0 (2026-09-11, 笔记 §507.5)**: five artefacts per `presentation-playbook.md` in `deliverables/showcase/talk/` —
`index.html` (8 slides, slide 2 = demo iframe, `check_talk.py` PASS) · `rehearsal-script.md` (735 words, six locked sentences) ·
`RUNBOOK.md` · `fallback.html` (6 shots) · `talk_130.webm` · `talk.pdf`. Left: author rewrites the script aloud; template skin; rehearsal §6.

**09-15 收尾（笔记 §511–§517）**：演讲电脑 = MacBook + Chrome，quark 全天在展板；Zekun：接自己电脑、提前发片子。
片子按 playbook v3 与学长意见重排：question 页一个问题 + 四路标 · behaviour / failure · hindsight = Claude Code 前端自检场景 + 三个数字（`talk/hindsight_efficiency.py`）· learned 大数字 0 of 8 · why = scaling law · not-yet · close；开场加导师。
板前走读已按印出来的 v9.10 重写（`SHOWCASE_PREP.md §2`，含「海报说过头处」口径表）。当天完整手册 `deliverables/showcase/day-of.html`（已发布为 artifact）。

**09-15 晚（ROADMAP D19）**：user 09:50 到场，先帮忙布置展板；下午学生演讲第二个讲（约 14:55，以主持人为准）。手册、RUNBOOK、ROADMAP Phase 6 已同步。

**09-15 晚收尾（ROADMAP D20–D23，笔记 §518–§526）**：why 换真实数据图 · learned 散点图简化 + 改说 perfect router · 开场 agents 用真实抓取（放进 Claude Code / Codex 窗口框）+ prize 数字卡 · 删书桌页 · Google 日历已建。user 仍不太满意，开新 session；交接见 next_steps §0。

**09-15 晚 D24（笔记 §527）**：user 说清结构 —— 标题 → 用 Claude Code / Codex 讲 web agent + 三种看法 → demo，且一开始就报数字。落地：opening → prize（三卡，删用时卡）→ agents（三块真实抓取按 demo 泳道顺序与颜色，BOTH = 截图 + 我们 SoM 代码画的 25 个编号框，`real_capture.py` 同一会话）→ demo（演讲版隐藏 demo 自己的 header、task 句放大、*sunset* 高亮）。check_talk PASS、PDF 重导、离线包 17.5 MB、手册重发。待 user 过目与掐表。
