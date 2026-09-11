---
type: task
status: active
priority: P0
horizon: now
order: 0
blocker: ""
eta: "talk v0 (deck+script+runbook+fallback) done 09-11 · author rewrites script + §2/§5 rewrite 09-12–13 · template skin ≤09-14 · rehearsal 09-14–15 · event 09-16 (votes 13:15–14:35, talk 14:45)"
detail: deliverables/showcase
created: 2026-09-02
updated: 2026-09-11
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
