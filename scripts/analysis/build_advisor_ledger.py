#!/usr/bin/env python3
"""Build the advisor-review ledger — every conclusion, with its coverage proof.

The 2026-08-05 REALM submission needs one working session with the advisor to
decide what goes in 8 pages. The material is spread across a 13-file conclusion
layer (~9.4k lines), a 2057-record ledger, 41 diag digests and three canvases,
and the standing requirement for that session is "no omissions, no errors".

Hand-assembling such a document is exactly the failure mode this repo keeps
hitting: a human-curated subset silently becomes the universe. So the ledger is
GENERATED — every section of the conclusion layer is enumerated mechanically, and
the script reports its own coverage. If a section is missing from the output, the
count at the top says so.

Six views, organised by what the meeting actually does rather than by document
structure:

  decide      open questions, each with options and consequences  ← the agenda
  decided     adjudications already made (so they are not re-litigated)
  evidence    measured results, by theme
  coverage    what was checked, what was not, what is unknown
  retracted   claims that died (so nothing dead gets cited)
  running     work in flight, whose state must be queried not trusted

`decide` is the only hand-authored view — it is judgement, not extraction, and is
marked as such in the output so the advisor can see which part depends on my
reading. Everything else is derived.

Usage:
  .venv/bin/python3 scripts/analysis/build_advisor_ledger.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONC = REPO / "docs/reference/known/conclusions"
LEDGER = REPO / "docs/reference/known/ledger.jsonl"
NAV_FILES = {"INDEX.md", "PROGRESS.md"}  # navigation, not content

SEC_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$", re.MULTILINE)
# Excerpts are read in a dense table, where raw Markdown emphasis is noise rather
# than signal — `**x**` and backticks carry no meaning once the text is no longer
# a Markdown document. Stripped for legibility; nothing else about the text changes.
MD_NOISE = [(re.compile(r"\*\*(.+?)\*\*"), r"\1"),
            (re.compile(r"`([^`]+)`"), r"\1"),
            (re.compile(r"^[-*]\s+", re.MULTILINE), ""),
            (re.compile(r"\|"), " ")]
REF_RE = re.compile(r"§\s?(\d+(?:\.\d+)*[a-z]?)")
DEAD_RE = re.compile(r"RETRACT|作废|已死|superseded|SUPERSEDED|不再成立|已收回")
OPEN_RE = re.compile(r"待\s?(user|advisor|裁定|定|决策)|待议|未定|裁定待|TBD|待核")

BATCH = {
    "adjudicated_A1.md": ("裁定", "§5–§119 工程建设 + framing 成形"),
    "adjudicated_A2.md": ("裁定", "§121–§164 pre-fire 审计密集"),
    "adjudicated_A3.md": ("裁定", "§165–§240 fire 前冲刺 + Fire-1~6"),
    "adjudicated_A4.md": ("裁定", "§241–§397 Protocol Reset + 治理 + 投稿"),
    "measured_D1.md": ("实测", "§1–§128.5"),
    "measured_D2.md": ("实测", "§128.6–§207.6"),
    "measured_D3.md": ("实测", "§207.4–§311"),
    "measured_D4.md": ("实测", "§312.2–§397.10 + 附录 A(§398) / B(diag)"),
    "measured_qualitative.md": ("实测", "全程 · 无数字实测"),
    "data_inventory.md": ("数据", "全程 · 数据资产"),
    "retracted.md": ("作废", "全程 · 作废与待验 + M1–M11 错误模式"),
}


def _clean(body: str) -> str:
    """Collapse a section body to one readable line for the table."""
    txt = body.strip()
    for pat, rep in MD_NOISE:
        txt = pat.sub(rep, txt)
    return re.sub(r"\s+", " ", txt).strip()


def sections() -> list[dict]:
    """Every heading in the conclusion layer, with its body."""
    out: list[dict] = []
    for path in sorted(CONC.glob("*.md")):
        if path.name in NAV_FILES:
            continue
        txt = path.read_text(encoding="utf-8")
        marks = list(SEC_RE.finditer(txt))
        kind, era = BATCH.get(path.name, ("其他", ""))
        for i, m in enumerate(marks):
            body = txt[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(txt)]
            refs = sorted(set(REF_RE.findall(m.group(2) + body)),
                          key=lambda s: [int(p) for p in re.findall(r"\d+", s)] or [0])
            out.append({
                "file": path.name, "kind": kind, "era": era,
                "level": len(m.group(1)), "title": m.group(2).strip(),
                "refs": refs[:12],
                "body_chars": len(body.strip()),
                "excerpt": _clean(body)[:400],
                "is_dead": bool(DEAD_RE.search(body)),
                "has_open_question": bool(OPEN_RE.search(body)),
            })
    return out


def ledger_rows() -> list[dict]:
    rows = []
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def ledger_by_section(rows: list[dict]) -> dict[str, list[dict]]:
    idx: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        for ref in REF_RE.findall(str(r.get("source_section") or "")):
            idx[ref].append(r)
    return idx


# ── the one hand-authored view ───────────────────────────────────────────────
# Judgement, not extraction. Kept in the producer (not a side file) so it is
# versioned with the derived material and cannot drift out of sync unnoticed.
DECISIONS = [
    {"id": "D1", "title": "合并后的论文骨架 —— 8 页装什么",
     "why": "REBUILD_PLAN 待办 1。它决定下游三项补到什么精度；不定则待办 3/5 是盲做。",
     "options": ["先定骨架再补细节（原计划）", "边写边定", "先写 §4 结果再倒推结构"],
     "recommend": "先定骨架。今天 07-29，deadline 08-05。",
     "cost_of_not_deciding": "下游三项全部盲做，最后 2 天堆在一起。",
     "evidence": ["REBUILD_PLAN Phase 3", "§398.8 两篇合一的焊接点"]},
    {"id": "D2", "title": "9 条「旧稿写过、拆两篇时丢了」进不进",
     "why": "工具捞的，不靠我判断。合并回一篇时最容易永久丢失。",
     "options": ["逐条过（约 15 分钟）", "只过我标必进的 4 条", "全部不进"],
     "recommend": "逐条过。其中 §109.17（novelty 防御）和 §338（B2 配置辩护）不进就是留攻击面。",
     "cost_of_not_deciding": "合并时静默丢失，且没人会再发现。",
     "evidence": ["find_unlanded_allDrafts_2026-07-29.txt", "§401 差集分析"]},
    {"id": "D3", "title": "噪声地板写不写进主文、怎么写",
     "why": "同条件重跑地板 4.9–7.6pp > H3 两轴的 1.35/2.09pp。写=承认正面结果站不住；不写=隐瞒。",
     "options": ["主文明写 + 据此收窄 claim", "只进 limitations", "进 appendix"],
     "recommend": "必须写。位置和措辞是取舍，建议主文一句 + limitations 展开。",
     "cost_of_not_deciding": "审稿人自己算出来 = 致命；上一稿已有过一次假陈述被抓的记录（§397.10）。",
     "evidence": ["§398.2 Phase 0b", "measured_D4 附录 A/Z1"]},
    {"id": "D4", "title": "canvas 里学长 5/5 提的 3 件，至今未 lock",
     "why": "2026-05-05 提出，到今天近三个月。其中「env-side pilot 进 paper-1 还是 paper-2」在 8 页 scope 下需重裁。",
     "options": ["会上逐条 lock", "全部推 paper-2", "只 lock hook 那条"],
     "recommend": "会上 lock。这是学长自己提的，不该由我们单方面决定。",
     "cost_of_not_deciding": "第四个月继续悬着。",
     "evidence": ["dual_track_taxonomy.canvas 末节", "conclusions/INDEX §1 裁定"]},
    {"id": "D5", "title": "mechanistic sweep 跑完了怎么用 + WA 进不进 Phase 1",
     "why": "sweep 08-01 完成，论文 08-05 交；mechanism 线 2026-05-14 已暂搁。user 已定「到时候看」，与 WA 一并对。",
     "options": ["跑完存档不用", "进 appendix", "重启 mechanism 线"],
     "recommend": "存档不用 —— 8 页装不下，且 scope 已收窄。",
     "cost_of_not_deciding": "最后 4 天变成干扰。",
     "evidence": ["_status/tasks/task_analysis_gating.md", "§402.7"]},
    {"id": "D6", "title": "codex 指控「router 实验用错了学习器」",
     "why": "codex 说该用 per-cell LR head + TF-IDF/MI（§216.1 规格），我用了 pooled head + 20 raw features。",
     "options": ["接受并重跑", "维持 + 在文中说明理由", "作为 appendix sensitivity"],
     "recommend": "维持。池化实验本质上必须 pooled head，否则不是池化；TF-IDF 省略有 §367 先例。但这条该你们裁，不该我自裁。",
     "cost_of_not_deciding": "审稿人可能提同样的问题。",
     "evidence": ["§401.4", "§216.1 CV 协议", "§367 TF-IDF 无增益"]},
    {"id": "D7", "title": "sidebar 泄漏的披露口径",
     "why": "user 已定归 benchmark bug paper、scored universe 保持 203。主 paper 那一句怎么写待定。",
     "options": ["一句话 + 指针", "limitations 一段", "完全不提（bug paper 单独讲）"],
     "recommend": "一句话 + 指针。实质只影响 B2·DOM 一格（8→5）。",
     "cost_of_not_deciding": "bug paper 和主 paper 都不提 = 漏报。",
     "evidence": ["reddit_sidebar_leakage_audit.md", "§402.6/§402.7"]},
]

RUNNING = [
    {"what": "mechanistic canonical sweep（24 cell）", "where": "DGX",
     "how_to_check": "cat logs/mechanistic_canonical/.sweep.pid && ps -p $(cat ...)",
     "note": "driver pid 38603，deadline 08-01。查驱动 pid 不查 worker pid（§397.10(4) 教训）"},
    {"what": "WA reddit 全量 6 模式", "where": "A100",
     "how_to_check": "docs/checkpoints/_status/tasks/*.md frontmatter",
     "note": "现为 future work，是否进 Phase 1 待对"},
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-out", type=Path,
                    default=REPO / "docs/checkpoints/deliverables/advisor_ledger.json")
    ap.add_argument("--html-out", type=Path,
                    default=REPO / "docs/checkpoints/deliverables/advisor_ledger.html")
    ap.add_argument("--template", type=Path,
                    default=REPO / "scripts/analysis/templates/advisor_ledger.html.tmpl")
    a = ap.parse_args()

    secs = sections()
    rows = ledger_rows()
    by_sec = ledger_by_section(rows)

    for s in secs:
        hits = [r for ref in s["refs"] for r in by_sec.get(ref, [])]
        s["n_ledger_records"] = len(hits)
        s["ledger_types"] = dict(Counter(r["type"] for r in hits))

    files = sorted({s["file"] for s in secs})
    coverage = {
        "conclusion_files_scanned": files,
        "n_conclusion_files": len(files),
        "n_sections_total": len(secs),
        "n_sections_l2": sum(1 for s in secs if s["level"] == 2),
        "n_sections_l3": sum(1 for s in secs if s["level"] == 3),
        "n_sections_dead": sum(1 for s in secs if s["is_dead"]),
        "n_sections_with_open_question": sum(1 for s in secs if s["has_open_question"]),
        "n_ledger_records": len(rows),
        "ledger_types": dict(Counter(r["type"] for r in rows)),
        "nav_files_excluded": sorted(NAV_FILES),
    }

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "advisor review session — 2026-08-05 REALM submission",
        "coverage": coverage,
        "decisions": DECISIONS,
        "running": RUNNING,
        "sections": secs,
        "note": ("`decisions` is hand-authored judgement; every other view is "
                 "extracted mechanically from the conclusion layer and ledger."),
    }
    a.json_out.parent.mkdir(parents=True, exist_ok=True)
    a.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                          encoding="utf-8")

    # Self-contained local page. Compact key names keep the inlined payload near
    # 225 KB; `</` is escaped so a string containing `</script>` cannot close the
    # data block early.
    compact = {
        "gen": payload["generated_utc"], "cov": coverage,
        "dec": DECISIONS, "run": RUNNING,
        "sec": [{"f": s["file"].replace("_", " ").replace(".md", ""), "k": s["kind"],
                 "l": s["level"], "t": s["title"], "r": s["refs"][:6],
                 "x": s["excerpt"][:240], "d": int(s["is_dead"]),
                 "o": int(s["has_open_question"]), "n": s["n_ledger_records"]}
                for s in secs],
    }
    blob = json.dumps(compact, ensure_ascii=False,
                      separators=(",", ":")).replace("</", "<\\/")
    tmpl = a.template.read_text(encoding="utf-8")
    if "__DATA__" not in tmpl:
        raise SystemExit(f"template lacks __DATA__ placeholder: {a.template}")
    a.html_out.write_text(tmpl.replace("__DATA__", blob), encoding="utf-8")

    print(f"conclusion files scanned : {len(files)}")
    print(f"sections extracted       : {len(secs)}  "
          f"(L2 {coverage['n_sections_l2']} · L3 {coverage['n_sections_l3']})")
    print(f"  of which marked dead   : {coverage['n_sections_dead']}")
    print(f"  with an open question  : {coverage['n_sections_with_open_question']}")
    print(f"ledger records           : {len(rows)}  {coverage['ledger_types']}")
    print(f"hand-authored decisions  : {len(DECISIONS)}")
    print(f"\nwrote {a.json_out}")
    print(f"wrote {a.html_out}  ({a.html_out.stat().st_size // 1024} KB, self-contained)")
    print(f"\n本地打开: xdg-open {a.html_out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
