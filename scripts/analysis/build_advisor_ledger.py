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
    {"id": "D1", "title": "8 页里放什么、砍什么",
     "why": "投的是 8 页的会议论文，而手上的材料远超 8 页。先定好放什么再动笔，"
            "还是边写边砍？这一条不定，下面几件都只能瞎补。",
     "options": ["先把结构定死，再按结构补细节",
                 "边写边定，写到哪算哪",
                 "先把结果那一章写出来，再倒推前后文该写什么"],
     "recommend": "先定结构。今天 7 月 29，8 月 5 号截稿，只剩一周。",
     "cost_of_not_deciding": "补细节的三件事全是瞎补，最后两天全堆在一起。",
     "evidence": ["REBUILD_PLAN 第 3 阶段", "§398.8 两篇合一的理由"]},

    {"id": "D2", "title": "以前写过、拆成两篇时弄丢的 9 个内容，捡不捡回来",
     "why": "论文原本是两篇，后来决定合成一篇。工具比对发现有 9 处内容"
            "在更早的稿子里写过，拆成两篇的时候丢了。合并回一篇是把它们捡回来的"
            "最后机会——再不捡就永久没了，而且没人会再发现。",
     "options": ["9 条逐个过，大概 15 分钟",
                 "只看我标了「必须捡」的那 4 条",
                 "都不捡，就按现在两篇的内容合"],
     "recommend": "逐个过。其中两条是防守用的：一条挡「你这个业界早就做了」，"
                  "一条挡「B2 模型才 1% 是不是你配错了」。不捡就是留着挨打。",
     "cost_of_not_deciding": "合并的时候静默丢失，之后不会再有人发现。",
     "evidence": ["find_unlanded 全稿扫描结果", "§401 差集分析"]},

    {"id": "D3", "title": "「同样条件重跑两次结果就不一样」这件事写不写进正文",
     "why": "同一个模型、同一批题目、同样设置，重跑一遍，成功率会差 4.9 到 7.6 个百分点。"
            "而我们论文里那个正面发现只有 1.35 和 2.09 个百分点——比重跑的波动还小。"
            "写进去等于自己承认那个发现站不住；不写就是隐瞒。",
     "options": ["正文写明，并据此把结论说得更保守",
                 "只在「局限」那一节提",
                 "放到附录"],
     "recommend": "必须写，不写是硬伤。写在哪、怎么措辞是可以商量的——"
                  "我倾向正文一句带过，局限那节展开。",
     "cost_of_not_deciding": "审稿人自己就能算出来，那时候是致命的。"
                             "上一版稿子已经因为一句不实陈述被抓过一次。",
     "evidence": ["§398.2 重跑地板实测", "结论层 measured_D4 附录 A"]},

    {"id": "D4", "title": "学长 5 月 5 号提的三个问题，到现在还没定",
     "why": "那三个问题记在一张 canvas 图里，从 5 月 5 号挂到今天，快三个月。"
            "其中一个是「要不要做一个服务器端的小实验放进这篇论文」——"
            "现在只有 8 页，这个问题得重新答一遍。",
     "options": ["会上三条一次性定掉",
                 "全部推到下一篇论文",
                 "只定其中关于论文卖点那一条"],
     "recommend": "会上定掉。这是学长自己提的问题，不该我们单方面替他决定。",
     "cost_of_not_deciding": "挂到第四个月。",
     "evidence": ["dual_track_taxonomy.canvas 最后一节", "结论层 INDEX §1"]},

    {"id": "D5", "title": "还在跑的那个实验，跑完了用不用",
     "why": "一个 24 组的机理实验在服务器上跑着，8 月 1 号完成，论文 8 月 5 号交。"
            "但这条研究线 5 月中就决定暂时搁置了。跑完的数据是用、是存档、还是重启这条线？"
            "另外 WebArena 那批数据要不要算进来，一起定。",
     "options": ["跑完存档，这篇不用",
                 "放进附录",
                 "重新把机理这条线捡起来"],
     "recommend": "存档不用。8 页装不下，而且论文范围已经收窄了。",
     "cost_of_not_deciding": "最后 4 天它会变成干扰项。",
     "evidence": ["_status/tasks/task_analysis_gating.md", "§402.7"]},

    {"id": "D6", "title": "外部审计说我那个路由实验方法用错了",
     "why": "我做了一个实验：把两个模型的数据合到一起训练一个「该用哪种输入格式」的分类器。"
            "外部 AI 审计说，按原定方案应该给每个模型单独训一个，不该合训。"
            "我的看法是：这个实验的目的就是测「合起来训行不行」，分开训就不是这个实验了。"
            "但这条该你们裁，不该我自己说了算。",
     "options": ["接受意见，按分开训重跑",
                 "维持现在的做法，在文中说明为什么",
                 "两种都做，一个进正文一个进附录"],
     "recommend": "维持。但这是我的一面之词，需要你们判断。",
     "cost_of_not_deciding": "审稿人可能提一模一样的问题。",
     "evidence": ["§401.4", "§216.1 原定方案", "§367 相关先例"]},

    {"id": "D7", "title": "reddit 有 6 道题的成败取决于跑的顺序",
     "why": "reddit 网站在每道题之间不会重置状态，而有几道题是靠「用户订阅了哪些板块」"
            "来判分的。于是前面题目留下的订阅会让后面的题「白捡」一个成功。"
            "已经逐个查过：真正受影响的是 6 个 episode，其中 3 个集中在同一格。"
            "你已经定了这归到另一篇 bug 论文，主论文只需一句话带过——那句话怎么写？",
     "options": ["一句话说明 + 指向 bug 论文",
                 "在局限那节写一段",
                 "主论文完全不提，全放 bug 论文"],
     "recommend": "一句话 + 指针。影响面确实小（只有一格从 8 降到 5）。",
     "cost_of_not_deciding": "两边都不提就是漏报。",
     "evidence": ["reddit_sidebar_leakage_audit.md", "§402.6 / §402.7"]},
]

RUNNING = [
    {"what": "WebArena（WA）reddit — B1 × 6 个输入格式", "where": "A100（不是 DGX）",
     "how_to_check": "ssh condense-a100 'pgrep -af run_experiment; "
                     "ls results/webarena/phase1/'",
     "note": "🔄 2026-07-29 14:11 实测在跑。整条链 queue_chain：dom ✅106 题完成 · "
             "som ✅106 完成 · vision 🔄 92/106 在跑 · 余 P-text / P-prompt / P-SoM。"
             "每个 mode 约 15 小时 ⇒ **预计 07-31～08-01 全部跑完，赶在 08-05 交稿前**。"
             "⚠️ 这批数据是否进论文，需要和学长定。"},
    {"what": "机理实验 sweep（24 组）", "where": "DGX 本机",
     "how_to_check": "ps -p $(cat logs/mechanistic_canonical/.sweep.pid) -o pid,etime,cmd",
     "note": "✅ 2026-07-29 15:10 实测在跑（driver pid 38603，已跑 2 天）。"
             "预计 08-01 完成。这条研究线 5 月中已暂搁 —— 跑完是用、是存档、"
             "还是重启，待和学长定。"
             "⚠️ 查 driver pid，不要查 worker pid（§397.10(4) 的教训）。"},
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
