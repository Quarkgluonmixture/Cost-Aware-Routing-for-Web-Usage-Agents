"""The talk's `--check` (presentation-playbook.md Appendix E, step 3, plus the §3 greps).

Four things, red = do not project:
  1. word budget — every non-backup slide ≤ 50 words, the opening slide ≤ 15.
     "Word" has ONE definition, the playbook's regex below; do not count by hand.
  2. every number on a slide is traceable to SHOWCASE_PREP.md (the numbers' authority).
  3. jargon grep — internal words never reach the screen or the spoken lines
     (ROADMAP.md §4 word list); poster sentences are whitelisted verbatim.
  4. meta grep — no listener names, no "how to talk to them" layer on the slides.
Also prints the script's English word count — a reminder only; the timing that counts is a timed read-aloud (playbook v3 §4).

Usage:  .venv/bin/python3 deliverables/showcase/talk/check_talk.py
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECK = HERE / "index.html"
SCRIPT = HERE / "rehearsal-script.md"
PREP = HERE.parent / "SHOWCASE_PREP.md"

WORD = re.compile(r"[A-Za-z0-9][\w',.%/§-]*")          # the playbook's one definition of a word
NUMBER = re.compile(r"(?<![\w.@])\d[\d,.]*%?")           # a number on a slide; not digits inside an e-mail / identifier

# ROADMAP.md §4 末 — internal words that must not reach an outsider. Whole-word, case-insensitive.
JARGON = ["oracle", "router", "routers", "mode", "modes", "pp", "AXTree", "DOM", "cell", "cells",
          "condition", "conditions", "replicate", "replicates", "episode", "episodes", "SR",
          "P-text", "P-SoM", "P-prompt", "canonical", "phantom", "baseline"]
# Poster sentences and words the sheet itself prints, allowed verbatim (the audience has seen them).
WHITELIST = ["Learned routers buy success only by spending more",
             "hindsight oracle", "routing"]
META = ["Zekun", "Maria", "Emre", "Adriano", "Ask, do not", "audience", "listener"]


def slide_texts(deck: str) -> list[tuple[int, bool, str]]:
    body = deck.split("<body", 1)[1]
    body = re.sub(r"<script.*?</script>", "", body, flags=re.S)
    out = []
    for n, (tag, inner) in enumerate(re.findall(r"(<section[^>]*>)(.*?)</section>", body, flags=re.S), 1):
        inner = re.sub(r"<!--.*?-->", " ", inner, flags=re.S)
        text = html.unescape(re.sub(r"<[^>]+>", " ", inner))
        out.append((n, "backup" in tag, " ".join(text.split())))
    return out


def scrub(text: str) -> str:
    for w in WHITELIST:
        text = text.replace(w, " ")
    return text


def jargon_hits(text: str) -> list[str]:
    hits = []
    for j in JARGON:
        if re.search(rf"(?<![\w-]){re.escape(j)}(?![\w-])", scrub(text), flags=re.I if j.islower() else 0):
            hits.append(j)
    return hits


def main() -> int:
    fail = 0
    deck = DECK.read_text(encoding="utf-8")
    prep = PREP.read_text(encoding="utf-8")
    slides = slide_texts(deck)

    print("── 1/4 word budget: non-backup slides ≤ 50, opening ≤ 15")
    for n, backup, text in slides:
        words = len(WORD.findall(text))
        cap = 15 if n == 1 else 50
        ok = backup or words <= cap
        fail += not ok
        print(f"  {'✓' if ok else '✗'} slide {n}: {words} words{' (backup, exempt)' if backup else f' / {cap}'}")

    print("── 2/4 every number on a slide is in SHOWCASE_PREP.md")
    for n, backup, text in slides:
        if backup:
            continue
        for num in sorted(set(NUMBER.findall(text))):
            if num in ("6", "8", "100"):          # counts of views / settings / 'in 100' — words, not results
                continue
            ok = num in prep
            fail += not ok
            print(f"  {'✓' if ok else '✗'} slide {n}: {num}")

    print("── 3/4 jargon grep (ROADMAP §4 word list) — deck, and the script's English lines")
    for n, backup, text in slides:
        hits = jargon_hits(text)
        fail += bool(hits)
        print(f"  {'✓' if not hits else '✗'} slide {n}: {', '.join(hits) if hits else 'clean'}")
    script = SCRIPT.read_text(encoding="utf-8")
    acts = script.split("# 被问到时")[0]
    lines = [l[2:] for l in acts.splitlines() if l.startswith("> ")]
    hits = sorted({h for l in lines for h in jargon_hits(l)})
    fail += bool(hits)
    print(f"  {'✓' if not hits else '✗'} script (acts): {', '.join(hits) if hits else 'clean'}")

    print("── 4/4 meta grep — listener names / 'how to say it' layer on slides")
    hits = [m for m in META if re.search(re.escape(m), " ".join(t for _, _, t in slides), flags=re.I)]
    fail += bool(hits)
    print(f"  {'✓' if not hits else '✗'} deck: {', '.join(hits) if hits else 'clean'}")

    n_words = sum(len(WORD.findall(l)) for l in lines)
    print(f"── script: {n_words} English words in the acts (an estimate only; time it aloud — playbook v3 §4)")

    print("CHECK PASS" if not fail else "CHECK FAIL")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
