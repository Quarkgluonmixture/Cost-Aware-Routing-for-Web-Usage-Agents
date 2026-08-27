#!/usr/bin/env python3
"""Fail the build when a dissertation figure carries prose inside the image.

Rationale
---------
House rule R1 in ``_style.py`` says a figure may contain labels but not claims.
That rule decayed silently in the 2026-08 draft because nothing checked it: a
``set_title`` holding a full sentence looks like ordinary plotting code at the
call site, and the damage is only visible once the PDF is placed on a page.

This script reads the *rendered* PDFs rather than the source, so it also catches
prose that arrives through a data file, a caption string assembled at runtime, or
an f-string that grew.

Detection
---------
A text run is reported as prose when any of the following holds:

* it contains a sentence boundary, meaning a full stop followed by a space and a
  letter, or a trailing full stop on a run of five or more tokens (a label does
  not end in a full stop, and counting tokens rather than prose words here is
  what catches a provenance footnote made mostly of file paths);
* it contains an em dash or a double hyphen, which in this project is always a
  sentence connective rather than a range;
* it runs to eight or more words of prose, where tokens carrying a digit, a
  separator dot, an equals sign or an underscore are identifiers rather than
  words, so a row of tick labels is not mistaken for a sentence.

Ranges written with an en dash (``2.0-7.6``) are not flagged: the test looks for
dashes with whitespace around them.

Exit status is 1 when anything is reported, so ``make figures-check`` and the
pre-submission sweep both fail loudly.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
FIG_DIR = REPO / "final_dissertation/tex/figures"

# Figures inherited verbatim from the conference submissions. They were written,
# reviewed and submitted under the venue's own figure conventions, and the
# decision on 2026-08-26 was to carry them across unaltered rather than re-cut
# artwork that has already been through review. They are therefore out of scope
# for R1 and are listed here explicitly rather than skipped by a name pattern,
# so that adding a new figure never lands in the exempt set by accident.
INHERITED = {
    "fig_overview",
    "fig_f1_diamond_schematic",
    "fig_ceilings",
}
# Four further conference figures were considered and are deliberately not in
# the document, so they are not in this set either:
#   fig_fusion_forest   draws the raw measured rerun range as a band and reads
#                       premiums against it, which Appendix B names explicitly as
#                       the comparison the main text must not make ("the main
#                       text uses the threshold, not the band"), and it draws one
#                       cell's band across all eight rows.
#   fig_f2_h1_forest    carries an "INTERIM (PARTIAL_DATA), NOT A VERDICT"
#                       watermark across the whole plot.
#   fig_partition_forest depends on a regex-flagged task partition that this
#                       dissertation never defines, so including it would assert
#                       something the text does not support.
#   fig_sr_by_class     was replaced by fig_f6_sr_by_class.py: its palette was
#                       the reverse of every other figure here, and its
#                       generating script is not in this repository.

# Long-but-legitimate axis labels. Each entry is a full run that the author has
# looked at and accepted as a label rather than a claim. Keep this list short;
# an entry here is a promise that the run names a quantity.
ALLOWED = {
    "oracle success lost if this arm is removed [pp]",
    "tasks whose outcome flips between two runs of one mode [%]",
    "ceiling gain from adding one arm to the best single mode [pp]",
    "success-rate gain of the 6-mode oracle over the best single mode [pp]",
}

SENTENCE_BOUNDARY = re.compile(r"\.\s+[A-Za-z]")
CONNECTIVE_DASH = re.compile(r"(\s[—–]\s|\s--\s|—)")

# A "word" for the purpose of the length rule means a word of prose. Tick
# labels and cell codes are not: a row of eight tick labels reads as one long
# line to pdftotext, and counting `cls-B0`, `n=224` or `P-SoM` as words made the
# check fire on the x axis of the design matrix. Tokens carrying a digit, a
# separator dot, an equals sign or an underscore are identifiers, and are
# excluded before the count.
WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
IDENTIFIER = re.compile(r"[0-9=_·\[\]%$]")


def _prose_words(text: str) -> list[str]:
    """Words of running prose in a text run, identifiers removed."""
    return [w for tok in text.split() if not IDENTIFIER.search(tok)
            for w in WORD.findall(tok)]


def _normalise(run: str) -> str:
    return re.sub(r"\s+", " ", run).strip()


def _is_prose(run: str) -> str | None:
    """Return the rule that the run violates, or None when the run is a label."""
    text = _normalise(run)
    if not text:
        return None
    # Prefix matching, not equality: a long axis label can reach pdftotext with
    # its unit split off into a separate run when the plot is narrow, so
    # "... of one mode" must still match "... of one mode [%]".
    low = text.lower()
    if any(low.startswith(a) or a.startswith(low) for a in ALLOWED):
        return None
    words = _prose_words(text)
    if CONNECTIVE_DASH.search(text):
        return "connective dash"
    if SENTENCE_BOUNDARY.search(text):
        return "sentence boundary"
    # A label does not end in a full stop. Counting ALL tokens here, not just
    # prose words, is deliberate: a provenance footnote is mostly file paths, so
    # excluding identifiers left it with four prose words and it slipped through.
    if text.endswith(".") and len(text.split()) >= 5:
        return "terminal full stop"
    if len(words) >= 8:
        return f"{len(words)} words"
    return None


def _runs(pdf: Path) -> list[str]:
    """Extract text runs from a PDF, one per visual line."""
    try:
        out = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                             capture_output=True, text=True, check=True).stdout
    except FileNotFoundError:
        raise SystemExit("pdftotext not found; install poppler-utils")
    runs: list[str] = []
    for line in out.splitlines():
        # A laid-out line can hold several independent labels separated by wide
        # gaps. Splitting on runs of three or more spaces keeps them apart, so a
        # row of six short tick labels is not mistaken for one long sentence.
        runs.extend(part for part in re.split(r"\s{3,}", line) if part.strip())
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=FIG_DIR)
    ap.add_argument("--include-inherited", action="store_true",
                    help="also check the figures carried over from the papers")
    a = ap.parse_args()

    pdfs = sorted(a.dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDFs under {a.dir}")

    findings: list[tuple[str, str, str]] = []
    checked = 0
    for pdf in pdfs:
        if pdf.stem in INHERITED and not a.include_inherited:
            continue
        checked += 1
        for run in _runs(pdf):
            rule = _is_prose(run)
            if rule:
                findings.append((pdf.stem, rule, _normalise(run)))

    for stem, rule, text in findings:
        shown = text if len(text) <= 96 else text[:93] + "..."
        print(f"{stem}: [{rule}] {shown}")

    print(f"\n{checked} figures checked, {len(pdfs) - checked} inherited and "
          f"skipped, {len(findings)} prose runs found")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
