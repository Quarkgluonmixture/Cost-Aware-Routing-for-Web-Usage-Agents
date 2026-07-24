#!/usr/bin/env python3
"""Lexical invariant gate: detect drift between two versions of a paper file.

A deslop/humanize rewrite must change HOW things are said, never WHAT is
said. This gate compares OLD and NEW and fails when protected lexical
material drifts. It is a tripwire, not a proof of semantic equivalence:
hedge strength and argument logic are the diff reviewer's job.

Checks:
  1. numbers      -- numeric literals with sign, unit, and percent attached
                     ("-4.23%", "5ms", ".05", ranges "10--12"), as a multiset
  2. citations    -- \\cite/\\citep/\\citet/\\parencite/... keys
                     (plus pandoc @key citations in .md files)
  3. crossrefs    -- \\ref/\\eqref/\\autoref/\\cref keys and \\label keys
  4. structrefs   -- literal structural pointers ("§8", "Section 7.2",
                     "Table 3"), normalized to kind:number and compared as a
                     multiset. They are pointers, not data: a rewrite may
                     move one into a completely new sentence, so they are
                     deliberately kept out of the number and anchor checks
                     (see structref_tokens)
  5. terms        -- occurrence counts of whitelisted domain terms
                     (terms.txt), matched on word boundaries -- see
                     term_pattern for the boundary/plural/case rules
  6. protected blocks -- exact (whitespace-normalized) comparison of math
                     ($...$, \\(...\\), \\[...\\], $$...$$, display envs),
                     macro-definition lines, verbatim/lstlisting/\\verb,
                     comments, and the preamble (before \\begin{document})
  7. anchors      -- each number/citation must stay in a sentence sharing at
                     least one content word with its original sentence
                     (catches "92% and 4.23% swapped between claims" and
                     "citation moved to a different claim")

Comment and verbatim syntax is per format, decided by the NEW file's
suffix. LaTeX files use "%" comments and verbatim/lstlisting/\\verb;
Markdown files (.md/.markdown/.qmd/.rmd) use "<!-- -->" comments, fenced
code blocks, and inline `code` -- and crucially do NOT treat "%" as a
comment, since "42.0% [@key]" is ordinary Markdown prose.

Usage:
    invariant_check.py OLD NEW [--terms terms.txt] [--report-only]

Exit codes: 0 = no drift detected, 1 = drift found (always 0 with
--report-only), 2 = usage error.

Known blind spots (documented, deliberate):
  - spelled-out numbers ("twenty");
  - a whitelist term pluralized in place (regular plurals are folded into one
    count on purpose: rewording around number is legitimate editing);
  - hedge strengthening/weakening and any purely semantic rewording --
    out of scope for a lexical gate; the rewrite layer must emit diffs.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

CITE_RE = re.compile(
    r"\\(?:[Cc]ite[tp]?|parencite|textcite|autocite|citeauthor|citeyear)\*?"
    r"(?:\[[^\]]*\])*\{([^}]*)\}"
)
REF_RE = re.compile(r"\\(?:ref|eqref|autoref|[Cc]ref|pageref|vref)\*?\{([^}]*)\}")
LABEL_RE = re.compile(r"\\label\{([^}]*)\}")
PANDOC_CITE_RE = re.compile(r"(?<![\w.])@([A-Za-z][\w:-]*)")

# Ranges first (10--12 stays one token), then signed/unsigned numbers with
# optional leading decimal. The sign is captured only when it cannot be part
# of an en-dash range or an identifier (gpt-4.1).
NUM_RE = re.compile(
    r"(?P<range>\d+(?:\.\d+)?--\d+(?:\.\d+)?)"
    r"|(?P<num>(?<![\w.\-])-(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
    r"|(?<![\w.])(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)"
)
PERCENT_RE = re.compile(r"\s*(?:\\%|%)")
UNIT_RE = re.compile(
    r"\s*(ms|µs|us|ns|sec|min|hr|Hz|kHz|MHz|GHz|KB|MB|GB|TB|dB|px|pt|FLOPs?|[shB])"
    r"(?![a-zA-Z])"
)

DISPLAY_MATH_RE = re.compile(
    r"(?s)(\\\[.*?\\\]|\$\$.*?\$\$"
    r"|\\begin\{(?:equation|align|gather|multline|eqnarray|math|displaymath)\*?\}"
    r".*?\\end\{(?:equation|align|gather|multline|eqnarray|math|displaymath)\*?\})"
)
INLINE_MATH_RE = re.compile(r"(\\\(.*?\\\)|\$[^$\n]+\$)")
VERBATIM_RE = re.compile(
    r"(?s)(\\begin\{(?:verbatim|lstlisting|minted)\*?\}.*?"
    r"\\end\{(?:verbatim|lstlisting|minted)\*?\}|\\verb\*?(\S).*?\2)"
)
MACRO_LINE_RE = re.compile(r"^\s*\\(?:(?:re)?newcommand|providecommand|def)\b")
COMMENT_RE = re.compile(r"(?<!\\)%(.*)")

# Markdown counterparts. "%" is NOT a comment in Markdown -- treating it as
# one silently truncated every line containing a percentage, hiding the
# numbers and citations after it from every check.
MD_SUFFIXES = frozenset({".md", ".markdown", ".qmd", ".rmd"})
HTML_COMMENT_RE = re.compile(r"(?s)<!--(.*?)-->")
FENCE_RE = re.compile(r"(?sm)^[ \t]*(`{3,}|~{3,})[^\n]*\n.*?^[ \t]*\1[ \t]*$")
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")

# Literal structural pointers: "§8", "§~8", "Section 7.2", "Sections 3--4",
# "Table 3", "Fig. 2b". These are cross-references, not data. In LaTeX they
# are usually \ref{...} (already covered); Markdown papers write them out.
STRUCTREF_KINDS = {
    "§": "sec", "§§": "sec", "section": "sec", "sections": "sec",
    "sec": "sec", "secs": "sec",
    "chapter": "chap", "chapters": "chap", "chap": "chap", "chaps": "chap",
    "ch": "chap",
    "figure": "fig", "figures": "fig", "fig": "fig", "figs": "fig",
    "table": "tab", "tables": "tab", "tab": "tab", "tabs": "tab",
    "equation": "eq", "equations": "eq", "eq": "eq", "eqs": "eq",
    "appendix": "app", "appendices": "app", "app": "app",
    "algorithm": "alg", "algorithms": "alg", "alg": "alg",
    "listing": "lst", "listings": "lst",
    "theorem": "thm", "lemma": "lem", "proposition": "prop",
    "corollary": "cor", "definition": "defn", "remark": "rem",
}
STRUCTREF_RE = re.compile(
    r"(?:(§§?)|\b(Sections?|Secs?|Chapters?|Chaps?|Ch|Figures?|Figs?|Tables?"
    r"|Tabs?|Equations?|Eqs?|Appendi(?:x|ces)|App|Algorithms?|Alg|Listings?"
    r"|Theorems?|Lemmas?|Propositions?|Corollar(?:y|ies)|Definitions?"
    r"|Remarks?)\.?)"
    r"[\s~]*(\d+(?:\.\d+)*[a-z]?(?:\s*--\s*\d+(?:\.\d+)*[a-z]?)?)",
    re.I,
)

STOPWORDS = frozenset("""
    that this these those with from than which have been were into over under
    both each could might would should shall also only very much many when
    where while such more most some then they their there here about after
    before between through during against because does done being other
""".split())


def strip_comments(text: str, latex: bool = True) -> str:
    """Blank out comments. In Markdown "%" is a percent sign, not a comment;
    stripping to end-of-line there hid every number and citation that
    followed a percentage on the same line."""
    if latex:
        return re.sub(r"(?<!\\)%.*", "", text)
    return HTML_COMMENT_RE.sub(" ", text)


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------- protected
def extract_protected(raw: str, latex: bool = True) -> dict[str, Counter]:
    """Exact-match inventories of blocks a rewrite must never touch."""
    blocks: dict[str, Counter] = {}
    comment_re = COMMENT_RE if latex else HTML_COMMENT_RE
    blocks["comment"] = Counter(
        normalize_ws(m.group(1)) for m in comment_re.finditer(raw)
    )
    text = strip_comments(raw, latex)

    if "\\begin{document}" in text:
        preamble = text.split("\\begin{document}", 1)[0]
        blocks["preamble"] = Counter([normalize_ws(preamble)])
    else:
        blocks["preamble"] = Counter()

    verbatim: Counter = Counter()
    for regex in verbatim_regexes(latex):
        for m in regex.finditer(text):
            verbatim[normalize_ws(m.group(0))] += 1
        text = regex.sub(" ", text)
    blocks["verbatim"] = verbatim

    math: Counter = Counter()
    for m in DISPLAY_MATH_RE.finditer(text):
        math[normalize_ws(m.group(1))] += 1
    text = DISPLAY_MATH_RE.sub(" ", text)
    for m in INLINE_MATH_RE.finditer(text):
        math[normalize_ws(m.group(1))] += 1
    blocks["math"] = math

    blocks["macro"] = Counter(
        normalize_ws(line)
        for line in strip_comments(raw, latex).splitlines()
        if MACRO_LINE_RE.match(line)
    )
    return blocks


def verbatim_regexes(latex: bool) -> tuple[re.Pattern, ...]:
    """Code that must survive verbatim: LaTeX verbatim/lstlisting/\\verb, and
    in Markdown fenced blocks and inline `code`. Fences come first so that a
    "$" or "%" inside a code block is never read as markup."""
    if latex:
        return (VERBATIM_RE,)
    return (FENCE_RE, VERBATIM_RE, INLINE_CODE_RE)


def prose_only(raw: str, latex: bool = True) -> str:
    """Comment-free text with protected blocks blanked out."""
    text = strip_comments(raw, latex)
    if "\\begin{document}" in text:
        text = text.split("\\begin{document}", 1)[1]
    for regex in verbatim_regexes(latex):
        text = regex.sub(" ", text)
    text = DISPLAY_MATH_RE.sub(" MATH ", text)
    text = INLINE_MATH_RE.sub(" MATH ", text)
    text = "\n".join(
        line for line in text.splitlines() if not MACRO_LINE_RE.match(line)
    )
    return text


# ------------------------------------------------------------------ tokens
def number_tokens(text: str):
    """Yield (token, span) with sign, unit, and percent folded in."""
    for m in NUM_RE.finditer(text):
        tok = m.group(0)
        end = m.end()
        if m.lastgroup == "num":
            pm = PERCENT_RE.match(text, end)
            if pm:
                tok += "%"
                end = pm.end()
            else:
                um = UNIT_RE.match(text, end)
                if um:
                    tok += um.group(1)
                    end = um.end()
        yield tok, (m.start(), end)


def structref_tokens(text: str) -> Counter:
    """Structural pointers as a normalized multiset: "Section 7" , "§7", and
    "Sec. 7" all become "sec:7", while "Figure 7" stays distinct. A range
    ("§§7--8") counts as its endpoints, so splitting it into "§7 and §8" is
    not drift.

    These are counted but NOT anchored. A pointer carries no claim: moving
    "§8" into a freshly written sentence is exactly what Pass 1 of a deslop
    rewrite does (split, merge, reposition clauses), and treating "8" as a
    data point made the gate fire on its own recommended edit. Retargeting
    (§8 -> §9) still fails here, on the count.
    """
    counts: Counter = Counter()
    for m in STRUCTREF_RE.finditer(text):
        word = (m.group(1) or m.group(2)).rstrip(".").lower()
        kind = STRUCTREF_KINDS.get(word)
        if kind is None:
            continue
        for part in re.split(r"\s*--\s*", m.group(3)):
            counts[f"{kind}:{part.strip()}"] += 1
    return counts


def strip_refs(text: str) -> str:
    """Remove everything that is a pointer rather than a claim: citation
    keys, cross-references, labels, and literal structural pointers."""
    text = CITE_RE.sub(" ", text)
    text = REF_RE.sub(" ", text)
    text = LABEL_RE.sub(" ", text)
    return STRUCTREF_RE.sub(" ", text)


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?;])\s+", text) if s.strip()]


def content_words(sentence: str) -> frozenset[str]:
    sentence = strip_refs(sentence)
    sentence = re.sub(r"\\[a-zA-Z@]+", " ", sentence)
    words = re.findall(r"[a-zA-Z]{4,}", sentence.lower())
    return frozenset(w for w in words if w not in STOPWORDS)


def anchored_tokens(prose: str):
    """Map each number token / citation key -> list of anchor word-sets."""
    num_anchors: dict[str, list[frozenset]] = {}
    cite_anchors: dict[str, list[frozenset]] = {}
    for sent in sentences(prose):
        anchor = content_words(sent)
        for m in CITE_RE.finditer(sent):
            for key in m.group(1).split(","):
                if key.strip():
                    cite_anchors.setdefault(key.strip(), []).append(anchor)
        for tok, _ in number_tokens(strip_refs(sent)):
            num_anchors.setdefault(tok, []).append(anchor)
    return num_anchors, cite_anchors


def rebound_violations(kind: str, old: dict, new: dict, lines: list[str]) -> int:
    """For tokens with matching counts, each old occurrence must share at
    least one content word with some new occurrence (greedy assignment)."""
    n = 0
    for tok, old_sets in sorted(old.items()):
        new_sets = list(new.get(tok, []))
        if len(new_sets) != len(old_sets):
            continue  # count drift already reported by the multiset check
        for old_anchor in old_sets:
            if not old_anchor:
                continue
            best_i, best_overlap = -1, -1
            for i, new_anchor in enumerate(new_sets):
                overlap = len(old_anchor & new_anchor)
                if overlap > best_overlap:
                    best_i, best_overlap = i, overlap
            if best_i >= 0 and best_overlap == 0 and new_sets[best_i]:
                lines.append(
                    f"    ! rebound {kind}: {tok!r} (no shared context words "
                    f"with its original sentence)"
                )
                n += 1
            if best_i >= 0:
                new_sets.pop(best_i)
    return n


# ---------------------------------------------------------------- plumbing
def load_terms(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _acronymish(word: str) -> bool:
    """True for DOM, SoM, AXTree, US -- a capital anywhere but the first
    letter. 'Transformer' is not acronymish: only its first letter is
    capitalized, so it may legitimately appear lowercase mid-sentence."""
    return any(c.isupper() for c in word[1:])


@lru_cache(maxsize=None)
def term_pattern(term: str) -> re.Pattern:
    """Word-boundary-anchored pattern for one whitelist term.

    Rules, each one paying for a false-positive class seen on real drafts:
      - spaces and hyphens inside the term match any run of whitespace,
        tilde, or hyphen, so a hard-wrapped "cost-accuracy trade-off"
        still counts as one occurrence;
      - the match must stand on word boundaries, so "DOM" no longer fires
        inside "random", "domain", or "dominant" and "SoM" no longer fires
        inside "some" -- without this, short acronyms could not be
        whitelisted at all, and those are exactly the tokens a rewrite is
        tempted to touch;
      - a regular plural or possessive is folded into the same count,
        keeping the documented "pluralized in place" blind spot rather
        than turning legitimate rewording into a violation (irregular
        plurals -- policy/policies, matrix/matrices -- are NOT folded and
        do show up as drift);
      - a word with a capital anywhere but the first letter is matched
        case-sensitively (otherwise a whitelisted "US" or "IT" would count
        every "us" and "it"); ordinary words stay case-insensitive so that
        sentence-initial capitalization cannot shift a count.
    """
    words = [w for w in re.split(r"[\s-]+", term) if w]
    if not words:
        return re.compile(r"(?!)")  # never matches
    core = r"[\s~-]+".join(
        re.escape(w) if _acronymish(w) else f"(?i:{re.escape(w)})" for w in words
    )
    lead = r"(?<!\w)" if re.match(r"\w", words[0][0]) else ""
    trail = r"(?:['’]s|e?s)?(?!\w)" if re.match(r"\w", words[-1][-1]) else ""
    return re.compile(lead + core + trail)


def term_counts(text: str, terms: list[str], latex: bool = True) -> Counter:
    text = strip_comments(text, latex)
    counts: Counter = Counter()
    for term in terms:
        n = len(term_pattern(term).findall(text))
        if n:
            counts[term] = n
    return counts


def diff_counters(label: str, old: Counter, new: Counter, lines: list[str]) -> int:
    removed = old - new
    added = new - old
    for key, n in sorted(removed.items()):
        lines.append(f"    - removed {label}: {_preview(key)!r} x{n}")
    for key, n in sorted(added.items()):
        lines.append(f"    + added   {label}: {_preview(key)!r} x{n}")
    return len(removed) + len(added)


def _preview(s: str, limit: int = 60) -> str:
    return s if len(s) <= limit else s[: limit - 3] + "..."


def report_section(name: str, count_note: str, lines: list[str], n: int) -> None:
    status = "OK" if n == 0 else f"{n} violation(s)"
    print(f"  {name + ':':<12}{status} ({count_note})")
    for line in lines:
        print(line)


def term_audit(terms, terms_path, old_raw: str, new_raw: str, args,
               latex: bool = True) -> int:
    """Per-term hit counts, so a whitelist can be curated against a real
    draft before the gate is trusted. Entries that never occur are usually
    typos or aspirational; a term with a surprising count is worth a look
    (it may be matching somewhere you did not expect)."""
    if not terms:
        print(f"term audit: no terms file found ({terms_path})")
        return 0
    old_c = term_counts(old_raw, terms, latex)
    new_c = term_counts(new_raw, terms, latex)
    print(f"term audit: {terms_path} ({len(terms)} terms)")
    print(f"  {'OLD':>6} {'NEW':>6}  term      "
          f"[{args.old.name} -> {args.new.name}]")
    dead = 0
    for term in terms:
        o, n = old_c.get(term, 0), new_c.get(term, 0)
        note = ""
        if o == 0 and n == 0:
            note, dead = "   <- never occurs (typo, or drop it)", dead + 1
        elif o != n:
            note = f"   <- drift {o - n:+d}"
        print(f"  {o:>6} {n:>6}  {term}{note}")
    if dead:
        print(f"  {dead} term(s) never occur in either file")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("old", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--terms", type=Path, default=None,
                    help="whitelist file (default: terms.txt at the repo root, if present)")
    ap.add_argument("--report-only", action="store_true",
                    help="print the report but always exit 0")
    ap.add_argument("--term-audit", action="store_true",
                    help="print per-term hit counts for both files and exit 0 "
                         "(whitelist curation; pass the same file twice to "
                         "audit terms.txt against one draft)")
    args = ap.parse_args()

    for p in (args.old, args.new):
        if not p.is_file():
            print(f"error: {p} is not a file", file=sys.stderr)
            return 2

    terms_path = args.terms
    if terms_path is None:
        default = Path(__file__).resolve().parent.parent / "terms.txt"
        terms_path = default if default.is_file() else None
    terms = load_terms(terms_path) if terms_path and terms_path.is_file() else []

    old_raw = args.old.read_text(errors="replace")
    new_raw = args.new.read_text(errors="replace")
    # Comment and verbatim syntax follow the format; "%" is a comment in
    # LaTeX and a percent sign in Markdown.
    pandoc = args.new.suffix.lower() in MD_SUFFIXES
    latex = not pandoc

    if args.term_audit:
        return term_audit(terms, terms_path, old_raw, new_raw, args, latex)

    old_prose = prose_only(old_raw, latex)
    new_prose = prose_only(new_raw, latex)

    print(f"lexical-invariant-gate: {args.old} -> {args.new}"
          f"  [{'latex' if latex else 'markdown'}]")
    violations = 0

    # 1. numbers (multiset over prose, sign/unit/percent folded in)
    old_nums = Counter(tok for tok, _ in number_tokens(strip_refs(old_prose)))
    new_nums = Counter(tok for tok, _ in number_tokens(strip_refs(new_prose)))
    lines: list[str] = []
    n = diff_counters("number", old_nums, new_nums, lines)
    report_section("numbers", f"{sum(old_nums.values())} in old", lines, n)
    violations += n

    # 2. citations
    def cite_counter(prose: str) -> Counter:
        c: Counter = Counter()
        for m in CITE_RE.finditer(prose):
            for key in m.group(1).split(","):
                if key.strip():
                    c[key.strip()] += 1
        if pandoc:
            for m in PANDOC_CITE_RE.finditer(prose):
                c[m.group(1)] += 1
        return c

    old_cites, new_cites = cite_counter(old_prose), cite_counter(new_prose)
    lines = []
    n = diff_counters("citation", old_cites, new_cites, lines)
    report_section("citations", f"{sum(old_cites.values())} in old", lines, n)
    violations += n

    # 3. crossrefs and labels
    for name, regex in (("crossrefs", REF_RE), ("labels", LABEL_RE)):
        old_c = Counter(m.group(1).strip() for m in regex.finditer(old_prose))
        new_c = Counter(m.group(1).strip() for m in regex.finditer(new_prose))
        lines = []
        n = diff_counters(name.rstrip("s"), old_c, new_c, lines)
        report_section(name, f"{sum(old_c.values())} in old", lines, n)
        violations += n

    # 4. structural pointers ("§8", "Section 7.2", "Table 3"): counted, never
    #    anchored -- a pointer has no claim to be rebound to.
    old_sr, new_sr = structref_tokens(old_prose), structref_tokens(new_prose)
    lines = []
    n = diff_counters("structref", old_sr, new_sr, lines)
    report_section("structrefs", f"{sum(old_sr.values())} in old", lines, n)
    violations += n

    # 5. whitelist terms
    if terms:
        lines = []
        n = diff_counters("term count", term_counts(old_raw, terms, latex),
                          term_counts(new_raw, terms, latex), lines)
        report_section("terms", f"whitelist: {len(terms)} terms", lines, n)
        violations += n
    else:
        print("  terms:      skipped (no terms file found)")

    # 6. protected blocks (exact, whitespace-normalized)
    old_blocks = extract_protected(old_raw, latex)
    new_blocks = extract_protected(new_raw, latex)
    for kind in ("math", "macro", "preamble", "comment", "verbatim"):
        lines = []
        n = diff_counters(kind, old_blocks[kind], new_blocks[kind], lines)
        report_section(kind, f"{sum(old_blocks[kind].values())} in old", lines, n)
        violations += n

    # 7. anchors: numbers/citations must keep some original sentence context
    old_num_a, old_cite_a = anchored_tokens(old_prose)
    new_num_a, new_cite_a = anchored_tokens(new_prose)
    lines = []
    n = rebound_violations("number", old_num_a, new_num_a, lines)
    n += rebound_violations("citation", old_cite_a, new_cite_a, lines)
    report_section("anchors", "sentence-context binding", lines, n)
    violations += n

    if violations:
        print(f"RESULT: FAIL ({violations} violation(s))"
              + (" [report-only]" if args.report_only else ""))
        return 0 if args.report_only else 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
