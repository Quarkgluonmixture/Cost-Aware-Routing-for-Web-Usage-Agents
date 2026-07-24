#!/usr/bin/env python3
"""Crude LaTeX -> plain text conversion for grammar checking.

Not a full detex. The goal is prose with sentence structure intact and
markup noise removed, so LanguageTool sees natural sentences instead of
commands. Math becomes a placeholder token, citations become "[1]",
figure/table bodies are dropped but their captions are kept.

Usage: tex_to_text.py [FILE]   (reads stdin without a FILE)
"""
from __future__ import annotations

import re
import sys

FORMAT_CMDS = r"emph|textbf|textit|texttt|textsc|underline|mbox|text|textrm|footnote"


def convert(tex: str) -> str:
    m = re.search(r"\\begin\{document\}(.*)\\end\{document\}", tex, re.S)
    if m:
        tex = m.group(1)
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    # Display math -> placeholder.
    tex = re.sub(
        r"(?s)\\begin\{(equation|align|gather|multline|eqnarray|displaymath|math)\*?\}"
        r".*?\\end\{\1\*?\}",
        " the equation ",
        tex,
    )
    tex = re.sub(r"\$\$[^$]*\$\$", " the equation ", tex)
    tex = re.sub(r"\$[^$\n]*\$", "X", tex)
    # Keep captions, drop the rest of float environments.
    captions = re.findall(r"\\caption\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", tex)
    tex = re.sub(
        r"(?s)\\begin\{(figure|table|algorithm|tikzpicture|tabular)\*?\}.*?\\end\{\1\*?\}",
        " ",
        tex,
    )
    # Citations and cross-references -> readable placeholders.
    tex = re.sub(
        r"\\(?:[Cc]ite[tp]?|parencite|textcite|autocite)\*?(?:\[[^\]]*\])*\{[^}]*\}",
        "[1]",
        tex,
    )
    tex = re.sub(r"\\(?:ref|eqref|autoref|[Cc]ref|pageref|vref)\*?\{[^}]*\}", "1", tex)
    tex = re.sub(r"\\label\{[^}]*\}", "", tex)
    # Section headings on their own lines, ended like sentences.
    tex = re.sub(r"\\(?:sub)*section\*?\{([^}]*)\}", r"\n\n\1.\n\n", tex)
    # Unwrap formatting commands, keeping their argument (nested up to 3 deep).
    for _ in range(3):
        tex = re.sub(rf"\\(?:{FORMAT_CMDS})\{{([^{{}}]*)\}}", r"\1", tex)
    tex = re.sub(r"\\item\b", "\n- ", tex)
    tex = re.sub(r"\\begin\{[^}]*\}(?:\[[^\]]*\])?|\\end\{[^}]*\}", " ", tex)
    tex = tex.replace("~", " ")
    tex = re.sub(r"\\[a-zA-Z@]+\s*(?:\[[^\]]*\])?", " ", tex)
    tex = re.sub(r"\\([%&#$_{}])", r"\1", tex)
    tex = re.sub(r"[{}]", "", tex)
    tex = re.sub(r"[ \t]+", " ", tex)
    tex = re.sub(r" +([.,;:)])", r"\1", tex)
    tex = re.sub(r"\n{3,}", "\n\n", tex)
    if captions:
        tex += "\n\n" + "\n".join(
            c.strip() + ("" if c.strip().endswith(".") else ".") for c in captions
        )
    return tex.strip() + "\n"


if __name__ == "__main__":
    source = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()
    sys.stdout.write(convert(source))
