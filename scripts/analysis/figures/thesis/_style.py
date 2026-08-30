#!/usr/bin/env python3
"""Shared drawing style for the dissertation figures.

Why this file exists
--------------------
The figures shipped in the 2026-08 draft carried their argument *inside the
image*: a full-sentence ``set_title``, a two-to-four line explanatory paragraph
pinned just above the axes, and a small-grey provenance footnote along the
bottom edge. The eye has nowhere to land, and the reader ends up *reading* the
figure instead of *seeing* it.

The governing rule adopted 2026-08-26 is the one Zekun stated plainly: a figure
exists to lower the reader's effort, so a figure that costs more effort than the
sentence it replaces is strictly worse than that sentence. Everything below is
that rule made mechanical.

House rules enforced here
-------------------------
R1  No prose inside the image. Text drawn on an axes must be a *label* (a noun
    phrase, a number, a unit), never a claim. Claims live in the LaTeX caption,
    where they are typeset in the body font and counted as body text.
R2  One figure answers one question. If two questions need answering, that is
    two figures, not two panels bolted together.
R3  One reading direction. Left to right, or top to bottom, never both and never
    outward from a centre. Ranked data is sorted so that position itself carries
    the ordering.
R4  Direct labelling beats a legend. A legend forces a saccade between the mark
    and the key; a label next to the mark does not. Legends are permitted only
    when a series appears in several panels.
R5  No decoration that does not encode data: no gridlines behind bars, no box
    around the axes, no shadow, no third colour introduced for variety.

``check_no_prose.py`` verifies R1 against the rendered PDFs, so a regression is
caught by the build rather than by a reader.
"""
from __future__ import annotations

from pathlib import Path

import re as _re
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --------------------------------------------------------------------------
# Palette. Carried over from the conference figures so that a reader who has
# seen the paper reads the same colour as the same thing here. Orange is the
# text-only side, green is text plus image, blue is image only; the reference
# marks are greys so they never compete with the data.
# --------------------------------------------------------------------------
C_TEXT = "#E8720C"      # DOM and the phantom arms: text payload, no screenshot
C_BOTH = "#14855F"      # SoM: marked screenshot plus text
C_IMAGE = "#1F5FD6"     # Vision: screenshot only
C_INK = "#222222"       # primary ink for marks and labels
C_MUTED = "#8A8A8A"     # axis furniture, reference lines, de-emphasised marks
C_ACCENT = "#C2352B"    # reserved for the single element the figure is about
C_FILL = "#D9D9D9"      # neutral fill for bars that carry no categorical meaning

# One display spelling per mode, for every figure. The source artefacts carry
# the pipeline slugs (`som`, `ptext`, `pprompt`), and a figure that prints them
# raw contradicts the Reader's Guide, which tells the examiner these are called
# SoM, P-text and P-prompt. This table lived in fig_f10_rerun_discordance.py and
# its sibling f10b did not import it, which is exactly how the two figures came
# to spell the same six modes two different ways.
# One display spelling per cell, for the same reason. Three figures spelled the
# same eight cells three ways --- `VWA-cls · B0` (F10b), `classifieds·B0` (F16)
# and `cls·B0` (F7) --- because each parsed its own artefact. The VWA-/WA- prefix
# is not decoration: both benchmarks have a site called reddit, and `red·B0`
# alone does not say which one. Unrecognised cells pass through unchanged.
_CELL_SITE = {
    "classifieds": "VWA-cls", "cls": "VWA-cls", "vwa-cls": "VWA-cls",
    "reddit": "VWA-red", "red": "VWA-red", "vwa-red": "VWA-red",
    "wa_reddit": "WA-red", "wa_red": "WA-red", "wa-red": "WA-red",
}
_CELL_RE = _re.compile(r"^\s*(.+?)\s*[\u00b7]\s*(B\d+)\s*$")


def cell_label(cell) -> str:
    """Display spelling for a (site, baseline) cell; unknown forms pass through."""
    m = _CELL_RE.match(str(cell))
    if not m:
        return cell
    site, baseline = m.group(1), m.group(2)
    return f"{_CELL_SITE.get(site.strip().lower(), site)}\u00b7{baseline}"


MODE_LABEL = {"dom": "DOM", "vision": "Vision", "som": "SoM",
              "ptext": "P-text", "pprompt": "P-prompt", "psom": "P-SoM"}


def mode_label(slug: str) -> str:
    """Display spelling for a mode slug; unknown slugs pass through unchanged."""
    return MODE_LABEL.get(str(slug).strip().lower(), slug)


MODE_COLOUR = {
    "DOM": C_TEXT,
    "P-text": C_TEXT,
    "P-prompt": C_TEXT,
    "P-SoM": C_TEXT,
    "SoM": C_BOTH,
    "Vision": C_IMAGE,
}

# --------------------------------------------------------------------------
# Print width, and why every figure must be authored at it.
#
# The UCL template sets 40mm binding margins on A4, leaving a 13cm text block:
#
#     21.0cm  page
#    - 4.0cm  left margin
#    - 4.0cm  right margin
#    = 13.0cm = 5.12in of text
#
# A figure authored 9.8in wide and placed at \textwidth is scaled to 0.52, and
# an 8.5pt label lands on the page at 4.4pt. Measured in the built PDF before
# this constant existed: body text 10.56pt of glyph height, figure labels
# 4.99pt. That is unreadable, and it is the same failure the prose removal was
# meant to fix, arriving through a different door.
#
# The rule is therefore: author at the printed width. A figure is drawn 5.12in
# wide (or 2x that when it goes on a landscape page), placed at \textwidth, and
# the type sizes below are then what the reader actually gets.
PRINT_W_IN = 5.12          # \textwidth under the template, in inches
LANDSCAPE_W_IN = 10.12     # \textheight, for figures on a pdflscape page

# Type sizes, in points as printed. Because figures are authored at the printed
# width these are literal: FS_VALUE 8.5 renders 8.5pt on the page.
FS_TICK = 8.0
FS_LABEL = 8.5
FS_VALUE = 7.5
FS_PANEL = 9.0


def apply() -> None:
    """Install the house rcParams. Call once at the top of ``main``."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": C_MUTED,
        "axes.linewidth": 0.8,
        "axes.labelcolor": C_INK,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_PANEL,
        "xtick.color": C_INK,
        "ytick.color": C_INK,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": FS_TICK,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": False,
        "pdf.fonttype": 42,
    })


def panel_label(ax, text: str) -> None:
    """Put a short noun-phrase label above a panel of a multi-panel figure.

    Multi-panel figures are the one place a label is unavoidable, because the
    reader must be able to name the panel the caption is talking about. Keep it
    to a noun phrase: ``Held-out AUROC``, not ``Held-out AUROC shows that ...``.
    """
    ax.set_title(text, fontsize=FS_PANEL, fontweight="bold", loc="left",
                 color=C_INK, pad=8)


def value_label(ax, x, y, text: str, *, dx: float = 0.0, **kw) -> None:
    """Write a number next to the mark it belongs to, rather than in a legend."""
    kw.setdefault("fontsize", FS_VALUE)
    kw.setdefault("color", C_INK)
    kw.setdefault("va", "center")
    ax.text(x + dx, y, text, **kw)


def reference_line(ax, value: float, *, axis: str = "x", label: str = "") -> None:
    """Draw a zero line, a chance line, or a threshold, plus a short tag.

    The tag is deliberately a fragment (``chance``, ``0``, ``noise floor``) so it
    reads as an axis annotation and not as a sentence about the finding.
    """
    if axis == "x":
        ax.axvline(value, color=C_MUTED, lw=0.9, ls=(0, (4, 3)), zorder=0)
    else:
        ax.axhline(value, color=C_MUTED, lw=0.9, ls=(0, (4, 3)), zorder=0)
    if not label:
        return
    trans = ax.get_xaxis_transform() if axis == "x" else ax.get_yaxis_transform()
    if axis == "x":
        ax.text(value, 1.005, label, transform=trans, ha="center", va="bottom",
                fontsize=FS_VALUE, color=C_MUTED)
    else:
        ax.text(1.002, value, label, transform=trans, ha="left", va="center",
                fontsize=FS_VALUE, color=C_MUTED)


# Where the figures land, and why this module does not write them.
#
# Each figure script writes only ``final_dissertation/figures/``. The copy LaTeX
# actually embeds lives in ``final_dissertation/tex/figures/`` and is refreshed
# by ``make thesis-figures``, which regenerates, copies, and then runs
# ``check_no_prose.py``.
#
# That means running a figure script on its own leaves the embedded copy stale,
# and the next build silently ships the previous version of that figure. An
# earlier draft of this module carried a ``save()`` helper that wrote both trees
# and claimed in its docstring to have closed that hole; nothing ever called it,
# so the hole stayed open behind a comment saying it was shut. The helper is
# gone rather than wired up, because one path that is always correct beats two
# paths of which one is usually skipped.
#
#     ALWAYS:  make thesis-figures
#     NEVER:   python scripts/analysis/figures/thesis/fig_fN_*.py   (alone)
