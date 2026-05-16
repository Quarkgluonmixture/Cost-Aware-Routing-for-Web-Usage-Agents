"""Codex-judged non-visual / uncertain task ids for VisualWebArena.

⚠️ **Provenance correction (2026-04-26)**: original docstring claimed
"Manually audited / human audit" but the classification was actually generated
by an LLM (codex). Treat the lists below as a single-judge LLM audit, not an
authoritative manual benchmark — known to under-classify visual cues (see
experiment notes §101). A re-audit with explicit per-task reasoning is planned;
new output will go to ``docs/analysis/cross_sites/codex_audit_{site}.json``.

A task is *non-visual* if it can be solved purely from DOM/AXTree text without
inspecting any image — i.e. every fact required by the intent is reachable as
text in the page tree (titles, prices, descriptions, listing labels, etc.).

A task is *uncertain* when textual cues exist but the task may still degrade
on text-only agents (color/material/visual property described only in image,
visual-counting tasks, emoji rendering, etc.). The accompanying note captures
the audit reasoning so future re-audits can revisit the call.

Use cases:
- Paper framing: rebut "DOM looks bad because the benchmark needs vision" by
  showing per-mode SR on the non_visual subset for sites with N >= ~30.
- Routing analysis: lower bound on tasks where the cheap (DOM) mode should be
  competitive with the rich (SoM/vision) mode.

Sample sizes (as of audit 2026-Q1):
- classifieds: 9 / 234 (~3.8%)
- reddit:      1 / 210 (~0.5%) — too small for statistics, kept for completeness
- shopping:   33 / 466 (~7.1%)

WA tasks are entirely non-visual by construction (no reference images),
so this file does not enumerate them; treat all WA task ids as non_visual.
"""
from __future__ import annotations


NON_VISUAL_TASK_IDS: dict[str, list[int]] = {
    "classifieds": [206, 209, 210, 211, 212, 215, 217, 218, 219],
    "reddit": [75],
    "shopping": [
        0, 3, 4, 5, 7, 12, 19, 30, 31, 45,
        72, 73, 74, 188, 192, 221, 223, 236, 237, 263,
        267, 271, 316, 317, 345, 387, 421, 422, 424, 426,
        428, 444, 449,
    ],
}


UNCERTAIN_TASK_IDS: dict[str, dict[int, str]] = {
    "classifieds": {
        40: "stainless steel may be in listing text, but can also require visual material recognition",
        203: "USB-C cable compatibility may be described in text or only visible from the cable image",
        221: "bowl count may be in listing text, but likely requires counting items in the listing image",
    },
    "reddit": {
        43: "yellow projector might be encoded in product text, but target product name does not include yellow",
        44: "posting product image may be possible from image URL metadata, but task still depends on image handling",
        162: "GIF could be identified by post title, but the requested contrast is visual content",
        172: "crispy chicken sandwich may appear in title text, but likely identifies food image content",
        173: "emoji is visible as Unicode in DOM text, but the task asks for rendered emoji content",
        174: "emoji is visible as Unicode in DOM text, but the task asks for rendered emoji content",
    },
    "shopping": {
        187: "opaque phone case may be inferred from title/material text, but opacity is a visual property",
        250: "soccer ball earbuds may be a product-title phrase or a visual design cue",
        252: "white Wii remotes may be title/variant text, but color is the discriminator",
        323: "N/A could be reached textually, but proving no purple drink on the page is visual",
        385: "beer in a box may be product-type text or packaging recognition",
        423: "white humidifier likely comes from image appearance; target text does not include white",
        448: "no armrests may be inferable from titles such as stool/backless, but can require visual inspection",
    },
}
