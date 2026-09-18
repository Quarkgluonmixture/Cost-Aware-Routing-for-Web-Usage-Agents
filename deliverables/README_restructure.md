# REALM @ EMNLP 2026 — restructured submission skeleton

Created 2026-08-04 from the Overleaf draft (`../6a59017b04233a73ed5ec570/realm_body.tex`,
commit `89fdd0a`). Deadline: **2026-08-05 11:59pm AoE** (= Beijing 08-06 19:59), OpenReview
direct submission, ACL 2026 style, double-blind, 8 pages content + unlimited refs/appendix.

## Layout

- `main_restructured.tex` — driver (`[review]` mode). Content ends **page 4** of the 8-page budget;
  ~4 pages of headroom for real prose. 35 pages total with appendix (appendix is unlimited).
- `sections/1_intro.tex … 8_discussion.tex` — section skeleton per the 2026-08-03 four-role
  review (AC storyline, corrected by reviewer/statistician/auditor). Every unwritten piece
  is a visible yellow `\todo`.
- `tables/tab01.tex … tab40.tex` — all 40 evidence tables split out, **source-order
  numbering preserved in the filename and `\label{tab:tNN}`** (matches the reading guide
  and Jiaming's inventory numbering). LaTeX renumbers display numbers automatically.
- Main body carries 7 tables: t01, t03, t16, t17, t18, t27, t28.
  Appendix groups the other 33 (A classes / B matrices / C noise / D efficiency /
  E routing / F failures+mechanisms / G audits / H evidence inventory).

## Repairs applied to the source (mechanical only, no numbers touched)

- 7 Pandoc-damaged captions fixed: stray `}word*` / `}word\emph{` closed the caption
  early and dumped the rest as loose float text → now proper `\emph{…}` with the caption
  closed at line end (t13, t16, t18, t20, t26, t39, t40).
- 6 stray `.*` junk lines removed.
- 25 wide tables (≥6 columns) converted to `table*` (full-width) — overfull hboxes
  1399 → 23.

## Flagged, not fixed (decisions pending the Zekun–Jiaming alignment)

- t20/t21/t22/t33: overlap with paper B (cut candidates per AC).
- t33: caption truncated in source, damaged cells.
- t35: table body **empty** — fill from `evaluator_score_granularity.json` or cut.
- Abstract/intro/discussion prose; title choice (3 AC candidates in `main_restructured.tex` comments).
- The evidence inventory (old abstract) is preserved verbatim as Appendix H, including
  its now-stale "WA cells are unaudited" line (WA audit landed 08-03, 0 leaked).

Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
