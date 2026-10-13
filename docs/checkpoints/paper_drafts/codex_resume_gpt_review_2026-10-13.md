# Codex resume prompt — GPT-review close-out (2026-10-13)

> **Status after submission (ARR #7345, 2026-10-13):** part 3 is DONE in a reduced form — appendix
> `app:otherrouters` (`sections/appendix_routers.tex`) is in the submitted paper, built from the recovered
> §368 / §374 products (E: backup) and existing products; only the tracked-product step and the LLM-pilot
> rescoring remain. Parts 1 and 2 were stopped mid-build: the unreviewed half-edits are in the working
> tree and in `wf_halfedits_2026-10-13.patch` next to this file. The paper is now for the author
> response / camera-ready, not the submission — do not push paper changes to Overleaf without asking.

You are continuing work another agent (Claude Code) started in this repo and may not have finished.
Work out where it stopped, finish everything below, and verify each step. Do not redo finished
work; do not trust a step as done until you have checked its artefact.

## Setting

- Repo: `C:/Workspace/Cost-Aware-Routing-for-Web-Usage-Agents`, branch `evidence/routing-frontier-20261007`
  (the COLING work lives here; do NOT switch to or merge into `master`). The GitHub repo is PUBLIC.
- Paper source: `deliverables/coling2027/` (main.tex, sections/, tables/, figures/). The same files are
  mirrored in the Overleaf git clone `~/overleaf-coling2027` (branch `main`).
- Python: ONLY `~/.venvs/car-analysis/Scripts/python.exe` (3.12), run from the repo root with
  `PYTHONUTF8=1`. The system python is 3.11 and cannot parse this code. Verified: re-running
  `scripts/analysis/routing_three_arm.py` reproduces its committed JSON exactly (~5.5 min).
- LaTeX: `latexmk -pdf` works (TeX Live 2025). Build in a scratch copy, never in the repo dir.
- A senior co-author (学长) and another session edit the paper on Overleaf concurrently. Division of labour:
  the senior co-author owns prose wording; we own numbers, figures, tables, the appendix material
  below, and factual corrections. Keep sentence edits minimal and list every one you make.

## The task (three parts, user approved all three)

A GPT review said: (C1) only three cells were fully rerun; (C2) rejecting the additive-difficulty
(Rasch) null does not establish narrow per-task representation preference; (C3) routers never
gained reliably and only two lightweight methods were tested. Evidence inventory, with a
do-not-cite list: `docs/checkpoints/paper_drafts/gpt_review_evidence_2026-10-13.md` — read it first.

1. **Fourth fully replicated cell (C1).** cls_B1 has registered reruns for dom/som/vision, so for the
   three-mode readings it is a fourth complete cell (cls_B2 lacks dom; it does not qualify).
   `scripts/analysis/lib/replicate_pairs.py: full_paired_cells()` required all six modes. Add a mode-subset
   path without changing six-mode callers; switch `routing_three_arm.py`'s null model / D-study / A→B to
   the three-mode paired cells; regenerate `docs/analysis/cross_sites/routing_three_arm.{json,md}`
   (+ check the `_wallclock` / `_gpu_time` siblings); extend `fig_dstudy` in
   `scripts/analysis/figures/coling/make_figures.py` to all null cells (it zipped only three styles —
   a fourth cell would be silently dropped); regenerate figures/tables with make_figures.py.
2. **Mode-specific-slope null (C2).** New `scripts/analysis/task_mode_slope_null.py` →
   `docs/analysis/cross_sites/task_mode_slope_null.{json,md}`: 2PL `logit P = a_m θ_i + b_m`, θ ~ N(0,1),
   marginal ML (Gauss–Hermite), runs share parameters; Rasch special case fitted the same way as a
   calibration against the existing curveball null; parametric bootstrap B=2000 with a fixed seed that
   preserves the task-difficulty distribution; recompute exactly the statistics of
   `task_mode_interaction_null.py` / `routing_three_arm.py` (sigma2_int, stable strict preference counts)
   — prove equivalence by reproducing the observed values; Holm across cells; six modes on cls_B0/red_B0/
   wared_B1, three modes on those plus cls_B1; secondary: opposite-direction preference counts within
   equal-total-success strata; LR/AIC 2PL vs Rasch and fitted a_m. Report honestly whether the
   interaction exceeds the 2PL null.
3. **Other routing constructions (C3).** Tracked products for: the §368 classifier sweep
   (`scripts/analysis/router_model_sweep.py`; notebook §368 says 15/15 below always-SoM 27.23 %,
   best LR 25.45 %, MLP 25.00 %), the §374 kNN / FrugalGPT-style baselines, and an offline rescoring of the
   §505 GPT-5.6 prompt-router pilot (`results/router_llm_pilot_20260909/`, no API calls, canonical task
   universe, EXCLUDE every row involving cls_B5 Vision). Then an appendix subsection with one table
   (construction, what it chooses, cells, result vs fixed modes / mixtures, source product), every number
   from a product; draft at `deliverables/coling2027/sections/appendix_routers_draft.tex`.

## Find out where the previous agent stopped

- `git status` / `git diff --stat` / `git log --oneline -5` on the branch, and the same in `~/overleaf-coling2027`.
- Which products / scripts above exist, and whether they regenerate identically.
- Previous agents' reports, if any, are in the workflow journal:
  `C:/Users/Administrator/.claude/projects/C--Workspace/5524f7f1-3aa8-485c-ab3b-023d1dc116ec/subagents/workflows/wf_fdc8f0cd-d10/journal.jsonl`
  (one `{"type":"result",...}` line per finished agent; build reports and verifier verdicts). A missing
  result line means that agent did not finish; its files may be half-written — check, do not assume.
- If a part looks done, still re-run its script once and confirm the product is byte-identical.

## Then integrate (after all three parts are verified)

1. `git pull --ff-only` the branch and `git pull --ff-only` in `~/overleaf-coling2027` first. If the
   Overleaf copy differs from `deliverables/coling2027` in files you need to touch, merge their edits in
   (Overleaf is the newer text); never overwrite someone else's wording.
2. Paper edits: update every number that changed (grep sections/ and tables/ for the old values);
   update §5 / appendix to state the fourth cell and the 2PL result (sober wording; if the 2PL null is not
   rejected, say so plainly); `\input` the routers appendix subsection in `sections/appendix.tex`, and
   replace the Limitations sentence claiming language-model routers "may do better" with an accurate
   mention of the pilot and its defects. Never print the words "arm" or "control" in the paper.
3. Build in a scratch copy: 0 errors, 0 undefined refs; the main body (before Limitations) must end by
   page 8; check that Figure 1 is still at the top of page 2 and inspect the changed figure visually.
4. Supplement: add the new products and entry scripts to `PRODUCTS` / `ENTRY_SCRIPTS` and the README
   table in `scripts/release/build_coling_supplement.py` (also `scripts/analysis/figures/coling/make_fig_system.py`
   for Figure 1, and whatever the LLM-pilot rescoring needs to be reproducible from the supplement), then
   rebuild `deliverables/coling2027/coling2027_supplement.zip` and run its scan/self-check.
5. Notebook: append ONE new section to `docs/checkpoints/实验笔记.md` (next free § number; Chinese;
   what / why / result / caveats for all three parts). Use an editor tool, not a shell heredoc.
6. Commit on the branch with explicit paths (never `git add -A` / `git add .`), message in Chinese,
   ending with a Co-Authored-By line for yourself. Push the branch to origin. Copy the changed paper files
   (sections/, tables/, figures/, paper.bib if touched) into `~/overleaf-coling2027`, commit, and
   `git push origin main` there.
7. Report to the user in Chinese, short: what changed in the paper, the key numbers, anything that did
   not reproduce, and every prose edit made (so the senior co-author can review).

## Gotchas (all verified on this machine)

- Overleaf push: the remote URL embeds a token — never print it; mask `//...@` in any output. Direct
  connections fail from Beijing; the clone already has `http.proxy=http://127.0.0.1:7897`. Pushes can stall
  or report failure although they landed: after any failure run `git ls-remote origin main` and compare
  with local HEAD before retrying. Retry 2–3 times with a timeout. Overleaf may record our commit as
  "Update on Overleaf."; check content, not message.
- The Overleaf clone checks files out with CRLF; compare with `diff --strip-trailing-cr`.
- Tables in `deliverables/coling2027/tables/` are GENERATED by make_figures.py — edit the generator, never
  the .tex by hand. The generator was updated 2026-10-12 (sources moved to `% sources:` comments); make sure
  regenerated tables keep that.
- Figure PDFs regenerated on this machine differ in bytes but not pixels from committed ones; only commit
  the figures whose content changed (compare with `pdftoppm` renders).
- `docs/cross-ai-audits/` is untracked and not ours: do not commit it.
- Do not reference the senior co-author or the other session by name in commits.
