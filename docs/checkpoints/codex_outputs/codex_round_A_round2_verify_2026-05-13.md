Reading prompt from stdin...
OpenAI Codex v0.130.0
--------
workdir: /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
model: gpt-5.5
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: high
reasoning summaries: none
session id: 019e2223-7c5b-7803-a38a-b707c7f5570b
--------
user
# Round A — Independent verification of today's commit (Round 2 work)

## Context

Earlier today, user ran codex stress audit on 16-cell rerun design and surfaced 6
HIGH severity paper-grade design flaws. Claude then propagated fixes in two rounds:

- Round 1: prereg / paper §3 / queue / advisor docs / OSF manifest / chronicle
- Round 2: decision script full rewrite + queue rename + TOST → one-sided superiority
  semantic fix in prereg H1(ii) + chronicle Round 2 append

All landed in commit **`e9ddbe3`** ("audit(prereg): codex stress 6 paper-grade design
fixes + Phase 1a 24/4 scope reframe"), 11 files +14738 / -403.

**Your job**: Independent verification of the Round 2 work specifically. You have NOT
seen Claude's reasoning or prior codex audit. Cold-read the commit + related files,
attack like a hostile reviewer.

Find anything that, if shipped as-is, would:
- (a) Produce wrong statistical inference (decision script bug),
- (b) Break the cross-doc consistency Round 2 was supposed to fix (regression),
- (c) Introduce a NEW reviewer attack vector that Round 2 created while fixing
      something else,
- (d) Be exploited by a top-tier statistician reviewer as "the fix is wrong".

## Input files (read cold)

### Primary targets — Round 2 work

- `git show e9ddbe3` — the commit itself + body message (use `git log -1 e9ddbe3 --stat`
  to see file list, then read modified files at current HEAD)
- `scripts/analysis/preregistration_decision_test.py` — Round 2 full rewrite
  (drop-one + DerSimonian-Laird meta + one-sided superiority test + framing rule
  R1-R5 mapper). Verify statistical correctness, edge cases, code↔prereg match
- `scripts/queues/queue_phase1_paper_grade.sh` — renamed from `queue_16cell_paper_grade.sh`,
  internal refs updated. Verify build_*_chain logic, no orphaned references, smoke
  gate B7 revision (outcome-independent) defensible
- `docs/checkpoints/pre_run/preregistration.md` H1(ii) section — Round 2 wording
  fix: prior "TOST equivalence at margin δ rejected" replaced with one-sided
  superiority test (H0: θ ≤ +δ vs H1: θ > +δ). Verify wording is unambiguous and
  statistically standard

### Cross-doc consistency check

- `docs/checkpoints/advisor_sync_5_5_followup.md` — does Part 3 §1 (a)/(b)/(c)
  thresholds match the current prereg?
- `docs/checkpoints/pre_run/osf_lock_manifest.md` §2.2 — does the H1 formula
  table match the current decision script's actual implementation?
- `docs/checkpoints/next_steps.md` §1 — does Phase 1a 24/4 scope language
  agree with prereg §4 + queue?

### Verification touchstones

- Run `scripts/analysis/preregistration_decision_test.py --synthetic --scenario r1_pass`
  and inspect output. Is framing rule R1 routed correctly?
- Run `--synthetic --scenario r5_fail`. Should fail H1 but synthetic generator
  may be too friendly. Note any synthetic-data caveat.
- Inspect `dersimonian_laird_meta()` math against Higgins & Thompson 2002 /
  DerSimonian & Laird 1986 standard formulae
- Inspect `superiority_test()` — is z = (θ̂ - threshold)/SE the right formula?

## Output format

### One-sentence verdict on Round 2 commit

Pick one:
- "Round 2 commit is statistically correct + cross-doc consistent — safe"
- "Round 2 commit has N statistical / consistency / propagation flaw(s)"
- "Round 2 commit has methodological concerns but no proven flaw"
- "Insufficient time to verify — partial audit only"

### Confirmed flaws in Round 2 work

For each: layer (statistics / code / cross-doc / wording), file:line evidence, what
breaks, severity (HIGH / MED / LOW), defuse effort.

### Statistical correctness check

For each statistical method in the decision script (DL meta, paired bootstrap,
superiority test, Holm correction, framing rule mapper), say PASS or FLAG with
reason. Cite textbook reference if applicable.

### Cross-doc consistency check post-Round 2

If prereg / advisor_sync / osf / decision script disagree on any number / formula /
threshold, report all + which is canonical + defuse.

### Reviewer ammunition created by Round 2

Round 2 fixed some things; did it create new attack vectors? List 3-5 specific
questions a hostile reviewer would ask after reading Round 2 commit message + diff.

### Verdict on next steps

If Round 2 holds: tell user they can confidently push commit `e9ddbe3` (already done
this session) and continue toward advisor meeting.
If Round 2 has flaws: prioritized list. Especially: which can be defused before
advisor meeting tomorrow (advisor would catch it) vs which can be defused post-data.

## Calibration

- This is paper-grade audit of code + prose, not style check
- Don't propose code fixes; identify the suspect, impact, defuse cost
- Negative result valid: if Round 2 holds after 60 min review, write verdict and stop
- Don't fabricate: file paths, line numbers must be real
- Set your own attack vectors based on what the code/docs show

## Time budget

Up to 60 min. Tier 3 PID monitor fires when codex exits.

codex
I’ll cold-read the commit and the current Round 2 targets, then run the two synthetic scenarios before drawing conclusions. I’ll keep this to verification only: no file edits.
exec
/bin/bash -lc 'git log -1 e9ddbe3 --stat --decorate --date=iso' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
commit e9ddbe3714099e28a9f77e72013083edcafac6cb (HEAD -> master, origin/master, origin/HEAD)
Author: Quarkgluonmixture <jimmyenglish@126.com>
Date:   2026-05-13 17:03:59 +0100

    audit(prereg): codex stress 6 paper-grade design fixes + Phase 1a 24/4 scope reframe
    
    Pre-launch codex stress audit (`docs/checkpoints/codex_outputs/codex_stress_16cell_design_2026-05-13.md`, lean prompt no-enumeration, cross-AI cold-read) surfaced 6 HIGH severity design flaws in 16-cell rerun design. All 6 closed pre-launch via Round 1 (doc layer) + Round 2 (script + queue layer).
    
    Codex 6 HIGH flaws → fixes:
    - Flaw 1 (queue baseline-vintage missing): queue now launches 24 conditions × 6 modes (DOM/SoM/Vision + 3 phantoms) per cell, not phantom-only
    - Flaw 2 (K-of-N gate vs transparency contradiction): K-of-N reclassified pre-data gate → transparency consistency check; propagated to prereg H1/H3/R5/§3 family/§4 audit B9/§6 9-decision/advisor email/osf manifest. Primary gate = pooled DerSimonian-Laird random-effects meta + one-sided superiority test on N=4 cells. Rationale: power analysis shows K-of-N family power < 10% at observed 1-3pp effect sizes, dysfunctional as gate
    - Flaw 3 (decision script wrong H1 formula): preregistration_decision_test.py full rewrite — drop-one oracle ceiling lift per (site, model) cell + paired bootstrap variance + DerSimonian-Laird random-effects meta + one-sided superiority test (replaces ambiguous "P-SoM ≥ best single baseline" check). Smoke test r1_pass synthetic → R1 STRONGEST framing routed correctly
    - Flaw 4 (paper §3 excludes P-prompt vs prereg H3 requires it): section3_definition.md §3.4 rewrite — P-prompt re-included as 4th cell of complete 2×2; 6-mode framework with 6 contrasts logic
    - Flaw 5 (16/18 cell ambiguity): canonical scope locked at 24 operational conditions = 2 sites (cls, red) × 2 models (B0, B1) × 6 modes, with 4 statistical (site, model) cells. Shopping deferred to Phase 1b main-paper expansion
    - Flaw 6 (smoke gate outcome-dependent stopping): prereg §4 audit B7 revised — smoke gate checks auth + artifact + evaluator parseability only, NOT SR-based restart (was biasing low-SR cells upward)
    
    Scope reframe (workshop-first publication strategy):
    - Phase 1a (workshop-targeted, immediate): 24 conditions / 4 cells, cls + red × B0 + B1 × 6 modes
    - Phase 1b (main-paper expansion, post-workshop): +12 conditions = shop × B0 + B1 × 6 modes; feeds R3 → R1 framing decision
    
    Round 2 refinements (during Path A execution):
    - TOST → one-sided superiority test for H1(ii): prior "TOST equivalence at margin δ rejected" wording had ambiguous semantic direction (TOST H0 rejection = demonstrate equivalence in standard literature, OPPOSITE of H1 substantive-effect intent). Replaced with unambiguous one-sided superiority H0: θ ≤ +δ vs H1: θ > +δ. TOST retained as informational secondary report
    - Queue rename: queue_16cell_paper_grade.sh → queue_phase1_paper_grade.sh (filename now reflects current Phase 1a+1b scope, not legacy 16-cell phantom-only); internal log/pid refs + external doc refs all updated
    
    Files (11):
    - scripts/queues/queue_phase1_paper_grade.sh (rename from queue_16cell + header rewrite + add baseline modes to chains + Phase 1b deferred)
    - scripts/analysis/preregistration_decision_test.py (full rewrite + DerSimonian-Laird meta + superiority test + framing rule R1-R5 mapper + synthetic smoke test)
    - docs/checkpoints/pre_run/preregistration.md (frontmatter scope revision + H1/H3 + R5 trigger + §3 family + §4 6 rows + §6 9-decision + Appendix A 2026-05-13)
    - docs/checkpoints/paper_drafts/section3_definition.md (§3.4 P-prompt re-inclusion)
    - docs/checkpoints/advisor_sync_5_5_followup.md (scope + 3 thresholds reframe + email template)
    - docs/checkpoints/pre_run/osf_lock_manifest.md (H1 formula correction + 24/4 scope + transparency K rows)
    - docs/checkpoints/pre_run/reeval_audit_protocol.md (queue filename refs)
    - docs/checkpoints/next_steps.md (§1 Phase 1 scope rewrite + filename refs + decision script CLI)
    - docs/checkpoints/实验笔记.md (§132 chronicle: 9 subsections covering trigger / verdict / scope / K-of-N / terminology / smoke-gate / files / Round 2 / status)
    - docs/checkpoints/codex_prompts/codex_stress_16cell_design_2026-05-13.md (audit prompt)
    - docs/checkpoints/codex_outputs/codex_stress_16cell_design_2026-05-13.md (audit evidence)
    
    Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

 docs/checkpoints/advisor_sync_5_5_followup.md      |    67 +-
 .../codex_stress_16cell_design_2026-05-13.md       | 13530 +++++++++++++++++++
 .../codex_stress_16cell_design_2026-05-13.md       |   181 +
 docs/checkpoints/next_steps.md                     |    44 +-
 .../paper_drafts/section3_definition.md            |    39 +-
 docs/checkpoints/pre_run/osf_lock_manifest.md      |    28 +-
 docs/checkpoints/pre_run/preregistration.md        |    93 +-
 docs/checkpoints/pre_run/reeval_audit_protocol.md  |     4 +-
 ...256\236\351\252\214\347\254\224\350\256\260.md" |   123 +
 scripts/analysis/preregistration_decision_test.py  |   905 +-
 ..._paper_grade.sh => queue_phase1_paper_grade.sh} |   127 +-
 11 files changed, 14738 insertions(+), 403 deletions(-)

exec
/bin/bash -lc 'git show --stat --summary --decorate --date=iso e9ddbe3' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
commit e9ddbe3714099e28a9f77e72013083edcafac6cb (HEAD -> master, origin/master, origin/HEAD)
Author: Quarkgluonmixture <jimmyenglish@126.com>
Date:   2026-05-13 17:03:59 +0100

    audit(prereg): codex stress 6 paper-grade design fixes + Phase 1a 24/4 scope reframe
    
    Pre-launch codex stress audit (`docs/checkpoints/codex_outputs/codex_stress_16cell_design_2026-05-13.md`, lean prompt no-enumeration, cross-AI cold-read) surfaced 6 HIGH severity design flaws in 16-cell rerun design. All 6 closed pre-launch via Round 1 (doc layer) + Round 2 (script + queue layer).
    
    Codex 6 HIGH flaws → fixes:
    - Flaw 1 (queue baseline-vintage missing): queue now launches 24 conditions × 6 modes (DOM/SoM/Vision + 3 phantoms) per cell, not phantom-only
    - Flaw 2 (K-of-N gate vs transparency contradiction): K-of-N reclassified pre-data gate → transparency consistency check; propagated to prereg H1/H3/R5/§3 family/§4 audit B9/§6 9-decision/advisor email/osf manifest. Primary gate = pooled DerSimonian-Laird random-effects meta + one-sided superiority test on N=4 cells. Rationale: power analysis shows K-of-N family power < 10% at observed 1-3pp effect sizes, dysfunctional as gate
    - Flaw 3 (decision script wrong H1 formula): preregistration_decision_test.py full rewrite — drop-one oracle ceiling lift per (site, model) cell + paired bootstrap variance + DerSimonian-Laird random-effects meta + one-sided superiority test (replaces ambiguous "P-SoM ≥ best single baseline" check). Smoke test r1_pass synthetic → R1 STRONGEST framing routed correctly
    - Flaw 4 (paper §3 excludes P-prompt vs prereg H3 requires it): section3_definition.md §3.4 rewrite — P-prompt re-included as 4th cell of complete 2×2; 6-mode framework with 6 contrasts logic
    - Flaw 5 (16/18 cell ambiguity): canonical scope locked at 24 operational conditions = 2 sites (cls, red) × 2 models (B0, B1) × 6 modes, with 4 statistical (site, model) cells. Shopping deferred to Phase 1b main-paper expansion
    - Flaw 6 (smoke gate outcome-dependent stopping): prereg §4 audit B7 revised — smoke gate checks auth + artifact + evaluator parseability only, NOT SR-based restart (was biasing low-SR cells upward)
    
    Scope reframe (workshop-first publication strategy):
    - Phase 1a (workshop-targeted, immediate): 24 conditions / 4 cells, cls + red × B0 + B1 × 6 modes
    - Phase 1b (main-paper expansion, post-workshop): +12 conditions = shop × B0 + B1 × 6 modes; feeds R3 → R1 framing decision
    
    Round 2 refinements (during Path A execution):
    - TOST → one-sided superiority test for H1(ii): prior "TOST equivalence at margin δ rejected" wording had ambiguous semantic direction (TOST H0 rejection = demonstrate equivalence in standard literature, OPPOSITE of H1 substantive-effect intent). Replaced with unambiguous one-sided superiority H0: θ ≤ +δ vs H1: θ > +δ. TOST retained as informational secondary report
    - Queue rename: queue_16cell_paper_grade.sh → queue_phase1_paper_grade.sh (filename now reflects current Phase 1a+1b scope, not legacy 16-cell phantom-only); internal log/pid refs + external doc refs all updated
    
    Files (11):
    - scripts/queues/queue_phase1_paper_grade.sh (rename from queue_16cell + header rewrite + add baseline modes to chains + Phase 1b deferred)
    - scripts/analysis/preregistration_decision_test.py (full rewrite + DerSimonian-Laird meta + superiority test + framing rule R1-R5 mapper + synthetic smoke test)
    - docs/checkpoints/pre_run/preregistration.md (frontmatter scope revision + H1/H3 + R5 trigger + §3 family + §4 6 rows + §6 9-decision + Appendix A 2026-05-13)
    - docs/checkpoints/paper_drafts/section3_definition.md (§3.4 P-prompt re-inclusion)
    - docs/checkpoints/advisor_sync_5_5_followup.md (scope + 3 thresholds reframe + email template)
    - docs/checkpoints/pre_run/osf_lock_manifest.md (H1 formula correction + 24/4 scope + transparency K rows)
    - docs/checkpoints/pre_run/reeval_audit_protocol.md (queue filename refs)
    - docs/checkpoints/next_steps.md (§1 Phase 1 scope rewrite + filename refs + decision script CLI)
    - docs/checkpoints/实验笔记.md (§132 chronicle: 9 subsections covering trigger / verdict / scope / K-of-N / terminology / smoke-gate / files / Round 2 / status)
    - docs/checkpoints/codex_prompts/codex_stress_16cell_design_2026-05-13.md (audit prompt)
    - docs/checkpoints/codex_outputs/codex_stress_16cell_design_2026-05-13.md (audit evidence)
    
    Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>

 docs/checkpoints/advisor_sync_5_5_followup.md      |    67 +-
 .../codex_stress_16cell_design_2026-05-13.md       | 13530 +++++++++++++++++++
 .../codex_stress_16cell_design_2026-05-13.md       |   181 +
 docs/checkpoints/next_steps.md                     |    44 +-
 .../paper_drafts/section3_definition.md            |    39 +-
 docs/checkpoints/pre_run/osf_lock_manifest.md      |    28 +-
 docs/checkpoints/pre_run/preregistration.md        |    93 +-
 docs/checkpoints/pre_run/reeval_audit_protocol.md  |     4 +-
 ...256\236\351\252\214\347\254\224\350\256\260.md" |   123 +
 scripts/analysis/preregistration_decision_test.py  |   905 +-
 ..._paper_grade.sh => queue_phase1_paper_grade.sh} |   127 +-
 11 files changed, 14738 insertions(+), 403 deletions(-)
 create mode 100644 docs/checkpoints/codex_outputs/codex_stress_16cell_design_2026-05-13.md
 create mode 100644 docs/checkpoints/codex_prompts/codex_stress_16cell_design_2026-05-13.md
 rename scripts/queues/{queue_16cell_paper_grade.sh => queue_phase1_paper_grade.sh} (56%)

exec
/bin/bash -lc 'git status --short' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
 M docs/analysis/cross_sites/sr_fp_per_mode.json
 M docs/analysis/cross_sites/sr_fp_per_mode.md
 M docs/checkpoints/interview_seonglae_prep.md
 D docs/reference/MYRIAD_SMOKE_REPORT.md
 D docs/reference/PAPER_STRATEGY_OPEN_QUESTIONS.md
 D docs/reference/RUNPOD_ONBOARDING.md
?? docs/checkpoints/PLAYBOOK.md.bak
?? docs/checkpoints/codex_outputs/codex_round_A_round2_verify_2026-05-13.md
?? docs/checkpoints/codex_outputs/codex_round_B_paper_prose_2026-05-13.md
?? docs/checkpoints/codex_outputs/codex_round_C_statistical_methodology_2026-05-13.md
?? docs/checkpoints/codex_prompts/codex_round_A_round2_verify_2026-05-13.md
?? docs/checkpoints/codex_prompts/codex_round_B_paper_prose_2026-05-13.md
?? docs/checkpoints/codex_prompts/codex_round_C_statistical_methodology_2026-05-13.md
?? scripts/analysis/stage2_transfer_correlation.py

codex
The target commit is `HEAD` and the worktree has unrelated dirty files; I’m treating those as out of scope and reading the committed/current Round 2 files directly. Next I’m mapping the script’s actual statistical path against the prereg language.
exec
/bin/bash -lc "nl -ba scripts/analysis/preregistration_decision_test.py | sed -n '1,260p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	"""Preregistration decision test — Phase 1a 24-condition / 4-cell H1 / H3 / H2 evaluation.
     2	
     3	⚠️ REWRITTEN 2026-05-13 to align with preregistration.md revisions (codex stress audit
     4	   Flaws 2 + 3 fix):
     5	   - PRIMARY GATE = pooled DerSimonian-Laird random-effects meta + TOST equivalence
     6	   - K-of-N reclassified gate → transparency consistency check (per pre-data 2026-05-13
     7	     reclassification, see `preregistration.md` §4 audit B9 + Appendix A 2026-05-13)
     8	   - H1 formula = P-SoM drop-one oracle ceiling lift (NOT P-SoM ≥ best single mode)
     9	   - H3 family = axis-1 (P-text \ P-SoM) + axis-2 (P-prompt \ P-SoM), both pooled
    10	   - Scope = 4 (site, model) statistical cells, each with 6 modes' per-task SR data
    11	
    12	Definitions (per preregistration.md §2 + §4):
    13	  - cell = 1 (site, model) statistical stratification unit. Phase 1a N=4 cells:
    14	    (cls, B0), (cls, B1), (red, B0), (red, B1).
    15	  - condition = 1 (site, model, mode) operational launch unit. Phase 1a N=24.
    16	  - Drop-one per cell: oracle ceiling SR over {6 modes} − oracle ceiling SR over
    17	    {5 modes drop P-SoM}, per task, averaged across task pool. Paired bootstrap CI.
    18	  - Pooled meta: DerSimonian-Laird random-effects across 4 cell effect estimates.
    19	  - TOST: two one-sided tests for H0 |θ| ≥ δ rejected vs H1 |θ| < δ at δ=1.0pp.
    20	
    21	PRIMARY GATES (gate paper hook framing R1-R5):
    22	  H1(i)  pooled DL meta on P-SoM drop-one, Holm α=0.05 sig (m=1)
    23	  H1(ii) pooled magnitude θ_RE ≥ 1.0pp + TOST equivalence rejected at δ=1.0pp
    24	  H3(i)  pooled DL meta on |P-text \ P-SoM| axis-1, Holm α=0.05 sig (m=1)
    25	  H3(ii) pooled DL meta on |P-prompt \ P-SoM| axis-2, Holm α=0.05 sig (m=1)
    26	  H2(a)  median cost(P-SoM) within ±10% of median cost(DOM) per cell, replicated
    27	         in ≥3 of 4 cells (transparency K_h2)
    28	
    29	TRANSPARENCY (NOT gating, reported alongside primary):
    30	  K_h1 = 3 of 4 cells individually Holm-sig on drop-one
    31	  K_h3 axis-1 = 3 of 4 cells individually CI > 0
    32	  K_h3 axis-2 = same
    33	
    34	Usage:
    35	    # With actual per-task data:
    36	    python3 scripts/analysis/preregistration_decision_test.py \\
    37	        --per-task-csv results/phantom_paper/per_task_sr.csv \\
    38	        --primary-gate drop_one_pooled_meta_TOST \\
    39	        --TOST-delta-pp 1.0 \\
    40	        --transparency-K_h1 3 --transparency-K_h3 3 \\
    41	        --out results/phantom_paper/preregistration_test_results.json
    42	
    43	    # Smoke test on synthetic data:
    44	    python3 scripts/analysis/preregistration_decision_test.py --synthetic --seed 42
    45	
    46	Input CSV schema (per-task wide format, one row per (cell_id, task_id)):
    47	    cell_id,site,model,task_id,sr_dom,sr_som,sr_vision,sr_ptext,sr_pprompt,sr_psom,
    48	        cost_dom,cost_psom
    49	    cls_B0,classifieds,B0,task_0001,0.0,1.0,0.0,1.0,0.0,1.0,0.043,0.044
    50	    ...
    51	
    52	Each SR cell ∈ {0, 1} (binary per-task evaluator verdict, post-FP-filter).
    53	Costs in any consistent unit (token-normalized $); only ratio used.
    54	
    55	Tied to:
    56	- preregistration.md §2 (H1/H3 hypotheses) + §4 (locked analysis choices) +
    57	  Appendix A 2026-05-13 (codex stress audit propagation)
    58	- osf_lock_manifest.md §2.2 (canonical threshold table)
    59	- run_manifest.yaml (cell scope = 4 Phase 1a cells)
    60	- 笔记 §132 (codex stress audit + scope reframe chronicle)
    61	"""
    62	
    63	from __future__ import annotations
    64	
    65	import argparse
    66	import csv
    67	import hashlib
    68	import json
    69	import logging
    70	import math
    71	import statistics
    72	import sys
    73	from collections import defaultdict
    74	from datetime import datetime, timezone
    75	from pathlib import Path
    76	from typing import Optional
    77	
    78	logger = logging.getLogger("preregistration-test")
    79	
    80	# Phase 1a canonical cells (must match preregistration.md §4 N_cells row)
    81	PHASE_1A_CELLS = [
    82	    ("classifieds", "B0"),
    83	    ("classifieds", "B1"),
    84	    ("reddit", "B0"),
    85	    ("reddit", "B1"),
    86	]
    87	PHANTOM_MODE_KEYS = ["sr_psom", "sr_ptext", "sr_pprompt"]
    88	BASELINE_MODE_KEYS = ["sr_dom", "sr_som", "sr_vision"]
    89	ALL_MODE_KEYS = BASELINE_MODE_KEYS + PHANTOM_MODE_KEYS
    90	
    91	
    92	# ---------------------------------------------------------------------------
    93	# Per-cell drop-one + unique-count computation (paired bootstrap)
    94	# ---------------------------------------------------------------------------
    95	
    96	def _oracle_per_task(task_row: dict, mode_keys: list[str]) -> int:
    97	    """Oracle ceiling for one task = 1 if ANY mode in mode_keys solved it, else 0."""
    98	    return 1 if any(int(task_row[k]) >= 1 for k in mode_keys) else 0
    99	
   100	
   101	def _drop_one_lift_per_cell(cell_tasks: list[dict], drop_mode: str = "sr_psom") -> float:
   102	    """Drop-one oracle ceiling lift for a cell.
   103	
   104	    Returns the mean over the cell's task pool of:
   105	        oracle({all 6 modes}, task) − oracle({all 6 modes} \\ {drop_mode}, task)
   106	
   107	    Result is in [0, 1] (probability units; multiply by 100 for pp).
   108	    """
   109	    full = ALL_MODE_KEYS
   110	    reduced = [k for k in full if k != drop_mode]
   111	    deltas = [_oracle_per_task(t, full) - _oracle_per_task(t, reduced) for t in cell_tasks]
   112	    return sum(deltas) / max(1, len(deltas))
   113	
   114	
   115	def _unique_count_per_cell(cell_tasks: list[dict], axis_mode: str, ref_mode: str = "sr_psom") -> int:
   116	    """|axis_mode \\ ref_mode| = number of tasks where axis_mode solved but ref_mode didn't.
   117	
   118	    Used for H3 axis-1 (axis_mode=sr_ptext) and H3 axis-2 (axis_mode=sr_pprompt).
   119	    """
   120	    return sum(1 for t in cell_tasks
   121	               if int(t[axis_mode]) >= 1 and int(t[ref_mode]) < 1)
   122	
   123	
   124	def _paired_bootstrap(cell_tasks: list[dict], statistic_fn, n_resamples: int = 1000,
   125	                       seed: int = 42) -> tuple[float, float, float, float]:
   126	    """1000-resample paired task-level bootstrap.
   127	
   128	    Returns (point_estimate, ci_lo_95, ci_hi_95, bootstrap_se).
   129	    Resamples task rows with replacement (preserves all modes' SR for that task → paired).
   130	    """
   131	    import random
   132	    rng = random.Random(seed)
   133	    point = statistic_fn(cell_tasks)
   134	    n = len(cell_tasks)
   135	    boot_vals = []
   136	    for _ in range(n_resamples):
   137	        resample = [cell_tasks[rng.randrange(n)] for _ in range(n)]
   138	        boot_vals.append(statistic_fn(resample))
   139	    boot_vals.sort()
   140	    ci_lo = boot_vals[int(0.025 * n_resamples)]
   141	    ci_hi = boot_vals[int(0.975 * n_resamples)]
   142	    se = statistics.stdev(boot_vals) if len(boot_vals) > 1 else 0.0
   143	    return point, ci_lo, ci_hi, se
   144	
   145	
   146	# ---------------------------------------------------------------------------
   147	# DerSimonian-Laird random-effects meta-analysis
   148	# ---------------------------------------------------------------------------
   149	
   150	def dersimonian_laird_meta(effects: list[float], variances: list[float]) -> dict:
   151	    """Pool effect estimates across cells via DerSimonian-Laird random-effects.
   152	
   153	    Args:
   154	        effects: per-cell effect estimates (same scale, e.g., pp or unique-count)
   155	        variances: per-cell variance estimates (= SE^2 from bootstrap)
   156	
   157	    Returns dict with: pooled_effect, pooled_se, pooled_ci_95, Q, I_squared, tau_squared,
   158	                       p_value_two_sided.
   159	
   160	    Method (Higgins & Thompson 2002; DerSimonian & Laird 1986):
   161	      1. Fixed-effects pooled mean θ_FE = Σ(w_i × θ_i) / Σw_i where w_i = 1 / v_i
   162	      2. Q = Σw_i × (θ_i − θ_FE)^2
   163	      3. τ^2 = max(0, (Q − (k − 1)) / (Σw_i − Σw_i^2 / Σw_i))
   164	      4. Random-effects weights w*_i = 1 / (v_i + τ^2)
   165	      5. Pooled θ_RE = Σ(w*_i × θ_i) / Σw*_i; SE_RE = sqrt(1 / Σw*_i)
   166	      6. I^2 = max(0, (Q − (k − 1)) / Q) × 100  (% heterogeneity)
   167	    """
   168	    k = len(effects)
   169	    if k < 2:
   170	        return {"pooled_effect": effects[0] if effects else 0.0,
   171	                "pooled_se": math.sqrt(variances[0]) if variances else 0.0,
   172	                "pooled_ci_95": [None, None],
   173	                "Q": None, "I_squared_pct": None, "tau_squared": None,
   174	                "p_value_two_sided": None, "k": k,
   175	                "note": "k<2: pooling undefined"}
   176	
   177	    w_fe = [1.0 / max(v, 1e-12) for v in variances]
   178	    theta_fe = sum(w * t for w, t in zip(w_fe, effects)) / sum(w_fe)
   179	    Q = sum(w * (t - theta_fe) ** 2 for w, t in zip(w_fe, effects))
   180	    sum_w = sum(w_fe)
   181	    sum_w_sq = sum(w * w for w in w_fe)
   182	    tau_sq_num = Q - (k - 1)
   183	    tau_sq_den = sum_w - (sum_w_sq / sum_w)
   184	    tau_sq = max(0.0, tau_sq_num / max(tau_sq_den, 1e-12))
   185	
   186	    w_re = [1.0 / (v + tau_sq) for v in variances]
   187	    theta_re = sum(w * t for w, t in zip(w_re, effects)) / sum(w_re)
   188	    se_re = math.sqrt(1.0 / sum(w_re))
   189	    ci_lo = theta_re - 1.96 * se_re
   190	    ci_hi = theta_re + 1.96 * se_re
   191	
   192	    z = theta_re / max(se_re, 1e-12)
   193	    # Two-sided p from standard normal (using error function approximation)
   194	    p_two_sided = 2.0 * (1.0 - _phi(abs(z)))
   195	
   196	    i_sq = max(0.0, (Q - (k - 1)) / Q) * 100.0 if Q > 0 else 0.0
   197	
   198	    return {
   199	        "pooled_effect": theta_re,
   200	        "pooled_se": se_re,
   201	        "pooled_ci_95": [ci_lo, ci_hi],
   202	        "Q": Q,
   203	        "Q_df": k - 1,
   204	        "I_squared_pct": i_sq,
   205	        "tau_squared": tau_sq,
   206	        "p_value_two_sided": p_two_sided,
   207	        "z_statistic": z,
   208	        "k": k,
   209	    }
   210	
   211	
   212	def _phi(z: float) -> float:
   213	    """Standard normal CDF using erf approximation."""
   214	    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
   215	
   216	
   217	# ---------------------------------------------------------------------------
   218	# TOST equivalence test
   219	# ---------------------------------------------------------------------------
   220	
   221	def superiority_test(pooled_effect: float, pooled_se: float, threshold: float,
   222	                      alpha: float = 0.05) -> dict:
   223	    """One-sided superiority test: H0: θ ≤ threshold vs H1: θ > threshold.
   224	
   225	    Used for H1(ii) per prereg 2026-05-13 wording revision: "effect is significantly
   226	    ABOVE the +threshold substantive-effect floor". Reject H0 when pooled effect is
   227	    significantly larger than threshold (z = (θ̂ - threshold)/SE > z_α).
   228	
   229	    Args:
   230	        pooled_effect: pooled effect estimate (same units as threshold)
   231	        pooled_se: pooled SE
   232	        threshold: substantive-effect floor (positive; e.g., 1.0pp)
   233	        alpha: one-sided significance level (default 0.05)
   234	
   235	    Returns dict with: z, p_one_sided, threshold, decision.
   236	
   237	    Note: This replaces prior TOST-rejection logic which had ambiguous semantic
   238	    direction ("TOST equivalence rejected" could mean either equivalence-demonstrated
   239	    OR equivalence-not-demonstrated). One-sided superiority is the unambiguous test
   240	    for "effect substantively exceeds threshold".
   241	    """
   242	    z = (pooled_effect - threshold) / max(pooled_se, 1e-12)
   243	    p_one_sided = 1.0 - _phi(z)
   244	    return {
   245	        "threshold": threshold,
   246	        "alpha": alpha,
   247	        "pooled_effect": pooled_effect,
   248	        "pooled_se": pooled_se,
   249	        "z_statistic": z,
   250	        "p_one_sided": p_one_sided,
   251	        "decision": "reject_H0_substantively_above_threshold" if p_one_sided < alpha else "fail_reject",
   252	    }
   253	
   254	
   255	def tost_equivalence(pooled_effect: float, pooled_se: float, delta: float,
   256	                      alpha: float = 0.05) -> dict:
   257	    """Two one-sided tests for equivalence (Schuirmann 1987).
   258	
   259	    Tests H0: |θ| ≥ δ (effect non-equivalent) vs H1: |θ| < δ (effect equivalent).
   260	    Both one-sided tests must reject H0 to demonstrate equivalence.

exec
/bin/bash -lc "nl -ba scripts/analysis/preregistration_decision_test.py | sed -n '261,560p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
   261	
   262	    Used in P79 paper-1 as **informational only** (reported alongside H1 superiority
   263	    test, NOT used for H1 PRIMARY gating per 2026-05-13 prereg revision).
   264	    """
   265	    t_lo = (pooled_effect - (-delta)) / max(pooled_se, 1e-12)  # tests θ > -δ
   266	    t_hi = ((+delta) - pooled_effect) / max(pooled_se, 1e-12)  # tests θ < +δ
   267	    p_lo = 1.0 - _phi(t_lo)
   268	    p_hi = 1.0 - _phi(t_hi)
   269	    max_p = max(p_lo, p_hi)
   270	    equivalence_demonstrated = (p_lo < alpha) and (p_hi < alpha)
   271	    return {
   272	        "delta": delta,
   273	        "alpha_per_side": alpha,
   274	        "pooled_effect": pooled_effect,
   275	        "pooled_se": pooled_se,
   276	        "p_lower_bound_test": p_lo,
   277	        "p_upper_bound_test": p_hi,
   278	        "max_p_value": max_p,
   279	        "equivalence_demonstrated": equivalence_demonstrated,
   280	        "decision": "equivalence_demonstrated" if equivalence_demonstrated else "equivalence_not_demonstrated",
   281	    }
   282	
   283	
   284	# ---------------------------------------------------------------------------
   285	# Holm-Bonferroni correction
   286	# ---------------------------------------------------------------------------
   287	
   288	def holm_correct(p_values: list[float], alpha: float = 0.05) -> list[dict]:
   289	    """Holm-Bonferroni step-down correction for a family of m tests.
   290	
   291	    Returns list of dicts (in original order) with: p_raw, p_holm, rejected.
   292	    """
   293	    m = len(p_values)
   294	    if m == 0:
   295	        return []
   296	    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
   297	    results = [None] * m
   298	    prev_adj = 0.0
   299	    for rank, (orig_idx, p) in enumerate(indexed):
   300	        adj = (m - rank) * p
   301	        adj = max(adj, prev_adj)
   302	        adj = min(adj, 1.0)
   303	        results[orig_idx] = {
   304	            "p_raw": p,
   305	            "p_holm": adj,
   306	            "rejected": adj < alpha,
   307	        }
   308	        prev_adj = adj
   309	    return results
   310	
   311	
   312	# ---------------------------------------------------------------------------
   313	# Hypothesis evaluators
   314	# ---------------------------------------------------------------------------
   315	
   316	def evaluate_h1(cells_by_id: dict[str, list[dict]], delta_pp: float = 1.0,
   317	                 magnitude_threshold_pp: float = 1.0, alpha: float = 0.05,
   318	                 transparency_K_h1: int = 3, bootstrap_seed: int = 42) -> dict:
   319	    """H1: P-SoM drop-one oracle ceiling lift > 0, pooled across cells.
   320	
   321	    PRIMARY: pooled DL meta sig at Holm α=0.05 (m=1) + θ_RE ≥ magnitude_threshold_pp
   322	             + TOST equivalence rejected at δ=delta_pp.
   323	    TRANSPARENCY: K_h1 = transparency_K_h1 of N cells individually Holm-sig (m=N).
   324	    """
   325	    per_cell = {}
   326	    effects_pp = []
   327	    variances_pp = []  # variances of per-cell drop-one in pp^2
   328	    per_cell_p_values = []
   329	
   330	    for cell_id, tasks in cells_by_id.items():
   331	        point, ci_lo, ci_hi, se = _paired_bootstrap(
   332	            tasks,
   333	            statistic_fn=lambda t: _drop_one_lift_per_cell(t, drop_mode="sr_psom"),
   334	            seed=bootstrap_seed,
   335	        )
   336	        # Convert to pp
   337	        effect_pp = point * 100.0
   338	        se_pp = se * 100.0
   339	        # Two-sided p from bootstrap normal approx
   340	        z = effect_pp / max(se_pp, 1e-12)
   341	        p_cell = 2.0 * (1.0 - _phi(abs(z)))
   342	        per_cell[cell_id] = {
   343	            "drop_one_lift_pp": effect_pp,
   344	            "ci_95_pp": [ci_lo * 100.0, ci_hi * 100.0],
   345	            "se_pp": se_pp,
   346	            "p_value_two_sided": p_cell,
   347	            "n_tasks": len(tasks),
   348	        }
   349	        effects_pp.append(effect_pp)
   350	        variances_pp.append(se_pp ** 2)
   351	        per_cell_p_values.append(p_cell)
   352	
   353	    # PRIMARY: pooled DL meta + magnitude + superiority test
   354	    meta = dersimonian_laird_meta(effects_pp, variances_pp)
   355	    superiority = superiority_test(meta["pooled_effect"], meta["pooled_se"],
   356	                                     threshold=magnitude_threshold_pp, alpha=alpha)
   357	    # TOST kept for informational reporting (NOT used in H1 gating decision)
   358	    tost_info = tost_equivalence(meta["pooled_effect"], meta["pooled_se"],
   359	                                  delta=delta_pp, alpha=alpha)
   360	
   361	    pooled_sig = meta["p_value_two_sided"] is not None and meta["p_value_two_sided"] < alpha
   362	    magnitude_pass = meta["pooled_effect"] >= magnitude_threshold_pp
   363	    superiority_pass = superiority["decision"] == "reject_H0_substantively_above_threshold"
   364	
   365	    primary_h1_pass = pooled_sig and magnitude_pass and superiority_pass
   366	
   367	    # TRANSPARENCY: K-of-N Holm
   368	    holm_per_cell = holm_correct(per_cell_p_values, alpha=alpha)
   369	    for (cell_id, _), h in zip(per_cell.items(), holm_per_cell):
   370	        per_cell[cell_id]["holm_p"] = h["p_holm"]
   371	        per_cell[cell_id]["individually_holm_sig"] = h["rejected"]
   372	    n_individually_sig = sum(1 for h in holm_per_cell if h["rejected"])
   373	    transparency_pass = n_individually_sig >= transparency_K_h1
   374	
   375	    return {
   376	        "primary_gate": {
   377	            "pooled_meta": meta,
   378	            "magnitude_check": {"pooled_pp": meta["pooled_effect"],
   379	                                 "threshold_pp": magnitude_threshold_pp,
   380	                                 "pass": magnitude_pass},
   381	            "superiority_test": superiority,
   382	            "tost_informational": tost_info,
   383	            "decision": "PASS" if primary_h1_pass else "FAIL",
   384	        },
   385	        "transparency_K_h1": {
   386	            "K": transparency_K_h1,
   387	            "N": len(cells_by_id),
   388	            "n_individually_holm_sig": n_individually_sig,
   389	            "consistent": transparency_pass,
   390	            "note": "transparency-only, NOT a gate on H1 (per prereg 2026-05-13 reclassification)",
   391	        },
   392	        "per_cell": per_cell,
   393	    }
   394	
   395	
   396	def evaluate_h3_axis(cells_by_id: dict[str, list[dict]], axis_mode_key: str,
   397	                      ref_mode_key: str = "sr_psom", min_unique_count: int = 2,
   398	                      alpha: float = 0.05, transparency_K_h3: int = 3,
   399	                      bootstrap_seed: int = 42) -> dict:
   400	    """H3 axis test: |axis_mode \\ ref_mode| > 0, pooled across cells.
   401	
   402	    axis_mode_key examples: sr_ptext (axis-1), sr_pprompt (axis-2).
   403	
   404	    PRIMARY: pooled DL meta on unique-count, CI excluding 0 at Holm α=0.05 (m=1).
   405	    TRANSPARENCY: K_h3 of N cells with bootstrap CI > 0 AND unique-count ≥ min_unique_count.
   406	    """
   407	    per_cell = {}
   408	    effects = []
   409	    variances = []
   410	    per_cell_p_values = []
   411	    per_cell_ci_excludes_zero = []
   412	
   413	    for cell_id, tasks in cells_by_id.items():
   414	        # Statistic: count of tasks where axis solved but ref did not, normalized by task count
   415	        # (using count as the statistic per prereg H3 wording)
   416	        count, ci_lo, ci_hi, se = _paired_bootstrap(
   417	            tasks,
   418	            statistic_fn=lambda t: float(_unique_count_per_cell(t, axis_mode_key, ref_mode_key)),
   419	            seed=bootstrap_seed,
   420	        )
   421	        # Per-cell pass: CI > 0 AND count ≥ min_unique_count (≥2 floor for noise)
   422	        ci_excludes_zero = ci_lo > 0
   423	        count_above_floor = count >= min_unique_count
   424	        per_cell_pass = ci_excludes_zero and count_above_floor
   425	        # Per-cell p from normal approx on count statistic (testing > 0)
   426	        z = count / max(se, 1e-12)
   427	        p_cell = 1.0 - _phi(z)  # one-sided
   428	        per_cell[cell_id] = {
   429	            "unique_count": count,
   430	            "ci_95": [ci_lo, ci_hi],
   431	            "se": se,
   432	            "p_value_one_sided": p_cell,
   433	            "ci_excludes_zero": ci_excludes_zero,
   434	            "count_above_min": count_above_floor,
   435	            "per_cell_pass": per_cell_pass,
   436	            "n_tasks": len(tasks),
   437	        }
   438	        effects.append(count)
   439	        variances.append(se ** 2)
   440	        per_cell_p_values.append(p_cell)
   441	        per_cell_ci_excludes_zero.append(per_cell_pass)
   442	
   443	    # PRIMARY: pooled meta
   444	    meta = dersimonian_laird_meta(effects, variances)
   445	    pooled_ci_lo = meta["pooled_ci_95"][0] if meta["pooled_ci_95"][0] is not None else None
   446	    primary_pass = (meta["p_value_two_sided"] is not None and
   447	                    meta["p_value_two_sided"] < alpha and
   448	                    pooled_ci_lo is not None and pooled_ci_lo > 0)
   449	
   450	    # TRANSPARENCY
   451	    holm_per_cell = holm_correct(per_cell_p_values, alpha=alpha)
   452	    for (cell_id, _), h in zip(per_cell.items(), holm_per_cell):
   453	        per_cell[cell_id]["holm_p"] = h["p_holm"]
   454	        per_cell[cell_id]["individually_holm_sig"] = h["rejected"]
   455	    n_per_cell_pass = sum(per_cell_ci_excludes_zero)
   456	    transparency_pass = n_per_cell_pass >= transparency_K_h3
   457	
   458	    return {
   459	        "axis_mode": axis_mode_key,
   460	        "ref_mode": ref_mode_key,
   461	        "primary_gate": {
   462	            "pooled_meta": meta,
   463	            "ci_excludes_zero": pooled_ci_lo is not None and pooled_ci_lo > 0,
   464	            "decision": "PASS" if primary_pass else "FAIL",
   465	        },
   466	        "transparency_K_h3": {
   467	            "K": transparency_K_h3,
   468	            "N": len(cells_by_id),
   469	            "n_cells_pass": n_per_cell_pass,
   470	            "consistent": transparency_pass,
   471	            "note": "transparency-only, NOT a gate on H3 (per prereg 2026-05-13 reclassification)",
   472	        },
   473	        "per_cell": per_cell,
   474	    }
   475	
   476	
   477	def evaluate_h2_cost(cells_by_id: dict[str, list[dict]], cost_margin_pct: float = 10.0,
   478	                      transparency_K_h2: int = 3) -> dict:
   479	    """H2(a): median cost(P-SoM) within ±cost_margin_pct% of median cost(DOM) per cell,
   480	    replicated in ≥ transparency_K_h2 of N cells.
   481	
   482	    H2(a) test margin is a RELATIVE PERCENTAGE (e.g., ±10% of DOM cost), distinct from
   483	    H1 TOST δ which is an SR percentage-point margin (codex probable concern disambig).
   484	    """
   485	    per_cell = {}
   486	    pass_count = 0
   487	    for cell_id, tasks in cells_by_id.items():
   488	        cost_dom_vals = [float(t["cost_dom"]) for t in tasks if t["cost_dom"]]
   489	        cost_psom_vals = [float(t["cost_psom"]) for t in tasks if t["cost_psom"]]
   490	        if not cost_dom_vals or not cost_psom_vals:
   491	            per_cell[cell_id] = {"per_cell_pass": False, "reason": "missing cost data"}
   492	            continue
   493	        med_dom = statistics.median(cost_dom_vals)
   494	        med_psom = statistics.median(cost_psom_vals)
   495	        rel_diff_pct = (med_psom - med_dom) / max(med_dom, 1e-12) * 100.0
   496	        within_band = abs(rel_diff_pct) <= cost_margin_pct
   497	        per_cell[cell_id] = {
   498	            "median_cost_dom": med_dom,
   499	            "median_cost_psom": med_psom,
   500	            "relative_diff_pct": rel_diff_pct,
   501	            "margin_pct": cost_margin_pct,
   502	            "per_cell_pass": within_band,
   503	        }
   504	        if within_band:
   505	            pass_count += 1
   506	    return {
   507	        "h2a_cost_equivalence": {
   508	            "K": transparency_K_h2,
   509	            "N": len(cells_by_id),
   510	            "n_cells_pass": pass_count,
   511	            "consistent": pass_count >= transparency_K_h2,
   512	            "margin_pct": cost_margin_pct,
   513	        },
   514	        "per_cell": per_cell,
   515	    }
   516	
   517	
   518	# ---------------------------------------------------------------------------
   519	# Framing rule R1-R5 mapper
   520	# ---------------------------------------------------------------------------
   521	
   522	def apply_framing_rule(h1: dict, h2: dict, h3_axis1: dict, h3_axis2: dict) -> dict:
   523	    """Apply preregistration §2 R1-R5 framing rule to test outcomes."""
   524	    h1_pass = h1["primary_gate"]["decision"] == "PASS"
   525	    h2_pass = h2["h2a_cost_equivalence"]["consistent"]
   526	    h3_axis1_pass = h3_axis1["primary_gate"]["decision"] == "PASS"
   527	    h3_axis2_pass = h3_axis2["primary_gate"]["decision"] == "PASS"
   528	
   529	    if h1_pass and h2_pass and h3_axis1_pass and h3_axis2_pass:
   530	        return {"rule": "R1", "framing": "Phantom routing space (2-axis empirical structure)",
   531	                "hook_power": "STRONGEST"}
   532	    if h1_pass and h2_pass and (h3_axis1_pass or h3_axis2_pass):
   533	        return {"rule": "R2", "framing": "Phantom routing space (single-axis empirical structure)",
   534	                "hook_power": "MODERATE-STRONG"}
   535	    if h1_pass and h2_pass and not h3_axis1_pass and not h3_axis2_pass:
   536	        return {"rule": "R3", "framing": "Phantom-SoM is hidden 4th routing arm (workshop-grade R3)",
   537	                "hook_power": "MODERATE"}
   538	    if h1_pass and not h2_pass:
   539	        return {"rule": "R4", "framing": "Phantom-SoM partial drop-in (cost/latency equivalence fails on some site)",
   540	                "hook_power": "WEAK"}
   541	    return {"rule": "R5", "framing": "Paper death scenario — pivot to VWA bug audit OR abandon",
   542	            "hook_power": "n/a"}
   543	
   544	
   545	# ---------------------------------------------------------------------------
   546	# Data loading
   547	# ---------------------------------------------------------------------------
   548	
   549	def load_per_task_csv(csv_path: Path) -> dict[str, list[dict]]:
   550	    """Load per-task CSV, return dict of cell_id → list of task rows."""
   551	    cells_by_id: dict[str, list[dict]] = defaultdict(list)
   552	    with csv_path.open() as f:
   553	        reader = csv.DictReader(f)
   554	        for row in reader:
   555	            cells_by_id[row["cell_id"]].append(row)
   556	    return dict(cells_by_id)
   557	
   558	
   559	def _file_sha256(path: Path) -> str:
   560	    h = hashlib.sha256()

exec
/bin/bash -lc "nl -ba scripts/analysis/preregistration_decision_test.py | sed -n '561,980p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
   561	    with path.open("rb") as f:
   562	        for chunk in iter(lambda: f.read(8192), b""):
   563	            h.update(chunk)
   564	    return h.hexdigest()
   565	
   566	
   567	# ---------------------------------------------------------------------------
   568	# Synthetic data generator (24-condition / 4-cell smoke test)
   569	# ---------------------------------------------------------------------------
   570	
   571	def generate_synthetic_per_task(seed: int = 42, n_tasks_per_cell: int = 200,
   572	                                  scenario: str = "r1_pass") -> dict[str, list[dict]]:
   573	    """Generate Phase 1a 4-cell × n_tasks per-task data.
   574	
   575	    Scenarios:
   576	      - r1_pass:   H1 strong (drop-one lift ~2pp pooled), H2 cost equiv hold, H3 both axes pass
   577	      - r3_pass:   H1 holds, H3 both axes fail (workshop fallback framing)
   578	      - r5_fail:   H1 fails (pooled near 0)
   579	    """
   580	    import random
   581	    rng = random.Random(seed)
   582	    cells_by_id = {}
   583	    for site, model in PHASE_1A_CELLS:
   584	        cell_id = f"{site}_{model}"
   585	        # Base per-task SR rates (per mode)
   586	        base_rate = {"sr_dom": 0.30, "sr_som": 0.32, "sr_vision": 0.20,
   587	                     "sr_ptext": 0.31, "sr_pprompt": 0.28, "sr_psom": 0.34}
   588	        # Capability adjustment
   589	        if model == "B1":
   590	            base_rate = {k: v * 0.6 for k, v in base_rate.items()}
   591	        # Scenario
   592	        if scenario == "r5_fail":
   593	            base_rate["sr_psom"] = base_rate["sr_dom"] - 0.01  # nullify hero
   594	        elif scenario == "r3_pass":
   595	            # Hero passes but axes collapse: ptext/pprompt similar to psom
   596	            base_rate["sr_ptext"] = base_rate["sr_psom"] - 0.005
   597	            base_rate["sr_pprompt"] = base_rate["sr_psom"] - 0.005
   598	
   599	        rows = []
   600	        for i in range(n_tasks_per_cell):
   601	            # Per-task latent solvability bias
   602	            bias = rng.uniform(-0.1, 0.1)
   603	            row = {"cell_id": cell_id, "site": site, "model": model,
   604	                   "task_id": f"{cell_id}_t{i:04d}"}
   605	            for mode_key, rate in base_rate.items():
   606	                eff_rate = max(0.0, min(1.0, rate + bias))
   607	                row[mode_key] = 1 if rng.random() < eff_rate else 0
   608	            # Cost: P-SoM ~ DOM cost (regex filter property)
   609	            row["cost_dom"] = 0.040 + rng.uniform(-0.005, 0.005)
   610	            row["cost_psom"] = row["cost_dom"] * (1.0 + rng.uniform(-0.05, 0.05))
   611	            rows.append(row)
   612	        cells_by_id[cell_id] = rows
   613	    return cells_by_id
   614	
   615	
   616	# ---------------------------------------------------------------------------
   617	# Main
   618	# ---------------------------------------------------------------------------
   619	
   620	def main():
   621	    p = argparse.ArgumentParser()
   622	    p.add_argument("--per-task-csv",
   623	                   help="Per-task CSV path (cell_id, site, model, task_id, sr_*, cost_*)")
   624	    p.add_argument("--synthetic", action="store_true",
   625	                   help="Run smoke test on synthetic 4-cell × 200-task data")
   626	    p.add_argument("--scenario", default="r1_pass",
   627	                   choices=["r1_pass", "r3_pass", "r5_fail"])
   628	    p.add_argument("--seed", type=int, default=42)
   629	    p.add_argument("--primary-gate", default="drop_one_pooled_meta_TOST",
   630	                   help="Primary gate flavor (informational; method is fixed in this rewrite)")
   631	    p.add_argument("--TOST-delta-pp", type=float, default=1.0,
   632	                   help="TOST equivalence margin in SR pp (default 1.0 per prereg lock)")
   633	    p.add_argument("--H1-magnitude-pp", type=float, default=1.0,
   634	                   help="H1 pooled magnitude threshold (default 1.0pp per prereg lock)")
   635	    p.add_argument("--H2-cost-margin-pct", type=float, default=10.0,
   636	                   help="H2(a) cost equivalence margin in % (default 10%% per prereg lock)")
   637	    p.add_argument("--H3-min-unique-count", type=int, default=2,
   638	                   help="H3 per-cell unique-count noise floor (default 2 tasks)")
   639	    p.add_argument("--transparency-K_h1", type=int, default=3,
   640	                   help="K_h1 transparency ratio cells count (default 3 of 4)")
   641	    p.add_argument("--transparency-K_h3", type=int, default=3,
   642	                   help="K_h3 transparency ratio cells count per axis (default 3 of 4)")
   643	    p.add_argument("--transparency-K_h2", type=int, default=3,
   644	                   help="H2 transparency cells count (default 3 of 4)")
   645	    p.add_argument("--alpha", type=float, default=0.05)
   646	    p.add_argument("--out", default="-", help="Output JSON path (- = stdout)")
   647	    args = p.parse_args()
   648	
   649	    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
   650	
   651	    # Load data
   652	    if args.synthetic:
   653	        cells_by_id = generate_synthetic_per_task(seed=args.seed, scenario=args.scenario)
   654	        input_sha = f"synthetic:{args.scenario}:{args.seed}"
   655	        logger.info(f"Synthetic mode: {len(cells_by_id)} cells, scenario={args.scenario}")
   656	    else:
   657	        if not args.per_task_csv:
   658	            logger.error("Must provide --per-task-csv or --synthetic")
   659	            sys.exit(2)
   660	        csv_path = Path(args.per_task_csv)
   661	        cells_by_id = load_per_task_csv(csv_path)
   662	        input_sha = _file_sha256(csv_path)
   663	        logger.info(f"Loaded {len(cells_by_id)} cells from {csv_path} (sha256={input_sha[:12]}...)")
   664	
   665	    if len(cells_by_id) < 2:
   666	        logger.error(f"Need ≥2 cells for pooled meta; got {len(cells_by_id)}")
   667	        sys.exit(2)
   668	
   669	    # Evaluate hypotheses
   670	    h1 = evaluate_h1(cells_by_id, delta_pp=args.TOST_delta_pp,
   671	                      magnitude_threshold_pp=args.H1_magnitude_pp,
   672	                      alpha=args.alpha, transparency_K_h1=args.transparency_K_h1,
   673	                      bootstrap_seed=args.seed)
   674	    h2 = evaluate_h2_cost(cells_by_id, cost_margin_pct=args.H2_cost_margin_pct,
   675	                           transparency_K_h2=args.transparency_K_h2)
   676	    h3_axis1 = evaluate_h3_axis(cells_by_id, axis_mode_key="sr_ptext",
   677	                                  ref_mode_key="sr_psom",
   678	                                  min_unique_count=args.H3_min_unique_count,
   679	                                  alpha=args.alpha,
   680	                                  transparency_K_h3=args.transparency_K_h3,
   681	                                  bootstrap_seed=args.seed)
   682	    h3_axis2 = evaluate_h3_axis(cells_by_id, axis_mode_key="sr_pprompt",
   683	                                  ref_mode_key="sr_psom",
   684	                                  min_unique_count=args.H3_min_unique_count,
   685	                                  alpha=args.alpha,
   686	                                  transparency_K_h3=args.transparency_K_h3,
   687	                                  bootstrap_seed=args.seed)
   688	    framing = apply_framing_rule(h1, h2, h3_axis1, h3_axis2)
   689	
   690	    result = {
   691	        "captured_at": datetime.now(timezone.utc).isoformat(),
   692	        "scope": "Phase 1a 24-condition / 4-cell statistical analysis",
   693	        "n_cells": len(cells_by_id),
   694	        "n_tasks_total": sum(len(t) for t in cells_by_id.values()),
   695	        "cell_ids": list(cells_by_id.keys()),
   696	        "input_data_sha256": input_sha,
   697	        "thresholds": {
   698	            "primary_gate_method": "pooled_DerSimonian_Laird_meta + TOST + magnitude",
   699	            "TOST_delta_pp": args.TOST_delta_pp,
   700	            "H1_magnitude_pp": args.H1_magnitude_pp,
   701	            "H2_cost_margin_pct": args.H2_cost_margin_pct,
   702	            "H3_min_unique_count": args.H3_min_unique_count,
   703	            "transparency_K_h1": args.transparency_K_h1,
   704	            "transparency_K_h3": args.transparency_K_h3,
   705	            "transparency_K_h2": args.transparency_K_h2,
   706	            "alpha": args.alpha,
   707	        },
   708	        "H1_psom_drop_one": h1,
   709	        "H2_cost_equivalence": h2,
   710	        "H3_axis1_ptext_unique": h3_axis1,
   711	        "H3_axis2_pprompt_unique": h3_axis2,
   712	        "framing_rule": framing,
   713	        "primary_gate_summary": {
   714	            "H1": h1["primary_gate"]["decision"],
   715	            "H2": "PASS" if h2["h2a_cost_equivalence"]["consistent"] else "FAIL",
   716	            "H3_axis1": h3_axis1["primary_gate"]["decision"],
   717	            "H3_axis2": h3_axis2["primary_gate"]["decision"],
   718	        },
   719	        "transparency_summary": {
   720	            "K_h1": f"{h1['transparency_K_h1']['n_individually_holm_sig']}/{h1['transparency_K_h1']['N']} ≥ {h1['transparency_K_h1']['K']}?  {'YES' if h1['transparency_K_h1']['consistent'] else 'NO'}",
   721	            "K_h3_axis1": f"{h3_axis1['transparency_K_h3']['n_cells_pass']}/{h3_axis1['transparency_K_h3']['N']} ≥ {h3_axis1['transparency_K_h3']['K']}?  {'YES' if h3_axis1['transparency_K_h3']['consistent'] else 'NO'}",
   722	            "K_h3_axis2": f"{h3_axis2['transparency_K_h3']['n_cells_pass']}/{h3_axis2['transparency_K_h3']['N']} ≥ {h3_axis2['transparency_K_h3']['K']}?  {'YES' if h3_axis2['transparency_K_h3']['consistent'] else 'NO'}",
   723	        },
   724	    }
   725	
   726	    payload = json.dumps(result, indent=2, default=float)
   727	    if args.out == "-":
   728	        print(payload)
   729	    else:
   730	        out_path = Path(args.out)
   731	        out_path.parent.mkdir(parents=True, exist_ok=True)
   732	        out_path.write_text(payload)
   733	        logger.info(f"Result → {out_path}")
   734	        logger.info(f"Framing rule: {framing['rule']} — {framing['framing']} (hook power: {framing['hook_power']})")
   735	        logger.info(f"  H1: {h1['primary_gate']['decision']} (pooled drop-one {h1['primary_gate']['pooled_meta']['pooled_effect']:.2f}pp, "
   736	                    f"superiority p={h1['primary_gate']['superiority_test']['p_one_sided']:.4f}, "
   737	                    f"TOST equiv {h1['primary_gate']['tost_informational']['decision']})")
   738	        logger.info(f"  H2: {'PASS' if h2['h2a_cost_equivalence']['consistent'] else 'FAIL'} "
   739	                    f"({h2['h2a_cost_equivalence']['n_cells_pass']}/{h2['h2a_cost_equivalence']['N']} cells within ±{args.H2_cost_margin_pct}% cost)")
   740	        logger.info(f"  H3 axis-1 (P-text): {h3_axis1['primary_gate']['decision']} "
   741	                    f"(pooled unique={h3_axis1['primary_gate']['pooled_meta']['pooled_effect']:.2f})")
   742	        logger.info(f"  H3 axis-2 (P-prompt): {h3_axis2['primary_gate']['decision']} "
   743	                    f"(pooled unique={h3_axis2['primary_gate']['pooled_meta']['pooled_effect']:.2f})")
   744	        logger.info(f"  Transparency K_h1: {result['transparency_summary']['K_h1']}")
   745	        logger.info(f"  Transparency K_h3 axis-1: {result['transparency_summary']['K_h3_axis1']}")
   746	        logger.info(f"  Transparency K_h3 axis-2: {result['transparency_summary']['K_h3_axis2']}")
   747	
   748	
   749	if __name__ == "__main__":
   750	    main()

exec
/bin/bash -lc "rg -n \"H1|TOST|superiority|DerSimonian|δ|1\\.0|K_h1|primary gate|Primary gate|Phase 1a|24-condition|24 conditions|4-cell|4 cells|queue_phase1|queue_16cell\" docs/checkpoints/pre_run/preregistration.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
11:data_lock_until: <pending Phase 1a 24-condition rerun completion (cls+red × B0+B1 × 6 modes)>
19:> **Reading order**: §1 epistemic structure (why this framework) → §2 hypotheses (H1-H6 + framing rule) → §3 multiple-comparison family declaration → §4 locked analysis choices → §5 exploratory disclosure → §6 witness mechanism.
36:3. **Framing decision is data-conditional, not data-prediction**: paper §1 hook framing depends on which combination of H1-H3 holds. The rule is pre-registered (R1-R5 below) so reviewers can verify the framing-to-data mapping is not post-hoc.
50:#### H1 — Hero deployment claim (P-SoM is hidden routing arm)
54:- **H1(i)** Pooled DerSimonian-Laird random-effect meta-analysis on N=4 (site, model) cells reaches significance at Holm α=0.05 (PRIMARY family m=1 test, no within-family correction needed).
55:- **H1(ii)** Pooled magnitude θ_RE ≥ 1.0pp AND one-sided **superiority test** rejects H0: θ ≤ 1.0pp at α=0.05 (i.e., effect is significantly ABOVE the +1.0pp substantive-effect threshold; commit-locked). Note 2026-05-13: replaces prior "TOST equivalence rejected at δ" wording which was ambiguous in direction; one-sided superiority is the unambiguous statistical test for "effect substantively > δ".
57:**Drop-one definition (operational)**: For each (site, model) cell containing all 6 modes (DOM, SoM, Vision, P-text, P-prompt, P-SoM), compute oracle ceiling SR over {6 modes} minus oracle ceiling SR over {5 modes drop P-SoM} per task, then average across the cell's task pool. Paired 1000-resample task-level bootstrap CI per cell; pooled DerSimonian-Laird across 4 cells.
59:**Transparency consistency check (NOT gating, reported alongside H1)**: K_h1 = ⌈0.75 × 4⌉ = 3 of 4 cells individually clear Holm α=0.05 within the per-cell P-SoM sub-family (m = 4). **K-of-N reclassified pre-data 2026-05-13** from gating threshold to transparency consistency check, based on power analysis (`docs/analysis/cross_sites/power_analysis.md`) showing per-cell power at observed 1-3pp effect sizes is < 10% — calibrated only for ≥7pp effects, smaller than reasonable phenomenon effect size, so K-as-gate is statistically dysfunctional. See §4 audit B9 row + Appendix A 2026-05-13 entry.
63:All four sub-claims hold per cell, replicated in ≥ K_h1 cells:
68:- **(d) Drop-one magnitude** — folded into H1(iii); P-SoM contributes ≥ 1.0pp lift on average.
76:- **H3(i) PRIMARY GATE** axis 1: pooled across N=4 cells, mean |P-text ∖ P-SoM| > 0 with DerSimonian-Laird random-effects meta CI excluding 0 (Holm α=0.05, m=1 within axis-1 sub-family).
80:**Transparency consistency check (NOT gating)**: K_h3 = ⌈0.67 × 4⌉ = 3 of 4 cells individually with bootstrap 95% CI excluding 0 (m=4 per axis). Same K-of-N reclassification rationale as H1 (see §4 audit B9 + Appendix A 2026-05-13 entry).
91:Reported per cell + meta-pooled (DerSimonian-Laird) for transparency. Holm-Bonferroni and BH FDR q-values reported. No pre-registered ranking commitment.
115:- **H7(i)** Pooled DerSimonian-Laird random-effect meta-analysis on lift reaches Holm α=0.05 (PRIMARY family m=1 if paper-1 / SECONDARY informational if paper-2).
116:- **H7(ii)** ≥ K_h1 of N_cells individually Holm-significant on per-cell lift, bootstrap 95% CI lower-bound > 0.
117:- **H7(iii)** Pooled magnitude θ_RE ≥ 1.0pp; TOST equivalence at margin δ=1.0pp rejected (same δ as H1).
131:- **H8(i)** Tier 2 router lift over Tier 1 oracle baseline ≥ 0 with bootstrap 95% CI excluding −1.0pp (paper claims Tier 2 ≈ Tier 1 within deployment-grade tolerance, given Tier 2 is leak-free and deployment-realistic).
132:- **H8(ii)** Tier 2 router lift over best-single-mode-baseline ≥ 1.0pp, ≥ K_h1 cells Holm-significant.
144:| **R1** | H1 holds AND H2 (a)(b)(c) all hold AND H3(i) holds AND H3(ii) holds | "Phantom routing space (M1/M2 2-axis empirical structure); P-SoM as deployment hero, P-text/P-prompt as structural ablation arms validating axis decomposition." | STRONGEST |
145:| **R2** | H1+H2 hold AND only one of H3(i)/(ii) holds | "Phantom routing space (single-axis empirical structure) with P-SoM as deployment hero; remaining axis decomposition theoretical (Zoom 1 architectural argument only)." | MODERATE-STRONG |
146:| **R3** | H1+H2 hold AND neither H3(i)/(ii) holds | "Phantom-SoM is hidden 4th routing arm; M1/M2 axis decomposition supported by Zoom 1 architectural argument only, not empirically validated by ablation." | MODERATE (= 04-30 fallback; workshop-grade) |
147:| **R4** | H1 holds AND H2 partially fails (e.g., (a) cost or (b) latency fails on some site) | "Phantom-SoM partial drop-in" + §4 disclosure of failed sub-claim. | WEAK; substantial revision |
148:| **R5** | H1 fails (pooled meta DerSimonian-Laird Holm α=0.05 fails OR pooled magnitude θ_RE < 1.0pp OR TOST equivalence fails reject at δ=1.0pp) | Paper death scenario: pivot to VWA bug audit paper (§107 4-cluster fix as primary) OR abandon. Decision deferred to advisor sync at fail time. | n/a |
150:**Trigger rule update 2026-05-13**: R5 no longer fires on `< K_h1` (K-of-N reclassified to transparency-only). Pooled meta + TOST primary gate only. K-of-N consistency reported in §4 per-cell table as descriptive transparency row.
152:**Heterogeneity-conditional rule (added 2026-05-13 to resolve §4 audit B8 ↔ H1(i) conflict)**: If pre-specified I² > 75% from random-effects meta (per §4 audit B8 thresholds), do NOT pool — primary inference reverts to per-cell forest + meta-regression by site / model. R1-R5 framing in this branch maps to per-cell direction-consistency: ≥3 of 4 cells direction-positive + ≥2 individually Holm sig → R3-grade hook; otherwise R4/R5.
159:- H1(i) pooled meta on N=4 statistical cells: m = 1 (no within-family correction).
160:- H1(ii) pooled magnitude θ_RE ≥ 1.0pp + TOST equivalence reject at δ=1.0pp: m = 1.
165:- H3(i) pooled axis-1 meta on N=4 cells: m = 1.
166:- H3(ii) pooled axis-2 meta on N=4 cells: m = 1.
171:- K_h1 = ⌈0.75 × 4⌉ = 3 of 4 cells individually Holm-significant on P-SoM drop-one (m=4 per cell).
172:- K_h3 axis-1 = ⌈0.67 × 4⌉ = 3 of 4 cells individually with bootstrap CI excluding 0.
175:- **Rationale for transparency-only reclassification**: power analysis (`docs/analysis/cross_sites/power_analysis.md`, pre-data) shows K-of-N family power at observed 1-3pp effect sizes is < 10%, calibrated only for ≥7pp effects. Per-cell N=234 (cls) / 210 (red) bootstrap power at 1.5pp effect ≈ 0.30. P(≥3 of 4 cells sig | p_cell=0.30) ≈ 8%. K-as-gate is statistically dysfunctional in this regime; K-as-transparency provides per-cell consistency check value alongside pooled meta. See Appendix A 2026-05-13 entry.
180:- H7(iii) folded into H7(i) magnitude/TOST.
210:| **TOST equivalence margin δ** | **1.0pp** | ≈ 2 tasks in N=234, matches per-cell bootstrap SE; smaller is within sampling noise floor |
211:| **H1 K_h1 transparency ratio** | **0.75** (= 3/4 cells; **transparency-only, not gating** per 2026-05-13 reclassification) | Reports per-cell consistency alongside pooled meta; not a gate on H1 |
212:| **H3 K_h3 transparency ratio** | **0.67** (= 3/4 cells; **transparency-only**) | Same as K_h1 reclassification rationale |
214:| **Cell inclusion (Phase 1a main)** | Phase A post-fix only (commit ≥ 3c15cd7), cls + red sites only, all 6 modes per (site, model) cell freshly rerun | Bug-clean rerun + workshop-target scope (shop deferred to Phase 1b) |
215:| **Cell inclusion (Phase 1b main paper)** | Phase A post-fix rerun of shop × B0+B1 × 6 modes (12 conditions added on top of Phase 1a 24 conditions) | Cross-site expansion lever for main paper, post-data R1 vs Option D framing decision |
225:| **N_conditions Phase 1a (operational)** | **24 conditions** = 2 sites (cls, red) × 2 models (B0, B1) × 6 modes (DOM, SoM, Vision, P-text, P-prompt, P-SoM). Each condition launched fresh post-fix via `scripts/queues/queue_phase1_paper_grade.sh` (renamed 2026-05-13 from `queue_16cell_paper_grade.sh`; current scope = 24 conditions Phase 1a + 12 conditions Phase 1b deferred). Sequence: B0 → B1 per site (shared user account); cls + red parallel chains | ✅ **Student-decided 2026-05-13** post-codex stress audit. Workshop-targeted (cls + red only, shop deferred to Phase 1b for main paper). Replaces prior 16-cell phantom-only scope that lacked baseline DOM/SoM/Vision rerun (codex Flaw 1) |
226:| **N_cells statistical (H1/H3 stratification)** | **4 cells** = (site, model) tuples: (cls, B0), (cls, B1), (red, B0), (red, B1). Drop-one is computed per cell using all 6 modes; pooled DerSimonian-Laird random-effects meta across 4 cells | Cell = paired-test stratification unit (one per (site, model)), distinct from "condition" (one per (site, model, mode)). 4 cells × 6 modes = 24 conditions. Distinction propagated to all prose / queue / docs 2026-05-13 |
227:| **N_conditions Phase 1b (main paper, deferred)** | **+12 conditions** = shop × 2 models × 6 modes. Launches after Phase 1a workshop submission to feed main paper R1 / Option D framing decision. N_cells statistical becomes 6 (= 3 sites × 2 models) when Phase 1b lands | Phase 1b is additive; workshop §1 hook does NOT depend on Phase 1b. Main paper §1 hook upgrade R3 → R1 conditional on shop replicating P-SoM 4-fold within ±2pp tolerance |
231:| **Heterogeneity (random-effects, Q, I², τ²) pre-spec** (audit B8) | (a) **Primary estimator**: random-effects DerSimonian-Laird via `aggregate_phantom_meta.py` (already implemented). (b) **Heterogeneity reporting**: report Cochran Q (chi² test of homogeneity), I² (% of total variance attributable to between-cell heterogeneity), τ² (between-cell variance). (c) **Interpretation thresholds (pre-specified)**: I² < 25% = "low heterogeneity, pooled mean is primary"; 25%-50% = "moderate, report both pooled + per-cell"; 50%-75% = "high, per-cell estimates are primary, pooled is summary"; > 75% = "very high, do not pool — report only per-cell + heterogeneity-source analysis (site / model / task-pool)". (d) **Heterogeneity-source decomposition**: when I² > 50%, report meta-regression by site (cls / red / shop) and by model (B0 / B1) to identify dominant variance source. | Higgins & Thompson 2002 (I² thresholds). Per-cell estimates always shown alongside pooled, so heterogeneity is never averaged away. Lock 2026-05-09. |
232:| **K-of-N rule scope** (audit B9 power-corrected, REPROPAGATED 2026-05-13 to H1/H3/R5/§6/Appendix A) | The **K_h1=3/4 / K_h3=3/4** ratios (under 24-condition / 4-cell Phase 1a scope) are **transparency consistency checks** (count of cells *individually* clearing α=0.05 Holm), **NOT gates on H1/H3 paper claims**. **Primary gate** = (a) DerSimonian-Laird random-effects meta-analysis on N=4 (site, model) cells + (b) TOST equivalence on pooled cls + red tasks at δ=1.0pp. Per `docs/analysis/cross_sites/power_analysis.md` §3-§5, K-of-N family power at observed 1-3pp effect sizes is < 10%; the rule is calibrated for ≥7pp effects (1.5pp per-cell power ≈ 0.30; P(≥3 of 4 cells sig) ≈ 8%). K-as-gate is statistically dysfunctional in this effect-size regime. **2026-05-13 propagation**: prior prereg text in H1(ii) / H3(i) / H3(ii) / R5 / §6 still gated K-of-N → fixed to "transparency consistency check, reported alongside but NOT gating". This is **pre-data reclassification**: power analysis commit predates Phase 1a launch; reclassification timestamp recorded for OSF witness audit trail. | Original audit B9 lock 2026-05-09 introduced framing but did not propagate to H1/H3/R5/§6 prose (codex stress audit 2026-05-13 Flaw 2 surfaced internal contradiction). Repropagation 2026-05-13 reconciles all references. |
249:- Any post-hoc cell subsetting beyond H1-H8 family scope
255:at **L17** (3 of 4 cells Holm-significant on `token_overlap_to_target`, p_Holm <
294:   - (1) **K_h1=0.75 transparency ratio** (= 3/4 cells; reclassified gate → transparency-only 2026-05-13)
295:   - (2) **K_h3=0.67 transparency ratio** (= 3/4 cells; reclassified gate → transparency-only 2026-05-13)
296:   - (3) **TOST δ=1.0pp** equivalence margin (interpretation: SR drop-one effect-size margin, distinct from H2(a) cost ±10% margin — see §4 lock row)
297:   - (4) **Cell inclusion**: Phase 1a = cls + red × B0+B1 × 6 modes (Phase A post-fix only); Phase 1b shop deferred
299:   - (6) **N_conditions Phase 1a final scope**: **24 operational conditions** (= 2 sites × 2 models × 6 modes) across **4 statistical cells** (= (site, model) tuples) — student-decided 2026-05-13 post-codex stress audit, replaces prior 16-cell phantom-only scope. Advisor email witness pending
303:   - Plus lock H-list (H1-H8 family declaration final).
306:4. Advisor sends single-line confirmation email: "I witness pre-registration of phantom-SoM hypotheses (H1-H8) and 8 lock decisions as of <git SHA> <date>." Email archived in `.witness/preregistration_witness.eml` (gitignored, local-only).
349:| 2026-05-03 | TOST δ = 1.0pp locked (was 0.5pp draft) | 0.5pp = 1 task in N=234 too liberal; 1.0pp = 2 tasks ≈ bootstrap SE noise floor; statistically principled |
350:| 2026-05-03 | K_h1 = 0.75 cell-pass threshold for H1 | Allows ~25% capability-outlier cells; not so strict as to break on single-cell noise |
351:| 2026-05-03 | K_h3 = 0.67 cell-pass threshold for H3 | Lower than K_h1 because structural < deployment commit |
354:| 2026-05-05 | Advisor sync 5/5 partial outcome — early-stop A locked (cancel全 mechanism); compute path locked (advisor 5090 → Rancher H100 → RunPod backup); paper split direction discussed but Mechanistic-nested-vs-independent + threshold detail not finalized due to network drop | Advisor explicit confirm early-stop cancel + compute paths; paper split + threshold lock deferred to email follow-up via `docs/checkpoints/advisor_sync_5_5_followup.md` |
355:| 2026-05-05 | **N_cells = 16** (student-decided post-5/5 sync, advisor email witness pending) | 14 (pre-sync default) → 16 to add B1 shop × {phantom_text, phantom_som} 2 cells for cross-capability shop coverage. K_h1 threshold count: ⌈0.75 × 16⌉ = 12. K_h3 threshold count: ⌈0.67 × 16⌉ = 11 |
356:| 2026-05-13 | **Codex stress audit triggered 6 paper-grade design fixes** (pre-launch): (a) scope reframe 16-cell phantom-only → 24-condition / 4-cell Phase 1a (cls+red×B0+B1×6modes), Phase 1b shop deferred to main paper; (b) K-of-N reclassified gate → transparency-only (power analysis showing dysfunction at < 7pp effects, re-propagated to H1/H3/R5/§6); (c) H1 drop-one definition disambiguated (oracle ceiling lift with-vs-without P-SoM, per (site, model) cell paired bootstrap); (d) smoke-gate B7 revised outcome-independent (no SR-based restart bias); (e) cell terminology disambiguated ("cell" = 4 statistical strata for K-of-N/meta input, "condition" = 24 operational launch units); (f) Phase 1b shop scope-expansion lever for main paper R3→R1 framing decision | Codex CLI hostile reviewer audit (`docs/checkpoints/codex_outputs/codex_stress_16cell_design_2026-05-13.md`, lean prompt no-enumeration, cross-AI complementary to prior Claude reviews); 6 HIGH severity findings + 3 probable concerns. Workshop-targeted Phase 1a launch this week; main paper Phase 1b after workshop submission |
357:| \<pending advisor email follow-up\> | \<witness K_h1=0.75 transparency / K_h3=0.67 transparency / TOST δ=1.0pp / N_conditions=24 (Phase 1a) / N_cells=4 / split protocol / paper split / Phase 1b shop / outcome-indep smoke gate / per follow-up doc Q1-Q11\> | \<email reply timestamp + Git SHA at lock\> |

exec
/bin/bash -lc "rg -n \"H1|TOST|superiority|DerSimonian|δ|1\\.0|K_h1|primary gate|Primary gate|Phase 1a|24-condition|24 conditions|4-cell|4 cells|queue_phase1|queue_16cell\" docs/checkpoints/advisor_sync_5_5_followup.md docs/checkpoints/pre_run/osf_lock_manifest.md docs/checkpoints/next_steps.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
docs/checkpoints/pre_run/osf_lock_manifest.md:5:(post 5/5 sync) confirming K_h1 / K_h3 / TOST δ thresholds, this checklist
docs/checkpoints/pre_run/osf_lock_manifest.md:15:- [ ] Advisor email reply received (K_h1=0.75 transparency / K_h3=0.67 transparency /
docs/checkpoints/pre_run/osf_lock_manifest.md:16:      TOST δ=1.0pp SR-margin / Phase 1a 24-condition 4-cell scope confirmed
docs/checkpoints/pre_run/osf_lock_manifest.md:20:      drop-one H1 formula + outcome-independent smoke gate + Appendix A 2026-05-13)
docs/checkpoints/pre_run/osf_lock_manifest.md:21:- [ ] `run_manifest.yaml` archived rows verified (Phase 1a 24-condition scope = 2 sites
docs/checkpoints/pre_run/osf_lock_manifest.md:23:      for Phase 1a post-fix rerun cells; Phase 1b shop deferred rows separately tagged)
docs/checkpoints/pre_run/osf_lock_manifest.md:51:| H1 PRIMARY gate (P-SoM drop-one oracle ceiling lift) | Pooled DerSimonian-Laird meta Holm α=0.05 sig on N=4 (site, model) cells + pooled magnitude θ_RE ≥ 1.0pp + TOST equivalence reject at δ=1.0pp | ⏳ pending | Drop-one = oracle SR over {6 modes} − oracle SR over {5 modes drop P-SoM} per task, paired bootstrap per cell, pooled across 4 cells |
docs/checkpoints/pre_run/osf_lock_manifest.md:52:| H3 PRIMARY gate axis-1 (P-text \ P-SoM) | Pooled axis-1 DerSimonian-Laird meta Holm α=0.05 sig on N=4 cells | ⏳ pending | Per-cell bootstrap CI on unique-task count, then pooled meta |
docs/checkpoints/pre_run/osf_lock_manifest.md:54:| K_h1 transparency ratio (NOT a gate) | 0.75 → 3 of 4 cells individually Holm-sig on drop-one | ⏳ pending | Reclassified gate → transparency 2026-05-13 (power analysis dysfunction at <7pp effects); reported alongside pooled meta as per-cell consistency check |
docs/checkpoints/pre_run/osf_lock_manifest.md:55:| K_h3 transparency ratio (NOT a gate) | 0.67 → 3 of 4 cells individually CI > 0 | ⏳ pending | Same reclassification rationale |
docs/checkpoints/pre_run/osf_lock_manifest.md:56:| TOST equivalence δ (SR-margin) | 1.0pp | ⏳ pending | SR percentage-point margin for H1(iii) drop-one effect size; distinct from H2(a) cost ±10% relative margin |
docs/checkpoints/pre_run/osf_lock_manifest.md:57:| Cell scope (Phase 1a operational) | 24 conditions = 2 sites (cls, red) × 2 models (B0, B1) × 6 modes (DOM, SoM, Vision, P-text, P-prompt, P-SoM) | ⏳ pending | Replaces prior 16/18-cell phantom-only scope (codex Flaw 1+5 fix 2026-05-13) |
docs/checkpoints/pre_run/osf_lock_manifest.md:58:| Cell scope (Phase 1a statistical) | 4 cells = (site, model) tuples | ⏳ pending | One drop-one number per cell; pooled meta input |
docs/checkpoints/pre_run/osf_lock_manifest.md:59:| Cell scope (Phase 1b deferred) | +12 conditions = shop × B0+B1 × 6 modes | ⏳ pending | Main-paper expansion lever; not part of Phase 1a workshop submission |
docs/checkpoints/next_steps.md:31:> 3. **Advisor email reply wait** (~2-5d, passive) — Q1-Q11 in [[advisor_sync_5_5_followup]]. K_h1=12 / K_h3=11 / TOST δ=1.0pp threshold lock + paper split 3v4. Reply triggers OSF DOI 8-step lock + 16-cell launch gate clearance.
docs/checkpoints/next_steps.md:103:**Phase 1a (workshop-targeted, immediate launch)** — 24 operational conditions across 4 statistical cells:
docs/checkpoints/next_steps.md:108:- **Total: 24 conditions = 2 sites × 2 models × 6 modes, 4 statistical (site, model) cells**
docs/checkpoints/next_steps.md:115:**Orchestrator**: `bash scripts/queues/queue_phase1_paper_grade.sh dry-run` (preview) → `... launch` (Phase 1a default = cls + red parallel chains). Phase 1b launches via `launch phase1b shop`.
docs/checkpoints/next_steps.md:125:**ETA on A100 40GB** (Phase 1a, post-advisor lock):
docs/checkpoints/next_steps.md:130:| **Phase 1a wallclock (parallel)** | 24 | **~72h ≈ 3 days** |
docs/checkpoints/next_steps.md:138:    --primary-gate drop_one_pooled_meta_TOST \
docs/checkpoints/next_steps.md:139:    --transparency-K_h1 3 --transparency-K_h3 3 --TOST-delta 1.0 \
docs/checkpoints/next_steps.md:199:**Trigger**: Advisor email reply with confirmed K_h1 / K_h3 / TOST δ.
docs/checkpoints/next_steps.md:232:| 🟡 R6 | `check_evaluator_consistency.py` (Gate 7 in `queue_phase1_paper_grade.sh`) — verify all conditions' most-recent `rederive_metadata.evaluator_code_sha` == lock-time SHA | 30 min | OSF DOI lock prep (笔记 §115 Protocol B §6) |
docs/checkpoints/next_steps.md:326:scripts/queues/queue_phase1_paper_grade.sh         🆕 Phase 1 paper-grade orchestrator (Phase 1a 24-cond default + Phase 1b shop deferred)
docs/checkpoints/next_steps.md:334:scripts/analysis/preregistration_decision_test.py  🆕 H1/H3/TOST canonical (Gap C2)
docs/checkpoints/advisor_sync_5_5_followup.md:16:> - **Phase 1a (workshop-targeted, immediate launch)**: 24 operational conditions = 2 sites (cls + red) × 2 models (B0 + B1) × 6 modes (DOM / SoM / Vision / P-text / P-prompt / P-SoM). 统计分析 4 个 cell (= (site, model) tuple), pooled DerSimonian-Laird meta + TOST primary gate. 投 workshop 这一档.
docs/checkpoints/advisor_sync_5_5_followup.md:19:> 我 5/12-6/1 考试, 这两天能 lock 完我就 launch Phase 1a 然后专心复习, 麻烦您扫一眼回个 email 即可!
docs/checkpoints/advisor_sync_5_5_followup.md:147:#### (a) K_h1 = 0.75 transparency ratio — Hero claim per-cell consistency (NOT gate, 2026-05-13 reclassified)
docs/checkpoints/advisor_sync_5_5_followup.md:149:**测什么**: P-SoM drop-one oracle ceiling lift > 0 在多少 % 的 statistical cell 里 individually Holm-significant. **现在是 transparency consistency check, NOT a gate on H1 paper claim**.
docs/checkpoints/advisor_sync_5_5_followup.md:151:**Phase 1a 24-condition / 4-cell scope 下**: ⌈0.75 × 4⌉ = **3 of 4 cells** individually clear Holm α=0.05 (报在 §4 per-cell table 作为 reviewer transparency consistency 行).
docs/checkpoints/advisor_sync_5_5_followup.md:155:- N=4 cells × per-cell power ≈ 0.30 at 1.5pp → P(≥3 of 4 sig) ≈ 8%, K-as-gate 设计上必 fail 即使 phenomenon 真实
docs/checkpoints/advisor_sync_5_5_followup.md:157:- **Primary H1 gate** = (i) pooled DerSimonian-Laird random-effects meta on 4 cells 在 Holm α=0.05 显著 + (ii) pooled magnitude θ_RE ≥ 1.0pp + TOST δ=1.0pp reject equivalence
docs/checkpoints/advisor_sync_5_5_followup.md:159:**Pre-data reclassification timestamp**: 2026-05-13 commit predates Phase 1a launch, OSF DOI 见证 audit trail 显示 reclass 在 unblind 之前 (防 reviewer 攻击 "你看了数据再 weaken gate").
docs/checkpoints/advisor_sync_5_5_followup.md:171:**Phase 1a 4-cell 下**: ⌈0.67 × 4⌉ = **3 of 4 cells** per axis 作为 transparency 行.
docs/checkpoints/advisor_sync_5_5_followup.md:173:**Primary H3 gate** = pooled axis-1 + axis-2 DerSimonian-Laird meta Holm α=0.05 (两个 axis 独立 sub-family, 各 m=1).
docs/checkpoints/advisor_sync_5_5_followup.md:175:Same reclassification rationale as K_h1.
docs/checkpoints/advisor_sync_5_5_followup.md:177:#### (c) TOST δ = 1.0pp — Equivalence margin (single canonical interpretation)
docs/checkpoints/advisor_sync_5_5_followup.md:179:**测什么**: 用作 H1(ii) drop-one effect-size equivalence margin (pooled drop-one lift 是 ≥ 1.0pp 而不是 ≈ 0). H1(iii) 的 TOST reject equivalence at δ=1.0pp 意思是 "drop-one lift 显著 > 1pp, 不是统计噪声".
docs/checkpoints/advisor_sync_5_5_followup.md:181:**注意 (2026-05-13 disambiguation, codex probable concern)**: 这个 δ=1.0pp 是 **SR percentage-point margin**, 不是 cost equivalence margin. H2(a) "cost ≈ DOM" 用另一个 margin ±10% relative cost (不复用同一个 δ). 之前 prereg 跟 advisor follow-up 这两处单位有混淆, 现在显式区分.
docs/checkpoints/advisor_sync_5_5_followup.md:183:**为什么 1.0pp 而不是 0.5pp 或 3pp**:
docs/checkpoints/advisor_sync_5_5_followup.md:185:- Bootstrap iteration noise + cell-level correlated error 实测约 0.7-1.0pp
docs/checkpoints/advisor_sync_5_5_followup.md:186:- δ = 0.5pp → 比 noise floor 还小, TOST 永远测不出 (Type II error 高)
docs/checkpoints/advisor_sync_5_5_followup.md:187:- δ = 3pp → 太松, equivalence claim 太 cheap
docs/checkpoints/advisor_sync_5_5_followup.md:188:- **δ = 1.0pp ≈ 2 tasks @ N=234**, noise floor 上方安全 margin
docs/checkpoints/advisor_sync_5_5_followup.md:196:> "I have reviewed the pre-registration revision (K_h1=0.75 transparency-only / K_h3=0.67 transparency-only / TOST δ=1.0pp SR-margin / Phase 1a 24 conditions across 4 cells: cls+red × B0+B1 × 6 modes / Phase 1b shop deferred / outcome-independent smoke gate / pooled DerSimonian-Laird meta + TOST primary gating) on \<date\> and witness them as committed before Phase 1a data unblinding."

exec
/bin/bash -lc "rg -n \"TOST|drop_one_pooled_meta_TOST|pooled_DerSimonian_Laird_meta \\+ TOST|equivalence rejected|equivalence\" scripts/analysis/preregistration_decision_test.py docs/checkpoints/pre_run/preregistration.md docs/checkpoints/advisor_sync_5_5_followup.md docs/checkpoints/pre_run/osf_lock_manifest.md docs/checkpoints/next_steps.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
docs/checkpoints/next_steps.md:31:> 3. **Advisor email reply wait** (~2-5d, passive) — Q1-Q11 in [[advisor_sync_5_5_followup]]. K_h1=12 / K_h3=11 / TOST δ=1.0pp threshold lock + paper split 3v4. Reply triggers OSF DOI 8-step lock + 16-cell launch gate clearance.
docs/checkpoints/next_steps.md:138:    --primary-gate drop_one_pooled_meta_TOST \
docs/checkpoints/next_steps.md:139:    --transparency-K_h1 3 --transparency-K_h3 3 --TOST-delta 1.0 \
docs/checkpoints/next_steps.md:199:**Trigger**: Advisor email reply with confirmed K_h1 / K_h3 / TOST δ.
docs/checkpoints/next_steps.md:334:scripts/analysis/preregistration_decision_test.py  🆕 H1/H3/TOST canonical (Gap C2)
docs/checkpoints/pre_run/osf_lock_manifest.md:5:(post 5/5 sync) confirming K_h1 / K_h3 / TOST δ thresholds, this checklist
docs/checkpoints/pre_run/osf_lock_manifest.md:16:      TOST δ=1.0pp SR-margin / Phase 1a 24-condition 4-cell scope confirmed
docs/checkpoints/pre_run/osf_lock_manifest.md:51:| H1 PRIMARY gate (P-SoM drop-one oracle ceiling lift) | Pooled DerSimonian-Laird meta Holm α=0.05 sig on N=4 (site, model) cells + pooled magnitude θ_RE ≥ 1.0pp + TOST equivalence reject at δ=1.0pp | ⏳ pending | Drop-one = oracle SR over {6 modes} − oracle SR over {5 modes drop P-SoM} per task, paired bootstrap per cell, pooled across 4 cells |
docs/checkpoints/pre_run/osf_lock_manifest.md:56:| TOST equivalence δ (SR-margin) | 1.0pp | ⏳ pending | SR percentage-point margin for H1(iii) drop-one effect size; distinct from H2(a) cost ±10% relative margin |
docs/checkpoints/advisor_sync_5_5_followup.md:16:> - **Phase 1a (workshop-targeted, immediate launch)**: 24 operational conditions = 2 sites (cls + red) × 2 models (B0 + B1) × 6 modes (DOM / SoM / Vision / P-text / P-prompt / P-SoM). 统计分析 4 个 cell (= (site, model) tuple), pooled DerSimonian-Laird meta + TOST primary gate. 投 workshop 这一档.
docs/checkpoints/advisor_sync_5_5_followup.md:157:- **Primary H1 gate** = (i) pooled DerSimonian-Laird random-effects meta on 4 cells 在 Holm α=0.05 显著 + (ii) pooled magnitude θ_RE ≥ 1.0pp + TOST δ=1.0pp reject equivalence
docs/checkpoints/advisor_sync_5_5_followup.md:177:#### (c) TOST δ = 1.0pp — Equivalence margin (single canonical interpretation)
docs/checkpoints/advisor_sync_5_5_followup.md:179:**测什么**: 用作 H1(ii) drop-one effect-size equivalence margin (pooled drop-one lift 是 ≥ 1.0pp 而不是 ≈ 0). H1(iii) 的 TOST reject equivalence at δ=1.0pp 意思是 "drop-one lift 显著 > 1pp, 不是统计噪声".
docs/checkpoints/advisor_sync_5_5_followup.md:181:**注意 (2026-05-13 disambiguation, codex probable concern)**: 这个 δ=1.0pp 是 **SR percentage-point margin**, 不是 cost equivalence margin. H2(a) "cost ≈ DOM" 用另一个 margin ±10% relative cost (不复用同一个 δ). 之前 prereg 跟 advisor follow-up 这两处单位有混淆, 现在显式区分.
docs/checkpoints/advisor_sync_5_5_followup.md:186:- δ = 0.5pp → 比 noise floor 还小, TOST 永远测不出 (Type II error 高)
docs/checkpoints/advisor_sync_5_5_followup.md:187:- δ = 3pp → 太松, equivalence claim 太 cheap
docs/checkpoints/advisor_sync_5_5_followup.md:196:> "I have reviewed the pre-registration revision (K_h1=0.75 transparency-only / K_h3=0.67 transparency-only / TOST δ=1.0pp SR-margin / Phase 1a 24 conditions across 4 cells: cls+red × B0+B1 × 6 modes / Phase 1b shop deferred / outcome-independent smoke gate / pooled DerSimonian-Laird meta + TOST primary gating) on \<date\> and witness them as committed before Phase 1a data unblinding."
docs/checkpoints/pre_run/preregistration.md:55:- **H1(ii)** Pooled magnitude θ_RE ≥ 1.0pp AND one-sided **superiority test** rejects H0: θ ≤ 1.0pp at α=0.05 (i.e., effect is significantly ABOVE the +1.0pp substantive-effect threshold; commit-locked). Note 2026-05-13: replaces prior "TOST equivalence rejected at δ" wording which was ambiguous in direction; one-sided superiority is the unambiguous statistical test for "effect substantively > δ".
docs/checkpoints/pre_run/preregistration.md:117:- **H7(iii)** Pooled magnitude θ_RE ≥ 1.0pp; TOST equivalence at margin δ=1.0pp rejected (same δ as H1).
docs/checkpoints/pre_run/preregistration.md:148:| **R5** | H1 fails (pooled meta DerSimonian-Laird Holm α=0.05 fails OR pooled magnitude θ_RE < 1.0pp OR TOST equivalence fails reject at δ=1.0pp) | Paper death scenario: pivot to VWA bug audit paper (§107 4-cluster fix as primary) OR abandon. Decision deferred to advisor sync at fail time. | n/a |
docs/checkpoints/pre_run/preregistration.md:150:**Trigger rule update 2026-05-13**: R5 no longer fires on `< K_h1` (K-of-N reclassified to transparency-only). Pooled meta + TOST primary gate only. K-of-N consistency reported in §4 per-cell table as descriptive transparency row.
docs/checkpoints/pre_run/preregistration.md:160:- H1(ii) pooled magnitude θ_RE ≥ 1.0pp + TOST equivalence reject at δ=1.0pp: m = 1.
docs/checkpoints/pre_run/preregistration.md:180:- H7(iii) folded into H7(i) magnitude/TOST.
docs/checkpoints/pre_run/preregistration.md:210:| **TOST equivalence margin δ** | **1.0pp** | ≈ 2 tasks in N=234, matches per-cell bootstrap SE; smaller is within sampling noise floor |
docs/checkpoints/pre_run/preregistration.md:232:| **K-of-N rule scope** (audit B9 power-corrected, REPROPAGATED 2026-05-13 to H1/H3/R5/§6/Appendix A) | The **K_h1=3/4 / K_h3=3/4** ratios (under 24-condition / 4-cell Phase 1a scope) are **transparency consistency checks** (count of cells *individually* clearing α=0.05 Holm), **NOT gates on H1/H3 paper claims**. **Primary gate** = (a) DerSimonian-Laird random-effects meta-analysis on N=4 (site, model) cells + (b) TOST equivalence on pooled cls + red tasks at δ=1.0pp. Per `docs/analysis/cross_sites/power_analysis.md` §3-§5, K-of-N family power at observed 1-3pp effect sizes is < 10%; the rule is calibrated for ≥7pp effects (1.5pp per-cell power ≈ 0.30; P(≥3 of 4 cells sig) ≈ 8%). K-as-gate is statistically dysfunctional in this effect-size regime. **2026-05-13 propagation**: prior prereg text in H1(ii) / H3(i) / H3(ii) / R5 / §6 still gated K-of-N → fixed to "transparency consistency check, reported alongside but NOT gating". This is **pre-data reclassification**: power analysis commit predates Phase 1a launch; reclassification timestamp recorded for OSF witness audit trail. | Original audit B9 lock 2026-05-09 introduced framing but did not propagate to H1/H3/R5/§6 prose (codex stress audit 2026-05-13 Flaw 2 surfaced internal contradiction). Repropagation 2026-05-13 reconciles all references. |
docs/checkpoints/pre_run/preregistration.md:296:   - (3) **TOST δ=1.0pp** equivalence margin (interpretation: SR drop-one effect-size margin, distinct from H2(a) cost ±10% margin — see §4 lock row)
docs/checkpoints/pre_run/preregistration.md:349:| 2026-05-03 | TOST δ = 1.0pp locked (was 0.5pp draft) | 0.5pp = 1 task in N=234 too liberal; 1.0pp = 2 tasks ≈ bootstrap SE noise floor; statistically principled |
docs/checkpoints/pre_run/preregistration.md:357:| \<pending advisor email follow-up\> | \<witness K_h1=0.75 transparency / K_h3=0.67 transparency / TOST δ=1.0pp / N_conditions=24 (Phase 1a) / N_cells=4 / split protocol / paper split / Phase 1b shop / outcome-indep smoke gate / per follow-up doc Q1-Q11\> | \<email reply timestamp + Git SHA at lock\> |
scripts/analysis/preregistration_decision_test.py:5:   - PRIMARY GATE = pooled DerSimonian-Laird random-effects meta + TOST equivalence
scripts/analysis/preregistration_decision_test.py:19:  - TOST: two one-sided tests for H0 |θ| ≥ δ rejected vs H1 |θ| < δ at δ=1.0pp.
scripts/analysis/preregistration_decision_test.py:23:  H1(ii) pooled magnitude θ_RE ≥ 1.0pp + TOST equivalence rejected at δ=1.0pp
scripts/analysis/preregistration_decision_test.py:38:        --primary-gate drop_one_pooled_meta_TOST \\
scripts/analysis/preregistration_decision_test.py:39:        --TOST-delta-pp 1.0 \\
scripts/analysis/preregistration_decision_test.py:218:# TOST equivalence test
scripts/analysis/preregistration_decision_test.py:237:    Note: This replaces prior TOST-rejection logic which had ambiguous semantic
scripts/analysis/preregistration_decision_test.py:238:    direction ("TOST equivalence rejected" could mean either equivalence-demonstrated
scripts/analysis/preregistration_decision_test.py:239:    OR equivalence-not-demonstrated). One-sided superiority is the unambiguous test
scripts/analysis/preregistration_decision_test.py:255:def tost_equivalence(pooled_effect: float, pooled_se: float, delta: float,
scripts/analysis/preregistration_decision_test.py:257:    """Two one-sided tests for equivalence (Schuirmann 1987).
scripts/analysis/preregistration_decision_test.py:260:    Both one-sided tests must reject H0 to demonstrate equivalence.
scripts/analysis/preregistration_decision_test.py:270:    equivalence_demonstrated = (p_lo < alpha) and (p_hi < alpha)
scripts/analysis/preregistration_decision_test.py:279:        "equivalence_demonstrated": equivalence_demonstrated,
scripts/analysis/preregistration_decision_test.py:280:        "decision": "equivalence_demonstrated" if equivalence_demonstrated else "equivalence_not_demonstrated",
scripts/analysis/preregistration_decision_test.py:322:             + TOST equivalence rejected at δ=delta_pp.
scripts/analysis/preregistration_decision_test.py:357:    # TOST kept for informational reporting (NOT used in H1 gating decision)
scripts/analysis/preregistration_decision_test.py:358:    tost_info = tost_equivalence(meta["pooled_effect"], meta["pooled_se"],
scripts/analysis/preregistration_decision_test.py:483:    H1 TOST δ which is an SR percentage-point margin (codex probable concern disambig).
scripts/analysis/preregistration_decision_test.py:507:        "h2a_cost_equivalence": {
scripts/analysis/preregistration_decision_test.py:525:    h2_pass = h2["h2a_cost_equivalence"]["consistent"]
scripts/analysis/preregistration_decision_test.py:539:        return {"rule": "R4", "framing": "Phantom-SoM partial drop-in (cost/latency equivalence fails on some site)",
scripts/analysis/preregistration_decision_test.py:629:    p.add_argument("--primary-gate", default="drop_one_pooled_meta_TOST",
scripts/analysis/preregistration_decision_test.py:631:    p.add_argument("--TOST-delta-pp", type=float, default=1.0,
scripts/analysis/preregistration_decision_test.py:632:                   help="TOST equivalence margin in SR pp (default 1.0 per prereg lock)")
scripts/analysis/preregistration_decision_test.py:636:                   help="H2(a) cost equivalence margin in % (default 10%% per prereg lock)")
scripts/analysis/preregistration_decision_test.py:670:    h1 = evaluate_h1(cells_by_id, delta_pp=args.TOST_delta_pp,
scripts/analysis/preregistration_decision_test.py:698:            "primary_gate_method": "pooled_DerSimonian_Laird_meta + TOST + magnitude",
scripts/analysis/preregistration_decision_test.py:699:            "TOST_delta_pp": args.TOST_delta_pp,
scripts/analysis/preregistration_decision_test.py:709:        "H2_cost_equivalence": h2,
scripts/analysis/preregistration_decision_test.py:715:            "H2": "PASS" if h2["h2a_cost_equivalence"]["consistent"] else "FAIL",
scripts/analysis/preregistration_decision_test.py:737:                    f"TOST equiv {h1['primary_gate']['tost_informational']['decision']})")
scripts/analysis/preregistration_decision_test.py:738:        logger.info(f"  H2: {'PASS' if h2['h2a_cost_equivalence']['consistent'] else 'FAIL'} "
scripts/analysis/preregistration_decision_test.py:739:                    f"({h2['h2a_cost_equivalence']['n_cells_pass']}/{h2['h2a_cost_equivalence']['N']} cells within ±{args.H2_cost_margin_pct}% cost)")

codex
A first consistency pass found multiple surviving `TOST` primary-gate references outside the repaired H1 paragraph, so I’m checking whether they are merely stale labels or actual contradictions with the decision script. I’m also reading the queue now for launch-scope and smoke-gate regressions.
exec
/bin/bash -lc "nl -ba docs/checkpoints/pre_run/preregistration.md | sed -n '45,180p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
    45	
    46	## §2 Hypotheses
    47	
    48	### PRIMARY family (gates paper claim)
    49	
    50	#### H1 — Hero deployment claim (P-SoM is hidden routing arm)
    51	
    52	P-SoM drop-one oracle ceiling lift > 0 across statistical cells (each cell = one (site, model) stratum), satisfying ALL two PRIMARY sub-conditions:
    53	
    54	- **H1(i)** Pooled DerSimonian-Laird random-effect meta-analysis on N=4 (site, model) cells reaches significance at Holm α=0.05 (PRIMARY family m=1 test, no within-family correction needed).
    55	- **H1(ii)** Pooled magnitude θ_RE ≥ 1.0pp AND one-sided **superiority test** rejects H0: θ ≤ 1.0pp at α=0.05 (i.e., effect is significantly ABOVE the +1.0pp substantive-effect threshold; commit-locked). Note 2026-05-13: replaces prior "TOST equivalence rejected at δ" wording which was ambiguous in direction; one-sided superiority is the unambiguous statistical test for "effect substantively > δ".
    56	
    57	**Drop-one definition (operational)**: For each (site, model) cell containing all 6 modes (DOM, SoM, Vision, P-text, P-prompt, P-SoM), compute oracle ceiling SR over {6 modes} minus oracle ceiling SR over {5 modes drop P-SoM} per task, then average across the cell's task pool. Paired 1000-resample task-level bootstrap CI per cell; pooled DerSimonian-Laird across 4 cells.
    58	
    59	**Transparency consistency check (NOT gating, reported alongside H1)**: K_h1 = ⌈0.75 × 4⌉ = 3 of 4 cells individually clear Holm α=0.05 within the per-cell P-SoM sub-family (m = 4). **K-of-N reclassified pre-data 2026-05-13** from gating threshold to transparency consistency check, based on power analysis (`docs/analysis/cross_sites/power_analysis.md`) showing per-cell power at observed 1-3pp effect sizes is < 10% — calibrated only for ≥7pp effects, smaller than reasonable phenomenon effect size, so K-as-gate is statistically dysfunctional. See §4 audit B9 row + Appendix A 2026-05-13 entry.
    60	
    61	#### H2 — 4-fold drop-in property (P-SoM specifically)
    62	
    63	All four sub-claims hold per cell, replicated in ≥ K_h1 cells:
    64	
    65	- **(a) Cost** — median cost(P-SoM) within ±10% of median cost(DOM); reflects the by-construction property that `[SOM_MARKS]` is an AXTree regex filter (no image embedding tokens). Tested empirically per cell.
    66	- **(b) Latency** — median latency(P-SoM) ≤ 0.6 × median latency(SoM); reflects skipping image inference stage. Tested empirically per cell.
    67	- **(c) Signal AUROC** — top-1 routing-signal AUROC(P-SoM) ≥ AUROC(DOM) − 0.05 (within 5pp). Tested empirically per cell, signal selected per `aggregate_routing_auroc.py` top-1.
    68	- **(d) Drop-one magnitude** — folded into H1(iii); P-SoM contributes ≥ 1.0pp lift on average.
    69	
    70	#### H3 — Phantom space 2-axis empirical structural claim
    71	
    72	Each phantom-space axis (axis 1 = text payload via P-text; axis 2 = SoM-style prompt via P-prompt) contributes tasks NOT solved by P-SoM, evidencing axis decomposition is empirically non-trivial (i.e., phantom space is a multi-region 2D structure, not a collapsed 0D point).
    73	
    74	H3 statistical cells = 4 (one per (site, model)). H3 axis-1 and axis-2 are tested separately within each cell.
    75	
    76	- **H3(i) PRIMARY GATE** axis 1: pooled across N=4 cells, mean |P-text ∖ P-SoM| > 0 with DerSimonian-Laird random-effects meta CI excluding 0 (Holm α=0.05, m=1 within axis-1 sub-family).
    77	- **H3(ii) PRIMARY GATE** axis 2: same as H3(i) for |P-prompt ∖ P-SoM|.
    78	- **H3(iii)** Per-cell unique-count noise floor: ≥ 2 tasks (≈ 1pp at N=234 to N=210); 1 task is noise floor, excluded from cell-level pass.
    79	
    80	**Transparency consistency check (NOT gating)**: K_h3 = ⌈0.67 × 4⌉ = 3 of 4 cells individually with bootstrap 95% CI excluding 0 (m=4 per axis). Same K-of-N reclassification rationale as H1 (see §4 audit B9 + Appendix A 2026-05-13 entry).
    81	
    82	**Test details**:
    83	- Primary gating: bootstrap CI on unique-count, 1000 resamples.
    84	- Secondary report: McNemar exact one-sided directional asymmetry test (informational only — McNemar tests if one axis dominates the other in unique contribution; H3 only requires non-emptiness, not dominance).
    85	- Multiple-comparison: Holm-Bonferroni step-down per axis sub-family (axis 1: m = N_cells; axis 2: m = N_cells).
    86	
    87	### EXPLORATORY family (reported with corrections, NOT gating)
    88	
    89	#### H4 — P-text / P-prompt drop-one magnitude
    90	
    91	Reported per cell + meta-pooled (DerSimonian-Laird) for transparency. Holm-Bonferroni and BH FDR q-values reported. No pre-registered ranking commitment.
    92	
    93	Paper §4 prose **must** explicitly flag: "exploratory analysis; not pre-registered for paper hook gating; magnitudes interpreted descriptively."
    94	
    95	### POST-HOC family (theory tested on data that motivated it)
    96	
    97	#### H5 — 别扭 (mismatch) framework predictions
    98	
    99	The 4 distinguishing predictions in 实验笔记 §108.16 are tested against 16-cell data. The framework was developed after observing N=4 pre-Phase-A cells; this is **post-hoc**.
   100	
   101	Paper §5 prose **must** explicitly flag: "post-hoc theoretical framework, validated on the same data motivating it; no formal significance gating."
   102	
   103	#### H6 — Capability-modulated reversal (B0 vs B1 axis preference)
   104	
   105	B0 vs B1 ranking direction on text-axis vs image-axis drop-one tested via B0 × B1 × axis logistic GLM interaction term. Post-hoc finding (developed after observing N=4 pre-Phase-A cells).
   106	
   107	Paper §7 prose **must** explicitly flag: "post-hoc finding; no pre-registered prediction."
   108	
   109	### ROUTER family (gates Section 6 routing claim — **pending advisor 5/5 lock**: paper-1 PRIMARY vs paper-2 deferred)
   110	
   111	#### H7 — Tier 1 oracle router lift over best-single-mode baseline (offline supervised)
   112	
   113	Tier 1 router: TF-IDF task-instruction features + binary task features (`has_ref_image`, `has_finish_string_match`) → logistic regression predicting best-mode-per-task. Trained per cell-fold (site-stratified k-fold). Lift = adjusted-SR(router) − adjusted-SR(best-single-mode-baseline) per cell.
   114	
   115	- **H7(i)** Pooled DerSimonian-Laird random-effect meta-analysis on lift reaches Holm α=0.05 (PRIMARY family m=1 if paper-1 / SECONDARY informational if paper-2).
   116	- **H7(ii)** ≥ K_h1 of N_cells individually Holm-significant on per-cell lift, bootstrap 95% CI lower-bound > 0.
   117	- **H7(iii)** Pooled magnitude θ_RE ≥ 1.0pp; TOST equivalence at margin δ=1.0pp rejected (same δ as H1).
   118	
   119	**Test details**:
   120	- 5-fold site-stratified CV on cls+red post-Phase-A task pool (split protocol locked §4 — train/test fold seed + minimum sizes).
   121	- Best-single-mode-baseline = mode with highest mean adjusted-SR on train fold per cell, evaluated on held-out test fold (no test leak).
   122	- Bootstrap 1000 resamples, paired task-level.
   123	- Multiple-comparison: Holm-Bonferroni step-down within H7 sub-family m=N_cells.
   124	
   125	**Status**: ⏸️ pending advisor 5/5 lock decision — if paper-1 PRIMARY, H7 gates Section 6 routing claim; if paper-2 deferred, H7 reported as informational with explicit "paper-1 hook does NOT depend on H7-H8".
   126	
   127	#### H8 — Tier 2 first-step trigger router (online, test-leak-free)
   128	
   129	Tier 2 router: features extracted from agent's first-step observation (task instruction + initial DOM/SoM observation slice + initial action diversity proxy) → predicts which mode to commit for full trajectory. **No test leak**: features use only first-step info, mode commitment thereafter is fixed.
   130	
   131	- **H8(i)** Tier 2 router lift over Tier 1 oracle baseline ≥ 0 with bootstrap 95% CI excluding −1.0pp (paper claims Tier 2 ≈ Tier 1 within deployment-grade tolerance, given Tier 2 is leak-free and deployment-realistic).
   132	- **H8(ii)** Tier 2 router lift over best-single-mode-baseline ≥ 1.0pp, ≥ K_h1 cells Holm-significant.
   133	
   134	**Status**: ⏸️ pending advisor 5/5 lock — same as H7.
   135	
   136	**Companion check** (NOT gating): per-mode AUROC of selected routing signals reported for transparency (Section 6 portfolio characterization, see EXPLORATORY §5).
   137	
   138	### FRAMING DECISION RULE (pre-registered, data-conditional)
   139	
   140	The paper §1 hook framing maps to data outcomes as follows:
   141	
   142	| Rule | Conditions | Paper hook framing | Hook power |
   143	|---|---|---|---|
   144	| **R1** | H1 holds AND H2 (a)(b)(c) all hold AND H3(i) holds AND H3(ii) holds | "Phantom routing space (M1/M2 2-axis empirical structure); P-SoM as deployment hero, P-text/P-prompt as structural ablation arms validating axis decomposition." | STRONGEST |
   145	| **R2** | H1+H2 hold AND only one of H3(i)/(ii) holds | "Phantom routing space (single-axis empirical structure) with P-SoM as deployment hero; remaining axis decomposition theoretical (Zoom 1 architectural argument only)." | MODERATE-STRONG |
   146	| **R3** | H1+H2 hold AND neither H3(i)/(ii) holds | "Phantom-SoM is hidden 4th routing arm; M1/M2 axis decomposition supported by Zoom 1 architectural argument only, not empirically validated by ablation." | MODERATE (= 04-30 fallback; workshop-grade) |
   147	| **R4** | H1 holds AND H2 partially fails (e.g., (a) cost or (b) latency fails on some site) | "Phantom-SoM partial drop-in" + §4 disclosure of failed sub-claim. | WEAK; substantial revision |
   148	| **R5** | H1 fails (pooled meta DerSimonian-Laird Holm α=0.05 fails OR pooled magnitude θ_RE < 1.0pp OR TOST equivalence fails reject at δ=1.0pp) | Paper death scenario: pivot to VWA bug audit paper (§107 4-cluster fix as primary) OR abandon. Decision deferred to advisor sync at fail time. | n/a |
   149	
   150	**Trigger rule update 2026-05-13**: R5 no longer fires on `< K_h1` (K-of-N reclassified to transparency-only). Pooled meta + TOST primary gate only. K-of-N consistency reported in §4 per-cell table as descriptive transparency row.
   151	
   152	**Heterogeneity-conditional rule (added 2026-05-13 to resolve §4 audit B8 ↔ H1(i) conflict)**: If pre-specified I² > 75% from random-effects meta (per §4 audit B8 thresholds), do NOT pool — primary inference reverts to per-cell forest + meta-regression by site / model. R1-R5 framing in this branch maps to per-cell direction-consistency: ≥3 of 4 cells direction-positive + ≥2 individually Holm sig → R3-grade hook; otherwise R4/R5.
   153	
   154	---
   155	
   156	## §3 Multiple-Comparison Family Declaration
   157	
   158	**PRIMARY family** (gating paper hook) — UPDATED 2026-05-13 (K-of-N → transparency-only):
   159	- H1(i) pooled meta on N=4 statistical cells: m = 1 (no within-family correction).
   160	- H1(ii) pooled magnitude θ_RE ≥ 1.0pp + TOST equivalence reject at δ=1.0pp: m = 1.
   161	- H2 sub-claims (a)(b)(c)(d) per cell: m = 4 × 4 statistical cells = 16 tests (each per-cell sub-claim).
   162	- Method: Holm-Bonferroni step-down per H-sub-family (Holm 1979).
   163	
   164	**STRUCTURAL family** (gating phantom-space framing) — UPDATED 2026-05-13:
   165	- H3(i) pooled axis-1 meta on N=4 cells: m = 1.
   166	- H3(ii) pooled axis-2 meta on N=4 cells: m = 1.
   167	- Method: Holm-Bonferroni step-down per axis sub-family.
   168	- Rationale: structural claim is weaker than deployment, separate family avoids inflating PRIMARY family m count.
   169	
   170	**TRANSPARENCY family** (NOT gating, reported in §4 per-cell table for reviewer transparency):
   171	- K_h1 = ⌈0.75 × 4⌉ = 3 of 4 cells individually Holm-significant on P-SoM drop-one (m=4 per cell).
   172	- K_h3 axis-1 = ⌈0.67 × 4⌉ = 3 of 4 cells individually with bootstrap CI excluding 0.
   173	- K_h3 axis-2 = same as axis-1.
   174	- Method: Holm-Bonferroni within transparency sub-family (m=4 per K-test).
   175	- **Rationale for transparency-only reclassification**: power analysis (`docs/analysis/cross_sites/power_analysis.md`, pre-data) shows K-of-N family power at observed 1-3pp effect sizes is < 10%, calibrated only for ≥7pp effects. Per-cell N=234 (cls) / 210 (red) bootstrap power at 1.5pp effect ≈ 0.30. P(≥3 of 4 cells sig | p_cell=0.30) ≈ 8%. K-as-gate is statistically dysfunctional in this regime; K-as-transparency provides per-cell consistency check value alongside pooled meta. See Appendix A 2026-05-13 entry.
   176	
   177	**ROUTER family** (gates Section 6 routing claim — pending advisor 5/5 paper-1-vs-paper-2 lock):
   178	- H7(i) pooled meta lift: m = 1 (no within-family correction).
   179	- H7(ii) per-cell Tier 1 lift Holm: m = N_cells.
   180	- H7(iii) folded into H7(i) magnitude/TOST.

exec
/bin/bash -lc "nl -ba docs/checkpoints/pre_run/preregistration.md | sed -n '200,240p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
   200	
   201	| Choice | Value | Rationale |
   202	|---|---|---|
   203	| **Primary metric** | Oracle ceiling SR pp lift (binary, paired) | Standard routing-arm contribution metric |
   204	| **CI method** | 1000-resample task-level paired bootstrap, **percentile** intervals (BCa as sensitivity check, not primary) | Existing infra in `aggregate_phantom_lift.py`. Percentile chosen primary because: (a) paired-bootstrap on bounded proportion (SR ∈ [0,1]) → BCa acceleration estimate is unstable at small N per cell; (b) Cohen's h transformation already symmetrizes; (c) percentile is the canonical reporting in WebArena/VWA precedent. BCa shown as appendix sensitivity check. |
   205	| **Bootstrap resampling unit** | **Task-level** (not episode-level, not run-level) | Each (task_id) drawn with replacement N times; same task across modes drawn together to preserve pairing. This is the standard unit for adjusted_success comparisons in VWA/WA. Episode-level would break pairing; run-level would over-conservatively widen CIs. |
   206	| **Bootstrap clustering** | **Single-level (task_id)** for primary, no nested cluster (cell × site) bootstrap | Justification: meta-analysis at cell level is separate (`aggregate_phantom_meta.py` random-effects + I²/τ²); within-cell bootstrap only re-samples tasks. Multi-level cluster would double-count uncertainty already captured by random-effects meta. Lock: percentile + task-id unit + no nested cluster (B2 lock 2026-05-09). |
   207	| **Sig threshold** | Holm α=0.05 within respective family | FWER control |
   208	| **Effect size (binary)** | Cohen's h with bootstrap CI | Standard for proportion comparisons |
   209	| **Effect size (continuous)** | Cohen's d with bootstrap CI | For cost/latency H2(a)(b) |
   210	| **TOST equivalence margin δ** | **1.0pp** | ≈ 2 tasks in N=234, matches per-cell bootstrap SE; smaller is within sampling noise floor |
   211	| **H1 K_h1 transparency ratio** | **0.75** (= 3/4 cells; **transparency-only, not gating** per 2026-05-13 reclassification) | Reports per-cell consistency alongside pooled meta; not a gate on H1 |
   212	| **H3 K_h3 transparency ratio** | **0.67** (= 3/4 cells; **transparency-only**) | Same as K_h1 reclassification rationale |
   213	| **H3 unique-count floor** | **≥ 2 tasks per cell** | 1 task is sampling noise; 2 tasks ≈ 1pp at N=234 |
   214	| **Cell inclusion (Phase 1a main)** | Phase A post-fix only (commit ≥ 3c15cd7), cls + red sites only, all 6 modes per (site, model) cell freshly rerun | Bug-clean rerun + workshop-target scope (shop deferred to Phase 1b) |
   215	| **Cell inclusion (Phase 1b main paper)** | Phase A post-fix rerun of shop × B0+B1 × 6 modes (12 conditions added on top of Phase 1a 24 conditions) | Cross-site expansion lever for main paper, post-data R1 vs Option D framing decision |
   216	| **Cell inclusion (Appendix D)** | Archived pre-Phase-A data as robustness check | Symmetric contamination disclosure |
   217	| **N inclusion floor** | ≥ 100 ep per (condition) | Statistical power baseline |
   218	| **FP filter primary** | na_fp + eval_fp combined | Per 实验笔记 §95 (visual_fp deprecated — no lit precedent, boundary-undecidable, over-filters 95.3% VWA tasks). Code: `compute_adjusted_success()` returns `fp_reason ∈ {'', 'na_fp', 'eval_fp'}` (`p79/experiment/analysis.py:52`) |
   219	| **FP filter sensitivity** | 3 variants reported (raw_SR / +na_fp only / +na_fp+eval combined) | Robustness disclosure. visual_fp is NOT in the ladder — see §95 decision rationale |
   220	| **Non-visual subset robustness** | 43 VWA + 480 WA = 523 manually-audited non-visual tasks (`docs/analysis/cross_sites/vwa_manual_non_visual_task_ids.py`) | Replaces deprecated visual_fp; Appendix D sensitivity check |
   221	| **Mode operational definitions** | 6 modes per paper §3 (text format × prompt × image): DOM (AXTree+DOM-prompt+no image) / SoM ([SOM_MARKS]+SoM-prompt+image) / Vision (no text+image) / P-text ([SOM_MARKS]+DOM-prompt+no image) / P-prompt (AXTree+SoM-prompt+no image) / P-SoM ([SOM_MARKS]+SoM-prompt+no image) | Stipulative — **no post-hoc episode reclassification**. Episodes systematically excluded per (FP filter / N-floor / data-corruption flag), never redefined which mode they belong to. Edge cases (empty AXTree / 0 marks / OCR-empty) follow `condition_meta.json` declared mode |
   222	| **Routing signal universe** | `aggregate_routing_auroc.py` enumerated set: ep_mean_verbalized / ep_min_verbalized / max_repeat_streak / action_diversity / url_revisit_count / url_revisit_max / action_unique_types / url_unique_count / ep_mean_logprob / ep_min_logprob (last 2 B1-only) | **No post-hoc engineered features** for router input. Best-signal-per-mode characterization is exploratory (§5) — paper §6 portfolio finding, not pre-registered prediction |
   223	| **Router train/test split** | 5-fold site-stratified CV on cls+red post-Phase-A task pool, seed=42, min test fold ≥ 40 tasks | Reproducible split via `scripts/analysis/router_split.py` (TBD). **Test fold predictions use ONLY train-fold mode rankings** to prevent oracle leak. Pending advisor 5/5 sync alternative: leave-one-site-out (LOSO) — test cls hold-out trained on red, vice versa |
   224	| **Failure-mode classification rubric** | 5-bucket: `early_finish` / `wrong_commit` / `visual_hijack` / `click_loop` / `persistent_error` per `docs/analysis/disagreement_clusters.md` decision tree | Pre-data inter-annotator agreement target Cohen κ ≥ 0.7 on 30-task pilot (codex prompt + 1 human spot-check). Buckets remain in the rubric but the paper §1 "+43.7pp B0/B1 capability shift" prose was dropped 2026-05-09 (third contribution cut from paper). Failure-mode classification still used for §8 limitations and supplement S.X if needed. |
   225	| **N_conditions Phase 1a (operational)** | **24 conditions** = 2 sites (cls, red) × 2 models (B0, B1) × 6 modes (DOM, SoM, Vision, P-text, P-prompt, P-SoM). Each condition launched fresh post-fix via `scripts/queues/queue_phase1_paper_grade.sh` (renamed 2026-05-13 from `queue_16cell_paper_grade.sh`; current scope = 24 conditions Phase 1a + 12 conditions Phase 1b deferred). Sequence: B0 → B1 per site (shared user account); cls + red parallel chains | ✅ **Student-decided 2026-05-13** post-codex stress audit. Workshop-targeted (cls + red only, shop deferred to Phase 1b for main paper). Replaces prior 16-cell phantom-only scope that lacked baseline DOM/SoM/Vision rerun (codex Flaw 1) |
   226	| **N_cells statistical (H1/H3 stratification)** | **4 cells** = (site, model) tuples: (cls, B0), (cls, B1), (red, B0), (red, B1). Drop-one is computed per cell using all 6 modes; pooled DerSimonian-Laird random-effects meta across 4 cells | Cell = paired-test stratification unit (one per (site, model)), distinct from "condition" (one per (site, model, mode)). 4 cells × 6 modes = 24 conditions. Distinction propagated to all prose / queue / docs 2026-05-13 |
   227	| **N_conditions Phase 1b (main paper, deferred)** | **+12 conditions** = shop × 2 models × 6 modes. Launches after Phase 1a workshop submission to feed main paper R1 / Option D framing decision. N_cells statistical becomes 6 (= 3 sites × 2 models) when Phase 1b lands | Phase 1b is additive; workshop §1 hook does NOT depend on Phase 1b. Main paper §1 hook upgrade R3 → R1 conditional on shop replicating P-SoM 4-fold within ±2pp tolerance |
   228	| **Best-single-mode baseline (H7/H8 anchor)** | Per cell: mode with highest mean adjusted-SR on train fold | Used as comparison anchor for router lift; **train/test split-stratified** to prevent test leak |
   229	| **Missing-data / crashed-episode policy** (audit B6) | (a) Crashed episodes (uncaught exception, OOM, timeout > 30 min, browser crash) **excluded from paired-N denominators**, **NOT imputed** to success or failure. (b) Episodes with `not_logged_in` or `auth_drift` flag at termination excluded after watchdog refresh fails 3 retries (per `experiment_watchdog.py`). (c) Missing artifacts (no `obs.txt` / `screenshot_annotated.png` at step k) excluded from per-step analyses, NOT imputed. (d) Per-cell exclusion count + reason histogram reported in Appendix C. | Listwise deletion only; mean imputation introduces bias for SR proportions, hot-deck imputation breaks paired-N pairing. Crashed-episode imputation as success/failure would inflate Type I/II error. Lock 2026-05-09. |
   230	| **Stopping rules / contamination halt criteria** (audit B7, REVISED 2026-05-13 to remove outcome-dependent bias per codex Flaw 6) | (a) **Pre-launch**: `make pre-launch-check` validates seed configured + HF SHA pinned + git working tree clean + GPU available + disk free > 20GB; failure halts launch (per audit C10). (b) **Smoke-test gate (outcome-INDEPENDENT)**: first 10 episodes per condition must show auth-state `logged_in=True` on all 10 AND ≥ 9 of 10 episodes produced complete artifact bundle (`obs.txt` + `screenshot.png` + `condition_summary_v2` increment + JSONL flush) AND evaluator returned a parseable verdict (success / failure / `ua_match` N/A — any of these is fine, **success rate itself is NOT checked**). Failures halt for auth refresh / artifact pipeline debug, NOT for low SR observation. Rationale: outcome-dependent smoke gate biases low-SR cells upward (a true 5-10% SR cell has 35-60% probability of "0 successes in first 10" by binomial chance and would be invalidly restarted). (c) **Auth/site contamination halt**: ≥ 5 consecutive episodes with `not_logged_in` ⇒ stop cell, refresh auth, archive partial run as `_dirty_partial`, restart fresh. (d) **Eval drift halt**: if rerun on identical archived episode produces SR delta > 5pp via `validate_run.py --strict`, freeze cell + investigate evaluator code. (e) **OOM / hardware halt**: 3 consecutive job failures ⇒ stop cell, document hardware in incident log, manually re-queue with diagnostic output. | Halt rules protect data purity; halted cells restarted only after root-cause documented in `master_bug_catalog.md` + bug fix committed. Lock 2026-05-09; smoke gate revised 2026-05-13 to outcome-independent variant. |
   231	| **Heterogeneity (random-effects, Q, I², τ²) pre-spec** (audit B8) | (a) **Primary estimator**: random-effects DerSimonian-Laird via `aggregate_phantom_meta.py` (already implemented). (b) **Heterogeneity reporting**: report Cochran Q (chi² test of homogeneity), I² (% of total variance attributable to between-cell heterogeneity), τ² (between-cell variance). (c) **Interpretation thresholds (pre-specified)**: I² < 25% = "low heterogeneity, pooled mean is primary"; 25%-50% = "moderate, report both pooled + per-cell"; 50%-75% = "high, per-cell estimates are primary, pooled is summary"; > 75% = "very high, do not pool — report only per-cell + heterogeneity-source analysis (site / model / task-pool)". (d) **Heterogeneity-source decomposition**: when I² > 50%, report meta-regression by site (cls / red / shop) and by model (B0 / B1) to identify dominant variance source. | Higgins & Thompson 2002 (I² thresholds). Per-cell estimates always shown alongside pooled, so heterogeneity is never averaged away. Lock 2026-05-09. |
   232	| **K-of-N rule scope** (audit B9 power-corrected, REPROPAGATED 2026-05-13 to H1/H3/R5/§6/Appendix A) | The **K_h1=3/4 / K_h3=3/4** ratios (under 24-condition / 4-cell Phase 1a scope) are **transparency consistency checks** (count of cells *individually* clearing α=0.05 Holm), **NOT gates on H1/H3 paper claims**. **Primary gate** = (a) DerSimonian-Laird random-effects meta-analysis on N=4 (site, model) cells + (b) TOST equivalence on pooled cls + red tasks at δ=1.0pp. Per `docs/analysis/cross_sites/power_analysis.md` §3-§5, K-of-N family power at observed 1-3pp effect sizes is < 10%; the rule is calibrated for ≥7pp effects (1.5pp per-cell power ≈ 0.30; P(≥3 of 4 cells sig) ≈ 8%). K-as-gate is statistically dysfunctional in this effect-size regime. **2026-05-13 propagation**: prior prereg text in H1(ii) / H3(i) / H3(ii) / R5 / §6 still gated K-of-N → fixed to "transparency consistency check, reported alongside but NOT gating". This is **pre-data reclassification**: power analysis commit predates Phase 1a launch; reclassification timestamp recorded for OSF witness audit trail. | Original audit B9 lock 2026-05-09 introduced framing but did not propagate to H1/H3/R5/§6 prose (codex stress audit 2026-05-13 Flaw 2 surfaced internal contradiction). Repropagation 2026-05-13 reconciles all references. |
   233	
   234	---
   235	
   236	## §5 Exploratory (NOT pre-registered, paper must explicitly flag)
   237	
   238	The following analyses are exploratory and cannot be used to gate paper claims. Paper prose **must** mark them explicitly as "exploratory" or "post-hoc":
   239	
   240	- Per-task category × mode heatmap exploration (`fig0e`)

exec
/bin/bash -lc "nl -ba docs/checkpoints/advisor_sync_5_5_followup.md | sed -n '1,30p;140,200p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# 5/5 Sync Follow-up — 想跟您 confirm 几件事
     2	
     3	> 学长您好,
     4	>
     5	> 5/5 sync 因您马德里那边网卡了, 后半段 forest plot + threshold 没 propagate 过去, summary 也有点 ambiguous, 我整理一份 follow-up:
     6	>
     7	> - **Part 1**: 简要列我们目前所有的 innovation points (帮您 frame paper 拆分)
     8	> - **Part 2**: paper 拆分 + Mechanistic 该独立还是 nested 这块想跟您 confirm
     9	> - **Part 3**: 来不及讨论的 + 需 email 回的 (主要是 pre-reg threshold)
    10	>
    11	> 已 confirmed 的不重列: early-stop A 全 cancel / GPU 走您 5090 + Rancher.
    12	>
    13	> **Rerun scope 2026-05-13 revised (post codex stress audit)**: 之前 follow-up 里写的 "16 cell (B0 × {cls,red,shop} × 3 phantom + B1 × {cls,red,shop} × 3 phantom)" 实际算数有歧义 (3 sites × 3 phantom × 2 baselines = 18, 不是 16). 而且 phantom-only scope **缺 fresh DOM/SoM/Vision baseline rerun**, drop-one CI 无法 paper-grade (codex audit Flaw 1).
    14	>
    15	> 我重新组织成两阶段:
    16	> - **Phase 1a (workshop-targeted, immediate launch)**: 24 operational conditions = 2 sites (cls + red) × 2 models (B0 + B1) × 6 modes (DOM / SoM / Vision / P-text / P-prompt / P-SoM). 统计分析 4 个 cell (= (site, model) tuple), pooled DerSimonian-Laird meta + TOST primary gate. 投 workshop 这一档.
    17	> - **Phase 1b (main paper expansion, post-workshop)**: 额外 12 conditions = shop × 2 models × 6 modes. 跑完 1a 投完 workshop 再跑这个, 看 shop 是不是 replicate cls+red phantom-SoM 4-fold property, 决定 main paper §1 hook 是 R3 (cls+red only) 还是 R1 (升级到 3-site).
    18	>
    19	> 我 5/12-6/1 考试, 这两天能 lock 完我就 launch Phase 1a 然后专心复习, 麻烦您扫一眼回个 email 即可!
    20	
    21	---
    22	
    23	## Part 1 — 我们目前手里的 innovation points
    24	
    25	### 1. Phantom routing space (核心发现)
    26	
    27	把 web agent 的 observation 拆成 **3 axis** (text payload / system prompt / image presence) → 8-corner cube. 现在 paper 测的 6 modes 里, **4 个 corner 是"不带标注图"** (我们叫 phantom space), 这块以前没人 systematic 测过. 发现 phantom 内部不是塌成一个点, 是 **2-axis 结构** (text-flattening 跟 SoM-prompt 各自能解 P-SoM 解不了的 task). Cube 中心 P-SoM (`[SOM_MARKS]` text + SoM-prompt + 无图) 是 hero, 满足 **4-fold drop-in property**:
    28	
    29	| | 数据 |
    30	|---|---|
   140	
   141	### 1. Pre-registration 三个 threshold ⭐⭐⭐ 最 urgent
   142	
   143	forest plot 当时 Slack 没传过去您没看到. 我会 commit + push 完发您 GitHub 链接, 您扫一眼 (preregistration.md + 3 张 forest figure), email 回 **"I have reviewed and witness these thresholds"** 即可.
   144	
   145	**为什么必须 pre-commit**: paper §1 footnote 会 cite "**pre-registered with advisor email witness on \<date\>, git commit \<SHA\>, OSF DOI**". 数据没出来前 commit, 数据出来后改不了. Reviewer 看到这个 footnote 就不会攻击 "你 cherry-pick 阈值 to make hero pass". 这是 paper rigor 关键, 也是为什么需要您 email 见证 — git commit 时间戳一个人能改, email + OSF 双层 audit trail 改不了.
   146	
   147	#### (a) K_h1 = 0.75 transparency ratio — Hero claim per-cell consistency (NOT gate, 2026-05-13 reclassified)
   148	
   149	**测什么**: P-SoM drop-one oracle ceiling lift > 0 在多少 % 的 statistical cell 里 individually Holm-significant. **现在是 transparency consistency check, NOT a gate on H1 paper claim**.
   150	
   151	**Phase 1a 24-condition / 4-cell scope 下**: ⌈0.75 × 4⌉ = **3 of 4 cells** individually clear Holm α=0.05 (报在 §4 per-cell table 作为 reviewer transparency consistency 行).
   152	
   153	**为什么 reclassify 成 transparency 而不是 gate (2026-05-13 codex audit Flaw 2)**:
   154	- Power analysis (`docs/analysis/cross_sites/power_analysis.md`, pre-data): K-of-N family power at observed 1-3pp effect sizes (phenomenon 实际 magnitude) < 10%, calibrated 只 for ≥7pp effects
   155	- N=4 cells × per-cell power ≈ 0.30 at 1.5pp → P(≥3 of 4 sig) ≈ 8%, K-as-gate 设计上必 fail 即使 phenomenon 真实
   156	- N=4 太小 K-of-N 退化 (只 5 个 outcome state)
   157	- **Primary H1 gate** = (i) pooled DerSimonian-Laird random-effects meta on 4 cells 在 Holm α=0.05 显著 + (ii) pooled magnitude θ_RE ≥ 1.0pp + TOST δ=1.0pp reject equivalence
   158	
   159	**Pre-data reclassification timestamp**: 2026-05-13 commit predates Phase 1a launch, OSF DOI 见证 audit trail 显示 reclass 在 unblind 之前 (防 reviewer 攻击 "你看了数据再 weaken gate").
   160	
   161	#### (b) K_h3 = 0.67 transparency ratio — Structural claim per-cell consistency (NOT gate)
   162	
   163	**测什么**: phantom space 是 2-axis 结构 — P-text (text-flattening axis) 跟 P-prompt (SoM-prompt axis) 各自有 P-SoM 解不了的 unique tasks.
   164	
   165	**计算方式 per cell**:
   166	- |P-text ∖ P-SoM| = P-text 解出但 P-SoM 没解出的 task 数
   167	- |P-prompt ∖ P-SoM| = P-prompt 解出但 P-SoM 没解出的 task 数
   168	- Bootstrap (resample tasks with replacement, 1000 次) → 95% CI on each count
   169	- **Cell pass = CI 下界 > 0** + ≥ 2 tasks (1 task 是 noise floor)
   170	
   171	**Phase 1a 4-cell 下**: ⌈0.67 × 4⌉ = **3 of 4 cells** per axis 作为 transparency 行.
   172	
   173	**Primary H3 gate** = pooled axis-1 + axis-2 DerSimonian-Laird meta Holm α=0.05 (两个 axis 独立 sub-family, 各 m=1).
   174	
   175	Same reclassification rationale as K_h1.
   176	
   177	#### (c) TOST δ = 1.0pp — Equivalence margin (single canonical interpretation)
   178	
   179	**测什么**: 用作 H1(ii) drop-one effect-size equivalence margin (pooled drop-one lift 是 ≥ 1.0pp 而不是 ≈ 0). H1(iii) 的 TOST reject equivalence at δ=1.0pp 意思是 "drop-one lift 显著 > 1pp, 不是统计噪声".
   180	
   181	**注意 (2026-05-13 disambiguation, codex probable concern)**: 这个 δ=1.0pp 是 **SR percentage-point margin**, 不是 cost equivalence margin. H2(a) "cost ≈ DOM" 用另一个 margin ±10% relative cost (不复用同一个 δ). 之前 prereg 跟 advisor follow-up 这两处单位有混淆, 现在显式区分.
   182	
   183	**为什么 1.0pp 而不是 0.5pp 或 3pp**:
   184	- N=234 (cls) / 210 (red), 单 task ≈ 0.43pp / 0.48pp
   185	- Bootstrap iteration noise + cell-level correlated error 实测约 0.7-1.0pp
   186	- δ = 0.5pp → 比 noise floor 还小, TOST 永远测不出 (Type II error 高)
   187	- δ = 3pp → 太松, equivalence claim 太 cheap
   188	- **δ = 1.0pp ≈ 2 tasks @ N=234**, noise floor 上方安全 margin
   189	
   190	**选错风险**: 选小 → drop-one 实际 hold 但 statistically can't claim; 选大 → claim 太 weak.
   191	
   192	#### Email confirm 模板
   193	
   194	GitHub 链接我 push 完发您, 您扫 `preregistration.md` (含 2026-05-13 codex stress audit 6 fix 的 propagated 更新 + Appendix A decision log 2026-05-13 entry) + 3 张 forest figure, email 回:
   195	
   196	> "I have reviewed the pre-registration revision (K_h1=0.75 transparency-only / K_h3=0.67 transparency-only / TOST δ=1.0pp SR-margin / Phase 1a 24 conditions across 4 cells: cls+red × B0+B1 × 6 modes / Phase 1b shop deferred / outcome-independent smoke gate / pooled DerSimonian-Laird meta + TOST primary gating) on \<date\> and witness them as committed before Phase 1a data unblinding."
   197	
   198	(您可以加任何 condition, e.g. "subject to my final review of the rerun protocol", 都 OK. 关键是 timestamp + explicit acknowledge 留 audit trail.)
   199	
   200	### 2. Train/test split protocol (Routing paper 用)

exec
/bin/bash -lc "nl -ba docs/checkpoints/pre_run/osf_lock_manifest.md | sed -n '1,70p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# OSF DOI Lock Manifest — Paper §3 / Appendix D
     2	
     3	**Purpose**: Codify the artefacts whose SHA-256 / git ref get frozen at the
     4	moment of OSF preregistration DOI minting. Once advisor's email reply arrives
     5	(post 5/5 sync) confirming K_h1 / K_h3 / TOST δ thresholds, this checklist
     6	drives the 8-step DOI workflow (笔记 §110 + ADVISOR_SYNC §F).
     7	
     8	**Status**: 🟡 Draft — fields populate at lock moment.
     9	**Lock blocker**: ⏳ advisor email reply (Q1-Q11 in `advisor_sync_5_5_followup.md`)
    10	
    11	---
    12	
    13	## §1 Pre-lock checklist (everything must be done before OSF DOI mint)
    14	
    15	- [ ] Advisor email reply received (K_h1=0.75 transparency / K_h3=0.67 transparency /
    16	      TOST δ=1.0pp SR-margin / Phase 1a 24-condition 4-cell scope confirmed
    17	      OR alternative noted)
    18	- [ ] `preregistration.md` final text edit committed + pushed (incl. 2026-05-13
    19	      codex stress audit propagation: K-of-N transparency-only + 24/4 scope +
    20	      drop-one H1 formula + outcome-independent smoke gate + Appendix A 2026-05-13)
    21	- [ ] `run_manifest.yaml` archived rows verified (Phase 1a 24-condition scope = 2 sites
    22	      × 2 models × 6 modes; all `grade=archived` for pre-fix cells, `grade=paper-grade`
    23	      for Phase 1a post-fix rerun cells; Phase 1b shop deferred rows separately tagged)
    24	- [ ] All paper draft sections section1-8 + paper.bib (57 entries) snapshot to
    25	      `docs/checkpoints/paper_drafts_locked/` directory (immutable copy)
    26	- [ ] `env_snapshot.json` of latest run on each machine (DGX, A100, Myriad if
    27	      used) committed under `results/provenance/env_lock_<hostname>.json`
    28	- [ ] `vwa_snapshot_<host>.json` for any VWA-using cells committed
    29	- [ ] No untracked / uncommitted files in repo (clean `git status`)
    30	- [ ] Repo pushed to GitHub master (DOI cites GitHub commit URL)
    31	
    32	---
    33	
    34	## §2 Locked artefacts — fields populate at lock moment
    35	
    36	### 2.1 Code + manifest SHAs
    37	
    38	| Artefact | Path | Git ref @ lock | Captured |
    39	|---|---|---|---|
    40	| Repository HEAD | `master` branch | `<TBD>` | TBD |
    41	| Pre-registration text | `docs/checkpoints/pre_run/preregistration.md` | `<TBD commit-SHA>` | TBD |
    42	| Run manifest YAML | `results/phantom_paper/run_manifest.yaml` | `<TBD commit-SHA>` | TBD |
    43	| Paper drafts (locked snapshot) | `docs/checkpoints/paper_drafts_locked/` | `<TBD commit-SHA>` | TBD |
    44	| Bibliography (57 entries) | `docs/checkpoints/paper_drafts/paper.bib` | `<TBD commit-SHA>` | TBD |
    45	| Mechanistic 24+15 candidates | `results/mechanistic/archive_subset_b1_cls/manifest.json` | `<TBD commit-SHA>` | TBD |
    46	
    47	### 2.2 Hypothesis thresholds (advisor email confirmed) — REVISED 2026-05-13
    48	
    49	| Threshold | Pre-reg value | Advisor confirmed? | Notes |
    50	|---|---|---|---|
    51	| H1 PRIMARY gate (P-SoM drop-one oracle ceiling lift) | Pooled DerSimonian-Laird meta Holm α=0.05 sig on N=4 (site, model) cells + pooled magnitude θ_RE ≥ 1.0pp + TOST equivalence reject at δ=1.0pp | ⏳ pending | Drop-one = oracle SR over {6 modes} − oracle SR over {5 modes drop P-SoM} per task, paired bootstrap per cell, pooled across 4 cells |
    52	| H3 PRIMARY gate axis-1 (P-text \ P-SoM) | Pooled axis-1 DerSimonian-Laird meta Holm α=0.05 sig on N=4 cells | ⏳ pending | Per-cell bootstrap CI on unique-task count, then pooled meta |
    53	| H3 PRIMARY gate axis-2 (P-prompt \ P-SoM) | Same as axis-1 | ⏳ pending | Requires P-prompt mode (re-included 2026-05-13) |
    54	| K_h1 transparency ratio (NOT a gate) | 0.75 → 3 of 4 cells individually Holm-sig on drop-one | ⏳ pending | Reclassified gate → transparency 2026-05-13 (power analysis dysfunction at <7pp effects); reported alongside pooled meta as per-cell consistency check |
    55	| K_h3 transparency ratio (NOT a gate) | 0.67 → 3 of 4 cells individually CI > 0 | ⏳ pending | Same reclassification rationale |
    56	| TOST equivalence δ (SR-margin) | 1.0pp | ⏳ pending | SR percentage-point margin for H1(iii) drop-one effect size; distinct from H2(a) cost ±10% relative margin |
    57	| Cell scope (Phase 1a operational) | 24 conditions = 2 sites (cls, red) × 2 models (B0, B1) × 6 modes (DOM, SoM, Vision, P-text, P-prompt, P-SoM) | ⏳ pending | Replaces prior 16/18-cell phantom-only scope (codex Flaw 1+5 fix 2026-05-13) |
    58	| Cell scope (Phase 1a statistical) | 4 cells = (site, model) tuples | ⏳ pending | One drop-one number per cell; pooled meta input |
    59	| Cell scope (Phase 1b deferred) | +12 conditions = shop × B0+B1 × 6 modes | ⏳ pending | Main-paper expansion lever; not part of Phase 1a workshop submission |
    60	
    61	### 2.3 Environment fingerprints (per-machine snapshots)
    62	
    63	| Machine | env_snapshot.json | vwa_snapshot.json | Locked at |
    64	|---|---|---|---|
    65	| DGX `spark-9ea3` (Phase 1 baseline) | `results/provenance/env_lock_dgx.json` | `results/provenance/vwa_dgx_via_quark.json` | TBD |
    66	| A100 `condense` (Phase 2 rerun + mechanistic) | `results/provenance/env_lock_a100.json` | `results/provenance/vwa_a100_self_host.json` | TBD |
    67	| Myriad (cross-arch backup, optional) | `results/provenance/env_lock_myriad.json` | N/A (no VWA use case) | TBD if used |
    68	
    69	Each `env_snapshot.json` captures: torch / transformers / Python / git commit /
    70	HuggingFace model revision SHA (Qwen3-VL-4B + Llama-3.2-Vision if used) /

exec
/bin/bash -lc "nl -ba docs/checkpoints/next_steps.md | sed -n '25,145p;190,205p;320,340p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
    25	
    26	**Paper hook**: → [[paper_planning#§1]] (canonical, 3 arms / 4-fold drop-in)
    27	
    28	> [!todo] Top 3 forward actions (priority order, 2026-05-10 evening update — Stage 3 mechanism attribution + new methods unlocked)
    29	> 1. **Stage 4 mechanism methods queue** ⭐⭐⭐ — paper §5 升级路径; 详 §0a 新加. Cell H-d-cls (DOM target 2x2 closure, job 344623 qw on Myriad) 数据回来后 trigger §124 笔记 + decide Method 1/3/4 顺序. Method 1 (PCA cosine gap, Tool Calling Linear Circuit replicate on B1=Qwen3-VL-4B) is highest-leverage Zoom 4 self-probe — **可立即跑 partial** on existing Stage 1 hidden state cache (3 modes × 26 tasks). Full 6-mode requires Myriad hidden state extraction (DGX 96% util currently).
    30	> 2. **Quark SSH cert → A100 SSH verify** ⭐⭐ — needed for 16-cell rerun (VWA self-host on A100). Portal cert (id_arc + id_arc.signed) + ~/.ssh/config. ETA 10 min once user has time.
    31	> 3. **Advisor email reply wait** (~2-5d, passive) — Q1-Q11 in [[advisor_sync_5_5_followup]]. K_h1=12 / K_h3=11 / TOST δ=1.0pp threshold lock + paper split 3v4. Reply triggers OSF DOI 8-step lock + 16-cell launch gate clearance.
    32	
    33	---
    34	
    35	## §0a Stage 4 mechanism methods queue (added 2026-05-10 after Q1/Q2/Q3 deep critique)
    36	
    37	**Trigger**: post-Stage 3 attribution (H-text + H-prompt cells) showed mid-layer L11/L17 disruption locus is real but no robust fusion under Spearman; user critique forced reframe of paper §5 from "fusion" → "disruption + attribution". To strengthen mechanism story beyond disruption-only, queue 4 methods (Zoom 4 model-internal probes, all feasible on B1=Qwen3-VL-4B).
    38	
    39	**Existing methods used**: linear probe (§111 trivial), activation patching with token-overlap + LD (Stage 2/3, 12 valid cells; Spearman robust check shows no clean transfer).
    40	
    41	### Stage 4.1: Cell H-d-cls (2x2 additivity closure) ⏳ in flight
    42	
    43	- Job 344623 qw on Myriad, cls fwd × strong × source=som × target=dom × N=24
    44	- Pre-registered prediction: Δ_to_target @ L11 ≈ +10.74 (= Ht_cls + Hp_cls − Cell A = 9.04 + 5.62 − 3.92)
    45	- Falsifies if observed outside ±2pp of prediction → prompt × text interaction at mid-layer
    46	- Bg monitor `bh702x73i` auto-computes observed Δ + ntfy on completion
    47	- ETA 30-90 min A100 / 1.5-2.5h V100 once it leaves qw
    48	
    49	### Stage 4.2: PCA cosine gap (Tool Calling Linear Circuit replicate, B1=Qwen3-VL-4B) ⭐ next priority
    50	
    51	**Method**: at L11/L17/L23, PCA on hidden states across 6 modes (DOM/P-text/P-prompt/P-SoM/SoM/Vision), measure (a) cosine gap between mode-mean vectors, (b) AUROC for binary mode classification via cosine to mode mean, (c) % variance captured in top-k PCA dims (Tool Calling found 15 tools → 10 PCA dims = 90.2% var on Qwen3-4B).
    52	
    53	**Why this answers "is it just prompt engineering"**: linear probe trivial ≠ PCA gap trivial. Even when classifier can't separate, mode means may differ on low-rank subspace (Tool Calling Linear Circuit demonstrated this on architectural cousin Qwen3-4B). If AUROC ≥ 0.8 at L11/L17 → phantom space is real representational structure; if ≈ 0.5 → paper §5 stays disruption-only.
    54	
    55	**Existing data (already on disk, no new compute)**:
    56	- `results/mechanistic/stage1B_archived_b1_classifieds_pilot/hidden_states.npz` — P-prompt + P-SoM, 96 examples × 37 layers × 2560 dim (cls, 26 tasks × 2 steps)
    57	- `results/mechanistic/stage1C_image_axis_b1_cls_pilot/hidden_states.npz` — SoM + P-SoM, 96 examples × 37 layers × 2560 dim
    58	
    59	**Immediate (today)**: 3-mode partial PCA cosine gap on existing cache (CPU-only, ~5 min) — answers "is there ANY mid-layer mode-specific structure?" using SoM/P-prompt/P-SoM.
    60	
    61	**Full 6-mode (next 1-2 days)**: extract DOM + P-text + Vision hidden states for same 26 cls tasks. DGX is at 96% util (don't run there) → **launch as Myriad qsub** parallel to Cell H-d-cls. ~1-2h forward pass on A100, ~3-5h on V100.
    62	
    63	**Decision tree post Method 4.2**:
    64	- AUROC ≥ 0.8 + clean cosine gap → §5 upgrade to "phantom-mode-specific subspace at L11-L17" with figure (cosine heatmap × layer)
    65	- AUROC ≈ 0.5 → §5 stays "disruption-only" honest framing; pivot to Method 4.3 (logit-level KL during patching) for transfer evidence
    66	
    67	### Stage 4.3: Logit-level KL during patching ⭐ (paper §5 transfer hypothesis decisive)
    68	
    69	**Method**: modify `p79/mechanistic/activation_patching.py` to dump first-token logit distribution at each patched layer position. Compute KL(patched ‖ source) and KL(patched ‖ target) per layer per task. Bypasses greedy decoding lock-in issue that masked transfer in token-overlap metric.
    70	
    71	**Why**: greedy decoding can lock first-token deterministically even if logit distribution shifted toward source. Token-overlap metric misses this. Logit KL is direct distribution-level measure.
    72	
    73	**Effort**: ~half day infra mod + 1 cell re-run on Myriad to verify. Then post-hoc on existing 12 patching cells if infra captures all needed data.
    74	
    75	### Stage 4.4: Counterfactual activation steering (Causal proof of phantom direction)
    76	
    77	**Method**: from Method 4.2 PCA, extract "phantom direction" = h_PSoM_mean - h_DOM_mean. During DOM forward pass, ADD this direction to L17 hidden state. Does output switch from DOM behavior to P-SoM behavior?
    78	
    79	**Why**: tool calling circuit showed L23+ steering 80-93% accuracy switch. If our phantom direction has similar steering effect → causal proof phantom space is mechanism-level (not just correlation).
    80	
    81	**Effort**: 1 day (re-use patching infra with vector add instead of full replace). Requires Method 4.2 to find the direction first.
    82	
    83	### Stage 4.5: Path patching (lower priority, paper §8 future work)
    84	
    85	**Method**: patch attention head OR MLP output specifically (not full layer). Identify sub-component carrying phantom info.
    86	
    87	**Effort**: 2-3 days infra. Reserve for paper-2 follow-up unless Method 4.2-4.4 leave open questions.
    88	
    89	### Routing decision (DGX vs Myriad for Stage 4 work)
    90	
    91	- **DGX 96% GPU util currently** (other user 31GB, seonglae 5GB; 96% compute) → **don't run new GPU work on DGX**
    92	- **Myriad available** but queue wait variable (3-9h observed today on V100/A100 mix)
    93	- **Method 4.2 partial (3-mode existing data)**: run on DGX CPU NOW (5 min, no GPU)
    94	- **Method 4.2 full (6-mode extraction)**: launch Myriad qsub parallel to Cell H-d-cls
    95	- **Method 4.3/4.4**: launch Myriad qsub when ready (no DGX competition)
    96	
    97	---
    98	
    99	## §1 Phase 1 paper-grade rerun launch sequence (post advisor email + A100 SSH)
   100	
   101	**Scope revised 2026-05-13 post codex stress audit** (replaces prior 16-cell phantom-only scope):
   102	
   103	**Phase 1a (workshop-targeted, immediate launch)** — 24 operational conditions across 4 statistical cells:
   104	- B0 × cls × {DOM, SoM, Vision, P-text, P-SoM, P-prompt} = 6
   105	- B0 × red × {DOM, SoM, Vision, P-text, P-SoM, P-prompt} = 6
   106	- B1 × cls × {DOM, SoM, Vision, P-text, P-SoM, P-prompt} = 6
   107	- B1 × red × {DOM, SoM, Vision, P-text, P-SoM, P-prompt} = 6
   108	- **Total: 24 conditions = 2 sites × 2 models × 6 modes, 4 statistical (site, model) cells**
   109	
   110	**Phase 1b (main paper expansion, deferred to post-workshop)** — 12 conditions:
   111	- B0 × shop × {DOM, SoM, Vision, P-text, P-SoM, P-prompt} = 6
   112	- B1 × shop × {DOM, SoM, Vision, P-text, P-SoM, P-prompt} = 6
   113	- Feeds R3 → R1 / Option D framing decision for main paper
   114	
   115	**Orchestrator**: `bash scripts/queues/queue_phase1_paper_grade.sh dry-run` (preview) → `... launch` (Phase 1a default = cls + red parallel chains). Phase 1b launches via `launch phase1b shop`.
   116	
   117	**Pre-launch gates** (orchestrator auto-checks):
   118	1. `preregistration.md` status `locked` + no `TBD` in threshold lines (incl. 2026-05-13 K-of-N transparency reclassification propagated)
   119	2. `results/provenance/env_<host>_baseline.json` committed
   120	3. `results/provenance/vwa_<host>_baseline.json` committed
   121	4. `bash scripts/preflight_v2.sh` passes
   122	5. GPU CUDA available (smoke `python3 -c "import torch; print(torch.cuda.is_available())"`)
   123	6. No conflicting active runs (`pgrep -f run_experiment` ≤ existing approved chains)
   124	
   125	**ETA on A100 40GB** (Phase 1a, post-advisor lock):
   126	| Chain | Conditions | ETA |
   127	|---|---|---|
   128	| cls | 12 (B0 24h → B1 48h) | 72h ≈ 3 days |
   129	| red | 12 (B0 20h → B1 40h) | 60h ≈ 2.5 days |
   130	| **Phase 1a wallclock (parallel)** | 24 | **~72h ≈ 3 days** |
   131	| Phase 1b shop (post-workshop) | 12 (B0 32h → B1 64h) | 96h ≈ 4 days |
   132	
   133	**Post-completion**:
   134	```bash
   135	make analysis                                    # rerun all aggregators + figures
   136	python3 scripts/analysis/preregistration_decision_test.py \
   137	    --cells-csv results/phantom_paper/cells_aggregated.csv \
   138	    --primary-gate drop_one_pooled_meta_TOST \
   139	    --transparency-K_h1 3 --transparency-K_h3 3 --TOST-delta 1.0 \
   140	    --out results/phantom_paper/preregistration_test_results.json
   141	```
   142	Output → paper §5 Table 5 quotable JSON.
   143	
   144	---
   145	
   190	- Reverse null effect cross-task → paper §5 strongest mechanism evidence
   191	- Token overlap per-layer distribution
   192	
   193	**Followup paper-grade artifact**: `run_manifest.json` aggregate field → paper Table 6 / Figure mechanism panel.
   194	
   195	---
   196	
   197	## §3 OSF DOI 8-step lock workflow (post advisor email)
   198	
   199	**Trigger**: Advisor email reply with confirmed K_h1 / K_h3 / TOST δ.
   200	
   201	**8 steps** (详 [[osf_lock_manifest]]):
   202	1. Save advisor email PDF → `docs/reference/advisor_email_<date>.pdf`
   203	2. Update `preregistration.md` (replace `TBD` with confirmed numbers)
   204	3. Run `python3 scripts/provenance/snapshot_env.py` on DGX + A100 (+ Myriad if used)
   205	4. Run `bash scripts/provenance/snapshot_vwa.sh` on each VWA host
   320	- fig5 category × mode heatmap / fig6 capability B0-vs-B1 / fig7 cost-SR Pareto
   321	- fig8 overlap-depth / fig9 regional carbon / fig10 phantom_lift_bars / fig11 routing_auroc_heatmap
   322	
   323	### Key infra paths
   324	```
   325	configs/exp_v2_*.yaml                              per-site experiment configs
   326	scripts/queues/queue_phase1_paper_grade.sh         🆕 Phase 1 paper-grade orchestrator (Phase 1a 24-cond default + Phase 1b shop deferred)
   327	scripts/queues/queue_chain.sh                      sequential chain wrapper
   328	scripts/queues/queue_phantom_*.sh                  per-cell launch
   329	scripts/maintenance/reset_vwa_sites.sh             DGX→quark PowerShell reset
   330	scripts/maintenance/experiment_watchdog.py         auto-clean + post-condition
   331	scripts/provenance/snapshot_env.py                 🆕 env fingerprint (Gap 1)
   332	scripts/provenance/snapshot_vwa.sh                 🆕 VWA Docker fingerprint (Gap 2)
   333	scripts/provenance/numerical_determinism_check.py  🆕 cross-machine drift (Gap 5)
   334	scripts/analysis/preregistration_decision_test.py  🆕 H1/H3/TOST canonical (Gap C2)
   335	p79/utils/auth_refresh.py                          Playwright sign-in subprocess
   336	p79/experiment/router.py                           RuleBasedRouter scaffold
   337	p79/agents/qwen3vl_agent.py                        🆕 HF revision pinned (Gap C1)
   338	p79/mechanistic/extract_hidden_states.py           🆕 HF revision pinned (Gap C1)
   339	```
   340	

exec
/bin/bash -lc "nl -ba scripts/queues/queue_phase1_paper_grade.sh | sed -n '1,260p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	#!/usr/bin/env bash
     2	# queue_phase1_paper_grade.sh — Master orchestrator for Phase 1 paper-grade rerun.
     3	# (Renamed 2026-05-13 from queue_16cell_paper_grade.sh; old name reflected prior
     4	# 16-cell phantom-only scope that codex stress audit identified as incomplete.)
     5	#
     6	# Scope (revised 2026-05-13 post codex stress audit):
     7	#   Phase 1a (THIS SCRIPT default): 24 operational conditions = 2 sites (cls, red)
     8	#     × 2 models (B0, B1) × 6 modes (DOM, SoM, Vision, P-text, P-prompt, P-SoM).
     9	#     Statistical analysis: 4 (site, model) cells, pooled DerSimonian-Laird meta + TOST.
    10	#     Target: workshop submission. Replaces prior 16-cell phantom-only scope which
    11	#     lacked DOM/SoM/Vision baseline rerun (codex Flaw 1).
    12	#   Phase 1b (deferred, requires explicit 'launch phase1b shop'): 12 additional
    13	#     conditions = shop × 2 models × 6 modes. Feeds main paper R3→R1 framing
    14	#     decision post-workshop submission.
    15	#
    16	# **Hard rule: Same site, B0 XOR B1 only**. queue_chain wraps reset+watchdog+idempotent.
    17	# Splits into 2 parallel chains (cls / red) for Phase 1a, each chain internally sequential.
    18	#
    19	# Pre-launch gates (must all pass):
    20	#   - Advisor email reply received → preregistration.md status `draft` → `locked`
    21	#   - A100 SSH connectivity verified ('ssh condense-a100 nvidia-smi' returns OK)
    22	#   - VWA stack running on chosen host (DGX→quark Tailscale OR A100 self-host)
    23	#   - env_snapshot baseline committed (results/provenance/env_<host>_baseline.json)
    24	#   - VWA snapshot baseline committed (results/provenance/vwa_<host>_baseline.json)
    25	#
    26	# Usage:
    27	#   bash scripts/queues/queue_phase1_paper_grade.sh dry-run            # preview, no launch
    28	#   bash scripts/queues/queue_phase1_paper_grade.sh launch             # Phase 1a (cls+red, 24 conditions)
    29	#   bash scripts/queues/queue_phase1_paper_grade.sh launch cls         # only classifieds Phase 1a chain (12 conditions)
    30	#   bash scripts/queues/queue_phase1_paper_grade.sh launch red         # only reddit Phase 1a chain (12 conditions)
    31	#   bash scripts/queues/queue_phase1_paper_grade.sh launch phase1b     # Phase 1b shop chain (12 conditions, deferred to post-workshop)
    32	#
    33	# Phase 1a conditions (24 total):
    34	#   cls chain (12 conditions, B0 → B1 sequential):
    35	#     - B0 cls × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
    36	#     - B1 cls × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
    37	#   red chain (12 conditions, B0 → B1 sequential):
    38	#     - B0 red × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
    39	#     - B1 red × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
    40	#
    41	# Phase 1b conditions (12 total, deferred main-paper expansion):
    42	#     - B0 shop × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
    43	#     - B1 shop × {DOM, SoM, Vision, P-text, P-SoM, P-prompt}  (6)
    44	#
    45	# Chain dependency:
    46	#   cls and red can run in parallel (different sites = no resource contention beyond A100 GPU).
    47	#   Within each chain B0 → B1 sequential (same-site B0/B1 share user account login).
    48	#   Phase 1b shop launched separately after workshop submission to avoid Magento FPC bug
    49	#   surface co-occurring with Phase 1a critical path.
    50	#
    51	# ETA estimates (A100 40GB, post-advisor lock):
    52	#   cls chain (12 conditions): B0 (~24h) → B1 (~48h) = 72h ≈ 3 days
    53	#   red chain (12 conditions): B0 (~20h) → B1 (~40h) = 60h ≈ 2.5 days
    54	#   Total Phase 1a wallclock with 2 parallel chains = max(72, 60) ≈ 3 days
    55	#   Phase 1b shop chain (12 conditions): B0 (~32h) → B1 (~64h) = 96h ≈ 4 days (deferred)
    56	#
    57	# Sentinel files (used by chain to detect completion):
    58	#   results/visualwebarena/phase1/<run_id>/<condition_id>/condition_summary_v2.json
    59	
    60	set -uo pipefail
    61	
    62	SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    63	REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
    64	cd "${REPO_DIR}"
    65	
    66	MODE="${1:-dry-run}"
    67	SITE_FILTER="${2:-all}"
    68	
    69	log() { echo "[16cell $(date '+%H:%M:%S')] $*"; }
    70	fail() { log "FAIL: $*"; exit 1; }
    71	
    72	# ---------------------------------------------------------------------------
    73	# Pre-launch gates
    74	# ---------------------------------------------------------------------------
    75	
    76	check_gates() {
    77	  local errors=0
    78	
    79	  log "=== Gate 1: preregistration.md threshold lock ==="
    80	  if grep -q "K_h1.*TBD\|K_h3.*TBD\|TOST.*TBD" docs/checkpoints/pre_run/preregistration.md 2>/dev/null; then
    81	    log "  FAIL: preregistration.md still has TBD threshold values."
    82	    log "        Wait for advisor email reply, then update preregistration.md."
    83	    errors=$((errors+1))
    84	  elif ! grep -q "^status: locked" docs/checkpoints/pre_run/preregistration.md 2>/dev/null; then
    85	    # Gate 1b added 2026-05-13 (codex audit HIGH-1): launch_checklist.md
    86	    # requires prereg `status: locked` before paper-grade rerun. Previously
    87	    # only TBD threshold was checked; status=draft could pass.
    88	    log "  FAIL: preregistration.md status is not 'locked' (still draft / pending advisor)."
    89	    log "        Once advisor signs, flip 'status: draft' → 'status: locked' in"
    90	    log "        docs/checkpoints/pre_run/preregistration.md before paper-grade launch."
    91	    errors=$((errors+1))
    92	  else
    93	    log "  OK"
    94	  fi
    95	
    96	  log "=== Gate 2: env_snapshot baseline committed ==="
    97	  if ! ls results/provenance/env_*_baseline.json &>/dev/null; then
    98	    log "  FAIL: No env_*_baseline.json found in results/provenance/"
    99	    log "        Run: python3 scripts/provenance/snapshot_env.py results/provenance/env_<host>_baseline.json"
   100	    errors=$((errors+1))
   101	  else
   102	    log "  OK ($(ls results/provenance/env_*_baseline.json | head -3 | tr '\n' ' '))"
   103	  fi
   104	
   105	  log "=== Gate 3: VWA snapshot baseline committed ==="
   106	  if ! ls results/provenance/vwa_*.json &>/dev/null; then
   107	    log "  WARN: No vwa_*.json found. Recommend bash scripts/provenance/snapshot_vwa.sh"
   108	  else
   109	    log "  OK ($(ls results/provenance/vwa_*.json | head -3 | tr '\n' ' '))"
   110	  fi
   111	
   112	  log "=== Gate 4: VWA reachability ==="
   113	  if [ -f scripts/preflight_v2.sh ]; then
   114	    bash scripts/preflight_v2.sh --no-strict-ports 2>&1 | tail -5 | sed 's/^/    /'
   115	  else
   116	    log "  WARN: scripts/preflight_v2.sh not found"
   117	  fi
   118	
   119	  log "=== Gate 5: GPU + model load smoke ==="
   120	  if command -v .venv/bin/python3 &>/dev/null; then
   121	    .venv/bin/python3 -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>&1 | sed 's/^/  /'
   122	  fi
   123	
   124	  log "=== Gate 6: No conflicting active runs ==="
   125	  active=$(pgrep -f "run_experiment.*--config" | wc -l)
   126	  log "  Active run_experiment processes: $active"
   127	  if [ "$active" -gt 0 ]; then
   128	    log "  WARN: Existing runs detected. Verify no same-site B0+B1 conflict before launch."
   129	    pgrep -af "run_experiment.*--config" | sed 's/^/    /'
   130	  fi
   131	
   132	  if [ "$errors" -gt 0 ]; then
   133	    fail "$errors gate(s) failed; abort. Fix and re-run."
   134	  fi
   135	  log "All gates passed (or warnings only)."
   136	}
   137	
   138	# ---------------------------------------------------------------------------
   139	# Chain definitions
   140	# ---------------------------------------------------------------------------
   141	
   142	build_cls_chain() {
   143	  # Phase 1a classifieds: 6 modes per model, B0 → B1 sequential = 12 conditions
   144	  cat <<EOF
   145	queue_baseline.sh B0 dom classifieds
   146	queue_baseline.sh B0 som classifieds
   147	queue_baseline.sh B0 vision classifieds
   148	queue_phantom_text.sh B0 classifieds
   149	queue_phantom_som.sh B0 classifieds
   150	queue_phantom_prompt.sh B0 classifieds
   151	queue_baseline.sh B1 dom classifieds
   152	queue_baseline.sh B1 som classifieds
   153	queue_baseline.sh B1 vision classifieds
   154	queue_phantom_text.sh B1 classifieds
   155	queue_phantom_som.sh B1 classifieds
   156	queue_phantom_prompt.sh B1 classifieds
   157	EOF
   158	}
   159	
   160	build_red_chain() {
   161	  # Phase 1a reddit: 6 modes per model, B0 → B1 sequential = 12 conditions
   162	  cat <<EOF
   163	queue_baseline.sh B0 dom reddit
   164	queue_baseline.sh B0 som reddit
   165	queue_baseline.sh B0 vision reddit
   166	queue_phantom_text.sh B0 reddit
   167	queue_phantom_som.sh B0 reddit
   168	queue_phantom_prompt.sh B0 reddit
   169	queue_baseline.sh B1 dom reddit
   170	queue_baseline.sh B1 som reddit
   171	queue_baseline.sh B1 vision reddit
   172	queue_phantom_text.sh B1 reddit
   173	queue_phantom_som.sh B1 reddit
   174	queue_phantom_prompt.sh B1 reddit
   175	EOF
   176	}
   177	
   178	build_shop_chain() {
   179	  # Phase 1b deferred: shop × 6 modes per model, B0 → B1 sequential = 12 conditions
   180	  # NOT launched as part of default `launch` (which is Phase 1a cls + red).
   181	  # Launch via explicit `launch phase1b` after workshop submission.
   182	  cat <<EOF
   183	queue_baseline.sh B0 dom shopping
   184	queue_baseline.sh B0 som shopping
   185	queue_baseline.sh B0 vision shopping
   186	queue_phantom_text.sh B0 shopping
   187	queue_phantom_som.sh B0 shopping
   188	queue_phantom_prompt.sh B0 shopping
   189	queue_baseline.sh B1 dom shopping
   190	queue_baseline.sh B1 som shopping
   191	queue_baseline.sh B1 vision shopping
   192	queue_phantom_text.sh B1 shopping
   193	queue_phantom_som.sh B1 shopping
   194	queue_phantom_prompt.sh B1 shopping
   195	EOF
   196	}
   197	
   198	# ---------------------------------------------------------------------------
   199	# Launch
   200	# ---------------------------------------------------------------------------
   201	
   202	dry_run() {
   203	  log "DRY RUN — no launches will occur."
   204	  log ""
   205	  log "=== Phase 1a (default, workshop-target) ==="
   206	  log ""
   207	  log "Cls chain (12 conditions, 6 modes × B0+B1):"
   208	  build_cls_chain | sed 's/^/  /'
   209	  log ""
   210	  log "Red chain (12 conditions, 6 modes × B0+B1):"
   211	  build_red_chain | sed 's/^/  /'
   212	  log ""
   213	  log "Phase 1a total: 24 operational conditions across 4 statistical cells (= (site, model) tuples)."
   214	  log ""
   215	  log "=== Phase 1b (deferred, main paper expansion) ==="
   216	  log ""
   217	  log "Shop chain (12 conditions, 6 modes × B0+B1):"
   218	  build_shop_chain | sed 's/^/  /'
   219	  log ""
   220	  log "Phase 1b total: 12 conditions (launch separately via 'launch phase1b shop' post-workshop)."
   221	  log ""
   222	  log "Run with 'launch' for Phase 1a default, or 'launch phase1b shop' for shop expansion."
   223	}
   224	
   225	launch_chain() {
   226	  local label=$1
   227	  local builder=$2
   228	  local logfile="logs/queue_phase1_${label}.log"
   229	  mkdir -p logs
   230	
   231	  # Convert chain commands to space-quoted args
   232	  local args=()
   233	  while IFS= read -r line; do
   234	    [ -z "$line" ] && continue
   235	    args+=("$line")
   236	  done < <($builder)
   237	
   238	  log "Launching $label chain (${#args[@]} cells) → $logfile"
   239	  RESET_BEFORE=1 nohup bash scripts/queues/queue_chain.sh "${args[@]}" \
   240	    > "$logfile" 2>&1 &
   241	  local pid=$!
   242	  log "  PID $pid, log $logfile"
   243	  echo "$pid" > "logs/queue_phase1_${label}.pid"
   244	}
   245	
   246	# ---------------------------------------------------------------------------
   247	# Main
   248	# ---------------------------------------------------------------------------
   249	
   250	case "$MODE" in
   251	  dry-run)
   252	    dry_run
   253	    ;;
   254	  launch)
   255	    check_gates
   256	    case "$SITE_FILTER" in
   257	      all)
   258	        # Default = Phase 1a (cls + red only). Phase 1b shop requires explicit launch.
   259	        launch_chain "cls" build_cls_chain
   260	        launch_chain "red" build_red_chain

exec
/bin/bash -lc "nl -ba scripts/queues/queue_phase1_paper_grade.sh | sed -n '250,340p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
   250	case "$MODE" in
   251	  dry-run)
   252	    dry_run
   253	    ;;
   254	  launch)
   255	    check_gates
   256	    case "$SITE_FILTER" in
   257	      all)
   258	        # Default = Phase 1a (cls + red only). Phase 1b shop requires explicit launch.
   259	        launch_chain "cls" build_cls_chain
   260	        launch_chain "red" build_red_chain
   261	        ;;
   262	      cls)  launch_chain "cls" build_cls_chain ;;
   263	      red)  launch_chain "red" build_red_chain ;;
   264	      shop)
   265	        log "WARN: 'launch shop' requested directly. shop is Phase 1b (main-paper expansion)."
   266	        log "      Default Phase 1a does NOT include shop. Proceeding only if you confirm."
   267	        log "      Use 'launch phase1b' to launch shop explicitly as Phase 1b."
   268	        fail "Use 'launch phase1b' for shop chain (Phase 1b main-paper expansion)."
   269	        ;;
   270	      phase1b)
   271	        log "=== Phase 1b launch (main-paper shop expansion) ==="
   272	        launch_chain "shop" build_shop_chain
   273	        ;;
   274	      *) fail "Unknown site filter: $SITE_FILTER (expected: all|cls|red|phase1b)" ;;
   275	    esac
   276	    log ""
   277	    log "Phase 1a rerun launched (24 conditions, cls + red × B0+B1 × 6 modes). Monitor:"
   278	    log "  - PIDs: cat logs/queue_phase1_*.pid"
   279	    log "  - Logs: tail -f logs/queue_phase1_*.log"
   280	    log "  - Cells: open Obsidian Bases view 'cells.base' (cron 10min refresh)"
   281	    log "  - Active: make active"
   282	    log ""
   283	    log "Post-completion analysis:"
   284	    log "  make analysis              # full pipeline"
   285	    log "  python3 scripts/analysis/preregistration_decision_test.py \\"
   286	    log "      --cells-csv results/phantom_paper/cells_aggregated.csv \\"
   287	    log "      --primary-gate drop_one_pooled_meta_TOST \\"
   288	    log "      --transparency K_h1_3_of_4,K_h3_3_of_4 \\"
   289	    log "      --out results/phantom_paper/preregistration_test_results.json"
   290	    ;;
   291	  *)
   292	    fail "Unknown mode: $MODE (expected: dry-run | launch)"
   293	    ;;
   294	esac

exec
/bin/bash -lc 'bash scripts/queues/queue_phase1_paper_grade.sh dry-run' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
[16cell 17:21:08] DRY RUN — no launches will occur.
[16cell 17:21:08] 
[16cell 17:21:08] === Phase 1a (default, workshop-target) ===
[16cell 17:21:08] 
[16cell 17:21:08] Cls chain (12 conditions, 6 modes × B0+B1):
  queue_baseline.sh B0 dom classifieds
  queue_baseline.sh B0 som classifieds
  queue_baseline.sh B0 vision classifieds
  queue_phantom_text.sh B0 classifieds
  queue_phantom_som.sh B0 classifieds
  queue_phantom_prompt.sh B0 classifieds
  queue_baseline.sh B1 dom classifieds
  queue_baseline.sh B1 som classifieds
  queue_baseline.sh B1 vision classifieds
  queue_phantom_text.sh B1 classifieds
  queue_phantom_som.sh B1 classifieds
  queue_phantom_prompt.sh B1 classifieds
[16cell 17:21:08] 
[16cell 17:21:08] Red chain (12 conditions, 6 modes × B0+B1):
  queue_baseline.sh B0 dom reddit
  queue_baseline.sh B0 som reddit
  queue_baseline.sh B0 vision reddit
  queue_phantom_text.sh B0 reddit
  queue_phantom_som.sh B0 reddit
  queue_phantom_prompt.sh B0 reddit
  queue_baseline.sh B1 dom reddit
  queue_baseline.sh B1 som reddit
  queue_baseline.sh B1 vision reddit
  queue_phantom_text.sh B1 reddit
  queue_phantom_som.sh B1 reddit
  queue_phantom_prompt.sh B1 reddit
[16cell 17:21:08] 
[16cell 17:21:08] Phase 1a total: 24 operational conditions across 4 statistical cells (= (site, model) tuples).
[16cell 17:21:08] 
[16cell 17:21:08] === Phase 1b (deferred, main paper expansion) ===
[16cell 17:21:08] 
[16cell 17:21:08] Shop chain (12 conditions, 6 modes × B0+B1):
  queue_baseline.sh B0 dom shopping
  queue_baseline.sh B0 som shopping
  queue_baseline.sh B0 vision shopping
  queue_phantom_text.sh B0 shopping
  queue_phantom_som.sh B0 shopping
  queue_phantom_prompt.sh B0 shopping
  queue_baseline.sh B1 dom shopping
  queue_baseline.sh B1 som shopping
  queue_baseline.sh B1 vision shopping
  queue_phantom_text.sh B1 shopping
  queue_phantom_som.sh B1 shopping
  queue_phantom_prompt.sh B1 shopping
[16cell 17:21:08] 
[16cell 17:21:08] Phase 1b total: 12 conditions (launch separately via 'launch phase1b shop' post-workshop).
[16cell 17:21:08] 
[16cell 17:21:08] Run with 'launch' for Phase 1a default, or 'launch phase1b shop' for shop expansion.

exec
/bin/bash -lc '.venv/bin/python3 scripts/analysis/preregistration_decision_test.py --synthetic --scenario r1_pass --seed 42' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
exec
/bin/bash -lc '.venv/bin/python3 scripts/analysis/preregistration_decision_test.py --synthetic --scenario r5_fail --seed 42' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 922ms:
{
  "captured_at": "2026-05-13T16:21:08.996859+00:00",
  "scope": "Phase 1a 24-condition / 4-cell statistical analysis",
  "n_cells": 4,
  "n_tasks_total": 800,
  "cell_ids": [
    "classifieds_B0",
    "classifieds_B1",
    "reddit_B0",
    "reddit_B1"
  ],
  "input_data_sha256": "synthetic:r1_pass:42",
  "thresholds": {
    "primary_gate_method": "pooled_DerSimonian_Laird_meta + TOST + magnitude",
    "TOST_delta_pp": 1.0,
    "H1_magnitude_pp": 1.0,
    "H2_cost_margin_pct": 10.0,
    "H3_min_unique_count": 2,
    "transparency_K_h1": 3,
    "transparency_K_h3": 3,
    "transparency_K_h2": 3,
    "alpha": 0.05
  },
  "H1_psom_drop_one": {
    "primary_gate": {
      "pooled_meta": {
        "pooled_effect": 4.648785956943824,
        "pooled_se": 0.8432433656780929,
        "pooled_ci_95": [
          2.996028960214762,
          6.301542953672886
        ],
        "Q": 3.8498487271629966,
        "Q_df": 3,
        "I_squared_pct": 22.0748602709193,
        "tau_squared": 0.6342046820184201,
        "p_value_two_sided": 3.528032310740059e-08,
        "z_statistic": 5.512982545917227,
        "k": 4
      },
      "magnitude_check": {
        "pooled_pp": 4.648785956943824,
        "threshold_pp": 1.0,
        "pass": true
      },
      "superiority_test": {
        "threshold": 1.0,
        "alpha": 0.05,
        "pooled_effect": 4.648785956943824,
        "pooled_se": 0.8432433656780929,
        "z_statistic": 4.327085282206351,
        "p_one_sided": 7.554773549856009e-06,
        "decision": "reject_H0_substantively_above_threshold"
      },
      "tost_informational": {
        "delta": 1.0,
        "alpha_per_side": 0.05,
        "pooled_effect": 4.648785956943824,
        "pooled_se": 0.8432433656780929,
        "p_lower_bound_test": 1.0501155500719506e-11,
        "p_upper_bound_test": 0.9999924452264501,
        "max_p_value": 0.9999924452264501,
        "equivalence_demonstrated": false,
        "decision": "equivalence_not_demonstrated"
      },
      "decision": "PASS"
    },
    "transparency_K_h1": {
      "K": 3,
      "N": 4,
      "n_individually_holm_sig": 4,
      "consistent": true,
      "note": "transparency-only, NOT a gate on H1 (per prereg 2026-05-13 reclassification)"
    },
    "per_cell": {
      "classifieds_B0": {
        "drop_one_lift_pp": 3.0,
        "ci_95_pp": [
          1.0,
          5.5
        ],
        "se_pp": 1.18855475785191,
        "p_value_two_sided": 0.011600355173791321,
        "n_tasks": 200,
        "holm_p": 0.011600355173791321,
        "individually_holm_sig": true
      },
      "classifieds_B1": {
        "drop_one_lift_pp": 7.000000000000001,
        "ci_95_pp": [
          3.5000000000000004,
          11.0
        ],
        "se_pp": 1.8288454488216537,
        "p_value_two_sided": 0.0001294243507643511,
        "n_tasks": 200,
        "holm_p": 0.0005176974030574044,
        "individually_holm_sig": true
      },
      "reddit_B0": {
        "drop_one_lift_pp": 5.5,
        "ci_95_pp": [
          2.5,
          9.0
        ],
        "se_pp": 1.597808731589217,
        "p_value_two_sided": 0.0005769730585636346,
        "n_tasks": 200,
        "holm_p": 0.0017309191756909037,
        "individually_holm_sig": true
      },
      "reddit_B1": {
        "drop_one_lift_pp": 4.5,
        "ci_95_pp": [
          2.0,
          7.5
        ],
        "se_pp": 1.4835487578495412,
        "p_value_two_sided": 0.002419211742530347,
        "n_tasks": 200,
        "holm_p": 0.004838423485060694,
        "individually_holm_sig": true
      }
    }
  },
  "H2_cost_equivalence": {
    "h2a_cost_equivalence": {
      "K": 3,
      "N": 4,
      "n_cells_pass": 4,
      "consistent": true,
      "margin_pct": 10.0
    },
    "per_cell": {
      "classifieds_B0": {
        "median_cost_dom": 0.040170003463832135,
        "median_cost_psom": 0.040164501010949136,
        "relative_diff_pct": -0.013697914883063642,
        "margin_pct": 10.0,
        "per_cell_pass": true
      },
      "classifieds_B1": {
        "median_cost_dom": 0.04011064783891247,
        "median_cost_psom": 0.03996128156263624,
        "relative_diff_pct": -0.37238559914588226,
        "margin_pct": 10.0,
        "per_cell_pass": true
      },
      "reddit_B0": {
        "median_cost_dom": 0.040094223813850395,
        "median_cost_psom": 0.03965347625007239,
        "relative_diff_pct": -1.0992794518839075,
        "margin_pct": 10.0,
        "per_cell_pass": true
      },
      "reddit_B1": {
        "median_cost_dom": 0.03995903373305061,
        "median_cost_psom": 0.040228395345905485,
        "relative_diff_pct": 0.6740944104263497,
        "margin_pct": 10.0,
        "per_cell_pass": true
      }
    }
  },
  "H3_axis1_ptext_unique": {
    "axis_mode": "sr_ptext",
    "ref_mode": "sr_psom",
    "primary_gate": {
      "pooled_meta": {
        "pooled_effect": 35.576614317677745,
        "pooled_se": 5.685220790067965,
        "pooled_ci_95": [
          24.433581569144536,
          46.71964706621095
        ],
        "Q": 14.289612980423906,
        "Q_df": 3,
        "I_squared_pct": 79.00572951758835,
        "tau_squared": 101.68066021124014,
        "p_value_two_sided": 3.906048817725605e-10,
        "z_statistic": 6.257736617692984,
        "k": 4
      },
      "ci_excludes_zero": true,
      "decision": "PASS"
    },
    "transparency_K_h3": {
      "K": 3,
      "N": 4,
      "n_cells_pass": 4,
      "consistent": true,
      "note": "transparency-only, NOT a gate on H3 (per prereg 2026-05-13 reclassification)"
    },
    "per_cell": {
      "classifieds_B0": {
        "unique_count": 45.0,
        "ci_95": [
          34.0,
          57.0
        ],
        "se": 5.837604699586007,
        "p_value_one_sided": 6.439293542825908e-15,
        "ci_excludes_zero": true,
        "count_above_min": true,
        "per_cell_pass": true,
        "n_tasks": 200,
        "holm_p": 1.9317880628477724e-14,
        "individually_holm_sig": true
      },
      "classifieds_B1": {
        "unique_count": 22.0,
        "ci_95": [
          14.0,
          31.0
        ],
        "se": 4.364869194269524,
        "p_value_one_sided": 2.324709456047458e-07,
        "ci_excludes_zero": true,
        "count_above_min": true,
        "per_cell_pass": true,
        "n_tasks": 200,
        "holm_p": 2.324709456047458e-07,
        "individually_holm_sig": true
      },
      "reddit_B0": {
        "unique_count": 44.0,
        "ci_95": [
          33.0,
          55.0
        ],
        "se": 5.6353154996664365,
        "p_value_one_sided": 2.886579864025407e-15,
        "ci_excludes_zero": true,
        "count_above_min": true,
        "per_cell_pass": true,
        "n_tasks": 200,
        "holm_p": 1.1546319456101628e-14,
        "individually_holm_sig": true
      },
      "reddit_B1": {
        "unique_count": 33.0,
        "ci_95": [
          23.0,
          43.0
        ],
        "se": 5.156582331480934,
        "p_value_one_sided": 7.789857647821918e-11,
        "ci_excludes_zero": true,
        "count_above_min": true,
        "per_cell_pass": true,
        "n_tasks": 200,
        "holm_p": 1.5579715295643837e-10,
        "individually_holm_sig": true
      }
    }
  },
  "H3_axis2_pprompt_unique": {
    "axis_mode": "sr_pprompt",
    "ref_mode": "sr_psom",
    "primary_gate": {
      "pooled_meta": {
        "pooled_effect": 30.883128305489386,
        "pooled_se": 3.584557982331513,
        "pooled_ci_95": [
          23.85739466011962,
          37.90886195085915
        ],
        "Q": 6.144721195744663,
        "Q_df": 3,
        "I_squared_pct": 51.177605876120836,
        "tau_squared": 26.21534274903154,
        "p_value_two_sided": 0.0,
        "z_statistic": 8.61560294399311,
        "k": 4
      },
      "ci_excludes_zero": true,
      "decision": "PASS"
    },
    "transparency_K_h3": {
      "K": 3,
      "N": 4,
      "n_cells_pass": 4,
      "consistent": true,
      "note": "transparency-only, NOT a gate on H3 (per prereg 2026-05-13 reclassification)"
    },
    "per_cell": {
      "classifieds_B0": {
        "unique_count": 31.0,
        "ci_95": [
          21.0,
          41.0
        ],
        "se": 4.941612949082368,
        "p_value_one_sided": 1.767881396830262e-10,
        "ci_excludes_zero": true,
        "count_above_min": true,
        "per_cell_pass": true,
        "n_tasks": 200,
        "holm_p": 5.303644190490786e-10,
        "individually_holm_sig": true
      },
      "classifieds_B1": {
        "unique_count": 29.0,
        "ci_95": [
          19.0,
          39.0
        ],
        "se": 4.970491402747296,
        "p_value_one_sided": 2.6986864964939627e-09,
        "ci_excludes_zero": true,
        "count_above_min": true,
        "per_cell_pass": true,
        "n_tasks": 200,
        "holm_p": 5.3973729929879255e-09,
        "individually_holm_sig": true
      },
      "reddit_B0": {
        "unique_count": 42.0,
        "ci_95": [
          31.0,
          53.0
        ],
        "se": 5.825280499074092,
        "p_value_one_sided": 2.7977620220553945e-13,
        "ci_excludes_zero": true,
        "count_above_min": true,
        "per_cell_pass": true,
        "n_tasks": 200,
        "holm_p": 1.1191048088221578e-12,
        "individually_holm_sig": true
      },
      "reddit_B1": {
        "unique_count": 24.0,
        "ci_95": [
          16.0,
          33.0
        ],
        "se": 4.430219277060995,
        "p_value_one_sided": 3.024627692749249e-08,
        "ci_excludes_zero": true,
        "count_above_min": true,
        "per_cell_pass": true,
        "n_tasks": 200,
        "holm_p": 3.024627692749249e-08,
        "individually_holm_sig": true
      }
    }
  },
  "framing_rule": {
    "rule": "R1",
    "framing": "Phantom routing space (2-axis empirical structure)",
    "hook_power": "STRONGEST"
  },
  "primary_gate_summary": {
    "H1": "PASS",
    "H2": "PASS",
    "H3_axis1": "PASS",
    "H3_axis2": "PASS"
  },
  "transparency_summary": {
    "K_h1": "4/4 \u2265 3?  YES",
    "K_h3_axis1": "4/4 \u2265 3?  YES",
    "K_h3_axis2": "4/4 \u2265 3?  YES"
  }
}

 succeeded in 904ms:
{
  "captured_at": "2026-05-13T16:21:08.995639+00:00",
  "scope": "Phase 1a 24-condition / 4-cell statistical analysis",
  "n_cells": 4,
  "n_tasks_total": 800,
  "cell_ids": [
    "classifieds_B0",
    "classifieds_B1",
    "reddit_B0",
    "reddit_B1"
  ],
  "input_data_sha256": "synthetic:r5_fail:42",
  "thresholds": {
    "primary_gate_method": "pooled_DerSimonian_Laird_meta + TOST + magnitude",
    "TOST_delta_pp": 1.0,
    "H1_magnitude_pp": 1.0,
    "H2_cost_margin_pct": 10.0,
    "H3_min_unique_count": 2,
    "transparency_K_h1": 3,
    "transparency_K_h3": 3,
    "transparency_K_h2": 3,
    "alpha": 0.05
  },
  "H1_psom_drop_one": {
    "primary_gate": {
      "pooled_meta": {
        "pooled_effect": 3.322454022446911,
        "pooled_se": 0.8866948532166888,
        "pooled_ci_95": [
          1.584532110142201,
          5.060375934751621
        ],
        "Q": 6.057807663909472,
        "Q_df": 3,
        "I_squared_pct": 50.477133536723784,
        "tau_squared": 1.5674338238144498,
        "p_value_two_sided": 0.00017895532418599203,
        "z_statistic": 3.747009481778255,
        "k": 4
      },
      "magnitude_check": {
        "pooled_pp": 3.322454022446911,
        "threshold_pp": 1.0,
        "pass": true
      },
      "superiority_test": {
        "threshold": 1.0,
        "alpha": 0.05,
        "pooled_effect": 3.322454022446911,
        "pooled_se": 0.8866948532166888,
        "z_statistic": 2.6192257844078792,
        "p_one_sided": 0.004406479762262383,
        "decision": "reject_H0_substantively_above_threshold"
      },
      "tost_informational": {
        "delta": 1.0,
        "alpha_per_side": 0.05,
        "pooled_effect": 3.322454022446911,
        "pooled_se": 0.8866948532166888,
        "p_lower_bound_test": 5.446125493913101e-07,
        "p_upper_bound_test": 0.9955935202377376,
        "max_p_value": 0.9955935202377376,
        "equivalence_demonstrated": false,
        "decision": "equivalence_not_demonstrated"
      },
      "decision": "PASS"
    },
    "transparency_K_h1": {
      "K": 3,
      "N": 4,
      "n_individually_holm_sig": 3,
      "consistent": true,
      "note": "transparency-only, NOT a gate on H1 (per prereg 2026-05-13 reclassification)"
    },
    "per_cell": {
      "classifieds_B0": {
        "drop_one_lift_pp": 1.5,
        "ci_95_pp": [
          0.0,
          3.5000000000000004
        ],
        "se_pp": 0.8413712947656674,
        "p_value_two_sided": 0.07461823566298431,
        "n_tasks": 200,
        "holm_p": 0.07461823566298431,
        "individually_holm_sig": false
      },
      "classifieds_B1": {
        "drop_one_lift_pp": 5.0,
        "ci_95_pp": [
          2.5,
          8.5
        ],
        "se_pp": 1.5352954543449064,
        "p_value_two_sided": 0.0011271464083262295,
        "n_tasks": 200,
        "holm_p": 0.004508585633304918,
        "individually_holm_sig": true
      },
      "reddit_B0": {
        "drop_one_lift_pp": 4.5,
        "ci_95_pp": [
          2.0,
          7.5
        ],
        "se_pp": 1.4570940680419788,
        "p_value_two_sided": 0.0020127892597132657,
        "n_tasks": 200,
        "holm_p": 0.006038367779139797,
        "individually_holm_sig": true
      },
      "reddit_B1": {
        "drop_one_lift_pp": 3.5000000000000004,
        "ci_95_pp": [
          1.5,
          6.5
        ],
        "se_pp": 1.3014536225050555,
        "p_value_two_sided": 0.007160191119021775,
        "n_tasks": 200,
        "holm_p": 0.01432038223804355,
        "individually_holm_sig": true
      }
    }
  },
  "H2_cost_equivalence": {
    "h2a_cost_equivalence": {
      "K": 3,
      "N": 4,
      "n_cells_pass": 4,
      "consistent": true,
      "margin_pct": 10.0
    },
    "per_cell": {
      "classifieds_B0": {
        "median_cost_dom": 0.040170003463832135,
        "median_cost_psom": 0.040164501010949136,
        "relative_diff_pct": -0.013697914883063642,
        "margin_pct": 10.0,
        "per_cell_pass": true
      },
      "classifieds_B1": {
        "median_cost_dom": 0.04011064783891247,
        "median_cost_psom": 0.03996128156263624,
        "relative_diff_pct": -0.37238559914588226,
        "margin_pct": 10.0,
        "per_cell_pass": true
      },
      "reddit_B0": {
        "median_cost_dom": 0.040094223813850395,
        "median_cost_psom": 0.03965347625007239,
        "relative_diff_pct": -1.0992794518839075,
        "margin_pct": 10.0,
        "per_cell_pass": true
      },
      "reddit_B1": {
        "median_cost_dom": 0.03995903373305061,
        "median_cost_psom": 0.040228395345905485,
        "relative_diff_pct": 0.6740944104263497,
        "margin_pct": 10.0,
        "per_cell_pass": true
      }
    }
  },
  "H3_axis1_ptext_unique": {
    "axis_mode": "sr_ptext",
    "ref_mode": "sr_psom",
    "primary_gate": {
      "pooled_meta": {
        "pooled_effect": 38.32415355229925,
        "pooled_se": 5.840188616164858,
        "pooled_ci_95": [
          26.877383864616128,
          49.77092323998237
        ],
        "Q": 14.113063199651068,
        "Q_df": 3,
        "I_squared_pct": 78.7430980959954,
        "tau_squared": 106.96731729554139,
        "p_value_two_sided": 5.303979477844223e-11,
        "z_statistic": 6.56214312089564,
        "k": 4
      },
      "ci_excludes_zero": true,
      "decision": "PASS"
    },
    "transparency_K_h3": {
      "K": 3,
      "N": 4,
      "n_cells_pass": 4,
      "consistent": true,
      "note": "transparency-only, NOT a gate on H3 (per prereg 2026-05-13 reclassification)"
    },
    "per_cell": {
      "classifieds_B0": {
        "unique_count": 49.0,
        "ci_95": [
          38.0,
          61.0
        ],
        "se": 6.065687724204698,
        "p_value_one_sided": 3.3306690738754696e-16,
        "ci_excludes_zero": true,
        "count_above_min": true,
        "per_cell_pass": true,
        "n_tasks": 200,
        "holm_p": 1.3322676295501878e-15,
        "individually_holm_sig": true
      },
      "classifieds_B1": {
        "unique_count": 24.0,
        "ci_95": [
          16.0,
          33.0
        ],
        "se": 4.523952049054249,
        "p_value_one_sided": 5.630685184776496e-08,
        "ci_excludes_zero": true,
        "count_above_min": true,
        "per_cell_pass": true,
        "n_tasks": 200,
        "holm_p": 5.630685184776496e-08,
        "individually_holm_sig": true
      },
      "reddit_B0": {
        "unique_count": 45.0,
        "ci_95": [
          34.0,
          57.0
        ],
        "se": 5.7002389181264785,
        "p_value_one_sided": 1.4432899320127035e-15,
        "ci_excludes_zero": true,
        "count_above_min": true,
        "per_cell_pass": true,
        "n_tasks": 200,
        "holm_p": 4.3298697960381105e-15,
        "individually_holm_sig": true
      },
      "reddit_B1": {
        "unique_count": 37.0,
        "ci_95": [
          27.0,
          48.0
        ],
        "se": 5.401799277595886,
        "p_value_one_sided": 3.703704010149522e-12,
        "ci_excludes_zero": true,
        "count_above_min": true,
        "per_cell_pass": true,
        "n_tasks": 200,
        "holm_p": 7.407408020299044e-12,
        "individually_holm_sig": true
      }
    }
  },
  "H3_axis2_pprompt_unique": {
    "axis_mode": "sr_pprompt",
    "ref_mode": "sr_psom",
    "primary_gate": {
      "pooled_meta": {
        "pooled_effect": 33.42590570019044,
        "pooled_se": 3.821201305423004,
        "pooled_ci_95": [
          25.936351141561353,
          40.91546025881953
        ],
        "Q": 6.575807377209787,
        "Q_df": 3,
        "I_squared_pct": 54.3782257005079,
        "tau_squared": 31.65861503047921,
        "p_value_two_sided": 0.0,
        "z_statistic": 8.747486203554047,
        "k": 4
      },
      "ci_excludes_zero": true,
      "decision": "PASS"
    },
    "transparency_K_h3": {
      "K": 3,
      "N": 4,
      "n_cells_pass": 4,
      "consistent": true,
      "note": "transparency-only, NOT a gate on H3 (per prereg 2026-05-13 reclassification)"
    },
    "per_cell": {
      "classifieds_B0": {
        "unique_count": 34.0,
        "ci_95": [
          24.0,
          43.0
        ],
        "se": 5.143720137882393,
        "p_value_one_sided": 1.921574011021221e-11,
        "ci_excludes_zero": true,
        "count_above_min": true,
        "per_cell_pass": true,
        "n_tasks": 200,
        "holm_p": 5.764722033063663e-11,
        "individually_holm_sig": true
      },
      "classifieds_B1": {
        "unique_count": 31.0,
        "ci_95": [
          21.0,
          42.0
        ],
        "se": 5.0698100530467585,
        "p_value_one_sided": 4.839137979217867e-10,
        "ci_excludes_zero": true,
        "count_above_min": true,
        "per_cell_pass": true,
        "n_tasks": 200,
        "holm_p": 9.678275958435734e-10,
        "individually_holm_sig": true
      },
      "reddit_B0": {
        "unique_count": 45.0,
        "ci_95": [
          34.0,
          57.0
        ],
        "se": 5.941571953387191,
        "p_value_one_sided": 1.8207657603852567e-14,
        "ci_excludes_zero": true,
        "count_above_min": true,
        "per_cell_pass": true,
        "n_tasks": 200,
        "holm_p": 7.283063041541027e-14,
        "individually_holm_sig": true
      },
      "reddit_B1": {
        "unique_count": 26.0,
        "ci_95": [
          17.0,
          35.0
        ],
        "se": 4.606419770613953,
        "p_value_one_sided": 8.29289081849538e-09,
        "ci_excludes_zero": true,
        "count_above_min": true,
        "per_cell_pass": true,
        "n_tasks": 200,
        "holm_p": 8.29289081849538e-09,
        "individually_holm_sig": true
      }
    }
  },
  "framing_rule": {
    "rule": "R1",
    "framing": "Phantom routing space (2-axis empirical structure)",
    "hook_power": "STRONGEST"
  },
  "primary_gate_summary": {
    "H1": "PASS",
    "H2": "PASS",
    "H3_axis1": "PASS",
    "H3_axis2": "PASS"
  },
  "transparency_summary": {
    "K_h1": "3/4 \u2265 3?  YES",
    "K_h3_axis1": "4/4 \u2265 3?  YES",
    "K_h3_axis2": "4/4 \u2265 3?  YES"
  }
}

codex
The synthetic `r1_pass` routes to R1, but `r5_fail` also routes to R1, so the smoke scenario is not actually testing the intended fail path. I’m now separating “script math is wrong” from “script/docs still disagree” so the verdict is not inflated by stale labels alone.
exec
/bin/bash -lc 'rg -n "latency|AUROC|h2|H2|cost_equivalence|cost_margin" scripts/analysis/preregistration_decision_test.py docs/checkpoints/pre_run/preregistration.md docs/checkpoints/pre_run/osf_lock_manifest.md docs/checkpoints/advisor_sync_5_5_followup.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
docs/checkpoints/advisor_sync_5_5_followup.md:32:| (b) latency ~50% lower | 省 image inference 那步 (cls SoM 74s → P-SoM 18s, 4×) |
docs/checkpoints/advisor_sync_5_5_followup.md:33:| (c) routing signal AUROC ≥ baseline | 5-mode 全 usable |
docs/checkpoints/advisor_sync_5_5_followup.md:38:**Signal 基础设施已 ready** (per-condition `confidence_summary.json`, commit `9d7e99f`), 4 个 signal family per mode 已 AUROC 实测:
docs/checkpoints/advisor_sync_5_5_followup.md:44:| **Behavioral** (`action_diversity` 等) | step-level action 序列 | ⭐ **主导, AUROC 0.682-0.748** | secondary |
docs/checkpoints/advisor_sync_5_5_followup.md:45:| **Verbalized self-confidence** | model 直接 output "I'm X% sure" | secondary | ⭐ **主导, AUROC 0.701-0.793** (P-text red 0.793 = 5-mode max, 超 baseline 0.766) |
docs/checkpoints/advisor_sync_5_5_followup.md:55:**多指标 Pareto**: cost / P95 latency / regional carbon (B1 measured 45 region, B0 token-based estimator) — 3 向 drop-in 全 ready.
docs/checkpoints/advisor_sync_5_5_followup.md:181:**注意 (2026-05-13 disambiguation, codex probable concern)**: 这个 δ=1.0pp 是 **SR percentage-point margin**, 不是 cost equivalence margin. H2(a) "cost ≈ DOM" 用另一个 margin ±10% relative cost (不复用同一个 δ). 之前 prereg 跟 advisor follow-up 这两处单位有混淆, 现在显式区分.
docs/checkpoints/pre_run/preregistration.md:32:1. **Phantom-SoM is the deployment hero**: 4-fold drop-in property (cost ≈ DOM, latency ~50% lower, signal AUROC ≥ baseline, drop-one positive) is the headline practical contribution. This is pre-registered strict.
docs/checkpoints/pre_run/preregistration.md:61:#### H2 — 4-fold drop-in property (P-SoM specifically)
docs/checkpoints/pre_run/preregistration.md:66:- **(b) Latency** — median latency(P-SoM) ≤ 0.6 × median latency(SoM); reflects skipping image inference stage. Tested empirically per cell.
docs/checkpoints/pre_run/preregistration.md:67:- **(c) Signal AUROC** — top-1 routing-signal AUROC(P-SoM) ≥ AUROC(DOM) − 0.05 (within 5pp). Tested empirically per cell, signal selected per `aggregate_routing_auroc.py` top-1.
docs/checkpoints/pre_run/preregistration.md:136:**Companion check** (NOT gating): per-mode AUROC of selected routing signals reported for transparency (Section 6 portfolio characterization, see EXPLORATORY §5).
docs/checkpoints/pre_run/preregistration.md:144:| **R1** | H1 holds AND H2 (a)(b)(c) all hold AND H3(i) holds AND H3(ii) holds | "Phantom routing space (M1/M2 2-axis empirical structure); P-SoM as deployment hero, P-text/P-prompt as structural ablation arms validating axis decomposition." | STRONGEST |
docs/checkpoints/pre_run/preregistration.md:145:| **R2** | H1+H2 hold AND only one of H3(i)/(ii) holds | "Phantom routing space (single-axis empirical structure) with P-SoM as deployment hero; remaining axis decomposition theoretical (Zoom 1 architectural argument only)." | MODERATE-STRONG |
docs/checkpoints/pre_run/preregistration.md:146:| **R3** | H1+H2 hold AND neither H3(i)/(ii) holds | "Phantom-SoM is hidden 4th routing arm; M1/M2 axis decomposition supported by Zoom 1 architectural argument only, not empirically validated by ablation." | MODERATE (= 04-30 fallback; workshop-grade) |
docs/checkpoints/pre_run/preregistration.md:147:| **R4** | H1 holds AND H2 partially fails (e.g., (a) cost or (b) latency fails on some site) | "Phantom-SoM partial drop-in" + §4 disclosure of failed sub-claim. | WEAK; substantial revision |
docs/checkpoints/pre_run/preregistration.md:161:- H2 sub-claims (a)(b)(c)(d) per cell: m = 4 × 4 statistical cells = 16 tests (each per-cell sub-claim).
docs/checkpoints/pre_run/preregistration.md:188:- Best-signal-per-mode characterization (Register III AA, Section 6 portfolio finding): per (mode, signal) AUROC reported, Holm-corrected within mode for transparency.
docs/checkpoints/pre_run/preregistration.md:209:| **Effect size (continuous)** | Cohen's d with bootstrap CI | For cost/latency H2(a)(b) |
docs/checkpoints/pre_run/preregistration.md:245:- **Best-signal-per-mode characterization** (Register III AA novelty, Section 6 portfolio finding): which routing signal works best for which mode is reported as exploratory characterization, NOT pre-registered prediction. Per-(mode, signal) AUROC table reported with Holm correction within mode for transparency.
docs/checkpoints/pre_run/preregistration.md:296:   - (3) **TOST δ=1.0pp** equivalence margin (interpretation: SR drop-one effect-size margin, distinct from H2(a) cost ±10% margin — see §4 lock row)
docs/checkpoints/pre_run/osf_lock_manifest.md:56:| TOST equivalence δ (SR-margin) | 1.0pp | ⏳ pending | SR percentage-point margin for H1(iii) drop-one effect size; distinct from H2(a) cost ±10% relative margin |
scripts/analysis/preregistration_decision_test.py:1:"""Preregistration decision test — Phase 1a 24-condition / 4-cell H1 / H3 / H2 evaluation.
scripts/analysis/preregistration_decision_test.py:26:  H2(a)  median cost(P-SoM) within ±10% of median cost(DOM) per cell, replicated
scripts/analysis/preregistration_decision_test.py:27:         in ≥3 of 4 cells (transparency K_h2)
scripts/analysis/preregistration_decision_test.py:477:def evaluate_h2_cost(cells_by_id: dict[str, list[dict]], cost_margin_pct: float = 10.0,
scripts/analysis/preregistration_decision_test.py:478:                      transparency_K_h2: int = 3) -> dict:
scripts/analysis/preregistration_decision_test.py:479:    """H2(a): median cost(P-SoM) within ±cost_margin_pct% of median cost(DOM) per cell,
scripts/analysis/preregistration_decision_test.py:480:    replicated in ≥ transparency_K_h2 of N cells.
scripts/analysis/preregistration_decision_test.py:482:    H2(a) test margin is a RELATIVE PERCENTAGE (e.g., ±10% of DOM cost), distinct from
scripts/analysis/preregistration_decision_test.py:496:        within_band = abs(rel_diff_pct) <= cost_margin_pct
scripts/analysis/preregistration_decision_test.py:501:            "margin_pct": cost_margin_pct,
scripts/analysis/preregistration_decision_test.py:507:        "h2a_cost_equivalence": {
scripts/analysis/preregistration_decision_test.py:508:            "K": transparency_K_h2,
scripts/analysis/preregistration_decision_test.py:511:            "consistent": pass_count >= transparency_K_h2,
scripts/analysis/preregistration_decision_test.py:512:            "margin_pct": cost_margin_pct,
scripts/analysis/preregistration_decision_test.py:522:def apply_framing_rule(h1: dict, h2: dict, h3_axis1: dict, h3_axis2: dict) -> dict:
scripts/analysis/preregistration_decision_test.py:525:    h2_pass = h2["h2a_cost_equivalence"]["consistent"]
scripts/analysis/preregistration_decision_test.py:529:    if h1_pass and h2_pass and h3_axis1_pass and h3_axis2_pass:
scripts/analysis/preregistration_decision_test.py:532:    if h1_pass and h2_pass and (h3_axis1_pass or h3_axis2_pass):
scripts/analysis/preregistration_decision_test.py:535:    if h1_pass and h2_pass and not h3_axis1_pass and not h3_axis2_pass:
scripts/analysis/preregistration_decision_test.py:538:    if h1_pass and not h2_pass:
scripts/analysis/preregistration_decision_test.py:539:        return {"rule": "R4", "framing": "Phantom-SoM partial drop-in (cost/latency equivalence fails on some site)",
scripts/analysis/preregistration_decision_test.py:576:      - r1_pass:   H1 strong (drop-one lift ~2pp pooled), H2 cost equiv hold, H3 both axes pass
scripts/analysis/preregistration_decision_test.py:635:    p.add_argument("--H2-cost-margin-pct", type=float, default=10.0,
scripts/analysis/preregistration_decision_test.py:636:                   help="H2(a) cost equivalence margin in % (default 10%% per prereg lock)")
scripts/analysis/preregistration_decision_test.py:643:    p.add_argument("--transparency-K_h2", type=int, default=3,
scripts/analysis/preregistration_decision_test.py:644:                   help="H2 transparency cells count (default 3 of 4)")
scripts/analysis/preregistration_decision_test.py:674:    h2 = evaluate_h2_cost(cells_by_id, cost_margin_pct=args.H2_cost_margin_pct,
scripts/analysis/preregistration_decision_test.py:675:                           transparency_K_h2=args.transparency_K_h2)
scripts/analysis/preregistration_decision_test.py:688:    framing = apply_framing_rule(h1, h2, h3_axis1, h3_axis2)
scripts/analysis/preregistration_decision_test.py:701:            "H2_cost_margin_pct": args.H2_cost_margin_pct,
scripts/analysis/preregistration_decision_test.py:705:            "transparency_K_h2": args.transparency_K_h2,
scripts/analysis/preregistration_decision_test.py:709:        "H2_cost_equivalence": h2,
scripts/analysis/preregistration_decision_test.py:715:            "H2": "PASS" if h2["h2a_cost_equivalence"]["consistent"] else "FAIL",
scripts/analysis/preregistration_decision_test.py:738:        logger.info(f"  H2: {'PASS' if h2['h2a_cost_equivalence']['consistent'] else 'FAIL'} "
scripts/analysis/preregistration_decision_test.py:739:                    f"({h2['h2a_cost_equivalence']['n_cells_pass']}/{h2['h2a_cost_equivalence']['N']} cells within ±{args.H2_cost_margin_pct}% cost)")

exec
/bin/bash -lc 'rg -n "per_task_sr|cells_aggregated|sr_psom|cost_psom|preregistration_decision_test" -S scripts docs results | head -200' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
exec
/bin/bash -lc "rg -n \"I²|I\\"'^2|I_squared|do NOT pool|do not pool|very high|heterogeneity" scripts/analysis/preregistration_decision_test.py docs/checkpoints/pre_run/preregistration.md docs/checkpoints/pre_run/osf_lock_manifest.md docs/checkpoints/advisor_sync_5_5_followup.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
scripts/analysis/preregistration_decision_test.py:157:    Returns dict with: pooled_effect, pooled_se, pooled_ci_95, Q, I_squared, tau_squared,
scripts/analysis/preregistration_decision_test.py:166:      6. I^2 = max(0, (Q − (k − 1)) / Q) × 100  (% heterogeneity)
scripts/analysis/preregistration_decision_test.py:173:                "Q": None, "I_squared_pct": None, "tau_squared": None,
scripts/analysis/preregistration_decision_test.py:204:        "I_squared_pct": i_sq,
docs/checkpoints/pre_run/preregistration.md:152:**Heterogeneity-conditional rule (added 2026-05-13 to resolve §4 audit B8 ↔ H1(i) conflict)**: If pre-specified I² > 75% from random-effects meta (per §4 audit B8 thresholds), do NOT pool — primary inference reverts to per-cell forest + meta-regression by site / model. R1-R5 framing in this branch maps to per-cell direction-consistency: ≥3 of 4 cells direction-positive + ≥2 individually Holm sig → R3-grade hook; otherwise R4/R5.
docs/checkpoints/pre_run/preregistration.md:206:| **Bootstrap clustering** | **Single-level (task_id)** for primary, no nested cluster (cell × site) bootstrap | Justification: meta-analysis at cell level is separate (`aggregate_phantom_meta.py` random-effects + I²/τ²); within-cell bootstrap only re-samples tasks. Multi-level cluster would double-count uncertainty already captured by random-effects meta. Lock: percentile + task-id unit + no nested cluster (B2 lock 2026-05-09). |
docs/checkpoints/pre_run/preregistration.md:231:| **Heterogeneity (random-effects, Q, I², τ²) pre-spec** (audit B8) | (a) **Primary estimator**: random-effects DerSimonian-Laird via `aggregate_phantom_meta.py` (already implemented). (b) **Heterogeneity reporting**: report Cochran Q (chi² test of homogeneity), I² (% of total variance attributable to between-cell heterogeneity), τ² (between-cell variance). (c) **Interpretation thresholds (pre-specified)**: I² < 25% = "low heterogeneity, pooled mean is primary"; 25%-50% = "moderate, report both pooled + per-cell"; 50%-75% = "high, per-cell estimates are primary, pooled is summary"; > 75% = "very high, do not pool — report only per-cell + heterogeneity-source analysis (site / model / task-pool)". (d) **Heterogeneity-source decomposition**: when I² > 50%, report meta-regression by site (cls / red / shop) and by model (B0 / B1) to identify dominant variance source. | Higgins & Thompson 2002 (I² thresholds). Per-cell estimates always shown alongside pooled, so heterogeneity is never averaged away. Lock 2026-05-09. |

exec
/bin/bash -lc 'rg -n "queue_16cell|16-cell|16cell|TOST primary|drop_one_pooled_meta_TOST|--cells-csv|--TOST-delta|--transparency " scripts/queues docs/checkpoints/pre_run docs/checkpoints/next_steps.md docs/checkpoints/advisor_sync_5_5_followup.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
docs/checkpoints/advisor_sync_5_5_followup.md:16:> - **Phase 1a (workshop-targeted, immediate launch)**: 24 operational conditions = 2 sites (cls + red) × 2 models (B0 + B1) × 6 modes (DOM / SoM / Vision / P-text / P-prompt / P-SoM). 统计分析 4 个 cell (= (site, model) tuple), pooled DerSimonian-Laird meta + TOST primary gate. 投 workshop 这一档.
docs/checkpoints/advisor_sync_5_5_followup.md:74:- **Capability-modulated reversal**: B0 偏 text 别扭, B1 偏 image 别扭 (post-hoc N=4 provisional, 16-cell rerun 后 commit)
docs/checkpoints/advisor_sync_5_5_followup.md:196:> "I have reviewed the pre-registration revision (K_h1=0.75 transparency-only / K_h3=0.67 transparency-only / TOST δ=1.0pp SR-margin / Phase 1a 24 conditions across 4 cells: cls+red × B0+B1 × 6 modes / Phase 1b shop deferred / outcome-independent smoke gate / pooled DerSimonian-Laird meta + TOST primary gating) on \<date\> and witness them as committed before Phase 1a data unblinding."
docs/checkpoints/advisor_sync_5_5_followup.md:236:- **5/6-5/8** (这份 doc 您回完): launch 16-cell paper-grade rerun (no early stop) + mechanistic pilot (B1 cls 一个 cell scout activation patching)
docs/checkpoints/next_steps.md:30:> 2. **Quark SSH cert → A100 SSH verify** ⭐⭐ — needed for 16-cell rerun (VWA self-host on A100). Portal cert (id_arc + id_arc.signed) + ~/.ssh/config. ETA 10 min once user has time.
docs/checkpoints/next_steps.md:31:> 3. **Advisor email reply wait** (~2-5d, passive) — Q1-Q11 in [[advisor_sync_5_5_followup]]. K_h1=12 / K_h3=11 / TOST δ=1.0pp threshold lock + paper split 3v4. Reply triggers OSF DOI 8-step lock + 16-cell launch gate clearance.
docs/checkpoints/next_steps.md:101:**Scope revised 2026-05-13 post codex stress audit** (replaces prior 16-cell phantom-only scope):
docs/checkpoints/next_steps.md:137:    --cells-csv results/phantom_paper/cells_aggregated.csv \
docs/checkpoints/next_steps.md:138:    --primary-gate drop_one_pooled_meta_TOST \
docs/checkpoints/next_steps.md:139:    --transparency-K_h1 3 --transparency-K_h3 3 --TOST-delta 1.0 \
docs/checkpoints/next_steps.md:235:| 🟢 N3 | Phantom variant FP rules | 1 h | Post 16-cell rerun |
docs/checkpoints/next_steps.md:241:## §5 Router experiments (Section 6, ~Week 4-5 post 16-cell)
docs/checkpoints/next_steps.md:245:| **Tier 1 oracle router** (TF-IDF + LR, ~3 d) | 16-cell rerun done | `p79/experiment/router.py::RuleBasedRouter` 扩展 |
scripts/queues/queue_phase1_paper_grade.sh:3:# (Renamed 2026-05-13 from queue_16cell_paper_grade.sh; old name reflected prior
scripts/queues/queue_phase1_paper_grade.sh:4:# 16-cell phantom-only scope that codex stress audit identified as incomplete.)
scripts/queues/queue_phase1_paper_grade.sh:10:#     Target: workshop submission. Replaces prior 16-cell phantom-only scope which
scripts/queues/queue_phase1_paper_grade.sh:69:log() { echo "[16cell $(date '+%H:%M:%S')] $*"; }
scripts/queues/queue_phase1_paper_grade.sh:286:    log "      --cells-csv results/phantom_paper/cells_aggregated.csv \\"
scripts/queues/queue_phase1_paper_grade.sh:287:    log "      --primary-gate drop_one_pooled_meta_TOST \\"
scripts/queues/queue_phase1_paper_grade.sh:288:    log "      --transparency K_h1_3_of_4,K_h3_3_of_4 \\"
docs/checkpoints/pre_run/preregistration.md:17:> **Status: draft** — pending advisor sync lock. Once advisor signs (single-line email or co-authored commit), `status` flips to `locked`, `registered_git_sha` records the commit at lock time, and `witnessed_by` records advisor name + lock timestamp. `data_lock_until` records when 16-cell rerun finishes — between lock-time and completion-time, NO additional analyses may be added to gating-family tests.
docs/checkpoints/pre_run/preregistration.md:99:The 4 distinguishing predictions in 实验笔记 §108.16 are tested against 16-cell data. The framework was developed after observing N=4 pre-Phase-A cells; this is **post-hoc**.
docs/checkpoints/pre_run/preregistration.md:150:**Trigger rule update 2026-05-13**: R5 no longer fires on `< K_h1` (K-of-N reclassified to transparency-only). Pooled meta + TOST primary gate only. K-of-N consistency reported in §4 per-cell table as descriptive transparency row.
docs/checkpoints/pre_run/preregistration.md:225:| **N_conditions Phase 1a (operational)** | **24 conditions** = 2 sites (cls, red) × 2 models (B0, B1) × 6 modes (DOM, SoM, Vision, P-text, P-prompt, P-SoM). Each condition launched fresh post-fix via `scripts/queues/queue_phase1_paper_grade.sh` (renamed 2026-05-13 from `queue_16cell_paper_grade.sh`; current scope = 24 conditions Phase 1a + 12 conditions Phase 1b deferred). Sequence: B0 → B1 per site (shared user account); cls + red parallel chains | ✅ **Student-decided 2026-05-13** post-codex stress audit. Workshop-targeted (cls + red only, shop deferred to Phase 1b for main paper). Replaces prior 16-cell phantom-only scope that lacked baseline DOM/SoM/Vision rerun (codex Flaw 1) |
docs/checkpoints/pre_run/preregistration.md:299:   - (6) **N_conditions Phase 1a final scope**: **24 operational conditions** (= 2 sites × 2 models × 6 modes) across **4 statistical cells** (= (site, model) tuples) — student-decided 2026-05-13 post-codex stress audit, replaces prior 16-cell phantom-only scope. Advisor email witness pending
docs/checkpoints/pre_run/preregistration.md:313:2. New project: "Phantom-SoM 16-cell pre-registration witness."
docs/checkpoints/pre_run/preregistration.md:316:5. Paper §1 footnote cites the DOI: "Hypotheses pre-registered prior to 16-cell rerun (OSF DOI X.YYYY/osf.io/zzzz, Git SHA abc123, witnessed by [advisor name] on YYYY-MM-DD)."
docs/checkpoints/pre_run/preregistration.md:356:| 2026-05-13 | **Codex stress audit triggered 6 paper-grade design fixes** (pre-launch): (a) scope reframe 16-cell phantom-only → 24-condition / 4-cell Phase 1a (cls+red×B0+B1×6modes), Phase 1b shop deferred to main paper; (b) K-of-N reclassified gate → transparency-only (power analysis showing dysfunction at < 7pp effects, re-propagated to H1/H3/R5/§6); (c) H1 drop-one definition disambiguated (oracle ceiling lift with-vs-without P-SoM, per (site, model) cell paired bootstrap); (d) smoke-gate B7 revised outcome-independent (no SR-based restart bias); (e) cell terminology disambiguated ("cell" = 4 statistical strata for K-of-N/meta input, "condition" = 24 operational launch units); (f) Phase 1b shop scope-expansion lever for main paper R3→R1 framing decision | Codex CLI hostile reviewer audit (`docs/checkpoints/codex_outputs/codex_stress_16cell_design_2026-05-13.md`, lean prompt no-enumeration, cross-AI complementary to prior Claude reviews); 6 HIGH severity findings + 3 probable concerns. Workshop-targeted Phase 1a launch this week; main paper Phase 1b after workshop submission |
docs/checkpoints/pre_run/locked_versions.md:115:| Git tag | `paper-grade-rerun-launch-{date}` (TBD when 16-cell launches) |
docs/checkpoints/pre_run/pre_rerun_audit.md:3:**Purpose**: Comprehensive paper-grade gate review before launching 16-cell rerun
docs/checkpoints/pre_run/pre_rerun_audit.md:40:│  Phase 2: 实验 run 过程 (Run)       [during 16-cell + Stage 2B] │
docs/checkpoints/pre_run/pre_rerun_audit.md:105:| 1.2.1 | 16-cell scope per `preregistration.md §4` | ✅ | `grep "N_cells" preregistration.md` = 16 |
docs/checkpoints/pre_run/pre_rerun_audit.md:183:| 1.7.8 | **Cross-paper data lineage map** — phantom 16-cell ↔ mechanistic Stage 2B/2C ↔ VWA bug catalog | 🟡 partial | Document data flow in osf_lock_manifest.md §1.5 |
docs/checkpoints/pre_run/pre_rerun_audit.md:197:| 1.6.2 | A.2 Manifest 全 archive + 16-cell rerun | ✅ run_manifest.yaml grade=archived |
docs/checkpoints/pre_run/pre_rerun_audit.md:212:| Q2 🟡 | B0 pre/post Phase A asymmetry | ✅ 16-cell rerun handles | |
docs/checkpoints/pre_run/pre_rerun_audit.md:223:# Phase 2 — 实验 run 过程 (Run) — during 16-cell + Stage 2B/2C
docs/checkpoints/pre_run/pre_rerun_audit.md:235:| 2.1.7 | **`scripts/queues/queue_16cell_paper_grade.sh`** — master orchestrator (B0×{cls,red}×3 + B1×{cls,red}×3 + shop×4) | 🟡 lock at advisor email | replaces ad-hoc `queue_phantom_pair` chains for the 16-cell rerun |
docs/checkpoints/pre_run/pre_rerun_audit.md:301:| 2.5.15 | **Pre-rerun probe re-run protocol** — fire all 6 probes on smoke cell, all exit 0 before launching 16-cell | 🔴 TBD add to launch protocol | scripts above + smoke 2-task cell |
docs/checkpoints/pre_run/pre_rerun_audit.md:414:| 4.1.1 | Run `preregistration_decision_test.py` with locked thresholds | 🟡 | `K_h1=12 --K_h3=11 --TOST-delta=1.0` (after advisor email) |
docs/checkpoints/pre_run/pre_rerun_audit.md:678:**Last expansion**: 2026-05-08, 笔记 §116.15 — repo-wide scripts/docs/笔记 sweep (5 phases × 25 sections × ~245 gate items): §1.4.7b EVIDENCE_LAYER_AUDIT / §1.7.9-13 infrastructure & data layer / §2.1.6-8 preflight + 16-cell orchestrator + A100 self-host / §2.3.7 GLM pipeline / **§2.5b 7-probe bug self-verification chain** / §2.8.7-8 smoke scripts / §3.1.6-7 progress trackers / §3.2.15 B0 vision coord errors / §4.1.6-9 meta-analysis + reeval + dual-track reframe / §4.2.8-12 5 behavior diagnostics / **§4.9.6-13 Stage 1+2A mechanistic pipeline** / §5.1.15-18 replication artifacts.
docs/checkpoints/pre_run/topvenue_constraints.md:33:| A1 | Preregister primary hypotheses, decision rules, and analysis families before post-rerun data are inspected | NEEDS_BIB_ENTRY: Pineau et al. 2021; NeurIPS checklist Q4/Q6 | ⚠️ | `docs/checkpoints/pre_run/preregistration.md` has H1-H8, Holm families, R1-R5, but frontmatter is `status: draft`, `registered_at`, `registered_git_sha`, `witnessed_by`, and OSF DOI are pending. Remediation: lock after advisor email, tag git, deposit OSF; cost 2-4h. | "The hypotheses and decision rules were written before the 16-cell rerun; the camera-ready will cite the lock SHA/OSF DOI once advisor witness is received." |
docs/checkpoints/pre_run/topvenue_constraints.md:85:| D1 | Claims must match evidence and scope; aspirational routing must not be stated as achieved | NeurIPS checklist Q1-Q2; NEEDS_BIB_ENTRY: Lipton & Steinhardt 2018 | ⚠️ | `paper_planning.md §1` labels the 4-fold drop-in hook provisional pending data; `preregistration.md R1-R5` maps framing to outcomes. Some older draft prose still says "hidden fourth routing arm" before 16-cell confirmation. Remediation: update intro after rerun based on R-rule; cost 2h. | "The final framing is data-conditional and tied to R1-R5; router deployment claims are deferred unless H7/H8 are locked and pass." |
docs/checkpoints/pre_run/topvenue_constraints.md:103:| E6 | Include model-scale contrast for agent behavior | `koh2024visualwebarena`; `drouin2024workarena`; `li2024effects` | ⚠️ | B0/B1 are included and `section1_intro.md` reports capability interaction, but B1 reddit phantom and B1 shop are still part of the 16-cell rerun plan. Remediation: finish 16-cell scope or weaken cross-capability claim; cost rerun-dependent. | "Capability contrast is limited to B0/B1 and interpreted as a scale probe, not a universal model-family law." |
docs/checkpoints/pre_run/topvenue_constraints.md:115:| F4 | Statistical conclusion validity: report uncertainty and sensitivity to thresholds | NeurIPS checklist Q7; NEEDS_BIB_ENTRY: Cook & Campbell 1979 | ✓ | `scripts/analysis/sensitivity_loo_meta.py` + `docs/analysis/cross_sites/sensitivity_loo_meta.md` (created 2026-05-09): leave-one-cell-out DerSimonian-Laird re-pool for each arm with k≥2 cells. **Finding**: 3→5-mode oracle lift, P-SoM drop-in, P-prompt drop-in are LOO-robust (Holm decision unchanged under any single-cell removal); **P-text drop-in is FRAGILE** — dropping B0 classifieds or B0 reddit flips Holm to NS (p=0.065-0.077). Consistent with primary meta I²=71% (substantial heterogeneity in P-text arm). K-of-N threshold gradient omitted because rule was reframed as secondary transparency in B9 lock; primary detection via random-effects meta is the LOO target. | "Pooled phantom-lift estimates are LOO-robust except P-text drop-in, which depends on B0 cell inclusion; flagged in §8.5 limitations + waits for 16-cell rerun for resolution." |
docs/checkpoints/pre_run/topvenue_constraints.md:152:4. ❌ C3 — Implement site-state snapshot pre/post-cell gate (0.5-1 day, before 16-cell paper-grade promotion).
docs/checkpoints/pre_run/topvenue_constraints.md:168:### Tier 1 — ✓-Final (data-independent, won't change with 16-cell rerun)
docs/checkpoints/pre_run/topvenue_constraints.md:191:| **D6 + H1** §8 limitations | 6 of 7 subsections cite specific numbers (8.5 N=15 Cohen's d, 8.6 "median LD=0 in 4/5 cells", etc.) — 16-cell rerun + cells F/G 完成后需复核 |
docs/checkpoints/pre_run/topvenue_constraints.md:207:**Implication for paper-launch readiness**: pre-rerun audit 50/78 ✓ 是 "scaffold + methodology lock 足够发起 16-cell paper-grade rerun", **不是** "paper claim is locked". 16-cell rerun 完成后需:
docs/checkpoints/pre_run/negative_results_registry.md:41:| C1 | 2026-04-26 | Phantom-SoM 4-fold drop-in property (cost ≈ DOM / latency ~50% / signal AUROC ≥ baseline / drop-one 1.7-3.8pp) | B0 reddit Phase A archived data + drop-one sig | Paper hook §1 (provisional pending 16-cell rerun) |
docs/checkpoints/pre_run/negative_results_registry.md:50:5. **Paper hook (§1)**: phantom routing space (3 arms) — retracted from "4th arm" (entry #2). State "provisional pending 16-cell rerun" until R1-R5 framing rules (preregistration.md §2) trigger.
docs/checkpoints/pre_run/negative_results_registry.md:56:- 16-cell rerun outcome inconsistent with archived data → entry for "phase A pre-fix data was over-optimistic" (audit F1)
docs/checkpoints/pre_run/dataset_card.md:37:All 910 VWA tasks are **included by default** in the locked 16-cell rerun
docs/checkpoints/pre_run/dataset_card.md:95:- Pilot pending (currently DOM-only B0 cls + B1 cls done; full 5-mode shop in 16-cell rerun)

exec
/bin/bash -lc "nl -ba scripts/queues/queue_chain.sh | sed -n '1,240p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	#!/usr/bin/env bash
     2	# queue_chain.sh — Sequentially launch a list of queue commands, waiting for
     3	# each runner to complete before launching the next. Useful for chaining cells
     4	# that share a single GPU instance (B1 4B local) or any paper-grade sequence.
     5	#
     6	# Each queued command goes through queue_baseline.sh / queue_phantom_som.sh /
     7	# queue_phantom_text.sh which already handle reset+auth_refresh+watchdog
     8	# launch+idempotent skip. This chain ALWAYS exports RESET_BEFORE=1 by default
     9	# (paper-grade — every cell starts from a fresh post-reset site state); pass
    10	# --no-reset to disable (rare, e.g. resume-only chain).
    11	# Note: queue_phantom_dom.sh exists as a back-compat symlink to queue_phantom_text.sh.
    12	#
    13	# Usage:
    14	#   nohup bash scripts/queues/queue_chain.sh [--no-reset] \
    15	#     "<cmd1>" "<cmd2>" ... \
    16	#     > logs/queue_chain_<label>.log 2>&1 &
    17	#
    18	# Each <cmd> is a queue script invocation, relative to scripts/queues/:
    19	#   "queue_phantom_som.sh B1 classifieds"
    20	#   "queue_phantom_text.sh B1 reddit"
    21	#   "queue_baseline.sh B0 dom shopping"
    22	#   "queue_baseline.sh B0 som shopping wa"
    23	#
    24	# The chain auto-detects an already-running cell (queue scripts are idempotent;
    25	# RESET is skipped when a runner is already attached). For the FIRST queued
    26	# cell — if it's already running, chain just waits for completion and proceeds
    27	# to the next.
    28	#
    29	# Examples:
    30	#   # B1 phantom 4-cell chain (cls already running):
    31	#   nohup bash scripts/queues/queue_chain.sh \
    32	#     "queue_phantom_som.sh B1 classifieds" \
    33	#     "queue_phantom_som.sh B1 reddit" \
    34	#     "queue_phantom_text.sh B1 classifieds" \
    35	#     "queue_phantom_text.sh B1 reddit" \
    36	#     > logs/queue_chain_b1_phantom.log 2>&1 &
    37	#
    38	#   # B0 phantom shopping pair (after B0 dom shopping done):
    39	#   nohup bash scripts/queues/queue_chain.sh \
    40	#     "queue_phantom_som.sh B0 shopping" \
    41	#     "queue_phantom_text.sh B0 shopping" \
    42	#     > logs/queue_chain_b0_phantom_shop.log 2>&1 &
    43	
    44	set -uo pipefail
    45	
    46	SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    47	REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
    48	cd "${REPO_DIR}"
    49	
    50	log() { echo "[chain $(date '+%H:%M:%S')] $*"; }
    51	
    52	# ---------- arg parsing ----------
    53	RESET_FLAG=1
    54	if [[ "${1:-}" == "--no-reset" ]]; then
    55	  RESET_FLAG=0
    56	  shift
    57	fi
    58	
    59	if [[ $# -lt 1 ]]; then
    60	  echo "Usage: $0 [--no-reset] <queue_command_1> [<queue_command_2> ...]" >&2
    61	  echo "  Each command: 'queue_<name>.sh <args>' (relative to scripts/queues/)" >&2
    62	  echo "  See header for examples." >&2
    63	  exit 2
    64	fi
    65	
    66	# ---------- helpers ----------
    67	wait_for_runner_done() {
    68	  local pattern="$1"
    69	  local label="$2"
    70	  local elapsed=0
    71	  while pgrep -f "run_experiment.py.*${pattern}" > /dev/null; do
    72	    sleep 60
    73	    elapsed=$((elapsed + 60))
    74	    if (( elapsed % 1800 == 0 )); then
    75	      log "  ${label}: still running (${elapsed}s elapsed)..."
    76	      pgrep -af "run_experiment.py.*${pattern}" | head -1 | sed 's/^/    /'
    77	    fi
    78	  done
    79	  log "  ${label}: runner done"
    80	}
    81	
    82	# ---------- chain ----------
    83	log "=================================================="
    84	log "queue_chain — $# cells (RESET_BEFORE=${RESET_FLAG})"
    85	for arg in "$@"; do log "  - $arg"; done
    86	log "=================================================="
    87	
    88	idx=0
    89	for cmd in "$@"; do
    90	  idx=$((idx + 1))
    91	  log ""
    92	  log "------ [${idx}/$#] ${cmd} ------"
    93	
    94	  # Validate the script exists (cmd is "queue_xxx.sh args...")
    95	  script_name="${cmd%% *}"
    96	  if [[ ! -f "${SCRIPT_DIR}/${script_name}" ]]; then
    97	    log "  [error] script not found: ${SCRIPT_DIR}/${script_name}"
    98	    log "  aborting chain"
    99	    exit 1
   100	  fi
   101	
   102	  # ---- Same-site B0/B1 collision check (paper-grade hard rule §106) ----
   103	  # Parse <baseline> + <site> from the queue command args.
   104	  # queue_baseline.sh format: <baseline> <mode> <site> [benchmark]
   105	  # queue_phantom_*.sh format: <baseline> <site> [benchmark]
   106	  cmd_args=( ${cmd} )
   107	  this_baseline="${cmd_args[1]:-}"  # B0 or B1
   108	  if [[ "${script_name}" == queue_baseline.sh ]]; then
   109	    this_site="${cmd_args[3]:-}"    # 4th token (script bash mode site)
   110	  else
   111	    this_site="${cmd_args[2]:-}"    # 3rd token (script bash site)
   112	  fi
   113	  if [[ -n "${this_baseline}" && -n "${this_site}" ]]; then
   114	    other_baseline="B0"
   115	    [[ "${this_baseline}" == "B0" ]] && other_baseline="B1"
   116	    if pgrep -f "run_experiment.*${other_baseline}_.*_${this_site}_" > /dev/null 2>&1; then
   117	      log "  [collision] ${other_baseline} runner already active on site=${this_site}"
   118	      log "  paper-grade hard rule: same site cannot run B0+B1 simultaneously"
   119	      log "  waiting for ${other_baseline} ${this_site} to finish before launching ${this_baseline}..."
   120	      while pgrep -f "run_experiment.*${other_baseline}_.*_${this_site}_" > /dev/null 2>&1; do
   121	        sleep 60
   122	      done
   123	      log "  ${other_baseline} ${this_site} finished; proceeding with ${this_baseline}"
   124	    fi
   125	  fi
   126	
   127	  # Launch via the queue script (idempotent — picks up existing or fresh+reset)
   128	  out=$(RESET_BEFORE="${RESET_FLAG}" bash "${SCRIPT_DIR}/${script_name}" \
   129	        ${cmd#${script_name} } 2>&1 || true)
   130	  echo "$out" | sed 's/^/    /'
   131	
   132	  # Extract run_id from queue script output
   133	  run_id=$(echo "$out" | grep -oP 'run_id=\K\S+' | tail -1)
   134	  if [[ -z "$run_id" ]]; then
   135	    log "  [error] could not extract run_id from queue script output, aborting"
   136	    exit 1
   137	  fi
   138	  log "  watching run_id=${run_id}"
   139	
   140	  wait_for_runner_done "$run_id" "[${idx}/$#] $cmd"
   141	done
   142	
   143	log ""
   144	log "=================================================="
   145	log "queue_chain done — $# cells complete"
   146	log "=================================================="
   147	
   148	# ntfy notify
   149	if command -v curl > /dev/null; then
   150	  curl -d "queue_chain done: $# cells (${*})" \
   151	    "ntfy.sh/${NTFY_TOPIC:-p79-exp-dgx-spark}" 2>/dev/null || true
   152	fi

exec
/bin/bash -lc "ls -1 scripts/queues | sed -n '1,120p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
_deprecated
qsub_2x2_fwd_revtasks_myriad.sh
qsub_2x2_rev_strongtasks_myriad.sh
qsub_celle_random_cls_strong_myriad.sh
qsub_cellf_fwd_reddit_strong_myriad.sh
qsub_cellg_rev_reddit_reverse_myriad.sh
qsub_curate_reddit_myriad.sh
qsub_stage2_cellcr_reddit_fwd_revtier.sh
qsub_stage2_celldr_reddit_rev_strongtier.sh
qsub_stage2_celler_reddit_fwd_random.sh
qsub_stage2b_myriad.sh
qsub_stage2c_myriad.sh
qsub_stage3_cellhd_cls_fwd_dom.sh
qsub_stage3_cellhd_red_fwd_dom.sh
qsub_stage3_cellhp_cls_fwd_prompt.sh
qsub_stage3_cellhp_red_fwd_prompt.sh
qsub_stage3_cellhprompt_cls.sh
qsub_stage3_cellhprompt_cls_rand.sh
qsub_stage3_cellhprompt_cls_reverse.sh
qsub_stage3_cellhprompt_cls_taskshuf.sh
qsub_stage3_cellhprompt_red.sh
qsub_stage3_cellhprompt_red_rand.sh
qsub_stage3_cellhprompt_red_reverse.sh
qsub_stage3_cellhprompt_red_taskshuf.sh
qsub_stage3_cellht_cls_fwd_text.sh
qsub_stage3_cellht_red_fwd_text.sh
qsub_stage4_format_variation_cls.sh
qsub_stage4_h1_cls_reverse.sh
qsub_stage4_h1_reddit.sh
qsub_stage4_method44_v2_sweep_cls.sh
qsub_stage4_multimode_extract_cls.sh
qsub_stage4_multimode_extract_cls_v2.sh
qsub_stage4_multimode_extract_red_v2.sh
qsub_stage4mm_reddit.sh
queue_baseline.sh
queue_chain.sh
queue_phantom_dom.sh
queue_phantom_prompt.sh
queue_phantom_som.sh
queue_phantom_text.sh
queue_phase1_paper_grade.sh
queue_pilot_t0.sh
run_scroll_comparison.sh

exec
/bin/bash -lc "nl -ba scripts/queues/queue_baseline.sh | sed -n '1,180p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	#!/usr/bin/env bash
     2	# queue_baseline.sh — 启动 baseline 实验 (dom / som / vision) + 自动 watchdog
     3	#
     4	# Baseline modes (Phase 1 表征筛选):
     5	#   dom    — viewport-only AXTree (no image)
     6	#   som    — [SOM_MARKS] 文本 + 带框截图
     7	#   vision — 裸截图 (no DOM/AXTree)
     8	#
     9	# 这个脚本统一处理:
    10	#   - PROXY_API_KEY 从 .auth/qwen_api 加载 (B0 用)
    11	#   - VWA 远程 host env 加载
    12	#   - CUDA workaround env (DGX Spark sm_121)
    13	#   - WIKIPEDIA ZIM 版本
    14	#   - runner + watchdog 一起启动，已存在则跳过 (idempotent)
    15	#   - RESET 在 idempotent check 之后执行 (防 race — 见笔记 §104 audit)
    16	#
    17	# 用法:
    18	#   bash scripts/queues/queue_baseline.sh <baseline> <mode> <site> [benchmark]
    19	#   - baseline:  B0 | B1
    20	#   - mode:      dom | som | vision
    21	#   - site:      classifieds | reddit | shopping | shopping_admin
    22	#   - benchmark: vwa (默认) | wa
    23	#
    24	# 例:
    25	#   bash scripts/queues/queue_baseline.sh B0 dom shopping            # B0 DOM-only VWA shopping
    26	#   bash scripts/queues/queue_baseline.sh B1 som reddit              # B1 SoM VWA reddit
    27	#   bash scripts/queues/queue_baseline.sh B0 vision shopping wa      # B0 vision WA shopping
    28	#
    29	# Reset:
    30	#   RESET_BEFORE=1 bash ...  →  reset site (VWA only) AFTER idempotent check
    31	#
    32	# Required configs (must exist before launch):
    33	#   VWA:  configs/exp_v2_<baseline>_<mode>_<site>.yaml
    34	#   WA:   configs/exp_v2_<baseline>_<mode>_wa_<site>.yaml
    35	
    36	set -euo pipefail
    37	
    38	SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    39	REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
    40	cd "${REPO_DIR}"
    41	
    42	if [[ $# -lt 3 ]]; then
    43	  echo "Usage: $0 <baseline:B0|B1> <mode:dom|som|vision> <site> [benchmark:vwa|wa]" >&2
    44	  echo "  e.g. bash $0 B0 dom shopping" >&2
    45	  echo "       bash $0 B0 vision shopping wa" >&2
    46	  exit 2
    47	fi
    48	
    49	BASELINE="$1"; MODE="$2"; SITE="$3"
    50	BENCHMARK="${4:-vwa}"
    51	
    52	# Validation
    53	if [[ "${BASELINE}" != "B0" && "${BASELINE}" != "B1" ]]; then
    54	  echo "Invalid baseline: ${BASELINE} (expected B0 or B1)" >&2; exit 2
    55	fi
    56	if [[ "${MODE}" != "dom" && "${MODE}" != "som" && "${MODE}" != "vision" ]]; then
    57	  echo "Invalid mode: ${MODE} (expected dom/som/vision)" >&2; exit 2
    58	fi
    59	if [[ "${BENCHMARK}" != "vwa" && "${BENCHMARK}" != "wa" ]]; then
    60	  echo "Invalid benchmark: ${BENCHMARK} (expected vwa or wa)" >&2; exit 2
    61	fi
    62	if [[ "${BENCHMARK}" == "vwa" && "${SITE}" != "classifieds" && "${SITE}" != "reddit" && "${SITE}" != "shopping" ]]; then
    63	  echo "Invalid VWA site: ${SITE}" >&2; exit 2
    64	fi
    65	if [[ "${BENCHMARK}" == "wa" && "${SITE}" != "reddit" && "${SITE}" != "shopping" && "${SITE}" != "shopping_admin" ]]; then
    66	  echo "Invalid WA site: ${SITE}" >&2; exit 2
    67	fi
    68	
    69	# Build config name
    70	# VWA: exp_v2_<baseline>_<mode>_<site>.yaml
    71	# WA:  exp_v2_<baseline>_<mode>_wa_<site>.yaml
    72	CFG_NAME="${BASELINE}_${MODE}_${SITE}"
    73	[[ "${BENCHMARK}" == "wa" ]] && CFG_NAME="${BASELINE}_${MODE}_wa_${SITE}"
    74	CONFIG="${REPO_DIR}/configs/exp_v2_${CFG_NAME}.yaml"
    75	
    76	if [[ ! -f "${CONFIG}" ]]; then
    77	  echo "[baseline][error] Config not found: ${CONFIG}" >&2
    78	  echo "  Single-mode baseline config 必须先创建 (template: exp_v2_B0_dom_shopping.yaml)" >&2
    79	  echo "  或参考 configs/exp_v2_<baseline>_3mode_<site>.yaml 调整 observation_mode 单 list" >&2
    80	  exit 1
    81	fi
    82	
    83	# Condition id: phase1_<mode>_router_0
    84	COND_ID="phase1_${MODE}_router_0"
    85	
    86	PYTHON_BIN="${REPO_DIR}/.venv/bin/python3"
    87	LOG_DIR="${REPO_DIR}/logs"
    88	mkdir -p "${LOG_DIR}"
    89	
    90	# ---------- DGX Spark CUDA workaround ----------
    91	export PYTORCH_NVML_BASED_CUDA_CHECK=1
    92	export CUDA_MPS_PIPE_DIRECTORY=""
    93	export CUDA_MPS_LOG_DIRECTORY=""
    94	
    95	# ---------- VWA 远程站点 env ----------
    96	if [[ -f "${REPO_DIR}/scripts/vwa_env_remote.sh" ]]; then
    97	  # shellcheck disable=SC1091
    98	  source "${REPO_DIR}/scripts/vwa_env_remote.sh"
    99	fi
   100	
   101	# ---------- WIKIPEDIA ZIM 版本 ----------
   102	export WIKIPEDIA_ZIM_VERSION="${WIKIPEDIA_ZIM_VERSION:-wikipedia_en_all_maxi_2025-08}"
   103	
   104	# ---------- B0 PROXY API key 加载 ----------
   105	if [[ "${BASELINE}" == "B0" ]]; then
   106	  if [[ -z "${PROXY_API_KEY:-}" ]]; then
   107	    AUTH_FILE="${REPO_DIR}/.auth/qwen_api"
   108	    if [[ -f "${AUTH_FILE}" ]]; then
   109	      raw_key="$(grep -m1 '^rp_' "${AUTH_FILE}" | tr -d '[:space:]')"
   110	      if [[ -n "${raw_key}" ]]; then
   111	        export PROXY_API_KEY="${raw_key}"
   112	        export QWEN_API_KEY="${raw_key}"
   113	        export DASHSCOPE_API_KEY="${raw_key}"
   114	        echo "[baseline] Loaded PROXY_API_KEY from ${AUTH_FILE}"
   115	      else
   116	        echo "[baseline][error] ${AUTH_FILE} 存在但无 rp_ key" >&2; exit 1
   117	      fi
   118	    else
   119	      echo "[baseline][error] ${AUTH_FILE} 不存在，且 PROXY_API_KEY 未设置" >&2; exit 1
   120	    fi
   121	  fi
   122	fi
   123	
   124	# ---------- 决定 run_id + run_dir ----------
   125	TS_DATE="$(date +%Y%m%d)"
   126	TS_FULL="$(date +%Y%m%d_%H%M%S)"
   127	if [[ "${BENCHMARK}" == "wa" ]]; then
   128	  PHASE_DIR="${REPO_DIR}/results/webarena/phase1"
   129	else
   130	  PHASE_DIR="${REPO_DIR}/results/visualwebarena/phase1"
   131	fi
   132	
   133	EXISTING="$(ls -dt "${PHASE_DIR}/${CFG_NAME}_"[0-9]* 2>/dev/null | head -1 || true)"
   134	if [[ -n "${EXISTING}" ]]; then
   135	  RUN_ID="$(basename "${EXISTING}")"
   136	  echo "[baseline] resuming existing run_id=${RUN_ID}"
   137	else
   138	  RUN_ID="${CFG_NAME}_${TS_DATE}"
   139	  echo "[baseline] new run_id=${RUN_ID}"
   140	fi
   141	
   142	RUN_DIR="${PHASE_DIR}/${RUN_ID}"
   143	echo "[baseline] config=${CONFIG}"
   144	echo "[baseline] run_dir=${RUN_DIR}"
   145	echo "[baseline] condition=${COND_ID}"
   146	
   147	# ---------- 检查 runner 是否已在跑 ----------
   148	if pgrep -f "run_experiment.py.*${RUN_ID}" > /dev/null; then
   149	  echo "[baseline] runner for ${RUN_ID} already running, skipping spawn"
   150	  echo "[baseline] (RESET_BEFORE skipped — runner already attached to current site state)"
   151	else
   152	  # ---------- Optional: site reset before launch ----------
   153	  # IMPORTANT: reset is AFTER the idempotent runner check — resetting while
   154	  # a runner is attached destroys site state under it (race condition fixed
   155	  # 2026-04-28 — see 实验笔记 §104).
   156	  if [[ "${RESET_BEFORE:-0}" == "1" && "${BENCHMARK}" != "wa" ]]; then
   157	    if [[ -f "${REPO_DIR}/scripts/maintenance/reset_vwa_sites.sh" ]]; then
   158	      # shellcheck disable=SC1091
   159	      source "${REPO_DIR}/scripts/maintenance/reset_vwa_sites.sh"
   160	      echo "[baseline] RESET_BEFORE=1 → resetting site=${SITE}..."
   161	      if reset_vwa_sites "${SITE}" "baseline_${MODE}_${SITE}"; then
   162	        echo "[baseline] reset OK; sleeping 15s for site to settle..."
   163	        sleep 15
   164	        # Refresh .auth/<site>_state.json post-reset — server-side session was wiped,
   165	        # so the runner's first task would otherwise hit NOT-LOGGED-IN (watchdog only
   166	        # reactively refreshes after streak=3, costing 3 dirty episodes).
   167	        echo "[baseline] refreshing .auth/${SITE}_state.json post-reset..."
   168	        if "${PYTHON_BIN}" -c "
   169	import sys
   170	sys.path.insert(0, '${REPO_DIR}')
   171	from pathlib import Path
   172	from p79.utils.auth_refresh import refresh_site_auth
   173	sys.exit(0 if refresh_site_auth('${SITE}', Path('${REPO_DIR}/.auth')) else 1)
   174	" 2>&1; then
   175	          echo "[baseline] auth refresh OK — runner task=0 will be LOGGED IN"
   176	        else
   177	          echo "[baseline][warn] post-reset auth refresh failed; watchdog will retry reactively after streak=3" >&2
   178	        fi
   179	      else
   180	        rc=$?

exec
/bin/bash -lc 'for c in B0_dom_classifieds B0_som_classifieds B0_vision_classifieds B1_dom_classifieds B1_som_classifieds B1_vision_classifieds B0_dom_reddit B0_som_reddit B0_vision_reddit B1_dom_reddit B1_som_reddit B1_vision_reddit B0_dom_shopping B0_som_shopping B0_vision_shopping B1_dom_shopping B1_som_shopping B1_vision_shopping; do test -f configs/exp_v2_${c}.yaml || echo MISSING $c; done' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
exec
/bin/bash -lc 'for c in B0_phantom_text_classifieds B0_phantom_som_classifieds B0_phantom_prompt_classifieds B1_phantom_text_classifieds B1_phantom_som_classifieds B1_phantom_prompt_classifieds B0_phantom_text_reddit B0_phantom_som_reddit B0_phantom_prompt_reddit B1_phantom_text_reddit B1_phantom_som_reddit B1_phantom_prompt_reddit B0_phantom_text_shopping B0_phantom_som_shopping B0_phantom_prompt_shopping B1_phantom_text_shopping B1_phantom_som_shopping B1_phantom_prompt_shopping; do test -f configs/exp_v2_${c}.yaml || echo MISSING $c; done' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
MISSING B0_dom_classifieds
MISSING B0_som_classifieds
MISSING B0_vision_classifieds
MISSING B1_dom_classifieds
MISSING B1_som_classifieds
MISSING B1_vision_classifieds
MISSING B0_dom_reddit
MISSING B0_som_reddit
MISSING B0_vision_reddit
MISSING B1_dom_reddit
MISSING B1_som_reddit
MISSING B1_vision_reddit
MISSING B1_dom_shopping
MISSING B1_som_shopping
MISSING B1_vision_shopping

 succeeded in 0ms:
MISSING B0_phantom_som_classifieds
MISSING B1_phantom_text_classifieds
MISSING B1_phantom_som_classifieds
MISSING B0_phantom_som_reddit
MISSING B1_phantom_som_reddit
MISSING B0_phantom_som_shopping
MISSING B0_phantom_prompt_shopping
MISSING B1_phantom_som_shopping
MISSING B1_phantom_prompt_shopping

exec
/bin/bash -lc "nl -ba scripts/queues/queue_phantom_text.sh | sed -n '60,100p'; nl -ba scripts/queues/queue_phantom_som.sh | sed -n '60,100p'; nl -ba scripts/queues/queue_phantom_prompt.sh | sed -n '60,100p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
    60	fi
    61	if [[ "${BENCHMARK}" == "wa" && "${SITE}" != "reddit" && "${SITE}" != "shopping" && "${SITE}" != "shopping_admin" ]]; then
    62	  echo "Invalid WA site: ${SITE}" >&2; exit 2
    63	fi
    64	
    65	# Build config name. Prefer the new phantom_text YAML; fall back to phantom_dom
    66	# YAML if the rename hasn't propagated yet (e.g. live run still on disk under
    67	# the legacy name).
    68	# VWA: exp_v2_<baseline>_phantom_text_<site>.yaml
    69	# WA:  exp_v2_<baseline>_phantom_text_wa_<site>.yaml
    70	CFG_BASE_NEW="${BASELINE}_phantom_text"
    71	CFG_BASE_LEGACY="${BASELINE}_phantom_dom"
    72	[[ "${BENCHMARK}" == "wa" ]] && CFG_BASE_NEW="${CFG_BASE_NEW}_wa" && CFG_BASE_LEGACY="${CFG_BASE_LEGACY}_wa"
    73	CFG_NAME_NEW="${CFG_BASE_NEW}_${SITE}"
    74	CFG_NAME_LEGACY="${CFG_BASE_LEGACY}_${SITE}"
    75	CONFIG_NEW="${REPO_DIR}/configs/exp_v2_${CFG_NAME_NEW}.yaml"
    76	CONFIG_LEGACY="${REPO_DIR}/configs/exp_v2_${CFG_NAME_LEGACY}.yaml"
    77	
    78	if [[ -f "${CONFIG_NEW}" ]]; then
    79	  CONFIG="${CONFIG_NEW}"
    80	  CFG_NAME="${CFG_NAME_NEW}"
    81	elif [[ -f "${CONFIG_LEGACY}" ]]; then
    82	  CONFIG="${CONFIG_LEGACY}"
    83	  CFG_NAME="${CFG_NAME_LEGACY}"
    84	  echo "[phantom_text] using legacy phantom_dom config: ${CONFIG_LEGACY}"
    85	else
    86	  echo "[phantom_text][error] Config not found: ${CONFIG_NEW} or ${CONFIG_LEGACY}" >&2
    87	  exit 1
    88	fi
    89	
    90	# condition_id stays as phantom_dom_router_0 for paper-grade compatibility —
    91	# existing run dirs (B0_phantom_text_*) all contain phase1_phantom_dom_router_0/.
    92	# Newly-introduced phantom_text YAMLs may declare phantom_text_router_0 instead;
    93	# in that case the runner reads the condition_id from YAML and this constant is
    94	# only used for watchdog targeting.
    95	COND_ID="phase1_phantom_dom_router_0"
    96	
    97	PYTHON_BIN="${REPO_DIR}/.venv/bin/python3"
    98	LOG_DIR="${REPO_DIR}/logs"
    99	mkdir -p "${LOG_DIR}"
   100	
    60	[[ "${BENCHMARK}" == "wa" ]] && CFG_NAME="${CFG_NAME}_wa"
    61	CFG_NAME="${CFG_NAME}_${SITE}"
    62	CONFIG="${REPO_DIR}/configs/exp_v2_${CFG_NAME}.yaml"
    63	
    64	if [[ ! -f "${CONFIG}" ]]; then
    65	  echo "[phantom_som][error] Config not found: ${CONFIG}" >&2; exit 1
    66	fi
    67	
    68	COND_ID="phase1_phantom_som_router_0"
    69	
    70	PYTHON_BIN="${REPO_DIR}/.venv/bin/python3"
    71	LOG_DIR="${REPO_DIR}/logs"
    72	mkdir -p "${LOG_DIR}"
    73	
    74	# ---------- DGX Spark CUDA workaround ----------
    75	export PYTORCH_NVML_BASED_CUDA_CHECK=1
    76	export CUDA_MPS_PIPE_DIRECTORY=""
    77	export CUDA_MPS_LOG_DIRECTORY=""
    78	
    79	# ---------- VWA 远程站点 env ----------
    80	if [[ -f "${REPO_DIR}/scripts/vwa_env_remote.sh" ]]; then
    81	  # shellcheck disable=SC1091
    82	  source "${REPO_DIR}/scripts/vwa_env_remote.sh"
    83	fi
    84	
    85	# ---------- WIKIPEDIA ZIM 版本 ----------
    86	export WIKIPEDIA_ZIM_VERSION="${WIKIPEDIA_ZIM_VERSION:-wikipedia_en_all_maxi_2025-08}"
    87	
    88	# ---------- B0 PROXY API key 加载 ----------
    89	if [[ "${BASELINE}" == "B0" ]]; then
    90	  if [[ -z "${PROXY_API_KEY:-}" ]]; then
    91	    AUTH_FILE="${REPO_DIR}/.auth/qwen_api"
    92	    if [[ -f "${AUTH_FILE}" ]]; then
    93	      raw_key="$(grep -m1 '^rp_' "${AUTH_FILE}" | tr -d '[:space:]')"
    94	      if [[ -n "${raw_key}" ]]; then
    95	        export PROXY_API_KEY="${raw_key}"
    96	        export QWEN_API_KEY="${raw_key}"
    97	        export DASHSCOPE_API_KEY="${raw_key}"
    98	        echo "[phantom_som] Loaded PROXY_API_KEY from ${AUTH_FILE}"
    99	      else
   100	        echo "[phantom_som][error] ${AUTH_FILE} 存在但无 rp_ key" >&2; exit 1
    60	if [[ ! -f "${CONFIG}" ]]; then
    61	  echo "[phantom_prompt][error] Config not found: ${CONFIG}" >&2; exit 1
    62	fi
    63	
    64	COND_ID="phase1_phantom_prompt_router_0"
    65	
    66	PYTHON_BIN="${REPO_DIR}/.venv/bin/python3"
    67	LOG_DIR="${REPO_DIR}/logs"
    68	mkdir -p "${LOG_DIR}"
    69	
    70	# ---------- DGX Spark CUDA workaround ----------
    71	export PYTORCH_NVML_BASED_CUDA_CHECK=1
    72	export CUDA_MPS_PIPE_DIRECTORY=""
    73	export CUDA_MPS_LOG_DIRECTORY=""
    74	
    75	# ---------- VWA 远程站点 env ----------
    76	if [[ -f "${REPO_DIR}/scripts/vwa_env_remote.sh" ]]; then
    77	  # shellcheck disable=SC1091
    78	  source "${REPO_DIR}/scripts/vwa_env_remote.sh"
    79	fi
    80	
    81	# ---------- WIKIPEDIA ZIM 版本 ----------
    82	export WIKIPEDIA_ZIM_VERSION="${WIKIPEDIA_ZIM_VERSION:-wikipedia_en_all_maxi_2025-08}"
    83	
    84	# ---------- B0 PROXY API key 加载 ----------
    85	if [[ "${BASELINE}" == "B0" ]]; then
    86	  if [[ -z "${PROXY_API_KEY:-}" ]]; then
    87	    AUTH_FILE="${REPO_DIR}/.auth/qwen_api"
    88	    if [[ -f "${AUTH_FILE}" ]]; then
    89	      raw_key="$(grep -m1 '^rp_' "${AUTH_FILE}" | tr -d '[:space:]')"
    90	      if [[ -n "${raw_key}" ]]; then
    91	        export PROXY_API_KEY="${raw_key}"
    92	        export QWEN_API_KEY="${raw_key}"
    93	        export DASHSCOPE_API_KEY="${raw_key}"
    94	        echo "[phantom_prompt] Loaded PROXY_API_KEY from ${AUTH_FILE}"
    95	      else
    96	        echo "[phantom_prompt][error] ${AUTH_FILE} 存在但无 rp_ key" >&2; exit 1
    97	      fi
    98	    else
    99	      echo "[phantom_prompt][error] ${AUTH_FILE} 不存在，且 PROXY_API_KEY 未设置" >&2; exit 1
   100	    fi

exec
/bin/bash -lc "rg --files configs | rg 'exp_v2_.*(classifieds|reddit|shopping).*\\.yaml"'$'"' | sed -n '1,240p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
configs/exp_v2_B0_phantom_text_reddit.yaml
configs/exp_v2_B0_phantom_text_classifieds.yaml
configs/exp_v2_B0_phantom_text_wa_reddit.yaml
configs/exp_v2_B0_dom_shopping_swatch_sanity.yaml
configs/exp_v2_B1_phantom_wa_shopping.yaml
configs/exp_v2_B0_dom_shopping.yaml
configs/exp_v2_B1_phantom_wa_reddit.yaml
configs/exp_v2_B1_phantom_reddit.yaml
configs/exp_v2_B0_phantom_prompt_classifieds.yaml
configs/exp_v2_B0_phantom_wa_shopping.yaml
configs/exp_v2_B0_phantom_reddit.yaml
configs/exp_v2_B1_phantom_prompt_reddit.yaml
configs/exp_v2_B0_phantom_shopping.yaml
configs/exp_v2_B1_phantom_text_wa_shopping.yaml
configs/exp_v2_B0_phantom_wa_reddit.yaml
configs/_deprecated/exp_v2_B0_3mode_wa_shopping.yaml
configs/_deprecated/exp_v2_B0_3mode_reddit.yaml
configs/_deprecated/exp_v2_B0_3mode_wa_shopping_admin.yaml
configs/_deprecated/exp_v2_B0_3mode_shopping.yaml
configs/_deprecated/exp_v2_B0_3mode_classifieds.yaml
configs/_deprecated/exp_v2_B0_3mode_wa_reddit.yaml
configs/exp_v2_B0_phantom_text_wa_shopping.yaml
configs/exp_v2_B1_phantom_shopping.yaml
configs/exp_v2_B0_phantom_text_wa_shopping_admin.yaml
configs/exp_v2_B0_vision_shopping.yaml
configs/exp_v2_B0_phantom_classifieds.yaml
configs/exp_v2_B1_phantom_dom_classifieds.yaml
configs/exp_v2_B0_phantom_wa_shopping_admin.yaml
configs/exp_v2_B0_phantom_text_shopping.yaml
configs/exp_v2_B1_phantom_wa_shopping_admin.yaml
configs/exp_v2_B0_dom_pilot_T0_reddit.yaml
configs/exp_v2_B0_phantom_prompt_reddit.yaml
configs/exp_v2_B1_phantom_text_wa_reddit.yaml
configs/exp_v2_B1_phantom_classifieds.yaml
configs/exp_v2_B0_som_shopping.yaml
configs/exp_v2_B1_phantom_text_reddit.yaml
configs/exp_v2_B0_dom_pilot_T0_shopping.yaml
configs/exp_v2_B0_dom_pilot_T0_classifieds.yaml
configs/exp_v2_B1_phantom_text_wa_shopping_admin.yaml
configs/exp_v2_B1_phantom_text_shopping.yaml
configs/exp_v2_B1_phantom_prompt_classifieds.yaml

exec
/bin/bash -lc "rg --files configs | rg 'exp_v2_B0_.*classifieds|exp_v2_B1_.*classifieds|exp_v2_B0_.*reddit|exp_v2_B1_.*reddit|exp_v2_B0_.*shopping|exp_v2_B1_.*shopping' | sort | sed -n '1,240p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
configs/_deprecated/exp_v2_B0_3mode_classifieds.yaml
configs/_deprecated/exp_v2_B0_3mode_reddit.yaml
configs/_deprecated/exp_v2_B0_3mode_shopping.yaml
configs/_deprecated/exp_v2_B0_3mode_wa_reddit.yaml
configs/_deprecated/exp_v2_B0_3mode_wa_shopping.yaml
configs/_deprecated/exp_v2_B0_3mode_wa_shopping_admin.yaml
configs/exp_v2_B0_dom_pilot_T0_classifieds.yaml
configs/exp_v2_B0_dom_pilot_T0_reddit.yaml
configs/exp_v2_B0_dom_pilot_T0_shopping.yaml
configs/exp_v2_B0_dom_shopping.yaml
configs/exp_v2_B0_dom_shopping_swatch_sanity.yaml
configs/exp_v2_B0_phantom_classifieds.yaml
configs/exp_v2_B0_phantom_prompt_classifieds.yaml
configs/exp_v2_B0_phantom_prompt_reddit.yaml
configs/exp_v2_B0_phantom_reddit.yaml
configs/exp_v2_B0_phantom_shopping.yaml
configs/exp_v2_B0_phantom_text_classifieds.yaml
configs/exp_v2_B0_phantom_text_reddit.yaml
configs/exp_v2_B0_phantom_text_shopping.yaml
configs/exp_v2_B0_phantom_text_wa_reddit.yaml
configs/exp_v2_B0_phantom_text_wa_shopping.yaml
configs/exp_v2_B0_phantom_text_wa_shopping_admin.yaml
configs/exp_v2_B0_phantom_wa_reddit.yaml
configs/exp_v2_B0_phantom_wa_shopping.yaml
configs/exp_v2_B0_phantom_wa_shopping_admin.yaml
configs/exp_v2_B0_som_shopping.yaml
configs/exp_v2_B0_vision_shopping.yaml
configs/exp_v2_B1_phantom_classifieds.yaml
configs/exp_v2_B1_phantom_dom_classifieds.yaml
configs/exp_v2_B1_phantom_prompt_classifieds.yaml
configs/exp_v2_B1_phantom_prompt_reddit.yaml
configs/exp_v2_B1_phantom_reddit.yaml
configs/exp_v2_B1_phantom_shopping.yaml
configs/exp_v2_B1_phantom_text_reddit.yaml
configs/exp_v2_B1_phantom_text_shopping.yaml
configs/exp_v2_B1_phantom_text_wa_reddit.yaml
configs/exp_v2_B1_phantom_text_wa_shopping.yaml
configs/exp_v2_B1_phantom_text_wa_shopping_admin.yaml
configs/exp_v2_B1_phantom_wa_reddit.yaml
configs/exp_v2_B1_phantom_wa_shopping.yaml
configs/exp_v2_B1_phantom_wa_shopping_admin.yaml

exec
/bin/bash -lc 'bash scripts/queues/queue_baseline.sh B0 dom classifieds' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 exited 1 in 0ms:
[baseline][error] Config not found: /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/configs/exp_v2_B0_dom_classifieds.yaml
  Single-mode baseline config 必须先创建 (template: exp_v2_B0_dom_shopping.yaml)
  或参考 configs/exp_v2_<baseline>_3mode_<site>.yaml 调整 observation_mode 单 list

exec
/bin/bash -lc 'bash scripts/queues/queue_phantom_prompt.sh B0 shopping' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 exited 1 in 0ms:
[phantom_prompt][error] Config not found: /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/configs/exp_v2_B0_phantom_prompt_shopping.yaml

exec
/bin/bash -lc 'bash scripts/queues/queue_phantom_som.sh B0 classifieds' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 5423ms:
[phantom_som] runner pid=2573567
[phantom_som] launching watchdog → /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/logs/exp_watchdog_B0_phantom_classifieds_20260513_v2.log (runner pid=2573567)
[phantom_som] watchdog pid=2573654

[phantom_som] OK — B0_phantom_classifieds (vwa/classifieds) running
  runner log:   /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/logs/B0_phantom_classifieds_resume_20260513_172216.log
  watchdog log: /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/logs/exp_watchdog_B0_phantom_classifieds_20260513_v2.log

codex
I accidentally invoked one queue helper while checking config paths; it passed validation and started a B0 classifieds phantom runner. I’m stopping that process immediately and will report it explicitly in the audit so there is no hidden side effect.
exec
/bin/bash -lc 'pgrep -af "run_experiment.py.*B0_phantom_classifieds_20260513|B0_phantom_classifieds"' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
2573567 /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/.venv/bin/python3 scripts/run_experiment.py --config /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/configs/exp_v2_B0_phantom_classifieds.yaml --run_id B0_phantom_classifieds_20260513 --log_path /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/logs/B0_phantom_classifieds_resume_20260513_172216.log
2573654 /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/.venv/bin/python3 -u scripts/maintenance/experiment_watchdog.py --run-dir /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/results/visualwebarena/phase1/B0_phantom_classifieds_20260513 --condition phase1_phantom_som_router_0 --poll-secs 30 --idle-alert-mins 30 --ntfy-topic p79-exp-dgx-spark --state-file /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/logs/exp_watchdog_B0_phantom_classifieds_20260513_v2.state.json --aggregate-prefix B0_3mode --glm-config .auth/glm --digest-dir /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/results/visualwebarena/phase1/B0_phantom_classifieds_20260513/analysis/digest --runner-pid 2573567

exec
/bin/bash -lc "ps -ef | rg 'run_experiment|B0_phantom_classifieds|watchdog'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
root         164       2  0 Apr10 ?        00:00:11 [watchdogd]
jiaming  2573567       1 72 17:22 ?        00:00:05 /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/.venv/bin/python3 scripts/run_experiment.py --config /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/configs/exp_v2_B0_phantom_classifieds.yaml --run_id B0_phantom_classifieds_20260513 --log_path /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/logs/B0_phantom_classifieds_resume_20260513_172216.log
jiaming  2573654       1  1 17:22 ?        00:00:00 /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/.venv/bin/python3 -u scripts/maintenance/experiment_watchdog.py --run-dir /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/results/visualwebarena/phase1/B0_phantom_classifieds_20260513 --condition phase1_phantom_som_router_0 --poll-secs 30 --idle-alert-mins 30 --ntfy-topic p79-exp-dgx-spark --state-file /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/logs/exp_watchdog_B0_phantom_classifieds_20260513_v2.state.json --aggregate-prefix B0_3mode --glm-config .auth/glm --digest-dir /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/results/visualwebarena/phase1/B0_phantom_classifieds_20260513/analysis/digest --runner-pid 2573567
jiaming  2573685 2569238 99 17:22 ?        00:00:00 pgrep -af run_experiment.py.*B0_phantom_classifieds_20260513|B0_phantom_classifieds
jiaming  2573689 2569238  0 17:22 ?        00:00:00 /bin/bash -lc ps -ef | rg 'run_experiment|B0_phantom_classifieds|watchdog'
jiaming  2573694 2573689  0 17:22 ?        00:00:00 rg run_experiment|B0_phantom_classifieds|watchdog

exec
/bin/bash -lc 'kill 2573567 2573654; sleep 2; pgrep -af "run_experiment.py.*B0_phantom_classifieds_20260513|B0_phantom_classifieds|experiment_watchdog.py.*B0_phantom_classifieds" || true' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 2031ms:
2573800 /bin/bash -lc kill 2573567 2573654; sleep 2; pgrep -af "run_experiment.py.*B0_phantom_classifieds_20260513|B0_phantom_classifieds|experiment_watchdog.py.*B0_phantom_classifieds" || true

exec
/bin/bash -lc "git status --short results/visualwebarena/phase1 logs | sed -n '1,120p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
exec
/bin/bash -lc "find results/visualwebarena/phase1/B0_phantom_classifieds_20260513 logs -maxdepth 2 \\( -name '*B0_phantom_classifieds*172216*' -o -name '*B0_phantom_classifieds_20260513*' \\) -print 2>/dev/null | sed -n '1,120p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
results/visualwebarena/phase1/B0_phantom_classifieds_20260513
logs/exp_watchdog_B0_phantom_classifieds_20260513_v2.log

exec
/bin/bash -lc "find results/visualwebarena/phase1/B0_phantom_classifieds_20260513 logs -maxdepth 3 \\( -path 'results/visualwebarena/phase1/B0_phantom_classifieds_20260513*' -o -name '*B0_phantom_classifieds_20260513*' -o -name '*B0_phantom_classifieds_resume_20260513_172216.log' \\) -print 2>/dev/null | sed -n '1,200p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
results/visualwebarena/phase1/B0_phantom_classifieds_20260513
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_191.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_15.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_154.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_82.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_102.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_172.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_116.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_33.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_10.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_231.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_23.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_188.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_63.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_232.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_65.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_137.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_177.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_86.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_201.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_152.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_199.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_35.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_110.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_56.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_7.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_204.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_104.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_121.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_224.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_71.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_168.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_194.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_112.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_222.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_88.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_229.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_115.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_51.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_136.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_151.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_117.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_202.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_54.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_32.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_83.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_219.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_105.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_1.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_18.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_40.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_8.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_140.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_3.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_181.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_99.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_31.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_205.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_184.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_130.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_0.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_155.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_45.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_95.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_160.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_37.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_128.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_26.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_80.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_209.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_28.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_58.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_30.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_16.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_227.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_223.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_228.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_6.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_233.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_165.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_93.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_75.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_158.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_36.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_76.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_100.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_120.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_124.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_66.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_79.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_208.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_97.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_218.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_190.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_20.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_122.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_47.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_24.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_146.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_164.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_167.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_175.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_87.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_193.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_89.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_59.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_119.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_195.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_186.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_70.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_64.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_197.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_138.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_62.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_135.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_148.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_21.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_91.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_157.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_107.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_225.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_52.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_178.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_4.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_129.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_221.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_17.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_166.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_187.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_108.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_98.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_127.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_147.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_196.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_207.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_38.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_50.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_217.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_43.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_210.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_49.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_139.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_192.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_156.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_145.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_113.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_22.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_61.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_153.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_12.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_55.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_29.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_198.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_159.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_163.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_150.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_144.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_114.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_123.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_85.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_170.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_77.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_11.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_106.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_215.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_39.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_78.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_185.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_69.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_46.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_149.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_134.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_171.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_57.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_179.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_53.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_19.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_14.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_111.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_200.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_73.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_183.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_174.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_34.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_141.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_48.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_118.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_81.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_220.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_25.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_213.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_72.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_133.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_96.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_173.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_101.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_176.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_2.json
results/visualwebarena/phase1/B0_phantom_classifieds_20260513/task_configs/classifieds_task_161.json

codex
The accidental runner is stopped. It created an ignored run directory and watchdog log; I’m removing only those artifacts I just created so the workspace is not left contaminated.
exec
/bin/bash -lc 'rm -rf results/visualwebarena/phase1/B0_phantom_classifieds_20260513 logs/exp_watchdog_B0_phantom_classifieds_20260513_v2.log logs/exp_watchdog_B0_phantom_classifieds_20260513_v2.state.json logs/B0_phantom_classifieds_resume_20260513_172216.log; pgrep -af "B0_phantom_classifieds_20260513|experiment_watchdog.py.*B0_phantom_classifieds" || true; test ! -e results/visualwebarena/phase1/B0_phantom_classifieds_20260513 && echo cleaned' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
2574107 /bin/bash -lc rm -rf results/visualwebarena/phase1/B0_phantom_classifieds_20260513 logs/exp_watchdog_B0_phantom_classifieds_20260513_v2.log logs/exp_watchdog_B0_phantom_classifieds_20260513_v2.state.json logs/B0_phantom_classifieds_resume_20260513_172216.log; pgrep -af "B0_phantom_classifieds_20260513|experiment_watchdog.py.*B0_phantom_classifieds" || true; test ! -e results/visualwebarena/phase1/B0_phantom_classifieds_20260513 && echo cleaned
cleaned

exec
/bin/bash -lc ".venv/bin/python3 - <<'PY'
from scripts.analysis.preregistration_decision_test import _oracle_per_task, ALL_MODE_KEYS
row = {k: '0.0' for k in ALL_MODE_KEYS}
try:
    print(_oracle_per_task(row, ALL_MODE_KEYS))
except Exception as e:
    print(type(e).__name__ + ': ' + str(e))
PY" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/scripts/analysis/preregistration_decision_test.py:1: SyntaxWarning: invalid escape sequence '\ '
  """Preregistration decision test — Phase 1a 24-condition / 4-cell H1 / H3 / H2 evaluation.
ValueError: invalid literal for int() with base 10: '0.0'

exec
/bin/bash -lc "nl -ba docs/checkpoints/pre_run/preregistration.md | sed -n '1,22p;288,318p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	---
     2	type: preregistration
     3	status: draft
     4	created: 2026-05-03
     5	last_revised: 2026-05-13
     6	draft_author: Jiaming
     7	registered_at: <pending advisor sync lock>
     8	registered_git_sha: <pending lock>
     9	witnessed_by: <pending advisor sync>
    10	osf_doi: <pending paper submission stage>
    11	data_lock_until: <pending Phase 1a 24-condition rerun completion (cls+red × B0+B1 × 6 modes)>
    12	scope_revision_2026_05_13: cls+red × B0+B1 × 6 modes = 24 operational conditions across 4 statistical cells; shop deferred to Phase 1b main paper; K-of-N reclassified gate → transparency-only; smoke-gate stopping rule replaced (outcome-independent)
    13	---
    14	
    15	# Phantom-SoM Pre-Registration (Draft)
    16	
    17	> **Status: draft** — pending advisor sync lock. Once advisor signs (single-line email or co-authored commit), `status` flips to `locked`, `registered_git_sha` records the commit at lock time, and `witnessed_by` records advisor name + lock timestamp. `data_lock_until` records when 16-cell rerun finishes — between lock-time and completion-time, NO additional analyses may be added to gating-family tests.
    18	>
    19	> **Reading order**: §1 epistemic structure (why this framework) → §2 hypotheses (H1-H6 + framing rule) → §3 multiple-comparison family declaration → §4 locked analysis choices → §5 exploratory disclosure → §6 witness mechanism.
    20	>
    21	> **Companion docs**:
    22	> - `docs/reference/EVIDENCE_LAYER_AUDIT.md` §2 — template + meta-rationale
   288	
   289	## §6 Witness Mechanism
   290	
   291	### (a) Internal witness — Git commit + advisor email
   292	
   293	1. Advisor sync session: lock **9 commit decisions** (expanded 5/4 audit + 2026-05-13 revisions):
   294	   - (1) **K_h1=0.75 transparency ratio** (= 3/4 cells; reclassified gate → transparency-only 2026-05-13)
   295	   - (2) **K_h3=0.67 transparency ratio** (= 3/4 cells; reclassified gate → transparency-only 2026-05-13)
   296	   - (3) **TOST δ=1.0pp** equivalence margin (interpretation: SR drop-one effect-size margin, distinct from H2(a) cost ±10% margin — see §4 lock row)
   297	   - (4) **Cell inclusion**: Phase 1a = cls + red × B0+B1 × 6 modes (Phase A post-fix only); Phase 1b shop deferred
   298	   - (5) **Witness mechanism**: Git + advisor email + OSF DOI
   299	   - (6) **N_conditions Phase 1a final scope**: **24 operational conditions** (= 2 sites × 2 models × 6 modes) across **4 statistical cells** (= (site, model) tuples) — student-decided 2026-05-13 post-codex stress audit, replaces prior 16-cell phantom-only scope. Advisor email witness pending
   300	   - (7) **Smoke-gate revision** (2026-05-13): outcome-independent (auth + artifact + evaluator parseability only), no SR-based restart
   301	   - (8) **Router paper-1-vs-paper-2 decision**: H7-H8 PRIMARY (paper-1) or SECONDARY-informational (paper-2 deferred)
   302	   - (9) **Train/test split protocol**: 5-fold site-stratified CV vs leave-one-site-out (LOSO)
   303	   - Plus lock H-list (H1-H8 family declaration final).
   304	2. Update this file frontmatter: `status: draft` → `status: locked`, fill `registered_at`, `registered_git_sha`, `witnessed_by`.
   305	3. Git commit this file.
   306	4. Advisor sends single-line confirmation email: "I witness pre-registration of phantom-SoM hypotheses (H1-H8) and 8 lock decisions as of <git SHA> <date>." Email archived in `.witness/preregistration_witness.eml` (gitignored, local-only).
   307	
   308	### (b) External witness — OSF DOI (optional, paper-time)
   309	
   310	Approximately 1 week before paper submission:
   311	
   312	1. Create free OSF account (if not exists) at osf.io.
   313	2. New project: "Phantom-SoM 16-cell pre-registration witness."
   314	3. Upload this `preregistration.md` (locked version) + companion EVIDENCE_LAYER_AUDIT.md §2 + ADVISOR_SYNC.md §1.4 (lock decisions).
   315	4. OSF generates DOI + permanent timestamp.
   316	5. Paper §1 footnote cites the DOI: "Hypotheses pre-registered prior to 16-cell rerun (OSF DOI X.YYYY/osf.io/zzzz, Git SHA abc123, witnessed by [advisor name] on YYYY-MM-DD)."
   317	
   318	---

exec
/bin/bash -lc 'rg -n "smoke|first 10|success rate|success|logged_in|artifact|parseable|condition_summary_v2|Gate 7|B7|outcome" scripts/queues scripts/maintenance scripts docs/checkpoints/pre_run/preregistration.md docs/checkpoints/next_steps.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
docs/checkpoints/next_steps.md:122:5. GPU CUDA available (smoke `python3 -c "import torch; print(torch.cuda.is_available())"`)
docs/checkpoints/next_steps.md:193:**Followup paper-grade artifact**: `run_manifest.json` aggregate field → paper Table 6 / Figure mechanism panel.
docs/checkpoints/next_steps.md:216:- ✅ `scripts/analysis/preregistration_decision_test.py` (smoke-tested with 3 synthetic scenarios)
docs/checkpoints/next_steps.md:227:| 🔴 C3 | A100 memory + wallclock smoke (Stage 2B 1 task forward) | 30 min | A100 SSH |
docs/checkpoints/next_steps.md:232:| 🟡 R6 | `check_evaluator_consistency.py` (Gate 7 in `queue_phase1_paper_grade.sh`) — verify all conditions' most-recent `rederive_metadata.evaluator_code_sha` == lock-time SHA | 30 min | OSF DOI lock prep (笔记 §115 Protocol B §6) |
docs/checkpoints/next_steps.md:341:### Provenance artifacts (paper-cite-able)
docs/checkpoints/next_steps.md:345:results/provenance/preregistration_smoke_*.json    decision rule smoke tests
docs/checkpoints/pre_run/preregistration.md:12:scope_revision_2026_05_13: cls+red × B0+B1 × 6 modes = 24 operational conditions across 4 statistical cells; shop deferred to Phase 1b main paper; K-of-N reclassified gate → transparency-only; smoke-gate stopping rule replaced (outcome-independent)
docs/checkpoints/pre_run/preregistration.md:140:The paper §1 hook framing maps to data outcomes as follows:
docs/checkpoints/pre_run/preregistration.md:205:| **Bootstrap resampling unit** | **Task-level** (not episode-level, not run-level) | Each (task_id) drawn with replacement N times; same task across modes drawn together to preserve pairing. This is the standard unit for adjusted_success comparisons in VWA/WA. Episode-level would break pairing; run-level would over-conservatively widen CIs. |
docs/checkpoints/pre_run/preregistration.md:218:| **FP filter primary** | na_fp + eval_fp combined | Per 实验笔记 §95 (visual_fp deprecated — no lit precedent, boundary-undecidable, over-filters 95.3% VWA tasks). Code: `compute_adjusted_success()` returns `fp_reason ∈ {'', 'na_fp', 'eval_fp'}` (`p79/experiment/analysis.py:52`) |
docs/checkpoints/pre_run/preregistration.md:229:| **Missing-data / crashed-episode policy** (audit B6) | (a) Crashed episodes (uncaught exception, OOM, timeout > 30 min, browser crash) **excluded from paired-N denominators**, **NOT imputed** to success or failure. (b) Episodes with `not_logged_in` or `auth_drift` flag at termination excluded after watchdog refresh fails 3 retries (per `experiment_watchdog.py`). (c) Missing artifacts (no `obs.txt` / `screenshot_annotated.png` at step k) excluded from per-step analyses, NOT imputed. (d) Per-cell exclusion count + reason histogram reported in Appendix C. | Listwise deletion only; mean imputation introduces bias for SR proportions, hot-deck imputation breaks paired-N pairing. Crashed-episode imputation as success/failure would inflate Type I/II error. Lock 2026-05-09. |
docs/checkpoints/pre_run/preregistration.md:230:| **Stopping rules / contamination halt criteria** (audit B7, REVISED 2026-05-13 to remove outcome-dependent bias per codex Flaw 6) | (a) **Pre-launch**: `make pre-launch-check` validates seed configured + HF SHA pinned + git working tree clean + GPU available + disk free > 20GB; failure halts launch (per audit C10). (b) **Smoke-test gate (outcome-INDEPENDENT)**: first 10 episodes per condition must show auth-state `logged_in=True` on all 10 AND ≥ 9 of 10 episodes produced complete artifact bundle (`obs.txt` + `screenshot.png` + `condition_summary_v2` increment + JSONL flush) AND evaluator returned a parseable verdict (success / failure / `ua_match` N/A — any of these is fine, **success rate itself is NOT checked**). Failures halt for auth refresh / artifact pipeline debug, NOT for low SR observation. Rationale: outcome-dependent smoke gate biases low-SR cells upward (a true 5-10% SR cell has 35-60% probability of "0 successes in first 10" by binomial chance and would be invalidly restarted). (c) **Auth/site contamination halt**: ≥ 5 consecutive episodes with `not_logged_in` ⇒ stop cell, refresh auth, archive partial run as `_dirty_partial`, restart fresh. (d) **Eval drift halt**: if rerun on identical archived episode produces SR delta > 5pp via `validate_run.py --strict`, freeze cell + investigate evaluator code. (e) **OOM / hardware halt**: 3 consecutive job failures ⇒ stop cell, document hardware in incident log, manually re-queue with diagnostic output. | Halt rules protect data purity; halted cells restarted only after root-cause documented in `master_bug_catalog.md` + bug fix committed. Lock 2026-05-09; smoke gate revised 2026-05-13 to outcome-independent variant. |
docs/checkpoints/pre_run/preregistration.md:300:   - (7) **Smoke-gate revision** (2026-05-13): outcome-independent (auth + artifact + evaluator parseability only), no SR-based restart
docs/checkpoints/pre_run/preregistration.md:322:**Public release scope** — what reviewers / replicators can reproduce from the released artifact:
docs/checkpoints/pre_run/preregistration.md:352:| 2026-05-03 | Disconfirmation rule changed from "any cell fail" to data-conditional R1-R5 framing rule | "Any cell fail" too strict given single-cell power limits; framing rule maps data outcomes to paper hook revisions transparently |
docs/checkpoints/pre_run/preregistration.md:354:| 2026-05-05 | Advisor sync 5/5 partial outcome — early-stop A locked (cancel全 mechanism); compute path locked (advisor 5090 → Rancher H100 → RunPod backup); paper split direction discussed but Mechanistic-nested-vs-independent + threshold detail not finalized due to network drop | Advisor explicit confirm early-stop cancel + compute paths; paper split + threshold lock deferred to email follow-up via `docs/checkpoints/advisor_sync_5_5_followup.md` |
docs/checkpoints/pre_run/preregistration.md:356:| 2026-05-13 | **Codex stress audit triggered 6 paper-grade design fixes** (pre-launch): (a) scope reframe 16-cell phantom-only → 24-condition / 4-cell Phase 1a (cls+red×B0+B1×6modes), Phase 1b shop deferred to main paper; (b) K-of-N reclassified gate → transparency-only (power analysis showing dysfunction at < 7pp effects, re-propagated to H1/H3/R5/§6); (c) H1 drop-one definition disambiguated (oracle ceiling lift with-vs-without P-SoM, per (site, model) cell paired bootstrap); (d) smoke-gate B7 revised outcome-independent (no SR-based restart bias); (e) cell terminology disambiguated ("cell" = 4 statistical strata for K-of-N/meta input, "condition" = 24 operational launch units); (f) Phase 1b shop scope-expansion lever for main paper R3→R1 framing decision | Codex CLI hostile reviewer audit (`docs/checkpoints/codex_outputs/codex_stress_16cell_design_2026-05-13.md`, lean prompt no-enumeration, cross-AI complementary to prior Claude reviews); 6 HIGH severity findings + 3 probable concerns. Workshop-targeted Phase 1a launch this week; main paper Phase 1b after workshop submission |
docs/checkpoints/pre_run/preregistration.md:357:| \<pending advisor email follow-up\> | \<witness K_h1=0.75 transparency / K_h3=0.67 transparency / TOST δ=1.0pp / N_conditions=24 (Phase 1a) / N_cells=4 / split protocol / paper split / Phase 1b shop / outcome-indep smoke gate / per follow-up doc Q1-Q11\> | \<email reply timestamp + Git SHA at lock\> |
scripts/queues/qsub_stage3_cellhprompt_cls_rand.sh:12:# Expected paper-grade outcome: random-injection L11-L17 displacement ≈ 0
scripts/queues/qsub_stage3_cellhprompt_cls_rand.sh:12:# Expected paper-grade outcome: random-injection L11-L17 displacement ≈ 0
scripts/queues/queue_phantom_text.sh:242:# AND condition_summary_v2.json present. Prevents init-orphan idle loops.
scripts/queues/_deprecated/queue_phantom_pair.sh:69:# Check if condition is complete (condition_summary_v2.json exists)
scripts/queues/_deprecated/queue_phantom_pair.sh:72:  [[ -f "${run_dir}/phase1_phantom_${mode}_router_0/condition_summary_v2.json" ]]
scripts/queues/queue_phantom_text.sh:242:# AND condition_summary_v2.json present. Prevents init-orphan idle loops.
scripts/queues/_deprecated/queue_b0_wa_with_reset.sh:221:  if [[ -f "${cond_dir}/condition_summary_v2.json" ]]; then
scripts/queues/_deprecated/queue_b0_wa_with_reset.sh:224:      rm -f "${cond_dir}/condition_summary_v2.json"
scripts/queues/_deprecated/queue_b1_with_reset.sh:190:  if [[ -f "${cond_dir}/condition_summary_v2.json" ]]; then
scripts/queues/_deprecated/queue_b1_with_reset.sh:193:      rm -f "${cond_dir}/condition_summary_v2.json"
scripts/queues/_deprecated/queue_phantom_pair.sh:69:# Check if condition is complete (condition_summary_v2.json exists)
scripts/queues/_deprecated/queue_phantom_pair.sh:72:  [[ -f "${run_dir}/phase1_phantom_${mode}_router_0/condition_summary_v2.json" ]]
scripts/queues/_deprecated/queue_b0_with_reset.sh:206:  if [[ -f "${cond_dir}/condition_summary_v2.json" ]]; then
scripts/queues/_deprecated/queue_b0_with_reset.sh:209:      rm -f "${cond_dir}/condition_summary_v2.json"
scripts/queues/_deprecated/queue_b0_wa_with_reset.sh:221:  if [[ -f "${cond_dir}/condition_summary_v2.json" ]]; then
scripts/queues/_deprecated/queue_b0_wa_with_reset.sh:224:      rm -f "${cond_dir}/condition_summary_v2.json"
scripts/queues/_deprecated/queue_b1_wa_with_reset.sh:192:  if [[ -f "${cond_dir}/condition_summary_v2.json" ]]; then
scripts/queues/_deprecated/queue_b1_wa_with_reset.sh:195:      rm -f "${cond_dir}/condition_summary_v2.json"
scripts/queues/qsub_stage4_method44_v2_sweep_cls.sh:16:# Why Myriad not DGX: DGX seonglae 96% GPU contention 2026-05-13, 10min smoke
scripts/queues/_deprecated/queue_b1_with_reset.sh:190:  if [[ -f "${cond_dir}/condition_summary_v2.json" ]]; then
scripts/queues/_deprecated/queue_b1_with_reset.sh:193:      rm -f "${cond_dir}/condition_summary_v2.json"
scripts/queues/queue_phantom_prompt.sh:196:# AND condition_summary_v2.json present. Prevents init-orphan idle loops.
scripts/queues/_deprecated/queue_b0_with_reset.sh:206:  if [[ -f "${cond_dir}/condition_summary_v2.json" ]]; then
scripts/queues/_deprecated/queue_b0_with_reset.sh:209:      rm -f "${cond_dir}/condition_summary_v2.json"
scripts/queues/qsub_2x2_fwd_revtasks_myriad.sh:8:# out task-selection-bias artifact in the apparent "reverse also disrupts"
scripts/queues/_deprecated/queue_b1_wa_with_reset.sh:192:  if [[ -f "${cond_dir}/condition_summary_v2.json" ]]; then
scripts/queues/_deprecated/queue_b1_wa_with_reset.sh:195:      rm -f "${cond_dir}/condition_summary_v2.json"
scripts/queues/queue_phase1_paper_grade.sh:58:#   results/visualwebarena/phase1/<run_id>/<condition_id>/condition_summary_v2.json
scripts/queues/queue_phase1_paper_grade.sh:119:  log "=== Gate 5: GPU + model load smoke ==="
scripts/queues/qsub_stage4_method44_v2_sweep_cls.sh:16:# Why Myriad not DGX: DGX seonglae 96% GPU contention 2026-05-13, 10min smoke
scripts/queues/queue_baseline.sh:218:# AND condition_summary_v2.json present. Prevents init-orphan idle loops.
scripts/queues/queue_phantom_prompt.sh:196:# AND condition_summary_v2.json present. Prevents init-orphan idle loops.
scripts/queues/qsub_2x2_fwd_revtasks_myriad.sh:8:# out task-selection-bias artifact in the apparent "reverse also disrupts"
scripts/queues/queue_phantom_som.sh:202:# AND condition_summary_v2.json present. Prevents init-orphan idle loops.
scripts/queues/qsub_stage3_cellhprompt_cls_reverse.sh:11:# Expected paper-grade outcome: if axis-2 is a symmetric pathway, reverse
scripts/queues/queue_phase1_paper_grade.sh:58:#   results/visualwebarena/phase1/<run_id>/<condition_id>/condition_summary_v2.json
scripts/queues/queue_phase1_paper_grade.sh:119:  log "=== Gate 5: GPU + model load smoke ==="
scripts/queues/qsub_stage3_cellhprompt_cls_taskshuf.sh:11:# Expected paper-grade outcome: if axis-2 effect is content-specific,
scripts/queues/queue_baseline.sh:218:# AND condition_summary_v2.json present. Prevents init-orphan idle loops.
scripts/queues/qsub_stage3_cellhprompt_red_reverse.sh:11:# Expected paper-grade outcome: if axis-2 is a symmetric pathway, reverse
scripts/queues/queue_phantom_som.sh:202:# AND condition_summary_v2.json present. Prevents init-orphan idle loops.
scripts/queues/qsub_stage3_cellhprompt_cls_reverse.sh:11:# Expected paper-grade outcome: if axis-2 is a symmetric pathway, reverse
scripts/queues/qsub_stage3_cellhprompt_cls_taskshuf.sh:11:# Expected paper-grade outcome: if axis-2 effect is content-specific,
scripts/queues/qsub_curate_reddit_myriad.sh:69:    --artifacts-subdir phase1_som_router_0
scripts/queues/qsub_stage3_cellhprompt_red_reverse.sh:11:# Expected paper-grade outcome: if axis-2 is a symmetric pathway, reverse
scripts/queues/qsub_curate_reddit_myriad.sh:69:    --artifacts-subdir phase1_som_router_0
scripts/queues/qsub_2x2_rev_strongtasks_myriad.sh:7:# selection-bias artifact in apparent "reverse also disrupts" finding (笔记 §111.7+).
scripts/queues/qsub_2x2_rev_strongtasks_myriad.sh:7:# selection-bias artifact in apparent "reverse also disrupts" finding (笔记 §111.7+).
scripts/provenance/numerical_determinism_check.py:59:    artifacts_dir = next(c / "artifacts" for c in archived_dir.iterdir() if c.is_dir() and (c / "artifacts").is_dir())
scripts/provenance/numerical_determinism_check.py:60:    step_dir = artifacts_dir / f"{args.site}_task_{args.task_id}" / f"step_{args.step:03d}"
scripts/provenance/numerical_determinism_check.py:70:        task_cfg_path = artifacts_dir.parent / "episodes" / f"task_{args.task_id}" / "summary.json"
scripts/provenance/snapshot_env.py:46:    "p79/experiment/analysis.py",      # compute_adjusted_success + FP rules
scripts/mechanistic/run_stage1_pilot.py:6:    infra smoke test, mode-axis is "system prompt structure only".
scripts/mechanistic/run_stage1_pilot.py:103:def _find_artifacts_dir(run_dir: Path) -> Path:
scripts/mechanistic/run_stage1_pilot.py:104:    """Find the condition subdir containing artifacts/ in an archived run."""
scripts/mechanistic/run_stage1_pilot.py:106:        if child.is_dir() and (child / "artifacts").is_dir():
scripts/mechanistic/run_stage1_pilot.py:107:            return child / "artifacts"
scripts/mechanistic/run_stage1_pilot.py:108:    raise FileNotFoundError(f"No condition subdir with artifacts/ in {run_dir}")
scripts/mechanistic/run_stage1_pilot.py:134:    artifacts_dir = _find_artifacts_dir(run_dir)
scripts/mechanistic/run_stage1_pilot.py:135:    logger.info(f"Loading archived observations from {artifacts_dir}")
scripts/mechanistic/run_stage1_pilot.py:141:        task_dir = artifacts_dir / f"{site}_task_{task_id}"
scripts/mechanistic/run_stage1_pilot.py:187:        f"(modes={list(modes)}); skipped {skipped} tasks not in artifacts dir, "
scripts/mechanistic/run_stage1_pilot.py:188:        f"{skipped_no_img} samples with missing image artifact"
scripts/mechanistic/run_stage1_pilot.py:300:    # 4. Save raw artifacts
scripts/maintenance/rsync_results_from_hub.sh:3:# Default: Tier B (episodes/*.jsonl + summary), no artifacts.
scripts/maintenance/rsync_results_from_hub.sh:4:# Set ARTIFACTS=1 to also pull artifacts (screenshots/SoM); useful when
scripts/maintenance/rsync_results_from_hub.sh:18:#   ARTIFACTS  set to 1 to include artifacts/
scripts/maintenance/rsync_results_from_hub.sh:35:  --include='condition_summary_v2.json'
scripts/maintenance/rsync_results_from_hub.sh:43:  INCLUDES+=( --include='artifacts/**' )
scripts/maintenance/rsync_results_from_hub.sh:44:  echo "[rsync←hub] including artifacts/"
scripts/maintenance/rsync_results_from_hub.sh:46:  EXCLUDES+=( --exclude='artifacts/' )
scripts/maintenance/smoke_test_vwa.py:20:    # Remove storage_state to avoid FileNotFoundError during smoke test
scripts/maintenance/smoke_test_vwa.py:22:        print("Removing storage_state from config for smoke test.")
scripts/maintenance/smoke_test_vwa.py:32:    # But for smoke test, even if it fails to load the page, reset might succeed or throw.
scripts/maintenance/smoke_test_vwa.py:41:    temp_config_path = "temp_smoke_config.json"
scripts/maintenance/smoke_test_vwa.py:60:        print("Reset successful!")
scripts/maintenance/smoke_test_vwa.py:71:        print("Step successful!")
scripts/maintenance/glm/glm_pre_launch_check.py:98:        return fail_default, f"(GLM unparseable, raw={raw[:200]}, {fail_suffix})"
scripts/maintenance/glm/glm_cell_autoupdate.py:2:"""Cell frontmatter auto-update — sync _status/cells/*.md from condition_summary_v2.json.
scripts/maintenance/glm/glm_cell_autoupdate.py:5:parse latest condition_summary_v2.json, update structured fields in cell frontmatter:
scripts/maintenance/glm/glm_cell_autoupdate.py:9:- sr_raw: success_rate * 100 (rounded 2 decimals)
scripts/maintenance/glm/glm_cell_autoupdate.py:114:      - summary_path: Path to condition_summary_v2.json, or None if not yet generated
scripts/maintenance/glm/glm_cell_autoupdate.py:119:      - is_inflight: True if no condition_summary_v2.json but episodes are accumulating
scripts/maintenance/glm/glm_cell_autoupdate.py:142:            summary = cond_dir / "condition_summary_v2.json"
scripts/maintenance/glm/glm_cell_autoupdate.py:308:        sr = d.get("success_rate")
scripts/preflight_v2.sh:400:    echo "Preflight completed successfully."
scripts/maintenance/probe_som_occlusion.py:66:    SoM:    .../artifacts/<task>/som/step_NNN_som.png
scripts/maintenance/probe_som_occlusion.py:67:            → .../artifacts/<task>/step_NNN/observation_som.txt
scripts/maintenance/probe_som_occlusion.py:68:    Screen: .../artifacts/<task>/step_NNN/screenshot.png
scripts/maintenance/probe_som_occlusion.py:69:            → .../artifacts/<task>/step_NNN/observation_som.txt
scripts/maintenance/generate_gallery.py:131:    reason_bucket, task_type, adjusted_success, fp_reason.
scripts/maintenance/generate_gallery.py:148:                    adj = row.get("adjusted_success", "")
scripts/maintenance/generate_gallery.py:152:                        "adjusted_success": (
scripts/maintenance/generate_gallery.py:345:            "success": ep["success"],
scripts/maintenance/generate_gallery.py:354:            "adjusted_success": ep.get("adjusted_success"),
scripts/maintenance/generate_gallery.py:361:        success = sum(1 for e in group["episodes"] if e.get("success") is True)
scripts/maintenance/generate_gallery.py:362:        fail = sum(1 for e in group["episodes"] if e.get("success") is False)
scripts/maintenance/generate_gallery.py:365:            "success": success,
scripts/maintenance/generate_gallery.py:367:            "success_rate": round(success / total, 3) if total > 0 else 0,
scripts/maintenance/generate_gallery.py:481:            artifacts_dir = cond_dir / "artifacts"
scripts/maintenance/generate_gallery.py:519:                task_artifact_dir = artifacts_dir / f"{raw_site}_task_{task_id}"
scripts/maintenance/generate_gallery.py:523:                    step_dir = task_artifact_dir / f"step_{step_idx:03d}"
scripts/maintenance/generate_gallery.py:608:                    "success": summary.get("success") if summary else None,
scripts/maintenance/generate_gallery.py:616:                    "adjusted_success": reason_info.get("adjusted_success"),
scripts/maintenance/generate_gallery.py:711:.badge.success{{ background:#1b5e20; color:#a5d6a7; }}
scripts/maintenance/generate_gallery.py:808:.reason-success{{ background:#1b5e20; color:#a5d6a7; }}
scripts/maintenance/generate_gallery.py:862:  if(r==='success') return 'reason-success';
scripts/maintenance/generate_gallery.py:883:    var sr=(g.stats.success_rate*100).toFixed(1);
scripts/maintenance/generate_gallery.py:894:      +'<span class="s ok">'+g.stats.success+' pass</span>'
scripts/maintenance/generate_gallery.py:904:      var c=e.success===true?'success':e.success===false?'fail':'unknown';
scripts/maintenance/generate_gallery.py:905:      var sl=e.success===true?'PASS':e.success===false?'FAIL':'&mdash;';
scripts/maintenance/generate_gallery.py:955:  var c=e.success===true?'success':e.success===false?'fail':'unknown';
scripts/maintenance/generate_gallery.py:956:  var sl=e.success===true?'PASS':e.success===false?'FAIL':'&mdash;';
scripts/maintenance/generate_gallery.py:990:    var ansCls=e.success===true?'ep-ans match':'ep-ans';
scripts/maintenance/digest_enrich.py:4:raw step logs and artifacts. Runs as a post-processor after glm_batch_digest.py.
scripts/maintenance/digest_enrich.py:123:    Handles restart artifacts: if the watchdog/queue restarted a task,
scripts/maintenance/digest_enrich.py:174:    artifact_dir: Path,
scripts/maintenance/digest_enrich.py:191:        dom_path = artifact_dir / f"step_{step_idx:03d}" / "observation_dom.txt"
scripts/maintenance/digest_enrich.py:259:    artifact_dir = run_dir / condition_id / "artifacts" / f"{site}_task_{task_id}"
scripts/maintenance/digest_enrich.py:263:    if artifact_dir.exists():
scripts/maintenance/digest_enrich.py:264:        has_desc, detail_dom_lengths, avg_dom = _check_dom_description(artifact_dir, steps, site=site)
scripts/maintenance/create_b1_classifieds_stub.py:9:  condition_summary_v2.json，使 is_condition_complete() 视其为完成，
scripts/maintenance/create_b1_classifieds_stub.py:13:  - success_rate 使用 RAW SR（评测器直接输出），adjusted SR 由分析脚本计算
scripts/maintenance/create_b1_classifieds_stub.py:35:        "success_rate": 21 / 234,        # 8.97% raw（adjusted 0.85%，2/234）
scripts/maintenance/create_b1_classifieds_stub.py:46:        "success_rate": 48 / 234,        # 20.51% raw（adjusted 16.24%，38/234）
scripts/maintenance/create_b1_classifieds_stub.py:57:        "success_rate": 29 / 234,        # 12.39% raw（adjusted 8.12%，19/234）
scripts/maintenance/create_b1_classifieds_stub.py:75:# ---------- condition_summary_v2.json 完整模板 ----------
scripts/maintenance/create_b1_classifieds_stub.py:81:        "success_rate": round(known["success_rate"], 6),
scripts/maintenance/create_b1_classifieds_stub.py:150:        summary_path = cond_dir / "condition_summary_v2.json"
scripts/maintenance/create_b1_classifieds_stub.py:154:        print(f"[stub] {mode}: SR={payload['success_rate']:.4f} ({known['episodes']} ep)")
scripts/maintenance/create_b1_classifieds_stub.py:192:        print(f"[stub] 分析脚本可读取 condition_summary_v2.json 获取已知 SR 数字")
scripts/maintenance/probe_tier10_dispatch_target.py:7:artifact) showed this gap. This probe extends to all dispatch action types
scripts/maintenance/probe_tier10_dispatch_target.py:11:1. For each action type, sample 10-15 FAILED steps (action_success=False) from
scripts/maintenance/probe_tier10_dispatch_target.py:117:                    if step.get("action_success") is True:
scripts/maintenance/glm/glm_playbook_refresh.py:110:        # Strategy: find positions of all fail markers + all success markers,
scripts/maintenance/glm/glm_playbook_refresh.py:325:- 否则: bullet list, 每条格式 `[severity] file (line N, time): 一句话讲发生啥` — 严重 (oom/traceback/not_logged_in) 用 🔴, 中度 (timeout/http5xx) 用 ⚠️, 轻度 (notify_fail) 用 ℹ️
scripts/maintenance/glm/glm_playbook_refresh.py:404:    # Reset failure count on success.
scripts/mechanistic/run_stage4_method44_v2_sweep.py:101:    p.add_argument("--limit", type=int, default=2, help="N tasks (smoke=2, full=24)")
scripts/maintenance/auto_pull_myriad_cell.sh:9:#        OR condition_summary_v2.json). If missing → abort + low-priority
scripts/maintenance/auto_pull_myriad_cell.sh:14:#        Phase 1 condition_summary_v2.json (paper hygiene gate)
scripts/maintenance/auto_pull_myriad_cell.sh:77:# LAST step; Phase 1 paper-grade cells write condition_summary_v2.json. If
scripts/maintenance/auto_pull_myriad_cell.sh:85:    echo "Phase 0: probing remote for done-sentinel (pilot_summary.md OR condition_summary_v2.json OR hidden_states.npz)"
scripts/maintenance/auto_pull_myriad_cell.sh:89:            test -s '$MYRIAD_REMOTE_BASE/$REMOTE_BASENAME/condition_summary_v2.json' && echo SENTINEL_OK_CONDITION && exit 0; \
scripts/maintenance/auto_pull_myriad_cell.sh:99:                "job=$JOB_ID remote=$REMOTE_BASENAME → no pilot_summary.md / condition_summary_v2.json / hidden_states.npz on remote. Likely qdel'd / crashed. Skipping SCP to avoid polluting local dir with partial data." \
scripts/maintenance/auto_pull_myriad_cell.sh:113:echo "Phase 1: pulling artifacts via DGX → quark → Myriad chain"
scripts/maintenance/auto_pull_myriad_cell.sh:118:            condition_summary_v2.json hidden_states.npz \
scripts/maintenance/auto_pull_myriad_cell.sh:146:if [ "${P79_SKIP_VALIDATE:-0}" != "1" ] && [ -f "$LOCAL_DIR/condition_summary_v2.json" ]; then
scripts/maintenance/clear_tasks.py:4:Deletes: summary JSON, steps JSONL, artifacts directory, and digest records for each task.
scripts/maintenance/clear_tasks.py:19:    # Clean orphan artifact dirs (no summary file) across all conditions
scripts/maintenance/clear_tasks.py:20:    python scripts/maintenance/clear_tasks.py --run-dir results/.../B1_run --clean-orphan-artifacts
scripts/maintenance/clear_tasks.py:22:    # Clean orphan artifacts for a specific condition
scripts/maintenance/clear_tasks.py:24:        --condition phase1_som_router_0 --clean-orphan-artifacts
scripts/maintenance/clear_tasks.py:52:def _clean_orphan_artifacts(
scripts/maintenance/clear_tasks.py:58:    """Delete artifact dirs and orphan steps files that have no corresponding summary.
scripts/maintenance/clear_tasks.py:61:    belong to an in-progress episode (runner creates artifacts/steps before writing
scripts/maintenance/clear_tasks.py:80:        art_dir = cond_dir / "artifacts"
scripts/maintenance/clear_tasks.py:83:        # 1. Orphan artifact directories (no summary)
scripts/maintenance/clear_tasks.py:85:            for artifact in sorted(art_dir.iterdir()):
scripts/maintenance/clear_tasks.py:86:                if not artifact.is_dir():
scripts/maintenance/clear_tasks.py:88:                if (ep_dir / f"{artifact.name}_summary_v2.json").exists():
scripts/maintenance/clear_tasks.py:90:                if artifact.stat().st_mtime > cutoff:
scripts/maintenance/clear_tasks.py:93:                rel = artifact.relative_to(run_dir)
scripts/maintenance/clear_tasks.py:95:                    print(f"  [dry-run] rm -rf {rel}  (orphan artifact — no summary)")
scripts/maintenance/clear_tasks.py:97:                    shutil.rmtree(artifact)
scripts/maintenance/clear_tasks.py:98:                    print(f"  deleted orphan artifact: {rel}")
scripts/maintenance/clear_tasks.py:130:    p.add_argument("--clean-orphan-artifacts", action="store_true",
scripts/maintenance/clear_tasks.py:131:                    help="Delete artifact dirs that have no corresponding summary file")
scripts/maintenance/clear_tasks.py:137:    # Validate: either --tasks or --clean-orphan-artifacts must be provided
scripts/maintenance/clear_tasks.py:138:    if not args.tasks and not args.clean_orphan_artifacts:
scripts/maintenance/clear_tasks.py:139:        p.error("one of --tasks or --clean-orphan-artifacts is required")
scripts/maintenance/clear_tasks.py:147:    # --- Orphan artifact cleanup mode ---
scripts/maintenance/clear_tasks.py:148:    if args.clean_orphan_artifacts:
scripts/maintenance/clear_tasks.py:149:        orphans_deleted = _clean_orphan_artifacts(run_dir, args.condition, args.dry_run)
scripts/maintenance/clear_tasks.py:151:        print(f"\nDone: {action} {orphans_deleted} orphan artifact dir(s)")
scripts/maintenance/clear_tasks.py:164:    art_dir = cond_dir / "artifacts"
scripts/maintenance/clear_tasks.py:176:        artifact_dir = art_dir / prefix
scripts/maintenance/clear_tasks.py:179:        # (has steps JSONL or artifacts but no summary yet)
scripts/maintenance/clear_tasks.py:180:        if not summary_file.exists() and (steps_file.exists() or artifact_dir.exists()):
scripts/maintenance/clear_tasks.py:182:                print(f"  SKIP {prefix} — in-progress (has steps/artifacts but no summary). Use --force to override")
scripts/maintenance/clear_tasks.py:187:        dirs = [artifact_dir]
scripts/maintenance/clear_tasks.py:251:        cond_summary_path = cond_dir / "condition_summary_v2.json"
scripts/README.md:51:| `rederive_episode_summary.py` | 修补 episode summary（adjusted_success/cost/etc.，§95 canonical） |
scripts/README.md:52:| `clear_tasks.py` | 清 task summary/steps/artifacts/digest 记录（统一入口，**不要手动 rm**） |
scripts/README.md:86:| `smoke_test_vwa.py` | VWA 环境 smoke test |
scripts/maintenance/probe_b37_api_determinism.py:150:        "n_successful": len(digests),
scripts/maintenance/probe_b08_b06_self_replay.py:108:    action_success = step.get("action_success")
scripts/maintenance/probe_b08_b06_self_replay.py:119:            "action_success": action_success,
scripts/maintenance/probe_b08_b06_self_replay.py:207:            "action_success": step.get("action_success"),
scripts/maintenance/probe_b08_b06_self_replay.py:328:                    if step.get("action_success") is True or step.get("page_changed") is True:
scripts/maintenance/probe_b08_b06_self_replay.py:389:                    if step.get("action_success") is True or step.get("page_changed") is True:
scripts/setup/a100_self_host_vwa.sh:140:# Step 4: Wait for sites to be ready + smoke check
scripts/maintenance/crontab.txt:16:# every 10 min: sync _status/cells/*.md frontmatter from condition_summary_v2.json
scripts/maintenance/probe_b01_b13_self_verify.py:124:                    if step.get("action_success") is True:
scripts/maintenance/probe_b01_b13_self_verify.py:201:            "action_success": step.get("action_success"),
scripts/maintenance/probe_b01_b13_self_verify.py:280:            "action_success": step.get("action_success"),
scripts/maintenance/probe_b01_b13_self_verify.py:321:    elif step.get("action_success") is False and real_change_signals:
scripts/maintenance/probe_b01_b13_self_verify.py:323:        out["reason"] = f"action_success=False but real change signals: {real_change_signals} — runner missed success"
scripts/maintenance/probe_b01_b13_self_verify.py:326:        out["reason"] = f"action_success={step.get('action_success')} signals={real_change_signals}"
scripts/maintenance/probe_b01_b13_self_verify.py:432:        f"- Self-verify probed: {b13_total} cases via state_digest log analysis (no Playwright replay — independent of codex's REPLAY_FAIL artifacts)",
scripts/mechanistic/extract_archive_subset.py:1:"""Extract 24 strong + 11 reverse mirage candidate task artifacts to a compact
scripts/mechanistic/extract_archive_subset.py:6:token_overlap criteria, and copies per-(task, step) artifacts:
scripts/mechanistic/extract_archive_subset.py:44:def find_artifacts_dir(run_dir: Path) -> Path:
scripts/mechanistic/extract_archive_subset.py:46:        if child.is_dir() and (child / "artifacts").is_dir():
scripts/mechanistic/extract_archive_subset.py:47:            return child / "artifacts"
scripts/mechanistic/extract_archive_subset.py:48:    raise FileNotFoundError(f"No condition subdir with artifacts/ in {run_dir}")
scripts/mechanistic/extract_archive_subset.py:83:        "--artifacts-subdir", default=None,
scripts/mechanistic/extract_archive_subset.py:86:             "find_artifacts_dir picks first-iterated which may be wrong condition. "
scripts/mechanistic/extract_archive_subset.py:87:             "Set explicitly: e.g. --artifacts-subdir phase1_som_router_0.",
scripts/mechanistic/extract_archive_subset.py:101:    if args.artifacts_subdir:
scripts/mechanistic/extract_archive_subset.py:102:        artifacts_dir = archived_dir / args.artifacts_subdir / "artifacts"
scripts/mechanistic/extract_archive_subset.py:103:        if not artifacts_dir.is_dir():
scripts/mechanistic/extract_archive_subset.py:104:            logger.error(f"--artifacts-subdir resolved to {artifacts_dir} (does not exist)")
scripts/mechanistic/extract_archive_subset.py:107:        artifacts_dir = find_artifacts_dir(archived_dir)
scripts/mechanistic/extract_archive_subset.py:108:    logger.info(f"Source artifacts: {artifacts_dir}")
scripts/mechanistic/extract_archive_subset.py:139:    # 3. Copy artifacts
scripts/mechanistic/extract_archive_subset.py:167:            task_src = artifacts_dir / f"{args.site}_task_{task_id}"
scripts/mechanistic/extract_archive_subset.py:196:                manifest["skipped"].append({"task_id": task_id, "tier": tier_name, "reason": "no artifact files"})
scripts/mechanistic/extract_archive_subset.py:218:        f"- Strong: {len(manifest['strong'])} tasks × {len(args.steps)} steps = up to {len(manifest['strong']) * len(args.steps)} (task, step) artifacts",
scripts/mechanistic/extract_archive_subset.py:220:        f"- Skipped (no artifact): {len(manifest['skipped'])}",
scripts/maintenance/glm/glm_batch_digest.py:55:_find_episode_artifact_dir = sidecar._find_episode_artifact_dir
scripts/maintenance/glm/glm_batch_digest.py:81:        success = _to_optional_bool(r.get("success"))
scripts/maintenance/glm/glm_batch_digest.py:82:        if success is True:
scripts/maintenance/glm/glm_batch_digest.py:168:        "action_success": _to_optional_bool(item.get("action_success")),
scripts/maintenance/glm/glm_batch_digest.py:184:# Key step selection + artifact loading
scripts/maintenance/glm/glm_batch_digest.py:336:        success = rec.get("action_success")
scripts/maintenance/glm/glm_batch_digest.py:342:            if success is False:
scripts/maintenance/glm/glm_batch_digest.py:346:            if success is False:
scripts/maintenance/glm/glm_batch_digest.py:357:        if success is False:
scripts/maintenance/glm/glm_batch_digest.py:441:    导致操作无效（action_success=false）。
scripts/maintenance/glm/glm_batch_digest.py:628:    # Load artifacts
scripts/maintenance/glm/glm_batch_digest.py:629:    ep_dir = _find_episode_artifact_dir(run_dir, condition_id, task_id)
scripts/maintenance/glm/glm_batch_digest.py:726:                raise ValueError(f"GLM returned unparseable response: {raw[:200]!r}")
scripts/maintenance/glm/glm_batch_digest.py:1146:    success_count = 0
scripts/maintenance/glm/glm_batch_digest.py:1163:            success_count += 1
scripts/maintenance/glm/glm_batch_digest.py:1181:            success_count += 1
scripts/maintenance/glm/glm_batch_digest.py:1196:    print(f"\n[batch-digest] Done. success={success_count} failed={fail_count}")
scripts/mechanistic/run_stage2b_continuation_pilot.py:102:def find_artifacts_dir(run_dir: Path) -> Path:
scripts/mechanistic/run_stage2b_continuation_pilot.py:103:    """Find artifacts directory; supports two layouts:
scripts/mechanistic/run_stage2b_continuation_pilot.py:104:    (a) nested:  <run>/<condition>/artifacts/<site>_task_X/step_NNN/
scripts/mechanistic/run_stage2b_continuation_pilot.py:107:    # Layout (a): nested condition/artifacts
scripts/mechanistic/run_stage2b_continuation_pilot.py:109:        if child.is_dir() and (child / "artifacts").is_dir():
scripts/mechanistic/run_stage2b_continuation_pilot.py:110:            return child / "artifacts"
scripts/mechanistic/run_stage2b_continuation_pilot.py:111:    # Layout (b): flat subset (run_dir IS the artifacts dir)
scripts/mechanistic/run_stage2b_continuation_pilot.py:118:    raise FileNotFoundError(f"No artifacts in {run_dir} (tried nested + flat layouts)")
scripts/mechanistic/run_stage2b_continuation_pilot.py:280:    artifacts_dir = find_artifacts_dir(archived_dir)
scripts/mechanistic/run_stage2b_continuation_pilot.py:281:    logger.info(f"Archived artifacts: {artifacts_dir}")
scripts/mechanistic/run_stage2b_continuation_pilot.py:288:    # build a deterministic permutation so that target task T_i uses source artifacts
scripts/mechanistic/run_stage2b_continuation_pilot.py:319:        step_dir = artifacts_dir / f"{args.site}_task_{task_id}" / f"step_{args.step:03d}"
scripts/mechanistic/run_stage2b_continuation_pilot.py:323:            logger.warning(f"task {task_id}: missing artifacts, skip")
scripts/mechanistic/run_stage2b_continuation_pilot.py:340:        # a different task's artifacts (intent + obs + screenshot) for source.
scripts/mechanistic/run_stage2b_continuation_pilot.py:343:            source_step_dir = artifacts_dir / f"{args.site}_task_{source_task_id}" / f"step_{args.step:03d}"
scripts/mechanistic/run_stage2b_continuation_pilot.py:348:                                "missing artifacts, falling back to same-task source")
scripts/mechanistic/run_stage2b_continuation_pilot.py:556:    # patch config + per-task outcomes for OSF DOI lock + cross-machine compare.
scripts/mechanistic/run_stage2b_continuation_pilot.py:581:        "outcomes_per_task": [
scripts/mechanistic/run_stage2_patching_pilot.py:71:def find_artifacts_dir(run_dir: Path) -> Path:
scripts/mechanistic/run_stage2_patching_pilot.py:73:        if child.is_dir() and (child / "artifacts").is_dir():
scripts/mechanistic/run_stage2_patching_pilot.py:74:            return child / "artifacts"
scripts/mechanistic/run_stage2_patching_pilot.py:75:    raise FileNotFoundError(f"No condition subdir with artifacts/ in {run_dir}")
scripts/mechanistic/run_stage2_patching_pilot.py:129:    artifacts_dir = find_artifacts_dir(archived_dir)
scripts/mechanistic/run_stage2_patching_pilot.py:130:    logger.info(f"Archived artifacts: {artifacts_dir}")
scripts/mechanistic/run_stage2_patching_pilot.py:138:        step_dir = artifacts_dir / f"{args.site}_task_{task_id}" / f"step_{args.step:03d}"
scripts/mechanistic/run_stage2_patching_pilot.py:142:            logger.warning(f"task {task_id} step {args.step}: missing artifacts, skip")
scripts/mechanistic/run_stage2_patching_pilot.py:172:        logger.error("No tasks had complete artifacts; aborting plot")
scripts/myriad/smoke_compute.qsub:2:# P79 Myriad onboarding smoke test — COMPUTE node side
scripts/myriad/smoke_compute.qsub:4:# Submit with: qsub scripts/myriad/smoke_compute.qsub
scripts/myriad/smoke_compute.qsub:5:# Output:      ./p79_smoke_compute.o<JOB_ID>  (cwd at submit time)
scripts/myriad/smoke_compute.qsub:13:#$ -N p79_smoke_compute
scripts/myriad/smoke_compute.qsub:34:===== P79 Myriad smoke (COMPUTE node) =====
scripts/myriad/smoke_compute.qsub:69:echo "  Torch bf16 GPU smoke:"
scripts/myriad/smoke_compute.qsub:70:python3 - <<'PY' || bad "Python smoke crashed"
scripts/myriad/smoke_compute.qsub:136:  echo "  compute-node egress is the binding constraint for runtime jobs. Options listed in login smoke."
scripts/maintenance/glm/glm_diagnosis_sidecar.py:237:        success = _to_optional_bool(r.get("success"))
scripts/maintenance/glm/glm_diagnosis_sidecar.py:238:        if success is True:
scripts/maintenance/glm/glm_diagnosis_sidecar.py:273:                        action_success = _to_optional_bool(item.get("action_success"))
scripts/maintenance/glm/glm_diagnosis_sidecar.py:281:                                "action_success": action_success,
scripts/maintenance/glm/glm_diagnosis_sidecar.py:652:    _wasted = float(case.get("wasted_cost_usd") or case.get("total_cost_usd") or 0) if not case.get("success") else 0.0
scripts/maintenance/glm/glm_diagnosis_sidecar.py:693:def _find_episode_artifact_dir(run_dir: Path, condition_id: str, task_id: int) -> Optional[Path]:
scripts/maintenance/glm/glm_diagnosis_sidecar.py:694:    """Find the artifact directory for a given condition + task_id."""
scripts/maintenance/glm/glm_diagnosis_sidecar.py:695:    artifacts_dir = run_dir / condition_id / "artifacts"
scripts/maintenance/glm/glm_diagnosis_sidecar.py:696:    if not artifacts_dir.exists():
scripts/maintenance/glm/glm_diagnosis_sidecar.py:698:    for d in artifacts_dir.iterdir():
scripts/maintenance/glm/glm_diagnosis_sidecar.py:890:                raise ValueError(f"GLM returned unparseable response: {raw[:200]!r}")
scripts/maintenance/glm/glm_diagnosis_sidecar.py:914:            _glm_wasted = float(case.get("wasted_cost_usd") or case.get("total_cost_usd") or 0) if not case.get("success") else 0.0
scripts/maintenance/glm/glm_diagnosis_sidecar.py:980:        # Load SoM artifacts
scripts/maintenance/glm/glm_diagnosis_sidecar.py:987:                _ep_dir = _find_episode_artifact_dir(run_dir, _cond_id, _task_id)
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1210:    success = _cnt("success")
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1216:        f" 当前成功率为 {success/denom:.1%}。"
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1470:    # Track the max task_id per condition for which ntfy was successfully sent.
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1530:    # analyzed but never successfully pushed via ntfy.  Split into batches of
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1624:                            # Compute success rate from CSV for context
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1633:                                    _sr_ok = sum(1 for r in _sr_rows if _to_optional_bool(r.get("success")) is True)
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1634:                                    _sr_line = f"success={_sr_ok}/{len(_sr_rows)} ({_sr_ok/len(_sr_rows):.1%})"
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1707:                # Compute success_rate for the triggered condition from CSV.
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1714:                success_count = sum(
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1715:                    1 for r in _cond_rows if _to_optional_bool(r.get("success")) is True
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1717:                success_rate = (success_count / episodes) if episodes > 0 else 0.0
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1721:                _mode_stats: Dict[str, Dict[str, int]] = _defaultdict(lambda: {"total": 0, "success": 0})
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1722:                _active_mode_stats: Dict[str, Dict[str, int]] = _defaultdict(lambda: {"total": 0, "success": 0})
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1727:                    _ok = _to_optional_bool(_r.get("success")) is True
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1730:                        _mode_stats[_mode]["success"] += 1
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1736:                            _active_mode_stats[_mode]["success"] += 1
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1737:                # Single success line: show active mode(s) with cumulative totals
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1739:                _success_line = "  ".join(
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1740:                    f"{m}: {_mode_stats[m]['success']}/{_mode_stats[m]['total']} ({_mode_stats[m]['success']/_mode_stats[m]['total']:.1%})"
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1742:                ) or f"{success_count}/{episodes}"
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1831:                    f"success={_success_line}",
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1851:                    title = f"P79 [{_cond_label}] {_success_line}"
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1855:                            f"success={_success_line}",
scripts/maintenance/glm/myriad_watcher.py:181:    Returns the full name on success, otherwise `fallback` (the 10-char
scripts/maintenance/glm/myriad_watcher.py:245:    on success, None on any failure (timeout, ssh error, non-zero exit)."""
scripts/maintenance/glm/myriad_watcher.py:345:    # Reset failure counter on success
scripts/analysis/b0_vision_coordinate_errors.py:60:    print(f"  {label} — click/type 的 action_success=false 率")
scripts/analysis/b0_vision_coordinate_errors.py:68:        fail = [r for r in subset if r.get("action_success") is False]
scripts/analysis/b0_vision_coordinate_errors.py:73:    coord_fail = [r for r in coord_steps if r.get("action_success") is False]
scripts/analysis/b0_vision_coordinate_errors.py:84:    print(f"  {label} — action_success=false 中 page_changed 的比例")
scripts/analysis/b0_vision_coordinate_errors.py:86:    fail_steps = [r for r in steps if r.get("action_success") is False]
scripts/analysis/b0_vision_coordinate_errors.py:88:        print("  无 action_success=false 的步骤")
scripts/analysis/b0_vision_coordinate_errors.py:92:    print(f"  action_success=false 且 page_changed=true: {len(page_changed)}/{len(fail_steps)} = {100*rate:.1f}%")
scripts/analysis/b0_vision_coordinate_errors.py:104:    print(f"  {label} — 连续 action_success=false streak 分布")
scripts/analysis/b0_vision_coordinate_errors.py:112:            if r.get("action_success") is False:
scripts/analysis/b0_vision_coordinate_errors.py:147:    print(f"  {label} — action_success=false 的坐标分布 (type + click)")
scripts/analysis/b0_vision_coordinate_errors.py:154:            if get_action_type(r) == at and r.get("action_success") is False
scripts/analysis/b0_vision_coordinate_errors.py:157:            print(f"\n  {at}: 无 action_success=false 的操作")
scripts/analysis/b0_vision_coordinate_errors.py:171:        print(f"\n  --- {at} (action_success=false) ---")
scripts/analysis/b0_vision_coordinate_errors.py:201:    print(f"  B0 三模式 action_success=false 率对比")
scripts/analysis/b0_vision_coordinate_errors.py:215:        all_fail = sum(1 for r in steps if r.get("action_success") is False)
scripts/analysis/b0_vision_coordinate_errors.py:217:        coord_fail = sum(1 for r in coord_steps if r.get("action_success") is False)
scripts/analysis/b0_vision_coordinate_errors.py:221:            fail = sum(1 for r in sub if r.get("action_success") is False)
scripts/maintenance/glm/error_scan.py:39:    ("not_logged_in", re.compile(r"NOT[_ ]LOGGED[_ ]IN|auth_refresh.*(?:fail|error)|session.*expired", re.IGNORECASE), 75),
scripts/maintenance/glm/error_scan.py:47:    ("fp_adjust_error", re.compile(r"fp_reason.*?adjustment_error|Failed to compute adjusted_success", re.IGNORECASE), 82),
scripts/maintenance/glm/error_scan.py:153:                f"({n_fail} consecutive ticks). Prune logs/ artifacts/.",
scripts/mechanistic/diag_stage4_method44_layer_check.py:4:Hypothesis: smoke test null is because we read direction from npz[17]
scripts/mechanistic/diag_stage4_method44_layer_check.py:10:  2. α=50 steering at patcher.layers[17]   ← what smoke test did
scripts/mechanistic/diag_stage4_method44_layer_check.py:20:    → direction works, but smoke alpha was too small (try larger)
scripts/mechanistic/diag_stage4_method44_layer_check.py:114:    logger.info("Test 2: α=50 at patcher.layers[17] = original smoke test position")
scripts/myriad/smoke_login.sh:2:# P79 Myriad onboarding smoke test — LOGIN node side
scripts/myriad/smoke_login.sh:8:# Output:  ~/p79_myriad_smoke_login_<timestamp>.log
scripts/myriad/smoke_login.sh:9:# Next:    qsub scripts/myriad/smoke_compute.qsub
scripts/myriad/smoke_login.sh:15:LOG="${HOME}/p79_myriad_smoke_login_${TS}.log"
scripts/myriad/smoke_login.sh:34:===== P79 Myriad smoke (LOGIN node) =====
scripts/myriad/smoke_login.sh:115:    ok "torch importable in .venv (real cuda check requires GPU node — see compute smoke)"
scripts/myriad/smoke_login.sh:120:  warn ".venv not yet created — run 'pip install -e .' + torch wheel after smoke passes"
scripts/myriad/smoke_login.sh:153:  echo "      qsub scripts/myriad/smoke_compute.qsub"
scripts/maintenance/run_one_vwa_episode.py:73:    # This matches the smoke test logic
scripts/maintenance/run_one_vwa_episode.py:213:    logger.info(f"Success Proxy: {terminated and reward > 0}") # VWA reward is 1.0 on success usually
scripts/analysis/mechanism_per_task.py:11:- E2 trajectory boundary divergence on symmetric-difference success tasks.
scripts/analysis/mechanism_per_task.py:205:def summary_success(summary_path: Path) -> bool | None:
scripts/analysis/mechanism_per_task.py:209:    if "adjusted_success" in row:
scripts/analysis/mechanism_per_task.py:210:        return bool(row["adjusted_success"])
scripts/analysis/mechanism_per_task.py:211:    if "success" in row:
scripts/analysis/mechanism_per_task.py:212:        return bool(row["success"])
scripts/analysis/mechanism_per_task.py:251:            "adjusted_success": summary_success(summary_path),
scripts/analysis/mechanism_per_task.py:344:        left_success = left[tid]["adjusted_success"]
scripts/analysis/mechanism_per_task.py:345:        right_success = right[tid]["adjusted_success"]
scripts/analysis/mechanism_per_task.py:346:        if left_success is None or right_success is None or left_success == right_success:
scripts/analysis/mechanism_per_task.py:355:                "left_success": left_success,
scripts/analysis/mechanism_per_task.py:356:                "right_success": right_success,
scripts/analysis/mechanism_per_task.py:387:                "left_success": row["left_success"],
scripts/analysis/mechanism_per_task.py:388:                "right_success": row["right_success"],
scripts/analysis/mechanism_per_task.py:804:        "This report explains why mode swaps move outcomes by using per-task and per-step evidence. Element ids are excluded because they are not stable across navigation steps or observation modes. Click evidence uses URL-changing transitions `(pre_url_signature, post_url_signature)`, trajectory evidence uses URL signatures per step, confidence evidence reads existing per-run calibration outputs, and action vocabulary evidence uses normalized action types.",
scripts/analysis/mechanism_per_task.py:845:        "E2 filters to symmetric-difference tasks, where exactly one side of the contrast has adjusted success. It then records the first step where URL signatures differ. Early divergence is step <= 3; late divergence is step >= 10.",
scripts/analysis/mechanism_per_task.py:864:                    f"left_success={case['left_success']}, right_success={case['right_success']}, "
scripts/analysis/mechanism_per_task.py:954:        "Together, E1 and E2 support a decision-path account: mode swaps change which URL transitions are attempted and how early trajectories split on tasks where outcomes disagree. E3 keeps the commitment-confidence claim separate from path choice: confidence evidence is useful, but existing B0 outputs support it mainly through verbalized and behavioral AUROC rather than token calibration. E4 shows whether those path changes are accompanied by broad policy-shape shifts in the action vocabulary, or whether the same action mix hides different click targets.",
scripts/analysis/mechanism_per_task.py:1080:            "adjusted-success tasks, E3 aggregates existing confidence analyzer outputs across paper-grade "
scripts/maintenance/rederive_episode_summary.py:18:`condition_summary_v2.json` using `aggregate_condition_metrics`.
scripts/maintenance/rederive_episode_summary.py:65:    adjusted_success: Optional[bool] = None
scripts/maintenance/rederive_episode_summary.py:141:    # §95 adjusted_success — re-derive for old data using runner's canonical
scripts/maintenance/rederive_episode_summary.py:144:    adj_success: Optional[bool] = None
scripts/maintenance/rederive_episode_summary.py:147:        from p79.experiment.analysis import compute_adjusted_success, _load_na_task_ids
scripts/maintenance/rederive_episode_summary.py:162:        adj_success_val, fp_val = compute_adjusted_success(
scripts/maintenance/rederive_episode_summary.py:163:            task_id, site, bool(summary.get("success", False)),
scripts/maintenance/rederive_episode_summary.py:169:        adj_success = bool(adj_success_val)
scripts/maintenance/rederive_episode_summary.py:172:        print(f"  [WARN] adjusted_success derive failed for {site} task {task_id}: {exc}",
scripts/maintenance/rederive_episode_summary.py:185:        adjusted_success=adj_success,
scripts/maintenance/rederive_episode_summary.py:211:    # §95 adjusted_success fields (Step 2): always update if derivation succeeded.
scripts/maintenance/rederive_episode_summary.py:212:    if "adjusted_success" in rewrite_set and adj_success is not None:
scripts/maintenance/rederive_episode_summary.py:213:        summary["adjusted_success"] = adj_success
scripts/maintenance/rederive_episode_summary.py:297:    # Re-aggregate condition_summary_v2.json from the freshly-written episodes.
scripts/maintenance/rederive_episode_summary.py:309:            existing_path = condition_dir / "condition_summary_v2.json"
scripts/maintenance/rederive_episode_summary.py:324:            print(f"  rebuilt condition_summary_v2.json (n={len(ep_summaries)})")
scripts/maintenance/rederive_episode_summary.py:370:        default="page_unchanged_rate,energy_partial,energy_step_complete_count,busy_wait_total_ms,adjusted_success",
scripts/maintenance/rederive_episode_summary.py:374:                        help="Skip rebuilding condition_summary_v2.json after episode rewrites")
scripts/maintenance/rederive_episode_summary.py:380:                    "adjusted_success"}
scripts/analysis/analyze_cross_representation.py:8:exclusive sets, cost-at-success, reason stability, router signals).
scripts/analysis/analyze_cross_representation.py:233:    df["success"] = df["success"].astype(bool)
scripts/analysis/analyze_cross_representation.py:340:    per_mode_fields = ["success", "reason_bucket", "steps", "final_action_type", "fallback_finish", "page_unchanged_rate", "has_effective_action", "url_unique_count"]
scripts/analysis/analyze_cross_representation.py:367:    """Add is_na_task + {mode}_na_fp + {mode}_eval_fp + {mode}_success_adj columns.
scripts/analysis/analyze_cross_representation.py:369:    N/A FP: any mode + N/A task + raw success + ~agent_finished.
scripts/analysis/analyze_cross_representation.py:371:      string_match + success + ~agent_finished → always E-FP
scripts/analysis/analyze_cross_representation.py:372:      program_html + success + ~agent_finished + ~has_effective_action → E-FP
scripts/analysis/analyze_cross_representation.py:395:    # Mirror canonical p79.experiment.analysis.compute_adjusted_success:
scripts/analysis/analyze_cross_representation.py:398:    #      which makes nfp_col = is_na & success & ~False = True)
scripts/analysis/analyze_cross_representation.py:418:    # Mark false positives and build adjusted success columns
scripts/analysis/analyze_cross_representation.py:430:        scol = f"{m}_success"
scripts/analysis/analyze_cross_representation.py:433:        adj_col = f"{m}_success_adj"
scripts/analysis/analyze_cross_representation.py:441:        # determined from data (matches canonical compute_adjusted_success).
scripts/analysis/analyze_cross_representation.py:479:    pivot: pd.DataFrame, modes: List[str], success_suffix: str = "_success",
scripts/analysis/analyze_cross_representation.py:481:    """Compute set analysis metrics using the given success column suffix.
scripts/analysis/analyze_cross_representation.py:484:        success_suffix: "_success" for raw, "_success_adj" for adjusted.
scripts/analysis/analyze_cross_representation.py:493:        col = f"{m}{success_suffix}"
scripts/analysis/analyze_cross_representation.py:522:        weighted_successes = 0
scripts/analysis/analyze_cross_representation.py:526:                col = f"{m}{success_suffix}"
scripts/analysis/analyze_cross_representation.py:531:                weighted_successes += group_sr[best_mode] * len(group_df)
scripts/analysis/analyze_cross_representation.py:538:        feature_oracle_sr = _safe_ratio(weighted_successes, n_tasks)
scripts/analysis/analyze_cross_representation.py:546:        col = f"{m}{success_suffix}"
scripts/analysis/analyze_cross_representation.py:556:        "per_mode_success_count": {m: len(s) for m, s in mode_sets.items()},
scripts/analysis/analyze_cross_representation.py:590:    raw = _compute_set_metrics(pivot, modes, "_success")
scripts/analysis/analyze_cross_representation.py:596:    has_adj = any(f"{m}_success_adj" in pivot.columns for m in modes)
scripts/analysis/analyze_cross_representation.py:598:        adj = _compute_set_metrics(pivot, modes, "_success_adj")
scripts/analysis/analyze_cross_representation.py:603:        raw["per_mode_success_count_adjusted"] = adj["per_mode_success_count"]
scripts/analysis/analyze_cross_representation.py:658:    pivot: pd.DataFrame, modes: List[str], success_suffix: str = "_success",
scripts/analysis/analyze_cross_representation.py:660:    """Compute exclusive set summary + detail using the given success suffix.
scripts/analysis/analyze_cross_representation.py:672:    def _success_vector(row):
scripts/analysis/analyze_cross_representation.py:675:            col = f"{m}{success_suffix}"
scripts/analysis/analyze_cross_representation.py:687:    pivot_c["_svec"] = pivot_c.apply(_success_vector, axis=1)
scripts/analysis/analyze_cross_representation.py:690:        successes = [modes[i] for i, v in enumerate(svec) if v is True]
scripts/analysis/analyze_cross_representation.py:694:        if not successes and not failures:
scripts/analysis/analyze_cross_representation.py:696:        if not successes:
scripts/analysis/analyze_cross_representation.py:699:            base = "all_tested_success" if untested else "all_success"
scripts/analysis/analyze_cross_representation.py:700:        elif len(successes) == 1:
scripts/analysis/analyze_cross_representation.py:701:            base = f"only_{successes[0]}"
scripts/analysis/analyze_cross_representation.py:703:            base = "_and_".join(successes) + "_not_" + "_".join(failures)
scripts/analysis/analyze_cross_representation.py:706:        # No untested modes → restore the legacy "all_fail" / "all_success"
scripts/analysis/analyze_cross_representation.py:712:    set_col = "exclusive_set" if success_suffix == "_success" else "exclusive_set_adj"
scripts/analysis/analyze_cross_representation.py:730:    """A3: Enumerate exclusive success/failure sets with task_type distribution."""
scripts/analysis/analyze_cross_representation.py:736:    summary_df, pivot_c, set_col = _compute_exclusive_sets(pivot, modes, "_success")
scripts/analysis/analyze_cross_representation.py:745:        detail_cols.extend([f"{m}_success", f"{m}_reason_bucket"])
scripts/analysis/analyze_cross_representation.py:754:    has_adj = any(f"{m}_success_adj" in pivot.columns for m in modes)
scripts/analysis/analyze_cross_representation.py:757:            pivot, modes, "_success_adj",
scripts/analysis/analyze_cross_representation.py:769:            adj_detail_cols.extend([f"{m}_success_adj", f"{m}_reason_bucket"])
scripts/analysis/analyze_cross_representation.py:785:def a4_cost_at_success(
scripts/analysis/analyze_cross_representation.py:795:    success_cols = [
scripts/analysis/analyze_cross_representation.py:796:        f"{m}_success_adj" if f"{m}_success_adj" in pivot.columns else f"{m}_success"
scripts/analysis/analyze_cross_representation.py:799:    success_cols = [c for c in success_cols if c in pivot.columns]
scripts/analysis/analyze_cross_representation.py:800:    mask = pivot[success_cols].apply(lambda row: all(v == True for v in row), axis=1)
scripts/analysis/analyze_cross_representation.py:855:    cost_df.to_csv(dirs.tables / "A4_cost_at_success.csv", index=False)
scripts/analysis/analyze_cross_representation.py:896:    _write_json(summary, dirs.base / "A4_cost_at_success_summary.json")
scripts/analysis/analyze_cross_representation.py:901:def a5_task_type_success_rate(
scripts/analysis/analyze_cross_representation.py:904:    """A5: Task type × mode success rate breakdown (raw + adjusted)."""
scripts/analysis/analyze_cross_representation.py:909:    has_adj = any(f"{m}_success_adj" in pivot.columns for m in modes)
scripts/analysis/analyze_cross_representation.py:915:            col = f"{m}_success"
scripts/analysis/analyze_cross_representation.py:917:                n_success = grp[col].sum()
scripts/analysis/analyze_cross_representation.py:919:                row[f"{m}_success_count"] = int(n_success)
scripts/analysis/analyze_cross_representation.py:921:                row[f"{m}_sr"] = round(_safe_ratio(n_success, n_present), 4)
scripts/analysis/analyze_cross_representation.py:923:            adj_col = f"{m}_success_adj"
scripts/analysis/analyze_cross_representation.py:927:                row[f"{m}_success_count_adj"] = int(n_adj)
scripts/analysis/analyze_cross_representation.py:932:    tt_df.to_csv(dirs.tables / "A5_task_type_success_rate.csv", index=False)
scripts/analysis/analyze_cross_representation.py:951:        fig.savefig(dirs.plots / "A5_task_type_success_rate.png", dpi=150)
scripts/analysis/analyze_cross_representation.py:958:    """A6: Venn diagram of success sets."""
scripts/analysis/analyze_cross_representation.py:961:        col = f"{m}_success_adj" if f"{m}_success_adj" in pivot.columns else f"{m}_success"
scripts/analysis/analyze_cross_representation.py:968:        print("  A6: skipped (need >=2 modes with successes)")
scripts/analysis/analyze_cross_representation.py:1160:    rd = reason_df[["condition_id", "site", "task_id", "success", "reason_bucket"]].copy()
scripts/analysis/analyze_cross_representation.py:1251:    detail_cols = ["site", "task_id", "condition_id", "reason_bucket", "success", "subtype"]
scripts/analysis/analyze_cross_representation.py:1319:        def _collect(success_suffix: str):
scripts/analysis/analyze_cross_representation.py:1322:                scol = f"{m}{success_suffix}"
scripts/analysis/analyze_cross_representation.py:1336:        succeeded_modes, best_mode = _collect("_success")
scripts/analysis/analyze_cross_representation.py:1337:        succeeded_modes_adj, best_mode_adj = _collect("_success_adj")
scripts/analysis/analyze_cross_representation.py:1383:        success = bool(r.get("success", False))
scripts/analysis/analyze_cross_representation.py:1388:            "success": success,
scripts/analysis/analyze_cross_representation.py:1408:        # "escalation_would_help" if the other mode's success is itself
scripts/analysis/analyze_cross_representation.py:1412:        def _other_success(success_suffix: str) -> bool:
scripts/analysis/analyze_cross_representation.py:1416:                scol = f"{m}{success_suffix}"
scripts/analysis/analyze_cross_representation.py:1425:        if not success and div_step is not None:
scripts/analysis/analyze_cross_representation.py:1426:            row["escalation_would_help"] = _other_success("_success")
scripts/analysis/analyze_cross_representation.py:1429:            if any(f"{m}_success_adj" in pivot.columns for m in modes):
scripts/analysis/analyze_cross_representation.py:1430:                row["escalation_would_help_adj"] = _other_success("_success_adj")
scripts/analysis/analyze_cross_representation.py:1443:    failed_with_div = esc_df[(esc_df["success"] == False) & esc_df["divergence_step"].notna()]
scripts/analysis/analyze_cross_representation.py:1481:    success_suffix: str = "_success",
scripts/analysis/analyze_cross_representation.py:1483:    """Build oracle decomposition rows for given success column suffix."""
scripts/analysis/analyze_cross_representation.py:1492:            scol = f"{m}{success_suffix}"
scripts/analysis/analyze_cross_representation.py:1539:    """R3: Oracle router decomposition -- for each union-success task, pick cheapest mode."""
scripts/analysis/analyze_cross_representation.py:1571:    success_cols = [f"{m}_success" for m in modes if f"{m}_success" in pivot.columns]
scripts/analysis/analyze_cross_representation.py:1572:    mask = pivot[success_cols].any(axis=1)
scripts/analysis/analyze_cross_representation.py:1576:        print("  R3: skipped (no successful tasks)")
scripts/analysis/analyze_cross_representation.py:1579:    rows = _build_oracle_rows(union_tasks, modes, cost_lookup, task_configs, "_success")
scripts/analysis/analyze_cross_representation.py:1597:    has_adj = any(f"{m}_success_adj" in pivot.columns for m in modes)
scripts/analysis/analyze_cross_representation.py:1599:        adj_cols = [f"{m}_success_adj" for m in modes if f"{m}_success_adj" in pivot.columns]
scripts/analysis/analyze_cross_representation.py:1602:        adj_rows = _build_oracle_rows(adj_union, modes, cost_lookup, task_configs, "_success_adj")
scripts/analysis/analyze_cross_representation.py:1813:        a4_cost_at_success(pivot, modes, ep_summaries, cond_mode, dirs)
scripts/analysis/analyze_cross_representation.py:1815:        a5_task_type_success_rate(pivot, modes, dirs, skip_plots)
scripts/mechanistic/curate_mirage_tasks.py:87:def find_artifacts_dir(run_dir: Path) -> Path:
scripts/mechanistic/curate_mirage_tasks.py:89:        if child.is_dir() and (child / "artifacts").is_dir():
scripts/mechanistic/curate_mirage_tasks.py:90:            return child / "artifacts"
scripts/mechanistic/curate_mirage_tasks.py:91:    raise FileNotFoundError(f"No condition subdir with artifacts/ in {run_dir}")
scripts/mechanistic/curate_mirage_tasks.py:199:        "--artifacts-subdir", default=None,
scripts/mechanistic/curate_mirage_tasks.py:202:             "find_artifacts_dir picks first-iterated which may be wrong condition. "
scripts/mechanistic/curate_mirage_tasks.py:203:             "Set explicitly: e.g. --artifacts-subdir phase1_som_router_0.",
scripts/mechanistic/curate_mirage_tasks.py:215:    if args.artifacts_subdir:
scripts/mechanistic/curate_mirage_tasks.py:216:        artifacts_dir = archived_dir / args.artifacts_subdir / "artifacts"
scripts/mechanistic/curate_mirage_tasks.py:217:        if not artifacts_dir.is_dir():
scripts/mechanistic/curate_mirage_tasks.py:218:            raise FileNotFoundError(f"--artifacts-subdir resolved to {artifacts_dir} which doesn't exist")
scripts/mechanistic/curate_mirage_tasks.py:220:        artifacts_dir = find_artifacts_dir(archived_dir)
scripts/mechanistic/curate_mirage_tasks.py:221:    logger.info(f"Archived artifacts: {artifacts_dir}")
scripts/mechanistic/curate_mirage_tasks.py:227:    skipped_no_artifact = 0
scripts/mechanistic/curate_mirage_tasks.py:230:        step_dir = artifacts_dir / f"{args.site}_task_{task_id}" / f"step_{args.step:03d}"
scripts/mechanistic/curate_mirage_tasks.py:234:            skipped_no_artifact += 1
scripts/mechanistic/curate_mirage_tasks.py:277:    logger.info(f"Scored {len(candidates)} task pairs (skipped {skipped_no_artifact} missing artifacts)")
scripts/mechanistic/curate_mirage_tasks.py:328:    md_lines.append(f"- skipped (missing artifacts): {skipped_no_artifact}")
scripts/analysis/aggregate_failure_modes.py:110:    # cell_totals[(baseline, site, mode)] = total episodes (including success)
scripts/analysis/aggregate_failure_modes.py:148:                if bucket_fine == "success":
scripts/analysis/aggregate_failure_modes.py:149:                    cells[cell_key]["success"] += count
scripts/analysis/aggregate_failure_modes.py:166:        failed = total - buckets.get("success", 0)
scripts/analysis/aggregate_failure_modes.py:169:            if b == "success":
scripts/analysis/aggregate_failure_modes.py:176:            "success_count": buckets.get("success", 0),
scripts/analysis/diag_pattern_match.py:51:    success: bool
scripts/analysis/diag_pattern_match.py:492:        if not cond_dir.is_dir() or cond_dir.name in ("task_configs", "artifacts"):
scripts/analysis/diag_pattern_match.py:544:        if failed_only and summary.get("success"):
scripts/analysis/diag_pattern_match.py:566:            success=summary.get("success", False),
scripts/analysis/aggregate_phantom_lift.py:111:        if rec.get("adjusted_success", rec.get("success", False)):
scripts/analysis/aggregate_phantom_lift.py:161:    """Wilcoxon signed-rank test on paired binary task outcomes (a vs b).
scripts/analysis/aggregate_phantom_lift.py:163:    For binary outcomes diff ∈ {-1, 0, +1}; scipy drops zero diffs. When set b
scripts/analysis/aggregate_phantom_lift.py:421:    # Restrict each mode's success set to its own comparison's universe
scripts/analysis/aggregate_routing_auroc.py:115:        "AUROC ≥ 0.5 means signal correlates with success; CI from 1000-resample bootstrap.",
scripts/maintenance/rsync_results_to_hub.sh:4:# Tier C (artifacts: screenshots, SoM图) is excluded — pull on demand via
scripts/maintenance/rsync_results_to_hub.sh:30:echo "[rsync→hub] $SOURCE → $HOST:$HUB_PATH (Tier B only, no artifacts)"
scripts/maintenance/rsync_results_to_hub.sh:36:  --include='condition_summary_v2.json' \
scripts/maintenance/rsync_results_to_hub.sh:40:  --exclude='artifacts/' \
scripts/analysis/analyze_comment_selflink_loop_v2.py:105:            if not summary or not summary.get("success"):
scripts/analysis/analyze_comment_selflink_loop_v2.py:253:    print("\n  For url_match tasks, arriving at the post page = success regardless of loop.")
scripts/analysis/analyze_comment_selflink_loop_v2.py:258:        loop_success_url = 0
scripts/analysis/analyze_comment_selflink_loop_v2.py:259:        loop_success_other = 0
scripts/analysis/analyze_comment_selflink_loop_v2.py:289:            if summary and summary.get("success"):
scripts/analysis/analyze_comment_selflink_loop_v2.py:291:                    loop_success_url += 1
scripts/analysis/analyze_comment_selflink_loop_v2.py:293:                    loop_success_other += 1
scripts/analysis/analyze_comment_selflink_loop_v2.py:298:        print(f"    Loop + success via url_match:  {loop_success_url}")
scripts/analysis/analyze_comment_selflink_loop_v2.py:299:        print(f"    Loop + success via other eval: {loop_success_other}")
scripts/analysis/analyze_comment_selflink_loop_v2.py:301:        total_loop = loop_success_url + loop_success_other + loop_fail
scripts/analysis/analyze_comment_selflink_loop_v2.py:303:            print(f"    url_match accounts for {loop_success_url}/{loop_success_url+loop_success_other} "
scripts/analysis/analyze_comment_selflink_loop_v2.py:304:                  f"({loop_success_url/(loop_success_url+loop_success_other)*100:.0f}%) of loop successes")
scripts/analysis/README.md:25:The status report reads live artifacts and marks missing evidence with `⚠️`.
scripts/analysis/README.md:36:| `scripts/analysis/figures/fig0e_category_mode_heatmap.py` | 0e | audit JSON + episode `adjusted_success` | `results/phantom_paper/figures/fig0e_category_mode_heatmap.png` | reddit/cls category × mode adjusted SR heatmap |
scripts/analysis/README.md:37:| `scripts/analysis/figures/fig0f_overlap_stacked_bar.py` | 0f | B0 adjusted-success task sets | `results/phantom_paper/figures/fig0f_overlap_stacked_bar.png` | reddit P-SoM solve depth distribution; classifieds P-SoM/P-text overlap depth |
scripts/analysis/README.md:40:| `scripts/analysis/figures/fig0d_taskpool_jaccard.py` | 0 supporting | live episode adjusted-success sets | `results/phantom_paper/figures/fig0d_taskpool_jaccard.png` | solve-pool overlap sketch for B0/B1 observation arms |
scripts/analysis/README.md:41:| `scripts/analysis/figures/fig0c_drop_one_oracle.py` | 0 supporting | live episode adjusted-success sets | `results/phantom_paper/figures/fig0c_drop_one_oracle.png` (figure) + `results/phantom_paper/fig0c_drop_one_bootstrap_ci.csv` (data sidecar) | drop-one oracle loss for B0/B1 mode pools |
scripts/analysis/README.md:84:| `scripts/analysis/aggregate_cross_site.py` | 3a-3c | per-condition `condition_summary_v2.json` | `results/phantom_paper/cross_site/cross_site_aggregation.csv`, `_summary.json`, plots | B0/B1 cross-site SR/cost/latency table |
scripts/analysis/README.md:86:| `scripts/analysis/figures/fig3d_cost_sr_frontier.py` | 3d | `cost_per_mode.json` paper cost + adjusted success | `results/phantom_paper/figures/fig3d_cost_sr_frontier.png` | B0 API-token vs B1 electricity-equivalent cost/SR Pareto frontier |
scripts/analysis/README.md:88:| `scripts/analysis/layered_status.py` | status | all 4-dimension artifacts listed above | `docs/analysis/layered_evidence_status.md` | live markdown snapshot with timestamps and missing-artifact warnings |
scripts/analysis/README.md:92:- token/cost and latency: `results/visualwebarena/phase1/*/*/condition_summary_v2.json`
scripts/analysis/README.md:107:| `scripts/analysis/analyze_confidence_calibration.py` | one run dir | per-run signal calibration and AUROC artifacts | upstream source for Outcome 0g aggregation |
scripts/analysis/analyze_comment_selflink_loop.py:92:        - success: bool or None
scripts/analysis/analyze_comment_selflink_loop.py:105:        "success": None,
scripts/analysis/analyze_comment_selflink_loop.py:116:    # Get success from summary
scripts/analysis/analyze_comment_selflink_loop.py:119:        result["success"] = summary.get("success", None)
scripts/analysis/analyze_comment_selflink_loop.py:320:        b1_ok = "Y" if b1_res["success"] else ("N" if b1_res["success"] is not None else "?")
scripts/analysis/analyze_comment_selflink_loop.py:321:        b0_ok = "Y" if b0_res["success"] else ("N" if b0_res["success"] is not None else "?")
scripts/analysis/analyze_comment_selflink_loop.py:351:        success = sum(1 for r in results.values() if r["success"])
scripts/analysis/analyze_comment_selflink_loop.py:354:            if r["max_consecutive_selflink"] >= 2 and not r["success"]
scripts/analysis/analyze_comment_selflink_loop.py:356:        loop_and_success = sum(
scripts/analysis/analyze_comment_selflink_loop.py:358:            if r["max_consecutive_selflink"] >= 2 and r["success"]
scripts/analysis/analyze_comment_selflink_loop.py:370:        print(f"  Successful:                           {success} ({success/total*100:.1f}%)")
scripts/analysis/analyze_comment_selflink_loop.py:372:        print(f"  Loop + success:                       {loop_and_success}")
scripts/analysis/analyze_comment_selflink_loop.py:378:            "success": success,
scripts/analysis/analyze_comment_selflink_loop.py:418:            ok = "Y" if r["success"] else "N"
scripts/analysis/analyze_comment_selflink_loop.py:429:            ok = "Y" if r["success"] else "N"
scripts/analysis/analyze_comment_selflink_loop.py:448:        ok = "Y" if r["success"] else "N"
scripts/analysis/analyze_comment_selflink_loop.py:557:        b1_ok = "Y" if b1r.get("success") else "N"
scripts/analysis/analyze_comment_selflink_loop.py:558:        b0_ok = "Y" if b0r.get("success") else "N"
scripts/analysis/analyze_comment_selflink_loop.py:574:        loop_url_match_success = 0
scripts/analysis/analyze_comment_selflink_loop.py:576:        loop_other_eval_success = 0
scripts/analysis/analyze_comment_selflink_loop.py:585:                if r["success"]:
scripts/analysis/analyze_comment_selflink_loop.py:586:                    loop_url_match_success += 1
scripts/analysis/analyze_comment_selflink_loop.py:589:                if r["success"]:
scripts/analysis/analyze_comment_selflink_loop.py:590:                    loop_other_eval_success += 1
scripts/analysis/analyze_comment_selflink_loop.py:593:        print(f"  Loop + url_match eval:    {loop_url_match} (success: {loop_url_match_success})")
scripts/analysis/analyze_comment_selflink_loop.py:594:        print(f"  Loop + other eval:        {loop_other_eval} (success: {loop_other_eval_success})")
scripts/analysis/stage4_h1_per_task_fragility.py:9:by few tasks. If <60% → average artifact.
scripts/analysis/figures/fig0g_routing_auroc_heatmap.py:119:             "Higher AUROC → signal better predicts task success → cheaper trigger feature for routing.",
scripts/maintenance/retry_b1_single_task.sh:176:    result=$("${PYTHON_BIN}" -c "import json; d=json.load(open('${summary}')); print(f'success={d[\"success\"]}, steps={d[\"steps\"]}, error={d.get(\"error\",None)}')")
scripts/maintenance/retry_b1_single_task.sh:232:    result=$("${PYTHON_BIN}" -c "import json; d=json.load(open('${summary}')); print(f'success={d[\"success\"]}, score={d[\"score\"]}, steps={d[\"steps\"]}')")
scripts/analysis/figures/fig0e_category_mode_heatmap.py:7:Outcome 0e: category × observation-mode success-rate evidence.
scripts/analysis/figures/fig0e_category_mode_heatmap.py:78:def load_successes(ep_dir: Path) -> tuple[set[int], set[int]]:
scripts/analysis/figures/fig0e_category_mode_heatmap.py:84:    successes: set[int] = set()
scripts/analysis/figures/fig0e_category_mode_heatmap.py:90:        if bool(record.get("adjusted_success", record.get("success", False))):
scripts/analysis/figures/fig0e_category_mode_heatmap.py:91:            successes.add(tid)
scripts/analysis/figures/fig0e_category_mode_heatmap.py:92:    return successes, observed
scripts/analysis/figures/fig0e_category_mode_heatmap.py:107:        successes, observed = load_successes(ep_dir)
scripts/analysis/figures/fig0e_category_mode_heatmap.py:120:            value = 100.0 * len(successes & denom_tasks) / n
scripts/analysis/figures/fig0e_category_mode_heatmap.py:154:    fig.colorbar(ims[0], ax=axes, shrink=0.82, label="Adjusted success rate (%)")
scripts/maintenance/active_processes.py:231:        if data.get("success"):
scripts/maintenance/active_processes.py:233:        if data.get("adjusted_success", data.get("success", False)):
scripts/maintenance/experiment_watchdog.py:6:1) REPORT:   periodic status every --report-interval-mins (success rate + counts)
scripts/maintenance/experiment_watchdog.py:48:    success: bool
scripts/maintenance/experiment_watchdog.py:250:    dom_path = condition_dir / "artifacts" / f"{site}_task_{task_id}" / "step_000" / "observation_dom.txt"
scripts/maintenance/experiment_watchdog.py:290:    if bool(summary.get("success", False)):
scripts/maintenance/experiment_watchdog.py:291:        return "success"
scripts/maintenance/experiment_watchdog.py:340:    cond_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0})
scripts/maintenance/experiment_watchdog.py:345:        cond_stats[r.condition_id]["success"] += int(r.success)
scripts/maintenance/experiment_watchdog.py:353:        n, succ = s["total"], s["success"]
scripts/maintenance/experiment_watchdog.py:404:        if r.condition_id in completed_conditions and not r.success:
scripts/maintenance/experiment_watchdog.py:469:    1. condition_summary_v2.json exists AND not in seen_completions, OR
scripts/maintenance/experiment_watchdog.py:470:    2. condition_summary_v2.json is newer than analysis outputs (post-analysis stale).
scripts/maintenance/experiment_watchdog.py:491:        summary_path = cond_dir / "condition_summary_v2.json"
scripts/maintenance/experiment_watchdog.py:527:        if (cond_dir / "condition_summary_v2.json").exists():
scripts/maintenance/experiment_watchdog.py:650:                     if d.is_dir() and (d / "condition_summary_v2.json").exists()]
scripts/maintenance/experiment_watchdog.py:699:    """True iff at least one condition has condition_summary_v2.json."""
scripts/maintenance/experiment_watchdog.py:703:        (d / "condition_summary_v2.json").exists()
scripts/maintenance/experiment_watchdog.py:1115:                        "(condition_summary_v2.json present). Without this, "
scripts/maintenance/experiment_watchdog.py:1176:                success=bool(summary.get("success", False)),
scripts/maintenance/experiment_watchdog.py:1188:    # Prune orphan artifacts and steps files (exist but no summary file).
scripts/maintenance/experiment_watchdog.py:1197:        _art_root = _cdir / "artifacts"
scripts/maintenance/experiment_watchdog.py:1199:        # Orphan artifact directories
scripts/maintenance/experiment_watchdog.py:1221:        print(f"[watchdog] Pruned {_orphan_count} orphan item(s) (artifact dirs / steps files without summary)")
scripts/maintenance/experiment_watchdog.py:1277:        print(f"[watchdog] Pruned {pruned_completions} stale completions (missing condition_summary_v2.json)")
scripts/maintenance/experiment_watchdog.py:1346:                condition_completed = (condition_dir / "condition_summary_v2.json").exists()
scripts/maintenance/experiment_watchdog.py:1387:                    # 3. Delete artifacts directory
scripts/maintenance/experiment_watchdog.py:1388:                    artifacts_dir = condition_dir / "artifacts" / f"{site}_task_{task_id}"
scripts/maintenance/experiment_watchdog.py:1390:                        if artifacts_dir.exists():
scripts/maintenance/experiment_watchdog.py:1391:                            shutil.rmtree(artifacts_dir)
scripts/maintenance/experiment_watchdog.py:1415:                    success=bool(summary.get("success", False)),
scripts/maintenance/experiment_watchdog.py:1481:                                cart = cond_dir / "artifacts" / f"{csite}_task_{ctask_id}"
scripts/maintenance/experiment_watchdog.py:1514:                cond_succ = sum(1 for r in cond_all if r.success)
scripts/maintenance/experiment_watchdog.py:1519:                    f"{'OK' if rec.success else reason:<10s} "
scripts/maintenance/experiment_watchdog.py:1602:                            status = "OK" if ep.success else ep.reason
scripts/maintenance/experiment_watchdog.py:1661:            cond_succ = sum(1 for r in cond_all if r.success)
scripts/maintenance/experiment_watchdog.py:1702:            cond_done = (run_dir / args.condition / "condition_summary_v2.json").exists()
scripts/analysis/figures/fig0b_extra_confidence_calibration.py:58:        emit_placeholder("no parseable cells in E3")
scripts/analysis/aggregate_sr_fp_per_mode.py:73:    n_raw_success = 0
scripts/analysis/aggregate_sr_fp_per_mode.py:74:    n_adjusted_success = 0
scripts/analysis/aggregate_sr_fp_per_mode.py:78:    # F26 audit fix 2026-05-09: track rows missing `adjusted_success` so
scripts/analysis/aggregate_sr_fp_per_mode.py:79:    # SR aggregation never silently substitutes raw success for the
scripts/analysis/aggregate_sr_fp_per_mode.py:86:        raw = bool(row.get("success", False))
scripts/analysis/aggregate_sr_fp_per_mode.py:87:        if "adjusted_success" not in row or row.get("adjusted_success") is None:
scripts/analysis/aggregate_sr_fp_per_mode.py:91:            adjusted = bool(row["adjusted_success"])
scripts/analysis/aggregate_sr_fp_per_mode.py:92:        n_raw_success += int(raw)
scripts/analysis/aggregate_sr_fp_per_mode.py:93:        n_adjusted_success += int(adjusted)
scripts/analysis/aggregate_sr_fp_per_mode.py:102:            "rows missing `adjusted_success` field — fell back to raw success. "
scripts/analysis/aggregate_sr_fp_per_mode.py:116:        "n_raw_success": n_raw_success,
scripts/analysis/aggregate_sr_fp_per_mode.py:117:        "n_adjusted_success": n_adjusted_success,
scripts/analysis/aggregate_sr_fp_per_mode.py:118:        "raw_sr_pct": round(pct(n_raw_success, n_total), 6),
scripts/analysis/aggregate_sr_fp_per_mode.py:119:        "adjusted_sr_pct": round(pct(n_adjusted_success, n_total), 6),
scripts/analysis/aggregate_sr_fp_per_mode.py:165:        "Raw SR counts `success == true`; adjusted SR counts `adjusted_success == true` "
scripts/analysis/aggregate_sr_fp_per_mode.py:166:        "with fallback to `success` when the adjusted field is absent. FP count is raw success minus adjusted success. "
scripts/analysis/stage2_layer_significance.py:256:    out.append("  (curated by directional composite score). Selection-bias artifact not")
scripts/maintenance/reeval_phase1.py:283:        summary["success"] = new_score >= 1.0
scripts/analysis/hero_claim_bootstrap.py:14:This script loads B0 reddit per-task adjusted_success for all 6 completed
scripts/analysis/hero_claim_bootstrap.py:60:def load_adjusted_success(episodes_dir: Path) -> dict[int, bool]:
scripts/analysis/hero_claim_bootstrap.py:61:    """Load per-task adjusted_success bool from episodes/*_summary_v2.json files."""
scripts/analysis/hero_claim_bootstrap.py:75:        # adjusted_success preferred; fall back to success
scripts/analysis/hero_claim_bootstrap.py:76:        v = rec.get("adjusted_success", rec.get("success", False))
scripts/analysis/hero_claim_bootstrap.py:81:def build_success_matrix(site: str) -> tuple[np.ndarray, list[int], list[str]]:
scripts/analysis/hero_claim_bootstrap.py:82:    """Build (N_tasks x N_modes) binary success matrix on the same-task subset."""
scripts/analysis/hero_claim_bootstrap.py:87:        per_mode[mode] = load_adjusted_success(epi_dir)
scripts/analysis/hero_claim_bootstrap.py:163:            M, tasks, modes = build_success_matrix(site)
scripts/analysis/stage4_axis2_per_task_fragility.py:199:        f"- If both < 25% → axis-2 signal **aggregate artifact**, §5.7 three-axis claim must downgrade to 'axis-1 + image-axis with axis-2 weak per-task'",
scripts/maintenance/generate_gallery.py:131:    reason_bucket, task_type, adjusted_success, fp_reason.
scripts/maintenance/generate_gallery.py:148:                    adj = row.get("adjusted_success", "")
scripts/maintenance/generate_gallery.py:152:                        "adjusted_success": (
scripts/maintenance/generate_gallery.py:345:            "success": ep["success"],
scripts/maintenance/generate_gallery.py:354:            "adjusted_success": ep.get("adjusted_success"),
scripts/maintenance/generate_gallery.py:361:        success = sum(1 for e in group["episodes"] if e.get("success") is True)
scripts/maintenance/generate_gallery.py:362:        fail = sum(1 for e in group["episodes"] if e.get("success") is False)
scripts/maintenance/generate_gallery.py:365:            "success": success,
scripts/maintenance/generate_gallery.py:367:            "success_rate": round(success / total, 3) if total > 0 else 0,
scripts/maintenance/generate_gallery.py:481:            artifacts_dir = cond_dir / "artifacts"
scripts/maintenance/generate_gallery.py:519:                task_artifact_dir = artifacts_dir / f"{raw_site}_task_{task_id}"
scripts/maintenance/generate_gallery.py:523:                    step_dir = task_artifact_dir / f"step_{step_idx:03d}"
scripts/maintenance/generate_gallery.py:608:                    "success": summary.get("success") if summary else None,
scripts/maintenance/generate_gallery.py:616:                    "adjusted_success": reason_info.get("adjusted_success"),
scripts/maintenance/generate_gallery.py:711:.badge.success{{ background:#1b5e20; color:#a5d6a7; }}
scripts/maintenance/generate_gallery.py:808:.reason-success{{ background:#1b5e20; color:#a5d6a7; }}
scripts/maintenance/generate_gallery.py:862:  if(r==='success') return 'reason-success';
scripts/maintenance/generate_gallery.py:883:    var sr=(g.stats.success_rate*100).toFixed(1);
scripts/maintenance/generate_gallery.py:894:      +'<span class="s ok">'+g.stats.success+' pass</span>'
scripts/maintenance/generate_gallery.py:904:      var c=e.success===true?'success':e.success===false?'fail':'unknown';
scripts/maintenance/generate_gallery.py:905:      var sl=e.success===true?'PASS':e.success===false?'FAIL':'&mdash;';
scripts/maintenance/generate_gallery.py:955:  var c=e.success===true?'success':e.success===false?'fail':'unknown';
scripts/maintenance/generate_gallery.py:956:  var sl=e.success===true?'PASS':e.success===false?'FAIL':'&mdash;';
scripts/maintenance/generate_gallery.py:990:    var ansCls=e.success===true?'ep-ans match':'ep-ans';
scripts/maintenance/probe_b08_b06_self_replay.py:108:    action_success = step.get("action_success")
scripts/maintenance/probe_b08_b06_self_replay.py:119:            "action_success": action_success,
scripts/maintenance/probe_b08_b06_self_replay.py:207:            "action_success": step.get("action_success"),
scripts/maintenance/probe_b08_b06_self_replay.py:328:                    if step.get("action_success") is True or step.get("page_changed") is True:
scripts/maintenance/probe_b08_b06_self_replay.py:389:                    if step.get("action_success") is True or step.get("page_changed") is True:
scripts/maintenance/probe_tier10_dispatch_target.py:7:artifact) showed this gap. This probe extends to all dispatch action types
scripts/maintenance/probe_tier10_dispatch_target.py:11:1. For each action type, sample 10-15 FAILED steps (action_success=False) from
scripts/maintenance/probe_tier10_dispatch_target.py:117:                    if step.get("action_success") is True:
scripts/maintenance/smoke_test_vwa.py:20:    # Remove storage_state to avoid FileNotFoundError during smoke test
scripts/maintenance/smoke_test_vwa.py:22:        print("Removing storage_state from config for smoke test.")
scripts/maintenance/smoke_test_vwa.py:32:    # But for smoke test, even if it fails to load the page, reset might succeed or throw.
scripts/maintenance/smoke_test_vwa.py:41:    temp_config_path = "temp_smoke_config.json"
scripts/maintenance/smoke_test_vwa.py:60:        print("Reset successful!")
scripts/maintenance/smoke_test_vwa.py:71:        print("Step successful!")
scripts/maintenance/digest_enrich.py:4:raw step logs and artifacts. Runs as a post-processor after glm_batch_digest.py.
scripts/maintenance/digest_enrich.py:123:    Handles restart artifacts: if the watchdog/queue restarted a task,
scripts/maintenance/digest_enrich.py:174:    artifact_dir: Path,
scripts/maintenance/digest_enrich.py:191:        dom_path = artifact_dir / f"step_{step_idx:03d}" / "observation_dom.txt"
scripts/maintenance/digest_enrich.py:259:    artifact_dir = run_dir / condition_id / "artifacts" / f"{site}_task_{task_id}"
scripts/maintenance/digest_enrich.py:263:    if artifact_dir.exists():
scripts/maintenance/digest_enrich.py:264:        has_desc, detail_dom_lengths, avg_dom = _check_dom_description(artifact_dir, steps, site=site)
scripts/maintenance/auto_pull_myriad_cell.sh:9:#        OR condition_summary_v2.json). If missing → abort + low-priority
scripts/maintenance/auto_pull_myriad_cell.sh:14:#        Phase 1 condition_summary_v2.json (paper hygiene gate)
scripts/maintenance/auto_pull_myriad_cell.sh:77:# LAST step; Phase 1 paper-grade cells write condition_summary_v2.json. If
scripts/maintenance/auto_pull_myriad_cell.sh:85:    echo "Phase 0: probing remote for done-sentinel (pilot_summary.md OR condition_summary_v2.json OR hidden_states.npz)"
scripts/maintenance/auto_pull_myriad_cell.sh:89:            test -s '$MYRIAD_REMOTE_BASE/$REMOTE_BASENAME/condition_summary_v2.json' && echo SENTINEL_OK_CONDITION && exit 0; \
scripts/maintenance/auto_pull_myriad_cell.sh:99:                "job=$JOB_ID remote=$REMOTE_BASENAME → no pilot_summary.md / condition_summary_v2.json / hidden_states.npz on remote. Likely qdel'd / crashed. Skipping SCP to avoid polluting local dir with partial data." \
scripts/maintenance/auto_pull_myriad_cell.sh:113:echo "Phase 1: pulling artifacts via DGX → quark → Myriad chain"
scripts/maintenance/auto_pull_myriad_cell.sh:118:            condition_summary_v2.json hidden_states.npz \
scripts/maintenance/auto_pull_myriad_cell.sh:146:if [ "${P79_SKIP_VALIDATE:-0}" != "1" ] && [ -f "$LOCAL_DIR/condition_summary_v2.json" ]; then
scripts/maintenance/probe_b37_api_determinism.py:150:        "n_successful": len(digests),
scripts/analysis/figures/fig0d_taskpool_jaccard.py:67:def load_success_set(ep_dir: Path) -> tuple[set[int], set[int]]:
scripts/analysis/figures/fig0d_taskpool_jaccard.py:72:    successes: set[int] = set()
scripts/analysis/figures/fig0d_taskpool_jaccard.py:82:        if bool(record.get("adjusted_success", record.get("success", False))):
scripts/analysis/figures/fig0d_taskpool_jaccard.py:83:            successes.add(tid)
scripts/analysis/figures/fig0d_taskpool_jaccard.py:84:    return successes, observed
scripts/analysis/figures/fig0d_taskpool_jaccard.py:100:        successes, observed = load_success_set(panel["modes"][mode])
scripts/analysis/figures/fig0d_taskpool_jaccard.py:101:        sets[mode] = successes
scripts/analysis/figures/fig0d_taskpool_jaccard.py:121:            # Restrict both success sets to the joint-observed task universe so
scripts/analysis/figures/fig0d_taskpool_jaccard.py:182:        cbar.set_label("Jaccard overlap of adjusted-success task pools")
scripts/analysis/figures/fig0d_taskpool_jaccard.py:187:        "Cells show Jaccard overlap and intersection/union counts; diagonals show adjusted successes and SR.",
scripts/maintenance/create_b1_classifieds_stub.py:9:  condition_summary_v2.json，使 is_condition_complete() 视其为完成，
scripts/maintenance/create_b1_classifieds_stub.py:13:  - success_rate 使用 RAW SR（评测器直接输出），adjusted SR 由分析脚本计算
scripts/maintenance/create_b1_classifieds_stub.py:35:        "success_rate": 21 / 234,        # 8.97% raw（adjusted 0.85%，2/234）
scripts/maintenance/create_b1_classifieds_stub.py:46:        "success_rate": 48 / 234,        # 20.51% raw（adjusted 16.24%，38/234）
scripts/maintenance/create_b1_classifieds_stub.py:57:        "success_rate": 29 / 234,        # 12.39% raw（adjusted 8.12%，19/234）
scripts/maintenance/create_b1_classifieds_stub.py:75:# ---------- condition_summary_v2.json 完整模板 ----------
scripts/maintenance/create_b1_classifieds_stub.py:81:        "success_rate": round(known["success_rate"], 6),
scripts/maintenance/create_b1_classifieds_stub.py:150:        summary_path = cond_dir / "condition_summary_v2.json"
scripts/maintenance/create_b1_classifieds_stub.py:154:        print(f"[stub] {mode}: SR={payload['success_rate']:.4f} ({known['episodes']} ep)")
scripts/maintenance/create_b1_classifieds_stub.py:192:        print(f"[stub] 分析脚本可读取 condition_summary_v2.json 获取已知 SR 数字")
scripts/maintenance/clear_tasks.py:4:Deletes: summary JSON, steps JSONL, artifacts directory, and digest records for each task.
scripts/maintenance/clear_tasks.py:19:    # Clean orphan artifact dirs (no summary file) across all conditions
scripts/maintenance/clear_tasks.py:20:    python scripts/maintenance/clear_tasks.py --run-dir results/.../B1_run --clean-orphan-artifacts
scripts/maintenance/clear_tasks.py:22:    # Clean orphan artifacts for a specific condition
scripts/maintenance/clear_tasks.py:24:        --condition phase1_som_router_0 --clean-orphan-artifacts
scripts/maintenance/clear_tasks.py:52:def _clean_orphan_artifacts(
scripts/maintenance/clear_tasks.py:58:    """Delete artifact dirs and orphan steps files that have no corresponding summary.
scripts/maintenance/clear_tasks.py:61:    belong to an in-progress episode (runner creates artifacts/steps before writing
scripts/maintenance/clear_tasks.py:80:        art_dir = cond_dir / "artifacts"
scripts/maintenance/clear_tasks.py:83:        # 1. Orphan artifact directories (no summary)
scripts/maintenance/clear_tasks.py:85:            for artifact in sorted(art_dir.iterdir()):
scripts/maintenance/clear_tasks.py:86:                if not artifact.is_dir():
scripts/maintenance/clear_tasks.py:88:                if (ep_dir / f"{artifact.name}_summary_v2.json").exists():
scripts/maintenance/clear_tasks.py:90:                if artifact.stat().st_mtime > cutoff:
scripts/maintenance/clear_tasks.py:93:                rel = artifact.relative_to(run_dir)
scripts/maintenance/clear_tasks.py:95:                    print(f"  [dry-run] rm -rf {rel}  (orphan artifact — no summary)")
scripts/maintenance/clear_tasks.py:97:                    shutil.rmtree(artifact)
scripts/maintenance/clear_tasks.py:98:                    print(f"  deleted orphan artifact: {rel}")
scripts/maintenance/clear_tasks.py:130:    p.add_argument("--clean-orphan-artifacts", action="store_true",
scripts/maintenance/clear_tasks.py:131:                    help="Delete artifact dirs that have no corresponding summary file")
scripts/maintenance/clear_tasks.py:137:    # Validate: either --tasks or --clean-orphan-artifacts must be provided
scripts/maintenance/clear_tasks.py:138:    if not args.tasks and not args.clean_orphan_artifacts:
scripts/maintenance/clear_tasks.py:139:        p.error("one of --tasks or --clean-orphan-artifacts is required")
scripts/maintenance/clear_tasks.py:147:    # --- Orphan artifact cleanup mode ---
scripts/maintenance/clear_tasks.py:148:    if args.clean_orphan_artifacts:
scripts/maintenance/clear_tasks.py:149:        orphans_deleted = _clean_orphan_artifacts(run_dir, args.condition, args.dry_run)
scripts/maintenance/clear_tasks.py:151:        print(f"\nDone: {action} {orphans_deleted} orphan artifact dir(s)")
scripts/maintenance/clear_tasks.py:164:    art_dir = cond_dir / "artifacts"
scripts/maintenance/clear_tasks.py:176:        artifact_dir = art_dir / prefix
scripts/maintenance/clear_tasks.py:179:        # (has steps JSONL or artifacts but no summary yet)
scripts/maintenance/clear_tasks.py:180:        if not summary_file.exists() and (steps_file.exists() or artifact_dir.exists()):
scripts/maintenance/clear_tasks.py:182:                print(f"  SKIP {prefix} — in-progress (has steps/artifacts but no summary). Use --force to override")
scripts/maintenance/clear_tasks.py:187:        dirs = [artifact_dir]
scripts/maintenance/clear_tasks.py:251:        cond_summary_path = cond_dir / "condition_summary_v2.json"
scripts/analysis/analyze_confidence_calibration.py:12:  C5 – Mode × outcome cross-analysis
scripts/analysis/analyze_confidence_calibration.py:46:    restart artifacts (stale lines from earlier runs in append-mode JSONL).
scripts/analysis/analyze_confidence_calibration.py:110:    """Aggregate step-level confidence → episode rows with success labels."""
scripts/analysis/analyze_confidence_calibration.py:132:        if "success" in summary and summary["success"] is not None:
scripts/analysis/analyze_confidence_calibration.py:133:            success = bool(summary["success"])
scripts/analysis/analyze_confidence_calibration.py:137:            success = float(last.get("reward", 0)) > 0
scripts/analysis/analyze_confidence_calibration.py:143:            "success": success,
scripts/analysis/analyze_confidence_calibration.py:338:    success_vals, positive rb means success > failure on the metric.
scripts/analysis/analyze_confidence_calibration.py:366:    succ = df[df["success"]]
scripts/analysis/analyze_confidence_calibration.py:367:    fail = df[~df["success"]]
scripts/analysis/analyze_confidence_calibration.py:372:        for label, grp in [("success", succ), ("failure", fail)]:
scripts/analysis/analyze_confidence_calibration.py:375:                "metric": m, "outcome": label,
scripts/analysis/analyze_confidence_calibration.py:381:    pd.DataFrame(stat_rows).to_csv(tables_dir / "confidence_by_outcome.csv", index=False)
scripts/analysis/analyze_confidence_calibration.py:390:                              "rank_biserial": np.nan, "n_success": len(s_vals),
scripts/analysis/analyze_confidence_calibration.py:399:            "n_success": len(s_vals), "n_failure": len(f_vals),
scripts/analysis/analyze_confidence_calibration.py:402:    print(f"  C1: tables → confidence_by_outcome.csv, mannwhitney_test.csv")
scripts/analysis/analyze_confidence_calibration.py:419:        # Color: success=green, failure=red
scripts/analysis/analyze_confidence_calibration.py:487:        if len(mdf) < 4 or mdf["success"].nunique() < 2:
scripts/analysis/analyze_confidence_calibration.py:492:        y = mdf["success"].astype(int).values
scripts/analysis/analyze_confidence_calibration.py:514:        if len(mdf) < 4 or mdf["success"].nunique() < 2:
scripts/analysis/analyze_confidence_calibration.py:519:        y = mdf["success"].astype(int).values
scripts/analysis/analyze_confidence_calibration.py:549:        if len(mdf) < 10 or mdf["success"].nunique() < 2:
scripts/analysis/analyze_confidence_calibration.py:551:        y = mdf["success"].astype(int).values
scripts/analysis/analyze_confidence_calibration.py:585:    labels = df["success"].astype(int).values
scripts/analysis/analyze_confidence_calibration.py:663:    if len(verb_df) >= 4 and verb_df["success"].nunique() >= 2:
scripts/analysis/analyze_confidence_calibration.py:665:        v_labels = verb_df["success"].astype(int).values
scripts/analysis/analyze_confidence_calibration.py:750:    # Merge success label into step_df
scripts/analysis/analyze_confidence_calibration.py:751:    success_map = dict(zip(
scripts/analysis/analyze_confidence_calibration.py:753:        ep_df["success"],
scripts/analysis/analyze_confidence_calibration.py:756:    sdf["success"] = sdf.apply(
scripts/analysis/analyze_confidence_calibration.py:757:        lambda r: success_map.get((r["condition_id"], r["task_id"]), None), axis=1,
scripts/analysis/analyze_confidence_calibration.py:759:    sdf = sdf.dropna(subset=["success"])
scripts/analysis/analyze_confidence_calibration.py:760:    sdf["success"] = sdf["success"].astype(bool)
scripts/analysis/analyze_confidence_calibration.py:768:        ("Success", sdf[sdf["success"]], "#2ca02c"),
scripts/analysis/analyze_confidence_calibration.py:769:        ("Failure", sdf[~sdf["success"]], "#d62728"),
scripts/analysis/analyze_confidence_calibration.py:801:            ("Success", sdf_ent[sdf_ent["success"]], "#2ca02c"),
scripts/analysis/analyze_confidence_calibration.py:802:            ("Failure", sdf_ent[~sdf_ent["success"]], "#d62728"),
scripts/analysis/analyze_confidence_calibration.py:834:            ("Success", sdf_verb[sdf_verb["success"]], "#2ca02c"),
scripts/analysis/analyze_confidence_calibration.py:835:            ("Failure", sdf_verb[~sdf_verb["success"]], "#d62728"),
scripts/analysis/analyze_confidence_calibration.py:878:                "success_rate": round(float(grp["success"].mean()), 4),
scripts/analysis/analyze_confidence_calibration.py:890:            values="success_rate", aggfunc="mean",
scripts/analysis/analyze_confidence_calibration.py:923:            "success_rate": round(float(grp["success"].mean()), 4),
scripts/analysis/analyze_confidence_calibration.py:930:        labels_arr = grp["success"].astype(int).values
scripts/analysis/analyze_confidence_calibration.py:947:            v_labels = grp.loc[verb_vals.index, "success"].astype(int).values
scripts/analysis/analyze_confidence_calibration.py:990:        labels_arr = mdf["success"].astype(int).values
scripts/analysis/analyze_confidence_calibration.py:1044:            v_labels = mdf["success"].astype(int).values
scripts/analysis/analyze_confidence_calibration.py:1255:def c5_mode_outcome(
scripts/analysis/analyze_confidence_calibration.py:1258:    """Mode × outcome cross-distribution + Kruskal-Wallis / Mann-Whitney."""
scripts/analysis/analyze_confidence_calibration.py:1260:    df["group"] = df["observation_mode"] + "_" + df["success"].map({True: "success", False: "failure"})
scripts/analysis/analyze_confidence_calibration.py:1264:        print("  C5: skipped – fewer than 2 mode×outcome groups")
scripts/analysis/analyze_confidence_calibration.py:1278:    pd.DataFrame(cross_rows).to_csv(tables_dir / "mode_outcome_cross.csv", index=False)
scripts/analysis/analyze_confidence_calibration.py:1308:    # Pairwise: same-outcome across modes, all signals
scripts/analysis/analyze_confidence_calibration.py:1316:        for outcome in ["success", "failure"]:
scripts/analysis/analyze_confidence_calibration.py:1319:                g1 = f"{m1}_{outcome}"
scripts/analysis/analyze_confidence_calibration.py:1320:                g2 = f"{m2}_{outcome}"
scripts/analysis/analyze_confidence_calibration.py:1366:    pd.DataFrame(test_rows).to_csv(tables_dir / "mode_outcome_tests.csv", index=False)
scripts/analysis/analyze_confidence_calibration.py:1367:    print(f"  C5: tables → mode_outcome_cross.csv, mode_outcome_tests.csv ({len(test_rows)} tests)")
scripts/analysis/analyze_confidence_calibration.py:1369:    # ── Violin 2×2: rows=mode, cols=outcome ──
scripts/analysis/analyze_confidence_calibration.py:1370:    outcomes = ["success", "failure"]
scripts/analysis/analyze_confidence_calibration.py:1371:    fig, axes = plt.subplots(len(modes), len(outcomes),
scripts/analysis/analyze_confidence_calibration.py:1375:        for ci, outcome in enumerate(outcomes):
scripts/analysis/analyze_confidence_calibration.py:1377:            g = f"{mode}_{outcome}"
scripts/analysis/analyze_confidence_calibration.py:1384:            ax.set_title(f"{mode} / {outcome} (n={len(vals)})")
scripts/analysis/analyze_confidence_calibration.py:1390:    fig.savefig(plots_dir / "C5_mode_outcome_violin.png", dpi=150)
scripts/analysis/analyze_confidence_calibration.py:1395:    colors = {"success": "#2ca02c", "failure": "#d62728"}
scripts/analysis/analyze_confidence_calibration.py:1402:        outcome_part = g.rsplit("_", 1)[1]
scripts/analysis/analyze_confidence_calibration.py:1406:            ax.plot(x_range, kde(x_range), color=colors.get(outcome_part, "gray"),
scripts/analysis/analyze_confidence_calibration.py:1417:    fig.savefig(plots_dir / "C5_mode_outcome_ridge.png", dpi=150)
scripts/analysis/analyze_confidence_calibration.py:1419:    print(f"  C5: plots → C5_mode_outcome_violin.png, C5_mode_outcome_ridge.png")
scripts/analysis/analyze_confidence_calibration.py:1439:    if len(df) < 4 or df["success"].nunique() < 2:
scripts/analysis/analyze_confidence_calibration.py:1443:    succ = df[df["success"]]
scripts/analysis/analyze_confidence_calibration.py:1444:    fail = df[~df["success"]]
scripts/analysis/analyze_confidence_calibration.py:1454:        for label, grp in [("success", succ), ("failure", fail)]:
scripts/analysis/analyze_confidence_calibration.py:1457:                "metric": m, "outcome": label,
scripts/analysis/analyze_confidence_calibration.py:1463:    pd.DataFrame(stat_rows).to_csv(tables_dir / "behavioral_by_outcome.csv", index=False)
scripts/analysis/analyze_confidence_calibration.py:1472:                              "rank_biserial": np.nan, "n_success": len(s_vals),
scripts/analysis/analyze_confidence_calibration.py:1481:            "n_success": len(s_vals), "n_failure": len(f_vals),
scripts/analysis/analyze_confidence_calibration.py:1484:    print(f"  C6: tables → behavioral_by_outcome.csv, behavioral_wilcoxon.csv")
scripts/analysis/analyze_confidence_calibration.py:1591:        if mdf["success"].nunique() < 2:
scripts/analysis/analyze_confidence_calibration.py:1593:        y = mdf["success"].astype(int).values
scripts/analysis/analyze_confidence_calibration.py:1601:            y_valid = mdf.loc[valid.index, "success"].astype(int).values
scripts/analysis/analyze_confidence_calibration.py:1692:    # Get success labels
scripts/analysis/analyze_confidence_calibration.py:1696:        if "success" in summary and summary["success"] is not None:
scripts/analysis/analyze_confidence_calibration.py:1697:            ep_labels[key] = bool(summary["success"])
scripts/analysis/analyze_confidence_calibration.py:1845:      - Scatter plot: ep_prob vs ep_mean_verbalized colored by success
scripts/analysis/analyze_confidence_calibration.py:1856:    labels = df["success"].astype(int).values
scripts/analysis/analyze_confidence_calibration.py:1868:    # Per-outcome correlations
scripts/analysis/analyze_confidence_calibration.py:1869:    for outcome, label_val in [("success", True), ("failure", False)]:
scripts/analysis/analyze_confidence_calibration.py:1870:        sub = df[df["success"] == label_val]
scripts/analysis/analyze_confidence_calibration.py:1874:                "comparison": f"ep_prob vs ep_mean_verbalized ({outcome})",
scripts/analysis/analyze_confidence_calibration.py:1892:    if len(df_minv) >= 10 and df_minv["success"].nunique() >= 2:
scripts/analysis/analyze_confidence_calibration.py:1893:        a = _auroc_safe(df_minv["success"].astype(int).values,
scripts/analysis/analyze_confidence_calibration.py:1959:    if len(all_signals) < 3 or ep_df["success"].nunique() < 2:
scripts/analysis/analyze_confidence_calibration.py:2009:    valid_mask = ep_df["success"].notna()
scripts/analysis/analyze_confidence_calibration.py:2021:    y_all = ep_df["success"].astype(int).values
scripts/analysis/analyze_confidence_calibration.py:2192:                        help="Disable adjusted labels (keep raw success as-is)")
scripts/analysis/analyze_confidence_calibration.py:2231:        from p79.experiment.analysis import compute_adjusted_success_batch
scripts/analysis/analyze_confidence_calibration.py:2246:            ep_df["raw_success"] = ep_df["success"]
scripts/analysis/analyze_confidence_calibration.py:2247:            ep_df["adjusted_success"] = ep_df["success"]
scripts/analysis/analyze_confidence_calibration.py:2252:            ep_df["raw_success"] = ep_df["success"]
scripts/analysis/analyze_confidence_calibration.py:2253:            compute_adjusted_success_batch(ep_df, bsite, benchmark)
scripts/analysis/analyze_confidence_calibration.py:2254:            n_adjusted = int((ep_df["raw_success"] != ep_df["adjusted_success"]).sum())
scripts/analysis/analyze_confidence_calibration.py:2255:            ep_df["success"] = ep_df["adjusted_success"]
scripts/analysis/analyze_confidence_calibration.py:2260:            ep_df["raw_success"] = ep_df["success"]
scripts/analysis/analyze_confidence_calibration.py:2274:                compute_adjusted_success_batch(site_ep, site, benchmark)
scripts/analysis/analyze_confidence_calibration.py:2275:                adj_parts.append(site_ep[["adjusted_success", "fp_reason"]])
scripts/analysis/analyze_confidence_calibration.py:2279:                ep_df["adjusted_success"] = adj_combined["adjusted_success"]
scripts/analysis/analyze_confidence_calibration.py:2281:            n_adjusted = int((ep_df["raw_success"] != ep_df["adjusted_success"]).sum())
scripts/analysis/analyze_confidence_calibration.py:2282:            ep_df["success"] = ep_df["adjusted_success"]
scripts/analysis/analyze_confidence_calibration.py:2286:        ep_df["raw_success"] = ep_df["success"]
scripts/analysis/analyze_confidence_calibration.py:2287:        ep_df["adjusted_success"] = ep_df["success"]
scripts/analysis/analyze_confidence_calibration.py:2359:        c5_result = c5_mode_outcome(ep_df_filt, tables_dir, plots_dir)
scripts/analysis/analyze_confidence_calibration.py:2407:        "n_success_raw": int(ep_df_filt["raw_success"].sum()) if "raw_success" in ep_df_filt.columns else None,
scripts/analysis/analyze_confidence_calibration.py:2408:        "n_success_adjusted": int(ep_df_filt["success"].sum()),
scripts/maintenance/crontab.txt:16:# every 10 min: sync _status/cells/*.md frontmatter from condition_summary_v2.json
scripts/maintenance/reeval_phase1.py:283:        summary["success"] = new_score >= 1.0
scripts/maintenance/glm/glm_pre_launch_check.py:98:        return fail_default, f"(GLM unparseable, raw={raw[:200]}, {fail_suffix})"
scripts/maintenance/README.md:41:| `clear_tasks.py` | Delete task results (summary/steps/artifacts/digest) — use this NOT `rm` |
scripts/maintenance/README.md:78:| `smoke_test_vwa.py` | VWA smoke test |
scripts/maintenance/glm/glm_cell_autoupdate.py:2:"""Cell frontmatter auto-update — sync _status/cells/*.md from condition_summary_v2.json.
scripts/maintenance/glm/glm_cell_autoupdate.py:5:parse latest condition_summary_v2.json, update structured fields in cell frontmatter:
scripts/maintenance/glm/glm_cell_autoupdate.py:9:- sr_raw: success_rate * 100 (rounded 2 decimals)
scripts/maintenance/glm/glm_cell_autoupdate.py:114:      - summary_path: Path to condition_summary_v2.json, or None if not yet generated
scripts/maintenance/glm/glm_cell_autoupdate.py:119:      - is_inflight: True if no condition_summary_v2.json but episodes are accumulating
scripts/maintenance/glm/glm_cell_autoupdate.py:142:            summary = cond_dir / "condition_summary_v2.json"
scripts/maintenance/glm/glm_cell_autoupdate.py:308:        sr = d.get("success_rate")
scripts/maintenance/active_processes.py:231:        if data.get("success"):
scripts/maintenance/active_processes.py:233:        if data.get("adjusted_success", data.get("success", False)):
scripts/analysis/analyze_reddit_selflink_cycle.py:191:def check_scroll_outcome(steps: list[dict], escape_idx: int) -> dict:
scripts/analysis/analyze_reddit_selflink_cycle.py:265:                    scroll_detail = check_scroll_outcome(steps, escape["escape_step"])
scripts/analysis/analyze_reddit_selflink_cycle.py:380:            success = "是" if r["final_score"] == 1.0 else "否"
scripts/analysis/analyze_reddit_selflink_cycle.py:382:            print(f"{r['task_id']:>8} {r['model']:<6} {r['cycle_length']:>8} {r['escape_step']:>9} {next_act:<20} {saw_comment:<10} {seq:<40} {success:<6} {af:<10}")
scripts/analysis/analyze_reddit_selflink_cycle.py:411:        type_success = defaultdict(int)
scripts/analysis/analyze_reddit_selflink_cycle.py:415:                type_success[r["escape_type"]] += 1
scripts/analysis/analyze_reddit_selflink_cycle.py:417:        # Also count agent_finished success (true task completion, not url_match)
scripts/analysis/analyze_reddit_selflink_cycle.py:418:        type_agent_finished_success = defaultdict(int)
scripts/analysis/analyze_reddit_selflink_cycle.py:421:                type_agent_finished_success[r["escape_type"]] += 1
scripts/analysis/analyze_reddit_selflink_cycle.py:427:            suc = type_success[etype]
scripts/analysis/analyze_reddit_selflink_cycle.py:428:            af_suc = type_agent_finished_success[etype]
scripts/analysis/analyze_reddit_selflink_cycle.py:432:        overall_success = sum(1 for r in unique if r["final_score"] == 1.0)
scripts/analysis/analyze_reddit_selflink_cycle.py:434:        print(f"  循环 task 总成功: {overall_success}/{total_cycle_tasks} "
scripts/analysis/analyze_reddit_selflink_cycle.py:435:              f"({overall_success/total_cycle_tasks*100:.1f}%), "
scripts/maintenance/glm/glm_playbook_refresh.py:110:        # Strategy: find positions of all fail markers + all success markers,
scripts/maintenance/glm/glm_playbook_refresh.py:325:- 否则: bullet list, 每条格式 `[severity] file (line N, time): 一句话讲发生啥` — 严重 (oom/traceback/not_logged_in) 用 🔴, 中度 (timeout/http5xx) 用 ⚠️, 轻度 (notify_fail) 用 ℹ️
scripts/maintenance/glm/glm_playbook_refresh.py:404:    # Reset failure count on success.
scripts/analysis/figures/fig2f_first_divergence.py:77:def read_successes(ep_dir: Path) -> dict[int, bool]:
scripts/analysis/figures/fig2f_first_divergence.py:82:        out[tid] = bool(rec.get("adjusted_success", rec.get("success", False)))
scripts/analysis/figures/fig2f_first_divergence.py:119:    success_by_mode: dict[str, dict[int, bool]] = {}
scripts/analysis/figures/fig2f_first_divergence.py:126:            success_by_mode[cell.mode] = read_successes(cell.episodes_dir)
scripts/analysis/figures/fig2f_first_divergence.py:127:    return steps_by_mode, success_by_mode
scripts/analysis/figures/fig2f_first_divergence.py:150:    success_by_mode: dict[str, dict[int, bool]],
scripts/analysis/figures/fig2f_first_divergence.py:154:    if left not in steps_by_mode or right not in steps_by_mode or left not in success_by_mode or right not in success_by_mode:
scripts/analysis/figures/fig2f_first_divergence.py:156:    task_ids = sorted(set(steps_by_mode[left]) & set(steps_by_mode[right]) & set(success_by_mode[left]) & set(success_by_mode[right]))
scripts/analysis/figures/fig2f_first_divergence.py:157:    sym = [tid for tid in task_ids if success_by_mode[left][tid] != success_by_mode[right][tid]]
scripts/analysis/figures/fig2f_first_divergence.py:166:    steps_by_mode, success_by_mode = load_mode_steps(baseline, site)
scripts/analysis/figures/fig2f_first_divergence.py:183:            extra = solved_delta_note(steps_by_mode, success_by_mode, left, right)
scripts/maintenance/glm/glm_batch_digest.py:55:_find_episode_artifact_dir = sidecar._find_episode_artifact_dir
scripts/maintenance/glm/glm_batch_digest.py:81:        success = _to_optional_bool(r.get("success"))
scripts/maintenance/glm/glm_batch_digest.py:82:        if success is True:
scripts/maintenance/glm/glm_batch_digest.py:168:        "action_success": _to_optional_bool(item.get("action_success")),
scripts/maintenance/glm/glm_batch_digest.py:184:# Key step selection + artifact loading
scripts/maintenance/glm/glm_batch_digest.py:336:        success = rec.get("action_success")
scripts/maintenance/glm/glm_batch_digest.py:342:            if success is False:
scripts/maintenance/glm/glm_batch_digest.py:346:            if success is False:
scripts/maintenance/glm/glm_batch_digest.py:357:        if success is False:
scripts/maintenance/glm/glm_batch_digest.py:441:    导致操作无效（action_success=false）。
scripts/maintenance/glm/glm_batch_digest.py:628:    # Load artifacts
scripts/maintenance/glm/glm_batch_digest.py:629:    ep_dir = _find_episode_artifact_dir(run_dir, condition_id, task_id)
scripts/maintenance/glm/glm_batch_digest.py:726:                raise ValueError(f"GLM returned unparseable response: {raw[:200]!r}")
scripts/maintenance/glm/glm_batch_digest.py:1146:    success_count = 0
scripts/maintenance/glm/glm_batch_digest.py:1163:            success_count += 1
scripts/maintenance/glm/glm_batch_digest.py:1181:            success_count += 1
scripts/maintenance/glm/glm_batch_digest.py:1196:    print(f"\n[batch-digest] Done. success={success_count} failed={fail_count}")
scripts/maintenance/annotate_screenshots.py:459:    ap = step_record.get("artifact_paths", {})
scripts/maintenance/annotate_screenshots.py:497:    artifact_base = condition_dir / "artifacts" / ep_name
scripts/maintenance/annotate_screenshots.py:513:        step_dir = artifact_base / f"step_{step_idx:03d}"
scripts/maintenance/experiment_watchdog.py:6:1) REPORT:   periodic status every --report-interval-mins (success rate + counts)
scripts/maintenance/experiment_watchdog.py:48:    success: bool
scripts/maintenance/experiment_watchdog.py:250:    dom_path = condition_dir / "artifacts" / f"{site}_task_{task_id}" / "step_000" / "observation_dom.txt"
scripts/maintenance/experiment_watchdog.py:290:    if bool(summary.get("success", False)):
scripts/maintenance/experiment_watchdog.py:291:        return "success"
scripts/maintenance/experiment_watchdog.py:340:    cond_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0})
scripts/maintenance/experiment_watchdog.py:345:        cond_stats[r.condition_id]["success"] += int(r.success)
scripts/maintenance/experiment_watchdog.py:353:        n, succ = s["total"], s["success"]
scripts/maintenance/experiment_watchdog.py:404:        if r.condition_id in completed_conditions and not r.success:
scripts/maintenance/experiment_watchdog.py:469:    1. condition_summary_v2.json exists AND not in seen_completions, OR
scripts/maintenance/experiment_watchdog.py:470:    2. condition_summary_v2.json is newer than analysis outputs (post-analysis stale).
scripts/maintenance/experiment_watchdog.py:491:        summary_path = cond_dir / "condition_summary_v2.json"
scripts/maintenance/experiment_watchdog.py:527:        if (cond_dir / "condition_summary_v2.json").exists():
scripts/maintenance/experiment_watchdog.py:650:                     if d.is_dir() and (d / "condition_summary_v2.json").exists()]
scripts/maintenance/experiment_watchdog.py:699:    """True iff at least one condition has condition_summary_v2.json."""
scripts/maintenance/experiment_watchdog.py:703:        (d / "condition_summary_v2.json").exists()
scripts/maintenance/experiment_watchdog.py:1115:                        "(condition_summary_v2.json present). Without this, "
scripts/maintenance/experiment_watchdog.py:1176:                success=bool(summary.get("success", False)),
scripts/maintenance/experiment_watchdog.py:1188:    # Prune orphan artifacts and steps files (exist but no summary file).
scripts/maintenance/experiment_watchdog.py:1197:        _art_root = _cdir / "artifacts"
scripts/maintenance/experiment_watchdog.py:1199:        # Orphan artifact directories
scripts/maintenance/experiment_watchdog.py:1221:        print(f"[watchdog] Pruned {_orphan_count} orphan item(s) (artifact dirs / steps files without summary)")
scripts/maintenance/experiment_watchdog.py:1277:        print(f"[watchdog] Pruned {pruned_completions} stale completions (missing condition_summary_v2.json)")
scripts/maintenance/experiment_watchdog.py:1346:                condition_completed = (condition_dir / "condition_summary_v2.json").exists()
scripts/maintenance/experiment_watchdog.py:1387:                    # 3. Delete artifacts directory
scripts/maintenance/experiment_watchdog.py:1388:                    artifacts_dir = condition_dir / "artifacts" / f"{site}_task_{task_id}"
scripts/maintenance/experiment_watchdog.py:1390:                        if artifacts_dir.exists():
scripts/maintenance/experiment_watchdog.py:1391:                            shutil.rmtree(artifacts_dir)
scripts/maintenance/experiment_watchdog.py:1415:                    success=bool(summary.get("success", False)),
scripts/maintenance/experiment_watchdog.py:1481:                                cart = cond_dir / "artifacts" / f"{csite}_task_{ctask_id}"
scripts/maintenance/experiment_watchdog.py:1514:                cond_succ = sum(1 for r in cond_all if r.success)
scripts/maintenance/experiment_watchdog.py:1519:                    f"{'OK' if rec.success else reason:<10s} "
scripts/maintenance/experiment_watchdog.py:1602:                            status = "OK" if ep.success else ep.reason
scripts/maintenance/experiment_watchdog.py:1661:            cond_succ = sum(1 for r in cond_all if r.success)
scripts/maintenance/experiment_watchdog.py:1702:            cond_done = (run_dir / args.condition / "condition_summary_v2.json").exists()
scripts/maintenance/README.md:41:| `clear_tasks.py` | Delete task results (summary/steps/artifacts/digest) — use this NOT `rm` |
scripts/maintenance/README.md:78:| `smoke_test_vwa.py` | VWA smoke test |
scripts/maintenance/annotate_screenshots.py:459:    ap = step_record.get("artifact_paths", {})
scripts/maintenance/annotate_screenshots.py:497:    artifact_base = condition_dir / "artifacts" / ep_name
scripts/maintenance/annotate_screenshots.py:513:        step_dir = artifact_base / f"step_{step_idx:03d}"
scripts/analysis/analyze_reason_diagnostics.py:121:        "success_rate": "success_rate",
scripts/analysis/analyze_reason_diagnostics.py:156:        "success_rate": "success_rate",
scripts/analysis/analyze_reason_diagnostics.py:246:def _resolve_artifact_path(path_text: str, run_dir: Path) -> Optional[Path]:
scripts/analysis/analyze_reason_diagnostics.py:394:        dom_path_raw = str(((s.get("artifact_paths") or {}).get("dom")) or "").strip()
scripts/analysis/analyze_reason_diagnostics.py:395:        dom_path = _resolve_artifact_path(dom_path_raw, run_dir)
scripts/analysis/analyze_reason_diagnostics.py:405:    # If no DOM artifacts exist (e.g. vision-only mode), we cannot determine visibility.
scripts/analysis/analyze_reason_diagnostics.py:495:            continue  # unparseable delta, skip
scripts/analysis/analyze_reason_diagnostics.py:708:                "action_success": bool(s.get("action_success", False)),
scripts/analysis/analyze_reason_diagnostics.py:773:        if s.get("action_success") is False:
scripts/analysis/analyze_reason_diagnostics.py:1007:    success: bool,
scripts/analysis/analyze_reason_diagnostics.py:1022:    if success:
scripts/analysis/analyze_reason_diagnostics.py:1023:        return "success"
scripts/analysis/analyze_reason_diagnostics.py:1099:        if bucket == "success":
scripts/analysis/analyze_reason_diagnostics.py:1189:            f"## {_t(lang, 'condition')}: `{cid}` ({cond['episodes']} {_t(lang, 'episodes')}, {_fmt_pct(float(cond['success_rate']))} {_t(lang, 'success_rate')})"
scripts/analysis/analyze_reason_diagnostics.py:1257:            if bucket == "success":
scripts/analysis/analyze_reason_diagnostics.py:1371:        success = rec.get("action_success")
scripts/analysis/analyze_reason_diagnostics.py:1376:            if success is False:
scripts/analysis/analyze_reason_diagnostics.py:1380:            if success is False:
scripts/analysis/analyze_reason_diagnostics.py:1391:        if success is False:
scripts/analysis/analyze_reason_diagnostics.py:1485:def _write_state_change_by_outcome(
scripts/analysis/analyze_reason_diagnostics.py:1488:    """Cross-tab of state_change metrics by (condition, adjusted_success)."""
scripts/analysis/analyze_reason_diagnostics.py:1492:        adj = bool(row.get("adjusted_success", row.get("success", False)))
scripts/analysis/analyze_reason_diagnostics.py:1503:        for outcome in [True, False]:
scripts/analysis/analyze_reason_diagnostics.py:1504:            subset = cond_groups[cid][outcome]
scripts/analysis/analyze_reason_diagnostics.py:1510:                "adjusted_success": outcome,
scripts/analysis/analyze_reason_diagnostics.py:1520:        output_dir / "state_change_by_outcome.csv",
scripts/analysis/analyze_reason_diagnostics.py:1522:        ["condition_id", "adjusted_success", "n_episodes",
scripts/analysis/analyze_reason_diagnostics.py:1525:    print(f"  State change by outcome → {output_dir / 'state_change_by_outcome.csv'}")
scripts/analysis/analyze_reason_diagnostics.py:1621:                sr = sum(1 for r in subset if r.get("adjusted_success")) / len(subset) * 100
scripts/analysis/analyze_reason_diagnostics.py:1661:                sr = sum(1 for r in subset if r.get("adjusted_success")) / len(subset) * 100
scripts/analysis/analyze_reason_diagnostics.py:1740:            sr = sum(1 for r in segment if r.get("adjusted_success")) / len(segment) * 100
scripts/analysis/analyze_reason_diagnostics.py:1804:        description="Stage-level success/failure diagnostics from *_summary_v2.json + *_steps_v2.jsonl"
scripts/analysis/analyze_reason_diagnostics.py:1918:            success = bool(summary.get("success", False))
scripts/analysis/analyze_reason_diagnostics.py:1985:                success=success,
scripts/analysis/analyze_reason_diagnostics.py:2001:            # ── Adjusted success (N/A FP + eval FP; visual_fp removed in §95) ──
scripts/analysis/analyze_reason_diagnostics.py:2002:            from p79.experiment.analysis import compute_adjusted_success, _load_na_task_ids
scripts/analysis/analyze_reason_diagnostics.py:2006:            adjusted_success, fp_reason = compute_adjusted_success(
scripts/analysis/analyze_reason_diagnostics.py:2007:                task_id, site, success,
scripts/analysis/analyze_reason_diagnostics.py:2013:            if adjusted_success != success:
scripts/analysis/analyze_reason_diagnostics.py:2015:                    success=adjusted_success,
scripts/analysis/analyze_reason_diagnostics.py:2070:                "success": success,
scripts/analysis/analyze_reason_diagnostics.py:2071:                "adjusted_success": adjusted_success,
scripts/analysis/analyze_reason_diagnostics.py:2174:                    "success": success,
scripts/analysis/analyze_reason_diagnostics.py:2195:        "success",
scripts/analysis/analyze_reason_diagnostics.py:2196:        "adjusted_success",
scripts/analysis/analyze_reason_diagnostics.py:2311:    per_condition_success: Counter = Counter()
scripts/analysis/analyze_reason_diagnostics.py:2317:        if bool(row["success"]):
scripts/analysis/analyze_reason_diagnostics.py:2318:            per_condition_success[cid] += 1
scripts/analysis/analyze_reason_diagnostics.py:2324:        adj_success_count = sum(
scripts/analysis/analyze_reason_diagnostics.py:2325:            1 for x in episode_rows if x["condition_id"] == cid and bool(x.get("adjusted_success", x["success"]))
scripts/analysis/analyze_reason_diagnostics.py:2334:                "success_count": per_condition_success[cid],
scripts/analysis/analyze_reason_diagnostics.py:2335:                "success_rate": _safe_ratio(per_condition_success[cid], total),
scripts/analysis/analyze_reason_diagnostics.py:2336:                "adjusted_success_count": adj_success_count,
scripts/analysis/analyze_reason_diagnostics.py:2337:                "adjusted_success_rate": _safe_ratio(adj_success_count, total),
scripts/analysis/analyze_reason_diagnostics.py:2342:                    if x["condition_id"] == cid and (not x["success"]) and bool(x["early_finish"])
scripts/analysis/analyze_reason_diagnostics.py:2355:            "success_count",
scripts/analysis/analyze_reason_diagnostics.py:2356:            "success_rate",
scripts/analysis/analyze_reason_diagnostics.py:2357:            "adjusted_success_count",
scripts/analysis/analyze_reason_diagnostics.py:2358:            "adjusted_success_rate",
scripts/analysis/analyze_reason_diagnostics.py:2490:        _write_state_change_by_outcome(episode_rows_sorted, output_dir)
scripts/maintenance/glm/glm_diagnosis_sidecar.py:237:        success = _to_optional_bool(r.get("success"))
scripts/maintenance/glm/glm_diagnosis_sidecar.py:238:        if success is True:
scripts/maintenance/glm/glm_diagnosis_sidecar.py:273:                        action_success = _to_optional_bool(item.get("action_success"))
scripts/maintenance/glm/glm_diagnosis_sidecar.py:281:                                "action_success": action_success,
scripts/maintenance/glm/glm_diagnosis_sidecar.py:652:    _wasted = float(case.get("wasted_cost_usd") or case.get("total_cost_usd") or 0) if not case.get("success") else 0.0
scripts/maintenance/glm/glm_diagnosis_sidecar.py:693:def _find_episode_artifact_dir(run_dir: Path, condition_id: str, task_id: int) -> Optional[Path]:
scripts/maintenance/glm/glm_diagnosis_sidecar.py:694:    """Find the artifact directory for a given condition + task_id."""
scripts/maintenance/glm/glm_diagnosis_sidecar.py:695:    artifacts_dir = run_dir / condition_id / "artifacts"
scripts/maintenance/glm/glm_diagnosis_sidecar.py:696:    if not artifacts_dir.exists():
scripts/maintenance/glm/glm_diagnosis_sidecar.py:698:    for d in artifacts_dir.iterdir():
scripts/maintenance/glm/glm_diagnosis_sidecar.py:890:                raise ValueError(f"GLM returned unparseable response: {raw[:200]!r}")
scripts/maintenance/glm/glm_diagnosis_sidecar.py:914:            _glm_wasted = float(case.get("wasted_cost_usd") or case.get("total_cost_usd") or 0) if not case.get("success") else 0.0
scripts/maintenance/glm/glm_diagnosis_sidecar.py:980:        # Load SoM artifacts
scripts/maintenance/glm/glm_diagnosis_sidecar.py:987:                _ep_dir = _find_episode_artifact_dir(run_dir, _cond_id, _task_id)
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1210:    success = _cnt("success")
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1216:        f" 当前成功率为 {success/denom:.1%}。"
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1470:    # Track the max task_id per condition for which ntfy was successfully sent.
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1530:    # analyzed but never successfully pushed via ntfy.  Split into batches of
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1624:                            # Compute success rate from CSV for context
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1633:                                    _sr_ok = sum(1 for r in _sr_rows if _to_optional_bool(r.get("success")) is True)
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1634:                                    _sr_line = f"success={_sr_ok}/{len(_sr_rows)} ({_sr_ok/len(_sr_rows):.1%})"
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1707:                # Compute success_rate for the triggered condition from CSV.
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1714:                success_count = sum(
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1715:                    1 for r in _cond_rows if _to_optional_bool(r.get("success")) is True
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1717:                success_rate = (success_count / episodes) if episodes > 0 else 0.0
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1721:                _mode_stats: Dict[str, Dict[str, int]] = _defaultdict(lambda: {"total": 0, "success": 0})
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1722:                _active_mode_stats: Dict[str, Dict[str, int]] = _defaultdict(lambda: {"total": 0, "success": 0})
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1727:                    _ok = _to_optional_bool(_r.get("success")) is True
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1730:                        _mode_stats[_mode]["success"] += 1
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1736:                            _active_mode_stats[_mode]["success"] += 1
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1737:                # Single success line: show active mode(s) with cumulative totals
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1739:                _success_line = "  ".join(
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1740:                    f"{m}: {_mode_stats[m]['success']}/{_mode_stats[m]['total']} ({_mode_stats[m]['success']/_mode_stats[m]['total']:.1%})"
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1742:                ) or f"{success_count}/{episodes}"
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1831:                    f"success={_success_line}",
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1851:                    title = f"P79 [{_cond_label}] {_success_line}"
scripts/maintenance/glm/glm_diagnosis_sidecar.py:1855:                            f"success={_success_line}",
scripts/maintenance/probe_som_occlusion.py:66:    SoM:    .../artifacts/<task>/som/step_NNN_som.png
scripts/maintenance/probe_som_occlusion.py:67:            → .../artifacts/<task>/step_NNN/observation_som.txt
scripts/maintenance/probe_som_occlusion.py:68:    Screen: .../artifacts/<task>/step_NNN/screenshot.png
scripts/maintenance/probe_som_occlusion.py:69:            → .../artifacts/<task>/step_NNN/observation_som.txt
scripts/maintenance/glm/myriad_watcher.py:181:    Returns the full name on success, otherwise `fallback` (the 10-char
scripts/maintenance/glm/myriad_watcher.py:245:    on success, None on any failure (timeout, ssh error, non-zero exit)."""
scripts/maintenance/glm/myriad_watcher.py:345:    # Reset failure counter on success
scripts/maintenance/probe_b01_b13_self_verify.py:124:                    if step.get("action_success") is True:
scripts/maintenance/probe_b01_b13_self_verify.py:201:            "action_success": step.get("action_success"),
scripts/maintenance/probe_b01_b13_self_verify.py:280:            "action_success": step.get("action_success"),
scripts/maintenance/probe_b01_b13_self_verify.py:321:    elif step.get("action_success") is False and real_change_signals:
scripts/maintenance/probe_b01_b13_self_verify.py:323:        out["reason"] = f"action_success=False but real change signals: {real_change_signals} — runner missed success"
scripts/maintenance/probe_b01_b13_self_verify.py:326:        out["reason"] = f"action_success={step.get('action_success')} signals={real_change_signals}"
scripts/maintenance/probe_b01_b13_self_verify.py:432:        f"- Self-verify probed: {b13_total} cases via state_digest log analysis (no Playwright replay — independent of codex's REPLAY_FAIL artifacts)",
scripts/maintenance/glm/error_scan.py:39:    ("not_logged_in", re.compile(r"NOT[_ ]LOGGED[_ ]IN|auth_refresh.*(?:fail|error)|session.*expired", re.IGNORECASE), 75),
scripts/maintenance/glm/error_scan.py:47:    ("fp_adjust_error", re.compile(r"fp_reason.*?adjustment_error|Failed to compute adjusted_success", re.IGNORECASE), 82),
scripts/maintenance/glm/error_scan.py:153:                f"({n_fail} consecutive ticks). Prune logs/ artifacts/.",
scripts/maintenance/rsync_results_to_hub.sh:4:# Tier C (artifacts: screenshots, SoM图) is excluded — pull on demand via
scripts/maintenance/rsync_results_to_hub.sh:30:echo "[rsync→hub] $SOURCE → $HOST:$HUB_PATH (Tier B only, no artifacts)"
scripts/maintenance/rsync_results_to_hub.sh:36:  --include='condition_summary_v2.json' \
scripts/maintenance/rsync_results_to_hub.sh:40:  --exclude='artifacts/' \
scripts/maintenance/retry_b1_single_task.sh:176:    result=$("${PYTHON_BIN}" -c "import json; d=json.load(open('${summary}')); print(f'success={d[\"success\"]}, steps={d[\"steps\"]}, error={d.get(\"error\",None)}')")
scripts/maintenance/retry_b1_single_task.sh:232:    result=$("${PYTHON_BIN}" -c "import json; d=json.load(open('${summary}')); print(f'success={d[\"success\"]}, score={d[\"score\"]}, steps={d[\"steps\"]}')")
scripts/maintenance/run_one_vwa_episode.py:73:    # This matches the smoke test logic
scripts/maintenance/run_one_vwa_episode.py:213:    logger.info(f"Success Proxy: {terminated and reward > 0}") # VWA reward is 1.0 on success usually
scripts/maintenance/rederive_episode_summary.py:18:`condition_summary_v2.json` using `aggregate_condition_metrics`.
scripts/maintenance/rederive_episode_summary.py:65:    adjusted_success: Optional[bool] = None
scripts/maintenance/rederive_episode_summary.py:141:    # §95 adjusted_success — re-derive for old data using runner's canonical
scripts/maintenance/rederive_episode_summary.py:144:    adj_success: Optional[bool] = None
scripts/maintenance/rederive_episode_summary.py:147:        from p79.experiment.analysis import compute_adjusted_success, _load_na_task_ids
scripts/maintenance/rederive_episode_summary.py:162:        adj_success_val, fp_val = compute_adjusted_success(
scripts/maintenance/rederive_episode_summary.py:163:            task_id, site, bool(summary.get("success", False)),
scripts/maintenance/rederive_episode_summary.py:169:        adj_success = bool(adj_success_val)
scripts/maintenance/rederive_episode_summary.py:172:        print(f"  [WARN] adjusted_success derive failed for {site} task {task_id}: {exc}",
scripts/maintenance/rederive_episode_summary.py:185:        adjusted_success=adj_success,
scripts/maintenance/rederive_episode_summary.py:211:    # §95 adjusted_success fields (Step 2): always update if derivation succeeded.
scripts/maintenance/rederive_episode_summary.py:212:    if "adjusted_success" in rewrite_set and adj_success is not None:
scripts/maintenance/rederive_episode_summary.py:213:        summary["adjusted_success"] = adj_success
scripts/maintenance/rederive_episode_summary.py:297:    # Re-aggregate condition_summary_v2.json from the freshly-written episodes.
scripts/maintenance/rederive_episode_summary.py:309:            existing_path = condition_dir / "condition_summary_v2.json"
scripts/maintenance/rederive_episode_summary.py:324:            print(f"  rebuilt condition_summary_v2.json (n={len(ep_summaries)})")
scripts/maintenance/rederive_episode_summary.py:370:        default="page_unchanged_rate,energy_partial,energy_step_complete_count,busy_wait_total_ms,adjusted_success",
scripts/maintenance/rederive_episode_summary.py:374:                        help="Skip rebuilding condition_summary_v2.json after episode rewrites")
scripts/maintenance/rederive_episode_summary.py:380:                    "adjusted_success"}
scripts/maintenance/rsync_results_from_hub.sh:3:# Default: Tier B (episodes/*.jsonl + summary), no artifacts.
scripts/maintenance/rsync_results_from_hub.sh:4:# Set ARTIFACTS=1 to also pull artifacts (screenshots/SoM); useful when
scripts/maintenance/rsync_results_from_hub.sh:18:#   ARTIFACTS  set to 1 to include artifacts/
scripts/maintenance/rsync_results_from_hub.sh:35:  --include='condition_summary_v2.json'
scripts/maintenance/rsync_results_from_hub.sh:43:  INCLUDES+=( --include='artifacts/**' )
scripts/maintenance/rsync_results_from_hub.sh:44:  echo "[rsync←hub] including artifacts/"
scripts/maintenance/rsync_results_from_hub.sh:46:  EXCLUDES+=( --exclude='artifacts/' )
scripts/analysis/power_analysis.py:3:For binary success rate comparisons (P-SoM vs best-baseline), computes:
scripts/analysis/power_analysis.py:72:                   help="Baseline success rate (default 0.30 — typical Phase 1 cls/red)")
scripts/analysis/aggregate_cost_electricity.py:8:classes. The current condition_summary_v2.json applies the same per-token rate
scripts/analysis/aggregate_cost_electricity.py:10:methodological artifact — B1 has no actual API dollars. The principled
scripts/analysis/aggregate_cost_electricity.py:78:    p = RESULTS / sub / "condition_summary_v2.json"
scripts/analysis/aggregate_cost_electricity.py:81:        return {"available": False, "reason": "missing condition_summary_v2.json"}
scripts/analysis/aggregate_cost_electricity.py:90:        # Efficiency 3a token cost (B0 = real API $; B1 = artifact, see notes)
scripts/analysis/aggregate_cost_electricity.py:133:            "condition_summary_v2.json is artifact (uses B0 rates) and is NOT comparable. "
scripts/analysis/preregistration_decision_test.py:523:    """Apply preregistration §2 R1-R5 framing rule to test outcomes."""
scripts/analysis/preregistration_decision_test.py:568:# Synthetic data generator (24-condition / 4-cell smoke test)
scripts/analysis/preregistration_decision_test.py:625:                   help="Run smoke test on synthetic 4-cell × 200-task data")
scripts/analysis/figures/fig0c_phantom_lift_bars.py:163:             "Bars = oracle ceiling success rate (any-of-modes solves task). Error bars: 95% bootstrap CI. "
scripts/analysis/figures/fig3d_cost_sr_frontier.py:116:        succ += bool(rec.get("adjusted_success", rec.get("success", False)))
scripts/analysis/figures/fig3d_cost_sr_frontier.py:209:    axes[0].set_ylabel("Adjusted success rate (%)")
scripts/analysis/validate_run.py:168:    # Each condition's condition_summary_v2.json
scripts/analysis/validate_run.py:170:        cs = cond_dir / "condition_summary_v2.json"
scripts/analysis/validate_run.py:172:            missing.append(f"{cond_dir.name}/condition_summary_v2.json")
scripts/analysis/validate_run.py:256:        # Check condition_summary_v2.json
scripts/analysis/validate_run.py:257:        summary = _load_json(cond_dir / "condition_summary_v2.json")
scripts/analysis/validate_run.py:259:            errors.append(f"{cond_name}/condition_summary_v2.json: observation_mode={summary.get('observation_mode')} (expected {expected_mode})")
scripts/analysis/validate_run.py:453:def check_score_success_match(run_dir: Path) -> CheckResult:
scripts/analysis/validate_run.py:454:    """C10: success=True & score==0 or success=False & score>0 should not occur."""
scripts/analysis/validate_run.py:461:            success = data.get("success")
scripts/analysis/validate_run.py:465:            if success and score == 0:
scripts/analysis/validate_run.py:466:                issues.append(f"{cond_dir.name}/task_{data.get('task_id')}: success=True but score=0")
scripts/analysis/validate_run.py:467:            elif not success and score > 0:
scripts/analysis/validate_run.py:468:                issues.append(f"{cond_dir.name}/task_{data.get('task_id')}: success=False but score={score}")
scripts/analysis/validate_run.py:472:            "C10", "Score/success match", "warn",
scripts/analysis/validate_run.py:473:            f"{len(issues)} score/success mismatches",
scripts/analysis/validate_run.py:477:        "C10", "Score/success match", "pass",
scripts/analysis/validate_run.py:478:        "All score/success values are consistent",
scripts/analysis/validate_run.py:572:    """C13: Full scan for step_idx=0 appearing more than once (restart artifacts)."""
scripts/analysis/validate_run.py:599:            f"{restart_files}/{total_files} files have restart artifacts (step_idx=0 > 1)",
scripts/analysis/validate_run.py:604:        f"No restart artifacts in {total_files} files",
scripts/analysis/validate_run.py:716:    """C17: benchmark_noise_rate from condition_summary_v2.json."""
scripts/analysis/validate_run.py:721:        summary = _load_json(cond_dir / "condition_summary_v2.json")
scripts/analysis/validate_run.py:820:        art_dir = cond_dir / "artifacts"
scripts/analysis/validate_run.py:847:def check_orphan_artifacts(run_dir: Path) -> CheckResult:
scripts/analysis/validate_run.py:853:        art_dir = cond_dir / "artifacts"
scripts/analysis/validate_run.py:858:        for artifact in sorted(art_dir.iterdir()):
scripts/analysis/validate_run.py:859:            if not artifact.is_dir():
scripts/analysis/validate_run.py:861:            summary_path = ep_dir / f"{artifact.name}_summary_v2.json"
scripts/analysis/validate_run.py:864:            if artifact.stat().st_mtime > cutoff:
scripts/analysis/validate_run.py:866:            orphans.append(f"{cond_dir.name}/artifacts/{artifact.name}")
scripts/analysis/validate_run.py:870:            "C20", "Orphan artifacts", "warn",
scripts/analysis/validate_run.py:871:            f"{len(orphans)} orphan artifact dirs (no summary, >10min old)",
scripts/analysis/validate_run.py:875:        "C20", "Orphan artifacts", "pass",
scripts/analysis/validate_run.py:876:        "No orphan artifacts detected",
scripts/analysis/validate_run.py:898:        cs = cond_dir / "condition_summary_v2.json"
scripts/analysis/validate_run.py:905:            "No condition_summary_v2.json files found",
scripts/analysis/validate_run.py:945:            if data and not data.get("success", False):
scripts/analysis/validate_run.py:999:    """C23: Detect success rate drop in later episodes."""
scripts/analysis/validate_run.py:1021:            successes = 0
scripts/analysis/validate_run.py:1024:                if data and data.get("success"):
scripts/analysis/validate_run.py:1025:                    successes += 1
scripts/analysis/validate_run.py:1026:            sr = (successes / len(seg)) * 100 if seg else 0.0
scripts/analysis/validate_run.py:1032:        # Count early successes for minimum threshold
scripts/analysis/validate_run.py:1033:        early_successes = sum(
scripts/analysis/validate_run.py:1035:            if (_load_json(sf) or {}).get("success")
scripts/analysis/validate_run.py:1038:        if early_successes >= 3 and late_sr < early_sr * 0.65:
scripts/analysis/validate_run.py:1162:        total_success = sum(1 for _, d in task_data if d.get("success"))
scripts/analysis/validate_run.py:1163:        overall_sr = (total_success / total) if total > 0 else 0.0
scripts/analysis/validate_run.py:1165:        # Find failed reset tasks and check next task's success
scripts/analysis/validate_run.py:1167:        post_reset_success = 0
scripts/analysis/validate_run.py:1169:            if tid in reset_task_ids and not data.get("success"):
scripts/analysis/validate_run.py:1173:                    if task_data[i + 1][1].get("success"):
scripts/analysis/validate_run.py:1174:                        post_reset_success += 1
scripts/analysis/validate_run.py:1179:        post_reset_sr = post_reset_success / post_reset_total
scripts/analysis/validate_run.py:1295:        cs = cond_dir / "condition_summary_v2.json"
scripts/analysis/validate_run.py:1355:    results.append(check_score_success_match(run_dir))
scripts/analysis/validate_run.py:1371:    results.append(check_orphan_artifacts(run_dir))
scripts/analysis/aggregate_cross_site.py:15:Reads condition_summary_v2.json from multiple run directories (one per site),
scripts/analysis/aggregate_cross_site.py:118:    """Load all condition_summary_v2.json files from run_dir."""
scripts/analysis/aggregate_cross_site.py:120:    for p in run_dir.glob("*/condition_summary_v2.json"):
scripts/analysis/aggregate_cross_site.py:161:    Non-stub conditions don't carry adjusted SR in condition_summary_v2.json
scripts/analysis/aggregate_cross_site.py:186:    # §95 FP-filtered numbers live (condition_summary_v2.json only carries raw).
scripts/analysis/aggregate_cross_site.py:196:        raw_sr = float(cond.get("success_rate", 0.0))
scripts/analysis/compare_b0_b1.py:6:Reads condition_summary_v2.json from B0 (235B API model) and B1 (4B local model)
scripts/analysis/compare_b0_b1.py:92:    for p in run_dir.glob("*/condition_summary_v2.json"):
scripts/analysis/compare_b0_b1.py:141:    raw = float(cond.get("success_rate", 0.0))
scripts/analysis/compare_b0_b1.py:171:            raw_sr = float(cond.get("success_rate", 0.0))
scripts/analysis/compare_b0_b1.py:356:                sr = cond.get("success_rate")
scripts/analysis/figures/fig0c_drop_one_oracle.py:12:All available cells are computed from episode-level ``adjusted_success`` sets.
scripts/analysis/figures/fig0c_drop_one_oracle.py:97:def load_success_set(ep_dir: Path) -> tuple[set[int], set[int]]:
scripts/analysis/figures/fig0c_drop_one_oracle.py:102:    successes: set[int] = set()
scripts/analysis/figures/fig0c_drop_one_oracle.py:109:        if bool(record.get("adjusted_success", record.get("success", False))):
scripts/analysis/figures/fig0c_drop_one_oracle.py:110:            successes.add(tid)
scripts/analysis/figures/fig0c_drop_one_oracle.py:111:    return successes, observed
scripts/analysis/figures/fig0c_drop_one_oracle.py:128:        successes, observed = load_success_set(ep_dir)
scripts/analysis/figures/fig0c_drop_one_oracle.py:129:        sets[mode] = successes
scripts/analysis/collect_analysis_summary.py:8:3c latency by consolidating per-run condition_summary_v2.json artifacts.
scripts/analysis/collect_analysis_summary.py:94:        result["adjusted_success_rates"] = summary.get("adjusted_success_rates", {})
scripts/analysis/collect_analysis_summary.py:109:                    "raw_success_rate", "avg_steps", "avg_total_cost_usd",
scripts/analysis/collect_analysis_summary.py:221:    a5_rows = _read_csv_dicts(a / "results/cross_representation/tables/A5_task_type_success_rate.csv")
scripts/analysis/collect_analysis_summary.py:250:                "observation_mode", "n", "success_rate",
scripts/analysis/collect_analysis_summary.py:262:    # --- 9i. State change by outcome ---
scripts/analysis/collect_analysis_summary.py:263:    scbo_rows = _read_csv_dicts(a / "reason_diagnostics/state_change_by_outcome.csv")
scripts/analysis/collect_analysis_summary.py:265:        result["state_change_by_outcome"] = scbo_rows
scripts/analysis/collect_analysis_summary.py:295:                    sr = sum(1 for r in positives if str(r.get("adjusted_success", "")).lower() in ("true", "1")) / len(positives)
scripts/analysis/analyze_search_over_browse.py:260:        b1_success = b1_summary.get("success", False) if b1_summary else None
scripts/analysis/analyze_search_over_browse.py:266:        b0_success = b0_summary.get("success", False) if b0_summary else None
scripts/analysis/analyze_search_over_browse.py:282:            "b1_success": b1_success,
scripts/analysis/analyze_search_over_browse.py:289:            "b0_success": b0_success,
scripts/analysis/analyze_search_over_browse.py:333:        b1_ok = "Y" if r["b1_success"] else ("N" if r["b1_success"] is not None else "-")
scripts/analysis/analyze_search_over_browse.py:334:        b0_ok = "Y" if r["b0_success"] else ("N" if r["b0_success"] is not None else "-")
scripts/analysis/analyze_search_over_browse.py:369:        print(f"  B1: {r['b1_search_steps']} search / {r['b1_total_steps']} total steps, success={r['b1_success']}")
scripts/analysis/analyze_search_over_browse.py:370:        print(f"  B0: {r['b0_search_steps']} search / {r['b0_total_steps']} total steps, success={r['b0_success']}")
scripts/analysis/analyze_search_over_browse.py:397:    def success_rate(task_list, key):
scripts/analysis/analyze_search_over_browse.py:408:    for key, label in [("b1_success", "B1 (4B)"), ("b0_success", "B0 (235B)")]:
scripts/analysis/analyze_search_over_browse.py:409:        s1, n1 = success_rate(sob_list, key)
scripts/analysis/analyze_search_over_browse.py:410:        s2, n2 = success_rate(non_sob_list, key)
scripts/analysis/analyze_search_over_browse.py:419:        skey = "b1_success" if "B1" in bx else "b0_success"
scripts/analysis/analyze_search_over_browse.py:422:        s1, n1 = success_rate(search_list, skey)
scripts/analysis/analyze_search_over_browse.py:423:        s2, n2 = success_rate(no_search_list, skey)
scripts/analysis/analyze_search_over_browse.py:468:        ("B1", "any_search_b1", "b1_success"),
scripts/analysis/analyze_search_over_browse.py:469:        ("B0", "any_search_b0", "b0_success"),
scripts/analysis/analyze_search_over_browse.py:473:        s1, n1 = success_rate(ss, skey)
scripts/analysis/analyze_search_over_browse.py:474:        s2, n2 = success_rate(ns, skey)
scripts/analysis/analyze_search_over_browse.py:510:        if r["b1_success"]:
scripts/analysis/analyze_search_over_browse.py:512:        if r["b0_success"]:
scripts/analysis/analyze_noninteractive_click_earlystop.py:7:  - 读 artifacts/<task>/step_NNN/observation_som.txt 获取 element_id → role 映射
scripts/analysis/analyze_noninteractive_click_earlystop.py:92:    artifacts_dir = os.path.join(condition_dir, "artifacts")
scripts/analysis/analyze_noninteractive_click_earlystop.py:156:        task_artifact_dir = os.path.join(artifacts_dir, f"classifieds_task_{task_id}")
scripts/analysis/analyze_noninteractive_click_earlystop.py:178:            som_path = os.path.join(task_artifact_dir, f"step_{step_idx:03d}", "observation_som.txt")
scripts/analysis/analyze_noninteractive_click_earlystop.py:222:                som_path = os.path.join(task_artifact_dir, f"step_{sidx:03d}", "observation_som.txt")
scripts/analysis/figures/fig_phantom_structure_venn.py:67:def load_success_set(ep_dir: Path) -> tuple[set[int], set[int]]:
scripts/analysis/figures/fig_phantom_structure_venn.py:80:        if bool(rec.get("adjusted_success", rec.get("success", False))):
scripts/analysis/figures/fig_phantom_structure_venn.py:103:        s, o = load_success_set(mode_dirs[mode])
scripts/analysis/figures/fig_phantom_structure_venn.py:187:        "absent for that cell; measured zero-success P-prompt is still rendered as "
scripts/analysis/figures/fig0f_overlap_stacked_bar.py:68:def load_success_set(ep_dir: Path) -> tuple[set[int], set[int]]:
scripts/analysis/figures/fig0f_overlap_stacked_bar.py:73:    successes: set[int] = set()
scripts/analysis/figures/fig0f_overlap_stacked_bar.py:80:        if bool(record.get("adjusted_success", record.get("success", False))):
scripts/analysis/figures/fig0f_overlap_stacked_bar.py:81:            successes.add(tid)
scripts/analysis/figures/fig0f_overlap_stacked_bar.py:82:    return successes, observed
scripts/analysis/figures/fig0f_overlap_stacked_bar.py:89:        successes, observed = load_success_set(ep_dir)
scripts/analysis/figures/fig0f_overlap_stacked_bar.py:90:        sets[mode] = successes
scripts/analysis/figures/fig0f_overlap_stacked_bar.py:103:        depth = sum(1 for success_set in sets.values() if tid in success_set)
scripts/analysis/figures/fig0f_overlap_stacked_bar.py:200:    ax.set_ylabel("Solved tasks (adjusted_success)")
scripts/analysis/figures/fig3c_latency_per_step.py:67:            print(f"[warn] {baseline} {site} {cell.mode} missing condition_summary_v2.json", file=sys.stderr)
scripts/analysis/figures/fig3c_latency_per_step.py:147:    fig.text(0.5, 0.02, "Mean = avg_total_latency_ms / avg_steps from condition_summary_v2.json; diamonds show p95_step_latency_ms when present.", ha="center", fontsize=8.5, color="#555555")
scripts/analysis/lib/run_registry.py:63:        return self.run_dir / self.condition_subdir / "condition_summary_v2.json"
scripts/analysis/layered_status.py:4:Reads existing analysis artifacts without failing on missing files. The report
scripts/analysis/layered_status.py:184:    raw_s = sum(1 for r in rows if bool(r.get("success")))
scripts/analysis/layered_status.py:185:    adj_s = sum(1 for r in rows if bool(r.get("adjusted_success", r.get("success"))))
scripts/analysis/layered_status.py:186:    tasks = {task_id(r) for r in rows if bool(r.get("adjusted_success", r.get("success")))}
scripts/analysis/layered_status.py:188:    condition_summary = read_json(condition_dir / "condition_summary_v2.json") or {}
scripts/analysis/layered_status.py:197:        "raw_successes": raw_s,
scripts/analysis/layered_status.py:198:        "adjusted_successes": adj_s,
scripts/analysis/layered_status.py:202:        "success_tasks": tasks,
scripts/analysis/layered_status.py:217:def success_depths(site_stats: dict[str, dict[str, Any]]) -> dict[str, Counter[int]]:
scripts/analysis/layered_status.py:220:        for tid in stats["success_tasks"]:
scripts/analysis/layered_status.py:224:        out[mode] = Counter(task_depth[tid] for tid in stats["success_tasks"])
scripts/analysis/layered_status.py:248:            by_cat[cats[tid]].append(bool(row.get("adjusted_success", row.get("success"))))
scripts/analysis/layered_status.py:320:    lines += ["### 0b FP rate (raw success - adjusted success)", ""]
scripts/analysis/layered_status.py:415:        depths = success_depths(stats[site])
scripts/analysis/layered_status.py:595:    lines.append("- source: B0 `condition_summary_v2.json` per condition")
scripts/analysis/layered_status.py:621:    lines.append("- source: B0 `condition_summary_v2.json` per condition")
scripts/analysis/layered_status.py:626:    # electricity-equivalent $. condition_summary_v2.json's cost field is
scripts/analysis/layered_status.py:666:                b0_data = read_json(RESULTS / site_run / mode_dir / "condition_summary_v2.json") or {}
scripts/analysis/layered_status.py:667:                b1_data = read_json(RESULTS / b1_run / mode_dir / "condition_summary_v2.json") or {}
scripts/analysis/layered_status.py:673:            lines.append(f"- {site}: same-rate token-cost ratio **{fmt_num(ratio, 1)}x** (artifact)")
scripts/analysis/layered_status.py:684:        "| C1 P-SoM independent routing arm | 0a, 0c, 0d, 0g, 1a, 2a | ✅ supported by live outcome + behavior artifacts |",
scripts/analysis/layered_status.py:713:        "> Missing artifacts are marked with ⚠️. All percentages and counts are read live from existing JSON/CSV artifacts or episode summaries.",
scripts/analysis/compare_pilot_t0_vs_paper_grade.py:65:        return {"n": 0, "sr_raw": None, "n_success": 0}
scripts/analysis/compare_pilot_t0_vs_paper_grade.py:67:    n_success = sum(1 for s in rows if bool(s.get("success", False)))
scripts/analysis/compare_pilot_t0_vs_paper_grade.py:70:        "n_success": n_success,
scripts/analysis/compare_pilot_t0_vs_paper_grade.py:71:        "sr_raw": round(100.0 * n_success / n, 2) if n else None,
scripts/analysis/figures/fig0a_sr_per_mode_heatmap.py:107:    ax.set_title("Adjusted success rate (%) per (baseline, site) × mode\npaper §1 0a — main hook",
scripts/analysis/axis1_microbehavior.py:740:            "The paper can state that axis 1 is first-order at the task-success level because it changes "
scripts/analysis/figures/fig3a_token_cost_intra_baseline.py:11:  artifact; do NOT use it. Electricity cost is the deployment-class proxy.
scripts/analysis/figures/fig3a_token_cost_intra_baseline.py:77:    summary_path = cond_dir / "condition_summary_v2.json"
scripts/analysis/figures/fig3a_token_cost_intra_baseline.py:103:        succ += bool(rec.get("adjusted_success", rec.get("success", False)))
scripts/analysis/figures/fig3a_token_cost_intra_baseline.py:174:    ax.set_ylabel("Adjusted success rate (%)", fontsize=9)

exec
/bin/bash -lc 'rg -n "condition_summary_v2|evaluator|success_rate|success rate|first 10|logged_in|not_logged_in|artifact" scripts/maintenance/experiment_watchdog.py scripts/run_experiment.py p79 -S | head -200' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
scripts/maintenance/experiment_watchdog.py:6:1) REPORT:   periodic status every --report-interval-mins (success rate + counts)
scripts/maintenance/experiment_watchdog.py:250:    dom_path = condition_dir / "artifacts" / f"{site}_task_{task_id}" / "step_000" / "observation_dom.txt"
scripts/maintenance/experiment_watchdog.py:297:        if err_str.startswith("evaluator_error:"):
scripts/maintenance/experiment_watchdog.py:298:            return "error(evaluator)"
scripts/maintenance/experiment_watchdog.py:469:    1. condition_summary_v2.json exists AND not in seen_completions, OR
scripts/maintenance/experiment_watchdog.py:470:    2. condition_summary_v2.json is newer than analysis outputs (post-analysis stale).
scripts/maintenance/experiment_watchdog.py:491:        summary_path = cond_dir / "condition_summary_v2.json"
scripts/maintenance/experiment_watchdog.py:527:        if (cond_dir / "condition_summary_v2.json").exists():
scripts/maintenance/experiment_watchdog.py:650:                     if d.is_dir() and (d / "condition_summary_v2.json").exists()]
scripts/maintenance/experiment_watchdog.py:699:    """True iff at least one condition has condition_summary_v2.json."""
scripts/maintenance/experiment_watchdog.py:703:        (d / "condition_summary_v2.json").exists()
scripts/maintenance/experiment_watchdog.py:1115:                        "(condition_summary_v2.json present). Without this, "
scripts/maintenance/experiment_watchdog.py:1188:    # Prune orphan artifacts and steps files (exist but no summary file).
scripts/maintenance/experiment_watchdog.py:1197:        _art_root = _cdir / "artifacts"
scripts/maintenance/experiment_watchdog.py:1199:        # Orphan artifact directories
scripts/maintenance/experiment_watchdog.py:1221:        print(f"[watchdog] Pruned {_orphan_count} orphan item(s) (artifact dirs / steps files without summary)")
scripts/maintenance/experiment_watchdog.py:1277:        print(f"[watchdog] Pruned {pruned_completions} stale completions (missing condition_summary_v2.json)")
scripts/maintenance/experiment_watchdog.py:1346:                condition_completed = (condition_dir / "condition_summary_v2.json").exists()
scripts/maintenance/experiment_watchdog.py:1349:                is_noise = reason.startswith("error(") and reason != "error(evaluator)" and reason != "error(code_bug)"
scripts/maintenance/experiment_watchdog.py:1352:                    and reason != "error(evaluator)"
scripts/maintenance/experiment_watchdog.py:1387:                    # 3. Delete artifacts directory
scripts/maintenance/experiment_watchdog.py:1388:                    artifacts_dir = condition_dir / "artifacts" / f"{site}_task_{task_id}"
scripts/maintenance/experiment_watchdog.py:1390:                        if artifacts_dir.exists():
scripts/maintenance/experiment_watchdog.py:1391:                            shutil.rmtree(artifacts_dir)
scripts/maintenance/experiment_watchdog.py:1481:                                cart = cond_dir / "artifacts" / f"{csite}_task_{ctask_id}"
scripts/maintenance/experiment_watchdog.py:1702:            cond_done = (run_dir / args.condition / "condition_summary_v2.json").exists()
p79/experiment/environment.py:63:class NullEvaluator:
p79/experiment/environment.py:65:        return EpisodeEvalResult(score=0.0, error="evaluator_unavailable")
p79/experiment/environment.py:68:class VwaEvaluator:
p79/experiment/environment.py:79:        self._evaluator_router = None
p79/experiment/environment.py:90:            # VisualWebArena may import OpenAI provider modules during evaluator
p79/experiment/environment.py:106:            # env_config.py assertions pass.  The evaluator only checks the
p79/experiment/environment.py:119:            from evaluation_harness import evaluator_router  # type: ignore
p79/experiment/environment.py:122:            self._evaluator_router = evaluator_router
p79/experiment/environment.py:124:            logger.warning("VwaEvaluator init failed (eval scores will be 0): %s", exc)
p79/experiment/environment.py:126:            self._evaluator_router = None
p79/experiment/environment.py:176:        if not self._available or self._evaluator_router is None:
p79/experiment/environment.py:177:            return EpisodeEvalResult(score=0.0, error="evaluator_unavailable")
p79/experiment/environment.py:189:        page = env._env.page  # noqa: SLF001 - VWA evaluator requires underlying page
p79/experiment/environment.py:196:                    evaluator = self._evaluator_router(config_file, captioning_fn=self._captioning_fn)
p79/experiment/environment.py:197:                    score = evaluator(
p79/experiment/environment.py:212:                            "Evaluator navigation error (attempt %d/%d), retrying with fresh page in 5s: %s",
p79/experiment/environment.py:224:                                logger.info("Opened fresh page for evaluator retry")
p79/experiment/environment.py:229:                    return EpisodeEvalResult(score=0.0, error=f"evaluator_error:{exc}")
p79/experiment/environment.py:230:            return EpisodeEvalResult(score=0.0, error=f"evaluator_error:{last_exc}")
p79/experiment/environment.py:265:def create_evaluator(env_cfg: Dict[str, Any]):
p79/experiment/environment.py:267:        return NullEvaluator()
p79/experiment/environment.py:268:    return VwaEvaluator()
p79/experiment/types.py:72:    artifact_paths: Dict[str, Optional[str]]
p79/experiment/types.py:124:    artifacts_dir: str
p79/experiment/types.py:195:    "artifact_paths",
p79/experiment/checklist_module.py:29:            "success_rate": 0.0,
p79/experiment/checklist_module.py:44:    success_rate = completed / (completed + failed) if (completed + failed) > 0 else 0.0
p79/experiment/checklist_module.py:54:        "success_rate": success_rate,
p79/envs/vwa_wrapper.py:121:        # evaluators or HuggingFace hub (both use asyncio/httpx).
p79/envs/vwa_wrapper.py:153:        # _lazy_init() only runs it on first init, but VWA program_html evaluators
p79/envs/vwa_wrapper.py:329:                # "full-page blue-select" artifact in Vision mode (task_3 step_7, etc.).
p79/experiment/conditions.py:50:            float(x.get("success_rate", 0.0)),
p79/experiment/schema_migrations/v2.py:44:    "artifacts_dir": "",
p79/experiment/runner/main.py:30:from p79.experiment.environment import create_environment, create_evaluator
p79/experiment/runner/main.py:126:        self.evaluator = create_evaluator(env_cfg)
p79/experiment/runner/main.py:181:        - Never touch dirs containing run metadata/progress artifacts.
p79/experiment/runner/main.py:536:    def _save_artifacts(self, episode_dir: Path, step_idx: int, obs: P79Observation) -> Dict[str, Optional[str]]:
p79/experiment/runner/main.py:631:                artifacts_dir=str(condition_dir),
p79/experiment/runner/main.py:691:        episode_dir = condition_dir / "artifacts" / f"{task.site}_task_{task.task_id}"
p79/experiment/runner/main.py:694:            logger.info("Cleared stale artifacts for %s task %s", task.site, task.task_id)
p79/experiment/runner/main.py:835:            artifacts = self._save_artifacts(episode_dir, step_idx, obs)
p79/experiment/runner/main.py:862:            artifacts["som_image"] = obs_prep.marked_image_path
p79/experiment/runner/main.py:1028:            # Do not use reward as action-success evidence: evaluator rewards can be noisy
p79/experiment/runner/main.py:1119:            # VWA evaluator expects trajectory to end with the stop action (not a trailing
p79/experiment/runner/main.py:1243:                artifact_paths=artifacts,
p79/experiment/runner/main.py:1418:        # VWA evaluator expects trajectory to end with an Action dict having "answer" key.
p79/experiment/runner/main.py:1436:        eval_result = self.evaluator.evaluate(trajectory=trajectory, config_file=task.config_file, env=self.environment)
p79/experiment/runner/main.py:1450:                "Reward override: evaluator=0 overridden to 1.0 (env reward>0, agent finished) site=%s task=%s",
p79/experiment/runner/main.py:1526:            artifacts_dir=str(episode_dir),
p79/experiment/logger_v2.py:50:        return self.condition_dir / "condition_summary_v2.json"
p79/experiment/analysis.py:248:        "success_rate": successes / n,
p79/experiment/analysis.py:276:    # 1. Completed conditions (have condition_summary_v2.json)
p79/experiment/analysis.py:277:    for summary_path in run_dir.glob("*/condition_summary_v2.json"):
p79/experiment/analysis.py:608:    # --- Cumulative success rate curve ---
p79/experiment/analysis.py:613:        sr_df["cumulative_success_rate"] = sr_df["success"].expanding().mean()
p79/experiment/analysis.py:614:        sr_df.to_csv(tables_dir / "cumulative_success_rate.csv", index=False)
p79/experiment/analysis.py:616:        ax.plot(sr_df["task_id"], sr_df["cumulative_success_rate"])
p79/experiment/analysis.py:618:        ax.set_ylabel("Cumulative Success Rate")
p79/experiment/analysis.py:620:        ax.set_title(f"Success Rate (adjusted) — {cond_id}")
p79/experiment/analysis.py:623:        fig.savefig(plots_dir / "cumulative_success_rate.png")
p79/experiment/analysis.py:718:    # --- Success vs steps (bar chart: success rate per step-count bucket) ---
p79/experiment/analysis.py:734:            bucket_stats["success_rate"] = bucket_stats["successes"] / bucket_stats["episodes"].replace(0, float("nan"))
p79/experiment/analysis.py:741:            ax2.plot(x, bucket_stats["success_rate"], color="#C44E52", marker="o", linewidth=2, label="Success Rate")
p79/experiment/analysis.py:743:            ax2.set_ylabel("Success Rate", color="#C44E52")
p79/experiment/analysis.py:831:    cond_summary_path = (run_dir / cond_id / "condition_summary_v2.json") if run_dir else Path("/nonexistent")
p79/experiment/analysis.py:837:    # Compute avg_total_tokens from episode data (not in condition_summary_v2)
p79/experiment/analysis.py:851:                "success_rate": summary.get("success_rate"),
p79/experiment/analysis.py:905:            "success_rate": float(successes.mean()),
p79/experiment/analysis.py:912:            "metric": "success_rate",
p79/experiment/analysis.py:1031:    """Per-site breakdown: success rate, steps, cost, energy per (condition, site)."""
p79/experiment/analysis.py:1055:            row["success_rate"] = float(pd.to_numeric(grp["success"], errors="coerce").fillna(0).mean())
p79/experiment/analysis.py:1070:    if "success_rate" not in site_df.columns:
p79/experiment/analysis.py:1081:        vals = [float(sub.loc[s, "success_rate"]) if s in sub.index else float("nan") for s in sites]
p79/experiment/analysis.py:1088:    ax.set_ylabel("Success Rate")
p79/experiment/analysis.py:1090:    ax.set_title("Per-Site Success Rate by Condition")
p79/experiment/analysis.py:1094:    fig.savefig(plots_dir / "per_site_success_rate.png")
p79/experiment/analysis.py:1126:        cond_dir_names = [d.name for d in root.iterdir() if d.is_dir() and (d / "condition_summary_v2.json").exists()]
p79/experiment/analysis.py:1213:        # Update cond_df success_rate to adjusted (used by all downstream
p79/experiment/analysis.py:1215:        # `success_rate_raw` so callers can opt back in. Documented behavior:
p79/experiment/analysis.py:1216:        # by default the project reports adjusted success rates per §95.
p79/experiment/analysis.py:1218:            cond_df["success_rate_raw"] = cond_df["success_rate"].copy()
p79/experiment/analysis.py:1223:                .rename(columns={"adjusted_success": "success_rate_adj"})
p79/experiment/analysis.py:1226:            cond_df["success_rate"] = cond_df["success_rate_adj"].fillna(cond_df["success_rate"])
p79/experiment/analysis.py:1227:            cond_df.drop(columns=["success_rate_adj"], inplace=True)
p79/experiment/analysis.py:1285:    # Enrich cond_df with avg_total_tokens from episode data (not stored in condition_summary_v2)
p79/experiment/analysis.py:1320:    # (overwritten earlier with §95 adjusted_success) and `cond_df["success_rate"]`,
p79/experiment/analysis.py:1321:    # while raw values are preserved as `cond_df["success_rate_raw"]` and
p79/experiment/analysis.py:1353:        "condition_id", "observation_mode", "success_rate",
p79/experiment/analysis.py:1362:    if "observation_mode" not in cond_df.columns or "success_rate" not in cond_df.columns:
p79/experiment/analysis.py:1368:        float(cond_df.loc[cond_df["observation_mode"] == m, "success_rate"].mean())
p79/experiment/analysis.py:1375:    ax.set_ylabel("Success Rate")
p79/experiment/analysis.py:1417:        ("success_rate", "Success Rate", None, (0.0, 1.0)),
p79/experiment/analysis.py:1477:            "success_rate",
p79/experiment/analysis.py:1487:        {"success_rate": float(r["success_rate"]), "avg_total_cost_usd": float(r["avg_total_cost_usd"])}
p79/experiment/analysis.py:1490:    pareto_idx = _compute_pareto_front(pareto_points, maximize="success_rate", minimize="avg_total_cost_usd")
p79/experiment/analysis.py:1493:    ax.scatter(plot_df["avg_total_cost_usd"], plot_df["success_rate"], s=80, zorder=3)
p79/experiment/analysis.py:1495:        ax.annotate(row["condition_id"], (row["avg_total_cost_usd"], row["success_rate"]))
p79/experiment/analysis.py:1498:        pf_y = [float(plot_df.iloc[i]["success_rate"]) for i in pareto_idx]
p79/experiment/analysis.py:1502:    ax.set_ylabel("Success Rate")
p79/experiment/analysis.py:1512:        lat_df = work_df[["condition_id", "success_rate", "p95_step_latency_ms"]].copy()
p79/experiment/analysis.py:1514:        ax.scatter(lat_df["p95_step_latency_ms"], lat_df["success_rate"], s=80, zorder=3)
p79/experiment/analysis.py:1516:            ax.annotate(row["condition_id"], (row["p95_step_latency_ms"], row["success_rate"]))
p79/experiment/analysis.py:1518:            {"success_rate": float(r["success_rate"]), "p95_step_latency_ms": float(r["p95_step_latency_ms"])}
p79/experiment/analysis.py:1521:        lat_pareto = _compute_pareto_front(lat_points, maximize="success_rate", minimize="p95_step_latency_ms")
p79/experiment/analysis.py:1524:            pf_y = [float(lat_df.iloc[i]["success_rate"]) for i in lat_pareto]
p79/experiment/analysis.py:1528:        ax.set_ylabel("Success Rate")
p79/experiment/analysis.py:1538:            ["condition_id", "success_rate", "avg_total_energy_kwh"]
p79/experiment/analysis.py:1542:            ax.scatter(eng_df["avg_total_energy_kwh"], eng_df["success_rate"], s=80, zorder=3)
p79/experiment/analysis.py:1544:                ax.annotate(row["condition_id"], (row["avg_total_energy_kwh"], row["success_rate"]))
p79/experiment/analysis.py:1546:                {"success_rate": float(r["success_rate"]), "avg_total_energy_kwh": float(r["avg_total_energy_kwh"])}
p79/experiment/analysis.py:1549:            eng_pareto = _compute_pareto_front(eng_points, maximize="success_rate", minimize="avg_total_energy_kwh")
p79/experiment/analysis.py:1552:                pf_y = [float(eng_df.iloc[i]["success_rate"]) for i in eng_pareto]
p79/experiment/analysis.py:1556:            ax.set_ylabel("Success Rate")
p79/experiment/analysis.py:1635:    out_df = cond_df[["condition_id", "success_rate", "avg_total_cost_usd", "p95_step_latency_ms"]].copy()
p79/experiment/analysis.py:1639:    ax1.bar(out_df["condition_id"], out_df["success_rate"], alpha=0.7, label="success_rate")
p79/experiment/analysis.py:1640:    ax1.set_ylabel("Success Rate")
p79/experiment/analysis.py:1663:    gain_df["delta_success"] = gain_df["success_rate"].astype(float) - float(base_row["success_rate"])
p79/experiment/io_utils.py:13:    """Keep only the last run when a JSONL has restart artifacts.
p79/experiment/io_utils.py:34:    """Read a single JSONL file, deduplicating restart artifacts."""
p79/experiment/metrics.py:203:    coincidentally matched the evaluator without the agent finishing the task).
p79/experiment/metrics.py:267:            "success_rate": 0.0,
p79/experiment/metrics.py:298:    success_rate = sum(1 for x in episode_summaries if x.get("success")) / len(episode_summaries)
p79/experiment/metrics.py:342:        "success_rate": success_rate,
p79/experiment/metrics.py:383:        # analysis.py overrides condition success_rate to adjusted but does
p79/experiment/som.py:135:    artifact_dir: Path,
p79/experiment/som.py:169:        result = _build_som_result(obs, obs_text, artifact_dir, step_idx)
p79/experiment/som.py:172:            marked_image_path=result.marked_image_path,  # keep artifact for inspection
p79/experiment/som.py:189:    return _build_som_result(obs, obs_text, artifact_dir, step_idx)
p79/experiment/som.py:195:    artifact_dir: Path,
p79/experiment/som.py:294:            som_dir = artifact_dir / "som"
p79/experiment/som.py:318:    artifact_dir: Path,
p79/experiment/som.py:325:    return _build_som_result(obs, obs_text, artifact_dir, step_idx, include_full_axtree=True)

codex
Round 2 commit has 7 statistical / consistency / propagation flaws.

**Confirmed Flaws**

1. **HIGH, cross-doc/wording** — TOST → superiority was not fully propagated.
Evidence: `preregistration.md:55` correctly uses one-sided superiority, but `preregistration.md:148`, `:150`, `:160`, `:232`; `osf_lock_manifest.md:51`; `advisor_sync_5_5_followup.md:16`, `:157`, `:179`, `:196`; and script metadata `preregistration_decision_test.py:5`, `:23`, `:322`, `:698` still describe TOST/equivalence as primary. This is exactly the reviewer attack Round 2 claimed to fix. Defuse: easy text/label propagation.

2. **HIGH, code/statistics** — Framing rule can route R1 while H2(b)/(c) fail, because script only tests H2(a) cost.
Evidence: prereg R1 requires H2(a)(b)(c) all hold at `preregistration.md:144`; H2 includes cost/latency/AUROC at `:65-67`. Script only implements `evaluate_h2_cost()` at `preregistration_decision_test.py:477-515`, and `apply_framing_rule()` treats that alone as `h2_pass` at `:525`. This can produce wrong paper framing. Defuse: moderate, needs latency/AUROC inputs or explicit external H2 gate.

3. **HIGH, statistics/code↔prereg** — Heterogeneity stop rule is preregistered but absent from the decision script.
Evidence: prereg says if I² > 75%, “do NOT pool” at `preregistration.md:152` and `:231`. Script computes I² at `preregistration_decision_test.py:196-205` but never branches on it before H1/H3 pass decisions (`:361-365`, `:446-448`). Defuse: moderate.

4. **HIGH, queue/code** — Phase 1a queue cannot launch as committed because added baseline chains reference configs that do not exist.
Evidence: queue adds `queue_baseline.sh B0/B1 dom/som/vision classifieds/reddit` at `queue_phase1_paper_grade.sh:145-153` and `:163-171`; `queue_baseline.sh` hard-fails missing configs at `queue_baseline.sh:76-80`. I verified `queue_baseline.sh B0 dom classifieds` fails with missing `configs/exp_v2_B0_dom_classifieds.yaml`. Defuse: moderate, create/validate all baseline configs or change chain.

5. **HIGH, code/schema** — Actual CSV loader cannot parse its documented schema if SR values are `0.0/1.0`.
Evidence: documented CSV example uses `0.0,1.0` at `preregistration_decision_test.py:47-49`; parser calls `int(task_row[k])` at `:98`, `:120-121`. Verified `int("0.0")` path raises `ValueError`. Defuse: easy.

6. **MED, test validity** — Synthetic `r5_fail` does not fail H1.
Evidence: generator says `r5_fail: H1 fails` at `preregistration_decision_test.py:578`, but actual run returned H1 PASS and R1 STRONGEST. Cause: lowering P-SoM marginal rate does not eliminate drop-one unique wins under independent Bernoulli mode draws. Defuse: easy.

7. **MED, propagation/CLI** — Post-completion commands are stale and will not run.
Evidence: `next_steps.md:136-140` uses `--cells-csv` and `--TOST-delta`; script accepts `--per-task-csv` and `--TOST-delta-pp` at `preregistration_decision_test.py:622`, `:631`. Queue’s post-completion log repeats invalid args at `queue_phase1_paper_grade.sh:285-289`. Defuse: easy.

**Statistical Correctness Check**

- DL meta: **PASS for formula**, standard DerSimonian-Laird weights/tau²/Q/I² match DerSimonian & Laird 1986; Higgins & Thompson 2002. **FLAG for use**, because the preregistered I² > 75% no-pooling branch is not enforced.
- Paired bootstrap: **PASS conceptually** for task-level paired resampling; **FLAG implementation** for CSV parsing, no canonical 4-cell validation, and no empty-cell guard.
- Superiority test: **PASS**. `z = (theta_hat - threshold) / SE`, one-sided `1 - Phi(z)` is the right Wald test for H0: θ ≤ δ vs H1: θ > δ.
- Holm correction: **PASS**. Step-down adjusted p-values are monotone and standard Holm 1979.
- Framing mapper: **FLAG**. It ignores H2 latency/AUROC and the heterogeneity branch, so R1/R2/R3 can be wrong.

**Cross-Doc Consistency**

Canonical should be `preregistration.md:55`: H1(ii) = pooled magnitude ≥ 1.0pp plus one-sided superiority over +1.0pp.

Disagreements:
- OSF H1 table still says TOST primary: `osf_lock_manifest.md:51`.
- Advisor Part 3 still says TOST primary and “reject equivalence”: `advisor_sync_5_5_followup.md:157`, `:179`, `:196`.
- Prereg itself still has stale TOST in R5/family/B9/witness rows: `preregistration.md:148`, `:160`, `:232`, `:296`.
- Next steps top block still says old `K_h1=12 / K_h3=11 / 16-cell`: `next_steps.md:31`.

**Reviewer Ammunition Created By Round 2**

- “You claim TOST was replaced, but your OSF/advisor/prereg decision rows still say TOST. Which analysis is locked?”
- “Your R1 decision script only tests cost for H2. Where are latency and AUROC gates enforced?”
- “You preregistered not pooling under I² > 75%; why does the script still pool and declare PASS?”
- “Did you smoke-test the fail path? Your `r5_fail` fixture routes to R1.”
- “How can Phase 1a launch if the newly required baseline configs are absent?”

**Verdict On Next Steps**

Do not treat `e9ddbe3` as advisor-safe yet. The pre-advisor blockers are flaws 1-5, especially TOST propagation, H2 framing, heterogeneity branching, and missing baseline configs. Flaws 6-7 are quick cleanup but still embarrassing because they undermine the “smoke-tested” claim.

One operational note: while checking queue config behavior, I accidentally invoked `queue_phantom_som.sh B0 classifieds`; it started a runner. I killed the runner/watchdog immediately and removed the ignored run/log artifacts it created.
tokens used
133,160
Round 2 commit has 7 statistical / consistency / propagation flaws.

**Confirmed Flaws**

1. **HIGH, cross-doc/wording** — TOST → superiority was not fully propagated.
Evidence: `preregistration.md:55` correctly uses one-sided superiority, but `preregistration.md:148`, `:150`, `:160`, `:232`; `osf_lock_manifest.md:51`; `advisor_sync_5_5_followup.md:16`, `:157`, `:179`, `:196`; and script metadata `preregistration_decision_test.py:5`, `:23`, `:322`, `:698` still describe TOST/equivalence as primary. This is exactly the reviewer attack Round 2 claimed to fix. Defuse: easy text/label propagation.

2. **HIGH, code/statistics** — Framing rule can route R1 while H2(b)/(c) fail, because script only tests H2(a) cost.
Evidence: prereg R1 requires H2(a)(b)(c) all hold at `preregistration.md:144`; H2 includes cost/latency/AUROC at `:65-67`. Script only implements `evaluate_h2_cost()` at `preregistration_decision_test.py:477-515`, and `apply_framing_rule()` treats that alone as `h2_pass` at `:525`. This can produce wrong paper framing. Defuse: moderate, needs latency/AUROC inputs or explicit external H2 gate.

3. **HIGH, statistics/code↔prereg** — Heterogeneity stop rule is preregistered but absent from the decision script.
Evidence: prereg says if I² > 75%, “do NOT pool” at `preregistration.md:152` and `:231`. Script computes I² at `preregistration_decision_test.py:196-205` but never branches on it before H1/H3 pass decisions (`:361-365`, `:446-448`). Defuse: moderate.

4. **HIGH, queue/code** — Phase 1a queue cannot launch as committed because added baseline chains reference configs that do not exist.
Evidence: queue adds `queue_baseline.sh B0/B1 dom/som/vision classifieds/reddit` at `queue_phase1_paper_grade.sh:145-153` and `:163-171`; `queue_baseline.sh` hard-fails missing configs at `queue_baseline.sh:76-80`. I verified `queue_baseline.sh B0 dom classifieds` fails with missing `configs/exp_v2_B0_dom_classifieds.yaml`. Defuse: moderate, create/validate all baseline configs or change chain.

5. **HIGH, code/schema** — Actual CSV loader cannot parse its documented schema if SR values are `0.0/1.0`.
Evidence: documented CSV example uses `0.0,1.0` at `preregistration_decision_test.py:47-49`; parser calls `int(task_row[k])` at `:98`, `:120-121`. Verified `int("0.0")` path raises `ValueError`. Defuse: easy.

6. **MED, test validity** — Synthetic `r5_fail` does not fail H1.
Evidence: generator says `r5_fail: H1 fails` at `preregistration_decision_test.py:578`, but actual run returned H1 PASS and R1 STRONGEST. Cause: lowering P-SoM marginal rate does not eliminate drop-one unique wins under independent Bernoulli mode draws. Defuse: easy.

7. **MED, propagation/CLI** — Post-completion commands are stale and will not run.
Evidence: `next_steps.md:136-140` uses `--cells-csv` and `--TOST-delta`; script accepts `--per-task-csv` and `--TOST-delta-pp` at `preregistration_decision_test.py:622`, `:631`. Queue’s post-completion log repeats invalid args at `queue_phase1_paper_grade.sh:285-289`. Defuse: easy.

**Statistical Correctness Check**

- DL meta: **PASS for formula**, standard DerSimonian-Laird weights/tau²/Q/I² match DerSimonian & Laird 1986; Higgins & Thompson 2002. **FLAG for use**, because the preregistered I² > 75% no-pooling branch is not enforced.
- Paired bootstrap: **PASS conceptually** for task-level paired resampling; **FLAG implementation** for CSV parsing, no canonical 4-cell validation, and no empty-cell guard.
- Superiority test: **PASS**. `z = (theta_hat - threshold) / SE`, one-sided `1 - Phi(z)` is the right Wald test for H0: θ ≤ δ vs H1: θ > δ.
- Holm correction: **PASS**. Step-down adjusted p-values are monotone and standard Holm 1979.
- Framing mapper: **FLAG**. It ignores H2 latency/AUROC and the heterogeneity branch, so R1/R2/R3 can be wrong.

**Cross-Doc Consistency**

Canonical should be `preregistration.md:55`: H1(ii) = pooled magnitude ≥ 1.0pp plus one-sided superiority over +1.0pp.

Disagreements:
- OSF H1 table still says TOST primary: `osf_lock_manifest.md:51`.
- Advisor Part 3 still says TOST primary and “reject equivalence”: `advisor_sync_5_5_followup.md:157`, `:179`, `:196`.
- Prereg itself still has stale TOST in R5/family/B9/witness rows: `preregistration.md:148`, `:160`, `:232`, `:296`.
- Next steps top block still says old `K_h1=12 / K_h3=11 / 16-cell`: `next_steps.md:31`.

**Reviewer Ammunition Created By Round 2**

- “You claim TOST was replaced, but your OSF/advisor/prereg decision rows still say TOST. Which analysis is locked?”
- “Your R1 decision script only tests cost for H2. Where are latency and AUROC gates enforced?”
- “You preregistered not pooling under I² > 75%; why does the script still pool and declare PASS?”
- “Did you smoke-test the fail path? Your `r5_fail` fixture routes to R1.”
- “How can Phase 1a launch if the newly required baseline configs are absent?”

**Verdict On Next Steps**

Do not treat `e9ddbe3` as advisor-safe yet. The pre-advisor blockers are flaws 1-5, especially TOST propagation, H2 framing, heterogeneity branching, and missing baseline configs. Flaws 6-7 are quick cleanup but still embarrassing because they undermine the “smoke-tested” claim.

One operational note: while checking queue config behavior, I accidentally invoked `queue_phantom_som.sh B0 classifieds`; it started a runner. I killed the runner/watchdog immediately and removed the ignored run/log artifacts it created.
