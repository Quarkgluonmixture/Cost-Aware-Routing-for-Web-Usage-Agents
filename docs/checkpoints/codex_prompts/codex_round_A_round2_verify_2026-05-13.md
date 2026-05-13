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
