# paper-deslop: improvements found by running it end-to-end on a real manuscript

Context for whoever picks this up: on 2026-08-05 the pipeline was run over a
13-file LaTeX manuscript (REALM 2026 submission, ~40 pages incl. appendix).
Result: **Vale errors 95 → 13, `invariant_check.py` PASS on all 13 files.**
The tool worked. Everything below is a defect or a gap found *while it was
working*, ordered by how much it costs a user. Each item says how to
reproduce it, so please confirm the behaviour before changing anything —
some of these may be config choices rather than bugs.

---

## P0-1. A file with one legitimate exception can never enter the ratchet

**The ratchet is the whole point of the tool** (`deslopped.txt`: "the blocking
set tightens by itself"). But entry requires zero error-level alerts, and
there is no way to mark an alert as intentional.

Two of the 13 files hit this:

- `1_intro.tex` — one `Paper.EmDash` **inside a LaTeX comment**
  (`% ... --- they would let it drift`). It cannot be edited, because
  `invariant_check.py` hashes comments and would fail. So Vale and the
  invariant checker of the *same pipeline* give contradictory instructions,
  and the file is locked out of the blocking set forever.
- `appendix_pb.tex` — 11 `Paper.EmDash` **inside tabular cells**
  (`1 (DOM) --- \textbf{no}`, a column separator, not prose) plus one
  `ai-tells.AffirmativeFormulas` on `the point is not one a deployment could
  occupy`, where the `not X` half is exactly the clause that bounds the
  claim. SKILL.md explicitly says to leave that kind of alert. Following the
  skill therefore guarantees the file can never be protected.

**Reproduce:**

```bash
printf 'A sentence with an em dash --- like this.\n\n%% vale Paper.EmDash = NO\nAnother --- suppressed?\n' > /tmp/t.tex
vale --config=.vale.ini /tmp/t.tex     # both lines still error
```

Vale's suppression comment does not take effect in `.tex` under this config.

**What to decide** (pick one, do not do all three):

1. Make Vale's `.tex` comment syntax a recognised suppression marker, and
   document it in SKILL.md next to the "invariant wins" paragraph, with the
   rule that a suppression must carry a reason on the same line.
2. Let `deslopped.txt` entries carry an allowance:
   `sections/appendix_pb.tex  allow=Paper.EmDash:11` — the file blocks on
   *new* alerts and on any change to the allowed count. This keeps the
   ratchet monotone, which the current design cares about.
3. Fix the scoping so the alerts never fire (see P0-2 and P0-3); then no
   escape hatch is needed for these two cases, though it would still be
   needed for the AffirmativeFormulas one.

Option 2 is the smallest change that makes the ratchet reachable for a real
manuscript, and it keeps the exception visible in a reviewed file rather than
scattered through the prose.

---

## P0-2. Vale lints LaTeX comments

`% b/h --- they would let it drift` produces an error. Comments are invisible
in the PDF, are protected by `invariant_check.py` (invariant 6: "edit prose
only"), and cannot be fixed without failing the gate.

**Reproduce:** any `.tex` file with `---` inside a `%` comment.

**Fix:** exclude `%`-comments in the LaTeX scoping. Note that a naive rule
breaks on `\%` (escaped percent inside prose) and on `%` at end of line used
to suppress a space, so the pattern needs to be comment-aware, not
`%`-to-EOL-aware.

---

## P0-3. Vale lints tabular cell content as prose

11 of the 13 surviving errors are `---` inside `tabular` rows, used as a
within-cell separator. Table cells are data presentation; the house style
rule about em dashes is about prose rhythm.

**Reproduce:** a `tabular` row containing `A --- B`.

**Fix:** treat `tabular`/`table`/`tabularx`/`longtable` bodies the way math
and verbatim are already treated. `invariant_check.py` already has the
protected-block machinery; the Vale scoping should agree with it. Worth
checking whether the two layers share a definition of "protected" at all — if
they do not, that divergence is itself the bug, and P0-2 and P0-3 are two
symptoms of it.

---

## P1-4. The shipped `terms.txt` is another paper's term list

`terms.txt` ships with 9 terms, of which **7 never occur** in a real
manuscript:

```
$ python3 scripts/invariant_check.py draft.tex draft.tex --term-audit
       0      0  cost-accuracy trade-off   <- never occurs (typo, or drop it)
       0      0  multimodal web agent      <- never occurs
       0      0  selective classification  <- never occurs
       ... 7 of 9
```

The audit works and says the right thing. The problem is that the file *looks
curated*, so the natural move is to trust it and start rewriting — at which
point the whitelist is protecting terms the paper does not contain and not
protecting the ones it does. The failure is silent: `invariant_check` passes
because zero occurrences cannot drift.

**Fix:** ship `terms.txt` containing only comments and an empty list, and
make `invariant_check.py` warn loudly when the whitelist has more
never-occurring entries than occurring ones. SKILL.md step 2 already says to
propose a list when the file "does not exist" — extend that to "does not
exist **or does not match the draft**".

---

## P1-5. `gen_vale_vocab.py` is a manual step with no staleness check

README says to run it "after every `terms.txt` edit". Nothing detects that it
was not run, and the symptom (spurious spelling alerts on domain terms) looks
like a prose problem rather than a stale-cache problem.

**Fix:** have `ratchet_lint.sh` compare the mtime or a hash of `terms.txt`
against the generated vocab and fail with a one-line instruction, or just
regenerate it at the start of the lint.

---

## P2-6. The ratchet cannot span two repositories

`ratchet_lint.sh` resolves `deslopped.txt` entries against **its own repo
root**. The documented usage ("from a repo containing this pipeline") assumes
the pipeline is vendored into the manuscript repo. That is a reasonable
design, but the common case for a paper on Overleaf is: manuscript in the
Overleaf git repo, pipeline cloned beside it. In that layout every entry in
`deslopped.txt` matches nothing.

The file's own comment says "an entry matching nothing is an error, not a
silent no-op", which is the right instinct — please confirm that check fires
in this layout, because if it does the user gets a clear error and this is
only a documentation gap.

**Fix:** either document the vendoring requirement prominently in the README
quick-start, or add `--root DIR` so the lint can point at a manuscript
checkout elsewhere.

---

## P2-7. `Paper.EmDash` at error level drowns the AI-tell signal

80 of the 95 initial errors were this one rule. The AI-tell rules that carry
the tool's actual thesis (`CataphoricForecasting`, `VerbTricolon`,
`ServesAsDodge`, `StackedAnaphora`) are mostly warnings, so on a first run the
error stream is ~85% house-style punctuation and the interesting findings are
in the warning stream that CI ignores.

This may well be intentional: `Paper.*` is the house-style namespace and em
dashes really are a strong AI tell in aggregate. But consider whether a first
run reads better with `EmDash` at warning and the structural AI-tells raised
to error, or with a `--summary` mode that reports counts per namespace so the
user sees "80 punctuation, 15 structural" before opening any file.

---

## P2-7b. Enumeration labels are counted as data numbers

Restructuring a figure from four panels to three meant deleting the caption's
`\textbf{(4)}` marker. `invariant_check.py` reported:

```
numbers:    1 violation(s) (11 in old)
  - removed number: '4' x1
RESULT: FAIL
```

The other ten numbers and all eleven cross-references were intact, and a
word-level diff confirmed the only change was `\textbf{(4)} The` → `Below
them sit the`. So the gate fired on a panel label, not on a measurement.

This is the right default (a gate that guesses which numbers matter is worse
than one that flags them all), but it is worth a targeted exemption, because
the failure teaches the user to skim `RESULT: FAIL` — which is the one habit
that makes the whole gate useless.

**Suggested fix:** do not count a bare integer that appears as an
enumeration marker in a caption or list context — `\textbf{(N)}`, `(N)`,
`\item[N]`, `(i)`/`(ii)` — when the same integer carries no unit and does not
appear elsewhere in the file. Failing that, print such cases under a separate
`enumeration:` heading so `RESULT` stays clean and the user still sees them.

---

## P3-8. Small things

- `invariant_check.py --term-audit` prints `OLD` and `NEW` columns even when
  both paths are the same file, which is the documented way to audit a term
  list. A single-column mode would read better for that use.
- SKILL.md's "never bare-delete" rule is the single most valuable instruction
  in the file for em-dash removal specifically (mechanical replacement makes
  prose *more* uniform, which is a net loss). Consider naming em dashes in
  that bullet, since they are the highest-volume rule and the one most likely
  to be fixed with `sed`.
- A worked before/after example in the README, taken from a real rewrite,
  would set the standard better than the pattern catalog alone. For instance:

  > **before** Measured within one system --- one scaffold, one prompt budget,
  > one action space --- across eight cells, the six representations we
  > compare are not redundant.
  >
  > **after** We hold one scaffold, one prompt budget and one action space
  > fixed, and compare the six representations across eight cells. They are
  > not redundant.

  Two sentences of different lengths replace one; the em dash is gone because
  the clause moved, not because it was deleted.

---

## What worked, and should not be "improved" away

- **The sentence-context anchor check.** Verifying that each number and
  citation still sits in the same sentence context is the check that makes it
  safe to let a model rewrite whole sections: it catches "all the numbers are
  still present, but one moved next to a different claim", which is the
  failure mode a human diff review misses. 41 rewrites across 13 files, zero
  anchor violations, and that guarantee came from the script rather than from
  my care.
- **"When a Vale alert and an invariant disagree, the invariant wins"**, with
  `ContrastiveFormulas` named as the worked example. That paragraph is the
  reason the rewrite did not quietly widen a bounded claim.
- **The diff-only rule for the rewrite layer.** Concentrating all semantic
  risk in the one layer that cannot run unattended is the correct split.
