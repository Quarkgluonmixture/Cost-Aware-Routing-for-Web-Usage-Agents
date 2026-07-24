---
name: deslop-paper
description: |
  Remove AI writing patterns from academic manuscripts (CS/ML focus) without
  changing what the paper claims. Use when asked to "deslop", "de-AI",
  "humanize", "去AI味", "润色论文", or to polish/rewrite a paper section, and
  before submitting a manuscript. Works on LaTeX and Markdown sources. Always
  produces a reviewable diff with reasons, and mechanically verifies rewrite
  invariants (numbers, citations, cross-refs, whitelisted terms) with
  tools/paper-deslop/scripts/invariant_check.py.
---

# Deslop-paper: de-AI an academic manuscript without changing its claims

## Repo layout (P79 vendored install)

The deterministic layers live under `tools/paper-deslop/`, not at the repo
root as upstream's README describes. All commands below are written for that
layout and are run **from the repo root**:

| Upstream path | This repo |
|---|---|
| `terms.txt` | `tools/paper-deslop/terms.txt` |
| `scripts/invariant_check.py` | `tools/paper-deslop/scripts/invariant_check.py` |
| `.vale.ini` | `tools/paper-deslop/.vale.ini` (Vale needs `--config=`) |
| `tests/run.sh` | `tools/paper-deslop/tests/run.sh` (self-contained, `cd`s itself) |

This skill's own files are tracked at `tools/paper-deslop/skill/deslop-paper/`
and symlinked into `.claude/skills/` (the repo gitignores `.claude/`). Edit the
tracked copy. Paper sources live in `docs/checkpoints/paper_drafts/` (Markdown
sections + `aaai27/latex/*.tex`). There are also `make` shortcuts:
`make deslop-lint [F=<file>]`, `make deslop-gate OLD= NEW=`,
`make deslop-selftest`. Upstream sync notes: `tools/paper-deslop/VENDORED.md`.

The goal is quality editing, not detector evasion: delete language that
carries no information, restore concrete subjects and causal boundaries,
keep claim strength aligned with evidence strength, and make the paper read
like a person who understands the research wrote it. Chasing "low AI-detection
scores" as a goal produces a different, equally recognizable artificial voice;
do not optimize for it.

## Non-negotiable invariants

NEVER change, in any rewrite:

1. **Numbers** — every statistic, measurement, percentage, count, and unit.
2. **Citations and cross-references** — `\cite`/`\citep`/`\citet` keys,
   `\ref`/`\eqref`/`\autoref` targets, `\label` definitions, pandoc `@keys`.
3. **Math** — anything inside `$...$` or math environments, verbatim.
4. **Whitelisted domain terms** — every term in `tools/paper-deslop/terms.txt`. No
   synonym substitution, no expansion of dash compounds, no "simplification".
   Term repetition is a feature of scientific writing, not a defect.
5. **Claim strength and direction** — "may reduce" must not become "reduces",
   and "reduces" must not become "may reduce". Hedges are calibrated (see
   patterns), never silently strengthened or weakened.
6. **Quoted text, LaTeX macros, preamble, and comments** — edit prose only.

Verification is mechanical where a machine can check it. Before editing a
file, keep a baseline (`git show HEAD:path/to/file.tex > /tmp/baseline.tex`,
or `cp` if the file is not committed). After editing, always run:

```bash
python3 tools/paper-deslop/scripts/invariant_check.py \
    /tmp/baseline.tex path/to/file.tex \
    --terms tools/paper-deslop/terms.txt
```

The gate catches lexical drift: numbers (with sign and unit), citation and
reference keys, whitelist-term counts, exact-hashed math/macros/comments, and
numbers or citations rebound to a different claim's sentence. If it fails,
fix the rewrite or revert. Never hand back a failing rewrite, and never
"explain away" a violation instead of repairing it.

The gate does NOT check hedge strength, claim direction, or argument logic —
invariant 5 is enforced by YOU and the user's diff review, not by the script.
Flag any hedge you touched in the diff notes explicitly.

## Workflow

1. **Scope one section at a time** (Abstract, Introduction, one subsection of
   Discussion, ...). Never batch-rewrite a whole paper in one pass; review
   quality collapses and register differences between sections get lost.
2. **Read `tools/paper-deslop/terms.txt`**. If it does not exist, propose one:
   scan the abstract and section headings for candidate domain terms, show the
   list, and ask the user to confirm before rewriting anything.
3. **Take the baseline copy** of the target file (see above).
4. **Edit in two passes** (rhythm first, then patterns — order matters, see
   the catalog below).
5. **Present the change as a unified diff** (`git diff` on the file, or an
   equivalent before/after listing per paragraph) with a one-line reason per
   change, grouped by pattern name. Do not present a rewritten wall of text
   without the diff.
6. **Run the checks**: `invariant_check.py` (must pass), and
   `vale --config=tools/paper-deslop/.vale.ini <file>` (error-level alerts
   must be zero on edited prose). Report both results honestly. The
   `--config` flag is required: Vale searches upward from the cwd for
   `.vale.ini`, and in this repo the config is vendored, not at the root.
7. If the session is interactive and the user has not pre-approved batch
   application, wait for their reaction to the diff before moving to the next
   section.

## Register by section

| Section | What changes | What is protected |
|---|---|---|
| Abstract | Strictest: every sentence carries data or a claim | Term definitions; the one headline number |
| Introduction | Kill significance inflation, canned openers, false ranges | Interrogative openers ("Why do...?"); the motivation logic |
| Related Work | Kill vague attribution; each claim gets a named citation | "X et al. show..." structure; dense citations are normal |
| Methods | Lightest touch: precision beats elegance | Passive voice is legitimate here; long qualified sentences may be necessary |
| Results | No evaluative adjectives; numbers speak | "significant" only with an actual test behind it |
| Discussion | Calibrate hedging (both directions); concrete limitations | Necessary epistemic qualifiers; do not compress caveats away |
| Conclusion | No generic upbeat endings | "In conclusion" itself is fine |

## Pattern catalog

### Pass 1 — rhythm and structure (do this FIRST)

Restructuring sentence rhythm is the highest-impact intervention; vocabulary
edits alone make prose worse if the cadence stays uniform.

- **Burstiness.** AI text converges on 15–25-word sentences with repeating
  subject-verb-object openings. Vary length (some under 15 words, some over
  30); open some sentences with a prepositional phrase, a subordinate clause,
  or a connective. Semicolons joining related clauses are a mark of skilled
  academic writing — use them.
- **Never bare-delete.** Removing an ornamental adverb or a transition without
  restructuring the sentence produces shorter, MORE uniform prose — a net
  loss. When you remove, restructure: split, merge, or reposition clauses.
- **Connective preservation.** Discourse markers are logic, not slop:
  Although / Whereas / Thus / However / In contrast / Based on these results /
  As expected / Moreover (unstacked) all stay. The test: does the phrase
  *inflate meaning* (delete) or *make logic explicit* (keep)? When an edit
  removes a sentence opener, restore the logical link — substitute a natural
  connective, echo a key noun from the previous sentence, or merge the
  sentences. Choppy connective-stripped prose is itself an AI-cleanup tell.
- **Paragraph cohesion.** After editing, per paragraph: (a) first sentence
  states what the paragraph claims; (b) each later sentence links to the
  previous one by a connective or an echoed key word; (c) contrast/continuity
  openers between paragraphs survive where the argument needs them.

### Pass 2a — content patterns

- **Significance inflation**: "pivotal challenge", "underscores the critical
  importance", "evolving landscape" → state the prevalence, the number, or
  the concrete consequence instead.
- **Promotional language**: "groundbreaking", "remarkable", "state-of-the-art
  performance" without a comparison → neutral statement with the comparison.
- **Superficial `-ing` tails**: "..., highlighting the importance of X" →
  delete the tail or replace with the actual mechanism/implication.
- **Vague attribution**: "Studies have shown", "Experts argue" with no
  citation → name the work. Exception: "Prior work~\cite{...} shows" followed
  by a citation is standard academic writing; keep it.
- **Formulaic challenges/outlook sections**: "Despite these challenges...
  future outlook" → the specific limitation, stated plainly.
- **Content-free evaluation sentences**: "This is an important finding." →
  delete, or show *why* it matters (mechanism, consequence, contrast).
- **Paraphrastic repetition**: the same claim restated via "In other words,"
  → keep the most precise version, delete the rest. Exception: translating a
  statistic into interpretation ("HR 0.65, i.e., a 35% relative reduction")
  adds information; keep it.
- **False ranges**: "from simple navigation to sophisticated reasoning" where
  no scale exists → list the actual cases.
- **Rule-of-three and negative parallelism**: forced triads and "not only X but
  also Y" — one per paragraph is natural, more is a pattern; two items often
  beat three.

### Pass 2b — vocabulary patterns

- **AI-frequent words** (delve, crucial, pivotal, holistic, multifaceted,
  intricate, tapestry, testament, underscore, showcase, foster, garner,
  landscape-as-abstraction...): replace with the specific or common word.
  **Technical-term test first**: would replacing the word lose a
  distinguishable, citable, operationalized concept? "Robust optimization",
  "representation alignment", "comprehensive evaluation suite" can be terms
  of art — check `terms.txt` and the surrounding usage before touching. When
  in doubt, leave it and flag it in the diff notes.
- **Copula avoidance**: "serves as / stands as / represents a" → "is".
- **Synonym cycling**: pick one name per concept and keep it everywhere
  (patients ≠ participants ≠ subjects; association ≠ relationship ≠ link).
- **Ornamental adverbs**: markedly, critically, remarkably, dramatically →
  delete the adverb and state the number or comparison it gestured at (and
  restructure the sentence — see "never bare-delete").
  Functional adverbs carry information and stay: approximately, slightly,
  modestly, consistently, "statistically significantly" with a test.
- **Hedge calibration, both directions**:
  - stacked hedges ("may suggest the potential to possibly...") → ONE
    calibrated hedge ("may reduce");
  - missing hedges on observational/exploratory claims → add one ("may help
    reduce", "was associated with");
  - RCT-grade / theorem-grade results → direct statement, no hedge.
- **Filler**: "In order to" → "To"; "due to the fact that" → "because";
  "It is important to note that X" → "X".

### Pass 2c — formatting and LaTeX

- **Em dashes**: house style is zero (`—` and `---` alike) → comma,
  parentheses, or period. LaTeX en-dash ranges (`pp. 10--12`) stay.
- **Title case**: follow the venue template; do not "fix" heading case the
  document class controls.
- **Quotes**: LaTeX uses ``...''; curly Unicode quotes in source are an
  artifact — normalize.
- **Bold-first bullets, inline-header lists, unicode arrows**: rewrite as
  prose or plain lists.
- Edit prose only: never touch the preamble, macro definitions, math,
  `\begin{}/\end{}` scaffolding, or comments.

## Preserve list (never flag these)

- Standard scholarly transitions (Moreover, Furthermore, However, In
  contrast, Nevertheless, Accordingly, Specifically) — only *stacking* them
  mechanically is a tell.
- Attribution phrases followed by citations or data.
- Interrogative sentence openers framing a research question.
- Passive voice in Methods.
- "we" for the authors' own work.
- Domain terms from `terms.txt` and their consistent repetition.
- Calibrated hedging on observational claims.
- "In conclusion" opening a Conclusion section.

## Self-audit before presenting the diff

1. Rhythm: does each edited paragraph have at least one sentence notably
   shorter and one notably longer than its average?
2. Openings: do three consecutive sentences start the same way anywhere?
3. Connective chain: did any edit leave two adjacent sentences with no
   logical link (connective or echoed noun)?
4. Em dash scan over the edited text: zero `—`/`---` remaining?
5. Whitelist scan: every `tools/paper-deslop/terms.txt` term still present,
   same count, same spelling?
6. Hedge audit: did any claim get stronger or weaker than its evidence?
7. `invariant_check.py` run and passing? `vale --config=...` error-clean?

## Attribution

Merged and adapted from [stephenturner/skill-deslop](https://github.com/stephenturner/skill-deslop)
and [matsuikentaro1/humanizer_academic](https://github.com/matsuikentaro1/humanizer_academic)
(both MIT), re-targeted at CS/ML manuscripts with LaTeX awareness and
mechanical invariant checking added. Detailed pattern catalogs with
before/after examples are vendored in [references/](references/):
[phrases.md](references/phrases.md), [structures.md](references/structures.md),
[tropes.md](references/tropes.md), [examples.md](references/examples.md).
