# paper-deslop

A de-AI-ify pipeline for academic manuscripts (CS/ML focus, LaTeX-aware).
Two tracks: an **interactive rewrite track** (a Claude Code skill,
human-in-the-loop, diff-only) and a **deterministic check track** (Vale +
invariant checker + LanguageTool, automatable in CI).

The goal is quality editing, not detector evasion: delete language with no
information, restore concrete subjects and causal boundaries, keep claim
strength aligned with evidence strength — and put a **mechanical tripwire**
behind every rewrite. The lexical invariant gate detects drift in protected
material (numbers with sign/unit, citation and reference keys, whitelist-term
counts, exact-hashed math/macros/comments, and the sentence context each
number and citation is bound to). It is a tripwire, not a proof of semantic
equivalence — hedge strength and argument logic stay with the human reviewing
the diff, which is why the rewrite layer only ever emits diffs.

```
While writing (interactive, track A)      On commit / CI (automatic, track B)
------------------------------------      -----------------------------------
/deslop-paper <section>                   1. invariant_check.py  (script)
  - one merged skill, per-section         2. Vale                (errors block)
  - outputs diff + reasons                3. grammar_check.py    (LanguageTool,
  - author accepts hunk by hunk              local server, grammar only)
```

## Layers

| Layer | Tool | Role | Mode |
|---|---|---|---|
| Rewrite | `.claude/skills/deslop-paper/` | sentence rhythm, canned phrases, hedging calibration | interactive, diff-only |
| Fidelity gate | `scripts/invariant_check.py` | lexical invariant gate: numbers (sign/unit-aware) / citations / cross-refs / terms / protected blocks (math, macros, comments, verbatim, preamble) / sentence-context anchors | blocking after any rewrite |
| Prose lint | Vale + `styles/` | deterministic AI-tell rules, academically re-tiered | CI, ratcheted: `deslopped.txt` blocks, rest advisory |
| Grammar | `scripts/grammar_check.py` | grammar/typos/punctuation via **local** LanguageTool | pre-submission pass |

### The one design rule

The LLM rewrite layer is never unattended: all semantic-fidelity risk lives
there, so it emits diffs and the invariant checker verifies every application.
The deterministic layers carry zero semantic risk, so they run in CI.

## Quick start

```bash
brew install vale          # macOS; see vale.sh for other platforms
python3 scripts/gen_vale_vocab.py   # after every terms.txt edit
bash tests/run.sh          # pipeline self-test, should print all ok
vale path/to/section.tex   # lint a paper file
scripts/ratchet_lint.sh    # lint the blocking set, exactly as CI does
scripts/ratchet_lint.sh --all   # advisory: every paper source
```

Rewrite loop (inside Claude Code, from a repo containing this pipeline):

```
/deslop-paper                # or just ask: "deslop section 4 of paper.tex"
```

The skill takes a baseline copy, rewrites one section, shows a diff with
per-change reasons, then runs:

```bash
python3 scripts/invariant_check.py BASELINE FILE --terms terms.txt
```

Grammar pass (before submission):

```bash
docker run --rm -d -p 8010:8010 erikvl87/languagetool
python3 scripts/grammar_check.py paper.tex
```

`grammar_check.py` only talks to localhost by default; a private remote
server needs an explicit `--allow-remote`, and the public languagetool.org
API is refused unconditionally — unpublished manuscripts stay on your
machine.

## Installing into a paper repo

Copy these into the paper repository (or start the paper inside this repo):

```
.vale.ini  styles/  terms.txt  deslopped.txt  scripts/
.claude/   .github/workflows/prose-lint.yml
```

Then:

1. Rewrite `terms.txt` for the paper's own vocabulary, check it against the
   real draft (`--term-audit`, below), and run
   `python3 scripts/gen_vale_vocab.py`.
2. Commit. CI now: self-tests the pipeline, blocks merge on error-level Vale
   alerts **in the files listed in `deslopped.txt`** (empty by default, so
   nothing blocks on day one), posts a full-repo alert summary as an
   advisory job summary, and posts an informational invariant-drift report on
   every PR.

## The ratchet: adopting this on a draft that is already written

An AI-drafted manuscript arrives with hundreds of error-level alerts — one
real draft opened at 551, over half of them em dashes. A gate that blocks on
all of them is red from the first push, so it gets ignored, and an ignored
gate protects nothing. The blocking set is therefore a list you grow:

```
deslopped.txt     # git pathspecs, one per line; empty = nothing blocks
```

Deslop a section, get it error-clean, add its path, commit. From then on a
regression in that file breaks the build, while the untouched chapters stay
merely advisory. `scripts/ratchet_lint.sh` is the same code path locally and
in CI. An entry that matches no tracked file is a hard error, not a silent
no-op — a typo there would quietly protect nothing.

## terms.txt: the single source of truth

Every domain term that must survive rewriting lives in `terms.txt`. It feeds
all three layers (skill constraint, invariant check, Vale vocabulary), so a
term added there is protected everywhere at once. The inclusion test:

> If this word were replaced by an everyday synonym, would the paper lose a
> distinguishable, citable, operationalized concept?

Matching is word-bounded, so short acronyms — `DOM`, `SoM`, `AXTree` — are
safe to list: they will not fire inside `random`, `domain`, `dominant`, or
`some`. A word with a capital anywhere but the first letter is matched
case-sensitively (listing `US` does not count every `us`); ordinary words are
case-insensitive, so sentence-initial capitalization cannot shift a count.
Regular plurals and possessives fold into the same count; irregular ones
(`policy`/`policies`) do not.

Curate the list against the actual draft before trusting the gate — a term
that never occurs is a typo, and a term with a surprising count is matching
somewhere you did not expect:

```bash
python3 scripts/invariant_check.py draft.tex draft.tex --term-audit
```

## Vale tiering philosophy

Rules come from the vendored [vale-ai-tells](https://github.com/tbhb/vale-ai-tells)
package (60+ AI-tell rules) plus a small `Paper` style. vale-ai-tells was
written for general prose where nearly everything is `error`; `.vale.ini`
re-tiers it for academic writing:

- **Disabled**: semicolon rule (semicolons are skilled academic prose),
  `--` em-dash rule (breaks LaTeX en-dash ranges; `Paper.EmDash` replaces it).
- **Suggestion**: Moreover/Furthermore, "In conclusion", "as noted above",
  colon caps — all legitimate scholarly usage; only mechanical stacking is a
  tell.
- **Warning**: possibly-technical vocabulary ("robust", "comprehensive",
  "novel") — a human judges whether it is a term of art or puffery. Also the
  tricolon rules: without lookarounds or POS tagging they cannot tell three
  parallel verbs from a plural-noun list ("contains models, datasets X, and
  metrics Y" fires), and they were the second-largest error source on a real
  draft.
- **Error**: canned phrases, em dashes, throat-clearing, vague attribution —
  things with no legitimate academic use.

## Known limitations

- Vale sees hard-wrapped LaTeX line by line for some multi-word upstream
  rules; the `Paper` style uses `\s+` joins to be wrap-safe, the vendored
  rules mostly are not. Soft-wrapped source lints more reliably.
- `tex = md` format mapping means LaTeX commands can leak into linting as
  tokens; `TokenIgnores`/`BlockIgnores` in `.vale.ini` strip math, citations,
  and display environments, but expect occasional noise in heavy markup.
- The gate is **lexical, not semantic**. Documented blind spots: spelled-out
  numbers ("twenty"); a whitelist term pluralized in place (regular plurals
  fold into one count on purpose, so rewording around number is not a
  violation); hedge strengthening/weakening and any purely semantic
  rewording. Those live in the diff review, by design.
- Whitelist matching is word-bounded but has no morphology beyond regular
  plurals: an irregular plural (`matrix`/`matrices`) or a term the rewrite
  legitimately re-inflects shows up as count drift. List both forms, or read
  the diff and move on.
- The sentence-context anchor check is a heuristic (each number/citation must
  keep ≥1 content word from its original sentence). It catches swapped
  numbers and rebound citations between claims; it cannot catch a rebind
  between two sentences that share vocabulary.
- Macro-definition tracking is line-based (`\newcommand`/`\def` on one line);
  multi-line macro bodies are only partially covered.
- `invariant_check.py` treats `4.23\%` vs `4.23 percent` as a violation
  (deliberately strict).
- Warning-tier vocabulary rules cannot tell "robust optimization" (term of
  art) from "robust and comprehensive solution" (slop). That judgment stays
  human.

## Attribution

This pipeline merges and adapts three MIT-licensed projects — see
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md):

- [stephenturner/skill-deslop](https://github.com/stephenturner/skill-deslop) —
  scientific-writing deslop skill; its reference catalogs are vendored under
  `.claude/skills/deslop-paper/references/`.
- [matsuikentaro1/humanizer_academic](https://github.com/matsuikentaro1/humanizer_academic) —
  academic AI-pattern taxonomy; its rhythm-first process, connective
  preservation, and hedge-calibration rules are folded into the skill.
- [tbhb/vale-ai-tells](https://github.com/tbhb/vale-ai-tells) — Vale rule
  package vendored under `styles/ai-tells/` (plus four rhythm-metric rules
  under `styles/Paper/`).
