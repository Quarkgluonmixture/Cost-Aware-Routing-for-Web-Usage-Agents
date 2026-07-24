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
| Prose lint | Vale + `styles/` | deterministic AI-tell rules, academically re-tiered | CI, error level blocks |
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
.vale.ini  styles/  terms.txt  scripts/  .claude/  .github/workflows/prose-lint.yml
```

Then:

1. Rewrite `terms.txt` for the paper's own vocabulary and run
   `python3 scripts/gen_vale_vocab.py`.
2. Commit. CI now: self-tests the pipeline, blocks merge on error-level Vale
   alerts, and posts an informational invariant-drift report on every PR.

## terms.txt: the single source of truth

Every domain term that must survive rewriting lives in `terms.txt`. It feeds
all three layers (skill constraint, invariant check, Vale vocabulary), so a
term added there is protected everywhere at once. The inclusion test:

> If this word were replaced by an everyday synonym, would the paper lose a
> distinguishable, citable, operationalized concept?

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
  "novel") — a human judges whether it is a term of art or puffery.
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
  numbers ("twenty"); a whitelist term pluralized in place (substring counts
  still match); hedge strengthening/weakening and any purely semantic
  rewording. Those live in the diff review, by design.
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
