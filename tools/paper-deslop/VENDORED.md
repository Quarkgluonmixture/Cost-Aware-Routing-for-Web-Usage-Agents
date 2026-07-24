# Vendored: paper-deslop

Upstream: https://github.com/Quarkgluonmixture/paper-deslop
Vendored at: `58253e9a8f2c61da3be41ab46696c6e21e06a462` (2026-07-25), MIT.
Lineage: `3adb2f5` → `af4bb36` (word-bounded terms, ratchet lint) → `58253e9`
(Markdown `%` truncation, structural pointers, error tier, vendored installs).
Both syncs retired local workarounds: the `%`-as-comment fix and the
`--root` / `PAPER_PATHSPEC` generalization were reported from here and are now
upstream, so those patches are gone. What remains is layout glue plus two
filename-robustness fixes found on this repo's own file names.

## Layout in this repo

Upstream's README installs the pipeline at the paper repo's root. P79 is not a
paper-only repo — `scripts/` holds the experiment runner and `docs/` holds 394
Markdown files — so the deterministic layers are vendored under
`tools/paper-deslop/`. The repo also gitignores `.claude/` wholesale, so the
skill itself is tracked under `tools/paper-deslop/skill/` and symlinked into
`.claude/skills/`, the same pattern the repo already uses for `caveman`,
`tdd`, and the other `.agents/skills/` links.

| Component | Path here | Upstream path |
|---|---|---|
| Rewrite skill | `tools/paper-deslop/skill/deslop-paper/` (symlinked to `.claude/skills/deslop-paper`) | `.claude/skills/deslop-paper/` |
| Vale config | `tools/paper-deslop/.vale.ini` | `.vale.ini` |
| Vale styles | `tools/paper-deslop/styles/` | `styles/` |
| Term whitelist | `tools/paper-deslop/terms.txt` | `terms.txt` |
| Ratchet list | `tools/paper-deslop/deslopped.txt` | `deslopped.txt` |
| Scripts | `tools/paper-deslop/scripts/` | `scripts/` |
| Self-test | `tools/paper-deslop/tests/` | `tests/` |
| CI | `.github/workflows/prose-lint.yml` | same, rewritten (see below) |

Two consequences of vendoring:

- Vale searches upward from the cwd for `.vale.ini` and will not find a
  vendored one, so **every** invocation from the repo root needs
  `--config=tools/paper-deslop/.vale.ini`. `ratchet_lint.sh` passes it
  automatically; `make deslop-lint` and the skill's own instructions carry it
  explicitly. `tests/run.sh` and `gen_vale_vocab.py` resolve paths relative to
  their own location and need no flag.
- Pathspecs in `deslopped.txt` are repo-relative, not pipeline-relative, so
  `ratchet_lint.sh` resolves them from the git repo root (`--root`, below).

Entry points: `make deslop-lint [F=]` · `make deslop-gate OLD= NEW=` ·
`make deslop-ratchet [ALL=1]` · `make deslop-audit [F=]` ·
`make deslop-selftest` · the `/deslop-paper` skill.

## Local modifications

Patches in `patches/`, regenerable with `diff -u <upstream> <local>`. The
first two are **upstream-worthy** (they fix or generalize upstream behaviour,
not P79 preferences); the rest are layout glue.

| Patch | Kind | What |
|---|---|---|
| `ratchet-lint-filename-safety.patch` | bug fix | Two ways a file name breaks the lint. (a) The file list was passed as an unquoted `$files`, so a tracked path with a space (`docs/literature/Cost-Aware Routing.md`) was word-split and vale aborted with `E100 [doLint] Runtime error` — which `--all`'s unconditional `exit 0` turned into a silent success. Now an argv array, and a vale runtime error is always reported. (b) `git ls-files` without `core.quotePath=false` returns non-ASCII paths octal-escaped and quoted, naming no existing file; this repo has 6 such tracked Markdown files. |
| `tests-run-sh-vendored-root.patch` | test fix | `check_ratchet` passes pathspecs that live inside the pipeline dir but never passes `--root`, so both cases resolve against the manuscript repo and fail in **any** vendored install. Adds `--root .`. Upstream's own repo cannot see this: there the pipeline *is* the root. |
| `skill-md-p79-paths.patch` | layout | SKILL.md: layout table + every `scripts/…`, `terms.txt`, `deslopped.txt` path repointed, `--config=` added to the `vale` step, `PAPER_PATHSPEC` noted. |

`PAPER_PATHSPEC=docs/checkpoints/paper_drafts` is exported by both the
Makefile and CI. Upstream's default is every tracked `.tex`/`.md` under the
root, which here is 394 files of lab notes rather than the manuscript.

`terms.txt` is **fully local** (P79 vocabulary) and deliberately not a patch —
take upstream's header wholesale when its matching rules change, keep the term
list. `.github/workflows/prose-lint.yml` is a rewrite, not a patch: same
blocking/advisory split as upstream, but path-filtered to
`docs/checkpoints/paper_drafts/**` + `tools/paper-deslop/**`, targeting
`master`, and calling the vendored `ratchet_lint.sh`.

## Sync procedure

```bash
git clone --depth 1 https://github.com/Quarkgluonmixture/paper-deslop /tmp/pd
rsync -a --exclude '.git/' --exclude '.claude/' --exclude '.github/' \
      --exclude 'skill/' --exclude 'VENDORED.md' --exclude 'patches/' \
      --exclude '.upstream-sha' /tmp/pd/ tools/paper-deslop/
rsync -a --delete /tmp/pd/.claude/skills/deslop-paper/ tools/paper-deslop/skill/deslop-paper/
git -C /tmp/pd rev-parse HEAD > tools/paper-deslop/.upstream-sha
for p in tools/paper-deslop/patches/*.patch; do patch -p0 < "$p" || echo "REAPPLY BY HAND: $p"; done
git checkout tools/paper-deslop/terms.txt        # local vocabulary wins
python3 tools/paper-deslop/scripts/gen_vale_vocab.py
make deslop-selftest                             # must print all ok
make deslop-audit                                # must report 0 dead terms
```

The rsync excludes keep the P79-only files and the CI rewrite from being
clobbered. `--exclude 'skill/'` on the first pass plus an explicit second
rsync means the skill has exactly one tracked copy and the `.claude/skills/`
symlink keeps resolving. Note that upstream may rewrite the same functions the
patches touch (it did between `3adb2f5` and `af4bb36`, where `term_counts`
gained `term_pattern`), in which case `patch` fails and the change is
reapplied by hand — the patch files then double as the changelog of what to
reapply.

## Environment notes (DGX Spark, aarch64, no sudo)

- Vale 3.15.2 at `~/.local/bin/vale` (arm64 tarball, already on PATH). Not in
  the repo; reinstall on a new host with the one-liner CI uses, swapping
  `Linux_64-bit` → `Linux_arm64`.
- `invariant_check.py` / `ratchet_lint.sh` / `tex_to_text.py` /
  `gen_vale_vocab.py` need only python3 + git + vale (python3.12 verified).
  `grammar_check.py` additionally needs a **local** LanguageTool server; it
  refuses the public API unconditionally, so unpublished manuscripts stay on
  the machine.
