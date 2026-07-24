# Vendored: paper-deslop

Upstream: https://github.com/Quarkgluonmixture/paper-deslop
Vendored at: `af4bb36d0ca42454b10e0d884e4f6354d80f122d` (2026-07-24), MIT.
Previously at `3adb2f5` (2026-07-24); that sync's field findings — substring
term matching and the all-or-nothing CI lint — are what upstream `af4bb36`
fixed, so the local workarounds for both are gone from this file.

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
| `invariant-check-md-percent.patch` | bug fix | `%` is a LaTeX comment but a percent sign in Markdown. Upstream strips `%`-to-end-of-line unconditionally, so on `.md` everything after the first percent sign escapes the number/citation/term checks. Adds a `tex: bool` parameter threaded from the existing `pandoc` suffix test. |
| `ratchet-lint-vendored-root.patch` | generalization | `ratchet_lint.sh` assumed the pipeline sits at the repo root: it `cd`s to its own parent and hardcodes an exclude list ending in `docs/`. Adds `--root DIR` (default: enclosing git repo root) and `PAPER_PATHSPEC` (default: `docs/checkpoints/paper_drafts` here), and passes `--config` to Vale. |
| `skill-md-p79-paths.patch` | layout | SKILL.md: layout table + every `scripts/…`, `terms.txt`, `deslopped.txt` path repointed, `--config=` added to the `vale` step. |
| `tests-run-sh-p79.patch` | layout + test | `--root .` for the ratchet fixtures (they are pathspecs inside the pipeline dir, not the repo), plus the `markdown percent signs do not blind the gate` regression case. |

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
