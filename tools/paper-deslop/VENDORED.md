# Vendored: paper-deslop

Upstream: https://github.com/Quarkgluonmixture/paper-deslop
Vendored at: `3adb2f5bcf4ccfab0dc318345bef265b05f89c48` (2026-07-24), MIT.

## Layout in this repo

Upstream's README installs the pipeline at the paper repo's root. P79 is not a
paper-only repo — `scripts/` holds the experiment runner and `docs/` holds 394
Markdown files — so the deterministic layers are vendored under
`tools/paper-deslop/` instead, and only the skill lives where Claude Code
discovers it.

This repo also gitignores `.claude/` wholesale, so the skill itself lives under
`tools/paper-deslop/skill/` (tracked) and is symlinked into `.claude/skills/`
where Claude Code discovers it — the same pattern the repo already uses for
`caveman`, `tdd`, and the other `.agents/skills/` links. One copy, version
controlled, no vendored duplicate to drift.

| Component | Path here | Upstream path |
|---|---|---|
| Rewrite skill | `tools/paper-deslop/skill/deslop-paper/` (symlinked to `.claude/skills/deslop-paper`) | `.claude/skills/deslop-paper/` |
| Vale config | `tools/paper-deslop/.vale.ini` | `.vale.ini` |
| Vale styles | `tools/paper-deslop/styles/` | `styles/` |
| Term whitelist | `tools/paper-deslop/terms.txt` | `terms.txt` |
| Scripts | `tools/paper-deslop/scripts/` | `scripts/` |
| Self-test | `tools/paper-deslop/tests/` | `tests/` |
| CI | `.github/workflows/prose-lint.yml` | same, rewritten (see below) |
| Ratchet list | `tools/paper-deslop/deslopped.txt` | (P79-only) |

Consequence: Vale searches upward from the cwd for `.vale.ini` and will not
find a vendored one, so **every** invocation from the repo root needs
`--config=tools/paper-deslop/.vale.ini`. `tests/run.sh` and
`gen_vale_vocab.py` resolve paths relative to their own location and need no
flag.

## Local modifications (reapply after an upstream sync)

Two patches in `patches/`, both regenerable with `diff -u <upstream> <local>`:

- `skill-md-p79-paths.patch` — SKILL.md: adds the layout table, repoints
  `scripts/invariant_check.py` → `tools/paper-deslop/scripts/invariant_check.py`,
  `terms.txt` → `tools/paper-deslop/terms.txt`, and adds `--config=` to the
  `vale` step.
- `terms-p79.patch` — terms.txt rewritten for P79 vocabulary (phantom routing
  space / P-SoM / AXTree / drop-one oracle / …), plus a warning about
  substring matching.

`.github/workflows/prose-lint.yml` is a rewrite, not a patch: upstream lints
the whole repo and blocks on any error-level alert. Here it is path-filtered
to `docs/checkpoints/paper_drafts/**`, targets `master` instead of `main`, and
splits into a blocking ratchet job plus a report-only full-corpus job.

## Sync procedure

```bash
git clone --depth 1 https://github.com/Quarkgluonmixture/paper-deslop /tmp/pd
rsync -a --exclude '.git/' --exclude '.claude/' --exclude '.github/' \
    --exclude 'skill/' --exclude 'deslopped.txt' --exclude 'VENDORED.md' \
    --exclude 'patches/' /tmp/pd/ tools/paper-deslop/
rsync -a --delete /tmp/pd/.claude/skills/deslop-paper/ tools/paper-deslop/skill/deslop-paper/
git -C /tmp/pd rev-parse HEAD > tools/paper-deslop/.upstream-sha
patch -p0 tools/paper-deslop/skill/deslop-paper/SKILL.md \
    < tools/paper-deslop/patches/skill-md-p79-paths.patch
# terms.txt: keep the local version; re-diff only if upstream changed the header
bash tools/paper-deslop/tests/run.sh   # must print all ok
```

The excludes keep the P79-only files (`deslopped.txt`, `VENDORED.md`,
`patches/`) and the local CI rewrite from being clobbered; the skill is synced
explicitly into `skill/` so the `.claude/skills/` symlink keeps resolving.

## Why terms.txt has no short abbreviations

`invariant_check.term_counts()` builds `[\s~-]+`-joined regexes with no word
boundary, so whitelist matching is substring matching. Measured on the P79
drafts on 2026-07-24: `DOM` matches random / dominance / dominant / dominated /
domain / domains **90** times beyond the real ones, and `SoM` matches some /
sometimes 7 times. Both are load-bearing edit vocabulary, so pinning their
counts would fire the gate on legitimate rewrites. Whitelist entries are
therefore multi-word phrases or long unique tokens only. `SoM` and `DOM`
themselves are protected in practice by `P-SoM`, `SOM_MARKS`, and
`accessibility tree` / `AXTree`.

## Environment notes (DGX Spark, aarch64, no sudo)

- Vale 3.15.2 installed at `~/.local/bin/vale` (arm64 tarball, already on PATH).
  Not in the repo; reinstall on a new host with the same one-liner the CI uses,
  swapping `Linux_64-bit` → `Linux_arm64`.
- `invariant_check.py` / `tex_to_text.py` / `gen_vale_vocab.py` are pure stdlib
  (python3.12 verified). `grammar_check.py` additionally needs a **local**
  LanguageTool server; it refuses the public API unconditionally, so
  unpublished manuscripts stay on the machine.
