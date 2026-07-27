# ACL 2026 style kit — provenance

Venue: **REALM @ EMNLP 2026**, direct submission 2026-08-05, double-blind, ACL
style, 8 pages of content with references and appendices uncounted.

## Where these files came from

| file | sha256 |
|---|---|
| `acl.sty` | `19dfeddc2c0e448f3926a0bef048a9db3f3611b46265b760caabd7ada4f361de` |
| `acl_natbib.bst` | `6fbb306202290f4b68e74ac1460a8b27398500cb6dfeb4492e74c457eae7cd1e` |

Downloaded 2026-07-27 from the official upstream repository:

```
https://github.com/acl-org/acl-style-files/archive/refs/heads/master.zip
zip sha256 = bead1eec1aa2c4de8596ddcdd5ac22917f38b2c8441b853ff13015c0b0b8b92e
```

Both files are copied byte-for-byte out of that archive. The kit's own README
states authors may not modify these files, so they are vendored unmodified and
any local need is met from `skeleton_acl.tex` instead. Verify with:

```bash
sha256sum docs/checkpoints/paper_drafts/latex/acl.sty \
          docs/checkpoints/paper_drafts/latex/acl_natbib.bst
```

## Two things about this kit that break a naive build

1. **`acl.sty` issues `\bibliographystyle{acl_natbib}` itself** (line 195). A
   second `\bibliographystyle` in the driver makes bibtex fail with
   `Illegal, another \bibstyle command`. `skeleton_acl.tex` therefore only sets
   the style in the proxy fallback, guarded on `\acl@finalcopytrue` being
   undefined. Same trap as `aaai2027.sty` had.
2. **`[review]` anonymises the author block itself** and adds line numbers and
   page numbers. Do not add `\author{Anonymous submission}` — the option handles
   it, and the skeleton's `\author` is likewise guarded so it only fires in proxy
   mode. `[final]` is the package default, so the option is not optional: leaving
   it off produces a non-anonymous PDF.

## Relationship to the retired AAAI-27 build

`../aaai27/latex/` is kept intact and unused. It holds the AAAI-27 author kit and
its own `convert.sh`, whose behaviour is pinned by a fixture regression in
`tests/test_dayaudit_rounda_20260714.py::test_f03_zero_todo_submission_fixture_runs_full_conversion_chain`.
AAAI-27 was withdrawn 2026-07-22; that directory is an archive, not a live path.

This directory is the live build root. `convert.sh` here differs from the AAAI one
in three ways: the venue template, a paper argument (`paperA` / `paperB`) instead
of a hardcoded source file, and per-paper build state under `build/<paper>/`.
