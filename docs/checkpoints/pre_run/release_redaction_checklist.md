# Release Redaction Checklist

> Pre-publication audit gate for the public release artifact (OSF / Zenodo
> deposit + GitHub repo cleanup). Addresses audit constraint **A11**
> (release artifact contains no credentials or private state).
>
> Run before final paper submission and any OSF DOI deposit.

## Sensitive items to verify EXCLUDED from release

### Credentials & secrets

- [ ] No `.env` file in any committed history
  - Verify: `git log --all --full-history -- .env` empty
- [ ] No `${PROXY_API_KEY}` literal value in any tracked file
  - Verify: `git grep -n -i "proxy_api_key.*=.*[a-zA-Z0-9_-]\{20,\}"` empty
- [ ] No `${ANTHROPIC_API_KEY}` / `${OPENAI_API_KEY}` literal values
  - Verify: `git grep -in -E "(api_key|api-key|apikey).{0,5}=.{0,5}[\"\']?[a-zA-Z0-9_-]{20,}"`  empty
- [ ] No `.auth/` directory in tracked files
  - Verify: `find . -path "./.git" -prune -o -name ".auth" -print` empty
- [ ] No `scripts/vwa_env_remote.sh` (gitignored, contains Tailscale-specific endpoints)
  - Verify: `git ls-files scripts/vwa_env_remote.sh` empty

### User-specific paths

- [ ] No hardcoded `/home/jiaming/` paths in committed files
  - Verify: `grep -rn "/home/jiaming" --include="*.py" --include="*.sh" --include="*.yaml" --include="*.md"` shows only DGX-specific docs (CLAUDE.md gitignored, runbooks are fine)
- [ ] No hardcoded `/home/ucab352/` paths outside `scripts/queues/qsub_*.sh` (Myriad-specific qsubs are fine)
  - Verify: `grep -rn "/home/ucab352" --include="*.py" --include="*.md" | grep -v "scripts/queues/qsub_"` empty

### Personal info

- [ ] No author email addresses in code comments
  - Verify: `git grep -E "[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}" -- "*.py" "*.md"` shows only paper.bib bibtex entries (which contain DOIs / arxiv IDs, not personal email)
- [ ] No personal Tailscale IPs (`100.95.81.103` etc.) outside `memory/` (gitignored) or `docs/reference/COMPUTE_INFRASTRUCTURE.md`
  - Note: `100.95.81.103` IS in some docs/reference files because that's documented infrastructure. Confirm OK for release or redact.

### Site auth state

- [ ] `.auth/` directory verified gitignored at root `.gitignore` line 28 ✓
- [ ] No site cookies / Playwright auth JSON files committed
  - Verify: `git ls-files | grep -iE "auth|cookie|session"` shows no matches except in test fixtures

### Sensitive logs

- [ ] No production-credential logs in `logs/` (gitignored)
- [ ] No debugging stdout/stderr captures with auth tokens
- [ ] No personal Slack / email content in chronicle / planning docs

## Items VERIFIED INCLUDED (paper-grade replication essentials)

- [x] All `scripts/` (Python + bash + qsub)
- [x] `p79/` (agent + experiment + analysis modules)
- [x] `configs/exp_v2_*.yaml` (locked seed=42 + per-site configs)
- [x] `pyproject.toml` (dep declarations)
- [x] `external/visualwebarena/` submodule SHA pinned
- [x] `docs/checkpoints/pre_run/` (preregistration + locked versions + audit + cards)
- [x] `docs/checkpoints/paper_drafts/` (camera-ready prose)
- [x] `results/mechanistic/archive_subset_b1_{cls,reddit}/` (curated mirage subsets, 16.5MB+35MB)
- [x] `results/mechanistic/curate_mirage_b1_{classifieds,reddit}/candidates.{jsonl,md}` (paper-cite-able)
- [x] `results/provenance/` (env snapshots, evaluator fingerprints)
- [x] `master_bug_catalog.md` (full 80+ entries, status + commit SHA per fix)

## Verification command (planned)

To be wired into `make pre-release-check`:

```bash
make pre-release-check
```

Will run:

1. `git grep` for `(api_key|password|secret|token).{0,5}=.{0,5}[\"\']?[a-zA-Z0-9_-]{20,}` across all tracked
2. `find` for accidentally-committed `.auth/`, `.env`, `*-key.*`, `*-secret.*`
3. `git ls-files | xargs grep -lE "100\.[0-9]+\.[0-9]+\.[0-9]+"` for IP addresses (manual review required)
4. Check that `external/visualwebarena/.git` is NOT in any tracked file
5. Exit non-zero on any match with redaction recommendation

## Manual review steps before OSF deposit

1. **Browse** `docs/checkpoints/paper_drafts/` for any `TODO_REDACT` markers
2. **Diff** released artifact tarball against private working tree to verify excluded items
3. **Cold-clone** the released artifact to a fresh temp dir, run `make replicate-smoke` to verify it works without local secrets
4. **Spot-check** 3 random episode JSON files for absence of API tokens / personal info in metadata

## Sign-off log

| Date | Reviewer | Result | Issues found |
|---|---|---|---|
| 2026-05-09 | Claude (Opus 4.7) | Initial checklist creation | n/a |
| (to be filled) | Author (manual review) | TBD | TBD |
| (to be filled) | Pre-OSF-deposit | TBD | TBD |

## References

- NeurIPS 2024 Reproducibility Checklist Q5/Q13 (release scope)
- `pre_run/ethics_license_coi_statements.md` (license + COI)
- `pre_run/preregistration.md §7` (reproducibility scope)
- `pre_run/locked_versions.md` (what's pinned)
- `master_bug_catalog.md` (known issues already public)
