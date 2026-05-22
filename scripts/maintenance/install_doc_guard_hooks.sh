#!/usr/bin/env bash
# One-time per clone — run on EACH machine (DGX / A100 / quark / Myriad).
# Points git at the versioned .githooks/ dir so the pre-push doc-numbering guard
# runs. Hook config (core.hooksPath) is per-clone and is NOT carried by git push,
# hence this must be run once on every working copy.
set -e
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

OLD="$(git config core.hooksPath 2>/dev/null || echo '(unset → default .git/hooks)')"
git config core.hooksPath .githooks
chmod +x .githooks/pre-push 2>/dev/null || true
chmod +x scripts/maintenance/check_doc_numbering.py 2>/dev/null || true

echo "[doc-guard] core.hooksPath: ${OLD}  ->  .githooks"
echo "[doc-guard] pre-push doc-numbering guard ACTIVE on this clone."
echo "[doc-guard] reminder: re-run this once on every other machine/clone."
