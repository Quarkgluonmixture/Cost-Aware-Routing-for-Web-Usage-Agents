#!/usr/bin/env bash
# B-1822 regression — flock fd-inheritance → condition-boundary self-collision.
#
# Reproduces the Fire-6 2026-05-21 21:21:12Z false "another chain holds lock /
# possible double-fire" ABORT, and verifies the queue_baseline.sh fix (close
# inherited paper-grade lock fds 9/8/7 when backgrounding the runner/watchdog
# daemons via `setsid ... 9>&- 8>&- 7>&- &`).
#
# Mechanism under test
# --------------------
# flock advisory locks bind to the open file description (OFD), not the fd
# number. A backgrounded daemon that INHERITS the chain's lock fd (fd 9) keeps
# that OFD alive after the chain's own `exec 9>&-` at the condition boundary
# (queue_chain.sh:549). So the chain's NEXT `flock -n 9` (next condition) fails
# even though the chain "released" → false double-fire ABORT. Closing fd 9 on
# the daemon makes the chain the SOLE OFD reference → release is real.
#
# Pre-B-1803 this was latent: every fire died inside condition [1/N] (task 4/75
# EvaluatorUnavailableError) and never reached a condition boundary.
#
# Exit 0 = both scenarios behaved as expected (bug reproduces pre-fix, gone
# post-fix). Exit 1 = regression. Self-contained; no repo state touched.
set -u

LOCK_PREFIX="$(mktemp -u -t b1822_XXXXXX)"
LOCK1="${LOCK_PREFIX}.1.lock"
LOCK2="${LOCK_PREFIX}.2.lock"
trap 'rm -f "$LOCK1" "$LOCK2"' EXIT
pass=0; fail=0

# Simulate one chain condition boundary.
#   $1 = "inherit" (pre-fix daemon: keeps fd 9) | "closed" (post-fix: 9>&-)
#   $2 = lock file path (independent per scenario for clean isolation)
# Returns 0 if the parent can RE-acquire the lock after releasing, 1 if blocked.
run_boundary() {
  local mode="$1" lf="$2" rc
  (
    exec 9>"$lf"
    flock -n 9 || { echo "setup-acquire-failed" >&2; exit 2; }
    # Spawn a long-lived daemon that out-lives the "runner" (mirrors watchdog).
    if [[ "$mode" == "inherit" ]]; then
      setsid sleep 20 >/dev/null 2>&1 &          # inherits fd 9 (pre-B-1822)
    else
      setsid sleep 20 9>&- >/dev/null 2>&1 &     # drops fd 9 (the B-1822 fix)
    fi
    local child=$!
    exec 9>&-                                     # parent release (condition boundary)
    exec 9>"$lf"                                  # parent re-acquire (next condition)
    if flock -n 9; then rc=0; else rc=1; fi
    exec 9>&-
    kill "$child" 2>/dev/null; wait "$child" 2>/dev/null
    exit $rc
  )
}

echo "=== B-1822 flock fd-inheritance regression ==="

# Scenario 1 — pre-fix daemon MUST reproduce the bug (re-acquire blocked).
if run_boundary inherit "$LOCK1"; then
  echo "[S1 pre-fix  repro ] re-acquire OK   — UNEXPECTED (bug not reproduced)"; fail=$((fail+1))
else
  echo "[S1 pre-fix  repro ] re-acquire FAIL — bug reproduced (inherited fd holds OFD) OK"; pass=$((pass+1))
fi

# Scenario 2 — post-fix daemon MUST succeed (re-acquire clean).
if run_boundary closed "$LOCK2"; then
  echo "[S2 post-fix verify] re-acquire OK   — fix works (daemon dropped fd 9) OK"; pass=$((pass+1))
else
  echo "[S2 post-fix verify] re-acquire FAIL — UNEXPECTED (fix ineffective)"; fail=$((fail+1))
fi

echo "=== result: pass=${pass} fail=${fail} ==="
[[ "$fail" -eq 0 && "$pass" -eq 2 ]]
