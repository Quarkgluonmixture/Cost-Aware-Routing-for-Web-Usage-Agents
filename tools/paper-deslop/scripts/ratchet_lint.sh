#!/usr/bin/env bash
# Ratcheted prose lint. Two modes, same file-selection logic in both, so
# what you run locally is what CI runs.
#
#   scripts/ratchet_lint.sh                 # blocking: error-level alerts in
#                                           # deslopped.txt files fail (exit 1)
#   scripts/ratchet_lint.sh --all           # advisory: every paper source,
#                                           # always exit 0
#   scripts/ratchet_lint.sh --list FILE     # use FILE instead of deslopped.txt
#   scripts/ratchet_lint.sh --output=line   # passed through to vale
#
# Why: an AI-drafted manuscript arrives with hundreds of error-level alerts.
# Blocking on all of them from day one trains everyone to ignore the gate.
# deslopped.txt is the blocking set and it only grows.
#   scripts/ratchet_lint.sh --root DIR      # resolve pathspecs from DIR
#                                           # (default: the enclosing git
#                                           # repo root, which differs from
#                                           # the pipeline dir in a vendored
#                                           # install like P79's tools/)
set -uo pipefail
pipeline=$(cd "$(dirname "$0")/.." && pwd)

# P79 vendored layout: the pipeline lives in tools/paper-deslop/ while the
# paper lives in docs/checkpoints/paper_drafts/, so pathspecs resolve from the
# repo root and Vale needs the vendored config explicitly. PAPER_PATHSPEC
# overrides which tracked files `--all` considers.
root=""
list="$pipeline/deslopped.txt"
mode=ratchet
vale_args=(--minAlertLevel=error --config="$pipeline/.vale.ini")
PAPER_PATHSPEC=${PAPER_PATHSPEC:-"docs/checkpoints/paper_drafts"}

while [ $# -gt 0 ]; do
    case "$1" in
        --all)       mode=all ;;
        --list)      list=$(realpath "${2:?--list needs a file}"); shift ;;
        --root)      root=${2:?--root needs a directory}; shift ;;
        --output=*)  vale_args+=("$1") ;;
        -h|--help)   sed -n '2,22p' "$0"; exit 0 ;;
        *)           echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

if [ -z "$root" ]; then
    root=$(git -C "$pipeline" rev-parse --show-toplevel 2>/dev/null || echo "$pipeline")
fi
cd "$root"

if ! command -v vale >/dev/null 2>&1; then
    echo "vale not found on PATH (brew install vale, or see vale.sh)" >&2
    exit 2
fi
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "not a git repository: file selection uses git ls-files" >&2
    exit 2
fi

# Paper sources = tracked .tex/.md under PAPER_PATHSPEC that are not part of
# the pipeline itself. Upstream excludes docs/ wholesale; P79 keeps its paper
# inside docs/, so the scope is an explicit pathspec instead of a blocklist.
all_sources() {
    git ls-files -- "$PAPER_PATHSPEC" \
        | grep -E '\.(tex|md)$' \
        | grep -vE '(^|/)(tests/|README|THIRD_PARTY)' || true
}

if [ "$mode" = all ]; then
    files=$(all_sources)
    if [ -z "$files" ]; then
        echo "no paper sources to lint (template repo)"
        exit 0
    fi
    # shellcheck disable=SC2086
    vale "${vale_args[@]}" $files
    echo "(advisory: the full-repo lint never blocks; the blocking set is $list)"
    exit 0
fi

if [ ! -f "$list" ]; then
    echo "$list not found: nothing is ratcheted yet, skipping the blocking lint"
    exit 0
fi

specs=$(grep -vE '^[[:space:]]*(#|$)' "$list" || true)
if [ -z "$specs" ]; then
    echo "$list is empty: nothing is ratcheted yet"
    echo "(add a file once it is error-clean, and regressions in it will block)"
    exit 0
fi

# Resolve every entry before linting anything: an entry that matches no
# tracked file is a typo silently protecting nothing, so it fails loudly.
files=""
bad=0
while IFS= read -r spec; do
    [ -n "$spec" ] || continue
    matched=$(git ls-files -- "$spec" | grep -E '\.(tex|md)$' || true)
    if [ -z "$matched" ]; then
        echo "error: $list entry matches no tracked .tex/.md file: $spec" >&2
        bad=1
        continue
    fi
    files+="$matched"$'\n'
done <<<"$specs"
[ "$bad" -eq 0 ] || exit 2

files=$(printf '%s' "$files" | sed '/^$/d' | sort -u)
count=$(printf '%s\n' "$files" | wc -l | tr -d ' ')
echo "ratchet: $count file(s) declared deslopped in $list; error-level alerts block"
# shellcheck disable=SC2086
vale "${vale_args[@]}" $files
