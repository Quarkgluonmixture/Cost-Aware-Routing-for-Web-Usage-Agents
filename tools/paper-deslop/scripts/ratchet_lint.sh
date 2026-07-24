#!/usr/bin/env bash
# Ratcheted prose lint. Two modes, same file-selection logic in both, so what
# you run locally is what CI runs.
#
#   scripts/ratchet_lint.sh                 # blocking: error-level alerts in
#                                           # deslopped.txt files fail (exit 1)
#   scripts/ratchet_lint.sh --all           # advisory: every paper source,
#                                           # always exit 0
#   scripts/ratchet_lint.sh --list FILE     # use FILE instead of deslopped.txt
#   scripts/ratchet_lint.sh --root DIR      # treat DIR as the paper repo root
#   scripts/ratchet_lint.sh --output=line   # passed through to vale
#
#   PAPER_PATHSPEC='docs/paper/*.md'        # override which files count as
#                                           # manuscript sources
#
# Why the ratchet: an AI-drafted manuscript arrives with hundreds of
# error-level alerts. Blocking on all of them from day one trains everyone to
# ignore the gate. deslopped.txt is the blocking set and it only grows.
#
# Vendored installs are first-class. The pipeline may live in a subdirectory
# (tools/deslop/, .deslop/, ...) of a repo whose manuscript sits anywhere,
# including docs/. Paper sources are therefore everything tracked under the
# root EXCEPT the pipeline's own files, and vale is always pointed at the
# pipeline's own .vale.ini (vale searches upward from the CWD and would
# otherwise miss, or mis-resolve, a vendored config).
set -uo pipefail

PIPE_DIR=$(cd "$(dirname "$0")/.." && pwd -P)

list=""
root=""
mode=ratchet
vale_args=(--minAlertLevel=error)

while [ $# -gt 0 ]; do
    case "$1" in
        --all)       mode=all ;;
        --list)      list=${2:?--list needs a file}; shift ;;
        --root)      root=${2:?--root needs a directory}; shift ;;
        --output=*)  vale_args+=("$1") ;;
        -h|--help)   sed -n '2,25p' "$0"; exit 0 ;;
        *)           echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

if ! command -v vale >/dev/null 2>&1; then
    echo "vale not found on PATH (brew install vale, or see vale.sh)" >&2
    exit 2
fi

# Root = the paper repo, which is NOT necessarily the pipeline directory.
if [ -z "$root" ]; then
    root=$(git -C "$PIPE_DIR" rev-parse --show-toplevel 2>/dev/null || true)
fi
if [ -z "$root" ]; then
    echo "not a git repository under $PIPE_DIR: pass --root DIR" >&2
    exit 2
fi
root=$(cd "$root" && pwd -P) || exit 2
if ! git -C "$root" rev-parse --git-dir >/dev/null 2>&1; then
    echo "$root is not a git repository (file selection uses git ls-files)" >&2
    exit 2
fi

# A vendored .vale.ini is not found by vale's upward search from the root.
[ -f "$PIPE_DIR/.vale.ini" ] && vale_args+=(--config "$PIPE_DIR/.vale.ini")

# The pipeline's own files, as a path prefix relative to the root ("" when the
# pipeline IS the root). Everything else under the root is fair game, so a
# manuscript in docs/ is linted instead of silently skipped.
prefix=""
case "$PIPE_DIR" in
    "$root") prefix="" ;;
    "$root"/*) prefix="${PIPE_DIR#"$root"/}/" ;;
esac
prefix_re=$(printf '%s' "$prefix" | sed 's/[.[\*^$]/\\&/g')
own_files="^(${prefix_re}tests/|${prefix_re}\.claude/|${prefix_re}\.github/\
|${prefix_re}docs/|${prefix_re}styles/|${prefix_re}scripts/\
|${prefix_re}README|${prefix_re}THIRD_PARTY)"

# Default: every tracked .tex/.md under the root. PAPER_PATHSPEC narrows it.
read -r -a pathspec <<<"${PAPER_PATHSPEC:-*.tex *.md}"

all_sources() {
    git -C "$root" -c core.quotePath=false ls-files -- "${pathspec[@]}" \
        | grep -E '\.(tex|md|markdown|qmd|rmd)$' \
        | grep -vE "$own_files" || true
}

# Pass the file list as an argv array, never as an unquoted $files: a tracked
# path containing a space ("docs/literature/Cost-Aware Routing.md") would be
# word-split, vale would abort with "E100 [doLint] Runtime error / argument
# ... does not exist", and in --all mode the unconditional exit 0 turned that
# abort into a silent success. A vale runtime error is always reported, even
# in advisory mode, because it means NOTHING was linted.
run_vale() {  # newline-separated file list
    local list=$1 status err
    local -a argv
    mapfile -t argv <<<"$list"
    err=$(mktemp)
    (cd "$root" && vale "${vale_args[@]}" "${argv[@]}") 2> >(tee "$err" >&2)
    status=$?
    if grep -qE 'Runtime error|does not exist' "$err"; then
        echo "error: vale aborted without linting (see above)" >&2
        rm -f "$err"
        return 2
    fi
    rm -f "$err"
    return $status
}

if [ "$mode" = all ]; then
    files=$(all_sources)
    if [ -z "$files" ]; then
        # Never silent: a zero-file lint that looks like success is how a
        # manuscript ends up unlinted for weeks.
        echo "no paper sources matched under $root"
        echo "  pathspec:  ${pathspec[*]}${PAPER_PATHSPEC:+  (from PAPER_PATHSPEC)}"
        echo "  excluding: the pipeline's own files (${prefix:-repo root})"
        if [ -n "${PAPER_PATHSPEC:-}" ]; then
            echo "PAPER_PATHSPEC matched nothing -- check it" >&2
            exit 2
        fi
        echo "  set PAPER_PATHSPEC to point at the manuscript, e.g."
        echo "  PAPER_PATHSPEC='docs/paper/*.md' $0 --all"
        exit 0
    fi
    run_vale "$files"
    echo "(advisory: the full-repo lint never blocks)"
    exit 0
fi

# Blocking mode: the list lives with the manuscript by default, with the
# pipeline directory as a fallback for vendored installs.
if [ -z "$list" ]; then
    if [ -f "$root/deslopped.txt" ]; then
        list="$root/deslopped.txt"
    else
        list="$PIPE_DIR/deslopped.txt"
    fi
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
    matched=$(git -C "$root" -c core.quotePath=false ls-files -- "$spec" \
        | grep -E '\.(tex|md|markdown|qmd|rmd)$' || true)
    if [ -z "$matched" ]; then
        echo "error: $list entry matches no tracked source under $root: $spec" >&2
        bad=1
        continue
    fi
    files+="$matched"$'\n'
done <<<"$specs"
[ "$bad" -eq 0 ] || exit 2

files=$(printf '%s' "$files" | sed '/^$/d' | sort -u)
count=$(printf '%s\n' "$files" | wc -l | tr -d ' ')
echo "ratchet: $count file(s) declared deslopped in $list; error-level alerts block"
run_vale "$files"
