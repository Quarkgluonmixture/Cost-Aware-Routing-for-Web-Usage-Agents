#!/usr/bin/env bash
# Ratcheted prose lint. Two modes, same file-selection logic in both, so what
# you run locally is what CI runs.
#
#   scripts/ratchet_lint.sh                 # blocking: error-level alerts in
#                                           # deslopped.txt files fail (exit 1)
#   scripts/ratchet_lint.sh --all           # advisory: every paper source;
#                                           # alerts exit 0, a vale crash does not
#   scripts/ratchet_lint.sh --list FILE     # use FILE instead of deslopped.txt
#   scripts/ratchet_lint.sh --root DIR      # treat DIR as the paper repo root
#   scripts/ratchet_lint.sh --output=line   # passed through to vale
#
#   PAPER_PATHSPEC='docs/paper/*.md'        # override which files count as
#                                           # manuscript sources. Whitespace-
#                                           # separated, unless it contains a
#                                           # newline, in which case one
#                                           # pathspec per line (use that when
#                                           # a path contains spaces).
#
# Exit codes: 0 = nothing blocking, 1 = error-level alerts in the blocking
# set, 2 = the lint could not be trusted (bad list entry, unusable pathspec,
# or vale itself failed). vale's own codes are read the same way: 0 clean,
# 1 alerts, >=2 runtime error (E100). A runtime error is never reported as a
# clean run -- "0 files linted, exit 0" is how a manuscript stays unlinted
# for weeks while CI stays green.
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
#
# Written for bash 3.2 (system bash on macOS): no mapfile, no ${x,,}, and
# empty arrays are never expanded under `set -u`.
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
        -h|--help)   sed -n '2,39p' "$0"; exit 0 ;;
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

# Which files count as manuscript sources.
pathspec=()
if [ -n "${PAPER_PATHSPEC:-}" ]; then
    if [[ $PAPER_PATHSPEC == *$'\n'* ]]; then
        while IFS= read -r p; do
            [ -n "$p" ] && pathspec+=("$p")
        done <<<"$PAPER_PATHSPEC"
    else
        read -r -a pathspec <<<"$PAPER_PATHSPEC"
    fi
else
    pathspec=('*.tex' '*.md' '*.markdown' '*.qmd' '*.rmd')
fi

# NUL-separated, so a path with spaces ("Cost-Aware Routing 5.1.md") survives
# instead of being word-split into arguments that do not exist -- vale answers
# that with E100 and exit 2. -z also bypasses git's octal quoting of non-ASCII
# paths ("\346\210\220..."), which points at files that do not exist either;
# core.quotePath=false covers any non -z use.
#
# The pipeline-exclusion applies ONLY to the automatic sweep. A path a human
# wrote into deslopped.txt is linted whatever it looks like -- otherwise a
# manuscript that happens to live under a pipeline-owned prefix (docs/) could
# never be ratcheted, which is the same silent-skip bug in the other mode.
collected=()
collect() {  # collect <sweep|explicit> <pathspec>...
    local sweep=$1
    shift
    collected=()
    local f
    while IFS= read -r -d '' f; do
        case "$f" in
            *.tex|*.TEX|*.md|*.MD|*.markdown|*.Markdown|*.qmd|*.QMD \
            |*.rmd|*.Rmd|*.RMD) ;;
            *) continue ;;
        esac
        if [ "$sweep" = sweep ] && [[ $f =~ $own_files ]]; then
            continue
        fi
        collected+=("$f")
    done < <(git -C "$root" -c core.quotePath=false ls-files -z -- "$@")
}

run_vale() {
    (cd "$root" && vale "${vale_args[@]}" "$@")
    local status=$?
    if [ "$status" -ge 2 ]; then
        echo "error: vale exited $status -- a runtime error, not an alert count." >&2
        echo "       $# file(s) were passed and none of them were linted." >&2
        exit "$status"
    fi
    return "$status"
}

if [ "$mode" = all ]; then
    collect sweep "${pathspec[@]}"
    if [ ${#collected[@]} -eq 0 ]; then
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
    run_vale "${collected[@]}"
    echo "(advisory: ${#collected[@]} file(s) linted; alerts here do not block)"
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
files=()
bad=0
while IFS= read -r spec; do
    spec=${spec#"${spec%%[![:space:]]*}"}        # trim leading whitespace
    spec=${spec%"${spec##*[![:space:]]}"}        # trim trailing whitespace
    [ -n "$spec" ] || continue
    collect explicit "$spec"
    if [ ${#collected[@]} -eq 0 ]; then
        echo "error: $list entry matches no tracked source under $root: $spec" >&2
        bad=1
        continue
    fi
    files+=("${collected[@]}")
done <<<"$specs"
[ "$bad" -eq 0 ] || exit 2

# Deduplicate without sort: paths may contain spaces, and the list is short.
unique=()
for f in ${files[@]+"${files[@]}"}; do
    dup=0
    for g in ${unique[@]+"${unique[@]}"}; do
        if [ "$f" = "$g" ]; then dup=1; break; fi
    done
    [ "$dup" -eq 0 ] && unique+=("$f")
done
if [ ${#unique[@]} -eq 0 ]; then
    echo "$list selects no tracked sources"
    exit 2
fi

echo "ratchet: ${#unique[@]} file(s) declared deslopped in $list; error-level alerts block"
run_vale "${unique[@]}"
exit $?
