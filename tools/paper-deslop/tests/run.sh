#!/usr/bin/env bash
# Self-test for the paper-deslop pipeline:
#   - Vale rules fire on the slop fixture and stay quiet on the faithful one;
#   - the lexical invariant gate passes the faithful rewrite;
#   - every single-fault adversarial fixture fails the gate with the
#     expected violation class (fixtures are copies of rewritten_good.tex
#     with exactly one planted fault each).
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0

echo "== vale reports errors on the slop fixture"
if vale --output=line --minAlertLevel=error tests/fixtures/slop.tex; then
    echo "FAIL: expected error-level alerts in slop.tex"
    fail=1
else
    echo "ok"
fi

echo "== vale is error-clean on the faithful rewrite"
if vale --output=line --minAlertLevel=error tests/fixtures/rewritten_good.tex; then
    echo "ok"
else
    echo "FAIL: rewritten_good.tex should have no error-level alerts"
    fail=1
fi

echo "== gate passes the faithful rewrite"
if python3 scripts/invariant_check.py tests/fixtures/slop.tex \
        tests/fixtures/rewritten_good.tex --terms terms.txt; then
    echo "ok"
else
    echo "FAIL: faithful rewrite should pass the lexical invariant gate"
    fail=1
fi

echo "== gate catches every adversarial fixture"
for f in tests/fixtures/bad_*.tex tests/fixtures/rewritten_bad.tex; do
    name=$(basename "$f" .tex)
    case "$name" in
        bad_number_swap)     expect="rebound number" ;;
        bad_sign_drop)       expect="removed number: '-0.8'" ;;
        bad_decimal_change)  expect="removed number: '.05'" ;;
        bad_unit_change)     expect="removed number: '5ms'" ;;
        bad_citation_rebind) expect="rebound citation" ;;
        bad_math_change)     expect="removed math" ;;
        bad_macro_change)    expect="removed macro" ;;
        rewritten_bad)       expect="removed citation" ;;
        *) echo "FAIL: no expectation registered for $name"; fail=1; continue ;;
    esac
    out=$(python3 scripts/invariant_check.py tests/fixtures/slop.tex "$f" --terms terms.txt)
    status=$?
    if [ "$status" -eq 0 ]; then
        echo "FAIL: $name should violate the gate"
        echo "$out"
        fail=1
    elif ! grep -qF "$expect" <<<"$out"; then
        echo "FAIL: $name: expected \"$expect\" in gate output, got:"
        echo "$out"
        fail=1
    else
        echo "ok: $name ($expect)"
    fi
done

echo "== markdown: gate passes a faithful rewrite (incl. a fully rewritten §8 sentence)"
if python3 scripts/invariant_check.py tests/fixtures/md_slop.md \
        tests/fixtures/md_good.md --terms terms.txt; then
    echo "ok"
else
    echo "FAIL: faithful Markdown rewrite should pass; structural pointers (§8,"
    echo "      Table 3) are references, not data, and repositioning a clause"
    echo "      around one is Pass 1 of the rewrite the skill prescribes"
    fail=1
fi

echo "== markdown: gate catches the faults '%'-as-comment used to hide"
for f in tests/fixtures/md_bad_*.md; do
    name=$(basename "$f" .md)
    case "$name" in
        # Both of these sit after a '%' on their line. Under LaTeX comment
        # rules the rest of the line vanished and the gate reported OK.
        md_bad_percent)   expect="removed number: '0.9'" ;;
        md_bad_cite)      expect="removed citation: 'liu2023'" ;;
        md_bad_structref) expect="removed structref: 'sec:8'" ;;
        *) echo "FAIL: no expectation registered for $name"; fail=1; continue ;;
    esac
    out=$(python3 scripts/invariant_check.py tests/fixtures/md_slop.md "$f" --terms terms.txt)
    status=$?
    if [ "$status" -eq 0 ]; then
        echo "FAIL: $name should violate the gate"
        echo "$out"
        fail=1
    elif ! grep -qF "$expect" <<<"$out"; then
        echo "FAIL: $name: expected \"$expect\" in gate output, got:"
        echo "$out"
        fail=1
    else
        echo "ok: $name ($expect)"
    fi
done

echo "== whitelist term matching is word-bounded"
if python3 - <<'PY'
import sys
sys.path.insert(0, "scripts")
from invariant_check import term_counts

# (text, term, expected count). Each case is a false-positive or
# false-negative class that word-boundary matching has to get right.
cases = [
    ("random dominance dominant domain randomly", "DOM", 0),
    ("The DOM tree; DOMs and DOM-based parsing, the DOM's root", "DOM", 4),
    ("some sometimes somebody handsome", "SoM", 0),
    ("SoM marks and SoMs", "SoM", 2),
    ("premature terminations happen", "premature termination", 1),
    ("we let us know about US policy", "US", 1),           # acronym: case-sensitive
    ("Representation routing and representation routing", "representation routing", 2),
    ("a cost-accuracy\ntrade-off and cost accuracy trade off",
     "cost-accuracy trade-off", 2),                        # wrap- and hyphen-safe
    ("Transformers and the transformer block", "Transformer", 2),
]
bad = 0
for text, term, want in cases:
    got = term_counts(text, [term]).get(term, 0)
    if got != want:
        print(f"  term {term!r}: want {want} occurrence(s), got {got}")
        bad += 1
sys.exit(1 if bad else 0)
PY
then
    echo "ok"
else
    echo "FAIL: term matching regressed (see above)"
    fail=1
fi

echo "== ratchet lint blocks only what deslopped.txt declares"
printf '# only comments\n\n'                      > /tmp/ratchet_empty.txt
printf 'tests/fixtures/rewritten_good.tex\n'      > /tmp/ratchet_clean.txt
printf 'tests/fixtures/slop.tex\n'                > /tmp/ratchet_dirty.txt
printf 'no/such/section.tex\n'                    > /tmp/ratchet_typo.txt
check_ratchet() {  # name, list file, expected exit code
    # --root is required, not decorative: these lists hold pathspecs relative
    # to the PIPELINE directory, and without it a vendored install resolves
    # them against the manuscript repo root and every entry misses.
    out=$(bash scripts/ratchet_lint.sh --root "$PWD" --list "$2" --output=line 2>&1)
    code=$?
    if [ "$code" -eq "$3" ]; then
        echo "ok: $1 (exit $code)"
    else
        echo "FAIL: $1 expected exit $3, got $code"
        echo "$out"
        fail=1
    fi
}
check_ratchet "empty list blocks nothing" /tmp/ratchet_empty.txt 0
check_ratchet "clean file passes"         /tmp/ratchet_clean.txt 0
check_ratchet "regression blocks"         /tmp/ratchet_dirty.txt 1
check_ratchet "unmatched entry fails loudly (never a silent no-op)" \
    /tmp/ratchet_typo.txt 2

echo "== vendored install: pipeline in a subdirectory, manuscript in docs/"
# The regression: all_sources() used to hard-code an exclusion of docs/ and to
# assume the pipeline sat at the repo root, so a repo that keeps its paper in
# docs/ linted exactly zero files and reported success.
vendor_parent=$(mktemp -d)
vendor="$vendor_parent/vendor"
mkdir -p "$vendor/tools/deslop/scripts" "$vendor/docs/literature/5.1"
git init -q "$vendor" 2>/dev/null
cp .vale.ini "$vendor/tools/deslop/"
cp -R styles "$vendor/tools/deslop/"
cp scripts/ratchet_lint.sh "$vendor/tools/deslop/scripts/"
cp tests/fixtures/md_slop.md "$vendor/docs/paper.md"   # has error-level alerts
cp tests/fixtures/md_good.md "$vendor/docs/clean.md"   # error-clean
# Real manuscripts have paths like these. A space used to be word-split into
# arguments that do not exist, and a non-ASCII name came back from git
# octal-quoted ("\346\210\220...md"); vale answers either with E100 and exit
# 2, which advisory mode then reported as a clean run over zero files.
cp tests/fixtures/md_slop.md "$vendor/docs/literature/5.1/Cost-Aware Routing 5.1.md"
cp tests/fixtures/md_slop.md "$vendor/docs/成本感知路由.md"
git -C "$vendor" add -- docs tools
vendor_lint="$vendor/tools/deslop/scripts/ratchet_lint.sh"

out=$(bash "$vendor_lint" --all --output=line 2>&1)
if grep -q 'docs/paper.md' <<<"$out"; then
    echo "ok: advisory lint reaches a manuscript in docs/"
else
    echo "FAIL: vendored --all found no manuscript (silent zero-file lint)"
    echo "$out"
    fail=1
fi

printf 'docs/clean.md\n' > "$vendor/deslopped.txt"
git -C "$vendor" add -- deslopped.txt
bash "$vendor_lint" >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "ok: root deslopped.txt is honoured from a vendored pipeline"
else
    echo "FAIL: clean ratcheted file should pass from a vendored install"
    fail=1
fi

printf 'docs/paper.md\n' > "$vendor/deslopped.txt"
git -C "$vendor" add -- deslopped.txt
bash "$vendor_lint" >/dev/null 2>&1
if [ $? -eq 1 ]; then
    echo "ok: regression in a ratcheted docs/ file blocks"
else
    echo "FAIL: dirty ratcheted file should exit 1 from a vendored install"
    fail=1
fi

PAPER_PATHSPEC='docs/nowhere/*.md' bash "$vendor_lint" --all >/dev/null 2>&1
if [ $? -eq 2 ]; then
    echo "ok: PAPER_PATHSPEC matching nothing fails loudly"
else
    echo "FAIL: a PAPER_PATHSPEC that matches nothing must not look like success"
    fail=1
fi

out=$(bash "$vendor_lint" --all --output=line 2>&1)
code=$?
if [ "$code" -eq 0 ] \
   && grep -q 'Cost-Aware Routing 5.1.md' <<<"$out" \
   && grep -q '成本感知路由.md' <<<"$out"; then
    echo "ok: paths with spaces and non-ASCII names reach vale intact"
else
    echo "FAIL: --all exited $code and did not lint the awkward paths; a space"
    echo "      was word-split, or git octal-quoted the non-ASCII name"
    printf '%s\n' "$out" | head -5
    fail=1
fi

printf 'docs/literature/5.1/Cost-Aware Routing 5.1.md\n' > "$vendor/deslopped.txt"
git -C "$vendor" add -- deslopped.txt
bash "$vendor_lint" >/dev/null 2>&1
if [ $? -eq 1 ]; then
    echo "ok: a ratcheted path with spaces reports alerts (1), not a crash (2)"
else
    echo "FAIL: a ratcheted path containing spaces must reach vale as one argument"
    fail=1
fi

# A vale runtime error (bad config, unreadable path) must never be laundered
# into a clean advisory run. This is the swallow that hid 394 unlinted files.
cp "$vendor/tools/deslop/.vale.ini" "$vendor_parent/vale.ini.bak"
printf 'StylesPath = styles\n\n[*.md]\nBasedOnStyles = NoSuchStyle\n' \
    > "$vendor/tools/deslop/.vale.ini"
bash "$vendor_lint" --all >/dev/null 2>&1
if [ $? -ge 2 ]; then
    echo "ok: a vale runtime error surfaces instead of exiting 0"
else
    echo "FAIL: advisory mode swallowed a vale crash and reported success"
    fail=1
fi
cp "$vendor_parent/vale.ini.bak" "$vendor/tools/deslop/.vale.ini"

# Pipeline-relative pathspecs (what the self-test itself uses) need --root.
mkdir -p "$vendor/tools/deslop/tests/fixtures"
cp tests/fixtures/md_good.md "$vendor/tools/deslop/tests/fixtures/"
git -C "$vendor" add -- tools
printf 'tests/fixtures/md_good.md\n' > "$vendor_parent/pipeline_list.txt"
bash "$vendor_lint" --root "$vendor/tools/deslop" \
    --list "$vendor_parent/pipeline_list.txt" >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "ok: --root makes pipeline-relative list entries resolve"
else
    echo "FAIL: --root DIR should resolve list entries relative to DIR"
    fail=1
fi
bash "$vendor_lint" --list "$vendor_parent/pipeline_list.txt" >/dev/null 2>&1
if [ $? -eq 2 ]; then
    echo "ok: the same list without --root fails loudly instead of passing empty"
else
    echo "FAIL: unresolvable list entries must exit 2, never 0"
    fail=1
fi
rm -rf "$vendor_parent"

echo "== vocabulary file is in sync with terms.txt"
before=$(cat styles/config/vocabularies/Paper/accept.txt 2>/dev/null || true)
python3 scripts/gen_vale_vocab.py >/dev/null
after=$(cat styles/config/vocabularies/Paper/accept.txt)
if [ "$before" = "$after" ]; then
    echo "ok"
else
    echo "FAIL: styles/config/vocabularies/Paper/accept.txt is stale; commit the regenerated file"
    fail=1
fi

exit $fail
