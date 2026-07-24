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
