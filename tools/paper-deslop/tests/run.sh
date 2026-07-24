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
    # --root .: the fixtures are pathspecs inside the pipeline dir, not the
    # enclosing repo (P79 vendors this under tools/paper-deslop/).
    out=$(bash scripts/ratchet_lint.sh --root . --list "$2" --output=line 2>&1)
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

echo "== markdown percent signs do not blind the gate"
# Regression: `%` is a LaTeX comment but a percent sign in Markdown. When the
# LaTeX rule was applied unconditionally, everything after "81%" on a line was
# stripped before the number/citation/term checks ran, so drift hiding behind a
# percent sign was invisible (it surfaced only as an unrelated "comment" alert).
out=$(python3 scripts/invariant_check.py tests/fixtures/md_percent_base.md \
        tests/fixtures/md_percent_drift.md --terms terms.txt)
if [ $? -eq 0 ]; then
    echo "FAIL: post-% number drift should violate the gate"
    echo "$out"
    fail=1
elif ! grep -qF "removed number: '42.0'" <<<"$out"; then
    echo "FAIL: expected the drift to be reported as a NUMBER violation, got:"
    echo "$out"
    fail=1
elif ! grep -qE "numbers: +[0-9]+ violation\(s\) \(5 in old\)" <<<"$out"; then
    # 5 = 4.2 (heading) + 81 + 19 + 42.0 + 3.1. Pre-fix only the first two were
    # visible; everything after the first `%` was stripped as a LaTeX comment.
    echo "FAIL: expected all 5 numbers visible to the checker, got:"
    echo "$out"
    fail=1
else
    echo "ok"
fi

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
