"""B-1913: keep scored RATES off the collected denominator in generated digests.

`write_digests.py` is legitimately a COLLECTED-set producer — a protocol-excluded
episode still exhibits a failure mode worth counting, which is why it sits in
`EPISODE_READER_EXEMPT` in `test_universe_consumption_lint.py`. What is NOT
legitimate is the handful of hardcoded narrative strings that quote a success
RATE against 205 instead of the 203-task scored set.

Two defects were fixed on 2026-07-27:
  * `B2_phantom_prompt`: "真实 SR = 0/205 = 0.00%" — scored rate, collected
    denominator. AMENDMENT_08 tier A had already removed task 160, so 0/203 is
    the canonical figure, not a digest-side correction.
  * `B1_som`: "扣除 task 160 后 som 7.80% vs dom 6.34%（17 vs 14 个成功, n=205）"
    — the numerator subtracted task 160 while the denominator stayed at 205, AND
    the parenthesised counts (17/14) contradicted the percentages themselves
    (7.80/6.34 = 16/205, 13/205). Same shape as the §387.9 6.37%-vs-6.40% error,
    i.e. its second instance, which is why this is now a test and not a note.

The guard is deliberately narrow: it looks for a PERCENTAGE adjacent to a /205
denominator, so genuine collected-set counts ("205 个 episode", "160/205 集",
"88/205 episode", "48/205 = 23.4%" no-hit coverage) stay allowed. Episode-level
coverage rates over the collected set are correct by construction; only success
rates must use the scored denominator.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WRITE_DIGESTS = REPO / "scripts" / "analysis" / "write_digests.py"
SR_JSON = REPO / "docs" / "analysis" / "cross_sites" / "sr_per_mode.json"

# "SR = 0/205", "SR 7.80% ... n=205" — a success rate tied to the collected N.
_SR_NEAR_205 = re.compile(
    r"(?:SR|success\s*rate)[^\n]{0,80}?/205\b|/205\s*=\s*\d+(?:\.\d+)?\s*%[^\n]{0,40}?(?:SR|真实)",
    re.IGNORECASE,
)


# A line may QUOTE the old wrong wording while explaining what was fixed. Those
# self-documenting lines are the opposite of the defect, so they are exempt —
# same convention as `test_no_manual_task_160_subtraction` below.
_HISTORICAL_MARKERS = ("旧写法", "B-1913", "pre-fix", "superseded")


def test_no_success_rate_on_collected_denominator() -> None:
    text = WRITE_DIGESTS.read_text(encoding="utf-8")
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        if any(m in line for m in _HISTORICAL_MARKERS):
            continue
        if _SR_NEAR_205.search(line):
            hits.append(f"{i}: {line.strip()[:120]}")
    assert not hits, (
        "a success RATE is quoted against the 205-task collected denominator; "
        "scored rates must use the AMENDMENT_08 scored set (reddit 203). "
        "Collected-set COVERAGE counts (episodes, walk_fail, no-hit) are fine — "
        "only rates are constrained (B-1913).\n  " + "\n  ".join(hits)
    )


def test_no_manual_task_160_subtraction() -> None:
    """AMENDMENT_08 removed task 160; subtracting it by hand double-counts."""
    text = WRITE_DIGESTS.read_text(encoding="utf-8")
    offenders = [
        f"{i}: {ln.strip()[:120]}"
        for i, ln in enumerate(text.splitlines(), 1)
        if "扣除 task 160" in ln and "旧写法" not in ln
    ]
    assert not offenders, (
        "task 160 is already outside the scored set (AMENDMENT_08 tier A), so a "
        "manual 'minus task 160' either double-subtracts or — as in the original "
        "B-1913 string — drops it from the numerator while leaving 205 in the "
        "denominator.\n  " + "\n  ".join(offenders)
    )


def test_quoted_b1_reddit_rates_match_canonical() -> None:
    """The B1 reddit figures in the narrative must equal sr_per_mode.json."""
    rows = json.loads(SR_JSON.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = next(v for v in rows.values() if isinstance(v, list))
    canon = {
        (r["baseline"], r["site"], r["mode"]): r
        for r in rows
    }
    text = WRITE_DIGESTS.read_text(encoding="utf-8")

    for mode, label in (("SoM", "som"), ("DOM", "dom"), ("Vision", "B1_vision")):
        row = canon[("B1", "reddit", mode)]
        pct = f"{row['sr_pct']:.2f}%"
        frac = f"({row['n_success']}/{row['sr_denominator_n']})"
        assert pct in text, (
            f"B1 reddit {mode} canonical SR {pct} not found in write_digests "
            f"narrative — the digest and sr_per_mode.json disagree (B-1913)."
        )
        assert frac in text, (
            f"B1 reddit {mode} canonical count {frac} not found in "
            f"write_digests narrative (B-1913)."
        )


def test_b2_phantom_prompt_is_zero_over_scored_set() -> None:
    rows = json.loads(SR_JSON.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = next(v for v in rows.values() if isinstance(v, list))
    row = next(
        r for r in rows
        if (r["baseline"], r["site"], r["mode"]) == ("B2", "reddit", "P-prompt")
    )
    # The arm's only success WAS task 160; post-AMENDMENT_08 it scores 0/203.
    assert row["n_success"] == 0, row
    assert row["sr_denominator_n"] == 203, row

    text = WRITE_DIGESTS.read_text(encoding="utf-8")
    assert "0/203" in text, (
        "B2 phantom_prompt narrative must state 0/203 (its single success was "
        "task 160, now outside the scored set) — B-1913."
    )
