"""B-1894 — the shell-side expected-N tables must track `scored_task_count`.

Two bash files carry a hardcoded (site -> N) table. They cannot import Python
mid-chain, so the numbers stay literal; this test is what keeps them honest.

The bug being locked out: the 2026-05-17 EXPECTED_N sweep updated the three VWA
rows to post-§139.8 values and left the three WA rows at their pre-exclusion
totals, justified by a comment claiming "WA has no N/A taxonomy (per prereg)".
笔记 §137 / task #76 (2026-05-14) says the opposite in the same breath as the
per-site counts — "一条统一 config 规则, 无 per-site edge case", wa-shop 19 /
wa-admin 6 / wa-red 2 — and the runner has excluded them ever since. A
full-scale WA reddit paper-grade chain would have produced 104 episodes against
an expected 106 and aborted at the post-condition sentinel. It went unnoticed
because WA had only ever run as a 10-task pilot with the partial bypass set.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from p79.experiment.analysis import scored_task_count

REPO = Path(__file__).resolve().parents[1]

# (bash table key, site arg, benchmark arg)
SITE_KEYS = [
    ("classifieds", "classifieds", "visualwebarena"),
    ("reddit", "reddit", "visualwebarena"),
    ("shopping", "shopping", "visualwebarena"),
    ("wa_shopping", "shopping", "webarena"),
    ("wa_shopping_admin", "shopping_admin", "webarena"),
    ("wa_reddit", "reddit", "webarena"),
]

TABLES = [
    (REPO / "scripts" / "queues" / "queue_chain.sh", "SITE_EXPECTED_N"),
    (REPO / "scripts" / "maintenance" / "launch.sh", "SITE_N"),
]


def _parse_bash_table(path: Path, name: str) -> dict[str, int]:
    src = path.read_text(encoding="utf-8")
    m = re.search(rf"declare -A {name}=\((.*?)\)", src, re.S)
    assert m, f"{name} table not found in {path.name}"
    return {k: int(v) for k, v in re.findall(r"\[(\w+)\]=(\d+)", m.group(1))}


@pytest.mark.parametrize("path,name", TABLES, ids=lambda p: getattr(p, "name", p))
def test_bash_expected_n_matches_scored_task_count(path: Path, name: str):
    table = _parse_bash_table(path, name)
    expected = {k: scored_task_count(site, bench) for k, site, bench in SITE_KEYS}
    assert table == expected, (
        f"{path.name}:{name} drifted from scored_task_count.\n"
        f"  in file : {table}\n"
        f"  expected: {expected}\n"
        "Update the literal table; it gates a paper-grade chain."
    )


def test_wa_sites_do_have_na_tasks():
    """Pins the fact the stale comment denied. If a future task-file refresh
    removes every WA N/A task this test should be revisited, not deleted — the
    point is that WA is not exempt from the exclusion by construction."""
    from p79.experiment.analysis import _load_na_task_ids

    counts = {
        s: len(_load_na_task_ids(s, "webarena"))
        for s in ("reddit", "shopping", "shopping_admin")
    }
    # 笔记 §137 task #76 (2026-05-14) counted exactly these.
    assert counts == {"reddit": 2, "shopping": 19, "shopping_admin": 6}


def test_no_file_still_claims_wa_has_no_na_taxonomy():
    """The false justification is what let the numbers drift for two months, so
    the sentence itself is what gets locked out — a corrected number under a
    surviving wrong rationale invites the next author to 'fix' it back."""
    offenders = []
    for rel in ("scripts/queues/queue_chain.sh", "scripts/maintenance/launch.sh"):
        for i, line in enumerate((REPO / rel).read_text(encoding="utf-8").splitlines(), 1):
            if "no N/A taxonomy" in line and "B-1894" not in line and "WRONG" not in line:
                # the B-1894 correction block quotes the old claim to refute it
                if "was WRONG" not in line and "counterfactual" not in line:
                    offenders.append(f"{rel}:{i}")
    assert not offenders, (
        "stale 'WA has no N/A taxonomy' justification still present at: "
        f"{offenders} — see 笔记 §137 task #76 for the refuting counts"
    )


def test_paper_grade_check_wa_counts_match_the_launch_side_table():
    """B-1941 (/stress Mode A F7, 2026-08-03): the analysis side must agree.

    B-1933 added a THIRD copy of the WA scored counts, in
    `scripts/maintenance/paper_grade_check.py::WA_SCORED_FALLBACK`, because the
    Phase 1a fire manifest deliberately carries only the three VWA sites (adding
    WA rows would edit a preregistration artifact to describe runs it does not
    bind). A comment there says "keep the two in sync" — a comment is not a
    guard, which is exactly how the launch-side numbers drifted for two months
    before B-1894. Drift here is worse than cosmetic: the launch side would
    accept a condition at 173 episodes while the verification side calls the
    same run incomplete, or vice versa.
    """
    import ast

    src = (REPO / "scripts" / "maintenance" / "paper_grade_check.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fallback = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "WA_SCORED_FALLBACK" for t in node.targets
        ):
            fallback = ast.literal_eval(node.value)
    assert fallback is not None, "WA_SCORED_FALLBACK not found in paper_grade_check.py"

    launch_side = _parse_bash_table(REPO / "scripts" / "queues" / "queue_chain.sh", "SITE_EXPECTED_N")
    for site, n in fallback.items():
        assert site in launch_side, (
            f"paper_grade_check knows site {site!r} but queue_chain.sh's "
            f"SITE_EXPECTED_N does not — one side gained a site the other lacks"
        )
        assert launch_side[site] == n, (
            f"{site}: paper_grade_check says {n}, queue_chain.sh says "
            f"{launch_side[site]} — the launch side and the verification side "
            f"disagree on how many episodes make a complete condition"
        )
