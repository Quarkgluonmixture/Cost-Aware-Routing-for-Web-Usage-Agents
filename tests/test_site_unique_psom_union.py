from __future__ import annotations

from scripts.analysis.aggregate_phase1_full_prereg_decision import _psom_unique_ids


def _cell(psom: dict, others: dict) -> dict:
    """Build per_task dict: psom/others map task_id -> success value."""
    modes = ["DOM", "SoM", "Vision", "P-text", "P-prompt"]
    per_task = {"P-SoM": {t: {"success": s} for t, s in psom.items()}}
    for m in modes:
        per_task[m] = {t: {"success": others.get(t, {}).get(m, 0.0)} for t in psom}
    return per_task


def test_unique_requires_psom_success_and_all_others_fail():
    per_task = _cell(
        psom={"1": 1.0, "2": 1.0, "3": 0.0},
        others={
            "1": {},                # all five default 0.0 -> unique
            "2": {"SoM": 1.0},     # SoM also solves -> not unique
            "3": {},                # P-SoM itself failed -> not unique
        },
    )
    assert _psom_unique_ids(per_task) == {1}


def test_missing_or_none_success_excludes_task_fail_closed():
    per_task = _cell(psom={"7": 1.0}, others={"7": {}})
    per_task["Vision"]["7"]["success"] = None
    assert _psom_unique_ids(per_task) == set()
    del per_task["DOM"]["7"]
    assert _psom_unique_ids(per_task) == set()


def test_ids_returned_as_ints():
    per_task = _cell(psom={"42": 1.0}, others={"42": {}})
    assert _psom_unique_ids(per_task) == {42}
