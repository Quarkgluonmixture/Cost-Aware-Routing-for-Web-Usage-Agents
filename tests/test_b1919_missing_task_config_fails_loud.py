"""B-1919: a /diag scan with no task configs must not look like a clean scan.

`scan_episodes` used to fall back to `config = {}` when the per-episode task
config was absent — silently, with no error, no warning, and nothing in the
output JSON. Every rule that reads the config then returns no hits, so a run
whose config-dependent rules were never evaluated is byte-indistinguishable
from a run where those rules genuinely found nothing.

That is not hypothetical: six WA reddit conditions sat on disk for a fortnight
reporting "0 success-side hits" (which read as "the presence-only defences are
clean") while 27 of 44 rules were switched off, because `results/webarena/`
had no sync path and every WA run dir arrived with an empty `task_configs/`.

These tests pin the three properties that make that failure visible:
  1. missing config raises by default,
  2. `--allow-missing-config` downgrades to a warning but records the count,
  3. a complete run still scans clean with `config_missing == 0`.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "diag_pattern_match", REPO / "scripts/analysis/diag_pattern_match.py")
dpm = importlib.util.module_from_spec(_spec)
sys.modules["diag_pattern_match"] = dpm
_spec.loader.exec_module(dpm)


def _make_run(tmp_path, *, with_configs: bool, n: int = 2) -> Path:
    """Minimal run dir: one condition, n episodes, optional task_configs."""
    run = tmp_path / "B9_dom_reddit_20260101_TEST"
    ep = run / "phase1_dom_router_0" / "episodes"
    ep.mkdir(parents=True)
    cfg_dir = run / "task_configs"
    cfg_dir.mkdir()

    for tid in range(n):
        stem = f"reddit_task_{tid}"
        step = {
            "step_idx": 0, "task_id": tid, "run_id": "R_TEST",
            "condition_id": "phase1_dom_router_0", "benchmark_site": "reddit",
            "benchmark": "webarena", "observation_mode": "dom",
            "obs_url": "http://localhost:9999/", "page_changed": True,
            "action_success": True,
            "action": {"action_type": "click", "element_id": 5, "thought": "t"},
        }
        (ep / f"{stem}_steps_v2.jsonl").write_text(
            json.dumps(step) + "\n", encoding="utf-8")
        (ep / f"{stem}_summary_v2.json").write_text(json.dumps({
            "task_id": tid, "condition_id": "phase1_dom_router_0",
            "benchmark_site": "reddit", "success": False, "steps": 1,
            "agent_finished": False, "sr_excluded": False,
        }), encoding="utf-8")
        if with_configs:
            (cfg_dir / f"{stem}.json").write_text(json.dumps({
                "task_id": tid, "intent": "do a thing",
                "sites": ["reddit"], "start_url": "http://localhost:9999",
                "eval": {"eval_types": ["string_match"],
                         "reference_answers": {"must_include": ["x"]}},
            }), encoding="utf-8")
    return run


def test_missing_config_raises_by_default(tmp_path):
    """The whole point: silence is not an option."""
    run = _make_run(tmp_path, with_configs=False)
    with pytest.raises(dpm.MissingTaskConfigError) as exc:
        dpm.scan_episodes(run)
    msg = str(exc.value)
    assert "task config" in msg
    # the operator needs to know WHAT went dark, not just that something did
    assert "P31" in msg, "error must name the rules that would silently no-op"
    assert "reddit_task_0" in msg, "error must name a concrete missing stem"


def test_allow_missing_config_warns_and_records_count(tmp_path, capsys):
    """Escape hatch still leaves evidence in the artifact itself."""
    run = _make_run(tmp_path, with_configs=False, n=3)
    res = dpm.scan_episodes(run, allow_missing_config=True)
    assert res["config_missing"] == 3
    assert "WARNING [B-1919]" in capsys.readouterr().err


def test_complete_run_scans_clean(tmp_path):
    """A healthy run must not trip the new guard, and reports zero."""
    run = _make_run(tmp_path, with_configs=True)
    res = dpm.scan_episodes(run)
    assert res["config_missing"] == 0
    assert res["total_episodes"] == 2


def test_config_dependent_rules_is_derived_not_hardcoded():
    """The rule list must track ALL_RULES so it cannot drift as rules are added."""
    affected = dpm._config_dependent_rules()
    assert set(affected) <= set(dpm.ALL_RULES), "must be a subset of the registry"
    # sanity: the config-reading rules we know about are in there
    for rid in ("P31", "P40", "P46"):
        assert rid in affected, f"{rid} reads config but was not detected"
    # and a purely step-based rule is not
    assert "P36" not in affected, "P36 is step-only; false positives make the error noisy"


def test_shipped_wa_scans_have_zero_config_missing():
    """Regression pin for the actual incident, if the scans are on this host."""
    scans = sorted((REPO / "results/diag_scans/v8_wa").glob("B1_*_wa_reddit.json"))
    if not scans:
        pytest.skip("v8_wa scans not present on this host")
    for f in scans:
        d = json.loads(f.read_text(encoding="utf-8"))
        assert d.get("config_missing") == 0, (
            f"{f.name} was scanned without task configs — its config-dependent "
            "rules did not really run (B-1919)"
        )
