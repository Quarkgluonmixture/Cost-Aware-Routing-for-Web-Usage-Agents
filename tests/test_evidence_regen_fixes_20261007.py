"""Regressions found by the 2026-10-07 full evidence-layer recompute (实验笔记 §536).

Each test names the bug it would have caught:
- routing_ceiling read the six WebArena pairs (`B1.wared.*`, registered 10-06) as VWA reddit,
  widening red_B1's rerun band to 0–7.69pp (a WA denominator).
- cross_mode_failure_signatures resolved 41 targets (B5 leaked in via `_discover_cls`, 09-11),
  and on 08-03 its newest-by-mtime fallback had read a still-running replicate as canonical.
- lib/atomic_io imported fcntl unconditionally, so its producers could not start on Windows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))


def test_routing_ceiling_keeps_wa_pairs_out_of_vwa_reddit(tmp_path, monkeypatch):
    import aggregate_routing_ceiling as rc
    floor = {"margins": {"x": {}}, "clean_pairs": [
        {"label": "B1.red.dom", "self_drop_a_to_b_pp": 0.49, "self_drop_b_to_a_pp": 1.97},
        {"label": "B1.wared.dom", "self_drop_a_to_b_pp": 1.92, "self_drop_b_to_a_pp": 7.69},
    ]}
    p = tmp_path / "noise_floor_inventory.json"
    p.write_text(json.dumps(floor), encoding="utf-8")
    monkeypatch.setattr(rc, "FLOOR_JSON", p)
    got = rc.load_floor()["reruns"]
    assert got["red_B1"] == [0.49, 1.97]
    assert got["wa_red_B1"] == [1.92, 7.69]


def test_routing_ceiling_unknown_site_label_fails_loud(tmp_path, monkeypatch):
    import pytest
    import aggregate_routing_ceiling as rc
    p = tmp_path / "noise_floor_inventory.json"
    p.write_text(json.dumps({"margins": {"x": {}}, "clean_pairs": [
        {"label": "B1.shop.som", "self_drop_a_to_b_pp": 1.0}]}), encoding="utf-8")
    monkeypatch.setattr(rc, "FLOOR_JSON", p)
    with pytest.raises(rc.MissingInput):
        rc.load_floor()


def test_cross_mode_targets_are_exactly_the_36_preregistered_conditions():
    import aggregate_cross_mode_failure_signatures as xm
    targets = xm.resolve_targets()
    assert len(targets) == 36
    assert not [k for k in targets if not k.startswith(("B0_", "B1_", "B2_"))]


def test_atomic_io_writes_and_locks_on_this_platform(tmp_path):
    from scripts.analysis.lib.atomic_io import atomic_write_text, exclusive_file_lock
    with exclusive_file_lock(tmp_path / "x.lock"):
        atomic_write_text(tmp_path / "a.txt", "ok")
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "ok"
