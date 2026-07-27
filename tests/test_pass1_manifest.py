"""Tests for canonical Pass-1/Pass-2 run discovery (router /stress B-1810 C2).

Guards against the bare-glob contamination where `*_smoke_*` / partial / stale runs
were silently folded into paper-grade labels + H10 (extract_50_features +
aggregate_h10_pareto both globbed `{baseline}_*_{site}_*` excluding only router_learned).
"""
import json

from p79.policies import pass1_manifest as pm


def test_b1810_is_non_canonical_flags_smoke_test_debug():
    assert pm.is_non_canonical("B0_dom_classifieds_smoke_20260521_R1")
    assert pm.is_non_canonical("B0_dom_reddit_test_R2")
    assert pm.is_non_canonical("B0_dom_classifieds_debug_R3")
    assert pm.is_non_canonical("B0_dom_reddit_dryrun_R4")
    # a real paper-grade run name is canonical
    assert not pm.is_non_canonical("B0_dom_classifieds_20260521_125142_R9755")


def test_b1810_discover_runs_rejects_smoke_and_router(tmp_path):
    root = tmp_path / "phase1"
    root.mkdir()
    canonical = root / "B0_dom_classifieds_20260521_R9755"
    smoke = root / "B0_dom_classifieds_smoke_20260521_R1"
    router = root / "B0_router_learned_classifieds_20260521_R2"
    for d in (canonical, smoke, router):
        d.mkdir()

    runs, prov = pm.discover_runs(root, "B0", "classifieds", router=False,
                             manifest_path=root / "_absent.json")
    names = [r.name for r in runs]
    assert canonical.name in names           # canonical kept
    assert smoke.name not in names           # smoke rejected (was silently included)
    assert router.name not in names          # router excluded from pass1 set
    assert prov["mode"] == "glob_reject_noncanonical"
    assert smoke.name in prov["rejected_non_canonical"]


def test_b1810_discover_runs_router_kind_selects_router(tmp_path):
    root = tmp_path / "phase1"
    root.mkdir()
    (root / "B0_dom_classifieds_20260521_R9755").mkdir()
    router = root / "B0_router_learned_classifieds_20260521_R2"
    router.mkdir()
    runs, _ = pm.discover_runs(root, "B0", "classifieds", router=True,
                        manifest_path=root / "_absent.json")
    assert [r.name for r in runs] == [router.name]


def test_b1810_discover_runs_manifest_whitelist(tmp_path):
    root = tmp_path / "phase1"
    root.mkdir()
    r1 = root / "B0_dom_classifieds_A_R1"
    r2 = root / "B0_dom_classifieds_B_R2"
    for d in (r1, r2):
        d.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"pass1": {"B0_classifieds": [r1.name]}}))

    runs, prov = pm.discover_runs(
        root, "B0", "classifieds", router=False, manifest_path=manifest
    )
    assert [r.name for r in runs] == [r1.name]   # only the whitelisted run
    assert prov["manifest_used"] is True
    assert prov["mode"] == "manifest"
    assert r2.name in prov["rejected_non_canonical"]


def test_b1810_discover_runs_multi_run_overwrite_warning(tmp_path):
    """No manifest + >1 canonical run → warn about newest-wins overwrite ambiguity."""
    root = tmp_path / "phase1"
    root.mkdir()
    for name in ("B0_dom_classifieds_A_R1", "B0_dom_classifieds_B_R2"):
        (root / name).mkdir()
    runs, prov = pm.discover_runs(root, "B0", "classifieds", router=False,
                             manifest_path=root / "_absent.json")
    assert len(runs) == 2
    assert any("overwritten" in w or "precedence" in w for w in prov["warnings"])
