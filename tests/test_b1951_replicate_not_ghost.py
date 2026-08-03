"""B-1951: a registered replicate arm is not a ghost — and must never be bound.

Context (实验笔记 §424): B0·cls·SoM was deliberately re-run to measure the
stochastic noise floor. Two earlier replicates (dom, vision) live under
`results/repro_replicates/` and are invisible to the ghost scanner by
construction; the SoM one was kept under `results/visualwebarena/phase1/`, where
the scanner sees it and — before this fix — could only call it a ghost, halting
every relaunch.

The inverse guard exists because the author of this fix first "resolved" that
halt by rebinding the manifest to the replicate. That is the one outcome that
destroys the measurement: canonical and replicate become the same run, and the
floor the pair exists to measure disappears.
"""
import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VALIDATOR = REPO / "scripts/analysis/validate_fire_manifest.py"
MANIFEST = REPO / "docs/checkpoints/pre_run/fire_manifest.json"

sys.path.insert(0, str(REPO / "scripts/analysis"))
from validate_fire_manifest import (  # noqa: E402
    registered_replicate_run_ids,
    registered_replicate_pairs,
)


def _clean_pairs():
    src = (REPO / "scripts/analysis/aggregate_noise_floor_inventory.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "CLEAN_PAIRS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("CLEAN_PAIRS not found — the replicate registry moved")


def test_registry_is_the_single_source():
    """The validator must derive replicates from CLEAN_PAIRS, not its own list."""
    pairs = _clean_pairs()
    prefix = "results/visualwebarena/phase1/"
    expected = {
        b[len(prefix):].split("/")[0] for _l, _a, b in pairs if b.startswith(prefix)
    }
    assert registered_replicate_run_ids() == frozenset(expected), (
        "validator's replicate set drifted from aggregate_noise_floor_inventory.CLEAN_PAIRS"
    )


def test_som_replicate_is_registered():
    """The concrete arm that answered §242 must be excused from ghost detection."""
    ids = registered_replicate_run_ids()
    assert any("R30696" in r for r in ids), (
        "B0.cls.som replicate no longer registered — the halt it caused would return"
    )
    labels = {lbl for lbl, _c, _r in registered_replicate_pairs()}
    assert "B0.cls.som" in labels


def test_clean_manifest_passes_and_names_the_replicate():
    r = subprocess.run([sys.executable, str(VALIDATOR)], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, f"clean manifest must pass, got {r.returncode}:\n{r.stdout}\n{r.stderr}"
    assert "registered replicate" in r.stdout, (
        "the excused run must be REPORTED, not silently skipped — a silent excuse is "
        "how a real ghost would hide"
    )


def test_binding_the_replicate_fails_closed():
    """THE load-bearing assertion: manifest must never name the replicate arm.

    Binding it collapses arm_a and arm_b onto the same run, so the noise floor
    (§424.1: net |ΔSR| 2.23pp against an exchangeability SD of 2.40pp) silently
    becomes a comparison of a run with itself.
    """
    pairs = registered_replicate_pairs()
    assert pairs, "no phase1/-resident replicate pairs — this test would be vacuous"
    _label, _canon, replicate = pairs[0]

    m = json.loads(MANIFEST.read_text())
    target = next(k for k, c in m["conditions"].items() if c.get("run_id") == _canon)
    m["conditions"][target]["run_id"] = replicate

    tmp = Path(tempfile.mkstemp(suffix=".json")[1])
    try:
        tmp.write_text(json.dumps(m))
        r = subprocess.run(
            [sys.executable, str(VALIDATOR), "--manifest", str(tmp)],
            capture_output=True, text=True, cwd=REPO,
        )
        assert r.returncode == 1, f"binding a replicate must fail-closed, got {r.returncode}"
        assert "REPLICATE arm as authoritative" in r.stdout
        assert "collapses the noise-floor comparison" in r.stdout
    finally:
        tmp.unlink()


def test_unreadable_registry_fails_closed():
    """If the registry cannot be parsed, every extra run stays a ghost."""
    import validate_fire_manifest as V

    orig = V.NOISE_INVENTORY
    try:
        V.NOISE_INVENTORY = REPO / "does_not_exist_xyz.py"
        assert V.registered_replicate_run_ids() == frozenset(), (
            "an unreadable registry must register NOTHING (fail closed), not everything"
        )
    finally:
        V.NOISE_INVENTORY = orig
