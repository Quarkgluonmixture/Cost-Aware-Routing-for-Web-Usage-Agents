"""B-1675 regression test: capture_doi1_witness.sh canonical schema invariant.

Protects against B-1670 pattern bug recurrence (capture script using
`*_summary.json` instead of canonical `*_summary_v2.json` per
`p79/experiment/logger_v2.py:111+114` + `analysis.py:209`).

Test strategy:
1. Create temp directory mimicking VWA run dir structure with canonical
   episode_summary_v2.json + steps_v2.jsonl + condition_summary_v2.json files.
2. Invoke capture_doi1_witness.sh with --run-dir-pattern pointing at temp dir.
3. Assert script reports correct nonzero counts for each tier.
4. Test known-positive probe: temp dir with WRONG-pattern files only.
   Script should detect schema-mismatch (target=0 + any-files>0) and exit 3.
5. Test empty dir: target=0 + any-files=0 → genuine pre-outcome-creation status.

Mirrors test_io_utils_strict_load.py style (per codex Mode B B-WIT-1 recommendation
2026-05-18 /stress witness pattern bug retraction wave).
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "maintenance" / "capture_doi1_witness.sh"


def _make_canonical_run_dir(base: Path, condition: str, n_episodes: int = 3) -> Path:
    """Create a VWA-like run dir with CANONICAL schema files.

    Layout:
        <base>/<condition>/episodes/
            classifieds_task_0_summary_v2.json
            classifieds_task_0_steps_v2.jsonl
            classifieds_task_1_summary_v2.json
            classifieds_task_1_steps_v2.jsonl
            ...
        <base>/<condition>/condition_summary_v2.json  (if condition complete)
    """
    run_dir = base / condition
    episodes_dir = run_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_episodes):
        summary_path = episodes_dir / f"classifieds_task_{i}_summary_v2.json"
        summary_path.write_text(json.dumps({"task_id": i, "success": True}))

        steps_path = episodes_dir / f"classifieds_task_{i}_steps_v2.jsonl"
        steps_path.write_text(json.dumps({"step": 0, "action": "click"}) + "\n")

    return run_dir


def _make_wrong_pattern_run_dir(base: Path, condition: str, n_episodes: int = 3) -> Path:
    """Create a run dir with WRONG-pattern files (B-1670 style schema mismatch).

    Files use suffixless schema `*_summary.json` / `*_steps.jsonl`
    instead of canonical `*_summary_v2.json` / `*_steps_v2.jsonl`.
    """
    run_dir = base / condition
    episodes_dir = run_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_episodes):
        # B-1670-style WRONG patterns
        (episodes_dir / f"classifieds_task_{i}_summary.json").write_text("{}")
        (episodes_dir / f"classifieds_task_{i}_steps.jsonl").write_text("{}")

    return run_dir


def _invoke_script(tmpdir: Path, pattern: str, label: str = "test") -> dict:
    """Invoke capture_doi1_witness.sh locally + return parsed status.

    Returns dict with keys: rc, stdout, stderr, outfile (Path | None),
    counts (dict of tier counts), status_line.
    """
    outfile = tmpdir / f"witness_{label}.txt"
    cmd = [
        "bash",
        str(SCRIPT_PATH),
        "--run-dir-pattern", pattern,
        "--output", str(outfile),
        "--label", label,
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    result = {
        "rc": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "outfile": outfile if outfile.exists() else None,
        "counts": {},
        "status_line": "",
    }

    if outfile.exists():
        content = outfile.read_text()
        for line in content.splitlines():
            if line.startswith("episode_summary_v2_count:"):
                result["counts"]["episode_summary_v2"] = int(line.split(":")[1].strip())
            elif line.startswith("episode_steps_v2_count:"):
                result["counts"]["episode_steps_v2"] = int(line.split(":")[1].strip())
            elif line.startswith("condition_summary_v2_count:"):
                result["counts"]["condition_summary_v2"] = int(line.split(":")[1].strip())
            elif line.startswith("any_json_in_run_dirs:"):
                result["counts"]["any_json"] = int(line.split(":")[1].strip())
            elif line.startswith("any_files_in_run_dirs:"):
                result["counts"]["any_files"] = int(line.split(":")[1].strip())
            elif line.startswith("STATUS:"):
                result["status_line"] = line

    return result


def test_capture_script_exists_and_executable():
    """Sanity: script file exists at expected path + is executable bash."""
    assert SCRIPT_PATH.exists(), f"capture script missing: {SCRIPT_PATH}"
    assert SCRIPT_PATH.is_file()
    # bash -n syntax check
    proc = subprocess.run(["bash", "-n", str(SCRIPT_PATH)], capture_output=True)
    assert proc.returncode == 0, f"bash syntax check failed: {proc.stderr.decode()}"


def test_canonical_schema_detected_correctly(tmp_path):
    """B-1675 core invariant: canonical *_summary_v2.json files MUST be counted.

    Creates run dir with canonical schema; asserts script reports nonzero counts.
    This guards against B-1670 pattern bug recurrence (suffixless `*_summary.json`).
    """
    _make_canonical_run_dir(tmp_path, "B0_dom_classifieds_test", n_episodes=3)

    result = _invoke_script(tmp_path, str(tmp_path / "B0_*"))

    assert result["rc"] == 0, (
        f"Script failed unexpectedly. stdout={result['stdout']}, stderr={result['stderr']}"
    )
    assert result["counts"]["episode_summary_v2"] == 3, (
        f"Expected 3 summary_v2 files, got {result['counts']['episode_summary_v2']}. "
        f"If 0 → CRITICAL: script using wrong pattern (B-1670 regression)"
    )
    assert result["counts"]["episode_steps_v2"] == 3
    # condition_summary_v2 not created in this test fixture
    assert result["counts"]["condition_summary_v2"] == 0


def test_schema_mismatch_detected(tmp_path):
    """B-1675 P1-4 known-positive probe: WRONG-pattern files trigger schema-mismatch exit.

    Creates run dir with B-1670-style suffixless files. Script should report
    target=0 (no canonical match) AND any-files>0 (files exist) → exit 3.
    """
    _make_wrong_pattern_run_dir(tmp_path, "B0_dom_classifieds_test", n_episodes=3)

    result = _invoke_script(tmp_path, str(tmp_path / "B0_*"))

    # Canonical counts all 0 (because files use wrong schema)
    assert result["counts"]["episode_summary_v2"] == 0
    assert result["counts"]["episode_steps_v2"] == 0
    # But any-files probe sees the wrong-pattern files
    assert result["counts"]["any_files"] > 0
    # Status detected mismatch
    assert "SCHEMA-MISMATCH-SUSPECTED" in result["status_line"], (
        f"Expected SCHEMA-MISMATCH-SUSPECTED status, got: {result['status_line']}"
    )
    # Exit code 3 per script doc
    assert result["rc"] == 3


def test_empty_run_dir_is_pre_outcome_creation(tmp_path):
    """Empty run dir (or no-files dir) returns pre-outcome-creation status.

    Distinguishes from schema-mismatch (where files exist with wrong pattern).
    """
    # Create EMPTY run dir (no files at all)
    (tmp_path / "B0_dom_classifieds_test").mkdir()

    result = _invoke_script(tmp_path, str(tmp_path / "B0_*"))

    assert result["rc"] == 0
    assert result["counts"]["episode_summary_v2"] == 0
    assert result["counts"]["episode_steps_v2"] == 0
    assert result["counts"]["condition_summary_v2"] == 0
    assert result["counts"]["any_files"] == 0
    assert "pre-outcome-creation" in result["status_line"]


def test_post_outcome_creation_tier_downgrade(tmp_path):
    """When episode summaries exist (no condition aggregate), tier downgrades to
    pre-outcome-inspection (not pre-outcome-creation).
    """
    _make_canonical_run_dir(tmp_path, "B0_dom_test", n_episodes=1)

    result = _invoke_script(tmp_path, str(tmp_path / "B0_*"))

    assert result["rc"] == 0
    assert result["counts"]["episode_summary_v2"] == 1
    # Tier should be inspection (per-episode present, no condition aggregate)
    assert (
        "pre-outcome-inspection" in result["status_line"]
        or "post-outcome-creation" in result["status_line"]
    ), f"Expected tier downgrade, got: {result['status_line']}"


def test_condition_summary_present_triggers_pre_analysis_tier(tmp_path):
    """When condition_summary_v2.json exists, tier downgrades to pre-analysis."""
    run_dir = _make_canonical_run_dir(tmp_path, "B0_dom_test", n_episodes=3)
    # Add condition-level aggregate
    (run_dir / "condition_summary_v2.json").write_text(
        json.dumps({"success_rate": 0.5, "n_tasks": 3})
    )

    result = _invoke_script(tmp_path, str(tmp_path / "B0_*"))

    assert result["rc"] == 0
    assert result["counts"]["condition_summary_v2"] == 1
    assert (
        "pre-analysis" in result["status_line"]
        or "post-outcome-aggregation" in result["status_line"]
    )


def test_witness_file_contains_provenance_footer(tmp_path):
    """Witness file MUST contain canonical pattern citations + git provenance + SHA-256."""
    _make_canonical_run_dir(tmp_path, "B0_dom_test", n_episodes=2)

    result = _invoke_script(tmp_path, str(tmp_path / "B0_*"))

    assert result["outfile"] is not None
    content = result["outfile"].read_text()

    # Canonical pattern citation
    assert "logger_v2.py:111" in content
    assert "_summary_v2.json" in content
    assert "_steps_v2.jsonl" in content

    # B-1675 known-positive probe documented
    assert "Known-positive sanity probe" in content
    assert "B-1675" in content

    # Cross-reference footer
    assert "DOI 1 README" in content
    assert "B-1670" in content  # Reference to original retraction context

    # SHA-256 self-doc
    assert "SHA-256" in content
    assert "## SHA-256 self-doc" in content


def test_dry_run_does_not_create_outfile(tmp_path):
    """--dry-run flag does NOT write outfile but reports intended action."""
    outfile = tmp_path / "should_not_exist.txt"
    cmd = [
        "bash", str(SCRIPT_PATH),
        "--dry-run",
        "--output", str(outfile),
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)

    assert proc.returncode == 0
    assert "[DRY-RUN]" in proc.stdout
    assert not outfile.exists()


def test_help_text_documents_canonical_patterns():
    """--help MUST cite canonical schema source (logger_v2.py / analysis.py)
    so future operators don't reinvent wrong patterns from mental model.
    """
    proc = subprocess.run(
        ["bash", str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    help_text = proc.stdout

    # Canonical schema sources cited
    assert "logger_v2.py" in help_text
    assert "analysis.py:209" in help_text or "analysis.py" in help_text

    # B-1670 retraction context referenced
    assert "B-1670" in help_text


if __name__ == "__main__":
    # Allow standalone invocation (not via pytest)
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
