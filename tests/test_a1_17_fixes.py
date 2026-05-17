"""/stress A1.17 cold-start fix anti-regression tests (B-744 ~ B-760).

Covers paper-grade contracts surfaced by the 3-AI cold-start audit
(Claude Mode A + codex Mode B + gemini Mode C) on the VWA setup substrate +
RESET_BEFORE protocol:

- B-744 P0-1 AB* OOB: `:?` source-time hardstop removed (A100 local mode unblock)
- B-745 P0-2 B* OOB:  site-aware reset timeout (reddit 240s > callee 180s)
- B-746 P0-3 C* OOB:  5-table sentinel + PHP cache + session cleanup (D')
- B-747 P1-4 AB* OOB: MYSQL_PWD env injection (5 callsites, B-717 sibling)
- B-748 P1-5+18 B* OOB: awk indexer (pipefail crash + empty-status false-Ready)
- B-749 P1-6 B:        substrate-missing FATAL (was warning-and-success)
- B-750 P1-7 B:        cls archive .zip + .tar.gz dual-format
- B-751 P1-8 AB* OOB:  template render for hostname patching (idempotent on change)
- B-752 P1-15 A:       cls compose-name broader regex (no accidental DB re-seed)
- B-753 P1-10 C* OOB:  P79_VWA_TZ unified env (was QUARK_TZ / VWA_REDDIT_TZ asymmetry)
- B-754 P1-9 B* OOB:   PAPER_GRADE_ALLOW_PARTIAL hard-block under PG=1
- B-755 P1-12 AC* OOB: assert_a100_url_locality logic invert (PG=1 default-on)
- B-756 P1-11 C:       Dirty Cell Backdoor FATAL (pre-existing runner under PG+RESET)
- B-757 P1-14 A:       CLASSIFIEDS_RESET_TOKEN from env / .auth/ (not literal)
- B-758 P1-13 C:       queue_chain sentinel parser FATAL on parse failure
- B-759 P1-16 ABC:     asset publish atomic .tmp + sha256 manifest scaffold
- B-760 P1-17 AC* OOB: REQ_GB 250 → 350 (mathematical underestimate)
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _strip_shell_comments(src: str) -> str:
    """Strip full-line `# ...` comments so contract greps don't trip on
    pre-fix literals quoted in B-### audit comments. Inline `#` (mid-line
    comment) is rare in bash; we tolerate ambiguity by stripping only lines
    where `#` is the first non-whitespace character."""
    return "\n".join(
        line for line in src.splitlines()
        if not re.match(r"^\s*#", line)
    )


RESET = REPO_ROOT / "scripts" / "maintenance" / "reset_vwa_sites.sh"
GATES = REPO_ROOT / "scripts" / "queues" / "_lib_paper_grade_gates.sh"
CHAIN = REPO_ROOT / "scripts" / "queues" / "queue_chain.sh"
BASELINE = REPO_ROOT / "scripts" / "queues" / "queue_baseline.sh"
SETUP = REPO_ROOT / "scripts" / "vwa" / "setup_vwa.sh"
START = REPO_ROOT / "scripts" / "vwa" / "start_vwa_docker.sh"
IMPORT_ = REPO_ROOT / "scripts" / "vwa" / "import_vwa_assets.sh"
A100 = REPO_ROOT / "scripts" / "setup" / "a100_self_host_vwa.sh"
MANIFEST = REPO_ROOT / "docs" / "reference" / "vwa_assets_manifest.json"


# ─── B-744 P0-1 AB* OOB: `:?` source-time hardstop removed ─────────────────
def test_b744_no_source_time_required_param_expansion():
    """`VWA_RESET_SSH_HOST` must not have `:?` requirement at source time.

    Pre-fix `${VWA_RESET_SSH_HOST:?...}` ran at script-load → A100 local mode
    (which doesn't need SSH host) collapsed every `source reset_vwa_sites.sh`.
    Post-fix uses `:-` empty default; validation pushed into remote branch.
    """
    src = _strip_shell_comments(RESET.read_text())
    assert "VWA_RESET_SSH_HOST:?" not in src, (
        "B-744 regression: source-time `:?` still requires VWA_RESET_SSH_HOST"
    )
    assert 'VWA_RESET_SSH_HOST="${VWA_RESET_SSH_HOST:-}"' in src, (
        "B-744 contract: empty-default `:-` must be used at script top"
    )


def test_b744_source_succeeds_without_ssh_host(tmp_path):
    """Sourcing reset_vwa_sites.sh under empty VWA_RESET_SSH_HOST + set -euo
    pipefail must NOT exit (the original A100 local-mode breakage)."""
    result = subprocess.run(
        ["bash", "-c", f"set -euo pipefail; unset VWA_RESET_SSH_HOST; source {RESET}; echo SOURCED"],
        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert result.returncode == 0, (
        f"B-744 regression: source failed under unset VWA_RESET_SSH_HOST rc={result.returncode}\nstderr: {result.stderr}"
    )
    assert "SOURCED" in result.stdout


# ─── B-745 P0-2 B* OOB: site-aware reset timeout ───────────────────────────
def test_b745_site_aware_timeout_exists():
    """`reset_and_auth_gate` must dispatch timeout by site (reddit gets ≥180s)."""
    src = GATES.read_text()
    # The case branch picks per-site timeout.
    assert re.search(r"case\s+\"\$\{site\}\"\s+in[^}]*reddit\)\s+_reset_timeout=2[34]0", src, re.DOTALL), (
        "B-745 contract: case block must assign reddit reset_timeout ≥ 240s"
    )
    assert "_reset_timeout=120" in src
    # Reddit must exceed the callee warm-up max (60 iters * 3s = 180s).
    m = re.search(r"reddit\)\s+_reset_timeout=(\d+)", src)
    assert m, "B-745 contract: reddit branch missing"
    assert int(m.group(1)) >= 180 + 30, (
        f"B-745 regression: reddit timeout {m.group(1)} < callee 180s + 30s buffer"
    )


# ─── B-746 P0-3 C* OOB: 5-table sentinel + PHP cache + session (D') ────────
def test_b746_sentinel_expanded_to_five_tables():
    src = RESET.read_text()
    for table in ("oc_t_item_comment", "oc_t_item", "oc_t_user", "oc_t_alerts", "oc_t_latest_searches"):
        assert table in src, (
            f"B-746 contract: sentinel must verify table {table!r} (gemini sentinel-theater defuse)"
        )


def test_b746_php_app_cache_cleanup():
    """Post-reset must rm OSClass file cache (oc-content/cache + runtime)."""
    src = RESET.read_text()
    assert "oc-content/cache" in src, "B-746b: PHP file cache cleanup missing"
    assert "oc-content/runtime" in src, "B-746b: PHP runtime cleanup missing"


def test_b746_php_session_cleanup():
    """Post-reset must rm PHP session files (tmpfs + custom session dir)."""
    src = RESET.read_text()
    assert "sess_*" in src, "B-746b: session file cleanup missing (find -name sess_*)"


# ─── B-747 P1-4 AB* OOB: MYSQL_PWD env (B-717 sibling) ─────────────────────
def test_b747_no_mysql_password_in_argv():
    """No `-pXxxx` mysql argv password pattern in any reset/start file (executable
    code only — comments may still reference pre-fix patterns for context)."""
    for path in (RESET, START):
        src = _strip_shell_comments(path.read_text())
        forbidden_patterns = [r"mysql\s+-u\s*\w+\s+-p\w+", r"-pMyPassword", r"-ppassword(?!\w)"]
        for pat in forbidden_patterns:
            matches = re.findall(pat, src)
            if matches:
                pytest.fail(f"B-747 regression in {path.name}: argv password pattern still present: {matches[:3]}")


def test_b747_mysql_pwd_env_used():
    """MYSQL_PWD env injection used in reset + start files (all 5 callsites)."""
    reset_src = RESET.read_text()
    start_src = START.read_text()
    # cls reset 3 callsites + start_shopping 2 callsites + cls DB seed 1 = 6 total
    # (some may collapse in the same line via `-e MYSQL_PWD=password`)
    combined = reset_src + start_src
    n_mysqlpwd = combined.count("MYSQL_PWD=")
    assert n_mysqlpwd >= 5, (
        f"B-747 contract: expected ≥5 MYSQL_PWD callsites, found {n_mysqlpwd}"
    )
    assert "docker exec -e MYSQL_PWD" in combined, (
        "B-747 contract: docker exec must pass MYSQL_PWD via -e"
    )


# ─── B-748 P1-5+18 B* OOB: awk replaces pipefail-fragile indexer ───────────
def test_b748_indexer_uses_awk_not_grep_pipeline():
    """start_vwa_docker.sh:start_shopping indexer poll must use awk, not
    `grep -E ":" | grep -vE "Ready|^$" | wc -l` (pipefail-crash on success).
    """
    src = _strip_shell_comments(START.read_text())
    # Old fragile pattern absent (comments stripped — may still appear in B-748 context comment)
    old_pattern = r"grep\s+-E\s+\":\"\s*\|\s*grep\s+-vE\s+\"Ready\|\^\$\"\s*\|\s*wc\s+-l"
    assert not re.search(old_pattern, src), (
        "B-748 regression: old grep|grep|wc indexer pattern still present (pipefail crash on success)"
    )
    # New awk pattern present (must include NF guard for empty-status protection)
    assert re.search(r"awk\s+-F:\s+'/:/\s*&&\s*NF", src), (
        "B-748 contract: awk -F: '/:/ && NF ...' pattern must replace grep pipeline"
    )


def test_b748_awk_pipeline_survives_pipefail_all_ready():
    """Empirical: under set -euo pipefail, awk pipeline does NOT crash on
    all-Ready status (the original B-748 reproduction)."""
    result = subprocess.run(
        ["bash", "-c", """
set -euo pipefail
idx_status='X: Ready
Y: Ready
Z: Ready'
total_rows=$(echo "${idx_status}" | awk -F: '/:/ && NF {n++} END {print n+0}')
non_ready=$(echo "${idx_status}" | awk -F: '/:/ && NF && $2 !~ /Ready/ {n++} END {print n+0}')
echo "total=${total_rows} non_ready=${non_ready}"
"""],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"B-748 regression: awk pipeline crashes under pipefail rc={result.returncode}\nstderr: {result.stderr}"
    )
    assert "total=3 non_ready=0" in result.stdout


# ─── B-749 P1-6 B: substrate-missing FATAL ─────────────────────────────────
def test_b749_setup_missing_configs_is_fatal():
    """setup_vwa.sh must `exit 1` when per-task split configs missing (was
    warning + 'Setup complete')."""
    src = SETUP.read_text()
    assert re.search(r"FATAL.*per-task split configs", src, re.DOTALL | re.IGNORECASE), (
        "B-749 contract: FATAL label on missing split configs"
    )
    assert "SETUP_ALLOW_MISSING_SPLIT_CONFIGS" in src, (
        "B-749 contract: escape env var must be documented for asset-only flows"
    )


# ─── B-750 P1-7 B: cls archive dual-format ─────────────────────────────────
def test_b750_classifieds_archive_dual_format():
    """import_vwa_assets.sh must detect .zip OR .tar.gz."""
    src = IMPORT_.read_text()
    assert "classifieds_docker_compose.zip" in src, "B-750 contract: .zip detection missing"
    assert "classifieds_docker_compose.tar.gz" in src, "B-750 contract: .tar.gz fallback missing"
    assert "CLASSIFIEDS_ARCHIVE" in src, "B-750 contract: archive detection var missing"


# ─── B-751 P1-8 AB* OOB: template render for hostname patching ─────────────
def test_b751_homepage_template_render_pattern():
    """start_vwa_docker.sh:start_homepage must use .template + cp + perl."""
    src = START.read_text()
    assert "index.html.template" in src, "B-751 contract: homepage template file path"
    assert re.search(r'cp\s+-f\s+"\$\{_homepage_template\}"', src), (
        "B-751 contract: cp template → working file pattern"
    )


def test_b751_classifieds_compose_template_render_pattern():
    """start_vwa_docker.sh:start_classifieds must use compose.yml.template."""
    src = START.read_text()
    assert "docker-compose.yml.template" in src, "B-751 contract: cls compose template missing"


# ─── B-752 P1-15 A: cls compose-name broader regex ─────────────────────────
def test_b752_classifieds_compose_name_broader_regex():
    """start_classifieds container detect must match compose v1 + v2 + bare."""
    src = START.read_text()
    # Pattern must match -app-1 (compose v2) AND _db_1 (compose v1) AND bare
    assert re.search(r"grep\s+-qE\s+'\^classifieds.*-app-1.*_db_1.*\$'", src) or \
           re.search(r"grep\s+-qE\s+'\^classifieds.*\$.*'", src) or \
           "classifieds(-app-1" in src, (
        "B-752 contract: broader regex matching compose-named containers"
    )


# ─── B-753 P1-10 C* OOB: P79_VWA_TZ unified ────────────────────────────────
def test_b753_p79_vwa_tz_unified():
    """P79_VWA_TZ must be first-layer env in BOTH reset + start TZ lookups."""
    for path in (RESET, START, GATES):
        src = path.read_text()
        assert "P79_VWA_TZ" in src, f"B-753 contract: P79_VWA_TZ missing from {path.name}"
    # init_paper_grade_env exports default
    gates_src = GATES.read_text()
    assert 'export P79_VWA_TZ="${P79_VWA_TZ:-Europe/London}"' in gates_src, (
        "B-753 contract: init_paper_grade_env must export P79_VWA_TZ default"
    )


# ─── B-754 P1-9 B* OOB: PAPER_GRADE_ALLOW_PARTIAL hard-block ───────────────
def test_b754_queue_chain_sources_paper_grade_lib():
    """queue_chain.sh must source _lib_paper_grade_gates.sh at start."""
    src = CHAIN.read_text()
    assert "source" in src and "_lib_paper_grade_gates.sh" in src, (
        "B-754 contract: queue_chain must source paper-grade lib"
    )
    assert "init_paper_grade_env" in src, "B-754 contract: must call init_paper_grade_env"


def test_b754_allow_partial_forbidden_under_paper_grade():
    """Sentinel python must reject PAPER_GRADE_ALLOW_PARTIAL=1 + P79_PAPER_GRADE=1."""
    src = CHAIN.read_text()
    assert "P79_PAPER_GRADE" in src, "B-754 contract: P79_PAPER_GRADE check in sentinel"
    assert "PAPER_GRADE_ALLOW_PARTIAL=1 is FORBIDDEN" in src, (
        "B-754 contract: explicit FORBIDDEN log on bypass+paper-grade combo"
    )


# ─── B-755 P1-12 AC* OOB: locality predicate logic invert ──────────────────
def test_b755_locality_gate_coupled_to_paper_grade():
    """assert_a100_url_locality must return early when P79_PAPER_GRADE != 1.

    Inverted from pre-fix OR-chain (hostname/CWD/dir whitelist) to PG-coupled.
    """
    src = GATES.read_text()
    # Find assert_a100_url_locality body + verify it has PG != 1 → return 0 pattern
    m = re.search(r"assert_a100_url_locality\(\)\s*\{(.*?)^\}", src, re.DOTALL | re.MULTILINE)
    assert m, "B-755 contract: assert_a100_url_locality function not found"
    body = m.group(1)
    # PG check appears
    assert 'P79_PAPER_GRADE' in body and '!=' in body and '"1"' in body, (
        "B-755 contract: PG != 1 check missing from function body"
    )
    # `return 0` must appear before the URL loop (within first ~25 lines of body)
    early_lines = "\n".join(body.splitlines()[:25])
    assert "return 0" in early_lines, (
        "B-755 contract: early `return 0` (under PG != 1) missing in first 25 body lines"
    )


# ─── B-756 P1-11 C: Dirty Cell Backdoor FATAL ──────────────────────────────
def test_b756_dirty_cell_backdoor_fatal_under_paper_grade():
    """queue_baseline.sh must FATAL when (P79_PAPER_GRADE=1 + RESET_BEFORE=1)
    AND a runner is already attached (gemini dirty-cell-backdoor attack)."""
    src = BASELINE.read_text()
    assert 'P79_PAPER_GRADE' in src and 'RESET_BEFORE' in src and 'FATAL' in src, (
        "B-756 contract: paper-grade + reset_before + pgrep hit → FATAL"
    )
    assert "dirty cell backdoor" in src.lower(), (
        "B-756 contract: explicit dirty-cell-backdoor naming in FATAL message"
    )


# ─── B-757 P1-14 A: cls reset token from env / .auth/ ──────────────────────
def test_b757_no_hardcoded_classifieds_reset_token():
    """Hardcoded `4b6...` literal removed; reads from env or .auth/cls_reset_token."""
    src = RESET.read_text()
    # Specific known-bad literal
    assert "4b61655535e7ed388f0d40a93600254c" not in src, (
        "B-757 regression: hardcoded CLASSIFIEDS_RESET_TOKEN literal still in source"
    )
    assert ".auth/cls_reset_token" in src, (
        "B-757 contract: fallback to .auth/cls_reset_token path"
    )
    assert 'CLASSIFIEDS_RESET_TOKEN:-' in src, (
        "B-757 contract: env-first lookup pattern"
    )


# ─── B-758 P1-13 C: queue_chain sentinel parser FATAL ──────────────────────
def test_b758_sentinel_parser_fatal_on_unknown_site():
    """When parsed_site empty after pattern match, FATAL (not silent expected_n=0)."""
    src = CHAIN.read_text()
    assert "cannot derive site from run_id" in src, (
        "B-758 contract: explicit FATAL message on parse failure"
    )
    # Confirm there's an `exit 1` in the new branch (not just log)
    assert re.search(r"\[FATAL\] cannot derive site.*\n.*\n.*exit 1", src, re.DOTALL), (
        "B-758 contract: exit 1 after FATAL log"
    )


# ─── B-759 P1-16 ABC: asset publish atomic + sha256 manifest ───────────────
def test_b759_verify_sha256_helper_exists():
    """setup_vwa.sh must define _verify_sha256 helper."""
    src = SETUP.read_text()
    assert "_verify_sha256()" in src, "B-759 contract: _verify_sha256 helper missing"


def test_b759_atomic_tmp_publish_pattern():
    """Download paths must write to `.tmp.$$` then atomic-promote."""
    src = SETUP.read_text()
    # All 4 downloads should have .tmp pattern
    n_tmp = src.count(".tmp.$$")
    assert n_tmp >= 4, (
        f"B-759 contract: expected ≥4 .tmp.$$ atomic-publish sites, found {n_tmp}"
    )


def test_b759_manifest_scaffold_valid():
    """manifest JSON must exist with 4 asset entries + schema fields."""
    assert MANIFEST.exists(), "B-759 contract: vwa_assets_manifest.json missing"
    d = json.loads(MANIFEST.read_text())
    assert "assets" in d, "B-759 contract: manifest needs 'assets' key"
    for asset in ("shopping_final_0712.tar", "postmill-populated-exposed-withimg.tar",
                  "wikipedia_en_all_maxi_2025-08.zim", "classifieds_docker_compose.zip"):
        assert asset in d["assets"], f"B-759 contract: manifest missing asset {asset!r}"
        entry = d["assets"][asset]
        assert "sha256" in entry, f"B-759 contract: {asset} missing sha256 field (may be empty scaffold)"
        assert "source" in entry, f"B-759 contract: {asset} missing source URL"


# ─── B-760 P1-17 AC* OOB: REQ_GB underestimate ─────────────────────────────
def test_b760_a100_runbook_req_gb_350():
    """a100_self_host_vwa.sh REQ_GB must be ≥ 350 (was 250, underestimated)."""
    src = A100.read_text()
    m = re.search(r"^REQ_GB=(\d+)$", src, re.MULTILINE)
    assert m, "B-760 contract: REQ_GB assignment missing"
    assert int(m.group(1)) >= 350, (
        f"B-760 regression: REQ_GB={m.group(1)} < 350 (working set 286 + 64 buffer)"
    )
