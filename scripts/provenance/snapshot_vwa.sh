#!/usr/bin/env bash
# VWA Docker stack fingerprint for cross-environment paper-grade reproducibility.
#
# Captures: docker container names + image tags + RepoDigests (primary, registry-canonical)
# + image IDs (local-only, non-portable) + per-site static-asset SHA-256 hash
# + VWA submodule source SHA (B-240 fix 2026-05-16 A1.16) + STOPPED containers
# + docker engine version + HTTP status sanity probe (A1.16 cold-start re-audit
# B-826 / B-835).
# Used for paper §3 / Appendix D byte-equivalence claim between Phase 1 quark
# deployment and A100 self-host deployment.
#
# Usage:
#   bash scripts/provenance/snapshot_vwa.sh                          # default out path
#   bash scripts/provenance/snapshot_vwa.sh /custom/out.json         # custom path
#   VWA_HOST=YOUR_HOST_IP bash scripts/provenance/snapshot_vwa.sh   # remote host (Tailscale)
#
# Environment variables:
#   VWA_HOST    — host IP/name for HTTP probes (default: localhost)
#   VWA_CLS_PORT  (default 9980)
#   VWA_RED_PORT  (default 9999)
#   VWA_SHOP_PORT (default 7770)
#   VWA_PROBE_PATH_<SITE>  — override static-asset probe path (default: /robots.txt)
#   P79_PAPER_GRADE=1  — fail-loud on submodule_dirty (B-825 A1.16-re P0-4)
#
# A1.16 fixes (2026-05-16):
#   - P0-1 (B-237 / B-273): replaced session-stateful `/` probe with static-asset URL.
#   - P1-4 (B-241 / B-279): added snap["vwa_source"] section (submodule SHA + Dockerfile +
#     compose sha256). RepoDigests primary, image_id local-only.
#   - P2-1 (B-243): bash heredoc switched to quoted `<<'PYEOF'` form + values
#     passed via os.environ (vs $VAR interpolation). Closes command-injection vector.
#
# A1.16 cold-start re-audit fixes (2026-05-17) — master_bug_catalog B-825 / B-826 / B-835:
#   - B-825 (P0-4-AB*): `vwa_source.submodule_dirty=True` now triggers
#     SystemExit(2) under `P79_PAPER_GRADE=1`. Pre-fix: dirty flag was captured
#     but never enforced; paper-grade fire could proceed with uncommitted
#     submodule patches that don't appear in any captured SBOM hash → tree-hash
#     chain claim contradicted at content level.
#   - B-826 (P0-5-A*): per-site `/robots.txt` probe now verifies HTTP status =
#     200 + Content-Type = text/plain. Pre-fix: probe silently captured 404
#     page body_sha256 when site didn't serve /robots.txt → cross-host
#     fingerprint comparison false-passed on matching 404 pages.
#   - B-835 (P1-9-A): `docker ps -a` includes STOPPED containers; RepoDigests
#     empty list now triggers warn-list entry (local-built images break
#     cross-host portability silently); docker engine version captured.
#
# Output is paper-grade quotable. Diff DGX baseline vs A100 self-host:
#   diff <(jq '.containers' dgx.json) <(jq '.containers' a100.json)
#   diff <(jq '.vwa_source.submodule_sha' dgx.json) <(jq '.vwa_source.submodule_sha' a100.json)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST="$(hostname)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${1:-${REPO_ROOT}/results/provenance/vwa_${HOST}_${TS}.json}"
mkdir -p "$(dirname "$OUT")"

VWA_HOST="${VWA_HOST:-localhost}"
CLS_PORT="${VWA_CLS_PORT:-9980}"
RED_PORT="${VWA_RED_PORT:-9999}"
SHOP_PORT="${VWA_SHOP_PORT:-7770}"
PROBE_PATH_CLS="${VWA_PROBE_PATH_CLS:-/robots.txt}"
PROBE_PATH_RED="${VWA_PROBE_PATH_RED:-/robots.txt}"
PROBE_PATH_SHOP="${VWA_PROBE_PATH_SHOP:-/robots.txt}"
P79_PAPER_GRADE="${P79_PAPER_GRADE:-0}"

echo "[snapshot-vwa] Output: $OUT"
echo "[snapshot-vwa] VWA host: $VWA_HOST"
echo "[snapshot-vwa] Probe paths: cls=${PROBE_PATH_CLS} red=${PROBE_PATH_RED} shop=${PROBE_PATH_SHOP}"
echo "[snapshot-vwa] Paper-grade mode: $P79_PAPER_GRADE"

# Export all values so Python heredoc (quoted form) reads them via os.environ.
# A1.16 P2-1 fix: switching to <<'PYEOF' kills bash $VAR interpolation inside the
# heredoc body, removing the command-injection vector via env vars.
export REPO_ROOT HOST TS OUT VWA_HOST
export CLS_PORT RED_PORT SHOP_PORT
export PROBE_PATH_CLS PROBE_PATH_RED PROBE_PATH_SHOP
export P79_PAPER_GRADE

python3 - <<'PYEOF'
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

errors = []
warnings = []

REPO_ROOT = os.environ["REPO_ROOT"]
HOST = os.environ["HOST"]
OUT = os.environ["OUT"]
VWA_HOST = os.environ["VWA_HOST"]
CLS_PORT = int(os.environ["CLS_PORT"])
RED_PORT = int(os.environ["RED_PORT"])
SHOP_PORT = int(os.environ["SHOP_PORT"])
P79_PAPER_GRADE = os.environ.get("P79_PAPER_GRADE", "0") == "1"
PROBE_PATHS = {
    "classifieds": os.environ["PROBE_PATH_CLS"],
    "reddit": os.environ["PROBE_PATH_RED"],
    "shopping": os.environ["PROBE_PATH_SHOP"],
}
PORTS = {"classifieds": CLS_PORT, "reddit": RED_PORT, "shopping": SHOP_PORT}


def run(cmd, default=""):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=10).decode().strip()
    except Exception as e:
        errors.append(f"{cmd[0] if cmd else 'cmd'}: {type(e).__name__}: {e}")
        return default


snap = {
    "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "host": HOST,
    "vwa_host_for_probe": VWA_HOST,
    "ports": PORTS,
    "probe_paths": PROBE_PATHS,
    "paper_grade_mode": P79_PAPER_GRADE,
    "schema_version": "2026-05-17-a1.16-re",  # B-825 + B-826 + B-835 fix marker
}

# ---------- Docker engine version (B-835 A1.16-re P1-9-A) ----------
docker_version = run(["docker", "version", "--format", "{{.Server.Version}}"])
snap["docker_engine_version"] = docker_version or "unavailable"

# ---------- Docker containers ----------
# B-835 A1.16-re P1-9-A: `docker ps -a` (include STOPPED). Some VWA stack
# components (wikipedia / homepage / shopping admin tools) may be STOPPED but
# task-required at runtime; their absence from fingerprint creates a paper-grade
# inventory gap.
containers = []
ps_out = run(["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Image}}\t{{.ID}}\t{{.Status}}"])
if ps_out:
    for line in ps_out.splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        name, image, cid, status = parts[0], parts[1], parts[2], parts[3]
        image_id_full = run(["docker", "inspect", "--format={{.Image}}", cid])
        repo_digest_raw = run(["docker", "inspect", "--format={{json .RepoDigests}}", image])
        try:
            repo_digests = json.loads(repo_digest_raw) if repo_digest_raw else []
        except json.JSONDecodeError:
            repo_digests = []
        # B-835 A1.16-re P1-9-A: warn when RepoDigests empty (locally-built
        # image, not registry-pulled). A100 self-host stack is build-from-source
        # so this will fire for VWA images — record as warning, not abort.
        if not repo_digests:
            warnings.append(
                f"container {name} (image {image}): empty RepoDigests "
                "→ locally-built, no cross-host registry-canonical fingerprint"
            )
        is_running = status.lower().startswith("up")
        containers.append({
            "name": name,
            "image": image,
            "container_id": cid,
            # B-241 (A1.16 P1-4): RepoDigests is registry-canonical SHA256, portable
            # across hosts. image_id_full is local layer storage hash — same registry
            # pull on different host can produce different value. Use repo_digests
            # for cross-host byte-equivalence; keep image_id for local debug.
            "repo_digests": repo_digests,  # PRIMARY for cross-host equivalence
            "image_id_full": image_id_full,  # local-only, non-portable
            "status": status,
            "is_running": is_running,
        })
snap["containers"] = containers
snap["running_container_count"] = sum(1 for c in containers if c.get("is_running"))
snap["stopped_container_count"] = sum(1 for c in containers if not c.get("is_running"))

# ---------- Per-site static-asset probe (B-237 / B-826 A1.16-re P0-5-A*) ----------
# Pre-fix: probed homepage `/` → captured session-stateful CSRF tokens / cart
# state / personalized feed → body_sha256 drifted between captures of the SAME
# VWA build. /robots.txt is server-config static.
#
# B-826 A1.16-re P0-5-A*: now ALSO verifies HTTP status = 200 + Content-Type =
# text/plain. Pre-fix: probe silently captured 404 page body_sha256 when site
# didn't serve /robots.txt → cross-host fingerprint comparison false-passed on
# matching 404 pages (both hosts return same nginx 404 page → fingerprint
# matches but is meaningless).
sites = {}
for site, port in PORTS.items():
    probe_path = PROBE_PATHS[site]
    url = f"http://{VWA_HOST}:{port}{probe_path}"
    # B-826: capture HTTP status + Content-Type via -w format
    status_raw = run(
        ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code} %{content_type}",
         "--max-time", "5", "-b", "/dev/null", "-c", "/dev/null", url],
        default="000 unknown",
    )
    parts = status_raw.split(" ", 1)
    http_status = parts[0] if parts else "000"
    content_type = parts[1] if len(parts) > 1 else "unknown"

    headers = run(["curl", "-sS", "-I", "--max-time", "5", url])
    # Combined HEAD + body in one curl call (atomic; no session drift between
    # the two requests). Disable cookies via -b /dev/null to ensure no carryover.
    body_raw = run(
        ["curl", "-sS", "--max-time", "10", "-b", "/dev/null", "-c", "/dev/null", url],
        default="",
    )
    body_hash = hashlib.sha256(body_raw.encode() if isinstance(body_raw, str) else body_raw).hexdigest() if body_raw else None

    # B-826 sanity gate: 200 + text/plain expected for static /robots.txt.
    # Record probe_sanity flag for caller / paper-grade gate to inspect.
    probe_ok = (http_status == "200" and content_type.lower().startswith("text/plain"))
    if not probe_ok:
        warnings.append(
            f"vwa-probe-{site}: status={http_status} content_type={content_type} "
            f"— expected 200 text/plain; body_sha256 may be meaningless (e.g. 404 page)"
        )

    sites[site] = {
        "url": url,
        "probe_kind": "static-asset",
        "http_status": http_status,
        "content_type": content_type,
        "probe_sanity_ok": probe_ok,  # B-826: True iff 200 + text/plain
        "headers_first_line": headers.splitlines()[0] if headers else "",
        "body_sha256": body_hash,
        "body_length": len(body_raw) if body_raw else 0,
    }
snap["sites"] = sites
snap["probe_sanity_all_ok"] = all(s.get("probe_sanity_ok") for s in sites.values())

# ---------- VWA source SHA (B-241 A1.16 P1-4 + B-825 A1.16-re P0-4-AB*) ----------
# Pre-fix B-241: docker image_id captured but VWA submodule SHA + Dockerfile +
# docker-compose source state completely unrecorded.
# B-825 A1.16-re P0-4-AB*: submodule_dirty=True now triggers SystemExit(2)
# under P79_PAPER_GRADE=1. Pre-fix: dirty flag captured but never enforced;
# paper-grade fire could proceed with uncommitted submodule patches that don't
# appear in any captured SBOM hash → tree-hash chain claim contradicted at
# content level.
vwa_dir = Path(REPO_ROOT) / "external" / "visualwebarena"
vwa_source = {}
if vwa_dir.exists():
    vwa_source["submodule_sha"] = run(["git", "-C", str(vwa_dir), "rev-parse", "HEAD"], default="unavailable")
    vwa_source["submodule_branch"] = run(
        ["git", "-C", str(vwa_dir), "rev-parse", "--abbrev-ref", "HEAD"],
        default="unavailable",
    )
    porcelain = run(["git", "-C", str(vwa_dir), "status", "--porcelain"], default="")
    submodule_dirty = bool(porcelain.strip())
    vwa_source["submodule_dirty"] = submodule_dirty
    vwa_source["submodule_dirty_files"] = porcelain.splitlines() if porcelain.strip() else []

    # Dockerfile + docker-compose fingerprint
    candidate_globs = ["Dockerfile", "Dockerfile.*", "docker-compose*.yml", "docker-compose*.yaml"]
    dockerfile_files = []
    for pattern in candidate_globs:
        dockerfile_files.extend(sorted(vwa_dir.rglob(pattern)))
    docker_fp = hashlib.sha256()
    docker_files_seen = []
    for f in sorted(dockerfile_files):
        if f.is_file():
            content = f.read_bytes()
            rel = str(f.relative_to(vwa_dir))
            docker_fp.update(rel.encode() + b"\x00" + content + b"\x00")
            docker_files_seen.append({"path": rel, "sha256": hashlib.sha256(content).hexdigest(), "size": len(content)})
    vwa_source["dockerfile_combined_sha256"] = docker_fp.hexdigest() if docker_files_seen else "no-dockerfiles-found"
    vwa_source["dockerfile_files"] = docker_files_seen
else:
    errors.append(f"vwa-source: submodule dir not found at {vwa_dir}")
    vwa_source = {"unavailable": True, "submodule_dirty": False}
snap["vwa_source"] = vwa_source

# ---------- Docker image global list (paper §3 disclosure) ----------
images_list = run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}"])
snap["images_inventory"] = [
    dict(zip(["repo_tag", "image_id", "size"], line.split("\t")))
    for line in images_list.splitlines() if line
]

snap["errors"] = errors
snap["warnings"] = warnings

Path(OUT).write_text(json.dumps(snap, indent=2))
print(
    f"Snapshot written: {OUT} "
    f"(errors: {len(errors)}, warnings: {len(warnings)}, "
    f"containers: {len(containers)} ({snap['running_container_count']} running / "
    f"{snap['stopped_container_count']} stopped), "
    f"sites probed: {len(sites)} ({sum(1 for s in sites.values() if s.get('probe_sanity_ok'))} sanity OK), "
    f"vwa_source: {'OK' if vwa_source.get('submodule_sha') else 'MISSING'}, "
    f"submodule_dirty: {vwa_source.get('submodule_dirty')})"
)

# B-825 A1.16-re P0-4-AB*: fail-loud on submodule_dirty under paper-grade mode.
# Other failure conditions (HTTP probe sanity / SBOM divergence) are recorded as
# warnings — caller queue script or run_experiment.py can promote to fail-loud
# via snapshot_has_critical_errors() helper.
if P79_PAPER_GRADE and vwa_source.get("submodule_dirty"):
    print(
        f"\n[FATAL paper-grade] VWA submodule dirty during snapshot — "
        f"{len(vwa_source.get('submodule_dirty_files', []))} uncommitted file(s). "
        "Paper-grade fire requires clean submodule (prereg §7 SBOM tree-hash chain "
        "covers committed HEAD only; dirty working tree adds uncaptured changes).",
        file=sys.stderr,
    )
    for f in vwa_source.get("submodule_dirty_files", [])[:10]:
        print(f"  {f}", file=sys.stderr)
    sys.exit(2)
PYEOF
