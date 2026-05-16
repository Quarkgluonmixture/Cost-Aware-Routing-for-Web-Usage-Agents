#!/usr/bin/env bash
# VWA Docker stack fingerprint for cross-environment paper-grade reproducibility.
#
# Captures: docker container names + image tags + RepoDigests (primary, registry-canonical)
# + image IDs (local-only, non-portable) + per-site static-asset SHA-256 hash
# + VWA submodule source SHA (B-240 fix 2026-05-16 A1.16).
# Used for paper §3 / Appendix D byte-equivalence claim between Phase 1 quark
# deployment and A100 self-host deployment.
#
# Usage:
#   bash scripts/provenance/snapshot_vwa.sh                          # default out path
#   bash scripts/provenance/snapshot_vwa.sh /custom/out.json         # custom path
#   VWA_HOST=100.95.81.103 bash scripts/provenance/snapshot_vwa.sh   # remote host (Tailscale)
#
# Environment variables:
#   VWA_HOST    — host IP/name for HTTP probes (default: localhost)
#   VWA_CLS_PORT  (default 9980)
#   VWA_RED_PORT  (default 9999)
#   VWA_SHOP_PORT (default 7770)
#   VWA_PROBE_PATH_<SITE>  — override static-asset probe path (default: /robots.txt)
#
# A1.16 fixes (2026-05-16):
#   - P0-1 (B-237): replaced session-stateful `/` probe with static-asset URL
#     (default /robots.txt). Homepages embed CSRF/session/cart tokens → body_sha256
#     drifted between captures even on same VWA build. /robots.txt is server-config
#     static, no session cookies, stable across captures.
#   - P1-4 (B-241): added snap["vwa_source"] section capturing submodule SHA +
#     Dockerfile + docker-compose sha256. Pre-fix: rebuild docker image with same
#     tag but new VWA source went undetected. RepoDigests now primary; image_id
#     marked "local-only, non-portable" per cross-host layer storage divergence.
#   - P2-1 (B-243): bash heredoc switched to quoted `<<'PYEOF'` form + values
#     passed via os.environ (vs $VAR interpolation into Python source). Closes
#     command-injection vector via env vars.
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

echo "[snapshot-vwa] Output: $OUT"
echo "[snapshot-vwa] VWA host: $VWA_HOST"
echo "[snapshot-vwa] Probe paths: cls=${PROBE_PATH_CLS} red=${PROBE_PATH_RED} shop=${PROBE_PATH_SHOP}"

# Export all values so Python heredoc (quoted form) reads them via os.environ.
# A1.16 P2-1 fix: switching to <<'PYEOF' kills bash $VAR interpolation inside the
# heredoc body, removing the command-injection vector via env vars.
export REPO_ROOT HOST TS OUT VWA_HOST
export CLS_PORT RED_PORT SHOP_PORT
export PROBE_PATH_CLS PROBE_PATH_RED PROBE_PATH_SHOP

python3 - <<'PYEOF'
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

errors = []

REPO_ROOT = os.environ["REPO_ROOT"]
HOST = os.environ["HOST"]
OUT = os.environ["OUT"]
VWA_HOST = os.environ["VWA_HOST"]
CLS_PORT = int(os.environ["CLS_PORT"])
RED_PORT = int(os.environ["RED_PORT"])
SHOP_PORT = int(os.environ["SHOP_PORT"])
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
    "schema_version": "2026-05-16-a1.16",  # B-237 + B-241 + B-243 fix marker
}

# ---------- Docker containers ----------
containers = []
ps_out = run(["docker", "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.ID}}\t{{.Status}}"])
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
        })
snap["containers"] = containers

# ---------- Per-site static-asset probe (B-237 A1.16 P0-1) ----------
# Pre-fix: probed homepage `/` → captured session-stateful CSRF tokens / cart
# state / personalized feed → body_sha256 drifted between captures of the SAME
# VWA build. /robots.txt is server-config static (no session cookies, no
# personalization, deterministic for given VWA submodule source).
sites = {}
for site, port in PORTS.items():
    probe_path = PROBE_PATHS[site]
    url = f"http://{VWA_HOST}:{port}{probe_path}"
    headers = run(["curl", "-sS", "-I", "--max-time", "5", url])
    # Combined HEAD + body in one curl call (atomic; no session drift between
    # the two requests). Disable cookies via -b /dev/null to ensure no carryover.
    body_raw = run(
        ["curl", "-sS", "--max-time", "10", "-b", "/dev/null", "-c", "/dev/null", url],
        default="",
    )
    body_hash = hashlib.sha256(body_raw.encode() if isinstance(body_raw, str) else body_raw).hexdigest() if body_raw else None
    sites[site] = {
        "url": url,
        "probe_kind": "static-asset",
        "headers_first_line": headers.splitlines()[0] if headers else "",
        "body_sha256": body_hash,
        "body_length": len(body_raw) if body_raw else 0,
    }
snap["sites"] = sites

# ---------- VWA source SHA (B-241 A1.16 P1-4) ----------
# Pre-fix: docker image_id captured but VWA submodule SHA + Dockerfile +
# docker-compose source state completely unrecorded. Rebuild docker with same
# tag but new VWA source → snap looks unchanged but evaluator code differs.
vwa_dir = Path(REPO_ROOT) / "external" / "visualwebarena"
vwa_source = {}
if vwa_dir.exists():
    vwa_source["submodule_sha"] = run(["git", "-C", str(vwa_dir), "rev-parse", "HEAD"], default="unavailable")
    vwa_source["submodule_branch"] = run(
        ["git", "-C", str(vwa_dir), "rev-parse", "--abbrev-ref", "HEAD"],
        default="unavailable",
    )
    porcelain = run(["git", "-C", str(vwa_dir), "status", "--porcelain"], default="")
    vwa_source["submodule_dirty"] = bool(porcelain.strip())
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
    vwa_source = {"unavailable": True}
snap["vwa_source"] = vwa_source

# ---------- Docker image global list (paper §3 disclosure) ----------
images_list = run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}"])
snap["images_inventory"] = [
    dict(zip(["repo_tag", "image_id", "size"], line.split("\t")))
    for line in images_list.splitlines() if line
]

snap["errors"] = errors

Path(OUT).write_text(json.dumps(snap, indent=2))
print(
    f"Snapshot written: {OUT} "
    f"(errors: {len(errors)}, containers: {len(containers)}, "
    f"sites probed: {len(sites)}, vwa_source: {'OK' if vwa_source.get('submodule_sha') else 'MISSING'})"
)
PYEOF
