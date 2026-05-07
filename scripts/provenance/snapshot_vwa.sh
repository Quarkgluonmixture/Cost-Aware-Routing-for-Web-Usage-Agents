#!/usr/bin/env bash
# VWA Docker stack fingerprint for cross-environment paper-grade reproducibility.
#
# Captures: docker container names + image tags + image digests + per-site
# HTML SHA-256 hash. Used for paper §3 / Appendix D byte-equivalence claim
# between Phase 1 quark deployment and A100 self-host deployment.
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
#
# Output is paper-grade quotable. Diff DGX baseline vs A100 self-host:
#   diff <(jq '.containers' dgx.json) <(jq '.containers' a100.json)

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

echo "[snapshot-vwa] Output: $OUT"
echo "[snapshot-vwa] VWA host: $VWA_HOST"

# Build JSON via python (jq syntax fragility avoided)
python3 - <<PYEOF
import json, subprocess, sys, datetime
from pathlib import Path

errors = []

def run(cmd, default=""):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=10).decode().strip()
    except Exception as e:
        errors.append(f"{cmd[0]}: {type(e).__name__}: {e}")
        return default

snap = {
    "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "host": "$HOST",
    "vwa_host_for_probe": "$VWA_HOST",
    "ports": {"classifieds": $CLS_PORT, "reddit": $RED_PORT, "shopping": $SHOP_PORT},
}

# Docker containers
containers = []
ps_out = run(["docker", "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.ID}}\t{{.Status}}"])
if ps_out:
    for line in ps_out.splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        name, image, cid, status = parts[0], parts[1], parts[2], parts[3]
        # Get image digest
        digest = run(["docker", "inspect", "--format={{.Image}}", cid])
        # Get image RepoDigests if available
        repo_digest = run(["docker", "inspect", "--format={{json .RepoDigests}}", image])
        try:
            repo_digest_parsed = json.loads(repo_digest) if repo_digest else []
        except json.JSONDecodeError:
            repo_digest_parsed = []
        containers.append({
            "name": name,
            "image": image,
            "container_id": cid,
            "image_id_full": digest,
            "repo_digests": repo_digest_parsed,
            "status": status,
        })
snap["containers"] = containers

# Per-site HTTP probe + content hash
sites = {}
for site, port in [("classifieds", $CLS_PORT), ("reddit", $RED_PORT), ("shopping", $SHOP_PORT)]:
    url = f"http://$VWA_HOST:{port}/"
    headers = run(["curl", "-sS", "-I", "--max-time", "5", url])
    body_hash = run(["bash", "-c", f"curl -sS --max-time 10 '{url}' | sha256sum | awk '{{print \$1}}'"])
    sites[site] = {
        "url": url,
        "headers_first_line": headers.splitlines()[0] if headers else "",
        "body_sha256": body_hash if body_hash else None,
    }
snap["sites"] = sites

# Docker image global list (paper §3 disclosure)
images_list = run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}"])
snap["images_inventory"] = [
    dict(zip(["repo_tag", "image_id", "size"], line.split("\t")))
    for line in images_list.splitlines() if line
]

snap["errors"] = errors

Path("$OUT").write_text(json.dumps(snap, indent=2))
print(f"Snapshot written: $OUT (errors: {len(errors)}, containers: {len(containers)}, sites probed: {len(sites)})")
PYEOF
