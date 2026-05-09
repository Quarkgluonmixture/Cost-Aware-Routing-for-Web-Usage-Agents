#!/usr/bin/env python3
"""Myriad job state watcher — push ntfy on qstat state transition.

Runs from DGX via SSH chain (DGX → quark Tailscale bastion → myriad). Diffs
current qstat output against last saved state in logs/cron/myriad_state.json
and pushes one ntfy notification per cron tick when transitions occur:
  - NEW jobs appearing (qsub submission)
  - state changes (qw → r, r → Eqw, etc.)
  - jobs disappearing (finished / killed) — includes tail of .err for context

Designed for cron schedule: */5 * * * *. SSH chain failures exit 0 silently
(transient network blip; experiment_watchdog handles persistent SSH outages).

ntfy topic: $NTFY_TOPIC env or default p79-exp-dgx-spark.
"""
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STATE_FILE = REPO / "logs" / "cron" / "myriad_state.json"
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "p79-exp-dgx-spark")

DGX_KEY = os.path.expanduser("~/.ssh/vwa_windows")
QUARK_USER = "Quark"
QUARK_HOST = "100.95.81.103"
MYRIAD_USER = "ucab352"
MYRIAD_HOST = "myriad.rc.ucl.ac.uk"
LOG_DIR_REMOTE = "/home/ucab352/Scratch/p79/logs"


def ssh_chain(remote_cmd: str, timeout: int = 25) -> str | None:
    """Run command on Myriad via DGX → quark → myriad chain. Returns stdout
    on success, None on any failure (timeout, ssh error, non-zero exit)."""
    inner = (
        "ssh -o IdentitiesOnly=yes -o BatchMode=yes "
        "-i $env:USERPROFILE\\.ssh\\id_rsa_myriad "
        f'{MYRIAD_USER}@{MYRIAD_HOST} "{remote_cmd}"'
    )
    cmd = ["ssh", "-i", DGX_KEY, "-o", "BatchMode=yes",
           f"{QUARK_USER}@{QUARK_HOST}", inner]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    return r.stdout


def parse_qstat(stdout: str) -> dict:
    """qstat -u $USER plain text → {job_id: {state, name}}."""
    state = {}
    for line in (stdout or "").splitlines():
        parts = line.split()
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        jid, _prio, name, _user, st = parts[:5]
        state[jid] = {"state": st, "name": name}
    return state


def diff_states(old: dict, new: dict) -> list[str]:
    events = []
    for jid, info in new.items():
        if jid not in old:
            events.append(f"NEW  {jid} ({info['name']}): {info['state']}")
        elif old[jid]["state"] != info["state"]:
            events.append(
                f"CHG  {jid} ({info['name']}): "
                f"{old[jid]['state']} → {info['state']}"
            )
    for jid, info in old.items():
        if jid not in new:
            events.append(f"GONE {jid} ({info['name']}): finished/killed")
    return events


def push_ntfy(title: str, body: str, priority: str = "default") -> None:
    url = f"https://ntfy.sh/{NTFY_TOPIC}"
    req = urllib.request.Request(
        url, data=body.encode("utf-8"),
        headers={"Title": title, "Priority": priority, "Tags": "gear"},
    )
    try:
        urllib.request.urlopen(req, timeout=10).read()
    except Exception as e:
        print(f"ntfy push failed: {e}", file=sys.stderr)


def main() -> int:
    # F36 audit fix 2026-05-09: persist consecutive SSH failure count
    # and notify after 3 failures. Previously exited 0 silently which
    # could hide hours of broken SSH chain.
    SSH_FAIL_FILE = STATE_FILE.with_suffix(".ssh_fail_count")
    stdout = ssh_chain("qstat -u ucab352")
    if stdout is None:
        try:
            n_fail = int(SSH_FAIL_FILE.read_text().strip()) if SSH_FAIL_FILE.exists() else 0
        except Exception:
            n_fail = 0
        n_fail += 1
        try:
            SSH_FAIL_FILE.write_text(str(n_fail))
        except Exception:
            pass
        if n_fail >= 3:
            push_ntfy(
                title=f"Myriad SSH chain broken ({n_fail} consecutive)",
                body="DGX → quark → Myriad SSH failed. Check Tailscale + Cisco. "
                     f"State file: {STATE_FILE}; fail counter: {SSH_FAIL_FILE}",
                priority="high",
            )
        return 0
    # Reset failure counter on success
    if SSH_FAIL_FILE.exists():
        try:
            SSH_FAIL_FILE.unlink()
        except Exception:
            pass

    new_state = parse_qstat(stdout)
    old_state = {}
    if STATE_FILE.exists():
        try:
            old_state = json.loads(STATE_FILE.read_text())
        except Exception:
            old_state = {}

    events = diff_states(old_state, new_state)

    if events:
        body_lines = list(events)
        for jid, info in old_state.items():
            if jid in new_state:
                continue
            err_tail = ssh_chain(
                f"tail -25 {LOG_DIR_REMOTE}/qsub_*{jid}.err 2>/dev/null",
                timeout=15,
            )
            if err_tail and err_tail.strip():
                snippet = err_tail.strip()[-500:]
                body_lines.append(f"\n--- {jid}.err tail ---\n{snippet}")
        push_ntfy("Myriad state change", "\n".join(body_lines))

    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(new_state, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
