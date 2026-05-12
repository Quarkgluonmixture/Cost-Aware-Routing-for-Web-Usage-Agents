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
from typing import Optional

REPO = Path(__file__).resolve().parents[3]
STATE_FILE = REPO / "logs" / "cron" / "myriad_state.json"
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "p79-exp-dgx-spark")

# Audit (A) 2026-05-09: GONE_HOOKS dispatch — when a Myriad job
# transitions to GONE (finished/killed), look up its name prefix here
# and fire the auto_pull script. Each entry maps prefix → (remote_dir,
# cell_md_path). Add new entries as cells get qsub'd; missing prefix
# falls through to ntfy-only (no SCP).
GONE_HOOKS: dict[str, tuple[str, str]] = {
    # cellf forward × reddit strong (336423 already pulled manually)
    "cellf_fwd_": (
        "stage2b_cellf_fwd_reddit_strong_myriad",
        "docs/checkpoints/_status/cells/cell_b1_red_stage2_cellf.md",
    ),
    # cellg reverse × reddit reverse-tier (336424 in flight)
    "cellg_rev_": (
        "stage2c_cellg_rev_reddit_reverse_myriad",
        "docs/checkpoints/_status/cells/cell_b1_red_stage2_cellg.md",
    ),
    # cellcr reddit fwd × reverse-tier (2x2 selection-bias control)
    "cellcr_": (
        "stage2b_cellcr_reddit_fwd_revtier_myriad",
        "",
    ),
    # celldr reddit rev × strong-tier (2x2 selection-bias control)
    "celldr_": (
        "stage2c_celldr_reddit_rev_strongtier_myriad",
        "",
    ),
    # celler reddit fwd × strong × random injection (negative control)
    "celler_": (
        "stage2b_celler_reddit_fwd_random_myriad",
        "",
    ),
    # Stage 3 mechanism attribution: SoM ↔ P-text / P-prompt patching
    "cellht_cls": (
        "stage3_cellht_cls_fwd_text_myriad",
        "",
    ),
    "cellhp_cls": (
        "stage3_cellhp_cls_fwd_prompt_myriad",
        "",
    ),
    "cellht_red": (
        "stage3_cellht_red_fwd_text_myriad",
        "",
    ),
    "cellhp_red": (
        "stage3_cellhp_red_fwd_prompt_myriad",
        "",
    ),
    # Cell H-d (DOM target) — closes 2x2 mechanism additivity test
    "cellhd_cls": (
        "stage3_cellhd_cls_fwd_dom_myriad",
        "",
    ),
    "cellhd_red": (
        "stage3_cellhd_red_fwd_dom_myriad",
        "",
    ),
    # Exp 5 axis-2 prompt-only patching (P-SoM → P-text, same flat text)
    "cellhprm_cls": (
        "stage3_cellhprompt_cls_fwd_ptext_myriad",
        "",
    ),
    "cellhprm_red": (
        "stage3_cellhprompt_red_fwd_ptext_myriad",
        "",
    ),
    # Exp 5 axis-2 random-injection negative control (/stress G3 specificity)
    "cellhprm_cls_rand": (
        "stage3_cellhprompt_cls_fwd_ptext_rand_myriad",
        "",
    ),
    "cellhprm_red_rand": (
        "stage3_cellhprompt_red_fwd_ptext_rand_myriad",
        "",
    ),
    # Stage 4 Method 4.2 multimode hidden state extraction (PCA cosine gap)
    "stage4mm_cls": (
        "stage4_multimode_b1_cls",
        "",
    ),
    # Stage 4 H1 test: text format variation across 8 industry-relevant indexed-list styles
    "stage4fv_cls": (
        "stage4_format_variation_b1_cls",
        "",
    ),
    # P4: H1 on cls reverse-tier (selection-bias defense)
    "stage4fv_clsrev": (
        "stage4_format_variation_b1_cls_reverse",
        "",
    ),
    # P5a: H1 on reddit strong-tier (cross-site defense)
    "stage4fv_red": (
        "stage4_format_variation_b1_reddit",
        "",
    ),
    # P5b: Method 4.2 multimode on reddit strong-tier (cross-site Mirage signature)
    "stage4mm_red": (
        "stage4_multimode_b1_reddit",
        "",
    ),
    # 16-cell rerun cells: register here as launched. Pattern:
    # "cellX_<descr>": ("<remote_subdir>", "<cell_md_relpath>"),
}
AUTO_PULL_SCRIPT = REPO / "scripts" / "maintenance" / "auto_pull_myriad_cell.sh"


def _dispatch_gone_hook(jid: str, name: str) -> Optional[str]:
    """If `name` matches a GONE_HOOKS prefix, fire auto_pull script
    in background and return the matched prefix; else None.
    """
    for prefix, (remote_dir, cell_md) in GONE_HOOKS.items():
        if name.startswith(prefix):
            try:
                subprocess.Popen(
                    [
                        "bash", str(AUTO_PULL_SCRIPT),
                        jid, name, remote_dir, cell_md,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                return prefix
            except OSError as e:
                print(f"[myriad_watcher] auto_pull dispatch failed for {jid}/{name}: {e}",
                      file=sys.stderr)
                return None
    return None

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


def _qstat_with_sentinel() -> str | None:
    """Run qstat via ssh_chain with stdout sentinel guard.

    Fix 2026-05-12 silent-miss bug: inner ssh (quark→myriad) can fail
    (Cisco VPN drop / hung session) while outer ssh (DGX→quark)
    returns exit 0 with empty stdout — powershell on quark doesn't
    propagate inner exit code. We append `&& echo __QSTAT_OK__`; if
    the sentinel is absent from stdout, qstat itself didn't run →
    treat as None (chain failure) instead of writing {} and losing
    GONE events for in-flight jobs.
    """
    stdout = ssh_chain("qstat -u ucab352 && echo __QSTAT_OK__")
    if stdout is None:
        return None
    if "__QSTAT_OK__" not in stdout:
        return None
    return stdout.replace("__QSTAT_OK__", "").rstrip()


def main() -> int:
    # F36 audit fix 2026-05-09: persist consecutive SSH failure count
    # and notify after 3 failures. Previously exited 0 silently which
    # could hide hours of broken SSH chain.
    SSH_FAIL_FILE = STATE_FILE.with_suffix(".ssh_fail_count")
    stdout = _qstat_with_sentinel()
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

    # Double-probe guard 2026-05-12: silent-miss bug postmortem. If we
    # had jobs last tick and now claim ZERO, re-probe before believing
    # — sentinel-passing-but-truncated qstat could still hide real jobs
    # under rare powershell pipe edge cases. Only accept empty result
    # when 2nd probe also confirms. Otherwise log + preserve old_state.
    if old_state and not new_state:
        stdout_recheck = _qstat_with_sentinel()
        if stdout_recheck is None:
            push_ntfy(
                title="Myriad qstat empty after non-empty + recheck failed",
                body=f"Last tick had {len(old_state)} jobs; this tick qstat empty; "
                     f"recheck SSH chain failed. Preserving old_state to avoid silent GONE-miss. "
                     f"Jobs: {sorted(old_state.keys())}",
                priority="high",
            )
            return 0
        recheck_state = parse_qstat(stdout_recheck)
        if recheck_state:
            # 2nd probe disagreed — real jobs still there
            new_state = recheck_state
        # else: 2nd probe also empty → accept; jobs really finished

    events = diff_states(old_state, new_state)

    if events:
        body_lines = list(events)
        # Audit (A) 2026-05-09: dispatch GONE_HOOKS auto_pull for
        # finalized jobs. Append dispatched-prefix info to ntfy body so
        # the user knows whether SCP+validate fired or just an alert.
        dispatched = []
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
            hooked = _dispatch_gone_hook(jid, info.get("name", ""))
            if hooked:
                dispatched.append(f"{jid}/{info.get('name','?')} → auto_pull[{hooked}]")
        if dispatched:
            body_lines.append("\n--- auto_pull dispatched ---\n" + "\n".join(dispatched))
        push_ntfy("Myriad state change", "\n".join(body_lines))

    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(new_state, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
