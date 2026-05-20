#!/usr/bin/env python3
"""Fire-4 RCA Wave 2 M6 — quarantine registry investigation gate.

User decision 2026-05-19 (post 3-AI /stress audit): cross-fire quarantine
memory is an **investigation gate**, NOT an auto skip-list. When the same
(site, task_id) quarantines across multiple fires, the next fire must
HALT until human reproduces + classifies the failure mode (Wave 4 M7).
Skip-list only after classification AND symmetric/global exclusion
decision documented in paper §4.

Registry format
---------------
Append-only JSONL at `docs/checkpoints/quarantine_registry.jsonl`. Each
line is a single event. Two event types:

  (1) `event_type=quarantine` — emitted by experiment_watchdog when a
      paper-grade run produces a needs_reevaluation=True summary. Carries
      run_id, site, task_id, url, error class/message, classify_timeout
      callsite, and capture timestamp.

  (2) `event_type=classification` — emitted by manual review tooling
      (Wave 4 M7 reproduction outcome) OR by `python quarantine_registry.py
      classify ...` CLI. Carries (site, task_id) + classification ∈
      {substrate, agent_induced, evaluator, transient_drift, undecided} +
      classified_by + rationale.

Investigation gate logic
------------------------
For a given (site, task_id), count unclassified quarantine events:
  unclassified_count = #quarantine_events - #classification_events
If any task in the upcoming fire has unclassified_count >= UNCLASSIFIED_HALT
(default 1, configurable via env QUARANTINE_HALT_THRESHOLD), preflight
gate G8 halts the fire with "investigation required for task X".

A classification can be revised by appending a new classification event;
the most recent timestamp wins for gate logic.

CLI
---
  python scripts/maintenance/quarantine_registry.py append --site=cls --task-id=75 ...
  python scripts/maintenance/quarantine_registry.py classify --site=cls --task-id=75 --as=substrate --rationale="..."
  python scripts/maintenance/quarantine_registry.py query --site=cls --task-id=75
  python scripts/maintenance/quarantine_registry.py preflight --site=cls --tasks=0-233  # exit 0 ok, 1 halt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Registry lives in docs/checkpoints/ (tracked in git so cross-machine memory
# persists via normal git pull on operator's laptop / DGX / A100).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REGISTRY_PATH = REPO_ROOT / "docs" / "checkpoints" / "quarantine_registry.jsonl"

VALID_CLASSIFICATIONS = (
    "substrate",       # site/docker/network — legitimate noise (verified_substrate_noise=True)
    "agent_induced",   # agent caused timeout (e.g., DOM deadlock from bad action sequence)
    "evaluator",       # evaluator harness failure (BLIP-2 / GPT judge / playwright eval timeout)
    "transient_drift", # one-off transient that doesn't reproduce (use ONLY when temporally-matched reproduce attempted under mid-fire context)
    # /stress 2026-05-20 P0-A1-Hub: epistemically honest tier when reproduce
    # attempted but only in isolated context (fresh chromium, post-event hours,
    # no cumulative GPU+chromium+cron load) — substrate health in mid-fire
    # context NOT proven. Re-tier from `transient_drift` when reproduction did
    # not replicate the mid-fire cumulative load condition (e.g., Wave 4 M7
    # playwright_mcp_reproduce from a different host, 30h+ post-event).
    "unreproducible_in_isolation",
    "undecided",       # reviewer needs more data
)

# Fire-6 RCA Stage C2 (/stress 2026-05-20): diagnostic-scoped Gate 8 override
# task-count cap. A diagnostic replay is targeted (a handful of tasks under
# matched-temporal-context investigation, e.g. cls task 4 + 75); a canonical
# fire enumerates a full site (0-233). Capping the override at a small task
# count is one of FOUR independent guards (env flag + --diagnostic-replay flag
# + non-canonical --output-path + task-scoped) that ALL must hold — so a leaked
# QUARANTINE_DIAGNOSTIC_REPLAY env var alone can never bypass canonical Gate 8.
DIAG_OVERRIDE_MAX_TASKS = 25

# Fire-6 RCA Stage C3a (/stress 2026-05-20): evidence-gated cross-fire-recurrence
# resolution. A recurrent task's Gate 8 Rule 2 block clears ONLY when EVERY
# distinct quarantine error_class has its OWN evidence-gated `resolution` event
# (the eval-goto C1 resolution must NOT clear the screenshot-timeout class — user
# directive 2026-05-20). All checks are fail-closed.
#
# resolved_by_commit floor: the fix must be `e9875cc` (C2 classification commit,
# which contains C1 eval isolation) OR a descendant — AND an ancestor of HEAD
# (the fix is actually present in the running code). Verified via git merge-base.
RESOLUTION_COMMIT_FLOOR = "e9875cc"
RESOLUTION_REQUIRED_VIA = "matched_temporal_context_diagnostic_replay"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_events() -> List[Dict[str, Any]]:
    """Read all events from registry. Empty list if file doesn't exist."""
    if not REGISTRY_PATH.exists():
        return []
    events: List[Dict[str, Any]] = []
    with REGISTRY_PATH.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                events.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                print(
                    f"[quarantine_registry] WARN: corrupt JSON at line {line_no}: {exc}",
                    file=sys.stderr,
                )
    return events


def _append_event(event: Dict[str, Any]) -> None:
    """Atomic append (best-effort: simple O_APPEND + newline)."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def append_quarantine(
    *,
    site: str,
    task_id: int,
    run_id: str,
    url: Optional[str] = None,
    error_class: Optional[str] = None,
    error_message: Optional[str] = None,
    callsite: Optional[str] = None,
    needs_reevaluation: bool = True,
) -> Dict[str, Any]:
    """Append a quarantine event (called by watchdog on QUARANTINE detection)."""
    event = {
        "event_type": "quarantine",
        "ts": _iso_now(),
        "run_id": run_id,
        "site": site,
        "task_id": int(task_id),
        "url": url,
        "error_class": error_class,
        "error_message": (error_message or "")[:500],  # cap to keep registry compact
        "callsite": callsite,
        "needs_reevaluation": bool(needs_reevaluation),
    }
    _append_event(event)
    return event


def append_classification(
    *,
    site: str,
    task_id: int,
    classification: str,
    classified_by: str,
    rationale: str,
    classified_via: Optional[str] = None,
) -> Dict[str, Any]:
    """Append a classification event (CLI or M7 reproduce tool)."""
    if classification not in VALID_CLASSIFICATIONS:
        raise ValueError(
            f"invalid classification {classification!r}; must be one of {VALID_CLASSIFICATIONS}"
        )
    event = {
        "event_type": "classification",
        "ts": _iso_now(),
        "site": site,
        "task_id": int(task_id),
        "classification": classification,
        "classified_by": classified_by,
        "classified_via": classified_via,
        "rationale": rationale,
    }
    _append_event(event)
    return event


def _task_events(site: str, task_id: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (quarantine_events, classification_events) for given (site, task_id)."""
    quar: List[Dict[str, Any]] = []
    classif: List[Dict[str, Any]] = []
    for ev in _read_events():
        if ev.get("site") != site or int(ev.get("task_id", -1)) != int(task_id):
            continue
        if ev.get("event_type") == "quarantine":
            quar.append(ev)
        elif ev.get("event_type") == "classification":
            classif.append(ev)
    return quar, classif


def count_unclassified(site: str, task_id: int) -> int:
    """Return unclassified count = #quarantine_events - #classification_events.

    Per-fire-per-event accounting: each fire's quarantine needs a matching
    classification event to be considered "resolved". This matches user's
    intent: "investigate task 75 first" means each occurrence is examined,
    even if same task quarantines multiple times across fires.

    Most-recent classification timestamp determines current state; older
    classifications are retained for audit trail.
    """
    quar, classif = _task_events(site, task_id)
    return max(0, len(quar) - len(classif))


def latest_classification(site: str, task_id: int) -> Optional[Dict[str, Any]]:
    """Return most recent classification event for (site, task_id), or None."""
    _, classif = _task_events(site, task_id)
    if not classif:
        return None
    return max(classif, key=lambda e: e.get("ts", ""))


# ===================== Fire-6 C3a: evidence-gated resolution =====================


def _git_commit_in_lock_range(commit: str, repo_root: Path = REPO_ROOT) -> Tuple[bool, str]:
    """Verify RESOLUTION_COMMIT_FLOOR ⪯ commit ⪯ HEAD (fail-closed).

    The fix a resolution references must be (a) at least as new as the floor
    (`e9875cc`, which contains C1 eval isolation) AND (b) actually present in the
    running code (an ancestor of HEAD). Git's content-addressable ancestry means
    a hand-written resolution cannot claim a fix absent from the deployed tree.
    """
    import subprocess
    if not commit:
        return False, "empty resolved_by_commit"

    def _is_ancestor(a: str, b: str) -> Optional[bool]:
        try:
            rc = subprocess.run(
                ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", a, b],
                capture_output=True, timeout=15,
            ).returncode
        except Exception:
            return None
        if rc == 0:
            return True
        if rc == 1:
            return False
        return None  # 128 = bad object / not a commit

    floor_ok = _is_ancestor(RESOLUTION_COMMIT_FLOOR, commit)
    if floor_ok is None:
        return False, f"git cannot evaluate {RESOLUTION_COMMIT_FLOOR}..{commit} (bad object / no git)"
    if not floor_ok:
        return False, f"resolved_by_commit {commit} is not {RESOLUTION_COMMIT_FLOOR}-or-descendant"
    head_ok = _is_ancestor(commit, "HEAD")
    if head_ok is None:
        return False, f"git cannot evaluate {commit}..HEAD"
    if not head_ok:
        return False, f"resolved_by_commit {commit} not ancestor of HEAD (fix not in running code)"
    return True, "ok"


def _error_class_evidence_profile(error_class: Optional[str]) -> str:
    """Map a quarantine error_class to its resolution evidence profile.

    eval_goto  — EvaluatorUnavailableError / Page.goto timeout → C1 eval isolation;
                 episode-verified (eval_context_mode=isolated + eval_goto_timeout=False).
    screenshot — Page.screenshot timeout → C1b non-fatal recovery; architectural
                 (resolved_by_commit must carry C1b + a clean diagnostic episode;
                 the load-dependent timeout cannot be forced in a small replay).
    unknown    — anything else → fail-closed (cannot resolve).
    """
    ec = (error_class or "").lower()
    if "page.goto" in ec or "evaluatorunavailable" in ec:
        return "eval_goto"
    if "page.screenshot" in ec:
        return "screenshot"
    return "unknown"


def _verify_resolution_episode(ep: Dict[str, Any], profile: str) -> Tuple[bool, str]:
    """Verify a diagnostic-replay episode meets the evidence conditions for the
    given error_class profile (fail-closed)."""
    if ep.get("diagnostic_replay") is not True:
        return False, "episode diagnostic_replay != True"
    if ep.get("sr_excluded") is not True:
        return False, "episode sr_excluded != True"
    if ep.get("needs_reevaluation") is not False:
        return False, "episode needs_reevaluation != False"
    if profile == "eval_goto":
        if ep.get("eval_context_mode") != "isolated_program_html_context":
            return False, "eval_context_mode != isolated_program_html_context"
        if ep.get("eval_goto_timeout") is not False:
            return False, "eval_goto_timeout != False"
        return True, "ok (eval_goto: isolated eval + no goto timeout)"
    if profile == "screenshot":
        return True, "ok (screenshot: clean episode + C1b architectural fix via commit)"
    return False, f"unknown error_class profile {profile!r}"


def append_resolution(
    *,
    site: str,
    task_id: int,
    error_class: str,
    resolved_by_commit: str,
    episode_summary_path: str,
    resolved_by: str,
    rationale: str,
) -> Dict[str, Any]:
    """Append an evidence-gated resolution event (verified at write).

    Refuses (ValueError) unless the profile is known, the episode satisfies the
    profile's evidence conditions, the episode task_id matches, and
    resolved_by_commit is e9875cc-or-descendant AND ancestor-of-HEAD. The gate
    (`is_error_class_resolved`) RE-verifies at preflight — two fail-closed checks.
    """
    profile = _error_class_evidence_profile(error_class)
    if profile == "unknown":
        raise ValueError(f"no evidence profile for error_class {error_class!r}")
    ep_path = Path(episode_summary_path)
    if not ep_path.exists():
        raise ValueError(f"episode summary not found: {ep_path}")
    ep = json.loads(ep_path.read_text(encoding="utf-8"))
    if int(ep.get("task_id", -1)) != int(task_id):
        raise ValueError(f"episode task_id={ep.get('task_id')} != resolution task_id={task_id}")
    ok, reason = _verify_resolution_episode(ep, profile)
    if not ok:
        raise ValueError(f"episode evidence rejected ({profile}): {reason}")
    commit_ok, commit_reason = _git_commit_in_lock_range(resolved_by_commit)
    if not commit_ok:
        raise ValueError(f"resolved_by_commit rejected: {commit_reason}")
    event = {
        "event_type": "resolution",
        "ts": _iso_now(),
        "site": site,
        "task_id": int(task_id),
        "error_class": error_class,
        "evidence_profile": profile,
        "classified_via": RESOLUTION_REQUIRED_VIA,
        "resolved_by_commit": resolved_by_commit,
        "episode_summary_path": str(ep_path),
        "evidence_snapshot": {
            "diagnostic_replay": ep.get("diagnostic_replay"),
            "sr_excluded": ep.get("sr_excluded"),
            "needs_reevaluation": ep.get("needs_reevaluation"),
            "eval_context_mode": ep.get("eval_context_mode"),
            "eval_goto_timeout": ep.get("eval_goto_timeout"),
            "success": ep.get("success"),
        },
        "resolved_by": resolved_by,
        "rationale": rationale,
    }
    _append_event(event)
    return event


def _resolution_events(site: str, task_id: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ev in _read_events():
        if ev.get("event_type") != "resolution":
            continue
        if ev.get("site") != site or int(ev.get("task_id", -1)) != int(task_id):
            continue
        out.append(ev)
    return out


def is_error_class_resolved(site: str, task_id: int, error_class: str) -> Tuple[bool, str]:
    """Is THIS (site, task, error_class) evidence-gated-resolved? Most-recent
    matching resolution wins and is RE-verified (episode re-read + git ancestry)
    so a hand-edited registry cannot bypass the gate. Returns (resolved, reason).
    """
    profile = _error_class_evidence_profile(error_class)
    if profile == "unknown":
        return False, f"unknown error_class profile for {error_class!r}"
    cands = [
        e for e in _resolution_events(site, task_id)
        if (e.get("error_class") or "").lower() == (error_class or "").lower()
    ]
    if not cands:
        return False, f"no resolution event for error_class {error_class!r}"
    res = max(cands, key=lambda e: e.get("ts", ""))
    if res.get("classified_via") != RESOLUTION_REQUIRED_VIA:
        return False, f"resolution classified_via != {RESOLUTION_REQUIRED_VIA}"
    commit_ok, commit_reason = _git_commit_in_lock_range(res.get("resolved_by_commit", ""))
    if not commit_ok:
        return False, f"resolved_by_commit re-check failed: {commit_reason}"
    ep_path = Path(res.get("episode_summary_path", ""))
    if not ep_path.exists():
        return False, f"resolution episode missing at preflight: {ep_path}"
    try:
        ep = json.loads(ep_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"resolution episode unreadable: {exc}"
    ok, reason = _verify_resolution_episode(ep, profile)
    if not ok:
        return False, f"episode re-verify failed ({profile}): {reason}"
    return True, f"resolved ({profile}) by {str(res.get('resolved_by_commit'))[:7]}"


def is_recurrence_resolved(site: str, task_id: int) -> Tuple[bool, List[str]]:
    """Are ALL distinct recurrence error_classes resolved? A task clears Rule 2
    ONLY when every distinct error_class across its quarantine fires has its own
    valid evidence-gated resolution (per-error-class — eval-goto resolution does
    NOT clear the screenshot-timeout class). Returns (all_resolved, unresolved).
    """
    quar, _ = _task_events(site, task_id)
    error_classes = sorted({q.get("error_class") for q in quar if q.get("error_class")})
    if not error_classes:
        return False, ["no error_class on quarantine events"]
    unresolved: List[str] = []
    for ec in error_classes:
        ok, reason = is_error_class_resolved(site, task_id, ec)
        if not ok:
            unresolved.append(f"{ec!r}: {reason}")
    return (not unresolved), unresolved


def detect_recurrent_failures(
    site: str,
    min_fires: int = 2,
) -> List[Dict[str, Any]]:
    """Return list of {task_id, fire_count, run_ids, error_classes} for tasks
    that have quarantined across >= min_fires distinct fires (distinct run_ids).

    /stress 2026-05-20 P0-A2-Hub: cross-fire recurrent same-task detection.
    Classification status does NOT unilaterally unblock — a task quarantining
    across multiple fires is a substrate-degradation signal independent of
    per-fire classification (a `transient_drift` event recurring 3× is no
    longer transient).

    Empirical anchor: Fire-3 + Fire-4 both quarantined cls task 75 (item
    id=84148). Even after Wave 4 M7 reproduce classified it (now
    `unreproducible_in_isolation` per /stress P0-A1-Hub revised tier), the
    cross-fire pattern itself warrants pre-Fire-6 halt + matched-temporal-
    context investigation. Default min_fires=2 catches Fire-3+Fire-4 task 75
    pattern at next pre-fire gate.
    """
    events = _read_events()
    by_task: Dict[int, List[Dict[str, Any]]] = {}
    for e in events:
        if e.get("event_type") != "quarantine":
            continue
        if e.get("site") != site:
            continue
        tid = e.get("task_id")
        if tid is None:
            continue
        by_task.setdefault(int(tid), []).append(e)

    recurrent: List[Dict[str, Any]] = []
    for tid, quar_events in by_task.items():
        run_ids = sorted({q.get("run_id") for q in quar_events if q.get("run_id")})
        if len(run_ids) >= min_fires:
            recurrent.append({
                "task_id": tid,
                "fire_count": len(run_ids),
                "run_ids": run_ids,
                "error_classes": sorted({q.get("error_class") for q in quar_events if q.get("error_class")}),
            })
    return sorted(recurrent, key=lambda x: -x["fire_count"])


def preflight_check(
    site: str,
    task_ids: List[int],
    halt_threshold: int = 1,
    min_recurrent_fires: int = 2,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Preflight gate G8 — returns (should_halt, blocking_tasks).

    blocking_tasks is a list of dicts with details on which tasks have
    unclassified_count >= halt_threshold (Rule 1: classification gap) OR
    recurrent_fires >= min_recurrent_fires (Rule 2: cross-fire recurrence
    pattern, independent of classification) — so operator can run M7 manual
    reproduction on those specific tasks BEFORE re-launch.

    halt_threshold default = 1: any single unclassified quarantine event
    halts the next fire. User decision 2026-05-19: "Fire-5 should not
    blindly rediscover" task 75. Stricter than "≥3 fires" rule because
    paper-grade should investigate at first occurrence; lenient ≥3 rule
    only applies to FE gate exclusion (Appendix E.1 terminal quarantine).

    min_recurrent_fires default = 2: /stress 2026-05-20 P0-A2-Hub —
    classification does NOT unilaterally unblock; a task quarantining
    across ≥ 2 fires halts regardless of classification status. Fire-3 +
    Fire-4 task 75 empirical recurrence trigger at default.
    """
    blocking: List[Dict[str, Any]] = []
    blocking_task_ids: set = set()

    # Rule 1: unclassified-count threshold (existing pre-/stress-2026-05-20)
    for tid in task_ids:
        count = count_unclassified(site, tid)
        if count >= halt_threshold:
            quar, _ = _task_events(site, tid)
            blocking.append({
                "site": site,
                "task_id": tid,
                "rule": "unclassified",
                "unclassified_count": count,
                "fires_quarantined": [q.get("run_id") for q in quar],
                "last_error_class": quar[-1].get("error_class") if quar else None,
                "last_callsite": quar[-1].get("callsite") if quar else None,
            })
            blocking_task_ids.add(tid)

    # Rule 2: cross-fire recurrence (P0-A2-Hub, 2026-05-20)
    recurrent = detect_recurrent_failures(site, min_fires=min_recurrent_fires)
    for r in recurrent:
        if r["task_id"] not in task_ids:
            continue
        if r["task_id"] in blocking_task_ids:
            # already blocked by Rule 1; annotate with recurrence info
            for b in blocking:
                if b["task_id"] == r["task_id"]:
                    b["recurrent_fires"] = r["fire_count"]
                    b["rule"] = b["rule"] + "+recurrent"
                    b["error_classes_across_fires"] = r["error_classes"]
                    break
            continue
        # Fire-6 C3a (/stress 2026-05-20): evidence-gated resolution clears the
        # Rule 2 cross-fire-recurrence block when EVERY distinct error_class has
        # its own valid resolution (per-error-class; re-verified episode + git
        # ancestry). This does NOT clear Rule 1 (unclassified count) — a task
        # blocked by Rule 1 above never reaches here. Fail-closed: any unresolved
        # class keeps the halt.
        _resolved, _unresolved = is_recurrence_resolved(site, r["task_id"])
        if _resolved:
            continue
        # not blocked by Rule 1, but recurrent across ≥ min_recurrent_fires
        # fires AND not evidence-gated-resolved — halt for matched-temporal-
        # context investigation
        quar, _ = _task_events(site, r["task_id"])
        blocking.append({
            "site": site,
            "task_id": r["task_id"],
            "rule": "cross_fire_recurrence",
            "unclassified_count": 0,  # may be classified but still recurrent
            "recurrent_fires": r["fire_count"],
            "fires_quarantined": r["run_ids"],
            "error_classes_across_fires": r["error_classes"],
            "last_error_class": quar[-1].get("error_class") if quar else None,
            "last_callsite": quar[-1].get("callsite") if quar else None,
        })
        blocking_task_ids.add(r["task_id"])

    should_halt = len(blocking) > 0
    return should_halt, blocking


# ============================== CLI ==============================


def _cmd_append(args: argparse.Namespace) -> int:
    ev = append_quarantine(
        site=args.site,
        task_id=args.task_id,
        run_id=args.run_id,
        url=args.url,
        error_class=args.error_class,
        error_message=args.error_message,
        callsite=args.callsite,
        needs_reevaluation=args.needs_reevaluation,
    )
    print(json.dumps(ev, ensure_ascii=False))
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    ev = append_classification(
        site=args.site,
        task_id=args.task_id,
        classification=getattr(args, "as"),
        classified_by=args.classified_by,
        rationale=args.rationale,
        classified_via=args.classified_via,
    )
    print(json.dumps(ev, ensure_ascii=False))
    return 0


def _cmd_resolve(args: argparse.Namespace) -> int:
    """Fire-6 C3a: append an evidence-gated resolution (verified at write)."""
    try:
        ev = append_resolution(
            site=args.site,
            task_id=args.task_id,
            error_class=args.error_class,
            resolved_by_commit=args.resolved_by_commit,
            episode_summary_path=args.episode_summary,
            resolved_by=args.resolved_by,
            rationale=args.rationale,
        )
    except ValueError as exc:
        print(f"[resolve] REFUSED (fail-closed): {exc}", file=sys.stderr)
        return 1
    print(json.dumps(ev, ensure_ascii=False))
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    quar, classif = _task_events(args.site, args.task_id)
    latest = latest_classification(args.site, args.task_id)
    result = {
        "site": args.site,
        "task_id": args.task_id,
        "quarantine_count": len(quar),
        "classification_count": len(classif),
        "unclassified_count": count_unclassified(args.site, args.task_id),
        "latest_classification": latest,
        "quarantine_events": quar,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _parse_task_range(spec: str) -> List[int]:
    """Parse '0-233' or '1,3,5' or '7' into list of task ids."""
    out: List[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def _diagnostic_override_active(
    args: argparse.Namespace, task_ids: List[int],
) -> Tuple[bool, List[str]]:
    """Fire-6 RCA Stage C2 (/stress 2026-05-20): diagnostic-scoped Gate 8
    override decision. Returns (active, reasons_failed).

    FAIL-CLOSED: the override is active ONLY if ALL FOUR independent guards
    hold; any single failure keeps canonical Gate 8 in force. The guards
    encode the user's STRICT constraint that this "must only work with
    --diagnostic-replay, explicit --tasks, non-canonical output path, and
    sr_excluded=True ... must not become a canonical Gate-8 bypass":

      1. env QUARANTINE_DIAGNOSTIC_REPLAY=1 — operator intent (queue wrapper).
      2. --diagnostic-replay CLI flag — a canonical queue's preflight call
         never passes this, so a leaked env var alone is inert.
      3. --output-path contains 'diagnostic_replay' — proves the run lands in
         the non-canonical results/diagnostic_replay/ tree (NOT phase1). This
         is also the runtime guarantee that every episode carries
         sr_excluded=True (the runner forces it whenever diagnostic_replay is
         on, which is what routes output here).
      4. task list is non-empty AND <= DIAG_OVERRIDE_MAX_TASKS — diagnostic
         replay is targeted; a full-site range (0-233) fails this guard.
    """
    reasons: List[str] = []
    env_on = os.environ.get("QUARANTINE_DIAGNOSTIC_REPLAY", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    if not env_on:
        reasons.append("QUARANTINE_DIAGNOSTIC_REPLAY env not set to 1/true")
    if not getattr(args, "diagnostic_replay", False):
        reasons.append("--diagnostic-replay flag not passed")
    out = getattr(args, "output_path", None) or ""
    if "diagnostic_replay" not in out:
        reasons.append(
            f"--output-path not non-canonical (must contain 'diagnostic_replay'): {out!r}"
        )
    if not task_ids:
        reasons.append("empty task list")
    elif len(task_ids) > DIAG_OVERRIDE_MAX_TASKS:
        reasons.append(
            f"task list too large for diagnostic scope "
            f"({len(task_ids)} > {DIAG_OVERRIDE_MAX_TASKS} cap)"
        )
    return (not reasons), reasons


def _cmd_preflight(args: argparse.Namespace) -> int:
    task_ids = _parse_task_range(args.tasks)
    threshold = int(os.environ.get("QUARANTINE_HALT_THRESHOLD", args.halt_threshold))
    # /stress 2026-05-20 P0-A2-Hub: cross-fire recurrent same-task detector
    # default min_fires=2 (Fire-3+Fire-4 task 75 empirical anchor); env override
    # QUARANTINE_MIN_RECURRENT_FIRES preserves operator control.
    min_recurrent = int(os.environ.get("QUARANTINE_MIN_RECURRENT_FIRES", 2))
    should_halt, blocking = preflight_check(
        args.site,
        task_ids,
        halt_threshold=threshold,
        min_recurrent_fires=min_recurrent,
    )
    if should_halt:
        # Fire-6 RCA Stage C2 (/stress 2026-05-20): diagnostic-scoped override.
        # Checked BEFORE the halt path. preflight_check() above stays pure (it
        # returned the REAL blocking state); the override is an explicit,
        # fail-closed CLI-layer decision so the gate logic remains honest +
        # testable. When all four guards hold, the diagnostic replay proceeds
        # on the SAME blocking tasks under matched-temporal-context — which is
        # exactly how a cross_fire_recurrence (Rule 2) task like cls 75 gets
        # the reproduce that unblocks it. Output is non-canonical + every
        # episode sr_excluded=True, so this can NEVER touch paper §1 SR.
        override_active, override_fail = _diagnostic_override_active(args, task_ids)
        if override_active:
            print(
                f"[preflight G8 DIAGNOSTIC OVERRIDE] site={args.site}: "
                f"{len(blocking)} blocking task(s) bypassed for diagnostic replay "
                f"(env QUARANTINE_DIAGNOSTIC_REPLAY=1 + --diagnostic-replay + "
                f"non-canonical output + {len(task_ids)} task(s) <= "
                f"{DIAG_OVERRIDE_MAX_TASKS}). NON-CANONICAL, sr_excluded — "
                f"NOT a paper-grade fire.",
                file=sys.stderr,
            )
            for b in blocking:
                print(
                    f"  • OVERRIDDEN task {b['task_id']} [{b.get('rule')}]: "
                    f"will run under matched-temporal-context diagnostic replay.",
                    file=sys.stderr,
                )
            return 0
        # --diagnostic-replay passed but override NOT granted → explain which
        # guard failed (fail-closed), then fall through to the canonical halt.
        if getattr(args, "diagnostic_replay", False):
            print(
                "[preflight G8] --diagnostic-replay passed but override NOT "
                "granted (fail-closed): " + "; ".join(override_fail),
                file=sys.stderr,
            )
        # /stress 2026-05-20 P0-A2-Hub: surface dual-rule blocking. Rule 1
        # = unclassified count; Rule 2 = cross-fire recurrence (independent
        # of classification).
        n_unclass = sum(1 for b in blocking if "unclassified" in b.get("rule", ""))
        n_recur = sum(1 for b in blocking if "cross_fire_recurrence" in b.get("rule", ""))
        print(
            f"[preflight G8 HALT] site={args.site}: {len(blocking)} task(s) blocked — "
            f"unclassified-rule: {n_unclass}, cross-fire-recurrence-rule: {n_recur} "
            f"(threshold={threshold} unclassified, min_recurrent_fires={min_recurrent}).",
            file=sys.stderr,
        )
        for b in blocking:
            rule = b.get("rule", "unclassified")
            tid = b["task_id"]
            fires = b.get("fires_quarantined", [])
            if "cross_fire_recurrence" in rule:
                fire_info = (
                    f"{b.get('recurrent_fires', len(fires))} fires across "
                    f"{len(b.get('error_classes_across_fires', []))} distinct error class(es)"
                )
            else:
                fire_info = f"{len(fires)} fire(s)"
            print(
                f"  • task {tid} [{rule}]: {fire_info}; "
                f"last error={b.get('last_error_class')!r} "
                f"callsite={b.get('last_callsite')!r}",
                file=sys.stderr,
            )
            if "cross_fire_recurrence" in rule and b.get("error_classes_across_fires"):
                for ec in b["error_classes_across_fires"]:
                    print(f"      error_class: {ec}", file=sys.stderr)
        # Build classification enum hint dynamically from VALID_CLASSIFICATIONS
        # so future enum expansions don't drift between code + CLI hint.
        enum_hint = "|".join(VALID_CLASSIFICATIONS)
        print(
            "[preflight G8 HALT] Required action: reproduce + classify via "
            "`python scripts/maintenance/quarantine_registry.py classify "
            f"--site=<site> --task-id=<N> --as=<{enum_hint}> "
            "--classified-by=<name> --rationale=<...>`. "
            "For cross-fire-recurrence rule: matched-temporal-context reproduce "
            "REQUIRED (not isolated playwright_mcp fresh-context reproduce — "
            "must replicate mid-fire cumulative load condition).",
            file=sys.stderr,
        )
        return 1
    print(
        f"[preflight G8 OK] site={args.site}: 0 tasks have unclassified quarantine events "
        f"above threshold={threshold}. Fire may proceed.",
        file=sys.stderr,
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Fire-4 RCA Wave 2 M6 quarantine registry CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_app = sub.add_parser("append", help="Append a quarantine event (called by watchdog)")
    p_app.add_argument("--site", required=True)
    p_app.add_argument("--task-id", type=int, required=True)
    p_app.add_argument("--run-id", required=True)
    p_app.add_argument("--url", default=None)
    p_app.add_argument("--error-class", default=None)
    p_app.add_argument("--error-message", default=None)
    p_app.add_argument("--callsite", default=None)
    p_app.add_argument("--needs-reevaluation", type=lambda s: s.lower() == "true", default=True)
    p_app.set_defaults(func=_cmd_append)

    p_cls = sub.add_parser("classify", help="Append a classification event (M7 manual review)")
    p_cls.add_argument("--site", required=True)
    p_cls.add_argument("--task-id", type=int, required=True)
    p_cls.add_argument("--as", required=True, choices=VALID_CLASSIFICATIONS)
    p_cls.add_argument("--classified-by", default="operator")
    p_cls.add_argument("--rationale", required=True)
    p_cls.add_argument("--classified-via", default=None,
                       help="e.g., playwright_manual_reproduce / agent_replay / docker_inspection")
    p_cls.set_defaults(func=_cmd_classify)

    # Fire-6 C3a: evidence-gated resolution (clears Gate 8 Rule 2 per error_class).
    p_res = sub.add_parser("resolve", help="Append evidence-gated resolution (Fire-6 C3a) — clears Rule 2 for one error_class")
    p_res.add_argument("--site", required=True)
    p_res.add_argument("--task-id", type=int, required=True)
    p_res.add_argument("--error-class", required=True,
                       help="exact quarantine error_class string this resolution addresses")
    p_res.add_argument("--episode-summary", required=True,
                       help="path to the matched-temporal-context diagnostic-replay episode summary JSON")
    p_res.add_argument("--resolved-by-commit", required=True,
                       help=f"fix commit; must be {RESOLUTION_COMMIT_FLOOR}-or-descendant AND ancestor of HEAD")
    p_res.add_argument("--resolved-by", default="operator")
    p_res.add_argument("--rationale", required=True)
    p_res.set_defaults(func=_cmd_resolve)

    p_qry = sub.add_parser("query", help="Query events for a (site, task_id)")
    p_qry.add_argument("--site", required=True)
    p_qry.add_argument("--task-id", type=int, required=True)
    p_qry.set_defaults(func=_cmd_query)

    p_pre = sub.add_parser("preflight", help="Gate G8 — exit 0 ok, exit 1 halt fire")
    p_pre.add_argument("--site", required=True)
    p_pre.add_argument("--tasks", required=True,
                       help="task range e.g. '0-233' or '1,3,5' or '75'")
    p_pre.add_argument("--halt-threshold", type=int, default=1,
                       help="halt if any task has unclassified_count >= threshold (default 1)")
    # Fire-6 RCA Stage C2 (/stress 2026-05-20): diagnostic-scoped Gate 8 override.
    # ALL of (env QUARANTINE_DIAGNOSTIC_REPLAY=1, this flag, non-canonical
    # --output-path, task-scoped --tasks) must hold — fail-closed. NEVER a
    # canonical bypass; canonical queue preflight calls omit these flags.
    p_pre.add_argument("--diagnostic-replay", action="store_true",
                       help="diagnostic-scoped Gate 8 override (requires "
                            "QUARANTINE_DIAGNOSTIC_REPLAY=1 + non-canonical "
                            "--output-path + <=25 tasks). NOT a canonical bypass.")
    p_pre.add_argument("--output-path", default=None,
                       help="output root path; for diagnostic override must "
                            "contain 'diagnostic_replay'.")
    p_pre.set_defaults(func=_cmd_preflight)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
