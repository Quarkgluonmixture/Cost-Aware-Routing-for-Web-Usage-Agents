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
    "transient_drift", # one-off transient that doesn't reproduce
    "undecided",       # reviewer needs more data
)


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


def preflight_check(
    site: str,
    task_ids: List[int],
    halt_threshold: int = 1,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Preflight gate G8 — returns (should_halt, blocking_tasks).

    blocking_tasks is a list of dicts with details on which tasks have
    unclassified_count >= halt_threshold so operator can run M7 manual
    reproduction on those specific tasks.

    halt_threshold default = 1: any single unclassified quarantine event
    halts the next fire. User decision 2026-05-19: "Fire-5 should not
    blindly rediscover" task 75. Stricter than "≥3 fires" rule because
    paper-grade should investigate at first occurrence; lenient ≥3 rule
    only applies to FE gate exclusion (Appendix E.1 terminal quarantine).
    """
    blocking: List[Dict[str, Any]] = []
    for tid in task_ids:
        count = count_unclassified(site, tid)
        if count >= halt_threshold:
            quar, _ = _task_events(site, tid)
            blocking.append({
                "site": site,
                "task_id": tid,
                "unclassified_count": count,
                "fires_quarantined": [q.get("run_id") for q in quar],
                "last_error_class": quar[-1].get("error_class") if quar else None,
                "last_callsite": quar[-1].get("callsite") if quar else None,
            })
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


def _cmd_preflight(args: argparse.Namespace) -> int:
    task_ids = _parse_task_range(args.tasks)
    threshold = int(os.environ.get("QUARANTINE_HALT_THRESHOLD", args.halt_threshold))
    should_halt, blocking = preflight_check(args.site, task_ids, halt_threshold=threshold)
    if should_halt:
        print(
            f"[preflight G8 HALT] site={args.site}: "
            f"{len(blocking)} task(s) have >= {threshold} unclassified quarantine event(s).",
            file=sys.stderr,
        )
        for b in blocking:
            print(
                f"  • task {b['task_id']}: {b['unclassified_count']} unclassified "
                f"(fires: {b['fires_quarantined']}); "
                f"last error={b['last_error_class']!r} callsite={b['last_callsite']!r}",
                file=sys.stderr,
            )
        print(
            "[preflight G8 HALT] Required action: reproduce + classify via "
            "`python scripts/maintenance/quarantine_registry.py classify "
            "--site=<site> --task-id=<N> --as=<substrate|agent_induced|evaluator|transient_drift|undecided> "
            "--classified-by=<name> --rationale=<...>`",
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
    p_pre.set_defaults(func=_cmd_preflight)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
