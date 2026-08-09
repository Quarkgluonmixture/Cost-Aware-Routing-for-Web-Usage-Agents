"""Free functions extracted from runner.py during §97 Step-3 split.

These are pure helpers (no class state) used by `runner.main.ExperimentRunner`.
Public API re-exported via `p79.experiment.runner.__init__` for backward
compatibility — external code that imported `from p79.experiment.runner
import _action_signature` still works.
"""
from __future__ import annotations

import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from p79.backends.action_utils import extract_candidate_query, first_element_id_by_keyword

logger = logging.getLogger(__name__)


def _parse_seeds(seed_value: Any) -> List[int]:
    """Accept seed as int or list of ints."""
    if isinstance(seed_value, (list, tuple)):
        return [int(s) for s in seed_value]
    return [int(seed_value)]


# B-1891: locator errors that mean "the agent's target was not actionable", as
# distinct from a transient or unrelated dispatch note. An allowlist rather than
# a truthy check on the error string, so a newly introduced locator error class
# has to be classified deliberately instead of silently counting as fulfilled.
# Sources: `p79/envs/locator_dispatch.py`.
_ACTION_INTENT_FAILURE_MARKERS = (
    "walk_fail",                      # no actionable ancestor within the walk
    "obs_nodes_info missing union_bound",  # referenced element absent from the observation
)


def _action_intent_fulfilled(
    action_success: bool, locator_route_meta: Optional[Dict[str, Any]]
) -> bool:
    """Was the agent's intent actually carried out on this step? — B-1891.

    `action_success` degraded into "the framework did not raise": on a locator
    `walk_fail` a fallback still executes and the top level records True. This
    returns False for those steps so a stuck episode is countable without
    parsing `locator_route_meta` at every call site.

    A step that already failed outright is unfulfilled by definition.
    """
    if not action_success:
        return False
    if not isinstance(locator_route_meta, dict):
        return True
    err = locator_route_meta.get("error")
    if not isinstance(err, str) or not err:
        return True
    return not any(m in err for m in _ACTION_INTENT_FAILURE_MARKERS)


def _action_signature(action: Dict[str, Any]) -> str:
    """Compact fingerprint of an action for cycle detection (strict: includes element_id).

    B-504 (/stress A1.5b Phase 1 P2-4-A, 2026-05-17): replace hardcoded 60-char
    text truncation with full-length hash + length suffix to eliminate
    long-prefix-same-suffix-different false positives. Pre-fix two `type`
    actions with text `"long search query truncated at position 60 mark: A"`
    vs `"...: B"` (both 60 chars) collapsed to the same prefix → false cycle
    detected → `_anti_repeat_control` fired spuriously → synthetic action
    injection. Edge case probability low (min_reps=3 protects), but the
    hash+length signature is strictly more precise + same compactness.
    """
    import hashlib
    atype = str(action.get("action_type", "")).lower()
    eid = action.get("element_id", "")
    text_full = str(action.get("text", ""))
    if text_full:
        text_sig = (
            hashlib.sha256(text_full.encode("utf-8")).hexdigest()[:10]
            + f"_{len(text_full)}"
        )
    else:
        text_sig = ""
    coord = action.get("coordinate", "")
    delta = action.get("delta", "")
    # tab_focus: include page_number so switching between different tabs is not
    # mistakenly treated as a cycle (e.g. 1→0→1 differs from 1→1→1).
    page_num = action.get("page_number", "") if atype == "tab_focus" else ""
    return f"{atype}|eid={eid}|t={text_sig}|c={coord}|d={delta}|pn={page_num}"


def _action_signature_soft(action: Dict[str, Any]) -> str:
    """Loose fingerprint ignoring element_id/coordinate (catches semantic loops
    where the same search query or click-type is repeated on re-rendered pages)."""
    atype = str(action.get("action_type", "")).lower()
    text = str(action.get("text", ""))[:60]
    delta = action.get("delta", "")
    # tab_focus: include page_number in soft signature for the same reason.
    page_num = action.get("page_number", "") if atype == "tab_focus" else ""
    return f"{atype}|t={text}|d={delta}|pn={page_num}"


def _action_signature_fuzzy(action: Dict[str, Any], obs_url: str = "") -> str:
    """Fuzzy fingerprint ignoring element_id/coordinate/text — catches semantic
    loops where agent varies query/target slightly but stays on same URL doing
    same action_type (B-11 search-loop with rephrased queries, B-17 repeat
    click-on-link, B-18 click-truncate-at-max-step).

    Uses (action_type, url_path_no_query) — max-aggressive fuzzy. Higher
    min_reps required at cycle-detect call site (5 instead of 3-4) to keep
    false-positive rate acceptable.
    """
    atype = str(action.get("action_type", "")).lower()
    # Strip query string from URL (search-loop varies query=foo&query=bar but
    # URL path stays /index.php). Also strip fragment.
    url = str(obs_url or "")
    url_path = url.split("?", 1)[0].split("#", 1)[0]
    # tab_focus: include page_number so different tab orchestration doesn't
    # collapse into one fuzzy bucket.
    page_num = action.get("page_number", "") if atype == "tab_focus" else ""
    return f"{atype}|u={url_path}|pn={page_num}"


def _detect_action_cycle(signatures: List[str], min_cycle: int = 1, max_cycle: int = 4,
                         min_reps: int = 3) -> int:
    """Return cycle length if the tail of *signatures* is a repeating cycle, else 0.

    Requires at least *min_reps* full repetitions of the cycle to trigger.
    E.g. [A,B,A,B,A,B] → cycle_len=2.  [A,A,A] → cycle_len=1.
    """
    n = len(signatures)
    for clen in range(min_cycle, max_cycle + 1):
        window = clen * min_reps
        if n < window:
            continue
        tail = signatures[-window:]
        pattern = tail[:clen]
        if all(tail[i] == pattern[i % clen] for i in range(window)):
            return clen
    return 0


# ---------------------------------------------------------------------------
# Diagnostic control helpers
# These functions are only active when `diagnostic_controls` is explicitly set
# in the experiment config (diagnostic_controls.enabled: true).  They are
# NOT enabled by default and must NOT be used in main baseline conditions.
# ---------------------------------------------------------------------------


def _sanitize_query_text(raw_query: str, max_words: int = 4, suspicious_word_threshold: int = 6) -> str:
    query = re.sub(r"\s+", " ", (raw_query or "").strip())
    if not query:
        return query

    suspicious = (
        ">" in query
        or "|" in query
        or "/" in query
        or query.count("&") >= 2
        or len(query.split()) >= max(1, suspicious_word_threshold)
    )
    if not suspicious:
        return query

    parts = [p.strip() for p in re.split(r"\s*(?:>|/|\|)\s*", query) if p.strip()]
    candidate = parts[-1] if parts else query
    candidate = re.sub(r"\([^)]*\)", " ", candidate)
    if "&" in candidate:
        left = candidate.split("&", 1)[0].strip()
        if left:
            candidate = left
    candidate = re.sub(r"[^A-Za-z0-9\-\s]", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()

    words = [w for w in candidate.split() if w]
    if len(words) > max(1, max_words):
        words = words[: max(1, max_words)]
    candidate = " ".join(words).strip()

    if len(candidate) < 2:
        return query
    return candidate


def _query_sanitization_control(action: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    if str(action.get("action_type", "")).lower() != "type":
        return action, None

    text = str(action.get("text", ""))
    had_newline = text.endswith("\n")
    core = text[:-1] if had_newline else text

    cleaned = _sanitize_query_text(
        raw_query=core,
        max_words=int(cfg.get("max_words", 4)),
        suspicious_word_threshold=int(cfg.get("suspicious_word_threshold", 6)),
    )
    if not cleaned or cleaned == core:
        return action, None

    patched = dict(action)
    patched["text"] = cleaned + ("\n" if had_newline else "")
    return patched, f"query_sanitized:{core}->{cleaned}"


def _repeat_hits_same_target(
    step_records: List[Dict[str, Any]],
    action: Dict[str, Any],
    window: int,
) -> int:
    atype = str(action.get("action_type", "")).lower()
    target_eid = action.get("element_id")
    if atype not in ("click", "type") or target_eid is None or not step_records:
        return 0

    hits = 0
    for rec in step_records[-max(1, window):]:
        if str(rec.get("action_type", "")).lower() != atype:
            continue
        prev_action = rec.get("action", {}) or {}
        if prev_action.get("element_id") != target_eid:
            continue
        if bool(rec.get("page_changed", True)):
            continue
        hits += 1
    return hits


def _build_exploration_fallback_action(
    obs_text: str,
    instruction: str,
    query_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    input_id = first_element_id_by_keyword(obs_text, ("textbox", "input", "search", "edit"))
    query = _sanitize_query_text(
        raw_query=extract_candidate_query(instruction),
        max_words=int(query_cfg.get("max_words", 4)),
        suspicious_word_threshold=int(query_cfg.get("suspicious_word_threshold", 6)),
    )
    if input_id is not None and query:
        return {
            "action_type": "type",
            "element_id": int(input_id),
            "text": f"{query}\n",
            "thought": "Diagnostic control: break loop with reformulated search.",
        }

    return {
        "action_type": "scroll",
        "delta": [0, 0.8],
        "coordinate_type": "normalized",
        "thought": "Diagnostic control: break loop with forced exploration scroll.",
    }


def _anti_repeat_control(
    action: Dict[str, Any],
    step_records: List[Dict[str, Any]],
    obs_text: str,
    instruction: str,
    cfg: Dict[str, Any],
    query_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[str]]:
    if str(action.get("action_type", "")).lower() not in ("click", "type"):
        return action, None

    window = int(cfg.get("window", 3))
    min_repeat_hits = int(cfg.get("min_repeat_hits", 2))
    hits = _repeat_hits_same_target(step_records, action, window=window)
    if hits < max(1, min_repeat_hits):
        return action, None

    fallback = _build_exploration_fallback_action(obs_text, instruction, query_cfg=query_cfg)
    return fallback, f"anti_repeat_blocked:hits={hits}"


def _no_early_finish_control(
    action: Dict[str, Any],
    step_records: List[Dict[str, Any]],
    obs_text: str,
    instruction: str,
    cfg: Dict[str, Any],
    query_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[str]]:
    atype = str(action.get("action_type", "")).lower()
    if atype not in ("finish", "stop"):
        return action, None

    min_exploration_steps = int(cfg.get("min_exploration_steps", 5))
    min_page_changes = int(cfg.get("min_page_changes", 2))
    min_search_attempts = int(cfg.get("min_search_attempts", 2))

    explored_steps = len(step_records)
    page_change_count = sum(1 for s in step_records if bool(s.get("page_changed", False)))
    search_attempts = sum(
        1
        for s in step_records
        if str(s.get("action_type", "")).lower() == "type"
        and bool(str((s.get("action", {}) or {}).get("text", "")).strip())
    )

    if (
        explored_steps >= min_exploration_steps
        and page_change_count >= min_page_changes
        and search_attempts >= min_search_attempts
    ):
        return action, None

    fallback = _build_exploration_fallback_action(obs_text, instruction, query_cfg=query_cfg)
    reason = (
        f"no_early_finish_blocked:"
        f"steps={explored_steps}/{min_exploration_steps},"
        f"page_changes={page_change_count}/{min_page_changes},"
        f"search={search_attempts}/{min_search_attempts}"
    )
    return fallback, reason


def _notify_retry_pass(
    condition_id: str,
    all_ids: List[int],
    ok_ids: List[int],
    fail_ids: List[int],
) -> None:
    """Log retry-pass results and optionally push ntfy notification."""
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        return
    n = len(all_ids)
    title = f"P79 Retry [{condition_id}] {len(ok_ids)}/{n} OK"
    lines = [f"Retried {n} tasks"]
    if ok_ids:
        lines.append(f"OK: {ok_ids}")
    if fail_ids:
        lines.append(f"FAIL: {fail_ids}")
    body = "\n".join(lines)
    priority = "default" if not fail_ids else "high"
    url = f"https://ntfy.sh/{topic}"
    req = urllib.request.Request(
        url, data=body.encode("utf-8"), method="POST",
        headers={"Title": title, "Priority": priority, "Markdown": "yes"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15):
            pass
        logger.info("Retry pass ntfy sent to %s", topic)
    except Exception as exc:
        logger.warning("Retry pass ntfy failed: %s", exc)


def _notify_transient_retry(
    condition_id: str,
    site: str,
    task_id: Any,
    transient_class: str,
    attempt: int,
    max_retries: int,
) -> None:
    """Push ntfy for a B-1881 transient-substrate episode-level retry.

    Transparency channel so the operator sees that a transient infra blip
    (auth / proxy_5xx / network) triggered an episode retry on fresh substrate
    INSTEAD of a condition-level fail-closed abort. Best-effort — never raises."""
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        return
    title = f"P79 transient-retry [{condition_id}] task {task_id} ({transient_class})"
    body = (
        f"transient substrate ({transient_class}) at site={site} task={task_id} "
        f"— episode-level retry {attempt}/{max_retries} on fresh substrate "
        f"(NOT condition abort; B-1881)"
    )
    url = f"https://ntfy.sh/{topic}"
    req = urllib.request.Request(
        url, data=body.encode("utf-8"), method="POST",
        headers={"Title": title, "Priority": "default", "Markdown": "yes"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15):
            pass
        logger.info("Transient-retry ntfy sent to %s", topic)
    except Exception as exc:
        logger.warning("Transient-retry ntfy failed: %s", exc)


# Recovery hints per abort class. Keyed by the `abort_class` passed to
# push_run_abort_ntfy; an unknown class still notifies, just without a hint.
_ABORT_RECOVERY_HINTS = {
    "paper_grade_quarantine": (
        "paper-grade 隔离：condition_summary 已带 abort 字段落盘，进程非零退出会让 "
        "chain 整体停住。先看 condition_summary.json 的 aborted_at_task / reason "
        "再决定是修 env 重发还是清该 task 重跑。"
    ),
    "proxy_quota": (
        "预算池耗尽。续额度后重跑同一条 queue 命令即可 —— resume:true 会从已完成的 "
        "episode 之后继续，不重跑已有数据。"
    ),
    "fatal_env": (
        "环境级错误，需要人工修复后从干净状态重发。不要直接重跑 —— 先按 log 里的 "
        "具体错误修 env。"
    ),
    "evaluator_unavailable": (
        "评测基建不可用（NLTK / OpenAI key / VWA submodule）。修好 env 再从干净状态"
        "重发；本次不写 needs_reevaluation summary。"
    ),
}


def push_run_abort_ntfy(
    condition_id: str,
    site: str,
    task_id: Any,
    abort_class: str,
    exc: Any,
) -> None:
    """Push ntfy when a run STOPS (not retries) on an unrecoverable condition.

    Why this exists: the three fail-fast branches in `_run_and_record_episode`
    (evaluator-unavailable / fatal-env / proxy-quota) all `raise` to stop the
    run — which is correct, since every subsequent task would fail the same way
    — but none of them told anyone. Empirically 2026-08-09: the shop B0 vision
    run hit proxy quota exhaustion at task 405 (374 episodes in) at 01:19 and
    sat dead for six hours before a human looked. The stop was by design; the
    silence was not.

    Priority is `urgent` on purpose. This is the one runner event where the
    difference between "operator finds out now" and "operator finds out at
    breakfast" is measured in wasted wall-clock on a booked machine.

    Best-effort — never raises, so it can be called on the way to a `raise`."""
    # EVERYTHING is inside the try, including body construction. `str(exc)` runs
    # arbitrary user code (`Exception.__str__`), and the whole point of this
    # helper is that it sits one line before a `raise` — an exception escaping
    # from here would replace the precise "quota exhausted" traceback with a
    # misleading one from the notifier. Verified 2026-08-09: an exception whose
    # __str__ raises made the pre-fix version propagate a RuntimeError.
    # (/stress Mode B P2-2.)
    try:
        topic = os.environ.get("NTFY_TOPIC", "").strip()
        if not topic:
            return
        title = f"🛑 P79 run STOPPED [{condition_id}] — {abort_class}"
        hint = _ABORT_RECOVERY_HINTS.get(abort_class, "")
        try:
            exc_text = str(exc)[:300]
        except Exception:  # noqa: BLE001
            exc_text = f"<{type(exc).__name__}.__str__ raised>"
        body = (
            f"**{abort_class}** at site={site} task={task_id} — run 已停止"
            f"（后续 task 会以同样方式失败）。\n\n"
            f"错误: {exc_text}\n\n"
            f"{hint}"
        )
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}", data=body.encode("utf-8"), method="POST",
            headers={"Title": title, "Priority": "urgent", "Markdown": "yes"},
        )
        with urllib.request.urlopen(req, timeout=15):
            pass
        logger.info("Run-abort ntfy sent to %s (%s)", topic, abort_class)
    except Exception as push_exc:  # noqa: BLE001
        logger.warning("Run-abort ntfy failed: %s", push_exc)
