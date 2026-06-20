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
