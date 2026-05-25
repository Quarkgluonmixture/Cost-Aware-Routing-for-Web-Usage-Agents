#!/usr/bin/env python3
"""Per-run diagnostic; not part of the 4-dimension evidence framework."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SUMMARY_RE = re.compile(r"^(?P<site>.+)_task_(?P<task_id>\d+)_summary_v2\.json$")


# /stress A1.10 P0-2-AB* (2026-05-16): canonical progress signal for analyzer.
# Diagnostics rollups (wasted scroll, stuck, page-change-rate, ax_page_change_rate)
# semantically want "did the agent perceive progress?" — agent_visible_changed,
# not raw runner-internal page_changed (which fires on form_value_changed /
# dom_complexity_changed / text_length_changed / interactive_elements_changed /
# form_fields_changed reasons agents cannot see). Pre-fix all sites used raw
# page_changed → diagnostics rollups polluted by RUNNER_INTERNAL_REASONS noise.
# Post-fix: prefer agent_visible_changed when present (current runner emits it
# per runner/main.py:1710); fall back to page_changed for legacy archive
# records that pre-date the B-09 split. Mode B F3 catch.
def _progress_changed(step: Dict[str, Any]) -> bool:
    """Return the agent-perceivable page-change boolean for a step record,
    falling back to runner-internal page_changed for legacy records that
    pre-date the B-09 agent_visible_changed split.
    """
    av = step.get("agent_visible_changed")
    if av is not None:
        return bool(av)
    return bool(step.get("page_changed", False))

NO_RESULT_PATTERNS = (
    "no result",
    "not available",
    "not found",
    "impossible",
    "cannot",
    "without access",
    "missing",
    "not present",
)

PRICE_RANGE_PATTERNS = (
    re.compile(
        r"(?:between|within|from)\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s*(?:to|and|-|–)\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\$\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s*(?:to|-|–)\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
        re.IGNORECASE,
    ),
)

ANSWER_AMOUNT_PATTERNS = (
    re.compile(r"\$\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)"),
    re.compile(r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s*\$"),
)

UNCERTAIN_PATTERNS = (
    "could be",
    "however",
    "might",
    "likely",
    "not specified",
    "unclear",
)

SIM_REPEAT_THRESHOLD = 0.90

COLLECTION_PATTERNS = (
    re.compile(r"\b(return|provide)\s+the\s+links?\b", re.IGNORECASE),
    re.compile(r"\b(find|list)\s+all\b", re.IGNORECASE),
    re.compile(r"\b(top\s*\d+|three|two)\s+(most\s+recent|items?|listings?)\b", re.IGNORECASE),
    re.compile(r"\b(mileage|size|price|date).*(for|of)\s+(each|all)\b", re.IGNORECASE),
)

ACTION_PATTERNS = (
    re.compile(r"\b(add|post|write)\s+(a\s+)?(comment|message)\b", re.IGNORECASE),
    re.compile(r"\bsend\s+(a\s+)?message\b", re.IGNORECASE),
    re.compile(r"\bdelete\b", re.IGNORECASE),
    re.compile(r"\bedit\b", re.IGNORECASE),
)

PAGE_READING_PATTERNS = (
    re.compile(r"\bon this page\b", re.IGNORECASE),
    re.compile(r"\bthis item\b", re.IGNORECASE),
    re.compile(r"\bthis listing\b", re.IGNORECASE),
    re.compile(r"\bthis exact item\b", re.IGNORECASE),
)

GRID_POSITION_PATTERNS = (
    re.compile(r"\bin\s+the\s+(first|second|third|fourth|[0-9]+(?:st|nd|rd|th)?)\s+row\b", re.IGNORECASE),
    re.compile(r"\b(first|second|third|fourth|[0-9]+(?:st|nd|rd|th)?)\s+(car|item|listing|bike|phone|painting)\s+in\s+the\s+(first|second|third|[0-9]+(?:st|nd|rd|th)?)\s+row\b", re.IGNORECASE),
    re.compile(r"\b(first|second|third|fourth|[0-9]+(?:st|nd|rd|th)?)\s+row\b", re.IGNORECASE),
)

DATE_COUNT_PATTERNS = (
    re.compile(r"\bhow\s+many\b.{0,60}\bposted\s+on\b", re.IGNORECASE),
    re.compile(r"\bposted\s+on\s+\d", re.IGNORECASE),
    re.compile(r"\bposted\s+on\s+\w+\s+\d{1,2}", re.IGNORECASE),
)

LANG_TEXT: Dict[str, Dict[str, str]] = {
    "zh": {
        "report_title": "失败归因报告",
        "generated_at": "生成时间",
        "run_dir": "运行目录",
        "filters": "筛选条件",
        "global_bucket": "全局 bucket 分布",
        "condition": "Condition",
        "bucket_breakdown": "Bucket 分解",
        "bucket": "Bucket",
        "count": "数量",
        "rate": "占比",
        "thought_similarity_summary": "按 bucket 的 thought 相似度摘要（全对全）",
        "pair_count": "pair 数",
        "pair_mean": "均值",
        "pair_p90": "P90",
        "pair_high_rate": ">=0.90 比例",
        "high_similarity_patterns": "高相似 thought 模式（按 bucket）",
        "sample_thoughts": "{bucket} — 样本 thought（末 3 步）",
        "sample_header": "task_{task_id} | steps={steps} | eval={eval_type} | final_action={final_action_type}",
        "sample_context": "final_url={final_url} | reference_url={reference_url} | early_finish={early_finish} | hit_max_steps={hit_max_steps} | task_type={task_type} | loop={loop_pattern} | click_back_pairs={click_back_pairs} | max_search_repeat={max_search_query_repeat}",
        "sample_task_intent": "task_intent={task_intent}",
        "sample_final_answer": "final_answer={final_answer}",
        "sample_step0_thought": "step0(plan): {thought}",
        "sample_visited_reference": "ever_visited_reference_url={ever_visited_reference_url}",
        "sample_price_range_match": "answer_in_intent_price_range={answer_in_intent_price_range}",
        "step_thought": "step{step_idx}: {thought}",
        "similarity_skipped": "已跳过 thought 全对全相似度计算（--skip-similarity）",
        "cross_condition": "跨 condition bucket 对照",
        "condition_overview": "condition 总览",
        "episodes": "episodes",
        "success_rate": "success_rate",
        "early_finish_fail": "early_finish_fail",
        "fallback_finish": "fallback_finish",
        "none": "无",
    },
    "en": {
        "report_title": "Failure Reason Diagnostics Report",
        "generated_at": "Generated at",
        "run_dir": "Run dir",
        "filters": "Filters",
        "global_bucket": "Global bucket breakdown",
        "condition": "Condition",
        "bucket_breakdown": "Bucket breakdown",
        "bucket": "Bucket",
        "count": "Count",
        "rate": "Rate",
        "thought_similarity_summary": "Thought similarity summary by bucket (all-pairs)",
        "pair_count": "Pair count",
        "pair_mean": "Mean",
        "pair_p90": "P90",
        "pair_high_rate": ">=0.90 rate",
        "high_similarity_patterns": "High-similarity thought patterns by bucket",
        "sample_thoughts": "{bucket} — Sample thoughts (final 3 steps)",
        "sample_header": "task_{task_id} | steps={steps} | eval={eval_type} | final_action={final_action_type}",
        "sample_context": "final_url={final_url} | reference_url={reference_url} | early_finish={early_finish} | hit_max_steps={hit_max_steps} | task_type={task_type} | loop={loop_pattern} | click_back_pairs={click_back_pairs} | max_search_repeat={max_search_query_repeat}",
        "sample_task_intent": "task_intent={task_intent}",
        "sample_final_answer": "final_answer={final_answer}",
        "sample_step0_thought": "step0(plan): {thought}",
        "sample_visited_reference": "ever_visited_reference_url={ever_visited_reference_url}",
        "sample_price_range_match": "answer_in_intent_price_range={answer_in_intent_price_range}",
        "step_thought": "step{step_idx}: {thought}",
        "similarity_skipped": "Skipped all-pairs thought similarity (--skip-similarity).",
        "cross_condition": "Cross-condition bucket comparison",
        "condition_overview": "Condition overview",
        "episodes": "episodes",
        "success_rate": "success_rate",
        "early_finish_fail": "early_finish_fail",
        "fallback_finish": "fallback_finish",
        "none": "None",
    },
}


def _t(lang: str, key: str) -> str:
    if lang == "bilingual":
        return f"{LANG_TEXT['zh'][key]} / {LANG_TEXT['en'][key]}"
    return LANG_TEXT[lang][key]


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


from p79.experiment.io_utils import read_jsonl_dedup as _read_jsonl
# B-1860: single-source coordinate normalizer — the pixel_coordinate_leak
# detector below must judge a coord through the SAME Qwen 0-1000 contract the
# runner applies (a canonical 0-1000 coord is NOT a "pixel leak"; only a coord
# still outside [0,1] AFTER normalization is a true grounding leak).
from p79.backends.action_utils import normalize_coordinate_pair as _normalize_coordinate_pair


def _sort_steps_by_idx(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(steps, key=lambda x: int(x.get("step_idx", -1)))


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_task_meta(run_dir: Path, site: str, task_id: int, cache: Dict[Tuple[str, int], Dict[str, Any]]) -> Dict[str, Any]:
    key = (site, task_id)
    if key in cache:
        return cache[key]
    task_cfg = run_dir / "task_configs" / f"{site}_task_{task_id}.json"
    if task_cfg.exists():
        data = _read_json(task_cfg)
    else:
        data = {}
    cache[key] = data
    return data


def _normalize_url_candidates(ref_url: Any) -> List[str]:
    if not isinstance(ref_url, str):
        return []
    if "|OR|" in ref_url:
        return [x.strip() for x in ref_url.split("|OR|") if x.strip()]
    return [ref_url.strip()] if ref_url.strip() else []


def _extract_item_ids_from_urls(urls: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for u in urls:
        m = re.search(r"[?&]id=(\d+)", str(u or ""))
        if not m:
            continue
        iid = m.group(1)
        if iid in seen:
            continue
        seen.add(iid)
        out.append(iid)
    return out


def _step_obs_url(step: Dict[str, Any]) -> str:
    url = str(step.get("obs_url", "") or "").strip()
    if url:
        return url
    digest = step.get("state_digest") or {}
    after = str(digest.get("url_after", "") or "").strip()
    if after:
        return after
    before = str(digest.get("url_before", "") or "").strip()
    return before


def _resolve_artifact_path(path_text: str, run_dir: Path) -> Optional[Path]:
    raw = str(path_text or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if p.exists():
        return p
    if p.is_absolute():
        return p if p.exists() else None
    cwd_p = (Path.cwd() / p).resolve()
    if cwd_p.exists():
        return cwd_p
    try:
        repo_root = run_dir.parents[3]
        repo_p = (repo_root / p).resolve()
        if repo_p.exists():
            return repo_p
    except Exception:
        pass
    return None


def _url_to_page_type(url: str) -> str:
    u = str(url or "").lower()
    if not u:
        return "unknown"
    # Classifieds
    if "page=item" in u:
        return "detail"
    if "page=search" in u:
        return "search"
    if "page=user" in u or "my+account" in u:
        return "account"
    # Shopping (Magento-style)
    if "/customer/account" in u or "/account/login" in u:
        return "account"
    if "/catalogsearch/" in u:
        return "search"
    if "/catalog/product" in u or "/product/" in u:
        return "detail"
    # Reddit
    if "/login" in u or "/register" in u:
        return "account"
    if "/r/" in u and "/comments/" in u:
        return "detail"
    if "/r/" in u and "/search" in u:
        return "search"
    # Wikipedia
    if "/wiki/" in u:
        return "detail"
    return "other"


def _to_number(num_text: str) -> Optional[float]:
    try:
        return float(str(num_text).replace(",", "").strip())
    except Exception:
        return None


def _extract_intent_price_ranges(intent: str) -> List[Tuple[float, float]]:
    ranges: List[Tuple[float, float]] = []
    text = str(intent or "")
    for pat in PRICE_RANGE_PATTERNS:
        for m in pat.finditer(text):
            lo = _to_number(m.group(1))
            hi = _to_number(m.group(2))
            if lo is None or hi is None:
                continue
            if lo > hi:
                lo, hi = hi, lo
            ranges.append((lo, hi))
    # Deduplicate while preserving order.
    seen = set()
    deduped: List[Tuple[float, float]] = []
    for lo, hi in ranges:
        key = (round(lo, 4), round(hi, 4))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((lo, hi))
    return deduped


def _extract_answer_amounts(answer: str) -> List[float]:
    amounts: List[float] = []
    text = str(answer or "")
    for pat in ANSWER_AMOUNT_PATTERNS:
        for m in pat.finditer(text):
            val = _to_number(m.group(1))
            if val is not None:
                amounts.append(val)
    # Deduplicate with stable order.
    seen = set()
    uniq: List[float] = []
    for x in amounts:
        k = round(x, 4)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(x)
    return uniq


def _answer_in_intent_price_range(answer: str, intent: str) -> Optional[bool]:
    ranges = _extract_intent_price_ranges(intent)
    if not ranges:
        return None
    amounts = _extract_answer_amounts(answer)
    if not amounts:
        return False
    for a in amounts:
        for lo, hi in ranges:
            if lo <= a <= hi:
                return True
    return False


def _ever_visited_reference_url(steps: List[Dict[str, Any]], ref_urls: List[str]) -> Optional[bool]:
    if not ref_urls:
        return None
    refs = {x.strip() for x in ref_urls if x and x.strip()}
    if not refs:
        return None
    for s in steps:
        digest = s.get("state_digest") or {}
        url_before = str(digest.get("url_before", "") or "")
        url_after = str(digest.get("url_after", "") or "")
        obs_url = _step_obs_url(s)
        if url_before in refs or url_after in refs or obs_url in refs:
            return True
    return False


def _target_item_ever_visible(steps: List[Dict[str, Any]], ref_urls: List[str], run_dir: Path) -> Optional[bool]:
    item_ids = _extract_item_ids_from_urls(ref_urls)
    if not item_ids:
        return None
    # Use stricter ID-context patterns than bare substring match: short IDs
    # (e.g. "12") would otherwise match arbitrary digits in DOM text — prices,
    # timestamps, other product IDs containing the digits as substring.
    # Match `id=NNNN` (URL query) or `?id=NNNN` or `/NNNN` (path segment).
    id_patterns = [
        re.compile(rf"(?:[?&]id={re.escape(iid)}\b|/{re.escape(iid)}\b|\bid=\"?{re.escape(iid)}\b)")
        for iid in item_ids
    ]
    dom_found_any = False
    for s in steps:
        dom_path_raw = str(((s.get("artifact_paths") or {}).get("dom")) or "").strip()
        dom_path = _resolve_artifact_path(dom_path_raw, run_dir)
        if not dom_path or not dom_path.exists():
            continue
        dom_found_any = True
        try:
            text = dom_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if any(p.search(text) for p in id_patterns):
            return True
    # If no DOM artifacts exist (e.g. vision-only mode), we cannot determine visibility.
    if not dom_found_any:
        return None
    return False


def _normalize_thought(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _first_words(text: str, max_words: int = 12) -> str:
    parts = _normalize_thought(text).split(" ")
    if not parts or parts == [""]:
        return ""
    return " ".join(parts[:max_words])


def _contains_any(text: str, patterns: Tuple[str, ...]) -> bool:
    low = (text or "").lower()
    return any(p in low for p in patterns)


def _is_finish(action_type: str) -> bool:
    return str(action_type).lower() in ("finish", "stop")


def _safe_ratio(numer: int, denom: int) -> float:
    return float(numer) / float(denom) if denom > 0 else 0.0


def _p90(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(0.9 * (len(ordered) - 1))
    return float(ordered[idx])


def _sequence_similarity(a: str, b: str) -> float:
    return float(SequenceMatcher(None, a, b).ratio())


def _compress_action_types(steps: List[Dict[str, Any]]) -> str:
    seq = [str((s.get("action") or {}).get("action_type", "")).lower() for s in steps]
    seq = [x for x in seq if x]
    if not seq:
        return ""
    chunks: List[str] = []
    cur = seq[0]
    cnt = 1
    for x in seq[1:]:
        if x == cur:
            cnt += 1
        else:
            chunks.append(f"{cur}×{cnt}")
            cur = x
            cnt = 1
    chunks.append(f"{cur}×{cnt}")
    return "|".join(chunks)


def _scroll_direction_stats(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze scroll direction usage per episode.

    Returns dict with:
        scroll_up: count of scroll actions with dy < 0 (up)
        scroll_down: count of scroll actions with dy >= 0 (down)
        scroll_direction_flips: number of consecutive direction changes
        scroll_wasted_steps: scroll actions where page_changed=False
    """
    scroll_up = 0
    scroll_down = 0
    direction_flips = 0
    wasted = 0
    prev_direction: Optional[str] = None

    for s in steps:
        atype = str(s.get("action_type", "") or "").lower()
        if atype != "scroll":
            continue
        act = s.get("action") or {}
        delta = act.get("delta")
        # Extract dy from various delta formats
        if isinstance(delta, (list, tuple)) and len(delta) >= 2:
            dy = delta[1]
        elif isinstance(delta, (list, tuple)) and len(delta) == 1:
            dy = delta[0]
        elif isinstance(delta, (int, float)):
            dy = delta
        else:
            continue  # unparseable delta, skip

        # dy == 0 is not a real scroll — skip rather than coerce to "down"
        # (which would inflate scroll_down stats).
        if dy == 0:
            continue
        direction = "down" if dy > 0 else "up"
        if direction == "up":
            scroll_up += 1
        else:
            scroll_down += 1

        if prev_direction is not None and direction != prev_direction:
            direction_flips += 1
        prev_direction = direction

        # /stress A1.10 P0-2-AB* — agent-visible progress for scroll-wasted attribution.
        if not _progress_changed(s):
            wasted += 1

    return {
        "scroll_up": scroll_up,
        "scroll_down": scroll_down,
        "scroll_direction_flips": direction_flips,
        "scroll_wasted_steps": wasted,
    }


def _page_unchanged_signals(steps: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    stuck_first_step = -1
    max_len = 0
    max_pos = -1
    cur_len = 0
    cur_pos = -1

    for s in steps:
        step_idx = int(s.get("step_idx", -1))
        # /stress A1.10 P0-2-AB* — agent-visible progress for stuck-detection.
        changed = _progress_changed(s)
        if not changed:
            if stuck_first_step == -1:
                stuck_first_step = step_idx
            if cur_len == 0:
                cur_pos = step_idx
            cur_len += 1
            if cur_len > max_len:
                max_len = cur_len
                max_pos = cur_pos
        else:
            cur_len = 0
            cur_pos = -1

    return stuck_first_step, max_len, max_pos


def _url_revisit_metrics(steps: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    """Count URL revisits across an episode.

    Returns:
        url_revisit_count: total number of steps that revisit a previously-seen URL
        url_unique_count: number of distinct URLs visited
        url_revisit_max: max times any single URL was visited
    """
    from collections import Counter as _Counter
    url_counts: _Counter = _Counter()
    for s in steps:
        # Use the same fallback chain as _step_obs_url for consistency:
        # obs_url → url_after → url_before. Previously this skipped url_before,
        # silently dropping visits when both other fields were empty.
        digest = s.get("state_digest") or {}
        url = str(
            s.get("obs_url", "")
            or digest.get("url_after", "")
            or digest.get("url_before", "")
            or ""
        ).strip()
        if url:
            url_counts[url] += 1
    url_unique_count = len(url_counts)
    url_revisit_count = sum(v - 1 for v in url_counts.values() if v > 1)
    url_revisit_max = max(url_counts.values()) if url_counts else 0
    return url_revisit_count, url_unique_count, url_revisit_max


def _action_diversity_metrics(steps: List[Dict[str, Any]]) -> Tuple[float, int]:
    """Measure action type diversity across an episode.

    Returns:
        action_diversity: ratio of unique action types to total steps (0-1)
        action_unique_types: number of distinct action types used
    """
    action_types = []
    for s in steps:
        act = s.get("action") or {}
        atype = str(act.get("action_type", "") or "").lower()
        if atype:
            action_types.append(atype)
    if not action_types:
        return 0.0, 0
    unique = len(set(action_types))
    return unique / len(action_types), unique


def _thought_snapshots(steps: List[Dict[str, Any]]) -> Tuple[str, str, str, str, List[Dict[str, Any]], List[Tuple[int, str, str]]]:
    step_to_thought: Dict[int, str] = {}
    trajectory: List[Dict[str, Any]] = []
    non_empty_norm: List[Tuple[int, str, str]] = []

    for s in steps:
        step_idx = int(s.get("step_idx", -1))
        action = s.get("action") or {}
        thought = str(action.get("thought", "") or "")
        thought_norm = _normalize_thought(thought)
        if thought.strip():
            step_to_thought[step_idx] = thought
        if thought_norm:
            non_empty_norm.append((step_idx, thought_norm, thought))

        trajectory.append(
            {
                "step_idx": step_idx,
                "action_type": str(s.get("action_type", "")).lower(),
                "page_changed": bool(s.get("page_changed", False)),
                "thought": thought,
                "thought_norm": thought_norm,
            }
        )

    thought_at_0 = step_to_thought.get(0, "")
    thought_at_5 = step_to_thought.get(5, "")
    thought_at_10 = step_to_thought.get(10, "")
    final_thought = step_to_thought.get(int(steps[-1].get("step_idx", -1)), "") if steps else ""
    return thought_at_0, thought_at_5, thought_at_10, final_thought, trajectory, non_empty_norm


def _thought_similarity_features(
    non_empty_norm: List[Tuple[int, str, str]],
    *,
    skip_pairwise: bool = False,
) -> Dict[str, Any]:
    thought_count = len(non_empty_norm)
    unique_count = len({x[1] for x in non_empty_norm})
    thought_diversity = _safe_ratio(unique_count, thought_count)

    adjacent_values: List[float] = []
    for i in range(len(non_empty_norm) - 1):
        adjacent_values.append(_sequence_similarity(non_empty_norm[i][1], non_empty_norm[i + 1][1]))

    max_adj = max(adjacent_values) if adjacent_values else 0.0
    repeat_rate = _safe_ratio(sum(1 for v in adjacent_values if v >= SIM_REPEAT_THRESHOLD), len(adjacent_values))

    pair_values: List[float] = []
    high_template_counter: Counter = Counter()
    if not skip_pairwise:
        for i in range(len(non_empty_norm)):
            for j in range(i + 1, len(non_empty_norm)):
                sim = _sequence_similarity(non_empty_norm[i][1], non_empty_norm[j][1])
                pair_values.append(sim)
                if sim >= SIM_REPEAT_THRESHOLD:
                    sig_i = _first_words(non_empty_norm[i][1], 12)
                    sig_j = _first_words(non_empty_norm[j][1], 12)
                    if sig_i:
                        high_template_counter[sig_i] += 1
                    if sig_j:
                        high_template_counter[sig_j] += 1

    return {
        "thought_count": thought_count,
        "thought_unique_count": unique_count,
        "thought_diversity": thought_diversity,
        "thought_similarity_max_adjacent": max_adj,
        "thought_similarity_repeat_rate": repeat_rate,
        "thought_similarity_mean_all_pairs": (sum(pair_values) / len(pair_values)) if pair_values else 0.0,
        "thought_similarity_p90_all_pairs": _p90(pair_values),
        "thought_similarity_high_rate_all_pairs": _safe_ratio(
            sum(1 for v in pair_values if v >= SIM_REPEAT_THRESHOLD), len(pair_values)
        ),
        "all_pair_values": pair_values,
        "high_template_counter": high_template_counter,
    }


def _collect_final_thoughts(steps: List[Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in steps[-k:]:
        action = s.get("action") or {}
        out.append(
            {
                "step_idx": int(s.get("step_idx", -1)),
                "thought": str(action.get("thought", "") or ""),
            }
        )
    return out


def _collect_select_events(steps: List[Dict[str, Any]], max_events: int = 24) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in steps:
        action = s.get("action") or {}
        action_type = str(action.get("action_type", "") or s.get("action_type", "")).lower()
        if action_type != "select":
            continue
        digest = s.get("state_digest") or {}
        option = (
            str(action.get("option", "") or "").strip()
            or str(action.get("value", "") or "").strip()
            or str(action.get("label", "") or "").strip()
            or str(action.get("text", "") or "").strip()
        )
        out.append(
            {
                "step_idx": int(s.get("step_idx", -1)),
                "element_id": str(action.get("element_id", "") or "").strip(),
                "option": option,
                "page_changed": bool(s.get("page_changed", False)),
                "action_success": bool(s.get("action_success", False)),
                "obs_url": str(s.get("obs_url", "") or digest.get("url_after", "") or "").strip(),
            }
        )
    if len(out) <= max_events:
        return out
    return out[-max_events:]


def _classify_task_type(intent: str) -> str:
    text = str(intent or "")
    for pat in GRID_POSITION_PATTERNS:
        if pat.search(text):
            return "grid_position"
    for pat in DATE_COUNT_PATTERNS:
        if pat.search(text):
            return "date_count"
    for pat in PAGE_READING_PATTERNS:
        if pat.search(text):
            return "page_reading"
    for pat in COLLECTION_PATTERNS:
        if pat.search(text):
            return "collection"
    for pat in ACTION_PATTERNS:
        if pat.search(text):
            return "action_on_item"
    return "single_navigation"


INTENT_FEATURE_PATTERNS = {
    "intent_has_sort": re.compile(r"\b(sort|order)\s+(by|from)\b", re.I),
    "intent_has_filter": re.compile(r"\b(filter|refine|narrow)\b", re.I),
    "intent_has_compare": re.compile(r"\b(compar|cheaper|cheapest|most\s+expensive|vs\.?)\b", re.I),
    "intent_has_count": re.compile(r"\b(how\s+many|count|number\s+of|total)\b", re.I),
    "intent_has_latest": re.compile(r"\b(latest|most\s+recent|newest|last\s+posted)\b", re.I),
    "intent_has_color": re.compile(r"\b(red|blue|green|black|white|yellow|purple|pink|grey|gray|brown|orange|silver|gold)\b", re.I),
    "intent_has_image": None,  # from config.get("image")
    "intent_has_price": re.compile(r"\$\s*\d|price|cost|budget|afford", re.I),
    "intent_has_location": re.compile(r"\b(from|in|near|located)\s+[A-Z]", re.I),
    "intent_needs_scroll": re.compile(r"\b(all|every|each|complete\s+list|bottom|end\s+of)\b", re.I),
}


def _extract_intent_features(intent: str, config: dict) -> Dict[str, bool]:
    """Extract boolean intent feature flags from task intent text."""
    features: Dict[str, bool] = {}
    for key, pattern in INTENT_FEATURE_PATTERNS.items():
        if key == "intent_has_image":
            features[key] = bool(config.get("image"))
        elif pattern is not None:
            features[key] = bool(pattern.search(intent))
        else:
            features[key] = False
    return features


def _compute_step_cost_breakdown(steps: List[Dict[str, Any]], loop_metrics: Dict[str, Any]) -> Dict[str, float]:
    """Compute fine-grained cost breakdown from step records."""
    total_cost = 0.0
    no_op_cost = 0.0
    page_unchanged_cost = 0.0
    for s in steps:
        # Defensive: cost_usd may be explicitly None on partial/error rows.
        step_cost = float((s.get("cost_usd") or {}).get("total", 0))
        total_cost += step_cost
        if s.get("action_success") is False:
            no_op_cost += step_cost
        action_type = str((s.get("action") or {}).get("action_type", "") or "").lower()
        # /stress A1.10 P0-2-AB* — agent-visible progress for page-unchanged cost attribution.
        if not _progress_changed(s) and action_type not in ("finish", "stop"):
            page_unchanged_cost += step_cost
    # Loop cost: estimate from loop_pattern
    loop_pattern = str(loop_metrics.get("loop_pattern", "") or "")
    loop_cost = 0.0
    if "repeat" in loop_pattern.lower() or "cycle" in loop_pattern.lower():
        # Rough heuristic: click_back_pairs * 2 steps worth of cost
        click_back = int(loop_metrics.get("click_back_pairs", 0))
        search_repeat = int(loop_metrics.get("max_search_query_repeat", 0))
        loop_steps = max(click_back * 2, (search_repeat - 1) * 2)
        if loop_steps > 0 and len(steps) > 0:
            avg_step_cost = total_cost / len(steps)
            loop_cost = min(avg_step_cost * loop_steps, total_cost)
    return {
        "no_op_cost_usd": no_op_cost,
        "page_unchanged_cost_usd": page_unchanged_cost,
        "loop_cost_usd": loop_cost,
        "effective_cost_usd": max(0.0, total_cost - no_op_cost - loop_cost),
    }


def _collection_overlap_score(final_answer: str, ref_answers: Any) -> Optional[float]:
    """For must_include reference_answers with multiple URLs, compute fraction found in final_answer."""
    if not ref_answers or not isinstance(ref_answers, dict):
        return None
    must_include = ref_answers.get("must_include")
    if not must_include or not isinstance(must_include, list) or len(must_include) < 2:
        return None
    answer_lower = final_answer.lower()
    found = sum(1 for item in must_include if str(item).lower() in answer_lower)
    return round(found / len(must_include), 3)


def _classify_stuck_subtype(
    *,
    reason_bucket: str,
    task_type: str,
    target_item_ever_visible: Optional[bool],
    loop_pattern: str,
    max_search_query_repeat: int,
    page_type_sequence: str,
    click_back_pairs: int,
    action_type_sequence: str = "",
) -> str:
    """Sub-classify fail_incomplete_or_stuck / fail_no_progress into meaningful subtypes."""
    if reason_bucket not in ("fail_incomplete_or_stuck", "fail_no_progress"):
        return ""
    # account/login loop: page type sequence has DOMINANT account visits.
    # Old check (any single "account" → loop) over-flagged single-visit
    # tasks (~33% FP rate observed). `_url_to_page_type` never returns
    # "login", so the "login in ..." branch was dead code — removed.
    _seq_lower = page_type_sequence.lower()
    if _seq_lower.count("account") >= 3:
        return "account_loop"
    if target_item_ever_visible is False:
        return "target_unreachable"
    if task_type == "page_reading":
        return "page_reading_mismatch"
    # Scroll-only stall: actions are dominated by scroll with no click/type
    _seq = str(action_type_sequence or "").lower()
    _scroll_dominant = "scroll" in _seq and "click" not in _seq and "type" not in _seq
    if _scroll_dominant:
        return "scroll_static"
    if int(max_search_query_repeat) >= 2:
        return "search_no_result"
    if int(click_back_pairs) >= 2:
        return "nav_loop"
    if target_item_ever_visible is True:
        return "target_visible_not_entered"
    return "unknown"


# Location names that appear as constraints in task intents (classifieds/shopping tasks)
_LOCATION_NAME_RE = re.compile(
    r"\b(delaware|ohio|california|new york|virginia|maryland|texas|florida|"
    r"pennsylvania|illinois|washington|oregon|michigan|georgia|colorado|"
    r"arizona|nevada|utah|connecticut|massachusetts|new jersey|hawaii|alaska)\b",
    re.IGNORECASE,
)
_VISUAL_MATCH_KWDS = (
    # Explicit image-comparison tasks (input image provided)
    "in the image", "same as", "same item", "same brand", "same product",
    "similar items", "of the image", "shown in", "like the product",
    # Visual-attribute tasks: attribute must be read from listing photo
    "on the cover", "cover image", "selfie", "taken as a selfie",
    "hard-case", "hard case", "color of", "colour of",
    "purple", "red", "blue", "green", "yellow", "orange", "pink", "white", "black",
    "pattern", "stripe", "floral", "checkered",
)

# Buckets where visual-DOM unreachability causes a loop rather than direct timeout
_LOOP_BUCKETS = frozenset((
    "fail_max_steps_target_unreachable",
    "fail_max_steps_click_back_loop",
    "fail_max_steps_search_repeat",
    "fail_incomplete_or_stuck",
    "fail_no_progress",
))


def _classify_unreachable_subtype(
    *,
    reason_bucket: str,
    task_intent: str,
    observation_mode: str,
    search_queries: List[Dict[str, Any]],
    degraded_som_steps: int = 0,
    total_steps: int = 0,
    has_image: bool = False,
) -> str:
    """Sub-classify target-unreachable structural defects.

    Covers both direct timeout (fail_max_steps_target_unreachable) and loop
    buckets where the underlying cause is a visual attribute invisible to DOM.
    """
    if reason_bucket not in _LOOP_BUCKETS:
        return ""
    intent_lower = str(task_intent or "").lower()
    obs = str(observation_mode or "").lower()

    # Visual-attribute unreachability: applies to DOM mode unconditionally, and
    # to SoM/hybrid mode when SoM is degraded for a DOMINANT share of steps.
    # Old check (any single degraded step → dom-like) over-flagged 30-step
    # episodes with one bad probe — switched to >=30% of steps degraded.
    # Vision mode is excluded: the model always receives the raw screenshot
    # and visual failures there are model capability issues, not scaffold defects.
    has_visual = any(k in intent_lower for k in _VISUAL_MATCH_KWDS)
    _DEGRADED_DOMINANT_RATIO = 0.30
    _som_dominant_degraded = (
        total_steps > 0
        and (degraded_som_steps / total_steps) >= _DEGRADED_DOMINANT_RATIO
    )
    is_dom_like = obs == "dom" or (obs in ("som", "hybrid") and _som_dominant_degraded)
    if has_visual and is_dom_like:
        # Tasks with reference images (has_image=True) provide the image in
        # the prompt — DOM mode can now see it.  The failure is a model
        # capability issue (can't match ref image to listings), not a
        # scaffold defect.  Keyword-only visual tasks (has_image=False)
        # remain structurally unreachable in DOM mode.
        if has_image:
            return "visual_has_ref_image"
        return "visual_dom_only"

    # Location constraint only applies to the direct-timeout bucket; for loop
    # buckets the primary cause is already the loop pattern, not location.
    if reason_bucket == "fail_max_steps_target_unreachable":
        has_location = bool(_LOCATION_NAME_RE.search(intent_lower))
        if has_location:
            loc_as_keyword = any(
                _LOCATION_NAME_RE.search(str(q.get("query", "")).lower())
                for q in (search_queries or [])
            )
            return "location_filter_keyword" if loc_as_keyword else "location_filter"

    return ""


def _normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\n", " ").strip().lower())


def _extract_search_queries(steps: List[Dict[str, Any]], max_queries: int = 40) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in steps:
        act = s.get("action") or {}
        action_type = str(act.get("action_type", "") or s.get("action_type", "")).lower()
        if action_type != "type":
            continue
        raw_text = str(act.get("text", "") or "").strip()
        q = _normalize_query(raw_text)
        if len(q) <= 2:
            continue
        out.append(
            {
                "step_idx": int(s.get("step_idx", -1)),
                "query": q[:160],
                "element_id": str(act.get("element_id", "") or "").strip(),
                "page_changed": bool(s.get("page_changed", False)),
                "obs_url": _step_obs_url(s),
            }
        )
    if len(out) <= max_queries:
        return out
    return out[-max_queries:]


def _detect_loops(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    action_types: List[str] = []
    for s in steps:
        act = s.get("action") or {}
        action_types.append(str(act.get("action_type", "") or s.get("action_type", "")).lower())
    click_back_pairs = sum(
        1
        for i in range(len(action_types) - 1)
        if action_types[i] == "click" and action_types[i + 1] == "back"
    )

    search_queries = _extract_search_queries(steps)
    query_counter: Counter = Counter(q["query"] for q in search_queries if str(q.get("query", "")).strip())
    max_repeat = max(query_counter.values()) if query_counter else 0
    most_repeated = query_counter.most_common(1)[0][0] if query_counter else ""
    unique_search_queries = len(query_counter)

    if click_back_pairs >= 3:
        loop_pattern = "click_back_loop"
    elif max_repeat >= 3:
        loop_pattern = "search_repeat_loop"
    else:
        loop_pattern = "none"

    return {
        "click_back_pairs": int(click_back_pairs),
        "max_search_query_repeat": int(max_repeat),
        "most_repeated_search_query": most_repeated[:80],
        "unique_search_queries": int(unique_search_queries),
        "loop_pattern": loop_pattern,
        "search_queries": search_queries,
    }


def _page_type_sequence(steps: List[Dict[str, Any]], max_items: int = 40) -> str:
    seq: List[str] = []
    for s in steps:
        seq.append(_url_to_page_type(_step_obs_url(s)))
    if len(seq) > max_items:
        seq = seq[-max_items:]
    return "|".join(seq)


def _classify_reason(
    *,
    success: bool,
    summary_error: Optional[str],
    final_action_type: str,
    final_error_category: Optional[str],
    final_answer: str,
    eval_type: str,
    early_finish: bool,
    hit_max_steps: bool,
    click_back_pairs: int,
    max_search_query_repeat: int,
    target_item_ever_visible: Optional[bool],
    final_url_match: Optional[bool],
    ever_visited_reference_url: Optional[bool],
    final_answer_in_intent_price_range: Optional[bool],
) -> str:
    if success:
        return "success"

    if summary_error:
        return "fail_summary_error"

    if final_error_category == "benchmark_noise":
        return "fail_benchmark_noise"
    if final_error_category == "env_error":
        return "fail_env_error"
    if final_error_category == "parse_error":
        return "fail_parse_error"
    # P1-5 (/stress accounting audit 2026-05-21): off-site goto blocked by the
    # VWA-origin whitelist (B-1782). A direct error_category → reason_bucket map
    # (like parse_error) so the new policy-blocked failure mode reaches the paper
    # §5 taxonomy instead of falling through to fail_max_steps / other-failure.
    if final_error_category == "policy_blocked_offsite":
        return "fail_policy_blocked_offsite"

    if _is_finish(final_action_type):
        if early_finish:
            return "fail_early_finish"
        if (final_answer or "").strip() == "":
            return "fail_finish_empty_answer"
        if _contains_any(final_answer, NO_RESULT_PATTERNS):
            return "fail_finish_claim_missing"
        if "url_match" in eval_type and final_url_match is False:
            if ever_visited_reference_url:
                return "fail_finish_wrong_url_left_target"
            if final_answer_in_intent_price_range is False:
                return "fail_finish_wrong_url_price_mismatch"
            return "fail_finish_wrong_url_not_found"
        return "fail_finish_eval_mismatch"

    if hit_max_steps:
        if "url_match" in eval_type and target_item_ever_visible is False:
            return "fail_max_steps_target_unreachable"
        if int(click_back_pairs) >= 3:
            return "fail_max_steps_click_back_loop"
        if int(max_search_query_repeat) >= 3:
            return "fail_max_steps_search_repeat"
        return "fail_max_steps"

    if final_error_category == "no_progress":
        return "fail_no_progress"

    return "fail_incomplete_or_stuck"


def _select_samples(rows: List[Dict[str, Any]], n: int) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(str(r["condition_id"]), str(r["reason_bucket"]))].append(r)

    selected: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for key, grp in grouped.items():
        grp_sorted = sorted(grp, key=lambda x: (-int(x.get("steps", 0)), int(x.get("task_id", 0))))
        selected[key] = grp_sorted[:n]
    return selected


def _render_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---" for _ in headers]) + "|")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _fmt_pct(v: float) -> str:
    return f"{100.0 * float(v):.1f}%"


def _write_bucket_thought_samples(
    path: Path,
    episode_rows: List[Dict[str, Any]],
    samples_per_bucket: int,
) -> None:
    samples = _select_samples(episode_rows, samples_per_bucket)
    lines: List[str] = []

    for (condition_id, bucket) in sorted(samples.keys()):
        if bucket == "success":
            continue
        grp_count = sum(
            1 for x in episode_rows if x["condition_id"] == condition_id and x["reason_bucket"] == bucket
        )
        lines.append(f"=== {condition_id} | {bucket} ({grp_count} cases) ===")
        for row in samples[(condition_id, bucket)]:
            lines.append(
                f"[task_{row['task_id']}, steps:{row['steps']}, eval:{row.get('eval_type','')}, final_action:{row.get('final_action_type','')}] final thoughts:"
            )
            lines.append(f"  task_intent: {str(row.get('task_intent', '') or '').strip()}")
            lines.append(
                f"  ever_visited_reference_url: {row.get('ever_visited_reference_url', None)}"
            )
            if row.get("answer_in_intent_price_range", None) is not None:
                lines.append(
                    f"  answer_in_intent_price_range: {row.get('answer_in_intent_price_range')}"
                )
            if _is_finish(str(row.get("final_action_type", "") or "")):
                lines.append(f"  final_answer: {str(row.get('final_answer_excerpt', '') or '').strip()}")
            step0_thought = str(row.get("thought_at_step_0", "") or "").strip()
            if step0_thought:
                lines.append(f"  step0(plan): {step0_thought}")
            for item in row.get("final_three_thoughts", []):
                step_idx = item.get("step_idx", -1)
                thought = str(item.get("thought", "") or "").strip()
                lines.append(f"  step{step_idx}: {thought}")
            lines.append(
                "  context: "
                f"final_url={row.get('final_url','')} | reference_url={row.get('reference_url','')} | "
                f"early_finish={row.get('early_finish', False)} | hit_max_steps={row.get('hit_max_steps', False)} | "
                f"task_type={row.get('task_type','')} | loop_pattern={row.get('loop_pattern','')} | "
                f"click_back_pairs={row.get('click_back_pairs', 0)} | max_search_query_repeat={row.get('max_search_query_repeat', 0)}"
            )
            lines.append("")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


def _build_report(
    *,
    output_path: Path,
    lang: str,
    run_dir: Path,
    filters: Dict[str, Any],
    episode_rows: List[Dict[str, Any]],
    cond_rows: List[Dict[str, Any]],
    bucket_rows: List[Dict[str, Any]],
    bucket_similarity_rows: List[Dict[str, Any]],
    bucket_template_rows: List[Dict[str, Any]],
    samples_per_bucket: int,
    skip_similarity: bool,
) -> None:
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines: List[str] = []

    lines.append(f"# {_t(lang, 'report_title')}")
    lines.append("")
    lines.append(f"- {_t(lang, 'generated_at')}: `{now}`")
    lines.append(f"- {_t(lang, 'run_dir')}: `{run_dir}`")
    lines.append(f"- {_t(lang, 'filters')}: `{json.dumps(filters, ensure_ascii=False)}`")
    lines.append("")

    global_bucket_counter = Counter(str(x["reason_bucket"]) for x in episode_rows)
    gb_rows = [[k, str(v), _fmt_pct(v / len(episode_rows))] for k, v in global_bucket_counter.most_common()]
    lines.append(f"## {_t(lang, 'global_bucket')}")
    lines.append(_render_markdown_table([_t(lang, "bucket"), _t(lang, "count"), _t(lang, "rate")], gb_rows))
    lines.append("")

    # Index for quick lookup
    bucket_by_condition: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in bucket_rows:
        bucket_by_condition[str(r["condition_id"])].append(r)

    sim_by_condition: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in bucket_similarity_rows:
        sim_by_condition[str(r["condition_id"])].append(r)

    tpl_by_condition_bucket: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in bucket_template_rows:
        tpl_by_condition_bucket[(str(r["condition_id"]), str(r["reason_bucket"]))].append(r)

    samples = _select_samples(episode_rows, samples_per_bucket)

    for cond in sorted(cond_rows, key=lambda x: str(x["condition_id"])):
        cid = str(cond["condition_id"])
        lines.append(
            f"## {_t(lang, 'condition')}: `{cid}` ({cond['episodes']} {_t(lang, 'episodes')}, {_fmt_pct(float(cond['success_rate']))} {_t(lang, 'success_rate')})"
        )
        lines.append("")

        # bucket breakdown
        cond_bucket_rows = sorted(
            bucket_by_condition.get(cid, []), key=lambda x: float(x.get("rate_in_condition", 0.0)), reverse=True
        )
        table_rows = [
            [
                str(r["reason_bucket"]),
                str(r["count"]),
                _fmt_pct(float(r.get("rate_in_condition", 0.0))),
            ]
            for r in cond_bucket_rows
        ]
        lines.append(f"### {_t(lang, 'bucket_breakdown')}")
        lines.append(_render_markdown_table([_t(lang, "bucket"), _t(lang, "count"), _t(lang, "rate")], table_rows))
        lines.append("")

        # similarity summary by bucket
        sim_rows = sorted(sim_by_condition.get(cid, []), key=lambda x: str(x.get("reason_bucket", "")))
        if skip_similarity:
            lines.append(f"### {_t(lang, 'thought_similarity_summary')}")
            lines.append(f"- {_t(lang, 'similarity_skipped')}")
            lines.append("")
        elif sim_rows:
            lines.append(f"### {_t(lang, 'thought_similarity_summary')}")
            lines.append(
                _render_markdown_table(
                    [_t(lang, "bucket"), _t(lang, "pair_count"), _t(lang, "pair_mean"), _t(lang, "pair_p90"), _t(lang, "pair_high_rate")],
                    [
                        [
                            str(r["reason_bucket"]),
                            str(r["pair_count"]),
                            f"{float(r['pair_similarity_mean']):.3f}",
                            f"{float(r['pair_similarity_p90']):.3f}",
                            _fmt_pct(float(r["pair_similarity_high_rate"])),
                        ]
                        for r in sim_rows
                    ],
                )
            )
            lines.append("")

        # high similarity patterns
        lines.append(f"### {_t(lang, 'high_similarity_patterns')}")
        if skip_similarity:
            lines.append(f"- {_t(lang, 'similarity_skipped')}")
            lines.append("")
        else:
            has_tpl = False
            for r in cond_bucket_rows:
                bucket = str(r["reason_bucket"])
                tpl_rows = tpl_by_condition_bucket.get((cid, bucket), [])[:5]
                if not tpl_rows:
                    continue
                has_tpl = True
                lines.append(f"- `{bucket}`")
                for t in tpl_rows:
                    lines.append(f"  - `{t['template_signature']}` ({t['count']})")
            if not has_tpl:
                lines.append(f"- {_t(lang, 'none')}")
            lines.append("")

        # samples by failure bucket
        for r in cond_bucket_rows:
            bucket = str(r["reason_bucket"])
            if bucket == "success":
                continue
            key = (cid, bucket)
            chosen = samples.get(key, [])
            if not chosen:
                continue
            lines.append(f"### {_t(lang, 'sample_thoughts').format(bucket=bucket)}")
            lines.append("")
            for item in chosen:
                lines.append(
                    "- "
                    + _t(lang, "sample_header").format(
                        task_id=item["task_id"],
                        steps=item["steps"],
                        eval_type=item.get("eval_type", ""),
                        final_action_type=item.get("final_action_type", ""),
                    )
                )
                lines.append(
                    "  - "
                    + _t(lang, "sample_context").format(
                        final_url=item.get("final_url", ""),
                        reference_url=item.get("reference_url", ""),
                        early_finish=item.get("early_finish", False),
                        hit_max_steps=item.get("hit_max_steps", False),
                        task_type=item.get("task_type", ""),
                        loop_pattern=item.get("loop_pattern", ""),
                        click_back_pairs=item.get("click_back_pairs", 0),
                        max_search_query_repeat=item.get("max_search_query_repeat", 0),
                    )
                )
                lines.append(
                    "  - "
                    + _t(lang, "sample_task_intent").format(
                        task_intent=str(item.get("task_intent", "") or "").strip()
                    )
                )
                if _is_finish(str(item.get("final_action_type", "") or "")):
                    lines.append(
                        "  - "
                        + _t(lang, "sample_final_answer").format(
                            final_answer=str(item.get("final_answer_excerpt", "") or "").strip()
                        )
                    )
                lines.append(
                    "  - "
                    + _t(lang, "sample_visited_reference").format(
                        ever_visited_reference_url=item.get("ever_visited_reference_url", None)
                    )
                )
                if item.get("answer_in_intent_price_range", None) is not None:
                    lines.append(
                        "  - "
                        + _t(lang, "sample_price_range_match").format(
                            answer_in_intent_price_range=item.get("answer_in_intent_price_range", None)
                        )
                    )
                step0_thought = str(item.get("thought_at_step_0", "") or "").strip()
                if step0_thought:
                    lines.append(
                        "  - "
                        + _t(lang, "sample_step0_thought").format(thought=step0_thought)
                    )
                for ft in item.get("final_three_thoughts", []):
                    lines.append(
                        "  - "
                        + _t(lang, "step_thought").format(
                            step_idx=ft.get("step_idx", -1),
                            thought=str(ft.get("thought", "") or ""),
                        )
                    )
            lines.append("")

    # cross-condition bucket comparison
    lines.append(f"## {_t(lang, 'cross_condition')}")
    lines.append("")
    cond_ids = [str(x["condition_id"]) for x in sorted(cond_rows, key=lambda x: str(x["condition_id"]))]
    cond_total = {str(x["condition_id"]): int(x["episodes"]) for x in cond_rows}

    bucket_set = sorted({str(x["reason_bucket"]) for x in bucket_rows})
    table_headers = [_t(lang, "bucket")] + cond_ids
    table_data: List[List[str]] = []
    bucket_count_map: Dict[Tuple[str, str], int] = {}
    for r in bucket_rows:
        bucket_count_map[(str(r["condition_id"]), str(r["reason_bucket"]))] = int(r["count"])
    for bucket in bucket_set:
        row = [bucket]
        for cid in cond_ids:
            c = bucket_count_map.get((cid, bucket), 0)
            rate = _safe_ratio(c, cond_total.get(cid, 0))
            row.append(f"{c} ({_fmt_pct(rate)})")
        table_data.append(row)
    lines.append(_render_markdown_table(table_headers, table_data))
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")


def _compute_action_execution_stats(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute action execution quality metrics from raw step records."""
    click_total = click_failed = 0
    type_total = type_failed = 0
    scroll_total = 0
    parse_error_count = 0
    page_changed_count = 0
    pixel_coordinate_leak = False
    # V-F3 (B-1860 codex verify P2, 2026-05-24): aggregate counter giving the
    # dead_zone tag a read path (raw dimension in (1.1, 10] — ambiguous 0-1000
    # near-corner vs out-of-[0,1] normalized; probes show none → a nonzero
    # count is a model-regression signal worth surfacing).
    coord_dead_zone_count = 0
    consecutive_fail = 0
    max_consecutive_fail_streak = 0

    for rec in steps:
        act = rec.get("action") or {}
        atype = str(act.get("action_type", "") or "").lower()
        success = rec.get("action_success")
        err_cat = rec.get("error_category") or ""

        if atype == "click":
            click_total += 1
            if success is False:
                click_failed += 1
        elif atype == "type":
            type_total += 1
            if success is False:
                type_failed += 1
        elif atype == "scroll":
            scroll_total += 1

        if err_cat == "parse_error":
            parse_error_count += 1

        # /stress A1.10 P0-2-AB* — agent-visible progress for ax_page_change_rate.
        if _progress_changed(rec):
            page_changed_count += 1

        if success is False:
            consecutive_fail += 1
            max_consecutive_fail_streak = max(max_consecutive_fail_streak, consecutive_fail)
        else:
            consecutive_fail = 0

        coord = act.get("coordinate")
        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
            # B-1860: normalize through the Qwen 0-1000 contract FIRST. A
            # canonical 0-1000 coord (e.g. [598, 125]) is the runner's accepted
            # format — NOT a pixel leak. Only a coord still outside [0,1] AFTER
            # normalization (raw > 1000 → true_oob, or raw < 0) is a genuine
            # leak. Pre-B-1860 this hard-checked `c > 1.0` → flagged EVERY
            # 0-1000 coord as a "pixel leak" (the same mislabel as the parse
            # -error 13.6% root cause). Malformed coords are not leaks (other
            # detectors handle them).
            if isinstance(coord[0], (int, float)) and isinstance(coord[1], (int, float)):
                # V-F2 (B-1860 codex verify P1): a raw negative coord is a
                # genuine off-page leak; the normalizer tags it `malformed` so
                # the post-normalize OOB test never sees it. Flag it explicitly
                # (bool excluded — bool is an int subclass).
                _c0_neg = not isinstance(coord[0], bool) and coord[0] < 0
                _c1_neg = not isinstance(coord[1], bool) and coord[1] < 0
                _x_n, _y_n, _tags = _normalize_coordinate_pair([coord[0], coord[1]])
                if _c0_neg or _c1_neg or (not _tags["malformed"] and (
                    _x_n > 1.0 or _x_n < 0.0 or _y_n > 1.0 or _y_n < 0.0
                )):
                    pixel_coordinate_leak = True
                # V-F3 (B-1860 codex verify P2): consume the dead_zone tag.
                if not _tags["malformed"] and _tags.get("dead_zone"):
                    coord_dead_zone_count += 1

    total_steps = len(steps)
    return {
        "ax_click_total": click_total,
        "ax_click_failed": click_failed,
        "ax_click_fail_rate": round(click_failed / click_total, 4) if click_total else 0.0,
        "ax_type_total": type_total,
        "ax_type_failed": type_failed,
        "ax_type_fail_rate": round(type_failed / type_total, 4) if type_total else 0.0,
        "ax_scroll_total": scroll_total,
        "ax_parse_error_count": parse_error_count,
        "ax_parse_error_rate": round(parse_error_count / total_steps, 4) if total_steps else 0.0,
        "ax_page_change_rate": round(page_changed_count / total_steps, 4) if total_steps else 0.0,
        "ax_max_consecutive_fail_streak": max_consecutive_fail_streak,
        "ax_pixel_coordinate_leak": pixel_coordinate_leak,
        "ax_coord_dead_zone_count": coord_dead_zone_count,
    }


def _write_action_execution_summary(
    episode_rows: List[Dict[str, Any]], output_dir: Path,
) -> None:
    """Aggregate action execution stats per condition and write CSV."""
    from collections import defaultdict as _dd

    cond_data: Dict[str, List[Dict[str, Any]]] = _dd(list)
    for row in episode_rows:
        cond_data[row["condition_id"]].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for cid in sorted(cond_data):
        rows = cond_data[cid]
        cfr = [r["ax_click_fail_rate"] for r in rows if r.get("ax_click_total", 0) > 0]
        tfr = [r["ax_type_fail_rate"] for r in rows if r.get("ax_type_total", 0) > 0]
        streaks = [r["ax_max_consecutive_fail_streak"] for r in rows]
        pixel_leak_count = sum(1 for r in rows if r.get("ax_pixel_coordinate_leak"))

        def _mean(vs):
            return round(sum(vs) / len(vs), 4) if vs else None

        def _median(vs):
            if not vs:
                return None
            s = sorted(vs)
            mid = len(s) // 2
            return round((s[mid] + s[mid - 1]) / 2, 4) if len(s) % 2 == 0 else round(s[mid], 4)

        def _p75(vs):
            if not vs:
                return None
            s = sorted(vs)
            idx = int(len(s) * 0.75)
            return round(s[min(idx, len(s) - 1)], 4)

        summary_rows.append({
            "condition_id": cid,
            "n_episodes": len(rows),
            "click_fail_rate_mean": _mean(cfr),
            "click_fail_rate_median": _median(cfr),
            "type_fail_rate_mean": _mean(tfr),
            "type_fail_rate_median": _median(tfr),
            "pixel_coordinate_leak_pct": round(pixel_leak_count / len(rows), 4) if rows else 0,
            "max_consecutive_fail_streak_mean": _mean(streaks),
            "max_consecutive_fail_streak_p75": _p75(streaks),
        })

    _write_csv(
        output_dir / "action_execution_summary.csv",
        summary_rows,
        [
            "condition_id", "n_episodes",
            "click_fail_rate_mean", "click_fail_rate_median",
            "type_fail_rate_mean", "type_fail_rate_median",
            "pixel_coordinate_leak_pct",
            "max_consecutive_fail_streak_mean", "max_consecutive_fail_streak_p75",
        ],
    )
    print(f"  Action execution summary → {output_dir / 'action_execution_summary.csv'}")


def _write_state_change_by_outcome(
    episode_rows: List[Dict[str, Any]], output_dir: Path,
) -> None:
    """Cross-tab of state_change metrics by (condition, success).

    Post-§139.8 + A1.6 (2026-05-16): `adjusted_success` alias retired —
    cross-tab now uses canonical `success`.
    """
    rows_out: List[Dict[str, Any]] = []
    cond_groups: Dict[str, Dict[bool, list]] = defaultdict(lambda: defaultdict(list))
    for row in episode_rows:
        cond_groups[row["condition_id"]][bool(row.get("success", False))].append(row)

    def _proper_median(vs):
        if not vs:
            return None
        s = sorted(vs)
        mid = len(s) // 2
        return round((s[mid] + s[mid - 1]) / 2, 4) if len(s) % 2 == 0 else round(s[mid], 4)

    for cid in sorted(cond_groups):
        for outcome in [True, False]:
            subset = cond_groups[cid][outcome]
            if not subset:
                continue
            pcr = [float(r.get("ax_page_change_rate") or 0) for r in subset]
            rows_out.append({
                "condition_id": cid,
                "success": outcome,
                "n_episodes": len(subset),
                "page_change_rate_mean": round(sum(pcr) / len(pcr), 4) if pcr else None,
                "page_change_rate_median": _proper_median(pcr),
                "avg_steps": round(sum(int(r.get("steps", 0)) for r in subset) / len(subset), 1),
            })

    _write_csv(
        output_dir / "state_change_by_outcome.csv",
        rows_out,
        ["condition_id", "success", "n_episodes",
         "page_change_rate_mean", "page_change_rate_median", "avg_steps"],
    )
    print(f"  State change by outcome → {output_dir / 'state_change_by_outcome.csv'}")


def _import_plt():
    """Lazy import matplotlib with Agg backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _generate_all_plots(episode_rows: List[Dict[str, Any]], plots_dir: Path) -> None:
    """Generate all 7 diagnostic plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt = _import_plt()

    _plot_reason_distribution(plt, episode_rows, plots_dir)
    _plot_cost_by_failure_mode(plt, episode_rows, plots_dir)
    _plot_intent_feature_sr(plt, episode_rows, plots_dir)
    _plot_task_type_mode_sr(plt, episode_rows, plots_dir)
    _plot_step_efficiency(plt, episode_rows, plots_dir)
    _plot_temporal_sr(plt, episode_rows, plots_dir)
    _plot_cost_decomposition(plt, episode_rows, plots_dir)
    print(f"  Plots saved to: {plots_dir}")


def _plot_reason_distribution(plt, rows, plots_dir):
    """Plot 1: Horizontal stacked bar of reason_bucket distribution per condition."""
    conditions = sorted(set(r["condition_id"] for r in rows))
    all_buckets = Counter(r["reason_bucket"] for r in rows)
    top_buckets = [b for b, _ in all_buckets.most_common(10)]

    data = {}
    for cid in conditions:
        cond_rows = [r for r in rows if r["condition_id"] == cid]
        total = len(cond_rows)
        bucket_counts = Counter(r["reason_bucket"] for r in cond_rows)
        data[cid] = {b: bucket_counts.get(b, 0) / max(total, 1) * 100 for b in top_buckets}

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(conditions))
    left = [0.0] * len(conditions)
    for bucket in top_buckets:
        widths = [data[cid].get(bucket, 0) for cid in conditions]
        ax.barh(list(y_pos), widths, left=left, label=bucket.replace("fail_", ""), height=0.6)
        left = [l + w for l, w in zip(left, widths)]
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([c.split("_")[1] for c in conditions])
    ax.set_xlabel("Percentage (%)")
    ax.set_title("Reason Bucket Distribution by Condition")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(plots_dir / "reason_distribution.png", dpi=150)
    plt.close(fig)


def _plot_cost_by_failure_mode(plt, rows, plots_dir):
    """Plot 2: Box plot of total_cost_usd by reason_bucket (top 8)."""
    all_buckets = Counter(r["reason_bucket"] for r in rows)
    top_buckets = [b for b, _ in all_buckets.most_common(8)]
    data = []
    labels = []
    for bucket in top_buckets:
        costs = [float(r.get("total_cost_usd") or 0) for r in rows if r["reason_bucket"] == bucket]
        if costs:
            data.append(costs)
            labels.append(bucket.replace("fail_", ""))

    if not data:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, tick_labels=labels, vert=True)
    ax.set_ylabel("Total Cost (USD)")
    ax.set_title("Cost Distribution by Failure Mode")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(plots_dir / "cost_by_failure_mode.png", dpi=150)
    plt.close(fig)


def _plot_intent_feature_sr(plt, rows, plots_dir):
    """Plot 3: Heatmap of intent_feature x mode -> adjusted SR."""
    intent_keys = [k for k in sorted(rows[0].keys()) if k.startswith("intent_")]
    if not intent_keys:
        return
    modes = sorted(set(r.get("observation_mode", "") for r in rows))
    if not modes:
        return

    grid = []
    y_labels = []
    for ik in intent_keys:
        row_vals = []
        for mode in modes:
            subset = [r for r in rows if r.get("observation_mode") == mode and r.get(ik) is True]
            if len(subset) >= 3:
                sr = sum(1 for r in subset if r.get("success")) / len(subset) * 100
            else:
                sr = float("nan")
            row_vals.append(sr)
        grid.append(row_vals)
        y_labels.append(ik.replace("intent_", ""))

    import numpy as np
    arr = np.array(grid)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=0, vmax=max(30, float(np.nanmax(arr) if not np.all(np.isnan(arr)) else 30)))
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    for i in range(len(y_labels)):
        for j in range(len(modes)):
            v = arr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("SR (%) by Intent Feature x Mode")
    fig.colorbar(im, ax=ax, label="SR %")
    fig.tight_layout()
    fig.savefig(plots_dir / "intent_feature_sr.png", dpi=150)
    plt.close(fig)


def _plot_task_type_mode_sr(plt, rows, plots_dir):
    """Plot 4: Heatmap of task_type x mode -> adjusted SR."""
    task_types = sorted(set(r.get("task_type", "") for r in rows))
    modes = sorted(set(r.get("observation_mode", "") for r in rows))
    if not task_types or not modes:
        return

    grid = []
    for tt in task_types:
        row_vals = []
        for mode in modes:
            subset = [r for r in rows if r.get("observation_mode") == mode and r.get("task_type") == tt]
            if len(subset) >= 3:
                sr = sum(1 for r in subset if r.get("success")) / len(subset) * 100
            else:
                sr = float("nan")
            row_vals.append(sr)
        grid.append(row_vals)

    import numpy as np
    arr = np.array(grid)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=0, vmax=max(30, float(np.nanmax(arr) if not np.all(np.isnan(arr)) else 30)))
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes)
    ax.set_yticks(range(len(task_types)))
    ax.set_yticklabels(task_types)
    for i in range(len(task_types)):
        for j in range(len(modes)):
            v = arr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("SR (%) by Task Type x Mode")
    fig.colorbar(im, ax=ax, label="SR %")
    fig.tight_layout()
    fig.savefig(plots_dir / "task_type_mode_sr.png", dpi=150)
    plt.close(fig)


def _plot_step_efficiency(plt, rows, plots_dir):
    """Plot 5: Grouped bar chart of no_op_rate / page_unchanged_rate by mode."""
    modes = sorted(set(r.get("observation_mode", "") for r in rows))
    if not modes:
        return

    no_op_rates = []
    pu_rates = []
    for mode in modes:
        subset = [r for r in rows if r.get("observation_mode") == mode]
        if subset:
            total_cost = sum(float(r.get("total_cost_usd") or 0) for r in subset)
            no_op = sum(float(r.get("no_op_cost_usd") or 0) for r in subset)
            pu = sum(float(r.get("page_unchanged_cost_usd") or 0) for r in subset)
            no_op_rates.append(no_op / max(total_cost, 1e-9) * 100)
            pu_rates.append(pu / max(total_cost, 1e-9) * 100)
        else:
            no_op_rates.append(0)
            pu_rates.append(0)

    import numpy as np
    x = np.arange(len(modes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, no_op_rates, width, label="No-op Cost %")
    ax.bar(x + width / 2, pu_rates, width, label="Page-unchanged Cost %")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Percentage of Total Cost (%)")
    ax.set_title("Step Efficiency: Wasted Cost Ratios by Mode")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "step_efficiency.png", dpi=150)
    plt.close(fig)


def _plot_temporal_sr(plt, rows, plots_dir):
    """Plot 6: Line chart of SR by task_id quintile x condition."""
    conditions = sorted(set(r["condition_id"] for r in rows))
    n_bins = 5

    csv_rows: List[Dict[str, Any]] = []
    fig, ax = plt.subplots(figsize=(10, 6))
    for cid in conditions:
        cond_rows = sorted([r for r in rows if r["condition_id"] == cid], key=lambda r: int(r["task_id"]))
        if len(cond_rows) < n_bins:
            continue
        bin_size = len(cond_rows) // n_bins
        srs = []
        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size if i < n_bins - 1 else len(cond_rows)
            segment = cond_rows[start:end]
            sr = sum(1 for r in segment if r.get("success")) / len(segment) * 100
            srs.append(sr)
            csv_rows.append({
                "condition_id": cid, "quintile": i + 1,
                "sr_pct": round(sr, 2), "n": len(segment),
            })
        ax.plot(range(1, n_bins + 1), srs, marker="o", label=cid.split("_")[1])

    ax.set_xlabel("Task ID Quintile (1=earliest, 5=latest)")
    ax.set_ylabel("SR (%)")
    ax.set_title("Temporal Success Rate Trend")
    ax.legend()
    ax.set_xticks(range(1, n_bins + 1))
    fig.tight_layout()
    fig.savefig(plots_dir / "temporal_sr.png", dpi=150)
    plt.close(fig)

    # Write CSV companion for the plot
    if csv_rows:
        csv_dir = plots_dir.parent  # reason_diagnostics/
        _write_csv(
            csv_dir / "temporal_sr.csv", csv_rows,
            ["condition_id", "quintile", "sr_pct", "n"],
        )
        print(f"  Temporal SR CSV → {csv_dir / 'temporal_sr.csv'}")


def _plot_cost_decomposition(plt, rows, plots_dir):
    """Plot 7: Stacked bar of effective / no_op / loop cost by condition."""
    conditions = sorted(set(r["condition_id"] for r in rows))

    effective = []
    no_op = []
    loop = []
    labels = []
    for cid in conditions:
        subset = [r for r in rows if r["condition_id"] == cid]
        eff = sum(float(r.get("effective_cost_usd") or 0) for r in subset)
        nop = sum(float(r.get("no_op_cost_usd") or 0) for r in subset)
        lp = sum(float(r.get("loop_cost_usd") or 0) for r in subset)
        effective.append(eff)
        no_op.append(nop)
        loop.append(lp)
        labels.append(cid.split("_")[1])

    import numpy as np
    x = np.arange(len(conditions))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, effective, label="Effective", color="#2ecc71")
    ax.bar(x, no_op, bottom=effective, label="No-op", color="#e74c3c")
    eff_nop = [e + n for e, n in zip(effective, no_op)]
    ax.bar(x, loop, bottom=eff_nop, label="Loop", color="#f39c12")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Total Cost (USD)")
    ax.set_title("Cost Decomposition by Condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "cost_decomposition.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage-level success/failure diagnostics from *_summary_v2.json + *_steps_v2.jsonl"
    )
    parser.add_argument("--run-dir", required=True, help="Run directory, e.g. results/.../B1_xxx")
    parser.add_argument("--task-min", type=int, default=None, help="Optional task_id lower bound (inclusive)")
    parser.add_argument("--task-max", type=int, default=None, help="Optional task_id upper bound (inclusive)")
    parser.add_argument("--condition", default=None, help="Optional condition_id exact match")
    parser.add_argument("--early-finish-steps", type=int, default=2, help="finish at <=N steps is early finish")
    parser.add_argument("--output-dir", default=None, help="Optional output dir (default: <run_dir>/analysis/reason_diagnostics)")
    parser.add_argument("--report", action="store_true", help="Generate failure_report.md")
    parser.add_argument(
        "--report-language",
        choices=["zh", "en", "bilingual"],
        default="zh",
        help="Language for failure_report.md",
    )
    parser.add_argument("--samples-per-bucket", type=int, default=5, help="Number of episode samples per reason bucket")
    parser.add_argument(
        "--skip-similarity",
        action="store_true",
        help="Skip O(N^2) all-pairs thought similarity calculations for faster runs",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating visualization plots",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run-dir not found: {run_dir}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else run_dir / "analysis" / "reason_diagnostics"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    max_steps = 30
    run_meta = run_dir / "run_meta.json"
    if run_meta.exists():
        try:
            meta = _read_json(run_meta)
            max_steps = int(meta.get("config", {}).get("runtime", {}).get("max_steps", max_steps))
        except Exception:
            pass

    episode_rows: List[Dict[str, Any]] = []
    trajectory_rows: List[Dict[str, Any]] = []
    task_cfg_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}

    # for phase2 aggregation
    bucket_pair_values: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    bucket_template_counter: Dict[Tuple[str, str], Counter] = defaultdict(Counter)

    for cond_dir in sorted([p for p in run_dir.iterdir() if p.is_dir()]):
        condition_id = cond_dir.name
        if args.condition and condition_id != args.condition:
            continue
        episodes_dir = cond_dir / "episodes"
        if not episodes_dir.exists():
            continue

        for summary_path in sorted(episodes_dir.glob("*_summary_v2.json")):
            m = SUMMARY_RE.match(summary_path.name)
            if not m:
                continue
            site = m.group("site")
            task_id = int(m.group("task_id"))
            if args.task_min is not None and task_id < args.task_min:
                continue
            if args.task_max is not None and task_id > args.task_max:
                continue

            steps_path = summary_path.with_name(summary_path.name.replace("_summary_v2.json", "_steps_v2.jsonl"))
            if not steps_path.exists():
                continue

            # B-549 (/stress A1.5 P0-2-AB* Claude+codex OOB sibling
            # propagation, 2026-05-17): switch summary read from plain
            # `_read_json` → `load_episode_summary_strict(reject_needs_
            # reevaluation=True)`. Pre-fix B-486 quarantined episodes
            # (crash-before-evaluator) would enter the reason-diagnostics
            # stage-level distribution as `success=False` with empty/missing
            # final_step → mis-categorized as "agent failure" rather than
            # "infra failure". paper §3 reason-diagnostics taxonomy table
            # was therefore polluted by infrastructure failures attributed
            # to model behavior. Lenient mode + reject_quarantine → loader
            # returns None for both quarantined rows AND type-mismatch
            # rows; skip both. Other `_read_json` call sites (L225 task
            # cfg, L1873 run_meta) read non-episode-summary files and
            # retain plain loader.
            from p79.experiment.io_utils import load_episode_summary_strict
            summary = load_episode_summary_strict(
                summary_path, mode="lenient", reject_needs_reevaluation=True,
            )
            if summary is None:
                continue  # corrupt / type-mismatch / B-486 quarantine
            # D-other fix 2026-05-24: pass summary_path + strict_identity=True so
            # read_jsonl_dedup (aliased as _read_jsonl at module-level via
            # `from p79.experiment.io_utils import read_jsonl_dedup as _read_jsonl`)
            # can validate segment identity against the episode summary. This catches
            # restart-crash bleed-through (stale prior-run steps leaking into the current
            # episode's JSONL) before the steps enter reason-diagnostics categorisation.
            # summary_path is already resolved above from the glob match on summary_path.
            steps = _sort_steps_by_idx(_read_jsonl(steps_path, summary_path=summary_path, strict_identity=True))
            if not steps:
                continue

            final_step = steps[-1]
            final_action = final_step.get("action") or {}
            final_action_type = str(final_action.get("action_type", "") or "").lower()
            final_answer = str(final_action.get("answer", "") or "")
            final_thought = str(final_action.get("thought", "") or "")
            final_error_category = final_step.get("error_category")
            parse_valid = final_step.get("parse_valid")
            parse_failure_reason = final_step.get("parse_failure_reason")
            fallback_finish = bool(final_step.get("fallback_finish", False))

            task_meta = _extract_task_meta(run_dir, site, task_id, task_cfg_cache)
            require_reset = bool(task_meta.get("require_reset", False))
            eval_cfg = (task_meta.get("eval") or {})
            task_intent = str(task_meta.get("intent", "") or "").strip()
            task_type = _classify_task_type(task_intent)
            eval_types = eval_cfg.get("eval_types") or []
            eval_type = "|".join(str(x) for x in eval_types)
            ref_url = eval_cfg.get("reference_url")
            ref_answers = eval_cfg.get("reference_answers")
            ref_urls = _normalize_url_candidates(ref_url)
            final_url = str(((final_step.get("state_digest") or {}).get("url_after")) or "").strip() or _step_obs_url(final_step)
            final_url_match: Optional[bool]
            if not ref_urls:
                final_url_match = None
            else:
                final_url_match = final_url in ref_urls
            ever_visited_ref_url = _ever_visited_reference_url(steps, ref_urls)
            target_item_ever_visible = _target_item_ever_visible(steps, ref_urls, run_dir)
            answer_in_intent_price_range = _answer_in_intent_price_range(final_answer, task_intent)

            success = bool(summary.get("success", False))
            steps_count = len(steps)
            # /stress A1.10 P0-2-AB* — agent-visible progress for per-task page_change_count.
            page_change_count = sum(1 for s in steps if _progress_changed(s))
            search_attempts = sum(
                1
                for s in steps
                if str((s.get("action") or {}).get("action_type", "")).lower() == "type"
                and bool(str((s.get("action") or {}).get("text", "")).strip())
            )
            early_finish = _is_finish(final_action_type) and steps_count <= int(args.early_finish_steps)
            hit_max_steps = steps_count >= max_steps

            max_repeat_streak = 0
            last_sig: Optional[str] = None
            streak = 0
            for s in steps:
                act = s.get("action") or {}
                sig = "|".join(
                    [
                        str(act.get("action_type", "")).lower(),
                        str(act.get("element_id", "")),
                        str(act.get("text", ""))[:80],
                        str(act.get("coordinate", "")),
                        str(act.get("delta", "")),
                    ]
                )
                if sig == last_sig:
                    streak += 1
                else:
                    streak = 1
                    last_sig = sig
                if streak > max_repeat_streak:
                    max_repeat_streak = streak

            no_result_language = _contains_any(final_answer, NO_RESULT_PATTERNS)
            uncertain_language = _contains_any(final_answer, UNCERTAIN_PATTERNS)
            final_thought_sig = _first_words(final_thought, max_words=12)

            thought_at_0, thought_at_5, thought_at_10, thought_final, trajectory, non_empty_norm = _thought_snapshots(steps)
            thought_metrics = _thought_similarity_features(
                non_empty_norm,
                skip_pairwise=bool(args.skip_similarity),
            )

            stuck_first_step, unchanged_streak_max_len, unchanged_streak_max_pos = _page_unchanged_signals(steps)
            scroll_stats = _scroll_direction_stats(steps)
            action_type_sequence = _compress_action_types(steps)
            final_three_thoughts = _collect_final_thoughts(steps, k=3)
            select_events = _collect_select_events(steps)
            loop_metrics = _detect_loops(steps)
            ax_stats = _compute_action_execution_stats(steps)
            page_type_seq = _page_type_sequence(steps)
            observation_mode = str(steps[0].get("observation_mode", "") or summary.get("observation_mode", "") or "").strip()
            # /stress A1.4 P0-2 (2026-05-17): canonical Path A signal is
            # `mark_count == 0` (zero-marks vision-fallback). Pre-fix the
            # `degraded_som` bool field was deleted from schema because it
            # overloaded three semantically distinct states (zero-marks /
            # render-fail / phantom inherit). Aggregator now counts on
            # SoM-mode steps where the extractor returned no marks; PIL
            # render-fail (Path B) is no longer schema-visible (logged via
            # logger.warning only — empirical 0/6471 archive fires).
            degraded_som_steps = sum(
                1
                for s in steps
                if str(s.get("observation_mode", "") or "").strip() == "som"
                and int((s.get("som") or {}).get("mark_count", 0) or 0) == 0
            )

            # ── URL revisit & action diversity signals ──
            url_revisit_count, url_unique_count, url_revisit_max = _url_revisit_metrics(steps)
            action_diversity, action_unique_types = _action_diversity_metrics(steps)

            reason_bucket = _classify_reason(
                success=success,
                summary_error=summary.get("error"),
                final_action_type=final_action_type,
                final_error_category=final_error_category,
                final_answer=final_answer,
                eval_type=eval_type,
                early_finish=early_finish,
                hit_max_steps=hit_max_steps,
                click_back_pairs=int(loop_metrics["click_back_pairs"]),
                max_search_query_repeat=int(loop_metrics["max_search_query_repeat"]),
                target_item_ever_visible=target_item_ever_visible,
                final_url_match=final_url_match,
                ever_visited_reference_url=ever_visited_ref_url,
                final_answer_in_intent_price_range=answer_in_intent_price_range,
            )

            # §139.8 + /stress A1.6 (2026-05-16) hard-delete: post-hoc
            # `adjusted_success` / `fp_reason` / `adjusted_reason_bucket`
            # alias trio removed; `success` + `reason_bucket` are canonical.

            collection_overlap_score = _collection_overlap_score(final_answer, ref_answers)
            stuck_subtype = _classify_stuck_subtype(
                reason_bucket=reason_bucket,
                task_type=task_type,
                target_item_ever_visible=target_item_ever_visible,
                loop_pattern=str(loop_metrics["loop_pattern"]),
                max_search_query_repeat=int(loop_metrics["max_search_query_repeat"]),
                page_type_sequence=page_type_seq,
                click_back_pairs=int(loop_metrics["click_back_pairs"]),
                action_type_sequence=action_type_sequence,
            )
            unreachable_subtype = _classify_unreachable_subtype(
                reason_bucket=reason_bucket,
                task_intent=task_intent,
                observation_mode=observation_mode,
                search_queries=loop_metrics["search_queries"],
                degraded_som_steps=degraded_som_steps,
                total_steps=steps_count,
                has_image=(task_meta.get("image") is not None),
            )

            # Phase2 aggregations
            bucket_key = (condition_id, reason_bucket)
            bucket_pair_values[bucket_key].extend(thought_metrics["all_pair_values"])
            bucket_template_counter[bucket_key].update(thought_metrics["high_template_counter"])

            # Cost breakdown
            _summary_cost = float(summary.get("total_cost_usd") or 0)
            _summary_tokens = int(summary.get("total_tokens") or 0)
            _waste = _compute_step_cost_breakdown(steps, loop_metrics)
            # Intent features
            _intent_features = _extract_intent_features(task_intent, task_meta)

            # /stress A1.10 P1-7-B (2026-05-16): per-task router metric rollups
            # from step records. Pre-fix runner condition_summary_v2.json had
            # escalation_count + trigger_distribution but diagnostics CSV
            # (episode_reason_rows.csv) dropped them — paper §3.5/§6 router
            # rollups required join-back to raw summary. Post-fix emits both
            # per-task fields so any failure-bucket / task-type slicing of
            # router behavior is direct from the CSV. Trigger distribution is
            # JSON-serialised because CSV is flat.
            _router_decisions = []
            _trigger_counter = Counter()
            _rule_router_skipped_count = 0
            for _s in steps:
                _r = _s.get("router") or {}
                _decision = str(_r.get("decision", "") or "")
                if _decision:
                    _router_decisions.append(_decision)
                # trigger_reason may be a list of strings, a single string, or absent
                _tr = _r.get("trigger_reason")
                if isinstance(_tr, list):
                    for _t in _tr:
                        if _t:
                            _trigger_counter[str(_t)] += 1
                elif _tr:
                    _trigger_counter[str(_tr)] += 1
                # rule_router_skipped is emitted as a flag in overhead dict by
                # P0-4-B* learned-router skip path (runner main.py).
                _overhead = _r.get("overhead") or {}
                if _overhead.get("rule_router_skipped"):
                    _rule_router_skipped_count += 1
            _task_escalation_count = sum(
                1 for _d in _router_decisions if _d and _d != observation_mode
            )

            episode_row: Dict[str, Any] = {
                "condition_id": condition_id,
                "site": site,
                "task_id": task_id,
                "success": success,
                "reason_bucket": reason_bucket,
                "eval_type": eval_type,
                "steps": steps_count,
                "hit_max_steps": hit_max_steps,
                "final_action_type": final_action_type,
                "early_finish": early_finish,
                "finish_answer_empty": _is_finish(final_action_type) and (final_answer.strip() == ""),
                "no_result_language": no_result_language,
                "uncertain_language": uncertain_language,
                "final_error_category": final_error_category,
                "summary_error": summary.get("error"),
                "parse_valid": parse_valid,
                "parse_failure_reason": parse_failure_reason,
                "fallback_finish": fallback_finish,
                "final_url": final_url,
                "reference_url": ref_url,
                "target_item_ever_visible": target_item_ever_visible,
                "final_url_match": final_url_match,
                "ever_visited_reference_url": ever_visited_ref_url,
                "page_change_count": page_change_count,
                "search_attempts": search_attempts,
                "page_unchanged_rate": summary.get("page_unchanged_rate"),
                "max_repeat_streak": max_repeat_streak,
                # /stress A1.10 P1-7-B router metric rollups
                "escalation_count": _task_escalation_count,
                "trigger_distribution_json": json.dumps(dict(_trigger_counter), ensure_ascii=False),
                "rule_router_skipped_steps": _rule_router_skipped_count,
                "task_intent": task_intent,
                "task_type": task_type,
                "observation_mode": observation_mode,
                "degraded_som_steps": degraded_som_steps,
                "scroll_up": scroll_stats["scroll_up"],
                "scroll_down": scroll_stats["scroll_down"],
                "scroll_direction_flips": scroll_stats["scroll_direction_flips"],
                "scroll_wasted_steps": scroll_stats["scroll_wasted_steps"],
                "url_revisit_count": url_revisit_count,
                "url_unique_count": url_unique_count,
                "url_revisit_max": url_revisit_max,
                "action_diversity": round(action_diversity, 4),
                "action_unique_types": action_unique_types,
                "require_reset": require_reset,
                "answer_in_intent_price_range": answer_in_intent_price_range,
                "reference_answers_json": json.dumps(ref_answers, ensure_ascii=False) if ref_answers is not None else "",
                "final_answer": final_answer,
                "final_answer_excerpt": final_answer[:200],
                "final_thought_signature": final_thought_sig,
                "thought_at_step_0": thought_at_0,
                "thought_at_step_5": thought_at_5,
                "thought_at_step_10": thought_at_10,
                "thought_final": thought_final,
                "stuck_first_step": stuck_first_step,
                "page_unchanged_streak_max_len": unchanged_streak_max_len,
                "page_unchanged_streak_max_pos": unchanged_streak_max_pos,
                "thought_count": thought_metrics["thought_count"],
                "thought_unique_count": thought_metrics["thought_unique_count"],
                "thought_diversity": thought_metrics["thought_diversity"],
                "action_type_sequence": action_type_sequence,
                "thought_similarity_max_adjacent": thought_metrics["thought_similarity_max_adjacent"],
                "thought_similarity_repeat_rate": thought_metrics["thought_similarity_repeat_rate"],
                "thought_similarity_mean_all_pairs": thought_metrics["thought_similarity_mean_all_pairs"],
                "thought_similarity_p90_all_pairs": thought_metrics["thought_similarity_p90_all_pairs"],
                "thought_similarity_high_rate_all_pairs": thought_metrics["thought_similarity_high_rate_all_pairs"],
                "high_similarity_templates": "|".join(sorted(thought_metrics["high_template_counter"].keys())),
                "final_three_thoughts": final_three_thoughts,
                "final_three_thoughts_json": json.dumps(final_three_thoughts, ensure_ascii=False),
                "select_events_json": json.dumps(select_events, ensure_ascii=False),
                "search_queries_json": json.dumps(loop_metrics["search_queries"], ensure_ascii=False),
                "click_back_pairs": int(loop_metrics["click_back_pairs"]),
                "max_search_query_repeat": int(loop_metrics["max_search_query_repeat"]),
                "most_repeated_search_query": str(loop_metrics["most_repeated_search_query"]),
                "unique_search_queries": int(loop_metrics["unique_search_queries"]),
                "loop_pattern": str(loop_metrics["loop_pattern"]),
                "page_type_sequence": page_type_seq,
                "collection_overlap_score": collection_overlap_score,
                "stuck_subtype": stuck_subtype,
                "unreachable_subtype": unreachable_subtype,
                "all_step_thoughts_json": json.dumps(
                    [
                        {"step_idx": int(t.get("step_idx", -1)), "thought": str(t.get("thought", "") or "")}
                        for t in trajectory
                        if str(t.get("thought", "") or "").strip()
                    ],
                    ensure_ascii=False,
                ),
                # --- Cost columns ---
                "total_cost_usd": _summary_cost,
                "total_tokens": _summary_tokens,
                "no_op_cost_usd": round(_waste["no_op_cost_usd"], 6),
                "page_unchanged_cost_usd": round(_waste["page_unchanged_cost_usd"], 6),
                "loop_cost_usd": round(_waste["loop_cost_usd"], 6),
                "effective_cost_usd": round(_waste["effective_cost_usd"], 6),
                # --- Intent features ---
                **_intent_features,
                # --- Action execution ---
                **ax_stats,
            }
            episode_rows.append(episode_row)

            trajectory_rows.append(
                {
                    "condition_id": condition_id,
                    "site": site,
                    "task_id": task_id,
                    "success": success,
                    "reason_bucket": reason_bucket,
                    "steps": steps_count,
                    "thought_count": thought_metrics["thought_count"],
                    "thought_unique_count": thought_metrics["thought_unique_count"],
                    "thought_diversity": thought_metrics["thought_diversity"],
                    "trajectory": trajectory,
                }
            )

    if not episode_rows:
        print("No episodes matched filters; nothing to write.")
        return

    episode_rows_sorted = sorted(episode_rows, key=lambda x: (x["condition_id"], int(x["task_id"])))

    # episode-level table
    episode_fields = [
        "condition_id",
        "site",
        "task_id",
        "success",
        "reason_bucket",
        "eval_type",
        "steps",
        "hit_max_steps",
        "final_action_type",
        "early_finish",
        "finish_answer_empty",
        "no_result_language",
        "uncertain_language",
        "final_error_category",
        "summary_error",
        "parse_valid",
        "parse_failure_reason",
        "fallback_finish",
        "final_url",
        "reference_url",
        "target_item_ever_visible",
        "final_url_match",
        "ever_visited_reference_url",
        "page_change_count",
        "search_attempts",
        "page_unchanged_rate",
        "max_repeat_streak",
        "task_intent",
        "task_type",
        "observation_mode",
        "degraded_som_steps",
        "scroll_up",
        "scroll_down",
        "scroll_direction_flips",
        "scroll_wasted_steps",
        "url_revisit_count",
        "url_unique_count",
        "url_revisit_max",
        "action_diversity",
        "action_unique_types",
        "require_reset",
        "answer_in_intent_price_range",
        "reference_answers_json",
        "final_answer",
        "stuck_first_step",
        "page_unchanged_streak_max_len",
        "page_unchanged_streak_max_pos",
        "action_type_sequence",
        "page_type_sequence",
        "click_back_pairs",
        "max_search_query_repeat",
        "most_repeated_search_query",
        "unique_search_queries",
        "loop_pattern",
        "collection_overlap_score",
        "stuck_subtype",
        "unreachable_subtype",
        "thought_count",
        "thought_unique_count",
        "thought_diversity",
        "thought_similarity_max_adjacent",
        "thought_similarity_repeat_rate",
        "thought_similarity_mean_all_pairs",
        "thought_similarity_p90_all_pairs",
        "thought_similarity_high_rate_all_pairs",
        "thought_at_step_0",
        "thought_at_step_5",
        "thought_at_step_10",
        "thought_final",
        "final_three_thoughts_json",
        "select_events_json",
        "search_queries_json",
        "all_step_thoughts_json",
        "final_answer_excerpt",
        "final_thought_signature",
        "high_similarity_templates",
        # Cost columns
        "total_cost_usd",
        "total_tokens",
        "no_op_cost_usd",
        "page_unchanged_cost_usd",
        "loop_cost_usd",
        "effective_cost_usd",
        # Intent features
        "intent_has_sort",
        "intent_has_filter",
        "intent_has_compare",
        "intent_has_count",
        "intent_has_latest",
        "intent_has_color",
        "intent_has_image",
        "intent_has_price",
        "intent_has_location",
        "intent_needs_scroll",
        # Action execution
        "ax_click_total",
        "ax_click_failed",
        "ax_click_fail_rate",
        "ax_type_total",
        "ax_type_failed",
        "ax_type_fail_rate",
        "ax_scroll_total",
        "ax_parse_error_count",
        "ax_parse_error_rate",
        "ax_page_change_rate",
        "ax_max_consecutive_fail_streak",
        "ax_pixel_coordinate_leak",
    ]
    _write_csv(output_dir / "episode_reason_rows.csv", episode_rows_sorted, episode_fields)

    # thought trajectories
    _write_jsonl(output_dir / "thought_trajectories.jsonl", sorted(trajectory_rows, key=lambda x: (x["condition_id"], int(x["task_id"]))))

    # condition summary (counts by bucket)
    per_condition_total: Counter = Counter()
    per_condition_success: Counter = Counter()
    per_condition_bucket: Dict[str, Counter] = defaultdict(Counter)

    for row in episode_rows:
        cid = str(row["condition_id"])
        per_condition_total[cid] += 1
        if bool(row["success"]):
            per_condition_success[cid] += 1
        per_condition_bucket[cid][str(row["reason_bucket"])] += 1

    cond_rows: List[Dict[str, Any]] = []
    for cid in sorted(per_condition_total.keys()):
        total = per_condition_total[cid]
        cond_rows.append(
            {
                "condition_id": cid,
                "episodes": total,
                "success_count": per_condition_success[cid],
                "success_rate": _safe_ratio(per_condition_success[cid], total),
                "early_finish_fail_count": sum(
                    1
                    for x in episode_rows
                    if x["condition_id"] == cid and (not x["success"]) and bool(x["early_finish"])
                ),
                "fallback_finish_count": sum(
                    1 for x in episode_rows if x["condition_id"] == cid and bool(x["fallback_finish"])
                ),
            }
        )
    _write_csv(
        output_dir / "condition_overview.csv",
        cond_rows,
        [
            "condition_id",
            "episodes",
            "success_count",
            "success_rate",
            "early_finish_fail_count",
            "fallback_finish_count",
        ],
    )

    bucket_rows: List[Dict[str, Any]] = []
    for cid in sorted(per_condition_bucket.keys()):
        total = per_condition_total[cid]
        for bucket, count in per_condition_bucket[cid].most_common():
            bucket_rows.append(
                {
                    "condition_id": cid,
                    "reason_bucket": bucket,
                    "count": count,
                    "rate_in_condition": _safe_ratio(count, total),
                }
            )
    _write_csv(
        output_dir / "condition_reason_summary.csv",
        bucket_rows,
        ["condition_id", "reason_bucket", "count", "rate_in_condition"],
    )

    # thought signatures (quick thinking-process summary)
    thought_rows: List[Dict[str, Any]] = []
    for cid in sorted(per_condition_total.keys()):
        c = Counter(
            x["final_thought_signature"]
            for x in episode_rows
            if x["condition_id"] == cid and x["final_thought_signature"]
        )
        for sig, count in c.most_common(50):
            thought_rows.append(
                {
                    "condition_id": cid,
                    "final_thought_signature": sig,
                    "count": count,
                }
            )
    _write_csv(
        output_dir / "final_thought_signature_summary.csv",
        thought_rows,
        ["condition_id", "final_thought_signature", "count"],
    )

    # Phase2: bucket-level thought similarity summary (all-pairs)
    bucket_similarity_rows: List[Dict[str, Any]] = []
    for (cid, bucket), values in sorted(bucket_pair_values.items(), key=lambda x: (x[0][0], x[0][1])):
        bucket_similarity_rows.append(
            {
                "condition_id": cid,
                "reason_bucket": bucket,
                "pair_count": len(values),
                "pair_similarity_mean": (sum(values) / len(values)) if values else 0.0,
                "pair_similarity_p90": _p90(values),
                "pair_similarity_high_rate": _safe_ratio(
                    sum(1 for v in values if v >= SIM_REPEAT_THRESHOLD), len(values)
                ),
            }
        )
    _write_csv(
        output_dir / "bucket_thought_similarity_summary.csv",
        bucket_similarity_rows,
        [
            "condition_id",
            "reason_bucket",
            "pair_count",
            "pair_similarity_mean",
            "pair_similarity_p90",
            "pair_similarity_high_rate",
        ],
    )

    bucket_template_rows: List[Dict[str, Any]] = []
    for (cid, bucket), ctr in sorted(bucket_template_counter.items(), key=lambda x: (x[0][0], x[0][1])):
        for sig, count in ctr.most_common(20):
            bucket_template_rows.append(
                {
                    "condition_id": cid,
                    "reason_bucket": bucket,
                    "template_signature": sig,
                    "count": count,
                }
            )
    _write_csv(
        output_dir / "bucket_high_similarity_templates.csv",
        bucket_template_rows,
        ["condition_id", "reason_bucket", "template_signature", "count"],
    )

    # Bucket thought samples (human-friendly plaintext)
    _write_bucket_thought_samples(
        output_dir / "bucket_thought_samples.txt",
        episode_rows_sorted,
        int(args.samples_per_bucket),
    )

    # machine-readable snapshot
    snapshot = {
        "run_dir": str(run_dir),
        "filters": {
            "condition": args.condition,
            "task_min": args.task_min,
            "task_max": args.task_max,
            "early_finish_steps": args.early_finish_steps,
            "report": bool(args.report),
            "report_language": args.report_language,
            "samples_per_bucket": int(args.samples_per_bucket),
            "skip_similarity": bool(args.skip_similarity),
        },
        "episodes": len(episode_rows),
        "reason_buckets_global": dict(Counter(str(x["reason_bucket"]) for x in episode_rows)),
        "outputs": {
            "episode_reason_rows_csv": "episode_reason_rows.csv",
            "condition_overview_csv": "condition_overview.csv",
            "condition_reason_summary_csv": "condition_reason_summary.csv",
            "final_thought_signature_summary_csv": "final_thought_signature_summary.csv",
            "thought_trajectories_jsonl": "thought_trajectories.jsonl",
            "bucket_thought_samples_txt": "bucket_thought_samples.txt",
            "bucket_thought_similarity_summary_csv": "bucket_thought_similarity_summary.csv",
            "bucket_high_similarity_templates_csv": "bucket_high_similarity_templates.csv",
            "failure_report_md": "failure_report.md" if args.report else None,
        },
    }
    with open(output_dir / "reason_diagnostics_summary.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    # Supplementary summary CSVs
    try:
        _write_action_execution_summary(episode_rows_sorted, output_dir)
        _write_state_change_by_outcome(episode_rows_sorted, output_dir)
    except Exception as e:
        print(f"Warning: supplementary summary write failed: {e}")

    # Visualization plots
    if not args.no_plots:
        try:
            _generate_all_plots(episode_rows_sorted, output_dir / "plots")
        except Exception as e:
            print(f"Warning: plot generation failed: {e}")

    # Markdown report
    if args.report:
        _build_report(
            output_path=output_dir / "failure_report.md",
            lang=args.report_language,
            run_dir=run_dir,
            filters=snapshot["filters"],
            episode_rows=episode_rows_sorted,
            cond_rows=cond_rows,
            bucket_rows=bucket_rows,
            bucket_similarity_rows=bucket_similarity_rows,
            bucket_template_rows=bucket_template_rows,
            samples_per_bucket=int(args.samples_per_bucket),
            skip_similarity=bool(args.skip_similarity),
        )

    print(f"Saved reason diagnostics to: {output_dir}")
    print(f"Episodes analyzed: {len(episode_rows)}")
    print("Global reason buckets:")
    for bucket, count in Counter(str(x["reason_bucket"]) for x in episode_rows).most_common():
        print(f"  - {bucket}: {count}")
    if args.report:
        print(f"Markdown report: {output_dir / 'failure_report.md'}")


if __name__ == "__main__":
    main()
