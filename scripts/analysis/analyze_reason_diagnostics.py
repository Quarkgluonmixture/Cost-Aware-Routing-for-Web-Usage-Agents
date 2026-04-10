#!/usr/bin/env python3
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
        if any(iid in text for iid in item_ids):
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


def _page_unchanged_signals(steps: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    stuck_first_step = -1
    max_len = 0
    max_pos = -1
    cur_len = 0
    cur_pos = -1

    for s in steps:
        step_idx = int(s.get("step_idx", -1))
        changed = bool(s.get("page_changed", False))
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
        digest = s.get("state_digest") or {}
        url = str(s.get("obs_url", "") or digest.get("url_after", "") or "").strip()
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
    # account/login loop: page type sequence has repeated auth pages (checked first — strongest signal)
    if "login" in page_type_sequence.lower() or "account" in page_type_sequence.lower():
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
    # to SoM/hybrid mode when SoM was degraded (zero marks → empty [SOM_MARKS] +
    # raw screenshot; no bounding-box marks means the model cannot locate items
    # visually, making visual-attribute tasks structurally unreachable).
    # Vision mode is excluded: the model always receives the raw screenshot and
    # visual failures there are model capability issues, not scaffold defects.
    has_visual = any(k in intent_lower for k in _VISUAL_MATCH_KWDS)
    is_dom_like = obs == "dom" or (obs in ("som", "hybrid") and degraded_som_steps > 0)
    if has_visual and is_dom_like:
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

            summary = _read_json(summary_path)
            steps = _sort_steps_by_idx(_read_jsonl(steps_path))
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
            page_change_count = sum(1 for s in steps if bool(s.get("page_changed", False)))
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
            action_type_sequence = _compress_action_types(steps)
            final_three_thoughts = _collect_final_thoughts(steps, k=3)
            select_events = _collect_select_events(steps)
            loop_metrics = _detect_loops(steps)
            page_type_seq = _page_type_sequence(steps)
            observation_mode = str(steps[0].get("observation_mode", "") or summary.get("observation_mode", "") or "").strip()
            degraded_som_steps = sum(
                1 for s in steps if bool((s.get("som") or {}).get("degraded_som", False))
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
            )

            # Phase2 aggregations
            bucket_key = (condition_id, reason_bucket)
            bucket_pair_values[bucket_key].extend(thought_metrics["all_pair_values"])
            bucket_template_counter[bucket_key].update(thought_metrics["high_template_counter"])

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
                "task_intent": task_intent,
                "task_type": task_type,
                "observation_mode": observation_mode,
                "degraded_som_steps": degraded_som_steps,
                "url_revisit_count": url_revisit_count,
                "url_unique_count": url_unique_count,
                "url_revisit_max": url_revisit_max,
                "action_diversity": round(action_diversity, 4),
                "action_unique_types": action_unique_types,
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
        "url_revisit_count",
        "url_unique_count",
        "url_revisit_max",
        "action_diversity",
        "action_unique_types",
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
