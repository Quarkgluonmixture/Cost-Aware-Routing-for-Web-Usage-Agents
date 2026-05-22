#!/usr/bin/env python3
"""Per-run diagnostic; not part of the 4-dimension evidence framework.

Batch pattern-matching diagnostics for P79 episodes.

Automates diag SKILL Step 2 (hard rules P1-P14) across all episodes in a run.
Outputs structured JSON with per-episode hits and aggregate summary.

Usage:
  python scripts/analysis/diag_pattern_match.py \
      --run-dir results/visualwebarena/phase1/B1_3mode_classifieds_20260413

  # Single task
  python scripts/analysis/diag_pattern_match.py \
      --run-dir results/visualwebarena/phase1/B1_3mode_classifieds_20260413 \
      --task-id 75

  # Failed only, specific rules
  python scripts/analysis/diag_pattern_match.py \
      --run-dir results/visualwebarena/phase1/B1_3mode_classifieds_20260413 \
      --failed-only --rules P1,P3,P5,P14
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Ruleset version — discover-then-freeze protocol (see diag SKILL.md
# "跨 condition / cross-mode 工作协议")
# ---------------------------------------------------------------------------
# Bump on ANY change to ALL_RULES: a new check_pN OR a regex/threshold edit
# inside an existing rule. `rules_applied` (in the output JSON) only records the
# rule-NAME set, so a P16 regex fix is invisible there — this version string
# makes such content changes explicit. After bumping, re-scan ALL existing
# conditions (diag_autorun.sh) so every per-condition digest carries the SAME
# ruleset_version BEFORE any cross-mode comparison.
#
# Current basis: P1-P18 discovered from B0 dom classifieds (R9755). The mode
# gates (`if mode != "dom"` in check_p6 / p15 / p16) are themselves provisional
# discover-products — NOT yet validated against som / vision / phantom modes.
RULESET_VERSION = "1-dom"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PatternHit:
    rule_id: str
    rule_name: str
    step_idx: Optional[int]  # None for episode-level rules
    detail: str
    is_scaffold: bool

@dataclass
class EpisodeDiagnosis:
    task_id: int
    condition_id: str
    site: str
    success: bool
    steps: int
    hits: List[PatternHit] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIEWPORT_W, VIEWPORT_H = 1280, 720

US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming",
}

SCROLL_DOWN_PHRASES = re.compile(
    r"scroll\s*down|further\s*down|see\s*more|below|next\s*page",
    re.IGNORECASE,
)

# Color keywords for visual task detection (P6)
VISUAL_COLOR_KEYWORDS = re.compile(
    r"\b(white|blue|green|orange|black|red|purple|yellow|pink|grey|gray|brown)\b",
    re.IGNORECASE,
)

# P6 extension (self-evolving 2026-05-22, diagnose Tier-2 task 21): color ADJECTIVES.
# The concrete-color list above misses "dark color" / "light colored" etc.
VISUAL_COLOR_ADJ = re.compile(
    r"\b(dark|light|pale|bright|deep)\s+colou?r(ed)?\b",
    re.IGNORECASE,
)

# P15: gallery row-position intent. DOM linearizes the visual grid → cannot know
# which items physically sit in row N (row width depends on viewport, absent from DOM).
GALLERY_ROW_RE = re.compile(
    r"\b(second|third|fourth|fifth|sixth|last|first|next|\d+(?:st|nd|rd|th))\s+(?:two\s+|three\s+)?rows?\b",
    re.IGNORECASE,
)

# P16: image-content tasks (book cover / item-image content filter). DOM has no pixels.
VISUAL_IMAGE_CONTENT_RE = re.compile(
    r"\b(on (?:the|its) cover|on the front|in (?:its|the|their) image|"
    r"(?:do not |don't )?include[^.]*\bimage\b|without[^.]*\bimage\b)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_center(bbox: List[float]) -> Tuple[float, float]:
    """Compute pixel center from [x, y, w, h] bbox."""
    return bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2


def _extract_numbers(text: str) -> List[float]:
    """Extract all numeric values from text."""
    return [float(m) for m in re.findall(r"\b\d+(?:\.\d+)?\b", text)]


def _obs_mode(step: Dict) -> str:
    return step.get("observation_mode", "dom")


# ---------------------------------------------------------------------------
# Rule implementations
# ---------------------------------------------------------------------------

def check_p1(steps: List[Dict], _summary: Dict, _config: Dict, _mode: str) -> List[PatternHit]:
    """P1: 元素中心越界 — click/type target outside viewport."""
    hits = []
    for s in steps:
        at = s.get("action_type", "")
        if at not in ("click", "type"):
            continue
        mode = _obs_mode(s)
        if mode == "vision":
            coord = s.get("action", {}).get("coordinate")
            if coord and len(coord) >= 2:
                if coord[1] > 1.0 or coord[1] < 0.0 or coord[0] > 1.0 or coord[0] < 0.0:
                    hits.append(PatternHit(
                        "P1", "元素中心越界", s["step_idx"],
                        f"Vision coord ({coord[0]:.3f}, {coord[1]:.3f}) outside [0,1]",
                        is_scaffold=False,
                    ))
        else:  # dom / som
            bbox = s.get("element_bbox")
            if bbox and len(bbox) >= 4:
                cx, cy = _get_center(bbox)
                if cy > VIEWPORT_H or cy < 0 or cx > VIEWPORT_W or cx < 0:
                    hits.append(PatternHit(
                        "P1", "元素中心越界", s["step_idx"],
                        f"Bbox center ({cx:.0f}, {cy:.0f}) outside viewport {VIEWPORT_W}x{VIEWPORT_H}",
                        is_scaffold=False,
                    ))
    return hits


def check_p2(steps: List[Dict], _summary: Dict, _config: Dict, _mode: str) -> List[PatternHit]:
    """P2: 容器节点误点 — wide element click with no page change (classifieds)."""
    hits = []
    for s in steps:
        if s.get("action_type") != "click":
            continue
        if _obs_mode(s) == "vision":
            continue
        bbox = s.get("element_bbox")
        if not bbox or len(bbox) < 4:
            continue
        w = bbox[2]
        cx = bbox[0] + w / 2
        if w >= 500 and 780 <= cx <= 860 and not s.get("page_changed", True):
            hits.append(PatternHit(
                "P2", "容器节点误点", s["step_idx"],
                f"Click on wide element (w={w:.0f}, cx={cx:.0f}), page unchanged",
                is_scaffold=False,
            ))
    return hits


def check_p3(steps: List[Dict], _summary: Dict, _config: Dict, _mode: str) -> List[PatternHit]:
    """P3: Thought-Action 解耦 — thought says scroll down but delta is negative."""
    hits = []
    for s in steps:
        if s.get("action_type") != "scroll":
            continue
        action = s.get("action", {})
        thought = action.get("thought", "")
        delta = action.get("delta")
        if not delta or len(delta) < 2:
            continue
        # delta[1] < 0 means scroll up in both pixel and normalized
        if delta[1] < 0 and SCROLL_DOWN_PHRASES.search(thought):
            hits.append(PatternHit(
                "P3", "Thought-Action 解耦", s["step_idx"],
                f"Thought says scroll down but delta[1]={delta[1]}",
                is_scaffold=False,
            ))
    return hits


def check_p4(steps: List[Dict], _summary: Dict, _config: Dict, _mode: str) -> List[PatternHit]:
    """P4: 根节点误操作 — action targets root element (id=0/1 or full-viewport bbox)."""
    hits = []
    for s in steps:
        at = s.get("action_type", "")
        if at not in ("click", "type"):
            continue
        if _obs_mode(s) == "vision":
            continue
        eid = s.get("action", {}).get("element_id")
        bbox = s.get("element_bbox")
        is_root = False
        reason = ""
        if eid is not None and eid in (0, 1):
            is_root = True
            reason = f"element_id={eid}"
        elif bbox and len(bbox) >= 4 and bbox[2] > 1200 and bbox[3] > 680:
            is_root = True
            reason = f"bbox covers nearly full viewport ({bbox[2]:.0f}x{bbox[3]:.0f})"
        if is_root:
            hits.append(PatternHit(
                "P4", "根节点误操作", s["step_idx"],
                f"{at} on root node ({reason})",
                is_scaffold=False,
            ))
    return hits


def check_p5(steps: List[Dict], _summary: Dict, _config: Dict, _mode: str) -> List[PatternHit]:
    """P5: 感知缺失循环 — 3+ consecutive steps same action+target, page unchanged."""
    hits = []
    if len(steps) < 3:
        return hits

    def _action_key(s: Dict) -> str:
        at = s.get("action_type", "")
        eid = s.get("action", {}).get("element_id", "")
        coord = s.get("action", {}).get("coordinate", "")
        # Include scroll direction so opposite scrolls don't merge
        delta = s.get("action", {}).get("delta")
        delta_dir = ""
        if at == "scroll" and delta and len(delta) >= 2:
            delta_dir = "up" if delta[1] < 0 else "down" if delta[1] > 0 else "0"
        return f"{at}|{eid}|{coord}|{delta_dir}"

    run_start = 0
    for i in range(1, len(steps)):
        same = (
            _action_key(steps[i]) == _action_key(steps[i - 1])
            and not steps[i].get("page_changed", True)
            and not steps[i - 1].get("page_changed", True)
        )
        if not same:
            if i - run_start >= 3:
                hits.append(PatternHit(
                    "P5", "感知缺失循环", run_start,
                    f"Steps {run_start}-{i-1}: repeated {_action_key(steps[run_start])}, page unchanged",
                    is_scaffold=False,
                ))
            run_start = i
    # Check tail
    if len(steps) - run_start >= 3:
        hits.append(PatternHit(
            "P5", "感知缺失循环", run_start,
            f"Steps {run_start}-{len(steps)-1}: repeated {_action_key(steps[run_start])}, page unchanged",
            is_scaffold=False,
        ))
    return hits


def check_p6(steps: List[Dict], _summary: Dict, config: Dict, _mode: str) -> List[PatternHit]:
    """P6: 视觉任务 DOM 必然失败 — DOM mode + visual task."""
    if not steps:
        return []
    mode = _obs_mode(steps[0])
    if mode != "dom":
        return []
    intent = config.get("intent", "")
    has_image = bool(config.get("image"))
    has_color = bool(VISUAL_COLOR_KEYWORDS.search(intent) or VISUAL_COLOR_ADJ.search(intent))
    if has_image:
        return [PatternHit(
            "P6", "视觉任务 DOM 必然失败", None,
            "DOM mode cannot see image reference (task has image field)",
            is_scaffold=False,
        )]
    if has_color:
        return [PatternHit(
            "P6", "视觉任务 DOM 必然失败", None,
            f"DOM mode cannot perceive color in intent: {intent[:80]}",
            is_scaffold=False,
        )]
    return []


def check_p7(steps: List[Dict], _summary: Dict, _config: Dict, _mode: str) -> List[PatternHit]:
    """P7: sCity=州名 — URL contains sCity= with a US state name."""
    hits = []
    seen: Set[str] = set()
    for s in steps:
        url = s.get("obs_url", "")
        m = re.search(r"sCity=([^&]+)", url)
        if not m:
            continue
        city_val = m.group(1).replace("+", " ").replace("%20", " ").strip().lower()
        if city_val in US_STATES and city_val not in seen:
            seen.add(city_val)
            hits.append(PatternHit(
                "P7", "sCity=州名", s["step_idx"],
                f"sCity={m.group(1)} is a US state, not a city",
                is_scaffold=False,
            ))
    return hits


def check_p8(steps: List[Dict], _summary: Dict, _config: Dict, _mode: str) -> List[PatternHit]:
    """P8: select 反馈缺失 — repeated select_option, page_changed but text_sim high."""
    hits = []
    for i in range(1, len(steps)):
        prev, cur = steps[i - 1], steps[i]
        if prev.get("action_type") != "select_option" or cur.get("action_type") != "select_option":
            continue
        prev_text = prev.get("action", {}).get("text", "")
        cur_text = cur.get("action", {}).get("text", "")
        if prev_text and prev_text == cur_text:
            ts = cur.get("text_similarity", 0)
            if cur.get("page_changed") and ts > 0.9:
                hits.append(PatternHit(
                    "P8", "select 反馈缺失", i,
                    f"Repeated select '{cur_text}', page_changed=True but text_sim={ts:.3f}",
                    is_scaffold=True,
                ))
    return hits


def check_p10(steps: List[Dict], _summary: Dict, _config: Dict, _mode: str) -> List[PatternHit]:
    """P10: 跨步数值记忆失败 — thought mentions number X, action uses different number Y."""
    hits = []
    for s in steps:
        at = s.get("action_type", "")
        if at not in ("type", "finish"):
            continue
        action = s.get("action", {})
        thought = action.get("thought", "")
        output_text = action.get("text", "") or action.get("answer", "") or ""
        if not thought or not output_text:
            continue
        thought_nums = _extract_numbers(thought)
        output_nums = _extract_numbers(output_text)
        if not thought_nums or not output_nums:
            continue
        # Check if any output number doesn't match any thought number (within +-10)
        for y in output_nums:
            if y == 0:
                continue
            matched = any(abs(x - y) <= 10 for x in thought_nums)
            if not matched and any(x > 10 for x in thought_nums):
                hits.append(PatternHit(
                    "P10", "跨步数值记忆失败", s["step_idx"],
                    f"Thought nums {thought_nums[:5]} vs output num {y}",
                    is_scaffold=False,
                ))
                break
    return hits


def check_p11(steps: List[Dict], _summary: Dict, config: Dict, _mode: str) -> List[PatternHit]:
    """P11: 最新+地点组合 — intent has latest+location, check if sCity is used."""
    intent = config.get("intent", "")
    has_latest = bool(re.search(r"latest|most recent|newest", intent, re.IGNORECASE))
    has_location = bool(re.search(r"from\s+\w+|in\s+\w+|posted\s+in", intent, re.IGNORECASE))
    if not (has_latest and has_location):
        return []
    # Check if any step URL has sCity
    for s in steps:
        url = s.get("obs_url", "")
        if "sCity=" in url:
            return [PatternHit(
                "P11", "最新+地点组合", s["step_idx"],
                f"Intent has latest+location, sCity used in URL; check P7 for state name",
                is_scaffold=False,
            )]
    return []


def check_p12(steps: List[Dict], summary: Dict, _config: Dict, _mode: str) -> List[PatternHit]:
    """P12: 从不翻页 — no scroll in episode with many steps and high no-change rate."""
    if len(steps) < 6:
        return []
    has_scroll = any(s.get("action_type") == "scroll" for s in steps)
    if has_scroll:
        return []
    unchanged = sum(1 for s in steps if not s.get("page_changed", True))
    pct = unchanged / len(steps)
    if pct > 0.5:
        return [PatternHit(
            "P12", "从不翻页", None,
            f"{len(steps)} steps, no scroll, {pct:.0%} page unchanged",
            is_scaffold=False,
        )]
    return []


def check_p13(steps: List[Dict], _summary: Dict, _config: Dict, _mode: str) -> List[PatternHit]:
    """P13: 搜索代替浏览 — starts with search, never clicks a listing."""
    if len(steps) < 3:
        return []
    # First meaningful action is type (search)
    if steps[0].get("action_type") != "type":
        return []
    # Count action types after first search
    post_search = steps[1:]
    click_count = sum(1 for s in post_search if s.get("action_type") == "click")
    type_count = sum(1 for s in post_search if s.get("action_type") == "type")
    scroll_count = sum(1 for s in post_search if s.get("action_type") == "scroll")
    if click_count == 0 and type_count >= 2 and len(post_search) >= 4:
        return [PatternHit(
            "P13", "搜索代替浏览", None,
            f"Starts with search, then {type_count} more type + {scroll_count} scroll, 0 clicks in {len(post_search)} steps",
            is_scaffold=False,
        )]
    return []


def check_p14(steps: List[Dict], _summary: Dict, _config: Dict, _mode: str) -> List[PatternHit]:
    """P14: URL 自环 — 3+ consecutive steps with identical obs_url (excluding start page)."""
    hits = []
    if len(steps) < 3:
        return hits
    # Use task config start_url (not steps[0].obs_url which is post-first-action)
    start_url = _config.get("start_url", "") or steps[0].get("state_digest", {}).get("url_before", "")
    run_start = 0
    for i in range(1, len(steps)):
        url_cur = steps[i].get("obs_url", "")
        url_prev = steps[i - 1].get("obs_url", "")
        if url_cur == url_prev and url_cur:
            pass  # continue run
        else:
            run_len = i - run_start
            if run_len >= 3:
                url = steps[run_start].get("obs_url", "")
                if url != start_url:
                    hits.append(PatternHit(
                        "P14", "URL 自环", run_start,
                        f"Steps {run_start}-{i-1}: stuck on {url[:80]}",
                        is_scaffold=False,
                    ))
            run_start = i
    # Check tail
    run_len = len(steps) - run_start
    if run_len >= 3:
        url = steps[run_start].get("obs_url", "")
        if url != start_url:
            hits.append(PatternHit(
                "P14", "URL 自环", run_start,
                f"Steps {run_start}-{len(steps)-1}: stuck on {url[:80]}",
                is_scaffold=False,
            ))
    return hits


def check_p15(_steps: List[Dict], _summary: Dict, config: Dict, mode: str) -> List[PatternHit]:
    """P15: gallery 行位置查询 — DOM 线性化网格无法定位视觉行 (self-evolving 2026-05-22, diagnose Tier-2 task 14/41/42)."""
    if mode != "dom":
        return []
    intent = config.get("intent", "")
    start_url = config.get("start_url", "")
    if "sShowAs=gallery" in start_url and GALLERY_ROW_RE.search(intent):
        return [PatternHit(
            "P15", "gallery行位置DOM不可定位", None,
            f"gallery view + row-position intent; DOM linearizes grid: {intent[:80]}",
            is_scaffold=False,
        )]
    return []


def check_p16(_steps: List[Dict], _summary: Dict, config: Dict, mode: str) -> List[PatternHit]:
    """P16: 视觉图像内容任务 — cover/image 内容过滤, DOM 无像素 (self-evolving 2026-05-22, diagnose Tier-2 task 80/81)."""
    if mode != "dom":
        return []
    intent = config.get("intent", "")
    if VISUAL_IMAGE_CONTENT_RE.search(intent):
        return [PatternHit(
            "P16", "视觉图像内容DOM必败", None,
            f"image-content task (cover/image filter); DOM has no pixels: {intent[:80]}",
            is_scaffold=False,
        )]
    return []


def check_p17(steps: List[Dict], _summary: Dict, _config: Dict, _mode: str) -> List[PatternHit]:
    """P17: click-back 振荡 — 同一 item 反复进入+退出, detail↔list 横跳无进展 (self-evolving 2026-05-22, diagnose Tier-2 task 40/111)."""
    from collections import Counter
    item_visits: Counter = Counter()
    for s in steps:
        url = s.get("obs_url", "")
        if "page=item" in url:
            m = re.search(r"[?&]id=(\d+)", url)
            if m:
                item_visits[m.group(1)] += 1
    n_back = sum(1 for s in steps if s.get("action_type") == "back")
    repeated = [(iid, c) for iid, c in item_visits.items() if c >= 3]
    if repeated and n_back >= 2:
        iid, c = max(repeated, key=lambda x: x[1])
        return [PatternHit(
            "P17", "click-back振荡", None,
            f"item id={iid} revisited {c}x with {n_back} back actions (detail↔list thrash)",
            is_scaffold=False,
        )]
    return []


def check_p18(steps: List[Dict], _summary: Dict, config: Dict, _mode: str) -> List[PatternHit]:
    """P18: cheapest 任务漏价格排序 — intent 要 cheapest 但全程从未按 i_price 排序 (self-evolving 2026-05-22, diagnose Tier-2 task 216)."""
    intent = config.get("intent", "")
    if not re.search(r"\b(cheapest|lowest[- ]price|least expensive)\b", intent, re.IGNORECASE):
        return []
    sorted_by_price = False
    has_search = False
    for s in steps:
        url = s.get("obs_url", "")
        if "page=search" in url:
            has_search = True
        if "i_price" in url:
            sorted_by_price = True
            break
        act = s.get("action", {}) or {}
        txt = str(act.get("text", "")).lower()
        if "lower price" in txt or "price first" in txt:
            sorted_by_price = True
            break
    if has_search and not sorted_by_price:
        return [PatternHit(
            "P18", "cheapest漏价格排序", None,
            f"cheapest/lowest intent but never sorted by price: {intent[:60]}",
            is_scaffold=False,
        )]
    return []


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

ALL_RULES: Dict[str, Any] = {
    "P1": check_p1,
    "P2": check_p2,
    "P3": check_p3,
    "P4": check_p4,
    "P5": check_p5,
    "P6": check_p6,
    "P7": check_p7,
    "P8": check_p8,
    "P10": check_p10,
    "P11": check_p11,
    "P12": check_p12,
    "P13": check_p13,
    "P14": check_p14,
    "P15": check_p15,
    "P16": check_p16,
    "P17": check_p17,
    "P18": check_p18,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_steps(path: Path) -> List[Dict]:
    try:
        from p79.experiment.io_utils import read_jsonl_dedup
        return read_jsonl_dedup(path)
    except ImportError:
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return rows


def _discover_episodes(
    run_dir: Path,
    condition_filter: Optional[str] = None,
    task_filter: Optional[int] = None,
) -> List[Tuple[Path, Path, Path]]:
    """Discover (steps_path, summary_path, config_path) triples."""
    results = []
    for cond_dir in sorted(run_dir.iterdir()):
        if not cond_dir.is_dir() or cond_dir.name in ("task_configs", "artifacts"):
            continue
        if condition_filter and cond_dir.name != condition_filter:
            continue
        ep_dir = cond_dir / "episodes"
        if not ep_dir.is_dir():
            continue
        for steps_path in sorted(ep_dir.glob("*_steps_v2.jsonl")):
            stem = steps_path.stem.replace("_steps_v2", "")
            summary_path = ep_dir / f"{stem}_summary_v2.json"
            config_path = run_dir / "task_configs" / f"{stem}.json"
            if not summary_path.exists():
                continue
            # Extract task_id from stem (e.g., "classifieds_task_42")
            m = re.search(r"_task_(\d+)$", stem)
            if not m:
                continue
            tid = int(m.group(1))
            if task_filter is not None and tid != task_filter:
                continue
            results.append((steps_path, summary_path, config_path))
    return results


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

def scan_episodes(
    run_dir: Path,
    *,
    condition_filter: Optional[str] = None,
    task_filter: Optional[int] = None,
    failed_only: bool = False,
    rule_filter: Optional[Set[str]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Scan all episodes and return structured results."""
    episodes = _discover_episodes(run_dir, condition_filter, task_filter)
    if not episodes:
        print(f"No episodes found in {run_dir}", file=sys.stderr)
        return {}

    rules_to_run = {k: v for k, v in ALL_RULES.items()
                    if rule_filter is None or k in rule_filter}

    all_diagnoses: List[Dict] = []
    hit_counts: Dict[str, int] = {r: 0 for r in rules_to_run}
    run_id_cache: Optional[str] = None

    for steps_path, summary_path, config_path in episodes:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if failed_only and summary.get("success"):
            continue

        steps = _load_steps(steps_path)
        if not steps:
            continue

        config: Dict = {}
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))

        task_id = summary.get("task_id", steps[0].get("task_id", -1))
        cond_id = summary.get("condition_id", steps[0].get("condition_id", ""))
        site = summary.get("benchmark_site", steps[0].get("benchmark_site", ""))
        mode = _obs_mode(steps[0]) if steps else "dom"
        if run_id_cache is None:
            run_id_cache = steps[0].get("run_id")

        diag = EpisodeDiagnosis(
            task_id=task_id,
            condition_id=cond_id,
            site=site,
            success=summary.get("success", False),
            steps=len(steps),
        )

        for rule_id, check_fn in rules_to_run.items():
            try:
                rule_hits = check_fn(steps, summary, config, mode)
            except Exception as e:
                if verbose:
                    print(f"  Rule {rule_id} error on task {task_id}: {e}", file=sys.stderr)
                continue
            diag.hits.extend(rule_hits)
            if rule_hits:
                hit_counts[rule_id] += 1

        # B-1829 (diag /stress P1-1): unconditional append. failed_only already
        # skips success above (line ~648), so appending all here keeps no-hit
        # FAILED episodes in results (the Tier-2 deep-dive target) AND makes
        # `total` = true failed count. Was `if diag.hits or not failed_only`,
        # which dropped no-hit failed from the denominator (--failed-only
        # reported Episodes:178 vs true failed=191 on R9755 → wrong pct).
        all_diagnoses.append(asdict(diag))

        if verbose and diag.hits:
            print(f"  Task {task_id} ({cond_id}): {len(diag.hits)} hits — "
                  + ", ".join(h.rule_id for h in diag.hits))

    total = len(all_diagnoses)
    with_hits = sum(1 for d in all_diagnoses if d["hits"])

    run_id = run_id_cache or run_dir.name

    result = {
        "run_id": run_id,
        "ruleset_version": RULESET_VERSION,
        "scan_time": datetime.now(timezone.utc).isoformat(),
        "rules_applied": sorted(rules_to_run.keys()),
        "total_episodes": total,
        "episodes_with_hits": with_hits,
        "results": sorted(all_diagnoses, key=lambda d: (d["condition_id"], d["task_id"])),
        "summary": {
            r: {"count": c, "pct": round(c / total * 100, 1) if total else 0}
            for r, c in sorted(hit_counts.items())
        },
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch pattern-matching diagnostics for P79 episodes",
    )
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Run directory (e.g. results/visualwebarena/phase1/B1_3mode_classifieds_20260413)")
    parser.add_argument("--condition", type=str, default=None,
                        help="Filter to specific condition_id (e.g. phase1_dom_router_0)")
    parser.add_argument("--task-id", type=int, default=None,
                        help="Filter to single task ID")
    parser.add_argument("--failed-only", action="store_true",
                        help="Only scan failed episodes")
    parser.add_argument("--rules", type=str, default=None,
                        help="Comma-separated rule IDs (e.g. P1,P3,P14)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path (default: stdout)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress to stderr")

    args = parser.parse_args()

    if not args.run_dir.is_dir():
        print(f"Error: {args.run_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    rule_filter = None
    if args.rules:
        rule_filter = set(args.rules.upper().split(","))
        unknown = rule_filter - set(ALL_RULES.keys())
        if unknown:
            print(f"Warning: unknown rules {unknown}, available: {sorted(ALL_RULES.keys())}", file=sys.stderr)
            rule_filter -= unknown

    result = scan_episodes(
        args.run_dir,
        condition_filter=args.condition,
        task_filter=args.task_id,
        failed_only=args.failed_only,
        rule_filter=rule_filter,
        verbose=args.verbose,
    )

    if not result:
        sys.exit(1)

    output_json = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_json, encoding="utf-8")
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(output_json)

    # Print summary table to stderr
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Run: {result['run_id']}", file=sys.stderr)
    print(f"Episodes: {result['total_episodes']}  |  With hits: {result['episodes_with_hits']}", file=sys.stderr)
    print(f"{'─'*60}", file=sys.stderr)
    for r, info in sorted(result["summary"].items()):
        if info["count"] > 0:
            bar = "█" * min(info["count"], 40)
            print(f"  {r:>4s}: {info['count']:4d} ({info['pct']:5.1f}%)  {bar}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    main()
