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
# Current basis: P1-P18 discovered from B0 dom classifieds R9755 (in-sample fit);
# P19-P23 + P6/P14 narrowing added from R31194 fresh-substrate Tier-2 (2026-05-23);
# P24-P30 + P10/P14/P20 FP-narrowing added from R9725 B0 som cls Tier-2 (2026-05-24,
# 2nd-mode discover). Now spans 2 modes (dom+som) — the mode gates (`if mode != "dom"`
# in check_p6 / p15 / p16) + ALL rules are provisional discover-products, NOT yet
# validated against vision / phantom modes.
# Cross-mode quantitative comparison remains FORBIDDEN until 6-mode freeze.
#
# B-1860 (2026-05-24): check_p1 vision-coord branch now normalizes through the
# Qwen 0-1000 contract (`normalize_coordinate_pair`) BEFORE the OOB test, so a
# canonical 0-1000 coord is no longer false-flagged. This is a CORRECTNESS edit
# to an existing rule (not a new discover), but per the bump-on-any-ALL_RULES-
# change contract the version is bumped to make the content change auditable.
# NOTE: no vision-mode condition has been Tier-2 discovered yet, so this fix is
# forward-looking — P1 vision had ZERO real inputs in the dom+som corpus, so
# re-scanning the existing dom/som digests is a no-op for P1 (their bbox branch
# is unchanged). Re-scan is still advised before any vision diag.
#
# ⚠️ SYNC: `.claude/skills/diag/SKILL.md` is gitignored (no CI guard) — when bumping
# this version OR editing ALL_RULES, MANUALLY update SKILL.md's "当前 P-rules" list +
# "当前相位" section. (R31194 session left them stale at "13 条 / 1-dom" for ~half a
# month because the skill doc has no git tracking to flag the drift.)
RULESET_VERSION = "4-domsomvis-b1860coord"

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

# B-1860: single-source coordinate normalizer. P1 (coord-OOB) must judge a
# vision coord through the SAME Qwen-0-1000 contract the runner applies, so a
# canonical 0-1000 coord (e.g. [598, 125]) is NOT false-flagged as out-of-[0,1]
# — only a coord that is STILL > 1 / < 0 AFTER normalization is a true OOB.
try:
    from p79.backends.action_utils import normalize_coordinate_pair as _normalize_coordinate_pair
except Exception:  # pragma: no cover - script may run without p79 on PYTHONPATH
    _normalize_coordinate_pair = None

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

# --- self-evolving 2026-05-23 (R31194 B0 dom cls Tier-2; ruleset 1-dom → 2-dom) ---

# P21: DOM-mode visual hallucination — finish answer/thought asserts perception of
# a PAGE/LISTING image that dom mode cannot see. MUST be tied to page content
# (listing/item context OR a photographic "image ... taken" claim). References to
# the TASK's OWN reference image (which IS passed to the multimodal model even in
# dom mode) are excluded — else this repeats P6's 88%-on-success over-fire.
DOM_PAGE_IMAGE_CLAIM_RE = re.compile(
    r"\b(?:listing|item|product|car|vehicle|this|its)\b[^.]{0,40}"
    r"\b(?:image|picture|photo)\b[^.]{0,30}"
    r"\b(?:taken|shows?|depicts?|appears?|reveals?|indicates?)\b"
    r"|\b(?:image|picture|photo)\b\s+(?:that\s+|which\s+)?"
    r"(?:is|was|appears?\s+to\s+be|seems?\s+to\s+be)\s+taken\b",
    re.IGNORECASE,
)
REF_IMAGE_RE = re.compile(
    r"\b(?:reference|provided|given|attached|task|target|uploaded|sample|example)\s+"
    r"(?:image|picture|photo)\b",
    re.IGNORECASE,
)

# P22: image-only number/quantity — the answer fact (a number printed on the
# listing image, or a count) exists only in the image, invisible to dom.
IMG_NUMBER_INTENT_RE = re.compile(
    r"number\s+(?:shown\s+)?(?:on|in)\s+(?:the\s+)?(?:image|picture|photo)|"
    r"(?:shown|displayed|visible)\s+(?:on|in)\s+(?:the\s+)?(?:image|picture|photo)",
    re.IGNORECASE,
)
COUNT_INTENT_RE = re.compile(r"\bhow many\b|\bnumber of\b", re.IGNORECASE)
GAVE_UP_RE = re.compile(
    r"not\s+specif|cannot\s+(?:determine|tell|find)|"
    r"does\s+not\s+(?:specify|mention|state|indicate)|"
    r"unable\s+to|no\s+(?:specific\s+)?(?:number|count|quantity)|isn'?t\s+specified",
    re.IGNORECASE,
)

# P23: oldest-listing intent solved with price-sort (no date-sort UI exists) —
# agent substitutes i_price ordering for chronology.
OLDEST_INTENT_RE = re.compile(r"\boldest\b|\bearliest\b", re.IGNORECASE)

# P6 narrowing: a task merely HAVING a reference image does NOT make it dom-blind
# (the multimodal model receives + can OCR the reference image even in dom mode —
# this drove P6's 88%-on-success over-fire). Only fire the image branch when the
# intent requires VISUALLY matching that image to page content.
P6_IMAGE_VISUAL_MATCH_RE = re.compile(
    r"\b(?:selfie|pictured|depicted|looks?\s+like|similar\s+to|matching|shown\s+in|"
    r"(?:this\s+)?exact\s+item|"
    r"in\s+(?:the|its|their|this)\s+(?:image|picture|photo)|taken\s+(?:on|in|from|at|during)|"
    r"on\s+(?:the|its)\s+(?:cover|front))\b",
    re.IGNORECASE,
)

# --- self-evolving 2026-05-24 (R9725 B0 som cls Tier-2; ruleset 2-dom → 3-domsom) ---

# P10 FP-narrowing: date numbers (16th November 2023) are NOT cross-step memory facts.
# R9725 task 25 fired P10 because thought "16th November 2023" → nums [16, 2023] vs
# answer "1" (a count). Strip date contexts before the number-mismatch check.
DATE_CONTEXT_RE = re.compile(
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b"
    r"|\b(?:19|20)\d{2}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}/\d{1,2}/\d{1,2}\b",
    re.IGNORECASE,
)

# P24: finish hedged with explicit uncertainty yet submitted anyway (agent knows the
# item may not match but visual search is exhausted → surrender-finish).
UNCERTAINTY_FINISH_RE = re.compile(
    r"not\s+explicitly|though\s+it'?s?\s+actually|while\s+(?:the\s+)?(?:description|listing|item)[^.]{0,25}does\s+not"
    r"|may\s+(?:match|be\b)|might\s+(?:match|be\b)|despite\s+(?:being|not)|not\s+a\s+perfect\s+match"
    r"|does\s+not\s+(?:explicitly\s+)?(?:mention|match|show)|although[^.]{0,40}\bnot\b",
    re.IGNORECASE,
)

# P27: finish answer abandons the task (gave up at a page instead of returning/retrying).
# R24792 vision discover: extended to cover "No <noun> ... is visible" (task 174) and
# "No <noun> ... was found" (task 186) — vision agents that scroll a wrong item page,
# fail to spot the target, and finish with a generic not-found phrasing.
ABANDONMENT_RE = re.compile(
    r"cannot\s+be\s+completed|does\s+not\s+display|(?:is|are)\s+not\s+(?:found|available|visible)"
    r"|task\s+cannot|unable\s+to\s+(?:find|complete|locate)|could\s+not\s+(?:find|locate|be\s+found)"
    r"|no\s+(?:such\s+)?(?:item|listing|result|page)\s+(?:found|exists|available)"
    r"|no\s+\w+(?:[^.]{0,50}?)\s+(?:was|were|is|are)\s+(?:found|visible|present|available)",
    re.IGNORECASE,
)

# P29 (benchmark-FP): semantic yes/no equivalents when reference literal is yes/no.
YESNO_SEMANTIC_RE = re.compile(
    r"\b(correct|incorrect|is\s+right|is\s+wrong|indeed|affirmative|"
    r"that'?s\s+true|that'?s\s+false|does\s+match|do\s+not\s+match)\b",
    re.IGNORECASE,
)

# P32: keyword text typed into the numeric price filter → malformed sPriceMin/Max URL
# value (agent put a search term into the price box). Self-evolving 2026-05-25 (R24792
# vision Tier-2 task 34: "painting animals" → sPriceMin=painti). Naturally success-safe:
# a malformed price-filter URL never appears in a successful trajectory.
PRICE_FILTER_TEXT_RE = re.compile(r"[?&]sPrice(?:Min|Max)=[^&]*[A-Za-z]")

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
                # B-1860: normalize through the Qwen-0-1000 contract BEFORE the
                # OOB check. A canonical 0-1000 coord (e.g. [598, 125]) is the
                # runner's accepted format, NOT an OOB — only a coord that is
                # STILL outside [0,1] after normalization (i.e. raw > 1000 →
                # true_oob, or raw < 0) is a genuine grounding miss. Pre-B-1860
                # this rule hard-checked `coord > 1.0` → false-flagged every
                # 0-1000 vision coord as OOB (the parse_error 13.6% mislabel).
                x_oob = y_oob = False
                x_raw, y_raw = coord[0], coord[1]
                # V-F2 (B-1860 codex verify P1, 2026-05-24): a raw negative
                # dimension points off the top/left of the page — a genuine
                # grounding miss (the comment above promised "raw < 0 → OOB").
                # The normalizer tags negatives `malformed`; pre-fix the
                # `malformed → continue` below SWALLOWED them, under-counting
                # coord failures. Flag negative as OOB explicitly, BEFORE the
                # malformed skip. (bool excluded — bool is an int subclass.)
                _x_num = isinstance(x_raw, (int, float)) and not isinstance(x_raw, bool)
                _y_num = isinstance(y_raw, (int, float)) and not isinstance(y_raw, bool)
                if (_x_num and x_raw < 0) or (_y_num and y_raw < 0):
                    hits.append(PatternHit(
                        "P1", "元素中心越界", s["step_idx"],
                        f"Vision coord raw ({x_raw}, {y_raw}) has a negative "
                        f"dimension — off-page grounding miss (true OOB)",
                        is_scaffold=False,
                    ))
                    continue
                if _normalize_coordinate_pair is not None:
                    x_n, y_n, _tags = _normalize_coordinate_pair([x_raw, y_raw])
                    if _tags["malformed"]:
                        # Non-negative malformed (NaN/inf/shape) — other rules.
                        continue
                    x_oob = x_n > 1.0 or x_n < 0.0
                    y_oob = y_n > 1.0 or y_n < 0.0
                else:
                    # Fallback (p79 not importable): inline the by-value contract
                    # (`> 1.1` → /1000) so the rule still does the right thing.
                    x_n = x_raw / 1000.0 if x_raw > 1.1 else x_raw
                    y_n = y_raw / 1000.0 if y_raw > 1.1 else y_raw
                    x_oob = x_n > 1.0 or x_n < 0.0
                    y_oob = y_n > 1.0 or y_n < 0.0
                if x_oob or y_oob:
                    hits.append(PatternHit(
                        "P1", "元素中心越界", s["step_idx"],
                        f"Vision coord raw ({x_raw}, {y_raw}) → norm "
                        f"({x_n:.3f}, {y_n:.3f}) outside [0,1] (true OOB)",
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
    if has_image and P6_IMAGE_VISUAL_MATCH_RE.search(intent):
        # Narrowed (R31194 FP audit): reference image alone ≠ dom-blind (model OCRs
        # it). Only fire when intent requires visually matching it to page content.
        return [PatternHit(
            "P6", "视觉任务 DOM 必然失败", None,
            "DOM mode cannot visually match reference image to page content",
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
        # P10 FP-narrowing (R9725 task 25): a number may legitimately echo a DATE in the
        # thought ("16th November") — match against FULL thought nums (so answer can
        # reference a date), but only count NON-date nums (thought_nodate) as facts that
        # should have carried over. Count answer "1" vs date "16th November 2023" then
        # no longer fires; genuine cross-step numeric-memory failures still fire.
        thought_nums = _extract_numbers(thought)
        thought_nodate = _extract_numbers(DATE_CONTEXT_RE.sub(" ", thought))
        output_nums = _extract_numbers(output_text)
        if not thought_nums or not output_nums:
            continue
        # Check if any output number doesn't match any thought number (within +-10)
        for y in output_nums:
            if y == 0:
                continue
            matched = any(abs(x - y) <= 10 for x in thought_nums)
            if not matched and any(x > 10 for x in thought_nodate):
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
    """P14: URL 自环 — 4+ consecutive steps with identical obs_url (excluding start page).

    Threshold raised 3→4 (R31194 FP audit). R9725 som FP audit (P14 fired on 8/8
    success-hit episodes = ~100% FP on success) further narrowed: a same-URL run is
    only "stuck" if it shows NO progress. Skip productive runs — any `type` action
    (form fill / search input / comment) OR a majority of page_changed=True steps
    (scrolling/browsing a long page). Genuine stuck loops are same-URL + no-type +
    mostly page-unchanged (e.g. repeated dead clicks).
    """
    hits = []
    if len(steps) < 4:
        return hits
    # Use task config start_url (not steps[0].obs_url which is post-first-action)
    start_url = _config.get("start_url", "") or steps[0].get("state_digest", {}).get("url_before", "")

    def _emit(run_start: int, run_end: int) -> None:  # run_end exclusive
        run = steps[run_start:run_end]
        if len(run) < 4:
            return
        url = run[0].get("obs_url", "")
        if not url or url == start_url:
            return
        # R9725 FP-narrowing: productive same-URL is not "stuck"
        if any(s.get("action_type") == "type" for s in run):
            return
        changed = sum(1 for s in run if s.get("page_changed"))
        if changed * 2 >= len(run):  # >=50% steps changed the page → making progress
            return
        hits.append(PatternHit(
            "P14", "URL 自环", run_start,
            f"Steps {run_start}-{run_end-1}: stuck (no progress) on {url[:80]}",
            is_scaffold=False,
        ))

    run_start = 0
    for i in range(1, len(steps)):
        url_cur = steps[i].get("obs_url", "")
        if url_cur == steps[i - 1].get("obs_url", "") and url_cur:
            continue  # extend run
        _emit(run_start, i)
        run_start = i
    _emit(run_start, len(steps))  # tail
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


def check_p19(steps: List[Dict], _summary: Dict, config: Dict, _mode: str) -> List[PatternHit]:
    """P19: url_match 任务过早在搜索/列表页 finish — agent 没进 item 详情页就 finish,
    而 url_match 比对的是当前页 URL (self-evolving 2026-05-23, R31194 Tier-2 task 210)."""
    ev = config.get("eval") or {}
    if "url_match" not in (ev.get("eval_types") or []):
        return []
    ref_url = ev.get("reference_url") or ""
    if "page=search" in ref_url:  # target itself is a search page → not premature
        return []
    finish_url = None
    for s in steps:
        if s.get("action_type") == "finish":
            finish_url = s.get("obs_url", "")
            break
    if finish_url is None and steps:
        finish_url = steps[-1].get("obs_url", "")
    if finish_url and "page=search" in finish_url:
        return [PatternHit(
            "P19", "url_match过早搜索页finish", None,
            f"url_match finished on search/list page (not item detail): {finish_url[:70]}",
            is_scaffold=False,
        )]
    return []


def check_p20(steps: List[Dict], _summary: Dict, config: Dict, _mode: str) -> List[PatternHit]:
    """P20: program_html 评测目标页从未访问 — agent 在错误 listing 上操作, eval_target_url
    全程未出现在 obs_url 历史 (self-evolving 2026-05-23, R31194 Tier-2 task 223)."""
    ev = config.get("eval") or {}
    if "program_html" not in (ev.get("eval_types") or []):
        return []
    targets = []
    for ph in ev.get("program_html") or []:
        # P20 FP-narrowing (R9725 task 5): delete-verification tasks check for "404"
        # via the evaluator's own goto — agent deletes from the LIST page (AJAX) and
        # never navigates to the item URL, so "never visited" is expected, not failure.
        rc = (ph.get("required_contents") or {}).get("must_include") or []
        if any("404" in str(x) for x in rc):
            continue
        u = ph.get("url", "")
        if isinstance(u, str) and u.startswith("http"):
            m = re.search(r"[?&]id=(\d+)", u)
            if m:
                targets.append(m.group(1))
    if not targets:
        return []
    visited: Set[str] = set()
    for s in steps:
        m = re.search(r"[?&]id=(\d+)", s.get("obs_url", ""))
        if m:
            visited.add(m.group(1))
    missing = [t for t in dict.fromkeys(targets) if t not in visited]
    if missing:
        return [PatternHit(
            "P20", "评测目标页从未访问", None,
            f"program_html target item id={missing[0]} never visited (acted on wrong listing)",
            is_scaffold=False,
        )]
    return []


def check_p21(steps: List[Dict], _summary: Dict, config: Dict, mode: str) -> List[PatternHit]:
    """P21: dom 模式视觉幻觉 — finish 声称看到 listing/page 图像内容, 但 dom 看不到页面像素.
    Gated on has_image==False (R31194 verify): a task WITH a reference image makes
    "the image" ambiguous (could be the legit ref image the model sees); WITHOUT one,
    any "image" claim must be about page content dom cannot see = hallucination.
    This cleanly removes the task-62/63 echo-the-intent FP (R31194 Tier-2 task 91)."""
    if mode != "dom":
        return []
    if config.get("image"):
        return []
    for s in steps:
        if s.get("action_type") != "finish":
            continue
        a = s.get("action", {}) or {}
        text = " ".join(str(a.get(k, "")) for k in ("thought", "answer"))
        if not text.strip():
            continue
        m = DOM_PAGE_IMAGE_CLAIM_RE.search(text)
        if m and not REF_IMAGE_RE.search(text):
            return [PatternHit(
                "P21", "dom模式视觉幻觉", s.get("step_idx"),
                f"finish claims page-image perception in dom mode: '...{m.group(0)[:60]}...'",
                is_scaffold=False,
            )]
    return []


def check_p22(steps: List[Dict], _summary: Dict, config: Dict, _mode: str) -> List[PatternHit]:
    """P22: 图像唯一信息 (图上数字 / 数量) — 答案 fact 仅在 listing 图中, dom 不可得
    (self-evolving 2026-05-23, R31194 Tier-2 task 100/221)."""
    ev = config.get("eval") or {}
    if "string_match" not in (ev.get("eval_types") or []):
        return []
    must = (ev.get("reference_answers") or {}).get("must_include") or []
    ref_nums: List[str] = []
    for r in must:
        for tok in str(r).split("|OR|"):
            tok = tok.strip()
            if re.fullmatch(r"\$?\d+(?:\.\d+)?", tok):
                ref_nums.append(re.sub(r"[^\d.]", "", tok))
    if not ref_nums:
        return []
    intent = config.get("intent", "")
    finish_ans = ""
    for s in steps:
        if s.get("action_type") == "finish":
            a = s.get("action", {}) or {}
            finish_ans = str(a.get("answer", "") or a.get("text", ""))
            break
    ans_nums = set(re.findall(r"\d+(?:\.\d+)?", finish_ans))
    hit_ref = any(rn in ans_nums for rn in ref_nums)
    if hit_ref:
        return []
    # (a) intent explicitly reads a number off the image; answer lacks ref number
    if IMG_NUMBER_INTENT_RE.search(intent):
        return [PatternHit(
            "P22", "图上数字dom不可读", None,
            f"intent reads number from image; answer lacks ref {ref_nums}: {finish_ans[:45]}",
            is_scaffold=False,
        )]
    # (b) count question + agent gave up (quantity only in image)
    if COUNT_INTENT_RE.search(intent) and GAVE_UP_RE.search(finish_ans):
        return [PatternHit(
            "P22", "图中数量dom不可数", None,
            f"count question, agent could not determine (image-only): {finish_ans[:45]}",
            is_scaffold=False,
        )]
    return []


def check_p23(steps: List[Dict], _summary: Dict, config: Dict, _mode: str) -> List[PatternHit]:
    """P23: oldest-listing 用价格排序代替日期排序 — UI 无 date-sort, agent 用 i_price 当
    chronology (self-evolving 2026-05-23, R31194 Tier-2 task 156)."""
    intent = config.get("intent", "")
    if not OLDEST_INTENT_RE.search(intent):
        return []
    for s in steps:
        if "sOrder=i_price" in s.get("obs_url", ""):
            return [PatternHit(
                "P23", "oldest误用价格排序", s.get("step_idx"),
                "oldest-listing intent but sorted by i_price (no date-sort substitute)",
                is_scaffold=False,
            )]
    return []


def check_p24(steps: List[Dict], _summary: Dict, config: Dict, _mode: str) -> List[PatternHit]:
    """P24: 不确定仍 finish — url_match 任务 finish thought/answer 含明确不确定限定语,
    且 finish 的 item id ≠ reference id (agent 知道不匹配但视觉搜索耗尽 → 投降式 finish)
    (self-evolving 2026-05-24, R9725 som Tier-2 task 101/176/201). success-safe: 选对
    (finish_id==ref_id) 不 fire。"""
    ev = config.get("eval") or {}
    ref_url = ev.get("reference_url") or ""
    ref_m = re.search(r"[?&]id=(\d+)", ref_url)
    if not ref_m or "page=search" in ref_url:
        return []
    ref_id = ref_m.group(1)
    for s in steps:
        if s.get("action_type") != "finish":
            continue
        a = s.get("action", {}) or {}
        text = " ".join(str(a.get(k, "")) for k in ("thought", "answer"))
        m = UNCERTAINTY_FINISH_RE.search(text)
        if not m:
            return []
        fin_m = re.search(r"[?&]id=(\d+)", s.get("obs_url", ""))
        if fin_m and fin_m.group(1) == ref_id:
            return []  # selected the right item; hedging is harmless
        return [PatternHit(
            "P24", "不确定仍finish", s.get("step_idx"),
            f"url_match finish hedged on wrong/unclear item: '...{m.group(0)[:45]}...'",
            is_scaffold=False,
        )]
    return []


def check_p25(steps: List[Dict], _summary: Dict, config: Dict, _mode: str) -> List[PatternHit]:
    """P25: 跨站任务跳过其中一站 — start_url 含 |AND| 多站, 但 obs_url 从未访问其中 ≥1 个
    host:port (agent 跳过跨站视觉推理) (self-evolving 2026-05-24, R9725 som Tier-2
    task 227/232). success-safe: 真做完跨站任务会访问所有站。"""
    start_url = config.get("start_url", "") or ""
    if "|AND|" not in start_url:
        return []

    def _hostport(u: str) -> Optional[str]:
        m = re.search(r"https?://([^/\s]+)", u)
        return m.group(1) if m else None

    sites: Set[str] = set()
    for part in start_url.split("|AND|"):
        hp = _hostport(part.strip())
        if hp:
            sites.add(hp)
    if len(sites) < 2:
        return []
    visited: Set[str] = set()
    for s in steps:
        hp = _hostport(s.get("obs_url", ""))
        if hp:
            visited.add(hp)
    missing = sites - visited
    if missing:
        return [PatternHit(
            "P25", "跨站任务跳过其中一站", None,
            f"multi-site task ({len(sites)} sites) never visited {sorted(missing)}",
            is_scaffold=False,
        )]
    return []


def check_p27(steps: List[Dict], _summary: Dict, config: Dict, _mode: str) -> List[PatternHit]:
    """P27: 找不到即放弃 — finish.answer 含放弃短语 (agent 到页面找不到目标即 finish,
    不返回上级重试) (self-evolving 2026-05-24, R9725 som Tier-2 task 118/163; ABANDONMENT_RE
    extended R24792 vision task 174/186).
    success-safe: 成功 ep 的 finish 不会说放弃 (N/A task 已 task-load 排除)。Carve-out
    (R24792 全量重扫 dom task 151): url_match agent 主观放弃 ("task cannot be completed")
    但 finish obs_url 实际停在 reference item → eval 凭 live url pass, 放弃措辞 harmless
    (同 P24/P30 finish==ref 不 fire)。"""
    ev = config.get("eval") or {}
    ref_m = re.search(r"[?&]id=(\d+)", ev.get("reference_url") or "")
    ref_id = ref_m.group(1) if ref_m else None
    for s in steps:
        if s.get("action_type") != "finish":
            continue
        a = s.get("action", {}) or {}
        ans = str(a.get("answer", "") or a.get("text", ""))
        m = ABANDONMENT_RE.search(ans)
        if not m:
            continue
        if ref_id:
            fin_m = re.search(r"[?&]id=(\d+)", s.get("obs_url", ""))
            if fin_m and fin_m.group(1) == ref_id:
                return []  # ended on reference item; url_match passes despite give-up phrasing
        return [PatternHit(
            "P27", "找不到即放弃", s.get("step_idx"),
            f"finish abandons task: '...{m.group(0)[:50]}...'",
            is_scaffold=False,
        )]
    return []


def check_p28(steps: List[Dict], _summary: Dict, config: Dict, _mode: str) -> List[PatternHit]:
    """P28 (benchmark-FP): NLTK 货币 tokenize 假阴性 — string_match must_include 全为纯
    整数, finish.answer 含该数的货币格式 ($N.NN) 但 NLTK word_tokenize('$5.00')=['$','5.00']
    致整数 token 缺失 → 误判 (self-evolving 2026-05-24, R9725 som Tier-2 task 42).
    success-safe: 答案含独立整数 token 时不 fire。"""
    ev = config.get("eval") or {}
    if "string_match" not in (ev.get("eval_types") or []):
        return []
    must = [str(r).strip() for r in ((ev.get("reference_answers") or {}).get("must_include") or [])]
    ints = [r for r in must if re.fullmatch(r"\d{1,6}", r)]
    if not must or len(ints) != len(must):
        return []
    finish_ans = ""
    for s in steps:
        if s.get("action_type") == "finish":
            a = s.get("action", {}) or {}
            finish_ans = str(a.get("answer", "") or a.get("text", ""))
            break
    if not finish_ans:
        return []
    flagged = []
    for n in ints:
        currency = re.search(rf"\${n}\.\d{{2}}|\b{n}\.\d{{2}}", finish_ans)
        token = re.search(rf"(?<![\d.$]){n}(?![\d.])", finish_ans)
        if currency and not token:
            flagged.append(n)
    if flagged:
        return [PatternHit(
            "P28", "benchmark-FP货币tokenize", None,
            f"must_include {flagged} present as $N.NN but NLTK splits → false-neg: {finish_ans[:42]}",
            is_scaffold=False,
        )]
    return []


def check_p29(steps: List[Dict], _summary: Dict, config: Dict, _mode: str) -> List[PatternHit]:
    """P29 (benchmark-FP): yes/no 语义等价不匹配 — must_include 是 yes/no, finish.answer
    用 correct/incorrect 等语义等价词但不含字面 yes/no (self-evolving 2026-05-24, R9725
    som Tier-2 task 222). success-safe: 含字面 yes/no 时不 fire。"""
    ev = config.get("eval") or {}
    if "string_match" not in (ev.get("eval_types") or []):
        return []
    must = [str(r).strip().lower() for r in ((ev.get("reference_answers") or {}).get("must_include") or [])]
    if not must or not all(m in ("yes", "no") for m in must):
        return []
    finish_ans = ""
    for s in steps:
        if s.get("action_type") == "finish":
            a = s.get("action", {}) or {}
            finish_ans = str(a.get("answer", "") or a.get("text", ""))
            break
    if not finish_ans:
        return []
    low = finish_ans.lower()
    if any(re.search(rf"\b{m}\b", low) for m in must):
        return []  # literal yes/no present → eval should pass
    if YESNO_SEMANTIC_RE.search(finish_ans):
        return [PatternHit(
            "P29", "benchmark-FP语义yes/no", None,
            f"ref={must} but answer uses semantic equiv: {finish_ans[:45]}",
            is_scaffold=False,
        )]
    return []


def check_p30(steps: List[Dict], _summary: Dict, config: Dict, _mode: str) -> List[PatternHit]:
    """P30: 到达正确 item 后离开 — obs_url 序列曾命中 reference item id 后又离开, 最终
    finish≠reference (som 标注图致 agent 过度自我否定) (self-evolving 2026-05-24, R9725
    som Tier-2 task 93). success-safe: finish==ref 时不 fire。"""
    ev = config.get("eval") or {}
    ref_m = re.search(r"[?&]id=(\d+)", ev.get("reference_url") or "")
    if not ref_m:
        return []
    ref_id = ref_m.group(1)
    seq = []
    for s in steps:
        m = re.search(r"[?&]id=(\d+)", s.get("obs_url", ""))
        seq.append(m.group(1) if m else None)
    if ref_id not in seq:
        return []
    fin_id = None
    for s in steps:
        if s.get("action_type") == "finish":
            m = re.search(r"[?&]id=(\d+)", s.get("obs_url", ""))
            fin_id = m.group(1) if m else None
            break
    if fin_id is None and seq:
        fin_id = seq[-1]
    if fin_id == ref_id:
        return []  # ended on the right item
    last_ref_idx = max(i for i, v in enumerate(seq) if v == ref_id)
    if last_ref_idx < len(seq) - 1:  # left the reference item after reaching it
        return [PatternHit(
            "P30", "到达正确item后离开", last_ref_idx,
            f"reached reference id={ref_id} (step {last_ref_idx}) then left, finished id={fin_id}",
            is_scaffold=False,
        )]
    return []


def check_p31(steps: List[Dict], summary: Dict, config: Dict, _mode: str) -> List[PatternHit]:
    """P31: budget 耗尽未完成 — trajectory_incomplete (agent 用尽 step budget 仍未到达
    valid finish / eval) (self-evolving 2026-05-25, R24792 vision Tier-2 task 77/136/203).
    FP-narrowing (R24792 task 5): delete-verification tasks (program_html checks "404"
    via evaluator goto) succeed on the DB side-effect even when the agent never reaches a
    valid finish — trajectory_incomplete is expected there, NOT a failure signal (same
    404 carve-out as P20)."""
    if not summary.get("trajectory_incomplete"):
        return []
    ev = config.get("eval") or {}
    for ph in ev.get("program_html") or []:
        rc = (ph.get("required_contents") or {}).get("must_include") or []
        if any("404" in str(x) for x in rc):
            return []
    return [PatternHit(
        "P31", "budget耗尽未完成", None,
        f"trajectory_incomplete after {len(steps)} steps "
        f"(no valid finish; agent_finished={summary.get('agent_finished')})",
        is_scaffold=False,
    )]


def check_p32(steps: List[Dict], _summary: Dict, _config: Dict, _mode: str) -> List[PatternHit]:
    """P32: 文本误入价格 filter — obs_url 出现 sPriceMin/sPriceMax 含字母 (agent 把搜索
    关键词打进数字价格框) (self-evolving 2026-05-25, R24792 vision Tier-2 task 34).
    天然 success-safe: malformed price-filter URL 不出现在成功轨迹。"""
    for s in steps:
        url = s.get("obs_url", "") or ""
        if PRICE_FILTER_TEXT_RE.search(url):
            return [PatternHit(
                "P32", "文本误入价格filter", s.get("step_idx"),
                f"non-numeric price filter (keyword typed into price box): {url[:80]}",
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
    "P19": check_p19,
    "P20": check_p20,
    "P21": check_p21,
    "P22": check_p22,
    "P23": check_p23,
    "P24": check_p24,  # 不确定仍finish (url_match wrong-item hedge)
    "P25": check_p25,  # 跨站任务跳过其中一站
    # P26 skipped (finish_at_search_page deferred — hard to separate from legit
    #   search-page count tasks without over-firing; see B0_som digest)
    "P27": check_p27,  # 找不到即放弃 (abandonment phrase)
    "P28": check_p28,  # benchmark-FP 货币 tokenize
    "P29": check_p29,  # benchmark-FP 语义 yes/no
    "P30": check_p30,  # 到达正确item后离开 (som self-doubt)
    "P31": check_p31,  # budget 耗尽未完成 (trajectory_incomplete; R24792 vision discover)
    "P32": check_p32,  # 文本误入价格 filter (sPriceMin/Max 含字母; R24792 vision discover)
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
