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
from urllib.parse import urlparse

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
# 2nd-mode discover); P31/P32 + P27 ext added from R24792 B0 vision cls Tier-2
# (2026-05-25, 3rd-mode discover); P33/P34 added from R32031 B0 phantom_som cls Tier-2
# (2026-05-28, 4th-mode discover): P33 phantom-img-nav ([SOM_MARKS] exposes a listing
# image href as clickable → agent lands on raw /oc-content/uploads/*.png, hallucinates;
# success-fire 1/17 on R32031 = clean). P34 phantom_som visual-blind (P6 color/ref-image
# twin) was PROPOSED but NOT landed: re-scan showed 21/106 success-fire (20%) = presence-
# only — the SAME lesson as P6's historic 88%-on-success dom over-fire. "navigate to my
# listing of the white car" (color = self-listing identifier, not a visual judgment) +
# "I recall seeing this exact item" (ref image IS OCR-able by the multimodal model → only
# needs OCR→search, no page-image match) both fire spuriously. Needs a success-safe narrow
# before landing (see B0_phantom_som digest § self-evolve). Now spans 4 modes
# (dom+som+vision+phantom_som) but phantom_som only contributes P33 so far. The dom-gated
# visual-blind DETAIL rules (P15 gallery-row / P16 image-content / P22 img-number) ALSO
# manifest on phantom_som but are NOT yet extended (need full phantom-family data to
# decide som-specific-twin vs no-image-family-wide gate — see B0_phantom_som digest).
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
RULESET_VERSION = "11-intent-text-fallback"

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

# P33: agent navigated to a RAW listing-image URL (/oc-content/uploads/<id>/<id>.png).
# In phantom_som the [SOM_MARKS] list exposes each listing image's href as a clickable
# element → agent "clicks the image" → lands on a near-empty DOM page showing only the
# raw image (still unreadable to the model) → hallucinates an answer. Signal is mode-
# agnostic (any mode landing on a bare image URL is lost) but structurally phantom_som-
# induced. Naturally success-safe: a raw image URL is never the item page url_match wants.
# 2026-07-27 (reddit discover batch): the classifieds-only `/oc-content/uploads/`
# path missed reddit's equivalent trap entirely. Postmill serves submission images
# at `/submission_images/<hash>.<ext>`, and reddit episodes land there constantly —
# clicking a post thumbnail (whose href IS the raw image file, not the post page)
# navigates to a near-empty DOM with the image the text-only modes cannot read.
# Verified present across B0/B1/B2 reddit Tier-2 samples (B1_vision task 172/176/177,
# B2_ptext task 103, B1_psom task 19). Union of both site conventions keeps cls
# behaviour byte-identical (the cls alternative is unchanged and reddit URLs never
# matched the old pattern).
RAW_IMAGE_URL_RE = re.compile(
    r"(?:/oc-content/uploads/|/submission_images/).*\.(?:png|jpe?g|gif|webp)(?:$|\?)",
    re.IGNORECASE,
)

# --- reddit discover batch 2026-07-27 (ruleset 7-* → 8-reddit-*) ---
# Sources: 9 Tier-2 sub-agents over reddit × {B0,B1,B2} × 6 modes, then every
# population-level claim re-verified by 0-token full scan (笔记 §387.6-§387.12).

# P43: intent needs visual information. Deliberately BROAD — the rule pairs it with
# "this mode delivers no page screenshot", and the combination is reported as a
# neutral (task × mode) label, NOT as a predicted failure. §387.10 measured the
# actual effect of restoring the screenshot on exactly this task set (dom→som:
# B0 +0.00 / B1 +1.56 / B2 +0.00 pp) — i.e. these tasks are NOT "structurally
# unsolvable without the image", they are hard for every representation. The label
# exists so the 64 reddit tasks in this bucket stop being invisible to Tier-1.
VISUAL_INTENT_RE = re.compile(
    r"\b(image|picture|photo|screenshot)\b|\bcolou?r of\b|"
    r"\bhow many\b[^.]{0,40}\bin (?:the|this)\b",
    re.IGNORECASE,
)

# P46: intents whose completion requires committing text to the site (a comment /
# reply). §387.8 measured these at 2.11% SR pooled over 18 cells vs 8.49% for the
# rest (4.0x), consistent in 18/18 cells. Word-bounded and deliberately NARROWER
# than MUTATION_INTENT_RE: widening to post/submit/create/edit/upvote erases the
# gap entirely (7.23% vs 6.01%), so "mutation task" is the wrong abstraction here.
COMMENT_INTENT_RE = re.compile(
    r"\b(comment|reply|replies|saying)\b",
    re.IGNORECASE,
)

# P44: the locator's OTHER error branch. `walk_fail:*` means "element resolved but
# no actionable ancestor"; this one means the referenced element_id was not in
# obs_nodes_info at all — i.e. a hallucinated reference. Four Tier-2 sub-agents
# independently reported this branch "never occurs"; that held in their 6-8 episode
# samples but NOT in the population (§387.12: B2 fires it on 7.84% of psom and
# 18.21% of dom action-steps). No existing rule covers it.
MISSING_UNION_BOUND_RE = re.compile(r"missing union_bound", re.IGNORECASE)

P34_GIVEUP_RE = re.compile(
    r"cannot verify|image not visible|not listed|\[\]",
    re.IGNORECASE,
)

P38_IMAGE_URL_INTENT_RE = re.compile(
    r"\bin the image\b|\bwebsite\b[^.]{0,40}\bimage\b|\bimage\b[^.]{0,40}\bwebsite\b",
    re.IGNORECASE,
)

MUTATION_INTENT_RE = re.compile(
    r"\b(delete|remove|edit|update|change|modify|submit|post|create|comment|reply|rate|"
    r"mark\s+as\s+sold|take\s+down)\b",
    re.IGNORECASE,
)

LUCKY_NUMERIC_TOKENS = {
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "none", "no", "not", "nothing", "false", "n/a",
}

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


def _benchmark_site(summary: Dict, steps: List[Dict]) -> str:
    """Authoritative episode site (mirrors scan_episodes site resolution)."""
    s = (summary or {}).get("benchmark_site")
    if not s and steps:
        s = steps[0].get("benchmark_site")
    return s or ""


def _find_finish_step(steps: List[Dict]) -> Optional[Dict]:
    for s in steps:
        if s.get("action_type") == "finish":
            return s
    return None


def _finish_answer(steps: List[Dict]) -> str:
    """The text the agent actually SUBMITTED as its answer.

    Deliberately does NOT fall back to `thought` — see `_finish_intent_text` for why
    that distinction matters. Rules that compare answer *content* (P37/P38: does the
    submitted answer contain example.com / localhost) and P46 (did the agent submit an
    answer at all) must keep using this.
    """
    s = _find_finish_step(steps)
    if not s:
        return ""
    a = s.get("action", {}) or {}
    return str(a.get("answer", "") or a.get("text", "") or "")


def _finish_intent_text(steps: List[Dict]) -> str:
    """Answer text, falling back to the finish step's `thought` when answer is empty.

    v11 (2026-08-03 数据质量审计). Motivation: `P48` was proposed FROM B1 data yet fires
    0 times on B1 and 15 times on B0. Root cause found during the audit — B0 (proxy 235B)
    habitually restates its conclusion into `answer`, while B1 (local 4B) leaves `answer`
    empty and keeps the reasoning in `thought`. A rule that detects a *stated belief*
    ("there are no results") therefore became a model-behaviour detector rather than a
    failure detector. 9 further B1 episodes already cleared P48's step gate and were
    excluded on this alone.

    ⚠️ Use ONLY for rules matching an INTENT/WORDING (P34 give-up phrasing, P48 negative
    conclusion). Do NOT use for rules comparing submitted answer CONTENT (P37/P38) —
    a model mentioning "example.com" while reasoning is not the same as submitting it —
    nor for P46, whose whole point is that `answer` is absent.
    """
    a = _finish_answer(steps)
    if a:
        return a
    s = _find_finish_step(steps)
    if not s:
        return ""
    return str((s.get("action", {}) or {}).get("thought", "") or "")


def _step_has_walk_fail(step: Dict) -> bool:
    for key in ("locator_route_meta", "locator_route_meta_primary", "locator_route_meta_retry"):
        meta = step.get(key)
        if isinstance(meta, dict) and "walk_fail" in str(meta.get("error", "")):
            return True
    return False


def _locator_errors(step: Dict) -> List[str]:
    """All non-empty locator error strings on a step, across primary/retry slots."""
    errors = []
    for key in ("locator_route_meta", "locator_route_meta_primary", "locator_route_meta_retry"):
        meta = step.get(key)
        if isinstance(meta, dict):
            err = str(meta.get("error", "") or "")
            if err:
                errors.append(err)
    return errors


def _action_type(step: Dict) -> str:
    """Action type, tolerating both the nested and the flattened step-record shapes."""
    act = step.get("action")
    if isinstance(act, dict) and act.get("action_type"):
        return str(act.get("action_type"))
    return str(step.get("action_type") or "")


def _action_element_id(step: Dict) -> Optional[int]:
    act = step.get("action")
    raw = act.get("element_id") if isinstance(act, dict) else step.get("element_id")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _count_mutating_steps(steps: List[Dict]) -> int:
    """Derive the count of state-mutating actions from the step records.

    B-1890 (2026-07-27): `summary["effective_mutating_action_count"]` — and its five
    siblings — are schema slots the runner has NEVER populated; every episode on disk
    carries 0 (`types.py:587-602` defers the aggregation "until action heuristic spec
    lock"). Two existing rules (P35, P39) guard on `!= 0: return []`, so that guard has
    always been a NO-OP: it never filtered anything, while reading as though the rule
    had verified an absence of mutation. Both rules were therefore looser than their
    docstrings claim.

    This derives the count the way `types.py:587` specifies — action_success AND
    page_changed AND a mutating action_type — so the guard becomes real. `click` is
    included because on Postmill the mutating controls (Subscribe, upvote, Save) are
    buttons/links, and a click that both succeeded and changed the page is the only
    signal available; the false-positive direction (counting a navigational click as a
    mutation) makes P35/P39 STRICTER than the broken version, never looser, so it
    cannot manufacture new hits.
    """
    n = 0
    for s in steps:
        if _action_type(s) not in ("type", "click", "select_option", "key_enter"):
            continue
        if s.get("action_success") is not True:
            continue
        if s.get("page_changed") is not True:
            continue
        n += 1
    return n


def _visited_hosts(steps: List[Dict]) -> Set[str]:
    """Distinct `host:port` values seen in obs_url over the episode."""
    hosts = set()
    for s in steps:
        url = str(s.get("obs_url") or "")
        m = re.match(r"https?://([^/]+)", url)
        if m:
            hosts.add(m.group(1))
    return hosts


def _walk_fail_errors(step: Dict) -> List[str]:
    errors = []
    for key in ("locator_route_meta", "locator_route_meta_primary", "locator_route_meta_retry"):
        meta = step.get(key)
        if isinstance(meta, dict):
            err = str(meta.get("error", "") or "")
            if "walk_fail" in err:
                errors.append(err)
    return errors


def _flatten_strings(obj: Any) -> List[str]:
    if obj is None:
        return []
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, (int, float, bool)):
        return [str(obj)]
    if isinstance(obj, dict):
        out: List[str] = []
        for v in obj.values():
            out.extend(_flatten_strings(v))
        return out
    if isinstance(obj, (list, tuple, set)):
        out = []
        for v in obj:
            out.extend(_flatten_strings(v))
        return out
    return [str(obj)]


def _eval_reference_strings(config: Dict) -> List[str]:
    ev = config.get("eval") or {}
    refs = []
    refs.extend(_flatten_strings(ev.get("reference_answers")))
    if ev.get("reference_url"):
        refs.append(str(ev.get("reference_url")))
    for ph in ev.get("program_html") or []:
        refs.extend(_flatten_strings(ph.get("url")))
        refs.extend(_flatten_strings(ph.get("required_contents")))
    return refs


def _string_match_reference_tokens(config: Dict) -> Set[str]:
    ev = config.get("eval") or {}
    refs = []
    refs.extend(_flatten_strings(ev.get("reference_answers")))
    for ph in ev.get("program_html") or []:
        rc = (ph.get("required_contents") or {}).get("must_include") or []
        refs.extend(_flatten_strings(rc))

    tokens: Set[str] = set()
    for ref in refs:
        for tok in re.findall(r"n/a|[A-Za-z]+|\d+", str(ref).lower()):
            tokens.add(tok)
    return tokens


def _program_html_locators(config: Dict) -> List[str]:
    ev = config.get("eval") or {}
    return [str(ph.get("locator", "") or "") for ph in ev.get("program_html") or []]


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


def check_p4(steps: List[Dict], summary: Dict, _config: Dict, _mode: str) -> List[PatternHit]:
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
            if summary.get("success") and _step_has_walk_fail(s):
                continue
            hits.append(PatternHit(
                "P4", "根节点误操作", s["step_idx"],
                f"{at} on root node ({reason})",
                is_scaffold=False,
            ))
    return hits


def check_p5(steps: List[Dict], _summary: Dict, _config: Dict, _mode: str) -> List[PatternHit]:
    """P5: 感知缺失循环 — 3+ consecutive steps same action+target, page unchanged."""
    if _summary.get("success"):
        return []
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


def check_p6(steps: List[Dict], summary: Dict, config: Dict, _mode: str) -> List[PatternHit]:
    """P6: 视觉任务 DOM 必然失败 — DOM mode + visual task (classifieds-gated)."""
    if not steps:
        return []
    mode = _obs_mode(steps[0])
    if mode != "dom":
        return []
    if _benchmark_site(summary, steps) != "classifieds":
        # H1 (2026-06-27, reddit discover): P6 is a classifieds-calibrated visual
        # rule. On reddit it produced only presence-only cross-site FPs — the task
        # reference image is delivered to the model even in dom mode (ref-image
        # channel), so "dom can't match ref-to-page" doesn't hold; genuine reddit
        # page-image blindness is owned by the eval_type=page_image_query detector,
        # not this intent-regex rule. Empirically P6 never fired on the real reddit
        # page-blind failures (they were no-hit). Gate to classifieds.
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
    finish_action = _find_finish_step(steps)
    if finish_action:
        answer = str((finish_action.get("action") or {}).get("answer", "") or "")
        if answer.startswith("http"):
            return []
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
    if summary.get("success"):
        return []
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
    if _summary.get("success"):
        return []
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
        # v10 carve-out (2026-08-03 数据质量审计): a run that spans essentially the
        # WHOLE of a short episode is a single "click in → read → finish" decision,
        # not a self-loop. Two VWA causal-verification samples (cls task 50 / 119,
        # both 4 steps) showed the opposite of "stuck": 119 navigated to the correct
        # item in ONE click and then misread a price from the image. Calling that
        # "URL 自环 / no progress" inverts the meaning. Require that the episode did
        # something outside this run before treating the run as a loop.
        if len(steps) <= 6 and (run_end - run_start) >= len(steps) - 1:
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


def check_p16(steps: List[Dict], summary: Dict, config: Dict, mode: str) -> List[PatternHit]:
    """P16: 视觉图像内容任务 — cover/image 内容过滤, DOM 无像素 (self-evolving 2026-05-22, diagnose Tier-2 task 80/81; classifieds-gated H1 2026-06-27)."""
    if mode != "dom":
        return []
    if _benchmark_site(summary, steps) != "classifieds":
        # H1 (2026-06-27, reddit discover): same cross-site FP as P6 — reddit
        # reference images are model-visible in dom; this intent-regex rule never
        # caught the real reddit page-image failures (no-hit), only mis-fired.
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
    if _summary.get("success"):
        return []
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
    if any("sOrder=i_price" in (s.get("obs_url", "") or "") for s in steps):
        return []
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
    if finish_url is None:
        return []
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
    if summary.get("success"):
        return []
    if not summary.get("trajectory_incomplete"):
        return []
    ev = config.get("eval") or {}
    eval_types = ev.get("eval_types") or []
    ref_url = ev.get("reference_url") or ""
    last_obs_url = steps[-1].get("obs_url", "") if steps else ""
    if ref_url and any(t in eval_types for t in ("url_match", "agent_page")):
        if urlparse(last_obs_url).path == urlparse(ref_url).path:
            return []
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


def check_p33(steps: List[Dict], _summary: Dict, _config: Dict, _mode: str) -> List[PatternHit]:
    """P33: 导航至裸图片 URL 幻觉 — obs_url 落到 /oc-content/uploads/*.png 近空 DOM 页.
    phantom_som 特有诱因: [SOM_MARKS] 把 listing 图片 href 暴露为带 ID 可点击元素, agent
    "点进图片" → 裸图片页无可读内容却幻觉作答 (self-evolving 2026-05-28, R32031 B0
    phantom_som cls Tier-2 task 128/187; 两 sub-agent 独立发现). 天然 success-safe:
    裸图片 URL ≠ item 页, url_match/program_html 不会 pass."""
    raw_idxs = [
        i for i, s in enumerate(steps)
        if RAW_IMAGE_URL_RE.search(s.get("obs_url", "") or "")
    ]
    if not raw_idxs:
        return []
    raw_set = set(raw_idxs)
    severity = "low"
    for i in raw_idxs:
        if i + 1 in raw_set and not any(s.get("action_type") == "back" for s in steps[i:i + 2]):
            severity = "high"
            break
    for i, s in enumerate(steps):
        url = s.get("obs_url", "") or ""
        if RAW_IMAGE_URL_RE.search(url):
            return [PatternHit(
                "P33", "导航至裸图片URL幻觉", i,
                f"step {i}: obs_url is raw listing image {url[:80]} "
                f"(clicked img href, lost; severity={severity})",
                is_scaffold=False,
            )]
    return []


def check_p34(steps: List[Dict], summary: Dict, config: Dict, mode: str) -> List[PatternHit]:
    """P34: image-task blind give-up — no image input, short run, explicit visual give-up."""
    if summary.get("success"):
        return []
    if mode != "dom" and not mode.startswith("phantom_"):
        return []
    if not config.get("image"):
        return []
    if len(steps) > 3:
        return []
    input_image = sum(int(((s.get("tokens") or {}).get("input_image")) or 0) for s in steps)
    if input_image != 0:
        return []
    answer = _finish_intent_text(steps)   # v11: intent wording, thought fallback OK
    if not answer or not P34_GIVEUP_RE.search(answer):
        return []
    return [PatternHit(
        "P34", "VISUAL_BLIND_IMAGE_TASK", None,
        f"image task in {mode}, input_image=0, short give-up finish: {answer[:80]}",
        is_scaffold=False,
    )]


def check_p35(steps: List[Dict], summary: Dict, config: Dict, _mode: str) -> List[PatternHit]:
    """P35: mutation missing — program_html side-effect task finished without mutation."""
    if summary.get("success"):
        return []
    ev = config.get("eval") or {}
    if "program_html" not in (ev.get("eval_types") or []):
        return []
    eval_source = str(summary.get("eval_source_agent_url") or "")
    locators = _program_html_locators(config)
    if "item_edit" not in eval_source and not any(".comments_list" in loc for loc in locators):
        return []
    # B-1890 fix (2026-07-27): was `summary["effective_mutating_action_count"]`, a
    # never-populated schema slot that is 0 for every episode on disk → this guard
    # was a no-op. Derived from step records instead (see _count_mutating_steps).
    if _count_mutating_steps(steps) != 0:
        return []
    if summary.get("agent_finished") is not True:
        return []
    return [PatternHit(
        "P35", "MUTATION_MISSING", None,
        "program_html side-effect task finished with 0 derived mutating steps",
        is_scaffold=False,
    )]


def _successfully_typed_element_ids(steps: List[Dict]) -> Set[Any]:
    """element_ids that a later `type` action reached successfully via the locator route.

    `dispatch_id_based_type` runs its OWN DOM walk-up (`_JS_RESOLVE_INPUT`), so a
    failed pre-focus *click* on the same element never blocks the subsequent type.
    Used by P36 to drop that harmless class of walk_fail.
    """
    out: Set[Any] = set()
    for s in steps:
        if s.get("action_type") != "type" or not s.get("action_success"):
            continue
        for key in ("locator_route_meta", "locator_route_meta_primary"):
            meta = s.get(key)
            if isinstance(meta, dict) and meta.get("success"):
                eid = (s.get("action") or {}).get("element_id")
                if eid is not None:
                    out.add(eid)
                break
    return out


def check_p36(steps: List[Dict], summary: Dict, _config: Dict, _mode: str) -> List[PatternHit]:
    """P36: mode-robust walk failure — locator walk_fail while trying to act.

    v10 carve-out (2026-08-03 数据质量审计): a `click` walk_fail whose element_id is
    LATER reached by a successful `type` is harmless — `dispatch_id_based_type` has an
    independent DOM walk-up, so the failed pre-focus click blocked nothing. Empirically
    16.3% of all P36 click hits (3862/23712) were of this class, and the share is
    mode-skewed (phantom_som 20.6% vs dom 6.2%), i.e. it injected a BIASED noise floor
    into every cross-mode comparison. Verified across 48 conditions.

    ⚠️ Known scope limits (documented, not fixed here):
      - vision mode: `click` carries no `locator_route_meta` at all (0/914–0/2270),
        so P36 only ever sees `type` steps there → the vision column has a DIFFERENT
        denominator and is not comparable to dom/som.
      - 10 cross-benchmark causal-verification samples judged P36 a *risk marker*,
        not a death cause. Do not read P36 counts as a death-cause distribution.
    """
    if summary.get("success"):
        return []
    hits = []
    seen: Set[Tuple[int, str]] = set()
    typed_ok = _successfully_typed_element_ids(steps)
    for s in steps:
        if s.get("action_type") not in ("click", "type"):
            continue
        if s.get("action_type") == "click":
            eid = (s.get("action") or {}).get("element_id")
            if eid is not None and eid in typed_ok:
                continue  # v10: superseded by a successful type on the same element
        for err in _walk_fail_errors(s):
            if not ("no_input_within_walk" in err or "no_actionable_within_walk" in err):
                continue
            key = (s.get("step_idx"), err)
            if key in seen:
                continue
            seen.add(key)
            hits.append(PatternHit(
                "P36", "WALK_FAIL_DEGENERATE", s.get("step_idx"),
                f"{s.get('action_type')} locator {err}",
                is_scaffold=False,
            ))
    return hits


def check_p37(steps: List[Dict], summary: Dict, config: Dict, _mode: str) -> List[PatternHit]:
    """P37: URL hallucination — example.com answer when task references localhost URL."""
    if summary.get("success"):
        return []
    answer = _finish_answer(steps)
    if "example.com" not in answer.lower():
        return []
    refs = _eval_reference_strings(config)
    if not any("localhost" in r for r in refs):
        return []
    return [PatternHit(
        "P37", "URL_HALLUCINATION", None,
        f"finish answer hallucinated example.com while reference contains localhost: {answer[:80]}",
        is_scaffold=False,
    )]


def check_p38(steps: List[Dict], summary: Dict, config: Dict, _mode: str) -> List[PatternHit]:
    """P38: DOM URL as image content — returns localhost URL for website-in-image task."""
    if summary.get("success"):
        return []
    intent = config.get("intent", "")
    if not P38_IMAGE_URL_INTENT_RE.search(intent):
        return []
    answer = _finish_answer(steps)
    if "localhost" not in answer.lower():
        return []
    refs = _eval_reference_strings(config)
    has_external_ref = any(
        "localhost" not in r.lower() and re.search(r"\b[a-z0-9-]+\.[a-z]{2,}\b", r, re.IGNORECASE)
        for r in refs
    )
    if not has_external_ref:
        return []
    return [PatternHit(
        "P38", "DOM_URL_AS_IMAGE", None,
        f"image/website intent answered with localhost URL instead of external reference: {answer[:80]}",
        is_scaffold=False,
    )]


def check_p39(steps: List[Dict], summary: Dict, config: Dict, _mode: str) -> List[PatternHit]:
    """P39: success without mutation — diagnostic benchmark-FP for mutation tasks."""
    if summary.get("success") is not True:
        return []
    ev = config.get("eval") or {}
    if "program_html" not in (ev.get("eval_types") or []):
        return []
    intent = config.get("intent", "")
    if not MUTATION_INTENT_RE.search(intent):
        return []
    # B-1890 fix (2026-07-27): see check_p35 — the old field is never populated.
    if _count_mutating_steps(steps) != 0:
        return []
    if summary.get("agent_finished") is not False:
        return []
    return [PatternHit(
        "P39", "SUCCESS_NO_MUTATION", None,
        "success=True mutation task with 0 derived mutating steps and agent_finished=False",
        is_scaffold=False,
    )]


def check_p40(steps: List[Dict], summary: Dict, config: Dict, _mode: str) -> List[PatternHit]:
    """P40: lucky numeric/string FP — trivial answer token succeeded without item-detail navigation."""
    if summary.get("success") is not True:
        return []
    ev = config.get("eval") or {}
    if ev.get("eval_types") != ["string_match"]:
        return []
    ref_tokens = _string_match_reference_tokens(config)
    if not ref_tokens or not ref_tokens.issubset(LUCKY_NUMERIC_TOKENS):
        return []
    detail_markers = ("page=item", "product_id=", "/product/")
    if any(any(marker in (s.get("obs_url", "") or "") for marker in detail_markers) for s in steps):
        return []
    return [PatternHit(
        "P40", "LUCKY_NUMERIC_FP", None,
        f"string_match success with trivial reference tokens {sorted(ref_tokens)} and no item-detail visit",
        is_scaffold=False,
    )]


# --- reddit discover batch P41-P46 (2026-07-27, ruleset 7-* → 8-reddit-*) ---
# Provenance: 9 Tier-2 sub-agents over reddit × {B0,B1,B2} × 6 modes. Every
# population-level claim below was re-verified by 0-token full scan before landing;
# three sub-agent claims were REJECTED at that step and are not encoded here
# (see 笔记 §387.10 / §387.12 for what was dropped and why).


def check_p41(steps: List[Dict], summary: Dict, config: Dict, _mode: str) -> List[PatternHit]:
    """P41: passive must_exclude FP — negative-only eval satisfiable by doing nothing.

    B-1889. A `program_html` check whose `required_contents` carries ONLY
    `must_exclude` is satisfied by an agent that never acts, because the post-reset
    state already excludes the listed distractors. reddit task 160 is the only such
    task in the 210-task pool, yet it was scored success in 13 of 18 conditions, and
    a trajectory hard-constraint check (Subscribe controls exist only on
    `/f/<forum>` pages) showed 13/13 could not have completed it: 10 never reached
    any forum page, the other 3 only reached forums not starting with 'i'.

    success-side diagnostic: marks the episode for review, does not alter SR.
    Deliberately NOT keyed on `effective_mutating_action_count` (B-1890 — always 0).

    Gating note (corrected 2026-07-27 after a first pass fired only 1 of the 13 known
    cases): the rule does NOT require zero derived mutating steps. "Passable by doing
    nothing" is a property of the EVAL SHAPE, not of the individual trajectory, and
    `_count_mutating_steps` counts navigational clicks as mutations — every task-160
    episode typed in the search box and clicked around, so a mutation gate suppressed
    12 of 13 true positives. The derived count is reported in the detail string
    instead, for the reviewer to weigh.
    """
    if summary.get("success") is not True:
        return []
    ev = config.get("eval") or {}
    blocks = ev.get("program_html") or []
    if not blocks:
        return []
    saw_required = False
    for blk in blocks:
        rc = (blk or {}).get("required_contents") or {}
        if not rc:
            continue
        saw_required = True
        if rc.get("must_include") or rc.get("exact_match") or rc.get("fuzzy_match"):
            return []          # has a positive check → not vacuously satisfiable
    if not saw_required:
        return []
    n_mut = _count_mutating_steps(steps)
    return [PatternHit(
        "P41", "PASSIVE_MUST_EXCLUDE_FP", None,
        f"success on a must_exclude-only program_html eval "
        f"(no positive check → satisfiable without acting); derived mutating steps={n_mut}",
        is_scaffold=False,
    )]


def check_p42(steps: List[Dict], summary: Dict, config: Dict, _mode: str) -> List[PatternHit]:
    """P42: multi-site task answered from one site — parametric-knowledge shortcut.

    B-1892. reddit task 58 (`sites: [wikipedia, reddit]`) asks for the author of the
    most popular novel-adapted anime of 2012; the answer is common knowledge, so the
    cross-site retrieval the task was designed to measure can be bypassed. 8 of the
    9 conditions that scored it success never loaded `localhost:8888` at all.

    Scope note: 40 reddit tasks are multi-site but they yield only 11 successes total
    across 18 cells, and 8 of those 11 are this one task — so this is a per-task
    selection issue, not a systematic property of multi-site tasks. Kept as a
    review-candidate flag, NOT an automatic exclusion: reading the answer off a
    reddit comment and then answering directly is legitimate.

    ⚠️ Known under-fire: the gate reads `config["sites"]`, which does not always
    declare every site an intent actually requires. cls task 233 declares
    `sites: ["classifieds"]` while its intent reads "the characters in the image ON
    REDDIT ... shown in the listing on the classifieds site" (found 2026-07-27 while
    verifying the P33 path extension, which caught that episode visiting
    localhost:9999). Tasks whose cross-site requirement lives only in the intent
    prose are invisible to this rule. Tightening it would need intent-side host
    detection; left as-is because a false ACCUSATION of an ungrounded answer is worse
    than a miss for a review-candidate flag.
    """
    if summary.get("success") is not True:
        return []
    sites = config.get("sites") or []
    if len(sites) < 2:
        return []
    hosts = _visited_hosts(steps)
    if len(hosts) >= len(sites):
        return []
    return [PatternHit(
        "P42", "MULTI_SITE_SINGLE_SITE_GROUNDING", None,
        f"success on {len(sites)}-site task ({','.join(map(str, sites))}) "
        f"but visited only {len(hosts)} host(s): {sorted(hosts)}",
        is_scaffold=False,
    )]


def check_p43(steps: List[Dict], summary: Dict, config: Dict, mode: str) -> List[PatternHit]:
    """P43: page-embedded visual info, mode delivers no page screenshot.

    Closes the structural gap in P34, which gates on `config["image"]` being truthy
    and therefore only ever fires for tasks carrying a TASK-LEVEL reference image.
    Tasks whose visual information lives in the page (a post's own attached image)
    have an empty `image` field, so P34 never saw them — 64 reddit tasks, previously
    invisible to every Tier-1 rule.

    ⚠️ Naming and framing are deliberate. Five sub-agents proposed calling this
    "structurally unsolvable / guaranteed fail". A controlled dom→som comparison on
    exactly this task set (same AXTree substrate, ± the annotated screenshot) measured
    B0 +0.00pp / B1 +1.56pp / B2 +0.00pp — restoring the screenshot barely helps, so
    the tasks are hard for every representation rather than blocked by the missing
    image (笔记 §387.10). This rule therefore emits a NEUTRAL (task × mode) label and
    must not be read as a predicted failure.

    Also relevant: reference images ARE delivered in every mode
    (`runner/main.py:2628-2631`), so "phantom == no visual input" is false; only the
    page screenshot is withheld.
    """
    if summary.get("success"):
        return []
    if config.get("image"):
        return []                      # task-level reference image present → P34's domain
    if mode not in ("dom", "phantom_text", "phantom_prompt", "phantom_som"):
        return []                      # som / vision do deliver a page screenshot
    if not VISUAL_INTENT_RE.search(str(config.get("intent") or "")):
        return []
    if any(int(((s.get("tokens") or {}).get("input_image")) or 0) for s in steps):
        return []                      # some image did reach the model → not blind
    return [PatternHit(
        "P43", "PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT", None,
        f"visual-intent task with no reference image, mode={mode} withholds the page "
        f"screenshot, input_image=0 across {len(steps)} steps (neutral label — "
        f"§387.10 measured ~0pp gain from restoring the screenshot)",
        is_scaffold=False,
    )]


def check_p44(steps: List[Dict], summary: Dict, _config: Dict, _mode: str) -> List[PatternHit]:
    """P44: hallucinated element reference — element_id absent from obs_nodes_info.

    The locator's second error branch, orthogonal to `walk_fail:*`. `walk_fail` means
    the element resolved but had no actionable ancestor; `missing union_bound` means
    the referenced id was not in the observation at all.

    Four Tier-2 sub-agents independently asserted this branch "never occurs" and
    concluded walk_fail can never indicate a hallucinated reference. That held in
    their 6-8 episode samples and is false in the population (笔记 §387.12):

        hallucinated-reference rate over action-steps
                     P-SoM     dom      SoM
        B0 (235B)     0.04%   0.39%    0.08%
        B1 (Qwen 4B)  0.12%   2.98%    0.45%
        B2 (Gemma 4B) 7.84%  18.21%    8.84%

    Monotone in model capability across all three modes, and dom is the worst mode for
    every model — consistent with dom asking the model to copy sparse 5-6 digit native
    AXTree ids (median 7839-18729) where [SOM_MARKS] uses compact 1..N (median 15-17).
    No prior rule covered this at all.
    """
    if summary.get("success"):
        return []
    hits = []
    seen: Set[Tuple[Optional[int], str]] = set()
    for s in steps:
        if _action_type(s) not in ("click", "type", "select_option"):
            continue
        for err in _locator_errors(s):
            if not MISSING_UNION_BOUND_RE.search(err):
                continue
            key = (s.get("step_idx"), err)
            if key in seen:
                continue
            seen.add(key)
            eid = _action_element_id(s)
            hits.append(PatternHit(
                "P44", "HALLUCINATED_ELEMENT_REF", s.get("step_idx"),
                f"{_action_type(s)} element_id={eid} not in obs_nodes_info ({err})",
                is_scaffold=False,
            ))
    return hits


def check_p45(steps: List[Dict], summary: Dict, _config: Dict, _mode: str) -> List[PatternHit]:
    """P45: identical failed action streak — same (action_type, element_id) ≥3× failing.

    Converged proposal from 4 sub-agents (B2 dom/ptext/psom/pprompt). The existing P36
    counts walk_fail step-wise without regard to consecutiveness, so a one-off locator
    miss and a 29-step deadlock look the same in aggregate. Observed deadlocks ran
    27-30 consecutive repeats of one failing (action_type, element_id) pair, consuming
    90-100% of the step budget, while the model kept receiving explicit FAILED feedback
    in its 8-step history window.

    Fires per streak (not per step) so one episode contributes one hit per deadlock.
    Threshold 3 is the point at which the history window has already shown the failure.
    """
    if summary.get("success"):
        return []
    hits = []
    run_key = None
    run_len = 0
    run_start = None
    for s in steps:
        at = _action_type(s)
        eid = _action_element_id(s)
        failed = (s.get("action_success") is not True) or bool(_locator_errors(s))
        key = (at, eid) if at in ("click", "type", "select_option") and eid is not None else None
        if key is not None and failed and key == run_key:
            run_len += 1
        else:
            if run_key is not None and run_len >= 3:
                hits.append(PatternHit(
                    "P45", "IDENTICAL_FAILED_ACTION_STREAK", run_start,
                    f"{run_key[0]} element_id={run_key[1]} repeated {run_len}x consecutively, all failing",
                    is_scaffold=False,
                ))
            run_key = key if (key is not None and failed) else None
            run_len = 1 if run_key is not None else 0
            run_start = s.get("step_idx") if run_key is not None else None
    if run_key is not None and run_len >= 3:
        hits.append(PatternHit(
            "P45", "IDENTICAL_FAILED_ACTION_STREAK", run_start,
            f"{run_key[0]} element_id={run_key[1]} repeated {run_len}x consecutively, all failing",
            is_scaffold=False,
        ))
    return hits


def check_p46(steps: List[Dict], summary: Dict, config: Dict, _mode: str) -> List[PatternHit]:
    """P46: comment/reply intent never committed text — action-modality mismatch.

    The agent treats "leave a comment saying X" as a question-answering task: it puts
    the answer in `finish(answer=...)` and never issues a `type`, so the evaluator
    (which reads the site's comment content) finds nothing. Observed with the answer
    semantically CORRECT and still scored 0 (B1_vision task 103's answer matched the
    reference verbatim), which is what separates this from a perception error.

    Scope: §387.8 measured comment/reply-intent tasks at 2.11% SR pooled over 18 cells
    vs 8.49% for the rest (4.0x), direction consistent in 18/18 cells. The intent
    regex is intentionally narrow — broadening it to any mutation verb collapses the
    gap (7.23% vs 6.01%), so this is about comment/reply specifically, not mutation.
    """
    if summary.get("success"):
        return []
    if not COMMENT_INTENT_RE.search(str(config.get("intent") or "")):
        return []
    if any(_action_type(s) == "type" and s.get("action_success") is True for s in steps):
        return []                      # it did commit text somewhere
    finish = _find_finish_step(steps)
    if finish is None:
        return []                      # ran out of budget instead — P31's domain
    answer = _finish_answer(steps)
    if not answer:
        return []
    return [PatternHit(
        "P46", "COMMENT_INTENT_NO_TYPE", finish.get("step_idx"),
        f"comment/reply intent finished with an answer but zero successful type actions: {answer[:70]}",
        is_scaffold=False,
    )]


# --- WA reddit Tier-2/3 batch P47-P48 (2026-08-02, ruleset 8-* -> 9-wa-*) ---
# Both were validated success-safe over the full 624-episode WA cell BEFORE landing: R1 fired on
# 24 failed / 0 success, R3 on 9 / 0. Two sibling candidates from the same round were rejected on
# exactly this test and are NOT here: R2 (`NOELEM_ACTION_STREAK`) fired on 15 successes = 17% of
# them, and R4 (`FORUM_NEVER_VISITED`) on 31 = 36% — R4 being the one the sub-agent rated first.
# Presence-only rules look strongest right up until the success side is checked.

# Postmill routes where a form is still open: new submission, new forum, edit.
_FORM_URL_RE = re.compile(r"/submit/|/create_forum|/-/edit", re.I)
# "there are no results" / "no submissions found" / "nothing was found" family.
_NEGATIVE_FINISH_RE = re.compile(
    r"\bno\s+(results?|submissions?|posts?|comments?|matches?)\b|"
    r"\bnothing\s+(was\s+)?found\b|\bcould\s+not\s+find\b|\bnone\s+found\b", re.I)


def check_p47(steps: List[Dict], summary: Dict, config: Dict, _mode: str) -> List[PatternHit]:
    """P47: typed into a form, then finished without submitting it.

    The last non-finish action is a `type`, no `click` follows it, and the page is still on a
    form route when `finish` is issued. The text was entered and never committed, so the
    evaluator reads a site that never changed. This is an agent-limit rather than a scaffold
    failure: the type itself succeeded and the submit control was present.

    Distinct from P46, which is about comment intents answered in `finish(answer=...)` with no
    `type` at all. Here the type happened; the submit did not.

    Firing rate on all three cells (mandatory before landing — a zero can be a site gate):
      VWA classifieds  0.00%  <- SITE GATE, not a measurement. osclass is not Postmill, so
                               /submit/ does not exist there: 0 of 224 episodes ever reach a
                               form URL. Never read this zero as evidence.
      VWA reddit       0.00%  <- a real measurement. Same Postmill app, 12 of 205 episodes DO
                               reach a form URL, and none of them finishes on one.
      WA reddit        3.70%  (22 of 594 failed episodes, 0 of 106 successes)

    Count note: the Tier-3 digest reported 24. That figure is the base condition alone
    (finish while on a form URL); adding the "last non-finish action is a type" clause the
    digest's own prose specifies brings it to 22. The narrower one is used because it is the
    stated mechanism — text entered, never committed. The third clause (no click after that
    type) filters nothing on current data and is kept only as a guard for future runs.
    """
    if summary.get("success"):
        return []
    finish = _find_finish_step(steps)
    if finish is None:
        return []
    if not _FORM_URL_RE.search(str(finish.get("obs_url") or "")):
        return []
    before = [s for s in steps if s is not finish and _action_type(s) != "finish"]
    if not before:
        return []
    if _action_type(before[-1]) != "type":
        return []
    # no click between that type and the finish
    idx = steps.index(before[-1])
    if any(_action_type(s) == "click" for s in steps[idx + 1:]):
        return []
    return [PatternHit(
        "P47", "PREMATURE_FINISH_ON_FORM", finish.get("step_idx"),
        f"typed into a form then finished without submitting; url still "
        f"{str(finish.get('obs_url') or '')[:70]}",
        is_scaffold=False,
    )]


def check_p49(steps: List[Dict], summary: Dict, _config: Dict, _mode: str) -> List[PatternHit]:
    """P49: on a /submit/ form page, repeatedly clicked an anchor (<A>) instead of the
    form's own submit BUTTON, and the episode never left /submit/.

    The site's top navbar carries a "Submit" LINK (`target_tag='A'`, bbox y=0) that sits
    next to nothing in the AXTree to distinguish it from the form's real submit control.
    Clicking it reloads a blank /submit/<forum> form, silently discarding whatever was
    typed — the model then re-types and re-clicks, producing a self-reinforcing loop.
    Confirmed as the death cause in causal verification of WA som task 610/614 and by
    the dom-B/psom-B Tier-2 batches.

    Gate design (2026-08-03, success-safe verified on all 48 conditions):
      >=2 anchor clicks while on /submit/   AND   final obs_url still under /submit/

    The terminal condition is what makes it clean: dropping it yields 106 failed / 14
    success hits (11.7% false-alarm); adding it yields **71 failed / 0 success**.
    Raising the click threshold instead does NOT help (>=3 → 47/0, >=5 → 18/0) — the
    win comes from the outcome gate, not from a stricter process threshold.

    Relation to existing rules: nearly disjoint from P47 (which REQUIRES no click after
    the last type) — overlap is 2 episodes. 55 of the 71 also carry P31, so the value is
    not extra coverage (net new vs P47+P31 is 14) but **attribution quality**: it turns
    "budget exhausted" (a verified risk-marker with no explanatory content) into a named
    mechanism. vision mode never fires it — its clicks carry no `locator_route_meta`.
    """
    if summary.get("success"):
        return []
    n_anchor = 0
    first_idx = None
    last_url = ""
    for s in steps:
        u = s.get("obs_url") or (s.get("state_digest") or {}).get("url_after") or ""
        if u:
            last_url = str(u)
        if s.get("action_type") != "click":
            continue
        url_before = s.get("obs_url") or (s.get("state_digest") or {}).get("url_before") or ""
        if "/submit/" not in str(url_before):
            continue
        for key in ("locator_route_meta", "locator_route_meta_primary"):
            meta = s.get(key)
            if isinstance(meta, dict) and str(meta.get("target_tag", "") or "").upper() == "A":
                n_anchor += 1
                if first_idx is None:
                    first_idx = s.get("step_idx")
                break
    if n_anchor >= 2 and "/submit/" in last_url:
        return [PatternHit(
            "P49", "SUBMIT_PAGE_ANCHOR_MISCLICK", first_idx,
            f"{n_anchor} clicks on anchor(<A>) elements while on /submit/; "
            f"episode ended still on {last_url[:70]}",
            is_scaffold=False,
        )]
    return []


def check_p48(steps: List[Dict], summary: Dict, config: Dict, _mode: str) -> List[PatternHit]:
    """P48: declared "no results" after a single search, in four steps or fewer.

    The agent searched once, read the first page, and finished asserting absence. The
    counter-example that motivates the rule is a matched pair in the same cell: on the same
    user and the same site version, one episode reached `/user/<name>/submissions` and found
    real posts while another asserted there were none.

    Deliberately narrow — coverage is ~1.7% of failures. Widening the step bound pulls in
    episodes that did search several ways before concluding absence, which is not this failure.

    Firing rate on all three cells (mandatory before landing):
      VWA classifieds  0.00%  (0 of 3621 failed)
      VWA reddit       0.29%  (10 of 3434 failed)
      WA reddit        0.00%  (0 of 594 failed) — the cell it was designed on, see below

    Success side: one success episode matches the pattern, `reddit_task_160` on B0/Vision. It is
    NOT a counter-example: task 160 is outside the scored universe under AMENDMENT_08, and its
    eval is `must_exclude`-only, i.e. passive-satisfiable — the "success" is the known false
    positive catalogued in `reddit_sidebar_leakage_audit.md`. Inside the scored universe the
    success-side hit count is 0.

    ⚠️ **It does not cover the episodes it was proposed from.** Measured on the WA cell: of 537
    failed episodes, 379 searched and 5 finished with a negative assertion — and all 5 run to
    8, 15 and 28 steps, so the four-step bound excludes every one of them. The Tier-3 digest
    reported 9 for this candidate; the landed regex finds 5, so its phrasing set was wider too.
    The rule is kept because the mechanism is real and the success side is clean, but its
    coverage belongs to VWA reddit (10 hits) and NOT to WA (0). Anyone widening the step bound
    to recover the motivating episodes must re-run the success-safe check first — that bound is
    the only thing currently separating this from "the agent searched and was right".
    """
    if summary.get("success"):
        return []
    if len(steps) > 4:
        return []
    if not any("/search?q=" in str(s.get("obs_url") or "") for s in steps):
        return []
    answer = _finish_intent_text(steps)   # v11: intent wording, thought fallback OK
    if not answer or not _NEGATIVE_FINISH_RE.search(answer):
        return []
    finish = _find_finish_step(steps)
    return [PatternHit(
        "P48", "PREMATURE_NEGATIVE_AFTER_SEARCH",
        (finish or steps[-1]).get("step_idx"),
        f"asserted absence after one search in {len(steps)} steps: {answer[:70]}",
        is_scaffold=False,
    )]


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
    "P33": check_p33,  # 导航至裸图片URL幻觉 (phantom_som SOM_MARKS img-href click; R32031)
    "P34": check_p34,
    "P35": check_p35,
    "P36": check_p36,
    "P37": check_p37,
    "P38": check_p38,
    "P39": check_p39,
    "P40": check_p40,
    # reddit discover batch 2026-07-27 (ruleset 8-reddit-*)
    "P41": check_p41,   # success-side benchmark-FP: must_exclude-only eval (B-1889)
    "P42": check_p42,   # success-side benchmark-FP: multi-site answered from 1 site (B-1892)
    "P43": check_p43,   # neutral label: page-embedded visual info, mode has no screenshot
    "P44": check_p44,   # hallucinated element ref (missing union_bound) — was uncovered
    "P45": check_p45,   # identical failed action streak >=3 (P36 consecutiveness)
    "P46": check_p46,   # comment/reply intent never committed text
    "P47": check_p47,   # typed into a form then finished without submitting
    "P48": check_p48,   # declared "no results" after a single search
    "P49": check_p49,   # /submit/ page: anchor(<A>) misclick loop, never left form
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

class MissingTaskConfigError(RuntimeError):
    """Raised when episodes have no task config and the caller did not opt in.

    B-1919 (2026-08-02): this used to be a silent `config = {}` fallback. Every
    rule that reads the task config then returns [] and the scan looks CLEAN —
    a scan whose config-dependent rules were never actually evaluated is
    indistinguishable from a scan that found nothing. That is how six WA reddit
    conditions sat on disk for a fortnight reporting "0 success-side hits" while
    28 of 44 rules were switched off. Fail loud instead.
    """


def _config_dependent_rules() -> List[str]:
    """Rule IDs whose check function reads `config` — i.e. what goes dark.

    Derived from the source at call time rather than hardcoded so it cannot
    drift as rules are added. Only ever invoked on the error path.
    """
    import inspect
    out = []
    for rule_id, fn in ALL_RULES.items():
        try:
            if re.search(r"\bconfig\b", inspect.getsource(fn)):
                out.append(rule_id)
        except (OSError, TypeError):      # source unavailable — assume affected
            out.append(rule_id)
    return sorted(out, key=lambda r: int(r[1:]))


def scan_episodes(
    run_dir: Path,
    *,
    condition_filter: Optional[str] = None,
    task_filter: Optional[int] = None,
    failed_only: bool = False,
    rule_filter: Optional[Set[str]] = None,
    verbose: bool = False,
    allow_missing_config: bool = False,
) -> Dict[str, Any]:
    """Scan all episodes and return structured results.

    Raises MissingTaskConfigError if any episode lacks its task config, unless
    `allow_missing_config` is set — in which case the count still lands in the
    returned dict under `config_missing` so downstream consumers can see that
    the config-dependent rules were not really evaluated.
    """
    episodes = _discover_episodes(run_dir, condition_filter, task_filter)
    if not episodes:
        print(f"No episodes found in {run_dir}", file=sys.stderr)
        return {}

    rules_to_run = {k: v for k, v in ALL_RULES.items()
                    if rule_filter is None or k in rule_filter}

    all_diagnoses: List[Dict] = []
    hit_counts: Dict[str, int] = {r: 0 for r in rules_to_run}
    run_id_cache: Optional[str] = None
    missing_config_stems: List[str] = []      # B-1919

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
        else:
            # B-1919: remember, don't shrug. Reported after the loop.
            missing_config_stems.append(config_path.stem)

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

    # B-1919: a scan missing its task configs is not a clean scan, it is a scan
    # with most of its rules switched off. Say so, and refuse to hand back a
    # result that looks authoritative unless the caller explicitly opted in.
    if missing_config_stems:
        affected = _config_dependent_rules()
        msg = (
            # `total` already counts every episode, including the config-less ones
            f"{len(missing_config_stems)}/{total} episodes have no "
            f"task config under {run_dir / 'task_configs'} — "
            f"{len(affected)} of {len(ALL_RULES)} rules read the task config and would "
            f"silently return no hits: {', '.join(affected)}. "
            f"First missing: {', '.join(missing_config_stems[:5])}"
            + (" ..." if len(missing_config_stems) > 5 else "")
        )
        if not allow_missing_config:
            raise MissingTaskConfigError(
                msg + ". Restore task_configs/ (the runner writes them via "
                "p79.experiment.tasks.load_tasks; on a synced mirror they may simply "
                "not have been transferred) or pass --allow-missing-config to accept "
                "a partial scan."
            )
        print(f"WARNING [B-1919]: {msg}", file=sys.stderr)

    run_id = run_id_cache or run_dir.name

    result = {
        "run_id": run_id,
        "ruleset_version": RULESET_VERSION,
        "scan_time": datetime.now(timezone.utc).isoformat(),
        "rules_applied": sorted(rules_to_run.keys()),
        "total_episodes": total,
        "episodes_with_hits": with_hits,
        # B-1919: 0 on a healthy scan. Non-zero means the config-dependent rules
        # were not really evaluated for that many episodes — the number travels
        # with the artifact so a downstream reader can tell.
        "config_missing": len(missing_config_stems),
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
    parser.add_argument("--allow-missing-config", action="store_true",
                        help="B-1919: accept episodes with no task config. Off by default — "
                             "without the config the 27-odd config-reading rules silently "
                             "return no hits and the scan looks clean. The count still lands "
                             "in the output JSON as `config_missing`.")

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

    try:
        result = scan_episodes(
            args.run_dir,
            condition_filter=args.condition,
            task_filter=args.task_id,
            failed_only=args.failed_only,
            rule_filter=rule_filter,
            verbose=args.verbose,
            allow_missing_config=args.allow_missing_config,
        )
    except MissingTaskConfigError as exc:
        print(f"Error [B-1919]: {exc}", file=sys.stderr)
        sys.exit(2)

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
