"""Single source of truth for L1 / learned-router feature & oracle definitions.

router /stress 2026-05-21 (B-1805 F1 + B-1806 F2 + B-1807 F6): consolidates the
difficulty parsing, observation-mode cost ordering, intent-regex banks, and
oracle-label derivation that were previously copy-pasted across:

  - scripts/analysis/extract_50_features.py   (Stage 1, train-time labels)
  - p79/policies/learned_router.py            (Pass-2 serve-time features)
  - scripts/analysis/l1_archive_simulation.py (archive simulation, §6)
  - (+ three diagnostic scripts, F6-followup)

Copy-paste drift between train and serve silently skews features (the F4
token-skew lesson) and biases oracle labels (F2 cost-order); routing the four
definitions through one import point makes train ≡ serve ≡ archive *by
construction* rather than by a "must match" comment that nobody re-checks.

Deliberately depends only on `re` + `typing` (no numpy / sklearn) so both the
analysis scripts and the runtime predictor can import it cheaply.
"""
from __future__ import annotations

import re
from typing import Optional

# ── F1 (B-1805): difficulty parsing — VWA stores ordinal STRINGS ──────────────
# VWA task configs store reasoning/visual/overall_difficulty as the ordinal
# strings "easy"/"medium"/"hard" (verified 234/234 cls configs 2026-05-21), NOT
# ints. `int("medium")` raises ValueError: the train path (extract_50_features
# :221) had no guard and crashed Stage 1 on the first labeled task; the serve
# path (runner main.py:2242) wrapped it in `except: pass` and silently zeroed the
# feature → train uses the real difficulty, serve always sees 0 = train/serve
# skew. Both sides now share this map so the feature is identical end to end.
DIFFICULTY_MAP = {"easy": 0, "medium": 1, "hard": 2}


def difficulty_to_int(raw: object, default: int = 0) -> int:
    """Parse a VWA difficulty annotation to an ordinal int — train ≡ serve safe.

    Accepts the ordinal strings ("easy"/"medium"/"hard"), numeric strings, ints
    (passed through), and missing/None (→ default). Unknown strings → default;
    difficulty is a soft feature, not worth a hard-fail (unlike artifact/universe
    errors), but the divergence is bounded to `default` rather than a crash.
    """
    if raw is None:
        return default
    if isinstance(raw, bool):  # guard: bool is an int subclass
        return default
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str):
        key = raw.strip().lower()
        if key in DIFFICULTY_MAP:
            return DIFFICULTY_MAP[key]
        if key.lstrip("-").isdigit():
            return int(key)
    return default


# ── F4 (B-1817): input-token estimate — train ≡ serve ─────────────────────────
def estimate_input_tokens(text_length: int) -> int:
    """Estimate input token count from char length (chars // 4).

    Used IDENTICALLY by train (extract_50_features step-0) and serve (runner). The real
    tokenizer count is unavailable at the serve dispatch point and is None for B0 at
    train time, so using the real count gave train=0 / serve=len//4 = train/serve skew
    (F4). A shared estimate makes the feature consistent end to end (consistency >
    accuracy for a routing feature).
    """
    return int(text_length) // 4


# ── F2 (B-1806): observation modes in ASCENDING prior-cost order ──────────────
# `derive_oracle_label` tie-breaks by picking the FIRST successful mode, so this
# list MUST be ordered cheapest-first for the "cheapest successful mode" label
# semantics to hold. Cost structure (paper §3 phantom drop-in property): the four
# text-only modes — dom + the three phantom arms ([SOM_MARKS] is a *flattened*
# AXTree, ≈1.00× chars, NO annotated image) — cost ≈ DOM; the two image-bearing
# modes (som = annotated screenshot, vision = raw screenshot) carry image-token
# cost per call. The previous order
# ["dom","som","vision","phantom_text","phantom_prompt","phantom_som"] ranked the
# two image modes ahead of the cheap phantoms and buried the
# deployment-representative HERO (phantom_som) last → labels biased toward
# expensive modes, undercutting the paper's cost argument (F2).
#
# F2 TODO RESOLVED (2026-06-09, /stress Mode A Q4 verify on landed B0+B1 cls,
# 12 conditions × 224 ep): measured episode-realized `total_billed_cost_usd`
# does NOT reproduce this prior order — success-only per-mode means are
# step-count-dominated and CELL-INVERTED (B0 cls cheapest→dearest:
# vision .033 < som .035 < psom .040 < pprompt .045 < ptext .046 < dom .051;
# B1 cls: dom .018 < vision .019 < ptext .023 < pprompt .032 < psom .047 <
# som .050; n_succ only 14-61 per mode). A measured-cost tie-break is therefore
# NOT viable: (a) the measured order is unstable across cells (B0/B1 inverted),
# (b) episode cost is endogenous to behavior (success path length), inviting a
# cost←outcome circularity into the label definition, (c) n_succ is too small
# to pin an order. DECISION: keep this PRIOR order (input-payload/per-call
# view: text-only modes carry no image tokens — the §3 drop-in (a) layer),
# unchanged as the locked tie-break; tie-break sensitivity is bounded by the
# multi-success task count (G1 `oracle_provenance.n_multi_success`) and is
# §6-disclosable from the same data if a reviewer asks. Do NOT silently switch
# to a measured tie-break — that is an oracle-label estimand change.
MODES = ["dom", "phantom_som", "phantom_text", "phantom_prompt", "som", "vision"]

# Coarse cost tiers (text-only vs image), for callers that want the grouping
# without depending on the exact intra-tier order. Lower = cheaper.
MODE_COST_TIER = {
    "dom": 0,
    "phantom_som": 0,
    "phantom_text": 0,
    "phantom_prompt": 0,
    "som": 1,
    "vision": 1,
}


# ── F6 (B-1807): single intent-regex bank (14 mechanism-anchored banks) ────────
# Previously duplicated verbatim in extract_50_features.py:51 and
# learned_router.py:58 (+ a 4-regex subset COLOR_RE/SEARCH_RE/COMPARE_RE/NAV_RE
# in learned_router and l1_archive_simulation). has_reference_image is the 15th
# binary feature, computed separately from the task config (not a regex).
INTENT_REGEX = {
    "intent_color": re.compile(
        r"\b(color|red|blue|green|yellow|black|white|orange|purple|pink|brown|gray|grey)\b",
        re.IGNORECASE,
    ),
    "intent_search": re.compile(r"\b(find|search|locate|how many|how much)\b", re.IGNORECASE),
    "intent_compare": re.compile(
        r"\b(cheapest|most expensive|highest|lowest|best|worst|biggest|smallest)\b",
        re.IGNORECASE,
    ),
    "intent_nav": re.compile(r"\b(go to|navigate|open|visit)\b", re.IGNORECASE),
    "intent_filter": re.compile(r"\b(filter|narrow|restrict|limit to|only)\b", re.IGNORECASE),
    "intent_sort": re.compile(r"\b(sort|rank|order by|by date|by price|newest|oldest)\b", re.IGNORECASE),
    "intent_aggregate": re.compile(r"\b(total|sum|average|count of|number of)\b", re.IGNORECASE),
    "intent_compose": re.compile(r"\b(compose|write|post|submit|reply|comment)\b", re.IGNORECASE),
    "intent_form_fill": re.compile(r"\b(fill|enter|type|input)\b", re.IGNORECASE),
    "intent_account_action": re.compile(
        r"\b(login|logout|account|profile|sign in|sign out|subscribe|unsubscribe)\b",
        re.IGNORECASE,
    ),
    "intent_visual_attribute": re.compile(
        r"\b(size|shape|appear|look|tall|wide|small|large|height|width)\b", re.IGNORECASE
    ),
    "intent_question": re.compile(r"\b(what|where|when|why|how)\b|\?", re.IGNORECASE),
    "intent_action_word": re.compile(r"\b(click|select|choose|press|tap)\b", re.IGNORECASE),
    "intent_temporal": re.compile(
        r"\b(today|yesterday|recent|latest|first|newest|oldest|2024|2025|2026)\b",
        re.IGNORECASE,
    ),
}

# Back-compat single-regex aliases (the deprecated 8-dim archive path + diagnostics
# import these). Kept pointing at the same compiled objects as INTENT_REGEX so the
# 8-dim path can never drift from the 15-dim bank.
COLOR_RE = INTENT_REGEX["intent_color"]
SEARCH_RE = INTENT_REGEX["intent_search"]
COMPARE_RE = INTENT_REGEX["intent_compare"]
NAV_RE = INTENT_REGEX["intent_nav"]


def compute_intent_binaries(intent: Optional[str]) -> dict[str, int]:
    """Apply the 14 intent-regex banks → {name: 0/1}. Order-independent (dict)."""
    return {
        name: int(bool(pattern.search(intent or "")))
        for name, pattern in INTENT_REGEX.items()
    }


def derive_oracle_label(outcomes: dict[str, bool]) -> Optional[str]:
    """Pick the oracle-best mode for a task: cheapest successful mode.

    Tie-break = MODES priority order, which is ascending prior cost (F2). Returns
    None if NO mode succeeded — B-995: do NOT fall back to "dom", which would
    collapse the label semantics (a no-success task is not a "dom is best" task).
    Callers filter None out of the *trainable* rows but must still route the task
    at serve time (see C1 universe-vs-trainable separation).
    """
    for m in MODES:
        if outcomes.get(m, False):
            return m
    return None
