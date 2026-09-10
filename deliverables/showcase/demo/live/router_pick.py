"""The learned router's pick for a task typed at the board.

The replay tabs show the router's fold-held-out pick for recorded tasks. A typed task
belongs to no fold, so all five fold models vote (each with its own threshold, exactly
as in `p79.policies.learned_router.predict_mode_fold_aware`: argmax if its top
probability clears the fold's τ, else the safe fallback view) and the majority is
reported together with how many of the five agreed.

Features are the router's own six (`extract_raw_features`), taken from the live run:
the typed intent, and the READ lane's first page (`state_digest` dom_complexity /
text_length — the same step-0 source the offline replay reads). One feature has no
live value: `reasoning_difficulty` is a human annotation on benchmark tasks. It is set
to the median of the 234 classifieds tasks (1 = medium), and the page says so.

There is no answer key for a typed task, so this module never says whether the pick
was right; the visitor judges each lane's answer on the page, and the ring colour on
the picked lane is that judgement.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from p79.policies.learned_router import (
    SAFE_FALLBACK_MODE,
    build_runtime_feature_vector,
    extract_raw_features,
    load_cell_meta,
    load_lr_pipeline_fold,
    load_selected_idx_fold,
    load_vectorizer_fold,
)
from p79.policies.router_features import estimate_input_tokens

REPO = Path(__file__).resolve().parents[4]
ARTIFACTS = REPO / "results/phantom_paper/l1_router_offline_20260715"
CELL = "B0_classifieds"
FOLDS = range(5)
# median reasoning_difficulty over the 234 classifieds tasks (easy 0: 68, medium 1: 76, hard 2: 90)
DIFFICULTY_IMPUTED = 1

MODE_LANE = {"vision": "LOOK", "dom": "READ", "som": "BOTH"}
OTHER_VIEW = {
    "phantom_prompt": "the text tree, read with the marked-screenshot instructions",
    "phantom_text": "the numbered element list, read with the plain-text instructions",
    "phantom_som": "the numbered element list, read with the marked-screenshot instructions",
}

_FOLD_CACHE: dict[int, tuple] = {}


def _fold(k: int) -> tuple:
    if k not in _FOLD_CACHE:
        mask, _ = load_selected_idx_fold(ARTIFACTS, k)
        parts = (load_vectorizer_fold(ARTIFACTS, k), mask, load_lr_pipeline_fold(ARTIFACTS, CELL, k))
        if any(p is None for p in parts):
            raise RuntimeError(f"router artifacts for fold {k} missing under {ARTIFACTS}")
        _FOLD_CACHE[k] = parts
    return _FOLD_CACHE[k]


def live_pick(intent: str, step0: dict) -> dict:
    digest = step0.get("state_digest") or {}
    text_length = int(digest.get("text_length", 0) or 0)
    raw = extract_raw_features(
        intent=intent,
        has_reference_image=False,
        dom_complexity=int(digest.get("dom_complexity", 0) or 0),
        text_length=text_length,
        tokens_input_text=estimate_input_tokens(text_length),
        reasoning_difficulty=DIFFICULTY_IMPUTED,
    )
    taus = load_cell_meta(ARTIFACTS, CELL)["thresholds_per_fold"]
    per_fold = []
    for k in FOLDS:
        vec, mask, pipe = _fold(k)
        probs = pipe.predict_proba(build_runtime_feature_vector(raw, vec, mask).reshape(1, -1))[0]
        tau = float(taus.get(str(k), taus.get(k)))
        top = float(probs.max())
        mode = str(pipe.classes_[int(probs.argmax())]) if top > tau else SAFE_FALLBACK_MODE
        per_fold.append({"fold": k, "mode": mode, "max_prob": round(top, 3), "tau": tau,
                         "fallback": top <= tau})
    votes = Counter(p["mode"] for p in per_fold)
    mode, agree = votes.most_common(1)[0]
    out = {
        "mode": mode, "agree": agree, "folds": len(per_fold), "votes": dict(votes),
        "per_fold": per_fold, "lane": MODE_LANE.get(mode),
        "fallback": mode == SAFE_FALLBACK_MODE and all(
            p["fallback"] for p in per_fold if p["mode"] == mode),
        "imputed": {"reasoning_difficulty": DIFFICULTY_IMPUTED},
    }
    if out["lane"] is None:
        out["name"] = OTHER_VIEW.get(mode, mode)
    return out
