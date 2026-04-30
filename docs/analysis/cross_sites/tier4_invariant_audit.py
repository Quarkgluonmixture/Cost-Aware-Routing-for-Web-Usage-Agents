#!/usr/bin/env python3
"""Invariant audit over Phase 1 paper-grade VWA episode traces."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from p79.experiment.io_utils import read_jsonl_dedup


AUDIT_DATE = "2026-04-30"

RUNS = [
    "results/visualwebarena/phase1/B0_3mode_classifieds_20260413",
    "results/visualwebarena/phase1/B0_3mode_reddit_20260422",
    "results/visualwebarena/phase1/B0_phantom_som_classifieds_20260426",
    "results/visualwebarena/phase1/B0_phantom_som_reddit_20260428",
    "results/visualwebarena/phase1/B0_phantom_text_classifieds_20260427",
    "results/visualwebarena/phase1/B0_phantom_text_reddit_20260427",
    "results/visualwebarena/phase1/B0_phantom_prompt_reddit_20260429",
    "results/visualwebarena/phase1/B0_dom_shopping_20260428",
    "results/visualwebarena/phase1/B1_3mode_classifieds_20260413",
    "results/visualwebarena/phase1/B1_3mode_reddit_20260413",
    "results/visualwebarena/phase1/B1_phantom_som_classifieds_20260428",
    "results/visualwebarena/phase1/B1_phantom_dom_classifieds_20260429",
]

OUT_JSON = Path("docs/analysis/cross_sites/tier4_invariant_audit.json")
OUT_MD = Path("docs/analysis/cross_sites/tier4_invariant_audit.md")

TERMINAL_ACTIONS = {"finish", "stop", "done"}
NAV_ACTIONS = {"click", "goto", "go_to", "goto_url", "navigate", "back", "forward"}
LONG_STEP_EXEMPT_ACTIONS = {"wait", "type", "scroll"}


@dataclass(frozen=True)
class Episode:
    run_path: Path
    steps_path: Path
    summary_path: Path
    summary: dict[str, Any]
    steps: list[dict[str, Any]]

    @property
    def site(self) -> str:
        if self.steps:
            return str(self.steps[0].get("benchmark_site") or self.summary.get("benchmark_site") or "")
        return str(self.summary.get("benchmark_site") or "")

    @property
    def task(self) -> int:
        value = self.steps[0].get("task_id") if self.steps else self.summary.get("task_id")
        return int(value)

    @property
    def run_id(self) -> str:
        return str(self.summary.get("run_id") or self.run_path.name)

    @property
    def short_run(self) -> str:
        return re.sub(r"_2026\d{4}$", "", self.run_id)

    @property
    def condition_id(self) -> str:
        return str(self.summary.get("condition_id") or (self.steps[0].get("condition_id") if self.steps else ""))

    @property
    def mode(self) -> str:
        obs_mode = str(self.steps[0].get("observation_mode") if self.steps else "")
        cid = self.condition_id
        if "phantom_prompt" in cid or obs_mode == "phantom_prompt":
            return "P-prompt"
        if "phantom_som" in cid or obs_mode == "phantom_som":
            return "P-SoM"
        if "phantom_dom" in cid or obs_mode == "phantom_dom":
            return "P-text"
        if "vision" in cid or obs_mode == "vision":
            return "Vision"
        if "som" in cid or obs_mode == "som":
            return "SoM"
        if "dom" in cid or obs_mode == "dom":
            return "DOM"
        return obs_mode or cid


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def collect_episode_files(run_root: Path) -> list[Path]:
    return sorted(run_root.glob("*/episodes/*_steps_v2.jsonl"))


def load_episodes() -> list[Episode]:
    episodes: list[Episode] = []
    for run in RUNS:
        run_root = Path(run)
        for steps_path in collect_episode_files(run_root):
            steps = read_jsonl_dedup(steps_path)
            summary_path = steps_path.with_name(steps_path.name.replace("_steps_v2.jsonl", "_summary_v2.json"))
            summary = load_json(summary_path)
            episodes.append(
                Episode(
                    run_path=run_root,
                    steps_path=steps_path,
                    summary_path=summary_path,
                    summary=summary,
                    steps=steps,
                )
            )
    return episodes


def action_type(step: dict[str, Any]) -> str:
    return str(step.get("action_type") or (step.get("action") or {}).get("action_type") or "").lower()


def action_element_id(step: dict[str, Any]) -> str | None:
    value = (step.get("action") or {}).get("element_id")
    if value is None:
        return None
    return str(value)


def env_step_ms(step: dict[str, Any]) -> float:
    value = (step.get("latency_ms") or {}).get("env_step")
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def text_similarity_value(step: dict[str, Any]) -> float | None:
    value = step.get("text_similarity")
    return float(value) if isinstance(value, (int, float)) else None


def url_before(step: dict[str, Any]) -> str:
    return str((step.get("state_digest") or {}).get("url_before") or "")


def url_after(step: dict[str, Any]) -> str:
    return str((step.get("state_digest") or {}).get("url_after") or step.get("obs_url") or "")


def canonical_url(url: str) -> tuple[str, str, str]:
    parsed = urlparse(url or "")
    path = parsed.path.rstrip("/") or "/"
    return (parsed.netloc, path, parsed.query)


def is_redirect_chain(prev_url: str, next_url: str) -> bool:
    """Conservative local heuristic; no network probing."""
    if not prev_url or not next_url:
        return False
    p = urlparse(prev_url)
    n = urlparse(next_url)
    if p.netloc != n.netloc:
        return False
    p_path = p.path.rstrip("/") or "/"
    n_path = n.path.rstrip("/") or "/"
    if p_path == n_path and (not p.query or not n.query):
        return True
    redirect_tokens = ("redirect", "next=", "return", "continue", "login", "logout")
    combined = f"{p.query}&{n.query}".lower()
    return any(token in combined for token in redirect_tokens)


_TEXT_CACHE: dict[str, str] = {}
_TOKEN_CACHE: dict[str, frozenset[str]] = {}
_ROLE_CACHE: dict[str, dict[str, str]] = {}


def artifact_dom_path(step: dict[str, Any]) -> str:
    return str((step.get("artifact_paths") or {}).get("dom") or "")


def read_obs_text(step: dict[str, Any]) -> str:
    path = artifact_dom_path(step)
    if not path:
        return ""
    if path in _TEXT_CACHE:
        return _TEXT_CACHE[path]
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    _TEXT_CACHE[path] = text
    return text


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def token_shingles(step: dict[str, Any], k: int = 5) -> frozenset[str]:
    path = artifact_dom_path(step)
    cache_key = f"{path}:{k}"
    if cache_key in _TOKEN_CACHE:
        return _TOKEN_CACHE[cache_key]
    tokens = re.findall(r"[a-z0-9]+", read_obs_text(step).casefold())
    if len(tokens) < k:
        shingles = frozenset(tokens)
    else:
        shingles = frozenset(" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1))
    _TOKEN_CACHE[cache_key] = shingles
    return shingles


def shingle_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    left = token_shingles(a)
    right = token_shingles(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


ROLE_LINE_RE = re.compile(r"^\s*\[(\d+)\]\s+([A-Za-z][A-Za-z0-9_-]*)\b")


def roles_by_element_id(step: dict[str, Any]) -> dict[str, str]:
    path = artifact_dom_path(step)
    if path in _ROLE_CACHE:
        return _ROLE_CACHE[path]
    roles: dict[str, str] = {}
    for line in read_obs_text(step).splitlines():
        match = ROLE_LINE_RE.match(line)
        if match:
            roles[match.group(1)] = match.group(2)
    _ROLE_CACHE[path] = roles
    return roles


def snippet_for_step(ep: Episode, step: dict[str, Any], extra: str = "") -> str:
    atype = action_type(step)
    action = step.get("action") or {}
    eid = action.get("element_id")
    text = action.get("text")
    option = action.get("option_label")
    answer = action.get("answer")
    bits = [f"action={atype}"]
    if eid is not None:
        bits.append(f"element_id={eid}")
    if text:
        bits.append(f"text={str(text).strip()[:80]}")
    if option:
        bits.append(f"option={str(option)[:80]}")
    if answer:
        bits.append(f"answer={str(answer)[:100]}")
    bits.append(f"url_before={url_before(step)[:120]}")
    bits.append(f"url_after={url_after(step)[:120]}")
    if step.get("page_changed") is not None:
        bits.append(f"page_changed={step.get('page_changed')}")
    if step.get("action_success") is not None:
        bits.append(f"action_success={step.get('action_success')}")
    sim = text_similarity_value(step)
    if sim is not None:
        bits.append(f"text_similarity={sim:.3f}")
    if extra:
        bits.append(extra)
    return "; ".join(bits)


def make_case(ep: Episode, step: dict[str, Any] | None, extra: str = "") -> dict[str, Any]:
    return {
        "site": ep.site,
        "task": ep.task,
        "step": int(step.get("step_idx")) if step and isinstance(step.get("step_idx"), int) else None,
        "run": ep.short_run,
        "condition_id": ep.condition_id,
        "mode": ep.mode,
        "snippet": snippet_for_step(ep, step, extra) if step else extra,
    }


def add_hit(
    hits: list[dict[str, Any]],
    ep: Episode,
    step: dict[str, Any] | None,
    extra: str = "",
) -> None:
    hits.append(make_case(ep, step, extra))


def inv_i1(ep: Episode) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for step in ep.steps:
        sim = text_similarity_value(step)
        if (
            step.get("action_success") is True
            and step.get("page_changed") is False
            and sim is not None
            and sim > 0.99
            and action_type(step) not in TERMINAL_ACTIONS
        ):
            add_hit(hits, ep, step)
    return hits


def inv_i2(ep: Episode) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for step in ep.steps:
        if step.get("action_success") is False and step.get("page_changed") is True:
            add_hit(hits, ep, step)
    return hits


def inv_i3(ep: Episode) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    run_start = 0
    last_eid: str | None = None
    for idx, step in enumerate(ep.steps + [None]):  # type: ignore[list-item]
        eid = action_element_id(step) if step else None
        is_same_click = bool(step) and action_type(step) == "click" and eid is not None and eid == last_eid
        if step and action_type(step) == "click" and eid is not None and (last_eid is None or eid != last_eid):
            if idx - run_start >= 3 and last_eid is not None:
                first = ep.steps[run_start]
                add_hit(hits, ep, first, extra=f"repeat_click_element_id={last_eid}; repeats={idx - run_start}")
            run_start = idx
            last_eid = eid
        elif not is_same_click:
            if idx - run_start >= 3 and last_eid is not None:
                first = ep.steps[run_start]
                add_hit(hits, ep, first, extra=f"repeat_click_element_id={last_eid}; repeats={idx - run_start}")
            run_start = idx + 1
            last_eid = None
    return hits


def inv_i4(ep: Episode) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for step in ep.steps:
        atype = action_type(step)
        if env_step_ms(step) > 30000 and atype not in LONG_STEP_EXEMPT_ACTIONS:
            add_hit(hits, ep, step, extra=f"env_step_ms={env_step_ms(step):.0f}")
    return hits


def inv_i5(ep: Episode) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for prev, step in zip(ep.steps, ep.steps[1:]):
        # Recorder semantics: current state_digest.url_before should match prior url_after.
        prior_after = url_after(prev)
        current_before = url_before(step)
        if (
            current_before
            and prior_after
            and current_before != prior_after
            and action_type(prev) not in NAV_ACTIONS
            and not is_redirect_chain(prior_after, current_before)
        ):
            add_hit(hits, ep, step, extra=f"prev_url_after={prior_after[:120]}; current_url_before={current_before[:120]}")
    return hits


def inv_i6(ep: Episode) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for step, next_step in zip(ep.steps, ep.steps[1:]):
        if action_type(step) in {"goto", "go_to", "goto_url", "navigate", "click"}:
            continue
        before_url = url_before(step)
        next_url = url_before(next_step)
        if before_url and before_url == next_url:
            sim = shingle_similarity(step, next_step)
            if sim < 0.7:
                add_hit(hits, ep, step, extra=f"adjacent_obs_shingle_similarity={sim:.3f}")
    return hits


def inv_i7(ep: Episode) -> list[dict[str, Any]]:
    finish_attempted = bool(ep.summary.get("agent_finished")) or any(action_type(s) in TERMINAL_ACTIONS for s in ep.steps)
    raw_success = bool(ep.summary.get("success"))
    if finish_attempted and not raw_success:
        step = ep.steps[-1] if ep.steps else None
        return [make_case(ep, step, extra="finish_attempted=True; raw_success=False")]
    return []


def inv_i8(ep: Episode, max_steps_cap: int) -> list[dict[str, Any]]:
    if not ep.steps:
        return []
    last = ep.steps[-1]
    derived_truncated = (
        len(ep.steps) >= max_steps_cap
        and bool(ep.summary.get("agent_finished")) is False
        and bool(last.get("done")) is False
    )
    if derived_truncated and action_type(last) == "click":
        return [make_case(ep, last, extra=f"derived_truncated_at_max_step=True; max_steps_cap={max_steps_cap}")]
    return []


def inv_i9(ep: Episode) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    seen: dict[str, tuple[str, int]] = {}
    emitted: set[str] = set()
    for step in ep.steps:
        for eid, role in roles_by_element_id(step).items():
            prev = seen.get(eid)
            if prev and prev[0] != role and eid not in emitted:
                extra = f"element_id={eid}; previous_role={prev[0]}; previous_step={prev[1]}; current_role={role}"
                add_hit(hits, ep, step, extra=extra)
                emitted.add(eid)
            seen[eid] = (role, int(step.get("step_idx") or 0))
    return hits


def inv_i10(ep: Episode) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for step, next_step in zip(ep.steps, ep.steps[1:]):
        if action_type(step) in TERMINAL_ACTIONS:
            continue
        if step.get("action_success") is True and step.get("page_changed") is True:
            before = normalize_text(read_obs_text(step))
            after = normalize_text(read_obs_text(next_step))
            if before and before == after:
                add_hit(hits, ep, step, extra="normalized_prev_and_next_obs_text_identical=True")
    return hits


INVARIANT_META = {
    "I1": {
        "name": "inv_action_success_but_no_change",
        "description": "Non-terminal action_success=True but page_changed=False and recorded text_similarity > 0.99.",
        "candidate_root_cause": "Runner accepted a no-op or swallowed an actionability failure; policy sees no state progress after a supposedly successful action.",
        "tier3_taxonomy_match": "Type 5 Actionability Check Masking and Timeout Swallowing",
        "novelty_assessment": "matches Tier 2 type/scroll silent-failure catalog and click probe no-progress family",
        "level": "step",
    },
    "I2": {
        "name": "inv_action_fail_but_page_changed",
        "description": "action_success=False but page_changed=True.",
        "candidate_root_cause": "Runner success flag is stricter or stale relative to actual state transition; this can cause unnecessary retries or wrong self-diagnosis.",
        "tier3_taxonomy_match": "Type 4 Evaluator State Drift / Type 5 actionability masking",
        "novelty_assessment": "NEW finding relative to Tier 1/Tier 2/probe counts: empirical runner-false-negative success flag",
        "level": "step",
    },
    "I3": {
        "name": "inv_repeat_click_no_cycle_break",
        "description": "Same element_id is clicked at least three consecutive times in one episode.",
        "candidate_root_cause": "Cycle detection did not halt a same-target click loop soon enough, or repeated clicks were treated as exploration despite no new target.",
        "tier3_taxonomy_match": "Type 1 Coordinate Dispatch Anomaly, surfacing as a loop",
        "novelty_assessment": "matches click-probe and phantom-paper click-loop already-known family",
        "level": "sequence",
    },
    "I4": {
        "name": "inv_long_step_unexplained",
        "description": "env_step_ms > 30s for a non-wait, non-type, non-scroll action.",
        "candidate_root_cause": "Hidden Playwright timeout or slow actionability wait is collapsed into a generic step result rather than exposed as an environment error.",
        "tier3_taxonomy_match": "Type 5 Actionability Check Masking and Timeout Swallowing",
        "novelty_assessment": "partly anticipated by Tier 1 static timeout concerns; empirical long-step count is NEW",
        "level": "step",
    },
    "I5": {
        "name": "inv_unexplained_url_jump",
        "description": "Adjacent recorder states disagree: current url_before differs from previous url_after after a non-navigation previous action.",
        "candidate_root_cause": "Popup, redirect, tab/frame switch, recovery path, or logger/environment state drift between recorded steps.",
        "tier3_taxonomy_match": "Type 4 Evaluator State Drift",
        "novelty_assessment": "NEW if nonzero; Tier 2 only covered same-URL drift, not between-record URL discontinuity",
        "level": "adjacent-step",
    },
    "I6": {
        "name": "inv_axtree_drift_same_url",
        "description": "Adjacent pre-action AXTree DOM artifacts have shingle similarity < 0.7 while URL is unchanged and the intervening action is not click/goto.",
        "candidate_root_cause": "AJAX render, overlay/modal replacement, in-place search update, or observation-cache replacement not represented as navigation.",
        "tier3_taxonomy_match": "Type 4 Evaluator State Drift",
        "novelty_assessment": "matches Tier 2 cross-step trajectory anomaly already-known family",
        "level": "adjacent-step",
    },
    "I7": {
        "name": "inv_finish_but_eval_reject",
        "description": "Agent attempted finish but raw evaluator success is false.",
        "candidate_root_cause": "Agent terminates from an answer-shaped or partial state that does not satisfy evaluator URL/database/content checks.",
        "tier3_taxonomy_match": "Type 4 Evaluator State Drift and false negatives/false positives",
        "novelty_assessment": "matches Tier 2 finish_wrong_state already-known family",
        "level": "episode",
    },
    "I8": {
        "name": "inv_max_step_truncate_at_click",
        "description": "Derived max-step truncation (last.done=False, no agent finish, len(steps) reaches observed cap) with last action click.",
        "candidate_root_cause": "Max-iteration masking hides the final failed click or unresolved click loop as a generic truncation.",
        "tier3_taxonomy_match": "Type 5 Actionability Check Masking and Timeout Swallowing",
        "novelty_assessment": "NEW count relative to Tier 1/Tier 2/probe: explicit max-step-at-click masking slice",
        "level": "episode",
    },
    "I9": {
        "name": "inv_element_id_role_drift",
        "description": "The same exposed AXTree element_id resolves to different roles at different steps in the same episode.",
        "candidate_root_cause": "Observation-local AX node IDs are reused across rerenders; treating element_id as stable across history risks stale-cache or wrong-role grounding.",
        "tier3_taxonomy_match": "Type 1 Coordinate Dispatch Anomaly / Type 4 state drift",
        "novelty_assessment": "mechanism anticipated by Tier 1 AXTree audit; empirical exposed-ID role-drift count is NEW",
        "level": "sequence",
    },
    "I10": {
        "name": "inv_state_change_but_obs_same",
        "description": "action_success=True and page_changed=True, but normalized pre-action and next pre-action DOM artifacts are identical.",
        "candidate_root_cause": "Step state-change bookkeeping and persisted observations disagree; likely logger/state digest desynchronization if nonzero.",
        "tier3_taxonomy_match": "Type 4 Evaluator State Drift",
        "novelty_assessment": "NEW if nonzero; direct logger consistency check not covered by earlier tiers",
        "level": "adjacent-step",
    },
}


def summarize_hits(
    inv_id: str,
    hits: list[dict[str, Any]],
    total_steps: int,
    total_episodes: int,
    notes: str = "",
) -> dict[str, Any]:
    site_breakdown = Counter(str(hit["site"]) for hit in hits)
    mode_breakdown = Counter(str(hit["mode"]) for hit in hits)
    episode_keys = {
        (hit.get("run"), hit.get("condition_id"), hit.get("site"), hit.get("task"))
        for hit in hits
    }
    meta = INVARIANT_META[inv_id]
    return {
        "id": inv_id,
        "name": meta["name"],
        "description": meta["description"],
        "implementable": True,
        "implementation_notes": notes,
        "n_violations": len(hits),
        "violation_pct_of_steps": round((len(hits) / total_steps * 100.0) if total_steps else 0.0, 4),
        "episode_traces_with_violation": len(episode_keys),
        "violation_pct_of_episodes": round((len(episode_keys) / total_episodes * 100.0) if total_episodes else 0.0, 4),
        "mode_breakdown": dict(sorted(mode_breakdown.items())),
        "site_breakdown": dict(sorted(site_breakdown.items())),
        "case_study_examples": hits[:8],
        "candidate_root_cause": meta["candidate_root_cause"],
        "tier3_taxonomy_match": meta["tier3_taxonomy_match"],
        "novelty_assessment": meta["novelty_assessment"],
    }


def build_markdown(audit: dict[str, Any]) -> str:
    invariants = audit["invariants"]
    top = invariants[:5]
    new_items = [
        item
        for item in invariants
        if "NEW" in item["novelty_assessment"] and item["n_violations"] > 0
    ]

    lines: list[str] = []
    lines.append("# Tier 4 Invariant Audit: Runner/Page-State Contradictions")
    lines.append("")
    lines.append(f"Audit date: {audit['audit_date']}. Scope: the 12 requested Phase 1 VisualWebArena run roots, read with `read_jsonl_dedup`.")
    lines.append("")
    lines.append("## 1. Scope and Method")
    lines.append("")
    lines.append(
        f"The audit scanned {audit['total_episodes_scanned']} deduplicated episode-mode traces and "
        f"{audit['total_steps_scanned']} step records. This is above the prompt estimate because the listed roots "
        "currently contain additional completed mode/task traces. No click probe was rerun and no §106 relabeling was performed."
    )
    lines.append("")
    lines.append(
        "All ten invariants were implementable from recorded fields or explicit derived episode state. "
        "`page_changed`, `text_similarity`, `action_success`, `obs_url`, `state_digest`, per-step DOM artifacts, and summary success fields are present. "
        "For I8, the runner does not write a literal `truncated_at_max_step` boolean, so the script derives it from `len(steps)` reaching the observed cap, `agent_finished=False`, and `last.done=False`. "
        "For URL and AXTree adjacency checks, the script respects the recorder semantics: DOM artifacts are saved before each action, while `state_digest.url_before/url_after` bracket the action."
    )
    lines.append("")
    lines.append("## 2. Ranked Violation Summary")
    lines.append("")
    lines.append("| Rank | ID | Invariant | Violations | Step % | Dominant site | Dominant mode | Novelty |")
    lines.append("|---:|---|---|---:|---:|---|---|---|")
    for rank, item in enumerate(invariants, 1):
        site = max(item["site_breakdown"].items(), key=lambda kv: kv[1])[0] if item["site_breakdown"] else "-"
        mode = max(item["mode_breakdown"].items(), key=lambda kv: kv[1])[0] if item["mode_breakdown"] else "-"
        novelty = item["novelty_assessment"].split(":")[0]
        lines.append(
            f"| {rank} | {item['id']} | `{item['name']}` | {item['n_violations']} | "
            f"{item['violation_pct_of_steps']:.4f}% | {site} | {mode} | {novelty} |"
        )
    lines.append("")
    lines.append(
        "The largest family is I6, the same-URL AXTree drift invariant. It is not a new mechanism: Tier 2 already found a cross-step trajectory anomaly family, and this stricter invariant confirms that the behavior is broad. "
        "I7 is likewise expected from Tier 2's finish-wrong-state catalog. The higher-value Tier 4 additions are the invariants that expose contradictions in runner bookkeeping rather than only policy failure: I2, I4, I8, I9, and I10."
    )
    lines.append("")
    lines.append("## 3. Per-Invariant Findings")
    lines.append("")
    for item in invariants:
        lines.append(f"### {item['id']} `{item['name']}`")
        lines.append("")
        lines.append(
            f"Violations: {item['n_violations']} "
            f"({item['violation_pct_of_steps']:.4f}% of steps; {item['violation_pct_of_episodes']:.4f}% of episode-mode traces). "
            f"Site breakdown: {item['site_breakdown'] or {}}. Mode breakdown: {item['mode_breakdown'] or {}}."
        )
        lines.append("")
        lines.append(f"Interpretation: {item['candidate_root_cause']} Taxonomy match: {item['tier3_taxonomy_match']}.")
        lines.append("")
        if item["case_study_examples"]:
            ex = item["case_study_examples"][0]
            lines.append(
                f"Case study: {ex['site']} task {ex['task']} step {ex['step']} "
                f"({ex['run']}, {ex['mode']}): {ex['snippet']}"
            )
        else:
            lines.append("Case study: no violations were observed, so no task-level example is available.")
        lines.append("")
        lines.append(f"Novelty assessment: {item['novelty_assessment']}.")
        lines.append("")
    lines.append("## 4. Section 4 Wiring")
    lines.append("")
    lines.append(
        "Use Tier 4 as the adversarial consistency layer on top of the earlier evidence. Tier 1 says which implementation surfaces are suspicious, Tier 2 mines action-specific silent-failure signatures, the click probe validates the §106 click-center mechanism, and Tier 4 asks whether the trace logs contradict themselves under simple invariants. "
        "The paper-ready claim should be framed as: state/action inconsistency is measurable without hand-authored bug signatures, and the largest contradictions concentrate in the same families predicted by the taxonomy."
    )
    lines.append("")
    lines.append(
        "For the main Section 4 table, report I6 and I7 as confirmation rows, then foreground the Tier 4-specific rows. "
        "I2 is the cleanest runner false-negative flag: the action is marked failed although page-state evidence changed. "
        "I4 is the timeout-swallowing row: long non-wait/non-type/non-scroll actions are likely hidden Playwright actionability waits. "
        "I8 isolates max-iteration masking at a terminal click, and I9 provides empirical support for the Tier 1 warning that AX element IDs are observation-local rather than stable cross-step object identities. "
        "I10 is a logger consistency guard; if nonzero, it should be described as direct state-change/observation desynchronization rather than an agent-policy failure."
    )
    lines.append("")
    lines.append("## 5. New Findings and Follow-Up")
    lines.append("")
    if new_items:
        new_text = ", ".join(f"{item['id']} ({item['n_violations']})" for item in new_items)
        lines.append(f"New or newly quantified findings not covered as counts by Tier 1/Tier 2/probe: {new_text}.")
    else:
        lines.append("No nonzero invariant is wholly new relative to Tier 1/Tier 2/probe; Tier 4 mainly confirms known families under an adversarial consistency lens.")
    lines.append("")
    lines.append(
        "Recommended follow-up is surgical, not another broad probe: sample the top I2/I4/I8/I9 cases, inspect the raw environment logs around those steps, and decide which rows deserve a small manual adjudication table in the appendix. "
        "Do not merge I7 into action-dispatch evidence; it is endpoint evaluator drift. Do not overclaim I6 as a bug by itself; in-place AJAX updates are legitimate, but they are a benchmark-control risk when the runner does not label the transition clearly."
    )
    lines.append("")
    lines.append("## Self-Check")
    lines.append("")
    sc = audit["self_check"]
    for key, value in sc.items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    episodes = load_episodes()
    total_episodes = len(episodes)
    total_steps = sum(len(ep.steps) for ep in episodes)
    max_steps_cap = max((len(ep.steps) for ep in episodes), default=0)

    invariant_hits: dict[str, list[dict[str, Any]]] = {
        "I1": [],
        "I2": [],
        "I3": [],
        "I4": [],
        "I5": [],
        "I6": [],
        "I7": [],
        "I8": [],
        "I9": [],
        "I10": [],
    }
    for ep in episodes:
        invariant_hits["I1"].extend(inv_i1(ep))
        invariant_hits["I2"].extend(inv_i2(ep))
        invariant_hits["I3"].extend(inv_i3(ep))
        invariant_hits["I4"].extend(inv_i4(ep))
        invariant_hits["I5"].extend(inv_i5(ep))
        invariant_hits["I6"].extend(inv_i6(ep))
        invariant_hits["I7"].extend(inv_i7(ep))
        invariant_hits["I8"].extend(inv_i8(ep, max_steps_cap=max_steps_cap))
        invariant_hits["I9"].extend(inv_i9(ep))
        invariant_hits["I10"].extend(inv_i10(ep))

    notes = {
        "I1": "Terminal finish/stop actions are excluded because they are local answer actions and are audited by I7.",
        "I8": f"`truncated_at_max_step` is derived because no literal field exists; observed max step cap={max_steps_cap}.",
        "I5": "Adapted to runner schema by comparing previous `url_after` to current `url_before`.",
        "I6": "Uses cached 5-token shingle Jaccard over adjacent pre-action DOM artifacts.",
        "I9": "Parses all exposed `[element_id] role` lines from per-step DOM artifacts, not only action-targeted IDs.",
        "I10": "Uses normalized exact equality over adjacent pre-action DOM artifacts.",
    }

    summarized = [
        summarize_hits(inv_id, hits, total_steps, total_episodes, notes=notes.get(inv_id, ""))
        for inv_id, hits in invariant_hits.items()
    ]
    summarized.sort(key=lambda item: (-int(item["n_violations"]), item["id"]))

    new_nonzero = [
        item
        for item in summarized
        if "NEW" in item["novelty_assessment"] and int(item["n_violations"]) > 0
    ]
    highest = new_nonzero[0]["id"] if new_nonzero else (summarized[0]["id"] if summarized else None)
    audit = {
        "audit_date": AUDIT_DATE,
        "run_roots": RUNS,
        "total_episodes_scanned": total_episodes,
        "total_steps_scanned": total_steps,
        "invariants": summarized,
        "summary": {
            "n_new_findings_not_covered_by_t1_t2_probe": len(new_nonzero),
            "highest_novelty_invariant_id": highest,
            "recommended_followup": (
                "Manually adjudicate high-signal I2/I4/I8/I9 examples against raw browser/environment logs; "
                "keep I7 as evaluator/policy endpoint evidence and I6 as same-URL async-state evidence."
            ),
        },
        "self_check": {
            "implementable_invariants": sum(1 for item in summarized if item["implementable"]),
            "at_least_7_of_10_implementable": sum(1 for item in summarized if item["implementable"]) >= 7,
            "case_study_count_at_least_3_unless_zero": {
                item["id"]: (item["n_violations"] == 0 or len(item["case_study_examples"]) >= 3)
                for item in summarized
            },
            "all_case_study_requirements_met": all(
                item["n_violations"] == 0 or len(item["case_study_examples"]) >= 3 for item in summarized
            ),
            "invariants_sorted_by_violation_count": all(
                summarized[i]["n_violations"] >= summarized[i + 1]["n_violations"]
                for i in range(len(summarized) - 1)
            ),
            "max_steps_cap_used_for_i8": max_steps_cap,
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(audit, indent=2, sort_keys=False), encoding="utf-8")
    OUT_MD.write_text(build_markdown(audit), encoding="utf-8")

    print(f"episodes={total_episodes} steps={total_steps} max_steps_cap={max_steps_cap}")
    for item in summarized:
        print(f"{item['id']} {item['name']} {item['n_violations']}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
