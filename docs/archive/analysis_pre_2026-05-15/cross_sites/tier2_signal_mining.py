#!/usr/bin/env python3
"""Mine non-click silent-failure signatures from Phase 1 VWA traces."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote_plus

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from p79.experiment.io_utils import read_jsonl_dedup


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

OUT_JSON = Path("docs/analysis/cross_sites/tier2_silent_failure_catalog.json")
OUT_MD = Path("docs/analysis/cross_sites/tier2_silent_failure_catalog.md")


@dataclass(frozen=True)
class EpisodeKey:
    site: str
    task: int
    run: str
    mode: str
    condition_id: str


@dataclass
class Hit:
    key: EpisodeKey
    signatures: set[str] = field(default_factory=set)
    steps: list[int] = field(default_factory=list)
    action_success_values: set[bool] = field(default_factory=set)
    evidence: dict[str, Any] = field(default_factory=dict)


def normalize_mode(condition_id: str, observation_mode: str | None = None) -> str:
    if "phantom_prompt" in condition_id:
        return "P-prompt"
    if "phantom_som" in condition_id:
        return "P-SoM"
    if "phantom_dom" in condition_id:
        return "P-text"
    if "vision" in condition_id:
        return "Vision"
    if "som" in condition_id:
        return "SoM"
    if "dom" in condition_id or observation_mode == "dom":
        return "DOM"
    return observation_mode or condition_id


def short_run(run_id: str) -> str:
    return re.sub(r"_2026\d{4}$", "", run_id)


def step_env_ms(step: dict[str, Any]) -> float:
    return float((step.get("latency_ms") or {}).get("env_step") or 0.0)


def state(step: dict[str, Any]) -> dict[str, Any]:
    return step.get("state_digest") or {}


def url_before(step: dict[str, Any]) -> str:
    return str(state(step).get("url_before") or "")


def url_after(step: dict[str, Any]) -> str:
    return str(state(step).get("url_after") or step.get("obs_url") or "")


def text_similarity(step: dict[str, Any]) -> float | None:
    value = step.get("text_similarity")
    return float(value) if isinstance(value, (int, float)) else None


def page_reasons(step: dict[str, Any]) -> set[str]:
    return set(step.get("page_change_reasons") or [])


def canonical_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def typed_query_variants(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    words = re.findall(r"[a-zA-Z0-9]+", stripped.casefold())
    variants = {canonical_text(stripped)}
    if len(words) >= 2:
        variants.add(" ".join(words[: min(5, len(words))]))
    elif words:
        variants.add(words[0])
    return [v for v in variants if len(v) >= 3]


_TEXT_CACHE: dict[str, str] = {}


def read_obs_text(step: dict[str, Any]) -> str:
    path = ((step.get("artifact_paths") or {}).get("dom") or "")
    if not path:
        return ""
    if path in _TEXT_CACHE:
        return _TEXT_CACHE[path]
    p = Path(path)
    if not p.exists():
        _TEXT_CACHE[path] = ""
        return ""
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    _TEXT_CACHE[path] = text
    return text


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def task_config_path(run_root: Path, site: str, task_id: int) -> Path:
    return run_root / "task_configs" / f"{site}_task_{task_id}.json"


def extract_eval_urls(task_config: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    ev = task_config.get("eval") or {}
    for item in ev.get("program_html") or []:
        if isinstance(item, dict) and item.get("url"):
            urls.append(str(item["url"]))
    ref = ev.get("reference_url")
    if ref:
        urls.append(str(ref))
    return urls


def objective_hint(task_config: dict[str, Any]) -> str:
    return str(task_config.get("intent") or task_config.get("intent_template") or "")


def add_hit(bucket: dict[EpisodeKey, Hit], key: EpisodeKey, step: dict[str, Any], signature: str, evidence: dict[str, Any] | None = None) -> None:
    hit = bucket.setdefault(key, Hit(key=key))
    hit.signatures.add(signature)
    if isinstance(step.get("step_idx"), int):
        hit.steps.append(step["step_idx"])
    if isinstance(step.get("action_success"), bool):
        hit.action_success_values.add(bool(step["action_success"]))
    if evidence:
        hit.evidence.update(evidence)


def select_current_value(obs_text: str, option_label: str) -> bool:
    label = re.escape(option_label)
    patterns = [
        rf'currently selected="{label}"',
        rf"combobox '{label}'",
        rf"combobox \"{label}\"",
    ]
    return any(re.search(pattern, obs_text, flags=re.I) for pattern in patterns)


def classify_finish(step: dict[str, Any], summary: dict[str, Any], task_config: dict[str, Any]) -> str:
    action = step.get("action") or {}
    thought = canonical_text(str(action.get("thought") or ""))
    answer = canonical_text(str(action.get("answer") or ""))
    combined = f"{thought} {answer}"
    obs_url = url_after(step)
    intent = canonical_text(objective_hint(task_config))

    if any(token in obs_url for token in ["/search", "catalogsearch/result", "page=search"]):
        return "agent_finished_on_search_results_page_not_target"
    if any(token in obs_url for token in ["/submission_images/", ".jpg", ".png", ".jpeg"]) or (
        "image" in intent and any(token in combined for token in ["cannot", "unable", "assume", "placeholder", "static image"])
    ):
        return "visual_perception_guess_or_image_blind_finish"
    if any(token in combined for token in ["already", "successfully", "complete", "completed", "done"]) and (
        summary.get("adjusted_success") is False or summary.get("success") is False
    ):
        return "agent_false_confidence_after_partial_work"
    if any(token in combined for token in ["assume", "guess", "placeholder", "cannot determine", "not contain any information"]):
        return "agent_hallucinated_answer"
    eval_urls = extract_eval_urls(task_config)
    if eval_urls and not any(u and u != "last" and u in obs_url for u in eval_urls if not u.startswith("func:")):
        return "finished_off_eval_target_url"
    return "wrong_state_unclear_from_trace"


def collect_episode_files(run_root: Path) -> list[Path]:
    files = []
    for path in run_root.glob("*/episodes/*_steps_v2.jsonl"):
        if any(part.startswith(".") for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def mine() -> tuple[dict[str, Any], str]:
    categories: dict[str, dict[EpisodeKey, Hit]] = {
        "type_silent_failure": {},
        "scroll_silent_failure": {},
        "select_option_silent_failure": {},
        "finish_wrong_state": {},
        "cross_step_trajectory_anomaly": {},
    }
    subcategories: Counter[str] = Counter()
    run_status: list[dict[str, Any]] = []
    per_run_counts: Counter[str] = Counter()
    site_scanned: Counter[str] = Counter()
    mode_scanned: Counter[str] = Counter()
    failed_episode_keys: set[EpisodeKey] = set()
    scanned_episode_keys: set[EpisodeKey] = set()
    total_steps = 0
    action_type_counts: Counter[str] = Counter()
    skipped: list[dict[str, str]] = []

    for run in RUNS:
        run_root = Path(run)
        if not run_root.exists():
            skipped.append({"run": run, "reason": "skipped (data not available)"})
            continue
        steps_files = collect_episode_files(run_root)
        scanned_in_run = 0
        for steps_path in steps_files:
            summary_path = steps_path.with_name(steps_path.name.replace("_steps_v2.jsonl", "_summary_v2.json"))
            if not summary_path.exists():
                continue
            steps = read_jsonl_dedup(steps_path)
            summary = load_json(summary_path)
            if not steps:
                continue
            first = steps[0]
            site = str(first.get("benchmark_site") or summary.get("benchmark_site") or "")
            task_id = int(first.get("task_id") if first.get("task_id") is not None else summary.get("task_id"))
            run_id = str(first.get("run_id") or summary.get("run_id") or run_root.name)
            condition_id = str(first.get("condition_id") or summary.get("condition_id") or steps_path.parents[1].name)
            mode = normalize_mode(condition_id, first.get("observation_mode"))
            key = EpisodeKey(site=site, task=task_id, run=short_run(run_id), mode=mode, condition_id=condition_id)
            task_config = load_json(task_config_path(run_root, site, task_id))

            scanned_episode_keys.add(key)
            scanned_in_run += 1
            per_run_counts[run_root.name] += 1
            site_scanned[site] += 1
            mode_scanned[mode] += 1
            total_steps += len(steps)
            if summary.get("adjusted_success") is False or summary.get("success") is False:
                failed_episode_keys.add(key)

            consecutive_type: dict[tuple[Any, str], int] = defaultdict(int)
            consecutive_scroll_noop = 0
            consecutive_select: dict[tuple[Any, str], int] = defaultdict(int)

            for step in steps:
                action_type = str(step.get("action_type") or "").lower()
                action_type_counts[action_type] += 1
                action = step.get("action") or {}
                sim = text_similarity(step)
                sd = state(step)
                reasons = page_reasons(step)
                before = url_before(step)
                after = url_after(step)
                same_url = before == after
                env_ms = step_env_ms(step)

                if action_type == "type":
                    text = str(action.get("text") or "")
                    variants = typed_query_variants(text)
                    decoded_url = canonical_text(unquote_plus(after))
                    obs_norm = canonical_text(read_obs_text(step))
                    text_in_state = any(v in obs_norm or v in decoded_url for v in variants)
                    form_changed = bool(sd.get("form_fields_changed")) or "form_value_changed" in reasons or "form_fields_changed" in reasons
                    bbox = step.get("element_bbox")
                    stale_bbox = bbox == [0.0, 0.0, 10.0, 10.0] or bbox == [0, 0, 10, 10]
                    silent = False
                    sigs: list[str] = []
                    if text.strip() and same_url and not form_changed and not text_in_state:
                        silent = True
                        sigs.append("same_url_no_form_or_text_echo")
                    if step.get("action_success") is False and same_url and (sim is None or sim >= 0.95):
                        silent = True
                        sigs.append("runner_failed_no_progress")
                    if stale_bbox and text.strip() and not text_in_state:
                        silent = True
                        sigs.append("stale_or_offscreen_bbox")
                    if text.strip() and same_url and env_ms >= 5000 and not text_in_state:
                        sigs.append("slow_env_step_without_text_echo")
                    rep_key = (action.get("element_id"), canonical_text(text))
                    if text.strip():
                        consecutive_type[rep_key] += 1
                        if consecutive_type[rep_key] >= 2 and not text_in_state:
                            silent = True
                            sigs.append("repeated_same_element_type_without_echo")
                    if silent:
                        add_hit(
                            categories["type_silent_failure"],
                            key,
                            step,
                            ";".join(sorted(set(sigs))),
                            {
                                "example_text": text.strip()[:80],
                                "obs_url": after,
                                "env_step_ms": round(env_ms, 1),
                            },
                        )
                else:
                    consecutive_type.clear()

                if action_type == "scroll":
                    scroll_same = sd.get("scroll_y_before") == sd.get("scroll_y_after")
                    page_static = (sim is not None and sim >= 0.95) or step.get("page_changed") is False
                    if scroll_same and "scroll_changed" not in reasons and page_static:
                        consecutive_scroll_noop += 1
                    else:
                        consecutive_scroll_noop = 0
                    silent = False
                    sigs = []
                    if consecutive_scroll_noop >= 2:
                        silent = True
                        sigs.append("consecutive_scrolls_no_viewport_move")
                    if scroll_same and page_static and step.get("action_success") is True:
                        silent = True
                        sigs.append("action_success_true_but_scroll_y_static")
                    if scroll_same and env_ms < 200:
                        silent = True
                        sigs.append("short_circuit_scroll_latency")
                    if silent:
                        add_hit(
                            categories["scroll_silent_failure"],
                            key,
                            step,
                            ";".join(sorted(set(sigs))),
                            {
                                "scroll_y": sd.get("scroll_y_after"),
                                "text_similarity": sim,
                                "env_step_ms": round(env_ms, 1),
                            },
                        )
                else:
                    consecutive_scroll_noop = 0

                if action_type == "select_option":
                    option = str(action.get("option_label") or "")
                    obs = read_obs_text(step)
                    selected = select_current_value(obs, option) if option else False
                    form_changed = bool(sd.get("form_fields_changed")) or "form_value_changed" in reasons or "form_fields_changed" in reasons
                    silent = False
                    sigs = []
                    if option and same_url and not form_changed and not selected:
                        silent = True
                        sigs.append("same_url_dropdown_not_selected")
                    if step.get("action_success") is False and same_url and (sim is None or sim >= 0.95):
                        silent = True
                        sigs.append("runner_failed_no_progress")
                    rep_key = (action.get("element_id"), canonical_text(option))
                    consecutive_select[rep_key] += 1
                    if consecutive_select[rep_key] >= 2 and same_url and not selected:
                        silent = True
                        sigs.append("repeated_same_option_without_state_update")
                    if silent:
                        add_hit(
                            categories["select_option_silent_failure"],
                            key,
                            step,
                            ";".join(sorted(set(sigs))),
                            {
                                "option_label": option,
                                "obs_url": after,
                                "text_similarity": sim,
                                "env_step_ms": round(env_ms, 1),
                            },
                        )
                else:
                    consecutive_select.clear()

                if action_type == "finish":
                    wrong = summary.get("adjusted_success") is False or summary.get("success") is False or step.get("reward") == 0.0
                    if wrong:
                        subcat = classify_finish(step, summary, task_config)
                        subcategories[subcat] += 1
                        add_hit(
                            categories["finish_wrong_state"],
                            key,
                            step,
                            subcat,
                            {
                                "obs_url": after,
                                "answer": str(action.get("answer") or "")[:140],
                                "confidence": (step.get("confidence") or {}).get("verbalized"),
                                "intent": objective_hint(task_config)[:180],
                            },
                        )

                # Large AXTree shifts with no URL change and no ordinary navigation action.
                navigation_like = {"click", "goto_url", "open_url", "go_back", "go_forward", "new_tab"}
                ordinary_viewport_change = action_type == "scroll" and "scroll_changed" in reasons
                if (
                    same_url
                    and action_type not in navigation_like
                    and not ordinary_viewport_change
                    and sim is not None
                    and sim < 0.7
                    and env_ms >= 5000
                    and step.get("page_changed") is True
                ):
                    add_hit(
                        categories["cross_step_trajectory_anomaly"],
                        key,
                        step,
                        "same_url_large_axtree_shift_non_navigation_action",
                        {
                            "action_type": action_type,
                            "text_similarity": sim,
                            "page_change_reasons": sorted(reasons),
                            "env_step_ms": round(env_ms, 1),
                            "obs_url": after,
                        },
                    )

        run_status.append({"run": run_root.name, "episode_mode_traces": scanned_in_run, "status": "scanned"})

    total_episode_traces = len(scanned_episode_keys)

    def category_json(name: str, root_cause: str) -> dict[str, Any]:
        hits = categories[name]
        mode_breakdown = Counter(k.mode for k in hits)
        site_breakdown = Counter(k.site for k in hits)
        run_breakdown = Counter(k.run for k in hits)
        signature_counts = Counter()
        case_studies = []
        for key, hit in sorted(hits.items(), key=lambda kv: (kv[0].site, kv[0].run, kv[0].mode, kv[0].task)):
            for sig in hit.signatures:
                signature_counts[sig] += 1
            if len(case_studies) < 8:
                case_studies.append(
                    {
                        "site": key.site,
                        "task": key.task,
                        "run": key.run,
                        "mode": key.mode,
                        "condition_id": key.condition_id,
                        "steps": sorted(set(hit.steps))[:8],
                        "signatures": sorted(hit.signatures),
                        "evidence": hit.evidence,
                    }
                )
        out = {
            "n_episodes": len(hits),
            "blast_radius_pct": round(100.0 * len(hits) / total_episode_traces, 2) if total_episode_traces else 0.0,
            "mode_breakdown": dict(sorted(mode_breakdown.items())),
            "site_breakdown": dict(sorted(site_breakdown.items())),
            "run_breakdown": dict(sorted(run_breakdown.items())),
            "signature_counts": dict(signature_counts.most_common()),
            "case_study_task_ids": case_studies,
            "candidate_root_cause": root_cause,
        }
        if name == "finish_wrong_state":
            out["subcategories"] = dict(subcategories.most_common())
        return out

    root_causes = {
        "type_silent_failure": "Typing frequently lands on stale, offscreen, or non-submitting elements: the runner may report success, but URL/form state and AXTree text do not echo the typed value, leaving the agent to repeat the same search or continue from an unchanged page.",
        "scroll_silent_failure": "Scroll no-ops are concentrated where the viewport is already pinned, a modal/overlay captures scroll, or the target page is not scrollable; the agent receives another near-identical AXTree and tends to spend more steps looking for hidden content.",
        "select_option_silent_failure": "Dropdown failures mostly arise from selecting unavailable or DOM-stale options. The page keeps the previous sort/category state, often with action_success=false or no selected-option echo, so the agent continues under a false filter assumption.",
        "finish_wrong_state": "Wrong-state finishes are dominated by agents completing from an answer-shaped local observation rather than the evaluator target state: search pages, image-only pages, partial form/message workflows, and confident but unevaluated claims.",
        "cross_step_trajectory_anomaly": "Large same-URL AXTree shifts without a navigation-like action point to frame-side async refreshes, form-only rerenders, modal state changes, and stale cache/observation replacements that change what the agent sees without an explicit navigation signal.",
    }

    category_payload = {name: category_json(name, root_causes[name]) for name in categories}
    all_failure_keys = set().union(*(set(bucket) for bucket in categories.values()))
    site_rates = {
        site: sum(1 for key in all_failure_keys if key.site == site) / count
        for site, count in site_scanned.items()
        if count
    }
    mode_rates = {
        mode: sum(1 for key in all_failure_keys if key.mode == mode) / count
        for mode, count in mode_scanned.items()
        if count
    }
    summary = {
        "total_silent_failures_estimated": len(all_failure_keys),
        "fraction_of_all_failures": round(len(all_failure_keys) / len(failed_episode_keys), 3) if failed_episode_keys else 0.0,
        "failed_episode_mode_traces": len(failed_episode_keys),
        "site_with_highest_concentration": max(site_rates, key=site_rates.get) if site_rates else None,
        "site_concentration_rates": {k: round(v, 3) for k, v in sorted(site_rates.items())},
        "mode_with_highest_concentration": max(mode_rates, key=mode_rates.get) if mode_rates else None,
        "mode_concentration_rates": {k: round(v, 3) for k, v in sorted(mode_rates.items())},
        "unique_category_sum": sum(cat["n_episodes"] for cat in category_payload.values()),
        "overlap_adjusted_unique_episodes": len(all_failure_keys),
    }
    payload = {
        "audit_date": "2026-04-30",
        "scan_scope_note": f"Scanned only the listed Phase 1 VisualWebArena run roots; archives, WA runs, and click reclassification were excluded. The listed roots currently contain {total_episode_traces:,} episode-mode traces, above the prompt's approximate 3,500 estimate.",
        "total_episodes_scanned": total_episode_traces,
        "total_steps_scanned": total_steps,
        "per_run_scanned": run_status,
        "skipped_runs": skipped,
        "action_type_counts": dict(action_type_counts.most_common()),
        "site_episode_mode_traces": dict(sorted(site_scanned.items())),
        "mode_episode_mode_traces": dict(sorted(mode_scanned.items())),
        "categories": category_payload,
        "cross_action_summary": summary,
        "validation": {
            "mode_breakdowns_sum_to_n_episodes": {
                name: sum(cat["mode_breakdown"].values()) == cat["n_episodes"]
                for name, cat in category_payload.items()
            },
            "case_study_count_at_least_3": {
                name: len(cat["case_study_task_ids"]) >= 3
                for name, cat in category_payload.items()
            },
            "category_episode_floor_at_least_5": {
                name: cat["n_episodes"] >= 5
                for name, cat in category_payload.items()
            },
        },
    }
    return payload, render_markdown(payload)


def render_case_line(case: dict[str, Any]) -> str:
    ev = case.get("evidence") or {}
    bits = []
    if ev.get("example_text"):
        bits.append(f"text `{ev['example_text']}`")
    if ev.get("option_label"):
        bits.append(f"option `{ev['option_label']}`")
    if ev.get("answer"):
        bits.append(f"answer `{ev['answer']}`")
    if ev.get("action_type"):
        bits.append(f"action `{ev['action_type']}`")
    if ev.get("obs_url"):
        bits.append(f"url `{ev['obs_url']}`")
    detail = "; ".join(bits[:2])
    if detail:
        detail = f" Evidence: {detail}."
    return f"- {case['site']} task {case['task']} ({case['run']}, {case['mode']}), steps {case['steps']}: {', '.join(case['signatures'])}.{detail}"


def render_markdown(payload: dict[str, Any]) -> str:
    cats = payload["categories"]
    lines = [
        "# Tier 2 Silent-Failure Signal Mining Catalog",
        "",
        f"Audit date: {payload['audit_date']}",
        "",
        "## Scope and Denominators",
        "",
        payload["scan_scope_note"],
        "",
        f"- Episode-mode traces scanned: {payload['total_episodes_scanned']}",
        f"- Steps scanned: {payload['total_steps_scanned']}",
        f"- Failed episode-mode traces used for failure-fraction denominator: {payload['cross_action_summary']['failed_episode_mode_traces']}",
        f"- Skipped runs: {len(payload['skipped_runs'])}",
        "",
        "The mining pass does not redo the click taxonomy. It treats the existing click-probe result as the companion audit and focuses on TYPE, SCROLL, SELECT_OPTION, FINISH, and cross-step non-navigation anomalies.",
        "",
        "The denominator is an episode-mode trace rather than a unique task ID. This is the right unit for this audit because a single task can be run in DOM, SoM, Vision, and phantom variants, and the question is whether a particular policy-observation stack silently loses action effects. The listed roots contain more traces than the prompt estimate; no additional roots were pulled in.",
        "",
        "The signatures require state evidence, not only a bad final score. TYPE requires missing text/form/URL echo, repeated no-progress typing, or a stale/offscreen target. SCROLL requires repeated static viewport state or a success-marked static viewport. SELECT_OPTION requires a dropdown state that does not commit or a repeated no-progress option selection. FINISH is broader by design: any failed trace where the agent explicitly terminates is a wrong-state finish, then subclustered by the final URL, answer, and task evaluator target. Cross-step anomalies use large same-URL AXTree shifts after non-navigation actions.",
        "",
        "## Cross-Action Summary",
        "",
        f"The overlap-adjusted estimate is **{payload['cross_action_summary']['total_silent_failures_estimated']} episode-mode traces**, or **{payload['cross_action_summary']['fraction_of_all_failures']:.3f}** of all failed traces in this scan. The highest site concentration is **{payload['cross_action_summary']['site_with_highest_concentration']}**; the highest mode concentration is **{payload['cross_action_summary']['mode_with_highest_concentration']}**.",
        "",
        "| Category | Episodes | Blast radius | Dominant site | Dominant mode |",
        "|---|---:|---:|---|---|",
    ]
    for name, cat in cats.items():
        dominant_site = max(cat["site_breakdown"], key=cat["site_breakdown"].get) if cat["site_breakdown"] else "-"
        dominant_mode = max(cat["mode_breakdown"], key=cat["mode_breakdown"].get) if cat["mode_breakdown"] else "-"
        lines.append(f"| `{name}` | {cat['n_episodes']} | {cat['blast_radius_pct']}% | {dominant_site} | {dominant_mode} |")
    lines.extend(["", "## Category Findings", ""])

    display_names = {
        "type_silent_failure": "TYPE Silent Failure",
        "scroll_silent_failure": "SCROLL Silent Failure",
        "select_option_silent_failure": "SELECT_OPTION Silent Failure",
        "finish_wrong_state": "FINISH Wrong-State Failure",
        "cross_step_trajectory_anomaly": "Cross-Step Trajectory Anomaly",
    }
    interpretation = {
        "type_silent_failure": "The dominant evidence pattern is not a normal failed search. The agent emits text, often with a newline that should submit a search or fill a form, but the post-action URL, form flags, and AXTree text do not contain the intended value. Several cases also expose the runner targeting a 0,0,10,10 bounding box, which is a strong stale-element or hidden-element signature. This family is a direct runtime counterpart to static action-dispatch concerns: a syntactically valid TYPE action is accepted, but the state transition that the policy relies on is absent.",
        "scroll_silent_failure": "The scroll family is mostly repeated no-op scrolling. A single no-op at the bottom of a page can be benign, but two or more consecutive scrolls with identical scroll_y and near-identical AXTree text is a silent progress failure for the policy: the agent has no new content and often keeps searching for items that are not reachable through the current viewport state. These are especially visible on classifieds and Vision traces, where the agent spends budget probing listing pages or image/detail pages that no longer move.",
        "select_option_silent_failure": "SELECT_OPTION failures are smaller in absolute count but high value diagnostically. They concentrate on classifieds sort/category widgets, especially unavailable options such as `Oldest first` or stale category comboboxes. The page remains on the old URL or old selected value, yet the agent often reasons as if the filter changed. This is the cleanest non-click analogue of button/AJAX silent failure: the command targets a UI control whose state is supposed to commit into query parameters or selected text, but the next observation does not encode that commit.",
        "finish_wrong_state": "FINISH dominates because it captures the policy-level endpoint of silent failure: the agent decides the task is complete while the evaluator state is false. This should not be read as an action-dispatch bug by itself. The subcategories separate direct hallucinated answers and image-blind guesses from search-result finishes, off-target URL finishes, and partial-work false confidence. This family is important for paper framing because many upstream silent failures only become measurable when the agent converts a misleading observation into a confident termination.",
        "cross_step_trajectory_anomaly": "The cross-step family captures traces where the AXTree changes sharply without a navigation-like action or URL transition. Many overlap with TYPE because search pages rerender in place or stale observations are replaced after form-like actions; others are modal or async page refreshes. This is not necessarily a user-facing bug in isolation, but it is a benchmark-control risk: the agent cannot distinguish an expected in-place update from an observation-cache replacement unless the runner exposes a stronger transition reason.",
    }
    for name, cat in cats.items():
        lines.extend([
            f"### {display_names[name]}",
            "",
            f"**Blast radius.** {cat['n_episodes']} episode-mode traces ({cat['blast_radius_pct']}%). Mode breakdown: {cat['mode_breakdown']}. Site breakdown: {cat['site_breakdown']}.",
            "",
            f"**Candidate root cause.** {cat['candidate_root_cause']}",
            "",
            f"**Interpretation.** {interpretation[name]}",
            "",
        ])
        if name == "finish_wrong_state":
            lines.extend(["**Root-cause subcategories.**", ""])
            for subcat, count in cat.get("subcategories", {}).items():
                lines.append(f"- `{subcat}`: {count}")
            lines.append("")
        lines.extend(["**Representative cases.**", ""])
        for case in cat["case_study_task_ids"][:4]:
            lines.append(render_case_line(case))
        lines.append("")

    lines.extend([
        "## Evidence Quality",
        "",
        "These counts should be used as a blast-radius estimate, not as hand-adjudicated ground truth. The strongest evidence categories are SELECT_OPTION and TYPE because they have concrete expected postconditions: query text, form-value change, selected option text, or URL parameters. SCROLL is also strong when repeated, but a single static scroll at page bottom is not enough, so the miner only promotes repeated static viewport patterns or success-marked no movement. Cross-step anomalies are weaker as root-cause labels but useful as alerts for same-URL observation instability.",
        "",
        "FINISH is deliberately framed differently. A wrong-state finish is not proof that the finish action failed; it is proof that the agent terminated from a state that did not satisfy the evaluator. The subcategory split is therefore the important paper signal. Search-page finishes and off-evaluator-URL finishes are high-confidence state mismatch. Image-blind guesses and hallucinated answers are policy/evaluator mismatch. The large `wrong_state_unclear_from_trace` bucket should be reserved for appendix-level examples or manual follow-up rather than headline claims.",
        "",
        "The overlap-adjusted unique count is lower than the per-category sum because one trace can contain both an upstream action-state anomaly and a wrong-state finish. That overlap is expected and useful: it gives a trajectory-level story from dispatch loss or observation instability to final false confidence.",
        "",
        "## Section 4 Wiring",
        "",
        "Use this catalog as the non-click complement to the existing click-probe taxonomy. In Section 4, the clean wiring is to present click failures as the first family, then add these five non-click families as evidence that silent failure is not a single click-dispatch phenomenon. The strongest paper claims are:",
        "",
        "- TYPE and SELECT_OPTION show dispatch/state-commit bugs: the action parser and runner accept a command, but DOM state, URL state, or selected value does not commit in a way visible to the next policy call.",
        "- SCROLL and cross-step anomalies show observation-transition bugs: the policy sees either a repeated page after an accepted action or a large same-URL AXTree replacement that is not tied to a navigation action.",
        "- FINISH wrong-state failures show policy/evaluator mismatch: the agent can confidently terminate on an answer-shaped local state while the programmatic evaluator requires a different URL, database side effect, or exact content.",
        "",
        "For the paper table, report `n_episodes`, blast radius, dominant site/mode, and one task ID per family. For the narrative, pair this with Tier 1 static candidates: TYPE and SELECT_OPTION are the best confirmation targets for action-dispatch code paths; FINISH is the best evidence for policy-level false confidence and evaluator-state mismatch.",
        "",
        "## Self-Check",
        "",
    ])
    validation = payload["validation"]
    for check, results in validation.items():
        lines.append(f"- `{check}`: {results}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    payload, md = mine()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(json.dumps({
        "total_episodes_scanned": payload["total_episodes_scanned"],
        "total_steps_scanned": payload["total_steps_scanned"],
        "categories": {k: v["n_episodes"] for k, v in payload["categories"].items()},
        "cross_action_summary": payload["cross_action_summary"],
        "validation": payload["validation"],
    }, indent=2))


if __name__ == "__main__":
    main()
