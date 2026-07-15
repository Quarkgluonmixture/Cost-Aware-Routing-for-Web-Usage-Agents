"""OFFLINE / NON-GATE / POST-HOC EXPLORATORY — cross-object Pareto decomposition.

Answers: can prior work whose routed object is NOT the observation
representation (e.g. RouteLLM-class model routing) be compared on the same
Pareto plane?  Yes — iff the axes share units and realized trajectories exist.
The 18 landed classifieds arms (3 backbones x 6 modes, one 224-task universe)
make model-choice, representation-choice, and joint-choice oracles all
computable from the same replayed data.

Unit discipline (prereg lock): billed USD is NEVER pooled across backbones
(api_usd vs electricity-derived bases differ by ~3 orders of magnitude per
token — a mixed axis is a unit artifact).  Cross-backbone planes use
``total_tokens`` and retry-adjusted latency, which share physical units.
USD planes are emitted within-backbone only.

Oracle semantics (this script): per task, pick the metric-cheapest successful
arm; if no arm succeeds, pick the metric-cheapest arm (success 0, cost still
paid).  This differs from ``derive_oracle_label`` (fixed priority order) in
tie-breaking only; documented rather than silently reused because the priority
list is a per-backbone mode order, undefined over 18 mixed arms.

Usage:
    .venv/bin/python3 scripts/analysis/cross_object_pareto.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.canonical_task_universe import expected_scored_ids, task_id_set_sha256
from router_pareto_analysis import PolicyPoint, pareto_frontier

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "results/phantom_paper/run_manifest.yaml"
OUT_DIR = REPO / "results/phantom_paper/l1_router_offline_20260715/cross_object"
SITE = "classifieds"
BACKBONES = ("B0", "B1", "B2")
MODES = ("DOM", "SoM", "Vision", "P-text", "P-prompt", "P-SoM")
HEADER = "OFFLINE / NON-GATE / POST-HOC EXPLORATORY"


def load_arms(manifest_path: Path = MANIFEST) -> dict[tuple[str, str], dict[int, dict[str, Any]]]:
    """Read all 18 paper-grade classifieds arms; fail closed on any gap."""
    manifest = yaml.safe_load(manifest_path.read_text())
    entries = [
        c for c in manifest["cells"]
        if c["site"] == SITE and c.get("grade") == "paper-grade"
    ]
    found = {(c["baseline"], c["mode"]) for c in entries}
    expected = {(b, m) for b in BACKBONES for m in MODES}
    if found != expected:
        raise ValueError(f"Expected 18 arms {sorted(expected)}, manifest has {sorted(found)}")

    ids, sha = expected_scored_ids(SITE)
    arms: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    for c in entries:
        ep_dir = REPO / "results/visualwebarena/phase1" / c["run_dir"] / c["condition_subdir"] / "episodes"
        rows: dict[int, dict[str, Any]] = {}
        for f in sorted(ep_dir.glob("*_summary_v2.json")):
            rec = json.loads(f.read_text())
            tid = int(rec["task_id"])
            if tid in rows:
                raise ValueError(f"Duplicate task {tid} in {ep_dir}")
            if not isinstance(rec["success"], bool):
                raise ValueError(f"Non-bool success for task {tid} in {ep_dir}")
            rows[tid] = {
                "success": rec["success"],
                "tokens": float(rec["total_tokens"]),
                "latency_s": float(rec["total_latency_minus_retry_ms"]) / 1000.0,
                "usd": float(rec["total_billed_cost_usd"]),
                "cost_unit_basis": str(rec.get("cost_unit_basis") or "unknown"),
            }
        if set(rows) != ids:
            raise ValueError(
                f"{c['baseline']}/{c['mode']}: task universe mismatch "
                f"({len(rows)} rows vs {len(ids)} canonical, sha {sha[:12]})"
            )
        bases = {r["cost_unit_basis"] for r in rows.values()}
        if len(bases) != 1:
            raise ValueError(f"{c['baseline']}/{c['mode']}: mixed cost_unit_basis {bases}")
        arms[(c["baseline"], c["mode"])] = rows
    return arms


def arm_stats(rows: dict[int, dict[str, Any]]) -> dict[str, float]:
    n = len(rows)
    return {
        "n_tasks": n,
        "sr": sum(r["success"] for r in rows.values()) / n,
        "mean_tokens": sum(r["tokens"] for r in rows.values()) / n,
        "mean_latency_s": sum(r["latency_s"] for r in rows.values()) / n,
        "mean_usd": sum(r["usd"] for r in rows.values()) / n,
        "cost_unit_basis": next(iter(rows.values()))["cost_unit_basis"],
    }


def hindsight_oracle(
    arms: dict[tuple[str, str], dict[int, dict[str, Any]]],
    subset: list[tuple[str, str]],
    metric: str,
) -> dict[str, Any]:
    """Metric-cheapest-success-else-metric-cheapest oracle over an arm subset."""
    task_ids = sorted(arms[subset[0]].keys())
    solved: set[int] = set()
    costs: list[float] = []
    picks: dict[int, str] = {}
    for tid in task_ids:
        succ = [a for a in subset if arms[a][tid]["success"]]
        pool = succ or subset
        pick = min(pool, key=lambda a: arms[a][tid][metric])
        if succ:
            solved.add(tid)
        costs.append(arms[pick][tid][metric])
        picks[tid] = f"{pick[0]}/{pick[1]}"
    return {
        "sr": len(solved) / len(task_ids),
        "n_solved": len(solved),
        f"mean_{metric}": sum(costs) / len(costs),
        "solved_task_ids": sorted(solved),
        "picks": picks,
    }


def union_solved(arms, subset) -> set[int]:
    out: set[int] = set()
    for a in subset:
        out |= {tid for tid, r in arms[a].items() if r["success"]}
    return out


def build(arms) -> dict[str, Any]:
    all18 = sorted(arms.keys())
    stats = {f"{b}/{m}": arm_stats(arms[(b, m)]) for b, m in all18}

    # --- SR decomposition (unit-free axis) -------------------------------
    best_arm = max(stats, key=lambda k: stats[k]["sr"])
    per_backbone_rep = {
        b: hindsight_oracle(arms, [(b, m) for m in MODES], "tokens") for b in BACKBONES
    }
    per_mode_model = {
        m: hindsight_oracle(arms, [(b, m) for b in BACKBONES], "tokens") for m in MODES
    }
    joint = {
        metric: hindsight_oracle(arms, all18, metric) for metric in ("tokens", "latency_s")
    }
    b0_solved = union_solved(arms, [("B0", m) for m in MODES])
    all_solved = union_solved(arms, all18)
    best_mode_model = max(per_mode_model, key=lambda m: per_mode_model[m]["sr"])

    decomposition = {
        "best_single_arm": {"arm": best_arm, "sr": stats[best_arm]["sr"]},
        "representation_only_oracle_per_backbone": {
            b: {"sr": v["sr"], "n_solved": v["n_solved"]} for b, v in per_backbone_rep.items()
        },
        "model_only_oracle_per_mode": {
            m: {"sr": v["sr"], "n_solved": v["n_solved"]} for m, v in per_mode_model.items()
        },
        "joint_18_arm_oracle": {"sr": joint["tokens"]["sr"], "n_solved": joint["tokens"]["n_solved"]},
        "marginals_pp": {
            "rep_within_B0_over_best_arm": 100 * (per_backbone_rep["B0"]["sr"] - stats[best_arm]["sr"]),
            "model_added_on_top_of_B0_rep": 100 * (joint["tokens"]["sr"] - per_backbone_rep["B0"]["sr"]),
            "rep_added_on_top_of_best_model_choice": 100 * (
                joint["tokens"]["sr"] - per_mode_model[best_mode_model]["sr"]
            ),
            "best_model_only_mode": best_mode_model,
        },
        "union_counts": {
            "solved_by_any_B0_mode": len(b0_solved),
            "solved_by_any_of_18": len(all_solved),
            "added_by_B1_B2_beyond_B0": sorted(all_solved - b0_solved),
        },
    }

    # --- cross-backbone Pareto planes (tokens / latency) -----------------
    planes: dict[str, Any] = {}
    n_univ = stats[f"B0/DOM"]["n_tasks"]
    for metric, unit in (("tokens", "tokens/task"), ("latency_s", "s/task")):
        pts = [
            PolicyPoint(
                policy_id=k, label=k, category="fixed_arm",
                mean_cost_usd=v[f"mean_{metric}"], success_rate_pct=100 * v["sr"],
                n_tasks=int(v["n_tasks"]), n_success=round(v["sr"] * v["n_tasks"]),
            )
            for k, v in stats.items()
        ]
        frontier = pareto_frontier(pts)
        o = joint[metric]
        planes[metric] = {
            "unit": unit,
            "points": {p.policy_id: {"sr_pct": p.success_rate_pct, "cost": p.mean_cost_usd} for p in pts},
            "fixed_arm_frontier": sorted(p.policy_id for p in frontier),
            "joint_oracle": {"sr": o["sr"], f"mean_{metric}": o[f"mean_{metric}"]},
        }

    # --- USD planes: within-backbone only ---------------------------------
    usd_planes = {}
    for b in BACKBONES:
        pts = [
            PolicyPoint(
                policy_id=f"{b}/{m}", label=f"{b}/{m}", category="fixed_arm",
                mean_cost_usd=stats[f"{b}/{m}"]["mean_usd"],
                success_rate_pct=100 * stats[f"{b}/{m}"]["sr"],
                n_tasks=int(stats[f"{b}/{m}"]["n_tasks"]),
                n_success=round(stats[f"{b}/{m}"]["sr"] * stats[f"{b}/{m}"]["n_tasks"]),
            )
            for m in MODES
        ]
        oracle_usd = hindsight_oracle(arms, [(b, m) for m in MODES], "usd")
        usd_planes[b] = {
            "cost_unit_basis": stats[f"{b}/DOM"]["cost_unit_basis"],
            "fixed_arm_frontier": sorted(p.policy_id for p in pareto_frontier(pts)),
            "rep_oracle": {"sr": oracle_usd["sr"], "mean_usd": oracle_usd["mean_usd"]},
        }

    ids, sha = expected_scored_ids(SITE)
    return {
        "header": HEADER,
        "site": SITE,
        "task_universe": {"n": len(ids), "sha256": sha},
        "arm_stats": stats,
        "sr_decomposition": decomposition,
        "cross_backbone_planes": planes,
        "usd_planes_within_backbone": usd_planes,
        "notes": [
            "USD never pooled across backbones (api_usd vs electricity bases).",
            "Oracle = metric-cheapest-success else metric-cheapest (hindsight).",
            "All quantities replayed from Pass-1 trajectories; no live routing.",
        ],
    }


def render_md(result: dict[str, Any]) -> str:
    d = result["sr_decomposition"]
    lines = [
        f"# Cross-object Pareto decomposition — {HEADER}",
        "",
        f"Site {result['site']}, task universe n={result['task_universe']['n']} "
        f"(sha `{result['task_universe']['sha256'][:12]}`).",
        "",
        "## SR decomposition (routed-object comparison, unit-free axis)",
        "",
        "| Choice axis | SR | Solved |",
        "|---|---:|---:|",
        f"| Best single arm ({d['best_single_arm']['arm']}) | {100*d['best_single_arm']['sr']:.2f}% | — |",
    ]
    for b, v in d["representation_only_oracle_per_backbone"].items():
        lines.append(f"| Representation-only oracle within {b} (6 modes) | {100*v['sr']:.2f}% | {v['n_solved']} |")
    for m, v in d["model_only_oracle_per_mode"].items():
        lines.append(f"| Model-only oracle at fixed {m} (3 backbones) | {100*v['sr']:.2f}% | {v['n_solved']} |")
    j = d["joint_18_arm_oracle"]
    lines.append(f"| Joint 18-arm oracle | {100*j['sr']:.2f}% | {j['n_solved']} |")
    mg = d["marginals_pp"]
    lines += [
        "",
        "## Marginal headroom (pp)",
        "",
        f"- Representation choice within B0, over best single arm: **{mg['rep_within_B0_over_best_arm']:+.2f}pp**",
        f"- Model choice added on top of B0 representation oracle: **{mg['model_added_on_top_of_B0_rep']:+.2f}pp**",
        f"- Representation added on top of best model-only choice ({mg['best_model_only_mode']}): "
        f"**{mg['rep_added_on_top_of_best_model_choice']:+.2f}pp**",
        f"- Tasks solved by B1/B2 but by no B0 mode: {len(d['union_counts']['added_by_B1_B2_beyond_B0'])} "
        f"(IDs {d['union_counts']['added_by_B1_B2_beyond_B0']})",
        "",
        "## Cross-backbone Pareto frontiers (unit-comparable axes)",
        "",
    ]
    for metric, plane in result["cross_backbone_planes"].items():
        o = plane["joint_oracle"]
        lines.append(
            f"- **{metric}** ({plane['unit']}): fixed-arm frontier = {plane['fixed_arm_frontier']}; "
            f"joint oracle {100*o['sr']:.2f}% @ {o[f'mean_{metric}']:.1f}"
        )
    lines += ["", "## USD planes (within-backbone only)", ""]
    for b, plane in result["usd_planes_within_backbone"].items():
        o = plane["rep_oracle"]
        lines.append(
            f"- **{b}** ({plane['cost_unit_basis']}): frontier = {plane['fixed_arm_frontier']}; "
            f"representation oracle {100*o['sr']:.2f}% @ ${o['mean_usd']:.4f}/task"
        )
    lines += ["", "### Notes", ""] + [f"- {n}" for n in result["notes"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    arms = load_arms()
    result = build(arms)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "cross_object_pareto.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    (OUT_DIR / "cross_object_pareto.md").write_text(render_md(result))
    print(render_md(result))


if __name__ == "__main__":
    main()
