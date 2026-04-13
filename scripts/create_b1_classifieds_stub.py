"""
create_b1_classifieds_stub.py
-----------------------------
从保留的分析文档（docs/analysis/classifieds/）恢复 classifieds B1 条件 stub。

背景：
  §34 中 `--clean` 误删了 classifieds 的全部 step-level 数据，但
  analysis 文档完整保留。本脚本将已知的 SR/cost 数字写入
  condition_summary_v2.json，使 is_condition_complete() 视其为完成，
  同时为分析管线提供有效的条件级汇总。

注意：
  - success_rate 使用 RAW SR（评测器直接输出），adjusted SR 由分析脚本计算
  - 成本字段为 0（classifieds B1 跑在 §35 成本模型补全之前，无成本 instrumentation）
  - 所有 episode-level 数据缺失，cross-representation 分析需从此 JSON 读取汇总数字

用法：
  python3 scripts/create_b1_classifieds_stub.py [--run_id B1_3mode_classifieds_20260413]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ---------- 已知数字（来自 docs/analysis/classifieds/ 和 MEMORY.md）----------
# Raw SR：评测器直接输出，未过滤 visual FP
# 来源：B1_findings.md + B1_*_digest.md
CLASSIFIEDS_KNOWN = {
    "dom": {
        "episodes": 234,
        "success_rate": 21 / 234,        # 8.97% raw（adjusted 0.85%，2/234）
        "avg_steps": 14.9,
        "avg_total_model_cost_usd": 0.0,  # 无成本 instrumentation
        "avg_total_cost_usd": 0.0,
        "observation_mode": "dom",
        "condition_id": "phase1_dom_router_0",
        "som_on": False,
        "_note": "Stub from analysis docs (§34 data loss). Raw SR=21/234. Adjusted SR=2/234 (visual FP filtered).",
    },
    "som": {
        "episodes": 234,
        "success_rate": 48 / 234,        # 20.51% raw（adjusted 16.24%，38/234）
        "avg_steps": 15.2,
        "avg_total_model_cost_usd": 0.0,
        "avg_total_cost_usd": 0.0,
        "observation_mode": "som",
        "condition_id": "phase1_som_router_0",
        "som_on": True,
        "_note": "Stub from analysis docs (§34 data loss). Raw SR=48/234. Adjusted SR=38/234 (visual FP filtered).",
    },
    "vision": {
        "episodes": 234,
        "success_rate": 29 / 234,        # 12.39% raw（adjusted 8.12%，19/234）
        "avg_steps": 21.3,
        "avg_total_model_cost_usd": 0.0,
        "avg_total_cost_usd": 0.0,
        "observation_mode": "vision",
        "condition_id": "phase1_vision_router_0",
        "som_on": False,
        "_note": "Stub from analysis docs (§34 data loss). Raw SR=29/234. Adjusted SR=19/234 (visual FP filtered).",
    },
}

# 条件 ID 映射
CONDITION_IDS = {
    "dom":    "phase1_dom_router_0",
    "som":    "phase1_som_router_0",
    "vision": "phase1_vision_router_0",
}

# ---------- condition_summary_v2.json 完整模板 ----------
def make_condition_summary(mode: str, known: dict) -> dict:
    """构建符合 aggregate_condition_metrics() 输出格式的 stub JSON。"""
    return {
        # --- aggregate_condition_metrics 标准字段 ---
        "episodes": known["episodes"],
        "success_rate": round(known["success_rate"], 6),
        "avg_steps": known["avg_steps"],
        "p95_step_latency_ms": 0.0,           # 无计时数据
        "avg_total_model_cost_usd": known["avg_total_model_cost_usd"],
        "avg_total_cost_usd": known["avg_total_cost_usd"],
        "avg_router_overhead_cost_usd": 0.0,
        "avg_total_energy_kwh": None,
        "avg_total_co2e_kg": None,
        "avg_retries": 0.0,
        "avg_no_op_rate": 0.0,
        "avg_page_unchanged_rate": 0.0,
        "avg_escalation_count": 0.0,
        "trigger_distribution": {},
        "state_change_reason_distribution": {},
        "avg_checklist_completion_rate": None,
        "checklist_failure_episode_rate": None,
        "benchmark_noise_rate": 0.0,
        "wasted_energy_kwh": None,
        "avg_wasted_cost_usd": 0.0,
        "avg_wasted_energy_kwh": 0.0,
        "cost_efficiency_ratio": None,
        # --- runner.py 追加的字段 ---
        "condition_id": known["condition_id"],
        "seed": 0,
        "phase": "phase1",
        "backend_id": "local_4b",
        "som_on": known["som_on"],
        "observation_mode": known["observation_mode"],
        "router_on": False,
        "module_flags": {},
        # --- stub 元数据（分析脚本应忽略未知字段）---
        "_stub": True,
        "_stub_note": known["_note"],
        "_stub_source": "docs/analysis/classifieds/B1_findings.md",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="创建 classifieds B1 条件 stub")
    parser.add_argument(
        "--run_id",
        default="B1_3mode_classifieds_20260413",
        help="stub 使用的 run_id（默认: B1_3mode_classifieds_20260413）",
    )
    parser.add_argument(
        "--results_base",
        default=None,
        help="results 根目录（默认: <repo_root>/results/visualwebarena/phase1）",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅打印，不写文件",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    results_base = Path(args.results_base) if args.results_base else (
        repo_root / "results" / "visualwebarena" / "phase1"
    )
    run_dir = results_base / args.run_id

    print(f"[stub] run_dir = {run_dir}")
    print(f"[stub] dry_run = {args.dry_run}")
    print()

    for mode, known in CLASSIFIEDS_KNOWN.items():
        cid = CONDITION_IDS[mode]
        cond_dir = run_dir / cid
        summary_path = cond_dir / "condition_summary_v2.json"

        payload = make_condition_summary(mode, known)

        print(f"[stub] {mode}: SR={payload['success_rate']:.4f} ({known['episodes']} ep)")
        print(f"       -> {summary_path}")

        if summary_path.exists():
            print(f"       [SKIP] 已存在，使用 --force 覆盖")
            if not getattr(args, "force", False):
                continue

        if not args.dry_run:
            cond_dir.mkdir(parents=True, exist_ok=True)
            # 创建空 episodes 目录（保持目录结构完整）
            (cond_dir / "episodes").mkdir(exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"       [OK] 已写入")
        else:
            print(f"       [DRY] 跳过写入")
        print()

    if not args.dry_run:
        # 写 condition_meta.json（runner 期望存在）
        for mode, known in CLASSIFIEDS_KNOWN.items():
            cid = CONDITION_IDS[mode]
            cond_dir = run_dir / cid
            meta_path = cond_dir / "condition_meta.json"
            if not meta_path.exists():
                meta = {
                    "condition_id": cid,
                    "observation_mode": mode,
                    "backend_id": "local_4b",
                    "site": "classifieds",
                    "_stub": True,
                }
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

        print(f"[stub] 完成。run_id={args.run_id}")
        print(f"[stub] is_condition_complete() 将对三个 condition 均返回 True")
        print(f"[stub] 分析脚本可读取 condition_summary_v2.json 获取已知 SR 数字")
    else:
        print("[stub] dry_run 完成，未写入任何文件")


if __name__ == "__main__":
    main()
