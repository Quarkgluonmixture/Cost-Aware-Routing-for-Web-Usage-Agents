#!/usr/bin/env python3
"""Cross-run same-condition task-level SR comparison + flip 根因三层分解.

对比**同一 condition** (site, model, mode) 在两个 run 之间的 SR,在 task 交集上
做公平比较 (两边都 not sr_excluded),并把每个 success-flip 归因到三层 (笔记 §282/§292):

  1. element_id flip (decision-harmless): step-0 同一 bbox (逐像素) 但 element_id 数字不同
     → agent 引用同一物理元素 = decision-level convergent (B-12 / B-1858 element-ID 非确定);
     **不破坏 SR**, 仅作附注。
  2. model nondeterm: step-0 decision 相同, 后续步 bbox 真分叉 (B0 多候选 listing 时 B-37
     非确定选不同分支) → 轨迹岔开。run-to-run noise floor 的主因。
  3. start-url mismatch: step-0 obs_url 与 config start_url 不一致 (且两 run 起始页不同)
     → env reset-goto 落点异常 (B-1581-adjacent reset fragility), 非 model/element_id。

典型用途: pre/post code-fix 一致性 check (e.g. B-1860 restart 前后 ptext archive↔current),
run-to-run reproducibility 审计, 或 restart 后验证 fix 无副作用。

Usage:
  .venv/bin/python3 scripts/analysis/compare_cross_run_same_condition.py \\
    --archive-run results/visualwebarena/phase1/_archive_b1860coord_R19776_ptext_partial180_20260525 \\
    --current-run results/visualwebarena/phase1/B0_phantom_text_classifieds_..._R2647 \\
    --site classifieds

run dir 可以是顶层 run dir (自动找 phase1_*_router_0 condition subdir) 或 condition dir 本身。
支持 partial run (current 仍在跑 → 交集自动取已完成 task)。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p79.experiment.io_utils import read_jsonl_dedup  # noqa: E402


def find_condition_dir(run_dir: Path) -> Path:
    """run dir 下找 condition subdir (有 episodes/); run_dir 本身是 condition dir 也接受。"""
    if (run_dir / "episodes").is_dir():
        return run_dir
    cands = sorted(run_dir.glob("phase1_*_router_0"))
    if not cands:
        cands = [d for d in run_dir.iterdir() if d.is_dir() and (d / "episodes").is_dir()]
    if not cands:
        raise SystemExit(f"no condition subdir with episodes/ under {run_dir}")
    return cands[0]


def load_summaries(cond_dir: Path, site: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for f in sorted((cond_dir / "episodes").glob(f"{site}_task_*_summary_v2.json")):
        try:
            s = json.loads(f.read_text())
        except Exception:
            continue
        tid = s.get("task_id")
        if tid is not None:
            out[int(tid)] = s
    return out


def load_steps(cond_dir: Path, site: str, tid: int) -> List[Dict[str, Any]]:
    p = cond_dir / "episodes" / f"{site}_task_{tid}_steps_v2.jsonl"
    return read_jsonl_dedup(p) if p.exists() else []


def decision_sig(r: Dict[str, Any]) -> Tuple:
    """decision signature: 忽略 element_id 数字, 用 action_type + bbox + scroll/text/answer。
    这是 §292 的核心 — 按物理 decision 比, 不按 thought 字符串 (后者含 eid 数字+措辞抖动 → 假分叉)。"""
    act = r.get("action") or {}
    atype = act.get("action_type") or r.get("action_type")
    bbox = r.get("element_bbox")
    bb = tuple(round(float(x)) for x in bbox) if bbox else None
    extra = ""
    if act.get("scroll_direction"):
        extra = f"scroll:{act['scroll_direction']}"
    elif act.get("text"):
        extra = f"text:{str(act['text'])[:24]}"
    elif isinstance(act.get("answer"), str):
        extra = f"ans:{act['answer'][:24]}"
    return (atype, bb, extra)


def norm_path(url: Optional[str]) -> str:
    """归一化到 path+query (去 scheme+host + __SITE__ placeholder), 用于 url_before↔start_url 比较。"""
    if not url:
        return ""
    u = re.sub(r"^https?://[^/]+", "", url)   # http://host:port/x → /x
    u = re.sub(r"^__[A-Z_]+__", "", u)        # __CLASSIFIEDS__/x → /x
    return u or "/"


def step0_landing(steps: List[Dict[str, Any]]) -> Optional[str]:
    """step-0 的 action-**前** url (reset 落点) = state_digest.url_before。
    这是真起始页; obs_url=state_digest.url_after 是 action **后** url (反映 agent 行为, 不能判起始污染)。"""
    if not steps:
        return None
    return (steps[0].get("state_digest") or {}).get("url_before")


def load_start_url(config_dir: Optional[Path], tid: int) -> Optional[str]:
    """读 VWA task config 原始 start_url (含 __SITE__ placeholder; 统一用 norm_path 归一化后再比)。"""
    if not config_dir:
        return None
    p = config_dir / f"{tid}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text()).get("start_url") or None
    except Exception:
        return None


def first_decision_diverge(sa: List[Dict], sc: List[Dict]) -> Optional[int]:
    m = min(len(sa), len(sc))
    for k in range(m):
        if decision_sig(sa[k]) != decision_sig(sc[k]):
            return k
    return m if len(sa) != len(sc) else None


def sr(summaries: Dict[int, Dict], ids: List[int]) -> Tuple[int, int]:
    scored = [summaries[i] for i in ids if not summaries[i].get("sr_excluded", False)]
    return sum(1 for s in scored if s.get("success")), len(scored)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--archive-run", required=True, type=Path, help="baseline/old run dir (or condition dir)")
    ap.add_argument("--current-run", required=True, type=Path, help="new run dir (or condition dir)")
    ap.add_argument("--site", default="classifieds", choices=["classifieds", "reddit", "shopping"])
    ap.add_argument(
        "--config-dir", type=Path, default=None,
        help="VWA task config dir for start_url scan (default: external/.../test_<site>; pass nonexistent to skip)",
    )
    ap.add_argument("--max-flip-detail", type=int, default=30)
    args = ap.parse_args()

    a_dir = find_condition_dir(args.archive_run)
    c_dir = find_condition_dir(args.current_run)
    cfg_dir = args.config_dir or (ROOT / f"external/visualwebarena/config_files/vwa/test_{args.site}")
    cfg_dir = cfg_dir if cfg_dir.exists() else None

    a = load_summaries(a_dir, args.site)
    c = load_summaries(c_dir, args.site)
    common = sorted(set(a) & set(c))
    common_scored = [i for i in common if not a[i].get("sr_excluded") and not c[i].get("sr_excluded")]

    print(f"archive : {a_dir}")
    print(f"          {len(a)} tasks" + (f" (range {min(a)}..{max(a)})" if a else ""))
    print(f"current : {c_dir}")
    print(f"          {len(c)} tasks" + (f" (range {min(c)}..{max(c)})" if c else ""))
    print(f"common scored (both not sr_excluded): {len(common_scored)} / {len(common)} intersection")
    if cfg_dir is None:
        print("(start_url scan skipped — config dir not found)")
    print()

    a_s, n = sr(a, common_scored)
    c_s, _ = sr(c, common_scored)
    if n:
        print(f"SR archive : {a_s}/{n} = {a_s / n * 100:.1f}%")
        print(f"SR current : {c_s}/{n} = {c_s / n * 100:.1f}%")
        print(f"Δ          : {(c_s - a_s) / n * 100:+.1f}pp")
    ax = {i for i in common if a[i].get("sr_excluded")}
    cx = {i for i in common if c[i].get("sr_excluded")}
    print(f"sr_excluded mismatch: {sorted(ax ^ cx) or 'none'}")
    print()

    flips = [i for i in common_scored if bool(a[i].get("success")) != bool(c[i].get("success"))]
    print(f"=== FLIPS: {len(flips)}/{n} ===")
    counts = {"model_nondeterm": 0, "start_url_mismatch": 0, "unclassified": 0}
    eid_flip = 0
    for tid in flips[: args.max_flip_detail]:
        sa, sc = load_steps(a_dir, args.site, tid), load_steps(c_dir, args.site, tid)
        av, cv = bool(a[tid].get("success")), bool(c[tid].get("success"))
        # 起始污染判定必须用 url_before (reset 落点); obs_url=action后 反映 agent 行为不能用 (§292 教训)
        a0, c0 = norm_path(step0_landing(sa)), norm_path(step0_landing(sc))
        if a0 and c0 and a0 != c0:
            klass, note = "start_url_mismatch", f"step0-landing archive={a0[:26]} current={c0[:26]}"
            su = load_start_url(cfg_dir, tid)
            if su:
                note += f" cfg={norm_path(su)[:26]}"
        else:
            div = first_decision_diverge(sa, sc)
            klass = "model_nondeterm" if div is not None else "unclassified"
            note = f"decision diverge @ step {div}"
        eid_note = ""
        if sa and sc:
            ba, bb = sa[0].get("element_bbox"), sc[0].get("element_bbox")
            ea = (sa[0].get("action") or {}).get("element_id")
            eb = (sc[0].get("action") or {}).get("element_id")
            if ba and bb and ba == bb and ea != eb:
                eid_note = f"  [eid flip step0 {ea}→{eb} same bbox=harmless]"
                eid_flip += 1
        counts[klass] += 1
        print(f"  task {tid:3d}: {'PASS' if av else 'fail'}→{'PASS' if cv else 'fail'} | {klass:18s} | {note}{eid_note}")

    print(f"\n=== reset-goto run-to-run scan (两 run step-0 landing[url_before] 是否一致, ALL {len(common)} common) ===")
    print("  (检测真起始页 run-to-run 异常; landing vs config 的确定性 osclass redirect 如 iPage/sShowAs 简化不算异常)")
    anomalies = []
    for tid in common:
        la = norm_path(step0_landing(load_steps(a_dir, args.site, tid)))
        lc = norm_path(step0_landing(load_steps(c_dir, args.site, tid)))
        if la and lc and la != lc:
            anomalies.append((tid, la, lc))
    if anomalies:
        for tid, la, lc in anomalies:
            print(f"  task {tid}: archive landing={la[:32]} ≠ current landing={lc[:32]}")
    else:
        print("  none (两 run step-0 landing 一致 → reset goto run-to-run 稳定, 无起始污染)")

    print("\n=== 三层分解 (flip subset) ===")
    print(f"  model_nondeterm (true divergence) : {counts['model_nondeterm']}")
    print(f"  start_url_mismatch (reset-goto)   : {counts['start_url_mismatch']}")
    print(f"  unclassified                      : {counts['unclassified']}")
    print(f"  └ (其中 step-0 含 element_id flip 但 decision-harmless: {eid_flip})")


if __name__ == "__main__":
    main()
