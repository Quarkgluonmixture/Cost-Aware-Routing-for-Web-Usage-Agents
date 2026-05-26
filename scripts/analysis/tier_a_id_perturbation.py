#!/usr/bin/env python3
"""Tier A — B1 (dense) element-id 扰动因果实验.

喂 B1 两版 step-0 obs (archive-id vs current-id, modulo-id 字节一致, 只差 [id=N] 数字),
B1 是 dense + temp=0 → 确定性 → 任何 greedy 决策翻转 = 纯 element-id 因果 (零 MoE 混淆).
= §282 deferred 的 replay 实验, 在 dense 模型上干净版 (实证 2026-05-26, 见 笔记 §298).

determinism 自检: 同 obs 喂两次 → 决策应一致 (验证 B1 在本机确定).

Usage:
  # DGX (默认路径 — current Phase 1a + repro_replicates 干净 replicate):
  .venv/bin/python3 scripts/analysis/tier_a_id_perturbation.py all   # 全 common task
  .venv/bin/python3 scripts/analysis/tier_a_id_perturbation.py 10 16 60  # 指定 task

  # Myriad / 其他机 — env 覆盖路径:
  P79_TIER_A_ARCH=/path/to/archive_run \
  P79_TIER_A_CURR=/path/to/current_run \
  python3 scripts/analysis/tier_a_id_perturbation.py all

DGX GB10 注意: 有 force_gpu_patch monkeypatch 绕过 device_map='auto' + 'Not Supported' 显存
查询 → CPU offload 坑 (强制 device_map={'':0})。A100 上 auto 本身就工作, patch 等价无害。
"""
from __future__ import annotations
import os, sys, json, re
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from p79.experiment.config import load_experiment_config  # noqa: E402
from p79.agents.qwen3vl_agent import Qwen3VLAgent  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

SITE = os.environ.get("P79_TIER_A_SITE", "classifieds")
ARCH = Path(os.environ.get(
    "P79_TIER_A_ARCH",
    str(REPO / "results/repro_replicates/B0_dom_classifieds_R31194_clean_replicate/phase1_dom_router_0"),
))
CURR = Path(os.environ.get(
    "P79_TIER_A_CURR",
    str(REPO / "results/visualwebarena/phase1/B0_dom_classifieds_20260525_194618_553890342_530647_R21557/phase1_dom_router_0"),
))
CFG = REPO / f"external/visualwebarena/config_files/vwa/test_{SITE}"
VWA = REPO / "external/visualwebarena"


def obs_text(cond: Path, tid: int):
    p = cond / "artifacts" / f"{SITE}_task_{tid}" / "step_000" / "observation_dom.txt"
    return p.read_text() if p.exists() else None


def task_meta(tid: int):
    j = json.loads((CFG / f"{tid}.json").read_text())
    intent = j.get("intent", "")
    imgs = []
    raw = j.get("image")
    if raw:
        for p in ([raw] if isinstance(raw, str) else raw):
            ip = VWA / p
            if ip.exists():
                im = PILImage.open(str(ip)).convert("RGB")
                if max(im.size) > 1024:
                    r = 1024 / max(im.size)
                    im = im.resize((int(im.size[0] * r), int(im.size[1] * r)))
                imgs.append(im)
    return intent, imgs


EIDLINE = re.compile(r"^\s*\[(\d+)\]\s*(.*)$")


def resolve(obs: str, eid):
    if eid is None:
        return None
    for ln in obs.splitlines():
        m = EIDLINE.match(ln)
        if m and int(m.group(1)) == int(eid):
            return m.group(2).strip()[:60]
    return f"<eid {eid} 不在 obs>"


def dsig(action: dict, obs: str):
    """id-agnostic decision signature: action_type + 物理目标 (按 obs 解析 eid)."""
    at = action.get("action_type")
    if at == "type":
        return (at, (action.get("text") or "").strip()[:40])
    if at in ("click", "hover", "select_option"):
        return (at, resolve(obs, action.get("element_id")), (action.get("option_label") or "")[:20])
    if at == "scroll":
        return (at, action.get("scroll_direction"))
    return (at,)


def common_tasks():
    """archive ∩ current 已完成 (both not sr_excluded), 用于全量 run."""
    def done(cond):
        out = set()
        for f in (cond / "episodes").glob(f"{SITE}_task_*_summary_v2.json"):
            try:
                s = json.loads(f.read_text())
            except Exception:
                continue
            if s.get("task_id") is not None and not s.get("sr_excluded", False):
                out.add(int(s["task_id"]))
        return out
    return sorted(done(ARCH) & done(CURR))


def force_gpu_patch():
    """绕过 agent 硬编 device_map='auto' (DGX GB10 显存查询 'Not Supported' → CPU offload).
    强制 device_map={'':0} 全放 GPU0. 仅 harness 改, 不碰 production agent.
    A100 上 auto 本身工作, patch 等价无害 (都是单 GPU 0 放置)."""
    import p79.agents.qwen3vl_agent as qa
    _orig = qa.Qwen3VLForConditionalGeneration.from_pretrained

    def _forced(*a, **k):
        k["device_map"] = {"": 0}
        return _orig(*a, **k)

    qa.Qwen3VLForConditionalGeneration.from_pretrained = _forced


def main():
    arg = sys.argv[1:]
    if arg and arg[0] == "all":
        tasks = common_tasks()
    else:
        tasks = [int(x) for x in arg] or [10, 16, 60, 61, 79]
    cfg_path = os.environ.get("P79_TIER_A_CFG", str(REPO / "configs/exp_v2_B1_dom_classifieds.yaml"))
    cfg = load_experiment_config(cfg_path)
    print(f"[load] forcing GPU0 + instantiating B1 ... ({len(tasks)} tasks)", flush=True)
    print(f"[paths] ARCH={ARCH}")
    print(f"[paths] CURR={CURR}", flush=True)
    force_gpu_patch()
    agent = Qwen3VLAgent(cfg)
    dev = next(agent.model.parameters()).device
    print(f"[load] B1 ready | model device={dev} | hf_device_map={getattr(agent.model, 'hf_device_map', None)}", flush=True)
    if dev.type != "cuda":
        print("[ABORT] 模型仍不在 GPU — 强制失败, 停")
        return
    ndet = nflip = n = 0
    for tid in tasks:
        oa, oc = obs_text(ARCH, tid), obs_text(CURR, tid)
        if not oa or not oc:
            print(f"task {tid}: obs 缺失, skip")
            continue
        intent, imgs = task_meta(tid)
        a1, _ = agent.step(intent, SimpleNamespace(text=oa, image=None), observation_mode="dom", reference_images=imgs)
        a1b, _ = agent.step(intent, SimpleNamespace(text=oa, image=None), observation_mode="dom", reference_images=imgs)
        ac, _ = agent.step(intent, SimpleNamespace(text=oc, image=None), observation_mode="dom", reference_images=imgs)
        sa, sab, sc = dsig(a1, oa), dsig(a1b, oa), dsig(ac, oc)
        det, flip = (sa == sab), (sa != sc)
        ndet += det; nflip += flip; n += 1
        print(f"task {tid:3d} | det={'OK' if det else '✗NONDET'} | id-flip={'★FLIP' if flip else 'same'} | imgs={len(imgs)}")
        print(f"    arch: {sa}")
        print(f"    curr: {sc}")
        if not det:
            print(f"    [⚠️ nondet] arch2: {sab}")
    print(f"\n=== n={n} | determinism OK {ndet}/{n} | id-perturbation FLIP {nflip}/{n} ===")


if __name__ == "__main__":
    main()
