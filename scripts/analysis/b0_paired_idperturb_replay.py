#!/usr/bin/env python3
"""路线 A — paired id-perturbation step-0 replay (offline). B0(AWS) + B1(local) 两层对照.

§302 RETRACT 了 cross-model 减法 (B0 总 noise − B1 id-channel): 235B-MoE 与 4B-dense
对 id token 的敏感度不可比 (category error)。本脚本改用 **paired** 隔离 element-id churn:
同一模型、同一 step-0 obs，只对 element-id 做受控扰动，配对差分把背景 noise 当 common
background 消掉。

  组 A (id-fixed):     current obs 原样,       跑 N 次
  组 B (id-perturbed): obs 行首 [N] within-obs shuffle (保 role/name/bbox/行序), 跑 N 次

  - id-agnostic 比较 (resolve/dsig): element_id 数字变了但点同一物理元素 = 不算 flip;
    只有点不同物理元素才算 id-churn 改变决策 (= element-id noise 的正确定义)。
  - mode_flip = (组A众数物理决策 ≠ 组B众数物理决策) = id 扰动改变了"去背景-noise 后的真决策"。
  - consistency_A/B = 组内众数占比 (组内一致性)。

两层对照:
  - B0 (AWS proxy, MoE): 有 serving 不确定性 → consistency_A < 1 = serving floor;
    id 边际 = consistency_drop (B−A) + mode_flip (受 serving 污染)。
  - B1 (local Qwen3-VL-4B, temp=0): determinism → consistency_A 应 = 1.0 (无 serving);
    组A≠组B (mode_flip) = **纯 id 效应** (§298 archive-churn 测得 ~10.5%, 此处 synthetic shuffle)。

NOT a VWA experiment: B0 纯 outbound HTTPS → AWS proxy; B1 纯本地 GPU forward。都读 cached
artifacts, 不连 live 站点。A100-down-safe (fire 在 A100; B0 用 AWS quota, B1 用 DGX/Myriad GPU)。

Usage:
  # B0 无偏大 N (压 serving floor):
  PYTORCH_NVML_BASED_CUDA_CHECK=1 .venv/bin/python3 scripts/analysis/b0_paired_idperturb_replay.py \
    --baseline B0 --task-ids sample:40 --n 12 --label b0_unbiased
  # B1 local 对照 (determinism, 小 N 够):
  PYTORCH_NVML_BASED_CUDA_CHECK=1 .venv/bin/python3 scripts/analysis/b0_paired_idperturb_replay.py \
    --baseline B1 --task-ids sample:40 --n 3 --label b1_paired
"""
from __future__ import annotations
import os, sys, re, json, time, random, argparse, datetime as _dt, statistics as st
from pathlib import Path
from types import SimpleNamespace
from collections import Counter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from p79.experiment.config import load_experiment_config  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

SITE = os.environ.get("P79_IDPERT_SITE", "classifieds")
CURR = Path(os.environ.get(
    "P79_IDPERT_RUN",
    str(REPO / "results/visualwebarena/phase1/"
        "B0_dom_classifieds_20260525_194618_553890342_530647_R21557/phase1_dom_router_0"),
))
CFG = REPO / f"external/visualwebarena/config_files/vwa/test_{SITE}"
VWA = REPO / "external/visualwebarena"
B0_CFG = os.environ.get("P79_IDPERT_CFG", str(REPO / "configs/exp_v2_B0_dom_classifieds.yaml"))
B1_CFG = str(REPO / "configs/exp_v2_B1_dom_classifieds.yaml")

EIDLINE = re.compile(r"^\s*\[(\d+)\]\s*(.*)$")


def _ensure_key():
    # B0 yaml 的 base_url 在 model 子层; 单独 load 不一定合并到 ProxyApiAgent 读取的层级
    # → 显式补 PROXY_API_ENDPOINT fallback (proxy_api_agent.py:46 OR-chain)。
    os.environ.setdefault(
        "PROXY_API_ENDPOINT",
        "https://i5xpracyci.execute-api.eu-west-2.amazonaws.com/model-api/invoke")
    if os.environ.get("PROXY_API_KEY"):
        return
    auth = REPO / ".auth/qwen_api"
    if auth.exists():
        for line in auth.read_text().splitlines():
            line = line.strip()
            if line.startswith("rp_"):
                os.environ["PROXY_API_KEY"] = line
                return


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


def permute_ids(obs: str, seed: int) -> str:
    """对 obs 行首 [N] 做 within-obs 双射 shuffle: 保 role/name/bbox/行序, 只改 id 数字。
    用真实 nodeId 值域 (shuffle 现有 id 集合) → 模拟 churn (同物理元素拿不同 id)。"""
    lines = obs.splitlines()
    eid_idx = [i for i, ln in enumerate(lines) if EIDLINE.match(ln)]
    ids = [int(EIDLINE.match(lines[i]).group(1)) for i in eid_idx]
    shuffled = ids[:]
    random.Random(seed).shuffle(shuffled)
    for k, i in enumerate(eid_idx):
        lines[i] = re.sub(r"^(\s*)\[\d+\]", rf"\g<1>[{shuffled[k]}]", lines[i], count=1)
    return "\n".join(lines)


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
    if at == "finish":
        return (at, (action.get("answer") or "")[:30])
    return (at,)


def common_tasks() -> list:
    out = set()
    for f in (CURR / "episodes").glob(f"{SITE}_task_*_summary_v2.json"):
        try:
            s = json.loads(f.read_text())
        except Exception:
            continue
        if s.get("task_id") is not None and not s.get("sr_excluded", False):
            out.add(int(s["task_id"]))
    return sorted(out)


def pick_tasks(spec: str) -> list:
    """显式 list | 'all' | 'sample:K' (seed=42 固定, B0/B1 同 sample apples-to-apples)."""
    if spec == "all":
        return common_tasks()
    if spec.startswith("sample:"):
        k = int(spec.split(":", 1)[1])
        pool = common_tasks()
        return sorted(random.Random(42).sample(pool, min(k, len(pool))))
    return [int(x) for x in spec.split(",") if x.strip()] or [10, 16, 60, 61, 79]


def load_agent(baseline: str):
    """B0 = AWS proxy (ProxyApiAgent); B1 = local Qwen3-VL-4B (Qwen3VLAgent + GB10 GPU patch).
    B1 temp=0 → determinism → 组A consistency 应=1.0 (无 serving); 组A≠组B = 纯 id 效应。"""
    cfg = load_experiment_config(B1_CFG if baseline == "B1" else B0_CFG)
    if baseline == "B1":
        import p79.agents.qwen3vl_agent as qa
        _orig = qa.Qwen3VLForConditionalGeneration.from_pretrained

        def _forced(*a, **k):  # GB10: 强制 device_map={'':0} 全放 GPU0 (tier_a 同 patch)
            k["device_map"] = {"": 0}
            return _orig(*a, **k)

        qa.Qwen3VLForConditionalGeneration.from_pretrained = _forced
        agent = qa.Qwen3VLAgent(cfg)
        dev = next(agent.model.parameters()).device
        print(f"[load] B1 ready | device={dev}", flush=True)
        if dev.type != "cuda":
            raise RuntimeError(f"B1 模型不在 GPU (device={dev}) — 停")
        return agent
    _ensure_key()
    from p79.agents.proxy_api_agent import ProxyApiAgent
    print("[load] B0 ready (AWS proxy)", flush=True)
    return ProxyApiAgent(cfg)


def replay_n(agent, intent, obs, imgs, n):
    acts = []
    for _ in range(n):
        try:
            a, _meta = agent.step(intent, SimpleNamespace(text=obs, image=None),
                                  observation_mode="dom", reference_images=imgs)
            acts.append(a if isinstance(a, dict) else {"action_type": "NONDICT"})
        except Exception as e:
            acts.append({"action_type": f"ERR:{type(e).__name__}"})
    return acts


def analyze_task(agent, tid, n):
    obs = obs_text(CURR, tid)
    if not obs:
        return {"task_id": tid, "skip": "obs 缺失"}
    intent, imgs = task_meta(tid)
    obs_B = permute_ids(obs, seed=tid)
    acts_A = replay_n(agent, intent, obs, imgs, n)
    acts_B = replay_n(agent, intent, obs_B, imgs, n)
    dA = [dsig(a, obs) for a in acts_A if a]
    dB = [dsig(a, obs_B) for a in acts_B if a]
    cA, cB = Counter(dA), Counter(dB)
    modeA, nA = cA.most_common(1)[0] if cA else (None, 0)
    modeB, nB = cB.most_common(1)[0] if cB else (None, 0)
    return {
        "task_id": tid, "n": n, "intent": intent[:80],
        "consistency_A_serving": round(nA / len(dA), 3) if dA else None,
        "consistency_B_serving_plus_id": round(nB / len(dB), 3) if dB else None,
        "unique_A_serving": len(cA),
        "unique_B_serving_plus_id": len(cB),
        "mode_flip_id_changed_decision": (modeA != modeB),
        "mode_A": str(modeA), "mode_B": str(modeB),
        "dsigs_A": [str(x) for x in dA],
        "dsigs_B": [str(x) for x in dB],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="B0", choices=["B0", "B1"])
    ap.add_argument("--task-ids", default="", help="comma-sep ids, 'all', or 'sample:K'")
    ap.add_argument("--n", type=int, default=8, help="replays per group (A and B each)")
    ap.add_argument("--out-dir", default="docs/checkpoints/probes")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    if args.baseline == "B0":
        _ensure_key()
        if not os.environ.get("PROXY_API_KEY"):
            print("ERROR: PROXY_API_KEY unset + .auth/qwen_api 无 rp_ key", file=sys.stderr)
            return 1

    tids = pick_tasks(args.task_ids.strip())
    agent = load_agent(args.baseline)
    print(f"[load] {args.baseline} | run={CURR.name} | tasks={len(tids)} | N={args.n}/group", flush=True)

    results = []
    t0 = time.time()
    for tid in tids:
        r = analyze_task(agent, tid, args.n)
        results.append(r)
        if "skip" in r:
            print(f"task {tid}: skip ({r['skip']})", flush=True)
            continue
        print(f"task {tid:3d} | consistA={r['consistency_A_serving']} "
              f"consistB={r['consistency_B_serving_plus_id']} "
              f"| uniqA={r['unique_A_serving']} uniqB={r['unique_B_serving_plus_id']} "
              f"| id-mode-flip={'★YES' if r['mode_flip_id_changed_decision'] else 'no'}", flush=True)

    wall = int(time.time() - t0)
    valid = [r for r in results if "skip" not in r]
    agg = {}
    if valid:
        cA = [r["consistency_A_serving"] for r in valid if r["consistency_A_serving"] is not None]
        cB = [r["consistency_B_serving_plus_id"] for r in valid if r["consistency_B_serving_plus_id"] is not None]
        agg = {
            "baseline": args.baseline,
            "n_tasks": len(valid),
            "mean_consistency_A": round(st.mean(cA), 3) if cA else None,
            "mean_consistency_B": round(st.mean(cB), 3) if cB else None,
            "frac_id_mode_flip": round(sum(1 for r in valid if r["mode_flip_id_changed_decision"]) / len(valid), 3),
            "mean_consistency_drop_B_minus_A": round(st.mean(cB) - st.mean(cA), 3) if (cA and cB) else None,
            "flip_task_ids": [r["task_id"] for r in valid if r["mode_flip_id_changed_decision"]],
        }
    summary = {
        "purpose": "路线A paired id-perturbation step-0 replay (serving vs id-churn 拆分)",
        "baseline": args.baseline,
        "ts_utc": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_dir": str(CURR), "n_per_group": args.n, "wall_seconds": wall,
        "aggregate": agg, "results": results,
    }
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    hhmmss = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    lab = f"_{args.label}" if args.label else ""
    out_path = out_dir / f"b0_paired_idperturb_{hhmmss}{lab}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n=== AGG === {json.dumps(agg, ensure_ascii=False)}")
    print(f"output: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
