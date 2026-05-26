#!/usr/bin/env python3
"""Tier B — element-id 扰动在 B1 哪一层 tip 决策 (activation patching layer sweep).

承 Tier A (笔记 §298): 14/133=10.5% step-0 flip 已坐实 id-channel 因果。本 Tier B 用
ActivationPatcher 做 layer-by-layer patching, 定位 id-effect 涌现层:

  对每个 flip task:
    cache: forward(O_arch) → hidden states per layer
    sweep L=0..n_layers-1:
      patched_action = patched_generate(L, source_hidden=cached_arch[L], **curr_inputs)
      if dsig(patched_action) == dsig(arch_action) → 层 L 携带 id-翻转的信息

  → 找最早的 L (id-effect 涌现层); 若没有 L 能 tip → 信息分布式 / 多层叠加。

机制层 read: 若 emergence L 早 (low layer) = id token 影响早期 token-rep; L 晚 (deep) =
高层语义/决策聚合处。Paper §5 advisor 暂搁 → Tier B 仅服务 reproducibility (Risk 6),
不再扩 §5 section。

Usage (单 task 测试):
  python3 scripts/analysis/tier_b_id_patching.py 10

Usage (Myriad qsub 全 14 flip):
  P79_TIER_A_ARCH=... P79_TIER_A_CURR=... python3 tier_b_id_patching.py 10 11 12 16 17 59 64 92 93 94 107 108 118 125

输出: JSON to OUT (default /tmp/tier_b_<tid>.json), 含 baseline + per-layer patched dsig + emergence layer。
"""
from __future__ import annotations
import os, sys, json, re
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from p79.experiment.config import load_experiment_config  # noqa: E402
from p79.agents.qwen3vl_agent import Qwen3VLAgent  # noqa: E402
from p79.mechanistic.activation_patching import ActivationPatcher  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
import torch  # noqa: E402

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
OUT_DIR = Path(os.environ.get("P79_TIER_B_OUT", "/tmp"))


def obs_text(cond, tid):
    p = cond / "artifacts" / f"{SITE}_task_{tid}" / "step_000" / "observation_dom.txt"
    return p.read_text() if p.exists() else None


def task_meta(tid):
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


def resolve(obs, eid):
    if eid is None:
        return None
    for ln in obs.splitlines():
        m = EIDLINE.match(ln)
        if m and int(m.group(1)) == int(eid):
            return m.group(2).strip()[:60]
    return f"<eid {eid} 不在 obs>"


def dsig(action, obs):
    at = action.get("action_type") if action else None
    if at == "type":
        return (at, (action.get("text") or "").strip()[:40])
    if at in ("click", "hover", "select_option"):
        return (at, resolve(obs, action.get("element_id")), (action.get("option_label") or "")[:20])
    if at == "scroll":
        return (at, action.get("scroll_direction"))
    return (at,) if at else ("PARSE_FAIL",)


def _extract_first_balanced_json(text: str):
    """Find first balanced {...} block (handles multi-line + nested)."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False; continue
        if c == "\\":
            esc = True; continue
        if c == '"':
            in_str = not in_str; continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_action(generated_text: str) -> dict:
    """抽 B1 greedy 输出里的 action JSON。优先 balanced-brace + json.loads;失败再 regex 兜底。"""
    js = _extract_first_balanced_json(generated_text)
    if js:
        try:
            return json.loads(js)
        except Exception:
            pass
    out = {}
    for k in ("action_type", "text", "scroll_direction", "option_label"):
        mm = re.search(rf'"{k}"\s*:\s*"([^"]*)"', generated_text)
        if mm:
            out[k] = mm.group(1)
    mm = re.search(r'"element_id"\s*:\s*(\d+)', generated_text)
    if mm:
        out["element_id"] = int(mm.group(1))
    return out


def force_gpu_patch():
    """强制 device_map={'':0} + eager attention.
    - device_map: DGX GB10 'Not Supported' 显存查询 → auto offload CPU 坑 (tier_a 同).
    - attn_implementation='eager': Myriad V100 + bf16 + SDPA → 'cutlassF: no kernel
      found to launch' (V100 SM7.0 cutlass kernel 路径缺); eager 跨 GPU-type 通用
      (慢但稳, A100/GB10 也 OK 仅 ~2x 慢)。"""
    import p79.agents.qwen3vl_agent as qa
    _orig = qa.Qwen3VLForConditionalGeneration.from_pretrained

    def _forced(*a, **k):
        k["device_map"] = {"": 0}
        k.setdefault("attn_implementation", "eager")
        return _orig(*a, **k)
    qa.Qwen3VLForConditionalGeneration.from_pretrained = _forced


def build_inputs(agent, intent: str, obs_str: str, imgs: list):
    """复现 agent.step() 的 prompt 构造 (qwen3vl_agent.py:175-275), 拿到 tokenized inputs。"""
    system_prompt = agent._system_prompts["dom"]
    obs_section = f"Accessibility Tree:\n{obs_str}"
    content = [{
        "type": "text",
        "text": f"Task: {intent}\nSystem: {system_prompt}\n{obs_section}",
    }]
    if imgs:
        for idx, im in enumerate(imgs):
            content.append({"type": "text", "text": f"[Reference image {idx+1}] target item; use to identify element."})
            content.append({"type": "image", "image": im})
    messages = [{"role": "user", "content": content}]
    text = agent.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if imgs:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = agent.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    else:
        inputs = agent.processor(text=[text], return_tensors="pt")
    return {k: v.to(agent.model.device) for k, v in inputs.items()}


def baseline_action(agent, intent, obs_str, imgs):
    """跑 agent.step 拿干净 baseline action (与 Tier A 同协议)。"""
    a, _ = agent.step(intent, SimpleNamespace(text=obs_str, image=None), observation_mode="dom", reference_images=imgs)
    return a


def patch_sweep(agent, patcher, intent, obs_arch, obs_curr, imgs, max_new_tokens=256):
    """对每层 L, 用 source=arch 的 hidden 替换 target=curr 的 last-token hidden, 看 patched 输出。"""
    arch_inputs = build_inputs(agent, intent, obs_arch, imgs)
    curr_inputs = build_inputs(agent, intent, obs_curr, imgs)
    # cache arch hidden states (per layer)
    cached_arch = patcher.cache_hidden_states(**arch_inputs)
    results = []
    for L in range(patcher.n_layers):
        gen_ids = patcher.patched_generate(L, cached_arch[L], max_new_tokens=max_new_tokens, **curr_inputs)
        text = agent.processor.decode(gen_ids, skip_special_tokens=True)
        a = parse_action(text)
        # 用 curr_obs 解析 (target 在 curr 空间)
        results.append({"layer": L, "dsig": list(dsig(a, obs_curr)), "raw": text[:200]})
    return results


def main():
    arg = sys.argv[1:]
    if not arg:
        print("Usage: tier_b_id_patching.py <tid> [<tid> ...]")
        return
    tasks = [int(x) for x in arg]
    cfg_path = os.environ.get("P79_TIER_A_CFG", str(REPO / "configs/exp_v2_B1_dom_classifieds.yaml"))
    cfg = load_experiment_config(cfg_path)
    print(f"[load] forcing GPU0 + instantiating B1 ... ({len(tasks)} tasks)", flush=True)
    print(f"[paths] ARCH={ARCH}")
    print(f"[paths] CURR={CURR}", flush=True)
    force_gpu_patch()
    agent = Qwen3VLAgent(cfg)
    dev = next(agent.model.parameters()).device
    print(f"[load] B1 ready | model device={dev}", flush=True)
    if dev.type != "cuda":
        print("[ABORT] not GPU"); return
    patcher = ActivationPatcher(agent.model, agent.processor)
    print(f"[load] patcher ready | n_layers={patcher.n_layers}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for tid in tasks:
        oa, oc = obs_text(ARCH, tid), obs_text(CURR, tid)
        if not oa or not oc:
            print(f"task {tid}: obs 缺失, skip"); continue
        intent, imgs = task_meta(tid)
        # baselines
        a_arch = baseline_action(agent, intent, oa, imgs)
        a_curr = baseline_action(agent, intent, oc, imgs)
        sa, sc = dsig(a_arch, oa), dsig(a_curr, oc)
        if sa == sc:
            print(f"task {tid}: baseline arch==curr (B1 没 flip on this task), skip patching"); continue
        print(f"\ntask {tid} | arch={sa} | curr={sc}", flush=True)
        per_layer = patch_sweep(agent, patcher, intent, oa, oc, imgs)
        # 找 emergence: 最早 L 使 patched dsig == arch dsig (id-effect tip 到 arch)
        emerge = next((r["layer"] for r in per_layer if tuple(r["dsig"]) == sa), None)
        rec = {
            "task_id": tid,
            "arch_dsig": list(sa),
            "curr_dsig": list(sc),
            "n_layers": patcher.n_layers,
            "emergence_layer": emerge,
            "per_layer": per_layer,
        }
        out_path = OUT_DIR / f"tier_b_task_{tid}.json"
        out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
        print(f"  emergence layer = {emerge}/{patcher.n_layers} | saved {out_path}", flush=True)
        summary.append({"task_id": tid, "emergence_layer": emerge, "n_layers": patcher.n_layers})
    (OUT_DIR / "tier_b_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n=== summary saved {OUT_DIR/'tier_b_summary.json'} | n_tasks_with_flip={len(summary)} ===")


if __name__ == "__main__":
    main()
