#!/usr/bin/env python3
"""Action-parity spike: do HF-eager and vLLM agree on the *action* across N real
dom steps?  This is the go/no-go datum for migrate-all SR risk — token-level
divergence is benign iff the parsed (action_type, element_id) stays identical.

Inputs = real `observation_dom.txt` AXTree dumps from a landed run, paired with
the task's real intent. HF dumps the exact input_ids; vLLM reuses them, so any
action difference is purely the HF->vLLM decode-kernel divergence (not input drift).

    .venv/bin/python      spike_action_parity.py --engine hf   --model qwen --run-dir <D> --n 40 --out ap_hf_qwen.jsonl
    .venv-vllm/bin/python spike_action_parity.py --engine vllm --model qwen --paired ap_hf_qwen.jsonl --out ap_vllm_qwen.jsonl
    .venv/bin/python      spike_action_parity.py --compare ap_hf_qwen.jsonl ap_vllm_qwen.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from p79.agents._shared_vl_utils import make_dom_prompt, format_history  # noqa: E402

try:
    from p79.backends.action_utils import parse_action_text  # noqa: E402
except Exception:
    parse_action_text = None

MODELS = {
    "qwen": ("Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17"),
    "gemma": ("google/gemma-3-4b-it", "093f9f388b31de276ce2de164bdc2081324b9767"),
}
SITE_CONFIG = {
    "classifieds": _REPO / "external/visualwebarena/config_files/vwa/test_classifieds.raw.json",
    "reddit": _REPO / "external/visualwebarena/config_files/vwa/test_reddit.raw.json",
    "shopping": _REPO / "external/visualwebarena/config_files/vwa/test_shopping.raw.json",
}


def load_intents(site: str) -> dict:
    p = SITE_CONFIG.get(site)
    if not p or not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        return {int(t["task_id"]): t.get("intent", "") for t in data if "task_id" in t}
    except Exception:
        return {}


def collect_steps(run_dir: str, n: int):
    files = sorted(Path(run_dir).glob("artifacts/*/step_*/observation_dom.txt"),
                   key=lambda p: p.stat().st_size)
    if not files:
        return []
    if len(files) <= n:
        picks = files
    else:
        idxs = sorted({round(i * (len(files) - 1) / (n - 1)) for i in range(n)})
        picks = [files[i] for i in idxs]
    out = []
    for f in picks:
        m = re.search(r"/(classifieds|reddit|shopping)_task_(\d+)/step_(\d+)/", str(f))
        site = m.group(1) if m else "classifieds"
        tid = int(m.group(2)) if m else -1
        sidx = int(m.group(3)) if m else -1
        out.append((f, site, tid, sidx))
    return out


def build_prompt_text(obs_text: str, instruction: str) -> str:
    """Byte-identical to the agent's dom-mode user-turn (empty history)."""
    return (f"Task: {instruction}\nSystem: {make_dom_prompt()}\n"
            f"{format_history([])}Accessibility Tree:\n{obs_text}")


def parse_act(txt: str):
    if parse_action_text is None:
        return (None, None, None)
    try:
        a, valid, _ = parse_action_text(txt)
        return (a.get("action_type"), a.get("element_id"), valid)
    except Exception:
        return ("PARSE_ERR", None, False)


def run_hf(args):
    import torch
    from transformers import AutoProcessor

    path, rev = MODELS[args.model]
    if args.model == "qwen":
        from transformers import Qwen3VLForConditionalGeneration as M
        model = M.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto",
                                  trust_remote_code=True, revision=rev).eval()
        proc = AutoProcessor.from_pretrained(path, trust_remote_code=True, revision=rev)
    else:
        from transformers import Gemma3ForConditionalGeneration as M
        model = M.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto",
                                  revision=rev).eval()
        proc = AutoProcessor.from_pretrained(path, revision=rev)

    if getattr(args, "compile", False):
        model.forward = torch.compile(model.forward)
        print(f"[hf {args.model}] torch.compile(model.forward) enabled (first steps compile)", flush=True)

    steps = collect_steps(args.run_dir, args.n)
    intents_cache = {}
    with open(args.out, "w") as fout:
        for i, (f, site, tid, sidx) in enumerate(steps):
            if site not in intents_cache:
                intents_cache[site] = load_intents(site)
            instr = intents_cache[site].get(tid) or "Complete the task for this page."
            text = build_prompt_text(f.read_text(), instr)
            messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
            if args.model == "gemma":
                inputs = proc.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
                input_ids = inputs["input_ids"]
            else:
                chat = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = proc(text=[chat], padding=True, return_tensors="pt").to(model.device)
                input_ids = inputs.input_ids
            if args.seed is not None:
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, return_dict_in_generate=True)
            n_in = int(input_ids.shape[1])
            gen = out.sequences[0][n_in:]
            otext = proc.batch_decode([gen], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0]
            at, eid, valid = parse_act(otext)
            fout.write(json.dumps({
                "idx": i, "dom_file": str(f), "site": site, "task_id": tid, "step_idx": sidx,
                "input_tokens": n_in, "input_ids": [int(x) for x in input_ids[0].tolist()],
                "action_type": at, "element_id": eid, "valid": valid, "output_text": otext,
            }) + "\n")
            fout.flush()
            print(f"[hf {args.model}] {i+1}/{len(steps)} task{tid}.s{sidx} in={n_in} -> {at}(id={eid})",
                  flush=True)


def run_vllm(args):
    from vllm import LLM, SamplingParams

    path, rev = MODELS[args.model]
    rows = [json.loads(l) for l in open(args.paired)]
    llm = LLM(model=path, revision=rev, dtype="bfloat16", seed=args.seed or 0,
              gpu_memory_utilization=args.gpu_mem_util, max_model_len=args.max_model_len,
              trust_remote_code=(args.model == "qwen"))
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, seed=args.seed or 0)
    with open(args.out, "w") as fout:
        for r in rows:
            out = llm.generate({"prompt_token_ids": r["input_ids"]}, sp, use_tqdm=False)
            otext = out[0].outputs[0].text
            at, eid, valid = parse_act(otext)
            fout.write(json.dumps({
                "idx": r["idx"], "dom_file": r["dom_file"], "site": r["site"],
                "task_id": r["task_id"], "step_idx": r["step_idx"],
                "action_type": at, "element_id": eid, "valid": valid, "output_text": otext,
            }) + "\n")
            fout.flush()
            print(f"[vllm {args.model}] idx{r['idx']} task{r['task_id']} -> {at}(id={eid})", flush=True)


def run_compare(pa: str, pb: str):
    A = [json.loads(l) for l in open(pa)]
    B = {json.loads(l)["idx"]: json.loads(l) for l in open(pb)}
    n = at_match = full_match = 0
    mism = []
    for ra in A:
        rb = B.get(ra["idx"])
        if rb is None:
            continue
        n += 1
        atm = ra["action_type"] == rb["action_type"]
        eidm = ra["element_id"] == rb["element_id"]
        at_match += atm
        if atm and eidm:
            full_match += 1
        else:
            mism.append((ra, rb))
    if not n:
        print("  no overlapping rows"); return
    print(f"  N steps compared          : {n}")
    print(f"  action_type match         : {at_match}/{n} = {at_match/n*100:.1f}%")
    print(f"  (type+element_id) match    : {full_match}/{n} = {full_match/n*100:.1f}%")
    print(f"  input_tokens range        : {min(r['input_tokens'] for r in A)}-{max(r['input_tokens'] for r in A)}")
    for ra, rb in mism[:25]:
        print(f"    MISMATCH idx{ra['idx']} task{ra['task_id']}.s{ra['step_idx']} "
              f"in={ra.get('input_tokens')}: "
              f"HF={ra['action_type']}(id={ra['element_id']},v={ra['valid']}) "
              f"vLLM={rb['action_type']}(id={rb['element_id']},v={rb['valid']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["hf", "vllm"])
    ap.add_argument("--model", choices=["qwen", "gemma"])
    ap.add_argument("--run-dir")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--paired")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu-mem-util", type=float, default=0.5)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--compile", action="store_true",
                    help="HF: torch.compile(model.forward) — test compile-vs-eager action divergence")
    args = ap.parse_args()

    if args.compare:
        run_compare(args.compare[0], args.compare[1])
    elif args.engine == "hf":
        run_hf(args)
    elif args.engine == "vllm":
        run_vllm(args)
    else:
        ap.error("need --engine hf|vllm or --compare A B")


if __name__ == "__main__":
    main()
