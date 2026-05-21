#!/usr/bin/env python3
"""Tier 0 inference spike: HF eager vs vLLM single-stream on a dom-mode step.

Answers three questions for the vLLM-migration decision (实验笔记 / inference
accel research 2026-05-20):
  1. tok/s  — model-only generate throughput (median over N trials, post-warmup)
  2. logprob availability — can vLLM populate the router confidence fields
       (mean/min logprob + mean/min margin)?  entropy needs full vocab -> expect None.
  3. token divergence — given the *same* input token ids, do HF-eager and vLLM
       greedy decode produce identical output token ids?  (paper-grade: vLLM != HF
       bitwise -> need to know the divergence rate before committing.)

Run the two engines in SEPARATE venvs so installing vLLM never perturbs the
paper-grade HF env:
    .venv/bin/python        scripts/spike/spike_infer.py --engine hf   --model qwen  --out /tmp/hf_qwen.json
    .venv-vllm/bin/python   scripts/spike/spike_infer.py --engine vllm --model qwen  --out /tmp/vllm_qwen.json --paired-input /tmp/hf_qwen.json
Then: .venv/bin/python scripts/spike/spike_compare.py /tmp/hf_qwen.json /tmp/vllm_qwen.json ...

dom mode = text-only path (obs.image is None) -> no multimodal needed here; a
second spike can cover the som/vision image path once dom is validated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

# repo root on sys.path so we consume the *canonical* cross-baseline prompt
# builders (byte-identical to the production agents).
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from p79.agents._shared_vl_utils import make_dom_prompt, format_history  # noqa: E402

try:
    from p79.backends.action_utils import parse_action_text  # noqa: E402
except Exception:  # pragma: no cover - keep spike runnable if import chain breaks
    parse_action_text = None

MODELS = {
    # (hf_path, pinned_revision_sha) — same SHAs as configs/exp_v2_base.yaml
    "qwen": ("Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17"),
    "gemma": ("google/gemma-3-4b-it", "093f9f388b31de276ce2de164bdc2081324b9767"),
}


def _ids_sha(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:16]


def build_dom_text(tokenizer, target_input_tokens: int) -> str:
    """Reconstruct the agent's dom-mode user-turn text, byte-identical to step().

    Uses a synthetic-but-realistic AXTree calibrated (by re-tokenizing) to land
    near `target_input_tokens`. dom-decode latency is driven by token count, not
    the specific AXTree content, so a calibrated synthetic page is representative.
    """
    system_prompt = make_dom_prompt()
    instruction = (
        "Find the cheapest used bicycle listed in the Sports category and open "
        "its detail page."
    )
    sample_rows = [
        "link 'Home'",
        "link 'Sports & Outdoors'",
        "button 'Search'",
        "textbox 'Search query' required: False",
        "link 'Bicycles'",
        "link 'Used'",
        "StaticText 'Showing 1-12 of 240 listings'",
        "link 'Mountain bike, good condition, barely used'",
        "StaticText '$120.00'",
        "link 'Road bike 54cm carbon frame'",
        "StaticText '$340.00'",
        "button 'Sort by: Price (low to high)'",
        "link 'Next page'",
        "heading 'Featured listings'",
        "img 'thumbnail'",
    ]
    rows = []
    idx = 100
    text = ""
    for _ in range(6000):  # hard upper bound on growth loop
        for s in sample_rows:
            rows.append(f"[{idx}] {s}")
            idx += 1
        axtree = "\n".join(rows)
        obs_section = f"Accessibility Tree:\n{axtree}"
        text = (
            f"Task: {instruction}\nSystem: {system_prompt}\n"
            f"{format_history([])}{obs_section}"
        )
        n = len(tokenizer(text, add_special_tokens=False).input_ids)
        if n >= target_input_tokens:
            break
    return text


# ----------------------------------------------------------------------
# HF eager path (main .venv) — mirrors the production agent step()
# ----------------------------------------------------------------------
def run_hf(model_key: str, args) -> dict:
    import torch
    import transformers
    from transformers import AutoProcessor
    from p79.agents._shared_vl_utils import compute_confidence

    path, rev = MODELS[model_key]
    if model_key == "qwen":
        from transformers import Qwen3VLForConditionalGeneration as ModelCls

        load_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto",
                           trust_remote_code=True, revision=rev)
        proc_kwargs = dict(trust_remote_code=True, revision=rev)
    else:
        from transformers import Gemma3ForConditionalGeneration as ModelCls

        load_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto", revision=rev)
        proc_kwargs = dict(revision=rev)

    t_load = time.perf_counter()
    model = ModelCls.from_pretrained(path, **load_kwargs).eval()
    processor = AutoProcessor.from_pretrained(path, **proc_kwargs)
    load_s = time.perf_counter() - t_load

    tokenizer = getattr(processor, "tokenizer", processor)
    text = build_dom_text(tokenizer, args.target_input_tokens)
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]

    # apply_chat_template exactly like the agent (gemma: tokenize=True dict;
    # qwen: tokenize=False then processor()). Normalize to input_ids tensor.
    if model_key == "gemma":
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)
        input_ids = inputs["input_ids"]
    else:
        chat = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[chat], padding=True, return_tensors="pt").to(model.device)
        input_ids = inputs.input_ids

    input_len = int(input_ids.shape[1])
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False,
                      return_dict_in_generate=True, output_scores=True)

    def _one():
        if model_key == "qwen" and args.seed is not None:
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**inputs, **gen_kwargs)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        n_out = int(out.sequences.shape[1]) - input_len
        return dt, n_out, out

    # warmup
    for _ in range(args.warmup):
        _one()

    trials, last_out = [], None
    for _ in range(args.n_trials):
        dt, n_out, out = _one()
        trials.append({"gen_s": dt, "output_tokens": n_out,
                       "tok_per_s": n_out / dt if dt else 0.0})
        last_out = out

    out_ids = [int(x) for x in last_out.sequences[0][input_len:].tolist()]
    out_text = processor.batch_decode(
        [last_out.sequences[0][input_len:]], skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    conf = compute_confidence(last_out.scores)

    return _assemble(
        engine="hf", model_key=model_key, path=path, rev=rev,
        versions={"torch": torch.__version__, "transformers": transformers.__version__,
                  "vllm": None},
        input_ids=[int(x) for x in input_ids[0].tolist()],
        input_len=input_len, trials=trials, out_ids=out_ids, out_text=out_text,
        conf=conf, load_s=load_s, args=args,
    )


# ----------------------------------------------------------------------
# vLLM single-stream path (isolated .venv-vllm)
# ----------------------------------------------------------------------
def run_vllm(model_key: str, args) -> dict:
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    path, rev = MODELS[model_key]

    # Strict divergence test: reuse HF's exact input_ids if provided.
    paired_ids = None
    if args.paired_input:
        with open(args.paired_input) as f:
            paired_ids = json.load(f)["input_ids"]

    t_load = time.perf_counter()
    llm = LLM(
        model=path, revision=rev, dtype="bfloat16", seed=args.seed or 0,
        gpu_memory_utilization=args.gpu_mem_util, max_model_len=args.max_model_len,
        trust_remote_code=(model_key == "qwen"), enforce_eager=args.enforce_eager,
    )
    load_s = time.perf_counter() - t_load

    if paired_ids is not None:
        input_ids = paired_ids
    else:
        tok = llm.get_tokenizer()
        text = build_dom_text(tok, args.target_input_tokens)
        input_ids = tok(text, add_special_tokens=False).input_ids
    input_len = len(input_ids)

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens,
                        logprobs=2, seed=args.seed or 0)
    prompt = {"prompt_token_ids": input_ids}

    def _one():
        t0 = time.perf_counter()
        outs = llm.generate(prompt, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        comp = outs[0].outputs[0]
        return dt, len(comp.token_ids), comp

    for _ in range(args.warmup):
        _one()

    trials, last = [], None
    for _ in range(args.n_trials):
        dt, n_out, comp = _one()
        trials.append({"gen_s": dt, "output_tokens": n_out,
                       "tok_per_s": n_out / dt if dt else 0.0})
        last = comp

    out_ids = [int(x) for x in last.token_ids]
    out_text = last.text

    # logprobs -> router confidence fields. vLLM returns per-position dict
    # {token_id: Logprob(logprob, rank, decoded_token)}; with logprobs=2 we get
    # top-2 (+ sampled, == top1 under greedy). entropy needs full vocab -> None.
    conf, lp_ok = _vllm_confidence(last)

    return _assemble(
        engine="vllm", model_key=model_key, path=path, rev=rev,
        versions={"torch": torch.__version__,
                  "transformers": _safe_tf_version(), "vllm": vllm.__version__},
        input_ids=[int(x) for x in input_ids], input_len=input_len, trials=trials,
        out_ids=out_ids, out_text=out_text, conf=conf, load_s=load_s, args=args,
        extra={"logprob_available": lp_ok, "enforce_eager": args.enforce_eager},
    )


def _safe_tf_version():
    try:
        import transformers
        return transformers.__version__
    except Exception:
        return None


def _vllm_confidence(comp):
    """Derive mean/min logprob + mean/min margin from vLLM top-k logprobs."""
    lps = getattr(comp, "logprobs", None)
    if not lps:
        return {}, False
    chosen, margins = [], []
    try:
        for pos, tok_id in enumerate(comp.token_ids):
            d = lps[pos]
            # rank-sorted top-k for this position
            ranked = sorted(d.values(), key=lambda lp: lp.logprob, reverse=True)
            top1 = ranked[0].logprob
            chosen.append(d[tok_id].logprob if tok_id in d else top1)
            if len(ranked) >= 2:
                margins.append(top1 - ranked[1].logprob)
        conf = {
            "mean_logprob": sum(chosen) / len(chosen),
            "min_logprob": min(chosen),
            "mean_margin": (sum(margins) / len(margins)) if margins else None,
            "min_margin": min(margins) if margins else None,
            "mean_entropy": None,  # full vocab unavailable from top-k
            "max_entropy": None,
        }
        return conf, True
    except Exception as e:  # pragma: no cover
        return {"error": repr(e)}, False


def _assemble(*, engine, model_key, path, rev, versions, input_ids, input_len,
              trials, out_ids, out_text, conf, load_s, args, extra=None):
    tps = [t["tok_per_s"] for t in trials]
    gms = [t["gen_s"] * 1000 for t in trials]
    action = None
    if parse_action_text is not None:
        try:
            a, valid, fail = parse_action_text(out_text)
            action = {"action_type": a.get("action_type"), "valid": valid,
                      "element_id": a.get("element_id"), "failure_reason": fail}
        except Exception as e:
            action = {"parse_error": repr(e)}
    rec = {
        "engine": engine, "model": model_key, "model_path": path, "revision": rev,
        "versions": versions, "host": platform.node(), "load_s": round(load_s, 1),
        "input_tokens": input_len, "input_ids_sha": _ids_sha(input_ids),
        "input_ids": input_ids,
        "n_trials": len(trials), "warmup": args.warmup,
        "median_tok_per_s": round(statistics.median(tps), 2) if tps else None,
        "median_gen_ms": round(statistics.median(gms), 1) if gms else None,
        "trials": [{"gen_ms": round(t["gen_s"] * 1000, 1),
                    "output_tokens": t["output_tokens"],
                    "tok_per_s": round(t["tok_per_s"], 2)} for t in trials],
        "output_tokens": len(out_ids), "output_ids": out_ids,
        "output_text": out_text, "action_parse": action,
        "confidence": conf,
    }
    if extra:
        rec.update(extra)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["hf", "vllm"])
    ap.add_argument("--model", required=True, choices=["qwen", "gemma"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-trials", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--target-input-tokens", type=int, default=2300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--paired-input", default=None,
                    help="vLLM: path to HF json; reuse its input_ids for strict divergence test")
    # vLLM knobs
    ap.add_argument("--gpu-mem-util", type=float, default=0.45)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--enforce-eager", action="store_true",
                    help="disable CUDA graphs (max determinism, slower) — default off")
    args = ap.parse_args()

    print(f"[spike] engine={args.engine} model={args.model} -> {args.out}", flush=True)
    t0 = time.perf_counter()
    rec = run_hf(args.model, args) if args.engine == "hf" else run_vllm(args.model, args)
    rec["wall_s"] = round(time.perf_counter() - t0, 1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[spike] DONE {args.engine}/{args.model}: "
          f"median {rec['median_tok_per_s']} tok/s "
          f"({rec['median_gen_ms']} ms/gen, {rec['output_tokens']} out tok, "
          f"input {rec['input_tokens']} tok); logprob_conf={ {k: v for k, v in rec['confidence'].items() if k in ('mean_logprob','mean_margin')} }",
          flush=True)


if __name__ == "__main__":
    main()
