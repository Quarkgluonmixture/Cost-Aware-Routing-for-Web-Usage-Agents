#!/usr/bin/env python3
"""Compare HF-eager vs vLLM spike outputs: tok/s, logprob availability, divergence.

Usage:
    python scripts/spike/spike_compare.py /tmp/hf_qwen.json /tmp/vllm_qwen.json \
                                          /tmp/hf_gemma.json /tmp/vllm_gemma.json
Pairs are matched by the "model" field; any number of files accepted.
"""
import json
import sys
from collections import defaultdict


def load(paths):
    by_model = defaultdict(dict)
    for p in paths:
        with open(p) as f:
            r = json.load(f)
        by_model[r["model"]][r["engine"]] = r
    return by_model


def divergence(hf, vllm):
    """First-divergence index over output_ids, given identical input."""
    same_input = hf["input_ids_sha"] == vllm["input_ids_sha"]
    a, b = hf["output_ids"], vllm["output_ids"]
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i] != b[i]), None)
    if first is None and len(a) == len(b):
        first_div = None  # identical
    elif first is None:
        first_div = n  # one is a prefix of the other
    else:
        first_div = first
    return {
        "same_input": same_input,
        "input_sha": (hf["input_ids_sha"], vllm["input_ids_sha"]),
        "hf_out_len": len(a), "vllm_out_len": len(b),
        "first_divergence_idx": first_div,
        "identical": first_div is None,
        "frac_matched": (n if first_div is None else first_div) / max(len(a), 1),
    }


def fmt(v, suffix=""):
    return f"{v}{suffix}" if v is not None else "—"


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    by_model = load(sys.argv[1:])

    print("=" * 74)
    print("TIER-0 SPIKE REPORT — HF eager vs vLLM single-stream (dom step)")
    print("=" * 74)

    for model, eng in by_model.items():
        hf, vllm = eng.get("hf"), eng.get("vllm")
        print(f"\n### {model.upper()}")
        if hf:
            v = hf["versions"]
            print(f"  versions: torch {v['torch']} | tf {v['transformers']} (HF env)")
        if vllm:
            v = vllm["versions"]
            print(f"            vllm {v['vllm']} | tf {v['transformers']} (vLLM env)"
                  f" | enforce_eager={vllm.get('enforce_eager')}")

        # --- 1. throughput ---
        print("  [1] throughput (median, post-warmup):")
        if hf:
            print(f"        HF eager : {fmt(hf['median_tok_per_s'],' tok/s')} "
                  f"({fmt(hf['median_gen_ms'],' ms/gen')}, {hf['output_tokens']} out tok, "
                  f"in {hf['input_tokens']} tok, load {hf['load_s']}s)")
        if vllm:
            print(f"        vLLM     : {fmt(vllm['median_tok_per_s'],' tok/s')} "
                  f"({fmt(vllm['median_gen_ms'],' ms/gen')}, {vllm['output_tokens']} out tok, "
                  f"load {vllm['load_s']}s)")
        if hf and vllm and hf["median_tok_per_s"] and vllm["median_tok_per_s"]:
            sp = vllm["median_tok_per_s"] / hf["median_tok_per_s"]
            print(f"        SPEEDUP  : {sp:.2f}x  (model-only; total wallclock capped"
                  f" by Amdahl — env ~35-45% of step)")

        # --- 2. logprob availability ---
        print("  [2] logprob / router-confidence availability:")
        for tag, r in (("HF", hf), ("vLLM", vllm)):
            if not r:
                continue
            c = r.get("confidence", {})
            have = [k for k in ("mean_logprob", "min_logprob", "mean_margin",
                                "min_margin", "mean_entropy", "max_entropy")
                    if c.get(k) is not None]
            print(f"        {tag:4}: {len(have)}/6 fields "
                  f"[{', '.join(have) if have else 'NONE'}]"
                  + (f"  available={r.get('logprob_available')}" if tag == "vLLM" else ""))

        # --- 3. token divergence ---
        if hf and vllm:
            d = divergence(hf, vllm)
            print("  [3] token divergence (same input ids):")
            if not d["same_input"]:
                print(f"        ⚠ INPUT MISMATCH {d['input_sha']} — rerun vLLM with "
                      f"--paired-input <hf.json>")
            if d["identical"]:
                print(f"        ✓ IDENTICAL output ({d['hf_out_len']} tok) — vLLM greedy"
                      f" == HF greedy for this step")
            else:
                print(f"        ✗ diverges at output idx {d['first_divergence_idx']} "
                      f"of {d['hf_out_len']}/{d['vllm_out_len']} (HF/vLLM); "
                      f"matched prefix {d['frac_matched']*100:.0f}%")
            ha = (hf.get("action_parse") or {})
            va = (vllm.get("action_parse") or {})
            print(f"        action: HF={ha.get('action_type')}"
                  f"(id={ha.get('element_id')},valid={ha.get('valid')}) "
                  f"vLLM={va.get('action_type')}(id={va.get('element_id')},"
                  f"valid={va.get('valid')}) "
                  f"-> {'SAME' if ha.get('action_type')==va.get('action_type') and ha.get('element_id')==va.get('element_id') else 'DIFFERENT'}")

    print("\n" + "=" * 74)
    print("Decision hooks: speedup>=3x single-stream confirms launch-bound thesis;")
    print("vLLM logprob 4/6 (no entropy) == B0 schema; action SAME across most")
    print("steps => migrate-all paper-grade risk is low. Divergence>0 but action")
    print("SAME => token-level diff is benign (sub-token JSON formatting).")
    print("=" * 74)


if __name__ == "__main__":
    main()
