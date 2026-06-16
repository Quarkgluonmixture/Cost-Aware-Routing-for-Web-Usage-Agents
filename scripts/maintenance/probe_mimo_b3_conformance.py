"""B3 pilot Stage 0 — MiMo-VL-7B-RL-2508 load + format-conformance smoke (DGX, 2026-06-16).

Tests the two unknowns the literature can't answer for MiMo as B3:
 (a) loads + runs on DGX (transformers compat via Qwen2.5-VL deployment class);
 (b) FORMAT CONFORMANCE under our fixed JSON action schema — does it emit parse-valid
     {"action_type":...} in OUR schema, can it produce `finish`, does it leak native
     tokens / markdown (= the GLM-432/432-zero-action lockout test, on MiMo).
Faithful: imports the byte-identical shared system prompt from _shared_vl_utils.
DGX only; does NOT touch the A100 paper-grade fire (R10175) or its VWA.
"""
import os, sys, json, re, time
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
os.environ.setdefault("CUDA_MPS_PIPE_DIRECTORY", "")
os.environ.setdefault("CUDA_MPS_LOG_DIRECTORY", "")
sys.path.insert(0, os.getcwd())
import torch
# DGX GB10 sm_121 nvrtc prod fallback — production agents (qwen3vl/gemma3vl) do this;
# MiMo's Qwen2.5-VL vision path calls image_grid_thw.prod(-1) which triggers the
# nvrtc --gpu-architecture crash on GB10 without this patch.
from p79.utils.torch_cuda_workarounds import apply_nvrtc_prod_fallback_if_needed
apply_nvrtc_prod_fallback_if_needed()
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# faithful shared prompt (same bytes all baselines see)
try:
    from p79.agents._shared_vl_utils import make_som_prompt as make_prompt
    PROMPT_KIND = "som"
except Exception:
    from p79.agents._shared_vl_utils import make_dom_prompt as make_prompt
    PROMPT_KIND = "dom"

MODEL = "XiaomiMiMo/MiMo-VL-7B-RL-2508"
ART = ("results/visualwebarena/phase1/B2_som_classifieds_20260611_210828_"
       "923656661_1218867_R3380/phase1_som_router_0/artifacts")
SAMPLES = [
    ("task_22", f"{ART}/classifieds_task_22/som/step_000_som.png",
     "How many miles does the red car in the second row have?"),
    ("task_42", f"{ART}/classifieds_task_42/som/step_000_som.png",
     "Find the listing of a collectible figurine and open its detail page."),
    ("task_184", f"{ART}/classifieds_task_184/som/step_000_som.png",
     "Find the powered PA speaker listing and open it."),
]
VALID_ACTIONS = {"click","type","select_option","scroll","wait","back","forward",
                 "finish","tab_focus","hover","press","new_tab","close_tab","goto"}

def cap(img, m=1024):
    if max(img.size) > m:
        r = m/max(img.size); return img.resize((int(img.size[0]*r),int(img.size[1]*r)), Image.Resampling.LANCZOS)
    return img

def parse_check(ans):
    """Mirror our parser's tolerance: find a JSON object with action_type."""
    leaked = bool(re.search(r"<\|.*?\|>|```|<tool_call>|<\|begin_of_box\|>", ans))
    m = re.search(r"\{.*\}", ans, re.S)
    if not m: return {"parse_valid": False, "action_type": None, "leaked_native": leaked, "raw_head": ans[:160]}
    try:
        obj = json.loads(m.group(0))
    except Exception:
        # try fenced/loose
        try: obj = json.loads(re.sub(r",\s*}", "}", m.group(0)))
        except Exception: return {"parse_valid": False, "action_type": None, "leaked_native": leaked, "raw_head": ans[:160]}
    at = obj.get("action_type")
    return {"parse_valid": at in VALID_ACTIONS, "action_type": at,
            "leaked_native": leaked, "has_thought": "thought" in obj,
            "has_confidence": "confidence" in obj}

def main():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading {MODEL} (prompt={PROMPT_KIND}, downloads ~16GB on first run)...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda:0").eval()
    metas = sum(1 for _, p in model.named_parameters() if p.device.type == "meta")
    print(f"  loaded in {time.time()-t0:.0f}s; class={model.__class__.__name__}; meta-device params={metas} (must be 0)", flush=True)
    if metas:
        print("  !! META-DEVICE OFFLOAD — invalid"); return
    sys_prompt = make_prompt()
    n_ok = 0
    for name, path, task in SAMPLES:
        if not os.path.exists(path):
            print(f"\n### {name}: MISSING {path}"); continue
        img = cap(Image.open(path).convert("RGB"))
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": f"TASK: {task}\n\n(Set-of-Marks screenshot above; numeric labels are element IDs.) What is your next action? Respond with ONLY the JSON action."},
            ]},
        ]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            from qwen_vl_utils import process_vision_info
            imgs, _ = process_vision_info(messages)
            inputs = proc(text=[text], images=imgs, return_tensors="pt").to("cuda:0")
        except Exception:
            inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        ans = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        chk = parse_check(ans)
        n_ok += int(chk["parse_valid"])
        print(f"\n{'='*64}\n### {name}  TASK={task[:48]}\n{'='*64}", flush=True)
        print(f"CONFORMANCE: {chk}", flush=True)
        print(f"--- raw output (first 400 chars) ---\n{ans[:400]}", flush=True)
    print(f"\n[{time.strftime('%H:%M:%S')}] SMOKE DONE: parse_valid {n_ok}/{len(SAMPLES)}; total {time.time()-t0:.0f}s", flush=True)
    print("CONFORMANCE GATE: 3/3 parse-valid + no native leak + at least one can emit finish => proceed to Stage 2 floor pilot", flush=True)

if __name__ == "__main__":
    main()
