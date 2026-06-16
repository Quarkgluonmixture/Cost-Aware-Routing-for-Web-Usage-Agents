"""Gemma3-4B 1024-cap vs uncapped A/B probe (GPT playbook P0 test, 2026-06-16).

Question: does removing the P79 image_max_size=1024 cap (before the Gemma
processor) rescue natural-photo recognition? Mirrors gemma3vl_agent.py:200-282
(cap -> two-step apply_chat_template -> processor(do_pan_and_scan=True,
add_special_tokens=False)). Only the cap differs between arms; P&S=True both.

Anchor image = task_22 cars gallery (§327 GT: thumbnails incl a RED Porsche;
capped+P&S Gemma said 'Dark Blue x4'). Original screenshots are 1280x720, so
the cap is only 1280->1024 (1.25x) — prior = small effect.
"""
import os, sys, time
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
os.environ.setdefault("CUDA_MPS_PIPE_DIRECTORY", "")
os.environ.setdefault("CUDA_MPS_LOG_DIRECTORY", "")
import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

MODEL = "google/gemma-3-4b-it"
REV = "093f9f388b31de276ce2de164bdc2081324b9767"
ART = ("results/visualwebarena/phase1/B2_som_classifieds_20260611_210828_"
       "923656661_1218867_R3380/phase1_som_router_0/artifacts")
IMAGES = {
    "task_22_cars": f"{ART}/classifieds_task_22/som/step_000_som.png",
    "task_42":      f"{ART}/classifieds_task_42/som/step_000_som.png",
    "task_184":     f"{ART}/classifieds_task_184/som/step_000_som.png",
}
# Q2 (photo content) is THE test; task_22 also gets label/price OCR controls.
Q_PHOTO = ("Look at the product/vehicle PHOTOS in the listing thumbnails (ignore "
           "the blue numbered labels). From left to right, top row then bottom "
           "row, name each item and its main color as specifically as you can.")
Q_LABEL = ("What number labels the 'Apply' button and what number labels the "
           "'Publish Ad' button? Answer with just the two numbers.")
Q_PRICE = ("What price is shown on the FIRST (top-left) car listing? Answer with "
           "just the number.")

def cap(img, max_size=1024):
    if max(img.size) > max_size:
        r = max_size / max(img.size)
        return img.resize((int(img.size[0]*r), int(img.size[1]*r)), Image.Resampling.LANCZOS)
    return img

def main():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading {MODEL} @ {REV[:8]} (bf16, cuda:0, no auto-offload)...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL, revision=REV)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL, revision=REV, torch_dtype=torch.bfloat16).to("cuda:0").eval()
    # guard against §327 meta-device trap
    metas = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    print(f"  loaded in {time.time()-t0:.0f}s; meta-device params={len(metas)} (must be 0)", flush=True)
    if metas:
        print("  !! META-DEVICE OFFLOAD DETECTED — results invalid", flush=True); return

    def ask(img, q):
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], images=[img], do_pan_and_scan=True,
                   add_special_tokens=False, return_tensors="pt").to("cuda:0")
        if "pixel_values" in inp:
            inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        n_img_tok = int((inp["input_ids"] == proc.tokenizer.convert_tokens_to_ids("<image_soft_token>")).sum()) \
            if "<image_soft_token>" in proc.tokenizer.get_vocab() else -1
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=256, do_sample=False)
        ans = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        return ans, n_img_tok

    for name, path in IMAGES.items():
        if not os.path.exists(path):
            print(f"\n### {name}: MISSING {path}", flush=True); continue
        base = Image.open(path).convert("RGB")
        print(f"\n{'='*72}\n### {name}  orig={base.size}\n{'='*72}", flush=True)
        qs = [("PHOTO", Q_PHOTO)] + ([("LABEL", Q_LABEL), ("PRICE", Q_PRICE)] if name == "task_22_cars" else [])
        for qtag, q in qs:
            for arm, img in [("CAPPED-1024", cap(base)), ("UNCAPPED", base)]:
                a, nt = ask(img, q)
                print(f"\n-- [{qtag}] [{arm}] feed_size={img.size} img_tok={nt} --\n{a}", flush=True)
    print(f"\n[{time.strftime('%H:%M:%S')}] DONE total {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
