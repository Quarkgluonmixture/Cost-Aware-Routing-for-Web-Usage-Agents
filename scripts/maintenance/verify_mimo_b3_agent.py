"""B3 integration verify — exercise the REAL MiMoVLAgent.step() path (NOT the
Stage-0 standalone probe) end-to-end: load → build messages → process_vision_info
→ generate → confidence → parse_action_text. De-risks the Explore-flagged
unknowns (generate() signature divergence; processor.image_token_id presence)
BEFORE the floor pilot. DGX only; does NOT touch the A100 fire or any VWA; HF
cache only (no network needed)."""
import os
import sys
import types

os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
os.environ.setdefault("CUDA_MPS_PIPE_DIRECTORY", "")
os.environ.setdefault("CUDA_MPS_LOG_DIRECTORY", "")
sys.path.insert(0, os.getcwd())

from PIL import Image, ImageDraw

REV = "4bfb270765825d2fa059011deb4c96fdd579be6f"
SOM_TEXT = (
    "[SOM_MARKS]\n"
    "[1] textbox 'Search for anything'\n"
    "[2] link 'Cars + Vehicles'\n"
    "[3] button 'Search'\n"
    "[4] link 'Electronics'\n"
    "[5] link 'a used Toyota sedan listing'\n"
    "[/SOM_MARKS]"
)


def synthetic_som_image():
    """A real 1280x720 PIL RGB image with mark boxes — exercises the same
    process_vision_info + image-token-counting path as a true SoM screenshot."""
    img = Image.new("RGB", (1280, 720), "white")
    d = ImageDraw.Draw(img)
    rows = [("[1] Search for anything", 40), ("[2] Cars + Vehicles", 110),
            ("[3] Search", 180), ("[4] Electronics", 250),
            ("[5] Used Toyota sedan", 320)]
    for label, y in rows:
        d.rectangle([40, y, 460, y + 50], outline="red", width=3)
        d.text((52, y + 16), label, fill="black")
    return img


def main():
    from p79.agents.mimo_vl_agent import MiMoVLAgent

    cfg = {
        "model": {
            "path": "XiaomiMiMo/MiMo-VL-7B-RL-2508",
            "revision": REV,
            "quantization": "none",
            "device": "cuda",
            "max_new_tokens": 4096,
            "min_free_vram_gb": 0,
        },
        "agent": {"image_max_size": 1024},
    }
    print("[verify] loading MiMoVLAgent (real agent class) ...", flush=True)
    agent = MiMoVLAgent(cfg)
    print(
        f"[verify] loaded OK; model class={agent.model.__class__.__name__}; "
        f"image_token_id={getattr(agent.processor, 'image_token_id', None)}",
        flush=True,
    )

    img = synthetic_som_image()
    obs = types.SimpleNamespace(image=img, text=SOM_TEXT)
    tasks = [
        "Find a used car listing and open its detail page.",
        "Search the site for electronics.",
    ]
    n_ok = 0
    for i, task in enumerate(tasks):
        action, meta = agent.step(task, obs, history=[], observation_mode="som")
        valid = bool(meta.get("valid"))
        n_ok += int(valid)
        print(f"\n=== task {i}: {task} ===", flush=True)
        print(f"  action      = {action}", flush=True)
        print(
            f"  valid={valid} source={meta.get('action_source')} "
            f"in_tok={meta.get('input_tokens')} out_tok={meta.get('output_tokens')} "
            f"img_tok={meta.get('input_image_tokens')} "
            f"think_stripped={'<think>' in (meta.get('raw_output') or '')}",
            flush=True,
        )
        print(f"  raw_head    = {(meta.get('raw_output') or '')[:240]!r}", flush=True)

    verdict = "PASS — full step() path works, integration ready-to-fire" if n_ok >= 1 \
        else "FAIL — investigate (no parse-valid action)"
    print(f"\n[verify] DONE: parse_valid {n_ok}/{len(tasks)}  =>  {verdict}", flush=True)
    print("MIMO_B3_VERIFY_DONE", flush=True)


if __name__ == "__main__":
    main()
