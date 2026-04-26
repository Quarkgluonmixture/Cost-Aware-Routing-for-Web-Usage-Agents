#!/usr/bin/env python3
"""Probe: 给 model 一张或多张 SoM 截图，让它列出可见 content 文字，
评估视觉理解能力（OCR + attention focus）。

Ground truth 自动从对应 observation_som.txt 提取。

用法：
  .venv/bin/python3 scripts/maintenance/probe_som_occlusion.py \
      --image PATH [--image PATH2 ...] \
      --backend b0|b1 \
      --output /tmp/probe.json
"""

from __future__ import annotations
import argparse, base64, json, os, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

PROBE_PROMPT_NO_MARKS = """Look at this screenshot of a web page. List all the visible link, button, heading, and content text — what a user would read on the page.

Output rules:
- One text item per line.
- Do NOT guess names you cannot actually read in the image.
- Read top-to-bottom, left-to-right.

Output ONLY the text content, no explanations."""

PROBE_PROMPT_WITH_MARKS = """Look at this screenshot of a web page. List all the visible link, button, heading, and content text — what a user would read on the page.

Output rules:
- One text item per line.
- Do NOT include the small numbered ID tags (the cyan rectangles with numbers like "111", "297"). Those are annotation labels, not content.
- Do NOT guess names you cannot actually read in the image.
- Read top-to-bottom, left-to-right.

Output ONLY the text content, no explanations."""

PROBE_PROMPT_WITH_SOM_TEXT = """You are given a screenshot of a web page AND a textual list of all interactive elements on the page (the [SOM_MARKS] block below).

Each element in the list has its full label. The screenshot shows the same elements with small numbered ID tags (cyan rectangles).

[SOM_MARKS]
{som_marks}
[/SOM_MARKS]

Task: List all the visible link, button, heading, and content text — what a user would read on the page.

Output rules:
- One text item per line.
- You may use the [SOM_MARKS] text above as ground-truth labels for elements you can locate visually.
- Do NOT include the small numbered ID tags themselves (e.g. "111", "297").
- Read top-to-bottom, left-to-right based on the screenshot layout.

Output ONLY the text content, no explanations."""


def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def find_obs_txt(image_path: Path) -> Path | None:
    """从 SoM 截图或原始 screenshot 路径找对应 observation_som.txt。
    SoM:    .../artifacts/<task>/som/step_NNN_som.png
            → .../artifacts/<task>/step_NNN/observation_som.txt
    Screen: .../artifacts/<task>/step_NNN/screenshot.png
            → .../artifacts/<task>/step_NNN/observation_som.txt
    """
    if image_path.name.endswith("_som.png"):
        m = re.match(r"step_(\d+)_som\.png$", image_path.name)
        if not m:
            return None
        step_idx = m.group(1)
        obs = image_path.parent.parent / f"step_{step_idx}" / "observation_som.txt"
    elif image_path.name == "screenshot.png":
        obs = image_path.parent / "observation_som.txt"
    else:
        return None
    return obs if obs.exists() else None


def extract_som_marks_text(obs_path: Path) -> str:
    """从 observation_som.txt 提取 [SOM_MARKS] 块的内容（不含包裹 tag）。"""
    text = obs_path.read_text()
    # som obs 整个文件就是 [SOM_MARKS]...[/SOM_MARKS]
    m = re.search(r"\[SOM_MARKS\]\n(.*?)\n\[/SOM_MARKS\]", text, re.DOTALL)
    if m:
        return m.group(1)
    # fallback: 整个 text 去掉 wrapper
    return text.replace("[SOM_MARKS]", "").replace("[/SOM_MARKS]", "").strip()


def extract_ground_truth(obs_path: Path) -> list[str]:
    """从 observation_som.txt 提取 visible content 文字（去重保序）。

    抽取所有 [id=N] role 'label' 中的 label，排除空 label 和 RootWebArea。
    Role 包含: link, button, heading, StaticText, searchbox, textbox, combobox, ...
    """
    text = obs_path.read_text()
    pat = re.compile(r"\[id=\d+\]\s+(\w+)\s+'([^']+)'")
    seen = set()
    items = []
    for role, label in pat.findall(text):
        if role == "RootWebArea":
            continue
        if not label.strip():
            continue
        # 去重（同 label 出现多次只取第一次）
        if label in seen:
            continue
        seen.add(label)
        items.append((role, label))
    return items


def call_b0_proxy(image_b64: str, prompt: str) -> str:
    import requests

    auth_file = REPO / ".auth" / "qwen_api"
    api_key = ""
    if auth_file.exists():
        for line in auth_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("rp_"):
                api_key = line
                break
    api_key = os.environ.get("PROXY_API_KEY", api_key)
    if not api_key:
        raise RuntimeError("No PROXY_API_KEY")

    endpoint = "https://i5xpracyci.execute-api.eu-west-2.amazonaws.com/model-api/invoke"
    data_url = f"data:image/png;base64,{image_b64}"
    payload = {
        "model": "qwen.qwen3-vl-235b-a22b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.1,
    }
    headers = {"X-Api-Key": api_key, "Content-Type": "application/json"}

    resp = requests.post(endpoint, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    rj = resp.json()
    if "choices" in rj:
        msg = rj["choices"][0]["message"]
        c = msg.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "".join(b.get("text", "") for b in c if isinstance(b, dict))
    if "content" in rj and isinstance(rj["content"], list):
        return "".join(b.get("text", "") for b in rj["content"] if isinstance(b, dict))
    if isinstance(rj.get("content"), str):
        return rj["content"]
    return json.dumps(rj)[:2000]


# Cached B1 model（避免每张图重新加载）
_B1_CACHE: dict = {}


def call_b1_local(image_path: Path, prompt: str) -> str:
    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    os.environ.setdefault("CUDA_MPS_PIPE_DIRECTORY", "")
    os.environ.setdefault("CUDA_MPS_LOG_DIRECTORY", "")
    from p79.utils.torch_cuda_workarounds import apply_nvrtc_prod_fallback_if_needed
    apply_nvrtc_prod_fallback_if_needed()
    import time
    import torch
    from PIL import Image

    if "model" not in _B1_CACHE:
        from transformers import AutoProcessor, AutoModelForImageTextToText

        model_path = "Qwen/Qwen3-VL-4B-Instruct"
        print(f"[B1] loading {model_path}...", file=sys.stderr, flush=True)
        t0 = time.time()
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
        )
        model.eval()
        _B1_CACHE["model"] = model
        _B1_CACHE["processor"] = processor
        print(f"[B1] model loaded in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)

    model = _B1_CACHE["model"]
    processor = _B1_CACHE["processor"]

    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
    ]
    print(f"[B1] preparing inputs for {image_path.name}...", file=sys.stderr, flush=True)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to("cuda")
    print(f"[B1] generating (input_tokens={inputs.input_ids.shape[1]})...", file=sys.stderr, flush=True)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    gen = out[0, inputs.input_ids.shape[1]:]
    elapsed = time.time() - t0
    print(f"[B1] generated in {elapsed:.1f}s ({len(gen)} tokens, {len(gen)/elapsed:.1f} tok/s)", file=sys.stderr, flush=True)
    return processor.tokenizer.decode(gen, skip_special_tokens=True).strip()


def parse_response_lines(response_text: str) -> list[str]:
    lines = [l.strip().strip("•-*").strip() for l in response_text.splitlines() if l.strip()]
    pred = []
    for l in lines:
        l = re.sub(r"^[\d]+[.)]\s*", "", l).strip()
        if not l or len(l) > 80 or len(l) < 2:
            continue
        # 排除 meta 信息
        if any(kw in l.lower() for kw in (
            "here are", "screenshot", "i can see", "list of", "following", "based on",
            "in this image", "the image shows", "i'll list",
        )):
            continue
        pred.append(l)
    return pred


def evaluate(response_text: str, ground_truth: list[tuple[str, str]]) -> dict:
    """Compare model response vs ground truth (list of (role, label))."""
    gt_labels = [label for _, label in ground_truth]
    gt_set = set(name.lower() for name in gt_labels)

    pred = parse_response_lines(response_text)
    pred_set = set(p.lower() for p in pred)

    exact = pred_set & gt_set
    pred_only = pred_set - gt_set
    gt_missed = gt_set - pred_set

    # 部分匹配
    partial = set()
    for p in pred_only:
        for g in gt_missed:
            if len(p) >= 3 and (p in g or g in p):
                partial.add((p, g))

    # 数字 ID 输出（attention hijack indicator）
    numeric_predictions = [p for p in pred if re.fullmatch(r"\d+", p.strip())]

    return {
        "n_predicted": len(pred),
        "n_ground_truth": len(gt_labels),
        "n_exact_match": len(exact),
        "n_predicted_only": len(pred_only),
        "n_gt_missed": len(gt_missed),
        "n_partial_match": len(partial),
        "n_numeric_predictions": len(numeric_predictions),
        "exact_recall": round(len(exact) / max(1, len(gt_set)), 4),
        "exact_precision": round(len(exact) / max(1, len(pred_set)), 4),
        "predicted_sample": sorted(pred)[:20],
        "predicted_only_sample": sorted(pred_only)[:20],
        "missed_gt_sample": sorted(gt_missed)[:20],
        "partial_pairs_sample": sorted(partial)[:20],
        "numeric_predictions": numeric_predictions,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", action="append", required=True, help="可重复多次指定多张图")
    ap.add_argument("--backend", required=True, choices=["b0", "b1"])
    ap.add_argument("--output", default="/tmp/probe_result.json")
    ap.add_argument("--mode", default="som", choices=["som", "no-marks", "with-som-text"],
                    help="som=带标记截图（默认）; no-marks=用原始 screenshot.png; with-som-text=带标记+在 prompt 里给 [SOM_MARKS] 文本列表")
    args = ap.parse_args()

    # 选 prompt
    if args.mode == "no-marks":
        base_prompt = PROBE_PROMPT_NO_MARKS
    elif args.mode == "with-som-text":
        base_prompt = PROBE_PROMPT_WITH_SOM_TEXT
    else:
        base_prompt = PROBE_PROMPT_WITH_MARKS

    results = []
    for img_path in args.image:
        img = Path(img_path)
        if not img.exists():
            print(f"[skip] {img}: file not found", file=sys.stderr)
            continue
        obs = find_obs_txt(img)
        if obs is None:
            print(f"[skip] {img}: no observation_som.txt", file=sys.stderr)
            continue

        gt = extract_ground_truth(obs)
        n_marks = len([l for l in obs.read_text().splitlines() if l.startswith("[id=")])

        # 构造实际 prompt
        if args.mode == "with-som-text":
            som_text = extract_som_marks_text(obs)
            prompt = base_prompt.format(som_marks=som_text)
        else:
            prompt = base_prompt

        print(f"\n{'='*70}\n{args.backend.upper()} [{args.mode}] | {img.name} ({n_marks} marks, {len(gt)} GT)\n{'='*70}", file=sys.stderr)

        if args.backend == "b0":
            b64 = encode_image(img)
            resp = call_b0_proxy(b64, prompt)
        else:
            resp = call_b1_local(img, prompt)

        m = evaluate(resp, gt)
        m["image"] = str(img)
        m["backend"] = args.backend
        m["mode"] = args.mode
        m["n_marks"] = n_marks
        m["raw_response"] = resp
        m["ground_truth"] = [{"role": r, "label": l} for r, l in gt]
        results.append(m)

        print(f"  predicted={m['n_predicted']} | exact={m['n_exact_match']} ({100*m['exact_recall']:.1f}%) | partial={m['n_partial_match']} | numeric_ids_in_output={m['n_numeric_predictions']}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # 汇总表
    print(f"\n\n{'='*100}\nSummary ({args.backend.upper()}, mode={args.mode})\n{'='*100}")
    print(f"{'image':55s} {'marks':>6s} {'GT':>4s} {'pred':>5s} {'exact':>6s} {'partial':>8s} {'num_ids':>8s} {'recall':>7s}")
    for r in results:
        tag = Path(r['image']).parent.parent.name + '/' + Path(r['image']).name
        print(f"  {tag:55s} {r['n_marks']:6d} {r['n_ground_truth']:4d} "
              f"{r['n_predicted']:5d} {r['n_exact_match']:6d} {r['n_partial_match']:8d} "
              f"{r['n_numeric_predictions']:8d} {100*r['exact_recall']:6.1f}%")

    print(f"\nFull JSON: {args.output}")


if __name__ == "__main__":
    main()
