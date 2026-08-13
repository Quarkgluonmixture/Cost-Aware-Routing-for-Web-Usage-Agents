#!/usr/bin/env python3
"""Zero-preset framing ask, sent straight to the provider APIs.

WHY NOT opencode: the opencode CLI answered a one-line smoke correctly on both models,
then hung on the full 7.7 KB prompt -- its local server reached CPU-idle with **zero
socket fds**, i.e. it never opened a connection to either provider. Two different
providers hanging identically rules out the provider side. Rather than keep debugging a
wrapper, this talks to the OpenAI-compatible endpoints directly.

WHAT CHANGES BY DOING SO: the models can no longer read the repo themselves, so the
artifacts are inlined **verbatim, in full** -- not summarised, not excerpted. That keeps
the zero-preset property (the model still forms its own view of which numbers matter)
while removing the tool-use dependency. Two artifacts are NOT inlined, for size, and the
prompt says so explicitly rather than pretending the inventory is complete.

Keys come from the opencode auth store the user already configured; they are never
printed.
"""

from __future__ import annotations

import json
import pathlib
import sys
import threading
import urllib.error
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[2]
AUTH = pathlib.Path.home() / ".local/share/opencode/auth.json"
PROMPT = REPO / "docs/checkpoints/codex_prompts/frame_zero_preset_2026-08-13.md"
OUT_DIR = REPO / "docs/checkpoints/codex_outputs"

# Inlined verbatim. Ordering is the reading order the prompt suggests, nothing more.
ARTIFACTS = [
    "docs/analysis/cross_sites/routing_ceiling.md",
    "docs/analysis/cross_sites/evaluator_score_granularity.md",
    "docs/analysis/cross_sites/noise_floor_inventory.md",
    "docs/analysis/cross_sites/fusion_premium.md",
    "docs/analysis/cross_sites/abstention_learnability.md",
    "docs/analysis/cross_sites/early_abort_B0_classifieds.md",
    "docs/analysis/cross_sites/retry_vs_switch_label_supply.md",
    "docs/analysis/cross_sites/representation_deployment_profile.md",
    "docs/analysis/cross_sites/latency_decomposition.md",
    "docs/analysis/cross_sites/energy_carbon_audit.md",
    "docs/analysis/layered_evidence_status.md",
]
NOT_INLINED = {
    "docs/reference/master_bug_catalog.md": "1.58 MB — the defect ledger",
    "docs/analysis/cross_sites/failure_modes_per_cell.json": "38 KB — raw per-cell failure buckets",
}

# Per-model params, both learned the hard way on 2026-08-13:
#   - kimi-k3 rejects any temperature but 1 (HTTP 400 "only 1 is allowed for this model")
#   - deepseek-v4-pro spent all 16000 output tokens on reasoning and returned EMPTY content
#     with finish_reason=length. Reasoning tokens come out of the same budget, so the cap
#     has to cover the thinking AND the answer.
MODELS = [
    ("deepseek", "https://api.deepseek.com/chat/completions", "deepseek-v4-pro", 1.0, 64000),
    ("moonshotai", "https://api.moonshot.ai/v1/chat/completions", "kimi-k3", 1.0, 32000),
]


def build_prompt() -> str:
    base = PROMPT.read_text()
    # The original prompt promised shell access; that is no longer true.
    base = base.replace(
        "Work in the repo `/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents`.\n"
        "You have full shell access. Read whatever you want. The pointers below are a starting\n"
        "inventory, not a reading list, and several of them contradict each other on purpose —\n"
        "part of the job is deciding which numbers actually carry weight.",
        "You have no shell. Every artifact named below is reproduced **verbatim and in full** at\n"
        "the end of this message — not summarised, not excerpted. Several of them contradict each\n"
        "other on purpose; deciding which numbers actually carry weight is part of the job.")
    parts = [base, "\n\n---\n\n# APPENDIX: the artifacts, verbatim\n"]
    for rel in ARTIFACTS:
        p = REPO / rel
        if not p.is_file():
            parts.append(f"\n## `{rel}`\n\n**MISSING ON DISK.**\n")
            continue
        parts.append(f"\n## `{rel}`\n\n```markdown\n{p.read_text()}\n```\n")
    parts.append("\n## Not inlined (size), paths given for completeness only\n\n")
    for rel, why in NOT_INLINED.items():
        parts.append(f"- `{rel}` — {why}. You have NOT seen its contents; do not infer them.\n")
    return "".join(parts)


def ask(provider: str, url: str, model: str, temp: float, max_tok: int,
        prompt: str, results: dict) -> None:
    key = json.loads(AUTH.read_text())[provider]["key"]
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tok,
        "temperature": temp,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.loads(r.read())
        msg = d["choices"][0]["message"]
        results[model] = {
            "content": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or "",
            "usage": d.get("usage"),
            "finish_reason": d["choices"][0].get("finish_reason"),
        }
    except urllib.error.HTTPError as e:
        results[model] = {"error": f"HTTP {e.code}: {e.read()[:400].decode(errors='replace')}"}
    except Exception as e:  # noqa: BLE001 - surface whatever went wrong, do not mask it
        results[model] = {"error": f"{type(e).__name__}: {e}"}


def main() -> int:
    prompt = build_prompt()
    print(f"prompt: {len(prompt)} chars (~{len(prompt)//4} tokens), "
          f"{len(ARTIFACTS)} artifacts inlined, {len(NOT_INLINED)} declared-not-inlined")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "frame_zero_preset_DIRECT_prompt.md").write_text(prompt)

    results: dict = {}
    threads = [threading.Thread(target=ask, args=(p, u, m, t, mt, prompt, results))
               for p, u, m, t, mt in MODELS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rc = 0
    for _, _, model, _, _ in MODELS:
        r = results.get(model, {"error": "no result"})
        out = OUT_DIR / f"frame_zero_preset_{model}_2026-08-13.md"
        if "error" in r:
            print(f"  ✗ {model}: {r['error']}")
            out.write_text(f"# FAILED\n\n{r['error']}\n")
            rc = 1
            continue
        if not r["content"].strip():
            print(f"  ✗ {model}: EMPTY content (finish={r['finish_reason']}, "
                  f"reasoning_tokens={(r['usage'] or {}).get('completion_tokens_details', {}).get('reasoning_tokens')})"
                  f" -- raising the cap is the fix, not retrying")
            rc = 1
        body = r["content"]
        if r["reasoning"]:
            body += f"\n\n---\n\n<!-- reasoning trace, {len(r['reasoning'])} chars -->\n\n{r['reasoning']}"
        out.write_text(body)
        u = r["usage"] or {}
        print(f"  ✅ {model}: {len(r['content'])} chars content, "
              f"finish={r['finish_reason']}, tokens={u.get('total_tokens')} "
              f"(reasoning={u.get('completion_tokens_details', {}).get('reasoning_tokens')})")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
