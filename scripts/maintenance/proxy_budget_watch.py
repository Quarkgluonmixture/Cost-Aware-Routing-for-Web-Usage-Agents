#!/usr/bin/env python3
"""Watch the shared AWS-proxy budget pool and alert on the transitions that matter.

Born 2026-08-09 out of two failures on the same night, both worth keeping in the
docstring because both are easy to re-introduce:

**Failure 1 — a health check that cried wolf.** v1 collapsed every non-200 into
"budget exhausted" and fired a false alarm within 20 minutes: the proxy had
answered `503 Service Unavailable`, which is the KNOWN B-1880 behaviour (API
Gateway -> Bedrock emits isolated 503s and sustained ~8-10min 503 clusters) and
says nothing about money. Outcomes are now separated, and only a quota-shaped
rejection — confirmed twice in a row — counts as exhaustion.

**Failure 2 — a health check that measured itself.** v2 probed with
`max_tokens=1`. The real shop run hit `403` and stopped at 01:19; this probe
kept reporting `ok` with $0.222 remaining until 02:33. Seventy-four minutes of
"healthy" while the thing being watched was dead.

  The working hypothesis is that the proxy RESERVES against `max_tokens`: a real
  VWA step asks for 4096 and must reserve ~$0.02; a 1-token probe reserves
  essentially nothing, so it slips through on a balance no real request can use.

  So the probe now sends the SAME `max_tokens` as production (4096) with a
  two-token prompt. Reservation matches the real workload; actual spend stays at
  ~$0.00002 because the model stops immediately. A probe that does not have the
  real request's shape only ever proves that the probe can run.

  ⚠️ The reservation mechanism is INFERRED from the 74-minute gap, not confirmed
  by the provider. `--verify-reservation` re-tests it the next time the pool is
  near-empty: it sends both probe shapes back to back and reports whether the
  small one survives the large one's rejection. Until that runs, treat the
  4096 choice as a defensive default, not an established fact.

Alerts (each fires once):
  LOW        — remaining < threshold (default $1.00; well above one real step)
  EXHAUSTED  — two consecutive quota-shaped rejections
  TOPPED_UP  — balance jumps > $10 above the running minimum

Usage:
    .venv/bin/python3 scripts/maintenance/proxy_budget_watch.py &
    .venv/bin/python3 scripts/maintenance/proxy_budget_watch.py --once
    .venv/bin/python3 scripts/maintenance/proxy_budget_watch.py --verify-reservation
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
URL = "https://i5xpracyci.execute-api.eu-west-2.amazonaws.com/model-api/invoke"
STATE = REPO / "logs" / "proxy_budget_watch_state.json"
MODEL = "qwen.qwen3-vl-235b-a22b"

# Match production so the reservation matches. exp_v2_base.yaml sets
# api_strong.max_new_tokens = 4096; proxy_api_agent.py:713 forwards it as
# `max_tokens`. If that default ever changes, change this with it.
PROD_MAX_TOKENS = 4096


def load_key() -> str:
    env = os.environ.get("PROXY_API_KEY", "")
    if env:
        return env
    auth = REPO / ".auth" / "qwen_api"
    if not auth.exists():
        return ""
    for line in auth.read_text().splitlines():
        if line.strip().startswith("rp_"):
            return line.strip()
    return ""


def notify(topic: str, title: str, msg: str, prio: str = "default") -> None:
    if not topic:
        return
    subprocess.run(
        ["curl", "-s", "-H", f"Title: {title}", "-H", f"Priority: {prio}",
         "-d", msg, f"https://ntfy.sh/{topic}"],
        capture_output=True, timeout=30,
    )


def probe(key: str, max_tokens: int = PROD_MAX_TOKENS) -> tuple[float | None, str]:
    """(remaining_usd, outcome) — outcome in {ok, proxy_outage:N, quota:N, network:*, other:N}."""
    try:
        r = requests.post(
            URL,
            headers={"X-Api-Key": key, "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            },
            timeout=90,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"network:{type(exc).__name__}"

    if r.status_code == 200:
        try:
            return float(r.json()["metadata"]["remaining_quota"]["remaining_budget"]), "ok"
        except Exception:  # noqa: BLE001
            return None, "ok_but_unparsable"
    # 5xx = B-1880 proxy outage, NOT a budget signal. The runner rides these out
    # with its own ~11min retry; the watcher must not cry wolf on them.
    if r.status_code in (500, 502, 503, 504):
        return None, f"proxy_outage:{r.status_code}"
    body = (r.text or "")[:300].lower()
    if r.status_code in (402, 403, 429) or any(
        w in body for w in ("quota", "budget", "exceeded", "limit reached", "insufficient")
    ):
        return None, f"quota:{r.status_code}"
    return None, f"other:{r.status_code}"


def decide_alerts(state: dict, rem: float | None, outcome: str,
                  low_threshold: float) -> tuple[dict, list[str]]:
    """Pure state machine: (state, poll) -> (new_state, alerts_to_fire).

    Extracted from the poll loop specifically because the TOPPED_UP rule was
    wrong on first write and a loop is not testable. Keys of `state`:
    `fired` (set), `lowest` (float|None), `consecutive_quota` (int),
    `seen_quota_rejection` (bool).
    """
    fired: set = set(state.get("fired", set()))
    lowest = state.get("lowest")
    consecutive_quota = int(state.get("consecutive_quota", 0))
    seen_quota = bool(state.get("seen_quota_rejection", False))
    alerts: list[str] = []

    if outcome.startswith("quota"):
        consecutive_quota += 1
        seen_quota = True
    elif outcome == "ok":
        consecutive_quota = 0

    if rem is not None:
        # Evaluate BEFORE folding rem into `lowest` — otherwise the first read
        # after a restart becomes its own baseline and TOPPED_UP never fires.
        recovered = seen_quota
        jumped = lowest is not None and rem > lowest + 10
        lowest = rem if lowest is None else min(lowest, rem)
        if rem < low_threshold and "LOW" not in fired:
            fired.add("LOW")
            alerts.append("LOW")
        if (recovered or jumped) and "TOPPED_UP" not in fired:
            fired.add("TOPPED_UP")
            alerts.append("TOPPED_UP")
            seen_quota = False
    elif consecutive_quota >= 2 and "EXHAUSTED" not in fired:
        fired.add("EXHAUSTED")
        alerts.append("EXHAUSTED")

    return (
        {"fired": fired, "lowest": lowest,
         "consecutive_quota": consecutive_quota,
         "seen_quota_rejection": seen_quota},
        alerts,
    )


def verify_reservation(key: str) -> int:
    """Test the reservation hypothesis: does a 1-token probe outlive a 4096 one?

    Only informative when the pool is near-empty. Run it then; it is the
    difference between a documented mechanism and a plausible story.
    """
    big_rem, big = probe(key, PROD_MAX_TOKENS)
    small_rem, small = probe(key, 1)
    print(f"max_tokens={PROD_MAX_TOKENS:<5} -> {big:<18} remaining={big_rem}")
    print(f"max_tokens=1     -> {small:<18} remaining={small_rem}")
    if big.startswith("quota") and small == "ok":
        print("\n✅ HYPOTHESIS SUPPORTED — the proxy reserves against max_tokens.")
        print("   A small probe reports healthy on a balance production cannot use.")
    elif big == "ok" and small == "ok":
        print("\n— inconclusive: pool still has headroom. Re-run when near empty.")
    elif big.startswith("quota") and small.startswith("quota"):
        print("\n❌ NOT SUPPORTED here — both shapes rejected (pool fully empty,")
        print("   or rejection is not reservation-based). Re-run at a partial balance.")
    else:
        print(f"\n— inconclusive: big={big} small={small}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=600, help="seconds between polls")
    ap.add_argument("--max-polls", type=int, default=144, help="ceiling (144 x 10min = 24h)")
    ap.add_argument("--low-threshold", type=float, default=1.00,
                    help="USD below which to fire LOW; keep well above one real step")
    ap.add_argument("--once", action="store_true", help="single poll, print, exit")
    ap.add_argument("--verify-reservation", action="store_true")
    args = ap.parse_args()

    key = load_key()
    if not key:
        print("ERROR: PROXY_API_KEY unset and .auth/qwen_api has no rp_ key")
        return 1
    topic = os.environ.get("NTFY_TOPIC", "p79-claude").strip()

    if args.verify_reservation:
        return verify_reservation(key)

    if args.once:
        rem, outcome = probe(key)
        print(json.dumps({"remaining": rem, "outcome": outcome}, indent=1))
        return 0

    state: dict = {"fired": set(), "lowest": None,
                   "consecutive_quota": 0, "seen_quota_rejection": False}
    history: list[dict] = []
    STATE.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(args.max_polls):
        rem, outcome = probe(key)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        state, alerts = decide_alerts(state, rem, outcome, args.low_threshold)

        for alert in alerts:
            if alert == "LOW":
                notify(topic, f"P79 proxy 预算 < ${args.low_threshold:.2f}",
                       f"[{stamp}] remaining=${rem:.4f}。注意: 探针用生产同款 max_tokens="
                       f"{PROD_MAX_TOKENS}, 所以这个余额是真实负载**还能用**的余额。", "high")
            elif alert == "TOPPED_UP":
                notify(topic, "P79 proxy 额度已到账",
                       f"[{stamp}] remaining=${rem:.2f} — chain 可续跑。"
                       f"resume:true 会跳过已完成 episode。"
                       f"\n\n⭐ 顺手做: `proxy_budget_watch.py --verify-reservation` "
                       f"在余额跌到 <$1 时重跑一次, 就能确认/推翻 max_tokens 预留假说。", "high")
            elif alert == "EXHAUSTED":
                notify(topic, "P79 proxy 预算耗尽（已二次确认）",
                       f"[{stamp}] 连续 2 轮 quota 型拒绝（{outcome}）。运行中的 B0 run 会 "
                       f"fail-fast 停止并各自推送 run-abort 告警。", "urgent")

        history.append({"t": stamp, "remaining": rem, "outcome": outcome})
        STATE.write_text(json.dumps(
            {"lowest": state["lowest"], "fired": sorted(state["fired"]),
             "consecutive_quota": state["consecutive_quota"],
             "seen_quota_rejection": state["seen_quota_rejection"],
             "probe_max_tokens": PROD_MAX_TOKENS,
             "history": history[-60:]}, indent=1))
        time.sleep(args.interval)

    notify(topic, "P79 预算 watcher 退出", f"轮询上限到达，最后一轮 {history[-1]}", "low")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
