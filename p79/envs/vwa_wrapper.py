from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)
import os
import re
from urllib.parse import urlparse
try:
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    np = None
from PIL import Image
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

# B-1860: single-source coordinate normalizer (Qwen 0-1000 contract). All
# coord sites below (click / type-focus / select_option / hover) call this so
# the per-dimension 0-1000↔[0,1] heuristic lives in ONE place (action_utils),
# never copy-pasted across the wrapper. action_utils is a stdlib-only leaf
# module → no circular import.
from p79.backends.action_utils import normalize_coordinate_pair

# B-422 (/stress A1.3 v9 Mode A P1-11, 2026-05-17): named injection
# distance thresholds. Pre-fix `_inject_css_dropdown_options` used 150 px
# inline and `_inject_select_options` used 100 px inline — same primitive
# (nearest-AXTree-node match), two magic numbers, no doc trail. Constants
# now module-level: CSS custom dropdowns have larger triggers (e.g. Reddit
# sort button) so 150 px tolerance is correct; native combobox lines are
# tightly bound to the SELECT so 100 px is correct. Renaming makes the
# intent visible; touching one is now a single-line edit.
_INJECT_DISTANCE_CSS_DROPDOWN_PX = 150
_INJECT_DISTANCE_NATIVE_SELECT_PX = 100


def _log_coord_normalization(tags: Dict[str, Any], context: str, raw_coord: Any) -> None:
    """B-1860: surface noteworthy coordinate-normalization events to the runner
    log (the full per-dimension tags also land in the step JSONL telemetry).

    Two events are worth a log line so a model regression is observable WITHOUT
    re-parsing every step record:
      * ``true_oob`` — a dimension > 1000 (after /1000 still > 1.0). We do NOT
        silently clamp the encoding away (the eps-clamp downstream only keeps
        the click inside the viewport); this is a grounding miss, logged at
        WARNING so it stands out.
      * ``dead_zone`` — a raw value in (1.1, 10], ambiguous between a genuine
        near-corner 0-1000 coord and a fat-fingered out-of-[0,1] normalized
        coord. B0/B1/B2 probes show NONE; logged at DEBUG for observability.
    """
    if tags.get("true_oob"):
        logger.warning(
            "B-1860 coord true_oob (%s): raw=%r regimes=(%s,%s) — grounding "
            "miss, NOT format-clamped",
            context, raw_coord, tags.get("x_regime"), tags.get("y_regime"),
        )
    elif tags.get("dead_zone"):
        logger.debug(
            "B-1860 coord dead_zone (%s): raw=%r in (1.1,10] — ambiguous "
            "0-1000 vs out-of-[0,1] normalized (probes show none)",
            context, raw_coord,
        )


@dataclass
class P79Observation:
    text: str
    image: Optional[Any] = None   # 可能是 PIL / np / base64 / path，先 Any
    url: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None
    # VWA obs_nodes_info: maps str(element_id) -> {"union_bound": [x,y,w,h], ...}
    # Populated from info["observation_metadata"]["text"]["obs_nodes_info"] by _to_p79_obs.
    obs_nodes_info: Optional[Dict[str, Any]] = None

# Fuzzy match JS function for select_option: exact → case-insensitive → keyword overlap.
# Injected into page.evaluate() calls to handle label mismatches between model output
# and actual option text (e.g. "Price: Low to High" vs "Lower price first").
#
# B-481 (/stress A1.25 GRL Chunk 2 P0-2-AB* + P1-1-BC* + parallel A1.4 B-453,
# 2026-05-17): `_fuzzyFind` now returns `{match, stage}` instead of bare
# candidate so the dispatch wrapper above can surface which fuzzy tier
# fired. Pre-fix the callsite couldn't distinguish exact / case-insensitive /
# keyword-overlap matches from a no-match fallthrough — paper §3.5 evidence
# layer couldn't audit prompt-vs-runtime "exact-text" contract drift.
# Stage tokens: 'exact' | 'ci' | 'fuzzy' | 'none' (paired with `match=null`).
_FUZZY_MATCH_JS = """
const _fuzzyFind = (candidates, label) => {
    // 1. Exact match
    const exact = candidates.find(c => c.text === label || (c.value && c.value === label));
    if (exact) return {match: exact, stage: 'exact'};
    // 2. Case-insensitive
    const lower = label.toLowerCase().trim();
    const ci = candidates.find(c =>
        c.text.toLowerCase().trim() === lower || (c.value && c.value.toLowerCase() === lower));
    if (ci) return {match: ci, stage: 'ci'};
    // 3. Keyword overlap
    const norm = s => s.toLowerCase().replace(/[^a-z0-9]/g, ' ').replace(/\\s+/g, ' ').trim();
    const stops = new Set(['the','a','an','to','by','of','in','on','for','and','or','is','it']);
    const kw = s => norm(s).split(' ').filter(w => w.length > 1 && !stops.has(w));
    const lkw = kw(label);
    if (!lkw.length) return {match: null, stage: 'none'};
    let best = null, bestS = 0;
    for (const c of candidates) {
        const ckw = kw(c.text);
        let s = 0;
        for (const l of lkw) {
            for (const o of ckw) {
                if (l === o || o.startsWith(l) || l.startsWith(o)) { s++; break; }
                let p = 0; const m = Math.min(l.length, o.length);
                while (p < m && l[p] === o[p]) p++;
                if (p >= 4) { s++; break; }
            }
        }
        if (s > bestS) { bestS = s; best = c; }
    }
    return bestS >= 2 ? {match: best, stage: 'fuzzy'} : {match: null, stage: 'none'};
};
"""


class VWAWrapper:
    """
    Thin wrapper around (Visual)WebArena ScriptBrowserEnv.

    - reset(options={"config_file": ...})
    - step(action)
    """

    def __init__(
        self,
        headless: bool = True,
        observation_type: str = "accessibility_tree",
        current_viewport_only: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        sleep_after_execution: float = 0.5,
        dry_run: bool = False,
        benchmark: str = "visualwebarena",
    ) -> None:
        self.headless = headless
        self.observation_type = observation_type
        self.current_viewport_only = current_viewport_only
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.sleep_after_execution = sleep_after_execution
        self.dry_run = dry_run
        self.benchmark = benchmark
        # Fire-6 RCA Stage C1b (/stress 2026-05-20): per-condition observation
        # mode, set by the runner before each condition. The single mode-gating
        # chokepoint for the submodule's screenshot-timeout recovery: dom =
        # screenshot is artifact-only → blank recovery is non-fatal; any other
        # mode = screenshot may be decision-input → re-raise (fatal, as before).
        # None (default) is treated as decision-input (fail-safe = fatal).
        self.observation_mode: Optional[str] = None

        self._env = None  # lazy init
        # 保存上一次 obs 的 obs_nodes_info，供 select_option element_id 路径使用
        self._last_obs_nodes_info: Optional[Dict[str, Any]] = None
        # Sequential SoM identifier contract (2026-05-25, codex review): which id
        # namespace the agent's element_id is in for the CURRENT step.
        #   "native" -> element_id IS the VWA AXTree nodeId (dom / p-prompt /
        #               default); native-id dispatch passes it through unchanged.
        #   "seq"    -> element_id is a 1..K sequential SoM id; native-id dispatch
        #               MUST translate via obs_nodes_info[seq]["native_element_id"]
        #               and FAIL-CLOSED (no-op) if the seq is absent — never pass a
        #               seq through as a native nodeId (would silently mis-click).
        # Set to "native" on every obs production (reset/step/navigate) and to
        # "seq" by set_dispatch_obs_nodes_info() (runner calls it for SoM-family).
        self._dispatch_id_namespace: str = "native"
        # B-509 (/stress A1.25 GRL Chunk 3 P1-2-BC*, 2026-05-17): per-step
        # dialog event accumulator. `_on_dialog` appends each accepted/
        # dismissed dialog; step() drains + clears at end + stamps into
        # info["dialog_meta"]. Playwright sync API runs dialog handler on
        # main thread (same as step()), so no lock needed.
        self._dialogs_this_step: list = []
        # B-158 (/stress A1.3 v8 codex P1-B2, 2026-05-16): dialog handler is
        # now registered at the BrowserContext level (auto-fires for every
        # new Page in the context) instead of per-Page. Previously the
        # ``_dialog_registered_page`` per-page tracker only attached the
        # listener to the initial page; new tabs opened via
        # ``target=_blank`` / ``window.open`` left their ``Page.dialog``
        # events unhandled → confirm/alert blocked navigation until timeout
        # (Classifieds delete operations hit this hard).
        self._dialog_registered_context: Optional[Any] = None

    def _lazy_init(self) -> None:
        if self._env is not None:
            return

        # Ensure environment variables are set to avoid crash on import
        # User should set these to real values for actual tasks
        dataset = "webarena" if self.benchmark == "webarena" else "visualwebarena"
        # Always set DATASET to match benchmark — shell env may have stale value
        # (e.g. vwa_env_remote.sh exports DATASET=visualwebarena for VWA runs)
        os.environ["DATASET"] = dataset

        if dataset == "webarena":
            required_vars = ["REDDIT", "SHOPPING", "SHOPPING_ADMIN", "GITLAB", "WIKIPEDIA", "MAP", "HOMEPAGE"]
        else:
            required_vars = ["REDDIT", "SHOPPING", "WIKIPEDIA", "HOMEPAGE", "CLASSIFIEDS", "CLASSIFIEDS_RESET_TOKEN"]
        for var in required_vars:
            if var not in os.environ:
                # Set dummy values if not present
                # Use example.com to allow page load without local server
                os.environ[var] = "https://example.com"

        # Workaround: playwright sync API raises if an asyncio event loop is
        # running on this thread. This can happen after VWA program_html
        # evaluators or HuggingFace hub (both use asyncio/httpx).
        #
        # B-1581 v2 (/stress A2.11 P0-2-A*B 2026-05-18, user Q1=A): replaces
        # B-1581 v1 (today fire-day hot fix) unconditional new_event_loop
        # band-aid with proper resource lifecycle. v1 left stale loops + their
        # Playwright resources (websocket connections, Frame instances) bound
        # in browser process — suspected root cause of red 99s busy-wait at
        # 2026-05-18 13:28:06 fire (orphaned websocket jamming page settle
        # channel). v2 explicitly close()s stale loop on detection. In
        # _lazy_init() there's no env yet (about to create), so loop-only
        # cleanup is sufficient; reset() path handles env force-rebuild.
        import asyncio as _asyncio
        try:
            _stale = _asyncio.get_running_loop()
            logger.warning(
                "B-1581 v2 (_lazy_init): stale asyncio loop bound to thread "
                "(likely from VWA browser_env/async_envs asyncio.run leak); "
                "closing stale loop + installing fresh. Stale=%r closed=%s",
                _stale, _stale.is_closed(),
            )
            try:
                if not _stale.is_closed():
                    _stale.close()
                    logger.info("B-1581 v2 (_lazy_init): stale loop closed")
            except Exception:
                logger.warning(
                    "B-1581 v2 (_lazy_init): failed to close stale loop",
                    exc_info=True,
                )
        except RuntimeError:
            pass  # Clean state — no stale loop
        _asyncio.set_event_loop(_asyncio.new_event_loop())

        from browser_env import ScriptBrowserEnv  # provided by (Visual)WebArena package

        self._env = ScriptBrowserEnv(
            headless=self.headless,
            observation_type=self.observation_type,
            current_viewport_only=self.current_viewport_only,
            viewport_size={"width": self.viewport_width, "height": self.viewport_height},
            sleep_after_execution=self.sleep_after_execution,
        )

    def reset(self, config_file: str) -> Tuple[P79Observation, Dict[str, Any]]:
        if self.dry_run:
            # Return dummy black image for dry run to satisfy agent
            dummy_img = Image.new('RGB', (self.viewport_width, self.viewport_height), color='black')
            # Sequential SoM contract (2026-05-25, codex round-3 P2): reset the
            # dispatch id-namespace to native on every obs production, including
            # the dev-only dry-run path, so it can never be left stale "seq".
            self._install_native_obs_nodes_info(None)
            return P79Observation(text="[DRY_RUN]", image=dummy_img), {"dry_run": True}

        # B-1581 v2 (/stress A2.11 P0-2-A*B 2026-05-18, user Q1=A): MUST run
        # BEFORE self._lazy_init() so env=None propagates to force-rebuild.
        # On stale loop detect: close stale env (releases Playwright websocket
        # refs to old loop) → close stale loop → self._env = None → install
        # fresh loop. lazy_init below then rebuilds ScriptBrowserEnv from clean
        # state. Pre-fix v1 only swapped the loop binding while keeping the
        # existing env's references to old-loop resources alive — empirical
        # red 99s busy-wait suspected to be orphaned websocket jamming page
        # settle. Cost amortization: stale detection rare (only after prior
        # asyncio.run leak from VWA browser_env/async_envs); per-reset
        # detection cost negligible.
        import asyncio as _asyncio
        _stale_loop = None
        try:
            _stale_loop = _asyncio.get_running_loop()
            logger.warning(
                "B-1581 v2 (reset): stale asyncio loop bound to thread (likely "
                "from prior episode's VWA browser_env/async_envs asyncio.run); "
                "forcing env+loop close + ScriptBrowserEnv rebuild. "
                "Stale=%r closed=%s",
                _stale_loop, _stale_loop.is_closed(),
            )
        except RuntimeError:
            pass  # Clean state — no stale loop, keep env

        if _stale_loop is not None:
            # Close stale env first — releases Playwright resource refs to old loop
            if self._env is not None:
                try:
                    self._env.close()
                    logger.info(
                        "B-1581 v2 (reset): closed stale env (forcing lazy_init rebuild)"
                    )
                except Exception:
                    logger.warning(
                        "B-1581 v2 (reset): failed to close stale env",
                        exc_info=True,
                    )
                self._env = None
            # Close stale loop
            try:
                if not _stale_loop.is_closed():
                    _stale_loop.close()
                    logger.info("B-1581 v2 (reset): closed stale loop")
            except Exception:
                logger.warning(
                    "B-1581 v2 (reset): failed to close stale loop",
                    exc_info=True,
                )

        # Always install fresh loop (covers both stale-rebuild and clean-state)
        _asyncio.set_event_loop(_asyncio.new_event_loop())

        # lazy_init rebuilds env if force-closed above; no-op if env survived
        self._lazy_init()
        assert self._env is not None

        # B-1831 (/stress 2026-05-22, user A): env.reset initial-navigation
        # Page.goto transient-timeout retry. The eval path has 3 retries
        # (environment.py evaluate); the reset path had 0 — a real robustness
        # asymmetry (Fire-6 B0 som cls task 76: env.reset homepage Page.goto 30s
        # timeout → PaperGradeAbort killed the whole condition for ONE transient
        # docker hiccup). Reset is PRE-episode: no model call, no env action, no
        # episode summary counted until reset succeeds → retry does NOT change
        # model behavior or SR. A recovered reset is infra recovery, NOT
        # benchmark_noise, NOT in the SR denominator. ONLY PlaywrightTimeoutError
        # (navigation/goto) is retried; any other exception keeps the original
        # close+raise. All retries exhausted → close + raise (PaperGradeAbort
        # upstream, as before). Does NOT change the VWA global goto timeout
        # (still 30s per attempt).
        #
        # B-1833 (2026-05-22): budget bumped 3 → 5 attempts after the transient
        # stall RECURRED — R21790 task 76 + R12265 task 106 both exhausted the
        # 3×30s = 90s window and PaperGradeAborted the whole cls som condition
        # (≈50% abort rate per full som run = burning B0 API $ on partial re-fires).
        # The stall is TRANSIENT not persistent: curl localhost:9980 = 0.2s healthy
        # between aborts, A100 idle with 52G RAM free → not resource pressure. More
        # attempts (5×30s = 150s + escalating 2/3/4/5s backoff) absorb longer
        # transient stalls WITHOUT weakening the persistent-failure fail-closed
        # (all 5 exhausted → close+raise as before). Per-attempt 30s unchanged
        # (user A: do not widen the VWA global goto timeout).
        import time as _time
        _max_reset_attempts = 5  # initial attempt + 4 retries (B-1833 budget bump)
        _reset_timeout_count = 0
        _reset_attempt_latencies_ms: list = []
        _reset_recovered = False
        _reset_retry_reason = None
        obs = info = None
        for _attempt in range(_max_reset_attempts):
            _t0 = _time.monotonic()
            try:
                obs, info = self._env.reset(options={"config_file": config_file})
                _reset_attempt_latencies_ms.append((_time.monotonic() - _t0) * 1000.0)
                _reset_recovered = _attempt > 0
                break
            except PlaywrightTimeoutError as _goto_to:
                _reset_timeout_count += 1
                _reset_attempt_latencies_ms.append((_time.monotonic() - _t0) * 1000.0)
                # B-1833 P1-3 (3-AI /stress 2026-05-22, codex F2): record the ACTUAL
                # timeout site, not a hardcoded "Page.goto". This except wraps the WHOLE
                # self._env.reset() (setup + page.goto + first observation/screenshot
                # capture), so a som/vision first-frame screenshot/CDP timeout is also
                # caught here — labeling every catch "Page.goto" was false telemetry.
                _reset_retry_reason = (
                    f"{type(_goto_to).__name__}: "
                    f"{(str(_goto_to).splitlines() or [''])[0][:140]}"
                )
                if _attempt < _max_reset_attempts - 1:
                    logger.warning(
                        "B-1831/B-1833: env.reset PlaywrightTimeoutError (attempt %d/%d) "
                        "— transient pre-episode retry after fresh-browser rebuild + backoff. %s",
                        _attempt + 1, _max_reset_attempts, _goto_to,
                    )
                    _time.sleep(2.0 + _attempt)  # 2s, 3s, 4s, 5s backoff
                    # B-1833 P1-1 (3-AI /stress, gemini G1 + codex F3): refresh the
                    # browser between retries. self.close() force-closes context_manager
                    # even when reset_finished=False (vwa_wrapper.py:1385-1392) — a
                    # timed-out reset leaves a half-built env whose Playwright context /
                    # websocket can be hung (B-1581), so a same-instance retry re-hits it
                    # (hollow) AND leaks the context. Each retry now gets a fresh browser,
                    # mirroring environment.py eval path's per-retry maximally-clean
                    # context. Without this the B-1833 budget bump is a hollow number.
                    try:
                        self.close()
                    except Exception:
                        logger.warning(
                            "B-1833: env close before retry-rebuild failed", exc_info=True
                        )
                    self._lazy_init()  # close() nulled self._env → rebuild fresh
                    continue
                # All retries exhausted → real substrate failure, fail-closed.
                self.close()
                raise
            except Exception:
                # Non-timeout reset failure: original behavior (close + raise).
                self.close()
                raise
        # Episode-level reset telemetry (read by runner → episode_summary).
        # Infra recovery only: does NOT set benchmark_noise, does NOT enter the
        # SR denominator (episode SR is decided by the agent run that follows a
        # SUCCESSFUL reset).
        self._reset_goto_telemetry = {
            "reset_goto_timeout_count": _reset_timeout_count,
            "reset_goto_retry_count": max(0, len(_reset_attempt_latencies_ms) - 1),
            "reset_goto_recovered": _reset_recovered,
            "reset_goto_latency_ms_per_attempt": _reset_attempt_latencies_ms,
            "reset_retry_reason": _reset_retry_reason,  # B-1833 P1-3: real exception, not hardcoded
        }

        # Auto-accept confirm/alert dialogs (e.g. Classifieds delete operations use
        # onclick="return confirm(...)" which blocks navigation if not dismissed).
        # B-158 (/stress A1.3 v8 codex P1-B2, 2026-05-16): register at the
        # BrowserContext level so every Page (including newly-opened tabs via
        # window.open / target=_blank from B-157) inherits the dialog handler
        # automatically. Identity-check guards against accumulating duplicate
        # listeners across episode resets (same context reused → skip).
        try:
            ctx = self._env.context
            if ctx is not self._dialog_registered_context:
                # Existing pages (the initial reset page)
                for _p in ctx.pages:
                    _p.on("dialog", self._on_dialog)
                # Future pages: ``context.on("page")`` fires on each new tab,
                # we attach the dialog listener there so the chain stays
                # current without manual per-step bookkeeping.
                ctx.on("page", lambda new_page: new_page.on("dialog", self._on_dialog))
                self._dialog_registered_context = ctx
        except Exception as _e:
            logger.warning("Failed to register dialog handler at context level: %s", _e)

        p79_obs = self._to_p79_obs(obs, info)
        self._install_native_obs_nodes_info(p79_obs.obs_nodes_info)
        return p79_obs, info

    def _install_native_obs_nodes_info(self, obs_nodes_info: Optional[Dict[str, Any]]) -> None:
        """Install the env's native (nodeId-keyed) obs_nodes_info for dispatch and
        reset the id-namespace to "native". Called on every obs production
        (reset / step / navigate); the runner may subsequently override with
        set_dispatch_obs_nodes_info() for SoM-family modes (Sequential SoM
        identifier contract, 2026-05-25)."""
        self._last_obs_nodes_info = obs_nodes_info
        self._dispatch_id_namespace = "native"

    def set_dispatch_obs_nodes_info(self, obs_nodes_info: Optional[Dict[str, Any]]) -> None:
        """Override the dispatch obs_nodes_info to a SEQUENTIAL-keyed map for the
        current step. The runner calls this for SoM-family modes after building
        the observation so that `click [seq]` resolves to the right bbox/native
        id. Marks the dispatch id-namespace "seq"; native-id fallback paths then
        translate + fail-closed (see _resolve_native_id). AXTree modes never call
        this, keeping the native nodeId map installed at obs-production time."""
        self._last_obs_nodes_info = obs_nodes_info
        self._dispatch_id_namespace = "seq"

    def _resolve_native_id(self, eid: Any) -> Optional[int]:
        """Translate a dispatch element_id to the native VWA AX nodeId for the
        native-id dispatch / serializer paths (create_id_based_action).

        - namespace "native": the id IS already the native nodeId -> int(eid).
        - namespace "seq": return the seq entry's embedded ``native_element_id``
          if present, else None = FAIL-CLOSED. A None return MUST make the caller
          no-op — a seq id (or a hallucinated id absent from the seq map) must
          never be passed through as a native nodeId (codex review 2026-05-25:
          validators accept any positive element_id, so a hallucinated
          ``click [1]`` would otherwise silently click real AX node 1)."""
        try:
            eid_str = str(int(eid))
        except (TypeError, ValueError):
            return None
        if self._dispatch_id_namespace != "seq":
            return int(eid)
        entry = (self._last_obs_nodes_info or {}).get(eid_str)
        if isinstance(entry, dict) and "native_element_id" in entry:
            try:
                return int(entry["native_element_id"])
            except (TypeError, ValueError):
                return None
        return None  # fail-closed: seq absent / no native id mapping

    def get_all_tab_titles(self) -> list[tuple[str, str]]:
        """Return (url, title) for every open tab. Used for start-URL health checks."""
        if self.dry_run or self._env is None:
            return []
        try:
            pages = self._env.context.pages
            return [(p.url, p.title()) for p in pages]
        except Exception:
            return []

    def _gate_screenshot_timeout(self, info: Dict[str, Any]) -> None:
        """Fire-6 RCA Stage C1b (/stress 2026-05-20): dom-only screenshot-timeout
        fatality gate — the SINGLE mode-gating chokepoint.

        async_envs.astep recovers a Page.screenshot timeout to a blank
        placeholder + ``info['screenshot_timeout_recovered']=True``, mode-
        agnostically. Fatality is decided HERE by observation_mode:
          - ``dom``: screenshot is artifact-only (agent decides on AXTree; the
            blank is discarded by prepare_observation_for_mode) → non-fatal,
            log + continue.
          - any other mode (som / vision / phantom_* / None): screenshot may be
            decision-input → restore the original FATAL behavior by raising a
            Page.screenshot Timeout (classify_timeout() tags agent_observation →
            runner quarantines). None is fail-safe-fatal. This preserves paper-
            grade integrity: a decision-input mode never silently consumes a
            blank image.
        """
        if not info.get("screenshot_timeout_recovered"):
            return
        if self.observation_mode == "dom":
            logger.warning(
                "C1b: Page.screenshot timeout recovered to blank placeholder "
                "(observation_mode=dom, artifact-only) — episode continues non-fatal."
            )
            return
        self.close()
        raise TimeoutError(
            "Page.screenshot Timeout 30000ms — C1b non-fatal recovery applies to "
            f"dom mode only; observation_mode={self.observation_mode!r} is "
            "(potential) decision-input, restoring fatal behavior."
        )

    def step(self, action_json: Dict[str, Any]) -> Tuple[P79Observation, float, bool, bool, Dict[str, Any]]:
        if self.dry_run:
            dummy_img = Image.new('RGB', (self.viewport_width, self.viewport_height), color='black')
            # Sequential SoM contract (2026-05-25, codex round-3 P2): reset
            # dispatch id-namespace to native on the dry-run step path too.
            self._install_native_obs_nodes_info(None)
            return P79Observation(text="[DRY_RUN]", image=dummy_img), 0.0, False, False, {"dry_run": True}

        self._lazy_init()
        assert self._env is not None

        from browser_env import (
            create_id_based_action,
            create_mouse_click_action,
            create_mouse_hover_action,
            create_scroll_action,
            create_stop_action,
            create_go_back_action,
            create_go_forward_action,
            create_page_focus_action,
            create_keyboard_type_action,
            create_none_action,
            create_playwright_action,
            create_goto_url_action,
        )

        action_type = (action_json.get("action_type") or "").lower().strip()
        action = None
        _type_needs_enter = False
        # B-156 (/stress A1.3 v8 Claude F5 + codex P2-B7, 2026-05-16):
        # locator-route dispatch result is captured here so it survives into
        # the post-step ``info`` dict (and from there into StepRecordV2 via
        # the runner). Paper §3 evidence layer (locator-route ON_TARGET rate)
        # depends on this telemetry being audit-able from JSONL alone.
        _locator_route_meta: Optional[Dict[str, Any]] = None
        # B-420 (/stress A1.3 v9 Mode B P1-5 OOB, 2026-05-17): same pattern
        # for select_option dispatch — separate dict so info contract stays
        # symmetric with locator_route_meta (action_kind discriminator).
        _select_option_meta: Optional[Dict[str, Any]] = None
        # Protocol Reset #5 (action-set restore, 2026-05-20): goto dispatch
        # telemetry. None unless the step is a goto; otherwise
        # {action_kind, url, host, allowed, error?}.
        _goto_meta: Optional[Dict[str, Any]] = None
        # B-510 (/stress A1.25 GRL Chunk 3 P1-3-AB, 2026-05-17): per-step
        # runtime settle-tax accumulator. Pre-fix `sleep_after_execution=0.5`
        # entered `latency_ms.total` per action mix (more TYPE/SELECT → more
        # settle-tax) — paper §4 phantom latency hero number could partially
        # attribute mode-delta to runtime-wait composition, not pure
        # model/representation efficiency. Locator-dispatch internal sleeps
        # excluded (they're inside dispatch helper); this counter covers only
        # wait_for_timeout calls in step() itself. Stamped into info at end.
        _runtime_sleep_ms = 0
        # B-512 (/stress A1.5b Phase 2 P0-1-C gemini OOB, 2026-05-17): wrapper-
        # normalized canonical action form. Pre-fix step_record["action"]
        # carried the agent's RAW emit (B0 enum `scroll_direction:"down"` vs
        # B1/B2 free-form `delta:[dx,dy]`) → cross-baseline evidence layer
        # asymmetry on action vocabulary. The wrapper at L395-414 already
        # collapses both to `create_scroll_action(direction=...)` (execution
        # identical since paper §67 schema reform), but the normalized form
        # was never recorded. `action_executed` exposes wrapper-level
        # alignment in step JSONL so reviewer reading evidence layer can
        # verify execution-layer parity from disk alone.
        #
        # B-553 (/stress A1.5 P1-3-AB* Claude+codex OOB, 2026-05-17): extended
        # from scroll-only (B-512) to click + type dispatch paths. Each
        # branch sets `_action_executed` with shape
        # `{"action_type", "dispatch_path", "fallback"}`:
        #   - click_eid:  element_id_locator_route / element_id_framework
        #   - click_coord: coord_mouse_click
        #   - type_coord:  coord_locator_route / coord_keyboard_fallback
        #   - type_eid:    element_id_locator_route / element_id_framework
        #                  / noop_invalid_element_id (B-506)
        #   - scroll:      {action_type, direction} (legacy B-512 shape kept)
        # `fallback=True` means the Cluster 1 locator-route walk-up FAILED
        # and the wrapper fell back to legacy framework path (interesting
        # for paper §3 cross-baseline taxonomy — B0 235B rarely falls back,
        # B1/B2 4B more often). `None` only when action_type is none of
        # {click, type, scroll} (e.g. back / forward / tab) — those have
        # no wrapper-level normalization layer.
        _action_executed: Optional[Dict[str, Any]] = None

        if action_type == "click" and "element_id" in action_json:
            # Prefer element_id click (id-based action via AXTree node)
            try:
                eid = int(action_json["element_id"])
                # B-01/02/33 Cluster 1 fix: locator-route bypasses framework's
                # mouse.click(union_bound_center) which hits child span/icon
                # instead of actionable parent (94.4% off-target on failed
                # clicks per Tier 10 sweep). Walks up DOM to find <a>/<button>/
                # [role=link]/[role=button]/<input type=submit/checkbox/radio>
                # and dispatches via Playwright locator.click() (real mouse
                # event + actionability check). Falls back to framework path
                # if walk-up fails (preserves existing behavior on edge cases).
                from p79.envs.locator_dispatch import dispatch_id_based_click as _lr_click
                # B-157 (/stress A1.3 v8 codex P1-B1, 2026-05-16): snapshot
                # context.pages count BEFORE the click so we can mimic VWA
                # framework's "switch to last opened tab" logic after a
                # successful locator-route dispatch. Pre-fix path used
                # ``create_none_action()`` to skip framework dispatch entirely,
                # which also bypassed VWA's `num_tabs_now > num_tabs_before`
                # tab-switch check (browser_env/actions.py:1417-1421) →
                # observation stayed bound to the old page when an
                # element-id click opened ``target=_blank`` / window.open.
                _num_tabs_before = 0
                try:
                    _num_tabs_before = len(self._env.context.pages)
                except Exception:
                    pass
                _lr_result = _lr_click(
                    self._env.page,
                    self._last_obs_nodes_info,
                    eid,
                    sleep_after_ms=int(self.sleep_after_execution * 1000),
                )
                # B-156: capture for step_record telemetry
                _locator_route_meta = dict(_lr_result)
                _locator_route_meta["action_kind"] = "click"
                if _lr_result.get("success"):
                    # B-157: switch to newly opened tab if click triggered one
                    # (window.open / target=_blank). Mirrors VWA framework
                    # `execute_action` tab-switch block at
                    # browser_env/actions.py:1417-1421.
                    try:
                        _pages_now = self._env.context.pages
                        if len(_pages_now) > _num_tabs_before:
                            _new_page = _pages_now[-1]
                            _new_page.bring_to_front()
                            self._env.page = _new_page
                            _locator_route_meta["new_tab_switched"] = True
                    except Exception as _e:
                        logger.warning("locator-route new-tab switch failed: %s", _e)
                    # JS dispatch already ran + slept. Skip framework dispatch
                    # via NONE action — env.step(NONE) just refreshes observation
                    # (now from the new page if a tab was opened).
                    action = create_none_action()
                    # B-553 (/stress A1.5 P1-3-AB* Claude+codex OOB, 2026-05-17):
                    # extend `action_executed` from scroll-only (B-512) to click
                    # dispatch path. Reviewer reading JSONL can now distinguish
                    # element_id-locator-route success (Cluster 1 path) from
                    # element_id-framework fallback (legacy VWA `create_id_based
                    # _action` path) without grepping `locator_route_meta.action
                    # _kind`. Paper §4.X.6 cross-baseline parity claim now
                    # auditable for all 3 wrapper-normalized action types.
                    _action_executed = {
                        "action_type": "click",
                        "dispatch_path": "element_id_locator_route",
                        "fallback": False,
                    }
                else:
                    logger.debug(
                        "locator-route click fallback: eid=%s reason=%s",
                        eid, _lr_result.get("error", "")[:80],
                    )
                    # Sequential SoM contract (2026-05-25): translate seq->native
                    # for the VWA framework dispatch; FAIL-CLOSED (no-op) if the
                    # seq is unresolved (seq mode, missing / hallucinated id) —
                    # never pass a seq id through as a native nodeId.
                    _nid = self._resolve_native_id(eid)
                    if _nid is None:
                        action = create_none_action()
                        _action_executed = {
                            "action_type": "click",
                            "dispatch_path": "seq_unresolved_noop",
                            "fallback": True,
                        }
                    else:
                        action = create_id_based_action(f"click [{_nid}]")
                        # B-553: fallback path — element_id click went through VWA
                        # framework's `create_id_based_action` after locator-route
                        # walk-up failed. Reviewer can grep
                        # `action_executed.fallback==True` to count cross-baseline
                        # fallback rate (B0 235B rarely; B1/B2 4B more often).
                        _action_executed = {
                            "action_type": "click",
                            "dispatch_path": "element_id_framework",
                            "fallback": True,
                        }
            except (TypeError, ValueError):
                action = None
        elif action_type == "click" and "coordinate" in action_json:
            coord = action_json.get("coordinate")
            if not (
                isinstance(coord, (list, tuple))
                and len(coord) == 2
                and coord[0] is not None
                and coord[1] is not None
            ):
                coord = None
            if coord is not None:
                # B-1860: Qwen 0-1000 contract via single-source normalizer.
                # Qwen3-VL natively emits a 0-1000 coordinate system (probe-
                # confirmed B0 + B1 2026-05-24); it also sometimes returns
                # normalized [0,1] (mixed-format probe). The normalizer judges
                # each dimension by value (`<= 1.1` kept / `> 1.1` divided by
                # 1000.0 — NOT viewport, which was the misclick root cause: a
                # 0-1000 value e.g. 728 / 1280 → 0.57 instead of 0.728) and
                # returns telemetry tags (regime / recovered / true_oob) so the
                # witness can quantify recovery rate. Format normalization only
                # — NO target snapping / element nearest-correction.
                left, top, _coord_tags = normalize_coordinate_pair(coord)
                _log_coord_normalization(_coord_tags, "click", coord)
                if _coord_tags["malformed"]:
                    # B-1860: defensive — the runner's validate_action already
                    # converts a malformed coord to {"action_type":"wait"}
                    # before env.step, so this branch is unreachable in the
                    # normal flow; guard anyway so a direct/edge caller can't
                    # crash on (None, None).
                    action = None
                elif _coord_tags["true_oob"]:
                    # V-F1 (B-1860 codex verify P1, 2026-05-24): a true_oob
                    # coord (a dimension > 1000 → after /1000 still > 1.0) is a
                    # GROUNDING miss, not a recoverable format. Pre-fix the
                    # eps-clamp in the else-branch mapped it onto the viewport
                    # edge and STILL ran create_mouse_click_action → a real
                    # corner-click that mutates page state (cart / nav) and
                    # contaminates the paper-grade episode. Fail-closed: no-op
                    # + telemetry so diag counts true_oob separately from parse
                    # errors and from executed clicks. NOT clamped.
                    action = create_none_action()
                    _action_executed = {
                        "action_type": "click",
                        "dispatch_path": "coord_true_oob_noop",
                        "fallback": True,
                        "coordinate_normalization": _coord_tags,
                    }
                else:
                    # Avoid 0.0 which triggers VWA create_mouse_click_action validation
                    eps = 1e-6
                    if left <= 0.0:
                        left = eps
                    elif left >= 1.0:
                        left = 1.0 - eps
                    if top <= 0.0:
                        top = eps
                    elif top >= 1.0:
                        top = 1.0 - eps
                    action = create_mouse_click_action(left=left, top=top)
                    # B-553 (/stress A1.5 P1-3-AB* Claude+codex OOB, 2026-05-17):
                    # vision-mode coord-click. No locator dispatch — direct
                    # framework pixel-click. Recorded so reviewer can confirm
                    # vision-mode B0/B1/B2 all take same code path.
                    _action_executed = {
                        "action_type": "click",
                        "dispatch_path": "coord_mouse_click",
                        "fallback": False,
                        # B-1860: coord normalization telemetry (per-dimension
                        # regime + whether a 0-1000→[0,1] rescale was applied).
                        "coordinate_normalization": _coord_tags,
                    }
            else:
                action = None
        elif action_type == "scroll" and ("delta" in action_json or "scroll_direction" in action_json):
            if "scroll_direction" in action_json:
                # Semantic scroll direction (from tool-calling schema).
                direction = "down" if action_json["scroll_direction"] == "down" else "up"
                action = create_scroll_action(direction=direction)
                # B-512: B0 already emits canonical enum; record post-normalize
                # form identically so paper §4 disclosure can show B0/B1/B2
                # parity at execution layer.
                _action_executed = {"action_type": "scroll", "direction": direction}
            else:
                delta = action_json["delta"]
                if isinstance(delta, (list, tuple)) and len(delta) >= 2:
                    dy = delta[1]
                elif isinstance(delta, (list, tuple)) and len(delta) == 1:
                    dy = delta[0]  # single-element delta: treat as dy
                else:
                    dy = delta if isinstance(delta, (int, float)) else 300
                # dy == 0 is not a scroll — treat as no-op rather than coercing
                # it into a "down" scroll (which would inflate scroll_down stats).
                if dy == 0:
                    action = create_none_action()
                    # B-512: explicit "noop" so reviewer can see dy=0
                    # collapsed (rare but legit edge case).
                    _action_executed = {"action_type": "scroll", "direction": "noop"}
                else:
                    direction = "down" if dy > 0 else "up"
                    action = create_scroll_action(direction=direction)
                    # B-512: B1/B2 raw `delta:[dx,dy]` collapsed to enum form
                    # (the gemini-flagged paper §4.X.6 asymmetry). Recording
                    # the post-normalize form here makes wrapper-level
                    # alignment auditable from step JSONL.
                    _action_executed = {"action_type": "scroll", "direction": direction}
        elif action_type == "type" and "text" in action_json and "element_id" not in action_json:
            # Type without element_id (vision mode): click coordinate first to focus, then keyboard type.
            # B-442 (/stress A1.25 P0-3-AC* OOB, 2026-05-17): vision-mode TYPE
            # focus-click was previously DIRECT `page.mouse.click(px, py)` —
            # exact B-01 bbox-pattern that locator-route was meant to retire
            # (DOM/SoM mode already got walk-up via `dispatch_id_based_type`;
            # vision was the cross-mode hole). Now routes through
            # `dispatch_coord_based_type` which walks up via _JS_RESOLVE_INPUT
            # to find the actionable INPUT/TEXTAREA/contenteditable ancestor,
            # then fills via `locator.fill()` (no global Meta+A, no 全选变蓝).
            # Fallback to original direct-click + keyboard.type path if
            # walk-up fails (preserves backward-compat on edge cases).
            coord = action_json.get("coordinate")
            # B-1860: Qwen 0-1000 contract via single-source normalizer
            # (type/vision focus-click coord). Same per-dimension by-value
            # judge as the click path (`<= 1.1` kept / `> 1.1` /1000.0, NOT
            # viewport — viewport division was the misclick root cause).
            # Telemetry tags stamped into both _action_executed branches.
            # Malformed (None,None) is gated by the `not malformed` guard — the
            # runner's validate_action already filters those upstream; this is
            # defensive for direct/edge callers.
            _norm_left, _norm_top, _coord_tags = normalize_coordinate_pair(coord)
            _log_coord_normalization(_coord_tags, "type_focus", coord)
            if _coord_tags["malformed"]:
                # B-1860: malformed coord (None,None) — validate_action filters
                # these upstream; defensive fallback to a coord-less keyboard
                # type into the current focus (preserves the behavior of the
                # pre-V-F1 `else` branch that this if-branch replaced).
                action = create_keyboard_type_action(action_json["text"])
            elif _coord_tags["true_oob"]:
                # V-F1 (B-1860 codex verify P1, 2026-05-24): true_oob grounding
                # miss → fail-closed no-op. Pre-fix the else-branch clamped to
                # the viewport edge and focus-clicked there (a real click that
                # can blur the page / type into the wrong field). Telemetry for
                # diag. NOT clamped.
                action = create_none_action()
                _action_executed = {
                    "action_type": "type",
                    "dispatch_path": "coord_true_oob_noop",
                    "fallback": True,
                    "coordinate_normalization": _coord_tags,
                }
            else:
                left, top = _norm_left, _norm_top
                eps = 1e-6
                left = max(eps, min(1.0 - eps, left))
                top = max(eps, min(1.0 - eps, top))
                _cx_px = left * self.viewport_width
                _cy_px = top * self.viewport_height
                from p79.envs.locator_dispatch import dispatch_coord_based_type as _lr_coord_type
                _lr_result = _lr_coord_type(
                    self._env.page,
                    _cx_px,
                    _cy_px,
                    str(action_json["text"]),
                    sleep_after_ms=int(self.sleep_after_execution * 1000),
                )
                # B-156/B-440 telemetry: stamp into info via _locator_route_meta
                _locator_route_meta = dict(_lr_result)
                _locator_route_meta["action_kind"] = "type_coord"
                if _lr_result.get("success"):
                    # locator dispatch already filled + optionally pressed Enter
                    # + slept. Convert to no-op for env.step (refreshes obs).
                    action = create_none_action()
                    # B-553 (/stress A1.5 P1-3-AB* Claude+codex OOB, 2026-05-17):
                    # vision-mode coord-type via locator route (Cluster 1 fix —
                    # bypasses 全选变蓝 §52/§64 risk).
                    _action_executed = {
                        "action_type": "type",
                        "dispatch_path": "coord_locator_route",
                        "fallback": False,
                        # B-1860: coord normalization telemetry.
                        "coordinate_normalization": _coord_tags,
                    }
                else:
                    # Walk-up failed → fall back to legacy direct-click + keyboard.type
                    # path. Preserves prior behavior on edge cases (e.g., coord
                    # falls outside any input/textarea/contenteditable element).
                    logger.debug(
                        "locator-route coord-type fallback: cx=%s cy=%s reason=%s",
                        _cx_px, _cy_px, _lr_result.get("error", "")[:80],
                    )
                    # B-553: fallback to direct-click + keyboard.type (with
                    # is_editable guard against 全选变蓝).
                    _action_executed = {
                        "action_type": "type",
                        "dispatch_path": "coord_keyboard_fallback",
                        "fallback": True,
                        # B-1860: coord normalization telemetry.
                        "coordinate_normalization": _coord_tags,
                    }
                    self._env.page.mouse.click(_cx_px, _cy_px)
                    _wait_ms = int(self.sleep_after_execution * 1000)
                    self._env.page.wait_for_timeout(_wait_ms)
                    _runtime_sleep_ms += _wait_ms  # B-510
                    # is_editable guard kept for the fallback path only (B-01
                    # Control+a-on-page-body guard against 全选变蓝). When the
                    # locator-route walk-up succeeded above, this code path is
                    # skipped entirely (action=NONE).
                    try:
                        is_editable = self._env.page.evaluate(
                            "() => { const el = document.activeElement; "
                            "return el != null && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable); }"
                        )
                    except Exception:
                        is_editable = False
                    if is_editable:
                        self._env.page.keyboard.press("Control+a")
                        self._env.page.keyboard.press("Backspace")
                    action = create_keyboard_type_action(action_json["text"])
        elif action_type == "type" and "text" in action_json and "element_id" in action_json:
            # B-506 (/stress A1.25 GRL Chunk 3 P0-1-B*, 2026-05-17): defense-
            # in-depth removal of the legacy `<=0` → keyboard fallback path.
            # Pre-fix model emitting sentinel `0` / `-1` had `parse_valid=true`
            # at the validator (B-506 fix at action_utils.py:308/342/367 now
            # also rejects this), and runtime here typed into whoever has focus
            # (silent typing-into-wrong-element). Post-fix: invalid element_id
            # → explicit no-op + locator_route_meta error tag, mirror of the
            # validator-side fix so any direct caller of vwa_wrapper.step()
            # that bypasses the validator gets the same fail-loud contract.
            try:
                element_id = int(action_json.get("element_id"))
            except (TypeError, ValueError):
                element_id = None
            if element_id is None or element_id <= 0:
                logger.warning(
                    "type action with invalid element_id=%s — no-op (B-506 fix)",
                    action_json.get("element_id"),
                )
                _locator_route_meta = {
                    "action_kind": "type",
                    "success": False,
                    "error": f"invalid_element_id_<=0:{action_json.get('element_id')}",
                }
                action = create_none_action()
                # B-553 (/stress A1.5 P1-3-AB*, 2026-05-17): invalid element_id
                # short-circuit. Recorded so taxonomy table can count B-506
                # noop-rescue rate per baseline.
                _action_executed = {
                    "action_type": "type",
                    "dispatch_path": "noop_invalid_element_id",
                    "fallback": False,
                }
            else:
                _type_needs_enter = bool(str(action_json.get("text", "")).endswith("\n"))
                # B-01 Cluster 1 fix: locator-route TYPE bypasses framework's
                # mouse.click(center) + Meta+A + Backspace + keyboard.type
                # pattern that causes 全选变蓝 (§52/§64) when click hits
                # non-input. Uses locator.fill() which auto-clears and dispatches
                # input event WITHOUT global Meta+A. press_enter handled by
                # text trailing-newline detection.
                from p79.envs.locator_dispatch import dispatch_id_based_type as _lr_type
                # B-418 (/stress A1.3 v9 Mode B P1-2 OOB sibling propagation
                # of B-157, 2026-05-17): snapshot tab count BEFORE dispatch so
                # press_enter that triggers form submit `target=_blank` /
                # `window.open` can be followed (mirrors click branch B-157
                # at lines 284-309). Pre-fix: search form Enter opened new
                # tab → observation stayed bound to old page → state_change
                # false-no-progress → cross-baseline taxonomy contamination.
                _num_tabs_before = 0
                try:
                    _num_tabs_before = len(self._env.context.pages)
                except Exception:
                    pass
                _lr_result = _lr_type(
                    self._env.page,
                    self._last_obs_nodes_info,
                    element_id,
                    str(action_json["text"]),
                    sleep_after_ms=int(self.sleep_after_execution * 1000),
                    press_enter=_type_needs_enter,
                )
                # B-156: capture for step_record telemetry
                _locator_route_meta = dict(_lr_result)
                _locator_route_meta["action_kind"] = "type"
                if _lr_result.get("success"):
                    # B-418: mirror click branch (B-157) new-tab switch logic.
                    try:
                        _pages_now = self._env.context.pages
                        if len(_pages_now) > _num_tabs_before:
                            _new_page = _pages_now[-1]
                            _new_page.bring_to_front()
                            self._env.page = _new_page
                            _locator_route_meta["new_tab_switched"] = True
                    except Exception as _e:
                        logger.warning("locator-route type+Enter new-tab switch failed: %s", _e)
                    action = create_none_action()
                    # Locator dispatch already pressed Enter if text ended with
                    # \n (see locator_dispatch.py:dispatch_id_based_type); avoid
                    # double Enter from post-step keyboard.press at line ~565.
                    _type_needs_enter = False
                    # B-553 (/stress A1.5 P1-3-AB*, 2026-05-17): element_id
                    # type via locator route (Cluster 1 fix). Reviewer can
                    # grep `action_executed.dispatch_path` to verify all
                    # baselines route DOM/SoM-mode type identically.
                    _action_executed = {
                        "action_type": "type",
                        "dispatch_path": "element_id_locator_route",
                        "fallback": False,
                    }
                else:
                    logger.debug(
                        "locator-route type fallback: eid=%s reason=%s",
                        element_id, _lr_result.get("error", "")[:80],
                    )
                    # Framework will build id-based action below; nothing to do.
                    # B-553: element_id type fallback — framework path.
                    _action_executed = {
                        "action_type": "type",
                        "dispatch_path": "element_id_framework",
                        "fallback": True,
                    }
                    pass
        elif action_type == "back":
            action = create_go_back_action()
        elif action_type == "forward":
            action = create_go_forward_action()
        elif action_type in ("tab", "tab_focus", "page_focus"):
            page_number = action_json.get("page_number")
            if page_number is None:
                page_number = action_json.get("tab_index")
            if page_number is None:
                thought = action_json.get("thought", "")
                match = re.search(r"tab\s*(\d+)", thought, re.IGNORECASE)
                if match:
                    page_number = int(match.group(1))
            if page_number is None:
                action = create_none_action()
            else:
                action = create_page_focus_action(page_number=int(page_number))
        elif action_type in ("finish", "stop"):
            answer = action_json.get("answer", "")
            action = create_stop_action("" if answer is None else str(answer))
        elif action_type == "select_option":
            option_label = str(
                action_json.get("option_label") or action_json.get("option_value") or ""
            )
            option_index = action_json.get("option_index")
            # B-420 (/stress A1.3 v9 Mode B P1-5 OOB, 2026-05-17):
            # select_option dispatch result telemetry. Pre-fix bare
            # `except: logger.warning + create_none_action()` swallowed
            # JS exceptions + missing-obs cases + no-match cases under a
            # single no-op tag. Empirical 195/738 archive rows
            # (action_success=False, page_change=False) were
            # taxonomy-blind. New `select_option_meta` mirrors
            # `locator_route_meta` and gets stamped into info dict so the
            # runner can persist it into StepRecordV2 (paper §3.5
            # select_option sub-taxonomy).
            # B-481 (/stress A1.25 GRL Chunk 2 P0-2-AB* + P1-1-BC* + parallel
            # A1.4 B-453 carry, 2026-05-17): expanded telemetry slots so
            # `success` carries the post-fix semantic ("an option matched and
            # was selected/clicked") rather than the legacy "JS evaluate did
            # not throw" — and downstream aggregators can compute true
            # ON_OPTION rates plus fuzzy-tier share per (site, model, mode).
            # `match_stage` ∈ {None, "exact", "ci", "fuzzy", "index", "none"};
            # `target_type` ∈ {None, "select", "css"}.
            _select_option_meta: Dict[str, Any] = {
                "action_kind": "select_option",
                "dispatch_path": None,  # "element_id" | "coordinate" | "missing_obs"
                "success": None,
                "matched": None,            # B-481: true iff a candidate was matched + dispatched
                "match_stage": None,        # B-481: which fuzzy tier (or 'index' / 'none')
                "target_type": None,        # B-481: 'select' (native) | 'css' (custom) | None
                "selected_text_before": None,  # B-481: native-select state-change evidence
                "selected_text_after": None,
                "clicked_text": None,       # B-481: CSS-menu state-change evidence
                "error": None,
            }

            if "element_id" in action_json:
                _select_option_meta["dispatch_path"] = "element_id"
                # DOM/SoM 路径：用 obs_nodes_info 像素坐标 + elementFromPoint 定位 SELECT
                try:
                    eid = int(action_json["element_id"])
                    # 从上一次 obs 的 obs_nodes_info 拿像素坐标（union_bound: [x, y, w, h]）
                    node_info = (self._last_obs_nodes_info or {}).get(str(eid))
                    if not node_info or "union_bound" not in node_info:
                        # obs_nodes_info 缺失时不瞎猜——视口中心几乎肯定不是
                        # 目标 SELECT，会随机选错下拉。降级为 no-op，让上层
                        # cycle/no-progress 检测处理。Skip the page.evaluate()
                        # below; action=create_none_action() is set at branch end.
                        logger.warning(
                            "select_option: obs_nodes_info missing for element_id=%s, "
                            "fallback to no-op (previously: viewport-center fallback that silently mis-selected)",
                            eid,
                        )
                        ub = None
                        _select_option_meta["dispatch_path"] = "missing_obs"
                        _select_option_meta["success"] = False
                        _select_option_meta["error"] = "obs_nodes_info_missing_union_bound"
                    else:
                        ub = node_info["union_bound"]
                    if ub is None:
                        pass  # no-op fallback handled above; skip the page.evaluate()
                    else:
                        x_px = ub[0] + ub[2] / 2
                        y_px = ub[1] + ub[3] / 2
                        # B-481: JS now returns structured
                        # {matched, match_stage, target_type, selected_text_before,
                        #  selected_text_after, clicked_text, error}
                        # so Python populates `_select_option_meta` from the result
                        # instead of blindly stamping `success=True` post-evaluate.
                        _js_result = self._env.page.evaluate(
                            _FUZZY_MATCH_JS + """([x, y, label, idx]) => {
                            const el = document.elementFromPoint(x, y);
                            // 1. Native <select> path
                            if (el && el.tagName === 'SELECT') {
                                const beforeOpt = el.options[el.selectedIndex];
                                const beforeText = beforeOpt ? beforeOpt.text.trim() : null;
                                if (idx !== null) {
                                    // B-511 (/stress A1.25 GRL Chunk 3 P1-4-B*,
                                    // 2026-05-17): bounds check before
                                    // selectedIndex assignment. Pre-fix idx=999
                                    // silently set selectedIndex=999 (clamped
                                    // by browser but `afterOpt` undefined) and
                                    // returned matched=true — false-positive
                                    // ON_OPTION pollution.
                                    if (idx < 0 || idx >= el.options.length) {
                                        return {matched: false, match_stage: 'none', target_type: 'select',
                                                selected_text_before: beforeText, selected_text_after: beforeText,
                                                clicked_text: null,
                                                error: 'index_out_of_bounds:' + idx + '/' + el.options.length};
                                    }
                                    el.selectedIndex = idx;
                                    el.dispatchEvent(new Event('change', {bubbles: true}));
                                    const afterOpt = el.options[el.selectedIndex];
                                    if (!afterOpt) {
                                        return {matched: false, match_stage: 'none', target_type: 'select',
                                                selected_text_before: beforeText, selected_text_after: beforeText,
                                                clicked_text: null,
                                                error: 'index_dispatch_no_after_option'};
                                    }
                                    return {matched: true, match_stage: 'index', target_type: 'select',
                                            selected_text_before: beforeText,
                                            selected_text_after: afterOpt.text.trim(),
                                            clicked_text: null, error: null};
                                }
                                const cands = Array.from(el.options).map(o => ({text: o.text.trim(), value: o.value, el: o}));
                                const result = _fuzzyFind(cands, label);
                                if (result && result.match) {
                                    el.value = result.match.value;
                                    el.dispatchEvent(new Event('change', {bubbles: true}));
                                    const afterOpt = el.options[el.selectedIndex];
                                    return {matched: true, match_stage: result.stage, target_type: 'select',
                                            selected_text_before: beforeText,
                                            selected_text_after: afterOpt ? afterOpt.text.trim() : null,
                                            clicked_text: null, error: null};
                                }
                                return {matched: false, match_stage: 'none', target_type: 'select',
                                        selected_text_before: beforeText, selected_text_after: beforeText,
                                        clicked_text: null, error: 'no_match_in_select'};
                            }
                            // 2. CSS custom dropdown fallback: scan hidden <ul>s near (x, y)
                            if (!label) return {matched: false, match_stage: 'none', target_type: null,
                                                selected_text_before: null, selected_text_after: null,
                                                clicked_text: null, error: 'no_label_no_select'};
                            for (const ul of document.querySelectorAll('ul')) {
                                const rect = ul.getBoundingClientRect();
                                if (rect.width > 0 || rect.height > 0) continue;
                                let trigger = ul.parentElement;
                                while (trigger && trigger !== document.body) {
                                    const tr = trigger.getBoundingClientRect();
                                    if (tr.width > 0 && tr.height > 0) break;
                                    trigger = trigger.parentElement;
                                }
                                if (!trigger) continue;
                                const tr = trigger.getBoundingClientRect();
                                const cx = tr.left + tr.width / 2;
                                const cy = tr.top + tr.height / 2;
                                if (Math.sqrt((cx - x) * (cx - x) + (cy - y) * (cy - y)) > 150) continue;
                                const items = Array.from(
                                    ul.querySelectorAll(':scope > li > a, :scope > li > button'));
                                const cands2 = items.map(i => ({text: i.textContent.trim(), el: i}));
                                const result = _fuzzyFind(cands2, label);
                                if (result && result.match) {
                                    const oldDisplay = ul.style.display;
                                    const oldVisibility = ul.style.visibility;
                                    ul.style.display = 'block';
                                    ul.style.visibility = 'visible';
                                    result.match.el.click();
                                    ul.style.display = oldDisplay;
                                    ul.style.visibility = oldVisibility;
                                    return {matched: true, match_stage: result.stage, target_type: 'css',
                                            selected_text_before: null, selected_text_after: null,
                                            clicked_text: result.match.text, error: null};
                                }
                            }
                            return {matched: false, match_stage: 'none', target_type: 'css',
                                    selected_text_before: null, selected_text_after: null,
                                    clicked_text: null, error: 'no_match_in_css_menus'};
                        }""",
                            [x_px, y_px, option_label, option_index],
                        )
                        _wait_ms = int(self.sleep_after_execution * 1000)
                        self._env.page.wait_for_timeout(_wait_ms)
                        _runtime_sleep_ms += _wait_ms  # B-510
                        # B-481: populate from structured JS result. `success`
                        # now means "matched and dispatched", not "JS didn't
                        # throw". Reviewer / aggregator can audit fuzzy-tier
                        # share + match_stage breakdown via paper §3 evidence
                        # layer.
                        if isinstance(_js_result, dict):
                            _select_option_meta["matched"] = bool(_js_result.get("matched"))
                            _select_option_meta["match_stage"] = _js_result.get("match_stage")
                            _select_option_meta["target_type"] = _js_result.get("target_type")
                            _select_option_meta["selected_text_before"] = _js_result.get("selected_text_before")
                            _select_option_meta["selected_text_after"] = _js_result.get("selected_text_after")
                            _select_option_meta["clicked_text"] = _js_result.get("clicked_text")
                            _select_option_meta["success"] = bool(_js_result.get("matched"))
                            if not _js_result.get("matched"):
                                _select_option_meta["error"] = _js_result.get("error") or "no_match"
                        else:
                            _select_option_meta["success"] = False
                            _select_option_meta["matched"] = False
                            _select_option_meta["error"] = "js_returned_non_dict"
                except Exception as _e:
                    logger.warning("select_option (element_id=%s) failed: %s", action_json.get("element_id"), _e)
                    _select_option_meta["success"] = False
                    _select_option_meta["matched"] = False
                    _select_option_meta["error"] = f"{type(_e).__name__}: {str(_e)[:160]}"
            elif "coordinate" in action_json:
                _select_option_meta["dispatch_path"] = "coordinate"
                # Vision 路径：通过坐标找元素，用 JS 设置选中值
                try:
                    coord = action_json["coordinate"]
                    # B-1860: Qwen 0-1000 contract via single-source normalizer
                    # (select_option coord, vision mode). The normalizer maps
                    # each dimension to [0,1] (`<= 1.1` kept / `> 1.1` /1000.0);
                    # we then multiply by viewport to get px. Pre-fix this site
                    # treated `> 1.0` as already-pixel (raw passthrough) which
                    # mis-placed a 0-1000 coord (e.g. 728 → 728px not 0.728*W).
                    x_norm, y_norm, _coord_tags = normalize_coordinate_pair(coord)
                    _log_coord_normalization(_coord_tags, "select_option", coord)
                    # B-1860: coord normalization telemetry (select_option path).
                    _select_option_meta["coordinate_normalization"] = _coord_tags
                    if _coord_tags["malformed"]:
                        # Defensive — validate_action filters malformed coords
                        # upstream; surface a clear error if a direct caller
                        # reaches here with (None, None).
                        raise ValueError("malformed select_option coordinate")
                    if _coord_tags["true_oob"]:
                        # V-F1 (B-1860 codex verify P1, 2026-05-24): true_oob
                        # grounding miss → fail-closed no-op. Mark dispatch +
                        # raise to skip the elementFromPoint JS below (at the
                        # clamped viewport edge it can resolve + mutate a wrong
                        # SELECT). The except clause stamps the error but does
                        # NOT touch dispatch_path, so coord_true_oob_noop
                        # survives. action falls through to create_none_action.
                        _select_option_meta["dispatch_path"] = "coord_true_oob_noop"
                        raise ValueError("coord_true_oob")
                    # B-1860 item 6: eps-clamp in normalized space (the click /
                    # type / hover coord paths all clamp to [eps, 1-eps]; this
                    # site previously did NOT, so a [1000,1000] coord → exactly
                    # (W, H) px = the viewport bottom-right corner, where
                    # elementFromPoint can return null / a wrong element).
                    # Clamping before the viewport multiply keeps px strictly
                    # inside the viewport. Format normalization only.
                    eps = 1e-6
                    x_norm = max(eps, min(1.0 - eps, x_norm))
                    y_norm = max(eps, min(1.0 - eps, y_norm))
                    x_px = x_norm * self.viewport_width
                    y_px = y_norm * self.viewport_height
                    # B-481: structured JS return mirrors element_id path above.
                    # B-511 (/stress A1.25 GRL Chunk 3 P1-4-B*, 2026-05-17):
                    # coord path now accepts `idx` arg symmetric with
                    # element_id path. Pre-fix coord path dropped
                    # `option_index` entirely — vision-mode index-dispatch
                    # was impossible. Now: same `[x,y,label,idx]` payload,
                    # same bounds + after-opt checks before matched=true.
                    _js_result = self._env.page.evaluate(
                        _FUZZY_MATCH_JS + """([x, y, label, idx]) => {
                            // 1. Native <select> path
                            const el = document.elementFromPoint(x, y);
                            if (el && el.tagName === 'SELECT') {
                                const beforeOpt = el.options[el.selectedIndex];
                                const beforeText = beforeOpt ? beforeOpt.text.trim() : null;
                                if (idx !== null) {
                                    // B-511: bounds + after-opt check (symmetric with element_id path)
                                    if (idx < 0 || idx >= el.options.length) {
                                        return {matched: false, match_stage: 'none', target_type: 'select',
                                                selected_text_before: beforeText, selected_text_after: beforeText,
                                                clicked_text: null,
                                                error: 'index_out_of_bounds:' + idx + '/' + el.options.length};
                                    }
                                    el.selectedIndex = idx;
                                    el.dispatchEvent(new Event('change', {bubbles: true}));
                                    const afterOpt = el.options[el.selectedIndex];
                                    if (!afterOpt) {
                                        return {matched: false, match_stage: 'none', target_type: 'select',
                                                selected_text_before: beforeText, selected_text_after: beforeText,
                                                clicked_text: null,
                                                error: 'index_dispatch_no_after_option'};
                                    }
                                    return {matched: true, match_stage: 'index', target_type: 'select',
                                            selected_text_before: beforeText,
                                            selected_text_after: afterOpt.text.trim(),
                                            clicked_text: null, error: null};
                                }
                                const cands = Array.from(el.options).map(o => ({text: o.text.trim(), value: o.value, el: o}));
                                const result = _fuzzyFind(cands, label);
                                if (result && result.match) {
                                    el.value = result.match.value;
                                    el.dispatchEvent(new Event('change', {bubbles: true}));
                                    const afterOpt = el.options[el.selectedIndex];
                                    return {matched: true, match_stage: result.stage, target_type: 'select',
                                            selected_text_before: beforeText,
                                            selected_text_after: afterOpt ? afterOpt.text.trim() : null,
                                            clicked_text: null, error: null};
                                }
                                return {matched: false, match_stage: 'none', target_type: 'select',
                                        selected_text_before: beforeText, selected_text_after: beforeText,
                                        clicked_text: null, error: 'no_match_in_select'};
                            }
                            // 2. CSS custom dropdown fallback: scan hidden <ul>s near (x, y)
                            for (const ul of document.querySelectorAll('ul')) {
                                const rect = ul.getBoundingClientRect();
                                if (rect.width > 0 || rect.height > 0) continue;
                                let trigger = ul.parentElement;
                                while (trigger && trigger !== document.body) {
                                    const tr = trigger.getBoundingClientRect();
                                    if (tr.width > 0 && tr.height > 0) break;
                                    trigger = trigger.parentElement;
                                }
                                if (!trigger) continue;
                                const tr = trigger.getBoundingClientRect();
                                const cx = tr.left + tr.width / 2;
                                const cy = tr.top + tr.height / 2;
                                if (Math.sqrt((cx - x) * (cx - x) + (cy - y) * (cy - y)) > 150) continue;
                                const items = Array.from(
                                    ul.querySelectorAll(':scope > li > a, :scope > li > button'));
                                const cands2 = items.map(i => ({text: i.textContent.trim(), el: i}));
                                const result = _fuzzyFind(cands2, label);
                                if (result && result.match) {
                                    const oldDisplay = ul.style.display;
                                    const oldVisibility = ul.style.visibility;
                                    ul.style.display = 'block';
                                    ul.style.visibility = 'visible';
                                    result.match.el.click();
                                    ul.style.display = oldDisplay;
                                    ul.style.visibility = oldVisibility;
                                    return {matched: true, match_stage: result.stage, target_type: 'css',
                                            selected_text_before: null, selected_text_after: null,
                                            clicked_text: result.match.text, error: null};
                                }
                            }
                            return {matched: false, match_stage: 'none', target_type: 'css',
                                    selected_text_before: null, selected_text_after: null,
                                    clicked_text: null, error: 'no_match_in_css_menus'};
                        }""",
                        # B-511: pass option_index symmetric with element_id path
                        [x_px, y_px, option_label, option_index],
                    )
                    _wait_ms = int(self.sleep_after_execution * 1000)
                    self._env.page.wait_for_timeout(_wait_ms)
                    _runtime_sleep_ms += _wait_ms  # B-510
                    # B-481: populate from structured JS result (same as element_id path).
                    if isinstance(_js_result, dict):
                        _select_option_meta["matched"] = bool(_js_result.get("matched"))
                        _select_option_meta["match_stage"] = _js_result.get("match_stage")
                        _select_option_meta["target_type"] = _js_result.get("target_type")
                        _select_option_meta["selected_text_before"] = _js_result.get("selected_text_before")
                        _select_option_meta["selected_text_after"] = _js_result.get("selected_text_after")
                        _select_option_meta["clicked_text"] = _js_result.get("clicked_text")
                        _select_option_meta["success"] = bool(_js_result.get("matched"))
                        if not _js_result.get("matched"):
                            _select_option_meta["error"] = _js_result.get("error") or "no_match"
                    else:
                        _select_option_meta["success"] = False
                        _select_option_meta["matched"] = False
                        _select_option_meta["error"] = "js_returned_non_dict"
                except Exception as _e:
                    logger.warning("select_option (coordinate) failed: %s", _e)
                    _select_option_meta["success"] = False
                    _select_option_meta["matched"] = False
                    _select_option_meta["error"] = f"{type(_e).__name__}: {str(_e)[:160]}"
            action = create_none_action()
        elif action_type == "wait":
            action = create_none_action()
        elif action_type == "hover":
            # Protocol Reset #5 (P1-1-BC cross-AI fix, 2026-05-20): explicit hover
            # branch. Pre-fix hover fell through to the escape-hatch, which only
            # serializes "hover [element_id]"; a coordinate hover (vision mode —
            # advertised in the vision prompt + validator-valid) raised
            # ValueError there → was caught → silent no-op (parse_valid=True but
            # no browser hover). Now element_id hovers via the upstream id-based
            # parser and coordinate hovers via create_mouse_hover_action,
            # mirroring the coord-click normalization. Both stamp action_executed
            # for the evidence layer (P2-2).
            _hover_eid = action_json.get("element_id")
            _hover_coord = action_json.get("coordinate")
            if _hover_eid is not None:
                # Sequential SoM contract (2026-05-25): seq->native + fail-closed.
                _nid = self._resolve_native_id(_hover_eid)
                if _nid is None:
                    action = create_none_action()
                    _action_executed = {"action_type": "hover", "dispatch_path": "seq_unresolved_noop", "fallback": True}
                else:
                    action = create_id_based_action(f"hover [{_nid}]")
                    _action_executed = {"action_type": "hover", "dispatch_path": "element_id", "fallback": False}
            elif (
                isinstance(_hover_coord, (list, tuple))
                and len(_hover_coord) == 2
                and _hover_coord[0] is not None
                and _hover_coord[1] is not None
            ):
                # B-1860: Qwen 0-1000 contract via single-source normalizer
                # (hover coord, vision mode). Per-dimension by-value judge
                # (`<= 1.1` kept / `> 1.1` /1000.0, NOT viewport). Mirrors the
                # coord-click normalization. Format normalization only.
                left, top, _coord_tags = normalize_coordinate_pair(_hover_coord)
                _log_coord_normalization(_coord_tags, "hover", _hover_coord)
                if _coord_tags["malformed"]:
                    # Defensive — validate_action filters malformed coords
                    # upstream; fall back to noop for a direct/edge caller.
                    action = create_none_action()
                    _action_executed = {"action_type": "hover", "dispatch_path": "noop_no_target", "fallback": True}
                elif _coord_tags["true_oob"]:
                    # V-F1 (B-1860 codex verify P1, 2026-05-24): true_oob
                    # grounding miss → fail-closed no-op (hover at the clamped
                    # viewport edge could trigger an unintended tooltip / menu).
                    # Telemetry for diag. NOT clamped.
                    action = create_none_action()
                    _action_executed = {
                        "action_type": "hover",
                        "dispatch_path": "coord_true_oob_noop",
                        "fallback": True,
                        "coordinate_normalization": _coord_tags,
                    }
                else:
                    eps = 1e-6
                    if left <= 0.0:
                        left = eps
                    elif left >= 1.0:
                        left = 1.0 - eps
                    if top <= 0.0:
                        top = eps
                    elif top >= 1.0:
                        top = 1.0 - eps
                    action = create_mouse_hover_action(left=left, top=top)
                    _action_executed = {
                        "action_type": "hover",
                        "dispatch_path": "coord_mouse_hover",
                        "fallback": False,
                        # B-1860: coord normalization telemetry.
                        "coordinate_normalization": _coord_tags,
                    }
            else:
                action = create_none_action()
                _action_executed = {"action_type": "hover", "dispatch_path": "noop_no_target", "fallback": True}
        elif action_type == "goto":
            # Protocol Reset #5 (action-set restore, 2026-05-20; P1-2-B* cross-AI
            # fix): goto with a VWA-origin whitelist. Allowed origins = open-tab
            # netlocs ∪ configured VWA site netlocs (env). Off-site goto → no-op
            # so the agent cannot leave the controlled site set (eval-breaking /
            # contamination). cls/reddit cross-site uses pre-opened |AND| tabs +
            # tab_focus so legitimate origins are already open; demand is ~0.
            #
            # P1-2-B* (codex OOB, 2026-05-20): match on `netloc` (host:port), NOT
            # `hostname`. On the A100 self-host every VWA site is localhost:<port>,
            # so a hostname-only whitelist collapsed to "any localhost port".
            # netloc keeps the port distinction. Relative URLs (empty netloc +
            # empty scheme = a path on the current origin) are inherently on-site
            # and allowed; non-empty schemes with empty netloc (javascript:/data:)
            # are NOT relative and stay blocked.
            _goto_url = str(action_json.get("url") or "").strip()
            _goto_parsed = urlparse(_goto_url)
            _goto_netloc = (_goto_parsed.netloc or "").lower()
            _goto_is_relative = _goto_netloc == "" and _goto_parsed.scheme == ""
            _allowed_netlocs = self._goto_allowed_hosts()
            _goto_meta = {
                "action_kind": "goto",
                "url": _goto_url,
                "netloc": _goto_netloc,
                "relative": _goto_is_relative,
                "allowed": False,
            }
            if _goto_url and (_goto_is_relative or _goto_netloc in _allowed_netlocs):
                action = create_goto_url_action(_goto_url)
                _goto_meta["allowed"] = True
                _action_executed = {
                    "action_type": "goto",
                    "dispatch_path": "relative" if _goto_is_relative else "whitelisted",
                    "fallback": False,
                }
            else:
                action = create_none_action()
                _goto_meta["error"] = "offsite_blocked"
                _action_executed = {"action_type": "goto", "dispatch_path": "offsite_blocked", "fallback": True}
                logger.warning(
                    "goto blocked (off-whitelist): netloc=%r allowed=%s",
                    _goto_netloc, sorted(_allowed_netlocs)[:8],
                )

        if action is None and action_type == "click" and "element_id" not in action_json:
            action = create_none_action()

        if action is None:
            # Fallback to action_str or ID based
            if "action_str" in action_json:
                action = create_playwright_action(str(action_json["action_str"]))
            else:
                # Sequential SoM contract (2026-05-25): this id-based escape hatch
                # serializes element_id actions (the type fallback routes here, plus
                # press/new_tab/close_tab which carry no element_id). If an
                # element_id is present, translate seq->native and FAIL-CLOSED
                # (no-op) when unresolved — before serializing — so a seq id is
                # never emitted as a native nodeId. element_id-free actions pass
                # through unchanged.
                _esc_eid = action_json.get("element_id")
                _esc_nid = self._resolve_native_id(_esc_eid) if _esc_eid is not None else None
                if _esc_eid is not None and _esc_nid is None:
                    action = create_none_action()
                    # FORCE-overwrite any pre-stamp (codex round-3 P2): the type
                    # element-id branch pre-stamps `element_id_framework` on its
                    # locator-route fallback before reaching this escape hatch; if
                    # the seq is then unresolved we no-op, so telemetry must report
                    # the no-op (what actually executed), not the superseded
                    # framework-dispatch intent. Unconditional, not `if None`.
                    _action_executed = {
                        "action_type": action_type,
                        "dispatch_path": "seq_unresolved_noop",
                        "fallback": True,
                    }
                else:
                    if _esc_eid is not None:
                        action_json = {**action_json, "element_id": _esc_nid}
                    try:
                        action_str = self._json_to_id_action_str(action_json)
                        action = create_id_based_action(action_str)
                        # P2-2 (cross-AI 2026-05-20): stamp escape-hatch dispatch into
                        # the wrapper-normalized telemetry so restored actions routed
                        # here (press / new_tab / close_tab) leave uniform evidence-
                        # layer proof of execution, parallel to the explicit branches.
                        if _action_executed is None:
                            _action_executed = {
                                "action_type": action_type,
                                "dispatch_path": "id_based_escape_hatch",
                                "fallback": False,
                            }
                    except Exception as _e:
                        logger.warning("create_id_based_action failed (%s), falling back to wait: %s", action_json, _e)
                        action = create_none_action()
                        if _action_executed is None:
                            _action_executed = {
                                "action_type": action_type,
                                "dispatch_path": "noop_serialize_fail",
                                "fallback": True,
                            }

        try:
            obs, reward, terminated, truncated, info = self._env.step(action)
        except Exception:
            # Reset underlying resources so next episode can re-initialize cleanly.
            self.close()
            raise

        # Post-type Enter: DOM/SOM type+element_id 路径下 VWA id-based action 会 strip \n，
        # 对于 \n 结尾的文本（搜索/表单提交），需补发 Enter 触发提交。
        if _type_needs_enter and self._env is not None:
            try:
                self._env.page.keyboard.press("Enter")
                _wait_ms = int(self.sleep_after_execution * 1000)
                self._env.page.wait_for_timeout(_wait_ms)
                _runtime_sleep_ms += _wait_ms  # B-510
                re_obs, _, _, _, re_info = self._env.step(create_none_action())
                obs, info = re_obs, re_info
            except Exception as _e:
                logger.warning("Post-type Enter press failed: %s", _e)

        # Fire-6 RCA Stage C1b: mode-gate the submodule screenshot-timeout
        # recovery (info["screenshot_timeout_recovered"]) on the final info.
        self._gate_screenshot_timeout(info)

        if action_type in ("finish", "stop"):
            terminated = True
        info["raw_action"] = action  # Expose the raw VWA action for trajectory recording
        # B-156: surface locator-route dispatch telemetry into info so the
        # runner can persist it into StepRecordV2 (paper §3 evidence layer).
        # None when the step did not invoke locator-route (scroll, wait,
        # coord-only click, etc.).
        info["locator_route_meta"] = _locator_route_meta
        # B-420 (/stress A1.3 v9 Mode B P1-5 OOB, 2026-05-17): select_option
        # dispatch telemetry. None when step did not invoke select_option;
        # otherwise {action_kind, dispatch_path, success, error}.
        info["select_option_meta"] = _select_option_meta
        # Protocol Reset #5 (action-set restore, 2026-05-20): goto dispatch
        # telemetry (whitelist allow/block). None unless step was a goto.
        info["goto_meta"] = _goto_meta
        # B-510 (/stress A1.25 GRL Chunk 3 P1-3-AB, 2026-05-17): per-step
        # runtime settle-tax (wrapper-level wait_for_timeout sum, excluding
        # locator-dispatch internal sleeps). Runner stamps into
        # step_record.latency_ms["runtime_sleep"] so paper §4 can report
        # both `total` and `total - runtime_sleep` columns to disentangle
        # mode-delta from runtime-wait composition.
        info["runtime_sleep_ms"] = _runtime_sleep_ms
        # B-509 (/stress A1.25 GRL Chunk 3 P1-2-BC*, 2026-05-17): drain
        # the dialog accumulator + stamp into info. None when no dialog
        # fired during this step (most common case); else a list of
        # {type, message, accepted} payloads. Runner stamps into
        # step_record.dialog_meta for paper §3.5.1 misclick-blast-radius
        # evidence layer.
        info["dialog_meta"] = (
            list(self._dialogs_this_step) if self._dialogs_this_step else None
        )
        self._dialogs_this_step.clear()
        # B-512 (scroll, /stress A1.5b Phase 2 P0-1-C gemini OOB) + B-553
        # (click/type extension, /stress A1.5 P1-3-AB*, 2026-05-17):
        # wrapper-normalized canonical action form including dispatch path.
        # Set in scroll/click/type branches above; None for back/forward/tab/
        # finish/stop (no wrapper-level normalization layer).
        info["action_executed"] = _action_executed
        p79_obs = self._to_p79_obs(obs, info)
        self._install_native_obs_nodes_info(p79_obs.obs_nodes_info)
        return p79_obs, float(reward), bool(terminated), bool(truncated), info

    def navigate_to(self, url: str) -> Tuple[P79Observation, float, bool, bool, Dict[str, Any]]:
        """Navigate to a URL using create_playwright_action, then return fresh observation."""
        self._lazy_init()
        assert self._env is not None
        from browser_env import create_playwright_action
        # B-160 (/stress A1.3 v8 Claude F1, 2026-05-16): VWA upstream
        # ``create_playwright_action`` evaluates the action_str as Python code
        # against the Playwright Page; previously the f-string ``f'page.goto("{url}")'``
        # broke out of the string literal if ``url`` contained a ``"`` character,
        # opening an arbitrary-Python-code injection vector. ``json.dumps`` emits a
        # JSON string literal which doubles as a syntactically-safe Python string
        # literal (escapes ``"``/``\``/control chars), closing the injection path.
        # Current Phase 1a callers pass trusted URLs from config files; this is
        # architectural hardening for any future caller that accepts arbitrary URLs.
        import json as _json
        action = create_playwright_action(f"page.goto({_json.dumps(url)})")
        obs, reward, terminated, truncated, info = self._env.step(action)
        # Fire-6 RCA Stage C1b: mode-gate screenshot-timeout recovery here too.
        self._gate_screenshot_timeout(info)
        p79_obs = self._to_p79_obs(obs, info)
        self._install_native_obs_nodes_info(p79_obs.obs_nodes_info)
        return p79_obs, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        self._dialog_registered_context = None  # B-158 (context-level handler)
        if self._env is not None:
            env = self._env
            self._env = None  # clear first to prevent re-entry
            try:
                env.close()  # VWA close(): calls context_manager.__exit__() only if reset_finished=True
            except Exception:
                logger.debug("env.close() raised (suppressed)", exc_info=True)
            # VWA's close() skips __exit__() when reset_finished=False (i.e. setup() failed
            # mid-way, e.g. ERR_CONNECTION_REFUSED during page.goto).  The Playwright event
            # loop started by __enter__() keeps running in its dispatcher greenlet, causing
            # every subsequent sync_playwright().__enter__() to raise "Sync API inside the
            # asyncio loop".  Force-close the context manager in that case.
            if not getattr(env, "reset_finished", True):
                ctx = getattr(env, "context_manager", None)
                pw = getattr(env, "playwright", None)
                if ctx is not None and pw is not None:
                    try:
                        ctx.__exit__(None, None, None)
                    except Exception:
                        logger.debug("ctx.__exit__() raised during force-close (suppressed)", exc_info=True)

    # ---------- form snapshot ----------

    # B-424 (/stress A1.3 v9 Mode B P2-2 / §147 P2-B5 closure, 2026-05-17):
    # Augment 200-char prefix with `value_len` + `value_djb2` (lightweight
    # JS hash). Pre-fix the bare prefix collapsed long-value edits beyond
    # 200 chars into "unchanged" → state_change false-no-progress on form
    # tasks with description / comment fields. Empirical archive
    # (codex Mode B receipts) had 25/7271 type actions >200 chars (max 807).
    # djb2 is sufficient for "did this value change" check; full SHA1 was
    # considered but per-step JS hashing overhead matters more for the
    # form snapshot hot path. Compare `(value_len, value_djb2)` in
    # state_change._form_fields_changed (separate fix); prefix kept for
    # human debug visibility.
    _FORM_SNAPSHOT_JS = """() => {
    const _djb2 = (s) => {
        let h = 5381;
        for (let i = 0; i < s.length; i++) {
            h = ((h << 5) + h) ^ s.charCodeAt(i);
        }
        return (h >>> 0).toString(16);
    };
    const fields = [];
    for (const el of document.querySelectorAll('input, textarea, select')) {
        const entry = {
            tag: el.tagName.toLowerCase(),
            type: (el.type || '').toLowerCase(),
            name: el.name || el.id || '',
            idx: Array.from(el.parentElement?.children || []).indexOf(el),
        };
        if (el.tagName === 'SELECT') {
            entry.selectedIndex = el.selectedIndex;
            entry.selectedText = (el.options[el.selectedIndex]?.text || '').trim();
            entry.value = el.value;
        } else if (el.type === 'checkbox' || el.type === 'radio') {
            entry.checked = el.checked;
            entry.value = el.value;
        } else {
            const _full = (el.value || '');
            entry.value = _full.substring(0, 200);
            // B-424: full-fidelity change detection without storing the
            // entire payload — len + djb2 hash captures suffix edits the
            // 200-char prefix would miss.
            entry.value_len = _full.length;
            entry.value_djb2 = _djb2(_full);
        }
        fields.push(entry);
    }
    const se = document.scrollingElement || document.body;
    return {
        fields: fields,
        scroll_y: Math.round(se.scrollTop),
        scroll_x: Math.round(se.scrollLeft),
        scroll_height: se.scrollHeight,
        client_height: window.innerHeight,
    };
}"""

    def snapshot_form_fields(self) -> Dict[str, Any]:
        """JS snapshot of all form field values + scroll position.

        B-419 (/stress A1.3 v9 Mode B P1-3 OOB, 2026-05-17): on exception
        the snapshot now stamps a `snapshot_error` sentinel so downstream
        `state_change.py` can distinguish "page genuinely has no fields"
        from "snapshot JS raised mid-navigation race". Pre-fix the bare
        empty dict collapsed both cases → `state_change` silently
        suppressed `form_value_changed` evidence → cross-baseline
        taxonomy contamination (B0 proxy higher network latency → wider
        race window → systematic SR bias).
        """
        empty: Dict[str, Any] = {
            "fields": [],
            "scroll_y": 0,
            "scroll_x": 0,
            "scroll_height": 0,
            "client_height": 0,
            "snapshot_error": None,
        }
        if self._env is None or self.dry_run:
            return empty
        try:
            result = self._env.page.evaluate(self._FORM_SNAPSHOT_JS)
            # Ensure snapshot_error key exists in the success path too so
            # downstream readers don't KeyError on missing-key vs explicit-
            # None disambiguation.
            if isinstance(result, dict) and "snapshot_error" not in result:
                result["snapshot_error"] = None
            return result
        except Exception as _e:
            # B-419: stamp typed error so state_change.py can flag the race
            # as a distinct page_change_reasons entry instead of silent
            # no-progress collapse.
            err_class = type(_e).__name__
            err_msg = str(_e)[:160]
            sentinel = dict(empty)
            sentinel["snapshot_error"] = f"{err_class}: {err_msg}"
            return sentinel

    # ---------- helpers ----------

    def _on_dialog(self, dialog: Any) -> None:
        """Auto-handle browser dialogs.

        - confirm / alert / beforeunload: accept (unblocks navigation +
          delete operations on Classifieds)
        - prompt: dismiss (no agent-supplied text input)

        B-423 (/stress A1.3 v9 Mode B P2-1 OOB, 2026-05-17): beforeunload
        moved from dismiss → accept. Pre-fix dismissing beforeunload meant
        "stay on page" — go_back / form-submit navigation after dirty form
        edit would silently cancel, taxonomy-blind. Most paper-grade
        agents WANT navigation to proceed; accept is the symmetric choice
        with confirm/alert. The only dialog type that should still dismiss
        is `prompt` (agent did not author a text-input response).

        B-509 (/stress A1.25 GRL Chunk 3 P1-2-BC* gemini + codex dual catch,
        2026-05-17): every dialog event is recorded into the per-step
        accumulator `_dialogs_this_step` so step_record carries
        `dialog_meta = [{type, message, accepted}, ...]`. Pre-fix dialogs
        were handled silently — reviewer cannot distinguish "agent
        intended destructive action" from "agent misclick that GRL
        amplified into destructive site mutation". VWA shared-account
        architecture (cls Blake / red Marvels) means misclick blast
        radius is cross-task; cross-baseline misclick rate differs →
        asymmetric SR contamination via state-mutation amplification.
        """
        accepted = False
        try:
            if dialog.type in ("confirm", "alert", "beforeunload"):
                dialog.accept()
                accepted = True
                logger.debug("Dialog auto-accepted: type=%s msg=%r", dialog.type, dialog.message)
            else:
                dialog.dismiss()
                logger.debug("Dialog dismissed: type=%s msg=%r", dialog.type, dialog.message)
        except Exception as _e:
            logger.warning("Dialog handler error: %s", _e)
        # B-509: record into per-step accumulator (drained at step() end + emit to info)
        try:
            self._dialogs_this_step.append({
                "type": str(dialog.type) if hasattr(dialog, "type") else "unknown",
                "message": (str(dialog.message)[:200]
                            if hasattr(dialog, "message") and dialog.message else None),
                "accepted": accepted,
            })
        except Exception:
            # Never let telemetry recording mask the actual dialog handling.
            pass

    def _goto_allowed_hosts(self) -> set:
        """Protocol Reset #5 (action-set restore, 2026-05-20): VWA-origin
        whitelist for the `goto` action.

        Allowed origins = `netloc` (host:port) of all currently-open browser
        tabs ∪ the configured VWA site origins (read from the same env vars
        upstream `browser_env/env_config.py` uses). This blocks the agent from
        issuing `goto` to an arbitrary off-VWA URL (which would break the eval
        harness / leave the controlled site set) while permitting every
        legitimate VWA origin — including cross-site tasks whose tabs are
        pre-opened via the `|AND|` start_url. Fails safe: any origin not
        provably a VWA origin is rejected (no-op) by the caller.

        P1-2-B* (codex OOB, 2026-05-20): match on `netloc` (host:port), NOT bare
        `hostname`. On the A100 self-host every VWA site is localhost:<port>, so
        a hostname-only whitelist collapsed to "any localhost port" — losing the
        per-site origin distinction the whitelist is supposed to enforce.
        """
        netlocs: set = set()
        try:
            for _p in self._env.context.pages:
                nl = (urlparse(_p.url).netloc or "").lower()
                if nl:
                    netlocs.add(nl)
        except Exception:
            pass
        for _var in (
            "REDDIT", "SHOPPING", "SHOPPING_ADMIN", "GITLAB",
            "WIKIPEDIA", "MAP", "HOMEPAGE", "CLASSIFIEDS",
        ):
            _v = os.environ.get(_var, "")
            if _v:
                nl = (urlparse(_v).netloc or "").lower()
                if nl:
                    netlocs.add(nl)
        return netlocs

    def _json_to_id_action_str(self, a: Dict[str, Any]) -> str:
        t = (a.get("action_type") or "").lower().strip()

        if t == "select_option":
            return "wait"   # 由 step() 直接处理，不经过此路径

        if t == "click":
            eid = a.get("element_id")
            if eid is None:
                raise ValueError(f"click requires element_id, got: {a}")
            return f"click [{int(eid)}]"

        if t == "type":
            eid = a.get("element_id")
            text = a.get("text", "")
            if eid is None:
                raise ValueError(f"type requires element_id, got: {a}")
            # VWA id-based parser cannot handle literal newlines inside text
            if isinstance(text, str):
                text = text.replace("\n", " ").replace("\r", " ")
            # 注意：文本里如果有 ']' 等符号，后续可以做转义；先跑通再说
            # Always include explicit enter flag [0] to avoid VWA parser ambiguity:
            # text="0" produces "type [406] [0]" which VWA misinterprets [0] as
            # enter_flag instead of text content (§task148 bug).
            return f"type [{int(eid)}] [{text}] [0]"

        if t == "scroll":
            direction = (a.get("direction") or "down").lower()
            # WebArena 常见方向：up/down/left/right
            return f"scroll [{direction}]"

        if t in ("stop", "finish", "done"):
            return "stop"

        if t == "wait":
            # 有些实现支持 wait；如果不支持就用 noop/stop 替代
            return "wait"

        # Protocol Reset #5 (action-set restore, 2026-05-20): hover / press /
        # new_tab / close_tab serialize to the upstream id-based action string
        # so the step() escape-hatch (`create_id_based_action`) executes them
        # via the upstream parser. goto is NOT here — it has an explicit step()
        # branch with a VWA-domain whitelist. Restored for upstream action-space
        # parity; cls/reddit demand ~0 (cross-site uses tab_focus).
        if t == "hover":
            eid = a.get("element_id")
            if eid is None:
                raise ValueError(f"hover requires element_id, got: {a}")
            return f"hover [{int(eid)}]"

        if t == "press":
            key = a.get("key") or a.get("key_comb") or ""
            key = str(key).strip()
            if not key:
                raise ValueError(f"press requires a key, got: {a}")
            return f"press [{key}]"

        if t == "new_tab":
            return "new_tab"

        if t == "close_tab":
            return "close_tab"

        # 兜底：如果 agent 直接给了 action_str
        if "action_str" in a:
            return str(a["action_str"])

        raise ValueError(f"Unknown action_type: {t}, raw={a}")

    def _to_p79_obs(self, obs: Dict[str, Any], info: Dict[str, Any]) -> P79Observation:
        # WebArena 文档提到可以从 obs["text"] 取文本观测（如 html / accessibility tree）:contentReference[oaicite:4]{index=4}
        text = ""
        if isinstance(obs, dict):
            text = obs.get("text", "") or ""

        # VWA 可能会包含 screenshot / image（不同版本 key 名可能不一样）
        image = None
        for k in ("image", "screenshot", "pixel", "rgb"):
            if isinstance(obs, dict) and k in obs:
                raw_img = obs[k]
                if np is not None and isinstance(raw_img, np.ndarray):
                    image = Image.fromarray(raw_img)
                else:
                    image = raw_img
                break

        url = None
        if isinstance(info, dict):
            url = info.get("url") or info.get("current_url")
            # VWA stores url inside info["page"].url (DetachedPage dataclass)
            if not url:
                page_obj = info.get("page")
                if page_obj is not None and hasattr(page_obj, "url"):
                    url = page_obj.url or None

        # Extract per-element bounding boxes from VWA observation metadata.
        # info["observation_metadata"]["text"]["obs_nodes_info"] maps str(element_id)
        # to {"union_bound": [x, y, width, height], ...} in pixel coordinates.
        obs_nodes_info: Optional[Dict[str, Any]] = None
        try:
            obs_nodes_info = (
                info.get("observation_metadata", {})
                    .get("text", {})
                    .get("obs_nodes_info")
            ) or None
        except Exception:
            logger.debug("Failed to extract obs_nodes_info", exc_info=True)

        # Inject select options into AXTree text.
        # VWA's AXTree shows combobox as "[N] combobox '' ... expanded: False" with no children.
        # We query the live page for all <select> elements and append their options after each
        # matching combobox line, so the agent can infer the correct option_label.
        if text and self._env is not None:
            try:
                text = self._inject_select_options(text, obs_nodes_info)
            except Exception as _e:
                logger.warning("select options injection failed: %s", _e)
            try:
                text = self._inject_css_dropdown_options(text, obs_nodes_info)
            except Exception as _e:
                logger.warning("css dropdown options injection failed: %s", _e)

        return P79Observation(text=text, image=image, url=url, raw=obs, obs_nodes_info=obs_nodes_info)

    def _inject_css_dropdown_options(self, axtree: str, obs_nodes_info: Optional[Dict[str, Any]]) -> str:
        """Inject [DROPDOWN OPTIONS] annotations for CSS/JS custom dropdowns (non-native <select>).

        Covers patterns like:
          - classifieds "Sort by": <span class="see_by"> + plain <ul><li><a>
          - reddit sort: <button class="dropdown__toggle"> + <ul class="dropdown__menu">

        Finds hidden ULs (getBoundingClientRect returns 0) with >=2 <li><a> items,
        locates the nearest visible ancestor as the "trigger", then injects [DROPDOWN OPTIONS]
        after the closest AXTree node to that trigger.
        """
        if not obs_nodes_info:
            return axtree

        try:
            dropdown_data = self._env.page.evaluate("""() => {
                const results = [];
                const seen = new Set();
                for (const ul of document.querySelectorAll('ul')) {
                    if (seen.has(ul)) continue;
                    seen.add(ul);
                    // Hidden ULs: zero bounding box
                    const ulRect = ul.getBoundingClientRect();
                    if (ulRect.width > 0 || ulRect.height > 0) continue;
                    // Must have 2-20 direct LI > A/BUTTON items (menu-like)
                    const items = Array.from(ul.querySelectorAll(':scope > li > a, :scope > li > button'))
                        .map(el => el.textContent.trim())
                        .filter(t => t.length > 0 && t.length < 100);
                    if (items.length < 2 || items.length > 20) continue;
                    // Find nearest visible ancestor (non-zero bounding box)
                    let trigger = null;
                    let el = ul.parentElement;
                    while (el && el !== document.body) {
                        const r = el.getBoundingClientRect();
                        if (r.width > 0 && r.height > 0 &&
                            r.top < window.innerHeight && r.bottom > 0) {
                            trigger = el;
                            break;
                        }
                        el = el.parentElement;
                    }
                    if (!trigger) continue;
                    const r = trigger.getBoundingClientRect();
                    results.push({
                        cx: r.left + r.width / 2,
                        cy: r.top + r.height / 2,
                        options: items
                    });
                }
                return results;
            }""")
        except Exception:
            return axtree

        if not dropdown_data:
            return axtree

        # Build node center lookup from obs_nodes_info
        node_centers: dict = {}
        for eid, node in obs_nodes_info.items():
            ub = node.get('union_bound')
            if ub:
                node_centers[eid] = (ub[0] + ub[2] / 2, ub[1] + ub[3] / 2)

        if not node_centers:
            return axtree

        # B-479 (/stress A1.25 GRL Chunk 2 P1-5-B carry of B-455, 2026-05-17):
        # accumulate menus per-eid instead of overwriting. Pre-fix
        # `injections: dict = {}` + `injections[best_eid] = dd['options']`
        # silently dropped all-but-last menu when multiple hidden `<ul>`s
        # clustered near one trigger (common: classifieds + reddit nav with
        # main + breadcrumb menus). Post-fix renders one `[DROPDOWN OPTIONS]`
        # line per accumulated menu so all options reach the agent. Parallel
        # A1.4 codex Mode B P1-5-B catch deferred to this session via commit
        # `901956d`; closes B-455 from the parallel-session deferral list.
        from collections import defaultdict
        # B-479: keys preserve original VWA `obs_nodes_info` str format
        # (str(element_id)) so `str(eid) in injections` lookup below matches.
        injections: Dict[str, list] = defaultdict(list)  # eid_str -> list[list[str]]
        for dd in dropdown_data:
            best_eid = min(
                node_centers,
                key=lambda e: (node_centers[e][0] - dd['cx']) ** 2 + (node_centers[e][1] - dd['cy']) ** 2,
            )
            cx, cy = node_centers[best_eid]
            dist = ((cx - dd['cx']) ** 2 + (cy - dd['cy']) ** 2) ** 0.5
            # B-422: named CSS dropdown threshold (was: inline 150).
            if dist > _INJECT_DISTANCE_CSS_DROPDOWN_PX:
                continue
            injections[best_eid].append(dd['options'])

        if not injections:
            return axtree

        # /stress A1.10 P1-2-AB* (2026-05-16): anchored mark-id extraction
        # via canonical helper from som.py, replacing unanchored
        # `re.search(r'\[(\d+)\]', line)`. Pre-fix the regex matched any
        # bracketed digit in the line including text-content references,
        # which could mis-inject [DROPDOWN OPTIONS] after a StaticText that
        # mentioned `[N]` rather than after the actual element row.
        from p79.experiment.som import extract_mark_id
        lines = axtree.splitlines()
        out = []
        for line in lines:
            out.append(line)
            eid = extract_mark_id(line)
            if eid is not None and str(eid) in injections:
                indent = len(line) - len(line.lstrip('\t'))
                prefix = '\t' * (indent + 1)
                # B-479: one `[DROPDOWN OPTIONS]` line per accumulated menu so
                # the agent sees all clustered menus rather than only the last.
                for opts in injections[str(eid)]:
                    opts_str = ', '.join(f'"{o}"' for o in opts)
                    out.append(f"{prefix}[DROPDOWN OPTIONS] {opts_str}")
        return '\n'.join(out)

    def _inject_select_options(self, axtree: str, obs_nodes_info: Optional[Dict[str, Any]]) -> str:
        """Append available options after each combobox line in the AXTree text.

        Matches each combobox element_id to a live <select> element via pixel coordinates
        from obs_nodes_info, then inserts an [OPTIONS] annotation immediately after the
        combobox line so the agent can use the correct option_label.
        """
        if not obs_nodes_info:
            return axtree

        # Collect all <select> elements on page with their bounding boxes, options, and selected value
        try:
            select_data = self._env.page.evaluate("""() => {
                return Array.from(document.querySelectorAll('select')).map(el => {
                    const r = el.getBoundingClientRect();
                    const selectedOpt = el.options[el.selectedIndex];
                    return {
                        cx: r.left + r.width / 2,
                        cy: r.top + r.height / 2,
                        options: Array.from(el.options)
                                      .filter(o => o.value !== '')
                                      .map(o => o.text.trim()),
                        selected: (selectedOpt && selectedOpt.value !== '')
                                  ? selectedOpt.text.trim() : null
                    };
                });
            }""")
        except Exception:
            return axtree

        if not select_data:
            return axtree

        # /stress A1.10 P1-2-AB* sibling propagation (2026-05-16): use canonical
        # anchored extractor instead of unanchored bracket-digit search.
        from p79.experiment.som import extract_mark_id
        lines = axtree.splitlines()
        out = []
        for line in lines:
            out.append(line)
            # Only process combobox lines that have an element id
            if 'combobox' not in line.lower():
                continue
            eid_int = extract_mark_id(line)
            if eid_int is None:
                continue
            eid = str(eid_int)
            node = obs_nodes_info.get(eid)
            if not node or 'union_bound' not in node:
                continue
            ub = node['union_bound']
            cx = ub[0] + ub[2] / 2
            cy = ub[1] + ub[3] / 2
            # Find the <select> whose center is closest to this combobox node
            best = min(select_data, key=lambda s: (s['cx'] - cx) ** 2 + (s['cy'] - cy) ** 2)
            dist = ((best['cx'] - cx) ** 2 + (best['cy'] - cy) ** 2) ** 0.5
            # B-422: named native select threshold (was: inline 100).
            if dist > _INJECT_DISTANCE_NATIVE_SELECT_PX or not best['options']:
                continue
            indent = len(line) - len(line.lstrip('\t'))
            prefix = '\t' * (indent + 1)
            opts_str = ', '.join(f'"{o}"' for o in best['options'])
            selected = best.get('selected')
            if selected:
                out.append(f"{prefix}[OPTIONS: currently selected=\"{selected}\"] {opts_str}")
            else:
                out.append(f"{prefix}[OPTIONS] {opts_str}")
        return '\n'.join(out)
