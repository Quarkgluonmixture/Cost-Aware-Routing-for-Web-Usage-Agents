from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)
import os
import re
try:
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    np = None
from PIL import Image

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
_FUZZY_MATCH_JS = """
const _fuzzyFind = (candidates, label) => {
    // 1. Exact match (current behavior)
    const exact = candidates.find(c => c.text === label || (c.value && c.value === label));
    if (exact) return exact;
    // 2. Case-insensitive
    const lower = label.toLowerCase().trim();
    const ci = candidates.find(c =>
        c.text.toLowerCase().trim() === lower || (c.value && c.value.toLowerCase() === lower));
    if (ci) return ci;
    // 3. Keyword overlap
    const norm = s => s.toLowerCase().replace(/[^a-z0-9]/g, ' ').replace(/\\s+/g, ' ').trim();
    const stops = new Set(['the','a','an','to','by','of','in','on','for','and','or','is','it']);
    const kw = s => norm(s).split(' ').filter(w => w.length > 1 && !stops.has(w));
    const lkw = kw(label);
    if (!lkw.length) return null;
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
    return bestS >= 2 ? best : null;
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

        self._env = None  # lazy init
        # 保存上一次 obs 的 obs_nodes_info，供 select_option element_id 路径使用
        self._last_obs_nodes_info: Optional[Dict[str, Any]] = None
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
        # Use get_running_loop() — it only returns a loop if one is *actively*
        # running, unlike get_event_loop() which returns closed/idle loops too.
        # B-159 (/stress A1.3 v8 codex P1-B3, 2026-05-16): if a loop is
        # actively running we now fail loud with an actionable error message
        # instead of falling through into VWA's ``sync_playwright().__enter__()``
        # and getting a cryptic "Sync API inside the asyncio loop" RuntimeError
        # mid-init. Phase 1a callers (queue scripts → run_experiment.py) never
        # have an active loop; this guard mostly catches pytest-asyncio /
        # notebook / service-runner contexts that should isolate via subprocess.
        import asyncio as _asyncio
        try:
            _asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — install a fresh one for Playwright sync API.
            _asyncio.set_event_loop(_asyncio.new_event_loop())
        else:
            raise RuntimeError(
                "VWAWrapper._lazy_init() detected an active asyncio loop on this "
                "thread; Playwright sync API will fail. Run the wrapper in a "
                "subprocess (or a thread without a running loop) — e.g. "
                "pytest-asyncio / notebook / service-runner contexts must isolate. "
                "See p79/envs/vwa_wrapper.py:_lazy_init for the upstream root cause."
            )

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
            return P79Observation(text="[DRY_RUN]", image=dummy_img), {"dry_run": True}

        self._lazy_init()
        assert self._env is not None

        # Re-apply asyncio event loop reset before every _env.reset().
        # _lazy_init() only runs it on first init, but VWA program_html evaluators
        # (httpx/asyncio) can leave a stale loop that causes Playwright sync API to
        # raise "Sync API inside the asyncio loop" on subsequent resets.
        # B-159: same fail-loud contract as _lazy_init — if a loop is running we
        # cannot safely proceed.
        import asyncio as _asyncio
        try:
            _asyncio.get_running_loop()
        except RuntimeError:
            _asyncio.set_event_loop(_asyncio.new_event_loop())
        else:
            raise RuntimeError(
                "VWAWrapper.reset() detected an active asyncio loop; Playwright "
                "sync API will fail. See B-159 in _lazy_init for context."
            )

        try:
            obs, info = self._env.reset(options={"config_file": config_file})
        except Exception:
            # Keep wrapper recoverable across episodes after init/reset failures.
            self.close()
            raise

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
        self._last_obs_nodes_info = p79_obs.obs_nodes_info
        return p79_obs, info

    def get_all_tab_titles(self) -> list[tuple[str, str]]:
        """Return (url, title) for every open tab. Used for start-URL health checks."""
        if self.dry_run or self._env is None:
            return []
        try:
            pages = self._env.context.pages
            return [(p.url, p.title()) for p in pages]
        except Exception:
            return []

    def step(self, action_json: Dict[str, Any]) -> Tuple[P79Observation, float, bool, bool, Dict[str, Any]]:
        if self.dry_run:
            dummy_img = Image.new('RGB', (self.viewport_width, self.viewport_height), color='black')
            return P79Observation(text="[DRY_RUN]", image=dummy_img), 0.0, False, False, {"dry_run": True}

        self._lazy_init()
        assert self._env is not None

        from browser_env import (
            create_id_based_action,
            create_mouse_click_action,
            create_scroll_action,
            create_stop_action,
            create_go_back_action,
            create_go_forward_action,
            create_page_focus_action,
            create_keyboard_type_action,
            create_none_action,
            create_playwright_action
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
                else:
                    logger.debug(
                        "locator-route click fallback: eid=%s reason=%s",
                        eid, _lr_result.get("error", "")[:80],
                    )
                    action = create_id_based_action(f"click [{eid}]")
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
                left = float(coord[0])
                top = float(coord[1])
                # Accept either normalized [0-1] or pixel coordinates.
                # Normalize each dimension independently to handle mixed formats
                # (e.g. [0.26, 330] where x is normalized but y is pixel).
                if left > 1.0:
                    left = left / float(self.viewport_width)
                if top > 1.0:
                    top = top / float(self.viewport_height)
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
            else:
                action = None
        elif action_type == "scroll" and ("delta" in action_json or "scroll_direction" in action_json):
            if "scroll_direction" in action_json:
                # Semantic scroll direction (from tool-calling schema).
                direction = "down" if action_json["scroll_direction"] == "down" else "up"
                action = create_scroll_action(direction=direction)
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
                else:
                    direction = "down" if dy > 0 else "up"
                    action = create_scroll_action(direction=direction)
        elif action_type == "type" and "text" in action_json and "element_id" not in action_json:
            # Type without element_id (vision mode): click coordinate first to focus, then keyboard type.
            # IMPORTANT: Use direct page.mouse.click() instead of env.step(click_action).
            # VWA's env.step() captures observations via CDP DOMSnapshot after each action,
            # which causes the focused INPUT element to lose focus (focus resets to BODY).
            # By clicking directly, we preserve focus for the subsequent keyboard.type().
            coord = action_json.get("coordinate")
            if coord is not None and isinstance(coord, (list, tuple)) and len(coord) == 2:
                left = float(coord[0])
                top = float(coord[1])
                # Normalize each dimension independently (same as click path)
                if left > 1.0:
                    left = left / float(self.viewport_width)
                if top > 1.0:
                    top = top / float(self.viewport_height)
                eps = 1e-6
                left = max(eps, min(1.0 - eps, left))
                top = max(eps, min(1.0 - eps, top))
                self._env.page.mouse.click(
                    left * self.viewport_width,
                    top * self.viewport_height,
                )
                self._env.page.wait_for_timeout(int(self.sleep_after_execution * 1000))
                # Select-all + delete to clear existing field content, but ONLY when the
                # click actually focused an editable element.  Applying Control+a to a
                # non-input element selects the entire page (blue highlight) and then
                # Backspace does nothing useful — this was the root cause of the
                # "full-page blue-select" artifact in Vision mode (task_3 step_7, etc.).
                try:
                    is_editable = self._env.page.evaluate(
                        "() => { const el = document.activeElement; "
                        "return el != null && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable); }"
                    )
                except Exception:
                    # Click may have triggered a navigation, destroying the
                    # execution context.  Safe to skip the clear – the field
                    # is gone anyway.
                    is_editable = False
                if is_editable:
                    self._env.page.keyboard.press("Control+a")
                    self._env.page.keyboard.press("Backspace")
            action = create_keyboard_type_action(action_json["text"])
        elif action_type == "type" and "text" in action_json and "element_id" in action_json:
            # Treat invalid/zero element_id as keyboard typing fallback
            try:
                element_id = int(action_json.get("element_id"))
            except (TypeError, ValueError):
                element_id = None
            if element_id is None or element_id <= 0:
                action = create_keyboard_type_action(action_json["text"])
            else:
                _type_needs_enter = bool(str(action_json.get("text", "")).endswith("\n"))
                # B-01 Cluster 1 fix: locator-route TYPE bypasses framework's
                # mouse.click(center) + Meta+A + Backspace + keyboard.type
                # pattern that causes 全选变蓝 (§52/§64) when click hits
                # non-input. Uses locator.fill() which auto-clears and dispatches
                # input event WITHOUT global Meta+A. press_enter handled by
                # text trailing-newline detection.
                from p79.envs.locator_dispatch import dispatch_id_based_type as _lr_type
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
                    action = create_none_action()
                    # Locator dispatch already pressed Enter if text ended with
                    # \n (see locator_dispatch.py:dispatch_id_based_type); avoid
                    # double Enter from post-step keyboard.press at line ~565.
                    _type_needs_enter = False
                else:
                    logger.debug(
                        "locator-route type fallback: eid=%s reason=%s",
                        element_id, _lr_result.get("error", "")[:80],
                    )
                    # Framework will build id-based action below; nothing to do.
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

            if "element_id" in action_json:
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
                    else:
                        ub = node_info["union_bound"]
                    if ub is None:
                        pass  # no-op fallback handled above; skip the page.evaluate()
                    else:
                        x_px = ub[0] + ub[2] / 2
                        y_px = ub[1] + ub[3] / 2
                        self._env.page.evaluate(
                            _FUZZY_MATCH_JS + """([x, y, label, idx]) => {
                            const el = document.elementFromPoint(x, y);
                            // 1. Native <select> path
                            if (el && el.tagName === 'SELECT') {
                                if (idx !== null) {
                                    el.selectedIndex = idx;
                                } else {
                                    const cands = Array.from(el.options).map(o => ({text: o.text.trim(), value: o.value, el: o}));
                                    const match = _fuzzyFind(cands, label);
                                    if (match) { el.value = match.value; }
                                }
                                el.dispatchEvent(new Event('change', {bubbles: true}));
                                return;
                            }
                            // 2. CSS custom dropdown fallback: scan hidden <ul>s near (x, y)
                            if (!label) return;
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
                                const match2 = _fuzzyFind(cands2, label);
                                const opt = match2 ? match2.el : null;
                                if (opt) {
                                    const oldDisplay = ul.style.display;
                                    const oldVisibility = ul.style.visibility;
                                    ul.style.display = 'block';
                                    ul.style.visibility = 'visible';
                                    opt.click();
                                    ul.style.display = oldDisplay;
                                    ul.style.visibility = oldVisibility;
                                    return;
                                }
                            }
                        }""",
                            [x_px, y_px, option_label, option_index],
                        )
                        self._env.page.wait_for_timeout(int(self.sleep_after_execution * 1000))
                except Exception as _e:
                    logger.warning("select_option (element_id=%s) failed: %s", action_json.get("element_id"), _e)
            elif "coordinate" in action_json:
                # Vision 路径：通过坐标找元素，用 JS 设置选中值
                try:
                    coord = action_json["coordinate"]
                    x_norm, y_norm = float(coord[0]), float(coord[1])
                    x_px = x_norm * self.viewport_width if x_norm <= 1.0 else x_norm
                    y_px = y_norm * self.viewport_height if y_norm <= 1.0 else y_norm
                    self._env.page.evaluate(
                        _FUZZY_MATCH_JS + """([x, y, label]) => {
                            // 1. Native <select> path
                            const el = document.elementFromPoint(x, y);
                            if (el && el.tagName === 'SELECT') {
                                const cands = Array.from(el.options).map(o => ({text: o.text.trim(), value: o.value, el: o}));
                                const match = _fuzzyFind(cands, label);
                                if (match) {
                                    el.value = match.value;
                                    el.dispatchEvent(new Event('change', {bubbles: true}));
                                }
                                return;
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
                                const match2 = _fuzzyFind(cands2, label);
                                const opt = match2 ? match2.el : null;
                                if (opt) {
                                    const oldDisplay = ul.style.display;
                                    const oldVisibility = ul.style.visibility;
                                    ul.style.display = 'block';
                                    ul.style.visibility = 'visible';
                                    opt.click();
                                    ul.style.display = oldDisplay;
                                    ul.style.visibility = oldVisibility;
                                    return;
                                }
                            }
                        }""",
                        [x_px, y_px, option_label],
                    )
                    self._env.page.wait_for_timeout(int(self.sleep_after_execution * 1000))
                except Exception as _e:
                    logger.warning("select_option (coordinate) failed: %s", _e)
            action = create_none_action()
        elif action_type == "wait":
            action = create_none_action()

        if action is None and action_type == "click" and "element_id" not in action_json:
            action = create_none_action()

        if action is None:
            # Fallback to action_str or ID based
            if "action_str" in action_json:
                action = create_playwright_action(str(action_json["action_str"]))
            else:
                try:
                    action_str = self._json_to_id_action_str(action_json)
                    action = create_id_based_action(action_str)
                except Exception as _e:
                    logger.warning("create_id_based_action failed (%s), falling back to wait: %s", action_json, _e)
                    action = create_none_action()

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
                self._env.page.wait_for_timeout(int(self.sleep_after_execution * 1000))
                re_obs, _, _, _, re_info = self._env.step(create_none_action())
                obs, info = re_obs, re_info
            except Exception as _e:
                logger.warning("Post-type Enter press failed: %s", _e)

        if action_type in ("finish", "stop"):
            terminated = True
        info["raw_action"] = action  # Expose the raw VWA action for trajectory recording
        # B-156: surface locator-route dispatch telemetry into info so the
        # runner can persist it into StepRecordV2 (paper §3 evidence layer).
        # None when the step did not invoke locator-route (scroll, wait,
        # coord-only click, etc.).
        info["locator_route_meta"] = _locator_route_meta
        p79_obs = self._to_p79_obs(obs, info)
        self._last_obs_nodes_info = p79_obs.obs_nodes_info
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
        p79_obs = self._to_p79_obs(obs, info)
        self._last_obs_nodes_info = p79_obs.obs_nodes_info
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

    _FORM_SNAPSHOT_JS = """() => {
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
            entry.value = (el.value || '').substring(0, 200);
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
        """JS snapshot of all form field values + scroll position."""
        empty: Dict[str, Any] = {"fields": [], "scroll_y": 0, "scroll_x": 0, "scroll_height": 0, "client_height": 0}
        if self._env is None or self.dry_run:
            return empty
        try:
            return self._env.page.evaluate(self._FORM_SNAPSHOT_JS)
        except Exception:
            return empty

    # ---------- helpers ----------

    def _on_dialog(self, dialog: Any) -> None:
        """Auto-handle browser dialogs.

        - confirm / alert: accept (unblocks delete operations on Classifieds)
        - prompt / beforeunload: dismiss (safer default)
        """
        try:
            if dialog.type in ("confirm", "alert"):
                dialog.accept()
                logger.debug("Dialog auto-accepted: type=%s msg=%r", dialog.type, dialog.message)
            else:
                dialog.dismiss()
                logger.debug("Dialog dismissed: type=%s msg=%r", dialog.type, dialog.message)
        except Exception as _e:
            logger.warning("Dialog handler error: %s", _e)

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

        # For each dropdown trigger, find nearest AXTree node (any type)
        injections: dict = {}  # eid -> list[str]
        for dd in dropdown_data:
            best_eid = min(
                node_centers,
                key=lambda e: (node_centers[e][0] - dd['cx']) ** 2 + (node_centers[e][1] - dd['cy']) ** 2,
            )
            cx, cy = node_centers[best_eid]
            dist = ((cx - dd['cx']) ** 2 + (cy - dd['cy']) ** 2) ** 0.5
            if dist > 150:  # sanity: within 150px
                continue
            injections[best_eid] = dd['options']

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
                opts = injections[str(eid)]
                indent = len(line) - len(line.lstrip('\t'))
                prefix = '\t' * (indent + 1)
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
            if dist > 100 or not best['options']:  # sanity: within 100px
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
