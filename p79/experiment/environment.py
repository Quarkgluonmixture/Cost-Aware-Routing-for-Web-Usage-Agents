from __future__ import annotations

import json
import logging
import os
import importlib.util
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from PIL import Image

from p79.envs.vwa_wrapper import P79Observation, VWAWrapper
from p79.experiment.metrics import classify_timeout


# B-1836 (Fire-5/6 eval-timeout unified root cause, 2026-05-22): evaluator
# retry tuning. The pre-fix code had max_eval_retries=3 but a keyword bug
# (is_nav_error used "timed out" spaced, never matching Playwright's
# "Timeout 30000ms exceeded") meant Playwright Page.goto timeouts got ZERO
# retries — one transient cls-docker degradation window aborted the whole
# condition (3-fire pattern). Fix: (1) eval_error_is_retryable() reuses the
# single-source classify_timeout(); (2) exponential LOCAL backoff so the
# retry sequence spans the empirically-observed ~8-10min transient window
# instead of ~100s back-to-back. The GLOBAL Page.goto timeout stays 30s on
# purpose — a wider single timeout would mask substrate degradation (user
# directive 2026-05-22). Whether local retry+backoff actually absorbs the
# window is a viability question for the B0-som-cls canary, NOT assumed here.
_EVAL_MAX_RETRIES = 5  # was 3; +2 attempts to span the transient window
_EVAL_RETRY_BACKOFF_BASE_S = 30.0  # exponential: 30, 60, 120, 180(cap)
_EVAL_RETRY_BACKOFF_CAP_S = 180.0


def eval_error_is_retryable(exc_str: Optional[str]) -> bool:
    """B-1836: whether an evaluator-phase exception warrants a fresh-context
    retry. Reuses the canonical classify_timeout() (single-source — already
    covers Playwright "Timeout 30000ms exceeded" / "timed out" / "deadline
    exceeded") PLUS non-timeout navigation-error keywords. The pre-B-1836
    inline list used only "timed out" (spaced) → Playwright's "timeout"
    (unspaced) never matched → eval Page.goto timeout never retried."""
    if not exc_str:
        return False
    is_timeout, _ = classify_timeout(exc_str)
    if is_timeout:
        return True
    low = exc_str.lower()
    return any(k in low for k in (
        "net::err_", "navigation failed", "target closed", "page closed",
    ))


@dataclass
class EpisodeEvalResult:
    score: float
    error: Optional[str] = None
    # Fire-6 RCA Stage C1 (/stress 2026-05-20, user-approved L2 program_html
    # isolation + timeout instrumentation). Evaluator-context provenance so
    # paper §3.5 can disclose which episodes used isolated-evaluator-context
    # navigation vs agent-page navigation, and forensic can correlate
    # eval Page.goto latency / timeout with the 3-fire stateful-edit pattern.
    #   eval_context_mode ∈ {agent_page, isolated_program_html_context,
    #                        no_browser_required}
    eval_context_mode: Optional[str] = None
    eval_isolated_context_used: Optional[bool] = None
    eval_goto_latency_ms: Optional[float] = None
    eval_goto_timeout: Optional[bool] = None
    eval_source_agent_url: Optional[str] = None
    eval_target_url: Optional[str] = None


# B-544 (/stress A1.5b Phase 2 P0-4-B codex OOB, 2026-05-17): paper-grade
# evaluator unavailability is a fatal infra failure, NOT silent agent failure.
# Pre-fix `VwaEvaluator.__init__` swallowed the import error + `evaluate()`
# returned `score=0.0,error="evaluator_unavailable"` → entire batch SR
# silently zeroed when evaluator harness / API key / BLIP-2 dep broken on
# A100. Cross-baseline infra-fragility confound. Now: paper_grade=True
# raises this exception so the runner exception path (B-486 hooks) flags
# the episode as needs_reevaluation=True instead of falsely scoring
# success=False. Dev mode keeps fail-open for iteration speed.
class EvaluatorUnavailableError(RuntimeError):
    """Raised when VwaEvaluator dependencies are missing / broken AND
    paper_grade=True.

    Two distinct failure modes with DIFFERENT semantics:

    (a) **init-time failure** (`VwaEvaluator.__init__` import error / missing
        OpenAI key / evaluator harness path broken). `ExperimentRunner.__init__`
        at `runner/main.py:145` has NO try/except around `create_evaluator()`
        — the exception propagates out of the constructor → CLI exits BEFORE
        any episode loop runs. **No** `needs_reevaluation=True` summary is
        written (no episode artifact exists yet). This path is **process-fatal**
        by design: fix env + restart runner (no point continuing with a broken
        scoring substrate). B-544 commit message intentionally framed init-time
        as halt-and-fix.

    (b) **evaluate-time failure** (post-init, evaluator dependencies broken
        when `evaluate()` is actually called). `_run_and_record_episode`
        (`runner/main.py:1185-1338`) catches inside the per-episode exception
        handler → writes `needs_reevaluation=True` summary via B-486 quarantine
        contract → continues the episode loop. The next restart will re-run
        quarantined episodes via the B-486 resume-gate force-rerun branch
        (`runner/main.py:818-830`).

    Per /stress A1.5 P1-1-AB* Claude+codex docstring clarification 2026-05-17.
    Pre-fix docstring claimed case (b) semantics for BOTH modes — case (a) is
    process-fatal (intentional, see B-544 commit message), not quarantine.
    Reviewer reading the docstring + finding no init-time `needs_reevaluation`
    forensic artifact would (correctly) conclude the contract was misstated."""


class PaperGradeAbortError(RuntimeError):
    """Raised when paper-grade mode catches the FIRST quarantine event
    (`needs_reevaluation=True` summary written) to enforce strict fail-closed
    condition abort — closes the Fire-4 RCA gap where R1 P1-10-B "preserve"
    semantic was pure tagging without runtime enforcement.

    **Trigger**: `_run_and_record_episode` writes a quarantined summary
    (B-486 path → `needs_reevaluation=True`) AND `cfg.paper_grade=True` →
    immediately after `write_episode_summary` succeeds, this exception is
    raised so the condition aborts rather than burning compute on
    potentially-compromised substrate.

    **Empirical motivation**: Fire-4 cls B0 dom 2026-05-19 — task 75
    Page.screenshot timeout at 20:48 wrote `needs_reevaluation=True`
    summary; runner CONTINUED tasks 76-101 (26 more tasks) until 4h
    wallclock kill at 21:46 (which itself only fired because the
    P1-16-AC `_B0_` regex regression denied B0 its 16h budget).
    Watchdog emitted `QUARANTINE-PRESERVED` + `PERSISTENT-ERROR` ntfy
    at 21:49 but had no abort authority. User core complaint: "并没有
    在 75 的时候自动暂停" — answered by this gate.

    **Contract**:
    - paper_grade=True + first `needs_reevaluation=True` → raise from
      runner; outer task-loop catch writes condition_summary with
      `condition_aborted=True, aborted_at_task=N, abort_reason="quarantine"`
      + re-raises so process exits non-zero rc, chain sentinel rc≠0 →
      master halt (P0-2-B sentinel-wait already in place).
    - paper_grade=False (dev mode) → no abort; legacy record-and-continue
      preserved for iteration speed.

    **Cross-AI overlap (3-AI confirmation)**: A Claude F1 / B codex F1 /
    C gemini A1-1 — highest-confidence Fire-4 RCA finding.

    Per /stress 3-AI Fire-4 RCA Wave 1 fix 2026-05-19. Bug catalog entry
    Fire-4 RCA Wave 1 M1 (depends on R1 B-486 quarantine substrate +
    R1 P1-10-B watchdog respect).

    **B-1881 (2026-06-20)**: carries structured provenance (`transient_class`,
    `steps`) so the transient-retry wrapper (`_run_and_record_episode`) can gate
    on the failure's class + step-count WITHOUT re-parsing the truncated message.
    The wrapper retries ONLY pre-flight (`steps == 0`) transient failures — where
    the agent took no action, so there is no site mutation, no stochastic-rollout
    redraw, and no agent-induced masking (3-AI /stress B-1881 consensus). Both
    fields default None/0 for legacy raises (back-compat)."""

    def __init__(
        self,
        message: str = "",
        *,
        transient_class: Optional[str] = None,
        steps: int = 0,
    ) -> None:
        super().__init__(message)
        # B-1881: structured failure provenance for the transient-retry gate.
        # transient_class ∈ {"auth","proxy_5xx","network",None}; steps = recovered
        # partial-step count (0 ⇒ pre-flight: auth gate / reset-goto / first model
        # call failed before any browser action ⇒ no mutation ⇒ clean to retry).
        self.transient_class = transient_class
        self.steps = int(steps or 0)


class MockEnvironment:
    def __init__(self, viewport_width: int = 1280, viewport_height: int = 720):
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self._step = 0

    def reset(self, config_file: str):
        self._step = 0
        text = "[1] Search textbox\n[2] Submit button\n[3] Product link"
        image = Image.new("RGB", (self.viewport_width, self.viewport_height), color="white")
        obs = P79Observation(text=text, image=image, raw={"text": text})
        return obs, {"url": "http://mock.local/start"}

    def step(self, action_json: Dict[str, Any]):
        self._step += 1
        text = "[1] Search textbox\n[2] Submit button\n[3] Product link"
        if self._step >= 2:
            text = "[4] Checkout button\n[5] Confirmation"
        image = Image.new("RGB", (self.viewport_width, self.viewport_height), color="white")
        obs = P79Observation(text=text, image=image, raw={"text": text})

        done = self._step >= 3 or action_json.get("action_type") in ("finish", "stop")
        reward = 1.0 if done else 0.0
        info = {
            "url": f"http://mock.local/step/{self._step}",
            "raw_action": {"action_type": "mock", "payload": action_json},
        }
        return obs, reward, done, False, info

    def get_all_tab_titles(self) -> list[tuple[str, str]]:
        return []

    def snapshot_form_fields(self) -> Dict[str, Any]:
        return {"fields": [], "scroll_y": 0, "scroll_x": 0, "scroll_height": 0, "client_height": 0}

    def set_dispatch_obs_nodes_info(self, obs_nodes_info):
        # Sequential SoM identifier contract (2026-05-25, codex round-3 P1):
        # protocol-compat no-op. The runner calls this for SoM-family modes to
        # push a seq-keyed dispatch map; the mock env does not dispatch by
        # element_id (its step() ignores the action target), so there is nothing
        # to override. Present so the runner can call it uniformly across the
        # VWAWrapper / MockEnvironment env protocol without an AttributeError.
        return None

    def close(self):
        return None


class NullEvaluator:
    """Mock-env evaluator stub. Returns score=0 with error="evaluator_unavailable".

    B-1407 (/stress A2.7 P1-7-A* Claude Mode A OOB, 2026-05-18): paper_grade=True
    callers MUST raise instead of silently scoring 0 — mock env under paper-grade
    flag is a misconfiguration (paper-grade orchestrator gates require env.type=vwa
    with localhost VWA URLs, NOT env.type=mock). Defense-in-depth pair with B-544
    fail-loud at VwaEvaluator init: even if mock env slips past the orchestrator
    Gate 4 (e.g., debug script + paper_grade=True yaml partial override), the
    last-line guard at this layer prevents silent cross-baseline SR=0 contamination.
    """

    def __init__(self, paper_grade: bool = False):
        self._paper_grade = paper_grade

    def evaluate(self, *args, **kwargs) -> EpisodeEvalResult:
        if self._paper_grade:
            raise EvaluatorUnavailableError(
                "NullEvaluator invoked under paper_grade=True — mock env under "
                "paper-grade flag is a misconfiguration. Mock env produces no real "
                "agent trajectory and would silently zero entire batch SR (cross-"
                "baseline infra-fragility confound). Set env.type=vwa for paper-"
                "grade runs OR clear paper_grade flag for dev-mode iteration. See "
                "B-1407 /stress A2.7 P1-7-A*."
            )
        return EpisodeEvalResult(score=0.0, error="evaluator_unavailable")


class VwaEvaluator:
    # Minimum free VRAM (GB) required before loading BLIP-2 on CUDA.
    # blip2-flan-t5-xl in float16 needs ~15 GB; add buffer.
    _BLIP2_MIN_FREE_VRAM_GB: float = 18.0
    _BLIP2_POLL_INTERVAL_S: int = 30
    # Give up waiting and fall back to CPU after this many seconds.
    # Must be well under the watchdog idle timeout (35 min = 2100 s).
    _BLIP2_MAX_WAIT_S: int = 10 * 60  # 10 minutes

    def __init__(self, paper_grade: bool = False):
        # B-544 (/stress A1.5b Phase 2 P0-4-B): paper-grade mode raises on
        # any evaluator dep failure (import / API key / module path) so the
        # runner can flag the episode for re-evaluation rather than
        # silently scoring `success=False` against missing infra.
        self._paper_grade = paper_grade
        self._available = False
        self._evaluator_router = None
        self._captioning_fn = None
        self._captioning_fn_ready = False  # lazy-load guard
        # B-797 (/stress A1.9 cold-start P2-4-C, 2026-05-17): BLIP-2 device
        # telemetry. Recorded at lazy-load time; None = not loaded yet, "cuda"
        # / "cpu" = loaded device. Surfaced into per-episode `evaluator_blip2_device`
        # so paper §3.5 audit can detect silent CPU fallback cross-baseline.
        self._blip2_device: Optional[str] = None
        try:
            import sys

            cwd = os.getcwd()
            candidate = os.path.join(cwd, "external/visualwebarena")
            if os.path.isdir(candidate):
                sys.path.append(candidate)

            # VisualWebArena may import OpenAI provider modules during evaluator
            # initialization even for non-LLM eval types. Keep import recoverable.
            # Load real OpenAI key from .auth/openai_key if available (not committed).
            # Override if the current value is a DUMMY placeholder (set by shell scripts
            # before Python starts, which prevents setdefault from working).
            _key_file = os.path.join(os.getcwd(), ".auth", "openai_key")
            if os.path.isfile(_key_file):
                with open(_key_file) as _kf:
                    _loaded_key = _kf.read().strip()
                if _loaded_key:
                    _cur_key = os.environ.get("OPENAI_API_KEY", "")
                    if not _cur_key or _cur_key.startswith("DUMMY"):
                        os.environ["OPENAI_API_KEY"] = _loaded_key
            os.environ.setdefault("OPENAI_API_KEY", "DUMMY_P79_NON_LLM_EVAL")

            # Fill dummy URLs for sites not used in this experiment so that
            # env_config.py assertions pass.  The evaluator only checks the
            # sites referenced by each task's config, so dummies are harmless.
            _DUMMY = "https://example.com"
            _dataset = os.environ.get("DATASET", "")
            if _dataset == "webarena":
                for _var in ("REDDIT", "SHOPPING", "SHOPPING_ADMIN",
                             "GITLAB", "WIKIPEDIA", "MAP", "HOMEPAGE"):
                    os.environ.setdefault(_var, _DUMMY)
            elif _dataset == "visualwebarena":
                for _var in ("REDDIT", "SHOPPING", "WIKIPEDIA", "HOMEPAGE",
                             "CLASSIFIEDS", "CLASSIFIEDS_RESET_TOKEN"):
                    os.environ.setdefault(_var, _DUMMY)

            from evaluation_harness import evaluator_router  # type: ignore

            self._available = True
            self._evaluator_router = evaluator_router
        except Exception as exc:
            logger.warning("VwaEvaluator init failed (eval scores will be 0): %s", exc)
            self._available = False
            self._evaluator_router = None
            # B-544: paper-grade mode = fail-loud. Dev mode = fail-open for
            # iteration speed (matches pre-fix legacy behaviour).
            if self._paper_grade:
                raise EvaluatorUnavailableError(
                    f"paper-grade evaluator init FAILED: {exc!r}. "
                    "VWA evaluator harness / OpenAI key / module path is broken; "
                    "scoring would silently zero entire batch SR (cross-baseline "
                    "infra-fragility confound). Fix env + restart runner."
                ) from exc

    def _ensure_captioning_fn(self) -> None:
        """Lazy-load BLIP-2 on first page_image_query task.

        Waits until enough GPU VRAM is free before loading on CUDA,
        so it doesn't compete with the main inference model.
        Falls back to CPU only if CUDA is unavailable.

        B-785 (/stress A1.9 cold-start P0-4-B* codex OOB, 2026-05-17):
        paper-grade mode = fail-loud on lazy-load failure. Pre-fix the
        `except Exception` branch unconditionally swallowed transformers /
        VRAM / dtype errors and continued with `captioning_fn=None` → any
        page_image_query task would silently score 0.0 across the entire
        run. This is a sibling path to B-544 init-time fail-loud: init
        was guarded but lazy load wasn't. Now `_paper_grade=True` raises
        `EvaluatorUnavailableError`; dev mode keeps the legacy
        captioning_fn=None (fail-open) for iteration speed.

        B-797 (/stress A1.9 cold-start P2-4-C gemini, 2026-05-17): also
        stamp `self._blip2_device` so the per-episode summary can record
        whether BLIP-2 ran on GPU or fell back to CPU — silent CPU
        fallback was a cross-baseline latency confound vector that had no
        forensic trail prior to this telemetry.
        """
        if self._captioning_fn_ready:
            return
        self._captioning_fn_ready = True
        _device = None
        try:
            import torch  # type: ignore
            from evaluation_harness.image_utils import get_captioning_fn  # type: ignore

            if torch.cuda.is_available():
                _deadline = time.monotonic() + self._BLIP2_MAX_WAIT_S
                while True:
                    torch.cuda.empty_cache()
                    free_gb = torch.cuda.mem_get_info(0)[0] / 1024 ** 3
                    if free_gb >= self._BLIP2_MIN_FREE_VRAM_GB:
                        _device, _dtype = "cuda", torch.float16
                        break
                    if time.monotonic() >= _deadline:
                        logger.warning(
                            "BLIP-2: VRAM wait timed out (%.1f GB free after %d s), "
                            "falling back to CPU",
                            free_gb, self._BLIP2_MAX_WAIT_S,
                        )
                        _device, _dtype = "cpu", torch.float32
                        break
                    logger.info(
                        "BLIP-2: waiting for VRAM (%.1f GB free, need %.1f GB), retry in %d s",
                        free_gb, self._BLIP2_MIN_FREE_VRAM_GB, self._BLIP2_POLL_INTERVAL_S,
                    )
                    time.sleep(self._BLIP2_POLL_INTERVAL_S)
            else:
                _device, _dtype = "cpu", torch.float32

            self._captioning_fn = get_captioning_fn(_device, _dtype)
            self._blip2_device = _device  # B-797 telemetry
            logger.info("BLIP-2 captioning model loaded on %s", _device)
        except Exception as exc:
            logger.warning(
                "BLIP-2 captioning fn unavailable (page_image_query tasks will score 0): %s",
                exc,
            )
            self._captioning_fn = None
            self._blip2_device = None  # B-797 telemetry
            # B-785: paper-grade mode raises so caller treats the failure
            # as infra (B-486 quarantine semantics), not a real agent score=0.
            if self._paper_grade:
                raise EvaluatorUnavailableError(
                    f"paper-grade BLIP-2 lazy-load FAILED: {exc!r}. "
                    "page_image_query tasks would silently score 0.0 across "
                    "the run (cross-baseline VLM evaluator-fragility confound). "
                    "Fix transformers / VRAM / dtype + restart runner. See B-785."
                ) from exc

    @staticmethod
    def _dump_eval_timeout_forensic(
        *,
        config_file: str,
        eval_context_mode: str,
        eval_isolated_context_used: bool,
        eval_source_agent_url: Optional[str],
        eval_target_url: Optional[str],
        attempt: int,
        exc: Exception,
    ) -> None:
        """Fire-6 RCA Stage C1b: capture mid-eval timeout forensic so even if
        the C1 isolation fix does not fully eliminate the 3-fire pattern, the
        next timeout has hard mid-fire evidence (A100 load + context mode +
        URLs + error). Best-effort: never raises (forensic must not mask the
        original eval exception). Writes one JSON per timeout to
        logs/eval_timeout_forensic/.
        """
        try:
            import os
            import platform
            from pathlib import Path as _Path

            repo_root = _Path(__file__).resolve().parent.parent.parent
            out_dir = repo_root / "logs" / "eval_timeout_forensic"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Derive task id from config_file name (e.g. .../test_classifieds/4.json)
            task_stem = _Path(config_file).stem
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

            loadavg = None
            try:
                loadavg = os.getloadavg()  # (1m, 5m, 15m)
            except (OSError, AttributeError):
                pass

            forensic = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "config_file": str(config_file),
                "task_stem": task_stem,
                "hostname": platform.node(),
                "eval_context_mode": eval_context_mode,
                "eval_isolated_context_used": eval_isolated_context_used,
                "eval_source_agent_url": eval_source_agent_url,
                "eval_target_url": eval_target_url,
                "attempt": attempt,
                "loadavg_1m_5m_15m": loadavg,
                "error_class": type(exc).__name__,
                "error_head": str(exc).splitlines()[0][:300] if str(exc) else "",
            }
            out_path = out_dir / f"eval_timeout_{task_stem}_{ts}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(forensic, f, indent=2, ensure_ascii=False)
            logger.warning(
                "Eval timeout forensic (C1b) captured → %s (mode=%s isolated=%s loadavg_1m=%s)",
                out_path, eval_context_mode, eval_isolated_context_used,
                loadavg[0] if loadavg else "n/a",
            )
        except Exception as _forensic_exc:
            logger.warning(
                "Eval timeout forensic (C1b) capture failed (non-fatal): %s",
                _forensic_exc,
            )

    @staticmethod
    def _classify_eval_context(config_file: str) -> tuple[str, Optional[str]]:
        """Fire-6 RCA Stage C1 (/stress 2026-05-20, user-approved): determine
        whether a task's eval can run in an isolated evaluator browser context
        (fresh page.goto to explicit target URL) vs MUST reuse the agent's page.

        Returns ``(eval_context_mode, first_program_html_target_url)``.

        eval_context_mode:
          * ``isolated_program_html_context`` — program_html with explicit
            target URLs only (no `__last_url__` / `"last"` / `func`-url). The
            VWA evaluator (`evaluators.py:368`) does `page.goto(target_url)`
            and reads from the server-rendered page — agent DOM is discarded
            by the goto, so a fresh isolated page is semantically identical
            AND avoids the runner-context cumulative-state hang (3-fire
            Fire-3/4/5 pattern: edit-form → public Page.goto 30s timeout).
            EMPIRICALLY VERIFIED 2026-05-20 (P2-1, B-1783, via
            scripts/maintenance/verify_eval_context_classification.py): cls has
            only **31** program_html tasks → **29 isolate** (2 have func/last
            targets → agent_page); red → **71 isolate**. task 4 + task 75 both
            isolate. (The earlier "Inspection: 232/234 cls + ~136 red qualify"
            figure was an ~8× / ~2× overcount — there are only 31 cls program_html
            tasks TOTAL; the bulk of cls tasks are url_match (131, read live
            page.url) or string_match (78), NOT program_html. Corrected B-1783.)
          * ``agent_page`` — needs the agent's live page: url_match (reads
            `page.url`), OR program_html with `__last_url__` / `"last"` /
            `func`-url target (URL derives from agent's final navigation).
            Verified 2026-05-20: cls 133 (131 url_match + 2 func/last program_html);
            red 62. url_match does NOT page.goto(target) so it is not exposed to
            the eval-goto cumulative-state timeout (that risk is program_html-goto
            specific — fully covered by isolation for the 29 cls / 71 red).
          * ``no_browser_required`` — string_match / ua_match (answer-based,
            no browser navigation).

        Detection is fail-safe: any config-read error → ``agent_page``
        (conservative — never isolate when unsure).
        """
        try:
            with open(config_file) as _f:
                cfg = json.load(_f)
        except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError):
            return "agent_page", None
        eval_block = cfg.get("eval", {}) or {}
        eval_types = eval_block.get("eval_types", []) or []
        if "program_html" not in eval_types:
            # url_match needs live page.url; string_match/ua_match no browser.
            if "url_match" in eval_types:
                return "agent_page", None
            return "no_browser_required", None
        # program_html present — isolate ONLY if every target uses an explicit
        # URL (no agent-page dependency).
        targets = eval_block.get("program_html", []) or []
        first_target_url: Optional[str] = None
        for t in targets:
            url = str(t.get("url", "") or "")
            if first_target_url is None and url:
                first_target_url = url
            if url == "last" or "__last_url__" in url or url.startswith("func"):
                # Any agent-page-dependent target forces agent_page for the
                # whole eval (EvaluatorComb runs all targets on one page).
                return "agent_page", first_target_url
        # url_match alongside program_html also needs agent page.
        if "url_match" in eval_types:
            return "agent_page", first_target_url
        return "isolated_program_html_context", first_target_url

    def evaluate(self, trajectory: List[Any], config_file: str, env: Any) -> EpisodeEvalResult:
        if not self._available or self._evaluator_router is None:
            # B-544: paper-grade mode raises so caller treats the episode
            # as needs_reevaluation=True (B-486 quarantine semantics).
            # Dev mode returns score=0 to preserve legacy iteration UX.
            if self._paper_grade:
                raise EvaluatorUnavailableError(
                    "paper-grade evaluator unavailable at evaluate() time "
                    f"(self._available={self._available}, router={self._evaluator_router!r}); "
                    "would have silently scored 0.0 — caller must treat as "
                    "infra failure, not real agent outcome."
                )
            return EpisodeEvalResult(score=0.0, error="evaluator_unavailable")

        # Lazy-load BLIP-2 only when the task actually needs it.
        # B-1701 (/stress A2.12 P0-2-B* OOB codex unique, 2026-05-18, user Q5=A):
        # narrow `except Exception: pass` → only swallow config-read errors
        # (FileNotFoundError / JSONDecodeError / KeyError) AND explicitly
        # re-raise EvaluatorUnavailableError from `_ensure_captioning_fn()`.
        # Pre-fix the broad except swallowed paper-grade B-785 fail-loud
        # (`raise EvaluatorUnavailableError` from BLIP-2 lazy-load failure at
        # `:277-285`) → captioning_fn stayed None → page_image_query tasks
        # silently scored 0.0 (paper §1 SR subset under-quote, cross-baseline
        # VLM-evaluator-fragility confound). Now: config-read errors still
        # swallowed (caller falls back to default eval_types); BLIP-2 lazy-load
        # exception (which is paper-grade infra-fail) propagates.
        try:
            with open(config_file) as _f:
                _eval_types = json.load(_f)["eval"]["eval_types"]
            if "page_image_query" in _eval_types:
                self._ensure_captioning_fn()
        except EvaluatorUnavailableError:
            raise
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass

        # B-329 (/stress A1.9 Mode A F6 OOB, 2026-05-16): skip fresh_page
        # retry for `program_html` eval_types. program_html validates the
        # DOM state from the agent's final navigation chain — fresh_page
        # (`page.context.new_page() + goto(target)`) produces a stateless
        # server-side render lacking the agent's uncommitted form values /
        # JS-mutated DOM / open dropdowns → retry false-negative on tasks
        # where the agent actually succeeded (paper §1 SR silent
        # under-quote). url_match / string_match / ua_match are unaffected:
        # url_match cares only about page.url (target same); string_match
        # is finish-answer-based; ua_match uses LLM judge on the answer.
        _is_program_html_task = False
        try:
            with open(config_file) as _ef:
                _cfg = json.load(_ef)
            _eval_types_for_retry = _cfg.get("eval", {}).get("eval_types", [])
            _is_program_html_task = "program_html" in _eval_types_for_retry
        # P1-9-AC* (/stress Phase 0 unified bug list 2026-05-19, Claude+Gemini
        # 2-AI overlap OOB): pre-fix bare `except Exception: pass` silently
        # disabled B-329 program_html escape hatch on any transient config-read
        # failure (fs hiccup / inode race) → fresh_page retry runs on real
        # program_html task → DOM state lost → eval false-negative OR fail-loud
        # at B-783 path. Fire-3 task 75 "All 3 retries exhausted" symptom
        # consistent with this silent disable. Narrow to expected config-read
        # exceptions; unexpected OSError / asyncio / IO failures propagate to
        # outer eval-phase handler.
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as _b329_exc:
            logger.warning(
                "B-329 program_html detection failed for %s: %s — fresh_page "
                "retry path will activate (eval false-negative risk if task IS "
                "program_html). Investigate config-file path / fs state.",
                config_file, _b329_exc,
            )

        max_eval_retries = _EVAL_MAX_RETRIES  # B-1836: was hardcoded 3
        page = env._env.page  # noqa: SLF001 - VWA evaluator requires underlying page

        # Fire-6 RCA Stage C1 (/stress 2026-05-20, user-approved L2 isolation):
        # determine eval context mode + proactively use an isolated fresh page
        # for program_html-safe tasks. Root cause (Z reproduce 2026-05-20):
        # agent's runner page accumulates 25-min cumulative state (heavy
        # edit-form DOM + A100 concurrent load) → evaluator `page.goto(target)`
        # hangs 30s × 3 → EvaluatorUnavailableError (Fire-3/4/5 task 75/4).
        # Z proved a FRESH page does the same goto in 639ms. So for tasks whose
        # program_html targets are explicit URLs (agent DOM discarded by goto
        # anyway), run the eval on an isolated page (shared auth cookies, clean
        # state). B-1803 (Fire-6 RCA C1b, 2026-05-21) UPGRADES this from a
        # same-context new_page to a FRESH browser CONTEXT — Fire-6 task 4 proved a
        # same-context fresh page STILL hangs 30s × 3 (the BrowserContext, not just
        # the page, degrades by task ~4); see _open_fresh_eval_page(). This
        # SUPERSEDES B-329's
        # skip-fresh-retry (which mistakenly assumed program_html needs agent
        # DOM — inspection 2026-05-20 confirmed evaluators.py:368 navigates away).
        eval_context_mode, eval_target_url = self._classify_eval_context(config_file)
        eval_source_agent_url: Optional[str] = None
        try:
            eval_source_agent_url = page.url
        except Exception:
            eval_source_agent_url = None
        eval_isolated_context_used = False
        eval_goto_latency_ms: Optional[float] = None
        eval_goto_timeout = False

        eval_page = page  # default: agent's page
        fresh_page = None  # track fresh page for cleanup
        fresh_context = None  # B-1803: track fresh eval CONTEXT for cleanup

        def _open_fresh_eval_page():
            """Fire-6 RCA C1b (B-1803, 2026-05-21): open the evaluator page in a
            FRESH browser CONTEXT — NOT page.context.new_page() (same context).

            Fire-6 (cls B0 dom task 4 / id=84144) proved the C1 same-context
            new_page is INSUFFICIENT: it timed out page.goto 30s × 3 even though
            the task isolated (eval_isolated_context_used=True) AND the target page
            was healthy (curl id=84144 = 0.17s; DB confirms item still active, NOT
            deleted). The agent's BrowserContext accumulates state by task ~4 (open
            dialogs / pending XHR / beforeunload / renderer pressure) and a
            same-context new_page inherits it. A fresh context is a clean Chromium
            profile (own cookie jar / cache / renderer) that loads the target in
            ~170ms like the curl. Carries the LIVE storage_state (auth cookies, so
            the eval stays logged-in) + viewport. Closes any prior fresh context
            first so each isolated attempt / retry gets a maximally-clean context.
            Returns the new page; sets the enclosing fresh_context for cleanup.
            """
            nonlocal fresh_context
            if fresh_context is not None:
                try:
                    fresh_context.close()
                except Exception:
                    pass
                fresh_context = None
            # Auth source: prefer the task config's storage_state FILE (the same
            # auth the env used at envs.py:206) over a live page.context.
            # storage_state() call — the agent context is the very thing that is
            # degraded/hung here, and storage_state() (CDP cookie read + localStorage
            # JS eval) could itself hang on a blocking modal. File read is inert.
            _storage = None
            try:
                with open(config_file) as _cf:
                    _storage = json.load(_cf).get("storage_state") or None
            except Exception:
                _storage = None
            if _storage is None:
                try:  # best-effort live fallback (may raise on a degraded context)
                    _storage = page.context.storage_state()
                except Exception:
                    _storage = None
            _kwargs = {}
            if _storage:
                _kwargs["storage_state"] = _storage
            try:
                _vp = page.viewport_size
                if _vp:
                    _kwargs["viewport"] = _vp
            except Exception:
                pass
            fresh_context = page.context.browser.new_context(**_kwargs)
            return fresh_context.new_page()

        if eval_context_mode == "isolated_program_html_context":
            try:
                # B-1803 (Fire-6 RCA C1b): FRESH CONTEXT, not new_page (same context).
                fresh_page = _open_fresh_eval_page()
                eval_page = fresh_page
                eval_isolated_context_used = True
                logger.info(
                    "Eval isolation (C1b): program_html-safe task → FRESH browser "
                    "context page (agent_url=%s target=%s)",
                    str(eval_source_agent_url)[:80], str(eval_target_url)[:80],
                )
            except Exception as _iso_exc:
                # Fail-safe: if fresh page creation fails, fall back to agent
                # page (degrades to pre-C1 behavior, no worse).
                logger.warning(
                    "Eval isolation (C1) fresh-page creation failed, falling "
                    "back to agent page: %s", _iso_exc,
                )
                eval_context_mode = "agent_page"
                eval_page = page

        def _eval_metadata() -> Dict[str, Any]:
            return {
                "eval_context_mode": eval_context_mode,
                "eval_isolated_context_used": eval_isolated_context_used,
                "eval_goto_latency_ms": eval_goto_latency_ms,
                "eval_goto_timeout": eval_goto_timeout,
                "eval_source_agent_url": eval_source_agent_url,
                "eval_target_url": eval_target_url,
            }

        last_exc: Optional[Exception] = None
        try:
            for attempt in range(max_eval_retries):
                try:
                    evaluator = self._evaluator_router(config_file, captioning_fn=self._captioning_fn)
                    _goto_t0 = time.monotonic()
                    score = evaluator(
                        trajectory=trajectory,
                        config_file=config_file,
                        page=eval_page,
                    )
                    eval_goto_latency_ms = (time.monotonic() - _goto_t0) * 1000.0
                    return EpisodeEvalResult(score=float(score), error=None, **_eval_metadata())
                except Exception as exc:  # pragma: no cover - depends on external environment
                    last_exc = exc
                    # B-1836 (Fire-5/6 eval-timeout unified root cause): reuse
                    # the single-source classify_timeout() for BOTH the forensic
                    # flag and the retry gate so they can never drift apart. The
                    # pre-fix inline "timed out" (spaced) never matched
                    # Playwright's "Timeout 30000ms exceeded" → ZERO retries.
                    _is_timeout_err, _timeout_callsite = classify_timeout(str(exc))
                    is_nav_error = eval_error_is_retryable(str(exc))
                    if _is_timeout_err:
                        # B-1836 P2-1: keep eval_goto_timeout goto-specific (the
                        # Fire-5/6 forensic signal is about Page.goto); non-goto
                        # timeouts (Page.screenshot/click/LLM-judge deadline)
                        # still dump forensic + stay retryable, but must NOT
                        # mislabel the goto flag.
                        if _timeout_callsite == "agent_navigation":
                            eval_goto_timeout = True
                        self._dump_eval_timeout_forensic(
                            config_file=config_file,
                            eval_context_mode=eval_context_mode,
                            eval_isolated_context_used=eval_isolated_context_used,
                            eval_source_agent_url=eval_source_agent_url,
                            eval_target_url=eval_target_url,
                            attempt=attempt,
                            exc=exc,
                        )
                    if is_nav_error and attempt < max_eval_retries - 1:
                        if _is_program_html_task and not eval_isolated_context_used:
                            # B-329 (legacy, agent_page program_html only): do NOT
                            # swap to fresh_page — DOM state would be lost. Bail
                            # out as evaluator_error. NOTE: with C1 isolation,
                            # program_html-safe tasks already run on a fresh page
                            # from the START (eval_isolated_context_used=True), so
                            # this branch only fires for agent-page-dependent
                            # program_html (`__last_url__` / `"last"` / func-url).
                            # B-1836 P0-1/P1-2: a TIMEOUT here is INFRA failure,
                            # not agent failure. B-329 forbids fresh-retry (DOM
                            # loss) AND silently scoring 0 would absorb the infra
                            # timeout as an agent error (deflate SR, B0-biased —
                            # proxy long-trajectory hits more eval timeouts).
                            # → fail-closed. Only non-timeout nav errors
                            # (net::err / target-closed) bail to score=0.0.
                            if _is_timeout_err and self._paper_grade:
                                raise EvaluatorUnavailableError(
                                    "paper-grade agent-page program_html evaluator "
                                    "TIMEOUT (B-329 forbids fresh-retry/DOM-loss; "
                                    "B-1836 forbids silent score=0.0 on infra "
                                    "timeout). needs_reevaluation=True (B-486 "
                                    "quarantine). err="
                                    f"{str(exc).splitlines()[0][:200] if str(exc) else 'unknown'}"
                                ) from exc
                            logger.warning(
                                "Evaluator nav error on agent-page program_html task; "
                                "skipping retry (B-329) to avoid DOM-state "
                                "loss on fresh_page. err=%s",
                                str(exc).split('\n')[0][:120],
                            )
                            return EpisodeEvalResult(
                                score=0.0,
                                error=f"evaluator_nav_error_program_html:{exc}",
                                **_eval_metadata(),
                            )
                        _backoff_s = min(
                            _EVAL_RETRY_BACKOFF_BASE_S * (2 ** attempt),
                            _EVAL_RETRY_BACKOFF_CAP_S,
                        )
                        logger.warning(
                            "Evaluator navigation error (attempt %d/%d), retrying with "
                            "fresh context in %.0fs (B-1836 local backoff; global 30s "
                            "goto timeout unchanged): %s",
                            attempt + 1, max_eval_retries, _backoff_s,
                            str(exc).split('\n')[0][:120],
                        )
                        # B-1803 (Fire-6 RCA C1b): the agent's page AND its
                        # long-lived BrowserContext may have dirty/degraded state
                        # (open dialogs, pending XHR, beforeunload, renderer
                        # pressure) that persistently blocks page.goto() — a fresh
                        # page in the SAME context (the old C1 behavior) still hung
                        # 30s × 3 in Fire-6. Open a fresh CONTEXT instead (clean
                        # Chromium profile + copied auth), and open a NEW one on
                        # every retry so each attempt gets a maximally-clean context.
                        try:
                            fresh_page = _open_fresh_eval_page()
                            eval_page = fresh_page
                            logger.info("Opened FRESH CONTEXT page for evaluator retry (C1b)")
                        except Exception as page_exc:
                            # B-1836 P1-1: fresh-context creation failed AFTER
                            # _open_fresh_eval_page closed the prior context →
                            # eval_page now points at a closed/stale page.
                            # Retrying would manufacture self-inflicted
                            # "Target closed" errors and burn retry budget on a
                            # known-bad page. Break to fail-closed instead.
                            logger.warning(
                                "Failed to open fresh eval context: %s — aborting "
                                "retry (B-1836 P1-1: avoid stale closed-page retry)",
                                page_exc,
                            )
                            break
                        time.sleep(_backoff_s)
                        continue
                    # B-783 (/stress A1.9 cold-start P0-2-AB* Claude+codex OOB,
                    # 2026-05-17): paper-grade mode = fail-loud on
                    # mid-evaluator-call exception. Pre-fix B-544 init-time
                    # raise was only half of the contract; ordinary evaluator
                    # crashes (Playwright timeout, OpenAI 503, helper_function
                    # exception) still returned score=0+error → silent agent
                    # failure → runner stamped success=False → paper §1 SR
                    # absorbed evaluator infra failure as real agent error.
                    # Cross-baseline confound: B0 (proxy API, long trajectories)
                    # > B1/B2 (local) on evaluator nav-error trigger rate, so
                    # mid-call exceptions are baseline-asymmetric.
                    if self._paper_grade:
                        raise EvaluatorUnavailableError(
                            f"paper-grade evaluator mid-call EXCEPTION: {exc!r}. "
                            f"Exhausted {attempt + 1}/{max_eval_retries} eval "
                            f"attempts (retryable={is_nav_error}); falling "
                            "back to score=0.0 would absorb evaluator infra "
                            "failure as agent error. Caller must treat as "
                            "needs_reevaluation=True (B-486 quarantine). "
                            f"err={str(exc).splitlines()[0][:200] if str(exc) else 'unknown'}"
                        ) from exc
                    return EpisodeEvalResult(score=0.0, error=f"evaluator_error:{exc}", **_eval_metadata())
            # B-783: same paper-grade fail-loud for the loop-exit no-retry path.
            if self._paper_grade and last_exc is not None:
                raise EvaluatorUnavailableError(
                    f"paper-grade evaluator loop-exit no retry: {last_exc!r}"
                ) from last_exc
            return EpisodeEvalResult(score=0.0, error=f"evaluator_error:{last_exc}", **_eval_metadata())
        finally:
            if fresh_page is not None:
                try:
                    fresh_page.close()
                except Exception:
                    pass
            # B-1803: close the fresh eval CONTEXT (clean Chromium profile) so it
            # does not leak across the 234-task run (one per isolated program_html eval).
            if fresh_context is not None:
                try:
                    fresh_context.close()
                except Exception:
                    pass


def create_environment(env_cfg: Dict[str, Any]):
    env_type = str(env_cfg.get("type", "vwa")).lower()

    if env_type == "mock":
        return MockEnvironment(
            viewport_width=int(env_cfg.get("viewport_width", 1280)),
            viewport_height=int(env_cfg.get("viewport_height", 720)),
        )

    if importlib.util.find_spec("browser_env") is None and not bool(env_cfg.get("dry_run", False)):
        raise RuntimeError(
            "env.type=vwa requires VisualWebArena/browser_env to be installed. "
            "Use env.type=mock for local smoke tests."
        )

    return VWAWrapper(
        headless=bool(env_cfg.get("headless", True)),
        observation_type=str(env_cfg.get("observation_type", "accessibility_tree")),
        viewport_width=int(env_cfg.get("viewport_width", 1280)),
        viewport_height=int(env_cfg.get("viewport_height", 720)),
        sleep_after_execution=float(env_cfg.get("sleep_after_execution", 0.5)),
        dry_run=bool(env_cfg.get("dry_run", False)),
        benchmark=str(env_cfg.get("benchmark", "visualwebarena")),
    )


def create_evaluator(env_cfg: Dict[str, Any], *, paper_grade: bool = False):
    # B-544 (/stress A1.5b Phase 2 P0-4-B): paper_grade flag propagation so
    # VwaEvaluator fail-loud when the harness / API key / dep is broken.
    # Runner (`main.py:139`) reads `paper_grade` from top-level cfg and
    # passes through.
    #
    # B-1407 (/stress A2.7 P1-7-A* Claude Mode A OOB, 2026-05-18): defense-in-
    # depth — even if env.type=mock slips into a paper-grade yaml (e.g.,
    # partial override + skipped base merge), construct-time RuntimeError
    # surfaces before any episode runs, preventing silent batch SR=0.
    # Orchestrator paper-grade gate 4 already requires env.type=vwa, but this
    # last-line guard catches misconfigured debug scripts + ad-hoc CLI yaml
    # paths that bypass the orchestrator.
    if str(env_cfg.get("type", "vwa")).lower() == "mock":
        if paper_grade:
            raise RuntimeError(
                "paper_grade=True with env.type=mock is a misconfiguration. "
                "Mock env produces no real agent trajectory; paper-grade runs "
                "MUST use env.type=vwa (with localhost VWA URLs for the A100 "
                "self-hosted docker stack). Set env.type=vwa OR clear "
                "paper_grade=False for dev-mode mock iteration. See "
                "B-1407 /stress A2.7 P1-7-A*."
            )
        return NullEvaluator(paper_grade=paper_grade)
    return VwaEvaluator(paper_grade=paper_grade)
