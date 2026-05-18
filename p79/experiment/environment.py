from __future__ import annotations

import json
import logging
import os
import importlib.util
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from PIL import Image

from p79.envs.vwa_wrapper import P79Observation, VWAWrapper


@dataclass
class EpisodeEvalResult:
    score: float
    error: Optional[str] = None


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
        try:
            with open(config_file) as _f:
                _eval_types = json.load(_f)["eval"]["eval_types"]
            if "page_image_query" in _eval_types:
                self._ensure_captioning_fn()
        except Exception:
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
        except Exception:
            pass

        max_eval_retries = 3
        page = env._env.page  # noqa: SLF001 - VWA evaluator requires underlying page
        eval_page = page  # first attempt uses agent's page
        fresh_page = None  # track fresh page for cleanup
        last_exc: Optional[Exception] = None
        try:
            for attempt in range(max_eval_retries):
                try:
                    evaluator = self._evaluator_router(config_file, captioning_fn=self._captioning_fn)
                    score = evaluator(
                        trajectory=trajectory,
                        config_file=config_file,
                        page=eval_page,
                    )
                    return EpisodeEvalResult(score=float(score), error=None)
                except Exception as exc:  # pragma: no cover - depends on external environment
                    last_exc = exc
                    err_lower = str(exc).lower()
                    is_nav_error = any(k in err_lower for k in (
                        "net::err_", "navigation failed", "timed out",
                        "target closed", "page closed",
                    ))
                    if is_nav_error and attempt < max_eval_retries - 1:
                        if _is_program_html_task:
                            # B-329: do NOT swap to fresh_page for program_html
                            # — DOM state would be lost. Bail out as
                            # evaluator_error so paper §1 SR analyzer can
                            # exclude (denominator-side) rather than counting
                            # as false agent failure.
                            logger.warning(
                                "Evaluator nav error on program_html task; "
                                "skipping retry (B-329) to avoid DOM-state "
                                "loss on fresh_page. err=%s",
                                str(exc).split('\n')[0][:120],
                            )
                            return EpisodeEvalResult(
                                score=0.0,
                                error=f"evaluator_nav_error_program_html:{exc}",
                            )
                        logger.warning(
                            "Evaluator navigation error (attempt %d/%d), retrying with fresh page in 5s: %s",
                            attempt + 1, max_eval_retries, str(exc).split('\n')[0][:120],
                        )
                        # Agent's page may have dirty state (open dialogs,
                        # pending XHR, beforeunload handlers) that persistently
                        # blocks page.goto().  Open a fresh page in the same
                        # browser context so cookies/auth are shared but state
                        # is clean.
                        if fresh_page is None:
                            try:
                                fresh_page = page.context.new_page()
                                eval_page = fresh_page
                                logger.info("Opened fresh page for evaluator retry")
                            except Exception as page_exc:
                                logger.warning("Failed to open fresh page: %s", page_exc)
                        time.sleep(5)
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
                            f"All {max_eval_retries} retries exhausted; falling "
                            "back to score=0.0 would absorb evaluator infra "
                            "failure as agent error. Caller must treat as "
                            "needs_reevaluation=True (B-486 quarantine). "
                            f"err={str(exc).splitlines()[0][:200] if str(exc) else 'unknown'}"
                        ) from exc
                    return EpisodeEvalResult(score=0.0, error=f"evaluator_error:{exc}")
            # B-783: same paper-grade fail-loud for the loop-exit no-retry path.
            if self._paper_grade and last_exc is not None:
                raise EvaluatorUnavailableError(
                    f"paper-grade evaluator loop-exit no retry: {last_exc!r}"
                ) from last_exc
            return EpisodeEvalResult(score=0.0, error=f"evaluator_error:{last_exc}")
        finally:
            if fresh_page is not None:
                try:
                    fresh_page.close()
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
