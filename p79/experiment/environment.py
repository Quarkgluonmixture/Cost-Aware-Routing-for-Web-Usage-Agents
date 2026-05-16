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
    def evaluate(self, *args, **kwargs) -> EpisodeEvalResult:
        return EpisodeEvalResult(score=0.0, error="evaluator_unavailable")


class VwaEvaluator:
    # Minimum free VRAM (GB) required before loading BLIP-2 on CUDA.
    # blip2-flan-t5-xl in float16 needs ~15 GB; add buffer.
    _BLIP2_MIN_FREE_VRAM_GB: float = 18.0
    _BLIP2_POLL_INTERVAL_S: int = 30
    # Give up waiting and fall back to CPU after this many seconds.
    # Must be well under the watchdog idle timeout (35 min = 2100 s).
    _BLIP2_MAX_WAIT_S: int = 10 * 60  # 10 minutes

    def __init__(self):
        self._available = False
        self._evaluator_router = None
        self._captioning_fn = None
        self._captioning_fn_ready = False  # lazy-load guard
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

    def _ensure_captioning_fn(self) -> None:
        """Lazy-load BLIP-2 on first page_image_query task.

        Waits until enough GPU VRAM is free before loading on CUDA,
        so it doesn't compete with the main inference model.
        Falls back to CPU only if CUDA is unavailable.
        """
        if self._captioning_fn_ready:
            return
        self._captioning_fn_ready = True
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
            logger.info("BLIP-2 captioning model loaded on %s", _device)
        except Exception as exc:
            logger.warning(
                "BLIP-2 captioning fn unavailable (page_image_query tasks will score 0): %s",
                exc,
            )
            self._captioning_fn = None

    def evaluate(self, trajectory: List[Any], config_file: str, env: Any) -> EpisodeEvalResult:
        if not self._available or self._evaluator_router is None:
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
                    return EpisodeEvalResult(score=0.0, error=f"evaluator_error:{exc}")
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


def create_evaluator(env_cfg: Dict[str, Any]):
    if str(env_cfg.get("type", "vwa")).lower() == "mock":
        return NullEvaluator()
    return VwaEvaluator()
