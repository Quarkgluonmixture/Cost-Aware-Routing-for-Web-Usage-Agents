from __future__ import annotations

import os
import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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

    def close(self):
        return None


class NullEvaluator:
    def evaluate(self, *args, **kwargs) -> EpisodeEvalResult:
        return EpisodeEvalResult(score=0.0, error="evaluator_unavailable")


class VwaEvaluator:
    def __init__(self):
        self._available = False
        self._evaluator_router = None
        try:
            import sys

            cwd = os.getcwd()
            candidate = os.path.join(cwd, "external/visualwebarena")
            if os.path.isdir(candidate):
                sys.path.append(candidate)

            from evaluation_harness import evaluator_router  # type: ignore

            self._available = True
            self._evaluator_router = evaluator_router
        except Exception:
            self._available = False
            self._evaluator_router = None

    def evaluate(self, trajectory: List[Any], config_file: str, env: Any) -> EpisodeEvalResult:
        if not self._available or self._evaluator_router is None:
            return EpisodeEvalResult(score=0.0, error="evaluator_unavailable")

        try:
            evaluator = self._evaluator_router(config_file)
            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=env._env.page,  # noqa: SLF001 - VWA evaluator requires underlying page
            )
            return EpisodeEvalResult(score=float(score), error=None)
        except Exception as exc:  # pragma: no cover - depends on external environment
            return EpisodeEvalResult(score=0.0, error=f"evaluator_error:{exc}")


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
        dry_run=bool(env_cfg.get("dry_run", False)),
    )


def create_evaluator(env_cfg: Dict[str, Any]):
    if str(env_cfg.get("type", "vwa")).lower() == "mock":
        return NullEvaluator()
    return VwaEvaluator()
