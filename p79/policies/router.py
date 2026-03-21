from p79.experiment.router import RuleBasedRouter, RouterState


class Router:
    """
    Backward-compatible wrapper around the new rule-based router.
    """

    def __init__(self, config):
        self._router = RuleBasedRouter(config)
        self._state = RouterState()

    def select_modality(self, step: int, obs_text: str = "", prev_action_success=None, prev_page_changed=None) -> str:
        decision, _, _, self._state = self._router.decide(
            router_enabled=True,
            preferred_mode="hybrid",
            obs_text=obs_text,
            state=self._state,
            prev_action_success=prev_action_success,
            prev_page_changed=prev_page_changed,
        )
        return decision
