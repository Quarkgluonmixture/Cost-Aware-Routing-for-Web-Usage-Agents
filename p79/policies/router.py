class Router:
    """
    Placeholder for policy routing logic.
    Decides which representation (DOM vs Vision) or agent to use.
    """
    def __init__(self, config):
        self.config = config
    
    def select_modality(self, step: int) -> str:
        # Simple heuristic: always vision for now
        return "vision"
