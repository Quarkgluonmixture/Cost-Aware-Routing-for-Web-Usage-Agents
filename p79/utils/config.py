import yaml
import os
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            try:
                config = yaml.safe_load(f)
                return config
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing config file: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the config using dot notation (e.g., 'model.path')."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def update(self, key: str, value: Any):
        """Update a config value using dot notation."""
        keys = key.split('.')
        target = self.config
        for k in keys[:-1]:
            target = target.setdefault(k, {})
        target[keys[-1]] = value

# Singleton instance for easy access
default_config = ConfigLoader().config

