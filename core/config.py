import yaml
import os
from typing import Any, Dict, Optional


class ConfigLoader:
    """
    Handles loading and accessing configuration from YAML files.
    """
    def __init__(self, config_path: Optional[str] = None):
        # Xác định đường dẫn tuyệt đối tới thư mục gốc của dự án
        # __file__ là đường dẫn tới core/config.py, ta đi ngược lên 1 cấp để ra gốc AlphaForge
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if config_path is None:
            self.config_path = os.path.join(base_dir, "configs", "base_config.yaml")
        else:
            self.config_path = config_path
            
        self.data = self._load_yaml()


    def _load_yaml(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get(self, key_path: str, default_value: Any = None) -> Any:
        """
        Access nested keys using dot notation (e.g., 'project.version').
        Returns default_value if the key is not found.
        """
        keys = key_path.split('.')
        value = self.data
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            if default_value is not None:
                return default_value
            raise KeyError(f"Configuration key '{key_path}' not found in {self.config_path}")

    def set(self, key_path: str, value: Any):
        """Sets a value in the config data (in-memory)."""
        keys = key_path.split('.')
        target = self.data
        for k in keys[:-1]:
            target = target.setdefault(k, {})
        target[keys[-1]] = value

# Global instance for easy access
config = ConfigLoader()

# Critical Constants - These MUST exist in the config file
VERSION = config.get("project.version")
PROJECT_NAME = config.get("project.name")
BOARD_SIZE = config.get("game.board_size")
ACTION_SIZE = config.get("game.action_size")
NET_INPUT_CHANNELS = config.get("network.input_channels")
MCTS_C_PUCT = 1.414 # This is a mathematical constant for AlphaZero, not a game config
MCTS_TEMPERATURE = config.get("selfplay.temp")

