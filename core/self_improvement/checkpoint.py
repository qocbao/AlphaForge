import os
import shutil
import torch
from typing import Optional, Tuple, List, Dict, Any
from core.network.model import AlphaForgeNet
from core.training.checkpoint import save_checkpoint, load_checkpoint
import torch.optim as optim

class ModelManager:
    """
    Handles versioning and persistence of the best and candidate models.
    """
    def __init__(self, base_dir: str = "models"):
        from core.config import config
        self.base_dir = base_dir
        self.version = config.get("project.version")
        self.best_dir = os.path.join(base_dir, "best")
        self.iter_dir = os.path.join(base_dir, f"iterations-{self.version}")
        
        os.makedirs(self.best_dir, exist_ok=True)
        os.makedirs(self.iter_dir, exist_ok=True)

    def list_all_checkpoints(self) -> List[Dict[str, Any]]:
        """
        Recursively scans all directories under base_dir to find all .pt files.
        """
        checkpoints = []
        
        if not os.path.exists(self.base_dir):
            return checkpoints

        # Use os.walk to scan recursively through all subdirectories
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith(".pt"):
                    full_path = os.path.join(root, file)
                    
                    # Determine a friendly name for the display
                    # Relative path from base_dir
                    rel_path = os.path.relpath(full_path, self.base_dir)
                    
                    category = "Best" if "best" in rel_path.lower() else "Candidate"
                    
                    checkpoints.append({
                        "path": full_path,
                        "name": rel_path,
                        "category": category
                    })
                            
        return checkpoints

    def get_model_info(self, path: str) -> Dict[str, Any]:
        """Reads a checkpoint file and returns its metadata."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at: {path}")
        
        # 1. Basic Model Info (from .pt file)
        model = AlphaForgeNet()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        try:
            step, _ = load_checkpoint(path, model, optimizer)
        except Exception:
            step = "Unknown"
        
        # 3. Network Architecture (from config)
        return {
            "path": path,
            "step": step,
            "model_type": "AlphaForgeNet",
            "version": self.version,
        }

    def best_model_exists(self) -> bool:
        """Checks if a best model checkpoint already exists."""
        return os.path.exists(self.get_best_model_path())

    def get_best_model_path(self) -> str:
        """Returns path to the current best model."""
        return os.path.join(self.best_dir, "best_model.pt")

    def load_custom_model(self, model: AlphaForgeNet, optimizer: torch.optim.Optimizer, path: str) -> int:
        """Loads a model from a specific custom path."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at: {path}")
        
        step, _ = load_checkpoint(path, model, optimizer)
        return step

    def save_candidate(self, model: AlphaForgeNet, optimizer: torch.optim.Optimizer, iteration: int, step: int) -> str:
        """Saves a candidate model for a specific iteration."""
        iter_path = os.path.join(self.iter_dir, f"iter_{iteration:04d}")
        os.makedirs(iter_path, exist_ok=True)
        
        path = os.path.join(iter_path, "candidate.pt")
        save_checkpoint(model, optimizer, step, path)
        return path

    def promote_candidate(self, candidate_path: str):
        """Promotes a candidate model to be the new best model."""
        best_path = self.get_best_model_path()
        shutil.copy2(candidate_path, best_path)

    def load_best_model(self, model: AlphaForgeNet, optimizer: torch.optim.Optimizer) -> int:
        """Loads the best model. Returns the step count."""
        path = self.get_best_model_path()
        if not os.path.exists(path):
            return 0
        
        step, _ = load_checkpoint(path, model, optimizer)
        return step

    def initialize_best_model(self, model: AlphaForgeNet, optimizer: torch.optim.Optimizer):
        """Sets the initial best model."""
        path = self.get_best_model_path()
        save_checkpoint(model, optimizer, 0, path)