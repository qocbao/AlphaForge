import torch
import os
from typing import Dict, Any, Tuple, Optional
from core.network.model import AlphaForgeNet

def save_checkpoint(
    model: AlphaForgeNet, 
    optimizer: torch.optim.Optimizer, 
    step: int, 
    path: str, 
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Saves the training state to a file.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'config': config
    }
    torch.save(checkpoint, path)

def load_checkpoint(
    path: str, 
    model: AlphaForgeNet, 
    optimizer: torch.optim.Optimizer
) -> Tuple[int, Optional[Dict[str, Any]]]:
    """
    Loads the training state from a file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")
        
    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    step = checkpoint.get('step', 0)
    config = checkpoint.get('config', None)
    
    return step, config