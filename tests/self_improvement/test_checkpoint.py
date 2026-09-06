import pytest
import os
import shutil
from core.self_improvement.checkpoint import ModelManager
from core.network.model import AlphaForgeNet
import torch.optim as optim

def test_model_manager_persistence(tmp_path):
    base_dir = str(tmp_path / "models")
    manager = ModelManager(base_dir=base_dir)
    
    model = AlphaForgeNet()
    optimizer = optim.AdamW(model.parameters())
    
    # Initialize best
    manager.initialize_best_model(model, optimizer)
    assert os.path.exists(manager.get_best_model_path())
    
    # Save candidate
    cand_path = manager.save_candidate(model, optimizer, iteration=1, step=100)
    assert "iter_0001" in cand_path
    assert os.path.exists(cand_path)
    
    # Promote
    manager.promote_candidate(cand_path)
    # Best path should now be the promoted candidate
    assert os.path.exists(manager.get_best_model_path())
