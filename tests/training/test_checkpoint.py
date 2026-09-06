import torch
import os
import pytest
from core.network.model import AlphaForgeNet
from core.training.checkpoint import save_checkpoint, load_checkpoint
from torch.optim import AdamW

def test_checkpoint_save_load(tmp_path):
    model = AlphaForgeNet()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    step = 42
    config = {"lr": 1e-3, "batch_size": 32}
    path = str(tmp_path / "checkpoint.pt")
    
    # Save
    save_checkpoint(model, optimizer, step, path, config)
    assert os.path.exists(path)
    
    # Modify model and optimizer to ensure they are restored
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)
            
    # Load
    loaded_step, loaded_config = load_checkpoint(path, model, optimizer)
    
    assert loaded_step == step
    assert loaded_config == config
    
def test_checkpoint_missing_file():
    model = AlphaForgeNet()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(FileNotFoundError):
        load_checkpoint("non_existent.pt", model, optimizer)
