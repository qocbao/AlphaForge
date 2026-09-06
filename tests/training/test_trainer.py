import torch
import pytest
import numpy as np
from core.network.model import AlphaForgeNet
from core.replay.buffer import ReplayBuffer
from core.replay.sample import ReplaySample
from core.training.trainer import Trainer

def test_trainer_step_updates_weights():
    model = AlphaForgeNet()
    buffer = ReplayBuffer(capacity=100)
    
    # Add a dummy sample
    real_state = [[[0]*10 for _ in range(10)] for _ in range(2)]
    real_policy = [0.01]*100
    # Normalize policy
    sum_p = sum(real_policy)
    real_policy = [p/sum_p for p in real_policy]
    
    sample = ReplaySample(state=real_state, policy=real_policy, value=1.0)
    buffer.add(sample)
    
    trainer = Trainer(model, buffer)
    
    # Record weight before step
    initial_param = next(model.parameters()).clone()
    
    # Train 1 step
    metrics = trainer.train_step(batch_size=1)
    
    # Check weights changed
    updated_param = next(model.parameters())
    assert not torch.equal(initial_param, updated_param)
    
    # Check metrics
    assert "total_loss" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert metrics["step"] == 1

def test_trainer_invalid_batch_size():
    model = AlphaForgeNet()
    buffer = ReplayBuffer(capacity=10)
    trainer = Trainer(model, buffer)
    
    with pytest.raises(ValueError):
        trainer.train_step(batch_size=1) # Buffer empty
