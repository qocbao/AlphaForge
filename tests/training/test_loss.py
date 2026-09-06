import torch
import pytest
from core.training.loss import calculate_policy_loss, calculate_value_loss, calculate_total_loss

def test_policy_loss_shapes():
    B, A = 4, 100
    logits = torch.randn(B, A)
    targets = torch.randn(B, A).softmax(dim=1)
    loss = calculate_policy_loss(logits, targets)
    assert loss.dim() == 0
    assert not torch.isnan(loss)

def test_value_loss_shapes():
    B = 4
    preds = torch.randn(B, 1)
    targets = torch.tensor([[1.0], [-1.0], [0.0], [1.0]])
    loss = calculate_value_loss(preds, targets)
    assert loss.dim() == 0
    assert not torch.isnan(loss)

def test_total_loss_weighting():
    p_loss = torch.tensor(1.0)
    v_loss = torch.tensor(2.0)
    
    # weight = 1.0
    total, _, _ = calculate_total_loss(p_loss, v_loss, value_weight=1.0)
    assert total.item() == 3.0
    
    # weight = 0.5
    total, _, _ = calculate_total_loss(p_loss, v_loss, value_weight=0.5)
    assert total.item() == 2.0

def test_policy_loss_stability():
    # Test with very large logits
    logits = torch.tensor([[100.0, -100.0], [-100.0, 100.0]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss = calculate_policy_loss(logits, targets)
    assert not torch.isinf(loss)
    assert not torch.isnan(loss)
