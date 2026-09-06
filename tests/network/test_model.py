import torch
from core.network.model import AlphaForgeNet
from core.config import NET_INPUT_CHANNELS, ACTION_SIZE

def test_model_shapes():
    """Verify output shapes for different batch sizes."""
    model = AlphaForgeNet()
    model.eval()
    
    batch_sizes = [1, 2, 8]
    for b in batch_sizes:
        x = torch.randn(b, NET_INPUT_CHANNELS, 10, 10)
        policy, value = model(x)
        
        assert policy.shape == (b, ACTION_SIZE), f"Policy shape mismatch for B={b}"
        assert value.shape == (b, 1), f"Value shape mismatch for B={b}"

def test_value_range():
    """Verify that value output is bounded by [-1, 1]."""
    model = AlphaForgeNet()
    model.eval()
    
    x = torch.randn(10, NET_INPUT_CHANNELS, 10, 10)
    _, value = model(x)
    
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0), "Value output out of range [-1, 1]"

def test_gradient_flow():
    """Verify that gradients can propagate through the model."""
    model = AlphaForgeNet()
    x = torch.randn(1, NET_INPUT_CHANNELS, 10, 10)
    policy, value = model(x)
    
    # Simple dummy loss
    loss = policy.sum() + value.sum()
    loss.backward()
    
    # Check if backbone weights have gradients
    for param in model.backbone.parameters():
        assert param.grad is not None, "Gradients did not reach backbone"

if __name__ == "__main__":
    try:
        test_model_shapes()
        test_value_range()
        test_gradient_flow()
        print("All network tests passed (mentally/statically)!")
    except Exception as e:
        print(f"Test failed: {e}")
