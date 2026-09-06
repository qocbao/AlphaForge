import torch
import torch.nn.functional as F
from typing import Tuple

def calculate_policy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculates the cross-entropy loss between predicted logits and soft target distribution.
    
    Args:
        logits: Network output [B, ACTION_SIZE]
        targets: MCTS visit counts distribution [B, ACTION_SIZE]
        
    Returns:
        Scalar policy loss.
    """
    # Use log_softmax for numerical stability
    log_probs = F.log_softmax(logits, dim=1)
    
    # Cross entropy for soft targets: -sum(target * log_prob)
    # targets are assumed to be normalized (sum=1)
    loss = -torch.sum(targets * log_probs, dim=1).mean()
    
    return loss

def calculate_value_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Mean Squared Error loss for the value head.
    
    Args:
        predictions: Network output [B, 1]
        targets: Game outcomes z in {-1, 0, 1} [B, 1]
        
    Returns:
        Scalar value loss.
    """
    # predictions: [B, 1], targets: [B, 1]
    return F.mse_loss(predictions, targets)

def calculate_total_loss(
    policy_loss: torch.Tensor, 
    value_loss: torch.Tensor, 
    value_weight: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combines policy and value losses into a single scalar for backpropagation.
    
    Args:
        policy_loss: Scalar tensor
        value_loss: Scalar tensor
        value_weight: Weight lambda for the value loss
        
    Returns:
        Tuple of (total_loss, policy_loss, value_loss)
    """
    total_loss = policy_loss + value_weight * value_loss
    return total_loss, policy_loss, value_loss
