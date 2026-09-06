import torch
import torch.nn as nn
from core.config import config
from .backbone import Backbone
from .policy_head import PolicyHead
from .value_head import ValueHead
 
class AlphaForgeNet(nn.Module):
    """
    Top-level neural network for AlphaForge.
    Integrates a shared convolutional backbone with separate Policy and Value heads.
    """
    def __init__(self):
        super().__init__()
        
        # Load network parameters from YAML config
        input_channels = config.get("network.input_channels")
        num_filters = config.get("network.num_filters")
        num_res_blocks = config.get("network.num_res_blocks")
        
        # Shared feature extractor
        self.backbone = Backbone(
            input_channels=input_channels,
            num_filters=num_filters,
            num_res_blocks=num_res_blocks
        )
        
        # Policy head: [B, C, H, W] -> [B, 100] logits
        self.policy_head = PolicyHead(num_filters=num_filters)
        
        # Value head: [B, C, H, W] -> [B, 1] scalar in [-1, 1]
        self.value_head = ValueHead(num_filters=num_filters)
 
    def forward(self, x: torch.Tensor):
        """
        Forward pass of the network.
        """
        # Feature extraction
        features = self.backbone(x)
        
        # Head outputs
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        
        return policy_logits, value
 
    def __repr__(self) -> str:
        return "AlphaForgeNet(Backbone -> PolicyHead & ValueHead)"
