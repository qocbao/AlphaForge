import torch
import torch.nn as nn
from core.config import config
 
class ResidualBlock(nn.Module):
    """
    A standard residual block for Gomoku feature extraction.
    Structure: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> (+) -> ReLU
    """
    def __init__(self, filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU()
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)
 
class Backbone(nn.Module):
    """
    Shared convolutional backbone for feature extraction.
    """
    def __init__(self, input_channels: int, num_filters: int, num_res_blocks: int):
        super().__init__()
        # Initial projection to match filter size
        self.init_conv = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        
        # Stack of residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.res_blocks(x)
        return x
