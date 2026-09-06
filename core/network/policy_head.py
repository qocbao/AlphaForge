import torch
import torch.nn as nn
from core.config import ACTION_SIZE

class PolicyHead(nn.Module):
    """
    Converts shared latent features into policy logits for the full action space.
    """
    def __init__(self, num_filters: int):
        super().__init__()
        # Reduce channels first to avoid parameter explosion
        self.conv = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        # Final linear layer to project to policy index space [0...99]
        self.fc = nn.Linear(2 * 10 * 10, ACTION_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.flatten(x)
        return self.fc(x)
