import torch
import torch.nn as nn

class ValueHead(nn.Module):
    """
    Converts shared latent features into a scalar value estimation.
    """
    def __init__(self, num_filters: int):
        super().__init__()
        # Reduce channels to a small number before pooling
        self.conv = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1, 1)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.tanh(x)
