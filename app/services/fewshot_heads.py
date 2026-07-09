"""PyTorch heads for user-defined few-shot downstream tasks."""

import torch
from torch import nn


class BinaryConv3x3ProbeHead(nn.Module):
    """Binary 3x3 conv probe with light local context."""

    def __init__(self, embed_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
