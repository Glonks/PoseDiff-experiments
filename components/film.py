import torch
import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, feature_dim: int, condition_dim: int):
        super().__init__()

        self.scale_transform = nn.Linear(condition_dim, feature_dim)
        self.bias_transform = nn.Linear(condition_dim, feature_dim)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale = self.scale_transform(condition)
        bias = self.bias_transform(condition)

        return scale * features + bias
