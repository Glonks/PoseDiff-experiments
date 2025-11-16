import math
import torch
import torch.nn as nn

from typing import Type


class TimestepEmbedder(nn.Module):
    def __init__(
            self,
            encoding_dim: int,
            hidden_dim: int,
            output_dim: int,
            max_period: int = 10000,
            activation: Type[nn.Module] = nn.SiLU
    ):
        super().__init__()

        self.embedding_dim = encoding_dim

        self.mlp = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, output_dim)
        )

        half_dim = encoding_dim // 2
        frequencies = torch.exp(
            -torch.arange(half_dim, dtype=torch.float32) * math.log(max_period) / half_dim
        )

        self.register_buffer('frequencies', frequencies)

    def get_timestep_encoding(self, timesteps: torch.Tensor) -> torch.Tensor:
        arguments = timesteps.unsqueeze(-1) * self.frequencies.unsqueeze(0)

        encoding = torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=-1)
        if self.embedding_dim % 2:
            encoding = torch.cat([encoding, torch.zeros_like(encoding[:, :1])], dim=-1)

        return encoding

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timestep_encoding = self.get_timestep_encoding(timesteps)

        timestep_embedding = self.mlp(timestep_encoding)

        return timestep_embedding
