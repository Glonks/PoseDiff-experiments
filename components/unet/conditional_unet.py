import torch
import torch.nn as nn

from .. import FiLM

from typing import Type


class FiLMConditionedResidualBlock(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            condition_dim: int,
            activation: Type[nn.Module] = nn.SiLU
    ):
        super().__init__()

        self.identity_projector = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )

        self.linear_1 = nn.Linear(input_dim, output_dim)
        self.activation_1 = activation()

        self.linear_2 = nn.Linear(output_dim, output_dim)
        self.film = FiLM(output_dim, condition_dim)
        self.activation_2 = activation()

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        identity = self.identity_projector(x)

        x = self.linear_1(x)
        x = self.activation_1(x)

        x = self.linear_2(x)
        x = self.film(x, condition)
        x = self.activation_2(x)

        x = x + identity

        return x


class ConditionalUNet(nn.Module):
    """
    A UNet implementation with FiLM conditioned blocks at every level.
    """
    def __init__(
            self,
            keypoint_dim: int,
            condition_dim: int,
            hidden_dims: list[int],
            activation: Type[nn.Module] = nn.SiLU
    ):
        super().__init__()

        # Encoder
        self.input_projector = nn.Linear(keypoint_dim, hidden_dims[0])

        self.encoder_blocks = nn.ModuleList([
            FiLMConditionedResidualBlock(
                hidden_dims[i],
                hidden_dims[i + 1],
                condition_dim,
                activation
            )
            for i in range(len(hidden_dims) - 1)
        ])

        # Bottleneck
        self.bottleneck_block = FiLMConditionedResidualBlock(
            hidden_dims[-1],
            hidden_dims[-1],
            condition_dim,
            activation
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList([
            FiLMConditionedResidualBlock(
                hidden_dims[i + 1] * 2,
                hidden_dims[i],
                condition_dim,
                activation
            )
            for i in reversed(range(len(hidden_dims) - 1))
        ])

        self.output_projector = nn.Linear(hidden_dims[0], keypoint_dim)

    def forward(
            self,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            condition: torch.Tensor
    ) -> torch.Tensor:
        x = self.input_projector(x)

        # Forward through encoder
        skip_connections = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, condition)

            skip_connections.append(x)

        # Forward through bottleneck
        x = self.bottleneck_block(x, condition)

        # Forward through decoder
        for decoder_block, skip_connection in zip(self.decoder_blocks, reversed(skip_connections)):
            x = torch.cat([x, skip_connection], dim=-1)
            x = decoder_block(x, condition)

        x = self.output_projector(x)

        return x
