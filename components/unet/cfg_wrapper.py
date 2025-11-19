import torch
import torch.nn as nn


class CFGWrapper(nn.Module):
    def __init__(self, model: nn.Module, guidance_strength: float):
        super().__init__()

        self.model = model
        self.guidance_strength = guidance_strength

    def forward_training(
            self,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            **kwargs
    ) -> torch.Tensor:
        return self.model(x, timesteps, **kwargs)

    def forward_evaluation(
            self,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            *,
            real_condition: torch.Tensor,
            null_condition: torch.Tensor
    ) -> torch.Tensor:
        predicted_noise_real = self.model(x, timesteps, condition=real_condition)
        predicted_noise_null = self.model(x, timesteps, condition=null_condition)

        predicted_noise = (
            (1 + self.guidance_strength) * predicted_noise_real -
            self.guidance_strength * predicted_noise_null
        )

        return predicted_noise

    def forward(
            self,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            **kwargs
    ) -> torch.Tensor:
        if self.training:
            return self.forward_training(x, timesteps, **kwargs)
        else:
            real_condition = kwargs["real_condition"]
            null_condition = kwargs["null_condition"]

            return self.forward_evaluation(x, timesteps, real_condition=real_condition, null_condition=null_condition)
