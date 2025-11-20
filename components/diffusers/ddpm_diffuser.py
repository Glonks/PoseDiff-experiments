import torch
import torch.nn as nn

from .schedulers import cosine_beta_schedule

from typing import Optional, Union, Callable


class DDPMDiffuser(nn.Module):
    def __init__(self, timesteps: int):
        super().__init__()

        self.timesteps = timesteps

        # Constants
        betas = cosine_beta_schedule(timesteps)

        alphas = 1 - betas

        alphas_cumprod = torch.cumprod(alphas, dim=0)

        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        # q posterior
        q_posterior_mean_x_0_coefficient = (
            (torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)) * betas
        )

        q_posterior_mean_x_t_coefficient = (
            (torch.sqrt(alphas) * (1 - alphas_cumprod_prev)) / (1 - alphas_cumprod)
        )

        q_posterior_variance = ((1 - alphas_cumprod_prev) / (1 - alphas_cumprod)) * betas

        # self.q_posterior_variance[0] is 0. We replace it with the second variance value
        # while computing the log variance (Consistent with
        # https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py)
        q_posterior_log_variance_clipped = torch.log(
            torch.cat([q_posterior_variance[1:2], q_posterior_variance[1:]], dim=0)
        )

        # Recovering x_0 (x_0_prediction)
        x_0_from_x_t_x_t_coefficient = 1 / torch.sqrt(alphas_cumprod)

        x_0_from_x_t_noise_coefficient = -torch.sqrt(1 - alphas_cumprod) / torch.sqrt(alphas_cumprod)

        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        self.register_buffer('q_posterior_mean_x_0_coefficient', q_posterior_mean_x_0_coefficient)
        self.register_buffer('q_posterior_mean_x_t_coefficient', q_posterior_mean_x_t_coefficient)
        self.register_buffer('q_posterior_variance', q_posterior_variance)
        self.register_buffer('q_posterior_log_variance_clipped', q_posterior_log_variance_clipped)

        self.register_buffer('x_0_from_x_t_x_t_coefficient', x_0_from_x_t_x_t_coefficient)
        self.register_buffer('x_0_from_x_t_noise_coefficient', x_0_from_x_t_noise_coefficient)

    @staticmethod
    def extract(
            tensor: torch.Tensor,
            indices: torch.Tensor,
            shape: torch.Size
    ) -> torch.Tensor:
        """
        Get tensor values in a shape suitable for broadcasting.
        """
        batch_size, *_ = tensor.shape

        out = tensor.gather(-1, indices)
        out = out.reshape(batch_size, *([1] * (len(shape) - 1)))

        # TODO: test if this works
        # out = tensor[indices].reshape(batch_size, *([1] * (len(shape) - 1)))

        return out

    def get_q_posterior_parameters(
            self,
            x_0: torch.Tensor,
            x_t: torch.Tensor,
            timesteps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters for q(x_{t - 1} | x_t, x_0)
        """
        # Mean
        q_posterior_mean_x_0_coefficient = self.extract(
            self.q_posterior_mean_x_0_coefficient, timesteps, x_0.shape
        )
        q_posterior_mean_x_t_coefficient = self.extract(
            self.q_posterior_mean_x_t_coefficient, timesteps, x_t.shape
        )

        q_posterior_mean = (
            q_posterior_mean_x_0_coefficient * x_0 +
            q_posterior_mean_x_t_coefficient * x_t
        )

        # Variance
        q_posterior_variance = self.extract(
            self.q_posterior_variance, timesteps, x_0.shape
        )

        # Log variance
        q_posterior_log_variance_clipped = self.extract(
            self.q_posterior_log_variance_clipped, timesteps, x_0.shape
        )

        return q_posterior_mean, q_posterior_variance, q_posterior_log_variance_clipped

    def q_sample(
            self,
            x_0: torch.Tensor,
            timesteps: torch.Tensor,
            noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample from q(x_t | x_0) - Forward process.
        Used only during training.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        alphas_cumprod_t = self.extract(
            self.alphas_cumprod, timesteps, x_0.shape
        )

        x_t = (
            torch.sqrt(alphas_cumprod_t) * x_0 +
            torch.sqrt(1 - alphas_cumprod_t) * noise
        )

        return x_t

    def predict_x_0_from_x_t(
            self,
            model: nn.Module,
            x_t: torch.Tensor,
            timesteps: torch.Tensor,
            **model_kwargs
    ) -> torch.Tensor:
        predicted_noise = model(x_t, timesteps, **model_kwargs)

        x_0_from_x_t_x_t_coefficient = self.extract(
            self.x_0_from_x_t_x_t_coefficient, timesteps, x_t.shape
        )
        x_0_from_x_t_noise_coefficient = self.extract(
            self.x_0_from_x_t_noise_coefficient, timesteps, x_t.shape
        )

        x_0_prediction = (
            x_0_from_x_t_x_t_coefficient * x_t +
            x_0_from_x_t_noise_coefficient * predicted_noise
        )

        return x_0_prediction

    def get_p_posterior_parameters(
            self,
            model: nn.Module,
            x_t: torch.Tensor,
            timesteps: torch.Tensor,
            **model_kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters for p_\theta(x_{t - 1} | x_t) which is an approximation of q(x_{t - 1} | x_t, x_0).
        """
        x_0_prediction = self.predict_x_0_from_x_t(model, x_t, timesteps, **model_kwargs)

        return self.get_q_posterior_parameters(x_0_prediction, x_t, timesteps)

    @torch.no_grad()
    def p_sample(
            self,
            model: Union[nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
            x_t: torch.Tensor,
            timesteps: torch.Tensor,
            **model_kwargs
    ) -> torch.Tensor:
        """
        Sample from p_\theta(x_{t - 1} | x_t) - A single step in the reverse process.
        """
        p_posterior_mean, p_posterior_variance, p_posterior_log_variance = self.get_p_posterior_parameters(
            model, x_t, timesteps, **model_kwargs
        )

        noise = torch.randn_like(x_t)

        nonzero_timesteps_mask = (
            (timesteps != 0)
            .float()
            .reshape(x_t.shape[0], *([1] * (x_t.ndim - 1)))  # Make it broadcastable
        )

        x_tm1 = (
            p_posterior_mean +
            nonzero_timesteps_mask * torch.exp(0.5 * p_posterior_log_variance) * noise
        )

        return x_tm1
