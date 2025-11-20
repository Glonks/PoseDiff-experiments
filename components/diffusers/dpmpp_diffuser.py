import torch
import torch.nn as nn
from typing import Optional, Union, Callable, Dict

from .schedulers import cosine_beta_schedule

class DPMPPDiffuser(nn.Module):
    def __init__(self, timesteps: int):
        super().__init__()
        self.timesteps = timesteps

        # 1. Calculate Schedules
        # We use the same beta schedule to remain consistent with the training
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # 2. Derive DPM-Solver++ specific parameters
        # alpha_t and sigma_t
        self.alpha_t = torch.sqrt(alphas_cumprod)
        self.sigma_t = torch.sqrt(1 - alphas_cumprod)
        
        # lambda_t = log(alpha_t) - log(sigma_t) (Log-SNR)
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)

        # Register buffers
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alpha_t_schedule', self.alpha_t)
        self.register_buffer('sigma_t_schedule', self.sigma_t)
        self.register_buffer('lambda_t_schedule', self.lambda_t)

        # State for the solver (to store previous noise prediction)
        # We use a simple dictionary to map batch indices or just assume 
        # strictly sequential single-batch-group processing. 
        # For safety in this architecture, we'll just store the last tensor.
        self.last_noise_prediction: Optional[torch.Tensor] = None

    @staticmethod
    def extract(tensor: torch.Tensor, indices: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        """Get tensor values in a shape suitable for broadcasting."""
        batch_size, *_ = shape
        out = tensor[indices].reshape(batch_size, *([1] * (len(shape) - 1)))
        return out

    def q_sample(self, x_0: torch.Tensor, timesteps: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion process (same as DDPM).
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        alphas_cumprod_t = self.extract(self.alphas_cumprod, timesteps, x_0.shape)
        
        x_t = (
            torch.sqrt(alphas_cumprod_t) * x_0 +
            torch.sqrt(1 - alphas_cumprod_t) * noise
        )
        return x_t

    def reset_solver_state(self):
        """Resets the history for the 2nd order solver."""
        self.last_noise_prediction = None

    @torch.no_grad()
    def p_sample(
        self,
        model: Union[nn.Module, Callable],
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        **model_kwargs
    ) -> torch.Tensor:
        """
        Sample from p(x_{t-1} | x_t) using DPM-Solver++ (2M).
        """
        # 1. Get Model Prediction (Epsilon)
        predicted_noise = model(x_t, timesteps, **model_kwargs)

        # 2. Identify current t and prev t
        # We assume timesteps is a batch of the same integer t.
        t_curr = timesteps[0].item()
        t_prev = t_curr - 1
        
        # Handle the last step (t=0 -> output) separately or clamp
        if t_prev < 0:
            t_prev = 0 # Should ideally be returning x_0 directly here

        # 3. Get parameters for t and t-1
        # Note: We construct indices for broadcasting
        indices_curr = timesteps
        indices_prev = torch.full_like(timesteps, t_prev)

        lambda_curr = self.extract(self.lambda_t_schedule, indices_curr, x_t.shape)
        lambda_prev = self.extract(self.lambda_t_schedule, indices_prev, x_t.shape)
        
        sigma_curr = self.extract(self.sigma_t_schedule, indices_curr, x_t.shape)
        sigma_prev = self.extract(self.sigma_t_schedule, indices_prev, x_t.shape)
        
        alpha_prev = self.extract(self.alpha_t_schedule, indices_prev, x_t.shape)

        # h = lambda_{t-1} - lambda_t
        h = lambda_prev - lambda_curr
        
        # 4. DPM-Solver++ (2M) Update
        
        # Reset state if we are at the start of generation (highest timestep)
        if t_curr == self.timesteps - 1:
            self.reset_solver_state()

        phi_1 = torch.expm1(-h) # e^{-h} - 1

        if self.last_noise_prediction is None:
            # First-order step (Euler/DDIM-like) for the very first step
            # x_{t-1} = (sigma_{t-1}/sigma_t) * x_t - alpha_{t-1} * phi_1 * epsilon_t
            denoised_update = predicted_noise
        else:
            # Second-order step (DPM-Solver++ 2M)
            # For uniform steps, r = 1, so D = 1.5 * eps_t - 0.5 * eps_{t+1}
            denoised_update = (1.5 * predicted_noise) - (0.5 * self.last_noise_prediction)

        x_tm1 = (
            (sigma_prev / sigma_curr) * x_t 
            - (alpha_prev * phi_1) * denoised_update
        )

        # Update state
        self.last_noise_prediction = predicted_noise

        return x_tm1