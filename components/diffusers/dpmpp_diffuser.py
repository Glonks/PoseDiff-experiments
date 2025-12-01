import torch
import torch.nn as nn
from typing import Optional, Union, Callable

# -----------------------------
# Log-SNR schedule (EDM style)
# -----------------------------
def log_snr_schedule_edm(timesteps: int,
                         sigma_min: float = 0.002,
                         sigma_max: float = 80.0) -> torch.Tensor:
    """
    Returns a log-SNR schedule of shape [timesteps].
    Compatible with DPM++ and EDM-style solvers.
    """
    t = torch.linspace(0, 1, timesteps)
    sigma_t = sigma_min * (sigma_max / sigma_min) ** t
    log_snr = torch.log(1.0 / (sigma_t ** 2 + 1e-12))
    return log_snr


# -----------------------------
# DPM++ Pseudo-Probability Diffuser
# -----------------------------
class DPMPPDiffuser(nn.Module):
    def __init__(self, timesteps: int):
        super().__init__()
        self.timesteps = timesteps

        # --- schedules ---
        log_snr = log_snr_schedule_edm(timesteps)
        alphas_cumprod = torch.sigmoid(log_snr)

        self.alpha_t = torch.sqrt(alphas_cumprod)
        self.sigma_t = torch.sqrt(1.0 - alphas_cumprod)
        self.lambda_t = log_snr  # log-SNR = log(alpha^2 / sigma^2)
        print(self.sigma_t)

        # register buffers
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alpha_t_schedule", self.alpha_t)
        self.register_buffer("sigma_t_schedule", self.sigma_t)
        self.register_buffer("lambda_t_schedule", self.lambda_t)

        # for 2nd-order solver
        self.last_noise_prediction: Optional[torch.Tensor] = None

    @staticmethod
    def extract(tensor: torch.Tensor, indices: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        """Extract values and reshape for broadcasting."""
        batch_size = shape[0]
        out = tensor[indices].reshape(batch_size, *([1] * (len(shape) - 1)))
        return out

    # -----------------------------
    # Forward diffusion (q_sample)
    # -----------------------------
    def q_sample(self, x_0: torch.Tensor, timesteps: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_0)

        alphas_cumprod_t = self.extract(self.alphas_cumprod, timesteps, x_0.shape)
        x_t = torch.sqrt(alphas_cumprod_t) * x_0 + torch.sqrt(1.0 - alphas_cumprod_t) * noise
        return x_t

    # -----------------------------
    # Reset solver state
    # -----------------------------
    def reset_solver_state(self):
        self.last_noise_prediction = None

    # -----------------------------
    # DPM++ 2M sampling
    # -----------------------------
    @torch.no_grad()
    def p_sample(
        self,
        model: Union[nn.Module, Callable],
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        **model_kwargs
    ) -> torch.Tensor:
        """
        Sample x_{t-1} from x_t using DPM-Solver++ 2M.
        """
        predicted_noise = model(x_t, timesteps, **model_kwargs)

        t_curr = timesteps[0].item()
        t_prev = max(t_curr - 1, 0)

        # Extract lambda, alpha, sigma
        indices_curr = timesteps
        indices_prev = torch.full_like(timesteps, t_prev)

        lambda_curr = self.extract(self.lambda_t_schedule, indices_curr, x_t.shape)
        lambda_prev = self.extract(self.lambda_t_schedule, indices_prev, x_t.shape)

        sigma_curr = self.extract(self.sigma_t_schedule, indices_curr, x_t.shape)
        sigma_prev = self.extract(self.sigma_t_schedule, indices_prev, x_t.shape)
        alpha_prev = self.extract(self.alpha_t_schedule, indices_prev, x_t.shape)

        # Log-SNR step
        h = lambda_prev - lambda_curr
        phi_1 = torch.expm1(-h)

        # Reset solver if starting new trajectory
        if t_curr == self.timesteps - 1:
            self.reset_solver_state()

        # 2nd-order multistep
        if self.last_noise_prediction is None:
            denoised_update = predicted_noise
        else:
            denoised_update = 1.5 * predicted_noise - 0.5 * self.last_noise_prediction

        # DPM++ update
        x_tm1 = (sigma_prev / sigma_curr) * x_t - alpha_prev * phi_1 * denoised_update

        # Store noise for next step
        self.last_noise_prediction = predicted_noise

        return x_tm1
