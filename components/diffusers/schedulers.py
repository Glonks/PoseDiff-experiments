import torch


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine beta scheduling consistent with "Improved Denoising Diffusion Probabilistic Models".
    """
    x = torch.arange(timesteps + 1, dtype=torch.float64)

    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * (torch.pi / 2)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]

    betas = 1 - alphas
    betas = torch.clip(betas, 0, 0.999)

    return betas.to(torch.float32)

def log_snr_schedule_edm(timesteps: int,
                         sigma_min: float = 0.002, 
                         sigma_max: float = 80.0) -> torch.Tensor:
    """
    Creates a log-SNR schedule compatible with EDM and DPM++ samplers.
    t goes from 0 to 1.
    Returns log_snr[t] values of length `timesteps`.
    """

    # Normalized time steps t in [0, 1]
    t = torch.linspace(0, 1, timesteps)

    # Exponential interpolation of sigma (EDM schedule)
    sigma_t = sigma_min * (sigma_max / sigma_min) ** t

    # Convert to log SNR = log(alpha^2 / sigma^2)
    # In EDM, alpha(t) = 1 (no explicit alpha), so:
    log_snr = torch.log(1.0 / (sigma_t ** 2 + 1e-12))

    return log_snr
