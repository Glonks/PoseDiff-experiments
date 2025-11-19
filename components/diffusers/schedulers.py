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


def log_snr_schedule() -> torch.Tensor:
    raise NotImplementedError()
