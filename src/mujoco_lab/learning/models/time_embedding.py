"""Fixed sinusoidal features for continuous flow time."""

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Embed unit-interval time using periods from 0.004 to 4.0."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 or dim % 2:
            raise ValueError("time embedding dimension must be a positive even integer")
        periods = torch.logspace(math.log10(0.004), math.log10(4.0), dim // 2)
        self.register_buffer("frequencies", 2 * math.pi / periods, persistent=False)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """Return sine and cosine features for a batch of scalar times."""
        if tau.ndim != 1:
            raise ValueError("tau must have shape (batch,)")
        phases = tau[:, None] * self.frequencies.to(device=tau.device, dtype=tau.dtype)
        return torch.cat((phases.sin(), phases.cos()), dim=-1)
