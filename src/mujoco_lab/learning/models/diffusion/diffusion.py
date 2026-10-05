"""Diffusion model definition (temporal U-Net).

Implements the ε-parameterized denoising model from Janner et al. (2022)
"Planning with Diffusion for Flexible Behavior Synthesis".
"""

from __future__ import annotations

import torch
from torch import nn

from .nn import TemporalUnet


class DiffusionModel(nn.Module):
    """Temporal U-Net that predicts the denoising noise ε for a trajectory.

    Replaces the original MLP backbone with the temporally-local 1D convolutional
    U-Net described in the paper.  Conditions on the diffusion timestep ``t`` via
    sinusoidal embeddings injected into every residual block.

    Args:
        state_dim: Dimension of each state (= ``transition_dim`` for the U-Net).
        horizon: Planning horizon (number of timesteps per trajectory).
        n_diffusion_steps: Total number of denoising steps.
        dim: Base hidden-channel dimension for the U-Net (default 32).
        dim_mults: Channel multipliers at each U-Net level (default (1, 2, 4, 8)).
    """

    def __init__(
        self,
        *,
        state_dim: int,
        horizon: int,
        n_diffusion_steps: int = 100,
        dim: int = 32,
        dim_mults: tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.horizon = int(horizon)
        self.n_diffusion_steps = int(n_diffusion_steps)
        self.dim = int(dim)
        self.dim_mults = tuple(dim_mults)

        self.unet = TemporalUnet(
            transition_dim=self.state_dim,
            dim=self.dim,
            dim_mults=self.dim_mults,
        )

        self.hparams = {
            "state_dim": self.state_dim,
            "horizon": self.horizon,
            "n_diffusion_steps": self.n_diffusion_steps,
            "dim": self.dim,
            "dim_mults": list(self.dim_mults),
            "model_class_path": "mujoco_lab.learning.models.diffusion.DiffusionModel",
        }

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise ε for a noisy trajectory at diffusion step t.

        Args:
            x: Noisy trajectory ``[B, H, state_dim]``.
            t: Diffusion timesteps ``[B]`` (integer).

        Returns:
            Predicted noise ``[B, H, state_dim]``.
        """
        if x.ndim < 3:
            raise ValueError("x must have shape [B, H, state_dim].")
        if t.ndim == 0:
            t = t.reshape(1).expand(x.shape[0])
        elif t.ndim == 1 and t.shape[0] == 1 and x.shape[0] != 1:
            t = t.expand(x.shape[0])
        return self.unet(x, t)

    def set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)


__all__ = ["DiffusionModel"]
