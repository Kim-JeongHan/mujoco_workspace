"""Conditional action diffusion with noise-prediction training and DDIM sampling."""

from __future__ import annotations

from typing import Any, Literal

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torch.nn import functional as F  # noqa: N812

from mujoco_lab.learning.models.diffusion.conditional_unet import ConditionalTemporalUnet
from mujoco_lab.learning.policies.base import BasePolicy


class DiffusionPolicy(BasePolicy):
    """Generate normalized action chunks conditioned on flattened observation history.

    Observation and action normalization belong to the shared BC data pipeline.
    DDIM uses epsilon prediction and no sample clipping because standardized
    action coordinates are not bounded by [-1, 1]. Sampling starts from fresh
    Gaussian noise; subsequent DDIM steps use ``eta=0``.
    """

    alphas_cumprod: torch.Tensor

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        *,
        down_dims: tuple[int, ...] = (128, 256, 512),
        time_embed_dim: int = 256,
        num_train_timesteps: int = 100,
        beta_schedule: Literal["linear", "squaredcos_cap_v2"] = "squaredcos_cap_v2",
        scheduler_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.denoiser = ConditionalTemporalUnet(
            action_dim, state_dim, down_dims=down_dims, time_embed_dim=time_embed_dim
        )
        if scheduler_config is None:
            self.scheduler = DDIMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_schedule=beta_schedule,
                prediction_type="epsilon",
                clip_sample=False,
                timestep_spacing="leading",
            )
        else:
            # Diffusers lacks an overload for the default return_unused_kwargs=False.
            self.scheduler: DDIMScheduler = DDIMScheduler.from_config(  # ty: ignore[invalid-assignment]
                scheduler_config
            )
        self.num_train_timesteps: int = self.scheduler.config["num_train_timesteps"]
        if self.num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive")
        if (
            self.scheduler.config["prediction_type"] != "epsilon"
            or self.scheduler.config["clip_sample"]
        ):
            raise ValueError("DiffusionPolicy requires epsilon prediction without sample clipping")
        # Keep training schedule tensors on the policy device without scheduler mutation in loss.
        self.register_buffer("alphas_cumprod", self.scheduler.alphas_cumprod, persistent=False)

    def forward(
        self, state: torch.Tensor, noisy_actions: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Predict noise in normalized action coordinates at each diffusion timestep."""
        return self.denoiser(noisy_actions, timesteps, state)

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Match the Gaussian noise added to demonstration action chunks."""
        if action_chunk.shape != (state.shape[0], self.chunk_size, self.action_dim):
            raise ValueError("action_chunk must have shape (B, chunk_size, action_dim)")
        timesteps = torch.randint(
            self.num_train_timesteps,
            (state.shape[0],),
            device=action_chunk.device,
        )
        noise = torch.randn_like(action_chunk)
        alpha = self.alphas_cumprod[timesteps].reshape(-1, 1, 1)
        noisy_actions = alpha.sqrt() * action_chunk + (1 - alpha).sqrt() * noise
        return F.mse_loss(self(state, noisy_actions, timesteps), noise)

    @torch.no_grad()
    def sample_actions(self, state: torch.Tensor, *, num_steps: int = 20) -> torch.Tensor:
        """Denoise Gaussian noise with DDIM and return a normalized action chunk."""
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        self.scheduler.set_timesteps(num_steps, device=state.device)
        self.scheduler.alphas_cumprod = self.alphas_cumprod
        self.scheduler.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(
            device=state.device, dtype=state.dtype
        )
        actions = torch.randn(
            state.shape[0], self.chunk_size, self.action_dim, device=state.device, dtype=state.dtype
        )
        for timestep in self.scheduler.timesteps:
            noise = self(state, actions, timestep.expand(state.shape[0]))
            actions = self.scheduler.step(noise, timestep, actions, eta=0.0, return_dict=False)[0]
        return actions
