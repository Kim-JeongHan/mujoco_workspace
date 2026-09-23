"""Conditional flow matching policy with fixed time features."""

from __future__ import annotations

import torch
from torch import nn

from mujoco_lab.learning.models import SinusoidalTimeEmbedding
from mujoco_lab.learning.policies.base import BasePolicy


class FlowMatchingPolicy(BasePolicy):
    """Predict action chunks with conditional flow matching."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
        time_embed_dim: int | None = 128,
    ) -> None:
        """Initialize the state-conditioned velocity model.

        Args:
            state_dim: Number of state features, N_s.
            action_dim: Number of action features per step, N_a.
            chunk_size: Number of actions in each predicted chunk, C.
            hidden_dims: Width of each hidden MLP layer.
            time_embed_dim: Sinusoidal feature width, or None for legacy scalar time.
        """
        super().__init__(state_dim, action_dim, chunk_size)

        self.time_embed_dim = time_embed_dim
        self.time_embedding = (
            SinusoidalTimeEmbedding(time_embed_dim) if time_embed_dim is not None else None
        )
        layers = []

        input_dim = state_dim + action_dim * chunk_size + (time_embed_dim or 1)

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())

            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim * chunk_size))

        self.net = nn.Sequential(*layers)
        self.criterion = nn.MSELoss()

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the conditional flow matching loss for expert action chunks.

        Args:
            state (batch, dim): B N_s
            action_chunk (batch, chunk, dim): B C N_a

        Returns:
            torch.Tensor: Scalar velocity matching loss with shape ().
        """
        batch = action_chunk.shape[0]
        action_0 = torch.randn_like(action_chunk)
        tau = torch.rand(batch, 1, 1, device=action_chunk.device, dtype=action_chunk.dtype)
        action_t_tua = tau * action_chunk + (1 - tau) * action_0
        v_data = action_chunk - action_0

        return self.criterion(self(state, action_t_tua, tau), v_data)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Integrate the predicted velocity field to sample action chunks.

        Args:
            state (batch, dim): B N_s
            num_steps: Number of Euler integration steps.

        Returns:
            torch.Tensor: Sampled action chunk with shape (B, C, N_a).
        """
        batch = state.shape[0]
        noisy_action = torch.randn(
            batch, self.chunk_size, self.action_dim, device=state.device, dtype=state.dtype
        )
        for i in range(num_steps):
            tau = torch.full((batch, 1, 1), i / num_steps, device=state.device, dtype=state.dtype)
            veloicty = self(state, noisy_action, tau)
            noisy_action = noisy_action + veloicty / num_steps

        return noisy_action

    def forward(
        self, state: torch.Tensor, noisy_action: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        """Predict the velocity for noisy action chunks at flow time ``tau``.

        Args:
            state (batch, dim): B N_s
            noisy_action (batch, chunk, dim): B C N_a
            tau (batch, 1, 1): B 1 1 interpolation times in [0, 1].

        Returns:
            torch.Tensor: Predicted velocity with shape (B, C, N_a).
        """
        batch_size = noisy_action.shape[0]
        noisy_action = noisy_action.reshape(batch_size, -1)
        tau = tau.reshape(batch_size)
        time_features = (
            self.time_embedding(tau) if self.time_embedding is not None else tau[:, None]
        )

        x = torch.cat([state, noisy_action, time_features], -1)
        action_chunk = self.net(x)
        action_chunk = action_chunk.view(-1, self.chunk_size, self.action_dim)
        return action_chunk
