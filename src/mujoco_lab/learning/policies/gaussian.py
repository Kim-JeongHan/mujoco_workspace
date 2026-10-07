"""Gaussian action chunks with a shared MLP backbone and a separate mean head."""

from __future__ import annotations

import torch
from torch import nn

from mujoco_lab.learning.infrastructure.utils import build_mlp
from mujoco_lab.learning.policies.base import BasePolicy


class GaussianPolicy(BasePolicy):
    """Predict Gaussian distributions over chunks of actions."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        """Initialize the state-conditioned Gaussian policy.

        Args:
            state_dim: Number of state features, N_s.
            action_dim: Number of action features per step, N_a.
            chunk_size: Number of actions in each predicted chunk, C.
            hidden_dims: Width of each hidden MLP layer.
        """
        super().__init__(state_dim, action_dim, chunk_size)

        mean_network = build_mlp(
            input_dim=state_dim,
            output_dim=action_dim * chunk_size,
            hidden_layers=hidden_dims,
        )
        # Retain the backbone/head names used by BC-to-PPO weight transfer.
        self.net = mean_network[:-1]
        self.mean_head = mean_network[-1]

        self.log_std = nn.Parameter(torch.zeros(chunk_size, action_dim))

    def forward(self, state: torch.Tensor) -> torch.distributions.Normal:
        """Return the Gaussian distribution conditioned on a batch of states.

        Args:
            state (batch, dim): B N_s

        Returns:
            torch.distributions.Normal: Distribution with mean and standard
                deviation broadcastable to shape (B, C, N_a).
        """
        batch = state.shape[0]

        h = self.net(state)

        mean = self.mean_head(h)
        mean = mean.view(batch, self.chunk_size, self.action_dim)

        std = self.log_std.exp()
        return torch.distributions.Normal(mean, std)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Sample action chunks of shape (B, C, N_a) with gradient support.

        ``num_steps`` is unused and retained for the shared policy interface.
        """
        return self(state).rsample()

    def log_prob(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return one joint log probability per sampled action chunk (B,)."""
        return self(state).log_prob(actions).sum(dim=(-1, -2))

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """Return joint action entropy per state (B,)."""
        return self(state).entropy().sum(dim=(-1, -2))

    def kl_from(
        self, state: torch.Tensor, old_mean: torch.Tensor, old_std: torch.Tensor
    ) -> torch.Tensor:
        """Return KL(old || current) per state using a detached distribution snapshot."""
        current = self(state)
        old_mean, old_std = old_mean.detach(), old_std.detach()
        kl = (
            current.scale.log()
            - old_std.log()
            + (old_std.square() + (old_mean - current.loc).square()) / (2 * current.scale.square())
            - 0.5
        )
        return kl.sum(dim=(-1, -2))

    def compute_loss(self, state, action_chunk):
        """Compute the negative log likelihood of expert action chunks.

        Args:
            state (batch, dim): B N_s
            action_chunk (batch, chunk, dim): B C N_a

        Returns:
            torch.Tensor: Mean negative log likelihood with shape ().
        """
        return -self.log_prob(state, action_chunk).mean()
