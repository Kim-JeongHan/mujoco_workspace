"""Deterministic action-chunk regression using the shared MLP builder."""

from __future__ import annotations

import torch
from torch import nn

from mujoco_lab.learning.infrastructure.utils import build_mlp
from mujoco_lab.learning.policies.base import BasePolicy


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        """Initialize the deterministic state-to-action MLP.

        Args:
            state_dim: Number of state features, N_s.
            action_dim: Number of action features per step, N_a.
            chunk_size: Number of actions in each predicted chunk, C.
            hidden_dims: Width of each hidden MLP layer.
        """
        super().__init__(state_dim, action_dim, chunk_size)

        self.net = build_mlp(
            input_dim=state_dim,
            output_dim=action_dim * chunk_size,
            hidden_layers=hidden_dims,
        )
        self.criterion = nn.MSELoss()

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the mean squared error against expert action chunks.

        Args:
            state (batch, dim): B N_s
            action_chunk (batch, chunk, dim): B C N_a

        Returns:
            torch.Tensor: Scalar mean squared error with shape ().
        """
        return self.criterion(action_chunk, self.sample_actions(state))

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Predict an action chunk; ``num_steps`` is unused by this policy.

        Args:
            state (batch, dim): B N_s
            num_steps: Unused; present to match the shared policy interface.

        Returns:
            torch.Tensor: Predicted action chunk with shape (B, C, N_a).
        """
        return self(state)

    def forward(self, state):
        """Map a batch of states to deterministic action chunks.

        Args:
            state (batch, dim): B N_s

        Returns:
            torch.Tensor: Predicted action chunk with shape (B, C, N_a).
        """
        action_chunk = self.net(state)
        action_chunk = action_chunk.view(-1, self.chunk_size, self.action_dim)
        return action_chunk
