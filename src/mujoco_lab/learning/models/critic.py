"""Independent state-value network for PPO."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ValueCritic(nn.Module):
    """Predict one scalar value per state with a randomly initialized MLP."""

    def __init__(
        self,
        ob_dim: int,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        """Create an independent value MLP and its Adam optimizer."""
        super().__init__()
        if ob_dim <= 0 or n_layers < 0 or layer_size <= 0:
            raise ValueError("ob_dim and layer_size must be positive; n_layers must be nonnegative")
        self.state_dim = ob_dim
        layers = []
        input_dim = ob_dim
        for _ in range(n_layers):
            layers.extend((nn.Linear(input_dim, layer_size), nn.ReLU()))
            input_dim = layer_size
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Map normalized state histories of shape (B, N_s) to values (B,)."""
        return self.network(obs).squeeze(-1)

    def update(self, obs: np.ndarray, q_values: np.ndarray) -> dict[str, float]:
        """Take one MSE regression step toward fixed return targets.

        ``obs`` must already use BC normalization and flattened observation
        history. ``q_values`` contains one return target per observation, with
        shape (B,) or (B, 1).
        """
        parameter = next(self.network.parameters())
        observations = torch.as_tensor(obs, dtype=parameter.dtype, device=parameter.device)
        targets = torch.as_tensor(q_values, dtype=parameter.dtype, device=parameter.device)
        if observations.ndim != 2 or observations.shape[1] != self.state_dim:
            raise ValueError("obs must have shape (B, ob_dim)")
        if targets.ndim == 2 and targets.shape[1] == 1:
            targets = targets.squeeze(-1)
        if targets.shape != (observations.shape[0],):
            raise ValueError("q_values must have shape (B,) or (B, 1)")
        loss = F.mse_loss(self(observations), targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"Baseline Loss": loss.item()}
