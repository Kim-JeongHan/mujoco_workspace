"""Value model definition (temporal encoder network).

Implements J_φ from Janner et al. (2022) "Planning with Diffusion for Flexible
Behavior Synthesis" — a differentiable trajectory value estimator used to guide
the reverse diffusion sampling process toward high-return trajectories.
"""

from __future__ import annotations

import torch
from torch import nn

from .nn import TemporalValueNet

_DEFAULT_DIM = 32
_DEFAULT_DIM_MULTS = (1, 2, 4, 8)


class ValueModel(nn.Module):
    """Trajectory value estimator J_φ.

    Encodes a full trajectory ``[B, H, state_dim]`` and produces a scalar
    value ``[B, 1]``.  During guided diffusion sampling (Algorithm 1 in the
    paper), its gradient w.r.t. the trajectory is used to bias the reverse
    process mean toward high-value regions.

    The model does not take an explicit conditioning vector; goal conditioning
    is handled through inpainting (fixing the first/last trajectory states
    during sampling).

    Args:
        state_dim: Dimension of each state.
        horizon: Planning horizon (number of timesteps per trajectory).
        dim: Base hidden-channel dimension (default 32).
        dim_mults: Channel multipliers at each U-Net level.
        **_: Accepts and silently ignores legacy keyword arguments
            (``n_hidden``, ``n_layers``, ``condition_dim``) to ease migration.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        horizon: int,
        dim: int = _DEFAULT_DIM,
        dim_mults: tuple[int, ...] = _DEFAULT_DIM_MULTS,
        **_: object,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.horizon = int(horizon)
        self.dim = int(dim)
        self.dim_mults = tuple(dim_mults)

        self.net = TemporalValueNet(
            transition_dim=self.state_dim,
            dim=self.dim,
            dim_mults=self.dim_mults,
        )

        self.hparams = {
            "state_dim": self.state_dim,
            "horizon": self.horizon,
            "dim": self.dim,
            "dim_mults": list(self.dim_mults),
            "model_class_path": "mujoco_lab.learning.models.diffusion.ValueModel",
        }

    def forward(
        self,
        x: torch.Tensor,
        condition: object = None,  # accepted but ignored
    ) -> torch.Tensor:
        """Estimate the value of a trajectory.

        Args:
            x: Trajectory tensor ``[B, H, state_dim]``.
            condition: Accepted for API compatibility; ignored.

        Returns:
            Scalar value per trajectory ``[B, 1]``.
        """
        if x.ndim < 3:
            raise ValueError("x must have shape [B, H, state_dim].")
        return self.net(x)

    # ------------------------------------------------------------------
    # Checkpoint interface
    # ------------------------------------------------------------------

    def load_state_dict(  # type: ignore[override]
        self,
        state_dict: dict[str, object],
        strict: bool = True,
    ) -> tuple[list[str], list[str]]:
        incompatible = super().load_state_dict(state_dict, strict=strict)  # type: ignore[arg-type]
        return incompatible.missing_keys, incompatible.unexpected_keys

    def state_dict(self) -> dict[str, object]:  # type: ignore[override]
        return super().state_dict()

    def set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)


__all__ = ["ValueModel"]
