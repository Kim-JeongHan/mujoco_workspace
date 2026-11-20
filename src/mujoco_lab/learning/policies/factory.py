"""Construct learning policies from their architecture settings."""

from mujoco_lab.learning.config.bc import (
    BCPolicyTrainConfig,
    DiffusionBCTrainConfig,
    FlowBCTrainConfig,
    MSEBCTrainConfig,
)
from mujoco_lab.learning.policies.base import BasePolicy
from mujoco_lab.learning.policies.diffusion import DiffusionPolicy
from mujoco_lab.learning.policies.flow import FlowMatchingPolicy
from mujoco_lab.learning.policies.mse import MSEPolicy


def build_policy(
    config: BCPolicyTrainConfig,
    *,
    state_dim: int,
    action_dim: int,
) -> BasePolicy:
    """Build a policy with the requested architecture and dimensions.

    Args:
        config: Policy-specific BC settings from training or a checkpoint.
        state_dim: Number of state features, N_s.
        action_dim: Number of action features per step, N_a.

    Returns:
        BasePolicy: Configured policy instance.

    Raises:
        ValueError: If the BC config type is not supported.
    """
    if isinstance(config, MSEBCTrainConfig):
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=config.chunk_size,
            hidden_dims=config.hidden_dims,
        )
    elif isinstance(config, FlowBCTrainConfig):
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=config.chunk_size,
            hidden_dims=config.hidden_dims,
            time_embed_dim=config.time_embed_dim,
        )
    elif isinstance(config, DiffusionBCTrainConfig):
        return DiffusionPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=config.chunk_size,
            down_dims=config.down_dims,
            time_embed_dim=config.time_embed_dim,
            scheduler_config=config.scheduler,
        )

    raise ValueError(f"Unsupported BC config: {type(config).__name__}")
