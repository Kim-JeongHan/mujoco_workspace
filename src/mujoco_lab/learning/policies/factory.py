"""Construct learning policies from their architecture settings."""

from typing import Literal

from mujoco_lab.learning.policies.base import BasePolicy
from mujoco_lab.learning.policies.flow import FlowMatchingPolicy
from mujoco_lab.learning.policies.gaussian import GaussianPolicy
from mujoco_lab.learning.policies.mse import MSEPolicy

type PolicyType = Literal["mse", "flow", "gaussian"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
    flow_time_embed_dim: int | None = 128,
) -> BasePolicy:
    """Build a policy with the requested architecture and dimensions.

    Args:
        policy_type: Policy architecture name: ``mse``, ``flow``, or
            ``gaussian``.
        state_dim: Number of state features, N_s.
        action_dim: Number of action features per step, N_a.
        chunk_size: Number of actions in each predicted chunk, C.
        hidden_dims: Width of each hidden MLP layer.
        flow_time_embed_dim: Sinusoidal time width for Flow, or None for scalar time.

    Returns:
        BasePolicy: Configured policy instance.

    Raises:
        ValueError: If ``policy_type`` is not supported.
    """
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    elif policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
            time_embed_dim=flow_time_embed_dim,
        )
    elif policy_type == "gaussian":
        return GaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )

    raise ValueError(f"Unknown policy type: {policy_type}")
