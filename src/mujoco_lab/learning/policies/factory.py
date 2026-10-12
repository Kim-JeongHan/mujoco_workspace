"""Construct learning policies from their architecture settings."""

from typing import Any, Literal

from mujoco_lab.learning.policies.base import BasePolicy
from mujoco_lab.learning.policies.diffusion import DiffusionPolicy
from mujoco_lab.learning.policies.flow import FlowMatchingPolicy
from mujoco_lab.learning.policies.gaussian import GaussianPolicy
from mujoco_lab.learning.policies.mse import MSEPolicy

type PolicyType = Literal["mse", "flow", "diffusion", "gaussian"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
    flow_time_embed_dim: int = 128,
    diffusion_down_dims: tuple[int, ...] = (128, 256, 512),
    diffusion_time_embed_dim: int = 256,
    diffusion_num_train_timesteps: int = 100,
    diffusion_beta_schedule: Literal["linear", "squaredcos_cap_v2"] = "squaredcos_cap_v2",
    diffusion_scheduler_config: dict[str, Any] | None = None,
) -> BasePolicy:
    """Build a policy with the requested architecture and dimensions.

    Args:
        policy_type: Policy architecture name: ``mse``, ``flow``, ``diffusion``,
            or ``gaussian``.
        state_dim: Number of state features, N_s.
        action_dim: Number of action features per step, N_a.
        chunk_size: Number of actions in each predicted chunk, C.
        hidden_dims: Width of each hidden MLP layer.
        flow_time_embed_dim: Sinusoidal time feature width for Flow.
        diffusion_down_dims: Temporal U-Net channel widths for Diffusion.
        diffusion_time_embed_dim: Diffusion timestep feature width.
        diffusion_num_train_timesteps: Number of training noise levels.
        diffusion_beta_schedule: Diffusion training noise schedule.
        diffusion_scheduler_config: Complete scheduler settings restored from a checkpoint.

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
    elif policy_type == "diffusion":
        return DiffusionPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            down_dims=diffusion_down_dims,
            time_embed_dim=diffusion_time_embed_dim,
            num_train_timesteps=diffusion_num_train_timesteps,
            beta_schedule=diffusion_beta_schedule,
            scheduler_config=diffusion_scheduler_config,
        )
    elif policy_type == "gaussian":
        return GaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )

    raise ValueError(f"Unknown policy type: {policy_type}")
