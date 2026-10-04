"""Initialize a BC-derived Gaussian PPO actor and an independent critic."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import tyro

from mujoco_lab.learning.checkpoint import load_checkpoint
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.models.critic import ValueCritic
from mujoco_lab.learning.policies.gaussian import GaussianPolicy
from mujoco_lab.learning.policies.mse import MSEPolicy


def initialize_actor(
    init_from: Path,
    *,
    initial_log_std: float = -2.0,
    device: str = "cpu",
) -> tuple[GaussianPolicy, Normalizer, dict[str, Any]]:
    """Copy BC weights into the Gaussian mean and retain BC normalization.

    The actor preserves BC observation and chunk dimensions. Returned metadata
    describes the source BC checkpoint, not a resumable PPO training state.
    ``initial_log_std`` sets exploration in normalized action coordinates.
    """
    if not math.isfinite(initial_log_std):
        raise ValueError("initial_log_std must be finite")
    bc_policy, normalizer, metadata = load_checkpoint(init_from)
    if not isinstance(bc_policy, MSEPolicy):
        raise ValueError("PPO actor initialization requires an MSE BC checkpoint")

    actor = GaussianPolicy(
        state_dim=bc_policy.state_dim,
        action_dim=bc_policy.action_dim,
        chunk_size=bc_policy.chunk_size,
        hidden_dims=tuple(metadata["architecture"]["hidden_dims"]),
    )
    actor.net.load_state_dict(bc_policy.net[:-1].state_dict())
    actor.mean_head.load_state_dict(bc_policy.net[-1].state_dict())
    with torch.no_grad():
        actor.log_std.fill_(initial_log_std)
    actor.to(device)
    actor.train()

    return actor, normalizer, metadata


def initialize_actor_critic(
    init_from: Path,
    *,
    initial_log_std: float = -2.0,
    critic_n_layers: int = 2,
    critic_layer_size: int = 128,
    critic_learning_rate: float = 3e-4,
    device: str = "cpu",
) -> tuple[GaussianPolicy, ValueCritic, Normalizer, dict[str, Any]]:
    """Load the BC actor and randomly initialize a separate state-value critic.

    Both networks consume the same normalized observation history. The critic
    has its own architecture and Adam optimizer and shares no actor parameters.
    """
    actor, normalizer, metadata = initialize_actor(
        init_from, initial_log_std=initial_log_std, device=device
    )
    critic = ValueCritic(
        ob_dim=actor.state_dim,
        n_layers=critic_n_layers,
        layer_size=critic_layer_size,
        learning_rate=critic_learning_rate,
        device=device,
    )

    # TODO: Connect a single CubeStackEnv using BC cadence, normalization, and observation history.
    # TODO: Define single-action PPO or action-chunk execution and log-probability semantics.
    # TODO: Initialize the actor optimizer; resume PPO with both optimizer states.
    # TODO: Collect rollouts, compute advantages, and run PPO updates.
    # TODO: Add evaluation, logging, checkpoint saving, and an environment-step budget.
    return actor, critic, normalizer, metadata


def main() -> None:
    """Initialize actor and critic only; online PPO training is still pending."""
    actor, critic, _, _ = tyro.cli(initialize_actor_critic)
    print(
        f"Initialized Gaussian actor: state_dim={actor.state_dim}, "
        f"action_dim={actor.action_dim}, chunk_size={actor.chunk_size}. "
        f"Initialized independent critic: state_dim={critic.state_dim}. "
        "PPO training is not implemented yet."
    )


if __name__ == "__main__":
    main()
