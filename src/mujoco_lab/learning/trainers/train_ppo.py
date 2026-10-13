"""Initialize a BC-derived PPO agent with an independent critic and optimizers."""

from pathlib import Path
from typing import Any

import tyro

from mujoco_lab.learning.agents.initialization import initialize_actor
from mujoco_lab.learning.agents.ppo import PPOAgent
from mujoco_lab.learning.config.critic import CriticConfig
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.models.critic import ValueCritic


def initialize_agent(
    init_from: Path,
    *,
    initial_log_std: float = -2.0,
    actor_learning_rate: float = 3e-4,
    critic_hidden_dims: tuple[int, ...] = (128, 128),
    critic_learning_rate: float = 3e-4,
    device: str = "cpu",
) -> tuple[PPOAgent, Normalizer, dict[str, Any]]:
    """Load a BC actor and return an agent owning both networks and optimizer state.

    PPO rollout execution and policy updates are still pending. The source BC
    observation, chunk, and normalization settings are retained at initialization.
    """
    actor, normalizer, metadata = initialize_actor(
        init_from, initial_log_std=initial_log_std, device=device
    )
    config = CriticConfig(hidden_dims=critic_hidden_dims, learning_rate=critic_learning_rate)
    critic = ValueCritic(actor.state_dim, config.hidden_dims, config.learning_rate, device=device)
    agent = PPOAgent(actor, critic, config, actor_learning_rate=actor_learning_rate)
    return agent, normalizer, metadata


def main() -> None:
    """Initialize PPO agent state; online PPO training is still pending."""
    agent, _, _ = tyro.cli(initialize_agent)
    print(
        f"Initialized PPO agent: state_dim={agent.policy.state_dim}, "
        f"action_dim={agent.policy.action_dim}, chunk_size={agent.policy.chunk_size}. "
        f"Initialized independent critic: state_dim={agent.critic.state_dim}. "
        "PPO training is not implemented yet."
    )


if __name__ == "__main__":
    main()
