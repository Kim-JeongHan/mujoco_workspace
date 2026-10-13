"""PPO actor-critic initialization and optimizer ownership."""

import torch

from mujoco_lab.learning.agents.base import OnPolicyAgent
from mujoco_lab.learning.config.critic import CriticConfig
from mujoco_lab.learning.models.critic import ValueCritic
from mujoco_lab.learning.policies.gaussian import GaussianPolicy
from mujoco_lab.learning.rollout.buffer import RolloutBatch


class PPOAgent(OnPolicyAgent):
    """Initialize PPO learning state; the PPO policy update is still pending."""

    def __init__(
        self,
        policy: GaussianPolicy,
        critic: ValueCritic,
        critic_config: CriticConfig,
        *,
        actor_learning_rate: float = 3e-4,
    ) -> None:
        super().__init__(policy, critic, critic_config)
        self.actor_optimizer = torch.optim.Adam(policy.parameters(), lr=actor_learning_rate)

    def update_policy(self, batch: RolloutBatch) -> dict[str, float]:
        raise NotImplementedError("PPO policy updates are not implemented yet")
