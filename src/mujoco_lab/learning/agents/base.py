"""Actor-critic ownership, update lifecycle, and checkpoint state."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from mujoco_lab.learning.config.critic import CriticConfig
from mujoco_lab.learning.models.critic import ValueCritic
from mujoco_lab.learning.policies.base import BasePolicy
from mujoco_lab.learning.policies.gaussian import GaussianPolicy
from mujoco_lab.learning.rollout.buffer import RolloutBatch, RolloutBuffer


class Agent[Update, Policy: BasePolicy, Critic: torch.nn.Module](ABC):
    """Own networks, optimizer state, and completed learning iterations."""

    def __init__(self, policy: Policy, critic: Critic) -> None:
        self.policy = policy
        self.critic = critic
        self.iteration = 0

    @abstractmethod
    def update(self, batch: Update, /) -> dict[str, float]:
        """Update learned state from the algorithm's training data."""

    def state_dict(self) -> dict[str, Any]:
        """Serialize networks and the completed update count."""
        return {
            "actor": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "iteration": self.iteration,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore networks and the completed update count."""
        self.policy.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.iteration = state["iteration"]


class OnPolicyAgent(Agent[RolloutBuffer, GaussianPolicy, ValueCritic]):
    """Fit a state-value baseline after a policy update on the current rollout."""

    def __init__(
        self, policy: GaussianPolicy, critic: ValueCritic, critic_config: CriticConfig
    ) -> None:
        super().__init__(policy, critic)
        self.critic_config = critic_config
        self.actor_optimizer: torch.optim.Optimizer | None = None

    def update(self, rollout: RolloutBuffer) -> dict[str, float]:
        """Update the policy once, fit the critic, and advance the learning iteration."""
        count = len(rollout) * rollout.rewards.shape[1]
        batch = next(rollout.minibatches(count))
        metrics = self.update_policy(batch)
        losses = []
        for _ in range(self.critic_config.num_epochs):
            for minibatch in rollout.minibatches(self.critic_config.batch_size):
                losses.append(
                    self.critic.update(minibatch.states, minibatch.returns)["Baseline Loss"]
                )
        metrics["critic/loss"] = float(np.mean(losses))
        self.iteration += 1
        return metrics

    @abstractmethod
    def update_policy(self, batch: RolloutBatch) -> dict[str, float]:
        """Apply the agent's policy update to a rollout collected by its current policy."""

    def state_dict(self) -> dict[str, Any]:
        """Serialize learned state using the existing actor-critic checkpoint keys."""
        state = {
            **super().state_dict(),
            "critic_optimizer": self.critic.optimizer.state_dict(),
        }
        if self.actor_optimizer is not None:
            state["actor_optimizer"] = self.actor_optimizer.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore network weights, optimizer history, and the learning iteration."""
        super().load_state_dict(state)
        self.critic.optimizer.load_state_dict(state["critic_optimizer"])
        if self.actor_optimizer is not None:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
