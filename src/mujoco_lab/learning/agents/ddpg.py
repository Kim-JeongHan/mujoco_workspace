"""Deterministic policy gradients with target networks and replay minibatches."""

from copy import deepcopy
from typing import Any

import torch
from torch.nn import functional as F

from mujoco_lab.learning.agents.base import Agent
from mujoco_lab.learning.config.ddpg import DDPGConfig
from mujoco_lab.learning.models.critic import QCritic
from mujoco_lab.learning.policies.mse import MSEPolicy
from mujoco_lab.learning.rollout.replay_buffer import ReplayBatch


class DDPGAgent(Agent[ReplayBatch, MSEPolicy, QCritic]):
    """Own online and target networks, Adam state, and the gradient update count."""

    def __init__(self, policy: MSEPolicy, critic: QCritic, config: DDPGConfig) -> None:
        if policy.chunk_size != 1 or not policy.bounded:
            raise ValueError("DDPG requires a bounded single-action policy")
        super().__init__(policy, critic)
        self.config = config
        self.target_policy = deepcopy(policy).requires_grad_(False)
        self.target_critic = deepcopy(critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(policy.parameters(), lr=config.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=config.critic_learning_rate
        )

    def update(self, batch: ReplayBatch) -> dict[str, float]:
        """Apply one Bellman regression, actor gradient step, and soft target update."""
        with torch.no_grad():
            next_actions = self.target_policy(batch.next_states)[:, 0]
            targets = batch.rewards + self.config.gamma * (~batch.terminated) * self.target_critic(
                batch.next_states, next_actions
            )
        values = self.critic(batch.states, batch.actions)
        critic_loss = F.mse_loss(values, targets)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic weights while retaining the action derivative for the actor.
        self.critic.requires_grad_(False)
        try:
            actor_loss = -self.critic(batch.states, self.policy(batch.states)[:, 0]).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
        finally:
            self.critic.requires_grad_(True)
        with torch.no_grad():
            for target, source in (
                (self.target_policy, self.policy),
                (self.target_critic, self.critic),
            ):
                for target_parameter, parameter in zip(
                    target.parameters(), source.parameters(), strict=True
                ):
                    target_parameter.lerp_(parameter, self.config.tau)
        self.iteration += 1
        return {
            "policy/loss": actor_loss.item(),
            "critic/loss": critic_loss.item(),
            "critic/mean_q": values.detach().mean().item(),
            "critic/mean_target": targets.mean().item(),
            "replay/her_fraction": batch.relabeled.float().mean().item(),
            "replay/goal_success_rate": (batch.rewards == 0).float().mean().item(),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            **super().state_dict(),
            "target_actor": self.target_policy.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.target_policy.load_state_dict(state["target_actor"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
