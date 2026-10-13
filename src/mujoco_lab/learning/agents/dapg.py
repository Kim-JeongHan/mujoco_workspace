"""Demonstration-augmented natural-gradient agent."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from mujoco_lab.learning.agents.npg import NPGAgent
from mujoco_lab.learning.config.dapg import DAPGConfig
from mujoco_lab.learning.models.critic import ValueCritic
from mujoco_lab.learning.policies.gaussian import GaussianPolicy
from mujoco_lab.learning.rollout.buffer import RolloutBatch


@dataclass
class Demonstrations:
    """Fixed normalized expert states (N, S) and single-action chunks (N, 1, A)."""

    states: torch.Tensor
    actions: torch.Tensor


class DAPGAgent(NPGAgent):
    """Own demonstrations and their decaying contribution to natural policy updates."""

    def __init__(self, policy: GaussianPolicy, critic: ValueCritic, config: DAPGConfig) -> None:
        super().__init__(policy, critic, config.npg, critic_config=config.critic)
        self.config = config
        self.demonstrations: Demonstrations | None = None

    def update_policy(
        self,
        batch: RolloutBatch,
        *,
        augmentation: Callable[[], torch.Tensor] | None = None,
    ) -> dict[str, float]:
        """Add a decaying imitation gradient using the agent's learning iteration.

        Follow the public mjrl implementation's constant demo weight after rollout
        advantage whitening, rather than the paper's maximum-advantage heuristic.
        The demo term is weight * sum(log_prob(demos)) / rollout_count. An optional
        uniform demo subsample estimates this sum with the same total-data scaling.
        Fisher products and KL use rollout states. Zero demo weight gives ordinary NPG.
        """
        config = self.config
        weight = config.demo_weight * config.demo_decay**self.iteration
        objective = augmentation
        if weight > 0:
            demonstrations = self.demonstrations
            if demonstrations is None:
                raise RuntimeError("Load demonstrations before updating a DAPG agent")
            count = demonstrations.states.shape[0]
            states, actions = demonstrations.states, demonstrations.actions
            if 0 < config.demo_batch_size < count:
                indices = torch.randperm(count, device=states.device)[: config.demo_batch_size]
                states, actions = states[indices], actions[indices]

            def demo_objective() -> torch.Tensor:
                result = (
                    weight
                    * count
                    / batch.states.shape[0]
                    * self.policy.log_prob(states, actions).mean()
                )
                return result if augmentation is None else result + augmentation()

            objective = demo_objective
        metrics = super().update_policy(batch, augmentation=objective)
        metrics["policy/demo_weight"] = weight
        return metrics
