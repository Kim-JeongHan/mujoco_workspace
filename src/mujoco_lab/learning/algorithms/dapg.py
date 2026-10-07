"""Demonstration-augmented natural policy gradient."""

from dataclasses import dataclass

import torch

from mujoco_lab.learning.algorithms.npg import NPGConfig, npg_step
from mujoco_lab.learning.policies.gaussian import GaussianPolicy
from mujoco_lab.learning.rollout.buffer import RolloutBatch


@dataclass
class Demonstrations:
    """Fixed normalized expert states (N, S) and single-action chunks (N, 1, A)."""

    states: torch.Tensor
    actions: torch.Tensor


def dapg_step(
    policy: GaussianPolicy,
    batch: RolloutBatch,
    demonstrations: Demonstrations,
    *,
    iteration: int,
    config: NPGConfig,
    demo_weight: float = 0.1,
    demo_decay: float = 0.95,
    demo_batch_size: int = 0,
) -> dict[str, float]:
    """Add a decaying imitation gradient; use rollout states for the Fisher metric.

    Follow the public mjrl implementation's constant demo weight after rollout
    advantage whitening, rather than the paper's maximum-advantage heuristic.
    The demo term is weight * sum(log_prob(demos)) / rollout_count. An optional
    uniform demo subsample estimates this sum with the same total-data scaling.
    Set demo_weight to zero for ordinary NPG. The trainer owns the iteration.
    """
    weight = demo_weight * demo_decay**iteration
    count = demonstrations.states.shape[0]
    states, actions = demonstrations.states, demonstrations.actions
    if weight > 0 and 0 < demo_batch_size < count:
        indices = torch.randperm(count, device=states.device)[:demo_batch_size]
        states, actions = states[indices], actions[indices]

    def augmentation() -> torch.Tensor:
        return weight * count / batch.states.shape[0] * policy.log_prob(states, actions).mean()

    metrics = npg_step(policy, batch, config, augmentation=augmentation if weight > 0 else None)
    metrics["policy/demo_weight"] = weight
    return metrics
