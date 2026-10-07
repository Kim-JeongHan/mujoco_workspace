"""On-policy action ownership, history resets, and final-observation bootstrapping."""

import numpy as np
import pytest
import torch
from gymnasium import spaces

from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.models.critic import ValueCritic
from mujoco_lab.learning.policies.gaussian import GaussianPolicy
from mujoco_lab.learning.rollout.buffer import RolloutBuffer
from mujoco_lab.learning.rollout.online import collect_rollout, evaluate_policy


class HistoryEnv:
    def __init__(self, *, terminal=False):
        self.action_space = spaces.Box(-1, 1, (1,), dtype=np.float32)
        self.terminal = terminal
        self.resets = 0
        self.actions = []

    def reset(self, *, seed=None):
        self.resets += 1
        self.steps = 0
        return np.array([10 * self.resets], dtype=np.float32), {}

    def step(self, action):
        self.actions.append(action.copy())
        self.steps += 1
        done = self.steps == 2
        return (
            np.array([10 * self.resets + self.steps], dtype=np.float32),
            1.0,
            done and self.terminal,
            done and not self.terminal,
            {"success": done and self.terminal},
        )


def setup_policy():
    policy = GaussianPolicy(2, 1, 1, hidden_dims=())
    critic = ValueCritic(2, 0, 8, 1e-3)
    with torch.no_grad():
        policy.mean_head.weight.zero_()
        policy.mean_head.bias.fill_(5)
        policy.log_std.fill_(-2)
        critic.network[0].weight.fill_(1)
        critic.network[0].bias.zero_()
    normalizer = Normalizer(
        np.array([10], dtype=np.float32),
        np.array([2], dtype=np.float32),
        np.array([3], dtype=np.float32),
        np.array([2], dtype=np.float32),
    )
    return policy, critic, normalizer


@pytest.mark.parametrize("terminal", [False, True])
def test_collection_preserves_raw_actions_history_and_boundary_values(terminal):
    torch.manual_seed(7)
    policy, critic, normalizer = setup_policy()
    env = HistoryEnv(terminal=terminal)
    buffer = RolloutBuffer(3, 2, (1, 1))
    metrics = collect_rollout(
        env, policy, critic, normalizer, buffer, steps=3, obs_horizon=2, seed=3
    )
    torch.testing.assert_close(
        buffer.states[:, 0], torch.tensor([[0.0, 0.0], [0.0, 0.5], [5.0, 5.0]])
    )
    torch.testing.assert_close(
        buffer.next_values[:, 0], torch.tensor([0.5, 0.0 if terminal else 1.5, 10.5])
    )
    assert buffer.truncated[-1, 0]
    assert buffer.terminated[1, 0].item() == terminal
    assert all(action[0] == 1 for action in env.actions)
    assert (buffer.actions > 4).all()
    torch.testing.assert_close(
        buffer.log_probs[:, 0], policy.log_prob(buffer.states[:, 0], buffer.actions[:, 0])
    )
    assert not buffer.actions.requires_grad
    assert metrics["rollout/completed_episodes"] == 1
    assert metrics["rollout/cutoff"] == 1


def test_evaluation_uses_mean_without_consuming_sampling_rng():
    policy, _, normalizer = setup_policy()
    env = HistoryEnv(terminal=True)
    rng = torch.get_rng_state().clone()
    metrics = evaluate_policy(
        env, policy, normalizer, obs_horizon=2, episodes=2, max_steps=4, seed=10
    )
    assert metrics == {"eval/success_rate": 1.0, "eval/mean_return": 2.0, "eval/mean_length": 2.0}
    assert all(action[0] == 1 for action in env.actions)
    torch.testing.assert_close(torch.get_rng_state(), rng)
