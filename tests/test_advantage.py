"""GAE estimates across episode boundaries and rollout cutoffs."""

import torch

from mujoco_lab.learning.algorithms.advantage import compute_gae


def test_gae_matches_discounted_terminal_returns():
    rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    values = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
    next_values = torch.tensor([1.0, 1.5, 100.0], dtype=torch.float64)
    terminated = torch.tensor([False, False, True])
    truncated = torch.zeros(3, dtype=torch.bool)

    advantages, returns = compute_gae(
        rewards, values, next_values, terminated, truncated, gamma=0.9, gae_lambda=1.0
    )

    expected_returns = torch.tensor([5.23, 4.7, 3.0], dtype=torch.float64)
    torch.testing.assert_close(returns, expected_returns)
    torch.testing.assert_close(advantages, expected_returns - values)


def test_gae_masks_each_environment_independently():
    rewards = torch.tensor([[1.0, 1.0], [2.0, 2.0], [100.0, 100.0]])
    values = torch.zeros_like(rewards)
    next_values = torch.tensor([[0.0, 0.0], [10.0, 10.0], [0.0, 0.0]])
    terminated = torch.tensor([[False, False], [True, False], [True, True]])
    truncated = torch.tensor([[False, False], [False, True], [False, False]])

    advantages, returns = compute_gae(
        rewards, values, next_values, terminated, truncated, gamma=0.9, gae_lambda=0.5
    )

    # Truncation bootstraps from the final observation; neither boundary lets
    # the next episode's reward of 100 propagate into the previous episode.
    expected = torch.tensor([[1.9, 5.95], [2.0, 11.0], [100.0, 100.0]])
    torch.testing.assert_close(advantages, expected)
    torch.testing.assert_close(returns, expected)


def test_gae_bootstraps_at_rollout_cutoff():
    rewards = torch.tensor([1.0, 2.0])
    values = torch.tensor([0.5, 1.0])
    next_values = torch.tensor([1.0, 3.0])
    flags = torch.zeros(2, dtype=torch.bool)

    advantages, returns = compute_gae(
        rewards, values, next_values, flags, flags, gamma=0.9, gae_lambda=0.5
    )

    torch.testing.assert_close(advantages, torch.tensor([3.065, 3.7]))
    torch.testing.assert_close(returns, torch.tensor([3.565, 4.7]))


def test_zero_lambda_returns_one_step_td_residuals():
    rewards = torch.tensor([1.0, 2.0])
    values = torch.tensor([0.5, 1.0])
    next_values = torch.tensor([1.0, 3.0])
    flags = torch.zeros(2, dtype=torch.bool)

    advantages, returns = compute_gae(
        rewards, values, next_values, flags, flags, gamma=0.9, gae_lambda=0.0
    )

    torch.testing.assert_close(advantages, torch.tensor([1.4, 3.7]))
    torch.testing.assert_close(returns, torch.tensor([1.9, 4.7]))


def test_terminal_takes_precedence_when_both_flags_are_set():
    rewards = torch.tensor([2.0])
    values = torch.tensor([0.5])
    next_values = torch.tensor([float("nan")])
    flags = torch.ones(1, dtype=torch.bool)

    advantages, returns = compute_gae(rewards, values, next_values, flags, flags)

    torch.testing.assert_close(advantages, torch.tensor([1.5]))
    torch.testing.assert_close(returns, rewards)


def test_gae_produces_fixed_targets_without_mutating_inputs():
    rewards = torch.tensor([1.0, 2.0], requires_grad=True)
    values = torch.tensor([0.5, 1.0], requires_grad=True)
    next_values = torch.tensor([1.0, 3.0], requires_grad=True)
    flags = torch.zeros(2, dtype=torch.bool)
    inputs = (rewards, values, next_values, flags)
    snapshots = [tensor.detach().clone() for tensor in inputs]

    advantages, returns = compute_gae(rewards, values, next_values, flags, flags)

    assert not advantages.requires_grad
    assert not returns.requires_grad
    for tensor, snapshot in zip(inputs, snapshots, strict=True):
        torch.testing.assert_close(tensor, snapshot)
