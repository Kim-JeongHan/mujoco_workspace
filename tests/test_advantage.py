"""Independent advantage and return estimates across rollout boundaries."""

import pytest
import torch

from mujoco_lab.learning.algorithms.advantage import compute_discounted_returns, compute_gae


def test_gae_matches_discounted_terminal_returns():
    rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    values = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
    next_values = torch.tensor([1.0, 1.5, 100.0], dtype=torch.float64)
    terminated = torch.tensor([False, False, True])
    truncated = torch.zeros(3, dtype=torch.bool)

    advantages = compute_gae(
        rewards, values, next_values, terminated, truncated, gamma=0.9, gae_lambda=1.0
    )

    expected_returns = torch.tensor([5.23, 4.7, 3.0], dtype=torch.float64)
    torch.testing.assert_close(advantages, expected_returns - values)


def test_gae_masks_each_environment_independently():
    rewards = torch.tensor([[1.0, 1.0], [2.0, 2.0], [100.0, 100.0]])
    values = torch.zeros_like(rewards)
    next_values = torch.tensor([[0.0, 0.0], [10.0, 10.0], [0.0, 0.0]])
    terminated = torch.tensor([[False, False], [True, False], [True, True]])
    truncated = torch.tensor([[False, False], [False, True], [False, False]])

    advantages = compute_gae(
        rewards, values, next_values, terminated, truncated, gamma=0.9, gae_lambda=0.5
    )

    # Truncation bootstraps from the final observation; neither boundary lets
    # the next episode's reward of 100 propagate into the previous episode.
    expected = torch.tensor([[1.9, 5.95], [2.0, 11.0], [100.0, 100.0]])
    torch.testing.assert_close(advantages, expected)


def test_gae_bootstraps_at_rollout_cutoff():
    rewards = torch.tensor([1.0, 2.0])
    values = torch.tensor([0.5, 1.0])
    next_values = torch.tensor([1.0, 3.0])
    flags = torch.zeros(2, dtype=torch.bool)

    advantages = compute_gae(rewards, values, next_values, flags, flags, gamma=0.9, gae_lambda=0.5)

    torch.testing.assert_close(advantages, torch.tensor([3.065, 3.7]))


def test_zero_lambda_returns_one_step_td_residuals():
    rewards = torch.tensor([1.0, 2.0])
    values = torch.tensor([0.5, 1.0])
    next_values = torch.tensor([1.0, 3.0])
    flags = torch.zeros(2, dtype=torch.bool)

    advantages = compute_gae(rewards, values, next_values, flags, flags, gamma=0.9, gae_lambda=0.0)

    torch.testing.assert_close(advantages, torch.tensor([1.4, 3.7]))


def test_terminal_takes_precedence_when_both_flags_are_set():
    rewards = torch.tensor([2.0])
    values = torch.tensor([0.5])
    next_values = torch.tensor([float("nan")])
    flags = torch.ones(1, dtype=torch.bool)

    advantages = compute_gae(rewards, values, next_values, flags, flags)

    torch.testing.assert_close(advantages, torch.tensor([1.5]))


def test_gae_produces_fixed_targets_without_mutating_inputs():
    rewards = torch.tensor([1.0, 2.0], requires_grad=True)
    values = torch.tensor([0.5, 1.0], requires_grad=True)
    next_values = torch.tensor([1.0, 3.0], requires_grad=True)
    flags = torch.zeros(2, dtype=torch.bool)
    inputs = (rewards, values, next_values, flags)
    snapshots = [tensor.detach().clone() for tensor in inputs]

    advantages = compute_gae(rewards, values, next_values, flags, flags)

    assert not advantages.requires_grad
    for tensor, snapshot in zip(inputs, snapshots, strict=True):
        torch.testing.assert_close(tensor, snapshot)


def test_discounted_returns_are_independent_of_gae_lambda_returns():
    rewards = torch.tensor([1.0, 1.0], dtype=torch.float64)
    values = torch.zeros_like(rewards)
    terminated = torch.tensor([False, True])
    truncated = torch.zeros(2, dtype=torch.bool)

    returns = compute_discounted_returns(rewards, terminated, truncated, gamma=0.9)
    advantages = compute_gae(
        rewards, values, values, terminated, truncated, gamma=0.9, gae_lambda=0.5
    )

    torch.testing.assert_close(returns, torch.tensor([1.9, 1.0], dtype=torch.float64))
    torch.testing.assert_close(advantages + values, torch.tensor([1.45, 1.0], dtype=torch.float64))


@pytest.mark.parametrize("bootstrap", [False, True])
def test_discounted_returns_stop_at_each_environment_boundary(bootstrap):
    rewards = torch.tensor([[1.0, 1.0], [2.0, 2.0], [100.0, 100.0]])
    terminated = torch.tensor([[False, False], [True, False], [True, True]])
    truncated = torch.tensor([[False, False], [False, True], [False, False]])
    next_values = torch.full_like(rewards, 10.0) if bootstrap else None

    returns = compute_discounted_returns(
        rewards, terminated, truncated, gamma=0.9, next_values=next_values
    )

    expected = torch.tensor(
        [[2.8, 10.9], [2.0, 11.0], [100.0, 100.0]]
        if bootstrap
        else [[2.8, 2.8], [2.0, 2.0], [100.0, 100.0]]
    )
    torch.testing.assert_close(returns, expected)


@pytest.mark.parametrize("bootstrap", [False, True])
def test_discounted_returns_bootstrap_at_rollout_cutoff_only_when_requested(bootstrap):
    rewards = torch.tensor([1.0, 2.0])
    flags = torch.zeros(2, dtype=torch.bool)
    next_values = torch.tensor([100.0, 3.0]) if bootstrap else None

    returns = compute_discounted_returns(rewards, flags, flags, gamma=0.9, next_values=next_values)

    # Interior next values do not affect the observed reward-to-go.
    expected = torch.tensor([5.23, 4.7] if bootstrap else [2.8, 2.0])
    torch.testing.assert_close(returns, expected)


def test_discounted_returns_ignore_bootstrap_when_both_boundary_flags_are_set():
    rewards = torch.tensor([2.0])
    flags = torch.ones(1, dtype=torch.bool)
    returns = compute_discounted_returns(
        rewards, flags, flags, next_values=torch.tensor([float("nan")])
    )
    torch.testing.assert_close(returns, rewards)


def test_discounted_returns_are_detached_without_mutating_inputs():
    rewards = torch.tensor([1.0, 2.0], requires_grad=True)
    next_values = torch.tensor([1.0, 3.0], requires_grad=True)
    flags = torch.zeros(2, dtype=torch.bool)
    inputs = (rewards, next_values, flags)
    snapshots = [tensor.detach().clone() for tensor in inputs]

    returns = compute_discounted_returns(rewards, flags, flags, next_values=next_values)

    assert not returns.requires_grad
    for tensor, snapshot in zip(inputs, snapshots, strict=True):
        torch.testing.assert_close(tensor, snapshot)
