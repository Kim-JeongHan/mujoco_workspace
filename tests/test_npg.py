"""Natural-gradient numerical checks and demonstration-guided policy improvement."""

from copy import deepcopy

import pytest
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from mujoco_lab.learning.algorithms.dapg import Demonstrations, dapg_step
from mujoco_lab.learning.algorithms.npg import (
    NPGConfig,
    conjugate_gradient,
    fisher_vector_product,
    npg_step,
)
from mujoco_lab.learning.policies.gaussian import GaussianPolicy
from mujoco_lab.learning.rollout.buffer import RolloutBatch


def make_batch(policy, *, zero_advantages=False):
    states = torch.ones(64, 2, dtype=next(policy.parameters()).dtype)
    with torch.no_grad():
        actions = policy(states).sample()
        rewards = -(actions[:, 0, 0] - 1).square()
        return RolloutBatch(
            states,
            actions,
            torch.zeros_like(rewards),
            policy.log_prob(states, actions),
            torch.zeros_like(rewards) if zero_advantages else rewards,
            rewards,
        )


def make_policy():
    policy = GaussianPolicy(2, 1, 1, hidden_dims=()).double()
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
    return policy


def test_conjugate_gradient_solves_spd_system_and_zero_gradient():
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    rhs = torch.tensor([1.0, 2.0], dtype=torch.float64)
    result = conjugate_gradient(lambda vector: matrix @ vector, rhs)
    torch.testing.assert_close(result, torch.linalg.solve(matrix, rhs))
    zero = torch.zeros_like(rhs)
    torch.testing.assert_close(conjugate_gradient(lambda vector: matrix @ vector, zero), zero)


def test_gaussian_rl_probabilities_and_kl_match_torch_distributions():
    torch.manual_seed(1)
    policy = GaussianPolicy(2, 3, 2).double()
    states = torch.randn(4, 2, dtype=torch.float64)
    old = deepcopy(policy)(states)
    actions = old.sample()
    with torch.no_grad():
        policy.mean_head.bias.add_(0.2)
        policy.log_std.add_(0.1)
    current = policy(states)
    torch.testing.assert_close(
        policy.log_prob(states, actions), current.log_prob(actions).sum((1, 2))
    )
    torch.testing.assert_close(policy.entropy(states), current.entropy().sum((1, 2)))
    torch.testing.assert_close(
        policy.kl_from(states, old.loc, old.scale),
        torch.distributions.kl_divergence(old, current).sum((1, 2)),
    )


def test_fisher_vector_product_matches_finite_difference_of_kl_gradient():
    torch.manual_seed(2)
    policy = make_policy()
    states = torch.randn(8, 2, dtype=torch.float64)
    dist = policy(states)
    old_mean, old_std = dist.loc.detach().clone(), dist.scale.detach().clone()
    parameters = tuple(policy.parameters())
    original = parameters_to_vector(parameters).detach().clone()
    vector = torch.randn_like(original)
    product = fisher_vector_product(policy, states, old_mean, old_std, vector, damping=0.1)

    def gradient_at(offset):
        with torch.no_grad():
            vector_to_parameters(original + offset * vector, parameters)
        gradients = torch.autograd.grad(
            policy.kl_from(states, old_mean, old_std).mean(), parameters
        )
        return torch.cat([value.reshape(-1) for value in gradients])

    eps = 1e-5
    expected = (gradient_at(eps) - gradient_at(-eps)) / (2 * eps) + 0.1 * vector
    torch.testing.assert_close(product, expected, rtol=1e-5, atol=1e-8)


def test_npg_improves_bandit_policy_within_kl_budget():
    torch.manual_seed(3)
    policy = make_policy()
    batch = make_batch(policy)
    old = deepcopy(policy)
    metrics = npg_step(policy, batch, NPGConfig(max_kl=0.02))

    assert metrics["policy/accepted"] == 1
    assert metrics["policy/objective_after"] > metrics["policy/objective_before"]
    assert policy(batch.states).mean.mean() > old(batch.states).mean.mean()
    kl = (
        torch.distributions.kl_divergence(old(batch.states), policy(batch.states))
        .sum((1, 2))
        .mean()
    )
    assert kl.item() <= 0.02
    assert metrics["policy/kl"] == pytest.approx(kl.item())


def test_npg_leaves_zero_gradient_policy_unchanged():
    policy = make_policy()
    original = parameters_to_vector(policy.parameters()).detach().clone()
    metrics = npg_step(policy, make_batch(policy, zero_advantages=True), NPGConfig())
    assert metrics["policy/accepted"] == 0
    torch.testing.assert_close(parameters_to_vector(policy.parameters()), original, rtol=0, atol=0)


def test_failed_candidate_restores_original_policy():
    torch.manual_seed(3)
    policy = make_policy()
    original = parameters_to_vector(policy.parameters()).detach().clone()
    calls = 0

    def augmentation():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("candidate failed")
        return policy.log_std.sum() * 0

    with pytest.raises(RuntimeError, match="candidate failed"):
        npg_step(policy, make_batch(policy), NPGConfig(), augmentation=augmentation)
    torch.testing.assert_close(parameters_to_vector(policy.parameters()), original, rtol=0, atol=0)


def test_dapg_uses_demos_for_gradient_but_only_rollout_states_for_fisher(monkeypatch):
    import mujoco_lab.learning.algorithms.npg as npg

    policy = make_policy()
    batch = make_batch(policy, zero_advantages=True)
    demos = Demonstrations(
        torch.full((8, 2), 2.0, dtype=torch.float64), torch.ones(8, 1, 1).double()
    )
    before = policy.log_prob(demos.states, demos.actions).mean().item()
    original_fvp = npg.fisher_vector_product
    fisher_calls = []

    def record_fvp(policy, states, *args, **kwargs):
        fisher_calls.append(states.clone())
        return original_fvp(policy, states, *args, **kwargs)

    monkeypatch.setattr(npg, "fisher_vector_product", record_fvp)
    metrics = dapg_step(
        policy,
        batch,
        demos,
        iteration=2,
        config=NPGConfig(),
        demo_weight=0.5,
        demo_decay=0.9,
        demo_batch_size=4,
    )
    assert metrics["policy/demo_weight"] == pytest.approx(0.405)
    assert policy.log_prob(demos.states, demos.actions).mean().item() > before
    assert fisher_calls
    for states in fisher_calls:
        torch.testing.assert_close(states, batch.states)


def test_zero_demo_weight_is_identical_to_npg():
    torch.manual_seed(5)
    policy = make_policy()
    other = deepcopy(policy)
    batch = make_batch(policy)
    demos = Demonstrations(torch.ones(3, 2).double(), torch.zeros(3, 1, 1).double())
    npg_step(policy, batch, NPGConfig())
    dapg_step(other, batch, demos, iteration=0, config=NPGConfig(), demo_weight=0)
    torch.testing.assert_close(
        parameters_to_vector(policy.parameters()),
        parameters_to_vector(other.parameters()),
        rtol=0,
        atol=0,
    )
