"""Partial action chunks supervise real targets without training padded positions."""

import pytest
import torch
from torch.nn import functional as F  # noqa: N812

from mujoco_lab.learning.config import DiffusionSchedulerConfig
from mujoco_lab.learning.policies.diffusion import DiffusionPolicy
from mujoco_lab.learning.policies.flow import FlowMatchingPolicy
from mujoco_lab.learning.policies.gaussian import GaussianPolicy
from mujoco_lab.learning.policies.mse import MSEPolicy


def test_mse_mask_uses_valid_coordinates_and_has_no_padded_output_gradient():
    policy = MSEPolicy(2, 2, 3, hidden_dims=())
    head = torch.nn.Linear(2, 6)
    policy.net[-1] = head
    bias = torch.nn.Parameter(torch.arange(1.0, 7.0))
    head.bias = bias
    with torch.no_grad():
        head.weight.zero_()
    state = torch.zeros(2, 2)
    mask = torch.tensor([[True, False, False], [True, True, False]])
    target = torch.zeros(2, 3, 2)
    target[~mask] = 1000
    loss = policy.compute_loss(state, target, valid_mask=mask)
    assert loss.item() == pytest.approx(35 / 6)
    loss.backward()
    assert bias.grad is not None
    torch.testing.assert_close(bias.grad.reshape(3, 2)[2], torch.zeros(2))
    assert bias.grad.reshape(3, 2)[:2].abs().sum() > 0


def make_policy(kind):
    if kind == "mse":
        return MSEPolicy(4, 2, 3, hidden_dims=(8,))
    if kind == "flow":
        return FlowMatchingPolicy(4, 2, 3, hidden_dims=(8,), time_embed_dim=8)
    if kind == "diffusion":
        return DiffusionPolicy(
            4,
            2,
            3,
            scheduler_config=DiffusionSchedulerConfig(),
            down_dims=(16, 32),
            time_embed_dim=16,
        )
    return GaussianPolicy(4, 2, 3, hidden_dims=(8,))


@pytest.mark.parametrize("kind", ["mse", "flow", "diffusion", "gaussian"])
def test_masked_targets_cannot_change_loss_or_parameter_gradients(kind):
    torch.manual_seed(7)
    policy = make_policy(kind)
    state = torch.randn(2, 4)
    target = torch.randn(2, 3, 2)
    mask = torch.tensor([[True, True, False], [True, False, False]])

    def loss_and_gradients(actions):
        policy.zero_grad(set_to_none=True)
        torch.manual_seed(11)
        loss = policy.compute_loss(state, actions, valid_mask=mask)
        loss.backward()
        return loss.detach(), [p.grad.clone() for p in policy.parameters()]

    before, gradients = loss_and_gradients(target)
    modified = target.clone()
    modified[~mask] = 1000
    after, updated = loss_and_gradients(modified)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    for expected, actual in zip(gradients, updated, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["mse", "flow", "diffusion", "gaussian"])
def test_full_mask_preserves_the_existing_objective(kind):
    torch.manual_seed(7)
    policy = make_policy(kind)
    state, target = torch.randn(2, 4), torch.randn(2, 3, 2)
    torch.manual_seed(11)
    expected = policy.compute_loss(state, target)
    torch.manual_seed(11)
    actual = policy.compute_loss(state, target, valid_mask=torch.ones(2, 3, dtype=torch.bool))
    torch.testing.assert_close(actual, expected)
    if kind == "mse":
        torch.testing.assert_close(actual, F.mse_loss(policy.sample_actions(state), target))
