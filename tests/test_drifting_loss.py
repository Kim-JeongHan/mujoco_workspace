"""Check drifting loss reductions and finite gradients for action chunks."""

import pytest
import torch

from mujoco_lab.learning.models.drifting import DriftingLoss, drift_loss


def test_generated_gradients_are_finite_and_other_inputs_are_detached():
    rng = torch.Generator().manual_seed(42)
    generated = torch.randn(2, 5, 6, generator=rng, requires_grad=True)
    positive = torch.randn(2, 3, 6, generator=rng, requires_grad=True)
    negative = torch.randn(2, 4, 6, generator=rng, requires_grad=True)
    weight = torch.rand(2, 3, generator=rng, requires_grad=True)
    loss, info = drift_loss(generated, positive, negative, weight_pos=weight)
    loss.mean().backward()
    assert loss.shape == (2,) and loss.dtype == torch.float32
    assert generated.grad is not None
    assert torch.isfinite(generated.grad).all() and generated.grad.abs().sum() > 0
    assert positive.grad is negative.grad is weight.grad is None
    assert all(not value.requires_grad for value in info.values())


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_action_chunks_keep_condition_and_sample_axes(reduction):
    generated = torch.randn(2, 5, 4, 3, requires_grad=True)
    positive = torch.randn(2, 1, 4, 3)
    expected, _ = drift_loss(generated.flatten(2), positive.flatten(2))
    if reduction != "none":
        expected = getattr(expected, reduction)()
    actual, _ = DriftingLoss(reduction=reduction)(generated, positive)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert generated.grad is not None
    assert torch.isfinite(generated.grad).all()


def test_zero_force_has_finite_zero_gradient():
    generated = torch.zeros(2, 4, 3, requires_grad=True)
    loss, _ = drift_loss(generated, torch.zeros(2, 1, 3))
    loss.mean().backward()
    assert generated.grad is not None
    assert torch.equal(loss, torch.zeros_like(loss))
    assert torch.equal(generated.grad, torch.zeros_like(generated))
