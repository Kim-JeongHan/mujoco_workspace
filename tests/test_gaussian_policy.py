"""Gaussian action sampling and deterministic BC weight transfer."""

import pytest
import torch

from mujoco_lab.learning.policies.gaussian import GaussianPolicy
from mujoco_lab.learning.policies.mse import MSEPolicy


@pytest.mark.parametrize("hidden_dims", [(), (16, 8)])
@pytest.mark.parametrize("chunk_size", [1, 3])
def test_mse_weights_initialize_gaussian_mean(hidden_dims, chunk_size):
    bc_policy = MSEPolicy(5, 2, chunk_size, hidden_dims=hidden_dims)
    actor = GaussianPolicy(5, 2, chunk_size, hidden_dims=hidden_dims)
    actor.net.load_state_dict(bc_policy.net[:-1].state_dict())
    actor.mean_head.load_state_dict(bc_policy.net[-1].state_dict())
    with torch.no_grad():
        actor.log_std.fill_(-2.0)

    state = torch.randn(4, 5)
    dist = actor(state)
    torch.testing.assert_close(dist.mean, bc_policy(state), rtol=0, atol=0)
    torch.testing.assert_close(dist.stddev, torch.full_like(dist.mean, -2.0).exp())

    actions = actor.sample_actions(state, num_steps=1)
    assert actions.shape == (4, chunk_size, 2)
    assert torch.isfinite(actions).all()
    loss = actor.compute_loss(state, bc_policy(state).detach())
    assert loss.shape == ()
    assert torch.isfinite(loss)
    loss.backward()
    assert actor.mean_head.weight.grad is not None
    assert actor.log_std.grad is not None
