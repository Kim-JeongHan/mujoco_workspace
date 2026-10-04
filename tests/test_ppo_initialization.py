"""BC checkpoint integration for Gaussian PPO actor initialization."""

import numpy as np
import pytest
import torch

from mujoco_lab.learning.checkpoint import save_checkpoint
from mujoco_lab.learning.config.config import TrainConfig
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.policies.factory import build_policy
from mujoco_lab.learning.trainers.train_ppo import initialize_actor, initialize_actor_critic


@pytest.mark.parametrize("hidden_dims", [(), (16, 8)])
def test_initialize_actor_from_bc_checkpoint(tmp_path, hidden_dims):
    config = TrainConfig(
        policy_type="mse",
        hidden_dims=hidden_dims,
        obs_horizon=2,
        chunk_size=3,
        execution_horizon=1,
    )
    bc = build_policy("mse", state_dim=6, action_dim=2, chunk_size=3, hidden_dims=hidden_dims)
    normalizer = Normalizer(
        state_mean=np.array([1, 2, 3], dtype=np.float32),
        state_std=np.array([2, 3, 4], dtype=np.float32),
        action_mean=np.array([5, 6], dtype=np.float32),
        action_std=np.array([7, 8], dtype=np.float32),
    )
    path = tmp_path / "bc.pt"
    save_checkpoint(path, bc, normalizer, config, optimizer_step=12)

    actor, critic, restored, metadata = initialize_actor_critic(path, initial_log_std=-1.5)
    state = torch.randn(4, 6)
    torch.testing.assert_close(actor(state).mean, bc(state), rtol=0, atol=0)
    torch.testing.assert_close(actor.log_std, torch.full((3, 2), -1.5))
    assert actor.training
    assert all(parameter.requires_grad for parameter in actor.parameters())
    for name in ("state_mean", "state_std", "action_mean", "action_std"):
        np.testing.assert_array_equal(getattr(restored, name), getattr(normalizer, name))
    assert metadata["architecture"]["obs_horizon"] == 2
    assert metadata["optimizer_step"] == 12

    actor_storage = {parameter.data_ptr() for parameter in actor.parameters()}
    assert all(parameter.data_ptr() not in actor_storage for parameter in critic.parameters())
    assert next(critic.parameters()).device == next(actor.parameters()).device
    values = critic(state)
    assert values.shape == (4,)
    assert critic(state[:1]).shape == (1,)
    assert torch.isfinite(values).all()
    values.square().mean().backward()
    assert critic.network[-1].weight.grad is not None
    assert all(parameter.grad is None for parameter in actor.parameters())


def test_initialize_actor_rejects_flow_checkpoint(tmp_path):
    config = TrainConfig(policy_type="flow", hidden_dims=(8,), obs_horizon=1, chunk_size=1)
    bc = build_policy("flow", state_dim=3, action_dim=2, chunk_size=1, hidden_dims=(8,))
    normalizer = Normalizer(
        state_mean=np.zeros(3, dtype=np.float32),
        state_std=np.ones(3, dtype=np.float32),
        action_mean=np.zeros(2, dtype=np.float32),
        action_std=np.ones(2, dtype=np.float32),
    )
    path = tmp_path / "flow.pt"
    save_checkpoint(path, bc, normalizer, config, optimizer_step=0)
    with pytest.raises(ValueError, match="MSE BC checkpoint"):
        initialize_actor(path)
