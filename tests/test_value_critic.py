"""Value regression updates and target shape handling."""

import numpy as np
import pytest
import torch

from mujoco_lab.learning.models.critic import ValueCritic


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("column_targets", [False, True])
def test_update_fits_return_targets(batch_size, column_targets):
    critic = ValueCritic(ob_dim=3, n_layers=0, layer_size=8, learning_rate=0.1)
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.zero_()
    obs = np.zeros((batch_size, 3), dtype=np.float64)
    targets = np.ones((batch_size, 1) if column_targets else (batch_size,), dtype=np.float64)
    first = critic.update(obs, targets)
    assert first == {"Baseline Loss": 1.0}
    for _ in range(4):
        result = critic.update(obs, targets)
    assert result["Baseline Loss"] < first["Baseline Loss"]
    values = critic(torch.from_numpy(obs).float())
    assert values.shape == (batch_size,)
    assert torch.isfinite(values).all()
    assert critic.optimizer.state


def test_update_rejects_targets_that_would_broadcast():
    critic = ValueCritic(3, 1, 8, 1e-3)
    with pytest.raises(ValueError, match="q_values must have shape"):
        critic.update(np.zeros((4, 3)), np.ones((4, 2)))


def test_tensor_targets_are_fixed_during_critic_update():
    critic = ValueCritic(3, 1, 8, 1e-3)
    targets = torch.ones(4, requires_grad=True)
    critic.update(torch.zeros(4, 3), targets)
    assert targets.grad is None
    assert critic.optimizer.state
