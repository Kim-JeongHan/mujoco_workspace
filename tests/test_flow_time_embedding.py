"""Sinusoidal flow time and checkpoint compatibility."""

import numpy as np
import pytest
import torch

from mujoco_lab.learning.checkpoint import load_checkpoint, save_checkpoint
from mujoco_lab.learning.config.config import TrainConfig
from mujoco_lab.learning.datasets.episode import Episode
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.models import SinusoidalTimeEmbedding
from mujoco_lab.learning.policies.factory import build_policy
from mujoco_lab.learning.trainers.train_bc import run_training


@pytest.mark.parametrize("dim", [0, -2, 3, True, 2.5])
def test_time_embedding_requires_positive_even_width(dim):
    with pytest.raises(ValueError, match="positive even integer"):
        SinusoidalTimeEmbedding(dim)


def test_unit_interval_embedding_varies_and_preserves_dtype_and_gradient():
    embedding = SinusoidalTimeEmbedding(128)
    times = torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=torch.float64, requires_grad=True)
    features = embedding(times)
    assert features.shape == (4, 128)
    assert features.dtype == times.dtype
    assert features.device == times.device
    assert torch.any(torch.abs(features[0] - features[1]) > 0.1)
    assert torch.any(torch.abs(features[1] - features[2]) > 0.1)
    assert torch.any(torch.abs(features[2] - features[3]) > 0.1)
    features[1].sum().backward()
    assert torch.isfinite(times.grad).all()
    assert times.grad[1] != 0


def test_flow_default_uses_embedding_and_keeps_training_sampling_shapes():
    model = build_policy("flow", state_dim=4, action_dim=2, chunk_size=3, hidden_dims=(8,))
    assert model.time_embed_dim == 128
    assert model.net[0].in_features == 4 + 3 * 2 + 128
    states = torch.randn(2, 4)
    actions = torch.randn(2, 3, 2)
    loss = model.compute_loss(states, actions)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.net[0].weight.grad is not None
    with torch.no_grad():
        assert model.sample_actions(states, num_steps=2).shape == actions.shape


def test_training_passes_configured_embedding_width(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = _config("flow", flow_time_embed_dim=64)
    config.num_epochs = 1
    config.eval_interval = 0
    config.log_interval = 1
    episode = Episode(
        states=np.zeros((3, 2), dtype=np.float32),
        actions=np.zeros((2, 1), dtype=np.float32),
    )
    model, _ = run_training(config, [episode])
    assert model.time_embed_dim == 64
    assert model.net[0].in_features == 4 + 2 + 64


def _normalizer():
    return Normalizer(
        state_mean=np.zeros(2, dtype=np.float32),
        state_std=np.ones(2, dtype=np.float32),
        action_mean=np.zeros(1, dtype=np.float32),
        action_std=np.ones(1, dtype=np.float32),
    )


def _config(policy_type, flow_time_embed_dim=128):
    return TrainConfig(
        policy_type=policy_type,
        flow_time_embed_dim=flow_time_embed_dim,
        hidden_dims=(8,),
        chunk_size=2,
        execution_horizon=1,
        physics_steps_per_action=1,
    )


@pytest.mark.parametrize("width", [64, 128])
def test_flow_embedding_checkpoint_round_trip(tmp_path, width):
    config = _config("flow", flow_time_embed_dim=width)
    model = build_policy(
        "flow",
        state_dim=4,
        action_dim=1,
        chunk_size=2,
        hidden_dims=(8,),
        flow_time_embed_dim=width,
    )
    path = tmp_path / "embedded.pt"
    save_checkpoint(path, model, _normalizer(), config, optimizer_step=3)
    loaded, _, metadata = load_checkpoint(path)
    assert metadata["architecture"]["flow_time_embed_dim"] == width
    assert loaded.time_embed_dim == width
    states = torch.randn(2, 4)
    torch.manual_seed(5)
    before = model.sample_actions(states, num_steps=3)
    torch.manual_seed(5)
    after = loaded.sample_actions(states, num_steps=3)
    torch.testing.assert_close(after, before)


def test_old_scalar_flow_checkpoint_loads_with_identical_prediction(tmp_path):
    model = build_policy(
        "flow",
        state_dim=4,
        action_dim=1,
        chunk_size=2,
        hidden_dims=(8,),
        flow_time_embed_dim=None,
    )
    path = tmp_path / "scalar.pt"
    save_checkpoint(path, model, _normalizer(), _config("flow", None), optimizer_step=3)
    payload = torch.load(path, weights_only=True)
    del payload["architecture"]["flow_time_embed_dim"]
    old_path = tmp_path / "old_scalar.pt"
    torch.save(payload, old_path)
    loaded, _, metadata = load_checkpoint(old_path)
    assert metadata["architecture"].get("flow_time_embed_dim") is None
    assert loaded.time_embed_dim is None
    assert loaded.net[0].in_features == 4 + 2 + 1
    states = torch.randn(2, 4)
    torch.manual_seed(5)
    before = model.sample_actions(states, num_steps=3)
    torch.manual_seed(5)
    after = loaded.sample_actions(states, num_steps=3)
    torch.testing.assert_close(after, before)


def test_mse_checkpoint_stays_unchanged(tmp_path):
    model = build_policy("mse", state_dim=4, action_dim=1, chunk_size=2, hidden_dims=(8,))
    path = tmp_path / "mse.pt"
    save_checkpoint(path, model, _normalizer(), _config("mse"), optimizer_step=3)
    loaded, _, metadata = load_checkpoint(path)
    assert "flow_time_embed_dim" not in metadata["architecture"]
    states = torch.randn(2, 4)
    torch.testing.assert_close(loaded.sample_actions(states), model.sample_actions(states))
