"""Sinusoidal flow time features and checkpoint round trips."""

import numpy as np
import pytest
import torch

from mujoco_lab.learning.checkpoint import load_checkpoint, save_checkpoint
from mujoco_lab.learning.config import BCTrainConfig
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.models import SinusoidalTimeEmbedding
from mujoco_lab.learning.policies.factory import build_policy


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


def _normalizer():
    return Normalizer(
        state_mean=np.zeros(2, dtype=np.float32),
        state_std=np.ones(2, dtype=np.float32),
        action_mean=np.zeros(1, dtype=np.float32),
        action_std=np.ones(1, dtype=np.float32),
    )


def _config(policy_type, flow_time_embed_dim=128):
    return BCTrainConfig(
        policy_type=policy_type,
        flow_time_embed_dim=flow_time_embed_dim,
        hidden_dims=(8,),
        chunk_size=2,
        execution_horizon=1,
        action_execution_hz=500,
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
