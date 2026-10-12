"""Conditional diffusion math, trainable denoising, and checkpoint reproduction."""

import numpy as np
import pytest
import torch
from torch import nn

from mujoco_lab.learning.checkpoint import load_checkpoint, save_checkpoint
from mujoco_lab.learning.config import BCTrainConfig
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.policies import DiffusionPolicy, build_policy


def make_policy(chunk_size=3, **kwargs):
    return DiffusionPolicy(
        state_dim=4,
        action_dim=2,
        chunk_size=chunk_size,
        down_dims=(16, 32),
        time_embed_dim=16,
        **kwargs,
    )


@pytest.mark.parametrize("chunk_size", [1, 3, 16])
def test_unet_preserves_chunk_length_and_uses_observation_and_time(chunk_size):
    torch.manual_seed(5)
    policy = make_policy(chunk_size)
    state = torch.randn(2, 4, requires_grad=True)
    actions = torch.randn(2, chunk_size, 2)
    timesteps = torch.tensor([10, 20])
    noise = policy(state, actions, timesteps)
    assert noise.shape == actions.shape
    assert not torch.allclose(noise, policy(state + 1, actions, timesteps))
    assert not torch.allclose(noise, policy(state, actions, timesteps + 10))
    loss = policy.compute_loss(state, actions)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(state.grad).all() and state.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters())


class KnownActionDenoiser(nn.Module):
    """Return exact noise when the clean action is known from the observation."""

    def __init__(self, alphas):
        super().__init__()
        self.register_buffer("alphas", alphas)

    def forward(self, noisy_actions, timesteps, state):
        clean = state[:, None, : noisy_actions.shape[-1]].expand_as(noisy_actions)
        alpha = self.alphas[timesteps].reshape(-1, 1, 1)
        return (noisy_actions - alpha.sqrt() * clean) / (1 - alpha).sqrt()


@pytest.mark.parametrize("num_steps", [1, 7, 20, 100])
def test_exact_denoiser_has_zero_loss_and_recovers_unclipped_actions(num_steps):
    policy = make_policy()
    policy.denoiser = KnownActionDenoiser(policy.alphas_cumprod)
    state = torch.tensor([[3.0, -2.5, 0.0, 0.0], [1.5, -4.0, 1.0, 1.0]])
    actions = state[:, None, :2].expand(-1, 3, -1)
    torch.manual_seed(6)
    assert policy.compute_loss(state, actions).item() < 1e-10
    sampled = policy.sample_actions(state, num_steps=num_steps)
    torch.testing.assert_close(sampled, actions, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_sampler_preserves_device_dtype_and_seed(dtype, device):
    policy = make_policy().to(device=device, dtype=dtype)
    state = torch.randn(1, 4, dtype=dtype, device=device)
    torch.manual_seed(9)
    first = policy.sample_actions(state, num_steps=3)
    torch.manual_seed(9)
    second = policy.sample_actions(state, num_steps=3)
    assert first.dtype == dtype and first.device == state.device
    assert first.shape == (1, 3, 2) and torch.isfinite(first).all()
    assert not first.requires_grad
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    loss = policy.compute_loss(state, torch.randn_like(first))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters())


def test_checkpoint_restores_actual_unet_and_complete_scheduler_settings(tmp_path):
    config = BCTrainConfig(
        policy_type="diffusion",
        chunk_size=3,
        execution_horizon=2,
        diffusion_down_dims=(16, 32),
        diffusion_time_embed_dim=16,
        diffusion_num_train_timesteps=40,
        diffusion_beta_schedule="linear",
        sampling_num_steps=7,
    )
    policy = build_policy(
        "diffusion",
        state_dim=4,
        action_dim=2,
        chunk_size=3,
        diffusion_down_dims=config.diffusion_down_dims,
        diffusion_time_embed_dim=config.diffusion_time_embed_dim,
        diffusion_num_train_timesteps=config.diffusion_num_train_timesteps,
        diffusion_beta_schedule=config.diffusion_beta_schedule,
    )
    normalizer = Normalizer(
        state_mean=np.zeros(2, dtype=np.float32),
        state_std=np.ones(2, dtype=np.float32),
        action_mean=np.array([0.1, 0.02], dtype=np.float32),
        action_std=np.array([0.3, 0.01], dtype=np.float32),
    )
    path = tmp_path / "diffusion.pt"
    save_checkpoint(path, policy, normalizer, config, optimizer_step=7)
    loaded, restored_normalizer, metadata = load_checkpoint(path)
    assert isinstance(loaded, DiffusionPolicy)
    architecture = metadata["architecture"]
    assert architecture["diffusion_down_dims"] == [16, 32]
    assert architecture["diffusion_time_embed_dim"] == 16
    scheduler_settings = architecture["diffusion_scheduler_config"]
    assert scheduler_settings["num_train_timesteps"] == 40
    assert scheduler_settings["beta_schedule"] == "linear"
    assert scheduler_settings["prediction_type"] == "epsilon"
    assert not scheduler_settings["clip_sample"]
    for name, value in scheduler_settings.items():
        assert loaded.scheduler.config[name] == value
    assert metadata["train_config"]["sampling_num_steps"] == 7
    np.testing.assert_array_equal(restored_normalizer.action_std, normalizer.action_std)
    state = torch.randn(2, 4)
    torch.manual_seed(11)
    before = policy.sample_actions(state, num_steps=7)
    torch.manual_seed(11)
    after = loaded.sample_actions(state, num_steps=7)
    torch.testing.assert_close(after, before, rtol=0, atol=0)


@pytest.mark.parametrize("num_steps", [0, 101])
def test_sampler_rejects_unsupported_step_counts(num_steps):
    with pytest.raises(ValueError):
        make_policy().sample_actions(torch.zeros(1, 4), num_steps=num_steps)


def test_loss_rejects_silently_broadcast_action_chunks():
    with pytest.raises(ValueError, match="action_chunk"):
        make_policy().compute_loss(torch.zeros(2, 4), torch.zeros(2, 1, 2))
