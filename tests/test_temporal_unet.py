"""Shared temporal denoising with optional global conditioning."""

import pytest
import torch

from mujoco_lab.learning.models.diffusion import DiffusionModel, TemporalUnet, ValueModel


@pytest.mark.parametrize(
    ("horizon", "down_dims", "condition_dim", "dtype"),
    [
        (1, (16,), 0, torch.float32),
        (3, (16, 32), 0, torch.float64),
        (3, (16, 32), 4, torch.float32),
        (16, (8, 16, 32), 4, torch.float64),
        (16, (8, 16, 32, 64), 0, torch.float32),
    ],
)
def test_conditioned_and_timestep_only_unet_preserve_sequences_and_gradients(
    horizon, down_dims, condition_dim, dtype
):
    torch.manual_seed(5)
    model = TemporalUnet(
        transition_dim=2,
        down_dims=down_dims,
        time_embed_dim=16,
        condition_dim=condition_dim,
    ).to(dtype=dtype)
    x = torch.randn(2, horizon, 2, dtype=dtype, requires_grad=True)
    timesteps = torch.tensor([10, 20])
    condition = (
        torch.randn(2, condition_dim, dtype=dtype, requires_grad=True) if condition_dim else None
    )

    noise = model(x, timesteps, condition)
    assert noise.shape == x.shape and noise.dtype == dtype
    assert not torch.allclose(noise, model(x, timesteps + 10, condition))
    if condition is not None:
        assert not torch.allclose(noise, model(x, timesteps, condition + 1))

    noise.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    if condition is not None:
        assert condition.grad is not None
        assert torch.isfinite(condition.grad).all() and condition.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("condition_dim", [0, 4])
def test_omitted_condition_matches_explicit_zero_features(condition_dim):
    model = TemporalUnet(2, down_dims=(16, 32), time_embed_dim=16, condition_dim=condition_dim)
    x = torch.randn(2, 3, 2)
    timesteps = torch.tensor([10, 20])
    with torch.no_grad():
        expected = model(x, timesteps, x.new_zeros((2, condition_dim)))
        torch.testing.assert_close(model(x, timesteps), expected, rtol=0, atol=0)
        torch.testing.assert_close(model(x, timesteps, condition=None), expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(1, 4), (2, 3), (2, 1, 4)])
def test_condition_shape_cannot_broadcast_across_sequences(shape):
    model = TemporalUnet(2, down_dims=(16,), time_embed_dim=16, condition_dim=4)
    with pytest.raises(ValueError, match="condition must have shape"):
        model(torch.randn(2, 3, 2), torch.tensor([10, 20]), torch.randn(shape))


def test_timestep_only_model_rejects_nonempty_condition():
    model = TemporalUnet(2, down_dims=(16,), time_embed_dim=16)
    with pytest.raises(ValueError, match="condition must have shape"):
        model(torch.randn(2, 3, 2), torch.tensor([10, 20]), torch.randn(2, 4))


@pytest.mark.parametrize("condition_dim", [0, 4])
def test_diffusion_model_supports_scalar_time_with_optional_condition(condition_dim):
    model = DiffusionModel(
        state_dim=2, horizon=3, dim=4, dim_mults=(1, 2), condition_dim=condition_dim
    )
    assert isinstance(model.unet, TemporalUnet)
    x = torch.randn(2, 3, 2)
    condition = torch.randn(2, condition_dim) if condition_dim else None
    noise = model(x, torch.tensor([10, 10]), condition)
    torch.testing.assert_close(model(x, torch.tensor(10), condition), noise)
    torch.testing.assert_close(model(x, torch.tensor([10]), condition), noise)
    assert not torch.allclose(noise, model(x, torch.tensor([10, 20]), condition))
    if condition is not None:
        assert not torch.allclose(noise, model(x, torch.tensor(10), condition + 1))
    assert model(x, torch.tensor(10)).shape == x.shape


def test_value_encoder_keeps_additive_residual_conditioning():
    model = ValueModel(state_dim=2, horizon=3, dim=4, dim_mults=(1, 2))
    x = torch.randn(2, 3, 2, requires_grad=True)
    value = model(x)
    assert value.shape == (2, 1)
    value.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
