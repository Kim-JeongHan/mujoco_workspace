"""Check replanning alignment, physical units, and isolated sampling probes."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "measure_flow_continuity",
    Path(__file__).resolve().parents[1] / "scripts/measure_flow_continuity.py",
)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_boundary_uses_pending_action_and_last_two_executed_targets():
    a2, a3, a4, b0 = (np.full(7, value, dtype=float) for value in (1, 3, 4, 6))
    result = diagnostic.boundary_errors(b0, a4, np.stack((a2, a3)), 0.01)
    np.testing.assert_array_equal(result["replan_vector_rad"], np.full(7, 2))
    np.testing.assert_array_equal(result["command_vector_rad"], np.full(7, 3))
    np.testing.assert_array_equal(result["smooth_vector_rad"], np.ones(7))
    assert result["replan_l2_rad"] == pytest.approx(2 * np.sqrt(7))
    assert result["command_l2_deg"] == pytest.approx(np.rad2deg(3 * np.sqrt(7)))
    assert result["acceleration_l2_rad_s2"] == pytest.approx(np.sqrt(7) / 0.01**2)


def test_constant_velocity_commands_have_zero_second_difference():
    result = diagnostic.boundary_errors(
        np.full(7, 5.0), np.full(7, 5.0), np.array([np.full(7, 1.0), np.full(7, 3.0)]), 0.02
    )
    assert result["smooth_l2_rad"] == 0
    assert result["acceleration_l2_rad_s2"] == 0


def test_probe_holds_full_input_fixed_and_preserves_rollout_rng():
    seen = []

    class ToyPolicy:
        def sample_actions(self, state, *, num_steps):
            seen.append(state.clone())
            assert num_steps == 20
            return torch.randn(len(state), 5, 8)

    state = torch.arange(12, dtype=torch.float32).reshape(1, -1)
    normalizer = SimpleNamespace(denormalize_action=lambda x: x)
    torch.manual_seed(42)
    expected_rng = torch.random.get_rng_state().clone()
    result = diagnostic.sampling_jitter(
        ToyPolicy(),
        state,
        normalizer,
        np.full(8, -0.1),
        np.full(8, 0.1),
        samples=64,
        seed=123,
        integration_steps=20,
    )
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    assert torch.equal(seen[0], state.repeat(64, 1))
    samples = result["raw_b0_arm_rad"]
    assert samples.shape == (64, 7)
    np.testing.assert_allclose(result["raw_variance_rad2"], samples.var(axis=0, ddof=1))
    np.testing.assert_allclose(result["raw_covariance_rad2"], np.cov(samples, rowvar=False))
    assert result["raw_rms_spread_rad"] == pytest.approx(np.sqrt(samples.var(axis=0, ddof=1).sum()))
    assert np.all(result["bounded_variance_rad2"] < result["raw_variance_rad2"])
