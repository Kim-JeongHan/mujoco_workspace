"""Cube-stack rot6d features remain raw through training and checkpoint inference."""

import numpy as np
import pytest
import torch

from mujoco_lab.learning.checkpoint import load_checkpoint, save_checkpoint
from mujoco_lab.learning.config.config import TrainConfig
from mujoco_lab.learning.datasets.episode import Episode
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.datasets.sequence import ChunkDataset
from mujoco_lab.learning.trainers.train_bc import run_training


def _rotation_indices(cubes: int) -> list[int]:
    return [
        *range(18, 24),
        *(index for cube in range(cubes) for index in range(27 + 15 * cube, 33 + 15 * cube)),
    ]


def _episode(cubes: int = 2) -> Episode:
    states = np.zeros((4, 24 + 15 * cubes), dtype=np.float32)
    states[:, 0] = [10, 12, 14, 1000]
    states[:3, _rotation_indices(cubes)] = np.array([[0.98], [0.99], [1.0]])
    states[3, _rotation_indices(cubes)] = -0.8
    return Episode(
        states=states,
        actions=np.array([[2.0], [4.0], [6.0]], dtype=np.float32),
        metadata={"replay": {"scene": "cube_stack", "cubes": cubes}},
    )


def test_cube_stack_rotations_are_raw_while_other_features_and_actions_are_scaled(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    episode = _episode()
    original = episode.states.copy()
    config = TrainConfig(
        policy_type="mse",
        obs_horizon=2,
        chunk_size=1,
        execution_horizon=1,
        physics_steps_per_action=1,
        hidden_dims=(8,),
        batch_size=3,
        num_epochs=1,
        log_interval=100,
    )
    model, normalizer = run_training(config, [episode])
    rotations = _rotation_indices(2)

    np.testing.assert_array_equal(normalizer.state_mean[rotations], 0)
    np.testing.assert_array_equal(normalizer.state_std[rotations], 1)
    np.testing.assert_allclose(normalizer.state_mean[0], 12)
    np.testing.assert_allclose(normalizer.state_std[0], np.std([10, 12, 14]))
    np.testing.assert_allclose(normalizer.action_mean, [4])
    np.testing.assert_allclose(normalizer.action_std, [np.std([2, 4, 6])])
    np.testing.assert_allclose(normalizer.normalize_action(np.array([6.0])), [np.sqrt(1.5)])

    held_out = normalizer.normalize_state(episode.states[-1])
    np.testing.assert_allclose(held_out[rotations], -0.8)
    np.testing.assert_allclose(normalizer.denormalize_state(held_out), episode.states[-1])
    history, _ = ChunkDataset([episode], 1, normalizer, obs_horizon=2)[1]
    np.testing.assert_allclose(history.reshape(2, -1)[:, rotations], original[:2, rotations])
    np.testing.assert_array_equal(episode.states, original)

    checkpoint = tmp_path / "raw.pt"
    save_checkpoint(checkpoint, model, normalizer, config, optimizer_step=1)
    _, restored, _ = load_checkpoint(checkpoint)
    np.testing.assert_array_equal(
        restored.normalize_state(episode.states[-1])[rotations], held_out[rotations]
    )
    np.testing.assert_array_equal(restored.action_mean, normalizer.action_mean)

    old = Normalizer(
        state_mean=np.ones(54),
        state_std=np.full(54, 2.0),
        action_mean=normalizer.action_mean,
        action_std=normalizer.action_std,
    )
    old_checkpoint = tmp_path / "old.pt"
    save_checkpoint(old_checkpoint, model, old, config, optimizer_step=1)
    _, restored_old, _ = load_checkpoint(old_checkpoint)
    np.testing.assert_array_equal(restored_old.state_mean, old.state_mean)
    np.testing.assert_array_equal(restored_old.state_std, old.state_std)
    np.testing.assert_allclose(
        restored_old.normalize_state(episode.states[-1])[rotations],
        (episode.states[-1, rotations] - 1) / 2,
    )


def test_generic_state_with_same_width_keeps_zscore():
    from mujoco_lab.learning.trainers.train_bc import _cube_stack_rotation_indices

    episode = _episode()
    episode.metadata = {}
    assert _cube_stack_rotation_indices([episode], episode.states.shape[1]) == []
    normalizer = Normalizer.from_data(episode.states[:-1], episode.actions)
    rotations = _rotation_indices(2)
    assert np.any(normalizer.state_mean[rotations] != 0)
    assert np.any(normalizer.state_std[rotations] != 1)


@pytest.mark.parametrize("cubes", [2, 3, 4])
def test_cube_count_selects_all_rotation_slices(cubes):
    from mujoco_lab.learning.trainers.train_bc import _cube_stack_rotation_indices

    episode = _episode(cubes)
    assert _cube_stack_rotation_indices([episode], episode.states.shape[1]) == _rotation_indices(
        cubes
    )


def test_cube_stack_metadata_must_match_observation_shape():
    from mujoco_lab.learning.trainers.train_bc import _cube_stack_rotation_indices

    episode = _episode()
    with pytest.raises(ValueError, match="dimension"):
        _cube_stack_rotation_indices([episode], 55)


def test_validation_layout_must_match_training_layout(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    training = _episode()
    validation = _episode()
    validation.metadata = {}
    config = TrainConfig(
        hidden_dims=(8,),
        chunk_size=1,
        execution_horizon=1,
        physics_steps_per_action=1,
        batch_size=3,
        num_epochs=1,
    )
    with pytest.raises(ValueError, match="mix cube-stack"):
        run_training(config, [training], [validation])
