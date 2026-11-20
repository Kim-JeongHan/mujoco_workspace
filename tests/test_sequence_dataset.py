"""Causal observation histories for offline action chunks."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from mujoco_lab.learning.config import DiffusionBCTrainConfig, FlowBCTrainConfig, MSEBCTrainConfig
from mujoco_lab.learning.datasets.episode import Episode
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.datasets.sequence import ChunkDataset
from mujoco_lab.learning.trainers.train_bc import run_training


def test_chunks_align_with_current_history_and_do_not_cross_episodes():
    first = Episode(states=np.array([[1], [2], [3], [4]]), actions=np.array([[10], [11], [12]]))
    second = Episode(states=np.array([[50], [60], [70]]), actions=np.array([[20], [21]]))
    dataset = ChunkDataset([first, second], chunk_size=2, obs_horizon=3)

    assert len(dataset) == 5
    np.testing.assert_array_equal(dataset[0][0], [1, 1, 1])
    np.testing.assert_array_equal(dataset[0][1], [[10], [11]])
    np.testing.assert_array_equal(dataset[1][0], [1, 1, 2])
    np.testing.assert_array_equal(dataset[1][1], [[11], [12]])
    np.testing.assert_array_equal(dataset[2][0], [1, 2, 3])
    np.testing.assert_array_equal(dataset[2][1], [[12], [0]])
    np.testing.assert_array_equal(dataset[2][2], [True, False])
    np.testing.assert_array_equal(dataset[3][0], [50, 50, 50])
    np.testing.assert_array_equal(dataset[3][1], [[20], [21]])
    np.testing.assert_array_equal(dataset[4][0], [50, 50, 60])
    np.testing.assert_array_equal(dataset[4][1], [[21], [0]])


def test_episode_shorter_than_chunk_retains_every_action_bearing_observation():
    episode = Episode(states=np.array([[1], [2], [3]]), actions=np.array([[10], [11]]))
    dataset = ChunkDataset([episode], chunk_size=4, obs_horizon=2)
    assert len(dataset) == 2
    state, actions, mask = dataset[1]
    np.testing.assert_array_equal(state, [1, 2])
    np.testing.assert_array_equal(actions, [[11], [0], [0], [0]])
    np.testing.assert_array_equal(mask, [True, False, False, False])
    assert mask.dtype == np.bool_


def test_tail_padding_is_zero_after_normalizing_real_targets():
    episode = Episode(states=np.array([[1], [2], [3]]), actions=np.array([[10], [14]]))
    normalizer = Normalizer(np.array([1]), np.array([2]), np.array([10]), np.array([2]))
    state, actions, mask = ChunkDataset([episode], 3, normalizer, obs_horizon=1)[1]
    np.testing.assert_array_equal(state, [0.5])
    np.testing.assert_array_equal(actions, [[2], [0], [0]])
    np.testing.assert_array_equal(mask, [True, False, False])
    assert state.dtype == actions.dtype == np.float32


def test_normalizer_applies_to_each_raw_frame_before_flattening_without_mutation():
    episode = Episode(
        states=np.array([[10, 100], [12, 104], [14, 108]], dtype=np.float64),
        actions=np.array([[5], [7]], dtype=np.float64),
    )
    before_states, before_actions = episode.states.copy(), episode.actions.copy()
    normalizer = Normalizer(
        state_mean=np.array([10, 100]),
        state_std=np.array([2, 4]),
        action_mean=np.array([5]),
        action_std=np.array([2]),
    )
    dataset = ChunkDataset([episode], chunk_size=1, normalizer=normalizer)

    np.testing.assert_array_equal(dataset[0][0], [0, 0, 0, 0])
    np.testing.assert_array_equal(dataset[1][0], [0, 0, 1, 1])
    np.testing.assert_array_equal(dataset[1][1], [[1]])
    assert dataset[1][0].dtype == dataset[1][1].dtype == np.float32
    np.testing.assert_array_equal(episode.states, before_states)
    np.testing.assert_array_equal(episode.actions, before_actions)


@pytest.mark.parametrize(
    "policy_config",
    [
        MSEBCTrainConfig(hidden_dims=(16,)),
        FlowBCTrainConfig(policy_type="flow", hidden_dims=(16,)),
        DiffusionBCTrainConfig(policy_type="diffusion", down_dims=(16, 32), time_embed_dim=16),
    ],
)
@pytest.mark.parametrize("chunk_size", [1, 3, 16])
def test_one_epoch_bc_uses_flattened_history_and_raw_frame_stats(
    policy_config, chunk_size, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rng = np.random.default_rng(7)
    episode = Episode(
        states=rng.normal(size=(5, 54)).astype(np.float32),
        actions=rng.normal(size=(4, 8)).astype(np.float32),
        metadata={
            "replay": {"physics_steps_per_action": 1},
            "observation": {"frame_dim": 54, "rotation_indices": []},
        },
    )
    initial_states, initial_actions = episode.states.copy(), episode.actions.copy()
    validation = Episode(
        states=np.full((3, 54), 1000.0),
        actions=np.zeros((2, 8)),
        metadata={
            "replay": {"physics_steps_per_action": 1},
            "observation": {"frame_dim": 54, "rotation_indices": []},
        },
    )
    episode.validate_training_data()
    validation.validate_training_data()
    config = replace(
        policy_config,
        obs_horizon=2,
        chunk_size=chunk_size,
        execution_horizon=1,
        action_execution_hz=500,
        batch_size=4,
        num_epochs=1,
        log_interval_steps=100,
    )

    model, normalizer = run_training(config, [episode], [validation])

    assert model.state_dim == 108 and model.action_dim == 8 and model.chunk_size == chunk_size
    assert normalizer.state_mean.shape == (54,) and normalizer.action_mean.shape == (8,)
    np.testing.assert_allclose(
        normalizer.state_mean, episode.states[:-1].mean(axis=0, dtype=np.float64)
    )
    state, _, _ = ChunkDataset([episode], 1, normalizer, obs_horizon=2)[0]
    with torch.no_grad():
        actions = model.sample_actions(torch.from_numpy(state[None]))
    assert actions.shape == (1, chunk_size, 8)
    assert torch.isfinite(actions).all()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    np.testing.assert_array_equal(episode.states, initial_states)
    np.testing.assert_array_equal(episode.actions, initial_actions)
