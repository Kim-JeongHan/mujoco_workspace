"""Causal observation histories for offline action chunks."""

import numpy as np
import pytest
import torch

from mujoco_lab.learning.config.config import TrainConfig
from mujoco_lab.learning.datasets.episode import Episode
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.datasets.sequence import ChunkDataset
from mujoco_lab.learning.trainers.train_bc import run_training


def test_history_pads_first_frame_and_never_uses_future_state():
    episode = Episode(
        states=np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.float64),
        actions=np.array([[100], [200], [300]], dtype=np.float64),
    )
    dataset = ChunkDataset([episode], chunk_size=1)

    np.testing.assert_array_equal(dataset[0][0], [1, 10, 1, 10])
    np.testing.assert_array_equal(dataset[1][0], [1, 10, 2, 20])
    np.testing.assert_array_equal(dataset[2][0], [2, 20, 3, 30])
    assert [dataset[index][1].item() for index in range(len(dataset))] == [100, 200, 300]
    assert dataset[0][0].dtype == dataset[0][1].dtype == np.float32
    np.testing.assert_array_equal(episode.states[-1], [4, 40])


def test_chunks_align_with_current_history_and_do_not_cross_episodes():
    first = Episode(states=np.array([[1], [2], [3], [4]]), actions=np.array([[10], [11], [12]]))
    second = Episode(states=np.array([[50], [60], [70]]), actions=np.array([[20], [21]]))
    dataset = ChunkDataset([first, second], chunk_size=2, obs_horizon=3)

    assert len(dataset) == 3
    np.testing.assert_array_equal(dataset[0][0], [1, 1, 1])
    np.testing.assert_array_equal(dataset[0][1], [[10], [11]])
    np.testing.assert_array_equal(dataset[1][0], [1, 1, 2])
    np.testing.assert_array_equal(dataset[1][1], [[11], [12]])
    np.testing.assert_array_equal(dataset[2][0], [50, 50, 50])
    np.testing.assert_array_equal(dataset[2][1], [[20], [21]])


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


def test_horizon_one_and_positive_size_guards():
    episode = Episode(states=np.array([[3], [4]]), actions=np.array([[9]]))
    state, action = ChunkDataset([episode], chunk_size=1, obs_horizon=1)[0]
    np.testing.assert_array_equal(state, [3])
    np.testing.assert_array_equal(action, [[9]])
    for chunk_size, obs_horizon in ((0, 1), (1, 0), (-1, 2)):
        with pytest.raises(ValueError, match="must be positive"):
            ChunkDataset([episode], chunk_size=chunk_size, obs_horizon=obs_horizon)


@pytest.mark.parametrize("policy_type", ["mse", "flow"])
def test_one_epoch_bc_uses_flattened_history_and_raw_frame_stats(policy_type, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rng = np.random.default_rng(7)
    episode = Episode(
        states=rng.normal(size=(5, 54)).astype(np.float32),
        actions=rng.normal(size=(4, 8)).astype(np.float32),
    )
    initial_states, initial_actions = episode.states.copy(), episode.actions.copy()
    validation = Episode(states=np.full((3, 54), 1000.0), actions=np.zeros((2, 8)))
    config = TrainConfig(
        policy_type=policy_type,
        obs_horizon=2,
        chunk_size=1,
        execution_horizon=1,
        physics_steps_per_action=1,
        hidden_dims=(16,),
        batch_size=4,
        num_epochs=1,
        log_interval=100,
    )

    model, normalizer = run_training(config, [episode], [validation])

    assert model.state_dim == 108 and model.action_dim == 8 and model.chunk_size == 1
    assert normalizer.state_mean.shape == (54,) and normalizer.action_mean.shape == (8,)
    np.testing.assert_allclose(normalizer.state_mean, episode.states[:-1].mean(axis=0))
    assert "step=1" in capsys.readouterr().err
    state, _ = ChunkDataset([episode], 1, normalizer, obs_horizon=2)[0]
    with torch.no_grad():
        actions = model.sample_actions(torch.from_numpy(state[None]))
    assert actions.shape == (1, 1, 8)
    assert torch.isfinite(actions).all()
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    np.testing.assert_array_equal(episode.states, initial_states)
    np.testing.assert_array_equal(episode.actions, initial_actions)
