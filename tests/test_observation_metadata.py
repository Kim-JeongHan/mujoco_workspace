"""Observation metadata selects rotation passthrough across scenes."""

import numpy as np
import pytest
import torch

from mujoco_lab.learning.checkpoint import load_checkpoint, save_checkpoint
from mujoco_lab.learning.config.config import TrainConfig
from mujoco_lab.learning.datasets.episode import Episode, load_episode, save_episode
from mujoco_lab.learning.trainers.train_bc import _observation_rotation_indices, run_training


def _episode():
    return Episode(
        states=np.arange(192, dtype=np.float32).reshape(4, 48),
        actions=np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        metadata={
            "replay": {"scene": "book"},
            "observation": {
                "frame_dim": 48,
                "rotation_indices": [*range(18, 24), *range(27, 33), *range(36, 42)],
            },
        },
    )


class SilentLogger:
    def log(self, values, *, step):
        pass


@pytest.mark.parametrize("policy_type", ["mse", "flow"])
def test_metadata_roundtrip_training_and_checkpoint(tmp_path, monkeypatch, policy_type):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch,
        "compile",
        lambda function=None, **kwargs: function if function is not None else lambda f: f,
    )
    episode = _episode()
    path = tmp_path / "episode.npz"
    save_episode(path, episode)
    restored_episode = load_episode(path)
    assert restored_episode.metadata == episode.metadata
    config = TrainConfig(
        policy_type=policy_type,
        obs_horizon=2,
        chunk_size=1,
        execution_horizon=1,
        hidden_dims=(8,),
        batch_size=3,
        num_epochs=1,
        log_interval=100,
        eval_interval=0,
    )
    model, normalizer = run_training(
        config, [restored_episode], [restored_episode], logger=SilentLogger()
    )
    indices = episode.metadata["observation"]["rotation_indices"]
    np.testing.assert_array_equal(normalizer.state_mean[indices], 0)
    np.testing.assert_array_equal(normalizer.state_std[indices], 1)
    np.testing.assert_array_equal(
        normalizer.normalize_state(episode.states)[..., indices], episode.states[..., indices]
    )
    checkpoint = tmp_path / "policy.pt"
    save_checkpoint(checkpoint, model, normalizer, config, optimizer_step=1)
    _, restored, _ = load_checkpoint(checkpoint)
    np.testing.assert_array_equal(restored.state_mean, normalizer.state_mean)
    np.testing.assert_array_equal(restored.state_std, normalizer.state_std)


@pytest.mark.parametrize(
    "observation",
    [
        None,
        {},
        {"frame_dim": 49, "rotation_indices": []},
        {"frame_dim": 48, "rotation_indices": [48]},
        {"frame_dim": 48, "rotation_indices": [-1]},
        {"frame_dim": 48, "rotation_indices": [2, 2]},
        {"frame_dim": 48, "rotation_indices": [True]},
        {"frame_dim": 48, "rotation_indices": [2.0]},
    ],
)
def test_invalid_observation_contract(observation):
    episode = _episode()
    episode.metadata["observation"] = observation
    with pytest.raises(ValueError, match="Observation"):
        _observation_rotation_indices(episode, 48)


@pytest.mark.parametrize("split", ["train", "validation"])
def test_mixed_layouts_fail_before_training(split):
    episode, different = _episode(), _episode()
    different.metadata["observation"]["rotation_indices"] = []
    train = [episode, different] if split == "train" else [episode]
    validation = [different] if split == "validation" else [episode]
    with pytest.raises(ValueError, match="layouts must match"):
        run_training(TrainConfig(), train, validation)


@pytest.mark.parametrize("scene", ["book", "cube_stack"])
def test_missing_observation_metadata_does_not_guess_layout(scene):
    episode = _episode()
    del episode.metadata["observation"]
    episode.metadata["replay"] = {"scene": scene, "cubes": 2}
    assert _observation_rotation_indices(episode, 48) == []


def test_collector_copies_observation_metadata(tmp_path):
    from mujoco_lab.learning.rollout.collector import collect_episode

    class Environment:
        observation_metadata = _episode().metadata["observation"]

        def reset(self, *, seed, options):
            return np.zeros(48, dtype=np.float32), {}

        def step(self, action):
            return np.ones(48, dtype=np.float32), 0.0, True, False, {"success": True}

    class Expert:
        failed = False

        def reset(self, observation, info):
            pass

        def act(self, observation):
            return np.zeros(1, dtype=np.float32)

    env = Environment()
    episode = collect_episode(env, Expert())
    assert episode.metadata["observation"] == env.observation_metadata
    env.observation_metadata["rotation_indices"].clear()
    assert episode.metadata["observation"]["rotation_indices"]
    path = tmp_path / "collected.npz"
    save_episode(path, episode)
    assert load_episode(path).metadata["observation"] == episode.metadata["observation"]


@pytest.mark.parametrize("feature", ["states", "actions"])
def test_mixed_feature_dimensions_fail_before_training(feature):
    episode, different = _episode(), _episode()
    values = getattr(different, feature)
    setattr(different, feature, np.column_stack([values, np.zeros(len(values))]))
    with pytest.raises(ValueError, match="feature dimensions must match"):
        run_training(TrainConfig(), [episode], [different])
