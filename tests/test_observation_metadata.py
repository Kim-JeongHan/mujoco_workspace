"""Observation metadata selects rotation passthrough across scenes."""

import numpy as np
import pytest

from mujoco_lab.learning.config.config import TrainConfig
from mujoco_lab.learning.datasets.episode import Episode, load_episode, load_episodes, save_episode
from mujoco_lab.learning.trainers.train_bc import run_training


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


@pytest.mark.parametrize(
    "observation",
    [
        None,
        {},
        {"frame_dim": 49, "rotation_indices": []},
        {"frame_dim": 48, "rotation_indices": [48]},
        {"frame_dim": 48, "rotation_indices": [-1]},
        {"frame_dim": 48, "rotation_indices": [True]},
        {"frame_dim": 48, "rotation_indices": [2.0]},
    ],
)
def test_invalid_observation_contract_is_rejected_when_loading(observation, tmp_path):
    episode = _episode()
    episode.metadata["observation"] = observation
    save_episode(tmp_path / "episode.npz", episode)
    with pytest.raises(ValueError, match="Observation"):
        load_episodes(tmp_path, success_only=False)


@pytest.mark.parametrize("split", ["train", "validation"])
def test_mixed_layouts_fail_before_training(split):
    episode, different = _episode(), _episode()
    different.metadata["observation"]["rotation_indices"] = []
    episode.validate_training_data()
    different.validate_training_data()
    train = [episode, different] if split == "train" else [episode]
    validation = [different] if split == "validation" else [episode]
    with pytest.raises(ValueError, match="layouts must match"):
        run_training(TrainConfig(), train, validation)


@pytest.mark.parametrize("scene", ["book_insertion", "cube_stack"])
def test_missing_observation_metadata_is_rejected_when_loading(scene, tmp_path):
    episode = _episode()
    del episode.metadata["observation"]
    episode.metadata["replay"] = {"scene": scene, "cubes": 2}
    save_episode(tmp_path / "episode.npz", episode)
    with pytest.raises(ValueError, match="Observation metadata"):
        load_episodes(tmp_path, success_only=False)


def test_loading_accepts_repeated_rotation_columns_without_rewriting_metadata(tmp_path):
    episode = _episode()
    episode.metadata["observation"] = {"frame_dim": 48.0, "rotation_indices": [2, 2]}
    save_episode(tmp_path / "episode.npz", episode)
    loaded = load_episodes(tmp_path, success_only=False)[0]
    assert loaded.rotation_indices == [2, 2]
    assert loaded.metadata == episode.metadata


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

        def act(self, observation, *, dt=0.0):
            return np.zeros(1, dtype=np.float32)

    env = Environment()
    episode = collect_episode(env, Expert())
    assert episode.metadata["observation"] == env.observation_metadata
    env.observation_metadata["rotation_indices"].clear()
    assert episode.metadata["observation"]["rotation_indices"]
    path = tmp_path / "collected.npz"
    save_episode(path, episode)
    assert load_episode(path).metadata["observation"] == episode.metadata["observation"]
