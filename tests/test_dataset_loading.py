"""Load and validate offline demonstrations."""

import numpy as np
import pytest

from mujoco_lab.learning.datasets import load_episodes
from mujoco_lab.learning.datasets.episode import Episode, save_episode


def make_episode(seed, *, success=True, steps=2, state_dim=54, action_dim=8):
    return Episode(
        states=np.zeros((steps + 1, state_dim), dtype=np.float32),
        actions=np.zeros((steps, action_dim), dtype=np.float32),
        metadata={"seed": seed, "success": success},
    )


def test_load_sorted_successes_and_preserve_episode_contents(tmp_path):
    episodes = [
        make_episode(3, success=None),
        make_episode(2, success=True),
        make_episode(1, success=False),
        make_episode(4, success=True),
    ]
    episodes[1].rewards = np.array([0.0, 1.0])
    episodes[1].qpos = np.ones((3, 23))
    for name, episode in zip(("c", "b", "a", "d"), episodes, strict=True):
        save_episode(tmp_path / f"{name}.npz", episode)

    successful = load_episodes(tmp_path)
    assert [episode.metadata["seed"] for episode in successful] == [2, 4]
    np.testing.assert_array_equal(successful[0].rewards, episodes[1].rewards)
    np.testing.assert_array_equal(successful[0].qpos, episodes[1].qpos)
    all_episodes = load_episodes(tmp_path, success_only=False)
    assert [episode.metadata["seed"] for episode in all_episodes] == [
        1,
        2,
        3,
        4,
    ]


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda episode: setattr(episode, "states", episode.states[:-1]), r"T\+1"),
        (lambda episode: setattr(episode, "actions", episode.actions[:0]), r"T\+1"),
        (lambda episode: episode.states.__setitem__((0, 0), np.nan), "finite"),
        (lambda episode: setattr(episode, "actions", episode.actions.reshape(-1)), r"T\+1"),
    ],
)
def test_load_rejects_malformed_kept_episode(tmp_path, change, match):
    episode = make_episode(1)
    change(episode)
    path = tmp_path / "bad.npz"
    save_episode(path, episode)
    with pytest.raises(ValueError, match=match):
        load_episodes(tmp_path)


def test_load_rejects_mixed_feature_dimensions_and_missing_data(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        load_episodes(tmp_path / "missing")
    with pytest.raises(ValueError, match="No NPZ"):
        load_episodes(tmp_path)

    save_episode(tmp_path / "a.npz", make_episode(1))
    save_episode(tmp_path / "b.npz", make_episode(2, state_dim=53))
    with pytest.raises(ValueError, match="feature dimensions"):
        load_episodes(tmp_path)


def test_load_rejects_corrupt_file_and_no_successful_episodes(tmp_path):
    (tmp_path / "corrupt.npz").write_bytes(b"not a zip archive")
    with pytest.raises(ValueError):
        load_episodes(tmp_path)

    (tmp_path / "corrupt.npz").unlink()
    save_episode(tmp_path / "failure.npz", make_episode(1, success=False))
    with pytest.raises(ValueError, match="No eligible episodes"):
        load_episodes(tmp_path)
    assert len(load_episodes(tmp_path, success_only=False)) == 1
