"""Streaming collection and safe resumption without a physics simulation."""

from unittest.mock import Mock
from zipfile import BadZipFile

import numpy as np
import pytest

from mujoco_lab.learning.datasets.episode import Episode, save_episode
from mujoco_lab.learning.rollout import iter_episodes


def episode(seed, replay=None):
    metadata = {"seed": seed, "success": seed % 2 == 0}
    if replay is not None:
        metadata["replay"] = replay
    return Episode(states=np.zeros((2, 1)), actions=np.zeros((1, 1)), metadata=metadata)


def test_interrupted_collection_resumes_absolute_indices_without_changes(tmp_path, monkeypatch):
    replay = {"scene": "cube_stack", "model_sha256": "same"}
    calls = []

    def collect(_env, _expert, *, seed, **_kwargs):
        calls.append(seed)
        if seed == 12 and calls.count(12) == 1:
            raise RuntimeError("interrupted")
        return episode(seed, replay)

    monkeypatch.setattr("mujoco_lab.learning.rollout.collector.collect_episode", collect)
    with pytest.raises(RuntimeError, match="interrupted"):
        list(iter_episodes(Mock(), Mock(), 4, seed=10, output_dir=tmp_path, replay_metadata=replay))
    originals = [path.read_bytes() for path in sorted(tmp_path.glob("*.npz"))]
    resumed = list(
        iter_episodes(
            Mock(), Mock(), 4, seed=10, output_dir=tmp_path, resume=True, replay_metadata=replay
        )
    )
    assert [item.metadata["seed"] for item in resumed] == [12, 13]
    assert calls == [10, 11, 12, 12, 13]
    assert [path.read_bytes() for path in sorted(tmp_path.glob("*.npz"))[:2]] == originals
    assert (
        list(
            iter_episodes(
                Mock(), Mock(), 4, seed=10, output_dir=tmp_path, resume=True, replay_metadata=replay
            )
        )
        == []
    )
    assert (
        list(
            iter_episodes(
                Mock(), Mock(), 3, seed=10, output_dir=tmp_path, resume=True, replay_metadata=replay
            )
        )
        == []
    )
    assert calls == [10, 11, 12, 12, 13]


def test_resume_rejects_missing_seed_gap_corruption_and_replay_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mujoco_lab.learning.rollout.collector.collect_episode",
        lambda _env, _expert, *, seed, **_kwargs: episode(
            seed, {"scene": "cube_stack", "model_sha256": "same"}
        ),
    )
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        list(iter_episodes(Mock(), Mock(), 2, seed=10, output_dir=missing, resume=True))
    directory = tmp_path / "episodes"
    list(iter_episodes(Mock(), Mock(), 2, seed=10, output_dir=directory))
    with pytest.raises(FileExistsError):
        list(iter_episodes(Mock(), Mock(), 3, seed=10, output_dir=directory))
    with pytest.raises(ValueError, match="different seed"):
        list(iter_episodes(Mock(), Mock(), 3, seed=11, output_dir=directory, resume=True))
    with pytest.raises(ValueError, match="different replay metadata"):
        list(
            iter_episodes(
                Mock(),
                Mock(),
                3,
                seed=10,
                output_dir=directory,
                resume=True,
                replay_metadata={"scene": "book_insertion", "model_sha256": "same"},
            )
        )
    assert list(iter_episodes(Mock(), Mock(), 1, seed=10, output_dir=directory, resume=True)) == []
    (directory / "episode_000001.npz").rename(directory / "episode_000002.npz")
    with pytest.raises(ValueError, match="gap"):
        list(iter_episodes(Mock(), Mock(), 3, seed=10, output_dir=directory, resume=True))
    (directory / "episode_000002.npz").rename(directory / "episode_000001.npz")
    (directory / "episode_000001.npz").write_bytes(b"incomplete archive")
    with pytest.raises(BadZipFile):
        list(iter_episodes(Mock(), Mock(), 3, seed=10, output_dir=directory, resume=True))


def test_resume_accepts_auxiliary_metadata_changes(tmp_path, monkeypatch):
    recorded = {
        "scene": "cube_stack",
        "model_sha256": "same",
        "visual_sha256": "legacy",
        "mujoco_version": "older",
    }
    current = {"scene": "cube_stack", "model_sha256": "same", "mujoco_version": "newer"}
    path = tmp_path / "episode_000000.npz"
    save_episode(path, episode(10, recorded))
    original = path.read_bytes()
    monkeypatch.setattr(
        "mujoco_lab.learning.rollout.collector.collect_episode",
        lambda _env, _expert, *, seed, replay_metadata, **_kwargs: episode(seed, replay_metadata),
    )
    resumed = list(
        iter_episodes(
            Mock(), Mock(), 2, seed=10, output_dir=tmp_path, resume=True, replay_metadata=current
        )
    )
    assert resumed[0].metadata["seed"] == 11
    assert resumed[0].metadata["replay"] == current
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    ("field", "recorded", "current"),
    [
        ("robot", "panda", "forte"),
        ("cubes", 1, 2),
        ("book", "small", "medium"),
        ("gripper_action_units", "joint_position", "opening_width_m"),
        ("dt", 0.001, 0.002),
        ("physics_steps_per_action", 1, 5),
        ("xy_range", 0.02, 0.04),
        ("cube_yaw_range_degrees", 0.0, 45.0),
        ("model_sha256", "old", "new"),
    ],
)
def test_resume_rejects_core_setting_and_model_changes(tmp_path, field, recorded, current):
    save_episode(tmp_path / "episode_000000.npz", episode(10, {field: recorded}))
    with pytest.raises(ValueError, match="different replay metadata"):
        list(
            iter_episodes(
                Mock(),
                Mock(),
                2,
                seed=10,
                output_dir=tmp_path,
                resume=True,
                replay_metadata={field: current},
            )
        )


def test_atomic_save_never_exposes_partial_final_or_replaces_existing(tmp_path, monkeypatch):
    path = tmp_path / "episode_000000.npz"
    original_save = np.savez_compressed

    def fail(file, **_arrays):
        file.write(b"partial")
        raise RuntimeError("write failed")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(RuntimeError, match="write failed"):
        save_episode(path, episode(10))
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(np, "savez_compressed", original_save)
    save_episode(path, episode(10))
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_episode(path, episode(11))
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
