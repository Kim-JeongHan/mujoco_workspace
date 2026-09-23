"""Streaming collection and safe resumption without a physics simulation."""

import gc
import weakref
from unittest.mock import Mock

import numpy as np
import pytest

from mujoco_lab.learning.datasets.episode import Episode, load_episode, save_episode
from mujoco_lab.learning.rollout import collect_episodes, iter_episodes


def episode(seed, replay=None):
    metadata = {"seed": seed, "success": seed % 2 == 0}
    if replay is not None:
        metadata["replay"] = replay
    return Episode(states=np.zeros((2, 1)), actions=np.zeros((1, 1)), metadata=metadata)


def test_stream_releases_previous_episode_and_list_api_remains(tmp_path, monkeypatch):
    references = []
    streaming = True

    def collect(_env, _expert, *, seed, **_kwargs):
        if streaming:
            gc.collect()
            assert all(reference() is None for reference in references)
        result = episode(seed)
        references.append(weakref.ref(result))
        return result

    monkeypatch.setattr("mujoco_lab.learning.rollout.collector.collect_episode", collect)
    for item in iter_episodes(Mock(), Mock(), 3, seed=10, output_dir=tmp_path):
        assert item.metadata["seed"] in (10, 11, 12)
        del item
    assert [
        load_episode(tmp_path / f"episode_{index:06d}.npz").metadata["seed"]
        for index in range(3)
    ] == [10, 11, 12]
    streaming = False
    retained = collect_episodes(Mock(), Mock(), 2, seed=20)
    assert [item.metadata["seed"] for item in retained] == [20, 21]


def test_interrupted_collection_resumes_absolute_indices_without_changes(tmp_path, monkeypatch):
    replay = {"model": "same"}
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
    resumed = collect_episodes(
        Mock(), Mock(), 4, seed=10, output_dir=tmp_path, resume=True, replay_metadata=replay
    )
    assert [item.metadata["seed"] for item in resumed] == [12, 13]
    assert calls == [10, 11, 12, 12, 13]
    assert [path.read_bytes() for path in sorted(tmp_path.glob("*.npz"))[:2]] == originals
    assert list(
        iter_episodes(
            Mock(), Mock(), 4, seed=10, output_dir=tmp_path, resume=True, replay_metadata=replay
        )
    ) == []
    assert list(
        iter_episodes(
            Mock(), Mock(), 3, seed=10, output_dir=tmp_path, resume=True, replay_metadata=replay
        )
    ) == []
    assert calls == [10, 11, 12, 12, 13]


def test_resume_accepts_legacy_repeat_one_but_rejects_new_cadence(tmp_path, monkeypatch):
    replay = {"model": "same"}
    monkeypatch.setattr(
        "mujoco_lab.learning.rollout.collector.collect_episode",
        lambda _env, _expert, *, seed, **_kwargs: episode(seed, replay),
    )
    collect_episodes(Mock(), Mock(), 1, seed=10, output_dir=tmp_path)
    assert list(
        iter_episodes(
            Mock(), Mock(), 1, seed=10, output_dir=tmp_path, resume=True,
            replay_metadata={**replay, "physics_steps_per_action": 1},
        )
    ) == []
    with pytest.raises(ValueError, match="different replay metadata"):
        list(
            iter_episodes(
                Mock(), Mock(), 2, seed=10, output_dir=tmp_path, resume=True,
                replay_metadata={**replay, "physics_steps_per_action": 5},
            )
        )


def test_resume_accepts_legacy_zero_yaw_but_rejects_changed_range(tmp_path):
    from mujoco_lab.learning.rollout.collector import _resume_index

    recorded = {"model": "same"}
    save_episode(tmp_path / "episode_000000.npz", episode(10, recorded))
    assert _resume_index(tmp_path, 10, {**recorded, "cube_yaw_range_degrees": 0.0}) == 1
    with pytest.raises(ValueError, match="different replay metadata"):
        _resume_index(tmp_path, 10, {**recorded, "cube_yaw_range_degrees": 45.0})


def test_resume_rejects_missing_seed_gap_corruption_and_replay_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mujoco_lab.learning.rollout.collector.collect_episode",
        lambda _env, _expert, *, seed, **_kwargs: episode(seed, {"model": "same"}),
    )
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        list(iter_episodes(Mock(), Mock(), 2, seed=10, output_dir=missing, resume=True))
    directory = tmp_path / "episodes"
    collect_episodes(Mock(), Mock(), 2, seed=10, output_dir=directory)
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
                replay_metadata={"model": "other"},
            )
        )
    assert list(iter_episodes(Mock(), Mock(), 1, seed=10, output_dir=directory, resume=True)) == []
    (directory / "episode_000001.npz").rename(directory / "episode_000002.npz")
    with pytest.raises(ValueError, match="gap"):
        list(iter_episodes(Mock(), Mock(), 3, seed=10, output_dir=directory, resume=True))
    (directory / "episode_000002.npz").rename(directory / "episode_000001.npz")
    (directory / "episode_000001.npz").write_bytes(b"incomplete archive")
    with pytest.raises(ValueError, match="Invalid episode file"):
        list(iter_episodes(Mock(), Mock(), 3, seed=10, output_dir=directory, resume=True))


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


@pytest.mark.parametrize(
    ("method", "cube_yaw_range_degrees", "max_steps"),
    [("heuristic", 0.0, 6000), ("sampling", 15.0, 18000)],
)
def test_cli_streams_resume_and_counts_only_new_episodes(
    tmp_path, monkeypatch, capsys, method, cube_yaw_range_degrees, max_steps
):
    from mujoco_lab.learning import collect as cli

    config = cli.Config(
        count=4,
        seed=10,
        output_dir=tmp_path,
        resume=True,
        method=method,
        cube_yaw_range_degrees=cube_yaw_range_degrees,
    )
    simulator = Mock()
    simulator.robots = {"forte": Mock()}
    iteration = Mock(return_value=iter([episode(12), episode(13)]))
    monkeypatch.setattr(cli.tyro, "cli", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(cli, "Simulator", lambda *_args, **_kwargs: simulator)
    monkeypatch.setattr(cli, "create_cube_stack", Mock())
    monkeypatch.setattr(cli, "cube_stack_metadata", lambda *_args, **_kwargs: {"model": "same"})
    monkeypatch.setattr(cli, "create_controller", Mock())
    monkeypatch.setattr(cli, "CubeStackTask", Mock())
    monkeypatch.setattr(cli, "CubeStackEnv", Mock())
    monkeypatch.setattr(cli, "create_expert", Mock())
    monkeypatch.setattr(cli, "iter_episodes", iteration)

    cli.main()

    assert iteration.call_args.args[2] == 4
    assert iteration.call_args.kwargs["resume"] is True
    assert iteration.call_args.kwargs["seed"] == 10
    assert iteration.call_args.kwargs["max_steps"] == max_steps
    assert iteration.call_args.kwargs["replay_metadata"] == {"model": "same"}
    assert cli.CubeStackEnv.call_args.kwargs["cube_yaw_range_degrees"] == cube_yaw_range_degrees
    assert cli.create_expert.call_args.kwargs["method"] == method
    captured = capsys.readouterr()
    assert captured.out == ""
    output = captured.err
    assert "Saved seed=12 success=True length=1; new episodes: 1, new successes: 1" in output
    assert "Saved 2 new episodes" in output
