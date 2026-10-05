"""Streaming collection and safe resumption without a physics simulation."""

import gc
import weakref
from unittest.mock import Mock
from zipfile import BadZipFile

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
        load_episode(tmp_path / f"episode_{index:06d}.npz").metadata["seed"] for index in range(3)
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


def test_resume_rejects_missing_or_different_action_cadence(tmp_path, monkeypatch):
    replay = {"model": "same"}
    monkeypatch.setattr(
        "mujoco_lab.learning.rollout.collector.collect_episode",
        lambda _env, _expert, *, seed, **_kwargs: episode(seed, replay),
    )
    collect_episodes(Mock(), Mock(), 1, seed=10, output_dir=tmp_path)
    for repeat in (1, 5):
        with pytest.raises(ValueError, match="different replay metadata"):
            list(
                iter_episodes(
                    Mock(),
                    Mock(),
                    2,
                    seed=10,
                    output_dir=tmp_path,
                    resume=True,
                    replay_metadata={**replay, "physics_steps_per_action": repeat},
                )
            )


@pytest.mark.parametrize(
    ("setting", "changed"),
    [("cube_yaw_range_degrees", 45.0), ("xy_range", 0.04), ("min_gap", 0.02)],
)
def test_resume_requires_matching_randomization_metadata(tmp_path, setting, changed):
    from mujoco_lab.learning.rollout.collector import _resume_index

    recorded = {"model": "same", "cube_yaw_range_degrees": 0.0, "xy_range": 0.02, "min_gap": 0.01}
    save_episode(tmp_path / "episode_000000.npz", episode(10, recorded))
    assert _resume_index(tmp_path, 10, recorded) == 1
    with pytest.raises(ValueError, match="different replay metadata"):
        _resume_index(
            tmp_path, 10, {key: value for key, value in recorded.items() if key != setting}
        )
    with pytest.raises(ValueError, match="different replay metadata"):
        _resume_index(tmp_path, 10, {**recorded, setting: changed})


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
    with pytest.raises(BadZipFile):
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
    ("method", "cube_yaw_range_degrees", "max_steps", "robot_name"),
    [
        ("heuristic", 0.0, 6000, "forte"),
        ("sampling", 15.0, 18000, "forte"),
        ("heuristic", 0.0, 6000, "panda"),
    ],
)
def test_cli_streams_resume_and_counts_only_new_episodes(
    tmp_path, monkeypatch, capsys, method, cube_yaw_range_degrees, max_steps, robot_name
):
    from mujoco_lab.learning import collect_cube as cli

    config = cli.Config(
        count=4,
        seed=10,
        output_dir=tmp_path,
        resume=True,
        robot=robot_name,
        method=method,
        cube_yaw_range_degrees=cube_yaw_range_degrees,
        xy_range=0.04,
        min_gap=0.02,
    )
    simulator = Mock()
    simulator.robots = {robot_name: Mock(robot_type=robot_name)}
    iteration = Mock(return_value=iter([episode(12), episode(13)]))
    monkeypatch.setattr(cli.tyro, "cli", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(cli, "Simulator", lambda *_args, **_kwargs: simulator)
    monkeypatch.setattr(cli, "create_cube_stack", Mock())
    metadata = Mock(return_value={"model": "same"})
    monkeypatch.setattr(cli, "cube_stack_metadata", metadata)
    monkeypatch.setattr(cli, "create_controller", Mock())
    monkeypatch.setattr(cli, "CubeStackTask", Mock())
    monkeypatch.setattr(cli, "CubeStackEnv", Mock())
    monkeypatch.setattr(cli, "CubeStackExpert", Mock())
    monkeypatch.setattr(cli, "load_cube_recipe", Mock())
    monkeypatch.setattr(cli, "iter_episodes", iteration)

    cli.main()

    assert iteration.call_args.args[2] == 4
    assert iteration.call_args.kwargs["resume"] is True
    assert iteration.call_args.kwargs["seed"] == 10
    assert iteration.call_args.kwargs["max_steps"] == max_steps
    assert iteration.call_args.kwargs["replay_metadata"] == {"model": "same"}
    assert (
        metadata.call_args.kwargs["xy_range"]
        == cli.CubeStackEnv.call_args.kwargs["xy_range"]
        == 0.04
    )
    assert (
        metadata.call_args.kwargs["min_gap"] == cli.CubeStackEnv.call_args.kwargs["min_gap"] == 0.02
    )
    assert cli.CubeStackEnv.call_args.kwargs["cube_yaw_range_degrees"] == cube_yaw_range_degrees
    assert cli.CubeStackExpert.call_args.kwargs["method"] == method
    cli.load_cube_recipe.assert_called_once_with(robot_name)
    assert cli.CubeStackExpert.call_args.kwargs["recipe"] is cli.load_cube_recipe.return_value
    if method == "sampling":
        assert cli.CubeStackExpert.call_args.kwargs["planning"] is config.planning
    else:
        assert cli.CubeStackExpert.call_args.kwargs["planning"] is None
    cli.create_controller.assert_called_once_with(
        simulator.robots[robot_name], cli.load_robot_config(robot_name).controller
    )
    simulator.robots[robot_name].change_controller.assert_called_once_with(
        cli.create_controller.return_value
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    output = captured.err
    assert "Saved seed=12 success=True length=1; new episodes: 1, new successes: 1" in output
    assert "Saved 2 new episodes" in output
