"""Recorded geometry round trips and native-viewer playback without physics."""

from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest
from controller_config import create_test_controller

from mujoco_lab import RobotSpec, Simulator, SimulatorManager, create_cube_stack
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import CubeStackExpert, CubeStackTask
from mujoco_lab.behaviors.cube_stack_recipe import load_recipe as load_cube_recipe
from mujoco_lab.learning import replay as replay_cli
from mujoco_lab.learning.config.replay import CubeStackReplayConfig
from mujoco_lab.learning.datasets.episode import Episode, load_episode, save_episode
from mujoco_lab.learning.datasets.replay import (
    capture_frame,
    capture_metadata,
)
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv
from mujoco_lab.learning.replay import EpisodeReplay
from mujoco_lab.learning.rollout import collect_episode


@pytest.fixture(scope="module")
def recording():
    simulator = Simulator(
        create_cube_stack(2),
        robots=[RobotSpec("forte", "forte", config=load_robot_config("forte"))],
    )
    metadata = capture_metadata(simulator, CubeStackReplayConfig(cubes=2, robot="forte"))
    robot = simulator.robots["forte"]
    robot.change_controller(create_test_controller(robot, controller="pd", frame="grasp"))
    task = CubeStackTask(simulator, 2)

    class MovingGoalEnv(CubeStackEnv):
        def step(self, action):
            self.data.mocap_pos[0, 0] += 0.001
            return super().step(action)

    env = MovingGoalEnv(task)
    expert = CubeStackExpert(
        task,
        recipe=load_cube_recipe("forte"),
        method="heuristic",
    )
    geometry = []

    def record():
        geometry.append(simulator.data.geom_xpos.copy())
        return capture_frame(simulator)

    episode = collect_episode(
        env, expert, max_steps=4, seed=7, record_frame=record, replay_metadata=metadata
    )
    return episode, geometry


def test_recorded_episode_round_trip_and_seek_restore_scene(recording, tmp_path, monkeypatch):
    episode, geometry = recording
    path = tmp_path / "episode.npz"
    save_episode(path, episode)
    loaded = load_episode(path)
    assert loaded.metadata == episode.metadata
    for name in ("qpos", "frame_times", "mocap_pos", "mocap_quat", "states", "actions"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(episode, name))
    assert loaded.qpos is not None
    assert loaded.frame_times is not None
    assert loaded.mocap_pos is not None
    assert loaded.mocap_quat is not None
    assert loaded.qpos.shape[0] == loaded.states.shape[0] == len(loaded) + 1
    assert loaded.qpos.shape[1] != loaded.states.shape[1]
    assert np.ptp(loaded.mocap_pos[:, 0, 0]) > 0
    replay = EpisodeReplay(loaded)
    monkeypatch.setattr(replay.simulator, "physics_step", Mock(side_effect=AssertionError))
    monkeypatch.setattr(mujoco, "mj_step", Mock(side_effect=AssertionError))
    for index in (4, 0, 2, 1):
        replay.set_frame(index)
        data = replay.simulator.data
        np.testing.assert_array_equal(data.qpos, loaded.qpos[index])
        np.testing.assert_array_equal(data.mocap_pos, loaded.mocap_pos[index])
        np.testing.assert_array_equal(data.mocap_quat, loaded.mocap_quat[index])
        np.testing.assert_allclose(data.geom_xpos, geometry[index], atol=1e-12)
        assert data.time == loaded.frame_times[index]
        assert not data.qvel.any()
    for index in (-1, replay.frame_count):
        with pytest.raises(IndexError):
            replay.set_frame(index)
    with pytest.raises(FileExistsError):
        save_episode(path, episode)


def test_old_episode_still_loads_for_learning_but_cannot_replay(tmp_path):
    episode = Episode(states=np.zeros((3, 5)), actions=np.zeros((2, 2)))
    path = tmp_path / "old.npz"
    save_episode(path, episode)
    loaded = load_episode(path)
    assert len(loaded) == 2
    assert loaded.qpos is None
    with pytest.raises(ValueError, match="learning states are not MuJoCo qpos"):
        EpisodeReplay(loaded)


def test_replay_rejects_a_different_recorded_model(recording):
    episode, _ = recording
    metadata = {
        **episode.metadata,
        "replay": {**episode.metadata["replay"], "model_sha256": "0" * 64},
    }
    with pytest.raises(ValueError, match="recorded robot model"):
        EpisodeReplay(replace(episode, metadata=metadata))


@pytest.mark.parametrize("missing", ["frame_times", "mocap_pos", "mocap_quat"])
def test_replay_rejects_incomplete_recorded_frames(recording, missing):
    episode, _ = recording
    with pytest.raises(ValueError, match="no replay data"):
        EpisodeReplay(replace(episode, **{missing: None}))


def shorter_recording(episode):
    """Create a compatible shorter episode with visibly different native frames."""
    return replace(
        episode,
        states=episode.states[:3].copy(),
        actions=episode.actions[:2].copy(),
        rewards=None,
        terminated=None,
        truncated=None,
        qpos=episode.qpos[:3].copy(),
        frame_times=episode.frame_times[:3].copy() + 10,
        mocap_pos=episode.mocap_pos[:3].copy() + 0.02,
        mocap_quat=episode.mocap_quat[:3].copy(),
    )


def test_switching_recordings_reuses_native_scene_and_restores_new_frames(recording):
    episode, _ = recording
    replay = EpisodeReplay(episode)
    simulator, model, data = replay.simulator, replay.simulator.model, replay.simulator.data
    shorter = shorter_recording(episode)
    replay.load_episode(shorter)
    assert replay.simulator is simulator
    assert replay.simulator.model is model
    assert replay.simulator.data is data
    assert replay.frame_count == 3
    replay.set_frame(2)
    np.testing.assert_array_equal(data.qpos, shorter.qpos[2])
    np.testing.assert_array_equal(data.mocap_pos, shorter.mocap_pos[2])
    assert data.time == shorter.frame_times[2]
    with pytest.raises(IndexError):
        replay.set_frame(3)
    replay.load_episode(episode)
    assert replay.frame_count == 5
    replay.set_frame(4)
    assert data.time == episode.frame_times[4]


def test_switching_recordings_rejects_incompatible_scene_before_replacing_frames(recording):
    episode, _ = recording
    replay = EpisodeReplay(episode)
    metadata = {**episode.metadata, "replay": {**episode.metadata["replay"], "cubes": 1}}
    with pytest.raises(ValueError, match="same recorded scene"):
        replay.load_episode(replace(episode, metadata=metadata))
    replay.set_frame(4)
    np.testing.assert_array_equal(replay.simulator.data.qpos, episode.qpos[4])


def test_directory_replay_loads_files_in_order_and_wraps_navigation(
    recording, tmp_path, monkeypatch
):
    episode, _ = recording
    shorter = shorter_recording(episode)
    save_episode(tmp_path / "episode_000001.npz", shorter)
    save_episode(tmp_path / "episode_000000.npz", episode)
    monkeypatch.setattr(
        replay_cli.tyro, "cli", lambda *_args, **_kwargs: replay_cli.Config(tmp_path)
    )
    calls = []

    def browse(manager, name, frame_count, frame_dt, set_frame, **options):
        simulator = manager.simulators[name]
        model, data = simulator.model, simulator.data
        assert frame_count == 5
        set_frame(0)
        assert data.time == episode.frame_times[0]
        switch = options["change_episode"]
        for direction, count, expected in ((1, 3, shorter), (1, 5, episode), (-1, 3, shorter)):
            new_count, new_dt, times = switch(direction)
            assert new_count == count
            assert new_dt == frame_dt
            np.testing.assert_array_equal(times, expected.frame_times)
            set_frame(0)
            np.testing.assert_array_equal(data.mocap_pos, expected.mocap_pos[0])
            assert simulator.model is model and simulator.data is data
            calls.append(direction)

    monkeypatch.setattr(SimulatorManager, "show_replay", browse)
    replay_cli.main()
    assert calls == [1, 1, -1]


def test_empty_replay_directory_has_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        replay_cli.tyro, "cli", lambda *_args, **_kwargs: replay_cli.Config(tmp_path)
    )
    with pytest.raises(ValueError, match="No episode NPZ"):
        replay_cli.main()


def playback(
    monkeypatch,
    *,
    schedule=None,
    frame_count=5,
    frame_dt=0.1,
    speed=1.0,
    failure=None,
    frame_times=None,
    change_episode=None,
):
    simulator = Simulator(mujoco.MjSpec.from_string("<mujoco/>"))
    simulator.physics_step = Mock(side_effect=AssertionError("Physics must not advance"))
    simulator.target_updater = Mock(side_effect=AssertionError("Control must not run"))
    manager = SimulatorManager()
    manager.add_simulator("replay", simulator)
    clock = [0.0]
    monkeypatch.setattr("mujoco_lab.simulator_manager.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("mujoco_lab.simulator_manager.time.sleep", lambda dt: None)
    schedule = schedule or [(0, []), (0.21, []), (1.0, []), (2.0, [])]
    steps = iter(schedule)
    viewer = Mock()
    viewer._sim = lambda: None
    viewer.lock.side_effect = nullcontext
    viewer.cam = mujoco.MjvCamera()
    callback = None
    frames = []

    def is_running():
        item = next(steps, None)
        if item is None:
            return False
        clock[0], keys = item
        assert callback is not None
        for key in keys:
            callback(key)
        return True

    def launch(*args, key_callback):
        nonlocal callback
        if failure == "launch":
            raise RuntimeError("launch failed")
        callback = key_callback
        return viewer

    def set_frame(index):
        frames.append(index)
        if failure == "frame" and len(frames) == 2:
            raise RuntimeError("frame failed")

    viewer.is_running.side_effect = is_running
    if failure == "sync":
        viewer.sync.side_effect = RuntimeError("sync failed")
    if failure == "close":
        viewer.close.side_effect = RuntimeError("close failed")
    monkeypatch.setattr("mujoco.viewer.launch_passive", launch)
    camera = mujoco.MjvCamera()
    camera.distance = 1.5
    if failure:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            manager.show_replay("replay", frame_count, frame_dt, set_frame, speed=speed)
    else:
        manager.show_replay(
            "replay",
            frame_count,
            frame_dt,
            set_frame,
            speed=speed,
            camera=camera,
            frame_times=frame_times,
            change_episode=change_episode,
        )
        assert viewer.cam.distance == camera.distance
    simulator.physics_step.assert_not_called()
    simulator.target_updater.assert_not_called()
    return simulator, viewer, frames


@pytest.mark.parametrize("frame_dt", [0.1, 0.0, -0.1, float("nan"), float("inf")])
def test_playback_uses_actual_frame_times_for_partial_final_action(monkeypatch, frame_dt):
    schedule = [(0.0, []), (0.015, []), (0.032, []), (0.035, [])]
    _, _, frames = playback(
        monkeypatch,
        schedule=schedule,
        frame_count=5,
        frame_dt=frame_dt,
        frame_times=[0.2, 0.21, 0.22, 0.23, 0.234],
    )
    assert frames == [0, 0, 1, 3, 4]


def test_next_previous_episode_keys_reset_playhead_and_use_new_timestamps(monkeypatch):
    directions = []

    def switch(direction):
        directions.append(direction)
        return (
            (3, 0.15, [20.0, 20.15, 20.31])
            if direction == 1
            else (5, 0.1, [0.0, 0.1, 0.2, 0.3, 0.4])
        )

    _, viewer, frames = playback(
        monkeypatch,
        schedule=[(0.0, []), (0.5, [298]), (0.7, []), (0.8, [297]), (1.01, [])],
        change_episode=switch,
    )
    assert directions == [1, -1]
    assert frames == [0, 0, 0, 1, 0, 2]
    viewer.close.assert_called_once()


def test_episode_change_failure_closes_viewer(monkeypatch):
    def fail_switch(direction):
        raise RuntimeError("episode change failed")

    with pytest.raises(RuntimeError, match="episode change failed"):
        playback(monkeypatch, schedule=[(0.0, [298])], change_episode=fail_switch)


def test_native_visualization_shortcuts_do_not_change_episodes(monkeypatch):
    switch = Mock()
    playback(monkeypatch, schedule=[(0.0, [78, 80]), (0.1, [])], change_episode=switch)
    switch.assert_not_called()


@pytest.mark.parametrize("failure", ["launch", "frame", "sync", "close"])
def test_playback_closes_viewer_and_recovers_lifecycle_on_failure(monkeypatch, failure):
    simulator, viewer, _ = playback(monkeypatch, failure=failure)
    if failure != "launch":
        viewer.close.assert_called_once()
    assert simulator._state.get_state() == "idle"
    assert not simulator._stop_requested
