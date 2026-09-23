"""Recorded geometry round trips and native-viewer playback without physics."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, SimulatorManager, create_cube_stack
from mujoco_lab.control import create_controller
from mujoco_lab.learning.collect import create_expert
from mujoco_lab.learning.datasets.episode import Episode, load_episode, save_episode
from mujoco_lab.learning.datasets.replay import (
    capture_frame,
    cube_stack_metadata,
    replay_action_repeat,
)
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv
from mujoco_lab.learning.replay import EpisodeReplay
from mujoco_lab.learning.rollout import collect_episode, collect_episodes
from mujoco_lab.tasks import CubeStackTask, default_planning


@pytest.fixture(scope="module")
def recording():
    simulator = Simulator(create_cube_stack(2), robots=[RobotSpec("forte", "forte")])
    metadata = cube_stack_metadata(simulator, cubes=2, robot="forte")
    robot = simulator.robots["forte"]
    robot.change_controller(create_controller("pd", robot, frame="grasp"))
    task = CubeStackTask(simulator, 2)

    class MovingGoalEnv(CubeStackEnv):
        def step(self, action):
            self.data.mocap_pos[0, 0] += 0.001
            return super().step(action)

    env = MovingGoalEnv(task)
    expert = create_expert(task, planning=default_planning())
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


def test_replay_action_repeat_defaults_legacy_and_validates_new_metadata(recording):
    episode, _ = recording
    assert replay_action_repeat({}) == 1
    assert replay_action_repeat(episode.metadata["replay"]) == 1
    for invalid in (True, 0, -1, 1.5, "5"):
        with pytest.raises(ValueError, match="physics_steps_per_action"):
            replay_action_repeat({"physics_steps_per_action": invalid})


def test_replay_accepts_partial_final_action_at_recorded_physics_tick(recording):
    episode, _ = recording
    metadata = deepcopy(episode.metadata)
    metadata["replay"]["physics_steps_per_action"] = 5
    frame_times = np.array([0.0, 0.01, 0.02, 0.03, 0.034])
    terminated = episode.terminated.copy()
    terminated[-1] = True
    episode = replace(episode, metadata=metadata, frame_times=frame_times, terminated=terminated)
    replay = EpisodeReplay(episode)
    assert replay.frame_dt == pytest.approx(0.01)
    np.testing.assert_array_equal(replay.frame_times, frame_times)
    replay.set_frame(4)
    assert replay.simulator.data.time == pytest.approx(0.034)
    with pytest.raises(ValueError, match="uniformly timed"):
        EpisodeReplay(replace(episode, terminated=np.zeros_like(terminated)))


@pytest.mark.parametrize("bad_field", ["qpos", "frame_times", "mocap_pos", "visual_sha256"])
def test_replay_rejects_incompatible_recordings(recording, bad_field):
    episode, _ = recording
    if bad_field == "visual_sha256":
        metadata = deepcopy(episode.metadata)
        metadata["replay"][bad_field] = "changed asset"
        episode = replace(episode, metadata=metadata)
        message = "model differs"
    elif bad_field == "frame_times":
        episode = replace(episode, frame_times=episode.frame_times * 2)
        message = "uniformly timed"
    else:
        episode = replace(episode, **{bad_field: getattr(episode, bad_field)[:-1]})
        message = bad_field
    with pytest.raises(ValueError, match=message):
        EpisodeReplay(episode)


def test_collect_episodes_forwards_recording_options(recording, tmp_path, monkeypatch):
    episode, _ = recording
    collect = Mock(return_value=episode)
    monkeypatch.setattr("mujoco_lab.learning.rollout.collector.collect_episode", collect)
    record = Mock()
    collect_episodes(
        Mock(),
        Mock(),
        1,
        output_dir=tmp_path,
        record_frame=record,
        replay_metadata=episode.metadata["replay"],
    )
    assert collect.call_args.kwargs["record_frame"] is record
    assert collect.call_args.kwargs["replay_metadata"] == episode.metadata["replay"]
    np.testing.assert_array_equal(load_episode(tmp_path / "episode_000000.npz").qpos, episode.qpos)


def playback(
    monkeypatch,
    *,
    schedule=None,
    frame_count=5,
    speed=1.0,
    failure=None,
    frame_times=None,
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
            manager.show_replay("replay", frame_count, 0.1, set_frame, speed=speed)
    else:
        manager.show_replay(
            "replay",
            frame_count,
            0.1,
            set_frame,
            speed=speed,
            camera=camera,
            frame_times=frame_times,
        )
        assert viewer.cam.distance == camera.distance
    simulator.physics_step.assert_not_called()
    simulator.target_updater.assert_not_called()
    return simulator, viewer, frames


def test_playback_uses_elapsed_time_and_speed_without_physics(monkeypatch):
    simulator, viewer, frames = playback(monkeypatch, frame_count=10, speed=2)
    assert frames == [0, 0, 4, 9, 9]
    assert simulator._state.get_state() == "idle"
    viewer.close.assert_called_once()
    assert all(call.kwargs == {"state_only": True} for call in viewer.sync.call_args_list)


def test_playback_uses_actual_frame_times_for_partial_final_action(monkeypatch):
    schedule = [(0.0, []), (0.015, []), (0.032, []), (0.035, [])]
    _, _, frames = playback(
        monkeypatch,
        schedule=schedule,
        frame_count=5,
        frame_times=[0.2, 0.21, 0.22, 0.23, 0.234],
    )
    assert frames == [0, 0, 1, 3, 4]


def test_playback_pause_seek_bounds_rewind_and_restart(monkeypatch):
    schedule = [
        (0, [32]),  # Pause at frame zero.
        (1, [263]),  # Cannot step before zero.
        (2, [262, 262]),
        (3, [262, 262, 262]),  # Cannot step beyond final frame.
        (4, []),  # Seeking leaves playback paused.
        (5, [82]),  # R rewinds while paused.
        (6, [32]),  # Resume.
        (6.21, []),
        (8, []),  # Automatically pause at the end.
        (9, [32]),  # Space at end restarts.
        (9.11, []),
        (10, [268]),  # Home rewinds.
    ]
    _, _, frames = playback(monkeypatch, schedule=schedule)
    assert frames == [0, 0, 0, 2, 4, 4, 0, 0, 2, 4, 0, 1, 0]


@pytest.mark.parametrize("failure", ["launch", "frame", "sync", "close"])
def test_playback_closes_viewer_and_recovers_lifecycle_on_failure(monkeypatch, failure):
    simulator, viewer, _ = playback(monkeypatch, failure=failure)
    if failure != "launch":
        viewer.close.assert_called_once()
    assert simulator._state.get_state() == ("viewing" if failure == "close" else "idle")
    assert not simulator._stop_requested


@pytest.mark.parametrize(
    ("frame_count", "frame_dt", "speed"),
    [(0, 0.1, 1), (1, 0, 1), (1, float("nan"), 1), (1, 0.1, 0), (1, 0.1, float("inf"))],
)
def test_invalid_playback_parameters_leave_simulator_idle(frame_count, frame_dt, speed):
    simulator = Simulator(mujoco.MjSpec.from_string("<mujoco/>"))
    manager = SimulatorManager()
    manager.add_simulator("replay", simulator)
    with pytest.raises(ValueError):
        manager.show_replay("replay", frame_count, frame_dt, Mock(), speed=speed)
    assert simulator._state.get_state() == "idle"
