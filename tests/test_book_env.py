"""Book learning observations, seeded physics resets, and joint-action cadence."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_book_insertion
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import BookTask, create_book_controller
from mujoco_lab.control import ControlTarget, min_jerk_target
from mujoco_lab.learning.envs import BookEnv
from mujoco_lab.learning.envs.book import book_observation_layout


def make_env(**kwargs):
    config = load_robot_config("forte")
    simulator = Simulator(
        create_book_insertion(), robots=[RobotSpec("forte", "forte", config=config)]
    )
    robot = simulator.robots["forte"]
    robot.change_controller(create_book_controller(robot, config.controller))
    return BookEnv(BookTask(simulator), **kwargs)


def test_reset_restores_physics_then_randomizes_and_resets_measurements():
    env = make_env(book_yaw_range_degrees=15)
    goal = env.data.site_xpos[env.task.target].copy()
    first, info = env.reset(seed=17)
    qpos = env.data.qpos.copy()
    target = env.robot.target.position.copy()
    env.data.qpos[env._book_qpos + 2] += 1
    env.data.qvel[:] = 1
    env.data.time = 4
    env.task._stable_since = 1
    second, repeat_info = env.reset(seed=17)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(env.data.qpos, qpos)
    np.testing.assert_array_equal(env.data.qvel, 0)
    np.testing.assert_array_equal(env.robot.target.position, target)
    np.testing.assert_array_equal(env.task.start_center, env.data.xpos[env.task.body])
    np.testing.assert_array_equal(env.data.site_xpos[env.task.target], goal)
    assert info == repeat_info
    assert env.data.time == 0
    assert env.task._stable_since is None
    assert np.all(np.abs(qpos[env._book_qpos : env._book_qpos + 2] - env._home_xy) <= 0.02)
    assert np.linalg.norm(qpos[env._book_qpos + 3 : env._book_qpos + 7]) == pytest.approx(1)
    assert abs(info["book_yaw_degrees"]) <= 15
    changed, _ = env.reset(seed=18)
    assert not np.array_equal(first, changed)


def test_book_observation_contains_live_goal_pose_and_offsets():
    env = make_env(xy_range=0, book_yaw_range_degrees=0)
    obs, _ = env.reset(seed=1)
    assert obs.dtype == np.float32
    assert obs.shape == (48,)
    width, rotations = book_observation_layout()
    assert env.observation_metadata == {"frame_dim": width, "rotation_indices": rotations}
    grasp = env.data.site_xpos[env._grasp_sites[env.robot.name]]
    book = env.data.xpos[env.task.body]
    goal = env.data.site_xpos[env.task.target]
    np.testing.assert_allclose(obs[15:18], grasp, atol=1e-7)
    np.testing.assert_allclose(obs[24:27], book, atol=1e-7)
    np.testing.assert_allclose(obs[33:36], goal, atol=1e-7)
    np.testing.assert_allclose(obs[42:45], book - grasp, atol=1e-7)
    np.testing.assert_allclose(obs[45:48], goal - book, atol=1e-7)
    for indices, matrix in zip(
        (rotations[:6], rotations[6:12], rotations[12:]),
        (
            env.data.site_xmat[env._grasp_sites[env.robot.name]],
            env.data.xmat[env.task.body],
            env.data.site_xmat[env.task.target],
        ),
        strict=True,
    ):
        np.testing.assert_allclose(obs[indices], matrix.reshape(3, 3)[:, :2].reshape(-1), atol=1e-7)
    assert np.isfinite(env.camera_lookat).all()


def test_book_action_preserves_target_derivatives_and_stops_on_success(monkeypatch):
    env = make_env(physics_steps_per_action=5, max_steps=1)
    env.reset(seed=4)
    start = ControlTarget(env.robot.target.position, np.full(7, 0.1), np.full(7, -0.2))
    env.robot.target = start
    action = np.r_[start.position - 0.001, 0.0]
    captured = []
    control = env.robot.control

    def record_control():
        captured.append(env.robot.target)
        return control()

    monkeypatch.setattr(env.robot, "control", record_control)
    calls = iter([False, True])
    monkeypatch.setattr(env.task, "status", lambda: SimpleNamespace(released_stable=next(calls)))
    _, reward, terminated, truncated, info = env.step(action)
    assert (reward, terminated, truncated) == (1.0, True, False)
    assert info["physics_steps"] == 2
    assert info["elapsed_dt"] == pytest.approx(2 * env.simulator.dt)
    for index, target in enumerate(captured):
        expected = min_jerk_target(
            start, ControlTarget(action[:7]), index * env.simulator.dt, env.action_dt
        )
        for field in ("position", "velocity", "acceleration"):
            np.testing.assert_allclose(getattr(target, field), getattr(expected, field))
    with pytest.raises(RuntimeError, match="Reset"):
        env.step(action)
    env.reset(seed=4)
    assert env._steps == 0 and not env._done
    assert env.robot.target.velocity is None and env.robot.target.acceleration is None


def test_hold_action_matches_simulator_physics_and_time_limit(monkeypatch):
    env = make_env(xy_range=0, physics_steps_per_action=5, max_steps=1)
    reference = make_env(xy_range=0)
    env.reset(seed=1)
    reference.reset(seed=1)
    monkeypatch.setattr(env.task, "status", lambda: SimpleNamespace(released_stable=False))
    action = np.r_[env.robot.target.position, 0.0]
    _, reward, terminated, truncated, info = env.step(action)
    reference.robot.gripper.set_target(0.0)
    reference.simulator.run_steps(5)
    mujoco.mj_forward(reference.model, reference.data)
    np.testing.assert_allclose(env.data.qpos, reference.data.qpos, atol=1e-12)
    np.testing.assert_allclose(env.data.qvel, reference.data.qvel, atol=1e-9)
    assert (reward, terminated, truncated) == (0.0, False, True)
    assert info["termination_reason"] == "time_limit"
    assert info["physics_steps"] == 5
    assert env.action_dt == pytest.approx(5 * env.simulator.dt)
