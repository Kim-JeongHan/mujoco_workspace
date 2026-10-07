"""Action cadence keeps physical control fresh and episode steps distinct."""

from types import SimpleNamespace

import numpy as np
import pytest
from controller_config import create_test_controller

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import CubeStackTask
from mujoco_lab.control import ControlTarget, min_jerk
from mujoco_lab.learning.datasets.replay import capture_frame
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv
from mujoco_lab.learning.rollout import collect_episode


def make_env(repeat):
    simulator = Simulator(
        create_cube_stack(2),
        robots=[RobotSpec("forte", "forte", config=load_robot_config("forte"))],
    )
    robot = simulator.robots["forte"]
    robot.change_controller(create_test_controller(robot, controller="pd", frame="grasp"))
    task = CubeStackTask(simulator, 2)
    return CubeStackEnv(task, physics_steps_per_action=repeat, max_steps=2), robot


@pytest.mark.parametrize("repeat", [1, 5])
def test_action_interpolates_target_with_fresh_control_each_physics_tick(monkeypatch, repeat):
    env, robot = make_env(repeat)
    env.reset(seed=7)
    monkeypatch.setattr(env.task, "status", lambda: SimpleNamespace(released_stable_stack=False))
    start_target = robot.target
    action = np.r_[start_target.position, robot.gripper.get_target()]
    action[1] -= 0.08
    targets = []
    qpos_samples = []
    original_control = robot.control

    def control():
        targets.append(robot.target.position.copy())
        qpos_samples.append(robot.state.snapshot().qpos.copy())
        return original_control()

    monkeypatch.setattr(robot, "control", control)
    start_time = env.data.time
    obs, _, terminated, truncated, info = env.step(action)
    assert obs.shape == (54,)
    assert not terminated and not truncated
    assert env._steps == 1
    assert env.action_dt == pytest.approx(0.002 * repeat)
    assert env.data.time - start_time == pytest.approx(env.action_dt)
    assert info["physics_steps"] == repeat
    assert info["elapsed_dt"] == pytest.approx(env.action_dt)
    assert len(targets) == len(qpos_samples) == repeat
    for index, target in enumerate(targets):
        expected = min_jerk(
            start_target, ControlTarget(action[:7]), index * env.simulator.dt, env.action_dt
        )
        np.testing.assert_allclose(target, expected.position)
    endpoint = min_jerk(start_target, ControlTarget(action[:7]), env.action_dt, env.action_dt)
    np.testing.assert_allclose(robot.target.position, endpoint.position)
    np.testing.assert_allclose(robot.target.velocity, endpoint.velocity)
    assert robot.target.position[1] < start_target.position[1]
    if repeat == 5:
        assert not np.array_equal(qpos_samples[0], qpos_samples[-1])


def test_success_can_end_partial_action_and_reset_clears_terminal_state(monkeypatch):
    env, _ = make_env(5)
    env.reset(seed=7)
    calls = 0

    def status():
        nonlocal calls
        calls += 1
        return SimpleNamespace(released_stable_stack=calls == 2)

    monkeypatch.setattr(env.task, "status", status)
    action = np.clip(np.zeros(env.action_space.shape), env.action_space.low, env.action_space.high)
    _, reward, terminated, truncated, info = env.step(action)
    assert (reward, terminated, truncated) == (1.0, True, False)
    assert info["physics_steps"] == 2
    assert info["elapsed_dt"] == pytest.approx(0.004)
    with pytest.raises(RuntimeError, match="Reset"):
        env.step(action)
    env.reset(seed=8)
    assert env._steps == 0
    assert not env._done


def test_collector_records_actual_repeat_and_action_boundary_frames(monkeypatch):
    env, _ = make_env(5)
    monkeypatch.setattr(env.task, "status", lambda: SimpleNamespace(released_stable_stack=False))

    class FixedExpert:
        failed = False

        def reset(self, obs, info):
            pass

        def act(self, obs, *, dt=0.0):
            return np.clip(
                np.zeros(env.action_space.shape), env.action_space.low, env.action_space.high
            ).astype(np.float32)

    episode = collect_episode(
        env,
        FixedExpert(),
        max_steps=2,
        seed=7,
        record_frame=lambda: capture_frame(env.simulator),
        replay_metadata={
            "dt": env.simulator.dt,
            "physics_steps_per_action": env.physics_steps_per_action,
            "cube_yaw_range_degrees": env.cube_yaw_range_degrees,
        },
    )
    assert episode.metadata["replay"]["physics_steps_per_action"] == 5
    assert episode.qpos.shape[0] == episode.states.shape[0] == len(episode) + 1 == 3
    np.testing.assert_allclose(np.diff(episode.frame_times), 0.01)
