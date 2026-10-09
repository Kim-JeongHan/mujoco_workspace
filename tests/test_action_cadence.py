"""Action cadence keeps physical control fresh and episode steps distinct."""

from types import SimpleNamespace

import numpy as np
import pytest
from controller_config import create_test_controller

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import CubeStackTask, Expert
from mujoco_lab.learning.datasets.replay import capture_frame
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv
from mujoco_lab.learning.rollout import collect_episode
from mujoco_lab.trajectory_generation import AnalyticJerkReference


def make_env(repeat, *, dt=0.002):
    simulator = Simulator(
        create_cube_stack(2),
        robots=[RobotSpec("forte", "forte", config=load_robot_config("forte"))],
        dt=dt,
    )
    robot = simulator.robots["forte"]
    robot.change_controller(create_test_controller(robot, controller="pd", frame="grasp"))
    task = CubeStackTask(simulator, 2)
    return CubeStackEnv(task, physics_steps_per_action=repeat, max_steps=2), robot


def test_zero_action_duration_is_rejected():
    with pytest.raises(ValueError, match="physics_steps_per_action must be positive"):
        make_env(0)


def test_action_interpolates_target_with_fresh_control_each_physics_tick(monkeypatch):
    repeat = 5
    env, robot = make_env(repeat, dt=0.003)
    env.reset(seed=7)
    monkeypatch.setattr(env.task, "status", lambda: SimpleNamespace(is_released_stable=False))
    start_target = robot.target
    action = np.r_[start_target.position, robot.gripper.get_target()]
    action[1] -= 0.08
    targets = []
    qpos_samples = []
    original_control = robot.control

    def control():
        targets.append(robot.target)
        qpos_samples.append(robot.state.snapshot().qpos.copy())
        return original_control()

    monkeypatch.setattr(robot, "control", control)
    start_time = env.data.time
    obs, _, terminated, truncated, info = env.step(action)
    assert obs.shape == (54,)
    assert not terminated and not truncated
    assert env._steps == 1
    assert env.action_dt == pytest.approx(env.simulator.dt * repeat)
    assert env.data.time - start_time == pytest.approx(env.action_dt)
    assert info["physics_steps"] == repeat
    assert info["elapsed_dt"] == pytest.approx(env.action_dt)
    assert len(targets) == len(qpos_samples) == repeat
    constraints = robot.config.constraints
    assert targets[0].position[1] < start_target.position[1]
    for target in [*targets, robot.target]:
        assert np.all(np.abs(target.velocity) <= constraints.velocity_limit)
        assert np.all(np.abs(target.acceleration) <= constraints.acceleration_limit)
    h = env.simulator.dt
    for before, after in zip([start_target, *targets[:-1]], targets, strict=True):
        # Constant jerk gives these independent trapezoidal/cubic integration identities.
        np.testing.assert_allclose(
            after.velocity, before.velocity + h * (before.acceleration + after.acceleration) / 2
        )
        np.testing.assert_allclose(
            after.position,
            before.position
            + h * before.velocity
            + h**2 * (2 * before.acceleration + after.acceleration) / 6,
        )
    assert robot.target.position[1] < start_target.position[1]
    assert robot.target.position[1] > action[1]
    assert not np.array_equal(qpos_samples[0], qpos_samples[-1])


def test_reference_uses_configured_motion_limits(monkeypatch):
    config = load_robot_config("forte")
    limits = config.constraints
    assert limits is not None
    limits.acceleration_limit[0] = 0.01
    simulator = Simulator(create_cube_stack(2), robots=[RobotSpec("forte", "forte", config=config)])
    robot = simulator.robots["forte"]
    robot.change_controller(create_test_controller(robot, controller="pd"))
    env = CubeStackEnv(CubeStackTask(simulator, 2), physics_steps_per_action=5)
    env.reset(seed=7)
    monkeypatch.setattr(env.task, "status", lambda: SimpleNamespace(is_released_stable=False))
    start = robot.target
    assert start is not None and robot.gripper is not None
    action = np.r_[start.position, robot.gripper.get_target()]
    action[0] += 0.005
    env.step(action)
    reference = robot.reference_generator
    assert isinstance(reference, AnalyticJerkReference)
    assert reference.amax[0] == pytest.approx(0.01)
    assert robot.target is not None
    assert 0 < robot.target.acceleration[0] <= 0.01 + 1e-12


def test_action_rejects_extra_values_before_advancing_physics():
    env, robot = make_env(5)
    env.reset(seed=7)
    target = robot.target
    with pytest.raises(ValueError, match="environment action shape"):
        env.step(np.zeros(env.action_space.shape[0] + 1))
    assert env.data.time == 0
    assert env._steps == 0
    assert robot.target is target


def test_success_can_end_partial_action_and_reset_clears_terminal_state(monkeypatch):
    env, robot = make_env(5)
    env.reset(seed=7)
    calls = 0

    def status():
        nonlocal calls
        calls += 1
        return SimpleNamespace(is_released_stable=calls == 2)

    monkeypatch.setattr(env.task, "status", status)
    action = np.clip(np.zeros(env.action_space.shape), env.action_space.low, env.action_space.high)
    _, reward, terminated, truncated, info = env.step(action)
    assert (reward, terminated, truncated) == (1.0, True, False)
    assert info["physics_steps"] == 2
    assert info["elapsed_dt"] == pytest.approx(0.004)
    assert not np.array_equal(robot.target.position, action[:7])
    reference = robot.reference_generator
    np.testing.assert_array_equal(robot.target.position, reference.state().position)
    with pytest.raises(RuntimeError, match="Reset"):
        env.step(action)
    env.reset(seed=8)
    assert env._steps == 0
    assert not env._done
    assert robot.reference_generator is reference
    np.testing.assert_array_equal(robot.target.velocity, 0)
    np.testing.assert_array_equal(robot.target.acceleration, 0)
    reset_reference = robot.reference_generator
    assert isinstance(reset_reference, AnalyticJerkReference)
    np.testing.assert_array_equal(reset_reference.goal, robot.target.position)


def test_collector_records_actual_repeat_and_action_boundary_frames(monkeypatch):
    env, _ = make_env(5)
    monkeypatch.setattr(env.task, "status", lambda: SimpleNamespace(is_released_stable=False))

    class FixedExpert(Expert):
        failed = False

        def reset(self):
            pass

        def act(self, *, dt=0.0):
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
    assert episode.qpos is not None and episode.frame_times is not None
    assert episode.qpos.shape[0] == episode.states.shape[0] == len(episode) + 1 == 3
    np.testing.assert_allclose(np.diff(episode.frame_times), 0.01)
