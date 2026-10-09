"""Robot-owned goals, reference clock ownership, and common execution paths."""

import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import CubeStackTask
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv
from mujoco_lab.trajectory_generation import AnalyticJerkReference


def make_robot():
    config = load_robot_config("forte")
    sim = Simulator(create_cube_stack(1), robots=[RobotSpec("arm", "forte", config=config)])
    robot = sim.robots["arm"]
    return sim, robot


def test_goal_bounds_joint_order_and_input_ownership():
    _, robot = make_robot()
    limits = robot.get_goal_limits()
    action = limits[:, 1] + 1
    names = tuple(reversed(robot.control_joint_names))
    action[:-1] = action[:-1][::-1]
    original = action.copy()
    robot.set_goal(action, joint_names=names)
    reference = robot.reference_generator
    assert isinstance(reference, AnalyticJerkReference)
    np.testing.assert_array_equal(reference.goal, limits[:-1, 1])
    assert robot.gripper is not None
    assert robot.gripper.get_target() == limits[-1, 1]
    np.testing.assert_array_equal(action, original)
    action[:] = -100
    np.testing.assert_array_equal(reference.goal, limits[:-1, 1])


def test_goal_rejects_extra_values_before_changing_reference():
    _, robot = make_robot()
    reference = robot.reference_generator
    assert isinstance(reference, AnalyticJerkReference)
    before = reference.goal.copy()
    expected = len(robot.control_joint_names) + int(robot.has_gripper())
    with pytest.raises(ValueError, match="goal must contain"):
        robot.set_goal(np.zeros(expected + 1))
    np.testing.assert_array_equal(reference.goal, before)


def test_control_does_not_advance_reference_and_simulator_advances_once(monkeypatch):
    sim, robot = make_robot()
    assert robot.target is not None and robot.gripper is not None
    initial = robot.target
    action = np.r_[initial.position, robot.gripper.get_target()]
    action[1] -= 0.1
    robot.set_goal(action)
    reference = robot.reference_generator
    assert reference is not None
    calls = []
    compute = reference.compute

    def record_compute():
        calls.append(sim.data.time)
        return compute()

    monkeypatch.setattr(reference, "compute", record_compute)
    robot.control()
    robot.control()
    assert calls == []
    assert robot.target is initial
    applied = []
    control = robot.control

    def record_control():
        applied.append(robot.target)
        return control()

    monkeypatch.setattr(robot, "control", record_control)
    sim.run_steps(3)
    np.testing.assert_allclose(calls, [0, sim.dt, 2 * sim.dt])
    assert len(applied) == 3
    assert applied[0] is not initial
    assert applied[-1] is robot.target
    assert robot.target is not None
    np.testing.assert_array_equal(robot.target.position, reference.state().position)


def test_simulator_reset_reuses_reference_and_clears_pending_goal():
    sim, robot = make_robot()
    assert robot.target is not None and robot.gripper is not None
    initial = robot.target.position.copy()
    action = np.r_[initial, robot.gripper.get_target()]
    action[1] -= 0.1
    robot.set_goal(action)
    reference = robot.reference_generator
    assert isinstance(reference, AnalyticJerkReference)
    sim.run_steps(5)
    sim.reset()
    assert robot.reference_generator is reference
    assert robot.target is not None
    np.testing.assert_array_equal(robot.target.position, initial)
    np.testing.assert_array_equal(reference.goal, initial)
    np.testing.assert_array_equal(robot.target.velocity, 0)
    np.testing.assert_array_equal(robot.target.acceleration, 0)


def test_direct_goal_and_environment_use_identical_physics_and_references():
    direct_sim, direct = make_robot()
    env_sim, observed = make_robot()
    env = CubeStackEnv(
        CubeStackTask(env_sim, 1), xy_range=0, physics_steps_per_action=5, max_steps=2
    )
    env.reset(seed=1)
    assert direct.target is not None and direct.gripper is not None
    action = np.r_[direct.target.position, direct.gripper.get_target()]
    action[1] -= 0.1
    direct.set_goal(action)
    direct_sim.run_steps(5)
    env.step(action)
    assert observed.target is not None and direct.target is not None
    for field in ("position", "velocity", "acceleration"):
        np.testing.assert_array_equal(
            getattr(direct.target, field), getattr(observed.target, field)
        )
    np.testing.assert_array_equal(direct_sim.data.qpos, env_sim.data.qpos)
    np.testing.assert_array_equal(direct_sim.data.qvel, env_sim.data.qvel)
