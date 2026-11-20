"""Robot YAML reference selection, tuning, and simulator-owned update intervals."""

from dataclasses import asdict

import numpy as np
import pytest
import yaml
from controller_config import forte_test_pd_config
from dacite import MissingValueError, WrongTypeError

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets import ROBOT_PATH
from mujoco_lab.assets.robot.robot import ReferenceConfig, RobotConfig
from mujoco_lab.trajectory_generation import (
    ControlTarget,
    JerkLQRReference,
    MinJerkReference,
    PositionHoldReference,
)


def load_settings(tmp_path, reference):
    settings = yaml.safe_load((ROBOT_PATH / "forte" / "robot.yaml").read_text())
    if reference is None:
        settings.pop("reference", None)
    else:
        settings["reference"] = reference
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(settings))
    return RobotConfig.load(path)


def test_missing_reference_block_uses_independent_jerk_lqr_defaults(tmp_path):
    first = load_settings(tmp_path, None)
    second = load_settings(tmp_path, None)
    assert first.reference == ReferenceConfig()
    assert first.reference is not second.reference


def test_yaml_requires_constraints(tmp_path):
    settings = yaml.safe_load((ROBOT_PATH / "forte" / "robot.yaml").read_text())
    del settings["constraints"]
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(MissingValueError, match="constraints"):
        RobotConfig.load(path)


def test_yaml_jerk_lqr_parameters_reach_the_robot_reference(tmp_path):
    config = load_settings(
        tmp_path,
        {"name": "jerk_lqr", "response_time": 0.04, "horizon": 0.08, "terminal_weight": 2.0},
    )
    sim = Simulator(
        create_cube_stack(1), robots=[RobotSpec("arm", "forte", config=config)], dt=0.003
    )
    robot = sim.robots["arm"]
    initial = robot.target
    expected = JerkLQRReference(
        initial,
        robot.get_reference_constraints(),
        control_dt=sim.dt,
        response_time=0.04,
        horizon=0.08,
        terminal_weight=2.0,
    )
    assert robot.gripper is not None
    goal = initial.position + 1e-6
    robot.set_goal(np.r_[goal, robot.gripper.get_target()])
    reference = robot.reference_generator
    assert isinstance(reference, JerkLQRReference)
    assert reference.dt == sim.dt
    assert reference.tau == 0.04
    expected.set_goal(ControlTarget(goal))
    sample = expected.compute()
    sim.run_steps(1)
    for field in ("position", "velocity", "acceleration"):
        np.testing.assert_allclose(getattr(robot.target, field), getattr(sample, field), atol=1e-16)


@pytest.mark.parametrize("constrain", [False, True])
def test_yaml_min_jerk_duration_is_independent_of_physics_dt(tmp_path, constrain):
    config = load_settings(
        tmp_path, {"name": "min_jerk", "duration": 0.018, "constrain": constrain}
    )
    sim = Simulator(
        create_cube_stack(1), robots=[RobotSpec("arm", "forte", config=config)], dt=0.003
    )
    robot = sim.robots["arm"]
    initial = robot.target.position.copy()
    goal = initial + 0.06
    assert robot.gripper is not None
    robot.set_goal(np.r_[goal, robot.gripper.get_target()])
    assert isinstance(robot.reference_generator, MinJerkReference)
    assert robot.reference_generator.dt == sim.dt
    sim.run_steps(3)
    if not constrain:
        np.testing.assert_allclose(robot.target.position, initial + 0.03, atol=1e-12)
    sim.run_steps(3)
    if constrain:
        assert np.all(robot.target.position < goal)
        assert config.constraints is not None
        assert np.all(np.abs(robot.target.velocity) <= config.constraints.velocity_limit)
        assert np.all(np.abs(robot.target.acceleration) <= config.constraints.acceleration_limit)
    else:
        np.testing.assert_allclose(robot.target.position, goal, atol=1e-12)


def test_yaml_rejects_unsupported_reference_name(tmp_path):
    with pytest.raises(WrongTypeError, match="reference.name"):
        load_settings(tmp_path, {"name": "unsupported"})


def test_position_hold_ignores_derivatives_and_owns_goals_and_snapshots():
    initial = ControlTarget([0.0, 0.1], [2.0, 3.0], [4.0, 5.0])
    reference = PositionHoldReference(initial, control_dt=0.002)
    goal = ControlTarget([0.3, -0.2], [1.0, 2.0], [3.0, 4.0])
    reference.set_goal(goal)
    goal.position[:] = 99
    for _ in range(5):
        target = reference.compute()
        np.testing.assert_array_equal(target.position, [0.3, -0.2])
        np.testing.assert_array_equal(target.velocity, 0)
        np.testing.assert_array_equal(target.acceleration, 0)
        target.position[:] = 88
    reference.reset(initial)
    np.testing.assert_array_equal(reference.state().position, initial.position)
    np.testing.assert_array_equal(reference.state().velocity, 0)
    np.testing.assert_array_equal(reference.state().acceleration, 0)


def test_yaml_pd_position_hold_keeps_zero_desired_velocity_between_actions(tmp_path):
    config = load_settings(tmp_path, {"name": "position_hold"})
    config.controller = forte_test_pd_config()
    sim = Simulator(
        create_cube_stack(1), robots=[RobotSpec("arm", "forte", config=config)], dt=0.002
    )
    robot = sim.robots["arm"]
    assert robot.controller.name == "joint-space PD"
    assert isinstance(robot.reference_generator, PositionHoldReference)
    assert robot.gripper is not None
    goal = robot.target.position.copy()
    goal[1] -= 0.01
    robot.set_goal(np.r_[goal, robot.gripper.get_target()])
    for _ in range(5):
        sim.run_steps(1)
        np.testing.assert_array_equal(robot.target.position, goal)
        np.testing.assert_array_equal(robot.target.velocity, 0)
        np.testing.assert_array_equal(robot.target.acceleration, 0)
    sim.reset()
    assert not np.array_equal(robot.target.position, goal)
    np.testing.assert_array_equal(robot.target.velocity, 0)


@pytest.mark.parametrize("desired_velocity", ["zero", "reference_velocity"])
def test_pd_min_jerk_uses_the_configured_velocity_target(tmp_path, desired_velocity):
    settings = yaml.safe_load((ROBOT_PATH / "forte" / "robot.yaml").read_text())
    controller = forte_test_pd_config()
    controller.desired_velocity = desired_velocity
    settings["controller"] = asdict(controller)
    settings["reference"] = {"name": "min_jerk", "duration": 0.01, "constrain": False}
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(settings))
    config = RobotConfig.load(path)
    sim = Simulator(create_cube_stack(1), robots=[RobotSpec("arm", "forte", config=config)])
    robot = sim.robots["arm"]
    assert robot.gripper is not None
    assert robot.controller.desired_velocity == desired_velocity
    goal = robot.target.position.copy()
    goal[1] -= 0.01
    robot.set_goal(np.r_[goal, robot.gripper.get_target()])
    sim.run_steps(1)
    assert np.any(robot.target.velocity != 0)
    state = robot.get_control_state()
    kp, kd = config.controller.pd_gain(robot.control_joint_names)
    velocity = robot.target.velocity if desired_velocity == "reference_velocity" else 0.0
    expected = (
        np.multiply(kp, robot.target.position - state.qpos)
        + np.multiply(kd, velocity - state.qvel)
        + state.bias_forces
    )
    np.testing.assert_allclose(robot.controller.compute(state, robot.target), expected)
