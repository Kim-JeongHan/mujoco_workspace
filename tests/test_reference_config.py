"""Robot YAML reference selection, tuning, and simulator-owned update intervals."""

import numpy as np
import pytest
import yaml
from dacite import MissingValueError, WrongTypeError

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets import ROBOT_PATH
from mujoco_lab.assets.robot.robot import ReferenceConfig, RobotConfig
from mujoco_lab.trajectory_generation import AnalyticJerkReference, ControlTarget, MinJerkReference


def load_settings(tmp_path, reference):
    settings = yaml.safe_load((ROBOT_PATH / "forte" / "robot.yaml").read_text())
    if reference is None:
        settings.pop("reference", None)
    else:
        settings["reference"] = reference
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(settings))
    return RobotConfig.load(path)


def test_missing_reference_block_uses_independent_analytic_defaults(tmp_path):
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


def test_yaml_analytic_parameters_reach_the_robot_reference(tmp_path):
    config = load_settings(
        tmp_path,
        {"name": "analytic_jerk", "response_time": 0.04, "horizon": 0.08, "terminal_weight": 2.0},
    )
    sim = Simulator(
        create_cube_stack(1), robots=[RobotSpec("arm", "forte", config=config)], dt=0.003
    )
    robot = sim.robots["arm"]
    initial = robot.target
    expected = AnalyticJerkReference(
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
    assert isinstance(reference, AnalyticJerkReference)
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
