"""External target behavior across scheduled, manual, and reset control."""

import numpy as np
import pytest
from controller_config import create_test_controller

from mujoco_lab import ControlTarget, RobotSpec, Simulator, create_environment
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.control import Controller
from mujoco_lab.utils import Transform


def test_default_target_derivatives_are_independent_zero_arrays():
    position = np.arange(6).reshape(2, 3)
    target = ControlTarget(position)
    other = ControlTarget(position)
    for derivative in (target.velocity, target.acceleration):
        assert derivative.shape == position.shape
        assert derivative.dtype == np.dtype(float)
        np.testing.assert_array_equal(derivative, np.zeros(position.shape))
    position[:] = -1
    np.testing.assert_array_equal(target.position, np.arange(6).reshape(2, 3))
    target.velocity[:] = 1
    np.testing.assert_array_equal(target.acceleration, np.zeros(position.shape))
    np.testing.assert_array_equal(other.velocity, np.zeros(position.shape))


def test_control_target_owns_supplied_derivative_arrays():
    position, velocity, acceleration = np.array([[1, 2], [3, 4], [5, 6]])
    target = ControlTarget(position, velocity, acceleration)
    position[:] = velocity[:] = acceleration[:] = -1
    np.testing.assert_array_equal(target.position, [1, 2])
    np.testing.assert_array_equal(target.velocity, [3, 4])
    np.testing.assert_array_equal(target.acceleration, [5, 6])


@pytest.mark.parametrize("derivative", ["velocity", "acceleration"])
def test_target_rejects_derivative_shape_that_would_broadcast(derivative):
    with pytest.raises(ValueError, match="match position shape"):
        ControlTarget(np.zeros(7), **{derivative: np.zeros(1)})


def forte_sim():
    sim = Simulator(
        create_environment("empty"),
        robots=[RobotSpec("arm", "forte", config=load_robot_config("forte"))],
        dt=0.001,
    )
    return sim, sim.robots["arm"]


class TargetRecorder(Controller):
    def __init__(self):
        self.seen = []

    def compute(self, state, target):
        self.seen.append((state.time, target.position.copy()))
        return np.zeros(state.qpos.shape)


def test_updater_runs_before_each_control_and_replays_after_reset():
    sim, robot = forte_sim()
    recorder = TargetRecorder()
    robot.change_controller(recorder)
    calls = []

    def update(current):
        calls.append(current.data.time)
        current.robots["arm"].update_target(ControlTarget(np.full(7, len(calls))))

    sim.target_updater = update
    sim.run_steps(10)
    np.testing.assert_allclose(calls, np.arange(10) * 0.001)
    assert [value[1][0] for value in recorder.seen] == list(range(1, 11))
    assert sim.target_updater is update
    sim.reset()
    np.testing.assert_array_equal(robot.target.position, robot.get_control_state().qpos)
    sim.step()
    assert calls[-1] == 0
    assert recorder.seen[-1][1][0] == 11


def test_external_targets_and_updater_are_robot_local():
    sim = Simulator(
        create_environment("empty"),
        robots=[
            RobotSpec(
                "left",
                "forte",
                Transform(translation=[-0.8, 0, 0]),
                config=load_robot_config("forte"),
            ),
            RobotSpec(
                "right",
                "forte",
                Transform(translation=[0.8, 0, 0]),
                config=load_robot_config("forte"),
            ),
        ],
    )
    left, right = sim.robots.values()
    left.change_controller(create_test_controller(left, controller="pd"))
    right.change_controller(create_test_controller(right, controller="pd"))
    original_right = right.target.position.copy()
    left_gripper, right_gripper = left.gripper, right.gripper
    assert left_gripper is not None and right_gripper is not None
    left.update_target(ControlTarget(np.zeros(7)), 0.02)
    original_right_gripper = right_gripper.get_target()

    def update(current):
        current.robots["left"].update_target(ControlTarget(np.full(7, 0.1)))

    sim.target_updater = update
    sim.step()
    np.testing.assert_array_equal(left.target.position, 0.1)
    np.testing.assert_array_equal(right.target.position, original_right)
    assert left_gripper.get_target() == 0.02
    assert left_gripper.is_active()
    assert right_gripper.get_target() == original_right_gripper
    assert not right_gripper.is_active()


def test_updater_and_controller_error_prevent_physics_step():
    sim, robot = forte_sim()
    robot.change_controller(TargetRecorder())
    before = sim.data.time

    def fail(current):
        assert current._state.get_state() == "running"
        raise RuntimeError("updater failed")

    sim.target_updater = fail
    with pytest.raises(RuntimeError, match="updater failed"):
        sim.step()
    assert sim.data.time == before
    assert sim._state.get_state() == "idle"
    assert not sim._stop_requested
    sim.reset()


def test_computed_torque_derivatives_change_command_without_changing_target_position():
    sim, robot = forte_sim()
    controller = create_test_controller(robot, controller="computed_torque")
    robot.change_controller(controller)
    robot.update_state()
    state = robot.get_control_state()
    position = robot.target.position.copy()
    zero = controller.compute(state, ControlTarget(position))
    velocity = np.zeros_like(position)
    acceleration = np.zeros_like(position)
    velocity[0], acceleration[1] = 0.01, 0.2
    derivative = controller.compute(
        state,
        ControlTarget(position, velocity=velocity, acceleration=acceleration),
    )
    assert not np.array_equal(zero, derivative)
    assert controller.get_tracking_error() == pytest.approx(0)
    np.testing.assert_array_equal(robot.target.position, position)
