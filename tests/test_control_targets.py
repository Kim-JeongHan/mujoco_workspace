"""External target behavior across scheduled, manual, and reset control."""

import numpy as np
import pytest
from controller_config import create_test_controller

from mujoco_lab import ControlTarget, RobotSpec, Simulator, create_environment
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.control import Controller
from mujoco_lab.utils import Transform


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
        current.robots["arm"].target = ControlTarget(np.full(7, len(calls)))

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

    def update(current):
        current.robots["left"].target = ControlTarget(np.full(7, 0.1))

    sim.target_updater = update
    sim.step()
    np.testing.assert_array_equal(left.target.position, 0.1)
    np.testing.assert_array_equal(right.target.position, original_right)


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


def test_osc_supplied_derivatives_change_command_without_changing_target_position():
    sim, robot = forte_sim()
    osc = create_test_controller(robot, controller="osc")
    robot.change_controller(osc)
    robot.update_state()
    state = robot.get_control_state()
    position = robot.target.position.copy()
    zero = osc.compute(state, ControlTarget(position))
    derivative = osc.compute(
        state,
        ControlTarget(position, velocity=np.array([0.01, 0, 0]), acceleration=[0, 0.2, 0]),
    )
    assert not np.array_equal(zero, derivative)
    assert osc.get_tracking_error() == pytest.approx(0)
    np.testing.assert_array_equal(osc._target, position)
