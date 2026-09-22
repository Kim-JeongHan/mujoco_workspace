"""External target behavior across scheduled, manual, and reset control."""

import numpy as np
import pytest

from mujoco_lab import ControlTarget, RobotSpec, Simulator, create_environment
from mujoco_lab.control import Controller, create_controller, demo_target_updater
from mujoco_lab.utils import Transform


def forte_sim():
    sim = Simulator(
        create_environment("empty"),
        robots=[RobotSpec("arm", "forte")],
        dt=0.001,
    )
    return sim, sim.robots["arm"]


class TargetRecorder(Controller):
    def __init__(self):
        self.seen = []

    def compute(self, state, target):
        self.seen.append((state.time, target.position.copy()))
        return np.zeros(8)


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
    np.testing.assert_array_equal(robot.target.position, robot.joint_state.qpos)
    sim.step()
    assert calls[-1] == 0
    assert recorder.seen[-1][1][0] == 11


def test_manual_target_persists_and_switch_reinitializes_target_space():
    sim, robot = forte_sim()
    robot.change_controller(create_controller("pd", robot))
    initial = robot.target.position.copy()
    commanded = initial.copy()
    commanded[0] += 0.03
    robot.target = ControlTarget(commanded)
    sim.run_steps(3)
    np.testing.assert_array_equal(robot.target.position, commanded)
    robot.change_controller(create_controller("osc", robot))
    assert robot.target.position.shape == (3,)
    np.testing.assert_allclose(robot.target.position, robot.state.get_frame_position("ee_site"))
    robot.change_controller(create_controller("pd", robot))
    robot.update_state()
    np.testing.assert_array_equal(robot.target.position, robot.joint_state.qpos[:7])
    sim.reset()
    np.testing.assert_array_equal(robot.target.position, initial)


def test_external_targets_and_updater_are_robot_local():
    sim = Simulator(
        create_environment("empty"),
        robots=[
            RobotSpec("left", "forte", Transform(translation=[-0.8, 0, 0])),
            RobotSpec("right", "forte", Transform(translation=[0.8, 0, 0])),
        ],
    )
    left, right = sim.robots.values()
    left.change_controller(create_controller("pd", left))
    right.change_controller(create_controller("pd", right))
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
        assert current._state.state == "running"
        raise RuntimeError("updater failed")

    sim.target_updater = fail
    with pytest.raises(RuntimeError, match="updater failed"):
        sim.step()
    assert sim.data.time == before
    assert sim._state.state == "running"


def test_no_controller_native_input_and_updater_each_tick():
    sim, robot = forte_sim()
    calls = []
    sim.target_updater = lambda current: calls.append(current.data.time)
    sim.data.ctrl[robot.actuator_ids] = 0.1
    sim.run_steps(9)
    np.testing.assert_allclose(calls, np.arange(9) * 0.001)
    np.testing.assert_array_equal(sim.data.ctrl[robot.actuator_ids], 0.1)


def test_osc_supplied_derivatives_change_command_without_changing_target_position():
    sim, robot = forte_sim()
    osc = create_controller("osc", robot)
    robot.change_controller(osc)
    robot.update_state()
    state = robot.joint_state
    position = robot.target.position.copy()
    zero = osc.compute(state, ControlTarget(position))
    derivative = osc.compute(
        state,
        ControlTarget(position, velocity=np.array([0.01, 0, 0]), acceleration=[0, 0.2, 0]),
    )
    assert not np.array_equal(zero, derivative)
    assert osc.tracking_error == pytest.approx(0)
    np.testing.assert_array_equal(osc._target, position)


@pytest.mark.parametrize("mode", ["pd", "osc"])
def test_demo_trajectory_replays_after_reset(mode):
    sim, robot = forte_sim()
    robot.change_controller(create_controller(mode, robot))
    updater = demo_target_updater(sim, {robot.name: mode})
    sim.target_updater = updater
    sim.run_steps(20)
    expected = (sim.data.qpos.copy(), sim.data.qvel.copy(), sim.data.ctrl.copy())
    sim.reset()
    assert sim.target_updater is updater
    sim.run_steps(20)
    for actual, reference in zip((sim.data.qpos, sim.data.qvel, sim.data.ctrl), expected):
        np.testing.assert_array_equal(actual, reference)
