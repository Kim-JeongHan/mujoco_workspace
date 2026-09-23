"""ForteV1_RobStride selected-joint control with a separate gripper target."""

import os
import subprocess
import sys

import mujoco
import numpy as np
import pytest

from mujoco_lab import ENVIRONMENT_NAMES, RobotSpec, Simulator, create_environment
from mujoco_lab.control import ControlTarget, create_controller, demo_target_updater
from mujoco_lab.control.osc import OperationalSpaceControl
from mujoco_lab.control.pd import JointSpacePD
from mujoco_lab.control.trajectory import HOME_QPOS, PD_WAYPOINTS, osc_circle_target
from mujoco_lab.rendering.annotations import annotate_controller
from mujoco_lab.state import JointState
from mujoco_lab.utils import Transform


def forte_simulator(environment="empty"):
    return Simulator(create_environment(environment), robots=[RobotSpec("robot", "forte")])


def test_seven_axis_controller_math_on_numpy_snapshots():
    state = JointState(0.0, HOME_QPOS.copy(), np.zeros(7), np.ones(7))
    np.testing.assert_array_equal(
        JointSpacePD(np.ones(7), np.ones(7)).compute(state, ControlTarget(HOME_QPOS)),
        np.ones(7),
    )

    class Dynamics:
        nv = 7

        def get_frame_position(self, frame):
            return np.array([0.65, 0.0, 0.53])

        def get_jacobian(self, frame):
            return np.eye(6, 7)

        def get_mass_matrix(self):
            return np.eye(7)

    np.testing.assert_array_equal(
        OperationalSpaceControl(Dynamics(), posture=HOME_QPOS).compute(
            state, ControlTarget([0.65, 0, 0.53])
        ),
        np.ones(7),
    )


@pytest.mark.parametrize("mode", ["pd", "osc"])
def test_connected_controller_uses_selected_state_and_maps_joint_commands(mode):
    sim = forte_simulator()
    robot = sim.robots["robot"]
    controller = create_controller(mode, robot)
    robot.change_controller(controller)
    assert robot.state.nq == robot.state.nv == 9
    assert robot.num_actuators == 8
    assert robot.target.position.shape == ((7,) if mode == "pd" else (3,))
    state = robot.state
    assert controller._owner is state
    assert robot.control_joint_names == tuple(state.joint_names[:7])
    joint_slots = robot.control_joint_slots
    np.testing.assert_array_equal(
        state.get_jacobian("ee_site", joint_slots),
        robot.state.get_jacobian("ee_site")[:, joint_slots],
    )
    np.testing.assert_array_equal(
        state.get_mass_matrix(joint_slots),
        robot.state.get_mass_matrix()[np.ix_(joint_slots, joint_slots)],
    )
    command = controller.compute(robot.get_control_state(), robot.target)
    assert command.shape == (7,)
    robot.control()
    assert sim.data.ctrl[robot.actuator_ids[-1]] == 0
    robot.gripper.set_target(-0.01)
    sim.step()
    assert sim.data.ctrl[robot.actuator_ids[-1]] == -0.01
    assert robot.state.snapshot().qpos.shape == (9,)
    with pytest.raises(ValueError, match="gripper target"):
        robot.gripper.set_target(-0.025)
    with pytest.raises(ValueError, match="gripper target"):
        robot.gripper.set_target(np.nan)
    sim.reset()
    assert robot.gripper.get_target() == 0


def test_pd_targets_respect_bounded_source_axes():
    assert HOME_QPOS.shape == (7,)
    assert PD_WAYPOINTS.shape[1] == 7
    sim = forte_simulator()
    ranges = sim.model.jnt_range[:3]
    assert np.all(PD_WAYPOINTS[:, :3] >= ranges[:, 0])
    assert np.all(PD_WAYPOINTS[:, :3] <= ranges[:, 1])


@pytest.mark.parametrize("mode", ["pd", "osc"])
@pytest.mark.parametrize("environment", ENVIRONMENT_NAMES)
def test_feedback_control_is_finite_in_each_environment(mode, environment):
    sim = forte_simulator(environment)
    robot = sim.robots["robot"]
    robot.change_controller(create_controller(mode, robot))
    sim.target_updater = demo_target_updater(sim, {robot.name: mode})
    stats = sim.run_steps(400)[robot.name]
    assert stats.steps == 400
    assert np.isfinite(sim.data.qpos).all()
    assert np.isfinite(sim.data.qvel).all()
    assert not sim.data.warning.number.any()
    assert max(stats.errors) < (0.15 if mode == "pd" else 0.05)


def test_osc_trajectory_and_annotations_follow_translated_rotated_mount():
    sim = forte_simulator()
    robot = sim.robots["robot"]
    moved_sim = Simulator(
        create_environment("empty"),
        robots=[RobotSpec("robot", "forte", Transform.from_pose_mmdeg([100, -200, 800, 0, 0, 90]))],
    )
    moved_model, moved_data = moved_sim.model, moved_sim.data
    moved_robot = moved_sim.robots["robot"]
    moved_robot.change_controller(create_controller("osc", moved_robot))
    moved_sim.target_updater = demo_target_updater(moved_sim, {"robot": "osc"})
    rotation = moved_data.body("robot/base_link").xmat.reshape(3, 3)
    translation = moved_data.body("robot/base_link").xpos
    entry = osc_circle_target(
        2.0,
        start_time=0,
        start_position=robot.state.get_frame_position("ee_site"),
        base_position=sim.data.body("robot/base_link").xpos,
        base_rotation=np.eye(3),
    )
    entry_distance = np.linalg.norm(entry.position - robot.state.get_frame_position("ee_site"))
    assert entry_distance == pytest.approx(0.025, abs=1e-5)
    for elapsed in [0.0, 1.25, 2.5]:
        reference = osc_circle_target(
            elapsed + 2,
            start_time=0,
            start_position=robot.state.get_frame_position("ee_site"),
            base_position=np.zeros(3),
            base_rotation=np.eye(3),
        )
        actual = osc_circle_target(
            elapsed + 2,
            start_time=0,
            start_position=moved_robot.state.get_frame_position("ee_site"),
            base_position=translation,
            base_rotation=rotation,
        )
        np.testing.assert_allclose(actual.position, translation + rotation @ reference.position)
        np.testing.assert_allclose(actual.velocity, rotation @ reference.velocity)
    moved_sim.physics_step()
    scene = mujoco.MjvScene(moved_model, maxgeom=100)
    scene.ngeom = 0
    annotate_controller(scene, moved_robot, moved_data, None)
    np.testing.assert_allclose(scene.geoms[0].pos, moved_robot.target.position, atol=1e-6)


@pytest.mark.parametrize("robot", ["panda"])
@pytest.mark.parametrize("mode", ["pd", "osc"])
def test_torque_control_rejects_position_actuators(robot, mode):
    sim = Simulator(create_environment("empty"), robots=[RobotSpec("robot", robot)])
    with pytest.raises(ValueError, match="requires torque/force actuators"):
        create_controller(mode, sim.robots["robot"])


@pytest.mark.parametrize("mode", ["pd", "osc"])
def test_cli_connects_controller_to_workspace_environment(mode, tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mujoco_lab",
            "--command",
            "simulate",
            "--robot",
            "forte",
            "--environment",
            "warehouse",
            "--controller",
            mode,
            "--steps",
            "100",
        ],
        cwd=tmp_path,
        env={**os.environ, "MUJOCO_GL": "disable"},
        capture_output=True,
        text=True,
        check=True,
    )


def test_cli_rejects_pd_on_position_controlled_robot(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mujoco_lab",
            "--command",
            "simulate",
            "--robot",
            "panda",
            "--controller",
            "pd",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "requires torque/force actuators" in result.stderr


@pytest.mark.parametrize("robot", ["panda"])
def test_cli_runs_native_position_controller(robot, tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mujoco_lab",
            "--command",
            "simulate",
            "--robot",
            robot,
            "--controller",
            "position",
            "--steps",
            "10",
        ],
        cwd=tmp_path,
        env={**os.environ, "MUJOCO_GL": "disable"},
        capture_output=True,
        text=True,
        check=True,
    )
