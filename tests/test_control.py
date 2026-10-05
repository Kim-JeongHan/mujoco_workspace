"""ForteV1_RobStride selected-joint control with a separate gripper target."""

import mujoco
import numpy as np
import pytest
from controller_config import create_test_controller

from mujoco_lab import ENVIRONMENT_NAMES, RobotSpec, Simulator, create_environment
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.assets.robot.forte_coordinates import HOME_RADIANS as HOME_QPOS
from mujoco_lab.control import ControlTarget, demo_target_updater
from mujoco_lab.control.osc import OperationalSpaceControl
from mujoco_lab.control.pd import JointSpacePD
from mujoco_lab.control.trajectory import osc_circle_target
from mujoco_lab.rendering.annotations import annotate_controller
from mujoco_lab.state import JointState
from mujoco_lab.utils import Transform


def forte_simulator(environment="empty"):
    return Simulator(
        create_environment(environment),
        robots=[RobotSpec("robot", "forte", config=load_robot_config("forte"))],
    )


@pytest.mark.parametrize("dt", [0.001, 0.002, 0.005])
def test_pd_period_matches_simulator_and_survives_reset(dt):
    sim = Simulator(
        create_environment("empty"),
        robots=[RobotSpec("robot", "forte", config=load_robot_config("forte"))],
        dt=dt,
    )
    robot = sim.robots["robot"]
    controller = create_test_controller(robot, controller="pd")
    robot.change_controller(controller)
    sim.step()
    sim.reset()
    assert controller.control_dt == dt


@pytest.mark.parametrize("dt", [0, -0.001, float("nan"), float("inf")])
def test_pd_rejects_invalid_control_period(dt):
    with pytest.raises(ValueError, match="control_dt must be finite and positive"):
        JointSpacePD(np.ones(7), np.ones(7), control_dt=dt)


def test_seven_axis_controller_math_on_numpy_snapshots():
    state = JointState(0.0, HOME_QPOS.copy(), np.zeros(7), np.ones(7))
    np.testing.assert_array_equal(
        JointSpacePD(np.ones(7), np.ones(7), control_dt=0.002).compute(
            state, ControlTarget(HOME_QPOS)
        ),
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
    controller = create_test_controller(robot, controller=mode)
    robot.change_controller(controller)
    assert robot.state.nq == robot.state.nv == 9
    assert robot.num_actuators == 8
    assert robot.target.position.shape == ((7,) if mode == "pd" else (3,))
    state = robot.state
    assert controller._owner is state
    assert robot.control_joint_names == tuple(state.joint_names[:7])
    joint_slots = robot.control_joint_slots
    np.testing.assert_array_equal(
        state.get_jacobian("grasp", joint_slots),
        robot.state.get_jacobian("grasp")[:, joint_slots],
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
        robot.gripper.set_target(robot.gripper.get_control_limits()[0] - 0.001)
    with pytest.raises(ValueError, match="gripper target"):
        robot.gripper.set_target(np.nan)
    sim.reset()
    assert robot.gripper.get_target() == 0


@pytest.mark.parametrize("mode", ["pd", "osc"])
@pytest.mark.parametrize("environment", ENVIRONMENT_NAMES)
def test_feedback_control_is_finite_in_each_environment(mode, environment):
    sim = forte_simulator(environment)
    robot = sim.robots["robot"]
    robot.change_controller(create_test_controller(robot, controller=mode))
    if mode == "osc":
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
        robots=[
            RobotSpec(
                "robot",
                "forte",
                Transform.from_pose_mmdeg([100, -200, 800, 0, 0, 90]),
                config=load_robot_config("forte"),
            )
        ],
    )
    moved_model, moved_data = moved_sim.model, moved_sim.data
    moved_robot = moved_sim.robots["robot"]
    moved_robot.change_controller(create_test_controller(moved_robot, controller="osc"))
    moved_sim.target_updater = demo_target_updater(moved_sim, {"robot": "osc"})
    rotation = moved_data.body("robot/base_link").xmat.reshape(3, 3)
    translation = moved_data.body("robot/base_link").xpos
    center = robot.state.get_frame_position("grasp") - sim.data.body("robot/base_link").xpos
    entry = osc_circle_target(
        2.0,
        start_time=0,
        start_position=robot.state.get_frame_position("grasp"),
        base_position=sim.data.body("robot/base_link").xpos,
        base_rotation=np.eye(3),
        center=center,
    )
    entry_distance = np.linalg.norm(entry.position - robot.state.get_frame_position("grasp"))
    assert entry_distance == pytest.approx(0.025, abs=1e-5)
    for elapsed in [0.0, 1.25, 2.5]:
        reference = osc_circle_target(
            elapsed + 2,
            start_time=0,
            start_position=robot.state.get_frame_position("grasp"),
            base_position=np.zeros(3),
            base_rotation=np.eye(3),
            center=center,
        )
        actual = osc_circle_target(
            elapsed + 2,
            start_time=0,
            start_position=moved_robot.state.get_frame_position("grasp"),
            base_position=translation,
            base_rotation=rotation,
            center=center,
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
    sim = Simulator(
        create_environment("empty"),
        robots=[RobotSpec("robot", robot, config=load_robot_config(robot))],
    )
    with pytest.raises(ValueError, match="requires torque/force actuators"):
        create_test_controller(sim.robots["robot"], controller=mode)
