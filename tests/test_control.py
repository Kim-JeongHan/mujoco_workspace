"""ForteV1_RobStride selected-joint control with a separate gripper target."""

import mujoco
import numpy as np
import pytest
from controller_config import create_test_controller

from mujoco_lab import RobotSpec, Simulator, create_environment
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.rendering.annotations import annotate_controller
from mujoco_lab.trajectory_generation.demo import osc_circle_target
from mujoco_lab.utils import Transform


def forte_simulator():
    return Simulator(
        create_environment("empty"),
        robots=[RobotSpec("robot", "forte", config=load_robot_config("forte"))],
    )


def test_computed_torque_tracks_joint_reference():
    sim = forte_simulator()
    robot = sim.robots["robot"]
    robot.change_controller(create_test_controller(robot, controller="computed_torque"))
    initial = robot.target.position.copy()
    assert robot.gripper is not None
    robot.set_goal(np.r_[initial + 0.05, robot.gripper.get_target()])
    for _ in range(400):
        sim.step()
        error = robot.controller.get_tracking_error()
        assert error is not None and error < 0.05
    assert sim.data.time == pytest.approx(400 * sim.dt)
    assert np.isfinite(sim.data.qpos).all()
    assert np.isfinite(sim.data.qvel).all()
    assert not sim.data.warning.number.any()
    assert np.max(np.abs(robot.get_control_state().qpos - initial)) > 0.01


def test_circle_targets_and_joint_annotations_follow_translated_rotated_mount():
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
    expected_grasp = moved_robot.state.get_frame_position("grasp").copy()
    moved_sim.step()
    scene = mujoco.MjvScene(moved_model, maxgeom=100)
    scene.ngeom = 0
    annotate_controller(scene, moved_robot, moved_data, mujoco.MjData(moved_model))
    np.testing.assert_allclose(scene.geoms[0].pos, expected_grasp, atol=1e-6)


@pytest.mark.parametrize("mode", ["pd", "computed_torque"])
def test_torque_control_rejects_position_actuators(mode):
    sim = Simulator(
        create_environment("empty"),
        robots=[RobotSpec("robot", "panda", config=load_robot_config("panda"))],
    )
    with pytest.raises(ValueError, match="requires torque/force actuators"):
        create_test_controller(sim.robots["robot"], controller=mode)
