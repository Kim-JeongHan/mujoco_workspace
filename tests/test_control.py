"""ForteV1_RobStride selected-joint control with a separate gripper target."""

import mujoco
import numpy as np
import pytest
from controller_config import create_test_controller

from mujoco_lab import RobotSpec, Simulator, create_environment
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.control import demo_target_updater
from mujoco_lab.control.trajectory import osc_circle_target
from mujoco_lab.rendering.annotations import annotate_controller
from mujoco_lab.utils import Transform


def forte_simulator():
    return Simulator(
        create_environment("empty"),
        robots=[RobotSpec("robot", "forte", config=load_robot_config("forte"))],
    )


@pytest.mark.parametrize("mode", ["pd", "osc", "computed_torque"])
def test_feedback_control_is_finite(mode):
    sim = forte_simulator()
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


@pytest.mark.parametrize("mode", ["pd", "osc", "computed_torque"])
def test_torque_control_rejects_position_actuators(mode):
    sim = Simulator(
        create_environment("empty"),
        robots=[RobotSpec("robot", "panda", config=load_robot_config("panda"))],
    )
    with pytest.raises(ValueError, match="requires torque/force actuators"):
        create_test_controller(sim.robots["robot"], controller=mode)
