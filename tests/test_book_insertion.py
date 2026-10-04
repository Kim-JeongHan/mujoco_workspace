"""Centered grasps, physical carry checks, and released sampling insertion."""

import mujoco
import numpy as np
import pytest

from mujoco_lab import BOOK_TYPES, RobotSpec, Simulator, create_book_insertion
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors.book import (
    BookTask,
    center_grasp_pose,
    create_book_controller,
)
from mujoco_lab.behaviors.book_expert import BookInsertionExpert
from mujoco_lab.behaviors.bookshelf_recipe import load_recipe as load_book_recipe
from mujoco_lab.control import create_controller
from mujoco_lab.planning import default_planning, planner_from_config


def make_expert(planner=None):
    sim = Simulator(
        create_book_insertion(),
        robots=[RobotSpec("forte", "forte", config=load_robot_config("forte"))],
    )
    robot = sim.robots["forte"]
    robot.change_controller(
        create_book_controller(robot, load_robot_config(robot.robot_type).controller)
    )
    task = BookTask(sim)
    expert = BookInsertionExpert(
        task,
        recipe=load_book_recipe(next(iter(task.simulator.robots.values())).robot_type),
        planner=planner or planner_from_config(default_planning()),
    )
    return sim, task, expert


@pytest.mark.parametrize("book_type", BOOK_TYPES)
def test_grasp_height_is_book_center_and_jaws_cross_thickness(book_type):
    sim = Simulator(create_book_insertion(book_type))
    center = sim.data.body("book").xpos
    rotation = sim.data.body("book").xmat.reshape(3, 3)
    size = 2 * sim.model.geom("book/collision").size
    pose = center_grasp_pose(center, rotation, size[0])
    assert pose.as_translation()[2] == pytest.approx(center[2])
    assert pose.as_translation()[2] < center[2] + size[2] / 2
    np.testing.assert_allclose(pose.as_rotation().as_matrix()[:, 0], rotation[:, 1], atol=1e-12)
    np.testing.assert_allclose(pose.as_rotation().as_matrix()[:, 2], rotation[:, 0], atol=1e-12)
    np.testing.assert_allclose(sim.data.site("book/grasp_target").xpos, pose.as_translation())


def test_planning_and_actions_do_not_modify_live_free_book():
    sim, task, expert = make_expert()
    before = sim.data.qpos.copy()
    trajectory, reason = expert.motion.make_trajectory(
        expert.motion_request("approach", expert.pick_pose)
    )
    assert reason is None
    assert trajectory.path.shape[1] == 8
    np.testing.assert_array_equal(sim.data.qpos, before)
    assert not task.has_grasp()
    assert sim.model.neq == 1  # Gripper parallelism only; no object weld.
    assert sim.model.jnt_type[task.joint] == mujoco.mjtJoint.mjJNT_FREE


def test_book_controller_tuning_preserves_asset_contact_settings():
    sim, _, expert = make_expert()
    assert expert.closed == pytest.approx(-0.037)
    base_controller = create_controller(
        expert.robot, load_robot_config(expert.robot.robot_type).controller
    )
    np.testing.assert_allclose(
        expert.robot.controller.kp, base_controller.kp * np.r_[np.full(4, 4), np.full(3, 24)]
    )
    np.testing.assert_allclose(
        expert.robot.controller.kd,
        base_controller.kd * np.r_[np.full(4, 2), np.full(3, 2 * np.sqrt(6))],
    )
    np.testing.assert_allclose(sim.model.geom("book/collision").friction, [0.9, 0.005, 0.0001])
    assert sim.model.geom("forte/gripper_left_pad").friction[1] == pytest.approx(0.05)
    fresh = Simulator(
        create_book_insertion(),
        robots=[RobotSpec("forte", "forte", config=load_robot_config("forte"))],
    )
    np.testing.assert_allclose(fresh.model.joint("forte/gripper_left_joint").range, [-0.037, 0])
    for side in ("left", "right"):
        np.testing.assert_allclose(
            fresh.model.geom(f"forte/gripper_{side}_pad").friction, [1.0, 0.05, 0.001]
        )
    friction = sim.model.geom_friction.copy()
    create_book_controller(expert.robot, expert.robot.config.controller)
    np.testing.assert_array_equal(sim.model.geom_friction, friction)


def test_missing_route_stops_with_reason_and_reset_clears_failure():
    class RejectPlanner:
        name = "reject"
        seed = 3

        def plan(self, *args, **kwargs):
            return None

    sim, _, expert = make_expert(RejectPlanner())
    sim.target_updater = expert.update
    sim.run_steps(10)
    assert expert.failed
    assert expert.failure_reason == "No reject route for approach"
    assert sim.data.time == pytest.approx(sim.dt)
    sim.reset()
    expert.reset()
    assert not expert.failed and expert.failure_reason is None
    assert expert.get_stage_name() == "approach"
    assert expert.motion.planning_epoch == 0


def test_no_physical_grasp_cannot_start_carry():
    sim, _, expert = make_expert()
    expert.stage = expert.STAGES.index("lift")
    # A closing target does not establish a physical grasp.
    expert.robot.gripper.set_target(expert.closed)
    expert.act()
    assert expert.failed
    assert expert.failure_reason == "No two-pad physical grasp for lift"


def test_sampling_medium_book_is_lifted_inserted_and_released():
    sim, task, expert = make_expert()
    sim.target_updater = expert.update
    sim.run_steps(60000)
    assert not expert.failed, expert.failure_reason
    assert expert.get_stage_name() == "settle"
    assert task.max_height > task.start_center[2] + 0.04
    status = task.status()
    assert status.released_stable
    assert status.supported and not status.touching_robot
    assert status.position_error < 0.015
    assert status.rotation_error < 0.12
    assert expert.motion.planning_epoch == 2
    assert not sim.data.warning.number.any()
