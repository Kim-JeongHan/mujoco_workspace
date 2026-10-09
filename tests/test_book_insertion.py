"""Centered grasps, physical carry checks, and released book insertion."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mujoco_lab import RobotSpec, Simulator, create_book_insertion
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors.book_expert import BookInsertionExpert
from mujoco_lab.behaviors.bookshelf_recipe import load_recipe as load_book_recipe
from mujoco_lab.behaviors.task import BookTask
from mujoco_lab.planning import default_planning
from mujoco_lab.trajectory_generation.trajectory import JointTrajectory


def make_expert(planning=None, *, method="sampling"):
    sim = Simulator(
        create_book_insertion(),
        robots=[RobotSpec("forte", "forte", config=load_robot_config("forte"))],
    )
    task = BookTask(sim)
    expert = BookInsertionExpert(
        task,
        recipe=load_book_recipe("forte"),
        method=method,
        planning=(planning or default_planning()) if method == "sampling" else None,
    )
    return sim, task, expert


def test_status_grasp_pose_tracks_book_and_preserves_reset_snapshot():
    sim, task, expert = make_expert()
    initial = task.status()
    initial_pose = initial.grasp_pose.as_matrix()
    np.testing.assert_allclose(expert.pick_pose.as_matrix(), initial_pose)
    address = int(sim.model.jnt_qposadr[task.joint])
    sim.data.qpos[address : address + 3] += [0.1, -0.1, 0.2]
    sim.data.qpos[address + 3 : address + 7] = Rotation.from_euler("x", np.pi / 4).as_quat(
        scalar_first=True
    )
    mujoco.mj_forward(sim.model, sim.data)
    moved = task.status()
    rotation = sim.data.xmat[task.body].reshape(3, 3)
    center = sim.data.xpos[task.body]
    np.testing.assert_allclose(
        rotation.T @ (moved.grasp_pose.as_translation() - center),
        [-task.depth / 2 + 0.012, 0, 0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        rotation.T @ moved.grasp_pose.as_rotation().as_matrix(),
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        atol=1e-12,
    )
    np.testing.assert_array_equal(initial.grasp_pose.as_matrix(), initial_pose)
    np.testing.assert_array_equal(expert.pick_pose.as_matrix(), initial_pose)
    assert not np.allclose(moved.grasp_pose.as_matrix(), initial_pose)
    expert.reset()
    np.testing.assert_allclose(expert.pick_pose.as_matrix(), moved.grasp_pose.as_matrix())


@pytest.mark.parametrize("method", ["sampling", "heuristic"])
def test_planning_and_actions_do_not_modify_live_free_book(method):
    sim, task, expert = make_expert(method=method)
    before = sim.data.qpos.copy()
    trajectory, reason = expert.motion.make_trajectory(
        expert.motion_request("approach", expert.pick_pose)
    )
    assert reason is None
    assert trajectory.path.shape[1] == 8
    if method == "heuristic":
        assert trajectory.smooth
        knot_speeds = [
            np.linalg.norm(trajectory.sample(time).velocity[:7])
            for time in trajectory.waypoint_times[1:-1]
        ]
        assert min(knot_speeds) > 1e-6
    np.testing.assert_array_equal(sim.data.qpos, before)
    assert not task.has_grasp()
    assert sim.model.neq == 1  # Gripper parallelism only; no object weld.
    assert sim.model.jnt_type[task.joint] == mujoco.mjtJoint.mjJNT_FREE


def test_no_physical_grasp_cannot_start_carry():
    sim, _, expert = make_expert()
    expert.stage = expert.STAGES.index("lift")
    # A closing target does not establish a physical grasp.
    expert.robot.gripper.set_target(expert.motion_request("close", expert.pick_pose).gripper_target)
    expert.act()
    assert expert.failed
    assert expert.failure_reason == "No two-finger physical grasp for lift"


def test_sampling_replaces_a_colliding_joint_goal_without_changing_its_grasp_pose():
    sim, _, expert = make_expert()
    request = expert.motion_request("approach", expert.pick_pose)
    # Sequential Cartesian IK reaches a folded wrist branch at this same goal.
    blocked = expert.motion.cartesian_path(request.waypoints[0])[-1]
    assert not request.checker.is_collision_free(blocked)
    before = sim.data.qpos.copy()
    blocked_pose = expert.motion.waypoint_poses((blocked,))[0]

    trajectory, reason = expert.motion.make_trajectory(replace(request, waypoints=(blocked,)))

    assert reason is None
    selected = trajectory.path[-1, :7]
    assert not np.allclose(selected, blocked)
    assert request.checker.is_collision_free(selected)
    selected_pose = expert.motion.waypoint_poses((selected,))[0]
    np.testing.assert_allclose(
        selected_pose.as_translation(), blocked_pose.as_translation(), atol=1e-5
    )
    np.testing.assert_allclose(
        selected_pose.as_rotation().as_matrix(), blocked_pose.as_rotation().as_matrix(), atol=1e-5
    )
    for time in np.linspace(0, trajectory.duration, 101):
        assert request.checker.is_collision_free(trajectory.sample(time).position[:7])
    np.testing.assert_array_equal(sim.data.qpos, before)


@pytest.mark.parametrize("method", ["sampling", "heuristic"])
def test_medium_book_is_lifted_inserted_and_released(method):
    sim, task, expert = make_expert(method=method)
    sim.target_updater = expert.update
    sim.run_steps(60000)
    assert not expert.failed, expert.failure_reason
    assert expert.get_stage_name() == "settle"
    assert task.max_height > task.start_center[2] + 0.04
    status = task.status()
    assert status.is_released_stable
    assert status.is_touch_shelf and not status.is_touch_robot
    assert status.position_error < 0.015
    assert status.rotation_error < 0.12
    if method == "heuristic":
        assert expert.motion.planning is None
    assert not sim.data.warning.number.any()


@pytest.mark.parametrize(
    ("method", "phase", "dwell"),
    [
        ("heuristic", "approach", 0.0),
        ("heuristic", "close", 0.4),
        ("sampling", "approach", 0.4),
    ],
)
def test_heuristic_waits_only_for_grasp_and_release(method, phase, dwell, monkeypatch):
    sim, task, expert = make_expert(method=method)
    expert.stage = expert.STAGES.index(phase)
    current = expert.robot.state.snapshot().qpos[:7].copy()
    width = expert.gripper.target_for_mode(expert.recipe.stages[expert.stage].gripper_mode)
    path = np.tile(np.r_[current, width], (2, 1))
    expert.execution.start(JointTrajectory(path, 1.0, 1.0), sim.data.time)
    if expert.recipe.stages[expert.stage].gripper_mode == 1:
        monkeypatch.setattr(task, "has_grasp", lambda: True)
    if dwell:
        assert not expert._advance_stage(current, path_complete=True)
        sim.data.time = dwell
    assert expert._advance_stage(current, path_complete=True)
