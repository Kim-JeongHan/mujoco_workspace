"""Centered grasps, physical carry checks, and released book insertion."""

import mujoco
import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_book_insertion
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors.book import (
    BookTask,
    create_book_controller,
)
from mujoco_lab.behaviors.book_expert import BookInsertionExpert
from mujoco_lab.behaviors.bookshelf_recipe import load_recipe as load_book_recipe
from mujoco_lab.control.trajectory import JointTrajectory
from mujoco_lab.planning import default_planning


def make_expert(planning=None, *, method="sampling"):
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
        method=method,
        planning=(planning or default_planning()) if method == "sampling" else None,
    )
    return sim, task, expert


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


def test_close_mode_waits_for_contact_then_completes(monkeypatch):
    _, task, expert = make_expert(method="heuristic")
    expert.stage = expert.STAGES.index("close")
    monkeypatch.setattr(task, "has_grasp", lambda: False)
    expert.act()
    assert not expert.failed
    current = expert.robot.target.position.copy()
    expert.simulator.data.time = expert.execution.trajectory.duration + expert.recipe.stage_dwell_s
    assert not expert._advance_stage(current, path_complete=True)
    monkeypatch.setattr(task, "has_grasp", lambda: True)
    assert expert._advance_stage(current, path_complete=True)


@pytest.mark.parametrize("phase", ["approach", "lift"])
@pytest.mark.parametrize("gripper_mode", [0, 1])
def test_gripper_mode_controls_grasp_requirement_independently_of_stage_name(
    phase, gripper_mode, monkeypatch
):
    _, task, expert = make_expert(method="heuristic")
    expert.stage = expert.STAGES.index(phase)
    expert.recipe = expert.recipe.model_copy(
        update={
            "stages": tuple(
                stage.model_copy(update={"gripper_mode": gripper_mode})
                if stage.name == phase
                else stage
                for stage in expert.recipe.stages
            ),
        }
    )
    monkeypatch.setattr(task, "has_grasp", lambda: False)
    expected = f"No two-finger physical grasp for {phase}" if gripper_mode == 1 else None
    assert expert._execution_failure() == expected


def test_required_grasp_allows_transient_loss_and_resets_on_recovery(monkeypatch):
    sim, task, expert = make_expert(method="heuristic")
    expert.stage = expert.STAGES.index("insert")
    current = expert.robot.state.snapshot().qpos[:7]
    path = np.tile(np.r_[current, 0.0], (2, 1))
    expert.execution.start(JointTrajectory(path, 1.0, 1.0), sim.data.time)
    grasped = False
    monkeypatch.setattr(task, "has_grasp", lambda: grasped)
    assert expert._execution_failure() is None
    sim.data.time = expert.recipe.lost_grasp_grace_s
    assert expert._execution_failure() is None
    grasped = True
    assert expert._execution_failure() is None
    grasped = False
    sim.data.time = 1.0
    assert expert._execution_failure() is None
    sim.data.time += expert.recipe.lost_grasp_grace_s + sim.dt
    assert expert._execution_failure() == "Lost two-finger grasp during insert"
    expert.stage = len(expert.STAGES)
    assert expert._execution_failure() is None


@pytest.mark.parametrize("method", ["sampling", "heuristic"])
def test_medium_book_is_lifted_inserted_and_released(method):
    sim, task, expert = make_expert(method=method)
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
    if method == "heuristic":
        assert expert.motion.planning is None
    assert not sim.data.warning.number.any()


@pytest.mark.parametrize(
    ("method", "phase", "dwell"),
    [
        *(
            ("heuristic", phase, 0.0)
            for phase in ("approach", "pick", "preinsert", "insert", "lower", "retract")
        ),
        ("heuristic", "close", 0.4),
        ("heuristic", "release", 0.4),
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
