"""Physical progress milestones remain observational and episode-local."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv
from mujoco_lab.learning.evaluation.evaluator import evaluation_log_metrics, summarize
from mujoco_lab.learning.evaluation.progress import CubeProgressTracker
from mujoco_lab.tasks import CubeStackTask
from mujoco_lab.tasks.cube_stack_motion import has_physical_grasp


class FakeTask:
    cubes = 2

    def __init__(self):
        self.simulator = SimpleNamespace(data=SimpleNamespace(time=0.0), robots={"arm": object()})
        self.heights = np.array([0.03, 0.07])
        self.distances = np.array([0.2, 0.3])
        self.stable = np.array([False, False])
        self.samples = 0

    def measurements(self):
        self.samples += 1
        return SimpleNamespace(
            centers=np.column_stack((np.zeros(2), np.zeros(2), self.heights)).copy(),
            goal_distances=self.distances.copy(),
            stable_placement=self.stable.copy(),
        )


def test_progress_requires_grasped_lift_and_held_released_placement(monkeypatch):
    task = FakeTask()
    gripped = {0: False, 1: False}
    monkeypatch.setattr(
        "mujoco_lab.learning.evaluation.progress.has_physical_grasp",
        lambda robot, index: gripped[index],
    )
    tracker = CubeProgressTracker(task)
    task.heights[0] += 0.05
    tracker.observe()  # A bounce cannot count as a carried lift.
    assert tracker.result()["cube_best_stage"] == [0, 0]

    task.heights[0] -= 0.05
    gripped[0] = True
    tracker.observe()
    assert tracker.result()["cube_best_stage"] == [1, 0]
    task.heights[0] += 0.05
    tracker.observe()
    assert tracker.result()["cube_best_stage"] == [2, 0]
    assert tracker.result()["best_progress"] == pytest.approx(1 / 3)
    gripped[0] = False
    task.stable[0] = True
    task.simulator.data.time = 1.0
    tracker.observe()
    task.simulator.data.time = 1.49
    tracker.observe()
    assert tracker.result()["cube_placed"] == [False, False]
    task.stable[0] = False
    task.simulator.data.time = 1.5
    tracker.observe()  # Interrupted contact starts the hold again.
    task.stable[0] = True
    task.simulator.data.time = 2.0
    tracker.observe()
    task.simulator.data.time = 2.5
    tracker.observe()
    task.stable[0] = False
    task.distances[0] = 0.6
    tracker.observe()  # History remains, while final distance records the drop.
    result = tracker.result()
    assert result["cube_best_stage"] == [3, 0]
    assert result["cube_placed"] == [True, False]
    assert result["best_progress"] == pytest.approx(0.5)
    assert result["final_goal_distance"] == pytest.approx(0.45)
    assert task.samples == 10  # Initial sample plus one per observe, no physics step.

    fresh = CubeProgressTracker(task)
    assert fresh.result()["cube_best_stage"] == [0, 0]


def test_summary_aggregates_progress_only_when_present():
    def row(progress):
        return {
            "success": False,
            "steps": 2,
            "clipped_actions": 1,
            "clipped_action_axes": [1, 0],
            "action_overrun_sum": [2.0, 0.0],
            "action_overrun_max": [2.0, 0.0],
            "termination_reason": "time_limit",
            **progress,
        }

    first = row(
        {
            "best_progress": 0.5,
            "final_goal_distance": 0.1,
            "grasp_fraction": 0.5,
            "lift_fraction": 0.5,
            "place_fraction": 0.5,
            "cube_grasped": [True, False],
            "cube_lifted": [True, False],
            "cube_placed": [True, False],
        }
    )
    second = row(
        {
            "best_progress": 0.0,
            "final_goal_distance": 0.3,
            "grasp_fraction": 0.0,
            "lift_fraction": 0.0,
            "place_fraction": 0.0,
            "cube_grasped": [False, False],
            "cube_lifted": [False, False],
            "cube_placed": [False, False],
        }
    )
    summary = summarize([first, second])
    assert summary["mean_best_progress"] == pytest.approx(0.25)
    assert summary["mean_final_goal_distance"] == pytest.approx(0.2)
    assert summary["cube_grasp_fraction"] == pytest.approx([0.5, 0.0])
    assert summary["action_clip_axis_fraction"] == pytest.approx([0.5, 0.0])
    assert evaluation_log_metrics(summary)["eval/mean_best_progress"] == pytest.approx(0.25)
    assert "mean_best_progress" not in summarize([row({})])


def test_physical_placement_predicates_and_measurement_do_not_advance_task():
    simulator = Simulator(
        create_cube_stack(2, environment="table_shelf"),
        robots=[RobotSpec("forte", "forte")],
        dt=0.002,
    )
    task = CubeStackTask(simulator, 2)
    env = CubeStackEnv(task, xy_range=0.02, min_gap=0.01, max_steps=10)
    env.reset(seed=10000)
    before_lift = task.max_lift.copy()
    before_hold = task._stable_since
    initial = task.measurements()
    np.testing.assert_allclose(task.max_lift, before_lift)
    assert task._stable_since is before_hold
    assert not initial.stable_placement.any()

    adr = int(simulator.model.joint("cube0/object_joint_0").qposadr[0])
    dof = int(simulator.model.joint("cube0/object_joint_0").dofadr[0])
    simulator.data.qpos[adr : adr + 3] = task.goals[0]
    mujoco.mj_forward(simulator.model, simulator.data)
    placed = task.measurements()
    assert placed.supported[0] and placed.stable_placement[0]
    simulator.data.qvel[dof] = 0.1
    assert not task.measurements().stable_placement[0]
    simulator.data.qvel[dof] = 0
    simulator.data.qpos[adr + 2] += 0.011
    mujoco.mj_forward(simulator.model, simulator.data)
    floating = task.measurements()
    assert floating.aligned[0] and floating.at_height[0]
    assert not floating.supported[0] and not floating.stable_placement[0]
    np.testing.assert_allclose(task.max_lift, before_lift)
    assert task._stable_since is before_hold


def test_both_forte_finger_pads_are_required_for_grasp():
    cube = 1
    left = 2
    right = 3
    names = {
        cube: "cube0/object_0",
        left: "forte/gripper_left_pad",
        right: "forte/gripper_right_pad",
    }
    # The real helper resolves the cube by name as well as contact endpoints.
    model = SimpleNamespace(
        geom=lambda index: SimpleNamespace(
            id=cube if isinstance(index, str) else index,
            name=names[index] if isinstance(index, int) else index,
        )
    )
    data = SimpleNamespace(contact=[SimpleNamespace(geom1=cube, geom2=left)])
    robot = SimpleNamespace(model=model, data=data, name="forte", robot_type="forte")
    assert not has_physical_grasp(robot, 0)
    data.contact.append(SimpleNamespace(geom1=right, geom2=cube))
    assert has_physical_grasp(robot, 0)
