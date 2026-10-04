"""Physical progress milestones remain observational and episode-local."""

from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from controller_config import create_test_controller

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import CubeStackTask
from mujoco_lab.behaviors.cube_stack import has_physical_grasp
from mujoco_lab.learning.config.config import RolloutConfig
from mujoco_lab.learning.datasets.normalizer import Normalizer
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv
from mujoco_lab.learning.evaluation.evaluator import (
    PolicyEvaluator,
    evaluation_log_metrics,
    summarize,
)
from mujoco_lab.learning.evaluation.progress import CubeProgressTracker
from mujoco_lab.learning.policies.factory import build_policy


class FakeTask(CubeStackTask):
    def __init__(self, cubes=2):
        simulator = Simulator(
            create_cube_stack(cubes),
            robots=[RobotSpec("forte", "forte", config=load_robot_config("forte"))],
        )
        super().__init__(simulator, cubes)
        self.heights = np.array([0.03, 0.07])[:cubes]
        self.distances = np.array([0.2, 0.3])[:cubes]
        self.stable = np.zeros(cubes, dtype=bool)
        self.samples = 0

    def measurements(self):
        self.samples += 1
        return SimpleNamespace(
            centers=np.column_stack((np.zeros(self.cubes), np.zeros(self.cubes), self.heights)),
            goal_distances=self.distances.copy(),
            stable_placement=self.stable.copy(),
            supported=self.stable.copy(),
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
    task.status()
    tracker.observe()
    task.simulator.data.time = 1.49
    task.status()
    tracker.observe()
    assert tracker.result()["cube_placed"] == [False, False]
    task.stable[0] = False
    task.simulator.data.time = 1.5
    task.status()
    tracker.observe()  # Interrupted contact starts the hold again.
    task.stable[0] = True
    task.simulator.data.time = 2.0
    task.status()
    tracker.observe()
    task.simulator.data.time = 2.5
    task.status()
    tracker.observe()
    task.stable[0] = False
    task.distances[0] = 0.6
    task.status()
    tracker.observe()  # History remains, while final distance records the drop.
    result = tracker.result()
    assert result["cube_best_stage"] == [3, 0]
    assert result["cube_placed"] == [True, False]
    assert result["best_progress"] == pytest.approx(0.5)
    assert result["final_goal_distance"] == pytest.approx(0.45)
    assert task.samples == 16  # Initial sample, nine observations, and six task updates.

    fresh = CubeProgressTracker(task)
    assert fresh.result()["cube_best_stage"] == [0, 0]


def test_task_tracks_independent_holds_and_resets_completion_history():
    task = FakeTask()
    task.stable[0] = True
    task.simulator.data.time = 1.0
    assert not task.status().released_stable_stack
    task.stable[1] = True
    task.simulator.data.time = 1.25
    assert not task.status().released_stable_stack
    task.simulator.data.time = 1.5
    assert not task.status().released_stable_stack
    assert task.completed_placements().tolist() == [True, False]
    task.simulator.data.time = 1.75
    assert task.status().released_stable_stack

    task.stable[0] = False
    task.simulator.data.time = 1.8
    assert not task.status().released_stable_stack
    assert task.completed_placements().tolist() == [True, True]
    task.stable[0] = True
    task.simulator.data.time = 2.0
    assert not task.status().released_stable_stack
    task.simulator.data.time = 2.49
    assert not task.status().released_stable_stack
    task.simulator.data.time = 2.5
    assert task.status().released_stable_stack

    # Callers cannot alter task history through the returned snapshot.
    completed = task.completed_placements()
    completed[:] = False
    assert task.completed_placements().all()
    task.reset()
    assert not task.completed_placements().any()
    assert not task.status().released_stable_stack


def test_progress_retains_placement_completed_between_observations():
    task = FakeTask()
    tracker = CubeProgressTracker(task)
    task.stable[0] = True
    task.simulator.data.time = 1.0
    task.status()
    task.simulator.data.time = 1.49
    tracker.observe()
    assert tracker.result()["cube_placed"] == [False, False]

    task.simulator.data.time = 1.5
    task.status()
    task.stable[0] = False
    task.simulator.data.time = 1.51
    task.status()
    before_hold = task._stable_since.copy()
    before_time = float(task.simulator.data.time)
    tracker.observe()
    assert tracker.result()["cube_placed"] == [True, False]
    assert task._stable_since == before_hold
    assert task.simulator.data.time == before_time


@pytest.mark.parametrize("cubes", [1, 2])
def test_success_inside_action_reports_complete_progress(cubes, monkeypatch):
    simulator = Simulator(
        create_cube_stack(cubes),
        robots=[RobotSpec("forte", "forte", config=load_robot_config("forte"))],
        dt=0.002,
    )
    robot = simulator.robots["forte"]
    robot.change_controller(create_test_controller(robot, controller="pd", frame="grasp"))
    task = CubeStackTask(simulator, cubes)
    env = CubeStackEnv(task, xy_range=0, max_steps=100, physics_steps_per_action=5)
    measure = task.measurements
    # Stable placement begins at the first physics tick, before the tracker runs.
    monkeypatch.setattr(
        task,
        "measurements",
        lambda: replace(measure(), stable_placement=np.full(cubes, simulator.data.time >= 0.002)),
    )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = build_policy(
        "mse", state_dim=state_dim, action_dim=action_dim, chunk_size=4, hidden_dims=()
    )
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
    normalizer = Normalizer(
        state_mean=np.zeros(state_dim, dtype=np.float32),
        state_std=np.ones(state_dim, dtype=np.float32),
        action_mean=np.zeros(action_dim, dtype=np.float32),
        action_std=np.ones(action_dim, dtype=np.float32),
    )
    rows, summary = PolicyEvaluator(
        env,
        RolloutConfig(num_episodes=1, max_steps=100, video_episodes=0),
        torch.device("cpu"),
    ).evaluate(
        policy,
        normalizer,
        {"architecture": {"obs_horizon": 1, "execution_horizon": 4}},
        flow_num_steps=1,
    )
    row = rows[0]
    assert row["success"]
    assert row["steps"] == 51
    assert row["sim_seconds"] == pytest.approx(0.502)
    assert row["cube_placed"] == [True] * cubes
    assert row["best_progress"] == row["place_fraction"] == 1.0
    assert summary["success_rate"] == summary["mean_place_fraction"] == 1.0
    assert summary["mean_best_progress"] == 1.0


def test_summary_aggregates_progress_only_when_present():
    def row(progress):
        return {
            "success": False,
            "steps": 2,
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
    assert evaluation_log_metrics(summary)["eval/mean_best_progress"] == pytest.approx(0.25)
    assert "mean_best_progress" not in summarize([row({})])


def test_physical_placement_predicates_and_measurement_do_not_advance_task():
    simulator = Simulator(
        create_cube_stack(2, environment="table_shelf"),
        robots=[RobotSpec("forte", "forte", config=load_robot_config("forte"))],
        dt=0.002,
    )
    task = CubeStackTask(simulator, 2)
    env = CubeStackEnv(task, xy_range=0.02, min_gap=0.01, max_steps=10)
    env.reset(seed=10000)
    before_lift = task.max_lift.copy()
    before_hold = task._stable_since.copy()
    before_placed = task.completed_placements()
    initial = task.measurements()
    np.testing.assert_allclose(task.max_lift, before_lift)
    assert task._stable_since == before_hold
    np.testing.assert_array_equal(task.completed_placements(), before_placed)
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
    assert task._stable_since == before_hold
    np.testing.assert_array_equal(task.completed_placements(), before_placed)


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
