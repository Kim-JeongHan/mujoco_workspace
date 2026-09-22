"""Sampling plans are executed through the physical cube stacking task."""

import mujoco
import numpy as np
import pytest

from mujoco_lab import create_cube_stack
from mujoco_lab.tasks import CubeStackTask


@pytest.mark.parametrize("planner", ["rrt_connect", "rrt", "prm"])
def test_sampling_planners_complete_released_two_cube_stack(planner):
    simulator = create_cube_stack(2)
    task = CubeStackTask(
        simulator,
        2,
        method="sampling",
        planner=planner,
        planning_budget=100 if planner == "prm" else 500,
    )
    simulator.run_steps(22000)
    status = task.status()
    assert not task.failed, task.failure_reason
    assert task._planning_epoch >= 12
    assert task.stage_name == "settle"
    assert status.released_stable_stack
    assert status.support_contacts
    assert np.all(task.max_lift > task.starts[:, 2] + 0.04)
    assert not simulator.data.warning.number.any()


def test_sampling_no_route_stops_cleanly_and_reset_replans(monkeypatch):
    simulator = create_cube_stack(2)
    task = CubeStackTask(simulator, 2, method="sampling")
    original = task._plan_stage
    monkeypatch.setattr(task, "_plan_stage", lambda *_: None)
    simulator.run_steps(10)
    assert task.failed
    assert task.failure_reason == "No rrt_connect route for cube0:above_pick"
    assert task.stage_name == "cube0:above_pick"
    assert simulator.data.time == pytest.approx(simulator.dt)

    task.reset()
    assert not task.failed
    assert task.failure_reason is None
    assert task._stage_path is None
    assert task._planning_epoch == 0
    monkeypatch.setattr(task, "_plan_stage", original)
    simulator.run_steps(1)
    assert task._stage_path is not None
    assert task._planning_epoch == 1
    first_path = task._stage_path.copy()
    task.reset()
    simulator.run_steps(1)
    np.testing.assert_allclose(task._stage_path, first_path)
    assert task._planning_epoch == 1


def test_sampling_targets_follow_each_planned_edge_in_order(monkeypatch):
    simulator = create_cube_stack(2)
    task = CubeStackTask(simulator, 2, method="sampling")
    start = task.robot.joint_state.qpos[:7].copy()
    middle = start.copy()
    middle[0] += 0.05
    goal = task._plan[0][1][:7]
    monkeypatch.setattr(task, "_plan_stage", lambda *_: np.vstack((start, middle, goal)))

    simulator.run_steps(5)
    assert task._path_vertex == 1
    assert start[0] < task.robot.target.position[0] < middle[0]
    np.testing.assert_allclose(task.robot.target.position[1:7], start[1:7])

    for _ in range(500):
        if task._path_vertex == 2:
            break
        simulator.run_steps(1)
    assert task._path_vertex == 2
    assert abs(task.robot.joint_state.qpos[0] - middle[0]) < 0.016
    simulator.run_steps(1)
    target = task.robot.target.position[:7]
    edge = goal - middle
    fraction = np.dot(target - middle, edge) / np.dot(edge, edge)
    assert 0 < fraction <= 1
    np.testing.assert_allclose(target, middle + fraction * edge)


def test_sampling_rejects_cube_moved_from_precomputed_pick_pose():
    simulator = create_cube_stack(2)
    task = CubeStackTask(simulator, 2, method="sampling")
    cube_joint = simulator.model.joint("cube0/object_joint_0")
    simulator.data.qpos[int(cube_joint.qposadr[0])] += 0.02
    mujoco.mj_forward(simulator.model, simulator.data)
    simulator.run_steps(1)
    assert task.failed
    assert "moved more than 1 cm" in task.failure_reason
    assert task._planning_epoch == 0
