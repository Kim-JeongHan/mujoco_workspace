"""Sampling plans are executed through the physical cube stacking task."""

import mujoco
import numpy as np
import pytest
from controller_config import create_test_controller

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import CubeStackExpert, CubeStackTask
from mujoco_lab.behaviors.cube_stack_recipe import load_recipe as load_cube_recipe
from mujoco_lab.planning import (
    PRMConfig,
    RRTConfig,
    RRTConnectConfig,
    default_planning,
)
from mujoco_lab.planning.sampling.sampler import GoalBiasedSampler


def make_sampling_expert(task, planning):
    robot = next(iter(task.simulator.robots.values()))
    robot.change_controller(create_test_controller(robot, controller="position", frame="grasp"))
    return CubeStackExpert(
        task,
        recipe=load_cube_recipe(next(iter(task.simulator.robots.values())).robot_type),
        method="sampling",
        planning=planning,
    )


@pytest.mark.parametrize(
    "planning",
    [
        RRTConnectConfig(max_iterations=500, step_size=0.2, goal_tolerance=0.04, seed=7),
        RRTConfig(max_iterations=500, step_size=0.2, goal_tolerance=0.04, goal_bias=0.8, seed=7),
        PRMConfig(
            sample_number=100,
            max_retries=2,
            radius=2.0,
            sampler=GoalBiasedSampler,
            goal_bias=0.3,
            seed=7,
        ),
    ],
)
def test_sampling_planners_complete_released_two_cube_stack(planning):
    simulator = Simulator(
        create_cube_stack(2),
        robots=[RobotSpec("panda", "panda", config=load_robot_config("panda"))],
    )
    task = CubeStackTask(simulator, 2)
    expert = make_sampling_expert(task, planning)
    simulator.target_updater = expert.update
    simulator.run_steps(22000)
    status = task.status()
    assert not expert.failed, expert.failure_reason
    assert expert.get_stage_name() == "settle"
    assert status.released_stable_stack
    assert status.support_contacts
    assert np.all(task.max_lift > task.starts[:, 2] + 0.04)
    assert not simulator.data.warning.number.any()


def test_sampling_no_route_stops_cleanly_and_reset_replans(monkeypatch):
    simulator = Simulator(
        create_cube_stack(2),
        robots=[RobotSpec("panda", "panda", config=load_robot_config("panda"))],
    )
    task = CubeStackTask(simulator, 2)
    expert = make_sampling_expert(task, default_planning())
    simulator.target_updater = expert.update
    original = expert.motion.plan_arm_path
    monkeypatch.setattr(expert.motion, "plan_arm_path", lambda *_: None)
    simulator.run_steps(10)
    assert expert.failed
    assert expert.failure_reason == "No rrt_connect route for cube0:pick"
    assert expert.get_stage_name() == "cube0:pick"
    assert simulator.data.time == pytest.approx(simulator.dt)

    task.reset()
    expert.reset()
    assert not expert.failed
    assert expert.failure_reason is None
    assert expert.execution.trajectory is None
    monkeypatch.setattr(expert.motion, "plan_arm_path", original)
    simulator.run_steps(1)
    assert expert.execution.trajectory is not None
    first_path = expert.execution.trajectory.path.copy()
    task.reset()
    expert.reset()
    simulator.run_steps(1)
    np.testing.assert_allclose(expert.execution.trajectory.path, first_path)


def test_sampling_rejects_cube_moved_from_precomputed_pick_pose():
    simulator = Simulator(
        create_cube_stack(2),
        robots=[RobotSpec("panda", "panda", config=load_robot_config("panda"))],
    )
    task = CubeStackTask(simulator, 2)
    expert = make_sampling_expert(task, default_planning())
    simulator.target_updater = expert.update
    cube_joint = simulator.model.joint("cube0/object_joint_0")
    simulator.data.qpos[int(cube_joint.qposadr[0])] += 0.02
    mujoco.mj_forward(simulator.model, simulator.data)
    simulator.run_steps(1)
    assert expert.failed
    assert "moved more than 1 cm" in expert.failure_reason


def test_sampling_missing_physical_grasp_returns_expected_failure(monkeypatch):
    simulator = Simulator(
        create_cube_stack(2),
        robots=[RobotSpec("panda", "panda", config=load_robot_config("panda"))],
    )
    task = CubeStackTask(simulator, 2)
    expert = make_sampling_expert(task, default_planning())
    stage = next(stage for stage in expert._plan if stage.recipe.name == "place")
    monkeypatch.setattr("mujoco_lab.behaviors.cube_stack.has_physical_grasp", lambda *_: False)

    trajectory, reason = expert.make_trajectory(stage)

    assert trajectory is None
    assert reason == f"No two-finger physical grasp for {stage.name}"


def test_sampling_unexpected_planner_error_propagates(monkeypatch):
    simulator = Simulator(
        create_cube_stack(2),
        robots=[RobotSpec("panda", "panda", config=load_robot_config("panda"))],
    )
    task = CubeStackTask(simulator, 2)
    expert = make_sampling_expert(task, default_planning())

    def fail_planning(*_):
        raise RuntimeError("planner bug")

    monkeypatch.setattr(expert.motion, "plan_arm_path", fail_planning)

    with pytest.raises(RuntimeError, match="planner bug"):
        expert.act()
