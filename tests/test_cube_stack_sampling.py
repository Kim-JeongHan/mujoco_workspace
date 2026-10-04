"""Sampling plans are executed through the physical cube stacking task."""

import mujoco
import numpy as np
import pytest
from controller_config import create_test_controller

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import CubeStackExpert, CubeStackTask
from mujoco_lab.behaviors.cube_stack_recipe import load_recipe as load_cube_recipe
from mujoco_lab.control.trajectory import JointTrajectory
from mujoco_lab.planning import (
    PRMConfig,
    RRTConfig,
    RRTConnectConfig,
    default_planning,
    planner_from_config,
)
from mujoco_lab.planning.sampling.sampler import GoalBiasedSampler
from mujoco_lab.state import IKError


def make_sampling_expert(task, planning):
    robot = next(iter(task.simulator.robots.values()))
    robot.change_controller(create_test_controller(robot, controller="position", frame="grasp"))
    return CubeStackExpert(
        task,
        recipe=load_cube_recipe(next(iter(task.simulator.robots.values())).robot_type),
        method="sampling",
        planner=planner_from_config(planning),
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
    assert expert.motion.planning_epoch == 8
    assert expert.get_stage_name() == "settle"
    assert status.released_stable_stack
    assert status.support_contacts
    assert np.all(task.max_lift > task.starts[:, 2] + 0.04)
    assert not simulator.data.warning.number.any()


def test_sampling_retracts_from_measured_release_state(monkeypatch):
    simulator = Simulator(
        create_cube_stack(2),
        robots=[RobotSpec("panda", "panda", config=load_robot_config("panda"))],
    )
    task = CubeStackTask(simulator, 2)
    expert = make_sampling_expert(task, default_planning())
    make_trajectory = expert.motion.make_trajectory
    retract_starts = []

    def record_retract_start(request):
        current = expert.robot.state.snapshot().qpos[:7].copy()
        trajectory, reason = make_trajectory(request)
        if request.name.endswith(":retract"):
            assert reason is None
            np.testing.assert_allclose(trajectory.path[0, :7], current)
            assert (
                abs(expert.robot.gripper.get_position() - request.gripper_target)
                < expert.recipe.gripper_tolerance
            )
            assert request.checker.is_collision_free(current)
            retract_starts.append(current)
        return trajectory, reason

    monkeypatch.setattr(expert.motion, "make_trajectory", record_retract_start)
    simulator.target_updater = expert.update
    simulator.run_steps(22000)

    assert not expert.failed, expert.failure_reason
    assert expert.get_stage_name() == "settle"
    assert task.status().released_stable_stack
    assert len(retract_starts) == 2
    assert np.all(task.max_lift > task.starts[:, 2] + 0.04)
    assert not simulator.data.warning.number.any()


def test_sampling_uses_pick_and_place_task_targets():
    simulator = Simulator(
        create_cube_stack(2),
        robots=[RobotSpec("panda", "panda", config=load_robot_config("panda"))],
    )
    expert = make_sampling_expert(CubeStackTask(simulator, 2), default_planning())
    assert [stage.recipe.name for stage in expert._plan] == [
        "pick",
        "close",
        "place",
        "release",
        "retract",
    ] * 2


def test_explicit_heuristic_method_uses_no_sampling_planner():
    simulator = Simulator(
        create_cube_stack(2),
        robots=[RobotSpec("panda", "panda", config=load_robot_config("panda"))],
    )
    task = CubeStackTask(simulator, 2)

    robot = simulator.robots["panda"]
    robot.change_controller(create_test_controller(robot, controller="position", frame="grasp"))
    expert = CubeStackExpert(
        task,
        recipe=load_cube_recipe(next(iter(task.simulator.robots.values())).robot_type),
        method="heuristic",
    )
    assert expert.method == "heuristic"
    assert expert.planner is None


def test_sampling_accepts_injected_planner_and_advances_stage_seed():
    simulator = Simulator(
        create_cube_stack(2),
        robots=[RobotSpec("panda", "panda", config=load_robot_config("panda"))],
    )
    task = CubeStackTask(simulator, 2)

    class DirectPlanner:
        name = "direct_test"
        seed = 13

        def __init__(self):
            self.seeds = []

        def plan(self, start, goal, bounds, collision_checker, *, seed):
            self.seeds.append(seed)
            return np.vstack((start, goal))

    planner = DirectPlanner()
    robot = simulator.robots["panda"]
    robot.change_controller(create_test_controller(robot, controller="position", frame="grasp"))
    expert = CubeStackExpert(
        task,
        recipe=load_cube_recipe(next(iter(task.simulator.robots.values())).robot_type),
        method="sampling",
        planner=planner,
    )
    assert expert.planner is planner
    assert expert.planner.name == "direct_test"
    stage = expert._plan[0]
    checker = expert.collision_checker(stage.cube_index, stage.recipe.name)
    assert (
        expert.motion.plan_arm_path(robot.state.snapshot().qpos[:7], stage.waypoints[-1], checker)
        is not None
    )
    assert (
        expert.motion.plan_arm_path(robot.state.snapshot().qpos[:7], stage.waypoints[-1], checker)
        is not None
    )
    assert planner.seeds == [13, 14]
    expert.reset()
    assert expert.motion.planning_epoch == 0
    stage = expert._plan[0]
    assert (
        expert.motion.plan_arm_path(
            robot.state.snapshot().qpos[:7],
            stage.waypoints[-1],
            expert.collision_checker(stage.cube_index, stage.recipe.name),
        )
        is not None
    )
    assert planner.seeds[-1] == 13


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
    assert expert.motion.planning_epoch == 0
    monkeypatch.setattr(expert.motion, "plan_arm_path", original)
    simulator.run_steps(1)
    assert expert.execution.trajectory is not None
    assert expert.motion.planning_epoch == 1
    first_path = expert.execution.trajectory.path.copy()
    task.reset()
    expert.reset()
    simulator.run_steps(1)
    np.testing.assert_allclose(expert.execution.trajectory.path, first_path)
    assert expert.motion.planning_epoch == 1


def test_sampling_targets_pass_checked_waypoints_in_order(monkeypatch):
    simulator = Simulator(
        create_cube_stack(2),
        robots=[RobotSpec("panda", "panda", config=load_robot_config("panda"))],
    )
    task = CubeStackTask(simulator, 2)
    expert = make_sampling_expert(task, default_planning())
    simulator.target_updater = expert.update
    start = expert.robot.joint_state.qpos[:7].copy()
    middle = start.copy()
    middle[0] += 0.05
    goal = expert._plan[0].waypoints[-1]
    monkeypatch.setattr(expert.motion, "plan_arm_path", lambda *_: np.vstack((start, middle, goal)))
    monkeypatch.setattr(expert.motion, "shortcut_path", lambda path, checker: path)

    simulator.run_steps(5)
    assert expert.execution.vertex == 1
    np.testing.assert_allclose(expert.execution.trajectory.path[1, :7], middle)
    waypoint_time = expert.execution.trajectory.waypoint_times[1]
    np.testing.assert_allclose(
        expert.execution.trajectory.sample(waypoint_time).position[:7], middle
    )

    for _ in range(500):
        if expert.execution.elapsed >= waypoint_time:
            break
        simulator.run_steps(1)
    assert expert.execution.vertex == 2
    simulator.run_steps(1)
    assert expert.execution.vertex >= 2
    assert not np.allclose(expert.robot.target.position[:7], middle)


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
    assert expert.motion.planning_epoch == 0


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


def test_sampling_unreachable_departure_reports_expected_failure(monkeypatch):
    simulator = Simulator(
        create_cube_stack(2),
        robots=[RobotSpec("panda", "panda", config=load_robot_config("panda"))],
    )
    expert = make_sampling_expert(CubeStackTask(simulator, 2), default_planning())
    stage = next(stage for stage in expert._plan if stage.recipe.name == "place")
    monkeypatch.setattr("mujoco_lab.behaviors.cube_stack.has_physical_grasp", lambda *_: True)

    def unreachable(*_):
        raise IKError("unreachable")

    monkeypatch.setattr(expert.motion, "solve_ik", unreachable)

    with pytest.raises(IKError, match="unreachable"):
        expert.make_trajectory(stage)
    expert.stage = next(i for i, candidate in enumerate(expert._plan) if candidate is stage)
    expert.act()
    assert expert.failed
    assert expert.failure_reason == "unreachable"


def test_sampling_reports_insufficient_lift_at_completed_place():
    simulator = Simulator(
        create_cube_stack(2),
        robots=[RobotSpec("panda", "panda", config=load_robot_config("panda"))],
    )
    task = CubeStackTask(simulator, 2)
    expert = make_sampling_expert(task, default_planning())
    expert.stage = next(i for i, stage in enumerate(expert._plan) if stage.recipe.name == "place")
    current = expert._hold_action()
    expert.execution.trajectory = JointTrajectory(current[None, :], 1.0, 1.0)
    expert.execution.vertex = 1

    expert.act()

    assert expert.failed
    assert expert.failure_reason == "Insufficient physical lift during cube0:place"


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
