"""Sampling plans are executed through the physical cube stacking task."""

import mujoco
import numpy as np
import pytest
from controller_config import create_test_controller
from scipy.spatial.transform import Rotation

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import CubeStackExpert, CubeStackTask
from mujoco_lab.behaviors.cube_stack_recipe import load_recipe as load_cube_recipe
from mujoco_lab.control import ControlTarget, create_controller
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv
from mujoco_lab.learning.rollout.collector import collect_episode
from mujoco_lab.planning import (
    PRMConfig,
    RRTConfig,
    RRTConnectConfig,
    default_planning,
)
from mujoco_lab.planning.motion import grasp_pose
from mujoco_lab.planning.sampling.sampler import GoalBiasedSampler


@pytest.fixture
def approach_expert():
    config = load_robot_config("forte")
    simulator = Simulator(create_cube_stack(1), robots=[RobotSpec("forte", "forte", config=config)])
    robot = simulator.robots["forte"]
    robot.change_controller(create_controller(robot, config.controller))
    return CubeStackExpert(
        CubeStackTask(simulator, 1),
        recipe=load_cube_recipe("forte"),
        method="sampling",
        planning=default_planning(),
    )


def test_sampling_reaches_above_pick_before_a_vertical_cartesian_descent(
    approach_expert, monkeypatch
):
    expert = approach_expert
    assert [stage.recipe.name for stage in expert._plan] == [
        "above_pick",
        "pick",
        "close",
        "place",
        "release",
        "retract",
    ]
    above, pick = expert._plan[:2]
    assert expert.sampling_request(above, expert.starts[0]).mode == "sampling"
    robot = expert.robot
    model, data = expert.simulator.model, expert.simulator.data
    data.qpos[robot.state.qpos_indices[:7]] = above.waypoints[-1]
    mujoco.mj_forward(model, data)
    robot.update_state()
    robot.target = ControlTarget(above.waypoints[-1].copy())
    measured = grasp_pose(robot)
    before = data.qpos.copy()

    def unexpected_sampling(*args, **kwargs):
        pytest.fail("The final pick descent must not use the sampling planner")

    monkeypatch.setattr(expert.motion, "plan_arm_path", unexpected_sampling)
    request = expert.sampling_request(pick, expert.starts[0])
    assert request.mode == "cartesian"
    trajectory, reason = expert.motion.make_trajectory(request)
    assert reason is None
    scratch = mujoco.MjData(model)
    mujoco.mj_copyData(scratch, model, data)
    positions = np.empty((101, 3))
    site = robot.state.site_id("grasp")
    for index, time in enumerate(np.linspace(0, trajectory.duration, len(positions))):
        scratch.qpos[robot.state.qpos_indices[:7]] = trajectory.sample(time).position[:7]
        mujoco.mj_forward(model, scratch)
        positions[index] = scratch.site_xpos[site]
        rotation = Rotation.from_matrix(scratch.site_xmat[site].reshape(3, 3))
        # Both operands are Rotations; SciPy also annotates multiplication as NotImplemented.
        error = measured.as_rotation() * rotation.inv()
        assert error.magnitude() < 0.001  # ty: ignore[unresolved-attribute]
    np.testing.assert_allclose(
        positions[:, :2], np.broadcast_to(measured.as_translation()[:2], (101, 2)), atol=1e-4
    )
    assert np.all(np.diff(positions[:, 2]) <= 1e-6)
    assert positions[-1, 2] == pytest.approx(
        expert.starts[0, 2] + pick.recipe.offset_xyz_m[2], abs=1e-5
    )
    np.testing.assert_array_equal(data.qpos, before)


@pytest.mark.parametrize("phase", ["close", "release"])
def test_sampling_gripper_only_stages_hold_the_current_arm_command(approach_expert, phase):
    expert = approach_expert
    selected = expert.robot.target
    assert selected is not None
    stage = next(stage for stage in expert._plan if stage.recipe.name == phase)
    assert not np.allclose(selected.position, stage.waypoints[-1])
    request = expert.sampling_request(stage, expert.starts[0])
    trajectory, reason = expert.motion.make_trajectory(request)
    assert reason is None
    np.testing.assert_array_equal(trajectory.path[:, :7], np.tile(selected.position, (2, 1)))


@pytest.mark.parametrize("seed", [44, 53, 91, 136])
def test_randomized_sampling_collection_handles_previous_pick_failures(approach_expert, seed):
    expert = approach_expert
    env = CubeStackEnv(
        expert.task,
        xy_range=0.02,
        cube_yaw_range_degrees=45,
        physics_steps_per_action=5,
        max_steps=18000,
    )
    episode = collect_episode(env, expert, seed=seed, max_steps=18000)
    assert episode.metadata["success"], episode.metadata["termination_reason"]
    assert not expert.failed, expert.failure_reason


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
    assert expert.failure_reason == "No rrt_connect route for cube0:above_pick"
    assert expert.get_stage_name() == "cube0:above_pick"
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
