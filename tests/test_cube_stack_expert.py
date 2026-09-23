"""Action-producing cube expert and its direct-simulation adapter."""

from itertools import pairwise

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.control import ControlTarget, create_controller
from mujoco_lab.learning.collect import create_expert
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv
from mujoco_lab.learning.rollout import Expert, collect_episode
from mujoco_lab.planning import planner_from_config
from mujoco_lab.tasks import (
    CubeStackExpert,
    CubeStackMotionGenerator,
    CubeStackTask,
    default_planning,
)


def _forte_expert(method="heuristic"):
    simulator = Simulator(create_cube_stack(2), robots=[RobotSpec("forte", "forte")])
    task = CubeStackTask(simulator, 2)
    robot = simulator.robots["forte"]
    robot.change_controller(create_controller("pd", robot, frame="grasp"))
    planner = planner_from_config(default_planning()) if method == "sampling" else None
    return simulator, task, CubeStackExpert(task, CubeStackMotionGenerator(task, planner=planner))


def test_constructor_and_reset_keep_caller_controller_and_targets():
    simulator = Simulator(create_cube_stack(2), robots=[RobotSpec("forte", "forte")])
    task = CubeStackTask(simulator, 2)
    robot = simulator.robots["forte"]
    controller = create_controller("pd", robot, kp=[19] * 7, kd=[3] * 7, frame="grasp")
    robot.change_controller(controller)
    robot.target = ControlTarget(robot.target.position + 0.001)
    robot.gripper.set_target(-0.001)
    target = robot.target.position.copy()
    ctrl = simulator.data.ctrl.copy()

    expert = CubeStackExpert(task, CubeStackMotionGenerator(task))
    expert.reset()

    assert robot.controller is controller
    np.testing.assert_array_equal(controller.kp, [19] * 7)
    np.testing.assert_array_equal(controller.kd, [3] * 7)
    np.testing.assert_array_equal(robot.target.position, target)
    np.testing.assert_array_equal(simulator.data.ctrl, ctrl)
    assert robot.gripper.get_target() == -0.001
    assert simulator.target_updater is None


def test_constructor_requires_joint_target_controller():
    simulator = Simulator(create_cube_stack(2), robots=[RobotSpec("forte", "forte")])
    task = CubeStackTask(simulator, 2)
    robot = simulator.robots["forte"]
    with pytest.raises(ValueError, match="configured seven-joint target controller"):
        CubeStackExpert(task, CubeStackMotionGenerator(task))
    robot.change_controller(create_controller("osc", robot, frame="grasp"))
    with pytest.raises(ValueError, match="configured seven-joint target controller"):
        CubeStackExpert(task, CubeStackMotionGenerator(task))


@pytest.mark.parametrize("method", ["heuristic", "sampling"])
def test_act_returns_physical_action_without_applying_it(method, monkeypatch):
    simulator, _, expert = _forte_expert(method)
    robot = expert.robot
    if method == "sampling":
        start = robot.state.snapshot().qpos[:7]
        goal = expert._plan[0].command[:7]
        monkeypatch.setattr(expert.generator, "_plan_stage", lambda *_: np.vstack((start, goal)))
    assert isinstance(expert, Expert)
    assert simulator.target_updater is None
    before_qpos = simulator.data.qpos.copy()
    before_qvel = simulator.data.qvel.copy()
    before_ctrl = simulator.data.ctrl.copy()
    before_target = robot.target.position.copy()
    before_gripper = robot.gripper.get_target()
    before_time = simulator.data.time

    action = expert.act(np.zeros(24, dtype=np.float32))

    assert action.shape == (8,)
    assert np.isfinite(action).all()
    assert expert._stage_start_time == before_time
    assert simulator.target_updater is None
    np.testing.assert_array_equal(simulator.data.qpos, before_qpos)
    np.testing.assert_array_equal(simulator.data.qvel, before_qvel)
    np.testing.assert_array_equal(simulator.data.ctrl, before_ctrl)
    np.testing.assert_array_equal(robot.target.position, before_target)
    assert robot.gripper.get_target() == before_gripper
    assert simulator.data.time == before_time
    assert not simulator._stop_requested


def test_reset_replans_current_cube_pose_without_resetting_physics_or_commands():
    simulator, task, expert = _forte_expert()
    simulator.run_steps(4)
    expert.robot.target = ControlTarget(expert.robot.state.snapshot().qpos[:7] + 0.001)
    expert.robot.gripper.set_target(-0.001)
    old_command = expert._plan[0].command.copy()
    cube_joint = simulator.model.joint("cube0/object_joint_0")
    simulator.data.qpos[int(cube_joint.qposadr[0])] += 0.01
    mujoco.mj_forward(simulator.model, simulator.data)
    expert.stage = 3
    expert.failed = True
    expert.failure_reason = "old failure"
    physical_qpos = simulator.data.qpos.copy()
    physical_ctrl = simulator.data.ctrl.copy()
    command_target = expert.robot.target.position.copy()
    grip_target = expert.robot.gripper.get_target()
    time = simulator.data.time
    assert time > 0

    expert.reset(np.zeros(24), {"seed": 7})

    assert expert.stage == 0
    assert expert._trajectory is None
    assert expert._stage_start_time == time
    assert not expert.failed
    assert expert.failure_reason is None
    np.testing.assert_allclose(expert.starts[0], simulator.data.body("cube0/object_0").xpos)
    assert not np.allclose(expert._plan[0].command[:7], old_command[:7])
    np.testing.assert_array_equal(simulator.data.qpos, physical_qpos)
    np.testing.assert_array_equal(simulator.data.ctrl, physical_ctrl)
    np.testing.assert_array_equal(expert.robot.target.position, command_target)
    assert expert.robot.gripper.get_target() == grip_target
    assert simulator.data.time == time
    assert task.starts[0, 0] != expert.starts[0, 0]


@pytest.mark.parametrize("method", ["heuristic", "sampling"])
def test_direct_callback_applies_same_first_action_as_manual_consumer(method, monkeypatch):
    direct, _, direct_expert = _forte_expert(method)
    manual, _, manual_expert = _forte_expert(method)
    if method == "sampling":
        for expert in (direct_expert, manual_expert):
            start = expert.robot.state.snapshot().qpos[:7]
            goal = expert._plan[0].command[:7]
            monkeypatch.setattr(
                expert.generator, "_plan_stage", lambda *_, s=start, g=goal: np.vstack((s, g))
            )
    direct.target_updater = direct_expert.update
    action = manual_expert.act()
    manual_expert.robot.target = ControlTarget(action[:7])
    manual_expert.robot.gripper.set_target(float(action[7]))

    direct.run_steps(1)
    manual.run_steps(1)

    np.testing.assert_allclose(direct.data.qpos, manual.data.qpos)
    np.testing.assert_allclose(direct.data.qvel, manual.data.qvel)
    np.testing.assert_allclose(direct.data.ctrl, manual.data.ctrl)
    np.testing.assert_allclose(direct_expert.robot.target.position, action[:7])
    assert direct_expert.robot.gripper.get_target() == pytest.approx(action[7])
    assert direct_expert._stage_start_time == manual_expert._stage_start_time == 0


def test_factory_accepts_both_methods_without_attaching_callback():
    simulator = Simulator(create_cube_stack(2), robots=[RobotSpec("forte", "forte")])
    task = CubeStackTask(simulator, 2)
    robot = simulator.robots["forte"]
    controller = create_controller("pd", robot, frame="grasp")
    robot.change_controller(controller)
    for method in ("heuristic", "sampling"):
        expert = create_expert(task, method=method, planning=default_planning())
        assert isinstance(expert, CubeStackExpert)
        if method == "sampling":
            assert expert.generator.planner.name == "rrt_connect"
        else:
            assert not hasattr(expert.generator, "planner")
        assert robot.controller is controller
        assert simulator.target_updater is None


def test_collector_accepts_another_expert_and_stops_after_its_failure():
    class FailingExpert(Expert):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def reset(self, initial_obs=None, info=None):
            self.calls = 0
            self.failed = False
            self.failure_reason = None

        def act(self, obs=None):
            self.calls += 1
            if self.calls == 2:
                self.failed = True
                self.failure_reason = "no route"
            return np.zeros(8, dtype=np.float32)

    class SmallEnv:
        def __init__(self):
            self.steps = 0
            self.obs = np.zeros(54, dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            self.steps = 0
            self.obs.fill(0)
            return self.obs, {}

        def step(self, action):
            self.steps += 1
            self.obs.fill(self.steps)
            return self.obs, 0.0, False, False, {}

    env = SmallEnv()
    episode = collect_episode(env, FailingExpert(), max_steps=10)
    assert env.steps == len(episode) == 2
    assert episode.states.shape == (3, 54)
    np.testing.assert_array_equal(
        episode.states, np.repeat(np.arange(3, dtype=np.float32)[:, None], 54, axis=1)
    )
    assert episode.actions.shape == (2, 8)
    np.testing.assert_array_equal(episode.terminated, [False, False])
    np.testing.assert_array_equal(episode.truncated, [False, True])
    assert episode.metadata["success"] is None
    assert episode.metadata["expert_failed"] is True
    assert episode.metadata["termination_reason"] == "no route"


def test_heuristic_first_approach_avoids_the_home_orientation_detour():
    simulator = Simulator(create_cube_stack(2), robots=[RobotSpec("forte", "forte")])
    task = CubeStackTask(simulator, 2)
    env = CubeStackEnv(task)
    robot = simulator.robots["forte"]
    robot.change_controller(create_controller("pd", robot, frame="grasp"))
    expert = CubeStackExpert(task, CubeStackMotionGenerator(task))
    obs, info = env.reset(seed=42)
    expert.reset(obs, info)
    stage = expert._plan[0]
    site = expert.robot.state.site_id("grasp")
    goal = expert.starts[stage.cube_index] + stage.recipe.offset_xyz_m
    above_pick = expert.starts[stage.cube_index] + expert.recipe.stages[0].offset_xyz_m
    rotation = Rotation.from_euler("xyz", expert.recipe.euler_xyz_degrees, degrees=True)
    positions, orientation_errors, rotations = [], [], []

    for _ in range(1000):
        positions.append(simulator.data.site_xpos[site].copy())
        current_rotation = Rotation.from_matrix(simulator.data.site_xmat[site].reshape(3, 3))
        orientation_errors.append((rotation * current_rotation.inv()).magnitude())
        rotations.append(current_rotation)
        if expert.stage != 0:
            break
        previous = expert.robot.target.position.copy()
        action = expert.act(obs)
        assert np.max(np.abs(action[:7] - previous)) <= (
            expert.recipe.arm_max_velocity * simulator.dt + 1e-12
        )
        obs, _, terminated, truncated, _ = env.step(action)
        assert not terminated and not truncated

    assert expert.stage == 1
    positions = np.asarray(positions)
    direction = goal - positions[0]
    fraction = (positions - positions[0]) @ direction / (direction @ direction)
    deviations = positions - (positions[0] + fraction[:, None] * direction)
    # Guard the physical path, not just the target interpolation arithmetic.
    assert np.linalg.norm(np.diff(positions, axis=0), axis=1).sum() < 0.45
    assert np.linalg.norm(deviations, axis=1).max() < 0.1
    assert np.linalg.norm(positions - above_pick, axis=1).min() < 0.02
    assert positions[:, 2].min() > goal[2] - 0.01
    assert max(orientation_errors) < np.deg2rad(10)
    angular_travel = sum(
        (second * first.inv()).magnitude() for first, second in pairwise(rotations)
    )
    assert angular_travel < np.deg2rad(20)
    assert np.max(np.abs(stage.command[5:7])) < np.deg2rad(45)
    assert not simulator.data.warning.number.any()
