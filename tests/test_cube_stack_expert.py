"""Action-producing cube expert and its direct-simulation adapter."""

import numpy as np
import pytest
import yaml
from controller_config import create_test_controller

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets import ROBOT_ASSETS, RobotAsset
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import CubeStackExpert, CubeStackTask
from mujoco_lab.behaviors.cube_stack_recipe import load_recipe as load_cube_recipe
from mujoco_lab.control import ControlTarget
from mujoco_lab.control.trajectory import JointTrajectory
from mujoco_lab.learning.rollout import Expert, collect_episode
from mujoco_lab.planning import default_planning


def _forte_expert(method="heuristic"):
    simulator = Simulator(
        create_cube_stack(2),
        robots=[RobotSpec("forte", "forte", config=load_robot_config("forte"))],
    )
    task = CubeStackTask(simulator, 2)
    robot = simulator.robots["forte"]
    robot.change_controller(create_test_controller(robot, controller="pd", frame="grasp"))
    planning = default_planning() if method == "sampling" else None
    return (
        simulator,
        task,
        CubeStackExpert(
            task,
            recipe=load_cube_recipe(next(iter(task.simulator.robots.values())).robot_type),
            method=method,
            planning=planning,
        ),
    )


def test_close_mode_waits_for_contact_then_completes(monkeypatch):
    _, task, expert = _forte_expert()
    expert.stage = next(i for i, stage in enumerate(expert._plan) if stage.recipe.name == "close")
    monkeypatch.setattr(task, "has_grasp", lambda _index: False)
    expert.act()
    assert not expert.failed
    current = expert.robot.target.position.copy()
    target = expert._plan[expert.stage].gripper_target
    assert not expert._advance_stage(current, target, path_complete=True)
    monkeypatch.setattr(task, "has_grasp", lambda _index: True)
    assert expert._advance_stage(current, target, path_complete=True)


def test_closed_mode_motion_cannot_start_without_a_grasp(monkeypatch):
    _, task, expert = _forte_expert()
    expert.stage = next(i for i, stage in enumerate(expert._plan) if stage.recipe.name == "place")
    monkeypatch.setattr(task, "has_grasp", lambda _index: False)
    expert.act()
    assert expert.failed
    assert expert.failure_reason == "No two-finger physical grasp for cube0:place"
    assert expert.execution.trajectory is None


def test_closed_mode_motion_monitors_grasp_loss_and_recovery(monkeypatch):
    sim, task, expert = _forte_expert()
    expert.stage = next(i for i, stage in enumerate(expert._plan) if stage.recipe.name == "place")
    current = expert.robot.state.snapshot().qpos[:7]
    path = np.tile(np.r_[current, 0.0], (2, 1))
    path[-1, 6] += 0.25
    expert.execution.start(JointTrajectory(path, 1.0, 1.0), sim.data.time)
    grasped = False
    monkeypatch.setattr(task, "has_grasp", lambda _index: grasped)
    expert.act()
    sim.data.time = expert.recipe.lost_grasp_grace_s
    expert.act()
    assert not expert.failed
    grasped = True
    expert.act()
    grasped = False
    sim.data.time += sim.dt
    expert.act()
    assert not expert.failed
    sim.data.time += expert.recipe.lost_grasp_grace_s + sim.dt
    expert.act()
    assert expert.failed
    assert expert.failure_reason == "Lost two-finger grasp during cube0:place"


def test_constructor_and_reset_keep_caller_controller_and_targets(tmp_path, monkeypatch):
    simulator = Simulator(
        create_cube_stack(2),
        robots=[RobotSpec("forte", "forte", config=load_robot_config("forte"))],
    )
    task = CubeStackTask(simulator, 2)
    robot = simulator.robots["forte"]
    gains = {name: {"kp": 19, "kd": 3} for name in robot.get_arm_joint_mapping()[0]}
    (tmp_path / "robot.yaml").write_text(
        yaml.safe_dump({"pose": {"default": []}, "controller": {"name": "pd", "pd_gains": gains}})
    )
    monkeypatch.setitem(ROBOT_ASSETS, "forte", RobotAsset(tmp_path / "robot.xml"))
    controller = create_test_controller(robot, controller="pd", frame="grasp")
    robot.change_controller(controller)
    robot.target = ControlTarget(robot.target.position + 0.001)
    robot.gripper.set_target(0.001)
    target = robot.target.position.copy()
    ctrl = simulator.data.ctrl.copy()

    expert = CubeStackExpert(
        task,
        recipe=load_cube_recipe(next(iter(task.simulator.robots.values())).robot_type),
        method="heuristic",
    )
    expert.reset()

    assert robot.controller is controller
    np.testing.assert_array_equal(controller.kp, [19] * 7)
    np.testing.assert_array_equal(controller.kd, [3] * 7)
    np.testing.assert_array_equal(robot.target.position, target)
    np.testing.assert_array_equal(simulator.data.ctrl, ctrl)
    assert robot.gripper.get_target() == 0.001
    assert simulator.target_updater is None


@pytest.mark.parametrize("method", ["heuristic", "sampling"])
def test_act_returns_physical_action_without_applying_it(method, monkeypatch):
    simulator, _, expert = _forte_expert(method)
    robot = expert.robot
    if method == "sampling":
        start = robot.state.snapshot().qpos[:7]
        goal = expert._plan[0].waypoints[-1]
        monkeypatch.setattr(expert.motion, "plan_arm_path", lambda *_: np.vstack((start, goal)))
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
    assert expert.execution.start_time == before_time
    assert simulator.target_updater is None
    np.testing.assert_array_equal(simulator.data.qpos, before_qpos)
    np.testing.assert_array_equal(simulator.data.qvel, before_qvel)
    np.testing.assert_array_equal(simulator.data.ctrl, before_ctrl)
    np.testing.assert_array_equal(robot.target.position, before_target)
    assert robot.gripper.get_target() == before_gripper
    assert simulator.data.time == before_time
    assert not simulator._stop_requested


@pytest.mark.parametrize(
    "method,use_planner,reason",
    [
        ("sampling", False, "requires an explicit planner configuration"),
        ("heuristic", True, "does not use a planner configuration"),
        ("unknown", False, "Unsupported cube execution method"),
    ],
)
def test_cube_execution_rejects_invalid_method_and_planner_combinations(
    method, use_planner, reason
):
    _, task, _ = _forte_expert()
    planning = default_planning() if use_planner else None
    with pytest.raises(ValueError, match=reason):
        CubeStackExpert(
            task,
            recipe=load_cube_recipe(next(iter(task.simulator.robots.values())).robot_type),
            method=method,
            planning=planning,
        )


def test_collector_accepts_another_expert_and_stops_after_its_failure():
    class FailingExpert(Expert):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def reset(self, initial_obs=None, info=None):
            self.calls = 0
            self.failed = False
            self.failure_reason = None

        def act(self, obs=None, *, dt=0.0):
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
