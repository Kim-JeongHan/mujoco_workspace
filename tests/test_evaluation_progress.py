"""Physical progress milestones remain observational and episode-local."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import CubeStackTask, CubeStatus
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv
from mujoco_lab.learning.evaluation.progress import CubeProgressTracker
from mujoco_lab.utils import Transform


class FakeTask(CubeStackTask):
    def __init__(self, cubes=2):
        simulator = Simulator(
            create_cube_stack(cubes),
            robots=[RobotSpec("forte", "forte", config=load_robot_config("forte"))],
        )
        super().__init__(simulator, cubes)
        self.heights = np.array([0.03, 0.07])[:cubes]
        self.position_errors = np.array([0.2, 0.3])[:cubes]
        self.stable = np.zeros(cubes, dtype=bool)
        self.samples = 0

    def status(self) -> CubeStatus:
        self.samples += 1
        self._track_heights(self.heights)
        held = [
            self._track_placement(index, stable, self.simulator.data.time)
            for index, stable in enumerate(self.stable)
        ]
        return CubeStatus(
            centers=np.column_stack((np.zeros(self.cubes), np.zeros(self.cubes), self.heights)),
            position_errors=self.position_errors.copy(),
            aligned=self.stable.copy(),
            at_height=self.stable.copy(),
            stable_placement=self.stable.copy(),
            supported=self.stable.copy(),
            is_final_success=bool(np.all(self.position_errors <= self.goal_tolerance_m)),
            is_released_stable=bool(np.all(held)),
            is_support_contacts=bool(np.all(self.stable)),
        )


def test_progress_requires_grasped_lift_and_held_released_placement(monkeypatch):
    task = FakeTask()
    gripped = {0: False, 1: False}
    monkeypatch.setattr(
        task,
        "has_grasp",
        lambda index, *, robot=None: gripped[index],
    )
    tracker = CubeProgressTracker(task)
    task.heights[0] += 0.05
    tracker.observe()  # A bounce cannot count as a carried lift.
    assert tracker.result()["best_stage"] == [0, 0]

    task.heights[0] -= 0.05
    gripped[0] = True
    tracker.observe()
    assert tracker.result()["best_stage"] == [1, 0]
    task.heights[0] += 0.05
    tracker.observe()
    assert tracker.result()["best_stage"] == [2, 0]
    assert tracker.result()["best_progress"] == pytest.approx(1 / 3)
    gripped[0] = False
    task.stable[0] = True
    task.simulator.data.time = 1.0
    task.status()
    tracker.observe()
    task.simulator.data.time = 1.49
    task.status()
    tracker.observe()
    assert tracker.result()["placed"] == [False, False]
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
    task.position_errors[0] = 0.6
    task.status()
    tracker.observe()  # History remains, while final position error records the drop.
    result = tracker.result()
    assert result["best_stage"] == [3, 0]
    assert result["placed"] == [True, False]
    assert result["best_progress"] == pytest.approx(0.5)
    assert result["final_position_error"] == pytest.approx(0.45)
    assert task.samples == 16  # Initial sample, nine observations, and six task updates.

    fresh = CubeProgressTracker(task)
    assert fresh.result()["best_stage"] == [0, 0]


def test_progress_observes_grasp_by_secondary_robot(monkeypatch):
    config = load_robot_config("forte")
    simulator = Simulator(
        create_cube_stack(1),
        robots=[
            RobotSpec("first", "forte", Transform(translation=[-0.8, 0, 0]), config=config),
            RobotSpec("second", "forte", Transform(translation=[0.8, 0, 0]), config=config),
        ],
    )
    task = CubeStackTask(simulator, 1)
    monkeypatch.setattr(
        task, "has_grasp", lambda index, *, robot=None: robot is simulator.robots["second"]
    )
    tracker = CubeProgressTracker(task)
    tracker.observe()
    assert tracker.result()["grasped"] == [True]


def test_progress_retains_placement_completed_between_observations():
    task = FakeTask()
    tracker = CubeProgressTracker(task)
    task.stable[0] = True
    task.simulator.data.time = 1.0
    task.status()
    task.simulator.data.time = 1.49
    tracker.observe()
    assert tracker.result()["placed"] == [False, False]

    task.simulator.data.time = 1.5
    task.status()
    task.stable[0] = False
    task.simulator.data.time = 1.51
    task.status()
    before_hold = task._stable_since.copy()
    before_time = float(task.simulator.data.time)
    tracker.observe()
    assert tracker.result()["placed"] == [True, False]
    assert task._stable_since == before_hold
    assert task.simulator.data.time == before_time


def test_status_reports_placement_conditions_and_updates_history_without_stepping_physics():
    simulator = Simulator(
        create_cube_stack(2, environment="table_shelf"),
        robots=[RobotSpec("forte", "forte", config=load_robot_config("forte"))],
        dt=0.002,
    )
    task = CubeStackTask(simulator, 2)
    env = CubeStackEnv(task, xy_range=0.02, min_gap=0.01, max_steps=10)
    env.reset(seed=10000)
    before_lift = task.max_height.copy()
    before_placed = task.completed_placements()
    before_time = simulator.data.time
    initial = task.status()
    assert isinstance(initial, CubeStatus)
    np.testing.assert_allclose(task.max_height, before_lift)
    np.testing.assert_array_equal(task.completed_placements(), before_placed)
    assert not initial.stable_placement.any()

    adr = int(simulator.model.joint("cube0/object_joint_0").qposadr[0])
    dof = int(simulator.model.joint("cube0/object_joint_0").dofadr[0])
    simulator.data.qpos[adr : adr + 3] = task.goals[0]
    mujoco.mj_forward(simulator.model, simulator.data)
    placed = task.status()
    assert placed.supported[0] and placed.stable_placement[0]
    assert task._stable_since[0] == before_time
    assert not placed.is_released_stable
    task.status()
    assert task._stable_since[0] == before_time
    simulator.data.qvel[dof] = 0.1
    assert not task.status().stable_placement[0]
    assert task._stable_since[0] is None
    simulator.data.qvel[dof] = 0
    simulator.data.qpos[adr + 2] += 0.011
    mujoco.mj_forward(simulator.model, simulator.data)
    floating = task.status()
    assert floating.aligned[0] and floating.at_height[0]
    assert not floating.supported[0] and not floating.stable_placement[0]
    np.testing.assert_allclose(task.max_height, np.maximum(before_lift, floating.centers[:, 2]))
    np.testing.assert_array_equal(task.completed_placements(), before_placed)
    assert simulator.data.time == before_time


@pytest.mark.parametrize("robot_type", ["forte", "panda"])
def test_contacts_on_distinct_fingers_are_required_for_grasp(robot_type, monkeypatch):
    simulator = Simulator(
        create_cube_stack(1),
        robots=[RobotSpec("robot", robot_type, config=load_robot_config(robot_type))],
    )
    actual = simulator.robots["robot"]
    task = CubeStackTask(simulator, 1)
    gripper = actual.gripper
    assert gripper is not None
    cube = simulator.model.geom("cube0/object_0").id
    left = next(iter(gripper.finger_geom_ids[0]))
    right = next(iter(gripper.finger_geom_ids[1]))
    data = SimpleNamespace(contact=[SimpleNamespace(geom1=cube, geom2=left)])
    with monkeypatch.context() as patch:
        patch.setattr(actual, "data", data)
        assert not task.has_grasp(0, robot=actual)
        for geom in gripper.finger_geom_ids[0]:
            data.contact.append(SimpleNamespace(geom1=geom, geom2=cube))
        assert not task.has_grasp(0, robot=actual)
        data.contact.append(SimpleNamespace(geom1=right, geom2=cube))
        assert task.has_grasp(0, robot=actual)
    assert not task.has_grasp(0)
