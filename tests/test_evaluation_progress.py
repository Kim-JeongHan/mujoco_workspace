"""Physical progress milestones remain observational and episode-local."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import CubeStackTask
from mujoco_lab.behaviors.cube_stack import has_physical_grasp
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv
from mujoco_lab.learning.evaluation.progress import CubeProgressTracker


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


@pytest.mark.parametrize("robot_type", ["forte", "panda"])
def test_contacts_on_distinct_fingers_are_required_for_grasp(robot_type):
    simulator = Simulator(
        create_cube_stack(1),
        robots=[RobotSpec("robot", robot_type, config=load_robot_config(robot_type))],
    )
    actual = simulator.robots["robot"]
    gripper = actual.gripper
    assert gripper is not None
    cube = simulator.model.geom("cube0/object_0").id
    left = next(iter(gripper.finger_geom_ids[0]))
    right = next(iter(gripper.finger_geom_ids[1]))
    data = SimpleNamespace(contact=[SimpleNamespace(geom1=cube, geom2=left)])
    robot = SimpleNamespace(model=simulator.model, data=data, gripper=gripper)
    assert not has_physical_grasp(robot, 0)
    for geom in gripper.finger_geom_ids[0]:
        data.contact.append(SimpleNamespace(geom1=geom, geom2=cube))
    assert not has_physical_grasp(robot, 0)
    data.contact.append(SimpleNamespace(geom1=right, geom2=cube))
    assert has_physical_grasp(robot, 0)
