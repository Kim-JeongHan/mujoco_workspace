"""Stage ratios scale configured limits without changing joint ordering."""

from copy import deepcopy
from math import ceil

import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.control import create_controller
from mujoco_lab.planning import RRTConnectConfig
from mujoco_lab.planning.collision.collision_checker import EmptyCollisionChecker
from mujoco_lab.planning.motion import MotionPlanner, MotionRequest
from mujoco_lab.trajectory_generation import ControlTarget, JointTrajectory


@pytest.fixture
def motion():
    simulator = Simulator(
        create_cube_stack(1),
        robots=[RobotSpec("panda", "panda", config=load_robot_config("panda"))],
    )
    robot = simulator.robots["panda"]
    robot.change_controller(create_controller(robot, robot.config.controller))
    return MotionPlanner(robot)


def test_arm_and_gripper_ratios_apply_independently_without_mutating_limits(motion):
    configured = motion.motion_constraints()
    before = deepcopy(motion.robot.config)
    start = motion.robot.state.snapshot().qpos[:7]
    trajectory = motion.trajectory(
        np.vstack((start, start + 0.1)),
        0.02,
        arm_ratio=(0.5, 0.2),
        gripper_ratio=(0.1, 0.3),
    )
    np.testing.assert_allclose(
        trajectory.max_velocity, np.array(configured.velocity_limit) * [*[0.5] * 7, 0.1]
    )
    np.testing.assert_allclose(
        trajectory.max_acceleration, np.array(configured.acceleration_limit) * [*[0.2] * 7, 0.3]
    )
    assert motion.robot.config == before


def test_collision_sampling_uses_actual_trajectory_peak_velocity(motion, monkeypatch):
    calls = []

    def record_velocity(checker, trajectory, velocity):
        calls.append(velocity.copy())
        np.testing.assert_array_equal(velocity, trajectory.peak_velocity[:-1])
        assert np.all(velocity < trajectory.max_velocity[:-1])
        return True

    monkeypatch.setattr(motion, "smooth_path_collision_free", record_velocity)
    start = motion.robot.state.snapshot().qpos[:7]
    motion.trajectory(np.vstack((start, start + 0.001)), 0.02, object(), arm_ratio=(0.3, 0.7))
    assert len(calls) == 1


@pytest.mark.parametrize("blocked", [None, "start", "middle", "end"])
def test_smooth_collision_sampling_checks_each_boundary_once(blocked):
    trajectory = JointTrajectory(np.array([[0.0, 0.0], [0.4, 0.02]]), 0.5, 1.0)
    calls = []

    class Checker:
        edge_resolution = 0.05

        def is_collision_free(self, state):
            calls.append(state.copy())
            position = state[0]
            return not (
                (blocked == "start" and position == 0)
                or (blocked == "middle" and 0.19 <= position <= 0.21)
                or (blocked == "end" and position == 0.4)
            )

    assert MotionPlanner.smooth_path_collision_free(
        Checker(), trajectory, trajectory.peak_velocity[:1]
    ) == (blocked is None)
    positions = np.concatenate(calls)
    assert len(np.unique(positions)) == len(positions)
    assert positions[0] == 0
    if blocked is None:
        assert positions[-1] == 0.4
        assert len(calls) == ceil(trajectory.duration / 0.02) + 1
        assert np.all(np.diff(positions) <= Checker.edge_resolution / 2)
    elif blocked == "start":
        assert len(calls) == 1
    elif blocked == "middle":
        assert 0.19 <= positions[-1] <= 0.21
    else:
        assert positions[-1] == 0.4


@pytest.mark.parametrize("arm_distance", [0.0, 0.01, 1.0])
def test_collision_sampling_preserves_time_and_joint_spacing_limits(arm_distance):
    trajectory = JointTrajectory(np.array([[0.0, 0.0], [arm_distance, 0.04]]), 100.0, 100.0)
    calls = []

    class Checker:
        edge_resolution = 0.05

        def is_collision_free(self, state):
            calls.append(state.copy())
            return True

    assert MotionPlanner.smooth_path_collision_free(
        Checker(), trajectory, trajectory.peak_velocity[:1]
    )
    if arm_distance == 0:
        assert len(calls) == 1  # A gripper-only trajectory keeps the checked arm fixed.
    else:
        positions = np.concatenate(calls)
        assert positions[0] == 0
        assert positions[-1] == arm_distance
        assert trajectory.duration / (len(calls) - 1) <= 0.02
        assert np.all(np.diff(positions) <= Checker.edge_resolution / 2 + 1e-12)
        configured_interval = Checker.edge_resolution / (2 * trajectory.max_velocity[0])
        assert len(calls) < ceil(trajectory.duration / configured_interval) + 1


def test_sampling_visits_mixed_waypoints_with_adjacent_queries(motion, monkeypatch):
    from mujoco_lab.planning.motion import grasp_pose

    current = motion.robot.state.snapshot().qpos[:7].copy()
    previous = current.copy()
    previous[0] -= 0.02
    motion.robot.target = ControlTarget(previous)
    motion.robot.gripper.set_target(0.03)
    departure = current.copy()
    departure[0] += 0.04
    goal = current.copy()
    goal[1] += 0.06
    pose = grasp_pose(motion.robot)
    calls = []
    ik_references = []

    class FreeChecker(EmptyCollisionChecker):
        bounds = motion.robot.state.get_joint_limits(list(range(7)))

        def is_path_collision_free(self, from_state, to_state, resolution=0.1):
            # Reconstructing the old command would collide in the current scene.
            return not np.array_equal(from_state, previous) and not np.array_equal(
                to_state, previous
            )

    def record_query(config, start, end, bounds, checker):
        assert config is motion.planning
        calls.append((start.copy(), end.copy()))
        # A free edge permits shortcutting this internal bend.
        bend = (start + end) / 2
        bend[2] += 0.02
        return np.vstack((start, bend, end))

    def solve_ik(destination, reference_q):
        assert destination is pose
        ik_references.append(reference_q.copy())
        return departure.copy()

    motion.planning = RRTConnectConfig(seed=19)
    monkeypatch.setattr("mujoco_lab.planning.motion.plan_path", record_query)
    monkeypatch.setattr(motion, "solve_ik", solve_ik)
    monkeypatch.setattr(motion, "smooth_path_collision_free", lambda *_: True)
    request = MotionRequest(
        name="transport",
        mode="sampling",
        waypoints=(pose, goal),
        gripper_target=0.02,
        checker=FreeChecker(),
    )
    trajectory, reason = motion.make_trajectory(request)
    assert reason is None
    assert len(calls) == 2
    assert motion.planning.seed == 19
    np.testing.assert_allclose(calls[0][0], current)
    np.testing.assert_allclose(calls[0][1], departure)
    np.testing.assert_allclose(calls[1][0], departure)
    np.testing.assert_allclose(calls[1][1], goal)
    np.testing.assert_allclose(ik_references, [current])
    # Per-leg shortcuts retain the measured start and required departure endpoint.
    np.testing.assert_allclose(trajectory.path[:, :7], [current, departure, goal])
    assert trajectory.path[0, 7] == pytest.approx(0.03)
    np.testing.assert_allclose(motion.robot.state.snapshot().qpos[:7], current)
    np.testing.assert_array_equal(motion.robot.target.position, previous)
    assert motion.robot.gripper.get_target() == 0.03


def test_pose_hold_keeps_commanded_arm_and_moves_only_gripper(motion, monkeypatch):
    from mujoco_lab.planning.motion import grasp_pose

    previous = motion.robot.target.position.copy()
    previous[0] += 0.05
    motion.robot.target = ControlTarget(previous)
    motion.robot.gripper.set_target(0.03)

    def unexpected_ik(*args, **kwargs):
        raise AssertionError("A hold must not solve pose IK")

    monkeypatch.setattr(motion, "solve_ik", unexpected_ik)
    trajectory, reason = motion.make_trajectory(
        MotionRequest(
            name="close",
            mode="hold",
            waypoints=(grasp_pose(motion.robot),),
            gripper_target=0.0,
        )
    )
    assert reason is None
    assert trajectory.duration > 0
    np.testing.assert_allclose(trajectory.path[:, :7], [previous, previous])
    np.testing.assert_allclose(trajectory.path[:, 7], [0.03, 0.0])
    np.testing.assert_allclose(trajectory.sample(trajectory.duration).position, [*previous, 0.0])
    assert motion.robot.gripper.get_target() == 0.03
