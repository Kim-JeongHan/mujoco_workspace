"""Stage ratios scale configured limits without changing joint ordering."""

from copy import deepcopy

import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.control import ControlTarget, create_controller
from mujoco_lab.planning import RRTConnectConfig
from mujoco_lab.planning.collision.collision_checker import EmptyCollisionChecker
from mujoco_lab.planning.motion import MotionPlanner, MotionRequest


@pytest.fixture
def motion():
    simulator = Simulator(
        create_cube_stack(1),
        robots=[RobotSpec("panda", "panda", config=load_robot_config("panda"))],
    )
    robot = simulator.robots["panda"]
    robot.change_controller(create_controller(robot, robot.config.controller))
    return MotionPlanner(robot)


@pytest.mark.parametrize("derivative", ["velocity", "acceleration"])
@pytest.mark.parametrize("smooth", [True, False])
def test_stage_ratios_bound_both_interpolation_modes(motion, derivative, smooth, monkeypatch):
    robot = motion.robot
    robot.target = ControlTarget(np.zeros(7))
    robot.gripper.set_target(0)
    config = robot.config
    config.constraints.velocity_limit = [100.0] * 7
    config.constraints.acceleration_limit = [100.0] * 7
    config.gripper.velocity_limit = 100.0
    config.gripper.acceleration_limit = 100.0
    field = f"{derivative}_limit"
    setattr(config.constraints, field, [0.1] * 7)
    setattr(config.gripper, field, 0.02)
    before = deepcopy(config)

    # Exercise the checked polyline fallback as well as the smooth curve.
    monkeypatch.setattr(motion, "smooth_path_collision_free", lambda *_: smooth)
    request = MotionRequest(
        name="hold",
        mode="hold",
        waypoints=(np.full(7, 0.1),),
        gripper_target=0.02,
        arm_ratio=(0.5, 0.5),
        gripper_ratio=(0.5, 0.5),
        checker=EmptyCollisionChecker(),
    )
    trajectory, reason = motion.make_trajectory(request)
    assert reason is None
    assert trajectory.smooth is smooth
    samples = [trajectory.sample(t) for t in np.linspace(0, trajectory.duration, 1001)]
    peaks = np.max(np.abs([getattr(sample, derivative) for sample in samples]), axis=0)
    assert np.all(peaks <= np.array([*[0.05] * 7, 0.01]) + 1e-10)
    assert peaks[-1] > 0.009  # The reduced gripper limit is actually active.
    np.testing.assert_allclose(trajectory.sample(trajectory.duration).position, [*[0.1] * 7, 0.02])
    assert config == before


def test_configured_joint_order_does_not_change_ratio_timing(motion):
    config = motion.robot.config.constraints
    config.velocity_limit = [0.1 * (i + 1) for i in range(7)]
    config.acceleration_limit = [0.2 * (i + 1) for i in range(7)]
    start = motion.robot.state.snapshot().qpos[:7]
    path = np.vstack((start, start + 0.1))
    baseline = motion.trajectory(path, 0.02, arm_ratio=(0.4, 0.7), gripper_ratio=(0.2, 0.8))
    for field in ("joint_names", "position_limit", "velocity_limit", "acceleration_limit"):
        setattr(config, field, list(reversed(getattr(config, field))))
    reordered = motion.trajectory(path, 0.02, arm_ratio=(0.4, 0.7), gripper_ratio=(0.2, 0.8))
    assert reordered.duration == baseline.duration
    for time in np.linspace(0, baseline.duration, 21):
        np.testing.assert_allclose(reordered.sample(time).position, baseline.sample(time).position)


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


def test_collision_sampling_uses_scaled_velocity(motion, monkeypatch):
    calls = []

    def record_velocity(checker, trajectory, velocity):
        calls.append(velocity.copy())
        return True

    monkeypatch.setattr(motion, "smooth_path_collision_free", record_velocity)
    start = motion.robot.state.snapshot().qpos[:7]
    motion.trajectory(np.vstack((start, start + 0.1)), 0.02, object(), arm_ratio=(0.3, 0.7))
    np.testing.assert_allclose(
        calls, [np.array(motion.motion_constraints().velocity_limit[:-1]) * 0.3]
    )


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
