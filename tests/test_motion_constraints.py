"""Stage motion limits respect configured limits and named-joint ordering."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.assets.robot.robot import MotionLimits
from mujoco_lab.control import ControlTarget, create_controller
from mujoco_lab.planning import RRTConnectConfig
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
def test_stage_and_robot_limits_bound_trajectory_derivatives(
    motion, derivative, smooth, monkeypatch
):
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
    stage = replace(
        motion.motion_constraints(),
        **{field: [0.05, *[1.0] * 6, 0.01]},
    )

    class FreeChecker:
        def is_path_collision_free(self, start, goal):
            return True

    # Exercise the checked polyline fallback as well as the smooth curve.
    monkeypatch.setattr(motion, "smooth_path_collision_free", lambda *_: smooth)
    request = MotionRequest(
        name="hold",
        mode="hold",
        waypoints=(np.full(7, 0.1),),
        gripper_target=0.02,
        constraints=stage,
        checker=FreeChecker(),
    )
    trajectory, reason = motion.make_trajectory(request)
    assert reason is None
    assert trajectory.smooth is smooth
    samples = [trajectory.sample(t) for t in np.linspace(0, trajectory.duration, 1001)]
    peaks = np.max(np.abs([getattr(sample, derivative) for sample in samples]), axis=0)
    assert np.all(peaks <= np.array([0.05, *[0.1] * 6, 0.01]) + 1e-10)
    assert peaks[-1] > 0.009  # The stage's lower gripper cap is actually active.
    np.testing.assert_allclose(trajectory.sample(trajectory.duration).position, [*[0.1] * 7, 0.02])
    assert config == before


def test_configured_and_stage_joint_order_do_not_change_timing(motion):
    robot = motion.robot
    config = robot.config.constraints
    config.velocity_limit = [0.1 * (i + 1) for i in range(7)]
    config.acceleration_limit = [0.2 * (i + 1) for i in range(7)]
    stage = replace(motion.motion_constraints(), velocity_limit=[0.15] * 7 + [0.05])
    start = robot.state.snapshot().qpos[:7]
    path = np.vstack((start, start + 0.1))
    baseline = motion.trajectory(path, 0.02, stage)

    for field in ("joint_names", "position_limit", "velocity_limit", "acceleration_limit"):
        setattr(config, field, list(reversed(getattr(config, field))))
    reordered_stage = replace(
        stage,
        **{
            field: list(reversed(getattr(stage, field)))
            for field in ("joint_names", "position_limit", "velocity_limit", "acceleration_limit")
        },
    )
    reordered = motion.trajectory(path, 0.02, reordered_stage)
    assert reordered.duration == baseline.duration
    for time in np.linspace(0, baseline.duration, 21):
        np.testing.assert_allclose(reordered.sample(time).position, baseline.sample(time).position)


def test_omitted_stage_constraints_use_configured_limits(motion):
    start = motion.robot.state.snapshot().qpos[:7]
    path = np.vstack((start, start + 0.1))
    configured = motion.trajectory(path, 0.02)
    faster_stage = replace(
        motion.motion_constraints(), velocity_limit=[100.0] * 8, acceleration_limit=[100.0] * 8
    )
    slower_stage = replace(motion.motion_constraints(), velocity_limit=[0.01] * 8)
    assert motion.trajectory(path, 0.02, faster_stage).duration == configured.duration
    assert motion.trajectory(path, 0.02, slower_stage).duration > configured.duration


@pytest.mark.parametrize("field", ["velocity_limit", "acceleration_limit"])
@pytest.mark.parametrize("value", [0.0, float("inf")])
def test_invalid_stage_limits_are_rejected_before_capping(motion, field, value):
    stage = replace(motion.motion_constraints())
    setattr(stage, field, [value] * 8)
    start = motion.robot.state.snapshot().qpos[:7]
    with pytest.raises(ValueError, match="positive finite"):
        motion.trajectory(np.vstack((start, start)), 0.02, stage)


def test_shared_limits_inherit_override_and_cap_without_mutating_config(motion):
    configured = motion.motion_constraints()
    before = deepcopy(configured)
    recipe = MotionLimits(velocity_limit=0.5, acceleration_limit=2.0)
    stage = MotionLimits(velocity_limit=100.0)
    requested = motion.stage_constraints(recipe.override(stage), MotionLimits())
    constraints = motion.resolve_constraints(requested)
    assert constraints.velocity_limit == configured.velocity_limit
    np.testing.assert_allclose(
        constraints.acceleration_limit,
        [
            *[min(value, 2.0) for value in configured.acceleration_limit[:-1]],
            configured.acceleration_limit[-1],
        ],
    )
    assert configured == before


def test_shared_limits_preserve_partial_joint_values_and_gripper_override(motion):
    configured = motion.motion_constraints()
    arm = MotionLimits(velocity_limit=[0.1 * (i + 1) for i in range(7)])
    requested = motion.stage_constraints(arm, MotionLimits(acceleration_limit=0.05))
    constraints = motion.resolve_constraints(requested)
    assert constraints.velocity_limit[:-1] == arm.velocity_limit
    assert constraints.velocity_limit[-1] == configured.velocity_limit[-1]
    assert constraints.acceleration_limit[:-1] == configured.acceleration_limit[:-1]
    assert constraints.acceleration_limit[-1] == 0.05


@pytest.mark.parametrize(
    "arm,gripper",
    [
        (MotionLimits(velocity_limit=[0.1] * 6), MotionLimits()),
        (MotionLimits(), MotionLimits(velocity_limit=[0.1, 0.2])),
    ],
)
def test_shared_limits_reject_incorrect_joint_counts(motion, arm, gripper):
    with pytest.raises(ValueError, match="one value per controlled joint"):
        motion.stage_constraints(arm, gripper)


def test_stage_groups_follow_joint_names_when_gripper_is_not_last(motion, monkeypatch):
    configured = motion.motion_constraints()
    reordered = replace(
        configured,
        **{
            field: list(reversed(getattr(configured, field)))
            for field in ("joint_names", "position_limit", "velocity_limit", "acceleration_limit")
        },
    )
    monkeypatch.setattr(motion, "motion_constraints", lambda: reordered)
    requested = motion.stage_constraints(
        MotionLimits(velocity_limit=[0.1 * (i + 1) for i in range(7)]),
        MotionLimits(velocity_limit=0.01),
    )
    by_name = dict(zip(requested.joint_names, requested.velocity_limit, strict=True))
    assert by_name[configured.joint_names[-1]] == 0.01
    for i, name in enumerate(configured.joint_names[:-1]):
        assert by_name[name] == pytest.approx(0.1 * (i + 1))


def test_trajectory_uses_current_robot_limits_after_request_is_built(motion):
    request = motion.stage_constraints(MotionLimits(velocity_limit=100.0), MotionLimits())
    motion.robot.config.constraints.velocity_limit = [0.01] * 7
    config = deepcopy(motion.robot.config)
    start = motion.robot.state.snapshot().qpos[:7]
    trajectory = motion.trajectory(np.vstack((start, start + 0.1)), 0.02, request)
    velocities = np.array(
        [trajectory.sample(time).velocity[:7] for time in np.linspace(0, trajectory.duration, 1001)]
    )
    assert np.max(np.abs(velocities)) <= 0.01 + 1e-10
    assert np.max(np.abs(velocities)) > 0.009
    assert motion.robot.config == config


def test_sampling_visits_mixed_waypoints_with_adjacent_calls_and_seeds(motion, monkeypatch):
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
    ik_seeds = []

    class FreeChecker:
        bounds = motion.robot.state.get_joint_limits(list(range(7)))

        def is_path_collision_free(self, start, end):
            # Reconstructing the old command would collide in the current scene.
            return not np.array_equal(start, previous) and not np.array_equal(end, previous)

    def record_query(config, start, end, bounds, checker, *, seed):
        assert config is motion.planning
        calls.append((start.copy(), end.copy(), seed))
        # A free edge permits shortcutting this internal bend.
        bend = (start + end) / 2
        bend[2] += 0.02
        return np.vstack((start, bend, end))

    def solve_ik(destination, seed):
        assert destination is pose
        ik_seeds.append(seed.copy())
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
    assert motion.planning_epoch == 2
    assert [call[2] for call in calls] == [19, 20]
    np.testing.assert_allclose(calls[0][0], current)
    np.testing.assert_allclose(calls[0][1], departure)
    np.testing.assert_allclose(calls[1][0], departure)
    np.testing.assert_allclose(calls[1][1], goal)
    np.testing.assert_allclose(ik_seeds, [current])
    # Per-leg shortcuts retain the measured start and required departure endpoint.
    np.testing.assert_allclose(trajectory.path[:, :7], [current, departure, goal])
    assert trajectory.path[0, 7] == pytest.approx(0.03)
    np.testing.assert_allclose(motion.robot.state.snapshot().qpos[:7], current)
    np.testing.assert_array_equal(motion.robot.target.position, previous)
    assert motion.robot.gripper.get_target() == 0.03


def test_cartesian_motion_starts_at_measured_arm_when_old_command_collides(motion, monkeypatch):
    from mujoco_lab.planning.motion import grasp_pose

    current = motion.robot.state.snapshot().qpos[:7].copy()
    previous = current.copy()
    previous[0] -= 0.02
    motion.robot.target = ControlTarget(previous)
    goal = current.copy()
    goal[0] += 0.04

    class Checker:
        def is_path_collision_free(self, start, end):
            return not np.array_equal(start, previous) and not np.array_equal(end, previous)

    monkeypatch.setattr(motion, "cartesian_path", lambda _: np.vstack((current, goal)))
    monkeypatch.setattr(motion, "smooth_path_collision_free", lambda *_: True)
    trajectory, reason = motion.make_trajectory(
        MotionRequest(
            name="insert",
            mode="cartesian",
            waypoints=(grasp_pose(motion.robot),),
            gripper_target=0.02,
            checker=Checker(),
        )
    )

    assert reason is None
    np.testing.assert_allclose(trajectory.path[:, :7], [current, goal])
    np.testing.assert_array_equal(motion.robot.target.position, previous)
    np.testing.assert_array_equal(motion.robot.state.snapshot().qpos[:7], current)


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
