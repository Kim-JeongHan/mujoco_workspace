"""Shared IK, sampling, and checked trajectory construction for manipulation."""

from dataclasses import dataclass, replace
from itertools import pairwise
from math import ceil
from typing import Literal, cast

import numpy as np
from scipy.spatial.transform import Rotation

from mujoco_lab.assets.robot.robot import Constraints, MotionLimits
from mujoco_lab.control.trajectory import JointTrajectory
from mujoco_lab.planning.collision.collision_checker import CollisionChecker
from mujoco_lab.planning.planners import PlannerConfig, plan_path, planner_name
from mujoco_lab.utils import Transform


@dataclass(frozen=True)
class MotionRequest:
    """Ordered motion inputs with explicit joint, sampling, or Cartesian execution.

    Hold and Cartesian requests use one waypoint. Sampling visits every
    waypoint in order; poses are resolved by IK without changing live physics.
    """

    name: str  # Stage label used in failure messages.
    mode: Literal["hold", "sampling", "cartesian"]  # Select the path construction method.
    waypoints: tuple[Transform | np.ndarray, ...]  # Ordered world grasp poses or arm joints (rad).
    gripper_target: float  # Desired gripper joint position (m).
    constraints: Constraints | None = None  # Stage limits, capped at trajectory creation.
    checker: CollisionChecker | None = None  # Stage-specific collision checker.
    wrap_angles: bool = False  # Keep continuous-joint sampling-goal IK angles near the seed.
    failure_reason: str | None = None  # Skip planning and return this failure reason when set.


def grasp_pose(robot) -> Transform:
    """Read the current world pose of the robot's grasp frame."""
    data = robot.data
    site = robot.state.site_id("grasp")
    return Transform(
        rotation=data.site_xmat[site].reshape(3, 3).copy(), translation=data.site_xpos[site].copy()
    )


class MotionPlanner:
    """Construct paths without moving live physics or handling caller exceptions.

    No route or invalid smoothing returns None. Invalid planner output raises
    RuntimeError, and IKError propagates to the expert's execution boundary.
    """

    def __init__(self, robot, planning: PlannerConfig | None = None) -> None:
        self.robot = robot
        self.planning = planning
        self.planning_epoch = 0

    def reset(self) -> None:
        self.planning_epoch = 0

    def motion_constraints(self) -> Constraints:
        """Combine configured arm and gripper constraints in trajectory joint order."""
        constraints = self.robot.state.constraints
        config = self.robot.config
        if (
            constraints is None
            or config is None
            or config.gripper is None
            or self.robot.gripper is None
        ):
            raise ValueError("Motion planning requires arm constraints and gripper motion limits")
        slots = self.robot.state.get_frame_joint_slots("grasp")
        names = [self.robot.state.joint_names[slot] for slot in slots]
        try:
            indices = [constraints.joint_names.index(name) for name in names]
        except ValueError as error:
            raise ValueError(
                "Motion constraints must include every grasp-frame arm joint"
            ) from error
        gripper_slot = self.robot.state.joint_ids.index(self.robot.gripper.joint_id)
        return Constraints(
            joint_names=[*names, self.robot.state.joint_names[gripper_slot]],
            position_limit=[
                *(constraints.position_limit[index].copy() for index in indices),
                self.robot.gripper.get_control_limits().tolist(),
            ],
            velocity_limit=[
                *(constraints.velocity_limit[index] for index in indices),
                config.gripper.velocity_limit,
            ],
            acceleration_limit=[
                *(constraints.acceleration_limit[index] for index in indices),
                config.gripper.acceleration_limit,
            ],
        )

    def stage_constraints(self, arm: MotionLimits, gripper: MotionLimits) -> Constraints:
        """Bind requested task limits to named joints; defer capping until trajectory creation."""
        constraints = self.motion_constraints()
        state = self.robot.state
        arm_names = [state.joint_names[slot] for slot in state.get_frame_joint_slots("grasp")]
        gripper_name = state.joint_names[state.joint_ids.index(self.robot.gripper.joint_id)]
        groups = ((arm, arm_names), (gripper, [gripper_name]))
        values = {}
        for field in ("velocity_limit", "acceleration_limit"):
            limits = list(getattr(constraints, field))
            for requested, names in groups:
                value = getattr(requested, field)
                if value is None:
                    continue
                expanded = value if isinstance(value, list) else [value] * len(names)
                if len(expanded) != len(names):
                    raise ValueError(f"{field} must contain one value per controlled joint")
                for name, limit in zip(names, expanded, strict=True):
                    limits[constraints.joint_names.index(name)] = limit
            values[field] = limits
        return replace(constraints, **values)

    def resolve_constraints(self, requested: Constraints | None = None) -> Constraints:
        """Align task limits by joint name and cap them once at the configured robot limits."""
        configured = self.motion_constraints()
        if requested is None:
            return configured
        if len(set(requested.joint_names)) != len(requested.joint_names):
            raise ValueError("Stage constraint joint names must be unique")
        try:
            indices = [requested.joint_names.index(name) for name in configured.joint_names]
        except ValueError as error:
            raise ValueError("Stage constraints must include every trajectory joint") from error
        values = {}
        for field in ("velocity_limit", "acceleration_limit"):
            limits = np.asarray(getattr(requested, field), dtype=float)
            if (
                limits.shape != (len(requested.joint_names),)
                or not np.isfinite(limits).all()
                or np.any(limits <= 0)
            ):
                raise ValueError(f"Stage {field} must contain one positive finite value per joint")
            values[field] = np.minimum(getattr(configured, field), limits[indices]).tolist()
        return replace(configured, **values)

    def make_trajectory(self, request: MotionRequest):
        """Resolve motion, rebasing moving stages at the measured arm state."""
        if request.failure_reason is not None:
            return None, request.failure_reason
        if not request.waypoints or (request.mode != "sampling" and len(request.waypoints) != 1):
            raise ValueError(
                "Hold and Cartesian requests require one waypoint; sampling requires at least one"
            )
        slots = self.robot.state.get_frame_joint_slots("grasp")
        current = self.robot.state.snapshot().qpos[slots].copy()
        previous = self.robot.target.position.copy()
        checker = request.checker
        if request.mode == "hold":
            target = request.waypoints[-1]
            target = target if isinstance(target, np.ndarray) else previous
            path = np.vstack((previous, target))
        elif request.mode == "cartesian":
            path = self.cartesian_path(cast(Transform, request.waypoints[-1]))
        else:
            if self.planning is None:
                raise ValueError("A sampling planner configuration is required")
            waypoints = []
            for index, waypoint in enumerate(request.waypoints):
                if isinstance(waypoint, Transform):
                    waypoint = (
                        self.solve_ik(waypoint, current, wrap_angles=request.wrap_angles)
                        if index == len(request.waypoints) - 1
                        else self.solve_ik(waypoint, current)
                    )
                waypoints.append(waypoint)
            legs = []
            # The old arm target can collide in the new scene after release.
            # Keep the measured start checked by the planner instead of bridging back.
            start = current
            for index, goal in enumerate(waypoints):
                vertices = self.plan_arm_path(start, goal, checker)
                label = " departure" if index < len(waypoints) - 1 else ""
                if vertices is None:
                    return None, f"No {planner_name(self.planning)}{label} route for {request.name}"
                vertices = self.shortcut_path(vertices, checker)
                legs.append(vertices if index == 0 else vertices[1:])
                start = goal
            path = np.vstack(legs)
        if checker is not None and any(
            not checker.is_path_collision_free(a, b) for a, b in pairwise(path)
        ):
            return None, f"Collision on {request.name} path"
        trajectory = self.trajectory(path, request.gripper_target, request.constraints, checker)
        return trajectory, None if trajectory is not None else f"Collision on {request.name} path"

    def solve_ik(self, pose: Transform, seed: np.ndarray, *, wrap_angles: bool = False):
        q = self.robot.state.solve_ik(pose, frame="grasp", seed=seed)
        if wrap_angles:
            limits = self.robot.state.get_joint_limits(list(range(len(q))))
            for index in range(len(q)):
                if not np.isfinite(limits[index]).all():
                    q[index] = seed[index] + (q[index] - seed[index] + np.pi) % (2 * np.pi) - np.pi
        return q

    def site_pose(self) -> Transform:
        return grasp_pose(self.robot)

    def cartesian_path(self, destination: Transform, *, resolution: float = 0.01) -> np.ndarray:
        measured = self.site_pose()
        start = measured.as_translation()
        goal = destination.as_translation()
        start_rotation = measured.as_rotation()
        delta_rotation = cast(
            Rotation, destination.as_rotation() * start_rotation.inv()
        ).as_rotvec()
        slots = self.robot.state.get_frame_joint_slots("grasp")
        vertices = [self.robot.state.snapshot().qpos[slots].copy()]
        steps = max(2, int(np.ceil(np.linalg.norm(goal - start) / resolution)) + 1)
        for fraction in np.linspace(0, 1, steps)[1:]:
            pose = Transform(
                rotation=cast(
                    Rotation, Rotation.from_rotvec(delta_rotation * fraction) * start_rotation
                ),
                translation=start + fraction * (goal - start),
            )
            vertices.append(self.solve_ik(pose, vertices[-1], wrap_angles=True))
        return np.asarray(vertices)

    def plan_arm_path(self, start: np.ndarray, goal: np.ndarray, checker) -> np.ndarray | None:
        if self.planning is None:
            raise ValueError("A sampling planner configuration is required")
        seed = self.planning.seed
        seed = seed + self.planning_epoch if seed is not None else None
        self.planning_epoch += 1
        vertices = plan_path(
            self.planning, start, goal, [tuple(row) for row in checker.bounds], checker, seed=seed
        )
        if vertices is None:
            return None
        if len(vertices) == 1:
            vertices = np.vstack((vertices, goal))
        return vertices

    @staticmethod
    def shortcut_path(path: np.ndarray, checker) -> np.ndarray:
        if len(path) <= 2:
            return path
        kept = [path[0]]
        index = 0
        while index < len(path) - 1:
            for successor in range(len(path) - 1, index, -1):
                if checker.is_path_collision_free(path[index], path[successor]):
                    break
            else:
                raise RuntimeError("Planner returned an arm edge in collision")
            kept.append(path[successor])
            index = successor
        return np.asarray(kept)

    @staticmethod
    def smooth_path_collision_free(checker, trajectory: JointTrajectory, velocity: np.ndarray):
        arm_size = len(velocity)
        if trajectory.duration == 0:
            return checker.is_collision_free(trajectory.path[0, :arm_size])
        interval = min(0.02, checker.edge_resolution / (2 * np.linalg.norm(velocity)))
        steps = max(1, ceil(trajectory.duration / interval))
        previous = trajectory.sample(0).position[:arm_size]
        for elapsed in np.linspace(trajectory.duration / steps, trajectory.duration, steps):
            current = trajectory.sample(float(elapsed)).position[:arm_size]
            if not checker.is_path_collision_free(previous, current):
                return False
            previous = current
        return True

    def trajectory(
        self, arm_path, gripper_target, constraints: Constraints | None = None, checker=None
    ):
        """Cap stage motion limits, then time and check the resulting trajectory.

        Position bounds belong to IK and collision checking. This method uses
        only velocity and acceleration limits for timing and smoothing checks.
        """
        resolved = self.resolve_constraints(constraints)
        velocity = np.asarray(resolved.velocity_limit, dtype=float)
        acceleration = np.asarray(resolved.acceleration_limit, dtype=float)
        gripper = np.full((len(arm_path), 1), gripper_target)
        gripper[0, 0] = self.robot.gripper.get_target()
        path = np.column_stack((arm_path, gripper))
        trajectory = JointTrajectory(path, velocity, acceleration)
        if checker is not None and not self.smooth_path_collision_free(
            checker, trajectory, velocity[:-1]
        ):
            if any(not checker.is_path_collision_free(a, b) for a, b in pairwise(arm_path)):
                return None
            return JointTrajectory(path, velocity, acceleration, smooth=False)
        return trajectory
