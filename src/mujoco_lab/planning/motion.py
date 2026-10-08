"""Shared IK, sampling, and checked trajectory construction for manipulation."""

from dataclasses import dataclass
from itertools import pairwise
from math import ceil
from typing import Literal

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from mujoco_lab.assets.robot.robot import Constraints
from mujoco_lab.control.trajectory import JointTrajectory, MotionRatio
from mujoco_lab.planning.collision.collision_checker import CollisionChecker
from mujoco_lab.planning.planners import PlannerConfig, plan_path, planner_name
from mujoco_lab.robot import Robot
from mujoco_lab.state import IKError
from mujoco_lab.utils import Transform


@dataclass(frozen=True)
class MotionRequest:
    """Ordered motion inputs with explicit joint, sampling, or Cartesian execution.

    Hold and Cartesian requests use one waypoint. Sampling visits every
    waypoint in order. Blocked routes retry alternate IK solutions without
    changing the requested grasp poses or live physics.
    """

    name: str  # Stage label used in failure messages.
    mode: Literal["hold", "sampling", "cartesian"]  # Select the path construction method.
    waypoints: tuple[Transform | np.ndarray, ...]  # Ordered world grasp poses or arm joints (rad).
    gripper_target: float  # Desired total gripper opening width (m).
    arm_ratio: MotionRatio = (1.0, 1.0)  # Fractions of configured arm velocity/acceleration.
    gripper_ratio: MotionRatio = (1.0, 1.0)  # Fractions of configured opening-width limits.
    checker: CollisionChecker | None = None  # Stage-specific collision checker.
    shape_preserving: bool = False  # Keep smooth joint curves inside each waypoint interval.
    cartesian_resolution: float = 0.01  # Maximum spacing of Cartesian IK waypoints (m).
    failure_reason: str | None = None  # Skip planning and return this failure reason when set.


def grasp_pose(robot) -> Transform:
    """Read the current world pose of the robot's grasp frame."""
    data = robot.data
    site = robot.state.site_id("grasp")
    return Transform(rotation=data.site_xmat[site].reshape(3, 3), translation=data.site_xpos[site])


class MotionPlanner:
    """Construct paths without moving live physics or handling caller exceptions.

    No route or invalid smoothing returns None. Invalid planner output raises
    RuntimeError, and IKError propagates to the expert's execution boundary.
    """

    def __init__(self, robot: Robot, planning: PlannerConfig | None = None) -> None:
        self.robot = robot
        self.planning = planning

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
        indices = [constraints.joint_names.index(name) for name in names]
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
        previous_target = self.robot.target
        if previous_target is None:
            raise ValueError("Motion planning requires a joint-target controller")
        previous = previous_target.position.copy()
        checker = request.checker
        if request.mode == "hold":
            target = request.waypoints[-1]
            target = target if isinstance(target, np.ndarray) else previous
            path = np.vstack((previous, target))
        elif request.mode == "cartesian":
            destination = request.waypoints[-1]
            if not isinstance(destination, Transform):
                raise TypeError("Cartesian motion requires a Transform waypoint")
            path = self.cartesian_path(destination, resolution=request.cartesian_resolution)
        else:
            if self.planning is None:
                raise ValueError("A sampling planner configuration is required")
            legs = []
            # The old arm target can collide in the new scene after release.
            # Keep the measured start checked by the planner instead of bridging back.
            start = current
            for index, waypoint in enumerate(request.waypoints):
                vertices = self.sampling_path(start, waypoint, checker)
                label = " departure" if index < len(request.waypoints) - 1 else ""
                if vertices is None:
                    return None, f"No {planner_name(self.planning)}{label} route for {request.name}"
                vertices = self.shortcut_path(vertices, checker)
                legs.append(vertices if index == 0 else vertices[1:])
                start = vertices[-1]
            path = np.vstack(legs)
        trajectory = self.trajectory(
            path,
            request.gripper_target,
            checker,
            arm_ratio=request.arm_ratio,
            gripper_ratio=request.gripper_ratio,
            shape_preserving=request.shape_preserving,
        )
        if trajectory is None and request.mode == "cartesian":
            trajectory = self.branch_trajectory(
                current,
                request.waypoints,
                request.gripper_target,
                checker,
                arm_ratio=request.arm_ratio,
                gripper_ratio=request.gripper_ratio,
                shape_preserving=request.shape_preserving,
            )
        return trajectory, None if trajectory is not None else f"Collision on {request.name} path"

    def ik_candidates(
        self, pose, reference_q, checker, *, use_default: bool = False
    ) -> tuple[list[np.ndarray], np.ndarray | None]:
        """Check alternate IK solutions without changing the requested grasp pose.

        Forte forearm and wrist seeds escape folded branches. Retry from the
        configured default posture only after local branches cannot make a route.
        Return the unchecked reference solution for deferred execution failures.
        """
        references = [reference_q]
        if self.robot.robot_type == "forte":
            base = reference_q
            offsets = (-np.pi / 2, np.pi / 2)
            if use_default:
                config = self.robot.config
                if config is None:
                    raise ValueError("IK retries require a configured default pose")
                base = np.asarray(config.pose.default[: len(reference_q)])
                references = [base]
                offsets = (*offsets, -np.pi, np.pi)
            for forearm in offsets:
                for wrist in offsets:
                    seed = base.copy()
                    seed[4] += forearm
                    seed[6] += wrist
                    references.append(seed)
        current = self.robot.state.snapshot().qpos[: len(reference_q)].copy()
        current_scheduled = any(np.array_equal(seed, current) for seed in references)
        candidates = []
        unchecked = None
        for seed in references:
            try:
                q = self.solve_ik(pose, seed, retry_current=False)
            except IKError:
                if not current_scheduled:
                    references.append(current)
                    current_scheduled = True
                continue
            if unchecked is None:
                unchecked = q
            if checker.is_collision_free(q):
                candidates.append(q)
        return sorted(candidates, key=lambda q: np.linalg.norm(q - reference_q)), unchecked

    def waypoint_poses(self, waypoints) -> list[Transform]:
        """Resolve joint waypoints with private forward kinematics."""
        model = self.robot.model
        data = mujoco.MjData(model)
        mujoco.mj_copyData(data, model, self.robot.data)
        slots = self.robot.state.get_frame_joint_slots("grasp")
        indices = [self.robot.state.qpos_indices[slot] for slot in slots]
        site = self.robot.state.site_id("grasp")
        poses = []
        for waypoint in waypoints:
            if isinstance(waypoint, Transform):
                poses.append(waypoint)
            else:
                data.qpos[indices] = waypoint
                mujoco.mj_kinematics(model, data)
                poses.append(
                    Transform(
                        rotation=data.site_xmat[site].reshape(3, 3),
                        translation=data.site_xpos[site],
                    )
                )
        return poses

    def sampling_path(self, start, waypoint, checker):
        """Try the original goal, then checked IK alternatives if its route fails."""
        goal = self.solve_ik(waypoint, start) if isinstance(waypoint, Transform) else waypoint
        if checker.is_collision_free(goal):
            path = self.plan_arm_path(start, goal, checker)
            if path is not None:
                return path
        pose = self.waypoint_poses((waypoint,))[0]
        for use_default in (False, True):
            candidates, _ = self.ik_candidates(pose, start, checker, use_default=use_default)
            for candidate in candidates:
                if np.allclose(candidate, goal):
                    continue
                path = self.plan_arm_path(start, candidate, checker)
                if path is not None:
                    return path
        return None

    def solve_ik(
        self,
        pose: Transform,
        reference_q: np.ndarray,
        *,
        retry_current: bool = True,
    ):
        return self.robot.state.solve_ik(pose, reference_q, retry_current=retry_current)

    def cartesian_path(self, destination: Transform, *, resolution: float = 0.01) -> np.ndarray:
        measured = grasp_pose(self.robot)
        start = measured.as_translation()
        goal = destination.as_translation()
        start_rotation = measured.as_rotation()
        relative_rotation = destination.as_rotation() * start_rotation.inv()
        # SciPy includes NotImplemented in the result type even for two Rotations.
        delta_rotation = relative_rotation.as_rotvec()  # ty: ignore[unresolved-attribute]
        slots = self.robot.state.get_frame_joint_slots("grasp")
        vertices = [self.robot.state.snapshot().qpos[slots].copy()]
        steps = max(2, int(np.ceil(np.linalg.norm(goal - start) / resolution)) + 1)
        for fraction in np.linspace(0, 1, steps)[1:]:
            rotation = Rotation.from_rotvec(delta_rotation * fraction) * start_rotation
            # Both operands are Rotations; retain the original composition operation.
            pose = Transform(
                rotation=rotation,  # ty: ignore[invalid-argument-type]
                translation=start + fraction * (goal - start),
            )
            vertices.append(self.solve_ik(pose, vertices[-1]))
        return np.asarray(vertices)

    def plan_arm_path(self, start: np.ndarray, goal: np.ndarray, checker) -> np.ndarray | None:
        if self.planning is None:
            raise ValueError("A sampling planner configuration is required")
        vertices = plan_path(
            self.planning, start, goal, [tuple(row) for row in checker.bounds], checker
        )
        if vertices is None:
            return None
        if len(vertices) == 1:
            vertices = np.vstack((vertices, goal))
        return vertices

    def orientation_path(self, path):
        """Refine joint edges around the Cartesian interpolation of their grasp poses."""
        poses = self.waypoint_poses(path)
        refined = [path[0]]
        for start, end, first, second in zip(
            path[:-1], path[1:], poses[:-1], poses[1:], strict=True
        ):
            interpolation = Slerp(
                [0, 1], Rotation.concatenate([first.as_rotation(), second.as_rotation()])
            )
            for fraction in (1 / 3, 2 / 3):
                pose = Transform(
                    rotation=interpolation(fraction),
                    translation=first.as_translation()
                    + fraction * (second.as_translation() - first.as_translation()),
                )
                reference_q = start + fraction * (end - start)
                refined.append(self.solve_ik(pose, reference_q))
            refined.append(end)
        return np.asarray(refined)

    def heuristic_trajectory(self, path, gripper_target, checker, *, arm_ratio, gripper_ratio):
        """Check a heuristic route and retry other IK branches when it is blocked."""
        refined = self.orientation_path(path) if self.robot.robot_type == "forte" else path
        trajectory = self.trajectory(
            refined,
            gripper_target,
            checker,
            arm_ratio=arm_ratio,
            gripper_ratio=gripper_ratio,
        )
        if trajectory is None:
            trajectory = self.branch_trajectory(
                path[0],
                tuple(path[1:]),
                gripper_target,
                checker,
                arm_ratio=arm_ratio,
                gripper_ratio=gripper_ratio,
            )
        return trajectory

    def branch_trajectory(
        self,
        start,
        waypoints,
        gripper_target,
        checker,
        *,
        arm_ratio,
        gripper_ratio,
        shape_preserving=False,
    ):
        """Connect grasp poses through checked IK branches, then check the trajectory."""
        poses = self.waypoint_poses(waypoints)
        for use_default in (False, True):
            routes = [(start,)]
            for pose in poses:
                candidates, _ = self.ik_candidates(pose, start, checker, use_default=use_default)
                routes = [
                    (*route, candidate)
                    for route in routes
                    for candidate in candidates
                    if checker.is_path_collision_free(route[-1], candidate)
                ]
                if not routes:
                    break
                routes.sort(key=lambda route: np.linalg.norm(np.diff(route, axis=0), axis=1).sum())
                routes = routes[:5]
            for route in routes:
                try:
                    path = self.orientation_path(route)
                except IKError:
                    continue
                trajectory = self.trajectory(
                    path,
                    gripper_target,
                    checker,
                    arm_ratio=arm_ratio,
                    gripper_ratio=gripper_ratio,
                    shape_preserving=shape_preserving,
                )
                if trajectory is not None:
                    return trajectory
        return None

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
        self,
        arm_path,
        gripper_target,
        checker=None,
        *,
        arm_ratio: MotionRatio = (1.0, 1.0),
        gripper_ratio: MotionRatio = (1.0, 1.0),
        shape_preserving: bool = False,
    ):
        """Pass configured limits and stage ratios to the checked joint trajectory.

        Position bounds belong to IK and collision checking. This method uses
        only velocity and acceleration limits for timing and smoothing checks.
        """
        resolved = self.motion_constraints()
        robot_gripper = self.robot.gripper
        if robot_gripper is None:
            raise ValueError("Motion planning requires a configured gripper")
        velocity = np.asarray(resolved.velocity_limit, dtype=float)
        acceleration = np.asarray(resolved.acceleration_limit, dtype=float)
        gripper = np.full((len(arm_path), 1), gripper_target)
        gripper[0, 0] = robot_gripper.get_target()
        path = np.column_stack((arm_path, gripper))
        ratios = np.array([*[arm_ratio] * (len(resolved.joint_names) - 1), gripper_ratio])
        trajectory = JointTrajectory(
            path, velocity, acceleration, ratio=ratios, shape_preserving=shape_preserving
        )
        if checker is not None and not self.smooth_path_collision_free(
            checker, trajectory, trajectory.max_velocity[:-1]
        ):
            if any(not checker.is_path_collision_free(a, b) for a, b in pairwise(arm_path)):
                return None
            return JointTrajectory(path, velocity, acceleration, ratio=ratios, smooth=False)
        return trajectory
