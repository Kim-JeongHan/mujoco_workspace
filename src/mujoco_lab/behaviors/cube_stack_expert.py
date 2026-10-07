"""Execute generated cube stacking stages as physical actions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from mujoco_lab.behaviors.cube_stack import CubeStackTask
from mujoco_lab.behaviors.cube_stack_recipe import CubeStackRecipe, StageRecipe
from mujoco_lab.behaviors.expert import Expert
from mujoco_lab.behaviors.grasp import GraspMonitor
from mujoco_lab.behaviors.trajectory import TrajectoryExecution
from mujoco_lab.control import ControlTarget
from mujoco_lab.control.trajectory import JointTrajectory, MotionRatio
from mujoco_lab.planning import PlannerConfig
from mujoco_lab.planning.collision.manipulation import ManipulationCollisionChecker
from mujoco_lab.planning.motion import MotionPlanner, MotionRequest, grasp_pose
from mujoco_lab.state import IKError
from mujoco_lab.utils import Transform


@dataclass(frozen=True)
class PlannedStage:
    """Ordered arm waypoints, gripper command, and readiness settings for one stage."""

    name: str
    waypoints: tuple[np.ndarray, ...]
    gripper_target: float
    recipe: StageRecipe
    cube_index: int
    settle_timeout: float | None = None


class CubeStackExpert(Expert):
    """Execute generated Panda or Forte stages as physical actions.

    ``act`` never applies its action. ``update`` adapts it to Simulator's
    target-updater callback for direct simulation; collectors call ``act`` and
    pass the returned action through their environment instead. The caller
    must bind a seven-joint target controller and gripper before construction.

    The application explicitly selects heuristic or sampling execution.
    Sampling requires an explicit planner configuration.
    """

    def __init__(
        self,
        task: CubeStackTask,
        *,
        recipe: CubeStackRecipe,
        method: Literal["heuristic", "sampling"],
        planning: PlannerConfig | None = None,
    ) -> None:
        super().__init__()
        if method not in ("heuristic", "sampling"):
            raise ValueError(f"Unsupported cube execution method: {method}")
        if method == "sampling" and planning is None:
            raise ValueError("Sampling execution requires an explicit planner configuration")
        if method == "heuristic" and planning is not None:
            raise ValueError("Heuristic execution does not use a planner configuration")
        self.method = method
        self.task = task
        self.simulator = task.simulator
        self.cubes = task.cubes
        if len(self.simulator.robots) != 1:
            raise ValueError("Cube stacking execution requires one robot")
        self.robot = next(iter(self.simulator.robots.values()))
        robot = self.robot
        if robot.gripper is None or robot.controller is None or robot.target is None:
            raise ValueError("Trajectory expert requires a joint controller and gripper")
        if robot.target.position.shape != (7,) or robot.control_joint_names != tuple(
            robot.state.joint_names[:7]
        ):
            raise ValueError("Trajectory expert requires a seven-joint target controller")

        if self.robot.robot_type not in ("panda", "forte"):
            raise ValueError("Cube stacking execution requires panda or forte")
        self.gripper = robot.gripper
        self.recipe = recipe
        self._grasp = GraspMonitor(recipe.lost_grasp_grace_s)
        self.planning = planning
        self.motion = MotionPlanner(self.robot, planning)
        self.reset()

    def reset(self, initial_obs: Any = None, info: dict[str, Any] | None = None) -> None:
        """Replan from current physics after env reset; observation and info are unused.

        The expert uses privileged simulator state. This method does not reset
        physics, task measurements, controllers, or physical command targets.
        """
        mujoco.mj_forward(self.simulator.model, self.simulator.data)
        self.starts = np.array(
            [self.simulator.data.body(f"cube{i}/object_0").xpos for i in range(self.cubes)]
        )
        self.goals = np.array(
            [self.simulator.data.body(f"cube{i}/object_target_0").xpos for i in range(self.cubes)]
        )
        self._plan = (
            self.build_stages(self.starts, self.goals)
            if self.method == "sampling"
            else self.group_stages(self.starts, self.goals)
        )
        self.stage = 0
        self.execution = TrajectoryExecution(self.simulator.data.time)
        self.failed = False
        self.failure_reason = None
        self._transport_peak = float("-inf")
        self._release_checker: ManipulationCollisionChecker | None = None
        self._grasp.reset()

    def get_stage_name(self):
        return self._plan[self.stage].name if self.stage < len(self._plan) else "settle"

    def _hold_action(self):
        target = self.robot.target
        if target is None:
            raise RuntimeError("Trajectory expert requires a control target")
        return np.r_[target.position[:7], self.gripper.get_target()]

    def act(self, obs: Any = None, *, dt: float = 0.0) -> np.ndarray:
        if self.failed or self.stage >= len(self._plan):
            return self._hold_action()
        now = self.simulator.data.time
        stage = self._plan[self.stage]
        reason = self._grasp_failure(stage, started=self.execution.trajectory is not None)
        if reason is not None:
            self.failed, self.failure_reason = True, reason
            return self._hold_action()
        current = self.robot.state.snapshot().qpos[:7]
        if self.execution.trajectory is None:
            try:
                trajectory, reason = self.make_trajectory(self._plan[self.stage])
            except IKError as error:
                trajectory, reason = None, str(error)
            if trajectory is None:
                self.failed, self.failure_reason = True, reason
                return self._hold_action()
            self.execution.start(trajectory, now)
        stage = self._plan[self.stage]
        required_lift = self.min_transport_lift(stage)
        if required_lift is not None:
            height = self.simulator.data.body(f"cube{stage.cube_index}/object_0").xpos[2]
            self._transport_peak = max(self._transport_peak, float(height))
        next_act, complete = self.execution.sample(now, current, self.recipe.arm_tolerance, dt=dt)
        trajectory = self.execution.trajectory
        if trajectory is None:
            raise RuntimeError("Trajectory expert has no active trajectory")
        if (
            complete
            and required_lift is not None
            and self._transport_peak <= self.starts[stage.cube_index, 2] + required_lift
        ):
            self.failed = True
            self.failure_reason = f"Insufficient physical lift during {stage.name}"
            return next_act
        if self._advance_stage(current, float(next_act[7]), path_complete=complete):
            return next_act
        timeout = self._plan[self.stage].settle_timeout
        if timeout is not None and now - self.execution.start_time > trajectory.duration + timeout:
            self.failed, self.failure_reason = True, f"Timed out executing {self.get_stage_name()}"
        return next_act

    def update(self, simulator) -> None:
        if simulator is not self.simulator:
            raise ValueError("Expert is bound to a different simulator")
        action = self.act()
        self.robot.target = ControlTarget(action[:7])
        self.gripper.set_target(float(action[7]))
        success = self.task.status().released_stable_stack
        if self.failed or (self.stage >= len(self._plan) and success):
            simulator.stop()

    def _advance_stage(
        self, current: np.ndarray, finger_target: float, *, path_complete: bool
    ) -> bool:
        """Advance after trajectory completion, reach, and requested physical checks."""
        stage = self._plan[self.stage]
        trajectory = self.execution.trajectory
        goal = trajectory.path[-1, :7] if trajectory is not None else stage.waypoints[-1]
        close_enough = np.max(np.abs(goal - current[:7])) < self.recipe.arm_tolerance
        # A cube can prevent the fingers from reaching the commanded closing gap.
        finger_close = abs(stage.gripper_target - finger_target) < self.recipe.gripper_tolerance
        if not (path_complete and close_enough and finger_close):
            return False
        physical_ready = self._grasp.ready(
            stage.recipe.gripper_mode, self.task.has_grasp(stage.cube_index)
        )
        if stage.recipe.name == "release":
            physical_ready = physical_ready and not self.task.touches_robot(stage.cube_index)
            if self.method == "sampling":
                # Retract planning freezes the fingers at their measured opening.
                physical_ready = physical_ready and (
                    abs(self.gripper.get_width() - stage.gripper_target)
                    < self.recipe.gripper_tolerance
                )
            if physical_ready and self.method == "sampling":
                # Live contacts can precede the latest integrated positions.
                # Check the same fresh scene that the next retract plan will use.
                if self._release_checker is None:
                    self._release_checker = self.collision_checker(stage.cube_index, "retract")
                else:
                    self._release_checker.refresh()
                physical_ready = self._release_checker.is_collision_free(current)
        if stage.recipe.min_lift_height_m is not None:
            physical_ready = physical_ready and (
                self.simulator.data.body(f"cube{stage.cube_index}/object_0").xpos[2]
                > self.starts[stage.cube_index, 2] + stage.recipe.min_lift_height_m
            )
        if physical_ready:
            self.stage += 1
            self.execution.trajectory = None
            self._release_checker = None
            self._grasp.reset()
            return True
        return False

    def _grasp_failure(self, stage: PlannedStage, *, started: bool) -> str | None:
        return self._grasp.failure(
            mode=stage.recipe.gripper_mode,
            hold=stage.recipe.name in ("close", "release"),
            started=started,
            grasped=self.task.has_grasp(stage.cube_index),
            now=self.simulator.data.time,
            stage_name=stage.name,
        )

    START_STAGES = frozenset(("above_pick", "pick", "close", "lift"))

    def stage_pose(
        self,
        stage: StageRecipe,
        index: int,
        starts: np.ndarray,
        goals: np.ndarray,
        rotation: Rotation | np.ndarray,
    ) -> Transform:
        """Align pick-side targets with measured cube yaw; keep place targets nominal."""
        reference = starts[index] if stage.name in self.START_STAGES else goals[index]
        if stage.name not in self.START_STAGES:
            return Transform(rotation=rotation, translation=reference + stage.offset_xyz_m)
        cube_rotation = self.simulator.data.body(f"cube{index}/object_0").xmat.reshape(3, 3)
        measured_yaw = np.arctan2(cube_rotation[1, 0], cube_rotation[0, 0])
        # A square cube admits the same grasp after any quarter turn.
        quarter_turn = np.pi / 2
        yaw = (measured_yaw + quarter_turn / 2) % quarter_turn - quarter_turn / 2
        if abs(yaw) < 1e-12:
            yaw = 0.0
        elif abs(abs(yaw) - quarter_turn / 2) < 1e-12:
            yaw = -quarter_turn / 2
        if yaw == 0:
            return Transform(rotation=rotation, translation=reference + stage.offset_xyz_m)
        yaw_rotation = Rotation.from_euler("z", yaw)
        nominal_rotation = (
            rotation if isinstance(rotation, Rotation) else Rotation.from_matrix(rotation)
        )
        rotated = yaw_rotation * nominal_rotation
        # Both operands are Rotations; SciPy's annotation also includes NotImplemented.
        return Transform(
            rotation=rotated,  # ty: ignore[invalid-argument-type]
            translation=reference + yaw_rotation.apply(stage.offset_xyz_m),
        )

    SAMPLING_STAGES = frozenset(("pick", "close", "place", "release", "retract"))

    def sampling_request(self, stage, planned_start):
        """Own cube workflow semantics; the planner receives only motion inputs."""
        robot = self.robot
        phase = stage.recipe.name
        reason = self._grasp_failure(stage, started=False)
        if phase == "pick":
            actual = robot.data.body(f"cube{stage.cube_index}/object_0").xpos
            if np.linalg.norm(actual - planned_start) > 0.01:
                reason = f"cube{stage.cube_index} moved more than 1 cm from its planned pick pose"
        hold = phase in ("close", "release")
        waypoints = stage.waypoints
        if phase == "place" and reason is None:
            measured = grasp_pose(robot)
            waypoints = (
                Transform(
                    rotation=measured.as_rotation(),
                    translation=measured.as_translation() + [0, 0, 0.06],
                ),
                *waypoints,
            )
        arm_ratio, gripper_ratio = self.stage_ratios(stage)
        return MotionRequest(
            name=stage.name,
            mode="hold" if hold else "sampling",
            waypoints=waypoints,
            gripper_target=stage.gripper_target,
            arm_ratio=arm_ratio,
            gripper_ratio=gripper_ratio,
            checker=None if hold else self.collision_checker(stage.cube_index, phase),
            failure_reason=reason,
        )

    def collision_checker(self, cube_index: int, phase: str):
        """Describe stage-specific support and grasp allowances for one cube."""
        return ManipulationCollisionChecker(
            self.robot,
            f"cube{cube_index}:{phase}",
            support_geom=self.task._supports[cube_index] if phase == "place" else None,
            departure_support_geom="table/box" if phase == "place" else None,
        )

    def min_transport_lift(self, stage: PlannedStage) -> float | None:
        if stage.recipe.name != "place":
            return None
        if self.method == "sampling":
            return 0.04
        return next(
            recipe.min_lift_height_m for recipe in self.recipe.stages if recipe.name == "lift"
        )

    def make_trajectory(self, stage):
        self._transport_peak = float("-inf")
        return (
            self.motion.make_trajectory(
                self.sampling_request(stage, self._starts[stage.cube_index])
            )
            if self.method == "sampling"
            else self.heuristic_trajectory(stage)
        )

    def forte_heuristic_ik_candidates(
        self, pose, reference_q, checker
    ) -> tuple[list[np.ndarray], np.ndarray | None]:
        """Return checked branches and the unchecked reference/current solution.

        Own retries here so the current posture is attempted at most once.
        Keep the reference/current solution for deferred collision failures.
        """
        references = [reference_q]
        # Forearm/wrist reference postures escape folded wrist branches without
        # changing the requested grasp pose or using a sampling path planner.
        for sign in (-1, 1):
            for wrist_sign in (-1, 1):
                candidate_reference_q = reference_q.copy()
                candidate_reference_q[4] += sign * np.pi / 2
                candidate_reference_q[6] += wrist_sign * np.pi / 2
                references.append(candidate_reference_q)
        current = self.robot.state.snapshot().qpos[:7].copy()
        current_scheduled = any(
            np.array_equal(candidate_reference_q, current) for candidate_reference_q in references
        )
        candidates = []
        unchecked = None
        for candidate_reference_q in references:
            try:
                q = self.motion.solve_ik(pose, candidate_reference_q, retry_current=False)
            except IKError:
                if not current_scheduled:
                    references.append(current)
                    current_scheduled = True
                continue
            if np.array_equal(candidate_reference_q, reference_q) or (
                unchecked is None and np.array_equal(candidate_reference_q, current)
            ):
                unchecked = q
            if checker.is_collision_free(q):
                candidates.append(q)
        return sorted(candidates, key=lambda q: np.linalg.norm(q - reference_q)), unchecked

    def build_stages(self, starts: np.ndarray, goals: np.ndarray) -> list[PlannedStage]:
        """Compute ordered targets from (cube_count, 3) poses; defer live paths."""
        self._starts = np.array(starts, dtype=float, copy=True)
        # Keep heuristic IK near the measured posture after joint-reference changes.
        state = self.robot.state
        reference = state.snapshot().qpos[:7].copy()
        reference_q = (
            self.simulator.model.qpos0[state.qpos_indices[:7]].copy()
            if self.robot.robot_type == "forte" and self.method == "sampling"
            else reference
        )
        if self.recipe.euler_xyz_degrees is None:
            rotation = self.simulator.data.site_xmat[state.site_id(self.recipe.frame)].reshape(3, 3)
            rotation = rotation.copy()
        else:
            rotation = Rotation.from_euler("xyz", self.recipe.euler_xyz_degrees, degrees=True)
        recipes = self.recipe.stages
        if self.method == "sampling":
            recipes = tuple(stage for stage in recipes if stage.name in self.SAMPLING_STAGES)
        stages = []
        for index in np.argsort(goals[:, 2]):
            checker = (
                self.collision_checker(int(index), "pick")
                if self.robot.robot_type == "forte" and self.method == "heuristic"
                else None
            )
            for stage in recipes:
                pose = self.stage_pose(stage, int(index), starts, goals, rotation)
                # Independent Forte poses share the same posture reference so a
                # pickup-side null-space drift cannot carry into place or the next cube.
                if checker is not None and stage.name in ("close", "release"):
                    # These gripper-only stages reuse the preceding pick/place arm pose.
                    reference_q = stages[-1].waypoints[-1]
                elif checker is not None:
                    candidates, unchecked = self.forte_heuristic_ik_candidates(
                        pose, reference, checker
                    )
                    # Let execution report a blocked pose as an expected route
                    # failure, so collectors can save it and continue to the next episode.
                    if candidates:
                        reference_q = candidates[0]
                    elif unchecked is not None:
                        reference_q = unchecked
                    else:
                        raise IKError(f"Unreachable IK pose for {self.robot.name}/grasp")
                else:
                    reference_q = self.motion.solve_ik(pose, reference_q)
                stages.append(
                    PlannedStage(
                        f"cube{index}:{stage.name}",
                        (reference_q.copy(),),
                        self.gripper.target_for_mode(stage.gripper_mode),
                        stage,
                        int(index),
                        12.0 if self.method == "sampling" else None,
                    )
                )
        return stages

    def stage_ratios(self, stage: PlannedStage) -> tuple[MotionRatio, MotionRatio]:
        """Use stage ratio pairs when supplied, otherwise inherit recipe defaults."""
        return (
            self.recipe.arm if stage.recipe.arm is None else stage.recipe.arm,
            self.recipe.gripper if stage.recipe.gripper is None else stage.recipe.gripper,
        )

    def group_stages(self, starts: np.ndarray, goals: np.ndarray) -> list[PlannedStage]:
        """Retain all recipe poses while executing five semantic stages per cube."""
        all_stages = self.build_stages(starts, goals)
        grouped = []
        count = len(self.recipe.stages)

        def motion(stages: dict[str, PlannedStage], final: str, *transit: str) -> PlannedStage:
            members = [stages[name] for name in (*transit, final)]
            ratios = [self.stage_ratios(member) for member in members]
            # A merged continuous motion uses the strictest member ratios.
            arm: MotionRatio = (
                min(pair[0][0] for pair in ratios),
                min(pair[0][1] for pair in ratios),
            )
            gripper: MotionRatio = (
                min(pair[1][0] for pair in ratios),
                min(pair[1][1] for pair in ratios),
            )
            recipe = stages[final].recipe.model_copy(update={"arm": arm, "gripper": gripper})
            return replace(
                stages[final],
                recipe=recipe,
                waypoints=tuple(point for member in members for point in member.waypoints),
            )

        for offset in range(0, len(all_stages), count):
            stages = {stage.recipe.name: stage for stage in all_stages[offset : offset + count]}

            grouped.extend(
                (
                    motion(stages, "pick", "above_pick"),
                    stages["close"],
                    motion(stages, "place", "lift", "above_place"),
                    stages["release"],
                    stages["retract"],
                )
            )
        return grouped

    def heuristic_trajectory(
        self, stage: PlannedStage
    ) -> tuple[JointTrajectory | None, str | None]:
        reason = self._grasp_failure(stage, started=False)
        if reason is not None:
            return None, reason
        target = self.robot.target
        if target is None:
            raise RuntimeError("Trajectory expert requires a control target")
        arm_path = np.vstack(
            (
                target.position[:7],
                *stage.waypoints,
            )
        )
        arm_ratio, gripper_ratio = self.stage_ratios(stage)
        if stage.recipe.name in ("close", "release") or (
            self.robot.robot_type != "forte" and len(stage.waypoints) == 1
        ):
            if self.robot.robot_type == "forte" and stage.recipe.name in ("close", "release"):
                arm_path = np.repeat(target.position[None, :7], 2, axis=0)
            return self.motion.trajectory(
                arm_path,
                stage.gripper_target,
                arm_ratio=arm_ratio,
                gripper_ratio=gripper_ratio,
            ), None
        checker = self.collision_checker(stage.cube_index, stage.recipe.name)
        if self.robot.robot_type == "forte":
            arm_path = self.forte_heuristic_orientation_path(arm_path)
        trajectory = self.motion.trajectory(
            arm_path,
            stage.gripper_target,
            checker,
            arm_ratio=arm_ratio,
            gripper_ratio=gripper_ratio,
        )
        if trajectory is None and self.robot.robot_type == "forte":
            trajectory = self.forte_heuristic_branch_path(stage, target.position[:7], checker)
        if trajectory is None:
            return None, f"No checked heuristic route for {stage.name}"
        return trajectory, None

    def forte_heuristic_orientation_path(self, path):
        """Keep the grasp orientation between recipe poses near its Cartesian interpolation."""
        model = self.simulator.model
        data = mujoco.MjData(model)
        mujoco.mj_copyData(data, model, self.simulator.data)
        slots = self.robot.state.qpos_indices[:7]
        site = self.robot.state.site_id("grasp")
        refined = [path[0]]
        for start, end in zip(path[:-1], path[1:], strict=True):
            positions, rotations = [], []
            for q in (start, end):
                data.qpos[slots] = q
                mujoco.mj_kinematics(model, data)
                positions.append(data.site_xpos[site].copy())
                rotations.append(data.site_xmat[site].reshape(3, 3).copy())
            interpolation = Slerp([0, 1], Rotation.from_matrix(rotations))
            for fraction in (1 / 3, 2 / 3):
                pose = Transform(
                    rotation=interpolation(fraction),
                    translation=positions[0] + fraction * (positions[1] - positions[0]),
                )
                reference_q = start + fraction * (end - start)
                refined.append(self.motion.solve_ik(pose, reference_q))
            refined.append(end)
        return np.asarray(refined)

    def forte_heuristic_branch_path(self, stage, start, checker):
        """Connect recipe poses through a bounded set of checked IK branches."""
        model = self.simulator.model
        data = mujoco.MjData(model)
        mujoco.mj_copyData(data, model, self.simulator.data)
        slots = self.robot.state.qpos_indices[:7]
        site = self.robot.state.site_id("grasp")
        routes = [(start,)]
        for waypoint in stage.waypoints:
            data.qpos[slots] = waypoint
            mujoco.mj_kinematics(model, data)
            pose = Transform(
                rotation=data.site_xmat[site].reshape(3, 3).copy(),
                translation=data.site_xpos[site].copy(),
            )
            candidates, _ = self.forte_heuristic_ik_candidates(pose, start, checker)
            routes = [
                (*route, candidate)
                for route in routes
                for candidate in candidates
                if checker.is_path_collision_free(route[-1], candidate)
            ]
            if not routes:
                return None
            routes.sort(key=lambda route: np.linalg.norm(np.diff(route, axis=0), axis=1).sum())
            routes = routes[:5]
        arm_ratio, gripper_ratio = self.stage_ratios(stage)
        for route in routes:
            try:
                path = self.forte_heuristic_orientation_path(np.asarray(route))
            except IKError:
                continue
            trajectory = self.motion.trajectory(
                path,
                stage.gripper_target,
                checker,
                arm_ratio=arm_ratio,
                gripper_ratio=gripper_ratio,
            )
            if trajectory is not None:
                return trajectory
        return None
