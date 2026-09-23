"""Generate cube stack stages and resolve their motion paths in the live scene."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from itertools import pairwise
from math import ceil

import numpy as np
from scipy.spatial.transform import Rotation

from mujoco_lab.control.trajectory import JointTrajectory
from mujoco_lab.planning import (
    PathPlanner,
    RRTConnectConfig,
)
from mujoco_lab.planning.collision.manipulation import CubeStackCollisionChecker
from mujoco_lab.tasks.cube_stack import CubeStackTask
from mujoco_lab.tasks.cube_stack_recipe import StageRecipe, load_recipe
from mujoco_lab.utils import Transform

START_STAGES = frozenset(("above_pick", "pick", "close", "lift"))


def default_planning() -> RRTConnectConfig:
    """Return the tuned RRT-Connect configuration used by sampling motion."""
    return RRTConnectConfig(max_iterations=500, step_size=0.2, goal_tolerance=0.04, seed=7)


@dataclass(frozen=True)
class PlannedStage:
    """A precomputed stage target whose path is resolved at stage entry."""

    name: str
    command: np.ndarray
    recipe: StageRecipe
    cube_index: int
    settle_timeout: float | None = None
    transit_commands: tuple[np.ndarray, ...] = ()


def has_physical_grasp(robot, cube_index: int) -> bool:
    """Return whether both physical finger pads touch the active cube."""
    model, data = robot.model, robot.data
    cube_geom = model.geom(f"cube{cube_index}/object_0").id
    sides = set()
    for contact in data.contact:
        if contact.geom1 == cube_geom:
            other = int(contact.geom2)
        elif contact.geom2 == cube_geom:
            other = int(contact.geom1)
        else:
            continue
        if robot.robot_type == "forte":
            for side in ("left", "right"):
                if model.geom(other).name == f"{robot.name}/gripper_{side}_pad":
                    sides.add(side)
        else:
            body = model.body(int(model.geom_bodyid[other])).name
            for side in ("left", "right"):
                if body == f"{robot.name}/panda_{side}finger":
                    sides.add(side)
    return sides == {"left", "right"}


def cube_touches_robot(robot, cube_index: int) -> bool:
    """Return whether the active cube contacts any geom owned by this robot."""
    model, data = robot.model, robot.data
    cube_geom = model.geom(f"cube{cube_index}/object_0").id
    for contact in data.contact:
        if contact.geom1 == cube_geom:
            other = int(contact.geom2)
        elif contact.geom2 == cube_geom:
            other = int(contact.geom1)
        else:
            continue
        if model.geom(other).name.startswith(robot.prefix):
            return True
    return False


class BaseCubeStackMotionGenerator(ABC):
    """Build cube stack targets and package live arm paths as timed trajectories."""

    def __init__(self, task: CubeStackTask) -> None:
        self.task = task
        self.simulator = task.simulator
        if len(self.simulator.robots) != 1:
            raise ValueError("Cube stacking execution requires one robot")
        self.robot = next(iter(self.simulator.robots.values()))
        if self.robot.robot_type not in ("panda", "forte"):
            raise ValueError("Cube stacking execution requires panda or forte")
        self.recipe = load_recipe(self.robot.robot_type)

    def stage_recipes(self) -> tuple[StageRecipe, ...]:
        """Select ordered task stages for this motion strategy."""
        return self.recipe.stages

    def settle_timeout(self, stage: StageRecipe) -> float | None:
        """Return an optional deadline after a stage trajectory finishes."""
        return None

    def min_transport_lift(self, stage: PlannedStage) -> float | None:
        """Return the measured lift required during this stage, if any."""
        return None

    def _stage_pose(
        self, stage: StageRecipe, index: int, reference: np.ndarray, rotation: Rotation | np.ndarray
    ) -> Transform:
        """Align pick-side targets with measured cube yaw; keep place targets nominal."""
        if stage.name not in START_STAGES:
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
        return Transform(
            rotation=yaw_rotation * nominal_rotation,
            translation=reference + yaw_rotation.apply(stage.offset_xyz_m),
        )

    def generate(self, starts: np.ndarray, goals: np.ndarray) -> list[PlannedStage]:
        """Compute ordered targets from (cube_count, 3) poses; defer live paths."""
        self._starts = np.array(starts, dtype=float, copy=True)
        # Keep Forte's tabletop IK on its reachable forward-facing branch.
        state = self.robot.state
        seed = (
            self.simulator.model.qpos0[state.qpos_indices[:7]].copy()
            if self.robot.robot_type == "forte"
            else state.snapshot().qpos[:7].copy()
        )
        if self.recipe.euler_xyz_degrees is None:
            rotation = self.simulator.data.site_xmat[state.site_id(self.recipe.frame)].reshape(3, 3)
            rotation = rotation.copy()
        else:
            rotation = Rotation.from_euler("xyz", self.recipe.euler_xyz_degrees, degrees=True)
        stages = []
        for index in np.argsort(goals[:, 2]):
            for stage in self.stage_recipes():
                reference = starts[index] if stage.name in START_STAGES else goals[index]
                pose = self._stage_pose(stage, int(index), reference, rotation)
                seed = state.solve_ik(pose, frame=self.recipe.frame, seed=seed)
                stages.append(
                    PlannedStage(
                        f"cube{index}:{stage.name}",
                        np.r_[seed, stage.gripper_target_m],
                        stage,
                        int(index),
                        self.settle_timeout(stage),
                    )
                )
        return stages

    @abstractmethod
    def make_trajectory(self, stage: PlannedStage) -> tuple[JointTrajectory | None, str | None]:
        """Resolve a stage path, returning a reason when expected checks fail."""

    def _trajectory(
        self,
        stage: PlannedStage,
        arm_path: np.ndarray,
        checker: CubeStackCollisionChecker | None = None,
    ) -> JointTrajectory | None:
        """Apply the stage's gripper target and velocity limits to an arm path."""
        gripper = np.full((len(arm_path), 1), stage.command[7])
        gripper[0, 0] = self.robot.gripper.get_target()
        path = np.column_stack((arm_path, gripper))
        arm_velocity = stage.recipe.arm_max_velocity or self.recipe.arm_max_velocity
        arm_acceleration = stage.recipe.arm_max_acceleration or self.recipe.arm_max_acceleration
        velocity = np.r_[np.full(7, arm_velocity), self.recipe.gripper_max_velocity]
        acceleration = np.r_[np.full(7, arm_acceleration), self.recipe.gripper_max_acceleration]
        trajectory = JointTrajectory(
            path,
            max_velocity=velocity,
            max_acceleration=acceleration,
        )
        if checker is not None and not self._smooth_path_collision_free(
            checker, trajectory, velocity[:7]
        ):
            if any(
                not checker.is_path_collision_free(start, goal)
                for start, goal in pairwise(arm_path)
            ):
                return None
            return JointTrajectory(path, velocity, acceleration, smooth=False)
        return trajectory

    @staticmethod
    def _smooth_path_collision_free(
        checker: CubeStackCollisionChecker, trajectory: JointTrajectory, velocity: np.ndarray
    ) -> bool:
        """Check dense chords of the curved arm path in the stage's frozen scene."""
        if trajectory.duration == 0:
            return checker.is_collision_free(trajectory.path[0, :7])
        interval = min(0.02, checker.edge_resolution / (2 * np.linalg.norm(velocity)))
        steps = max(1, ceil(trajectory.duration / interval))
        previous = trajectory.sample(0).position[:7]
        for elapsed in np.linspace(trajectory.duration / steps, trajectory.duration, steps):
            current = trajectory.sample(float(elapsed)).position[:7]
            if not checker.is_path_collision_free(previous, current):
                return False
            previous = current
        return True

    @staticmethod
    def _shortcut_path(path: np.ndarray, checker: CubeStackCollisionChecker) -> np.ndarray:
        """Keep the farthest collision-checked successor of each retained state."""
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

    def _collision_checker(self, stage: PlannedStage) -> CubeStackCollisionChecker:
        """Freeze one scene for stage path validation."""
        return CubeStackCollisionChecker(
            self.robot,
            stage.name,
            support_geom=(
                self.task._supports[stage.cube_index] if stage.recipe.name == "place" else None
            ),
            departure_support_geom=("table/box" if stage.recipe.name == "place" else None),
        )


class HeuristicCubeStackMotionGenerator(BaseCubeStackMotionGenerator):
    """Follow the recipe's explicit targets with direct joint-space paths."""

    def generate(self, starts: np.ndarray, goals: np.ndarray) -> list[PlannedStage]:
        """Retain all recipe poses while executing five semantic stages per cube."""
        all_stages = super().generate(starts, goals)
        grouped = []
        count = len(self.recipe.stages)

        def motion(stages: dict[str, PlannedStage], final: str, *transit: str) -> PlannedStage:
            members = [stages[name] for name in (*transit, final)]
            recipe = stages[final].recipe.model_copy(
                update={
                    "arm_max_velocity": min(
                        member.recipe.arm_max_velocity or self.recipe.arm_max_velocity
                        for member in members
                    ),
                    "arm_max_acceleration": min(
                        member.recipe.arm_max_acceleration or self.recipe.arm_max_acceleration
                        for member in members
                    ),
                    "require_grasp": any(member.recipe.require_grasp for member in members),
                }
            )
            return replace(
                stages[final],
                recipe=recipe,
                transit_commands=tuple(stages[name].command for name in transit),
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

    def min_transport_lift(self, stage: PlannedStage) -> float | None:
        if stage.recipe.name != "place":
            return None
        return next(
            recipe.min_lift_height_m for recipe in self.recipe.stages if recipe.name == "lift"
        )

    def make_trajectory(self, stage: PlannedStage) -> tuple[JointTrajectory | None, str | None]:
        arm_path = np.vstack(
            (
                self.robot.target.position[:7],
                *(command[:7] for command in stage.transit_commands),
                stage.command[:7],
            )
        )
        if not stage.transit_commands:
            return self._trajectory(stage, arm_path), None
        checker = self._collision_checker(stage)
        trajectory = self._trajectory(stage, arm_path, checker)
        if trajectory is None:
            return None, f"No checked heuristic route for {stage.name}"
        return trajectory, None


class SamplingCubeStackMotionGenerator(BaseCubeStackMotionGenerator):
    """Plan to pick and place poses with gripper and withdrawal actions."""

    _TASK_STAGES = frozenset(("pick", "close", "place", "release", "retract"))

    def __init__(
        self,
        task: CubeStackTask,
        *,
        planner: PathPlanner,
    ) -> None:
        super().__init__(task)
        self.planner = planner
        self._planning_epoch = 0

    def settle_timeout(self, stage: StageRecipe) -> float:
        return max(12.0, stage.min_duration_s + 4.0)

    def stage_recipes(self) -> tuple[StageRecipe, ...]:
        return tuple(stage for stage in self.recipe.stages if stage.name in self._TASK_STAGES)

    def min_transport_lift(self, stage: PlannedStage) -> float | None:
        return 0.04 if stage.recipe.name == "place" else None

    def generate(self, starts: np.ndarray, goals: np.ndarray) -> list[PlannedStage]:
        self._planning_epoch = 0
        return super().generate(starts, goals)

    def make_trajectory(self, stage: PlannedStage) -> tuple[JointTrajectory | None, str | None]:
        phase = stage.recipe.name
        checker = None
        if phase in ("close", "release"):
            arm_path = np.vstack((self.robot.target.position[:7], stage.command[:7]))
        else:
            if phase == "pick":
                actual = self.simulator.data.body(f"cube{stage.cube_index}/object_0").xpos
                if np.linalg.norm(actual - self._starts[stage.cube_index]) > 0.01:
                    reason = (
                        f"cube{stage.cube_index} moved more than 1 cm from its planned pick pose"
                    )
                    return None, reason
            if phase == "place" and not has_physical_grasp(self.robot, stage.cube_index):
                return None, f"No two-finger physical grasp for {stage.name}"
            checker = self._collision_checker(stage)
            if phase == "place":
                try:
                    clearance = self._clearance_target(0.06)
                except ValueError as error:
                    return None, f"No departure IK for {stage.name}: {error}"
                first_leg = self._plan_stage(stage, checker, destination_override=clearance)
                if first_leg is None:
                    return None, f"No {self.planner.name} departure route for {stage.name}"
                second_leg = self._plan_stage(stage, checker, start_override=clearance)
                if second_leg is None:
                    return None, f"No {self.planner.name} route for {stage.name}"
                first_leg = self._shortcut_path(first_leg, checker)
                second_leg = self._shortcut_path(second_leg, checker)
                arm_path = np.vstack((first_leg, second_leg[1:]))
            else:
                arm_path = self._plan_stage(stage, checker)
            if arm_path is None:
                return None, f"No {self.planner.name} route for {stage.name}"
            if phase != "place":
                arm_path = self._shortcut_path(arm_path, checker)
        trajectory = self._trajectory(stage, arm_path, checker)
        if trajectory is None:
            return None, f"No checked {self.planner.name} route for {stage.name}"
        return trajectory, None

    def _clearance_target(self, height: float) -> np.ndarray:
        """Solve a short upward departure from the measured grasp pose."""
        state = self.robot.state
        site = state.site_id(self.recipe.frame)
        pose = Transform(
            rotation=self.simulator.data.site_xmat[site].reshape(3, 3).copy(),
            translation=self.simulator.data.site_xpos[site].copy() + np.array((0.0, 0.0, height)),
        )
        return state.solve_ik(pose, frame=self.recipe.frame, seed=state.snapshot().qpos[:7])

    def _plan_stage(
        self,
        stage: PlannedStage,
        checker: CubeStackCollisionChecker,
        *,
        start_override: np.ndarray | None = None,
        destination_override: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Search the current scene for one joint-space stage transition."""
        measured = self.simulator.data.qpos[checker.qpos_indices].copy()
        start = measured if start_override is None else start_override
        destination = stage.command[:7] if destination_override is None else destination_override
        bounds = [tuple(row) for row in checker.bounds]
        seed = self.planner.seed
        stage_seed = seed + self._planning_epoch if seed is not None else None
        self._planning_epoch += 1
        vertices = self.planner.plan(start, destination, bounds, checker, seed=stage_seed)
        if vertices is None:
            return None
        vertices = np.asarray(vertices, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 7:
            raise RuntimeError("Planner returned an invalid arm path")
        if len(vertices) == 1:
            vertices = np.vstack((vertices, destination))
        if not np.allclose(vertices[0], start, atol=1e-9, rtol=0):
            raise RuntimeError("Planner path does not start at measured state")
        if not np.allclose(vertices[-1], destination, atol=1e-9, rtol=0):
            raise RuntimeError("Planner path does not end at stage target")
        # Reach the measured planner start without a command jump or unchecked edge.
        previous = self.robot.target.position[:7]
        if start_override is None and not np.array_equal(previous, start):
            if not checker.is_path_collision_free(previous, start):
                return None
            vertices = np.vstack((previous, vertices))
        return vertices


def CubeStackMotionGenerator(
    task: CubeStackTask,
    *,
    planner: PathPlanner | None = None,
) -> BaseCubeStackMotionGenerator:
    """Construct a heuristic generator or use the supplied sampling planner."""
    if planner is None:
        return HeuristicCubeStackMotionGenerator(task)
    return SamplingCubeStackMotionGenerator(task, planner=planner)
