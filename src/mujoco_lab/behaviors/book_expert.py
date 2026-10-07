"""Plan and execute centered book insertion with physical carry checks."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from mujoco_lab.behaviors.book import BookTask, center_grasp_pose
from mujoco_lab.behaviors.bookshelf_recipe import STAGE_ORDER, BookshelfRecipe
from mujoco_lab.behaviors.expert import Expert
from mujoco_lab.behaviors.grasp import GraspMonitor
from mujoco_lab.behaviors.trajectory import TrajectoryExecution
from mujoco_lab.control import ControlTarget
from mujoco_lab.planning import PlannerConfig
from mujoco_lab.planning.collision.manipulation import ManipulationCollisionChecker
from mujoco_lab.planning.motion import MotionPlanner, MotionRequest, grasp_pose
from mujoco_lab.state import IKError
from mujoco_lab.utils import Transform


class BookInsertionExpert(Expert):
    """Execute sampling or heuristic book stages with physical readiness checks.

    Heuristic execution follows Cartesian IK paths without a sampling planner.
    Both methods preserve stage-specific collision and physical grasp checks.
    """

    STAGES = STAGE_ORDER

    def __init__(
        self,
        task: BookTask,
        *,
        recipe: BookshelfRecipe,
        method: Literal["heuristic", "sampling"] = "sampling",
        planning: PlannerConfig | None = None,
    ) -> None:
        super().__init__()
        if method not in ("heuristic", "sampling"):
            raise ValueError(f"Unsupported book execution method: {method}")
        if method == "sampling" and planning is None:
            raise ValueError("Sampling book insertion requires an explicit planner configuration")
        if method == "heuristic" and planning is not None:
            raise ValueError("Heuristic execution does not use a planner configuration")
        self.method = method
        self.task = task
        self.simulator, self.robot = task.simulator, task.robot
        robot = self.robot
        if robot.gripper is None or robot.controller is None or robot.target is None:
            raise ValueError("Trajectory expert requires a joint controller and gripper")
        if robot.target.position.shape != (7,) or robot.control_joint_names != tuple(
            robot.state.joint_names[:7]
        ):
            raise ValueError("Trajectory expert requires a seven-joint target controller")

        self.gripper = robot.gripper
        self.recipe = recipe
        self._grasp = GraspMonitor(recipe.lost_grasp_grace_s)
        self.motion = MotionPlanner(self.robot, planning)
        self.reset()

    def reset(self, initial_obs: Any = None, info: dict[str, Any] | None = None) -> None:
        data = self.simulator.data
        self.pick_pose = center_grasp_pose(
            data.xpos[self.task.body].copy(),
            data.xmat[self.task.body].reshape(3, 3).copy(),
            self.task.depth,
        )
        self.stage = 0
        self.execution = TrajectoryExecution(data.time)
        self.failed = False
        self.failure_reason = None
        self._grasp.reset()

    def _hold_action(self):
        target = self.robot.target
        if target is None:
            raise RuntimeError("Trajectory expert requires a control target")
        return np.r_[target.position[:7], self.gripper.get_target()]

    def act(self, obs: Any = None, *, dt: float = 0.0) -> np.ndarray:
        if self.failed or self.stage >= len(self.STAGES):
            return self._hold_action()
        reason = self._execution_failure()
        if reason is not None:
            self.failed, self.failure_reason = True, reason
            return self._hold_action()
        now = self.simulator.data.time
        current = self.robot.state.snapshot().qpos[:7]
        trajectory = self.execution.trajectory
        if trajectory is None:
            try:
                trajectory, reason = self.motion.make_trajectory(
                    self.motion_request(self.get_stage_name(), self.pick_pose)
                )
            except IKError as error:
                trajectory, reason = None, str(error)
            if trajectory is None:
                self.failed, self.failure_reason = True, reason
                return self._hold_action()
            self.execution.start(trajectory, now)
        next_act, complete = self.execution.sample(
            now, current, self.recipe.waypoint_tolerance, dt=dt
        )
        if self._advance_stage(current, path_complete=complete):
            return next_act
        if now - self.execution.start_time > trajectory.duration + self.recipe.stage_timeout_s:
            self.failed, self.failure_reason = True, f"Timed out executing {self.get_stage_name()}"
        return next_act

    def update(self, simulator) -> None:
        if simulator is not self.simulator:
            raise ValueError("Expert is bound to a different simulator")
        action = self.act()
        self.robot.update_target(ControlTarget(action[:7]), float(action[7]))
        success = self.task.status().released_stable
        if self.failed or (self.stage >= len(self.STAGES) and success):
            simulator.stop()

    def get_stage_name(self) -> str:
        return self.STAGES[self.stage] if self.stage < len(self.STAGES) else "settle"

    def _execution_failure(self) -> str | None:
        if self.stage >= len(self.recipe.stages):
            self._grasp.reset()
            return None
        stage = self.recipe.stages[self.stage]
        return self._grasp.failure(
            mode=stage.gripper_mode,
            hold=self._motion_mode(stage.name) == "hold",
            started=self.execution.trajectory is not None,
            grasped=self.task.has_grasp(),
            now=self.simulator.data.time,
            stage_name=stage.name,
        )

    def _advance_stage(self, current, *, path_complete: bool) -> bool:
        phase, now = self.get_stage_name(), self.simulator.data.time
        trajectory = self.execution.trajectory
        if trajectory is None:
            raise RuntimeError("Trajectory expert has no active trajectory")
        reached = np.max(np.abs(current - trajectory.path[-1, :7])) < self.recipe.arm_tolerance
        ready = self._grasp.ready(
            self.recipe.stages[self.stage].gripper_mode, self.task.has_grasp()
        )
        if phase == "lift":
            ready = ready and (
                self.simulator.data.xpos[self.task.body, 2]
                > self.task.start_center[2] + self.recipe.min_lift_height_m
            )
        elif phase == "release":
            ready = ready and not self.task.status().touching_robot
        dwell = self.recipe.stage_dwell_s
        if self.method == "heuristic" and phase not in ("close", "release"):
            dwell = 0.0
        if (
            path_complete
            and reached
            and ready
            and now - self.execution.start_time >= trajectory.duration + dwell
        ):
            self.stage += 1
            self.execution.trajectory = None
            self._grasp.reset()
            return True
        return False

    def _motion_mode(self, phase: str) -> Literal["hold", "cartesian", "sampling"]:
        if phase in ("close", "release"):
            return "hold"
        if self.method == "heuristic" or phase in ("pick", "lift", "insert", "lower", "retract"):
            return "cartesian"
        return "sampling"

    def motion_request(self, phase: str, pick_pose: Transform) -> MotionRequest:
        """Own book stage meanings, poses, gripper commands, and contact policy."""
        stage = self.recipe.stages[self.STAGES.index(phase)]
        mode = self._motion_mode(phase)
        return MotionRequest(
            name=phase,
            mode=mode,
            waypoints=(self.destination(phase, pick_pose),),
            gripper_target=self.gripper.target_for_mode(stage.gripper_mode),
            shape_preserving=self.method == "heuristic" and phase == "approach",
            arm_ratio=self.recipe.arm if stage.arm is None else stage.arm,
            gripper_ratio=self.recipe.gripper if stage.gripper is None else stage.gripper,
            checker=None if mode == "hold" else self.collision_checker(phase),
        )

    def collision_checker(self, phase: str) -> ManipulationCollisionChecker:
        mapped = {
            "approach": "above_pick",
            "preinsert": "place",
            "insert": "place",
            "lower": "place",
        }.get(phase, phase)
        return ManipulationCollisionChecker(
            self.robot,
            f"book:{mapped}",
            object_body="book",
            target_site="book_target",
            edge_resolution=0.025,
            support_geom="large_shelf/shelf_1" if mapped == "place" else None,
            departure_support_geom="table/box" if mapped == "place" else None,
            grasp_penetration=0.003,
            support_xy_tolerance=0.02,
        )

    def destination(self, phase: str, pick_pose: Transform) -> Transform:
        if phase in ("approach", "pick", "close"):
            position = pick_pose.as_translation().copy()
            if phase == "approach":
                position -= (
                    pick_pose.as_rotation().as_matrix()[:, 2] * self.recipe.approach_distance_m
                )
            return Transform(rotation=pick_pose.as_rotation(), translation=position)
        measured = grasp_pose(self.robot)
        if phase == "lift":
            return Transform(
                rotation=measured.as_rotation(),
                translation=measured.as_translation() + [0, 0, self.recipe.lift_height_m],
            )
        data = self.simulator.data
        target_rotation = data.site_xmat[self.task.target].reshape(3, 3).copy()
        target_center = data.site_xpos[self.task.target].copy()
        if phase in ("preinsert", "insert", "lower"):
            book_rotation = data.xmat[self.task.body].reshape(3, 3)
            # T_world_site = T_world_book_goal * inverse(T_site_book_measured).
            local_site = book_rotation.T @ (measured.as_translation() - data.xpos[self.task.body])
            local_rotation = book_rotation.T @ measured.as_rotation().as_matrix()
            if phase != "lower":
                target_center[2] += self.recipe.insertion_clearance_m
            if phase == "preinsert":
                # Keep the entire book in front of the shelf entrance.
                target_center[1] = (
                    self.recipe.shelf_front_y_m
                    - self.task.depth / 2
                    - self.recipe.preinsert_clearance_m
                )
            return Transform(
                rotation=target_rotation @ local_rotation,
                translation=target_center + target_rotation @ local_site,
            )
        if phase == "retract":
            position = measured.as_translation().copy()
            position -= target_rotation[:, 0] * self.recipe.retract_distance_m
            return Transform(rotation=measured.as_rotation(), translation=position)
        return measured
