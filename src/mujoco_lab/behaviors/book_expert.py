"""Plan and execute centered book insertion with physical carry checks."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from mujoco_lab.behaviors.book import BookTask, center_grasp_pose
from mujoco_lab.behaviors.bookshelf_recipe import STAGE_ORDER, BookshelfRecipe
from mujoco_lab.behaviors.expert import Expert
from mujoco_lab.behaviors.trajectory import TrajectoryExecution
from mujoco_lab.control import ControlTarget
from mujoco_lab.control.trajectory import JointTrajectory
from mujoco_lab.planning import PathPlanner
from mujoco_lab.planning.collision.manipulation import ManipulationCollisionChecker
from mujoco_lab.planning.motion import MotionPlanner, MotionRequest, grasp_pose
from mujoco_lab.state import IKError
from mujoco_lab.utils import Transform


class BookInsertionExpert(Expert):
    """Plan each book stage and execute it with physical readiness checks."""

    STAGES = STAGE_ORDER
    CARRY = frozenset(("lift", "preinsert", "insert", "lower"))

    def __init__(self, task: BookTask, *, recipe: BookshelfRecipe, planner: PathPlanner) -> None:
        super().__init__()
        if planner is None:
            raise ValueError("Book insertion requires an explicit path planner")
        self.task = task
        self.simulator, self.robot = task.simulator, task.robot
        robot = self.robot
        if robot.gripper is None or robot.controller is None or robot.target is None:
            raise ValueError("Trajectory expert requires a joint controller and gripper")
        if robot.target.position.shape != (7,) or robot.control_joint_names != tuple(
            robot.state.joint_names[:7]
        ):
            raise ValueError("Trajectory expert requires a seven-joint target controller")

        self.recipe = recipe
        self.motion = MotionPlanner(self.robot, planner)
        limits = self.robot.gripper.get_control_limits()
        self.open, self.closed = float(limits[1]), float(limits[0])
        self.reset()

    def reset(self, initial_obs: Any = None, info: dict[str, Any] | None = None) -> None:
        self.motion.reset()
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
        self._lost_grasp_since = None

    def _hold_action(self):
        return np.r_[self.robot.target.position[:7], self.robot.gripper.get_target()]

    def act(self, obs: Any = None) -> np.ndarray:
        if self.failed or self.stage >= len(self.STAGES):
            return self._hold_action()
        reason = self._execution_failure()
        if reason is not None:
            self.failed, self.failure_reason = True, reason
            return self._hold_action()
        now = self.simulator.data.time
        current = self.robot.state.snapshot().qpos[:7]
        if self.execution.trajectory is None:
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
        action, complete = self.execution.sample(now, current, self.recipe.waypoint_tolerance)
        trajectory = cast(JointTrajectory, self.execution.trajectory)
        reason = self._execution_failure()
        if reason is not None:
            self.failed, self.failure_reason = True, reason
            return action
        if self._advance_stage(current, path_complete=complete):
            return action
        if now - self.execution.start_time > trajectory.duration + self.recipe.stage_timeout_s:
            self.failed, self.failure_reason = True, f"Timed out executing {self.get_stage_name()}"
        return action

    def update(self, simulator) -> None:
        if simulator is not self.simulator:
            raise ValueError("Expert is bound to a different simulator")
        action = self.act()
        self.robot.target = ControlTarget(action[:7])
        self.robot.gripper.set_target(float(action[7]))
        success = self.task.status().released_stable
        if self.failed or (self.stage >= len(self.STAGES) and success):
            simulator.stop()

    def get_stage_name(self) -> str:
        return self.STAGES[self.stage] if self.stage < len(self.STAGES) else "settle"

    def _execution_failure(self) -> str | None:
        phase, now = self.get_stage_name(), self.simulator.data.time
        if phase in self.CARRY and not self.task.has_grasp():
            if self.execution.trajectory is None:
                return f"No two-pad physical grasp for {phase}"
            if self._lost_grasp_since is None:
                self._lost_grasp_since = now
            elif now - self._lost_grasp_since > self.recipe.lost_grasp_grace_s:
                return f"Lost two-pad grasp during {phase}"
        else:
            self._lost_grasp_since = None
        return None

    def _advance_stage(self, current, *, path_complete: bool) -> bool:
        phase, now = self.get_stage_name(), self.simulator.data.time
        trajectory = cast(JointTrajectory, self.execution.trajectory)
        reached = np.max(np.abs(current - trajectory.path[-1, :7])) < self.recipe.arm_tolerance
        ready = True
        if phase == "close":
            ready = self.task.has_grasp()
        elif phase == "lift":
            ready = (
                self.simulator.data.xpos[self.task.body, 2]
                > self.task.start_center[2] + self.recipe.min_lift_height_m
            )
        elif phase == "release":
            ready = not self.task.status().touching_robot
        if (
            path_complete
            and reached
            and ready
            and now - self.execution.start_time >= trajectory.duration + self.recipe.stage_dwell_s
        ):
            self.stage += 1
            self.execution.trajectory = None
            return True
        return False

    def motion_request(self, phase: str, pick_pose: Transform) -> MotionRequest:
        """Own book stage meanings, poses, gripper commands, and contact policy."""
        stage = self.recipe.stages[self.STAGES.index(phase)]
        if phase in ("close", "release"):
            mode = "hold"
        elif phase in ("pick", "lift", "insert", "lower", "retract"):
            mode = "cartesian"
        else:
            mode = "sampling"
        limits = self.robot.gripper.get_control_limits()
        grip = limits[1] if phase in ("approach", "pick", "release", "retract") else limits[0]
        return MotionRequest(
            name=phase,
            mode=mode,
            waypoints=(self.destination(phase, pick_pose),),
            gripper_target=float(grip),
            wrap_angles=True,
            constraints=self.motion.stage_constraints(
                self.recipe.arm.override(stage.arm),
                self.recipe.gripper.override(stage.gripper),
            ),
            checker=None if mode == "hold" else self.collision_checker(phase),
        )

    def collision_checker(self, phase: str) -> ManipulationCollisionChecker:
        mapped = {
            "approach": "above_pick",
            "preinsert": "place",
            "insert": "place",
            "lower": "place",
        }.get(phase, phase)
        bounds = self.robot.state.get_joint_limits(list(range(7)))
        current = self.robot.state.snapshot().qpos[:7]
        for index in range(7):
            if not np.isfinite(bounds[index]).all():
                bounds[index] = [current[index] - 2 * np.pi, current[index] + 2 * np.pi]
        return ManipulationCollisionChecker(
            self.robot,
            f"book:{mapped}",
            object_body="book",
            target_site="book_target",
            bounds=bounds,
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
