"""Execute generated cube stacking stages as physical actions."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from mujoco_lab.control.target import ControlTarget
from mujoco_lab.simulation import Simulator
from mujoco_lab.tasks.cube_stack import CubeStackTask
from mujoco_lab.tasks.cube_stack_motion import (
    BaseCubeStackMotionGenerator,
    PlannedStage,
    cube_touches_robot,
    has_physical_grasp,
)
from mujoco_lab.tasks.expert import Expert


class CubeStackExpert(Expert):
    """Execute generated Panda or Forte stages as physical actions.

    ``act`` never applies its action. ``update`` adapts it to Simulator's
    target-updater callback for direct simulation; collectors call ``act`` and
    pass the returned action through their environment instead. The caller
    must bind a seven-joint target controller and gripper before construction.
    """

    def __init__(
        self,
        task: CubeStackTask,
        generator: BaseCubeStackMotionGenerator,
    ):
        super().__init__()
        if generator.task is not task:
            raise ValueError("Motion generator must use the expert's task")
        self.task = task
        self.generator = generator
        self.simulator = task.simulator
        self.cubes = task.cubes
        self.robot = generator.robot
        if self.robot.gripper is None:
            raise ValueError("CubeStackExpert requires a robot gripper")
        if (
            self.robot.controller is None
            or self.robot.target is None
            or self.robot.target.position.shape != (7,)
            or self.robot.control_joint_names != tuple(self.robot.state.joint_names[:7])
        ):
            raise ValueError("CubeStackExpert requires a configured seven-joint target controller")
        self.recipe = generator.recipe
        self.reset()

    def _reset_progress(self) -> None:
        self.stage = 0
        self._trajectory = None
        self._path_vertex = 1
        self._trajectory_elapsed = 0.0
        self._trajectory_last_time = self.simulator.data.time
        self._stage_start_time = self.simulator.data.time
        self._transport_peak = float("-inf")
        self.failed = False
        self.failure_reason = None

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
        self._plan = self.generator.generate(self.starts, self.goals)
        self._reset_progress()

    def get_stage_name(self):
        return self._plan[self.stage].name if self.stage < len(self._plan) else "settle"

    def act(self, obs: Any = None) -> np.ndarray:
        """Return an eight-value arm/gripper target without applying it.

        ``obs`` is unused because this expert reads its bound simulator. One
        call represents one physics tick and advances only planning progress.
        """
        if self.failed or self.stage >= len(self._plan):
            return self._hold_action()
        return self._trajectory_action()

    def _hold_action(self) -> np.ndarray:
        return np.r_[self.robot.target.position[:7], self.robot.gripper.get_target()]

    def update(self, simulator: Simulator) -> None:
        """Apply one expert action for direct Simulator use when explicitly attached."""
        if simulator is not self.simulator:
            raise ValueError("Expert is bound to a different simulator")
        action = self.act()
        self.robot.target = ControlTarget(action[:7])
        self.robot.gripper.set_target(float(action[7]))
        status = self.task.status()
        if self.failed or (self.stage >= len(self._plan) and status.released_stable_stack):
            simulator.stop()

    def _advance_stage(
        self, stage: PlannedStage, current: np.ndarray, finger_target: float, *, path_complete: bool
    ) -> bool:
        """Advance after time, reach, and requested physical checks."""
        command = stage.command
        close_enough = np.max(np.abs(command[:7] - current[:7])) < self.recipe.arm_tolerance
        # A cube can prevent the fingers from reaching the commanded closing gap.
        finger_close = abs(command[7] - finger_target) < self.recipe.gripper_tolerance
        physical_ready = True
        if stage.recipe.require_grasp:
            physical_ready = has_physical_grasp(self.robot, stage.cube_index)
        if stage.recipe.name == "release":
            physical_ready = physical_ready and not cube_touches_robot(self.robot, stage.cube_index)
        if stage.recipe.min_lift_height_m is not None:
            physical_ready = physical_ready and (
                self.simulator.data.body(f"cube{stage.cube_index}/object_0").xpos[2]
                > self.starts[stage.cube_index, 2] + stage.recipe.min_lift_height_m
            )
        if (
            path_complete
            and close_enough
            and finger_close
            and physical_ready
            and self.simulator.data.time - self._stage_start_time >= stage.recipe.min_duration_s
        ):
            self.stage += 1
            self._trajectory = None
            return True
        return False

    def _trajectory_action(self) -> np.ndarray:
        """Follow the current stage's checked timed path."""
        stage = self._plan[self.stage]
        now = self.simulator.data.time
        current = self.robot.state.snapshot().qpos
        if self._trajectory is None:
            self._trajectory, failure_reason = self.generator.make_trajectory(stage)
            if self._trajectory is None:
                self.failed = True
                self.failure_reason = failure_reason
                return self._hold_action()
            self._path_vertex = 1
            self._trajectory_elapsed = 0.0
            self._trajectory_last_time = now
            self._stage_start_time = now
            self._transport_peak = float("-inf")

        lift_requirement = self.generator.min_transport_lift(stage)
        if lift_requirement is not None:
            height = self.simulator.data.body(f"cube{stage.cube_index}/object_0").xpos[2]
            self._transport_peak = max(self._transport_peak, float(height))

        elapsed = self._trajectory_elapsed + max(0.0, now - self._trajectory_last_time)
        self._trajectory_last_time = now
        if not self._trajectory.smooth:
            # The exact polyline fallback waits for measured arrival at each corner.
            while self._path_vertex < len(self._trajectory.path):
                waypoint_time = self._trajectory.waypoint_times[self._path_vertex]
                if (
                    self._trajectory_elapsed < waypoint_time
                    or np.max(np.abs(self._trajectory.path[self._path_vertex, :7] - current[:7]))
                    >= self.recipe.arm_tolerance
                ):
                    break
                self._path_vertex += 1
            if self._path_vertex < len(self._trajectory.path):
                elapsed = min(elapsed, self._trajectory.waypoint_times[self._path_vertex])
        self._trajectory_elapsed = min(elapsed, self._trajectory.duration)
        if self._trajectory.smooth:
            self._path_vertex = min(
                len(self._trajectory.path),
                int(
                    np.searchsorted(
                        self._trajectory.waypoint_times, self._trajectory_elapsed, side="right"
                    )
                ),
            )
        action = self._trajectory.sample(self._trajectory_elapsed).position
        path_complete = (
            self._trajectory_elapsed >= self._trajectory.duration
            if self._trajectory.smooth
            else self._path_vertex == len(self._trajectory.path)
        )
        if (
            lift_requirement is not None
            and path_complete
            and self._transport_peak <= self.starts[stage.cube_index, 2] + lift_requirement
        ):
            self.failed = True
            self.failure_reason = f"Insufficient physical lift during {stage.name}"
            return action
        if self._advance_stage(stage, current, float(action[7]), path_complete=path_complete):
            return action
        if (
            stage.settle_timeout is not None
            and now - self._stage_start_time > self._trajectory.duration + stage.settle_timeout
        ):
            self.failed = True
            self.failure_reason = f"Timed out executing {stage.name}"
        return action
