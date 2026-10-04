"""Track timed trajectory progress and pause at unchecked waypoints."""

from typing import cast

import numpy as np

from mujoco_lab.control.trajectory import JointTrajectory


class TrajectoryExecution:
    """Advance a trajectory without commanding a robot or stepping physics."""

    def __init__(self, now: float):
        self.trajectory: JointTrajectory | None = None
        self.vertex = 1
        self.elapsed = 0.0
        self.last_time = now
        self.start_time = now

    def start(self, trajectory: JointTrajectory, now: float) -> None:
        self.trajectory = trajectory
        self.vertex = 1
        self.elapsed = 0.0
        self.last_time = self.start_time = now

    def sample(
        self, now: float, current: np.ndarray, arm_tolerance: float
    ) -> tuple[np.ndarray, bool]:
        trajectory = cast(JointTrajectory, self.trajectory)
        elapsed = self.elapsed + max(0.0, now - self.last_time)
        self.last_time = now
        if not trajectory.smooth:
            while self.vertex < len(trajectory.path):
                waypoint_time = trajectory.waypoint_times[self.vertex]
                if (
                    self.elapsed < waypoint_time
                    or np.max(np.abs(trajectory.path[self.vertex, :7] - current)) >= arm_tolerance
                ):
                    break
                self.vertex += 1
            if self.vertex < len(trajectory.path):
                elapsed = min(elapsed, trajectory.waypoint_times[self.vertex])
        self.elapsed = min(elapsed, trajectory.duration)
        if trajectory.smooth:
            self.vertex = min(
                len(trajectory.path),
                int(np.searchsorted(trajectory.waypoint_times, self.elapsed, side="right")),
            )
        action = trajectory.sample(self.elapsed).position
        complete = (
            self.elapsed >= trajectory.duration
            if trajectory.smooth
            else self.vertex == len(trajectory.path)
        )
        return action, complete
