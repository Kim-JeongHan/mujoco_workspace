"""Track timed trajectory progress and pause at unchecked waypoints."""

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
        self, now: float, current: np.ndarray, arm_tolerance: float, *, dt: float = 0.0
    ) -> tuple[np.ndarray, bool]:
        """Return a target dt seconds ahead and completion at the current time.

        Previewing never advances execution or crosses an unchecked stop waypoint.
        """
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("dt must be finite and nonnegative")
        trajectory = self.trajectory
        if trajectory is None:
            raise RuntimeError("Cannot sample a trajectory before it is started")
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
        sample_time = self.elapsed + dt
        if not trajectory.smooth and self.vertex < len(trajectory.path):
            sample_time = min(sample_time, trajectory.waypoint_times[self.vertex])
        next_act = trajectory.sample(sample_time).position
        complete = (
            self.elapsed >= trajectory.duration
            if trajectory.smooth
            else self.vertex == len(trajectory.path)
        )
        return next_act, complete
