"""Drive a checked joint-space waypoint path through a robot's native controller."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mujoco_lab.control import ControlTarget
from mujoco_lab.robot import Robot


class WaypointFollower:
    """Rate-limit commands on each planned edge and wait for measured convergence.

    The planner checks a static scene at sampled joint configurations. This
    follower does not turn that check into a continuous or dynamic guarantee.
    """

    def __init__(
        self,
        robot: Robot,
        joint_names: Sequence[str],
        waypoints: np.ndarray,
        *,
        max_steps: int = 6000,
        speed: float = 0.35,
        position_tolerance: float = 0.005,
        velocity_tolerance: float = 0.02,
        stop_on_finish: bool = False,
    ) -> None:
        controller = robot.controller
        if controller is None or robot.target is None:
            raise ValueError("Attach a joint controller before following waypoints")
        names = tuple(joint_names)
        path = np.asarray(waypoints, dtype=float)
        if (
            not names
            or len(set(names)) != len(names)
            or path.ndim != 2
            or path.shape[0] < 2
            or path.shape[1] != len(names)
            or not np.isfinite(path).all()
        ):
            raise ValueError("Waypoints need at least two finite rows in unique joint order")
        if (
            max_steps <= 0
            or not np.isfinite(speed)
            or speed <= 0
            or not np.isfinite(position_tolerance)
            or position_tolerance <= 0
            or not np.isfinite(velocity_tolerance)
            or velocity_tolerance <= 0
        ):
            raise ValueError("Step limit, speed, and convergence tolerances must be positive")
        controlled = controller.state.joint_names
        if not set(names).issubset(controlled):
            raise ValueError("Every planned joint needs a controlled actuator")
        if not np.allclose(
            path[0],
            robot.data.qpos[[robot.model.joint(robot.prefix + name).qposadr[0] for name in names]],
            atol=1e-6,
            rtol=0,
        ):
            raise ValueError("The first waypoint must match the robot's current pose")

        self.robot = robot
        self.joint_names = names
        self.waypoints = path.copy()
        self.qpos_indices = np.asarray(
            [robot.model.joint(robot.prefix + name).qposadr[0] for name in names], dtype=int
        )
        self.dof_indices = np.asarray(
            [robot.model.joint(robot.prefix + name).dofadr[0] for name in names], dtype=int
        )
        self.target_indices = np.asarray([controlled.index(name) for name in names], dtype=int)
        self.base_target = robot.target.position.copy()
        self.max_steps = max_steps
        self.speed = speed
        self.position_tolerance = position_tolerance
        self.velocity_tolerance = velocity_tolerance
        self.stop_on_finish = stop_on_finish
        self.steps = 0
        self.next_index = 1
        self.progress = 0.0
        self.complete = False
        self.failed = False
        self.max_tracking_error = 0.0
        self.initial_q = robot.data.qpos[self.qpos_indices].copy()

    @property
    def reached_waypoints(self) -> int:
        """Number of planned vertices reached in order, including the start."""
        return self.next_index

    def __call__(self, simulator) -> None:
        """Update one target before the simulator's control and physics step."""
        if self.complete or self.failed:
            return
        self.check_arrival(simulator)
        if self.complete:
            return
        self.steps += 1
        if self.steps > self.max_steps:
            self.failed = True
            simulator.stop()
            return

        source = self.waypoints[self.next_index - 1]
        destination = self.waypoints[self.next_index]
        difference = destination - source
        duration = max(np.max(np.abs(difference)) / self.speed, simulator.dt)
        self.progress = min(1.0, self.progress + simulator.dt / duration)
        selected_target = source + self.progress * difference

        target = self.base_target.copy()
        target[self.target_indices] = selected_target
        velocity = np.zeros_like(target)
        if self.progress < 1.0:
            velocity[self.target_indices] = difference / duration
        self.robot.target = ControlTarget(target, velocity)

        measured = simulator.data.qpos[self.qpos_indices]
        tracking_error = float(np.max(np.abs(selected_target - measured)))
        self.max_tracking_error = max(self.max_tracking_error, tracking_error)
        self.check_arrival(simulator)

    def check_arrival(self, simulator) -> None:
        """Accept the most recently stepped state if it reaches the current vertex."""
        if self.complete or self.failed or self.progress != 1.0:
            return
        measured = simulator.data.qpos[self.qpos_indices]
        measured_velocity = simulator.data.qvel[self.dof_indices]
        if (
            np.max(np.abs(self.waypoints[self.next_index] - measured)) <= self.position_tolerance
            and np.max(np.abs(measured_velocity)) <= self.velocity_tolerance
        ):
            self.next_index += 1
            self.progress = 0.0
            if self.next_index == len(self.waypoints):
                self.complete = True
                if self.stop_on_finish:
                    simulator.stop()

    def result(self) -> str:
        """Describe the measured result after the simulation has stopped."""
        final_q = self.robot.data.qpos[self.qpos_indices]
        error = float(np.max(np.abs(final_q - self.waypoints[-1])))
        speed = float(np.max(np.abs(self.robot.data.qvel[self.dof_indices])))
        travel = float(np.max(np.abs(final_q - self.initial_q)))
        status = "complete" if self.complete else "timeout" if self.failed else "interrupted"
        return (
            f"Execution {status}: {self.reached_waypoints}/{len(self.waypoints)} "
            f"waypoints reached in {self.steps} steps; final max joint error "
            f"{error:.4f} rad; final max joint speed {speed:.4f} rad/s; "
            f"measured travel {travel:.4f} rad; "
            f"peak tracking error {self.max_tracking_error:.4f} rad"
        )
