"""NumPy jerk regularization with continuous velocity and acceleration bounds."""

import numpy as np

from mujoco_lab.assets.robot.robot import Constraints
from mujoco_lab.trajectory_generation.references.base import ReferenceGenerator
from mujoco_lab.trajectory_generation.target import ControlTarget


class AnalyticJerkReference(ReferenceGenerator):
    """Track position goals with a finite-horizon jerk penalty and one-step bounds.

    A scaled triple-integrator Riccati recursion computes the feedback gain once.
    Each update projects its desired jerk onto a feasible interval, then integrates
    constant jerk exactly. Goals can change without resetting reference derivatives.
    This projection is not a constrained horizon optimizer. Position, jerk, torque,
    contact forces, and actual robot motion are not bounded by this generator.
    Positions use radians, velocities rad/s, and accelerations rad/s^2.
    """

    def __init__(
        self,
        initial: ControlTarget,
        constraints: Constraints,
        *,
        control_dt: float = 0.002,
        horizon: float = 0.6,
        response_time: float = 0.15,
        terminal_weight: float = 0.0,
    ) -> None:
        self.dt = float(control_dt)
        self.tau = float(response_time)
        if self.dt <= 0 or self.tau <= 0 or horizon <= 0:
            raise ValueError("control_dt, response_time, and horizon must be positive")
        self.vmax = np.array(constraints.velocity_limit, dtype=float, copy=True)
        self.amax = np.array(constraints.acceleration_limit, dtype=float, copy=True)
        self._speed_margin = np.maximum(1e-5, self.vmax * 1e-5)
        self.reset(initial)

        h = self.dt / self.tau
        matrix = np.array([[1, h, h * h / 2], [0, 1, h], [0, 0, 1]], dtype=float)
        input_vector = np.array([h**3 / 6, h * h / 2, h], dtype=float)
        position_cost = np.diag([1.0, 0.0, 0.0])
        value = terminal_weight * position_cost
        gain = np.zeros(3)
        for _ in range(int(np.ceil(horizon / self.dt - 1e-12))):
            future = position_cost + value
            weighted_input = future @ input_vector
            denominator = 1 + float(input_vector @ weighted_input)
            gain = (input_vector @ future @ matrix) / denominator
            value = (
                matrix.T @ future @ matrix
                - np.outer(matrix.T @ weighted_input, input_vector @ future @ matrix) / denominator
            )
            value = (value + value.T) / 2
        self._gain = gain

    def reset(self, initial: ControlTarget) -> None:
        """Restore reference state and goal while retaining the precomputed gain."""
        self.p = initial.position.copy()
        self.v = initial.velocity.copy()
        self.a = initial.acceleration.copy()
        self.goal = self.p.copy()
        self.last_jerk = np.zeros_like(self.p)

    def set_goal(self, target: ControlTarget) -> None:
        """Track the target's position while retaining reference derivatives.

        Goal velocity and acceleration are not used by the position tracking cost.
        """
        goal = target.position
        if goal.shape != self.p.shape or not np.isfinite(goal).all():
            raise ValueError("Goal must be finite and match position shape")
        self.goal = goal.copy()

    def state(self) -> ControlTarget:
        """Snapshot the reference without advancing time."""
        return ControlTarget(self.p, self.v, self.a)

    def _feasible_jerk_interval(self) -> tuple[np.ndarray, np.ndarray]:
        """Bound a constant jerk over the whole tick and retain next-tick viability."""
        h = self.dt
        v, a = self.v, self.a
        lower = (-self.amax - a) / h
        upper = (self.amax - a) / h
        lower = np.maximum(lower, 2 * (-self.vmax - v - a * h) / h**2)
        upper = np.minimum(upper, 2 * (self.vmax - v - a * h) / h**2)

        gap_upper = np.maximum(0.0, self.vmax - v)
        gap_lower = np.maximum(0.0, self.vmax + v)
        if np.any((gap_upper == 0) & (a > 0)) or np.any((gap_lower == 0) & (a < 0)):
            raise ValueError("Speed at its bound with outward acceleration is infeasible")
        upper_mask = (a > 0) & (gap_upper > 0) & (2 * gap_upper < a * h)
        lower_mask = (a < 0) & (gap_lower > 0) & (2 * gap_lower < -a * h)
        upper[upper_mask] = np.minimum(
            upper[upper_mask], -(a[upper_mask] ** 2) / (2 * gap_upper[upper_mask])
        )
        lower[lower_mask] = np.maximum(
            lower[lower_mask], a[lower_mask] ** 2 / (2 * gap_lower[lower_mask])
        )

        # Reserve enough speed margin to brake outward acceleration during the next tick.
        for sign in (1.0, -1.0):
            gap = np.maximum(0.0, self.vmax - sign * v - self._speed_margin)
            outward = sign * a
            speed_budget = 2 * gap / h
            linear = outward + self.amax - speed_budget
            constant = self.amax * (outward - speed_budget)
            discriminant = np.maximum(0.0, linear**2 - 8 * constant)
            allowed = np.clip((-linear + np.sqrt(discriminant)) / 4, 0, self.amax)
            if sign > 0:
                upper = np.minimum(upper, (allowed - a) / h)
            else:
                lower = np.maximum(lower, (-allowed - a) / h)
        if np.any(lower > upper + 1e-7):
            raise ValueError("No feasible constant-jerk update from the reference state")
        return lower, np.maximum(lower, upper)

    def compute(self) -> ControlTarget:
        """Integrate one control tick with the latest goal and return a snapshot."""
        p, v, a = self.p, self.v, self.a
        h, tau = self.dt, self.tau
        k0, k1, k2 = self._gain
        desired = -(k0 * (p - self.goal) + k1 * tau * v + k2 * tau**2 * a) / tau**3
        lower, upper = self._feasible_jerk_interval()
        jerk = np.clip(desired, lower, upper)
        self.p = p + h * v + h**2 * a / 2 + h**3 * jerk / 6
        self.v = np.clip(v + h * a + h**2 * jerk / 2, -self.vmax, self.vmax)
        self.a = np.clip(a + h * jerk, -self.amax, self.amax)
        self.a[np.abs(self.a) < 1e-10] = 0
        self.last_jerk = jerk
        return self.state()
