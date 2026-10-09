"""Quintic interpolation and duration selection under derivative limits."""

import numpy as np
from scipy.interpolate import BPoly, PPoly

from mujoco_lab.assets.robot.robot import Constraints
from mujoco_lab.trajectory_generation.target import ControlTarget


def min_jerk(
    start: ControlTarget,
    end: ControlTarget,
    elapsed: float,
    duration: float,
) -> ControlTarget:
    """Sample a minimum-jerk segment with position, velocity, and acceleration boundaries.

    Times are in seconds; elapsed is clamped to the segment's endpoints. Replan
    from the previous segment's sampled target to preserve position, velocity,
    and acceleration when a new goal arrives.
    This interpolation does not impose velocity, acceleration, or joint limits.
    """
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(elapsed):
        raise ValueError("duration must be finite and positive; elapsed must be finite")
    if start.position.shape != end.position.shape:
        raise ValueError("start and end must have matching position shapes")

    u = float(np.clip(elapsed / duration, 0.0, 1.0))
    p = start.position
    v = start.velocity * duration
    a = start.acceleration * duration**2 / 2
    position_delta = end.position - p - v - a
    velocity_delta = end.velocity * duration - v - 2 * a
    acceleration_delta = end.acceleration * duration**2 - 2 * a
    c3 = 10 * position_delta - 4 * velocity_delta + acceleration_delta / 2
    c4 = -15 * position_delta + 7 * velocity_delta - acceleration_delta
    c5 = 6 * position_delta - 3 * velocity_delta + acceleration_delta / 2
    position = p + u * (v + u * (a + u * (c3 + u * (c4 + u * c5))))
    velocity = (v + u * (2 * a + u * (3 * c3 + u * (4 * c4 + u * 5 * c5)))) / duration
    acceleration = (2 * a + u * (6 * c3 + u * (12 * c4 + u * 20 * c5))) / duration**2
    return ControlTarget(position, velocity, acceleration)


def min_jerk_duration(
    start: ControlTarget,
    end: ControlTarget,
    duration: float,
    constraint: Constraints,
) -> float:
    """Plan a derivative-bounded quintic once for repeated ``min_jerk`` samples."""
    velocity = np.asarray(constraint.velocity_limit, dtype=float)
    acceleration = np.asarray(constraint.acceleration_limit, dtype=float)
    boundaries = [
        (target.position, target.velocity, target.acceleration) for target in (start, end)
    ]
    for _, v, a in boundaries:
        if np.any(np.abs(v) > velocity) or np.any(np.abs(a) > acceleration):
            raise ValueError("Boundary derivatives exceed the motion limits")
    distance = np.abs(end.position - start.position)
    _, start_velocity, start_acceleration = boundaries[0]
    _, end_velocity, end_acceleration = boundaries[1]
    start_at_rest = np.all(start_velocity == 0) and np.all(start_acceleration == 0)
    end_at_rest = np.all(end_velocity == 0) and np.all(end_acceleration == 0)
    if start_at_rest and end_at_rest:
        duration = max(
            duration,
            float(np.max((15 / 8) * distance / velocity)),
            float(np.sqrt(np.max((10 / np.sqrt(3)) * distance / acceleration))),
        )
        return duration

    for _ in range(32):
        polynomial = PPoly.from_bernstein_basis(
            BPoly.from_derivatives(
                [0.0, 1.0], [[p, v * duration, a * duration**2] for p, v, a in boundaries]
            )
        )
        scale = 1.0
        for order, limits in ((1, velocity), (2, acceleration)):
            derivative = polynomial.derivative(order)
            roots = np.concatenate(derivative.derivative().roots(extrapolate=False))
            times = np.r_[0.0, roots[np.isfinite(roots)], 1.0]
            peak = np.max(np.abs(derivative(times)), axis=0) / duration**order
            scale = max(scale, float(np.max((peak / limits) ** (1 / order))))
        if scale <= 1 + 1e-10:
            return duration
        duration *= min(2.0, max(1.25, scale))
    raise ValueError("Could not find a single quintic satisfying the motion limits")
