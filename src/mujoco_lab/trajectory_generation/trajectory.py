"""Time and sample joint paths with configured derivative limits."""

from typing import Annotated

import numpy as np
from pydantic import Field
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

from mujoco_lab.trajectory_generation.interpolation import min_jerk
from mujoco_lab.trajectory_generation.target import ControlTarget

# Fractions of configured velocity and acceleration limits.
type MotionRatio = tuple[Annotated[float, Field(gt=0, le=1)], Annotated[float, Field(gt=0, le=1)]]


class JointTrajectory:
    """Time a C1 joint curve, with an exact stop-at-waypoints polyline fallback.

    Smooth segments may leave the polyline, and acceleration may jump at knots.
    Shape-preserving smoothing keeps each joint inside its waypoint intervals.
    Ratio pairs scale the supplied limits, uniformly or once per joint. Each
    fraction must be in (0, 1]; both interpolation modes use the scaled limits.
    ``peak_velocity`` stores the actual absolute peak speed of each timed joint.
    """

    def __init__(
        self,
        path: np.ndarray,
        max_velocity: float | np.ndarray,
        max_acceleration: float | np.ndarray,
        *,
        ratio: MotionRatio | np.ndarray = (1.0, 1.0),
        smooth: bool = True,
        shape_preserving: bool = False,
    ) -> None:
        waypoints = np.array(path, dtype=float, copy=True)
        if waypoints.ndim != 2 or not all(waypoints.shape) or not np.isfinite(waypoints).all():
            raise ValueError("path must be a nonempty finite 2D array of joint positions")
        ratios = np.array(ratio, dtype=float, copy=True)
        if ratios.shape not in ((2,), (waypoints.shape[1], 2)):
            raise ValueError("ratio must be a velocity/acceleration pair or one pair per joint")
        if not np.all((ratios > 0) & (ratios <= 1)):
            raise ValueError("ratio fractions must be finite and in (0, 1]")
        ratios.flags.writeable = False
        self.ratio = np.broadcast_to(ratios, (waypoints.shape[1], 2))
        limits = []
        for index, (name, value) in enumerate(
            (
                ("max_velocity", max_velocity),
                ("max_acceleration", max_acceleration),
            )
        ):
            limit = np.broadcast_to(value, (waypoints.shape[1],))
            limit = limit * self.ratio[:, index]
            if not np.all(np.isfinite(limit) & (limit > 0)):
                raise ValueError(f"scaled {name} must contain positive finite values")
            limit.flags.writeable = False
            limits.append(limit)
        velocity, acceleration = limits
        self.max_velocity = velocity
        self.max_acceleration = acceleration
        distance = np.abs(np.diff(waypoints, axis=0))
        durations = np.maximum(
            np.max((15 / 8) * distance / velocity, axis=1),
            np.max(np.sqrt((10 / np.sqrt(3)) * distance / acceleration), axis=1),
        )
        self.peak_velocity = np.max(
            np.divide(
                (15 / 8) * distance,
                durations[:, None],
                out=np.zeros_like(distance),
                where=durations[:, None] > 0,
            ),
            axis=0,
            initial=0.0,
        )
        self.path = waypoints
        self.path.flags.writeable = False
        self.waypoint_times = np.r_[0.0, np.cumsum(durations)]
        self.smooth = smooth
        self._coefficients = None
        if smooth and self.waypoint_times[-1] > 0:
            active = np.r_[True, np.any(np.diff(waypoints, axis=0) != 0, axis=1)]
            knots = waypoints[active]
            knot_times = self.waypoint_times[active]
            span = np.diff(knot_times)[:, None]
            secants = np.diff(knots, axis=0) / span
            slopes = np.zeros_like(knots)
            if len(knots) > 2:
                if shape_preserving:
                    slopes[1:-1] = PchipInterpolator(knot_times, knots, axis=0).derivative()(
                        knot_times[1:-1]
                    )
                else:
                    slopes[1:-1] = (secants[:-1] * span[1:] + secants[1:] * span[:-1]) / (
                        span[:-1] + span[1:]
                    )
            coefficients = CubicHermiteSpline(knot_times, knots, slopes, axis=0).c
            a, b, c = coefficients[:3]
            vertex = np.divide(-b, 3 * a, out=np.zeros_like(a), where=a != 0)
            vertex_velocity = (3 * a * vertex + 2 * b) * vertex + c
            peak_velocity = np.maximum.reduce(
                (
                    np.abs(c),
                    np.abs((3 * a * span + 2 * b) * span + c),
                    np.where((vertex > 0) & (vertex < span), np.abs(vertex_velocity), 0),
                )
            )
            peak_acceleration = np.maximum(np.abs(2 * b), np.abs(6 * a * span + 2 * b))
            scale = max(
                float(np.max(peak_velocity / velocity)),
                float(np.sqrt(np.max(peak_acceleration / acceleration))),
            )
            self.waypoint_times *= scale
            self._knot_times = knot_times * scale
            self._coefficients = coefficients
            self._coefficients[0] /= scale**3
            self._coefficients[1] /= scale**2
            self._coefficients[2] /= scale
            self.peak_velocity = np.max(peak_velocity, axis=0) / scale
        self.peak_velocity.flags.writeable = False
        self.waypoint_times.flags.writeable = False
        self.duration = float(self.waypoint_times[-1])

    def sample(self, elapsed: float) -> ControlTarget:
        """Return joint position, velocity, and acceleration at elapsed seconds."""
        if not np.isfinite(elapsed):
            raise ValueError("elapsed must be finite")
        if elapsed <= 0:
            position = self.path[0]
            return ControlTarget(position)
        if elapsed >= self.duration:
            position = self.path[-1]
            return ControlTarget(position)
        if self._coefficients is not None:
            index = int(np.searchsorted(self._knot_times, elapsed, side="right") - 1)
            local = elapsed - self._knot_times[index]
            a, b, c, d = self._coefficients[:, index]
            return ControlTarget(
                ((a * local + b) * local + c) * local + d,
                (3 * a * local + 2 * b) * local + c,
                6 * a * local + 2 * b,
            )
        index = int(np.searchsorted(self.waypoint_times, elapsed, side="right") - 1)
        duration = self.waypoint_times[index + 1] - self.waypoint_times[index]
        return min_jerk(
            ControlTarget(self.path[index]),
            ControlTarget(self.path[index + 1]),
            elapsed - self.waypoint_times[index],
            duration,
        )
