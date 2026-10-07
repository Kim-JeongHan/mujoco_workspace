"""Time-based joint trajectories and bundled robot demo references."""

from typing import Annotated

import numpy as np
from pydantic import Field
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator

from mujoco_lab.control.target import ControlTarget

CIRCLE_RADIUS = 0.025
CIRCLE_PERIOD = 5.0
LEAD_IN = 2.0

# A plain pair: fractions of the configured velocity and acceleration limits.
type MotionRatio = tuple[Annotated[float, Field(gt=0, le=1)], Annotated[float, Field(gt=0, le=1)]]


def min_jerk(
    start: ControlTarget,
    end: ControlTarget,
    elapsed: float,
    duration: float,
) -> ControlTarget:
    """Sample a minimum-jerk segment with position, velocity, and acceleration boundaries.

    Times are in seconds; elapsed is clamped to the segment's endpoints. Missing
    derivatives are zero. Replan from the previous segment's sampled target to
    preserve position, velocity, and acceleration when a new goal arrives.
    This interpolation does not impose velocity, acceleration, or joint limits.
    """
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(elapsed):
        raise ValueError("duration must be finite and positive; elapsed must be finite")
    if start.position.shape != end.position.shape:
        raise ValueError("start and end must have matching position shapes")

    def derivatives(target: ControlTarget) -> tuple[np.ndarray, np.ndarray]:
        velocity = np.zeros_like(target.position) if target.velocity is None else target.velocity
        acceleration = (
            np.zeros_like(target.position) if target.acceleration is None else target.acceleration
        )
        return velocity, acceleration

    start_velocity, start_acceleration = derivatives(start)
    end_velocity, end_acceleration = derivatives(end)
    u = float(np.clip(elapsed / duration, 0.0, 1.0))
    p = start.position
    v = start_velocity * duration
    a = start_acceleration * duration**2 / 2
    position_delta = end.position - p - v - a
    velocity_delta = end_velocity * duration - v - 2 * a
    acceleration_delta = end_acceleration * duration**2 - 2 * a
    c3 = 10 * position_delta - 4 * velocity_delta + acceleration_delta / 2
    c4 = -15 * position_delta + 7 * velocity_delta - acceleration_delta
    c5 = 6 * position_delta - 3 * velocity_delta + acceleration_delta / 2
    position = p + u * (v + u * (a + u * (c3 + u * (c4 + u * c5))))
    velocity = (v + u * (2 * a + u * (3 * c3 + u * (4 * c4 + u * 5 * c5)))) / duration
    acceleration = (2 * a + u * (6 * c3 + u * (12 * c4 + u * 20 * c5))) / duration**2
    return ControlTarget(position, velocity, acceleration)


class JointTrajectory:
    """Time a C1 joint curve, with an exact stop-at-waypoints polyline fallback.

    Smooth segments may leave the polyline, and acceleration may jump at knots.
    Shape-preserving smoothing keeps each joint inside its waypoint intervals.
    Ratio pairs scale the supplied limits, uniformly or once per joint. Each
    fraction must be in (0, 1]; both interpolation modes use the scaled limits.
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
        self.waypoint_times.flags.writeable = False
        self.duration = float(self.waypoint_times[-1])

    def sample(self, elapsed: float) -> ControlTarget:
        """Return joint position, velocity, and acceleration at elapsed seconds."""
        if not np.isfinite(elapsed):
            raise ValueError("elapsed must be finite")
        if elapsed <= 0:
            position = self.path[0]
            return ControlTarget(position, np.zeros_like(position), np.zeros_like(position))
        if elapsed >= self.duration:
            position = self.path[-1]
            return ControlTarget(position, np.zeros_like(position), np.zeros_like(position))
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


def osc_circle_target(
    time: float,
    *,
    start_time: float,
    start_position: np.ndarray,
    base_position: np.ndarray,
    base_rotation: np.ndarray,
    center: np.ndarray | None = None,
    radius: float = CIRCLE_RADIUS,
    period: float = CIRCLE_PERIOD,
    lead_in: float = LEAD_IN,
) -> ControlTarget:
    """Return the Forte lead-in and circle in world coordinates."""

    elapsed = time - start_time
    omega = 2.0 * np.pi / period
    if center is None:
        center = base_rotation.T @ (start_position - base_position)

    def circle(phase):
        angle = omega * phase
        offset = np.array([0.0, np.sin(angle), np.cos(angle)])
        tangent = np.array([0.0, np.cos(angle), -np.sin(angle)])
        return ControlTarget(
            base_position + base_rotation @ (center + radius * offset),
            base_rotation @ (radius * omega * tangent),
            base_rotation @ (-radius * omega**2 * offset),
        )

    if elapsed >= lead_in:
        return circle(elapsed - lead_in)
    entry = circle(0.0).position
    return min_jerk(ControlTarget(start_position), ControlTarget(entry), elapsed, lead_in)


def demo_target_updater(simulator, modes: dict[str, str]):
    """Build one explicit-time reference updater for the OSC circle demo.

    The start epoch and controlled poses are captured now. Simulator.reset restores
    the same initial time and pose, so this updater replays from the beginning.
    """
    from functools import partial

    epoch = simulator.data.time
    trajectories = {}
    for name, mode in modes.items():
        if mode != "osc":
            raise ValueError(f"No demo motion is defined for controller {mode!r}")
        robot = simulator.robots[name]
        frame = robot.config.controller.frame
        base = simulator.data.body(robot.state.root_body_id)
        start = robot.state.get_frame_position(frame).copy()
        rotation = base.xmat.reshape(3, 3).copy()
        center = rotation.T @ (start - base.xpos)
        trajectories[name] = partial(
            osc_circle_target,
            start_time=epoch,
            start_position=start,
            base_position=base.xpos.copy(),
            base_rotation=rotation,
            center=center,
        )

    def update(sim):
        for name, trajectory in trajectories.items():
            sim.robots[name].update_target(trajectory(sim.data.time))

    return update
