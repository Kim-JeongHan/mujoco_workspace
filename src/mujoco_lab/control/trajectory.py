"""Time-based joint trajectories and bundled robot demo references."""

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from mujoco_lab.control.target import ControlTarget

# The CAD zero pose is inside all three source-defined bounded arm joints.
HOME_QPOS = np.zeros(7)
PD_WAYPOINTS = np.array(
    [
        HOME_QPOS,
        [0.05, -0.15, 0.05, 0.15, 0.0, 0.0, 0.0],
        [-0.05, -0.10, -0.05, 0.20, 0.10, -0.10, 0.10],
        [0.0, 0.05, 0.05, 0.05, -0.10, 0.10, -0.10],
    ]
)
# Base-relative zero-pose ee_site position; the root's 0.058 m lift is added later.
CIRCLE_CENTER = np.array([0.6349, -0.04638, 0.40459])
CIRCLE_RADIUS = 0.025
CIRCLE_PERIOD = 5.0
LEAD_IN = 2.0


def min_jerk(fraction: float) -> tuple[float, float]:
    """Minimum-jerk blend and its derivative."""
    u = float(np.clip(fraction, 0.0, 1.0))
    blend = 10 * u**3 - 15 * u**4 + 6 * u**5
    slope = 30 * u**2 - 60 * u**3 + 30 * u**4
    return blend, slope


class JointTrajectory:
    """Time a C1 joint curve, with an exact stop-at-waypoints polyline fallback.

    Smooth segments may leave the polyline, and acceleration may jump at knots.
    """

    def __init__(
        self,
        path: np.ndarray,
        max_velocity: float | np.ndarray,
        max_acceleration: float | np.ndarray,
        *,
        smooth: bool = True,
    ) -> None:
        waypoints = np.array(path, dtype=float, copy=True)
        if waypoints.ndim != 2 or not all(waypoints.shape) or not np.isfinite(waypoints).all():
            raise ValueError("path must be a nonempty finite 2D array of joint positions")
        limits = []
        for name, value in (
            ("max_velocity", max_velocity),
            ("max_acceleration", max_acceleration),
        ):
            try:
                limit = np.broadcast_to(np.asarray(value, dtype=float), (waypoints.shape[1],))
            except ValueError as exc:
                raise ValueError(f"{name} must be scalar or one value per joint") from exc
            if not np.isfinite(limit).all() or np.any(limit <= 0):
                raise ValueError(f"{name} must contain positive finite values")
            limits.append(limit)
        velocity, acceleration = limits
        distance = np.abs(np.diff(waypoints, axis=0))
        durations = (
            np.maximum(
                np.max((15 / 8) * distance / velocity, axis=1),
                np.max(np.sqrt((10 / np.sqrt(3)) * distance / acceleration), axis=1),
            )
            if len(waypoints) > 1
            else np.empty(0)
        )
        if not np.isfinite(durations).all():
            raise ValueError("path and limits must yield finite segment durations")
        self.path = waypoints
        self.path.flags.writeable = False
        self.waypoint_times = np.r_[0.0, np.cumsum(durations)]
        if not np.isfinite(self.waypoint_times).all():
            raise ValueError("path and limits must yield a finite total duration")
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
            if (
                scale <= 0
                or not np.isfinite(scale)
                or not np.isfinite(self.waypoint_times * scale).all()
            ):
                raise ValueError("path and limits must yield a finite total duration")
            self.waypoint_times *= scale
            self._knot_times = knot_times * scale
            self._coefficients = coefficients.copy()
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
        fraction = (elapsed - self.waypoint_times[index]) / duration
        blend, slope = min_jerk(fraction)
        acceleration = 60 * fraction - 180 * fraction**2 + 120 * fraction**3
        delta = self.path[index + 1] - self.path[index]
        return ControlTarget(
            self.path[index] + delta * blend,
            delta * slope / duration,
            delta * acceleration / duration**2,
        )


def pd_waypoint_target(
    time: float,
    *,
    start_time: float,
    waypoints: np.ndarray = PD_WAYPOINTS,
    transit: float = 1.5,
    hold: float = 0.4,
) -> ControlTarget:
    """Return the cyclic Forte joint target at an explicit time epoch."""

    segment = transit + hold
    phase = (time - start_time) % (len(waypoints) * segment)
    index = int(phase // segment)
    local = phase - index * segment
    start = waypoints[index]
    end = waypoints[(index + 1) % len(waypoints)]
    if local >= transit:
        return ControlTarget(end)
    blend, slope = min_jerk(local / transit)
    delta = end - start
    return ControlTarget(start + delta * blend, delta * slope / transit)


def osc_circle_target(
    time: float,
    *,
    start_time: float,
    start_position: np.ndarray,
    base_position: np.ndarray,
    base_rotation: np.ndarray,
    center: np.ndarray = CIRCLE_CENTER,
    radius: float = CIRCLE_RADIUS,
    period: float = CIRCLE_PERIOD,
    lead_in: float = LEAD_IN,
) -> ControlTarget:
    """Return the Forte lead-in and circle in world coordinates."""

    elapsed = time - start_time
    omega = 2.0 * np.pi / period

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
    blend, slope = min_jerk(elapsed / lead_in)
    delta = entry - start_position
    return ControlTarget(start_position + delta * blend, delta * slope / lead_in)


def demo_target_updater(simulator, modes: dict[str, str]):
    """Build one explicit-time reference updater for bundled PD/OSC demos.

    The start epoch and OSC start pose are captured now. Simulator.reset restores
    the same initial time and pose, so this updater replays from the beginning.
    """
    from functools import partial

    epoch = simulator.data.time
    trajectories = {}
    for name, mode in modes.items():
        robot = simulator.robots[name]
        if mode == "pd":
            trajectories[name] = partial(pd_waypoint_target, start_time=epoch)
        elif mode == "osc":
            base = simulator.data.body(robot.state.root_body_id)
            trajectories[name] = partial(
                osc_circle_target,
                start_time=epoch,
                start_position=robot.state.get_frame_position("ee_site").copy(),
                base_position=base.xpos.copy(),
                base_rotation=base.xmat.reshape(3, 3).copy(),
            )
        else:
            raise ValueError(f"Unknown demo controller {mode!r}")

    def update(sim):
        for name, trajectory in trajectories.items():
            sim.robots[name].target = trajectory(sim.data.time)

    return update
