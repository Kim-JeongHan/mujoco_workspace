"""External seven-axis references for the curated ForteV1_RobStride model."""

import numpy as np

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
