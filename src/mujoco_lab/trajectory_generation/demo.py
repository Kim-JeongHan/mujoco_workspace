"""Explicit-time reference signals for the bundled controller demonstrations."""

import numpy as np

from mujoco_lab.trajectory_generation.interpolation import min_jerk
from mujoco_lab.trajectory_generation.target import ControlTarget

CIRCLE_RADIUS = 0.025
CIRCLE_PERIOD = 5.0
LEAD_IN = 2.0


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
