"""Construct online reference generators from typed robot settings."""

from mujoco_lab.assets.robot.robot import Constraints, ReferenceConfig
from mujoco_lab.assets.robot.robot import ReferenceType as ReferenceType
from mujoco_lab.trajectory_generation.references.base import ReferenceGenerator
from mujoco_lab.trajectory_generation.references.jerk_lqr import JerkLQRReference
from mujoco_lab.trajectory_generation.references.min_jerk import MinJerkReference
from mujoco_lab.trajectory_generation.references.position_hold import PositionHoldReference
from mujoco_lab.trajectory_generation.target import ControlTarget


def create_reference_generator(
    config: ReferenceConfig,
    initial: ControlTarget,
    constraints: Constraints,
    *,
    control_dt: float = 0.002,
) -> ReferenceGenerator:
    """Build a configured reference without advancing time.

    ``control_dt`` comes from Simulator. JerkLQR uses the configured
    horizon, response time, and terminal weight. Minimum jerk uses duration
    and constrain; its minimum segment duration is independent of update cadence.
    Position hold repeats the raw goal with zero desired derivatives.
    """
    if config.name == "position_hold":
        return PositionHoldReference(initial, control_dt=control_dt)
    if config.name == "jerk_lqr":
        return JerkLQRReference(
            initial,
            constraints,
            control_dt=control_dt,
            horizon=config.horizon,
            response_time=config.response_time,
            terminal_weight=config.terminal_weight,
        )
    if config.name == "min_jerk":
        return MinJerkReference(
            initial,
            constraints,
            control_dt=control_dt,
            duration=config.duration,
            constrain=config.constrain,
        )
    raise ValueError(f"Unknown reference type: {config.name}")
