"""Reference targets and joint or task-space trajectory generation."""

from mujoco_lab.trajectory_generation.demo import demo_target_updater, osc_circle_target
from mujoco_lab.trajectory_generation.interpolation import (
    min_jerk,
    min_jerk_duration,
)
from mujoco_lab.trajectory_generation.references import (
    AnalyticJerkReference,
    MinJerkReference,
    ReferenceGenerator,
    ReferenceType,
    create_reference_generator,
)
from mujoco_lab.trajectory_generation.target import ControlTarget
from mujoco_lab.trajectory_generation.trajectory import (
    JointTrajectory,
    MotionRatio,
)

__all__ = [
    "AnalyticJerkReference",
    "MinJerkReference",
    "ReferenceGenerator",
    "ReferenceType",
    "ControlTarget",
    "JointTrajectory",
    "MotionRatio",
    "create_reference_generator",
    "demo_target_updater",
    "min_jerk",
    "min_jerk_duration",
    "osc_circle_target",
]
