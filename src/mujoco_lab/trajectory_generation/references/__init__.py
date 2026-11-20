"""Interchangeable online reference generators."""

from mujoco_lab.trajectory_generation.references.base import ReferenceGenerator
from mujoco_lab.trajectory_generation.references.factory import (
    ReferenceType,
    create_reference_generator,
)
from mujoco_lab.trajectory_generation.references.jerk_lqr import JerkLQRReference
from mujoco_lab.trajectory_generation.references.min_jerk import MinJerkReference
from mujoco_lab.trajectory_generation.references.position_hold import PositionHoldReference

__all__ = [
    "ReferenceGenerator",
    "JerkLQRReference",
    "MinJerkReference",
    "PositionHoldReference",
    "ReferenceType",
    "create_reference_generator",
]
