"""Interchangeable online reference generators."""

from mujoco_lab.trajectory_generation.references.analytic_jerk import AnalyticJerkReference
from mujoco_lab.trajectory_generation.references.base import ReferenceGenerator
from mujoco_lab.trajectory_generation.references.factory import (
    ReferenceType,
    create_reference_generator,
)
from mujoco_lab.trajectory_generation.references.min_jerk import MinJerkReference

__all__ = [
    "ReferenceGenerator",
    "AnalyticJerkReference",
    "MinJerkReference",
    "ReferenceType",
    "create_reference_generator",
]
