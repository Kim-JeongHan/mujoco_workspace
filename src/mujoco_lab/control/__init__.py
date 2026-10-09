"""Controller algorithms and creation for MuJoCo Robots."""

from mujoco_lab.control.base import Controller
from mujoco_lab.control.computed_torque import JointSpaceComputedTorque
from mujoco_lab.control.factory import create_controller
from mujoco_lab.control.pd import JointSpacePD
from mujoco_lab.control.position import PositionController

__all__ = [
    "Controller",
    "JointSpacePD",
    "JointSpaceComputedTorque",
    "PositionController",
    "create_controller",
]
