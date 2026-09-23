"""Controller algorithms and creation for MuJoCo Robots."""

from mujoco_lab.control.base import Controller
from mujoco_lab.control.factory import CONTROLLER_NAMES, create_controller
from mujoco_lab.control.osc import OperationalSpaceControl
from mujoco_lab.control.pd import JointSpacePD
from mujoco_lab.control.position import PositionController
from mujoco_lab.control.target import ControlTarget
from mujoco_lab.control.trajectory import JointTrajectory, demo_target_updater

__all__ = [
    "CONTROLLER_NAMES",
    "Controller",
    "ControlTarget",
    "JointTrajectory",
    "JointSpacePD",
    "OperationalSpaceControl",
    "PositionController",
    "create_controller",
    "demo_target_updater",
]
