"""Joint position commands for native MuJoCo position servos."""

import numpy as np
from numpy.typing import ArrayLike

from mujoco_lab.control.base import Controller
from mujoco_lab.control.target import ControlTarget
from mujoco_lab.state import JointState


class PositionController(Controller):
    """Send joint positions with optional bias compensation using servo gains.

    ``kp`` is the effective stiffness in joint coordinates. A scalar or per-joint
    compensation mask can be supplied. The native position servo computes the
    force; velocity and acceleration targets are not used by this controller.
    """

    name = "joint position"
    output_kind = "position"

    def __init__(self, kp: ArrayLike, gravity_compensation: bool | ArrayLike = True):
        self.kp = np.array(kp, dtype=float, copy=True)
        self.gravity_compensation = np.array(gravity_compensation, dtype=bool, copy=True)
        self.tracking_error = 0.0

    def compute(self, state: JointState, target: ControlTarget) -> np.ndarray:
        self.tracking_error = float(np.linalg.norm(target.position - state.qpos))
        correction = np.where(self.gravity_compensation, state.bias_forces / self.kp, 0)
        return target.position + correction

    def reset(self) -> None:
        self.tracking_error = 0.0
