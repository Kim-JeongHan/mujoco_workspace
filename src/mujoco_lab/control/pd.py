"""Joint-space PD for any number of torque/force-controlled scalar joints."""

import numpy as np
from numpy.typing import ArrayLike

from mujoco_lab.control.base import Controller
from mujoco_lab.control.target import ControlTarget
from mujoco_lab.state import JointState


class JointSpacePD(Controller):
    name = "joint-space PD"
    output_kind = "torque"

    def __init__(
        self,
        kp: ArrayLike,
        kd: ArrayLike,
        gravity_compensation: bool = True,
        frame: str = "ee_site",
    ) -> None:
        self.kp = np.array(kp, dtype=float, copy=True)
        self.kd = np.array(kd, dtype=float, copy=True)
        self.gravity_compensation = gravity_compensation
        self.frame = frame
        self.tracking_error = 0.0
        self._desired = None

    def compute(self, state: JointState, target: ControlTarget) -> np.ndarray:
        position = target.position
        velocity = np.zeros_like(position) if target.velocity is None else target.velocity
        self._desired = position.copy()
        error = position - state.qpos
        self.tracking_error = float(np.abs(error).max())
        torque = self.kp * error + self.kd * (velocity - state.qvel)
        if self.gravity_compensation:
            torque = torque + state.bias_forces
        return torque

    def reset(self) -> None:
        self.tracking_error = 0.0
        self._desired = None

    def summary(self) -> str:
        return f"kp {self.kp}, kd {self.kd}"
