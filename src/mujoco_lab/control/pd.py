"""Joint-space PD for any number of torque/force-controlled scalar joints."""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from mujoco_lab.control.base import Controller
from mujoco_lab.state import JointState
from mujoco_lab.trajectory_generation.target import ControlTarget


class JointSpacePD(Controller):
    name = "joint-space PD"
    output_kind = "torque"

    def __init__(
        self,
        kp: ArrayLike,
        kd: ArrayLike,
        gravity_compensation: bool = True,
        frame: str = "grasp",
        *,
        control_dt: float,
        desired_velocity: Literal["reference_velocity", "zero"] = "reference_velocity",
    ) -> None:
        """Configure the PD evaluation period in seconds, separate from target cadence."""
        self.control_dt = float(control_dt)
        if not np.isfinite(self.control_dt) or self.control_dt <= 0:
            raise ValueError("control_dt must be finite and positive")
        self.kp = np.array(kp, dtype=float, copy=True)
        self.kd = np.array(kd, dtype=float, copy=True)
        self.gravity_compensation = gravity_compensation
        self.frame = frame
        if desired_velocity not in ("reference_velocity", "zero"):
            raise ValueError("desired_velocity must be reference_velocity or zero")
        self.desired_velocity = desired_velocity
        self.tracking_error = 0.0
        self._desired = None

    def compute(self, state: JointState, target: ControlTarget) -> np.ndarray:
        position = target.position
        velocity = target.velocity if self.desired_velocity == "reference_velocity" else 0.0
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
        return f"kp {self.kp}, kd {self.kd}, control_dt {self.control_dt:g}s"
