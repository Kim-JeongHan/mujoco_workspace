"""Position-only operational-space control with external targets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mujoco_lab.control.base import Controller
from mujoco_lab.control.target import ControlTarget
from mujoco_lab.state import JointState

if TYPE_CHECKING:
    from mujoco_lab.state import RobotState

TASK_KP = 900.0
TASK_KD = 60.0
POSTURE_KP = 20.0
POSTURE_KD = 4.0


class OperationalSpaceControl(Controller):
    name = "operational-space control"

    def __init__(
        self,
        robot_state: RobotState,
        frame: str = "ee_site",
        *,
        posture: np.ndarray,
        task_kp: float = TASK_KP,
        task_kd: float = TASK_KD,
        posture_kp: float = POSTURE_KP,
        posture_kd: float = POSTURE_KD,
        regularization: float = 1e-4,
    ) -> None:
        robot_state.get_frame_position(frame)
        self.robot_state = robot_state
        self.frame = frame
        self.task_kp = task_kp
        self.task_kd = task_kd
        self.posture_kp = posture_kp
        self.posture_kd = posture_kd
        self.regularization = regularization
        self.posture = np.array(posture, dtype=float, copy=True)
        self.tracking_error = 0.0
        self._force = np.zeros(3)
        self._target = None

    def bind(self, robot_state: RobotState) -> None:
        if self.robot_state is not robot_state:
            raise ValueError("OSC must be assigned to the RobotState it was constructed for")
        super().bind(robot_state)

    def initial_target(self, state: JointState) -> ControlTarget:
        """Hold the current world-frame end-effector position."""
        return ControlTarget(self.robot_state.get_frame_position(self.frame))

    def compute(self, state: JointState, target: ControlTarget) -> np.ndarray:
        jacobian = self.robot_state.get_jacobian(self.frame)[:3]
        mass = self.robot_state.get_mass_matrix()
        mass_inverse = np.linalg.inv(mass)

        position = target.position
        velocity = np.zeros(3) if target.velocity is None else target.velocity
        acceleration = np.zeros(3) if target.acceleration is None else target.acceleration
        self._target = position.copy()
        error = position - self.robot_state.get_frame_position(self.frame)
        self.tracking_error = float(np.linalg.norm(error))
        velocity_error = velocity - jacobian @ state.qvel
        command = acceleration + self.task_kp * error + self.task_kd * velocity_error

        task_inertia = np.linalg.inv(
            jacobian @ mass_inverse @ jacobian.T + self.regularization * np.eye(3)
        )
        self._force = task_inertia @ command
        torque = jacobian.T @ self._force

        pseudo_inverse = mass_inverse @ jacobian.T @ task_inertia
        null_space = np.eye(self.robot_state.nv) - jacobian.T @ pseudo_inverse.T
        posture = self.posture_kp * (self.posture - state.qpos) - self.posture_kd * state.qvel
        return torque + null_space @ posture + state.bias_forces

    def reset(self) -> None:
        self.tracking_error = 0.0
        self._force.fill(0)
        self._target = None

    def summary(self) -> str:
        return f"kp {self.task_kp:g}, kd {self.task_kd:g}"
