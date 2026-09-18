"""Original Forte OSC equations driven by MuJoCo dynamics getters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mujoco_lab.control.base import Controller
from mujoco_lab.control.trajectory import HOME_QPOS, min_jerk
from mujoco_lab.state import JointState

if TYPE_CHECKING:
    from mujoco_lab.state.dynamics import Dynamics

CIRCLE_CENTER = np.array([0.65, 0.0, 0.35])
CIRCLE_RADIUS = 0.18
CIRCLE_PERIOD = 5.0
LEAD_IN = 2.0

TASK_KP = 900.0
TASK_KD = 60.0
POSTURE_KP = 20.0
POSTURE_KD = 4.0


class OperationalSpaceControl(Controller):
    name = "operational-space control"

    def __init__(
        self,
        dynamics: Dynamics,
        frame: str = "ee_site",
        center: np.ndarray = CIRCLE_CENTER,
        radius: float = CIRCLE_RADIUS,
        period: float = CIRCLE_PERIOD,
        lead_in: float = LEAD_IN,
        task_kp: float = TASK_KP,
        task_kd: float = TASK_KD,
        posture_kp: float = POSTURE_KP,
        posture_kd: float = POSTURE_KD,
        regularization: float = 1e-4,
    ) -> None:
        dynamics.get_frame_position(frame)
        self.dynamics = dynamics
        self.frame = frame
        self.center = np.asarray(center, dtype=float)
        self.radius = radius
        self.period = period
        self.lead_in = lead_in
        self.task_kp = task_kp
        self.task_kd = task_kd
        self.posture_kp = posture_kp
        self.posture_kd = posture_kd
        self.regularization = regularization
        self.posture = HOME_QPOS.copy()
        self.tracking_error = 0.0
        self._start = None
        self._force = np.zeros(3)
        self._target = self.center.copy()

    def circle(self, elapsed: float) -> tuple[np.ndarray, ...]:
        omega = 2.0 * np.pi / self.period
        angle = omega * elapsed
        offset = np.array([0.0, np.sin(angle), np.cos(angle)])
        tangent = np.array([0.0, np.cos(angle), -np.sin(angle)])
        position = self.center + self.radius * offset
        velocity = self.radius * omega * tangent
        acceleration = -self.radius * omega**2 * offset
        return position, velocity, acceleration

    def target(self, elapsed: float) -> tuple[np.ndarray, ...]:
        """Lead-in from the start pose, then follow the circle."""
        entry, _, _ = self.circle(0.0)
        if elapsed >= self.lead_in:
            return self.circle(elapsed - self.lead_in)
        blend, slope = min_jerk(elapsed / self.lead_in)
        delta = entry - self._start
        return (
            self._start + delta * blend,
            delta * slope / self.lead_in,
            np.zeros(3),
        )

    def torques(self, state: JointState) -> np.ndarray:
        if self._start is None:
            self._start = self.dynamics.get_frame_position(self.frame).copy()

        jacobian = self.dynamics.get_jacobian(self.frame)[:3]
        mass = self.dynamics.get_mass_matrix()
        mass_inverse = np.linalg.inv(mass)

        position, velocity, acceleration = self.target(state.time)
        self._target = position
        error = position - self.dynamics.get_frame_position(self.frame)
        self.tracking_error = float(np.linalg.norm(error))
        velocity_error = velocity - jacobian @ state.qvel
        command = acceleration + self.task_kp * error + self.task_kd * velocity_error

        task_inertia = np.linalg.inv(
            jacobian @ mass_inverse @ jacobian.T + self.regularization * np.eye(3)
        )
        self._force = task_inertia @ command
        torque = jacobian.T @ self._force

        # Null-space term keeps unused joints near the home pose.
        pseudo_inverse = mass_inverse @ jacobian.T @ task_inertia
        null_space = np.eye(self.dynamics.nv) - jacobian.T @ pseudo_inverse.T
        posture = self.posture_kp * (self.posture - state.qpos) - self.posture_kd * state.qvel
        return torque + null_space @ posture + state.bias_forces

    def summary(self) -> str:
        return (
            f"circle radius {self.radius} m, period {self.period} s, "
            f"kp {self.task_kp:g}, kd {self.task_kd:g}"
        )


class ForteOSC(OperationalSpaceControl):
    """Use upstream OSC with its target trajectory expressed in the robot base frame."""

    def __init__(self, dynamics: Dynamics, base_position: np.ndarray, base_rotation: np.ndarray):
        self._base_rotation = base_rotation.copy()
        center = base_position + self._base_rotation @ CIRCLE_CENTER
        super().__init__(dynamics, center=center)

    def circle(self, elapsed: float) -> tuple[np.ndarray, ...]:
        position, velocity, acceleration = super().circle(elapsed)
        return (
            self.center + self._base_rotation @ (position - self.center),
            self._base_rotation @ velocity,
            self._base_rotation @ acceleration,
        )
