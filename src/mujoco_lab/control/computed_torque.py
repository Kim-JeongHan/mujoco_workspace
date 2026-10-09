"""Joint-space computed torque using the robot's selected inertia block."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from mujoco_lab.control.base import Controller
from mujoco_lab.state import JointState, RobotState
from mujoco_lab.trajectory_generation.target import ControlTarget


class JointSpaceComputedTorque(Controller):
    """Convert desired acceleration and PD feedback into joint torque.

    Natural frequencies are in rad/s and damping ratios are dimensionless.
    Scalars apply to every controlled joint; vectors follow ``dof_slots`` order.
    Bias compensation includes gravity, Coriolis, and centrifugal forces, but
    excludes passive damping, contact forces, and a separately grasped payload.
    The selected mass block assumes omitted coupled DOFs have zero acceleration.
    Dynamics must be refreshed by the caller at the same phase as the snapshot.
    """

    name = "joint-space computed torque"
    output_kind = "torque"

    def __init__(
        self,
        robot_state: RobotState,
        dof_slots: Sequence[int],
        natural_frequency: ArrayLike = 20.0,
        damping_ratio: ArrayLike = 1.0,
        *,
        gravity_compensation: bool = True,
        frame: str = "grasp",
    ) -> None:
        self.robot_state = robot_state
        self.dof_slots = tuple(dof_slots)
        self.natural_frequency = np.array(natural_frequency, dtype=float, copy=True)
        self.damping_ratio = np.array(damping_ratio, dtype=float, copy=True)
        self.kp = self.natural_frequency**2
        self.kd = 2 * self.damping_ratio * self.natural_frequency
        self.gravity_compensation = gravity_compensation
        self.frame = frame
        self.tracking_error = 0.0
        super().bind(robot_state)

    def compute(self, state: JointState, target: ControlTarget) -> np.ndarray:
        """Return M @ (desired acceleration + PD feedback) plus optional bias."""
        for value in (target.position, target.velocity, target.acceleration):
            if value.shape != state.qpos.shape:
                raise ValueError("Control targets must have one value per controlled joint")
        velocity = target.velocity
        acceleration = target.acceleration
        error = target.position - state.qpos
        self.tracking_error = float(np.max(np.abs(error)))
        command = acceleration + self.kp * error + self.kd * (velocity - state.qvel)
        mass = self.robot_state.get_mass_matrix(self.dof_slots)
        torque = mass @ command
        if self.gravity_compensation:
            torque = torque + state.bias_forces
        return torque

    def reset(self) -> None:
        self.tracking_error = 0.0

    def summary(self) -> str:
        return f"omega_n {self.natural_frequency} rad/s, zeta {self.damping_ratio}"
