"""Common controller binding and NumPy control interface."""

import numpy as np

from mujoco_lab.control.target import ControlTarget
from mujoco_lab.state import JointState, RobotState


class Controller:
    """Compute selected joint torque or position commands before Robot applies limits."""

    name = "controller"
    output_kind = "torque"
    tracking_error: float | None = None
    _owner = None

    def bind(self, robot_state: RobotState) -> None:
        """Validate ownership and bind this controller to one robot state."""
        if self._owner is not None and self._owner is not robot_state:
            raise ValueError("A controller instance can belong to only one Robot")
        self._owner = robot_state

    def initial_target(self, state: JointState) -> ControlTarget:
        """Hold the current joint posture when this controller is connected."""
        return ControlTarget(state.qpos)

    def compute(self, state: JointState, target: ControlTarget) -> np.ndarray:
        """Return one torque or position command per selected joint."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear runtime history while retaining controller configuration."""
        return

    def summary(self) -> str:
        return ""

    def get_tracking_error(self) -> float | None:
        """Return the latest controller tracking error, if available."""
        return self.tracking_error
