"""Base class for the NumPy torque controllers."""

import numpy as np

from mujoco_lab.state import JointState


class Controller:
    """Compute joint torques in Nm before the runner applies actuator limits."""

    name = "controller"

    def torques(self, state: JointState) -> np.ndarray:
        raise NotImplementedError

    def summary(self) -> str:
        return ""
