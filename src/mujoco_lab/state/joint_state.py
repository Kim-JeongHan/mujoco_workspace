"""Joint state data and direct access to the current MuJoCo state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mujoco


@dataclass(frozen=True)
class JointState:
    """One robot's state in controller joint order and SI units.

    Arrays may be live MuJoCo views; consume them during the current control step.
    Bias forces include gravity and velocity-dependent terms, matching qfrc_bias.
    """

    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    bias_forces: np.ndarray


def read_state(data: mujoco.MjData) -> JointState:
    """Expose the current state directly, preserving native values and joint order."""
    return JointState(data.time, data.qpos, data.qvel, data.qfrc_bias)
