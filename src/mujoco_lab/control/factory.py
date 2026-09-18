"""Validate native Forte torque actuators and construct workspace controllers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mujoco_lab.control.base import Controller
from mujoco_lab.control.osc import ForteOSC
from mujoco_lab.control.pd import JointSpacePD

if TYPE_CHECKING:
    import mujoco

CONTROLLER_NAMES = ("none", "pd", "osc")

_JOINTS = (
    "shoulder_yaw",
    "shoulder_pitch",
    "shoulder_roll",
    "elbow_pitch",
    "lower_arm_roll",
    "wrist_pitch",
    "wrist_roll",
)


def create_controller(name: str, model: mujoco.MjModel, data: mujoco.MjData) -> Controller | None:
    """Select no controller, original Forte PD, or base-frame-aware Forte OSC.

    The Forte controllers require its seven unit-gear torque motors. Position
    actuators and different joint layouts are rejected before controls are applied.
    """
    import mujoco

    from mujoco_lab.state.dynamics import Dynamics

    if name not in CONTROLLER_NAMES:
        raise ValueError(f"Unknown controller {name!r}. Available controllers: {CONTROLLER_NAMES}")
    if name == "none":
        return None
    valid = (
        model.nq == model.nv == model.nu == 7
        and tuple(model.joint(i).name for i in range(model.njnt)) == _JOINTS
        and np.all(model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
        and np.array_equal(model.actuator_trnid[:, 0], np.arange(7))
        and np.all(model.actuator_gaintype == mujoco.mjtGain.mjGAIN_FIXED)
        and np.all(model.actuator_biastype == mujoco.mjtBias.mjBIAS_NONE)
        and np.all(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_NONE)
        and np.allclose(model.actuator_gainprm[:, 0], 1)
        and np.allclose(model.actuator_gear[:, 0], 1)
        and np.allclose(model.actuator_gear[:, 1:], 0)
    )
    if not valid:
        raise ValueError(f"Controller {name!r} requires the Forte seven-joint torque model")
    if name == "pd":
        return JointSpacePD()
    base = data.body("base_link")
    return ForteOSC(Dynamics(model, data), base.xpos, base.xmat.reshape(3, 3))
