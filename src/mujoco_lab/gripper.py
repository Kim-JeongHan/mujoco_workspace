"""Robot-owned position-servo gripper target and measured joint position."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from mujoco_lab.robot import Robot


class Gripper:
    """Control one explicitly named position actuator in joint units."""

    def __init__(self, robot: Robot, actuator_name: str) -> None:
        if actuator_name not in robot.actuator_names:
            raise ValueError(f"Unknown gripper actuator {actuator_name!r}")
        self._state = robot.state
        self._model = robot.model
        self._data = robot.data
        self.slot = robot.actuator_names.index(actuator_name)
        self.actuator_id = robot.actuator_ids[self.slot]
        model = self._model
        actuator = self.actuator_id
        if model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError("Gripper requires a direct joint position actuator")
        joint = int(model.actuator_trnid[actuator, 0])
        if joint not in self._state.joint_ids:
            raise ValueError("Gripper actuator must drive a robot-owned joint")
        self.joint_id = joint
        gain = model.actuator_gainprm[actuator, 0]
        bias = model.actuator_biasprm[actuator]
        self.gear = float(model.actuator_gear[actuator, 0])
        if not (
            model.actuator_gaintype[actuator] == mujoco.mjtGain.mjGAIN_FIXED
            and model.actuator_biastype[actuator] == mujoco.mjtBias.mjBIAS_AFFINE
            and gain > 0
            and self.gear != 0
            and np.allclose(model.actuator_gear[actuator, 1:], 0)
            and np.isclose(bias[0], 0)
            and np.isclose(bias[1], -gain)
            and bias[2] <= 0
        ):
            raise ValueError("Gripper requires a scalar position-servo actuator")
        self._qpos_index = int(model.jnt_qposadr[joint])
        joints = {self.joint_id}
        links = [
            (int(model.eq_obj1id[index]), int(model.eq_obj2id[index]))
            for index in range(model.neq)
            if model.eq_type[index] == mujoco.mjtEq.mjEQ_JOINT
        ]
        while True:
            previous = len(joints)
            for first, second in links:
                if first in joints and second in self._state.joint_ids:
                    joints.add(second)
                if second in joints and first in self._state.joint_ids:
                    joints.add(first)
            if len(joints) == previous:
                break
        self._finger_joint_ids = [joint for joint in self._state.joint_ids if joint in joints]
        self._finger_qpos_indices = [
            int(model.jnt_qposadr[joint]) for joint in self._finger_joint_ids
        ]
        self._home_target = 0.0
        self._target = 0.0
        self._active = False

    def get_position(self) -> float:
        """Measured position of the driven joint in meters or radians."""
        return float(self._state.data.qpos[self._qpos_index])

    def get_target(self) -> float:
        """Stored joint target; active after set_target()."""
        return self._target

    def get_control_limits(self) -> np.ndarray:
        """Return physical target limits in the driven joint's native units."""
        if not self._model.actuator_ctrllimited[self.actuator_id]:
            return np.array([-np.inf, np.inf])
        return np.sort(self._model.actuator_ctrlrange[self.actuator_id] / self.gear)

    def get_width(self) -> float:
        """Sum displacement from lower limits for the driven and coupled fingers."""
        lower = self._model.jnt_range[self._finger_joint_ids, 0]
        return float(np.sum(self._data.qpos[self._finger_qpos_indices] - lower))

    def is_active(self) -> bool:
        return self._active

    def set_target(self, displacement: float) -> None:
        """Store a joint target, checked against the native actuator range."""
        value = float(displacement) * self.gear
        lower, upper = self._model.actuator_ctrlrange[self.actuator_id]
        limited = self._model.actuator_ctrllimited[self.actuator_id]
        if not np.isfinite(value) or (limited and not lower <= value <= upper):
            raise ValueError("gripper target is outside the actuator's control range")
        self._target = float(displacement)
        self._active = True

    def _capture_home(self) -> None:
        self._home_target = float(self._data.ctrl[self.actuator_id] / self.gear)
        self.reset()

    def reset(self) -> None:
        self._target = self._home_target
        self._active = False
