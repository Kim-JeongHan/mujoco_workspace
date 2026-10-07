"""Opening-width control over native position-servo gripper joints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np

if TYPE_CHECKING:
    from mujoco_lab.robot import Robot


class Gripper:
    """Control total opening width in meters: zero closed, positive open."""

    def __init__(self, robot: Robot, actuator_name: str, joint_names: list[str]) -> None:
        self._data = robot.data
        self.slot = robot.actuator_names.index(actuator_name)
        self.actuator_id = robot.actuator_ids[self.slot]
        model = robot.model
        actuator = self.actuator_id
        if model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError("Gripper requires a direct joint position actuator")
        joint = int(model.actuator_trnid[actuator, 0])
        if joint not in robot.state.joint_ids:
            raise ValueError("Gripper actuator must drive a robot-owned joint")
        self.joint_id = joint
        self.gear = float(model.actuator_gear[actuator, 0])
        self.finger_joint_ids = tuple(model.joint(robot.prefix + name).id for name in joint_names)
        if not set(self.finger_joint_ids).issubset(robot.state.joint_ids):
            raise ValueError("Gripper fingers must belong to the robot")
        self.finger_geom_ids = tuple(
            frozenset(
                geom
                for geom in range(model.ngeom)
                if model.geom_bodyid[geom] == model.jnt_bodyid[joint]
                and (model.geom_contype[geom] or model.geom_conaffinity[geom])
            )
            for joint in self.finger_joint_ids
        )
        self._finger_qpos_indices = [
            int(model.jnt_qposadr[joint]) for joint in self.finger_joint_ids
        ]
        lower, upper = 0.0, np.inf
        for joint in self.finger_joint_ids:
            if model.jnt_limited[joint]:
                upper = min(upper, model.jnt_range[joint, 1])
        if model.actuator_ctrllimited[actuator]:
            limits = np.sort(model.actuator_ctrlrange[actuator] / self.gear)
            lower, upper = max(lower, limits[0]), min(upper, limits[1])
        self._control_limits = np.array([lower, upper]) * len(self.finger_joint_ids)
        self._home_target = 0.0
        self._target = 0.0
        self._active = False

    def get_target(self) -> float:
        """Stored opening-width target in meters; active after set_target()."""
        return self._target

    def get_control_limits(self) -> np.ndarray:
        """Return opening-width target limits in meters."""
        return self._control_limits.copy()

    def target_for_mode(self, mode: Literal[0, 1]) -> float:
        """Resolve a validated recipe mode: 1 closes and 0 fully opens."""
        return float(self._control_limits[0 if mode == 1 else 1])

    def get_width(self) -> float:
        """Measured opening from the closed configuration, excluding any fixed finger gap."""
        return max(0.0, float(self._data.qpos[self._finger_qpos_indices].sum()))

    def get_actuator_target(self) -> float:
        """Convert the stored opening-width target to a native actuator input."""
        return self._target / len(self.finger_joint_ids) * self.gear

    def has_contact_on_all_fingers(self, contacts: set[int]) -> bool:
        """Require an object contact with a collision geom on every configured finger."""
        return bool(self.finger_geom_ids) and all(
            group & contacts for group in self.finger_geom_ids
        )

    def apply_initial_width(self, width: float) -> None:
        """Initialize all finger positions and their servo target without stepping physics."""
        self.set_target(width)
        self._data.qpos[self._finger_qpos_indices] = self._target / len(self.finger_joint_ids)
        self._data.ctrl[self.actuator_id] = self.get_actuator_target()
        self._active = False

    def is_active(self) -> bool:
        return self._active

    def set_target(self, width: float) -> None:
        """Store an opening-width target in meters, checked against its physical range."""
        value = float(width)
        lower, upper = self._control_limits
        if not np.isfinite(value):
            raise ValueError("gripper width is outside the control range")
        if not lower <= value <= upper:
            # Trajectory interpolation can exceed an endpoint by floating-point roundoff.
            bounded = float(np.clip(value, lower, upper))
            tolerance = 8 * np.finfo(float).eps * max(1.0, abs(value), abs(bounded))
            if abs(value - bounded) > tolerance:
                raise ValueError("gripper width is outside the control range")
            value = bounded
        self._target = value
        self._active = True

    def _capture_home(self) -> None:
        self._home_target = (
            self._data.ctrl[self.actuator_id] / self.gear * len(self.finger_joint_ids)
        )
        self.reset()

    def reset(self) -> None:
        self._target = self._home_target
        self._active = False
