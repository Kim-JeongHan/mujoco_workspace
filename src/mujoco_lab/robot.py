"""Robot assets, state/controller composition, and scoped actuator inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from mujoco_lab.assets.loader import AssetInfo
from mujoco_lab.control.target import ControlTarget
from mujoco_lab.gripper import Gripper
from mujoco_lab.state import JointState, RobotState
from mujoco_lab.utils import Transform

if TYPE_CHECKING:
    from mujoco_lab.control.base import Controller
    from mujoco_lab.simulation import Simulator


@dataclass(frozen=True)
class RobotSpec:
    """Place a named asset in world coordinates; a singleton may use the mount."""

    name: str
    robot_type: str
    pose: Transform | None = None
    gripper_actuator: str | None = None


class Robot:
    """Connect one robot's state and controller to its actuator inputs.

    RobotState owns state reads and dynamics. Simulator owns physics advancement.
    """

    def __init__(
        self,
        simulator: Simulator,
        info: AssetInfo,
        prefix: str,
        robot_type: str,
    ) -> None:
        # Simulator ownership and robot identity.
        self._simulator = simulator
        self.model, self.data = simulator.model, simulator.data
        self.name = prefix.removesuffix("/")
        self.robot_type = robot_type
        self.prefix = prefix

        # Robot state access.
        self.state = RobotState(
            self.model,
            self.data,
            name=self.name,
            prefix=prefix,
            root_name=info.root_name,
            joint_names=info.joint_names,
            site_names=info.site_names,
        )

        # Actuator and gripper access.
        self.actuator_names = info.actuator_names
        self.actuator_ids = [self.model.actuator(prefix + name).id for name in info.actuator_names]
        self.num_actuators = len(self.actuator_ids)
        self.gripper = (
            Gripper(self, info.gripper_actuator) if info.gripper_actuator is not None else None
        )

        # Arm actuator mapping and properties.
        self._arm_actuator_slots = np.asarray(
            [slot for slot in range(self.num_actuators) if self.gripper is None or slot != self.gripper.slot],
            dtype=int,
        )
        arm_ids = np.asarray(self.actuator_ids, dtype=int)[self._arm_actuator_slots]
        model = self.model
        self._arm_transmissions = model.actuator_trntype[arm_ids]
        self._arm_joint_ids = model.actuator_trnid[arm_ids, 0]
        self._arm_gear = model.actuator_gear[arm_ids, 0].copy()
        self._arm_gain = model.actuator_gainprm[arm_ids, 0].copy()
        self._arm_fixed_gain = model.actuator_gaintype[arm_ids] == mujoco.mjtGain.mjGAIN_FIXED
        self._arm_scalar_gear = np.allclose(model.actuator_gear[arm_ids, 1:], 0)
        bias = model.actuator_biasprm[arm_ids]
        self._arm_position = (
            (model.actuator_biastype[arm_ids] == mujoco.mjtBias.mjBIAS_AFFINE)
            & (self._arm_gain > 0)
            & np.isclose(bias[:, 0], 0)
            & np.isclose(bias[:, 1], -self._arm_gain)
            & (bias[:, 2] <= 0)
        )
        self._arm_motor = model.actuator_biastype[arm_ids] == mujoco.mjtBias.mjBIAS_NONE

        # Arm joint-to-state mapping.
        names, qpos_indices, joint_slots = [], [], []
        for transmission, joint in zip(self._arm_transmissions, self._arm_joint_ids, strict=True):
            joint = int(joint)
            if transmission == mujoco.mjtTrn.mjTRN_JOINT and joint in self.state.joint_ids:
                qpos_index = int(model.jnt_qposadr[joint])
                names.append(self.state.joint_names[self.state.joint_ids.index(joint)])
                qpos_indices.append(qpos_index)
                joint_slots.append(self.state.joint_ids.index(joint))
            else:
                names.append("")
                qpos_indices.append(-1)
                joint_slots.append(-1)
        self._arm_joint_names = tuple(names)
        self._arm_qpos_indices = qpos_indices
        self._arm_joint_slots = joint_slots

        # Active controller joint mapping.
        self.control_joint_names: tuple[str, ...] = ()
        self.control_qpos_indices: list[int] = []
        self.control_joint_slots: list[int] = []
        self.control_actuator_slots = np.empty(0, dtype=int)
        self.control_scale = np.empty(0)

        # Attached controller and cached control state.
        self.controller: Controller | None = None
        self.joint_state = self.state.snapshot()
        self.target: ControlTarget | None = None

    def _joint_actuators(self, output_kind: str) -> np.ndarray:
        """Validate the requested joint controller and select arm actuator positions."""
        if output_kind not in ("position", "torque"):
            raise ValueError(f"Unknown joint controller output kind {output_kind!r}")
        if not len(self._arm_actuator_slots):
            raise ValueError("Controller requires joint actuators")
        if not np.all(self._arm_transmissions == mujoco.mjtTrn.mjTRN_JOINT):
            raise ValueError("Controller requires direct joint transmissions")
        joints = self._arm_joint_ids
        if self.gripper is not None and self.gripper.joint_id in joints:
            raise ValueError("Arm actuators cannot also drive the configured gripper joint")
        if len(set(joints)) != len(joints) or any(j not in self.state.joint_ids for j in joints):
            raise ValueError("Controller requires one actuator per robot-owned joint")
        if not np.all(self._arm_fixed_gain):
            raise ValueError("Controller requires fixed-gain actuators")
        if np.any(self._arm_gear == 0) or np.any(self._arm_gain == 0) or not self._arm_scalar_gear:
            raise ValueError("Controller requires nonzero scalar joint transmission gains")
        if output_kind == "position":
            if not np.all(self._arm_position):
                raise ValueError("PositionController requires position-servo actuators")
            return np.arange(len(joints))
        if not np.all(self._arm_position | self._arm_motor):
            raise ValueError("Controller requires motor or position-servo actuators")
        selected = np.flatnonzero(self._arm_motor)
        if not len(selected):
            raise ValueError("PD/OSC requires torque/force actuators; use controller='position'")
        return selected

    def get_control_state(self) -> JointState:
        """Use the cached robot snapshot in the attached controller's joint order."""
        if self.controller is None:
            return self.joint_state
        return self.joint_state.select(self.control_joint_slots)

    def get_arm_joint_mapping(self) -> tuple[tuple[str, ...], list[int]]:
        """Return controlled joints, or valid unique arm actuator joints before binding."""
        if self.controller is not None:
            return self.control_joint_names, self.control_joint_slots.copy()
        names, slots = [], []
        seen = set()
        for name, slot in zip(self._arm_joint_names, self._arm_joint_slots, strict=True):
            if not name or slot < 0 or slot in seen:
                continue
            if self.gripper is not None and self.state.joint_ids[slot] == self.gripper.joint_id:
                continue
            names.append(name)
            slots.append(slot)
            seen.add(slot)
        return tuple(names), slots

    def get_control_target_order(self, action_names: tuple[str, ...]) -> list[int]:
        """Map stored joint action order into the attached controller's order."""
        if action_names and self.controller is None:
            raise ValueError(f"Robot {self.name!r} needs a joint-target controller")
        if set(self.control_joint_names) != set(action_names):
            raise ValueError(f"Robot {self.name!r} controller joint mapping changed")
        return [action_names.index(name) for name in self.control_joint_names]

    def _apply_home_keyframe(self) -> None:
        """Apply this robot's home state without forwarding the shared scene."""
        home = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, self.prefix + "home")
        if home < 0:
            return
        key = self.model.key(home)
        self.data.qpos[self.state.qpos_indices] = key.qpos[self.state.qpos_indices]
        self.data.qvel[self.state.dof_indices] = key.qvel[self.state.dof_indices]
        self.data.ctrl[self.actuator_ids] = key.ctrl[self.actuator_ids]

    def change_controller(self, controller: Controller | None) -> None:
        """Bind and reset a replacement controller, or detach it with None."""
        self._simulator._require_idle()
        if controller is self.controller:
            return
        names = ()
        qpos_indices = []
        slots = np.empty(0, dtype=int)
        scale = np.empty(0)
        joint_slots = []
        if controller is not None:
            kind = controller.output_kind
            selected = self._joint_actuators(kind)
            slots = self._arm_actuator_slots[selected]
            scale = (
                self._arm_gear[selected]
                if kind == "position"
                else 1 / (self._arm_gear[selected] * self._arm_gain[selected])
            )
            names = tuple(self._arm_joint_names[index] for index in selected)
            qpos_indices = [self._arm_qpos_indices[index] for index in selected]
            joint_slots = [self._arm_joint_slots[index] for index in selected]
            controller.bind(self.state)
            controller.reset()
            self.update_state()
            target = controller.initial_target(self.joint_state.select(joint_slots))
        else:
            target = None
        self.controller = controller
        self.control_joint_names = names
        self.control_qpos_indices = qpos_indices
        self.control_joint_slots = joint_slots
        self.control_actuator_slots = slots
        self.control_scale = scale
        self.target = target

    def initial_target(self) -> ControlTarget | None:
        """Make the attached controller's target from the cached robot snapshot."""
        if self.controller is None:
            return None
        return self.controller.initial_target(self.get_control_state())

    def update_state(self) -> None:
        """Refresh and return the owned joint snapshot without forwarding or stepping."""
        self.joint_state = self.state.snapshot()

    def control(self) -> bool:
        """Apply selected joint commands and gripper target; return saturation status.

        Control inputs use each actuator's native units. Only actuators with
        enabled control limits are clipped. This method never steps physics.
        """
        command = self.data.ctrl[self.actuator_ids].copy()
        if self.controller is not None:
            values = np.asarray(
                self.controller.compute(self.get_control_state(), self.target), dtype=float
            )
            required = len(self.control_actuator_slots)
            if values.shape != (required,):
                raise ValueError(f"Robot {self.name!r} needs {required} finite joint commands")
            command[self.control_actuator_slots] = values * self.control_scale
        if not np.isfinite(command).all():
            raise ValueError(f"Robot {self.name!r} needs {self.num_actuators} finite actuator inputs")
        if self.gripper is not None and self.gripper.is_active():
            command[self.gripper.slot] = self.gripper.get_target() * self.gripper.gear
        limits = self.model.actuator_ctrlrange[self.actuator_ids]
        limited = self.model.actuator_ctrllimited[self.actuator_ids]
        clipped = np.where(limited, np.clip(command, limits[:, 0], limits[:, 1]), command)

        # update control information into mujoco
        self.data.ctrl[self.actuator_ids] = clipped
        return bool(np.any(clipped != command))

    def get_tracking_error(self):
        return None if self.controller is None else self.controller.get_tracking_error()

    def has_gripper(self):
        return self.gripper is not None
