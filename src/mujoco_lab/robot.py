"""Robot assets, state/controller composition, and scoped actuator inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from mujoco_lab.assets.robot.robot import RobotConfig
from mujoco_lab.control.target import ControlTarget
from mujoco_lab.gripper import Gripper
from mujoco_lab.state import JointState, RobotState
from mujoco_lab.utils import Transform

if TYPE_CHECKING:
    from mujoco_lab.control.base import Controller
    from mujoco_lab.simulation import Simulator


@dataclass(frozen=True)
class RobotSpec:
    """Place a named asset with explicit robot settings.

    A singleton may use the scene's robot mounting site instead of an explicit pose.
    """

    name: str
    robot_type: str
    pose: Transform | None = None
    config: RobotConfig = field(kw_only=True)


class Robot:
    """Connect one robot's state and controller to its actuator inputs.

    RobotState owns state reads and dynamics. Simulator owns physics advancement.
    """

    def __init__(
        self,
        simulator: Simulator,
        config: RobotConfig,
        prefix: str,
        robot_type: str,
    ) -> None:
        # Simulator ownership and robot identity.
        self._simulator = simulator
        self.model, self.data = simulator.model, simulator.data
        self.name = prefix.removesuffix("/")
        self.robot_type = robot_type
        self.prefix = prefix
        self.config = config
        info = config.model_info

        # Robot state access.
        self.state = RobotState(
            self.model,
            self.data,
            name=self.name,
            prefix=prefix,
            root_name=info.root_name,
            joint_names=info.joint_names,
            site_names=info.site_names,
            constraints=config.constraints,
            ik_frame=config.controller.frame,
        )

        # Actuator and gripper access.
        self.actuator_names = info.actuator_names
        self.actuator_ids = [self.model.actuator(prefix + name).id for name in info.actuator_names]
        self.num_actuators = len(self.actuator_ids)
        self.gripper = (
            Gripper(self, config.gripper.actuator, config.gripper.joints)
            if config.gripper
            else None
        )

        # Arm actuator mapping and properties.
        self._arm_actuator_slots = np.asarray(
            [
                slot
                for slot in range(self.num_actuators)
                if self.gripper is None or slot != self.gripper.slot
            ],
            dtype=int,
        )
        arm_ids = np.asarray(self.actuator_ids, dtype=int)[self._arm_actuator_slots]
        model = self.model
        transmissions = model.actuator_trntype[arm_ids]
        self._arm_joint_ids = model.actuator_trnid[arm_ids, 0]
        self._arm_gear = model.actuator_gear[arm_ids, 0].copy()
        self._arm_gain = model.actuator_gainprm[arm_ids, 0].copy()
        bias = model.actuator_biasprm[arm_ids]
        self._arm_position = (
            (transmissions == mujoco.mjtTrn.mjTRN_JOINT)
            & (model.actuator_biastype[arm_ids] == mujoco.mjtBias.mjBIAS_AFFINE)
            & (self._arm_gain > 0)
            & np.isclose(bias[:, 0], 0)
            & np.isclose(bias[:, 1], -self._arm_gain)
            & (bias[:, 2] <= 0)
        )
        self._arm_motor = model.actuator_biastype[arm_ids] == mujoco.mjtBias.mjBIAS_NONE

        # Check the fixed model once; controller-free native inputs remain available.
        joints = self._arm_joint_ids
        self._arm_controller_error: str | None = None
        if not len(arm_ids):
            self._arm_controller_error = "Controller requires joint actuators"
        elif not np.all(transmissions == mujoco.mjtTrn.mjTRN_JOINT):
            self._arm_controller_error = "Controller requires direct joint transmissions"
        elif self.gripper is not None and self.gripper.joint_id in joints:
            self._arm_controller_error = (
                "Arm actuators cannot also drive the configured gripper joint"
            )
        elif len(set(joints)) != len(joints) or any(j not in self.state.joint_ids for j in joints):
            self._arm_controller_error = "Controller requires one actuator per robot-owned joint"
        elif not np.all(model.actuator_gaintype[arm_ids] == mujoco.mjtGain.mjGAIN_FIXED):
            self._arm_controller_error = "Controller requires fixed-gain actuators"
        elif (
            np.any(self._arm_gear == 0)
            or np.any(self._arm_gain == 0)
            or not np.allclose(model.actuator_gear[arm_ids, 1:], 0)
        ):
            self._arm_controller_error = (
                "Controller requires nonzero scalar joint transmission gains"
            )
        elif not np.all(self._arm_position | self._arm_motor):
            self._arm_controller_error = "Controller requires motor or position-servo actuators"

        # Arm joint-to-state mapping.
        self._arm_joint_slots = (
            [self.state.joint_ids.index(int(joint)) for joint in joints]
            if self._arm_controller_error is None
            else []
        )
        self._arm_joint_names = tuple(
            self.state.joint_names[slot] for slot in self._arm_joint_slots
        )
        self._arm_qpos_indices = [self.state.qpos_indices[slot] for slot in self._arm_joint_slots]

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
        """Select arm actuator positions for the requested controller output."""
        if output_kind not in ("position", "torque"):
            raise ValueError(f"Unknown joint controller output kind {output_kind!r}")
        if self._arm_controller_error is not None:
            raise ValueError(self._arm_controller_error)
        if output_kind == "position":
            if not np.all(self._arm_position):
                raise ValueError("PositionController requires position-servo actuators")
            return np.arange(len(self._arm_actuator_slots))
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
        """Return controlled joints, or the validated arm mapping before binding."""
        if self.controller is not None:
            return self.control_joint_names, self.control_joint_slots.copy()
        return self._arm_joint_names, self._arm_joint_slots.copy()

    def get_control_target_order(self, action_names: tuple[str, ...]) -> list[int]:
        """Map stored joint action order into the attached controller's order."""
        if action_names and self.controller is None:
            raise ValueError(f"Robot {self.name!r} needs a joint-target controller")
        if set(self.control_joint_names) != set(action_names):
            raise ValueError(f"Robot {self.name!r} controller joint mapping changed")
        return [action_names.index(name) for name in self.control_joint_names]

    def _apply_initial_pose(self) -> None:
        """Initialize arm coordinates and delegate opening width to the gripper."""
        gripper = self.gripper
        fingers = gripper.finger_joint_ids if gripper is not None else ()
        slots = [slot for slot, joint in enumerate(self.state.joint_ids) if joint not in fingers]
        pose = np.asarray(self.config.pose.default, dtype=float)
        expected = len(slots) + int(gripper is not None)
        if pose.shape != (expected,) or not np.isfinite(pose).all():
            raise ValueError(
                f"Robot {self.name!r} default pose must contain {expected} finite values"
            )
        arm_pose = pose[: len(slots)]
        if self.state.constraints is not None:
            limits = self.state.get_joint_limits(slots)
            if np.any(arm_pose < limits[:, 0]) or np.any(arm_pose > limits[:, 1]):
                raise ValueError(f"Robot {self.name!r} default arm pose exceeds joint limits")
        self.data.qpos[np.asarray(self.state.qpos_indices, dtype=int)[slots]] = arm_pose
        self.data.qvel[self.state.dof_indices] = 0
        self.data.ctrl[self.actuator_ids] = 0
        self._initialize_actuator_inputs()
        if gripper is not None:
            gripper.apply_initial_width(float(pose[-1]))

    def _initialize_actuator_inputs(self) -> None:
        """Hold position-servo arm joints at their initialized coordinates."""
        slots = self._arm_actuator_slots[self._arm_position]
        joints = self._arm_joint_ids[self._arm_position]
        gears = self._arm_gear[self._arm_position]
        actuators = np.asarray(self.actuator_ids, dtype=int)[slots]
        self.data.ctrl[actuators] = self.data.qpos[self.model.jnt_qposadr[joints]] * gears

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
            target = self.target
            if target is None:
                raise RuntimeError("An attached controller requires a control target")
            values = np.asarray(
                self.controller.compute(self.get_control_state(), target),
                dtype=float,
            )
            required = len(self.control_actuator_slots)
            if values.shape != (required,):
                raise ValueError(f"Robot {self.name!r} needs {required} finite joint commands")
            command[self.control_actuator_slots] = values * self.control_scale
        if not np.isfinite(command).all():
            raise ValueError(
                f"Robot {self.name!r} needs {self.num_actuators} finite actuator inputs"
            )
        if self.gripper is not None and self.gripper.is_active():
            command[self.gripper.slot] = self.gripper.get_actuator_target()
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
