"""Robot assets, state/controller composition, and scoped actuator inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from mujoco_lab.assets.robot.robot import Constraints, RobotConfig
from mujoco_lab.control.factory import create_controller
from mujoco_lab.gripper import Gripper
from mujoco_lab.state import JointState, RobotState
from mujoco_lab.trajectory_generation.references.base import ReferenceGenerator
from mujoco_lab.trajectory_generation.references.factory import create_reference_generator
from mujoco_lab.trajectory_generation.target import ControlTarget
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
    """Own one robot's controller, constrained reference, and actuator inputs.

    RobotState owns state reads and dynamics. Simulator owns physics advancement.
    """

    ####### initialization #######

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
        arm_ids = [self.actuator_ids[slot] for slot in self._arm_actuator_slots]
        model = self.model
        transmissions = model.actuator_trntype[arm_ids]
        self._arm_joint_ids = model.actuator_trnid[arm_ids, 0]
        self._arm_gear = model.actuator_gear[arm_ids, 0]
        self._arm_gain = model.actuator_gainprm[arm_ids, 0]
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

        # Check joint mapping and the fixed actuator scale used by controllers.
        joints = self._arm_joint_ids
        if not np.all(transmissions == mujoco.mjtTrn.mjTRN_JOINT):
            raise ValueError("Controller requires direct joint transmissions")
        if self.gripper is not None and self.gripper.joint_id in joints:
            raise ValueError("Arm actuators cannot also drive the configured gripper joint")
        if len(set(joints)) != len(joints):
            raise ValueError("Controller requires one actuator per joint")
        if not np.all(model.actuator_gaintype[arm_ids] == mujoco.mjtGain.mjGAIN_FIXED):
            raise ValueError("Controller requires fixed-gain actuators")
        if np.any(self._arm_gear == 0) or np.any(self._arm_gain == 0):
            raise ValueError("Controller requires nonzero joint transmission gains")

        # Arm joint-to-state mapping.
        self._arm_joint_slots = [self.state.joint_ids.index(int(joint)) for joint in joints]
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
        self.controller: Controller
        self.joint_state = self.state.snapshot()
        self._target: ControlTarget
        self.reference_generator: ReferenceGenerator
        self._apply_initial_pose()
        if self.gripper is not None:
            self.gripper._capture_home()
        mujoco.mj_forward(self.model, self.data)
        self.update_state()
        self.change_controller(create_controller(self, config.controller))

    ####### public methods: controller configuration #######

    def change_controller(self, controller: Controller) -> None:
        """Bind a replacement controller and reset its target and reference state."""
        self._simulator._require_idle()
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
        self.controller = controller
        self.control_joint_names = names
        self.control_qpos_indices = qpos_indices
        self.control_joint_slots = joint_slots
        self.control_actuator_slots = slots
        self.control_scale = scale
        self._target = target
        self.reference_generator = create_reference_generator(
            self.config.reference,
            target,
            self.get_reference_constraints(),
            control_dt=self._simulator.dt,
        )

    def get_arm_joint_mapping(self) -> tuple[tuple[str, ...], list[int]]:
        """Return the attached controller's joint mapping."""
        return self.control_joint_names, self.control_joint_slots.copy()

    def get_control_target_order(self, action_names: tuple[str, ...]) -> list[int]:
        """Map stored joint action order into the attached controller's order."""
        if set(self.control_joint_names) != set(action_names):
            raise ValueError(f"Robot {self.name!r} controller joint mapping changed")
        return [action_names.index(name) for name in self.control_joint_names]

    ####### public methods: control #######

    @property
    def target(self) -> ControlTarget:
        """Return the current controller target, also used as the reference snapshot."""
        return self._target

    @target.setter
    def target(self, target: ControlTarget) -> None:
        """Replace the sampled target and synchronize the reference state."""
        self._target = target
        self.reference_generator.reset(target)

    def set_goal(self, action: ArrayLike, *, joint_names: tuple[str, ...] | None = None) -> None:
        """Bound a physical joint action once and retain its goal until the next action.

        Values are arm positions followed by optional gripper opening width.
        An explicit joint order maps stored actions into the current control order.
        """
        values = np.asarray(action, dtype=float)
        names = self.control_joint_names if joint_names is None else joint_names
        expected = len(names) + int(self.has_gripper())
        if values.shape != (expected,):
            raise ValueError(f"Robot {self.name!r} goal must contain {expected} values")

        order = self.get_control_target_order(names)
        arm = values[: len(names)][order]
        values = np.r_[arm, values[-1]] if self.has_gripper() else arm
        limits = self.get_goal_limits()
        bounded = np.clip(values, limits[:, 0], limits[:, 1])
        self.reference_generator.set_goal(ControlTarget(bounded[: len(names)]))
        if self.gripper is not None:
            self.gripper.set_target(float(bounded[-1]))

    def update_target(self, target: ControlTarget, gripper_target: float | None = None) -> None:
        """Replace reference state and target; use set_goal() for physical joint actions."""
        self.target = target
        if self.gripper is not None and gripper_target is not None:
            self.gripper.set_target(gripper_target)

    def control(self) -> bool:
        """Apply selected joint commands and gripper target; return saturation status.

        Control inputs use each actuator's native units. Only actuators with
        enabled control limits are clipped. This method never steps physics.
        """
        command = self.data.ctrl[self.actuator_ids]
        values = self.controller.compute(self.get_control_state(), self.target)
        required = len(self.control_actuator_slots)
        if values.shape != (required,):
            raise ValueError(f"Robot {self.name!r} needs {required} joint commands")
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

    ####### public methods: reset #######

    def reset(self) -> None:
        """Reset control state from current physics without restoring or stepping the scene."""
        self.controller.reset()
        if self.gripper is not None:
            self.gripper.reset()
        self.update_state()
        self._target = self.controller.initial_target(self.get_control_state())
        self.reset_reference()

    ####### public methods: reference generation #######

    def get_goal_limits(self) -> np.ndarray:
        """Return physical arm position and optional gripper width bounds in control order."""
        _, slots = self.get_arm_joint_mapping()
        limits = self.state.get_joint_limits(slots)
        if self.gripper is not None:
            limits = np.vstack((limits, self.gripper.get_control_limits()))
        return limits

    def get_reference_constraints(self) -> Constraints:
        """Select configured motion limits in controller joint order."""
        constraints = self.state.constraints
        indices = [constraints.joint_names.index(name) for name in self.control_joint_names]
        return Constraints(
            joint_names=list(self.control_joint_names),
            position_limit=[constraints.position_limit[index] for index in indices],
            velocity_limit=[constraints.velocity_limit[index] for index in indices],
            acceleration_limit=[constraints.acceleration_limit[index] for index in indices],
        )

    def set_reference_generator(self, reference: ReferenceGenerator) -> None:
        """Install an experiment's reference and its current target without advancing time."""
        self._simulator._require_idle()
        target = reference.state()
        self.reference_generator = reference
        self._target = target

    def step_reference(self) -> None:
        """Advance the owned reference once before control for one physics tick."""
        self._target = self.reference_generator.compute()

    def reset_reference(self) -> None:
        """Reset the generator from the restored controller target."""
        self.reference_generator.reset(self.target)
        self._target = self.reference_generator.state()

    ####### public methods: state access #######

    def update_state(self) -> None:
        """Refresh the owned joint snapshot without forwarding or stepping."""
        self.joint_state = self.state.snapshot()

    def get_control_state(self) -> JointState:
        """Use the cached robot snapshot in the attached controller's joint order."""
        return self.joint_state.select(self.control_joint_slots)

    def get_tracking_error(self):
        return self.controller.get_tracking_error()

    def has_gripper(self) -> bool:
        """Return whether this robot has a configured gripper."""
        return self.gripper is not None

    ####### private methods: initialization #######

    def _apply_initial_pose(self) -> None:
        """Initialize arm coordinates and delegate opening width to the gripper."""
        gripper = self.gripper
        fingers = gripper.finger_joint_ids if gripper is not None else ()
        slots = [slot for slot, joint in enumerate(self.state.joint_ids) if joint not in fingers]
        pose = np.asarray(self.config.pose.default, dtype=float)
        expected = len(slots) + int(self.has_gripper())
        if pose.shape != (expected,) or not np.isfinite(pose).all():
            raise ValueError(
                f"Robot {self.name!r} default pose must contain {expected} finite values"
            )
        arm_pose = pose[: len(slots)]
        limits = self.state.get_joint_limits(slots)
        if np.any(arm_pose < limits[:, 0]) or np.any(arm_pose > limits[:, 1]):
            raise ValueError(f"Robot {self.name!r} default arm pose exceeds joint limits")
        qpos_indices = [self.state.qpos_indices[slot] for slot in slots]
        self.data.qpos[qpos_indices] = arm_pose
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
        actuators = [self.actuator_ids[slot] for slot in slots]
        self.data.ctrl[actuators] = self.data.qpos[self.model.jnt_qposadr[joints]] * gears

    ####### private methods: controller configuration #######

    def _joint_actuators(self, output_kind: str) -> np.ndarray:
        """Select arm actuator positions for the requested controller output."""
        if output_kind not in ("position", "torque"):
            raise ValueError(f"Unknown joint controller output kind {output_kind!r}")
        if output_kind == "position":
            if not np.all(self._arm_position):
                raise ValueError("PositionController requires position-servo actuators")
            return np.arange(len(self._arm_actuator_slots))
        selected = np.flatnonzero(self._arm_motor)
        if not len(selected):
            raise ValueError(
                "Controller requires torque/force actuators; use controller='position'"
            )
        return selected
