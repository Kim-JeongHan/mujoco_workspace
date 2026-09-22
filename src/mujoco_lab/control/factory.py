"""Connect controller algorithms to robot-owned joint actuators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from mujoco_lab.control.base import Controller
from mujoco_lab.control.osc import OperationalSpaceControl
from mujoco_lab.control.pd import JointSpacePD
from mujoco_lab.control.position import PositionController
from mujoco_lab.control.target import ControlTarget
from mujoco_lab.state import JointState, RobotState

if TYPE_CHECKING:
    from mujoco_lab.robot import Robot

CONTROLLER_NAMES = ("none", "position", "pd", "osc")

# Presets are robot configuration, not a restriction on the PD algorithm.
PD_GAINS = {
    "forte": {
        "shoulder_yaw": (147.0, 11.8),
        "shoulder_pitch": (222.0, 17.8),
        "shoulder_roll": (47.0, 3.8),
        "elbow_pitch": (74.0, 5.9),
        "lower_arm_roll": (6.8, 0.55),
        "wrist_pitch": (7.2, 0.57),
        "wrist_roll": (3.2, 0.25),
    },
}


class _ActuatorState:
    """Project state and dynamics into the selected actuator order."""

    def __init__(self, robot: Robot, mode: str):
        self.robot_state = state = robot.state
        model = robot.model
        self.actuator_ids = actuators = np.array(robot.actuator_ids)
        if not len(actuators):
            raise ValueError("Controller requires joint actuators")
        if not np.all(model.actuator_trntype[actuators] == mujoco.mjtTrn.mjTRN_JOINT):
            raise ValueError("Controller requires direct joint transmissions")
        joints = model.actuator_trnid[actuators, 0]
        if len(set(joints)) != len(joints) or any(j not in state.joint_ids for j in joints):
            raise ValueError("Controller requires one actuator per robot-owned joint")
        if not np.all(
            np.isin(
                model.jnt_type[joints],
                [
                    int(mujoco.mjtJoint.mjJNT_HINGE),
                    int(mujoco.mjtJoint.mjJNT_SLIDE),
                ],
            )
        ):
            raise ValueError("Controller supports hinge and slide joints only")
        if not (
            np.all(model.actuator_dyntype[actuators] == mujoco.mjtDyn.mjDYN_NONE)
            and np.all(model.actuator_gaintype[actuators] == mujoco.mjtGain.mjGAIN_FIXED)
        ):
            raise ValueError("Controller requires stateless fixed-gain actuators")
        self.gear = model.actuator_gear[actuators, 0].copy()
        gains = model.actuator_gainprm[actuators, 0]
        if (
            np.any(self.gear == 0)
            or np.any(gains == 0)
            or not np.allclose(model.actuator_gear[actuators, 1:], 0)
        ):
            raise ValueError("Controller requires nonzero scalar joint transmission gains")
        bias = model.actuator_biasprm[actuators]
        position = (
            (model.actuator_biastype[actuators] == mujoco.mjtBias.mjBIAS_AFFINE)
            & (gains > 0)
            & np.isclose(bias[:, 0], 0)
            & np.isclose(bias[:, 1], -gains)
            & (bias[:, 2] <= 0)
        )
        motor = model.actuator_biastype[actuators] == mujoco.mjtBias.mjBIAS_NONE
        if not np.all(position | motor):
            raise ValueError("Controller requires motor or position-servo actuators")
        if mode == "position":
            if not np.all(position):
                raise ValueError("PositionController requires position-servo actuators")
            self.slots = np.arange(len(actuators))
            self.command_scale = self.gear.copy()
        else:
            self.slots = np.flatnonzero(motor)
            if not len(self.slots):
                raise ValueError(
                    "PD/OSC requires torque/force actuators; use controller='position'"
                )
            self.command_scale = 1 / (self.gear[self.slots] * gains[self.slots])
        self.auxiliary_slots = np.setdiff1d(np.arange(len(actuators)), self.slots)
        self.joint_ids = joints[self.slots]
        self.joint_names = tuple(
            state.joint_names[state.joint_ids.index(joint)] for joint in self.joint_ids
        )
        self.qpos_indices = model.jnt_qposadr[self.joint_ids]
        self._qpos = [state.qpos_indices.index(index) for index in self.qpos_indices]
        self._dofs = [state.dof_indices.index(model.jnt_dofadr[joint]) for joint in self.joint_ids]
        self.nv = len(self.joint_ids)
        self.kp = gains[self.slots] * self.gear[self.slots] ** 2

    def snapshot(self, state: JointState) -> JointState:
        return JointState(
            state.time,
            state.qpos[self._qpos],
            state.qvel[self._dofs],
            state.bias_forces[self._dofs],
        )

    def get_frame_position(self, frame: str) -> np.ndarray:
        return self.robot_state.get_frame_position(frame)

    def get_jacobian(self, frame: str) -> np.ndarray:
        return self.robot_state.get_jacobian(frame)[:, self._dofs]

    def get_mass_matrix(self) -> np.ndarray:
        return self.robot_state.get_mass_matrix()[np.ix_(self._dofs, self._dofs)]


class ActuatorController(Controller):
    """Map an algorithm's joint outputs to native inputs, retaining auxiliary servos.

    PD/OSC control motor-driven joints. Other position servos retain their home
    commands; one auxiliary servo can be commanded with set_gripper_target().
    PositionController targets include all actuated joint positions, including
    a driven gripper joint, in the robot's actuator order.
    """

    def __init__(self, algorithm: Controller, state: _ActuatorState, robot: Robot, frame: str):
        self.algorithm = algorithm
        self.robot_state = robot.state
        self.state = state
        self.frame = frame
        self.name = algorithm.name
        self._initial_commands = robot.data.ctrl[robot.actuator_ids].copy()
        home = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_KEY, robot.prefix + "home")
        if home >= 0:
            self._initial_commands = robot.model.key_ctrl[home, robot.actuator_ids].copy()
        self._commands = self._initial_commands.copy()
        self._limits = robot.model.actuator_ctrlrange[robot.actuator_ids].copy()
        self._limited = robot.model.actuator_ctrllimited[robot.actuator_ids].copy()
        self.algorithm.bind(state)

    @property
    def tracking_error(self):
        return self.algorithm.tracking_error

    @tracking_error.setter
    def tracking_error(self, value):
        self.algorithm.tracking_error = value

    def __getattr__(self, name):
        return getattr(self.algorithm, name)

    @property
    def task_kp(self):
        return self.algorithm.task_kp

    @task_kp.setter
    def task_kp(self, value):
        self.algorithm.task_kp = value

    @property
    def gripper_target(self) -> float:
        if len(self.state.auxiliary_slots) != 1:
            raise ValueError("No single auxiliary position actuator is bound")
        slot = self.state.auxiliary_slots[0]
        return float(self._commands[slot] / self.state.gear[slot])

    def set_gripper_target(self, displacement: float) -> None:
        """Set the single auxiliary servo's joint position, in meters or radians."""
        if len(self.state.auxiliary_slots) != 1:
            raise ValueError("No single auxiliary position actuator is bound")
        slot = self.state.auxiliary_slots[0]
        value = float(displacement) * self.state.gear[slot]
        lower, upper = self._limits[slot]
        if not np.isfinite(value) or (self._limited[slot] and not lower <= value <= upper):
            raise ValueError("gripper target is outside the actuator's control range")
        self._commands[slot] = value

    def bind(self, robot_state: RobotState) -> None:
        if robot_state is not self.robot_state:
            raise ValueError("A controller instance can belong to only one Robot")
        super().bind(robot_state)

    def initial_target(self, state: JointState) -> ControlTarget:
        return self.algorithm.initial_target(self.state.snapshot(state))

    def compute(self, state: JointState, target: ControlTarget) -> np.ndarray:
        values = self.algorithm.compute(self.state.snapshot(state), target)
        command = self._commands.copy()
        command[self.state.slots] = values * self.state.command_scale
        return command

    def reset(self) -> None:
        self.algorithm.reset()
        self._commands[:] = self._initial_commands

    def summary(self) -> str:
        return self.algorithm.summary()


def create_controller(
    name: str,
    robot: Robot,
    *,
    kp=None,
    kd=None,
    frame: str = "ee_site",
    gravity_compensation=True,
) -> Controller | None:
    """Bind by actuator capabilities; custom torque robots supply PD gains explicitly."""
    if name not in CONTROLLER_NAMES:
        raise ValueError(f"Unknown controller {name!r}. Available controllers: {CONTROLLER_NAMES}")
    if name == "none":
        return None
    state = _ActuatorState(robot, name)
    if name == "position":
        algorithm = PositionController(state.kp, gravity_compensation=gravity_compensation)
    elif name == "pd":
        if kp is None or kd is None:
            preset = PD_GAINS.get(robot.robot_type, {})
            if not all(joint in preset for joint in state.joint_names):
                raise ValueError("Provide kp and kd for this robot's controlled joints")
            if kp is None:
                kp = [preset[joint][0] for joint in state.joint_names]
            if kd is None:
                kd = [preset[joint][1] for joint in state.joint_names]
        algorithm = JointSpacePD(kp, kd, gravity_compensation=gravity_compensation)
    else:
        posture = state.snapshot(robot.joint_state).qpos
        algorithm = OperationalSpaceControl(state, frame=frame, posture=posture)
    controller = ActuatorController(algorithm, state, robot, frame)
    controller.bind(robot.state)
    return controller
