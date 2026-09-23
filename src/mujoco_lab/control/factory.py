"""Create robot controller algorithms configured for owned joint actuators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mujoco_lab.control.base import Controller
from mujoco_lab.control.osc import OperationalSpaceControl
from mujoco_lab.control.pd import JointSpacePD
from mujoco_lab.control.position import PositionController

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
    selected = robot._joint_actuators("position" if name == "position" else "torque")
    if name == "position":
        algorithm = PositionController(
            robot._arm_gain[selected] * robot._arm_gear[selected] ** 2,
            gravity_compensation=gravity_compensation,
        )
    elif name == "pd":
        joint_names = tuple(robot._arm_joint_names[index] for index in selected)
        if kp is None or kd is None:
            preset = PD_GAINS.get(robot.robot_type, {})
            if not all(joint in preset for joint in joint_names):
                raise ValueError("Provide kp and kd for this robot's controlled joints")
            if kp is None:
                kp = [preset[joint][0] for joint in joint_names]
            if kd is None:
                kd = [preset[joint][1] for joint in joint_names]
        algorithm = JointSpacePD(kp, kd, gravity_compensation=gravity_compensation, frame=frame)
    else:
        joint_slots = [robot._arm_joint_slots[index] for index in selected]
        posture = robot.joint_state.qpos[joint_slots]
        algorithm = OperationalSpaceControl(
            robot.state,
            frame=frame,
            posture=posture,
            dof_slots=joint_slots,
        )
    if name != "osc":
        algorithm.bind(robot.state)
    return algorithm
