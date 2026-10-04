"""Create robot controller algorithms configured for owned joint actuators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mujoco_lab.assets.robot.robot import ControllerConfig
from mujoco_lab.control.base import Controller
from mujoco_lab.control.osc import OperationalSpaceControl
from mujoco_lab.control.pd import JointSpacePD
from mujoco_lab.control.position import PositionController

if TYPE_CHECKING:
    from mujoco_lab.robot import Robot


def create_controller(robot: Robot, config: ControllerConfig) -> Controller | None:
    """Create a controller for owned actuators from an already loaded controller config."""
    name = config.name
    if name == "none":
        return None
    selected = robot._joint_actuators("position" if name == "position" else "torque")
    if name == "position":
        algorithm = PositionController(
            robot._arm_gain[selected] * robot._arm_gear[selected] ** 2,
            gravity_compensation=config.gravity_compensation,
        )
    elif name == "pd":
        joint_names = tuple(robot._arm_joint_names[index] for index in selected)
        kp, kd = config.pd_gain(joint_names)
        algorithm = JointSpacePD(
            kp,
            kd,
            gravity_compensation=config.gravity_compensation,
            frame=config.frame,
            control_dt=robot._simulator.dt,
        )
    else:
        joint_slots = [robot._arm_joint_slots[index] for index in selected]
        posture = robot.joint_state.qpos[joint_slots]
        algorithm = OperationalSpaceControl(
            robot.state,
            frame=config.frame,
            posture=posture,
            dof_slots=joint_slots,
            gravity_compensation=config.gravity_compensation,
        )
    if name != "osc":
        algorithm.bind(robot.state)
    return algorithm
