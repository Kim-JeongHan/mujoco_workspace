"""Rated output torques and external gears for Forte's simulation actuators.

EL05 and RS06 rated output torques are 1.8 and 11 N.m, respectively. Their
internal 9:1 reduction is already included in those output specifications.
External ratios follow the Hardware Spec working assumptions, including the
unverified 6:1 shoulder-pitch ratio. Transmission efficiency is taken as one;
the scalar limits do not model the forearm/wrist transmission coupling.

Manufacturer references:
https://files.seeedstudio.com/products/RobStride/Product%20Literature/EL05/EL05User%20Manual251112.pdf
https://files.seeedstudio.com/products/RobStride/Product%20Literature/RS06/RS06User%20Manual251112.pdf
"""

import xml.etree.ElementTree as ET

RATED_OUTPUT_TORQUE = {"EL05": 1.8, "RS06": 11.0}
ARM_ACTUATORS = {
    "shoulder_yaw": ("EL05", 1.0),
    "shoulder_pitch": ("RS06", 6.0),
    "shoulder_roll": ("RS06", 1.0),
    "elbow_pitch": ("RS06", 1.0),
    "lower_arm_roll": ("EL05", 2.0),
    "wrist_pitch": ("EL05", 2.0),
    "wrist_roll": ("EL05", 1.0),
}
RATED_JOINT_TORQUE = {
    joint: RATED_OUTPUT_TORQUE[motor] * reduction
    for joint, (motor, reduction) in ARM_ACTUATORS.items()
}
TORQUE_LIMIT_NOTE = (
    "Rated actuator-output torque limits (N.m); gear specifies external reduction. "
    "Ideal transmission; shoulder_pitch assumes 6:1. Internal 9:1 is already in the motor specs."
)


def apply_rated_arm_torque_limits(actuators: ET.Element) -> None:
    """Set rated output torques and external gears; retain the gripper's force limit."""
    actuators.insert(0, ET.Comment(TORQUE_LIMIT_NOTE))
    for actuator in actuators.findall("motor"):
        motor, reduction = ARM_ACTUATORS[actuator.attrib["joint"]]
        torque = RATED_OUTPUT_TORQUE[motor]
        actuator.set("gear", f"{reduction:g}")
        actuator.set("ctrlrange", f"{-torque:g} {torque:g}")
