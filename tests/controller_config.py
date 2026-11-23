"""Select controller modes through parsed settings for controller-specific tests."""

from mujoco_lab.assets import ROBOT_ASSETS
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.assets.robot.robot import ControllerConfig, FortePDGain, PDGain
from mujoco_lab.control import create_controller


def forte_test_pd_config() -> ControllerConfig:
    """Use the established Forte torque PD gains in controller tests."""
    return ControllerConfig(
        name="pd",
        pd_gains=FortePDGain(
            shoulder_yaw=PDGain(147.0, 11.8),
            shoulder_pitch=PDGain(222.0, 17.8),
            shoulder_roll=PDGain(47.0, 5.9),
            elbow_pitch=PDGain(74.0, 5.9),
            lower_arm_roll=PDGain(6.8, 0.55),
            wrist_pitch=PDGain(7.2, 0.57),
            wrist_yaw=PDGain(3.2, 0.25),
        ),
    )


def create_test_controller(robot, controller, *, frame="grasp", gravity_compensation=None):
    path = ROBOT_ASSETS[robot.robot_type].path.with_name("robot.yaml")
    config = (
        load_robot_config(robot.robot_type).controller
        if path.is_file()
        else ControllerConfig(name=controller)
    )
    if robot.robot_type == "forte" and controller == "pd" and config.pd_gains is None:
        # Keep explicit PD tests independent of the live CTC default.
        config.pd_gains = forte_test_pd_config().pd_gains
    config.name = controller
    config.frame = frame
    if gravity_compensation is not None:
        config.gravity_compensation = gravity_compensation
    return create_controller(robot, config)
