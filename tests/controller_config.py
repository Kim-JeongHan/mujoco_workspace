"""Select controller modes through parsed settings for controller-specific tests."""

from mujoco_lab.assets import ROBOT_ASSETS, ROBOT_PATH
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.assets.robot.robot import ControllerConfig, RobotConfig
from mujoco_lab.control import create_controller


def create_test_controller(robot, controller, *, frame="grasp", gravity_compensation=None):
    path = ROBOT_ASSETS[robot.robot_type].path.with_name("robot.yaml")
    config = (
        load_robot_config(robot.robot_type).controller
        if path.is_file()
        else ControllerConfig(name=controller)
    )
    if robot.robot_type == "forte" and controller == "pd" and config.pd_gains is None:
        # Keep explicit PD tests independent of the live CTC default.
        config.pd_gains = RobotConfig.load(
            ROBOT_PATH / "forte_backup" / "robot.yaml"
        ).controller.pd_gains
    config.name = controller
    config.frame = frame
    if gravity_compensation is not None:
        config.gravity_compensation = gravity_compensation
    return create_controller(robot, config)
