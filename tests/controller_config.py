"""Select controller modes through parsed settings for controller-specific tests."""

from mujoco_lab.assets import ROBOT_ASSETS
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.assets.robot.robot import ControllerConfig
from mujoco_lab.control import create_controller


def create_test_controller(robot, controller, *, frame="grasp", gravity_compensation=None):
    path = ROBOT_ASSETS[robot.robot_type].path.with_name("robot.yaml")
    config = (
        load_robot_config(robot.robot_type).controller
        if path.is_file()
        else ControllerConfig(name=controller)
    )
    config.name = controller
    config.frame = frame
    if gravity_compensation is not None:
        config.gravity_compensation = gravity_compensation
    return create_controller(robot, config)
