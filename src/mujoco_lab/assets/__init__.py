"""Bundled robot and environment asset locations."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RobotAsset:
    path: Path
    gripper_actuator: str | None = None


ASSET_PATH = Path(__file__).parent
ROBOT_PATH = ASSET_PATH / "robot"
ROBOT_ASSETS = {
    "panda": RobotAsset(ROBOT_PATH / "panda" / "robot.xml", gripper_actuator="panda_finger_joint1"),
    "forte": RobotAsset(ROBOT_PATH / "forte" / "robot.xml", gripper_actuator="gripper_motor"),
}
ROBOT_NAMES = tuple(ROBOT_ASSETS)

ENVIRONMENT_PATH = ASSET_PATH / "environment"
ENVIRONMENT_SCENES = {
    "empty": ENVIRONMENT_PATH / "empty" / "scene.xml",
    "table_shelf": ENVIRONMENT_PATH / "table_shelf" / "scene.xml",
    "warehouse": ENVIRONMENT_PATH / "warehouse" / "scene.xml",
    **{
        f"cube_stack_{count}": ENVIRONMENT_PATH / f"cube_stack_{count}" / "scene.xml"
        for count in (2, 3, 4)
    },
    **{
        f"cube_stack_{count}_warehouse": ENVIRONMENT_PATH / f"cube_stack_{count}" / "warehouse.xml"
        for count in (2, 3, 4)
    },
}
ENVIRONMENT_NAMES = tuple(ENVIRONMENT_SCENES)
