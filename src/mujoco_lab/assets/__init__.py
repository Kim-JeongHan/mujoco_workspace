"""Bundled robot and environment asset locations."""

from pathlib import Path

ASSET_PATH = Path(__file__).parent
ROBOT_PATH = ASSET_PATH / "robot"
ROBOT_SCENES = {
    # "ur20": ROBOT_PATH / "ur20" / "robot.xml",
    # "ur30": ROBOT_PATH / "ur30" / "robot.xml",
    "panda": ROBOT_PATH / "panda" / "robot.xml",
    "forte": ROBOT_PATH / "forte" / "robot.xml",
}
ROBOT_NAMES = tuple(ROBOT_SCENES)

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
