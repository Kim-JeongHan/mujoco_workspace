"""Bundled robot and environment asset locations."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

type RobotName = Literal["panda", "forte"]
type CubeCount = Literal[1, 2, 3, 4]
CUBE_COUNTS: tuple[CubeCount, ...] = (1, 2, 3, 4)
type BookType = Literal["small", "medium", "thick", "large"]
BOOK_TYPES: tuple[BookType, ...] = ("small", "medium", "thick", "large")


@dataclass(frozen=True)
class RobotAsset:
    path: Path


ASSET_PATH = Path(__file__).parent
ROBOT_PATH = ASSET_PATH / "robot"
ROBOT_ASSETS = {
    "panda": RobotAsset(ROBOT_PATH / "panda" / "robot.xml"),
    "forte": RobotAsset(ROBOT_PATH / "forte" / "robot.xml"),
}
ROBOT_NAMES = tuple(ROBOT_ASSETS)

ENVIRONMENT_PATH = ASSET_PATH / "environment"
BOOK_SCENES = {name: ENVIRONMENT_PATH / "book_shelf" / f"{name}.xml" for name in BOOK_TYPES}
ENVIRONMENT_SCENES = {
    "empty": ENVIRONMENT_PATH / "empty" / "scene.xml",
    "table_shelf": ENVIRONMENT_PATH / "table_shelf" / "scene.xml",
    "warehouse": ENVIRONMENT_PATH / "warehouse" / "scene.xml",
    "book_shelf": ENVIRONMENT_PATH / "book_shelf" / "scene.xml",
    **{
        f"cube_stack_{count}": ENVIRONMENT_PATH / f"cube_stack_{count}" / "scene.xml"
        for count in CUBE_COUNTS
    },
    **{
        f"cube_stack_{count}_warehouse": ENVIRONMENT_PATH / f"cube_stack_{count}" / "warehouse.xml"
        for count in CUBE_COUNTS
    },
}
ENVIRONMENT_NAMES = tuple(ENVIRONMENT_SCENES)
