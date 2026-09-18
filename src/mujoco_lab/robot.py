"""Factory for creating the manipulators bundled with this workspace."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mujoco_lab.assets import ASSET_ROOT
from mujoco_lab.environment import create_environment
from mujoco_lab.simulation import initialize_data

if TYPE_CHECKING:
    import mujoco

ROBOT_SCENES = {
    "ur20": ASSET_ROOT / "ur20" / "robot.xml",
    "ur30": ASSET_ROOT / "ur30" / "robot.xml",
    "panda": ASSET_ROOT / "panda" / "robot.xml",
    "forte": ASSET_ROOT / "forte" / "robot.xml",
}
ROBOT_NAMES = tuple(ROBOT_SCENES)


def create_robot(name: str, *, environment: str = "empty") -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compose a robot and environment, then initialize the robot's home pose.

    Extend ``ROBOT_SCENES`` when adding a manipulator. This factory is the common
    creation point for the CLI, examples, and Python callers. Registry entries
    point to robot-only MJCF files; the environment supplies the floor and placement.
    """
    try:
        scene = ROBOT_SCENES[name]
    except KeyError:
        available = ", ".join(ROBOT_NAMES)
        raise ValueError(f"Unknown robot {name!r}. Available robots: {available}") from None
    import mujoco

    spec = create_environment(environment)
    robot = mujoco.MjSpec.from_file(str(scene))
    spec.attach(robot, prefix="", site=spec.site("robot_mount"))
    spec.modelname = f"{name}_{environment}"
    model = spec.compile()
    return model, initialize_data(model)
