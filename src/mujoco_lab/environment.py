"""Factory for editable environments composed from reusable object assets."""

from __future__ import annotations

import mujoco

from mujoco_lab.assets import ENVIRONMENT_SCENES
from mujoco_lab.robot import RobotSpec
from mujoco_lab.simulation import Simulator


def create_environment(name: str) -> mujoco.MjSpec:
    """Load a fresh environment specification with a named robot mounting site.

    Objects, placements, lights, and the floor belong to the environment. Returning
    an editable MjSpec allows a robot to be attached before the scene is compiled.
    """
    path = ENVIRONMENT_SCENES[name]

    spec = mujoco.MjSpec.from_file(str(path))
    if spec.site("robot_mount") is None:
        raise ValueError(f"Environment {name!r} must define a 'robot_mount' site")
    return spec


def create_cube_stack(
    cubes: int = 2,
    *,
    environment: str = "table_shelf",
    robot: str = "panda",
) -> Simulator:
    """Compose Panda or Forte with the requested XML cube environment."""
    if cubes not in (2, 3, 4):
        raise ValueError("cubes must be 2, 3, or 4")
    if environment not in ("table_shelf", "warehouse"):
        raise ValueError("Cube stacking requires table_shelf or warehouse")
    if robot not in ("panda", "forte"):
        raise ValueError("Cube stacking requires panda or forte")
    scene_name = f"cube_stack_{cubes}"
    if environment == "warehouse":
        scene_name += "_warehouse"
    scene = create_environment(scene_name)
    return Simulator(scene, robots=[RobotSpec(robot, robot)])
