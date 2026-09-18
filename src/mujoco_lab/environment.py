"""Factory for editable environments composed from reusable object assets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mujoco_lab.assets import ASSET_ROOT

if TYPE_CHECKING:
    import mujoco

_ENVIRONMENTS = ASSET_ROOT / "environments"
ENVIRONMENT_SCENES = {
    "empty": _ENVIRONMENTS / "empty" / "scene.xml",
    "table_shelf": _ENVIRONMENTS / "table_shelf" / "scene.xml",
    "warehouse": _ENVIRONMENTS / "warehouse" / "scene.xml",
}
ENVIRONMENT_NAMES = tuple(ENVIRONMENT_SCENES)


def create_environment(name: str) -> mujoco.MjSpec:
    """Load a fresh environment specification with a named robot mounting site.

    Objects, placements, lights, and the floor belong to the environment. Returning
    an editable MjSpec allows a robot to be attached before the scene is compiled.
    """
    try:
        path = ENVIRONMENT_SCENES[name]
    except KeyError:
        available = ", ".join(ENVIRONMENT_NAMES)
        raise ValueError(
            f"Unknown environment {name!r}. Available environments: {available}"
        ) from None
    import mujoco

    spec = mujoco.MjSpec.from_file(str(path))
    if spec.site("robot_mount") is None:
        raise ValueError(f"Environment {name!r} must define a 'robot_mount' site")
    return spec
