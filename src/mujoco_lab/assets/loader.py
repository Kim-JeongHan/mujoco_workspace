"""Load robot assets and editable environment scenes."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mujoco_lab.assets import ENVIRONMENT_SCENES, ROBOT_ASSETS
from mujoco_lab.assets.randomization import sample_cube_positions


@dataclass(frozen=True)
class AssetInfo:
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    site_names: tuple[str, ...]
    root_name: str
    gripper_actuator: str | None = None


def asset_info(spec: mujoco.MjSpec, gripper_actuator: str | None = None) -> AssetInfo:
    roots = spec.worldbody.bodies
    if len(roots) != 1:
        raise ValueError("Robot assets must have one fixed root body")
    return AssetInfo(
        tuple(joint.name for joint in spec.joints),
        tuple(actuator.name for actuator in spec.actuators),
        tuple(site.name for site in spec.sites if site.name and site.parent is not spec.worldbody),
        roots[0].name,
        gripper_actuator,
    )


def load_asset(name: str) -> tuple[mujoco.MjSpec, AssetInfo]:
    asset = ROBOT_ASSETS[name]

    spec = mujoco.MjSpec.from_file(str(asset.path))
    if (
        any(
            actuator.dyntype != mujoco.mjtDyn.mjDYN_NONE or actuator.actdim > 0
            for actuator in spec.actuators
        )
        or any(body.mocap for body in spec.bodies)
    ):
        raise ValueError("Robot assets must have no activation or mocap state")
    info = asset_info(spec, asset.gripper_actuator)
    if any(not name for name in info.joint_names + info.actuator_names):
        raise ValueError("Robot assets must name their joints and actuators")
    return spec, info


def create_environment(name: str) -> mujoco.MjSpec:
    """Load a fresh editable environment spec with a robot mounting site."""
    path = ENVIRONMENT_SCENES[name]

    spec = mujoco.MjSpec.from_file(str(path))
    if spec.site("robot_mount") is None:
        raise ValueError(f"Environment {name!r} must define a 'robot_mount' site")
    return spec


def create_cube_stack(
    cubes: int = 2,
    *,
    environment: str = "table_shelf",
) -> mujoco.MjSpec:
    """Load the requested cube scene for later composition with a robot."""
    if cubes not in (2, 3, 4):
        raise ValueError("cubes must be 2, 3, or 4")
    if environment not in ("table_shelf", "warehouse"):
        raise ValueError("Cube stacking requires table_shelf or warehouse")
    scene_name = f"cube_stack_{cubes}"
    if environment == "warehouse":
        scene_name += "_warehouse"
    return create_environment(scene_name)


def randomize_cube_positions(
    scene: mujoco.MjSpec,
    *,
    xy_range: float = 0.02,
    min_gap: float = 0.01,
    seed: int | None = None,
    max_attempts: int = 100,
) -> None:
    """Randomize cube XY positions in a fresh, uncompiled cube scene.

    Each cube is sampled within ``xy_range`` meters of its current position.
    The function assumes bundled, axis-aligned cube bodies share a coordinate
    frame and keeps their Z positions, rotations, target bodies, and robot
    unchanged. Call it on a fresh scene for each episode. A fixed seed makes
    sampling reproducible; this spacing check does not guarantee reachability.
    """
    if not np.isfinite(xy_range) or xy_range < 0:
        raise ValueError("xy_range must be finite and nonnegative")
    if not np.isfinite(min_gap) or min_gap < 0:
        raise ValueError("min_gap must be finite and nonnegative")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    cubes = sorted(
        (
            body
            for body in scene.bodies
            if body.name and body.name.startswith("cube") and body.name.endswith("/object_0")
        ),
        key=lambda body: body.name,
    )
    if not cubes:
        raise ValueError("Scene contains no cube object bodies")

    half_sizes = []
    for body in cubes:
        geom = scene.geom(body.name)
        if geom is None:
            raise ValueError(f"Cube body {body.name!r} has no same-named geom")
        half_sizes.append(geom.size[:2].copy())

    positions = sample_cube_positions(
        np.array([body.pos[:2].copy() for body in cubes]),
        np.array(half_sizes),
        np.random.default_rng(seed),
        xy_range,
        min_gap,
        max_attempts,
    )
    for body, xy in zip(cubes, positions, strict=True):
        position = body.pos.copy()
        position[:2] = xy
        body.pos = position
