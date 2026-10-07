"""Load robot assets and editable environment scenes."""

from __future__ import annotations

import mujoco
import numpy as np

from mujoco_lab.assets import BOOK_SCENES, CUBE_COUNTS, ENVIRONMENT_SCENES, ROBOT_ASSETS, BookType
from mujoco_lab.assets.randomization import sample_cube_positions
from mujoco_lab.assets.robot.robot import RobotConfig


def load_asset(name: str) -> mujoco.MjSpec:
    asset = ROBOT_ASSETS[name]

    spec = mujoco.MjSpec.from_file(str(asset.path))
    if any(
        actuator.dyntype != mujoco.mjtDyn.mjDYN_NONE or actuator.actdim > 0
        for actuator in spec.actuators
    ) or any(body.mocap for body in spec.bodies):
        raise ValueError("Robot assets must have no activation or mocap state")
    return spec


def load_robot_config(name: str) -> RobotConfig:
    """Parse robot-local YAML into typed controller and joint gain settings."""
    asset = ROBOT_ASSETS[name]
    path = asset.path.with_name("robot.yaml")
    return RobotConfig.load(path)


def create_environment(name: str) -> mujoco.MjSpec:
    """Load a fresh editable environment spec with a robot mounting site."""
    path = ENVIRONMENT_SCENES[name]

    spec = mujoco.MjSpec.from_file(str(path))
    if spec.site("robot_mount") is None:
        raise ValueError(f"Environment {name!r} must define a 'robot_mount' site")
    return spec


def create_book_insertion(book_type: BookType = "medium") -> mujoco.MjSpec:
    """Load a fixed book insertion preset, including furniture and goal sites.

    Book physics, initial placement, and matching goals are defined in XML.
    The scene is editable for later randomization and robot composition.
    """
    return mujoco.MjSpec.from_file(str(BOOK_SCENES[book_type]))


def create_cube_stack(
    cubes: int = 2,
    *,
    environment: str = "table_shelf",
) -> mujoco.MjSpec:
    """Load the requested cube scene for later composition with a robot."""
    if cubes not in CUBE_COUNTS:
        raise ValueError(f"cubes must be one of {CUBE_COUNTS}")
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
    yaw_range_degrees: float = 0.0,
    seed: int | None = None,
    max_attempts: int = 100,
) -> None:
    """Randomize cube XY positions and yaw in a fresh, uncompiled cube scene.

    Each cube is sampled within ``xy_range`` meters of its current position.
    The function assumes bundled, axis-aligned cube bodies share a coordinate
    frame. Yaw offsets are sampled within ``yaw_range_degrees`` of the initial
    orientation, accounting for rotated footprints when checking spacing.
    Z positions, target bodies, and the robot stay unchanged. Call it on a
    fresh scene for each episode. A fixed seed makes sampling reproducible;
    this spacing check does not guarantee reachability.
    """
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
        half_sizes.append(geom.size[:2])

    positions, yaws = sample_cube_positions(
        np.array([body.pos[:2] for body in cubes]),
        np.array(half_sizes),
        np.random.default_rng(seed),
        xy_range,
        min_gap,
        max_attempts,
        yaw_range_degrees=yaw_range_degrees,
    )
    for body, xy, yaw in zip(cubes, positions, yaws, strict=True):
        position = body.pos.copy()
        position[:2] = xy
        body.pos = position
        if yaw_range_degrees:
            half_angle = np.deg2rad(yaw) / 2
            yaw_quat = np.array([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)])
            quaternion = np.empty(4)
            mujoco.mju_mulQuat(quaternion, yaw_quat, body.quat)
            body.quat = quaternion
