"""Construct learning environments from scene settings without a BC checkpoint."""

from typing import overload

from mujoco_lab import RobotSpec, Simulator, create_book_insertion, create_cube_stack
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import BookTask, CubeStackTask
from mujoco_lab.learning.config.replay import BookReplayConfig, CubeStackReplayConfig
from mujoco_lab.learning.envs.book import BookEnv
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv


@overload
def create_env(
    scene: CubeStackReplayConfig,
    *,
    max_steps: int,
    dt: float = 0.002,
    robot_name: str | None = None,
    terminate_on_success: bool = True,
) -> CubeStackEnv: ...


@overload
def create_env(
    scene: BookReplayConfig,
    *,
    max_steps: int,
    dt: float = 0.002,
    robot_name: str | None = None,
    terminate_on_success: bool = True,
) -> BookEnv: ...


def create_env(
    scene: CubeStackReplayConfig | BookReplayConfig,
    *,
    max_steps: int,
    dt: float = 0.002,
    robot_name: str | None = None,
    terminate_on_success: bool = True,
) -> CubeStackEnv | BookEnv:
    """Share scene composition, robot configuration, and action cadence across trainers."""
    spec = (
        create_cube_stack(scene.cubes, environment=scene.environment)
        if isinstance(scene, CubeStackReplayConfig)
        else create_book_insertion(scene.book)
    )
    simulator = Simulator(
        spec,
        robots=[
            RobotSpec(robot_name or scene.robot, scene.robot, config=load_robot_config(scene.robot))
        ],
        dt=dt,
    )
    settings = {
        "max_steps": max_steps,
        "physics_steps_per_action": scene.physics_steps_per_action,
        "xy_range": scene.xy_range,
        "terminate_on_success": terminate_on_success,
    }
    if isinstance(scene, CubeStackReplayConfig):
        return CubeStackEnv(
            CubeStackTask(simulator, scene.cubes),
            min_gap=scene.min_gap,
            cube_yaw_range_degrees=scene.cube_yaw_range_degrees,
            **settings,
        )
    return BookEnv(
        BookTask(simulator),
        book_yaw_range_degrees=scene.book_yaw_range_degrees,
        **settings,
    )
