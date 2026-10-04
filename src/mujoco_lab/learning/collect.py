"""Collect physical cube stacking demonstrations as compressed episodes."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import tyro

from mujoco_lab import RobotSpec, Simulator, create_cube_stack
from mujoco_lab.assets import CubeCount, RobotName
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import (
    CubeStackExpert,
    CubeStackTask,
)
from mujoco_lab.behaviors.cube_stack_recipe import load_recipe
from mujoco_lab.control import create_controller
from mujoco_lab.learning.datasets.replay import capture_frame, cube_stack_metadata
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv
from mujoco_lab.learning.rollout.collector import iter_episodes
from mujoco_lab.planning import (
    PRMConfig,
    RRTConfig,
    RRTConnectConfig,
    default_planning,
    planner_from_config,
)
from mujoco_lab.utils import Logger


@dataclass
class Config:
    """Choose a Panda or Forte cube demonstration collection run."""

    count: int = 100  # Total target attempts, including existing episodes when resuming.
    seed: int = 42  # Starting seed for repeatable attempts.

    xy_range: float = 0.02  # Cube offset range, in meters.
    min_gap: float = 0.01  # Minimum cube edge gap, in meters.
    # Maximum absolute initial cube yaw, in degrees.
    cube_yaw_range_degrees: float = 45.0
    physics_steps_per_action: int = 5  # Physics steps per expert action.

    output_dir: Path = Path("data/demos")  # Directory for episode NPZ files.
    resume: bool = False  # Continue or extend a collection to count total attempts.
    robot: RobotName = "forte"  # Bundled robot for demonstration collection.
    cubes: CubeCount = 2  # Supported collection cube counts.
    method: Literal["heuristic", "sampling"] = "heuristic"  # Expert execution method.
    planning: RRTConnectConfig | RRTConfig | PRMConfig = field(
        default_factory=default_planning
    )  # Sampling planner settings.


def create_expert(
    task: CubeStackTask,
    *,
    planning: RRTConnectConfig | RRTConfig | PRMConfig,
    method: Literal["heuristic", "sampling"] = "heuristic",
) -> CubeStackExpert:
    """Build an action-producing expert bound to the environment's simulator."""
    if method not in ("heuristic", "sampling"):
        raise ValueError("method must be heuristic or sampling")
    robot = next(iter(task.simulator.robots.values()))
    return CubeStackExpert(
        task,
        recipe=load_recipe(robot.robot_type),
        method=method,
        planner=planner_from_config(planning) if method == "sampling" else None,
    )


def main() -> None:
    config = tyro.cli(Config, description="Collect Panda or Forte cube demonstrations")
    logger = Logger()
    logger.info(
        f"Collecting {config.count} total attempts to {config.output_dir.resolve()} "
        f"as compressed episode_<index:06d>.npz files (resume={config.resume})."
    )
    simulator = Simulator(
        create_cube_stack(config.cubes),
        robots=[RobotSpec(config.robot, config.robot, config=load_robot_config(config.robot))],
    )
    replay_metadata = cube_stack_metadata(
        simulator,
        cubes=config.cubes,
        robot=config.robot,
        physics_steps_per_action=config.physics_steps_per_action,
        cube_yaw_range_degrees=config.cube_yaw_range_degrees,
    )
    robot = simulator.robots[config.robot]
    controller = (
        create_controller(
            robot,
            replace(
                load_robot_config(robot.robot_type).controller,
                name="position",
                gravity_compensation=True,
                frame="grasp",
            ),
        )
        if config.robot == "panda"
        else create_controller(
            robot, replace(load_robot_config(robot.robot_type).controller, name="pd", frame="grasp")
        )
    )
    robot.change_controller(controller)
    task = CubeStackTask(simulator, config.cubes)
    physics_budget = 90_000 if config.method == "sampling" else 30_000
    max_steps = (
        physics_budget + config.physics_steps_per_action - 1
    ) // config.physics_steps_per_action
    env = CubeStackEnv(
        task,
        xy_range=config.xy_range,
        min_gap=config.min_gap,
        cube_yaw_range_degrees=config.cube_yaw_range_degrees,
        max_steps=max_steps,
        physics_steps_per_action=config.physics_steps_per_action,
    )
    expert = create_expert(task, method=config.method, planning=config.planning)

    saved = 0
    successes = 0
    for episode in iter_episodes(
        env,
        expert,
        config.count,
        seed=config.seed,
        max_steps=max_steps,
        output_dir=config.output_dir,
        resume=config.resume,
        record_frame=lambda: capture_frame(simulator),
        replay_metadata=replay_metadata,
    ):
        saved += 1
        successes += episode.metadata.get("success") is True
        logger.info(
            f"Saved seed={episode.metadata.get('seed')} "
            f"success={episode.metadata.get('success')} length={len(episode)}; "
            f"new episodes: {saved}, new successes: {successes}",
        )
        del episode
    logger.info(f"Saved {saved} new episodes to {config.output_dir}; new successes: {successes}")


if __name__ == "__main__":
    main()
