"""Collect physical cube stacking demonstrations as compressed episodes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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
from mujoco_lab.behaviors.cube_stack_recipe import load_recipe as load_cube_recipe
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

    count: int = 1800  # Total target attempts, including existing episodes when resuming.
    seed: int = 42  # Starting seed for repeatable attempts.

    xy_range: float = 0.02  # Cube offset range, in meters.
    min_gap: float = 0.01  # Minimum cube edge gap, in meters.
    # Maximum absolute initial cube yaw, in degrees.
    cube_yaw_range_degrees: float = 45.0
    simulation_hz: float = 500.0  # Physics and PD evaluations per simulated second.
    action_execution_hz: float = 100.0  # 500 Hz / 100 Hz = 5 physics steps/action.

    output_dir: Path = Path("data/forte_mixed_1")  # Directory for episode NPZ files.
    resume: bool = False  # Continue or extend a collection to count total attempts.
    robot: RobotName = "forte"  # Bundled robot for demonstration collection.
    cubes: CubeCount = 1  # Supported collection cube counts.
    method: Literal["heuristic", "sampling"] = "sampling"  # Expert execution method.
    planning: RRTConnectConfig | RRTConfig | PRMConfig = field(
        default_factory=default_planning
    )  # Sampling planner settings.

    @property
    def simulation_dt(self) -> float:
        return 1.0 / self.simulation_hz

    @property
    def physics_steps_per_action(self) -> int:
        return int(self.simulation_hz / self.action_execution_hz)

    def validate(self) -> None:
        """Reject invalid collection cadence and randomization parameters."""
        if not (
            math.isfinite(self.simulation_hz)
            and self.simulation_hz > 0
            and math.isfinite(self.action_execution_hz)
            and self.action_execution_hz > 0
        ):
            raise ValueError("simulation_hz and action_execution_hz must be finite and positive")
        if not (self.simulation_hz / self.action_execution_hz).is_integer():
            raise ValueError("simulation_hz / action_execution_hz must be a positive integer")
        if self.count <= 0:
            raise ValueError("count must be positive")
        for name in ("xy_range", "min_gap", "cube_yaw_range_degrees"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


def main() -> None:
    config = tyro.cli(Config, description="Collect Panda or Forte cube demonstrations")
    config.validate()
    repeat = config.physics_steps_per_action
    logger = Logger()
    logger.info(
        f"Collecting {config.count} total attempts to {config.output_dir.resolve()} "
        f"as compressed episode_<index:06d>.npz files (resume={config.resume})."
    )
    robot_config = load_robot_config(config.robot)
    simulator = Simulator(
        create_cube_stack(config.cubes),
        robots=[RobotSpec(config.robot, config.robot, config=robot_config)],
        dt=config.simulation_dt,
    )
    replay_metadata = cube_stack_metadata(
        simulator,
        cubes=config.cubes,
        robot=config.robot,
        physics_steps_per_action=config.physics_steps_per_action,
        cube_yaw_range_degrees=config.cube_yaw_range_degrees,
    )
    robot = simulator.robots[config.robot]
    controller = create_controller(robot, robot_config.controller)
    robot.change_controller(controller)
    task = CubeStackTask(simulator, config.cubes)
    duration = 180 if config.method == "sampling" else 60
    max_steps = math.ceil(duration * config.action_execution_hz)
    env = CubeStackEnv(
        task,
        xy_range=config.xy_range,
        min_gap=config.min_gap,
        cube_yaw_range_degrees=config.cube_yaw_range_degrees,
        max_steps=max_steps,
        physics_steps_per_action=repeat,
    )
    planner = planner_from_config(config.planning) if config.method == "sampling" else None
    recipe = load_cube_recipe(robot.robot_type)
    expert = CubeStackExpert(
        task,
        recipe=recipe,
        method=config.method,
        planner=planner,
    )

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
