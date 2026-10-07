"""Collect physical cube stacking demonstrations as compressed episodes."""

from __future__ import annotations

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
from mujoco_lab.learning.config.replay import CubeStackReplayConfig
from mujoco_lab.learning.datasets.replay import capture_frame, capture_metadata
from mujoco_lab.learning.envs.cube_stack import CubeStackEnv
from mujoco_lab.learning.rollout.collector import iter_episodes
from mujoco_lab.learning.timing import max_steps_for_seconds, physics_steps_per_action
from mujoco_lab.planning import (
    PRMConfig,
    RRTConfig,
    RRTConnectConfig,
    default_planning,
)
from mujoco_lab.utils import Logger


@dataclass
class Config:
    """Choose a Panda or Forte cube demonstration collection run."""

    count: int = 900  # Total target attempts, including existing episodes when resuming.
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
    method: Literal["heuristic", "sampling"] = "heuristic"  # Expert execution method.
    # Maximum simulated duration per attempt; None uses 60 s heuristic or 180 s sampling.
    max_seconds: float | None = None
    planning: RRTConnectConfig | RRTConfig | PRMConfig = field(
        default_factory=default_planning
    )  # Sampling planner settings.

    @property
    def simulation_dt(self) -> float:
        return 1.0 / self.simulation_hz

    @property
    def physics_steps_per_action(self) -> int:
        return physics_steps_per_action(self.simulation_hz, self.action_execution_hz)

    @property
    def max_steps(self) -> int:
        duration = self.max_seconds
        if duration is None:
            duration = 180.0 if self.method == "sampling" else 60.0
        return max_steps_for_seconds(duration, self.simulation_dt * self.physics_steps_per_action)


def main() -> None:
    config = tyro.cli(Config, description="Collect Panda or Forte cube demonstrations")
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
    replay_metadata = capture_metadata(
        simulator,
        CubeStackReplayConfig(
            cubes=config.cubes,
            robot=config.robot,
            physics_steps_per_action=repeat,
            cube_yaw_range_degrees=config.cube_yaw_range_degrees,
            xy_range=config.xy_range,
            min_gap=config.min_gap,
        ),
    )
    robot = simulator.robots[config.robot]
    controller = create_controller(robot, robot_config.controller)
    robot.change_controller(controller)
    task = CubeStackTask(simulator, config.cubes)
    max_steps = config.max_steps
    env = CubeStackEnv(
        task,
        xy_range=config.xy_range,
        min_gap=config.min_gap,
        cube_yaw_range_degrees=config.cube_yaw_range_degrees,
        max_steps=max_steps,
        physics_steps_per_action=repeat,
    )
    recipe = load_cube_recipe(config.robot)
    expert = CubeStackExpert(
        task,
        recipe=recipe,
        method=config.method,
        planning=config.planning if config.method == "sampling" else None,
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
