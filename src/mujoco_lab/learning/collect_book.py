"""Collect physical book insertion demonstrations as compressed episodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tyro

from mujoco_lab import RobotSpec, Simulator, create_book_insertion
from mujoco_lab.assets import BookType, RobotName
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors import (
    BookInsertionExpert,
    BookTask,
    create_book_controller,
)
from mujoco_lab.behaviors.bookshelf_recipe import load_recipe as load_book_recipe
from mujoco_lab.learning.config.replay import BookReplayConfig
from mujoco_lab.learning.datasets.replay import capture_frame, capture_metadata
from mujoco_lab.learning.envs.book import BookEnv
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
    """Choose a Panda or Forte book demonstration collection run."""

    count: int = 1800  # Total target attempts, including existing episodes when resuming.
    seed: int = 42  # Starting seed for repeatable attempts.

    xy_range: float = 0.0  # Book offset range, in meters; zero uses the fixed baseline.
    # Maximum absolute initial book yaw, in degrees.
    book_yaw_range_degrees: float = 0.0
    simulation_hz: float = 500.0  # Physics and PD evaluations per simulated second.
    action_execution_hz: float = 100.0  # 500 Hz / 100 Hz = 5 physics steps/action.

    output_dir: Path = Path("data/forte_book_medium")  # Directory for episode NPZ files.
    resume: bool = False  # Continue or extend a collection to count total attempts.
    robot: RobotName = "forte"  # Bundled robot for demonstration collection.
    book: BookType = "medium"  # Bundled book dimensions.
    max_seconds: float = 180.0  # Maximum simulated duration of each attempt.
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
        return max_steps_for_seconds(
            self.max_seconds, self.simulation_dt * self.physics_steps_per_action
        )


def main() -> None:
    config = tyro.cli(Config, description="Collect Panda or Forte book demonstrations")
    repeat = config.physics_steps_per_action
    logger = Logger()
    logger.info(
        f"Collecting {config.count} total attempts to {config.output_dir.resolve()} "
        f"as compressed episode_<index:06d>.npz files (resume={config.resume})."
    )
    robot_config = load_robot_config(config.robot)
    simulator = Simulator(
        create_book_insertion(config.book),
        robots=[RobotSpec(config.robot, config.robot, config=robot_config)],
        dt=config.simulation_dt,
    )
    replay_metadata = capture_metadata(
        simulator,
        BookReplayConfig(
            book=config.book,
            robot=config.robot,
            physics_steps_per_action=repeat,
            book_yaw_range_degrees=config.book_yaw_range_degrees,
            xy_range=config.xy_range,
        ),
    )
    robot = simulator.robots[config.robot]
    controller = create_book_controller(robot, robot_config.controller)
    robot.change_controller(controller)
    task = BookTask(simulator)
    max_steps = max_steps_for_seconds(config.max_seconds, simulator.dt * repeat)
    env = BookEnv(
        task,
        xy_range=config.xy_range,
        book_yaw_range_degrees=config.book_yaw_range_degrees,
        max_steps=max_steps,
        physics_steps_per_action=repeat,
    )
    recipe = load_book_recipe(config.robot)
    expert = BookInsertionExpert(
        task,
        recipe=recipe,
        planning=config.planning,
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
