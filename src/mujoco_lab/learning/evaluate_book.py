"""Build and evaluate book-insertion checkpoint environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import tyro

from mujoco_lab import RobotSpec, Simulator, create_book_insertion
from mujoco_lab.assets.loader import load_robot_config
from mujoco_lab.behaviors.book import BookTask, create_book_controller
from mujoco_lab.learning.config.config import EvalConfig, RolloutConfig
from mujoco_lab.learning.datasets.replay import replay_action_repeat
from mujoco_lab.learning.envs.book import BookEnv


@dataclass
class BookEvalConfig(EvalConfig):
    """Allow enough action steps for a complete book-insertion episode."""

    rollout: RolloutConfig = field(default_factory=lambda: RolloutConfig(max_steps=18_000))


def create_book_evaluation_env(
    metadata: dict[str, Any],
    *,
    xy_range: float | None = None,
    max_steps: int,
    book_yaw_range_degrees: float | None = None,
) -> tuple[BookEnv, dict[str, Any]]:
    """Restore the recorded book, robot, controller tuning, and action cadence."""
    replay = metadata["dataset_metadata"]["replay"]
    if replay["scene"] != "book_insertion" or replay["environment"] != "book_shelf":
        raise ValueError("Expected a recorded book-insertion scene in book_shelf")
    repeat = replay_action_repeat(replay)
    architecture_repeat = metadata.get("architecture", {}).get("physics_steps_per_action", repeat)
    if architecture_repeat != repeat:
        raise ValueError("Checkpoint action cadence differs from recorded book cadence")
    robot_type, robot_name = replay["robot"], replay["robot_name"]
    robot_config = load_robot_config(robot_type)
    simulator = Simulator(
        create_book_insertion(replay["book"]),
        robots=[RobotSpec(robot_name, robot_type, config=robot_config)],
        dt=replay["dt"],
    )
    robot = simulator.robots[robot_name]
    robot.change_controller(create_book_controller(robot, robot_config.controller))
    env = BookEnv(
        BookTask(simulator),
        xy_range=replay["xy_range"] if xy_range is None else xy_range,
        book_yaw_range_degrees=(
            replay["book_yaw_range_degrees"]
            if book_yaw_range_degrees is None
            else book_yaw_range_degrees
        ),
        max_steps=max_steps,
        physics_steps_per_action=repeat,
    )
    scene = {
        **replay,
        "xy_range": env.xy_range,
        "book_yaw_range_degrees": env.book_yaw_range_degrees,
        "simulation_hz": 1.0 / simulator.dt,
        "action_execution_hz": 1.0 / env.action_dt,
        "action_dt": env.action_dt,
    }
    return env, scene


def main() -> None:
    """Evaluate a checkpoint recorded from book insertion."""
    from mujoco_lab.learning.evaluate import run
    from mujoco_lab.utils.logger import Logger

    config = tyro.cli(BookEvalConfig, description="Evaluate a saved book-insertion BC policy")
    run_dir, summary = run(config, expected_scene="book_insertion")
    Logger().info(f"Saved {summary['attempted']} evaluation attempts to {run_dir}")


if __name__ == "__main__":
    main()
